from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch

from model import GPTConfig, MiniGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniGPT on train/val splits.")
    parser.add_argument("--config", type=str, default="configs/base_config.json")
    parser.add_argument("--spm_model", type=str, default="data/tokenizer/spm_bpe_12k.model")
    parser.add_argument("--train_bin", type=str, default="data/processed/train.bin")
    parser.add_argument("--val_bin", type=str, default="data/processed/val.bin")
    parser.add_argument("--meta", type=str, default="data/processed/data_meta.json")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    return parser.parse_args()


def choose_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_meta(meta: dict[str, Any]) -> np.dtype:
    dtype_name = meta.get("dtype", "uint16")
    if dtype_name == "uint16":
        return np.uint16
    if dtype_name == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported dtype in meta: {dtype_name}")


def load_config(config_path: Path, vocab_size: int) -> GPTConfig:
    if config_path.exists():
        return GPTConfig.from_json(config_path, vocab_size=vocab_size)
    return GPTConfig(vocab_size=vocab_size)


def extract_state_dict(raw_ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw_ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
                return raw_ckpt[key]
        if all(isinstance(k, str) for k in raw_ckpt.keys()):
            return raw_ckpt
    raise ValueError("Unsupported checkpoint format.")


def maybe_extract_config(raw_ckpt: Any, fallback: GPTConfig) -> GPTConfig:
    if isinstance(raw_ckpt, dict) and "model_config" in raw_ckpt and isinstance(raw_ckpt["model_config"], dict):
        return GPTConfig.from_dict(raw_ckpt["model_config"])
    return fallback


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("Dataset is too small for current block_size.")

    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64)) for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(np.array(data[i + 1 : i + block_size + 1], dtype=np.int64)) for i in ix.tolist()
    ])

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_split_loss(model: MiniGPT, data: np.memmap, batch_size: int, eval_iters: int, device: str) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    use_amp = device == "cuda"

    for k in range(eval_iters):
        xb, yb = get_batch(data, batch_size, model.config.block_size, device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, loss = model(xb, yb)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    spm_model_path = Path(args.spm_model)
    meta_path = Path(args.meta)
    train_bin_path = Path(args.train_bin)
    val_bin_path = Path(args.val_bin)
    ckpt_path = Path(args.ckpt)
    config_path = Path(args.config)

    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")
    if not train_bin_path.exists():
        raise FileNotFoundError(f"train.bin not found: {train_bin_path}")
    if not val_bin_path.exists():
        raise FileNotFoundError(f"val.bin not found: {val_bin_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    vocab_size = sp.get_piece_size()

    meta = load_meta(meta_path)
    dtype = dtype_from_meta(meta)
    fallback_config = load_config(config_path, vocab_size=vocab_size)

    raw_ckpt = torch.load(ckpt_path, map_location=device)
    config = maybe_extract_config(raw_ckpt, fallback=fallback_config)
    model = MiniGPT(config).to(device)
    model.load_state_dict(extract_state_dict(raw_ckpt), strict=True)

    train_data = np.memmap(train_bin_path, dtype=dtype, mode="r")
    val_data = np.memmap(val_bin_path, dtype=dtype, mode="r")

    num_params = model.num_parameters()
    train_loss = estimate_split_loss(model, train_data, args.batch_size, args.eval_iters, device)
    val_loss = estimate_split_loss(model, val_data, args.batch_size, args.eval_iters, device)

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    print("=" * 60)
    print("MiniGPT evaluation")
    print("=" * 60)
    print(f"device:           {device}")
    print(f"checkpoint:       {ckpt_path}")
    print(f"vocab_size:       {vocab_size}")
    print(f"model params:     {num_params / 1e6:.2f} M")
    print(f"block_size:       {config.block_size}")
    print(f"batch_size:       {args.batch_size}")
    print(f"eval_iters:       {args.eval_iters}")
    print("-" * 60)
    print(f"train_loss:       {train_loss:.4f}")
    print(f"val_loss:         {val_loss:.4f}")
    print(f"train_perplexity: {train_ppl:.4f}")
    print(f"val_perplexity:   {val_ppl:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
