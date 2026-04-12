from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from model import GPTConfig, MiniGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint.")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/base_config.json")
    parser.add_argument("--spm_model", type=str, default="data/tokenizer/spm_bpe_12k.model")
    parser.add_argument("--prompt", type=str, default="问题：为什么天空是蓝色的？\n回答：")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
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


def load_config(config_path: Path, vocab_size: int) -> GPTConfig:
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        data["vocab_size"] = vocab_size
        return GPTConfig.from_dict(data)
    return GPTConfig(vocab_size=vocab_size)


def extract_config(ckpt: Any, fallback: GPTConfig) -> GPTConfig:
    if isinstance(ckpt, dict) and "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
        return GPTConfig.from_dict(ckpt["model_config"])
    return fallback


def extract_state_dict(raw_ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw_ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
                return raw_ckpt[key]
        if all(isinstance(k, str) for k in raw_ckpt.keys()) and all(
            isinstance(v, torch.Tensor) for v in raw_ckpt.values()
        ):
            return raw_ckpt
    raise ValueError("Unsupported checkpoint format.")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt_path = Path(args.ckpt)
    config_path = Path(args.config)
    spm_model_path = Path(args.spm_model)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")

    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    vocab_size = sp.get_piece_size()

    fallback_config = load_config(config_path, vocab_size=vocab_size)
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    config = extract_config(raw_ckpt, fallback=fallback_config)
    model = MiniGPT(config).to(device)
    model.load_state_dict(extract_state_dict(raw_ckpt), strict=True)
    model.eval()

    prompt_ids = sp.encode(args.prompt, out_type=int)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print("=" * 60)
    print("Prompt:")
    print(args.prompt)
    print("-" * 60)
    print("Output:")
    print(sp.decode(out[0].tolist()))
    print("=" * 60)


if __name__ == "__main__":
    main()
