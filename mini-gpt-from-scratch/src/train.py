from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm

from model import GPTConfig, MiniGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniGPT.")
    parser.add_argument("--config", type=str, default="configs/base_config.json")
    parser.add_argument("--spm_model", type=str, default="data/tokenizer/spm_bpe_12k.model")
    parser.add_argument("--train_bin", type=str, default="data/processed/train.bin")
    parser.add_argument("--val_bin", type=str, default="data/processed/val.bin")
    parser.add_argument("--meta", type=str, default="data/processed/data_meta.json")
    parser.add_argument("--out_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log_csv", type=str, default="outputs/logs/metrics.csv")
    parser.add_argument("--sample_dir", type=str, default="outputs/samples")
    parser.add_argument("--resume", action="store_true")
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_meta(meta: dict[str, Any]) -> np.dtype:
    dtype_name = meta.get("dtype", "uint16")
    if dtype_name == "uint16":
        return np.uint16
    if dtype_name == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported dtype in meta: {dtype_name}")


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
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
def estimate_loss(
    model: MiniGPT,
    train_data: np.memmap,
    val_data: np.memmap,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    use_amp = device == "cuda"

    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, batch_size, model.config.block_size, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def ensure_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "best_val_loss", "learning_rate"])


def append_metrics_row(path: Path, row: list[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": model.config.to_dict(),
    }
    torch.save(ckpt, path)


def generate_sample(
    model: MiniGPT,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    device: str,
) -> str:
    prompt_ids = sp.encode(prompt, out_type=int)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return sp.decode(out[0].tolist())


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    config_path = Path(args.config)
    spm_model_path = Path(args.spm_model)
    train_bin_path = Path(args.train_bin)
    val_bin_path = Path(args.val_bin)
    meta_path = Path(args.meta)
    out_dir = Path(args.out_dir)
    log_csv_path = Path(args.log_csv)
    sample_dir = Path(args.sample_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")
    if not train_bin_path.exists():
        raise FileNotFoundError(f"train.bin not found: {train_bin_path}")
    if not val_bin_path.exists():
        raise FileNotFoundError(f"val.bin not found: {val_bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"data_meta.json not found: {meta_path}")

    cfg = load_json(config_path)
    meta = load_json(meta_path)
    dtype = dtype_from_meta(meta)

    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    vocab_size = sp.get_piece_size()

    model_config = GPTConfig.from_dict({**cfg, "vocab_size": vocab_size})
    model = MiniGPT(model_config).to(device)

    train_data = np.memmap(train_bin_path, dtype=dtype, mode="r")
    val_data = np.memmap(val_bin_path, dtype=dtype, mode="r")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_size = int(cfg.get("batch_size", 32))
    max_iters = int(cfg.get("max_iters", 50000))
    eval_interval = int(cfg.get("eval_interval", 500))
    eval_iters = int(cfg.get("eval_iters", 100))
    sample_interval = int(cfg.get("sample_interval", 1000))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    prompt = str(cfg.get("sample_prompt", "你好，世界"))
    sample_max_new_tokens = int(cfg.get("sample_max_new_tokens", 120))
    temperature = float(cfg.get("temperature", 0.8))
    top_k = cfg.get("top_k", 50)
    top_p = cfg.get("top_p", 0.9)
    repetition_penalty = cfg.get("repetition_penalty", 1.15)

    start_iter = 0
    best_val_loss = float("inf")
    last_ckpt_path = out_dir / "last_model.pth"
    best_ckpt_path = out_dir / "best_model.pth"

    if args.resume and last_ckpt_path.exists():
        raw_ckpt = torch.load(last_ckpt_path, map_location=device)
        state_dict = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
        model.load_state_dict(state_dict)
        if isinstance(raw_ckpt, dict) and "optimizer" in raw_ckpt:
            optimizer.load_state_dict(raw_ckpt["optimizer"])
        start_iter = int(raw_ckpt.get("step", 0)) + 1 if isinstance(raw_ckpt, dict) else 0
        best_val_loss = float(raw_ckpt.get("best_val_loss", float("inf"))) if isinstance(raw_ckpt, dict) else float("inf")
        print(f"Resumed from {last_ckpt_path} at step {start_iter}")

    ensure_metrics_csv(log_csv_path)
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniGPT training")
    print("=" * 60)
    print(f"device:       {device}")
    print(f"vocab_size:   {vocab_size}")
    print(f"model params: {model.num_parameters() / 1e6:.2f} M")
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens:   {len(val_data):,}")
    print("=" * 60)

    pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters)
    for step in pbar:
        xb, yb = get_batch(train_data, batch_size, model_config.block_size, device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        pbar.set_description(f"step {step} loss {loss.item():.4f}")

        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, eval_iters, device)
            train_loss = losses["train"]
            val_loss = losses["val"]
            lr = optimizer.param_groups[0]["lr"]
            best_val_loss = min(best_val_loss, val_loss)
            append_metrics_row(log_csv_path, [step, train_loss, val_loss, best_val_loss, lr])

            print(f"\nstep {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val_loss:.4f}")

            save_checkpoint(last_ckpt_path, model, optimizer, step, best_val_loss)
            if val_loss <= best_val_loss:
                save_checkpoint(best_ckpt_path, model, optimizer, step, best_val_loss)

        if (step % sample_interval == 0 or step == max_iters - 1) and step > 0:
            sample_text = generate_sample(
                model=model,
                sp=sp,
                prompt=prompt,
                max_new_tokens=sample_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device,
            )
            sample_path = sample_dir / f"sample_step_{step}.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            print(f"Saved sample to {sample_path}")

    print("Training finished.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Metrics CSV:     {log_csv_path}")


if __name__ == "__main__":
    main()
