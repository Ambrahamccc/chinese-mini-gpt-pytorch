from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train.bin / val.bin from raw text.")
    parser.add_argument("--input", type=str, default="data/raw/wiki_cleaned_train.txt")
    parser.add_argument("--spm_model", type=str, default="data/tokenizer/spm_bpe_12k.model")
    parser.add_argument("--train_bin", type=str, default="data/processed/train.bin")
    parser.add_argument("--val_bin", type=str, default="data/processed/val.bin")
    parser.add_argument("--meta", type=str, default="data/processed/data_meta.json")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--append_eos", action="store_true", help="Append EOS after each non-empty line.")
    return parser.parse_args()


def pick_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError(f"vocab_size={vocab_size} is too large for uint32.")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    spm_model_path = Path(args.spm_model)
    train_bin_path = Path(args.train_bin)
    val_bin_path = Path(args.val_bin)
    meta_path = Path(args.meta)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw text file not found: {input_path}")
    if not spm_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")

    train_bin_path.parent.mkdir(parents=True, exist_ok=True)
    val_bin_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(spm_model_path))
    vocab_size = sp.get_piece_size()
    dtype = pick_dtype(vocab_size)
    eos_id = sp.eos_id()

    token_chunks: list[np.ndarray] = []
    line_count = 0
    non_empty_line_count = 0

    print(f"Loading SentencePiece model: {spm_model_path}")
    print(f"vocab_size = {vocab_size}, dtype = {np.dtype(dtype).name}")
    print(f"Encoding file line by line: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line_count += 1
            line = raw_line.strip()
            if not line:
                continue

            ids = sp.encode(line, out_type=int)
            if args.append_eos and eos_id != -1:
                ids.append(eos_id)

            if ids:
                token_chunks.append(np.asarray(ids, dtype=np.int64))
                non_empty_line_count += 1

    if not token_chunks:
        raise ValueError("No tokens were produced. Check your input file and tokenizer.")

    all_tokens = np.concatenate(token_chunks, axis=0)
    total_tokens = int(all_tokens.shape[0])

    if not (0.0 < args.split_ratio < 1.0):
        raise ValueError("--split_ratio must be between 0 and 1.")

    split_idx = int(total_tokens * args.split_ratio)
    train_tokens = all_tokens[:split_idx].astype(dtype, copy=False)
    val_tokens = all_tokens[split_idx:].astype(dtype, copy=False)

    if len(train_tokens) == 0 or len(val_tokens) == 0:
        raise ValueError("Train/val split failed. Try a different --split_ratio or use more data.")

    train_tokens.tofile(train_bin_path)
    val_tokens.tofile(val_bin_path)

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": int(vocab_size),
        "tokenizer_model": str(spm_model_path.as_posix()),
        "raw_input": str(input_path.as_posix()),
        "total_tokens": total_tokens,
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "split_ratio": float(args.split_ratio),
        "append_eos": bool(args.append_eos),
        "line_count": int(line_count),
        "non_empty_line_count": int(non_empty_line_count),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Train tokens: {train_tokens.shape[0]:,} -> {train_bin_path}")
    print(f"Val tokens:   {val_tokens.shape[0]:,} -> {val_bin_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
