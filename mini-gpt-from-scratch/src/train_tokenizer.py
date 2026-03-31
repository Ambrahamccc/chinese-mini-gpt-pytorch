from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument("--input", type=str, default="data/raw/wiki_cleaned_train.txt")
    parser.add_argument("--model_prefix", type=str, default="data/tokenizer/spm_bpe_12k")
    parser.add_argument("--vocab_size", type=int, default=12000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", type=str, default="bpe")
    parser.add_argument("--input_sentence_size", type=int, default=1000000)
    parser.add_argument("--shuffle_input_sentence", action="store_true")
    parser.add_argument("--max_sentence_length", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    model_prefix = Path(args.model_prefix)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,
        max_sentence_length=args.max_sentence_length,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
    )

    print("Tokenizer training finished.")
    print(f"Model: {model_prefix}.model")
    print(f"Vocab: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
