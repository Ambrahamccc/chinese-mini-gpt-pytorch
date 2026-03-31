# Experiment Summary

| Version | Tokenizer | Data Pipeline | Eval | Generation | Engineering |
|---|---|---|---|---|---|
| v1 | char-level | full in-memory | no val set | basic sampling | low |
| v2 | SentencePiece | full tensor | partial | top-k / top-p | medium |
| v3 | SentencePiece | bin + memmap | train/val + best ckpt | controlled sampling | high |
