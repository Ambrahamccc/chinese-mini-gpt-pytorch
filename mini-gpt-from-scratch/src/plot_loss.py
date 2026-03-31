from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from metrics.csv.")
    parser.add_argument("--input", type=str, default="outputs/logs/metrics.csv")
    parser.add_argument("--output", type=str, default="outputs/figures/loss_curve.png")
    parser.add_argument("--title", type=str, default="Training and Validation Loss")
    return parser.parse_args()


def load_metrics(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    steps: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "train_loss", "val_loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            if not row.get("step"):
                continue
            steps.append(float(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not steps:
        raise ValueError("Metrics CSV is empty.")

    return steps, train_losses, val_losses


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps, train_losses, val_losses = load_metrics(input_path)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="train_loss")
    plt.plot(steps, val_losses, label="val_loss")
    plt.xlabel("Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.title(args.title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
