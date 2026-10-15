from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": payload["experiment"],
                "loss": payload["loss"],
                "warmup_ratio": payload["warmup_ratio"],
                "margin": payload["margin"],
                "seed": payload["seed"],
                "validation_auc": payload["best_validation"],
                "test_auc": payload["test"]["auc"],
                "path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no result.json files found under {root}")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "loss", "warmup_ratio", "margin"]
    summary = (
        frame.groupby(keys, dropna=False)
        .agg(
            runs=("seed", "size"),
            validation_auc_mean=("validation_auc", "mean"),
            validation_auc_std=("validation_auc", "std"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
        )
        .reset_index()
    )
    return summary.sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validation-selected runs.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()
    frame = collect(args.root)
    summary = summarize(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

