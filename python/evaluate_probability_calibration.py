"""
Evaluate calibration buckets for probability outputs such as
`prob_top20_60d` and `prob_top20_90d`.

Default behavior:
  - load predictions from CSV
  - if actual label columns are missing in the predictions file, join labels.csv
    by (date, code)
  - build calibration bucket tables
  - print summary tables
  - optionally save CSV outputs

Usage:
  python python/evaluate_probability_calibration.py
  python python/evaluate_probability_calibration.py --no-save
  python python/evaluate_probability_calibration.py --predictions-csv data/my_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
OUTPUT_DIR = Path("outputs") / "calibration_tables"

PROBABILITY_FIELDS: Dict[str, str] = {
    "prob_top20_60d": "target_60d_top20",
    "prob_top20_90d": "target_90d_top20",
}

NUM_BUCKETS = 10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate probability calibration buckets")
    ap.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    ap.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--buckets", type=int, default=NUM_BUCKETS)
    ap.add_argument("--no-save", action="store_true", help="Do not save CSV outputs")
    return ap.parse_args()


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path.resolve()}")
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError(f"{name} must contain 'date' and 'code' columns")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_inputs(predictions_csv: Path, labels_csv: Path) -> pd.DataFrame:
    preds = _load_csv(predictions_csv, "predictions_csv")

    required_label_cols = list(PROBABILITY_FIELDS.values())
    has_all_actuals = all(col in preds.columns for col in required_label_cols)
    if has_all_actuals:
        return preds

    labels = _load_csv(labels_csv, "labels_csv")
    keep_cols = ["date", "code"] + [c for c in required_label_cols if c in labels.columns]
    missing = [c for c in required_label_cols if c not in labels.columns]
    if missing:
        raise ValueError(f"labels_csv is missing required target columns: {missing}")

    merged = preds.merge(labels[keep_cols], on=["date", "code"], how="left")
    return merged


def build_bucket_table(df: pd.DataFrame, prob_col: str, actual_col: str, buckets: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    work = df[[prob_col, actual_col]].copy()
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(), {"rows": 0, "brier_score": float("nan")}

    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work = work.dropna()
    work = work[(work[prob_col] >= 0.0) & (work[prob_col] <= 1.0)]
    if work.empty:
        return pd.DataFrame(), {"rows": 0, "brier_score": float("nan")}

    edges = np.linspace(0.0, 1.0, buckets + 1)
    bucket_ids = pd.cut(
        work[prob_col],
        bins=edges,
        include_lowest=True,
        labels=False,
    )
    work["bucket_id"] = bucket_ids.astype("Int64")
    work = work[work["bucket_id"].notna()].copy()
    work["bucket_id"] = work["bucket_id"].astype(int)

    agg = (
        work.groupby("bucket_id", as_index=False)
        .agg(
            predicted_mean=(prob_col, "mean"),
            actual_hit_rate=(actual_col, "mean"),
            sample_count=(actual_col, "count"),
            prob_min=(prob_col, "min"),
            prob_max=(prob_col, "max"),
        )
        .sort_values("bucket_id")
        .reset_index(drop=True)
    )
    agg["bucket_label"] = agg["bucket_id"].apply(
        lambda i: f"[{edges[i]:.1f}, {edges[i + 1]:.1f}]"
    )
    agg["gap"] = agg["actual_hit_rate"] - agg["predicted_mean"]
    agg = agg[
        [
            "bucket_id",
            "bucket_label",
            "prob_min",
            "prob_max",
            "predicted_mean",
            "actual_hit_rate",
            "gap",
            "sample_count",
        ]
    ]

    brier_score = float(np.mean((work[prob_col] - work[actual_col]) ** 2))
    summary = {
        "rows": int(len(work)),
        "brier_score": brier_score,
    }
    return agg, summary


def print_table(prob_col: str, actual_col: str, table: pd.DataFrame, summary: Dict[str, float]) -> None:
    print("")
    print(f"[{prob_col}] actual={actual_col}")
    print(f"rows={summary['rows']}, brier_score={summary['brier_score']:.6f}")
    if table.empty:
        print("No valid rows.")
        return
    print(table.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()
    df = load_inputs(args.predictions_csv, args.labels_csv)

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for prob_col, actual_col in PROBABILITY_FIELDS.items():
        if prob_col not in df.columns:
            print(f"\n[{prob_col}] skipped: probability column not found")
            continue
        if actual_col not in df.columns:
            print(f"\n[{prob_col}] skipped: actual label column '{actual_col}' not found")
            continue

        table, summary = build_bucket_table(df, prob_col, actual_col, args.buckets)
        print_table(prob_col, actual_col, table, summary)

        if not args.no_save:
            out_path = args.output_dir / f"{prob_col}_calibration.csv"
            table.to_csv(out_path, index=False, encoding="utf-8")
            print(f"saved_csv={out_path}")


if __name__ == "__main__":
    main()
