"""
Snapshot ranking/prediction/feature outputs with a date stamp and emit basic metrics.

Usage:
  python scripts/snapshot_scores.py

Creates:
  outputs/snapshots/YYYYMMDD/
    - ranking_final_YYYYMMDD.csv
    - predictions_YYYYMMDD.csv
    - features_YYYYMMDD.csv
    - top20_YYYYMMDD.csv
    - metrics_YYYYMMDD.json
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOT_ROOT = ROOT / "outputs" / "snapshots"


def copy_with_date(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / f"{src.stem}_{dest_dir.name}{src.suffix}"
    shutil.copy2(src, dest)
    return dest


def score_distribution(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    series = df[score_col].dropna()
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "p99": float(series.quantile(0.99)),
    }


def top20_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    top = df.sort_values(score_col, ascending=False).head(20)
    metrics: Dict[str, float] = {}

    def add_if_exists(name: str, cols: List[str], fn) -> None:
        for col in cols:
            if col in top.columns:
                metrics[name] = float(fn(top[col].dropna()))
                return

    add_if_exists("pred_return_60d_mean", ["pred_return_60d"], lambda s: s.mean())
    add_if_exists("pred_return_90d_mean", ["pred_return_90d"], lambda s: s.mean())
    add_if_exists("pred_mdd_60d_min", ["pred_mdd_60d"], lambda s: s.min())
    add_if_exists("pred_mdd_90d_min", ["pred_mdd_90d"], lambda s: s.min())
    add_if_exists("vol_20_mean", ["vol_20"], lambda s: s.mean())
    add_if_exists("vol_60_mean", ["vol_60"], lambda s: s.mean())
    add_if_exists("risk_penalty_mean", ["risk_penalty"], lambda s: s.mean())
    return metrics


def main() -> None:
    today = datetime.today().strftime("%Y%m%d")
    dest_dir = SNAPSHOT_ROOT / today
    dest_dir.mkdir(parents=True, exist_ok=True)

    ranking_src = DATA / "ranking_final.csv"
    predictions_src = DATA / "predictions.csv"
    features_src = DATA / "features.csv"

    ranking_path = copy_with_date(ranking_src, dest_dir)
    copy_with_date(predictions_src, dest_dir)
    copy_with_date(features_src, dest_dir)

    df = pd.read_csv(ranking_path)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    score_col = "final_score" if "final_score" in df.columns else "score"

    metrics = {
        "snapshot_date": today,
        "score_distribution": score_distribution(df, score_col),
        "top20_metrics": top20_metrics(df, score_col),
    }

    top20_path = dest_dir / f"top20_{today}.csv"
    df.sort_values(score_col, ascending=False).head(20).to_csv(top20_path, index=False)

    metrics_path = dest_dir / f"metrics_{today}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Snapshot saved to {dest_dir}")
    print(f"Metrics -> {metrics_path}")
    print(f"Top20   -> {top20_path}")


if __name__ == "__main__":
    main()
