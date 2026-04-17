"""
Calibration report for prob_top20 style fields.

Steps:
 1) load predictions (research.run_id) + prices
 2) compute realized_return (horizon_days)
 3) label = 1 if realized_return is in top-N for that date (default 20)
 4) bin prob field (default 0~1, --bins)
 5) compute observed rate per bin + Brier score

Usage:
  python python/research/calibration_report.py `
    --run-id 4 `
    --start-date 2024-01-01 --end-date 2024-12-31 `
    --horizon-days 60 `
    --prob-field prob_top20_60d `
    --top-n 20 `
    --bins 10 `
    --price-source db `
    --output-dir outputs/calibration_run4
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_loader import load_predictions, load_prices
from outcomes import compute_realized_outcomes


def label_top_n(realized: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Assign label=1 to codes that are in the top-N by realized_return per date.
    """
    realized = realized.copy()
    realized["label"] = 0
    realized["rank_realized"] = (
        realized.groupby("date")["realized_return"]
        .rank(method="first", ascending=False)
    )
    realized.loc[realized["rank_realized"] <= top_n, "label"] = 1
    return realized


def brier_score(prob: pd.Series, label: pd.Series) -> float:
    mask = prob.notna() & label.notna()
    if not mask.any():
        return float("nan")
    return float(np.mean((prob[mask] - label[mask]) ** 2))


def calibration_table(df: pd.DataFrame, prob_field: str, bins: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df[prob_field].notna()]
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.cut(df[prob_field], bins=bins, include_lowest=True, labels=False)
    agg = (
        df.groupby("bin")
        .agg(
            prob_mean=(prob_field, "mean"),
            prob_min=(prob_field, "min"),
            prob_max=(prob_field, "max"),
            n=("label", "count"),
            observed_rate=("label", "mean"),
        )
        .reset_index(drop=True)
    )
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--start-date", required=True, type=str)
    ap.add_argument("--end-date", required=True, type=str)
    ap.add_argument("--horizon-days", type=int, default=60)
    ap.add_argument("--prob-field", type=str, default="prob_top20_60d")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--price-source", choices=["db", "csv"], default="db")
    ap.add_argument("--prices-csv", type=Path, default=Path("data/prices_daily_adjusted.csv"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = BacktestConfig(
        start_date=pd.to_datetime(args.start_date).date(),
        end_date=pd.to_datetime(args.end_date).date(),
        model_versions=["any"],  # load_predictions filters by list; set wildcard later
        run_id=args.run_id,
        horizon_days=args.horizon_days,
        top_n=args.top_n,
        price_source=args.price_source,
        prediction_source="research",
        prices_csv=str(args.prices_csv),
    )

    # Load preds/prices and compute realized_return
    preds = load_predictions(cfg)
    # override model_versions filter to allow all loaded versions
    if preds.empty:
        raise SystemExit("No predictions loaded for given run/window.")
    cfg.model_versions = preds["model_version"].unique().tolist()
    preds = preds[preds["model_version"].isin(cfg.model_versions)].copy()

    prices = load_prices(cfg)
    realized = compute_realized_outcomes(preds, prices, cfg)
    realized = label_top_n(realized, cfg.top_n)

    # Per model_version calibration
    all_tables = []
    summaries = []
    for mv, grp in realized.groupby("model_version"):
        table = calibration_table(grp, args.prob_field, args.bins)
        table.insert(0, "model_version", mv)
        table_path = args.output_dir / f"calibration_{mv}.csv"
        table.to_csv(table_path, index=False)
        all_tables.append(table)

        bs = brier_score(grp[args.prob_field], grp["label"])
        summaries.append({
            "model_version": mv,
            "brier_score": bs,
            "n": int(grp.shape[0]),
        })

    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(args.output_dir / "calibration_all.csv", index=False)
    summary_path = args.output_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[calibration] saved tables to {args.output_dir}")
    print(f"[calibration] summary -> {summary_path}")


if __name__ == "__main__":
    main()
