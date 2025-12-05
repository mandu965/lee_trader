"""
Combined 60d base + 30d auxiliary prediction backtest (fixed weights 90/10).
- Merges prediction histories from 60d and 30d horizons
- Builds combined_score (weighted normalized ranks per rebalance date)
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_scores(group: pd.DataFrame, col: str) -> pd.Series:
    # rank-based 0~1 normalization (higher is better)
    ranks = group[col].rank(method="dense", ascending=True)
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9)


def calc_equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def calc_mdd(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    eq = calc_equity_curve(returns)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calc_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    ex = returns - risk_free
    std = ex.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(ex.mean() / (std + 1e-9))


def select_topk(group: pd.DataFrame, k: int) -> pd.DataFrame:
    return group.sort_values("combined_score", ascending=False).head(k)


def load_preds(path: Path, expected_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"]).dt.normalize()
    # find prediction column
    pred_col = None
    for c in df.columns:
        if c == expected_col:
            pred_col = c
            break
    if pred_col is None:
        # fallback: choose first column that startswith 'pred'
        pred_candidates = [c for c in df.columns if c.startswith("pred")]
        if not pred_candidates:
            raise ValueError(f"Prediction column not found in {path}")
        pred_col = pred_candidates[0]
        df = df.rename(columns={pred_col: expected_col})
    return df[["split_id", "rebalance_date", "date", "code", expected_col]]


def main():
    parser = argparse.ArgumentParser(description="Combined 60d base + 30d auxiliary backtest (fixed weights 0.9/0.1)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rebalance-mode", type=str, default="biweekly", help="For filename context only")
    parser.add_argument("--pred-60d-csv", type=Path, default=Path("data/prediction_history_60d_biweekly.csv"))
    parser.add_argument("--pred-30d-csv", type=Path, default=Path("data/prediction_history_30d_biweekly.csv"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--output-prefix", type=Path, default=Path("data/backtest_walkforward_combined_90_10_top10"))
    args = parser.parse_args()

    setup_logging()

    # load predictions
    logging.info("Loading predictions 60d: %s", args.pred_60d_csv)
    pred60 = load_preds(args.pred_60d_csv, expected_col="pred_return_60d")
    logging.info("Loading predictions 30d: %s", args.pred_30d_csv)
    pred30 = load_preds(args.pred_30d_csv, expected_col="pred_return_30d")

    # merge on keys
    merged = pd.merge(
        pred60,
        pred30,
        on=["split_id", "rebalance_date", "date", "code"],
        how="inner",
        suffixes=("_60d", "_30d"),
    )
    logging.info("Merged prediction rows=%d", len(merged))
    if merged.empty:
        logging.warning("No merged predictions; exiting.")
        return

    # normalize per rebalance_date
    merged["score_60d_norm"] = (
        merged.groupby(["split_id", "rebalance_date"])
        .apply(lambda g: normalize_scores(g, "pred_return_60d"))
        .reset_index(level=[0, 1], drop=True)
    )
    merged["score_30d_norm"] = (
        merged.groupby(["split_id", "rebalance_date"])
        .apply(lambda g: normalize_scores(g, "pred_return_30d"))
        .reset_index(level=[0, 1], drop=True)
    )
    # load labels (realized 60d)
    labels_path = args.labels_csv
    if not labels_path.exists() and Path("data/labels.csv").exists():
        labels_path = Path("data/labels.csv")
    logging.info("Loading labels from %s", labels_path)
    labels = pd.read_csv(labels_path, dtype={"code": str})
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    realized_col = "realized_return_60d"
    if realized_col not in labels.columns:
        if "target_60d" in labels.columns:
            labels = labels.rename(columns={"target_60d": realized_col})
        else:
            raise ValueError("labels.csv requires realized_return_60d or target_60d.")

    merged = merged.merge(labels[["date", "code", realized_col]], on=["date", "code"], how="left")
    merged = merged.dropna(subset=[realized_col])
    logging.info("Merged with labels rows=%d", len(merged))
    if merged.empty:
        logging.warning("No rows after merging labels; exiting.")
        return

    # fixed weights
    w_base, w_aux = 0.9, 0.1
    merged["combined_score"] = w_base * merged["score_60d_norm"] + w_aux * merged["score_30d_norm"]
    topk_trades = (
        merged.groupby(["split_id", "rebalance_date"], as_index=False)
        .apply(lambda g: select_topk(g, args.top_k))
        .reset_index(drop=True)
    )
    logging.info("Weights (%.2f, %.2f) -> Top-k trades rows=%d", w_base, w_aux, len(topk_trades))
    if topk_trades.empty:
        logging.warning("No trades selected; exiting.")
        return

    rets = topk_trades[realized_col]
    summary = pd.DataFrame(
        [
            {
                "strategy": "combined_90_10",
                "W_BASE": w_base,
                "W_AUX": w_aux,
                "num_trades": len(topk_trades),
                "avg_trade_return": rets.mean(),
                "median_trade_return": rets.median(),
                "win_rate": (rets > 0).mean(),
                "total_return": (1 + rets).prod() - 1,
                "mdd": calc_mdd(rets),
                "sharpe": calc_sharpe(rets),
            }
        ]
    )

    trades_path = Path(f"{args.output_prefix}_trades.csv")
    summary_path = Path(f"{args.output_prefix}_summary.csv")
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    topk_trades.to_csv(trades_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logging.info("Saved trades -> %s (rows=%d)", trades_path, len(topk_trades))
    logging.info("Saved summary -> %s (rows=%d)", summary_path, len(summary))


if __name__ == "__main__":
    main()
