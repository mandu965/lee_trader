"""
Walk-forward prediction history + strategy backtest (Top-K, configurable horizon).

Goals:
 1) Generate prediction history per rebalance date using only past data.
 2) Evaluate a simple Top-K strategy using those predictions over holding_days.

Highlights (v5 refactor):
- CLI defaults: train_years=1.0, valid_months=3.0, rebalance=biweekly, top_k=10, holding_days=60.
- Split generation helper respects label_max_date and holding_days.
- Rebalance date helper (monthly / biweekly) centralized.
- Logs per split and overall sample-size warnings.
- Output filenames include horizon/top_k/rebalance to avoid clobbering.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA_DIR_DEFAULT = Path("data")

@dataclass
class SplitConfig:
    id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must have 'date' and 'code'")
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c not in ("date", "code"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


def load_labels(path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("labels.csv must have 'date' and 'code'")
    df["date"] = pd.to_datetime(df["date"])
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in labels.csv")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


def make_splits_from_data(
    df_all: pd.DataFrame,
    train_years: float,
    valid_months: float,
    holding_days: int,
    label_max_date: Optional[pd.Timestamp] = None,
) -> List[SplitConfig]:
    """
    Generate rolling splits based on data range.
    - train_years: length of train window in years (float)
    - valid_months: length of valid window in months (int)
    - holding_days: horizon; valid_end cannot exceed label_max_date - holding_days
    """
    if df_all.empty:
        return []
    dmin = df_all["date"].min()
    dmax = df_all["date"].max()

    if label_max_date is None:
        label_max_date = df_all["date"].max()
    else:
        label_max_date = pd.to_datetime(label_max_date)

    valid_end_max = label_max_date - pd.Timedelta(days=holding_days)
    if valid_end_max < dmin:
        return []

    splits = []
    idx = 1
    cur_train_start = dmin
    while True:
        train_start = cur_train_start
        train_end = train_start + pd.DateOffset(months=max(1, int(round(train_years * 12)))) - pd.Timedelta(days=1)
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = valid_start + pd.DateOffset(months=max(1, int(round(valid_months)))) - pd.Timedelta(days=1)

        # clamp valid_end to allowed max
        if valid_end > valid_end_max:
            valid_end = valid_end_max
        if valid_end < valid_start:
            break

        # stop if valid_end > dmax (no data)
        if valid_end > dmax:
            break

        split_id = f"s{idx}"
        splits.append(
            SplitConfig(
                id=split_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

        # slide window by valid_months (rolling)
        cur_train_start = valid_start
        idx += 1

    return splits


def generate_rebalance_dates(valid_start: pd.Timestamp, valid_end: pd.Timestamp, mode: str) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    if mode == "biweekly":
        d = valid_start
        while d <= valid_end:
            dates.append(d)
            d = d + pd.Timedelta(days=14)
    else:  # monthly (default)
        # monthly start within range
        pr = pd.period_range(valid_start, valid_end, freq="M")
        for p in pr:
            d = p.asfreq("D", "end").to_timestamp()  # month end
            if valid_start <= d <= valid_end:
                dates.append(d)
        # ensure sorted unique
        dates = sorted(set(dates))
    return dates


def train_regressor(X: pd.DataFrame, y: pd.Series, params: dict) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return model


def compute_equity_curve(rets: np.ndarray) -> np.ndarray:
    if len(rets) == 0:
        return np.array([])
    eq = np.ones(len(rets), dtype=float)
    eq[0] = 1.0 * (1.0 + rets[0])
    for i in range(1, len(rets)):
        eq[i] = eq[i - 1] * (1.0 + rets[i])
    return eq


def compute_mdd(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.where(run_max == 0, 1.0, run_max) - 1.0
    return float(dd.min())


def annualize_sharpe(rets: np.ndarray, holding_days: int, trading_days_per_year: int = 252, rf: float = 0.0) -> float:
    if rets.size == 0:
        return 0.0
    ex = rets - rf
    mu = np.nanmean(ex)
    sd = np.nanstd(ex, ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    factor = np.sqrt(trading_days_per_year / float(holding_days))
    return float(mu / sd * factor)


def run_walkforward_split(
    df_all: pd.DataFrame,
    split_id: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    feature_cols: List[str],
    target_col: str,
    top_k: int,
    holding_days: int,
    rebalance_mode: str,
    fee_pct: float,
    slippage_pct: float,
    model_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # train/valid filter
    train_df = df_all[(df_all["date"] >= train_start) & (df_all["date"] <= train_end)].copy()
    valid_df = df_all[(df_all["date"] >= valid_start) & (df_all["date"] <= valid_end)].copy()
    pred_col = f"pred_return_{holding_days}d"
    realized_col = f"realized_return_{holding_days}d"

    logging.info(
        f"[split {split_id}] "
        f"train {train_start.date()}~{train_end.date()} -> {len(train_df)} rows, {train_df['code'].nunique()} codes | "
        f"valid {valid_start.date()}~{valid_end.date()} -> {len(valid_df)} rows, {valid_df['code'].nunique()} codes"
    )
    if len(train_df) < 10_000:
        logging.warning(f"[split {split_id}] train samples < 10k; model may be unstable (possible no-split warnings).")
    if train_df.empty or valid_df.empty:
        logging.warning(f"[split {split_id}] empty train/valid after filtering; skipping split.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pred_history = []
    trades = []

    # rebalance dates
    rebalance_dates = generate_rebalance_dates(valid_start, valid_end, mode=rebalance_mode)
    logging.info(
        f"[split {split_id}] rebalances={len(rebalance_dates)}, top_k={top_k}, approx_trades~{len(rebalance_dates)*top_k}"
    )
    if not rebalance_dates:
        logging.warning(f"[split {split_id}] no rebalance dates within valid range; skipping split.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for r_date in rebalance_dates:
        # train subset: strictly before rebalance date
        train_cut = train_df[train_df["date"] < r_date]
        if len(train_cut) < 1000:
            continue
        X_train = train_cut[feature_cols]
        y_train = train_cut[target_col]
        model = train_regressor(X_train, y_train, model_params)

        # universe on rebalance date
        uni = valid_df[valid_df["date"] == r_date].copy()
        if uni.empty:
            continue
        preds = model.predict(uni[feature_cols])
        uni[pred_col] = preds
        uni["rebalance_date"] = r_date
        uni["split_id"] = split_id
        uni["model_version"] = "wf_reg_v1"

        # prediction history
        pred_history.append(uni[["split_id", "rebalance_date", "date", "code", pred_col, "model_version"]])

        # select top_k
        topk = uni.sort_values(pred_col, ascending=False).head(top_k)
        # realized return from label (already in df_all as target_col)
        trades.append(
            topk[
                [
                    "split_id",
                    "rebalance_date",
                    "date",
                    "code",
                    pred_col,
                    target_col,
                ]
            ].rename(columns={target_col: realized_col})
        )

    pred_hist_df = pd.concat(pred_history, ignore_index=True) if pred_history else pd.DataFrame()
    trades_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()

    # apply costs/slippage on realized returns
    if not trades_df.empty:
        cost_factor = 1.0 - float(fee_pct) - float(slippage_pct)
        trades_df[realized_col] = (1.0 + trades_df[realized_col]) * cost_factor - 1.0

    # summary metrics
    summary = []
    if not trades_df.empty:
        port_rets = trades_df.groupby(["split_id", "rebalance_date"])[realized_col].mean().reset_index()
        eq = compute_equity_curve(port_rets[realized_col].values)
        total_ret = float(eq[-1] - 1.0) if eq.size else 0.0
        mdd = compute_mdd(eq)
        sharpe = annualize_sharpe(port_rets[realized_col].values, holding_days=holding_days)
        win_rate = (trades_df[realized_col] > 0).mean()
        summary.append(
            {
                "split_id": split_id,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "valid_start": valid_start.date(),
                "valid_end": valid_end.date(),
                "num_trades": len(trades_df),
                "avg_trade_return": trades_df[realized_col].mean(),
                "median_trade_return": trades_df[realized_col].median(),
                "win_rate": win_rate,
                "total_return": total_ret,
                "mdd": mdd,
                "sharpe": sharpe,
            }
        )
    summary_df = pd.DataFrame(summary)
    return trades_df, summary_df, pred_hist_df


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest with prediction history.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--holding-days", type=int, default=60)
    parser.add_argument(
        "--rebalance-mode",
        type=str,
        default="biweekly",
        choices=["monthly", "biweekly"],
        help="Rebalance schedule",
    )
    parser.add_argument("--train-years", type=float, default=1.0, help="Train window length in years")
    parser.add_argument("--valid-months", type=float, default=3.0, help="Valid window length in months")
    parser.add_argument(
        "--label-max-date",
        type=str,
        default=None,
        help="Max date for labels (stop valid at label_max - holding_days). If omitted, use labels max date.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Label column to use. If omitted, auto-select target_{holding_days}d.",
    )
    parser.add_argument("--features-csv", type=Path, default=DATA_DIR_DEFAULT / "features.csv", help="Features CSV path")
    parser.add_argument("--labels-csv", type=Path, default=DATA_DIR_DEFAULT / "labels.csv", help="Labels CSV path")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR_DEFAULT, help="Output directory for backtest artifacts")
    parser.add_argument("--fee-pct", type=float, default=0.0003, help="Round-trip fee rate (e.g., 0.0003 for 0.03%)")
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.002,
        help="Total slippage rate (entry+exit). e.g., 0.002 for 0.2%",
    )
    args = parser.parse_args()

    setup_logging()
    feats = load_features(args.features_csv)
    target_col = args.target_col or f"target_{args.holding_days}d"
    labels = load_labels(args.labels_csv, target_col)

    # merge features + labels on (date, code)
    df_all = feats.merge(labels[["date", "code", target_col]], on=["date", "code"], how="inner")
    feature_cols = [c for c in feats.columns if c not in ("date", "code")]

    # auto-generate splits
    label_max_dt = pd.to_datetime(args.label_max_date) if args.label_max_date else labels["date"].max()
    splits = make_splits_from_data(
        df_all=df_all,
        train_years=args.train_years,
        valid_months=args.valid_months,
        holding_days=args.holding_days,
        label_max_date=label_max_dt,
    )
    if not splits:
        logging.warning("No splits generated; check date range or parameters.")

    model_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="regression",
    )

    all_trades = []
    all_summary = []
    all_preds = []

    for sc in splits:
        trades_df, summary_df, pred_hist_df = run_walkforward_split(
            df_all=df_all,
            split_id=sc.id,
            train_start=sc.train_start,
            train_end=sc.train_end,
            valid_start=sc.valid_start,
            valid_end=sc.valid_end,
            feature_cols=feature_cols,
            target_col=target_col,
            top_k=args.top_k,
            holding_days=args.holding_days,
            rebalance_mode=args.rebalance_mode,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
            model_params=model_params,
        )
        if not trades_df.empty:
            all_trades.append(trades_df)
        if not summary_df.empty:
            all_summary.append(summary_df)
        if not pred_hist_df.empty:
            all_preds.append(pred_hist_df)

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_out = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    preds_out = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    suffix = f"{args.holding_days}d_{args.rebalance_mode}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_dir / f"backtest_walkforward_{suffix}_top{args.top_k}_trades.csv"
    summary_path = args.out_dir / f"backtest_walkforward_{suffix}_top{args.top_k}_summary.csv"
    preds_path = args.out_dir / f"prediction_history_{suffix}.csv"

    trades_out.to_csv(trades_path, index=False, encoding="utf-8")
    summary_out.to_csv(summary_path, index=False, encoding="utf-8")
    preds_out.to_csv(preds_path, index=False, encoding="utf-8")

    logging.info(f"Saved trades to {trades_path} (rows={len(trades_out)})")
    logging.info(f"Saved summary to {summary_path} (rows={len(summary_out)})")
    logging.info(f"Saved prediction history to {preds_path} (rows of preds={len(preds_out)})")

    num_trades_total = len(trades_out)
    logging.info("Total trades=%d", num_trades_total)
    if num_trades_total < 100:
        logging.warning(f"[WARN] total trades < 100 ({num_trades_total}); low statistical confidence.")
    elif num_trades_total >= 200:
        logging.info(f"[INFO] total trades >= 200 ({num_trades_total}); more reliable sample size.")


if __name__ == "__main__":
    main()
