"""US Stock ML Model Training (LightGBM)

Loads features from DB, merges with labels, trains LightGBM models.

Regression targets:  future_ret_20d, future_ret_60d
Classification targets: label_top20_20d, label_top20_60d

Outputs: data/us_model.pkl
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier, LGBMRegressor
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_db import ensure_us_financial_feature_reported_date_column, get_us_engine

DATA_DIR = Path("data")
MODEL_PKL = DATA_DIR / "us_model.pkl"

N_SPLITS = 3

FEATURE_DAILY_TABLE = "feature.us_stock_feature_daily"
FEATURE_RS_TABLE = "feature.us_stock_relative_strength_daily"
FEATURE_FINANCIAL_TABLE = "feature.us_stock_financial_feature"
LABEL_TABLE = "label.us_stock_label_daily"

# Financial features to include (use annual period for stability)
FINANCIAL_FEATURE_COLS = [
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "eps_growth_yoy",
    "free_cash_flow_growth_yoy",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "roa",
    "debt_to_equity",
    "current_ratio",
    "per",
    "pbr",
    "psr",
    "ev_ebitda",
    "dividend_yield",
    "financial_quality_score",
    "financial_growth_score",
    "financial_value_score",
    "fcf_yield",
    "gross_margin_trend",
    "revenue_growth_accel",
    "roic_approx",
    "peg_ratio",
]

DAILY_FEATURE_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_252d",
    "ma_20",
    "ma_60",
    "ma_200",
    "close_over_ma20",
    "price_vs_ma200",
    "volatility_20d",
    "dollar_volume_20",
    "volume_avg_20d",
    "volume_ratio_20d",
    "rsi_14",
    "atr_14_norm",
    "bb_position",
    "high_52w_ratio",
    "price_above_ma20_flag",
    "price_above_ma60_flag",
    "sector_rel_ret_20d",
    "sector_rel_ret_60d",
    "sector_rank_pct",
]

RS_FEATURE_COLS = [
    "rs_spy_5d",
    "rs_spy_20d",
    "rs_spy_60d",
    "rs_spy_120d",
    "rs_spy_252d",
    "rs_qqq_5d",
    "rs_qqq_20d",
    "rs_qqq_60d",
    "rs_qqq_120d",
    "rs_qqq_252d",
    "rs_spy_20d_rank_pct",
    "rs_spy_60d_rank_pct",
    "rs_qqq_20d_rank_pct",
    "rs_qqq_60d_rank_pct",
]

REG_TARGETS = ["future_ret_20d", "future_ret_60d"]
CLS_TARGETS = ["label_top20_20d", "label_top20_60d"]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train US stock LightGBM models")
    p.add_argument("--train-end-date", type=str, default=None, help="Use rows up to this date (YYYY-MM-DD)")
    p.add_argument("--train-start-date", type=str, default=None, help="Use rows from this date (YYYY-MM-DD)")
    p.add_argument("--output-pkl", type=Path, default=MODEL_PKL)
    p.add_argument("--model-version", type=str, default=os.environ.get("US_MODEL_VERSION", "v1"))
    p.add_argument("--dry-run", action="store_true", help="Load and validate data only, skip training")
    return p.parse_args()


def load_daily_features(engine) -> pd.DataFrame:
    logging.info("[US_TRAIN] Loading daily features from %s", FEATURE_DAILY_TABLE)
    cols = ", ".join(f'"{c}"' for c in ["feature_date", "ticker"] + DAILY_FEATURE_COLS if True)
    query = f"SELECT {cols} FROM {FEATURE_DAILY_TABLE}"
    df = pd.read_sql(text(query), engine)
    df["trade_date"] = pd.to_datetime(df["feature_date"]).dt.date
    df = df.drop(columns=["feature_date"])
    # Cast any object columns that should be numeric
    for col in df.columns:
        if col not in {"ticker", "trade_date"} and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logging.info("[US_TRAIN] daily features rows=%d tickers=%d", len(df), df["ticker"].nunique())
    return df


def load_rs_features(engine) -> pd.DataFrame:
    logging.info("[US_TRAIN] Loading RS features from %s", FEATURE_RS_TABLE)
    cols = ", ".join(f'"{c}"' for c in ["trade_date", "ticker"] + RS_FEATURE_COLS)
    query = f"SELECT {cols} FROM {FEATURE_RS_TABLE}"
    df = pd.read_sql(text(query), engine)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    logging.info("[US_TRAIN] RS features rows=%d tickers=%d", len(df), df["ticker"].nunique())
    return df


def load_financial_features(engine) -> pd.DataFrame:
    """Load annual financial features for reported-date-aware as-of joins."""
    logging.info("[US_TRAIN] Loading financial features from %s", FEATURE_FINANCIAL_TABLE)
    ensure_us_financial_feature_reported_date_column()
    available_cols = []
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='feature' AND table_name='us_stock_financial_feature'
            """)
        )
        db_cols = {row[0] for row in result}
    available_cols = [c for c in FINANCIAL_FEATURE_COLS if c in db_cols]
    if not available_cols:
        logging.warning("[US_TRAIN] No financial feature columns found, skipping financial features")
        return pd.DataFrame()

    cols = ", ".join(f'"{c}"' for c in ["ticker", "fiscal_date", "reported_date", "period_type"] + available_cols)
    query = f"SELECT {cols} FROM {FEATURE_FINANCIAL_TABLE} WHERE period_type = 'annual'"
    df = pd.read_sql(text(query), engine)
    if df.empty:
        logging.warning("[US_TRAIN] No annual financial features found")
        return df
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"]).dt.date
    df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce").dt.date
    df = df.sort_values(["ticker", "fiscal_date"])
    reported_ratio = 0.0 if df.empty else float(df["reported_date"].notna().mean())
    logging.info(
        "[US_TRAIN] financial features rows=%d tickers=%d reported_date_coverage=%.4f",
        len(df),
        df["ticker"].nunique(),
        reported_ratio,
    )
    if reported_ratio < 1.0:
        logging.warning(
            "[US_TRAIN] financial features have missing reported_date rows=%d missing=%d",
            len(df),
            int(df["reported_date"].isna().sum()),
        )
    return df


def merge_financial_asof(daily_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
    """Left join financial features by ticker using reported_date-aware as-of merge."""
    if fin_df.empty:
        logging.info("[US_TRAIN] Skipping financial feature merge (empty)")
        return daily_df

    fin_cols = [c for c in fin_df.columns if c not in {"ticker", "fiscal_date", "reported_date", "period_type"}]
    fin_narrow = fin_df[["ticker", "fiscal_date", "reported_date"] + fin_cols].copy()
    fin_narrow["effective_date"] = fin_narrow["reported_date"].where(fin_narrow["reported_date"].notna(), fin_narrow["fiscal_date"])
    fin_narrow = fin_narrow[fin_narrow["effective_date"].notna()].copy()
    if fin_narrow.empty:
        logging.info("[US_TRAIN] Skipping financial feature merge (no effective financial dates)")
        return daily_df

    # merge_asof requires datetime for the keys
    daily_sorted = daily_df.copy()
    daily_sorted["_td"] = pd.to_datetime(daily_sorted["trade_date"])
    daily_sorted = daily_sorted.sort_values(["ticker", "_td"])

    fin_sorted = fin_narrow.copy()
    fin_sorted["_fd"] = pd.to_datetime(fin_sorted["effective_date"])
    fin_sorted = fin_sorted.sort_values(["ticker", "_fd", "fiscal_date"])

    out = pd.merge_asof(
        daily_sorted,
        fin_sorted.rename(columns={"_fd": "_td"})[["ticker", "_td"] + fin_cols],
        on="_td",
        by="ticker",
        direction="backward",
    )
    out = out.drop(columns=["_td"])
    logging.info("[US_TRAIN] After financial as-of merge: rows=%d", len(out))
    return out


def merge_financial_locf(daily_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for reported-date-aware financial merge."""
    return merge_financial_asof(daily_df, fin_df)


def load_labels(engine) -> pd.DataFrame:
    logging.info("[US_TRAIN] Loading labels from %s", LABEL_TABLE)
    cols = ["ticker", "trade_date"] + REG_TARGETS + CLS_TARGETS
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    query = f"SELECT {cols_sql} FROM {LABEL_TABLE}"
    df = pd.read_sql(text(query), engine)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    logging.info("[US_TRAIN] labels rows=%d tickers=%d", len(df), df["ticker"].nunique())
    return df


def build_merged_dataset(
    daily_df: pd.DataFrame,
    rs_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    logging.info("[US_TRAIN] Merging daily + RS features...")
    if rs_df.empty:
        merged = daily_df
    else:
        merged = pd.merge(daily_df, rs_df, on=["ticker", "trade_date"], how="left")
    logging.info("[US_TRAIN] After RS merge: rows=%d", len(merged))

    logging.info("[US_TRAIN] Merging financial features (as-of)...")
    merged = merge_financial_asof(merged, fin_df)

    logging.info("[US_TRAIN] Merging labels...")
    merged = pd.merge(merged, label_df, on=["ticker", "trade_date"], how="inner")
    logging.info("[US_TRAIN] After label merge: rows=%d", len(merged))

    exclude = {"ticker", "trade_date"}
    exclude.update(REG_TARGETS)
    exclude.update(CLS_TARGETS)
    feature_cols = [
        c for c in merged.columns
        if c not in exclude
        and not c.startswith("future_ret_")
        and not c.startswith("label_")
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    non_numeric = [
        c for c in merged.columns
        if c not in exclude and not pd.api.types.is_numeric_dtype(merged[c])
    ]
    if non_numeric:
        logging.info("[US_TRAIN] Excluded non-numeric cols: %s", non_numeric)

    logging.info("[US_TRAIN] Feature columns (%d): %s", len(feature_cols), feature_cols)
    return merged, feature_cols


def apply_date_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        cutoff = pd.to_datetime(start).date()
        df = df[df["trade_date"] >= cutoff]
        logging.info("[US_TRAIN] Applied start_date=%s rows=%d", start, len(df))
    if end:
        cutoff = pd.to_datetime(end).date()
        df = df[df["trade_date"] <= cutoff]
        logging.info("[US_TRAIN] Applied end_date=%s rows=%d", end, len(df))
    return df


def time_series_folds(dates: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.array(sorted(pd.Series(dates).dropna().unique()))
    if uniq.size < 20:
        return []
    eff = min(n_splits, max(2, uniq.size - 1))
    tscv = TimeSeriesSplit(n_splits=eff)
    folds = []
    for tr_idx, va_idx in tscv.split(uniq):
        folds.append((uniq[tr_idx], uniq[va_idx]))
    return folds


def train_regressors(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, LGBMRegressor]:
    models: dict[str, LGBMRegressor] = {}
    for target in REG_TARGETS:
        if target not in df.columns:
            logging.warning("[US_TRAIN] Regression target %s not found, skipping", target)
            continue
        df_t = df[df[target].notna()].copy()
        if df_t.empty:
            logging.warning("[US_TRAIN] No rows for %s, skipping", target)
            continue

        X = df_t[feature_cols]
        y = df_t[target].astype(float)
        dates = df_t["trade_date"].values

        logging.info("[US_TRAIN] Training regressor for %s (rows=%d)", target, len(df_t))
        reg = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=42,
            n_jobs=-1,
        )
        folds = time_series_folds(dates, N_SPLITS)
        if folds:
            rmses, maes = [], []
            for i, (tr_dates, va_dates) in enumerate(folds, 1):
                tr_mask = np.isin(dates, tr_dates)
                va_mask = np.isin(dates, va_dates)
                if tr_mask.sum() == 0 or va_mask.sum() == 0:
                    continue
                reg_cv = LGBMRegressor(**reg.get_params())
                reg_cv.fit(X[tr_mask], y[tr_mask])
                pred = reg_cv.predict(X[va_mask])
                rmse = float(np.sqrt(mean_squared_error(y[va_mask], pred)))
                mae_val = float(mean_absolute_error(y[va_mask], pred))
                rmses.append(rmse)
                maes.append(mae_val)
                logging.info("[US_TRAIN]   [%s][fold %d/%d] RMSE=%.4f MAE=%.4f", target, i, len(folds), rmse, mae_val)
            if rmses:
                logging.info("[US_TRAIN]   [%s] CV RMSE=%.4f±%.4f MAE=%.4f±%.4f",
                             target, np.mean(rmses), np.std(rmses), np.mean(maes), np.std(maes))
        reg.fit(X, y)
        models[target] = reg
        logging.info("[US_TRAIN] Regressor %s trained", target)
    return models


def train_classifiers(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, LGBMClassifier]:
    models: dict[str, LGBMClassifier] = {}
    for target in CLS_TARGETS:
        if target not in df.columns:
            logging.warning("[US_TRAIN] Classification target %s not found, skipping", target)
            continue
        df_t = df[df[target].notna()].copy()
        if df_t.empty:
            logging.warning("[US_TRAIN] No rows for %s, skipping", target)
            continue

        X = df_t[feature_cols]
        y = df_t[target].astype(int)
        dates = df_t["trade_date"].values
        pos_ratio = float(y.mean())
        logging.info("[US_TRAIN] Training classifier for %s (rows=%d, positive_rate=%.3f)", target, len(df_t), pos_ratio)

        cls = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            n_jobs=-1,
        )
        folds = time_series_folds(dates, N_SPLITS)
        if folds:
            aucs, losses = [], []
            for i, (tr_dates, va_dates) in enumerate(folds, 1):
                tr_mask = np.isin(dates, tr_dates)
                va_mask = np.isin(dates, va_dates)
                if tr_mask.sum() == 0 or va_mask.sum() == 0:
                    continue
                cls_cv = LGBMClassifier(**cls.get_params())
                cls_cv.fit(X[tr_mask], y[tr_mask])
                proba = cls_cv.predict_proba(X[va_mask])[:, 1]
                auc = float(roc_auc_score(y[va_mask], proba))
                loss = float(log_loss(y[va_mask], proba, labels=[0, 1]))
                aucs.append(auc)
                losses.append(loss)
                logging.info("[US_TRAIN]   [%s][fold %d/%d] AUC=%.4f logloss=%.4f", target, i, len(folds), auc, loss)
            if aucs:
                logging.info("[US_TRAIN]   [%s] CV AUC=%.4f±%.4f logloss=%.4f±%.4f",
                             target, np.mean(aucs), np.std(aucs), np.mean(losses), np.std(losses))
        cls.fit(X, y)
        models[target] = cls
        logging.info("[US_TRAIN] Classifier %s trained", target)
    return models


def main() -> int:
    setup_logging()
    args = parse_args()

    engine = get_us_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logging.info("[US_TRAIN] DB connected")

    daily_df = load_daily_features(engine)
    rs_df = load_rs_features(engine)
    fin_df = load_financial_features(engine)
    label_df = load_labels(engine)

    merged, feature_cols = build_merged_dataset(daily_df, rs_df, fin_df, label_df)
    merged = apply_date_filter(merged, args.train_start_date, args.train_end_date)

    if merged.empty:
        logging.error("[US_TRAIN] No training data after merge/filter")
        return 1

    logging.info("[US_TRAIN] Final training set: rows=%d tickers=%d date_range=%s..%s",
                 len(merged), merged["ticker"].nunique(),
                 merged["trade_date"].min(), merged["trade_date"].max())

    if args.dry_run:
        logging.info("[US_TRAIN] Dry run complete. Skipping model training.")
        return 0

    logging.info("[US_TRAIN] Training regressors...")
    reg_models = train_regressors(merged, feature_cols)

    logging.info("[US_TRAIN] Training classifiers...")
    cls_models = train_classifiers(merged, feature_cols)

    train_end_date = str(merged["trade_date"].max()) if args.train_end_date is None else args.train_end_date
    pack: dict[str, Any] = {
        "features": feature_cols,
        "reg_models": reg_models,
        "cls_models": cls_models,
        "reg_targets": list(reg_models.keys()),
        "cls_targets": list(cls_models.keys()),
        "model_version": args.model_version,
        "train_end_date": train_end_date,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market": "US",
    }

    out_path = args.output_pkl
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[2] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pack, f)

    logging.info("[US_TRAIN] Model saved to %s (version=%s, reg=%s, cls=%s)",
                 out_path, args.model_version,
                 list(reg_models.keys()), list(cls_models.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
