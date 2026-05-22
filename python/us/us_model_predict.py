"""US Stock ML Model Prediction

Loads the latest features for the given date, runs inference using the trained model,
and upserts predictions to recommend.us_stock_rank_daily.

Usage:
    python -m python.us.us_model_predict [--trade-date YYYY-MM-DD]

If --trade-date is omitted, uses the latest available feature date.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_db import ensure_us_financial_feature_reported_date_column, get_us_engine
from python.us.us_model_train import (
    FEATURE_DAILY_TABLE,
    FEATURE_FINANCIAL_TABLE,
    FEATURE_RS_TABLE,
    FINANCIAL_FEATURE_COLS,
    merge_financial_asof,
)

DATA_DIR = Path("data")
MODEL_PKL = DATA_DIR / "us_model.pkl"

RANK_TABLE = "recommend.us_stock_rank_daily"

# Prediction column names in the rank table
PRED_RET_20D_COL = "pred_ret_20d"
PRED_RET_60D_COL = "pred_ret_60d"
PROB_TOP20_20D_COL = "prob_top20_20d"
PROB_TOP20_60D_COL = "prob_top20_60d"
ML_SCORE_COL = "ml_score"
ML_RANK_COL = "ml_rank_no"
ML_GRADE_COL = "ml_grade"

ML_STRONG_BUY_SCORE = 80.0
ML_BUY_SCORE = 70.0
ML_WATCH_SCORE = 60.0


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US stock ML prediction for a single trade date")
    p.add_argument("--trade-date", type=str, default=None, help="Trade date YYYY-MM-DD (default: latest feature date)")
    p.add_argument("--model-pkl", type=Path, default=None)
    p.add_argument("--source", type=str, default="ml_v1", help="Source label for rank table")
    return p.parse_args()


def resolve_model_path(override: Path | None) -> Path:
    if override:
        return override if override.is_absolute() else Path(__file__).resolve().parents[2] / override
    root = Path(__file__).resolve().parents[2]
    return root / MODEL_PKL


def load_model(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        pack = pickle.load(f)
    if not isinstance(pack, dict) or "features" not in pack:
        raise ValueError("Invalid model pack structure")
    logging.info("[US_PREDICT] Loaded model version=%s trained_at=%s features=%d",
                 pack.get("model_version"), pack.get("trained_at"), len(pack["features"]))
    return pack


def resolve_trade_date(engine, cli_date: str | None) -> date:
    if cli_date:
        return date.fromisoformat(cli_date)
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT MAX(feature_date) FROM {FEATURE_DAILY_TABLE}"))
        max_date = result.scalar()
    if max_date is None:
        raise RuntimeError(f"No data in {FEATURE_DAILY_TABLE}")
    return max_date if isinstance(max_date, date) else max_date.date()


def load_features_for_date(engine, trade_date: date, feature_cols: list[str]) -> pd.DataFrame:
    """Load all feature rows for a single trade_date."""
    td_str = trade_date.isoformat()

    # Daily features
    daily_q = text(f"""
        SELECT * FROM {FEATURE_DAILY_TABLE}
        WHERE feature_date = :td
    """)
    with engine.connect() as conn:
        daily_df = pd.read_sql(daily_q, conn, params={"td": td_str})
    if daily_df.empty:
        logging.warning("[US_PREDICT] No daily features for %s", td_str)
        return pd.DataFrame()
    daily_df["trade_date"] = pd.to_datetime(daily_df["feature_date"]).dt.date
    daily_df = daily_df.drop(columns=["feature_date"], errors="ignore")
    for col in daily_df.columns:
        if col not in {"ticker", "trade_date"} and daily_df[col].dtype == object:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

    # RS features
    rs_q = text(f"""
        SELECT * FROM {FEATURE_RS_TABLE}
        WHERE trade_date = :td
    """)
    with engine.connect() as conn:
        rs_df = pd.read_sql(rs_q, conn, params={"td": td_str})
    if not rs_df.empty:
        rs_df["trade_date"] = pd.to_datetime(rs_df["trade_date"]).dt.date
        daily_df = pd.merge(daily_df, rs_df[["ticker", "trade_date"] + [
            c for c in rs_df.columns if c in feature_cols
        ]], on=["ticker", "trade_date"], how="left")

    # Financial features (last available before trade_date)
    fin_cols_available = [c for c in FINANCIAL_FEATURE_COLS if c in feature_cols]
    if fin_cols_available:
        ensure_us_financial_feature_reported_date_column()
        fin_q = text(f"""
            SELECT ticker, fiscal_date, reported_date, period_type, {", ".join(fin_cols_available)}
            FROM {FEATURE_FINANCIAL_TABLE}
            WHERE period_type = 'annual' AND COALESCE(reported_date, fiscal_date) <= :td
        """)
        with engine.connect() as conn:
            fin_df = pd.read_sql(fin_q, conn, params={"td": td_str})
        if not fin_df.empty:
            fin_df["fiscal_date"] = pd.to_datetime(fin_df["fiscal_date"]).dt.date
            fin_df["reported_date"] = pd.to_datetime(fin_df["reported_date"], errors="coerce").dt.date
            reported_ratio = float(fin_df["reported_date"].notna().mean())
            logging.info(
                "[US_PREDICT] financial feature coverage trade_date=%s reported_date_coverage=%.4f rows=%d",
                td_str,
                reported_ratio,
                len(fin_df),
            )
            if reported_ratio < 1.0:
                logging.warning(
                    "[US_PREDICT] financial features missing reported_date trade_date=%s missing=%d",
                    td_str,
                    int(fin_df["reported_date"].isna().sum()),
                )
            daily_df = merge_financial_asof(daily_df, fin_df)

    logging.info("[US_PREDICT] Feature rows for %s: %d tickers", td_str, len(daily_df))
    return daily_df


def score_from_predictions(
    pred_ret_20d: np.ndarray,
    pred_ret_60d: np.ndarray,
    prob_top20_20d: np.ndarray,
    prob_top20_60d: np.ndarray,
) -> np.ndarray:
    """
    Composite ML score 0-100:
      40% prob_top20_20d  (20d classification)
      40% prob_top20_60d  (60d classification)
      10% ret_20d percentile
      10% ret_60d percentile
    """
    def to_pct(arr: np.ndarray) -> np.ndarray:
        ranks = pd.Series(arr).rank(pct=True, method="average").values
        return ranks * 100.0

    score = (
        0.40 * prob_top20_20d * 100.0
        + 0.40 * prob_top20_60d * 100.0
        + 0.10 * to_pct(pred_ret_20d)
        + 0.10 * to_pct(pred_ret_60d)
    )
    return score


def grade_from_score(score: float) -> str:
    if score >= ML_STRONG_BUY_SCORE:
        return "STRONG_BUY"
    if score >= ML_BUY_SCORE:
        return "BUY"
    if score >= ML_WATCH_SCORE:
        return "WATCH"
    return "HOLD"


def ensure_ml_columns(engine) -> None:
    """Add ML prediction columns to rank table if missing."""
    new_cols = {
        PRED_RET_20D_COL: "NUMERIC",
        PRED_RET_60D_COL: "NUMERIC",
        PROB_TOP20_20D_COL: "NUMERIC",
        PROB_TOP20_60D_COL: "NUMERIC",
        ML_SCORE_COL: "NUMERIC",
        ML_RANK_COL: "INTEGER",
        ML_GRADE_COL: "TEXT",
    }
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'recommend' AND table_name = 'us_stock_rank_daily'
        """))
        existing = {row[0] for row in result}

    missing = {col: dtype for col, dtype in new_cols.items() if col not in existing}
    if not missing:
        return
    with engine.begin() as conn:
        for col, dtype in missing.items():
            conn.execute(text(f"ALTER TABLE {RANK_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}"))
    logging.info("[US_PREDICT] Added columns to %s: %s", RANK_TABLE, list(missing.keys()))


def upsert_predictions(engine, rows: list[dict], source: str) -> int:
    if not rows:
        return 0
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        normalized_rows.append(
            {
                "trade_date": row["trade_date"],
                "symbol": row["symbol"],
                "rank_no": row[ML_RANK_COL],
                "recommend_grade": row[ML_GRADE_COL],
                "total_score": row[ML_SCORE_COL],
                "momentum_score": None,
                "relative_strength_score": None,
                "fundamental_score": None,
                "growth_score": None,
                "valuation_score": None,
                "risk_score": 0.0,
                "feature_quality_score": None,
                "universe_group": None,
                "company_name": None,
                "sector": None,
                "industry": None,
                "market_cap": None,
                "avg_volume": None,
                "is_etf": False,
                "is_active": True,
                "data_status": "OK",
                "exclude_reason": None,
                "reason_summary": "ml_prediction_only",
                "score_detail_json": None,
                "source": source,
                PRED_RET_20D_COL: row[PRED_RET_20D_COL],
                PRED_RET_60D_COL: row[PRED_RET_60D_COL],
                PROB_TOP20_20D_COL: row[PROB_TOP20_20D_COL],
                PROB_TOP20_60D_COL: row[PROB_TOP20_60D_COL],
                ML_SCORE_COL: row[ML_SCORE_COL],
                ML_RANK_COL: row[ML_RANK_COL],
                ML_GRADE_COL: row[ML_GRADE_COL],
            }
        )

    cols = list(normalized_rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_names = ", ".join(cols)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in cols
        if c not in {"trade_date", "symbol", "source"}
    )
    sql = text(f"""
        INSERT INTO {RANK_TABLE} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (trade_date, symbol, source) DO UPDATE SET
            {update_set},
            updated_at = now()
    """)
    with engine.begin() as conn:
        conn.execute(sql, normalized_rows)
    return len(normalized_rows)


def run_prediction(
    engine,
    model_pack: dict[str, Any],
    trade_date: date,
    source: str,
) -> int:
    feature_cols: list[str] = model_pack["features"]
    reg_models: dict = model_pack.get("reg_models", {})
    cls_models: dict = model_pack.get("cls_models", {})

    df = load_features_for_date(engine, trade_date, feature_cols)
    if df.empty:
        logging.warning("[US_PREDICT] No feature data for %s, skipping", trade_date)
        return 0

    # Align feature columns: fill missing with NaN
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = df[col] if col in df.columns else np.nan

    # Regression predictions
    pred_ret_20d = np.zeros(len(df))
    pred_ret_60d = np.zeros(len(df))
    if "future_ret_20d" in reg_models:
        pred_ret_20d = reg_models["future_ret_20d"].predict(X)
    if "future_ret_60d" in reg_models:
        pred_ret_60d = reg_models["future_ret_60d"].predict(X)

    # Classification predictions
    prob_top20_20d = np.full(len(df), 0.5)
    prob_top20_60d = np.full(len(df), 0.5)
    if "label_top20_20d" in cls_models:
        prob_top20_20d = cls_models["label_top20_20d"].predict_proba(X)[:, 1]
    if "label_top20_60d" in cls_models:
        prob_top20_60d = cls_models["label_top20_60d"].predict_proba(X)[:, 1]

    scores = score_from_predictions(pred_ret_20d, pred_ret_60d, prob_top20_20d, prob_top20_60d)
    ranks = pd.Series(scores).rank(ascending=False, method="min").astype(int).values

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        score = float(scores[i])
        rows.append({
            "symbol": row.ticker,
            "trade_date": trade_date,
            "source": source,
            PRED_RET_20D_COL: float(pred_ret_20d[i]),
            PRED_RET_60D_COL: float(pred_ret_60d[i]),
            PROB_TOP20_20D_COL: float(prob_top20_20d[i]),
            PROB_TOP20_60D_COL: float(prob_top20_60d[i]),
            ML_SCORE_COL: round(score, 2),
            ML_RANK_COL: int(ranks[i]),
            ML_GRADE_COL: grade_from_score(score),
        })

    ensure_ml_columns(engine)
    upserted = upsert_predictions(engine, rows, source)
    logging.info("[US_PREDICT] Upserted %d predictions for %s (source=%s)", upserted, trade_date, source)
    return upserted


def main() -> int:
    setup_logging()
    args = parse_args()

    model_path = resolve_model_path(args.model_pkl)
    model_pack = load_model(model_path)

    engine = get_us_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    trade_date = resolve_trade_date(engine, args.trade_date)
    logging.info("[US_PREDICT] Running prediction for trade_date=%s source=%s", trade_date, args.source)

    count = run_prediction(engine, model_pack, trade_date, args.source)
    logging.info("[US_PREDICT] Done. rows=%d", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
