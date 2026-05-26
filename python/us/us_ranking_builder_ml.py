"""US Stock ML Ranking Builder

Generates daily ML-based rankings using trained LightGBM models.
Writes to recommend.us_stock_rank_daily with source='ml_v1'.

Prerequisites:
  - Trained model at data/us_model.pkl (run us_model_train.py first)
  - Feature tables populated for the target date

Usage:
    # Today's ranking
    python -m python.us.us_ranking_builder_ml

    # Historical backfill (for backtest comparison)
    python -m python.us.us_ranking_builder_ml --start-date 2021-08-02 --end-date 2026-05-18

    # Single date
    python -m python.us.us_ranking_builder_ml --trade-date 2026-05-18
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_db import ensure_us_financial_feature_reported_date_column, get_us_engine, upsert_us_rank_rows
from python.us.us_model_predict import (
    ML_GRADE_COL,
    ML_RANK_COL,
    ML_SCORE_COL,
    PRED_RET_20D_COL,
    PRED_RET_60D_COL,
    PROB_TOP20_20D_COL,
    PROB_TOP20_60D_COL,
    grade_from_score,
    load_model,
    resolve_model_path,
    run_prediction,
    score_from_predictions,
)
from python.us.us_model_train import (
    FEATURE_DAILY_TABLE,
    FEATURE_FINANCIAL_TABLE,
    FEATURE_RS_TABLE,
    FINANCIAL_FEATURE_COLS,
    merge_financial_asof,
)

RANK_TABLE = "recommend.us_stock_rank_daily"
META_TABLE = "meta.us_stock_universe"
DEFAULT_SOURCE = "ml_v1"

DATA_DIR = Path("data")
MODEL_PKL = DATA_DIR / "us_model.pkl"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML-based US stock rankings")
    p.add_argument("--trade-date", type=str, default=None, help="Single date YYYY-MM-DD")
    p.add_argument("--start-date", type=str, default=None, help="Backfill start date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default=None, help="Backfill end date YYYY-MM-DD")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Source tag (default: ml_v1)")
    p.add_argument("--model-pkl", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Calculate without writing")
    p.add_argument("--top-n", type=int, default=20, help="Top N to show in summary")
    return p.parse_args()


def ensure_source_in_pk(engine) -> None:
    """Migrate PK to include source column if not already done."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'recommend.us_stock_rank_daily'::regclass
              AND contype = 'p'
        """))
        pk_def = result.scalar() or ""

    if "source" in pk_def:
        return  # already migrated

    logging.info("[US_ML_RANK] Migrating PK to include source column...")
    with engine.begin() as conn:
        # Ensure source has a default and is NOT NULL
        conn.execute(text("""
            UPDATE recommend.us_stock_rank_daily
            SET source = 'rule_v1'
            WHERE source IS NULL
        """))
        conn.execute(text("""
            ALTER TABLE recommend.us_stock_rank_daily
            ALTER COLUMN source SET NOT NULL
        """))
        conn.execute(text("""
            ALTER TABLE recommend.us_stock_rank_daily
            ALTER COLUMN source SET DEFAULT 'rule_v1'
        """))
        # Drop old PK and create new one with source
        conn.execute(text("""
            ALTER TABLE recommend.us_stock_rank_daily
            DROP CONSTRAINT us_stock_rank_daily_pkey
        """))
        conn.execute(text("""
            ALTER TABLE recommend.us_stock_rank_daily
            ADD CONSTRAINT us_stock_rank_daily_pkey
            PRIMARY KEY (trade_date, symbol, source)
        """))
    logging.info("[US_ML_RANK] PK migration complete: (trade_date, symbol, source)")


def fetch_universe_meta(engine) -> pd.DataFrame:
    """Fetch company metadata from universe table."""
    try:
        q = text(f"""
            SELECT symbol AS ticker, company_name, sector, industry,
                   is_etf
            FROM {META_TABLE}
            WHERE is_active = true
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn)
        return df
    except Exception as e:
        logging.warning("[US_ML_RANK] Could not fetch universe meta: %s", e)
        return pd.DataFrame()


def resolve_date_range(engine, args) -> list[date]:
    """Resolve list of trade dates to process."""
    if args.trade_date:
        return [date.fromisoformat(args.trade_date)]

    # Get available feature dates
    with engine.connect() as conn:
        if args.start_date and args.end_date:
            q = text(f"""
                SELECT DISTINCT feature_date
                FROM {FEATURE_DAILY_TABLE}
                WHERE feature_date BETWEEN :start AND :end
                ORDER BY feature_date
            """)
            result = conn.execute(q, {"start": args.start_date, "end": args.end_date})
        elif args.start_date:
            q = text(f"""
                SELECT DISTINCT feature_date
                FROM {FEATURE_DAILY_TABLE}
                WHERE feature_date >= :start
                ORDER BY feature_date
            """)
            result = conn.execute(q, {"start": args.start_date})
        else:
            # Default: today/latest
            q = text(f"SELECT MAX(feature_date) FROM {FEATURE_DAILY_TABLE}")
            result = conn.execute(q)
            max_date = result.scalar()
            return [max_date if isinstance(max_date, date) else max_date.date()] if max_date else []
        rows = result.fetchall()

    return [r[0] if isinstance(r[0], date) else r[0].date() for r in rows]


def load_all_features(engine, start_date: date, end_date: date, feature_cols: list[str]) -> pd.DataFrame:
    """Bulk load features for a date range (more efficient than per-date loading)."""
    sd = start_date.isoformat()
    ed = end_date.isoformat()

    logging.info("[US_ML_RANK] Loading daily features %s..%s", sd, ed)
    daily_q = text(f"""
        SELECT * FROM {FEATURE_DAILY_TABLE}
        WHERE feature_date BETWEEN :sd AND :ed
        ORDER BY ticker, feature_date
    """)
    with engine.connect() as conn:
        daily_df = pd.read_sql(daily_q, conn, params={"sd": sd, "ed": ed})
    daily_df["trade_date"] = pd.to_datetime(daily_df["feature_date"]).dt.date
    daily_df = daily_df.drop(columns=["feature_date"], errors="ignore")
    for col in daily_df.columns:
        if col not in {"ticker", "trade_date"} and daily_df[col].dtype == object:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
    logging.info("[US_ML_RANK] Daily features: rows=%d", len(daily_df))

    logging.info("[US_ML_RANK] Loading RS features %s..%s", sd, ed)
    rs_q = text(f"""
        SELECT * FROM {FEATURE_RS_TABLE}
        WHERE trade_date BETWEEN :sd AND :ed
        ORDER BY ticker, trade_date
    """)
    with engine.connect() as conn:
        rs_df = pd.read_sql(rs_q, conn, params={"sd": sd, "ed": ed})
    rs_df["trade_date"] = pd.to_datetime(rs_df["trade_date"]).dt.date

    if not rs_df.empty:
        rs_keep = [c for c in ["ticker", "trade_date"] + [c for c in rs_df.columns if c in feature_cols]]
        daily_df = pd.merge(daily_df, rs_df[rs_keep], on=["ticker", "trade_date"], how="left")

    logging.info("[US_ML_RANK] Loading financial features (all annual)...")
    fin_cols_available = [c for c in FINANCIAL_FEATURE_COLS if c in feature_cols]
    if fin_cols_available:
        ensure_us_financial_feature_reported_date_column()
        fin_q = text(f"""
            SELECT ticker, fiscal_date, reported_date, period_type, {", ".join(fin_cols_available)}
            FROM {FEATURE_FINANCIAL_TABLE}
            WHERE period_type = 'annual'
            ORDER BY ticker, COALESCE(reported_date, fiscal_date), fiscal_date
        """)
        with engine.connect() as conn:
            fin_df = pd.read_sql(fin_q, conn)
        fin_df["fiscal_date"] = pd.to_datetime(fin_df["fiscal_date"]).dt.date
        fin_df["reported_date"] = pd.to_datetime(fin_df["reported_date"], errors="coerce").dt.date
        reported_ratio = float(fin_df["reported_date"].notna().mean()) if not fin_df.empty else 0.0
        logging.info(
            "[US_ML_RANK] Financial feature reported_date_coverage=%.4f rows=%d",
            reported_ratio,
            len(fin_df),
        )
        if reported_ratio < 1.0:
            logging.warning(
                "[US_ML_RANK] Financial features missing reported_date rows=%d missing=%d",
                len(fin_df),
                int(fin_df["reported_date"].isna().sum()),
            )
        daily_df = merge_financial_asof(daily_df, fin_df)

    for col in feature_cols:
        if col in daily_df.columns and daily_df[col].dtype == object:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

    return daily_df


def build_rank_rows(
    df: pd.DataFrame,
    *,
    trade_date: date,
    model_pack: dict[str, Any],
    meta_df: pd.DataFrame,
    source: str,
) -> list[dict[str, Any]]:
    """Build rank rows for a single date from pre-loaded feature data."""
    feature_cols: list[str] = model_pack["features"]
    reg_models = model_pack.get("reg_models", {})
    cls_models = model_pack.get("cls_models", {})

    date_df = df[df["trade_date"] == trade_date].copy()
    if date_df.empty:
        return []

    X = pd.DataFrame(index=date_df.index)
    for col in feature_cols:
        X[col] = date_df[col].values if col in date_df.columns else np.nan

    pred_ret_20d = np.zeros(len(date_df))
    pred_ret_60d = np.zeros(len(date_df))
    prob_top20_20d = np.full(len(date_df), 0.5)
    prob_top20_60d = np.full(len(date_df), 0.5)

    if "future_ret_20d" in reg_models:
        pred_ret_20d = reg_models["future_ret_20d"].predict(X)
    if "future_ret_60d" in reg_models:
        pred_ret_60d = reg_models["future_ret_60d"].predict(X)
    if "label_top20_20d" in cls_models:
        prob_top20_20d = cls_models["label_top20_20d"].predict_proba(X)[:, 1]
    if "label_top20_60d" in cls_models:
        prob_top20_60d = cls_models["label_top20_60d"].predict_proba(X)[:, 1]

    scores = score_from_predictions(pred_ret_20d, pred_ret_60d, prob_top20_20d, prob_top20_60d)
    ranks = pd.Series(scores).rank(ascending=False, method="min").astype(int).values

    # Merge metadata
    meta_idx = {}
    if not meta_df.empty and "ticker" in meta_df.columns:
        for _, mrow in meta_df.iterrows():
            meta_idx[mrow["ticker"]] = mrow

    rows: list[dict[str, Any]] = []
    for i, (_, frow) in enumerate(date_df.iterrows()):
        ticker = frow["ticker"]
        score = float(scores[i])
        meta = meta_idx.get(ticker, {})

        score_detail = {
            "pred_ret_20d": round(float(pred_ret_20d[i]), 6),
            "pred_ret_60d": round(float(pred_ret_60d[i]), 6),
            "prob_top20_20d": round(float(prob_top20_20d[i]), 4),
            "prob_top20_60d": round(float(prob_top20_60d[i]), 4),
            "ml_score": round(score, 2),
        }

        rows.append({
            "trade_date": trade_date,
            "symbol": ticker,
            "rank_no": int(ranks[i]),
            "recommend_grade": grade_from_score(score),
            "total_score": round(score, 4),
            "momentum_score": None,
            "relative_strength_score": None,
            "fundamental_score": None,
            "growth_score": None,
            "valuation_score": None,
            "risk_score": 0.0,  # CHECK constraint: risk_score <= 0
            "feature_quality_score": None,
            "universe_group": None,
            "company_name": meta.get("company_name") if isinstance(meta, (dict, pd.Series)) else None,
            "sector": meta.get("sector") if isinstance(meta, (dict, pd.Series)) else None,
            "industry": meta.get("industry") if isinstance(meta, (dict, pd.Series)) else None,
            "market_cap": meta.get("market_cap") if isinstance(meta, (dict, pd.Series)) else None,
            "avg_volume": meta.get("avg_volume") if isinstance(meta, (dict, pd.Series)) else None,
            "is_etf": bool(meta.get("is_etf", False)) if isinstance(meta, (dict, pd.Series)) else False,
            "is_active": True,
            "data_status": "ok",
            "exclude_reason": None,
            "reason_summary": None,
            "score_detail_json": json.dumps(score_detail),
            "source": source,
        })

    return rows


def main() -> int:
    setup_logging()
    args = parse_args()

    model_path = resolve_model_path(args.model_pkl)
    model_pack = load_model(model_path)

    engine = get_us_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    # Migrate PK to include source if needed
    ensure_source_in_pk(engine)

    trade_dates = resolve_date_range(engine, args)
    if not trade_dates:
        logging.error("[US_ML_RANK] No dates to process")
        return 1

    logging.info("[US_ML_RANK] Processing %d dates: %s .. %s",
                 len(trade_dates), trade_dates[0], trade_dates[-1])

    meta_df = fetch_universe_meta(engine)

    # Bulk load for efficiency when processing multiple dates
    if len(trade_dates) > 1:
        all_features = load_all_features(
            engine,
            trade_dates[0],
            trade_dates[-1],
            model_pack["features"],
        )
    else:
        # Single date: use per-date loader
        from python.us.us_model_predict import load_features_for_date
        all_features = load_features_for_date(engine, trade_dates[0], model_pack["features"])
        if not all_features.empty and "trade_date" not in all_features.columns:
            all_features["trade_date"] = trade_dates[0]

    total_written = 0
    for td in trade_dates:
        rows = build_rank_rows(
            all_features,
            trade_date=td,
            model_pack=model_pack,
            meta_df=meta_df,
            source=args.source,
        )
        if not rows:
            logging.warning("[US_ML_RANK] No rows for %s, skipping", td)
            continue

        if not args.dry_run:
            written = upsert_us_rank_rows(rows)
            total_written += written
        else:
            written = len(rows)

        # Show top N for the latest date
        if td == trade_dates[-1]:
            top = sorted(rows, key=lambda r: r["rank_no"])[:args.top_n]
            logging.info("[US_ML_RANK] Top %d for %s (source=%s):", args.top_n, td, args.source)
            for r in top:
                logging.info("  #%2d %s  score=%.1f  grade=%-11s  p20d=%.2f  p60d=%.2f",
                             r["rank_no"], r["symbol"],
                             r["total_score"], r["recommend_grade"],
                             r["score_detail_json"] and json.loads(r["score_detail_json"]).get("prob_top20_20d", 0),
                             r["score_detail_json"] and json.loads(r["score_detail_json"]).get("prob_top20_60d", 0))

        logging.info("[US_ML_RANK] %s: wrote %d rows (dry_run=%s)", td, written, args.dry_run)

    logging.info("[US_ML_RANK] Done. total_written=%d dates=%d", total_written, len(trade_dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
