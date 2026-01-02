"""
build_backtest_outcome.py

prediction_history(run_id) + labels를 join하여 realized_return / realized_mdd를
research.backtest_outcome에 적재한다.

기본 로직:
  - run_id (필수), horizon_days는 인자 > dim_model_run > env(HORIZON_DAYS) 순으로 결정
  - prediction_history(run_id, optional date range, optional horizon filter) 로드
  - labels에서 realized_return_{h}d, target_mdd_{h}d를 가져와 (as_of_date=date, code) 매칭
  - realized_return NaN이면 제외
  - label_source='from_labels'
  - 기존 run_id(+날짜범위) backtest_outcome 삭제 후 insert
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from db import get_engine
except Exception:
    get_engine = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill backtest_outcome from labels")
    p.add_argument("--run-id", type=int, required=True, help="run_id to backfill outcome")
    p.add_argument("--horizon-days", type=int, help="override horizon_days (default: dim_model_run or env)")
    p.add_argument("--start-date", type=str, help="inclusive start as_of_date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, help="inclusive end as_of_date (YYYY-MM-DD)")
    p.add_argument("--out-csv", type=Path, help="optional: save generated outcome to CSV")
    p.add_argument("--log-interval", type=int, default=20, help="log every N dates")
    return p.parse_args()


def resolve_horizon(run_id: int, override: Optional[int]) -> int:
    if override:
        return override
    if not get_engine:
        return int(os.environ.get("HORIZON_DAYS", "60"))
    try:
        eng = get_engine()
        with eng.connect() as conn:
            res = conn.execute(
                text("SELECT horizon_days FROM research.dim_model_run WHERE run_id = :rid"),
                {"rid": run_id},
            ).scalar()
            if res:
                return int(res)
    except Exception:
        logging.warning("horizon_days lookup failed; fallback to env")
    return int(os.environ.get("HORIZON_DAYS", "60"))


def load_predictions(run_id: int, start: Optional[str], end: Optional[str], horizon: int) -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("No DB engine available")
    eng = get_engine()
    clauses = [f"run_id = {run_id}"]
    if start:
        clauses.append(f"as_of_date >= '{start}'")
    if end:
        clauses.append(f"as_of_date <= '{end}'")
    clauses.append(f"horizon_days = {horizon}")
    where_clause = " AND ".join(clauses)
    query = (
        "SELECT run_id, as_of_date, code, horizon_days "
        f"FROM research.prediction_history WHERE {where_clause}"
    )
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    return df


def load_labels(horizon: int) -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("No DB engine available")
    eng = get_engine()
    ret_col = f"realized_return_{horizon}d"
    mdd_col = f"target_mdd_{horizon}d"  # label_builder naming
    query = f"SELECT date, code, {ret_col} AS realized_return, {mdd_col} AS realized_mdd FROM labels"
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["date"])
    return df


def build_outcome(preds: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if preds.empty or labels.empty:
        return pd.DataFrame()
    latest_label_date = labels["date"].max()
    preds = preds[preds["as_of_date"] <= latest_label_date].copy()
    if preds.empty:
        return pd.DataFrame()
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"])
    merged = preds.merge(
        labels,
        left_on=["as_of_date", "code"],
        right_on=["date", "code"],
        how="left",
    )
    merged = merged.drop(columns=["date"])
    merged["horizon_days"] = horizon
    merged["label_source"] = "from_labels"
    merged = merged.dropna(subset=["realized_return"])
    return merged[["run_id", "as_of_date", "code", "horizon_days", "realized_return", "realized_mdd", "label_source"]]


def save_outcome(run_id: int, df_out: pd.DataFrame, start: Optional[str], end: Optional[str]) -> None:
    if not get_engine:
        logging.info("No DB engine available -> skip DB save")
        return
    eng = get_engine()
    try:
        with eng.begin() as conn:
            if start or end:
                where = ["run_id = :rid"]
                params = {"rid": run_id}
                if start:
                    where.append("as_of_date >= :start")
                    params["start"] = start
                if end:
                    where.append("as_of_date <= :end")
                    params["end"] = end
                conn.execute(text(f"DELETE FROM research.backtest_outcome WHERE {' AND '.join(where)}"), params)
            else:
                conn.execute(text("DELETE FROM research.backtest_outcome WHERE run_id = :rid"), {"rid": run_id})
        df_out.to_sql("backtest_outcome", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Saved backtest_outcome (run_id=%s, rows=%d)", run_id, len(df_out))
    except Exception:
        logging.exception("Failed to save backtest_outcome")


def main() -> None:
    setup_logging()
    args = parse_args()
    horizon = resolve_horizon(args.run_id, args.horizon_days)
    preds = load_predictions(args.run_id, args.start_date, args.end_date, horizon)
    if preds.empty:
        logging.warning("No prediction_history rows for run_id=%s", args.run_id)
        return
    labels = load_labels(horizon)
    outcome = build_outcome(preds, labels, horizon)
    if outcome.empty:
        logging.warning("No outcome rows after join (run_id=%s)", args.run_id)
        return
    if args.out_csv:
        outcome.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved outcome CSV: %s (rows=%d)", args.out_csv, len(outcome))
    save_outcome(args.run_id, outcome, args.start_date, args.end_date)
    logging.info(
        "Outcome summary: rows=%d, min_as_of=%s, max_as_of=%s, horizon=%d",
        len(outcome),
        outcome["as_of_date"].min().date(),
        outcome["as_of_date"].max().date(),
        horizon,
    )


if __name__ == "__main__":
    main()
