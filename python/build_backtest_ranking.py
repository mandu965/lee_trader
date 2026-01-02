"""
build_backtest_ranking.py

prediction_history(run_id) 기반으로 날짜별 랭킹을 계산해
research.ranking_history에 적재한다.

로직:
  - run_id로 prediction_history를 로드 (필요 시 날짜 범위 필터)
  - final_score 내림차순 정렬 후 rank 부여 (method="first")
  - rank <= top_n -> in_top_n = true
  - top_n: 인자 우선, 없으면 dim_model_run.top_n 조회, 그래도 없으면 env TOP_N(기본 20)
  - 기존 run_id (+ 날짜범위) 데이터는 삭제 후 insert
"""
import argparse
import logging
import os
from datetime import datetime
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
    p = argparse.ArgumentParser(description="Backfill ranking_history from prediction_history")
    p.add_argument("--run-id", type=int, required=True, help="run_id to backfill ranking")
    p.add_argument("--start-date", type=str, help="inclusive start as_of_date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, help="inclusive end as_of_date (YYYY-MM-DD)")
    p.add_argument("--top-n", type=int, help="override top_n")
    p.add_argument("--out-csv", type=Path, help="optional: save generated ranking to CSV")
    p.add_argument("--log-interval", type=int, default=20, help="log every N dates")
    return p.parse_args()


def resolve_top_n(run_id: int, override: Optional[int]) -> int:
    if override:
        return override
    if not get_engine:
        return int(os.environ.get("TOP_N", "20"))
    try:
        eng = get_engine()
        with eng.connect() as conn:
            res = conn.execute(
                text("SELECT top_n FROM research.dim_model_run WHERE run_id = :rid"),
                {"rid": run_id},
            ).scalar()
            if res:
                return int(res)
    except Exception:
        logging.warning("top_n lookup failed; fallback to env")
    return int(os.environ.get("TOP_N", "20"))


def load_predictions(run_id: int, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("No DB engine available")
    eng = get_engine()
    clauses = [f"run_id = {run_id}"]
    if start:
        clauses.append(f"as_of_date >= '{start}'")
    if end:
        clauses.append(f"as_of_date <= '{end}'")
    where_clause = " AND ".join(clauses)
    query = (
        "SELECT run_id, as_of_date, code, model_version, horizon_days, "
        "       final_score, ret_score, prob_score, qual_score, tech_score, risk_penalty "
        f"FROM research.prediction_history WHERE {where_clause}"
    )
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    return df


def build_ranking(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["top_n"] = top_n
    df["rank"] = (
        df.sort_values(["as_of_date", "final_score"], ascending=[True, False])
        .groupby("as_of_date")["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df["in_top_n"] = df["rank"] <= top_n
    df["top_n"] = top_n
    needed = [
        "run_id",
        "as_of_date",
        "code",
        "model_version",
        "horizon_days",
        "rank",
        "final_score",
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "risk_penalty",
        "in_top_n",
        "top_n",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    return df[needed]


def save_ranking(run_id: int, df_rank: pd.DataFrame, start: Optional[str], end: Optional[str]) -> None:
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
                conn.execute(text(f"DELETE FROM research.ranking_history WHERE {' AND '.join(where)}"), params)
            else:
                conn.execute(text("DELETE FROM research.ranking_history WHERE run_id = :rid"), {"rid": run_id})
        df_rank.to_sql("ranking_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Saved ranking_history (run_id=%s, rows=%d)", run_id, len(df_rank))
    except Exception:
        logging.exception("Failed to save ranking_history")


def main() -> None:
    setup_logging()
    args = parse_args()
    top_n = resolve_top_n(args.run_id, args.top_n)
    preds = load_predictions(args.run_id, args.start_date, args.end_date)
    if preds.empty:
        logging.warning("No prediction_history rows for run_id=%s (date filter start=%s end=%s)", args.run_id, args.start_date, args.end_date)
        return
    df_rank = build_ranking(preds, top_n)
    if args.out_csv:
        df_rank.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved ranking CSV: %s (rows=%d)", args.out_csv, len(df_rank))
    save_ranking(args.run_id, df_rank, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
