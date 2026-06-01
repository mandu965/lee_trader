"""
Standardized walk-forward outcome builder.

This script computes row-level outcome directly from prediction history and
actual price data. It does not rely on precomputed labels for the realized
return / realized MDD values.

Rules:
- `realized_return` and `realized_mdd` are computed from actual stock price
  history using the prediction row's `as_of_date` and `horizon_days`.
- `is_matured=false` means the row does not yet have enough future trading rows
  in the real price history. In that case realized fields stay NULL.
- Maturity uses real trading-date availability per stock, not simple calendar
  math, so missing trading days are handled correctly.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text

from outcome_maturity import (
    attach_forward_outcomes,
    build_price_reference,
    evaluate_prediction_maturity_rows,
    load_price_history,
)

try:
    from db import get_engine
except Exception:
    get_engine = None


OUTCOME_DB_COLUMNS = [
    "run_id",
    "as_of_date",
    "code",
    "horizon_days",
    "realized_return",
    "realized_return_net",
    "realized_mdd",
    "label_source",
    "is_matured",
    "maturity_status",
    "available_future_trading_days",
    "required_future_trading_days",
    "last_available_price_date",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build standardized backtest_outcome rows from actual prices")
    p.add_argument("--run-id", type=int, required=True, help="run_id to backfill outcome")
    p.add_argument("--horizon-days", type=int, help="override horizon_days (default: dim_model_run or env)")
    p.add_argument("--start-date", type=str, help="inclusive start as_of_date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, help="inclusive end as_of_date (YYYY-MM-DD)")
    p.add_argument("--out-csv", type=Path, help="optional: save row-level outcome CSV")
    p.add_argument("--out-summary-csv", type=Path, help="optional: save run-level summary CSV")
    return p.parse_args()


def resolve_horizon(run_id: int, override: Optional[int]) -> int:
    if override:
        return override
    if not get_engine:
        return int(os.environ.get("HORIZON_DAYS", "60"))
    eng = get_engine()
    with eng.connect() as conn:
        res = conn.execute(
            text("SELECT horizon_days FROM research.dim_model_run WHERE run_id = :rid"),
            {"rid": run_id},
        ).scalar()
    if res:
        return int(res)
    return int(os.environ.get("HORIZON_DAYS", "60"))


def load_predictions(run_id: int, start: Optional[str], end: Optional[str], horizon: int) -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("No DB engine available")
    eng = get_engine()
    clauses = [f"run_id = {run_id}", f"horizon_days = {horizon}"]
    if start:
        clauses.append(f"as_of_date >= '{start}'")
    if end:
        clauses.append(f"as_of_date <= '{end}'")
    query = (
        "SELECT run_id, as_of_date, code, horizon_days "
        f"FROM research.prediction_history WHERE {' AND '.join(clauses)}"
    )
    with eng.connect() as conn:
        preds = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    preds["code"] = preds["code"].astype(str).str.zfill(6)
    preds["as_of_date"] = pd.to_datetime(preds["as_of_date"]).dt.normalize()
    return preds


def build_outcome_rows(preds: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(columns=OUTCOME_DB_COLUMNS)

    price_history = load_price_history()
    price_reference = build_price_reference(price_history[["code", "date"]])
    maturity_rows = evaluate_prediction_maturity_rows(
        preds,
        price_reference=price_reference,
        horizon_days=horizon,
        as_of_col="as_of_date",
        code_col="code",
    )
    outcome_ref = attach_forward_outcomes(price_history, horizon_days=horizon).rename(columns={"date": "as_of_date"})
    out = maturity_rows.merge(
        outcome_ref,
        on=["code", "as_of_date"],
        how="left",
    )
    out["label_source"] = "from_price_history"
    out.loc[~out["is_matured"], ["realized_return", "realized_return_net", "realized_mdd"]] = pd.NA
    return out[OUTCOME_DB_COLUMNS].copy()


def build_run_summary(outcome_rows: pd.DataFrame) -> pd.DataFrame:
    if outcome_rows.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "prediction_rows",
                "matured_rows",
                "unmatured_rows",
                "realized_rows",
                "maturity_coverage_pct",
                "missing_price_rows",
            ]
        )

    summary = (
        outcome_rows.groupby("run_id", as_index=False)
        .agg(
            prediction_rows=("code", "size"),
            matured_rows=("is_matured", lambda s: int(pd.Series(s).fillna(False).sum())),
            missing_price_rows=("maturity_status", lambda s: int((pd.Series(s) == "PRICE_DATE_MISSING").sum())),
            realized_rows=("realized_return", lambda s: int(pd.Series(s).notna().sum())),
        )
    )
    summary["unmatured_rows"] = summary["prediction_rows"] - summary["matured_rows"]
    summary["maturity_coverage_pct"] = (
        summary["matured_rows"] / summary["prediction_rows"].replace(0, pd.NA) * 100.0
    ).fillna(0.0)
    return summary[
        [
            "run_id",
            "prediction_rows",
            "matured_rows",
            "unmatured_rows",
            "realized_rows",
            "maturity_coverage_pct",
            "missing_price_rows",
        ]
    ]


def ensure_backtest_outcome_columns() -> None:
    if not get_engine:
        return
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS realized_return_net double precision"))
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS is_matured boolean"))
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS maturity_status character varying(32)"))
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS available_future_trading_days integer"))
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS required_future_trading_days integer"))
        conn.execute(text("ALTER TABLE research.backtest_outcome ADD COLUMN IF NOT EXISTS last_available_price_date date"))


def save_outcome(run_id: int, df_out: pd.DataFrame, start: Optional[str], end: Optional[str]) -> None:
    if not get_engine:
        logging.info("No DB engine available -> skip DB save")
        return
    eng = get_engine()
    ensure_backtest_outcome_columns()
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


def log_warnings(outcome_rows: pd.DataFrame) -> None:
    missing_price = outcome_rows[outcome_rows["maturity_status"] == "PRICE_DATE_MISSING"]
    if not missing_price.empty:
        sample = missing_price[["code", "as_of_date"]].head(5).to_dict(orient="records")
        logging.warning(
            "Found %d prediction rows with missing price dates; sample=%s",
            len(missing_price),
            sample,
        )

    matured_without_realized = outcome_rows[
        outcome_rows["is_matured"].fillna(False) & outcome_rows["realized_return"].isna()
    ]
    if not matured_without_realized.empty:
        sample = matured_without_realized[["code", "as_of_date"]].head(5).to_dict(orient="records")
        logging.warning(
            "Found %d matured rows without realized outcome values; sample=%s",
            len(matured_without_realized),
            sample,
        )


def main() -> None:
    setup_logging()
    args = parse_args()
    horizon = resolve_horizon(args.run_id, args.horizon_days)
    if horizon not in (60, 90):
        logging.warning("This standardized outcome builder is currently validated for horizon_days in {60, 90}; got %s", horizon)

    preds = load_predictions(args.run_id, args.start_date, args.end_date, horizon)
    if preds.empty:
        logging.warning("No prediction_history rows for run_id=%s", args.run_id)
        return

    outcome_rows = build_outcome_rows(preds, horizon)
    if outcome_rows.empty:
        logging.warning("No outcome rows were built for run_id=%s", args.run_id)
        return

    log_warnings(outcome_rows)
    summary = build_run_summary(outcome_rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        outcome_rows.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved row-level outcome CSV: %s (rows=%d)", args.out_csv, len(outcome_rows))

    if args.out_summary_csv:
        args.out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_summary_csv, index=False, encoding="utf-8")
        logging.info("Saved run-level outcome summary CSV: %s", args.out_summary_csv)

    save_outcome(args.run_id, outcome_rows, args.start_date, args.end_date)

    if not summary.empty:
        row = summary.iloc[0].to_dict()
        logging.info(
            "Maturity coverage: run_id=%s prediction_rows=%s matured_rows=%s unmatured_rows=%s realized_rows=%s coverage=%.2f%% missing_price_rows=%s",
            int(row["run_id"]),
            int(row["prediction_rows"]),
            int(row["matured_rows"]),
            int(row["unmatured_rows"]),
            int(row["realized_rows"]),
            float(row["maturity_coverage_pct"]),
            int(row["missing_price_rows"]),
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
