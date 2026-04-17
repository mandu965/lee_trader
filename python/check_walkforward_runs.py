"""
Check accumulated walk-forward runs with a common outcome-maturity rule.

The maturity rule is based on actual price availability per stock:
- row-level: a prediction is mature when that stock has at least `horizon_days`
  future trading rows after `as_of_date`
- run-level: aggregate mature vs unmatured prediction rows and compare them to
  saved `backtest_outcome` rows
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine
from outcome_maturity import (
    build_price_reference,
    evaluate_prediction_maturity_rows,
    load_price_availability,
    summarize_run_maturity,
)


DEFAULT_MIN_RUNS = 8


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check accumulated walk-forward backtest runs")
    p.add_argument("--out-csv", type=Path, help="Optional output CSV path")
    p.add_argument("--out-md", type=Path, help="Optional markdown summary output path")
    p.add_argument(
        "--min-runs",
        type=int,
        default=DEFAULT_MIN_RUNS,
        help="Minimum number of primary-group runs required for a sufficient accumulation judgment.",
    )
    return p.parse_args()


def _decode_config_json(value) -> dict:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            obj = json.loads(value)
        except Exception as e:
            raise ValueError(f"Failed to parse config_json: {e}") from e
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("config_json must decode to an object")
        return obj
    raise ValueError(f"Unsupported config_json type: {type(value)}")


def load_walkforward_runs() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        WITH pred AS (
            SELECT
                run_id,
                COUNT(*) AS prediction_rows,
                COUNT(DISTINCT as_of_date) AS prediction_dates
            FROM research.prediction_history
            GROUP BY run_id
        ),
        rank_hist AS (
            SELECT
                run_id,
                COUNT(*) AS ranking_rows,
                COUNT(DISTINCT as_of_date) AS ranking_dates
            FROM research.ranking_history
            GROUP BY run_id
        ),
        outcome_hist AS (
            SELECT
                run_id,
                COUNT(*) AS outcome_rows,
                COUNT(DISTINCT as_of_date) AS outcome_dates
            FROM research.backtest_outcome
            GROUP BY run_id
        )
        SELECT
            d.run_id,
            d.model_version,
            d.horizon_days,
            d.top_n,
            d.config_json,
            COALESCE(pred.prediction_rows, 0) AS prediction_rows,
            COALESCE(rank_hist.ranking_rows, 0) AS ranking_rows,
            COALESCE(pred.prediction_dates, 0) AS prediction_dates,
            COALESCE(rank_hist.ranking_dates, 0) AS ranking_dates,
            COALESCE(outcome_hist.outcome_rows, 0) AS outcome_rows,
            COALESCE(outcome_hist.outcome_dates, 0) AS outcome_dates
        FROM research.dim_model_run d
        LEFT JOIN pred ON pred.run_id = d.run_id
        LEFT JOIN rank_hist ON rank_hist.run_id = d.run_id
        LEFT JOIN outcome_hist ON outcome_hist.run_id = d.run_id
        WHERE d.run_type = 'walkforward_backtest'
        ORDER BY d.run_id
        """
    )
    with eng.connect() as conn:
        return pd.read_sql(query, conn)


def load_walkforward_predictions() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        SELECT
            p.run_id,
            p.as_of_date,
            p.code,
            p.horizon_days
        FROM research.prediction_history p
        JOIN research.dim_model_run d
          ON d.run_id = p.run_id
        WHERE d.run_type = 'walkforward_backtest'
        """
    )
    with eng.connect() as conn:
        return pd.read_sql(query, conn, parse_dates=["as_of_date"])


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "model_version",
                "horizon_days",
                "top_n",
                "train_start",
                "train_end",
                "predict_start",
                "predict_end",
                "prediction_rows",
                "ranking_rows",
                "prediction_dates",
                "ranking_dates",
                "outcome_rows",
                "outcome_dates",
                "outcome_status",
                "matured_prediction_rows",
                "unmatured_prediction_rows",
                "matured_prediction_dates",
                "unmatured_prediction_dates",
                "expected_outcome_rows",
                "rebalance_freq",
                "universe_version",
                "universe_mode",
                "uses_fixed_current_universe",
                "historical_universe_built",
                "score_formula_version",
                "score_weights",
                "warning",
                "comparison_group",
                "comparison_group_runs",
                "is_primary_group",
            ]
        )

    rows = []
    for rec in df.to_dict(orient="records"):
        cfg = _decode_config_json(rec.get("config_json"))
        train_start = cfg.get("train_start")
        train_end = cfg.get("train_end")
        predict_start = cfg.get("predict_start")
        predict_end = cfg.get("predict_end")
        rebalance_freq = cfg.get("rebalance_freq")
        universe_version = cfg.get("universe_version")
        universe_mode = cfg.get("universe_mode")
        uses_fixed_current_universe = cfg.get("uses_fixed_current_universe")
        historical_universe_built = cfg.get("historical_universe_built")
        score_formula_version = cfg.get("score_formula_version")
        score_weights = cfg.get("score_weights")

        warnings = []
        if int(rec["prediction_rows"]) == 0 or int(rec["ranking_rows"]) == 0:
            warnings.append("WARN_EMPTY_RUN")

        rows.append(
            {
                "run_id": int(rec["run_id"]),
                "model_version": rec["model_version"],
                "horizon_days": int(rec["horizon_days"]) if pd.notna(rec["horizon_days"]) else None,
                "top_n": int(rec["top_n"]) if pd.notna(rec["top_n"]) else None,
                "train_start": train_start,
                "train_end": train_end,
                "predict_start": predict_start,
                "predict_end": predict_end,
                "prediction_rows": int(rec["prediction_rows"]),
                "ranking_rows": int(rec["ranking_rows"]),
                "prediction_dates": int(rec["prediction_dates"]),
                "ranking_dates": int(rec["ranking_dates"]),
                "outcome_rows": int(rec.get("outcome_rows", 0)),
                "outcome_dates": int(rec.get("outcome_dates", 0)),
                "rebalance_freq": rebalance_freq,
                "universe_version": universe_version,
                "universe_mode": universe_mode,
                "uses_fixed_current_universe": uses_fixed_current_universe,
                "historical_universe_built": historical_universe_built,
                "score_formula_version": score_formula_version,
                "score_weights": json.dumps(score_weights, ensure_ascii=True, sort_keys=True)
                if isinstance(score_weights, dict)
                else score_weights,
                "warning": ",".join(warnings),
            }
        )

    summary = pd.DataFrame(rows)
    maturity_summary = build_run_maturity_summary(df["run_id"].astype(int).tolist(), summary)
    maturity_keep = [
        "run_id",
        "matured_prediction_rows",
        "unmatured_prediction_rows",
        "matured_prediction_dates",
        "unmatured_prediction_dates",
        "expected_outcome_rows",
        "actual_outcome_rows",
        "run_outcome_status",
    ]
    maturity_summary = maturity_summary[maturity_keep] if not maturity_summary.empty else pd.DataFrame(columns=maturity_keep)
    summary = summary.merge(maturity_summary, on="run_id", how="left")
    for col in [
        "matured_prediction_rows",
        "unmatured_prediction_rows",
        "matured_prediction_dates",
        "unmatured_prediction_dates",
        "expected_outcome_rows",
        "actual_outcome_rows",
    ]:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0).astype(int)
    summary["outcome_status"] = summary["run_outcome_status"].fillna("OUTCOME_EMPTY")
    summary = summary.drop(columns=["run_outcome_status", "actual_outcome_rows"], errors="ignore")

    comp_cols = [
        "model_version",
        "horizon_days",
        "top_n",
        "rebalance_freq",
        "universe_version",
        "score_formula_version",
    ]
    summary["comparison_group"] = summary[comp_cols].fillna("NA").astype(str).agg("|".join, axis=1)

    group_counts = summary["comparison_group"].value_counts()
    primary_group = group_counts.index[0]
    summary["comparison_group_runs"] = summary["comparison_group"].map(group_counts).astype(int)
    summary["is_primary_group"] = summary["comparison_group"] == primary_group

    non_primary_mask = ~summary["is_primary_group"]
    if non_primary_mask.any():
        summary.loc[non_primary_mask, "warning"] = summary.loc[non_primary_mask, "warning"].apply(
            lambda value: "WARN_EXCLUDED_FROM_COMPARISON"
            if not value
            else f"{value},WARN_EXCLUDED_FROM_COMPARISON"
        )

    mismatch_mask = summary["prediction_rows"] != summary["ranking_rows"]
    if mismatch_mask.any():
        summary.loc[mismatch_mask, "warning"] = summary.loc[mismatch_mask, "warning"].apply(
            lambda value: "WARN_ROW_COUNT_MISMATCH" if not value else f"{value},WARN_ROW_COUNT_MISMATCH"
        )

    outcome_gap_mask = summary["outcome_rows"] < summary["expected_outcome_rows"]
    if outcome_gap_mask.any():
        summary.loc[outcome_gap_mask, "warning"] = summary.loc[outcome_gap_mask, "warning"].apply(
            lambda value: "WARN_OUTCOME_ROW_GAP" if not value else f"{value},WARN_OUTCOME_ROW_GAP"
        )

    primary_mask = summary["is_primary_group"]
    if primary_mask.any():
        for col in ["prediction_rows", "ranking_rows"]:
            median_value = summary.loc[primary_mask, col].median()
            if pd.notna(median_value) and median_value > 0:
                low = median_value * 0.5
                high = median_value * 1.5
                outlier_mask = primary_mask & ((summary[col] < low) | (summary[col] > high))
                if outlier_mask.any():
                    summary.loc[outlier_mask, "warning"] = summary.loc[outlier_mask, "warning"].apply(
                        lambda value: f"WARN_{col.upper()}_OUTLIER" if not value else f"{value},WARN_{col.upper()}_OUTLIER"
                    )

    return summary


def build_run_maturity_summary(run_ids: list[int], run_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build run-level maturity status from prediction rows and actual price coverage.
    """
    if not run_ids:
        return pd.DataFrame(columns=["run_id", "run_outcome_status"])

    preds = load_walkforward_predictions()
    preds = preds[preds["run_id"].isin(run_ids)].copy()
    if preds.empty:
        return pd.DataFrame(columns=["run_id", "run_outcome_status"])

    price_reference = build_price_reference(load_price_availability())
    maturity_frames = []
    for horizon_days, group in preds.groupby("horizon_days"):
        maturity_frames.append(
            evaluate_prediction_maturity_rows(
                group,
                price_reference=price_reference,
                horizon_days=int(horizon_days),
                as_of_col="as_of_date",
                code_col="code",
            )
        )
    maturity_rows = pd.concat(maturity_frames, ignore_index=True) if maturity_frames else pd.DataFrame()
    outcome_rows_by_run = {
        int(row["run_id"]): int(row["outcome_rows"])
        for row in run_rows[["run_id", "outcome_rows"]].to_dict(orient="records")
    }
    return summarize_run_maturity(
        maturity_rows,
        run_id_col="run_id",
        as_of_col="as_of_date",
        outcome_rows_by_run=outcome_rows_by_run,
    )


def judge_run_accumulation(summary: pd.DataFrame, min_runs: int) -> str:
    if summary.empty:
        return "INSUFFICIENT: no walkforward_backtest runs found"

    primary = summary[summary["is_primary_group"]].copy()
    if primary.empty:
        return "INSUFFICIENT: no primary comparison group identified"

    primary_run_count = len(primary)
    empty_primary = primary[(primary["prediction_rows"] == 0) | (primary["ranking_rows"] == 0)]
    mature_primary = primary[primary["outcome_status"] == "OUTCOME_READY"]
    comparison_group_count = int(summary["comparison_group"].nunique())

    reasons = []
    if primary_run_count < min_runs:
        reasons.append(f"primary-group runs too small ({primary_run_count} < {min_runs})")
    if not empty_primary.empty:
        reasons.append(f"empty runs present in primary group ({len(empty_primary)})")

    if reasons:
        return "INSUFFICIENT: " + ", ".join(reasons)
    if comparison_group_count > 1:
        return (
            "SUFFICIENT_WITH_WARNINGS: primary group is large enough and internally comparable, "
            f"but other setting groups exist ({comparison_group_count} groups total)"
        )
    if len(mature_primary) < min_runs:
        return (
            "PARTIAL: primary group is structurally consistent, "
            f"but mature outcome runs are only {len(mature_primary)}"
        )
    return (
        "SUFFICIENT: primary group has enough runs, no empty runs, "
        "and settings are comparable"
    )


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.fillna("").astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return lines


def build_markdown_summary(summary: pd.DataFrame, min_runs: int) -> str:
    lines = ["# Walk-Forward Run Check", ""]
    lines.append(f"- total_runs: {len(summary)}")
    lines.append(f"- judgment: {judge_run_accumulation(summary, min_runs)}")
    lines.append("")

    if summary.empty:
        return "\n".join(lines) + "\n"

    primary_group = summary.loc[summary["is_primary_group"], "comparison_group"].iloc[0]
    primary = summary[summary["is_primary_group"]].copy()
    fixed_current_universe = bool(primary["uses_fixed_current_universe"].astype("boolean").fillna(False).any())
    historical_universe_built = bool(primary["historical_universe_built"].astype("boolean").fillna(False).all()) if not primary.empty else False
    lines.append(f"- primary_comparison_group: `{primary_group}`")
    lines.append(f"- current_universe_fixed_use: {'yes' if fixed_current_universe else 'no_or_unknown'}")
    lines.append(f"- historical_universe_built: {'yes' if historical_universe_built else 'no'}")
    lines.append("")
    lines.append("## Research Warnings")
    lines.append("")
    lines.append("- Current universe is treated as fixed when `universe_mode=fixed_current_universe`.")
    lines.append("- Historical universe snapshots are not yet built for this project.")
    lines.append("- Survivor bias is possible because past runs may include stocks chosen from the current universe definition.")
    lines.append("- Current walk-forward outputs should be interpreted as structure validation and temporary research results, not a production-grade performance claim.")
    lines.append("")
    lines.append("## Comparison Groups")
    lines.append("")
    lines.append("| comparison_group | runs | primary |")
    lines.append("|---|---:|---|")

    group_counts = (
        summary[["comparison_group", "comparison_group_runs", "is_primary_group"]]
        .drop_duplicates()
        .sort_values(["comparison_group_runs", "comparison_group"], ascending=[False, True])
    )
    for row in group_counts.to_dict(orient="records"):
        lines.append(
            f"| `{row['comparison_group']}` | {row['comparison_group_runs']} | {'yes' if row['is_primary_group'] else 'no'} |"
        )

    lines.append("")
    lines.append("## Runs")
    lines.append("")
    cols = [
        "run_id",
        "model_version",
        "horizon_days",
        "top_n",
        "train_start",
        "train_end",
        "predict_start",
        "predict_end",
        "prediction_rows",
        "ranking_rows",
        "matured_prediction_rows",
        "unmatured_prediction_rows",
        "outcome_rows",
        "outcome_status",
        "universe_mode",
        "comparison_group_runs",
        "is_primary_group",
        "warning",
    ]
    lines.extend(dataframe_to_markdown(summary[cols]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    args = parse_args()

    runs = load_walkforward_runs()
    summary = build_summary(runs)

    if summary.empty:
        logging.warning("No walkforward_backtest runs found")
    else:
        empty_runs = summary[(summary["prediction_rows"] == 0) | (summary["ranking_rows"] == 0)]
        if not empty_runs.empty:
            logging.warning(
                "Found %d walkforward runs with empty prediction or ranking rows",
                len(empty_runs),
            )

        excluded = summary[~summary["is_primary_group"]]
        if not excluded.empty:
            logging.warning(
                "Found %d runs outside the primary comparison group; exclude them from performance comparison",
                len(excluded),
            )

        mismatched = summary[summary["prediction_rows"] != summary["ranking_rows"]]
        if not mismatched.empty:
            logging.warning("Found %d runs with prediction/ranking row count mismatch", len(mismatched))

        outliers = summary[
            summary["warning"].fillna("").str.contains("OUTLIER", regex=False)
        ]
        if not outliers.empty:
            logging.warning("Found %d runs with row count outlier warnings", len(outliers))

        immature = summary[summary["outcome_status"].isin(["OUTCOME_NOT_MATURE", "OUTCOME_PARTIAL", "OUTCOME_EMPTY"])]
        if not immature.empty:
            logging.warning("Found %d runs with non-ready outcome status", len(immature))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved walkforward run summary CSV: %s", args.out_csv.resolve())

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(build_markdown_summary(summary, args.min_runs), encoding="utf-8")
        logging.info("Saved walkforward run summary markdown: %s", args.out_md.resolve())

    if not summary.empty:
        print(summary.to_string(index=False))

    print(judge_run_accumulation(summary, args.min_runs))


if __name__ == "__main__":
    main()
