"""
Check whether walk-forward runs have enough mature outcome coverage for Step 3.

This script focuses on outcome-evaluable coverage, not score quality itself.
It answers:
- how many prediction rows are mature per run
- what fraction of each run is outcome-evaluable
- which matured runs are eligible for official comparison
- how many matured runs exist per horizon

Outcome maturity uses the common trading-date coverage rule from
`outcome_maturity.py`, so weekends and missing price dates are handled via real
price availability rather than calendar-day arithmetic.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine
from outcome_maturity import (
    build_price_reference,
    classify_run_status,
    evaluate_prediction_maturity_rows,
    load_price_availability,
)
from walkforward_compare import annotate_comparison_groups, extract_comparison_metadata


DEFAULT_MIN_MATURED_RUNS_PER_HORIZON = 3
DEFAULT_MIN_RUNS_PER_GROUP = 3


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check walk-forward outcome maturity coverage")
    p.add_argument("--out-csv", type=Path, help="Optional output CSV path")
    p.add_argument("--out-md", type=Path, help="Optional markdown output path")
    p.add_argument(
        "--min-matured-runs-per-horizon",
        type=int,
        default=DEFAULT_MIN_MATURED_RUNS_PER_HORIZON,
        help="Minimum matured runs per horizon required for a sufficient Step 3 judgment.",
    )
    p.add_argument(
        "--min-runs-per-group",
        type=int,
        default=DEFAULT_MIN_RUNS_PER_GROUP,
        help="Minimum matured runs required inside a comparison_group for official comparison.",
    )
    return p.parse_args()


def load_walkforward_predictions() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        SELECT
            d.run_id,
            d.model_version,
            d.horizon_days,
            p.as_of_date,
            p.code
        FROM research.dim_model_run d
        JOIN research.prediction_history p
          ON p.run_id = d.run_id
        WHERE d.run_type = 'walkforward_backtest'
        ORDER BY d.run_id, p.as_of_date, p.code
        """
    )
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    if not df.empty:
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.normalize()
    return df


def load_run_metadata() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        SELECT
            run_id,
            model_version,
            horizon_days,
            top_n,
            config_json
        FROM research.dim_model_run
        WHERE run_type = 'walkforward_backtest'
        ORDER BY run_id
        """
    )
    with eng.connect() as conn:
        return pd.read_sql(query, conn)


def load_backtest_outcome_rows() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        SELECT
            d.run_id,
            d.horizon_days,
            COUNT(*) AS outcome_rows
        FROM research.dim_model_run d
        LEFT JOIN research.backtest_outcome o
          ON o.run_id = d.run_id
        WHERE d.run_type = 'walkforward_backtest'
        GROUP BY d.run_id, d.horizon_days
        ORDER BY d.run_id
        """
    )
    with eng.connect() as conn:
        return pd.read_sql(query, conn)


def build_run_coverage_summary(
    preds: pd.DataFrame,
    outcome_counts: pd.DataFrame,
    run_metadata: pd.DataFrame,
    *,
    min_runs_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "run_id",
        "model_version",
        "horizon_days",
        "top_n",
        "rebalance_freq",
        "universe_version",
        "score_formula_version",
        "comparison_group",
        "comparison_group_runs",
        "is_primary_group",
        "comparison_warning",
        "prediction_rows",
        "matured_rows",
        "maturity_ratio",
        "run_status",
        "run_status_reason",
        "matured_run_flag",
        "outcome_rows",
    ]
    if preds.empty:
        empty_groups = pd.DataFrame(
            columns=["comparison_group", "comparison_group_runs", "is_primary_group", "comparison_warning"]
        )
        return pd.DataFrame(columns=base_columns), empty_groups

    price_reference = build_price_reference(load_price_availability())
    maturity_frames = []
    for horizon_days, group in preds.groupby("horizon_days", sort=True):
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

    rows = []
    for (run_id, model_version, horizon_days), group in maturity_rows.groupby(
        ["run_id", "model_version", "horizon_days"],
        sort=True,
    ):
        prediction_rows = int(len(group))
        matured_rows = int(group["is_matured"].fillna(False).sum())
        maturity_ratio = 0.0 if prediction_rows == 0 else matured_rows / prediction_rows
        run_status, run_status_reason = classify_run_status(maturity_ratio)
        rows.append(
            {
                "run_id": int(run_id),
                "model_version": model_version,
                "horizon_days": int(horizon_days),
                "prediction_rows": prediction_rows,
                "matured_rows": matured_rows,
                "maturity_ratio": maturity_ratio,
                "run_status": run_status,
                "run_status_reason": run_status_reason,
                "matured_run_flag": run_status == "matured",
            }
        )

    summary = pd.DataFrame(rows).sort_values(["horizon_days", "run_id"]).reset_index(drop=True)
    if outcome_counts is not None and not outcome_counts.empty:
        summary = summary.merge(outcome_counts[["run_id", "outcome_rows"]], on="run_id", how="left")

    meta = extract_comparison_metadata(run_metadata)
    summary = summary.merge(
        meta[
            [
                "run_id",
                "top_n",
                "rebalance_freq",
                "universe_version",
                "score_formula_version",
                "comparison_group",
            ]
        ],
        on="run_id",
        how="left",
    )
    summary, group_summary = annotate_comparison_groups(summary, min_runs_per_group=min_runs_per_group)
    return summary[base_columns], group_summary


def build_horizon_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame(columns=["horizon_days", "total_runs", "matured_runs"])
    grouped = (
        run_summary.groupby("horizon_days", as_index=False)
        .agg(
            total_runs=("run_id", "size"),
            matured_runs=("matured_run_flag", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )
    return grouped


def judge_step3_sufficiency(horizon_summary: pd.DataFrame, min_matured_runs_per_horizon: int) -> str:
    if horizon_summary.empty:
        return "INSUFFICIENT: no walkforward_backtest runs found"

    insufficient = horizon_summary[horizon_summary["matured_runs"] < min_matured_runs_per_horizon]
    if not insufficient.empty:
        details = ", ".join(
            f"h{int(row['horizon_days'])}={int(row['matured_runs'])}"
            for row in insufficient.to_dict(orient="records")
        )
        return (
            "INSUFFICIENT: Step3 outcome-evaluable runs are too few for some horizons "
            f"(required>={min_matured_runs_per_horizon}; {details})"
        )

    return (
        "SUFFICIENT: Step3 outcome-evaluable run coverage is available for all observed horizons "
        f"(required>={min_matured_runs_per_horizon} matured runs per horizon)"
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


def build_markdown_report(
    run_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    *,
    min_matured_runs_per_horizon: int,
) -> str:
    lines = ["# Walk-Forward Outcome Coverage Check", ""]
    lines.append(f"- total_runs: {len(run_summary)}")
    lines.append(
        f"- judgment: {judge_step3_sufficiency(horizon_summary, min_matured_runs_per_horizon)}"
    )
    lines.append("")
    lines.append("## Maturity Rule")
    lines.append("")
    lines.append("- `matured_rows` counts prediction rows with enough future trading rows in actual price history.")
    lines.append("- `maturity_ratio = matured_rows / prediction_rows`.")
    lines.append("- Shared `run_status` rule: `unmatured`, `partial`, `matured`.")
    lines.append("- `matured_run_flag = true` only when `run_status = matured`.")
    lines.append("- Official comparison candidates are `run_status = matured` only.")
    lines.append("- `partial` runs remain in the report for reference only.")
    lines.append("- `unmatured` runs are excluded from official performance comparison and kept with reason.")
    lines.append("- `comparison_group` uses model_version, horizon_days, top_n, rebalance_freq, universe_version, score_formula_version.")
    lines.append("")
    lines.append("## Comparison Groups")
    lines.append("")
    lines.extend(dataframe_to_markdown(group_summary))
    lines.append("")
    lines.append("## Horizon Summary")
    lines.append("")
    lines.extend(dataframe_to_markdown(horizon_summary))
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    display_cols = [
        "run_id",
        "model_version",
        "horizon_days",
        "prediction_rows",
        "matured_rows",
        "maturity_ratio",
        "run_status",
        "run_status_reason",
        "matured_run_flag",
        "comparison_group",
        "comparison_group_runs",
        "is_primary_group",
        "comparison_warning",
    ]
    lines.extend(dataframe_to_markdown(run_summary[display_cols]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    args = parse_args()

    preds = load_walkforward_predictions()
    outcome_counts = load_backtest_outcome_rows()
    run_metadata = load_run_metadata()
    run_summary, group_summary = build_run_coverage_summary(
        preds,
        outcome_counts,
        run_metadata,
        min_runs_per_group=args.min_runs_per_group,
    )
    horizon_summary = build_horizon_summary(run_summary)

    matured_runs = run_summary[run_summary["matured_run_flag"]]
    if not matured_runs.empty:
        logging.info("Found %d matured walk-forward runs", len(matured_runs))
    primary_runs = matured_runs[matured_runs["is_primary_group"]]
    if not primary_runs.empty:
        logging.info("Primary comparison group matured runs: %d", len(primary_runs))
    reference_groups = group_summary[group_summary["comparison_group_runs"] < args.min_runs_per_group]
    if not reference_groups.empty:
        logging.warning(
            "Found %d comparison groups with fewer than %d matured runs; keep them as reference-only",
            len(reference_groups),
            args.min_runs_per_group,
        )
    partial_runs = run_summary[run_summary["run_status"] == "partial"]
    if not partial_runs.empty:
        logging.warning("Found %d partial runs; keep them as reference-only", len(partial_runs))
    unmatured_runs = run_summary[run_summary["run_status"] == "unmatured"]
    if not unmatured_runs.empty:
        sample = unmatured_runs[["run_id", "run_status_reason"]].head(5).to_dict(orient="records")
        logging.warning(
            "Found %d unmatured runs excluded from performance comparison; sample=%s",
            len(unmatured_runs),
            sample,
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        run_summary.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved outcome coverage CSV: %s", args.out_csv.resolve())

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            build_markdown_report(
                run_summary,
                horizon_summary,
                group_summary,
                min_matured_runs_per_horizon=args.min_matured_runs_per_horizon,
            ),
            encoding="utf-8",
        )
        logging.info("Saved outcome coverage markdown: %s", args.out_md.resolve())

    if not run_summary.empty:
        print(run_summary.to_string(index=False))
    if not horizon_summary.empty:
        print(horizon_summary.to_string(index=False))

    print(judge_step3_sufficiency(horizon_summary, args.min_matured_runs_per_horizon))


if __name__ == "__main__":
    main()
