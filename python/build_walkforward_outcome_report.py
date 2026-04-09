"""
Build a performance summary report for walk-forward runs using shared run-status
and comparison-group rules.

Scope:
- official performance comparison uses matured runs only
- partial runs remain as reference-only
- unmatured runs are excluded with explicit reason
- run-level performance metrics
- horizon-level aggregation
- comparison-group averages for like-for-like run comparison
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from check_walkforward_outcome_coverage import (
    build_run_coverage_summary,
    load_backtest_outcome_rows,
    load_run_metadata,
    load_walkforward_predictions,
)
from db import get_engine
from walkforward_compare import annotate_comparison_groups, extract_comparison_metadata


DEFAULT_MIN_RUNS_PER_GROUP = 3


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build walk-forward matured outcome performance report")
    p.add_argument("--out-csv", type=Path, help="Optional run-level CSV path")
    p.add_argument("--out-md", type=Path, help="Optional markdown report path")
    p.add_argument(
        "--min-runs-per-group",
        type=int,
        default=DEFAULT_MIN_RUNS_PER_GROUP,
        help="Minimum matured runs required in a comparison_group to consider it an official comparison group.",
    )
    return p.parse_args()


def load_joined_performance_rows() -> pd.DataFrame:
    eng = get_engine()
    query = text(
        """
        SELECT
            r.run_id,
            r.as_of_date,
            r.code,
            r.horizon_days,
            r.in_top_n,
            r.top_n,
            o.realized_return,
            o.realized_mdd,
            o.is_matured
        FROM research.ranking_history r
        JOIN research.backtest_outcome o
          ON o.run_id = r.run_id
         AND o.as_of_date = r.as_of_date
         AND o.code = r.code
        JOIN research.dim_model_run d
          ON d.run_id = r.run_id
        WHERE d.run_type = 'walkforward_backtest'
        """
    )
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    if not df.empty:
        df["is_matured"] = df["is_matured"].astype("boolean")
        df["in_top_n"] = df["in_top_n"].fillna(False).astype(bool)
    return df


def build_run_metrics(
    rows: pd.DataFrame,
    run_status_summary: pd.DataFrame,
    *,
    min_runs_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "run_id",
        "model_version",
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "comparison_group_runs",
        "is_primary_group",
        "comparison_warning",
        "run_status",
        "run_status_reason",
        "avg_realized_return",
        "median_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "topn_hit_ratio",
        "coverage",
    ]
    if rows.empty or run_status_summary.empty:
        empty_groups = pd.DataFrame(
            columns=["comparison_group", "comparison_group_runs", "is_primary_group", "comparison_warning"]
        )
        return pd.DataFrame(columns=base_columns), empty_groups

    matured_runs = run_status_summary[run_status_summary["run_status"] == "matured"].copy()
    matured_ids = set(matured_runs["run_id"].astype(int).tolist())
    df = rows[rows["run_id"].isin(matured_ids)].copy()
    df = df[df["realized_return"].notna()].copy()
    if df.empty:
        empty_groups = pd.DataFrame(
            columns=["comparison_group", "comparison_group_runs", "is_primary_group", "comparison_warning"]
        )
        return pd.DataFrame(columns=base_columns), empty_groups

    run_metrics = (
        df.groupby("run_id", as_index=False)
        .agg(
            avg_realized_return=("realized_return", "mean"),
            median_realized_return=("realized_return", "median"),
            avg_realized_mdd=("realized_mdd", "mean"),
            hit_ratio=("realized_return", lambda s: float((pd.Series(s) > 0).mean())),
        )
    )

    topn = df[df["in_top_n"]].copy()
    topn_metrics = (
        topn.groupby("run_id", as_index=False)
        .agg(
            topn_avg_return=("realized_return", "mean"),
            topn_hit_ratio=("realized_return", lambda s: float((pd.Series(s) > 0).mean())),
        )
        if not topn.empty
        else pd.DataFrame(columns=["run_id", "topn_avg_return", "topn_hit_ratio"])
    )

    coverage = matured_runs[["run_id", "maturity_ratio", "run_status", "run_status_reason", "comparison_group"]].rename(
        columns={"maturity_ratio": "coverage"}
    )
    run_metrics = run_metrics.merge(topn_metrics, on="run_id", how="left").merge(coverage, on="run_id", how="left")
    run_metrics, group_summary = annotate_comparison_groups(run_metrics, min_runs_per_group=min_runs_per_group)

    meta = extract_comparison_metadata(load_run_metadata())
    run_metrics = run_metrics.merge(
        meta[["run_id", "model_version", "horizon_days", "score_formula_version"]],
        on="run_id",
        how="left",
    )
    return run_metrics[base_columns].sort_values(["horizon_days", "run_id"]).reset_index(drop=True), group_summary


def build_comparison_group_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    if run_metrics.empty:
        return pd.DataFrame(
            columns=[
                "comparison_group",
                "horizon_days",
                "runs",
                "is_primary_group",
                "comparison_warning",
                "group_avg_realized_return",
                "group_median_realized_return",
                "group_avg_realized_mdd",
                "group_hit_ratio",
                "group_topn_avg_return",
                "group_topn_hit_ratio",
                "group_avg_coverage",
            ]
        )

    grouped = (
        run_metrics.groupby(["comparison_group", "horizon_days", "is_primary_group", "comparison_warning"], as_index=False)
        .agg(
            runs=("run_id", "size"),
            group_avg_realized_return=("avg_realized_return", "mean"),
            group_median_realized_return=("median_realized_return", "mean"),
            group_avg_realized_mdd=("avg_realized_mdd", "mean"),
            group_hit_ratio=("hit_ratio", "mean"),
            group_topn_avg_return=("topn_avg_return", "mean"),
            group_topn_hit_ratio=("topn_hit_ratio", "mean"),
            group_avg_coverage=("coverage", "mean"),
        )
        .sort_values(["horizon_days", "runs", "comparison_group"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return grouped


def build_horizon_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    if run_metrics.empty:
        return pd.DataFrame(
            columns=[
                "horizon_days",
                "runs",
                "avg_realized_return",
                "median_realized_return",
                "avg_realized_mdd",
                "hit_ratio",
                "topn_avg_return",
                "topn_hit_ratio",
                "avg_coverage",
            ]
        )
    return (
        run_metrics.groupby("horizon_days", as_index=False)
        .agg(
            runs=("run_id", "size"),
            avg_realized_return=("avg_realized_return", "mean"),
            median_realized_return=("median_realized_return", "mean"),
            avg_realized_mdd=("avg_realized_mdd", "mean"),
            hit_ratio=("hit_ratio", "mean"),
            topn_avg_return=("topn_avg_return", "mean"),
            topn_hit_ratio=("topn_hit_ratio", "mean"),
            avg_coverage=("coverage", "mean"),
        )
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )


def build_formula_version_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "score_formula_version",
        "runs",
        "avg_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "positive_run_ratio",
    ]
    if run_metrics.empty:
        return pd.DataFrame(columns=columns)
    out = (
        run_metrics.groupby(["horizon_days", "score_formula_version"], dropna=False, as_index=False)
        .agg(
            runs=("run_id", "size"),
            avg_realized_return=("avg_realized_return", "mean"),
            avg_realized_mdd=("avg_realized_mdd", "mean"),
            hit_ratio=("hit_ratio", "mean"),
            topn_avg_return=("topn_avg_return", "mean"),
            positive_run_ratio=("avg_realized_return", lambda s: float((pd.Series(s) > 0).mean())),
        )
        .sort_values(["horizon_days", "runs", "avg_realized_return", "score_formula_version"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    out["score_formula_version"] = out["score_formula_version"].fillna("NA")
    return out[columns]


def judge_step4_ready(run_metrics: pd.DataFrame, group_summary: pd.DataFrame, min_runs_per_group: int) -> str:
    if run_metrics.empty:
        return "INSUFFICIENT: no matured walk-forward runs available for Step4"
    eligible_groups = group_summary[group_summary["runs"] >= min_runs_per_group]
    if eligible_groups.empty:
        return (
            "PARTIAL: matured runs exist, but no comparison_group has enough runs "
            f"(required>={min_runs_per_group})"
        )
    return (
        "READY: Step4 run-by-run performance comparison is available "
        f"({len(eligible_groups)} comparison groups with runs>={min_runs_per_group})"
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
    run_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    formula_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    run_status_summary: pd.DataFrame,
    *,
    min_runs_per_group: int,
) -> str:
    lines = ["# Walk-Forward Outcome Performance Report", ""]
    lines.append(f"- matured_runs: {len(run_metrics)}")
    lines.append(f"- step4_judgment: {judge_step4_ready(run_metrics, group_summary, min_runs_per_group)}")
    lines.append("")
    lines.append("## Run Status Rule")
    lines.append("")
    lines.append("- `matured`: official performance comparison allowed")
    lines.append("- `partial`: kept for reference only")
    lines.append("- `unmatured`: excluded from performance comparison with reason")
    lines.append("- `comparison_group` uses model_version, horizon_days, top_n, rebalance_freq, universe_version, score_formula_version.")
    lines.append("- groups with fewer than the minimum matured runs are flagged as reference-only")
    lines.append("")
    lines.append("## Comparison Group Summary")
    lines.append("")
    lines.extend(dataframe_to_markdown(group_summary))
    lines.append("")
    lines.append("## Score Formula Summary")
    lines.append("")
    lines.extend(dataframe_to_markdown(formula_summary))
    lines.append("")
    lines.append("## Horizon Summary")
    lines.append("")
    lines.extend(dataframe_to_markdown(horizon_summary))
    lines.append("")
    lines.append("## Non-Matured Runs")
    lines.append("")
    ref_cols = [
        "run_id",
        "model_version",
        "horizon_days",
        "maturity_ratio",
        "run_status",
        "run_status_reason",
        "comparison_group",
        "comparison_group_runs",
        "comparison_warning",
    ]
    non_matured = run_status_summary[run_status_summary["run_status"] != "matured"].copy()
    if non_matured.empty:
        lines.append("- none")
    else:
        lines.extend(dataframe_to_markdown(non_matured[ref_cols]))
    lines.append("")
    lines.append("## Run Metrics")
    lines.append("")
    lines.extend(dataframe_to_markdown(run_metrics))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    args = parse_args()

    run_status_summary, _ = build_run_coverage_summary(
        load_walkforward_predictions(),
        load_backtest_outcome_rows(),
        load_run_metadata(),
        min_runs_per_group=args.min_runs_per_group,
    )
    rows = load_joined_performance_rows()
    run_metrics, annotation_summary = build_run_metrics(
        rows,
        run_status_summary,
        min_runs_per_group=args.min_runs_per_group,
    )
    group_summary = build_comparison_group_summary(run_metrics)
    if group_summary.empty and not annotation_summary.empty:
        group_summary = annotation_summary
    horizon_summary = build_horizon_summary(run_metrics)
    formula_summary = build_formula_version_summary(run_metrics)

    reference_groups = group_summary[group_summary["comparison_warning"] != ""]
    if not reference_groups.empty:
        logging.warning(
            "Found %d comparison groups with fewer than %d matured runs; keep them as reference-only",
            len(reference_groups),
            args.min_runs_per_group,
        )
    partial_runs = run_status_summary[run_status_summary["run_status"] == "partial"]
    if not partial_runs.empty:
        logging.warning("Found %d partial runs; keep them as reference-only", len(partial_runs))
    unmatured_runs = run_status_summary[run_status_summary["run_status"] == "unmatured"]
    if not unmatured_runs.empty:
        sample = unmatured_runs[["run_id", "run_status_reason"]].head(5).to_dict(orient="records")
        logging.warning(
            "Found %d unmatured runs excluded from official comparison; sample=%s",
            len(unmatured_runs),
            sample,
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        run_metrics.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved walkforward outcome report CSV: %s", args.out_csv.resolve())

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            build_markdown_report(
                run_metrics,
                horizon_summary,
                formula_summary,
                group_summary,
                run_status_summary,
                min_runs_per_group=args.min_runs_per_group,
            ),
            encoding="utf-8",
        )
        logging.info("Saved walkforward outcome report markdown: %s", args.out_md.resolve())

    if not run_metrics.empty:
        print(run_metrics.to_string(index=False))
    if not group_summary.empty:
        print(group_summary.to_string(index=False))

    print(judge_step4_ready(run_metrics, group_summary, args.min_runs_per_group))


if __name__ == "__main__":
    main()
