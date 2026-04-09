"""
Build comparison-group performance summaries from a run-level outcome summary.

Input is expected to be a CSV produced by the walk-forward outcome report, such
as `outputs/walkforward_outcome_report.csv`. Official group comparison uses
`run_status == matured` rows only.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd


DEFAULT_MIN_RUNS_PER_GROUP = 3
DEFAULT_INPUT_CSV = Path("outputs/walkforward_outcome_report.csv")
DEFAULT_OUT_CSV = Path("outputs/walkforward_group_comparison.csv")
DEFAULT_OUT_MD = Path("outputs/walkforward_group_comparison.md")
DEFAULT_OUT_RUN_CSV = Path("outputs/walkforward_group_comparison_runs.csv")
DEFAULT_OUT_WARNINGS_CSV = Path("outputs/walkforward_group_comparison_warnings.csv")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build walk-forward comparison-group summary from run-level outcome CSV")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Run-level outcome summary CSV path",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Comparison-group summary CSV output path",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Comparison-group summary markdown output path",
    )
    p.add_argument(
        "--out-run-csv",
        type=Path,
        default=DEFAULT_OUT_RUN_CSV,
        help="Run-level classified CSV output path",
    )
    p.add_argument(
        "--out-warnings-csv",
        type=Path,
        default=DEFAULT_OUT_WARNINGS_CSV,
        help="Warnings CSV output path",
    )
    p.add_argument(
        "--min-runs-per-group",
        type=int,
        default=DEFAULT_MIN_RUNS_PER_GROUP,
        help="Minimum matured runs required for an official comparison group.",
    )
    p.add_argument(
        "--horizon-days",
        type=str,
        default="",
        help="Optional comma-separated horizon filter, e.g. 60,90",
    )
    p.add_argument(
        "--universe-baseline-csv",
        type=Path,
        default=None,
        help="Optional universe-average baseline CSV. Supported keys: run_id or horizon_days/model_version/comparison_group.",
    )
    p.add_argument(
        "--benchmark-baseline-csv",
        type=Path,
        default=None,
        help="Optional benchmark-index baseline CSV. Supported keys: run_id or horizon_days/model_version/comparison_group.",
    )
    p.add_argument(
        "--disable-previous-model-baseline",
        action="store_true",
        help="Disable automatic previous-model_version baseline derivation.",
    )
    return p.parse_args()


def parse_horizon_filter(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"horizon_days must be > 0: {value}")
        values.append(value)
    return sorted(set(values))


def load_run_level_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    required = [
        "run_id",
        "horizon_days",
        "comparison_group",
        "avg_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "topn_hit_ratio",
        "median_realized_return",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in run-level summary: {missing}")

    if "run_status" in df.columns:
        df = df[df["run_status"].astype(str).str.lower() == "matured"].copy()
    else:
        logging.warning("`run_status` column is missing; all rows will be treated as matured reference rows")

    if df.empty:
        return df

    numeric_cols = [
        "run_id",
        "horizon_days",
        "avg_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "topn_hit_ratio",
        "median_realized_return",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["comparison_group"] = df["comparison_group"].fillna("NA").astype(str)
    if "score_formula_version" not in df.columns:
        df["score_formula_version"] = "NA"
    df["score_formula_version"] = df["score_formula_version"].fillna("NA").astype(str)
    return df


def make_warning_rows(scope: str, code: str, message: str, *, horizon_days: int | None = None) -> list[dict]:
    return [
        {
            "scope": scope,
            "horizon_days": horizon_days,
            "warning_code": code,
            "warning_message": message,
        }
    ]


def resolve_baseline_value_column(df: pd.DataFrame, preferred: str) -> str | None:
    candidates = [
        preferred,
        "baseline_return",
        "avg_return",
        "return",
        "realized_return",
        "benchmark_return",
        "universe_return",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_optional_baseline_csv(
    path: Path | None,
    *,
    baseline_name: str,
    preferred_value_col: str,
) -> tuple[pd.DataFrame | None, list[dict]]:
    if path is None:
        return None, make_warning_rows("baseline", f"WARN_{baseline_name.upper()}_BASELINE_MISSING", f"{baseline_name} baseline CSV not provided")
    if not path.exists():
        return None, make_warning_rows("baseline", f"WARN_{baseline_name.upper()}_BASELINE_NOT_FOUND", f"{baseline_name} baseline CSV not found: {path}")

    df = pd.read_csv(path)
    value_col = resolve_baseline_value_column(df, preferred_value_col)
    if value_col is None:
        return None, make_warning_rows(
            "baseline",
            f"WARN_{baseline_name.upper()}_BASELINE_INVALID",
            f"{baseline_name} baseline CSV has no supported return column",
        )

    out = df.copy()
    if value_col != preferred_value_col:
        out = out.rename(columns={value_col: preferred_value_col})
    for col in ["run_id", "horizon_days"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["model_version", "comparison_group"]:
        if col in out.columns:
            out[col] = out[col].fillna("NA").astype(str)
    out[preferred_value_col] = pd.to_numeric(out[preferred_value_col], errors="coerce")
    return out, []


def merge_baseline_by_keys(
    run_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    *,
    value_col: str,
    output_col: str,
    baseline_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    merged = run_df.copy()
    merged[output_col] = pd.NA
    warnings: list[dict] = []

    if baseline_df is None or baseline_df.empty:
        warnings.extend(
            make_warning_rows(
                "baseline",
                f"WARN_{baseline_name.upper()}_BASELINE_UNAVAILABLE",
                f"{baseline_name} baseline data is unavailable",
            )
        )
        return merged, warnings

    key_options = [
        ["run_id"],
        ["horizon_days", "model_version"],
        ["horizon_days", "comparison_group"],
        ["horizon_days"],
    ]
    matched = False
    for keys in key_options:
        if not all(key in baseline_df.columns for key in keys):
            continue
        subset = baseline_df[keys + [value_col]].copy()
        subset = subset.dropna(subset=[value_col]).drop_duplicates(subset=keys, keep="last")
        if subset.empty:
            continue
        merged = merged.drop(columns=[output_col], errors="ignore").merge(subset, on=keys, how="left")
        merged = merged.rename(columns={value_col: output_col})
        matched = True
        break

    if not matched:
        warnings.extend(
            make_warning_rows(
                "baseline",
                f"WARN_{baseline_name.upper()}_BASELINE_NO_MATCHABLE_KEYS",
                f"{baseline_name} baseline CSV does not contain supported keys",
            )
        )
        merged[output_col] = pd.NA
        return merged, warnings

    null_count = int(merged[output_col].isna().sum())
    if null_count > 0:
        warnings.extend(
            make_warning_rows(
                "baseline",
                f"WARN_{baseline_name.upper()}_BASELINE_PARTIAL_NULL",
                f"{baseline_name} baseline missing for {null_count} runs after merge",
            )
        )
    return merged, warnings


def attach_previous_model_baseline(run_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    merged = run_df.copy()
    merged["previous_model_version"] = pd.NA
    merged["previous_model_return"] = pd.NA
    warnings: list[dict] = []
    if merged.empty:
        return merged, warnings

    summaries = (
        merged.groupby(["horizon_days", "model_version"], as_index=False)
        .agg(
            model_avg_return=("avg_realized_return", "mean"),
            model_min_run_id=("run_id", "min"),
        )
        .sort_values(["horizon_days", "model_min_run_id", "model_version"])
        .reset_index(drop=True)
    )

    prev_frames: list[pd.DataFrame] = []
    for horizon_days, group in summaries.groupby("horizon_days", sort=True):
        ordered = group.sort_values(["model_min_run_id", "model_version"]).reset_index(drop=True)
        ordered["previous_model_version"] = ordered["model_version"].shift(1)
        ordered["previous_model_return"] = ordered["model_avg_return"].shift(1)
        prev_frames.append(ordered[["horizon_days", "model_version", "previous_model_version", "previous_model_return"]])
        if ordered["previous_model_version"].notna().sum() == 0:
            warnings.extend(
                make_warning_rows(
                    "baseline",
                    "WARN_PREVIOUS_MODEL_BASELINE_UNAVAILABLE",
                    "No previous model_version baseline available for this horizon",
                    horizon_days=int(horizon_days),
                )
            )
    previous_map = pd.concat(prev_frames, ignore_index=True) if prev_frames else pd.DataFrame()
    if previous_map.empty:
        merged["previous_model_version"] = pd.NA
        merged["previous_model_return"] = pd.NA
        return merged, warnings

    merged = merged.merge(previous_map, on=["horizon_days", "model_version"], how="left", suffixes=("", "_mapped"))
    if "previous_model_version_mapped" in merged.columns:
        merged["previous_model_version"] = merged["previous_model_version_mapped"]
        merged = merged.drop(columns=["previous_model_version_mapped"])
    if "previous_model_return_mapped" in merged.columns:
        merged["previous_model_return"] = merged["previous_model_return_mapped"]
        merged = merged.drop(columns=["previous_model_return_mapped"])

    null_count = int(merged["previous_model_return"].isna().sum())
    if null_count > 0:
        warnings.extend(
            make_warning_rows(
                "baseline",
                "WARN_PREVIOUS_MODEL_BASELINE_PARTIAL_NULL",
                f"Previous model_version baseline missing for {null_count} runs",
            )
        )
    return merged, warnings


def attach_baselines(
    run_df: pd.DataFrame,
    *,
    universe_baseline_csv: Path | None,
    benchmark_baseline_csv: Path | None,
    use_previous_model_baseline: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    enriched = run_df.copy()
    warnings: list[dict] = []

    universe_df, universe_warnings = load_optional_baseline_csv(
        universe_baseline_csv,
        baseline_name="universe",
        preferred_value_col="universe_baseline_return",
    )
    warnings.extend(universe_warnings)
    enriched, merge_warnings = merge_baseline_by_keys(
        enriched,
        universe_df,
        value_col="universe_baseline_return",
        output_col="universe_baseline_return",
        baseline_name="universe",
    )
    warnings.extend(merge_warnings)

    benchmark_df, benchmark_warnings = load_optional_baseline_csv(
        benchmark_baseline_csv,
        baseline_name="benchmark",
        preferred_value_col="benchmark_baseline_return",
    )
    warnings.extend(benchmark_warnings)
    enriched, merge_warnings = merge_baseline_by_keys(
        enriched,
        benchmark_df,
        value_col="benchmark_baseline_return",
        output_col="benchmark_baseline_return",
        baseline_name="benchmark",
    )
    warnings.extend(merge_warnings)

    if use_previous_model_baseline:
        enriched, prev_warnings = attach_previous_model_baseline(enriched)
        warnings.extend(prev_warnings)
    else:
        enriched["previous_model_version"] = pd.NA
        enriched["previous_model_return"] = pd.NA
        warnings.extend(
            make_warning_rows(
                "baseline",
                "WARN_PREVIOUS_MODEL_BASELINE_DISABLED",
                "Previous model_version baseline derivation is disabled",
            )
        )

    enriched["selected_baseline_name"] = pd.NA
    enriched["selected_baseline_return"] = pd.NA
    for baseline_name, value_col in [
        ("benchmark_index", "benchmark_baseline_return"),
        ("universe_average", "universe_baseline_return"),
        ("previous_model_version", "previous_model_return"),
    ]:
        mask = enriched["selected_baseline_return"].isna() & enriched[value_col].notna()
        enriched.loc[mask, "selected_baseline_name"] = baseline_name
        enriched.loc[mask, "selected_baseline_return"] = enriched.loc[mask, value_col]

    enriched["excess_return"] = pd.to_numeric(enriched["avg_realized_return"], errors="coerce") - pd.to_numeric(
        enriched["selected_baseline_return"], errors="coerce"
    )
    return enriched, warnings


def percentile_rank(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranked = s.rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.5).astype(float)


def assign_class_labels(
    df: pd.DataFrame,
    *,
    horizon_col: str,
    return_col: str,
    hit_ratio_col: str,
    topn_return_col: str,
    topn_hit_ratio_col: str,
    mdd_col: str,
    stability_col: str,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["class_label"] = pd.Series(dtype=object)
        return out

    frames: list[pd.DataFrame] = []
    for _, group in df.groupby(horizon_col, sort=True):
        scored = group.copy()

        ret_pct = percentile_rank(scored[return_col], ascending=True)
        hit_pct = percentile_rank(scored[hit_ratio_col], ascending=True)
        topn_ret_pct = percentile_rank(scored[topn_return_col], ascending=True)
        topn_hit_pct = percentile_rank(scored[topn_hit_ratio_col], ascending=True)
        mdd_good_pct = percentile_rank(scored[mdd_col], ascending=True)
        stability_pct = percentile_rank(scored[stability_col], ascending=False)

        scored["best_score"] = (
            0.45 * ret_pct
            + 0.20 * hit_pct
            + 0.20 * topn_ret_pct
            + 0.15 * topn_hit_pct
        )
        scored["stable_score"] = (
            0.35 * ret_pct
            + 0.35 * stability_pct
            + 0.30 * mdd_good_pct
        )
        scored["weak_score"] = (
            0.40 * (1.0 - ret_pct)
            + 0.20 * (1.0 - hit_pct)
            + 0.15 * (1.0 - topn_hit_pct)
            + 0.25 * (1.0 - mdd_good_pct)
        )

        best_gate = scored["best_score"].quantile(0.67)
        stable_gate = scored["stable_score"].quantile(0.50)
        weak_gate = scored["weak_score"].quantile(0.67)

        labels: list[str] = []
        for row in scored[["best_score", "stable_score", "weak_score"]].to_dict(orient="records"):
            if row["weak_score"] >= weak_gate and row["weak_score"] > row["best_score"] and row["weak_score"] >= row["stable_score"]:
                labels.append("weak")
            elif row["best_score"] >= best_gate and row["best_score"] >= row["stable_score"]:
                labels.append("best")
            elif row["stable_score"] >= stable_gate:
                labels.append("stable")
            else:
                labels.append(max(("best", row["best_score"]), ("stable", row["stable_score"]), ("weak", row["weak_score"]), key=lambda x: x[1])[0])

        scored["class_label"] = labels
        frames.append(scored)

    out = pd.concat(frames, ignore_index=True)
    return out


def build_candidate_selection(group_df: pd.DataFrame, *, min_runs_per_group: int) -> pd.DataFrame:
    columns = ["horizon_days", "comparison_group", "selection_label", "selection_reason"]
    if group_df.empty:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for _, group in group_df.groupby("horizon_days", sort=True):
        scored = group.copy()
        return_pct = percentile_rank(scored["group_avg_return"], ascending=True)
        hit_pct = percentile_rank(scored["group_avg_hit_ratio"], ascending=True)
        mdd_good_pct = percentile_rank(scored["group_avg_mdd"], ascending=True)
        stability_pct = percentile_rank(scored["group_std_return"], ascending=False)
        run_pct = percentile_rank(scored["runs_count"], ascending=True)
        excess_pct = percentile_rank(scored["group_avg_excess_return"], ascending=True)

        scored["production_score"] = (
            0.25 * run_pct
            + 0.25 * return_pct
            + 0.20 * hit_pct
            + 0.15 * mdd_good_pct
            + 0.15 * stability_pct
        )
        scored["drop_score"] = (
            0.35 * (1.0 - return_pct)
            + 0.20 * (1.0 - hit_pct)
            + 0.20 * (1.0 - mdd_good_pct)
            + 0.15 * (1.0 - stability_pct)
            + 0.10 * (1.0 - excess_pct)
        )

        production_gate = scored["production_score"].quantile(0.67)
        drop_gate = scored["drop_score"].quantile(0.67)
        median_runs = float(pd.to_numeric(scored["runs_count"], errors="coerce").median())

        labels: list[str] = []
        reasons: list[str] = []
        for row in scored.to_dict(orient="records"):
            low_sample = int(row["runs_count"]) < int(min_runs_per_group)
            weak_class = str(row.get("class_label", "")) == "weak"
            mixed_result = (
                str(row.get("class_label", "")) == "stable"
                or pd.isna(row.get("group_avg_excess_return"))
                or ("WARN_BENCHMARK_BASELINE_NULL" in str(row.get("warning", "")))
            )
            strong_production = (
                not low_sample
                and row["production_score"] >= production_gate
                and row["group_avg_return"] > 0
                and row["group_avg_hit_ratio"] >= 0.5
                and row["group_std_return"] <= float(pd.to_numeric(scored["group_std_return"], errors="coerce").median())
                and row["group_avg_mdd"] >= float(pd.to_numeric(scored["group_avg_mdd"], errors="coerce").median())
            )
            strong_drop = (
                (weak_class and row["drop_score"] >= drop_gate and row["group_avg_return"] <= 0)
                or (weak_class and row["group_avg_hit_ratio"] < 0.5 and row["group_avg_mdd"] < float(pd.to_numeric(scored["group_avg_mdd"], errors="coerce").quantile(0.33)))
            )

            if strong_production:
                labels.append("production_candidate")
                reasons.append("enough runs with strong return, hit ratio, and stable drawdown/volatility")
            elif low_sample or mixed_result or int(row["runs_count"]) <= median_runs:
                labels.append("research_candidate")
                reasons.append("sample is limited or results are mixed; keep for further validation")
            elif strong_drop:
                labels.append("drop_candidate")
                reasons.append("performance is persistently weak on return, hit ratio, and drawdown quality")
            elif weak_class:
                labels.append("drop_candidate")
                reasons.append("group remains weak versus peers and lacks enough evidence for promotion")
            else:
                labels.append("research_candidate")
                reasons.append("not strong enough for production and not weak enough for immediate drop")

        scored["selection_label"] = labels
        scored["selection_reason"] = reasons
        frames.append(scored)

    return pd.concat(frames, ignore_index=True)[columns]


def build_run_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "model_version",
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "avg_realized_return",
        "median_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "topn_hit_ratio",
        "universe_baseline_return",
        "benchmark_baseline_return",
        "previous_model_version",
        "previous_model_return",
        "selected_baseline_name",
        "selected_baseline_return",
        "excess_return",
        "class_label",
    ]
    if run_df.empty:
        return pd.DataFrame(columns=columns)

    classified = run_df.copy()
    if "median_realized_return" not in classified.columns:
        classified["median_realized_return"] = classified["avg_realized_return"]
    classified["stability_proxy"] = (
        (classified["avg_realized_return"] - classified["median_realized_return"]).abs()
    )
    classified = assign_class_labels(
        classified,
        horizon_col="horizon_days",
        return_col="avg_realized_return",
        hit_ratio_col="hit_ratio",
        topn_return_col="topn_avg_return",
        topn_hit_ratio_col="topn_hit_ratio",
        mdd_col="avg_realized_mdd",
        stability_col="stability_proxy",
    )
    return classified[columns].sort_values(["horizon_days", "run_id"]).reset_index(drop=True)


def build_group_summary(run_df: pd.DataFrame, *, min_runs_per_group: int) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "runs_count",
        "group_avg_return",
        "group_median_return",
        "group_std_return",
        "group_avg_mdd",
        "group_avg_hit_ratio",
        "group_avg_topn_return",
        "group_avg_topn_hit_ratio",
        "positive_run_ratio",
        "negative_run_ratio",
        "group_avg_excess_return",
        "group_median_excess_return",
        "benchmark_outperform_ratio",
        "is_primary_group",
        "is_candidate_group",
        "comparison_role",
        "class_label",
        "selection_label",
        "selection_reason",
        "warning",
    ]
    if run_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        run_df.groupby(["horizon_days", "score_formula_version", "comparison_group"], as_index=False)
        .agg(
            runs_count=("run_id", "size"),
            group_avg_return=("avg_realized_return", "mean"),
            group_median_return=("avg_realized_return", "median"),
            group_std_return=("avg_realized_return", lambda s: float(pd.Series(s).std(ddof=0))),
            group_avg_mdd=("avg_realized_mdd", "mean"),
            group_avg_hit_ratio=("hit_ratio", "mean"),
            group_avg_topn_return=("topn_avg_return", "mean"),
            group_avg_topn_hit_ratio=("topn_hit_ratio", "mean"),
            positive_run_ratio=("avg_realized_return", lambda s: float((pd.Series(s) > 0).mean())),
            negative_run_ratio=("avg_realized_return", lambda s: float((pd.Series(s) < 0).mean())),
            group_avg_excess_return=("excess_return", "mean"),
            group_median_excess_return=("excess_return", "median"),
            benchmark_outperform_ratio=("benchmark_baseline_return", lambda s: pd.NA),
        )
        .sort_values(["horizon_days", "runs_count", "group_avg_return", "comparison_group"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )

    benchmark_ratio_rows = []
    for (horizon_days, comparison_group), group in run_df.groupby(["horizon_days", "comparison_group"], sort=True):
        eligible = group[group["benchmark_baseline_return"].notna()].copy()
        ratio = pd.NA if eligible.empty else float((eligible["avg_realized_return"] > eligible["benchmark_baseline_return"]).mean())
        benchmark_ratio_rows.append(
            {
                "horizon_days": horizon_days,
                "comparison_group": comparison_group,
                "benchmark_outperform_ratio": ratio,
            }
        )
    if benchmark_ratio_rows:
        benchmark_ratio_df = pd.DataFrame(benchmark_ratio_rows)
        grouped = grouped.drop(columns=["benchmark_outperform_ratio"], errors="ignore").merge(
            benchmark_ratio_df,
            on=["horizon_days", "comparison_group"],
            how="left",
        )

    grouped["is_primary_group"] = False
    grouped["is_candidate_group"] = False
    grouped["comparison_role"] = "peer"
    grouped["warning"] = ""

    low_count_mask = grouped["runs_count"] < int(min_runs_per_group)
    grouped.loc[low_count_mask, "warning"] = f"WARN_LOW_RUNS_LT_{int(min_runs_per_group)}"

    for horizon_days, horizon_group in grouped.groupby("horizon_days", sort=True):
        eligible = horizon_group[horizon_group["runs_count"] >= int(min_runs_per_group)].copy()
        if eligible.empty:
            primary_idx = horizon_group.sort_values(
                ["runs_count", "group_avg_return", "comparison_group"],
                ascending=[False, False, True],
            ).index[0]
            grouped.loc[primary_idx, "is_primary_group"] = True
            grouped.loc[primary_idx, "comparison_role"] = "primary_reference"
            continue

        eligible_sorted = eligible.sort_values(
            ["runs_count", "group_avg_return", "positive_run_ratio", "comparison_group"],
            ascending=[False, False, False, True],
        )
        primary_idx = eligible_sorted.index[0]
        grouped.loc[primary_idx, "is_primary_group"] = True
        grouped.loc[primary_idx, "comparison_role"] = "primary"

        candidate_pool = eligible_sorted.drop(index=primary_idx)
        if not candidate_pool.empty:
            candidate_idx = candidate_pool.sort_values(
                ["group_avg_return", "positive_run_ratio", "runs_count", "comparison_group"],
                ascending=[False, False, False, True],
            ).index[0]
            grouped.loc[candidate_idx, "is_candidate_group"] = True
            grouped.loc[candidate_idx, "comparison_role"] = "candidate"

        reference_mask = (grouped["horizon_days"] == horizon_days) & (grouped["runs_count"] < int(min_runs_per_group))
        grouped.loc[reference_mask, "comparison_role"] = "reference_only"
        benchmark_null_mask = (grouped["horizon_days"] == horizon_days) & grouped["benchmark_outperform_ratio"].isna()
        grouped.loc[benchmark_null_mask & (grouped["warning"] == ""), "warning"] = "WARN_BENCHMARK_BASELINE_NULL"
        grouped.loc[
            benchmark_null_mask & (grouped["warning"] != "") & ~grouped["warning"].astype(str).str.contains("WARN_BENCHMARK_BASELINE_NULL", regex=False),
            "warning",
        ] = grouped.loc[
            benchmark_null_mask & (grouped["warning"] != "") & ~grouped["warning"].astype(str).str.contains("WARN_BENCHMARK_BASELINE_NULL", regex=False),
            "warning",
        ] + "|WARN_BENCHMARK_BASELINE_NULL"

    grouped = assign_class_labels(
        grouped,
        horizon_col="horizon_days",
        return_col="group_avg_return",
        hit_ratio_col="group_avg_hit_ratio",
        topn_return_col="group_avg_topn_return",
        topn_hit_ratio_col="group_avg_topn_hit_ratio",
        mdd_col="group_avg_mdd",
        stability_col="group_std_return",
    )
    selection_df = build_candidate_selection(grouped, min_runs_per_group=min_runs_per_group)
    grouped = grouped.merge(selection_df, on=["horizon_days", "comparison_group"], how="left")

    return grouped[columns].sort_values(
        ["horizon_days", "is_primary_group", "is_candidate_group", "runs_count", "group_avg_return", "comparison_group"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)


def build_formula_version_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "score_formula_version",
        "runs_count",
        "avg_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "positive_run_ratio",
    ]
    if run_df.empty:
        return pd.DataFrame(columns=columns)
    out = (
        run_df.groupby(["horizon_days", "score_formula_version"], as_index=False)
        .agg(
            runs_count=("run_id", "size"),
            avg_realized_return=("avg_realized_return", "mean"),
            avg_realized_mdd=("avg_realized_mdd", "mean"),
            hit_ratio=("hit_ratio", "mean"),
            topn_avg_return=("topn_avg_return", "mean"),
            positive_run_ratio=("avg_realized_return", lambda s: float((pd.Series(s) > 0).mean())),
        )
        .sort_values(["horizon_days", "runs_count", "avg_realized_return", "score_formula_version"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    return out[columns]


def build_formula_version_recommendations(formula_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "score_formula_version",
        "recommendation_type",
        "recommendation_reason",
    ]
    if formula_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for horizon_days, group in formula_df.groupby("horizon_days", sort=True):
        scored = group.copy()
        return_pct = percentile_rank(scored["avg_realized_return"], ascending=True)
        mdd_good_pct = percentile_rank(scored["avg_realized_mdd"], ascending=True)
        hit_pct = percentile_rank(scored["hit_ratio"], ascending=True)
        topn_pct = percentile_rank(scored["topn_avg_return"], ascending=True)
        positive_pct = percentile_rank(scored["positive_run_ratio"], ascending=True)
        run_pct = percentile_rank(scored["runs_count"], ascending=True)
        scored["operation_score"] = (
            0.25 * run_pct
            + 0.25 * return_pct
            + 0.20 * hit_pct
            + 0.15 * mdd_good_pct
            + 0.15 * positive_pct
        )
        scored["research_score"] = (
            0.35 * return_pct
            + 0.25 * topn_pct
            + 0.20 * positive_pct
            + 0.20 * (1.0 - run_pct)
        )

        op_idx = scored.sort_values(
            ["operation_score", "avg_realized_return", "hit_ratio", "score_formula_version"],
            ascending=[False, False, False, True],
        ).index[0]
        op_row = scored.loc[op_idx]
        rows.append(
            {
                "horizon_days": int(horizon_days),
                "score_formula_version": str(op_row["score_formula_version"]),
                "recommendation_type": "operation_candidate",
                "recommendation_reason": "best balance of return, hit ratio, drawdown quality, and sample size",
            }
        )

        research_pool = scored.drop(index=op_idx)
        if research_pool.empty:
            research_pool = scored.copy()
        research_idx = research_pool.sort_values(
            ["research_score", "topn_avg_return", "avg_realized_return", "score_formula_version"],
            ascending=[False, False, False, True],
        ).index[0]
        research_row = research_pool.loc[research_idx]
        rows.append(
            {
                "horizon_days": int(horizon_days),
                "score_formula_version": str(research_row["score_formula_version"]),
                "recommendation_type": "research_candidate",
                "recommendation_reason": "strong upside or top-n signal but still needs more validation versus the operation candidate",
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_horizon_overview(group_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_days",
        "groups_count",
        "eligible_groups_count",
        "reference_only_groups_count",
        "primary_comparison_group",
        "candidate_comparison_group",
    ]
    if group_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for horizon_days, group in group_df.groupby("horizon_days", sort=True):
        primary = group.loc[group["is_primary_group"], "comparison_group"]
        candidate = group.loc[group["is_candidate_group"], "comparison_group"]
        rows.append(
            {
                "horizon_days": int(horizon_days),
                "groups_count": int(len(group)),
                "eligible_groups_count": int((group["comparison_role"] != "reference_only").sum()),
                "reference_only_groups_count": int((group["comparison_role"] == "reference_only").sum()),
                "primary_comparison_group": primary.iloc[0] if not primary.empty else "",
                "candidate_comparison_group": candidate.iloc[0] if not candidate.empty else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.fillna("").astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return lines


def build_step4_completion_summary(
    run_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    *,
    min_runs_per_group: int,
) -> tuple[str, pd.DataFrame]:
    columns = ["criterion", "status", "evidence"]
    rows: list[dict[str, str]] = []

    matured_run_ready = not run_df.empty and not run_summary_df.empty
    rows.append(
        {
            "criterion": "matured run performance comparison",
            "status": "pass" if matured_run_ready else "fail",
            "evidence": (
                f"matured runs={len(run_df)}"
                if matured_run_ready
                else "no matured runs available for official comparison"
            ),
        }
    )

    eligible_groups = group_df[group_df["comparison_role"] != "reference_only"].copy() if not group_df.empty else pd.DataFrame()
    group_compare_ready = len(eligible_groups) >= 2
    rows.append(
        {
            "criterion": "comparison_group group comparison",
            "status": "pass" if group_compare_ready else "fail",
            "evidence": (
                f"eligible comparison groups={len(eligible_groups)} (required>=2, min_runs_per_group={min_runs_per_group})"
                if not group_df.empty
                else "no comparison groups available"
            ),
        }
    )

    observed_group_labels: list[str] = []
    observed_run_labels: list[str] = []
    if not group_df.empty and "class_label" in group_df.columns:
        observed_group_labels = sorted({label for label in group_df["class_label"].dropna().astype(str) if label})
    if not run_summary_df.empty and "class_label" in run_summary_df.columns:
        observed_run_labels = sorted({label for label in run_summary_df["class_label"].dropna().astype(str) if label})
    classification_ready = {"best", "stable", "weak"}.issubset(set(observed_group_labels) | set(observed_run_labels))
    rows.append(
        {
            "criterion": "best/stable/weak classification",
            "status": "pass" if classification_ready else "fail",
            "evidence": (
                f"run labels={','.join(observed_run_labels) if observed_run_labels else 'none'}; "
                f"group labels={','.join(observed_group_labels) if observed_group_labels else 'none'}"
            ),
        }
    )

    production_candidates = group_df[group_df["selection_label"] == "production_candidate"].copy() if not group_df.empty else pd.DataFrame()
    candidate_ready = not production_candidates.empty
    rows.append(
        {
            "criterion": "production candidate selection",
            "status": "pass" if candidate_ready else "fail",
            "evidence": (
                "selected groups="
                + ", ".join(production_candidates["comparison_group"].astype(str).tolist())
                if candidate_ready
                else "no production_candidate group selected"
            ),
        }
    )

    weak_groups = group_df[group_df["class_label"] == "weak"].copy() if not group_df.empty else pd.DataFrame()
    weak_exclusion_ready = False
    weak_evidence = "no weak group identified"
    if not weak_groups.empty:
        evidence_parts: list[str] = []
        for row in weak_groups.to_dict(orient="records"):
            reasons: list[str] = []
            if str(row.get("selection_label", "")) == "drop_candidate":
                reasons.append(f"selection={row['selection_label']}")
            if str(row.get("comparison_role", "")) == "reference_only":
                reasons.append("comparison_role=reference_only")
            warning = str(row.get("warning", "")).strip()
            if warning:
                reasons.append(f"warning={warning}")
            if reasons:
                weak_exclusion_ready = True
                evidence_parts.append(f"{row['comparison_group']} ({'; '.join(reasons)})")
        if evidence_parts:
            weak_evidence = "; ".join(evidence_parts)
        else:
            weak_evidence = "weak groups exist but no exclusion rationale was attached"
    rows.append(
        {
            "criterion": "weak group exclusion rationale",
            "status": "pass" if weak_exclusion_ready else "fail",
            "evidence": weak_evidence,
        }
    )

    overall = "COMPLETE" if all(row["status"] == "pass" for row in rows) else "PARTIAL"
    return overall, pd.DataFrame(rows, columns=columns)


def build_markdown_report(
    run_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    formula_df: pd.DataFrame,
    formula_60_df: pd.DataFrame,
    formula_rec_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    *,
    input_csv: Path,
    min_runs_per_group: int,
) -> str:
    step4_status, step4_summary_df = build_step4_completion_summary(
        run_df,
        run_summary_df,
        group_df,
        horizon_df,
        min_runs_per_group=min_runs_per_group,
    )
    lines = ["# Walk-Forward Group Comparison", ""]
    lines.append(f"- input_csv: `{input_csv.as_posix()}`")
    lines.append(f"- matured_runs: {len(run_df)}")
    lines.append(f"- comparison_groups: {len(group_df)}")
    lines.append(f"- min_runs_per_group: {min_runs_per_group}")
    lines.append(f"- step4_completion: {step4_status}")
    lines.append("")
    lines.append("## Step 4 Completion Check")
    lines.append("")
    lines.extend(dataframe_to_markdown(step4_summary_df))
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append("- Input is a run-level outcome summary CSV.")
    lines.append("- Official comparison uses matured runs only.")
    lines.append("- Aggregation is performed by `horizon_days` and `comparison_group`.")
    lines.append("- Groups with too few runs are marked `reference_only` with a warning.")
    lines.append("- `primary` is selected from eligible groups by highest `runs_count`, then higher `group_avg_return`.")
    lines.append("- `candidate` is the best non-primary eligible group by `group_avg_return`, then `positive_run_ratio`.")
    lines.append("- `class_label` is assigned for both run-level and group-level using horizon-local distribution scores.")
    lines.append("- `best` emphasizes higher return with stronger hit/top-n performance.")
    lines.append("- `stable` emphasizes return quality with lower volatility and lower MDD.")
    lines.append("- `weak` emphasizes weak return, weak hit ratio, and larger drawdown.")
    lines.append("- Run-level stability uses `abs(avg_realized_return - median_realized_return)` as a volatility proxy because the run summary lacks per-run return std.")
    lines.append("- `excess_return` uses baseline priority: benchmark index, then universe average, then previous model_version.")
    lines.append("")
    lines.append("## Horizon Overview")
    lines.append("")
    lines.extend(dataframe_to_markdown(horizon_df))
    lines.append("")
    lines.append("## 60d Formula Comparison")
    lines.append("")
    if formula_60_df.empty:
        lines.append("- none")
    else:
        lines.extend(dataframe_to_markdown(formula_60_df))
    lines.append("")
    lines.append("## Formula Version Summary")
    lines.append("")
    if formula_df.empty:
        lines.append("- none")
    else:
        lines.extend(dataframe_to_markdown(formula_df))
    lines.append("")
    lines.append("## Formula Version Recommendation")
    lines.append("")
    if formula_rec_df.empty:
        lines.append("- none")
    else:
        lines.extend(dataframe_to_markdown(formula_rec_df))
    lines.append("")
    lines.append("## Run Absolute Performance")
    lines.append("")
    run_absolute_cols = [
        "run_id",
        "model_version",
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "avg_realized_return",
        "median_realized_return",
        "avg_realized_mdd",
        "hit_ratio",
        "topn_avg_return",
        "topn_hit_ratio",
        "class_label",
    ]
    lines.extend(dataframe_to_markdown(run_summary_df[run_absolute_cols]))
    lines.append("")
    lines.append("## Run Excess Performance")
    lines.append("")
    run_excess_cols = [
        "run_id",
        "model_version",
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "selected_baseline_name",
        "selected_baseline_return",
        "universe_baseline_return",
        "benchmark_baseline_return",
        "previous_model_version",
        "previous_model_return",
        "excess_return",
    ]
    lines.extend(dataframe_to_markdown(run_summary_df[run_excess_cols]))
    lines.append("")
    lines.append("## Group Absolute Performance")
    lines.append("")
    group_absolute_cols = [
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "runs_count",
        "group_avg_return",
        "group_median_return",
        "group_std_return",
        "group_avg_mdd",
        "group_avg_hit_ratio",
        "group_avg_topn_return",
        "group_avg_topn_hit_ratio",
        "positive_run_ratio",
        "negative_run_ratio",
        "comparison_role",
        "class_label",
        "selection_label",
        "warning",
    ]
    lines.extend(dataframe_to_markdown(group_df[group_absolute_cols]))
    lines.append("")
    lines.append("## Group Excess Performance")
    lines.append("")
    group_excess_cols = [
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "runs_count",
        "group_avg_excess_return",
        "group_median_excess_return",
        "benchmark_outperform_ratio",
        "comparison_role",
        "warning",
    ]
    lines.extend(dataframe_to_markdown(group_df[group_excess_cols]))
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings_df.empty:
        lines.append("- none")
    else:
        lines.extend(dataframe_to_markdown(warnings_df))
    lines.append("")
    lines.append("## Candidate Selection")
    lines.append("")
    candidate_cols = [
        "horizon_days",
        "score_formula_version",
        "comparison_group",
        "runs_count",
        "group_avg_return",
        "group_avg_hit_ratio",
        "group_avg_mdd",
        "group_std_return",
        "group_avg_excess_return",
        "class_label",
        "selection_label",
        "selection_reason",
        "warning",
    ]
    lines.extend(dataframe_to_markdown(group_df[candidate_cols]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    args = parse_args()

    run_df = load_run_level_summary(args.input_csv)
    horizon_filter = parse_horizon_filter(args.horizon_days)
    if horizon_filter:
        run_df = run_df[run_df["horizon_days"].isin(horizon_filter)].copy()

    run_df, baseline_warnings = attach_baselines(
        run_df,
        universe_baseline_csv=args.universe_baseline_csv,
        benchmark_baseline_csv=args.benchmark_baseline_csv,
        use_previous_model_baseline=not args.disable_previous_model_baseline,
    )
    run_summary_df = build_run_summary(run_df)
    group_df = build_group_summary(run_df, min_runs_per_group=args.min_runs_per_group)
    horizon_df = build_horizon_overview(group_df)
    formula_df = build_formula_version_summary(run_df)
    formula_60_df = formula_df[formula_df["horizon_days"] == 60].reset_index(drop=True) if not formula_df.empty else formula_df
    formula_rec_base = formula_60_df if not formula_60_df.empty else formula_df
    formula_rec_df = build_formula_version_recommendations(formula_rec_base)
    warnings_df = pd.DataFrame(
        baseline_warnings,
        columns=["scope", "horizon_days", "warning_code", "warning_message"],
    ).drop_duplicates().reset_index(drop=True)

    if not group_df.empty:
        low_count = group_df[group_df["warning"].astype(str).str.contains("WARN_LOW_RUNS", regex=False, na=False)]
        if not low_count.empty:
            logging.warning(
                "Found %d low-count comparison groups below min_runs_per_group=%d",
                len(low_count),
                args.min_runs_per_group,
            )
    if not warnings_df.empty:
        for row in warnings_df.to_dict(orient="records"):
            logging.warning("%s: %s", row["warning_code"], row["warning_message"])

    args.out_run_csv.parent.mkdir(parents=True, exist_ok=True)
    run_summary_df.to_csv(args.out_run_csv, index=False, encoding="utf-8")
    logging.info("Saved run-level classified CSV: %s", args.out_run_csv.resolve())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    logging.info("Saved comparison-group CSV: %s", args.out_csv.resolve())

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_markdown_report(
            run_df,
            run_summary_df,
            group_df,
            horizon_df,
            formula_df,
            formula_60_df,
            formula_rec_df,
            warnings_df,
            input_csv=args.input_csv,
            min_runs_per_group=args.min_runs_per_group,
        ),
        encoding="utf-8",
    )
    logging.info("Saved comparison-group markdown: %s", args.out_md.resolve())

    args.out_warnings_csv.parent.mkdir(parents=True, exist_ok=True)
    warnings_df.to_csv(args.out_warnings_csv, index=False, encoding="utf-8")
    logging.info("Saved warnings CSV: %s", args.out_warnings_csv.resolve())

    if not horizon_df.empty:
        print(horizon_df.to_string(index=False))
    if not run_summary_df.empty:
        print(run_summary_df.to_string(index=False))
    if not group_df.empty:
        print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
