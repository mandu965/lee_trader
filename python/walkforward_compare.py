"""
Shared comparison-group utilities for walk-forward outcome coverage/reporting.

Official performance comparison must use matured runs only. This module turns
run metadata into a like-for-like comparison key and derives:
- comparison_group
- comparison_group_runs
- is_primary_group
- comparison_warning

The primary group is the matured comparison group with the largest run count.
Groups with fewer than `min_runs_per_group` matured runs are kept as reference
only and flagged with a warning.
"""
import json

import pandas as pd


COMPARISON_GROUP_FIELDS = [
    "model_version",
    "horizon_days",
    "top_n",
    "rebalance_freq",
    "universe_version",
    "score_formula_version",
]


def decode_config_json(value) -> dict:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        obj = json.loads(value)
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("config_json must decode to an object")
        return obj
    raise ValueError(f"Unsupported config_json type: {type(value)}")


def extract_comparison_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "model_version",
                "horizon_days",
                "top_n",
                "rebalance_freq",
                "universe_version",
                "score_formula_version",
                "comparison_group",
            ]
        )

    def extract_row(row: pd.Series) -> pd.Series:
        cfg = decode_config_json(row.get("config_json"))
        return pd.Series(
            {
                "rebalance_freq": cfg.get("rebalance_freq"),
                "universe_version": cfg.get("universe_version"),
                "score_formula_version": cfg.get("score_formula_version"),
            }
        )

    meta = metadata.copy()
    extra = meta.apply(extract_row, axis=1)
    meta = pd.concat([meta.drop(columns=["config_json"], errors="ignore"), extra], axis=1)
    meta["comparison_group"] = meta[COMPARISON_GROUP_FIELDS].fillna("NA").astype(str).agg("|".join, axis=1)
    return meta


def annotate_comparison_groups(
    runs: pd.DataFrame,
    *,
    min_runs_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annotate run rows with comparison-group metadata.

    Group counts and primary-group selection are based on matured runs only,
    because only matured runs are eligible for official performance comparison.
    """
    if runs.empty:
        empty_runs = runs.copy()
        for col in ["comparison_group_runs", "is_primary_group", "comparison_warning"]:
            empty_runs[col] = pd.Series(dtype=object)
        empty_groups = pd.DataFrame(
            columns=["comparison_group", "comparison_group_runs", "is_primary_group", "comparison_warning"]
        )
        return empty_runs, empty_groups

    annotated = runs.copy()
    matured = annotated[annotated["run_status"] == "matured"].copy()

    if matured.empty:
        annotated["comparison_group_runs"] = 0
        annotated["is_primary_group"] = False
        annotated["comparison_warning"] = "WARN_NO_MATURED_COMPARISON_GROUP"
        group_summary = (
            annotated[["comparison_group"]]
            .drop_duplicates()
            .assign(
                comparison_group_runs=0,
                is_primary_group=False,
                comparison_warning="WARN_NO_MATURED_COMPARISON_GROUP",
            )
            .reset_index(drop=True)
        )
        return annotated, group_summary

    group_counts = matured["comparison_group"].value_counts().sort_values(ascending=False)
    primary_group = sorted(group_counts[group_counts == group_counts.max()].index.tolist())[0]

    annotated["comparison_group_runs"] = annotated["comparison_group"].map(group_counts).fillna(0).astype(int)
    annotated["is_primary_group"] = annotated["comparison_group"] == primary_group
    annotated["comparison_warning"] = ""
    low_count_mask = annotated["comparison_group_runs"] < int(min_runs_per_group)
    annotated.loc[low_count_mask, "comparison_warning"] = (
        f"WARN_REFERENCE_ONLY_GROUP_LT_{int(min_runs_per_group)}"
    )

    group_summary = (
        annotated[["comparison_group", "comparison_group_runs", "is_primary_group", "comparison_warning"]]
        .drop_duplicates()
        .sort_values(["comparison_group_runs", "comparison_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return annotated, group_summary
