from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"

PIPELINE_STEPS = [
    ("theme_etf_daily", [sys.executable, "python/compute_theme_etf_daily.py"]),
    ("stock_theme_daily", [sys.executable, "python/build_stock_theme_daily.py"]),
    ("ranking_builder", [sys.executable, "python/ranking_builder.py"]),
    ("final_analysis", [sys.executable, "python/build_theme_overlay_final_analysis.py"]),
    ("driver_report", [sys.executable, "python/build_top20_vs_near_top20_driver_report.py"]),
    ("validation_check", [sys.executable, "python/check_theme_overlay_validation.py"]),
]

REQUIRED_OUTPUTS: dict[Path, dict[str, Any]] = {
    DATA_DIR / "top20_before_after_compare_v3.csv": {
        "kind": "csv",
        "required_columns": {
            "date",
            "code",
            "name",
            "base_rank",
            "v2_rank",
            "v3_rank",
            "dominant_theme",
            "theme_confidence",
            "score_diff_v3",
            "in_base_top20",
            "in_v3_top20",
            "explain_ko",
            "explain_en",
        },
        "min_rows": 20,
    },
    DATA_DIR / "theme_overlay_acceptance_summary.md": {
        "kind": "markdown",
        "required_text": [
            "decision_status:",
            "top20_churn_count:",
            "no_theme_displaced_count:",
            "top1_theme_count:",
            "top2_theme_count:",
        ],
    },
    DATA_DIR / "no_theme_displacement_report.md": {
        "kind": "markdown",
        "required_text": [
            "displaced_no_theme_count:",
            "warning_threshold:",
            "English Summary",
        ],
    },
    DATA_DIR / "theme_concentration_report.csv": {
        "kind": "csv",
        "required_columns": {
            "theme_label",
            "stock_count",
            "top20_share",
            "concentration_flag",
            "explain_ko",
            "explain_en",
        },
        "min_rows": 1,
    },
    DATA_DIR / "near_top20_theme_lift_report.csv": {
        "kind": "csv",
        "required_columns": {
            "date",
            "code",
            "name",
            "dominant_theme",
            "base_rank",
            "v3_rank",
            "lift_size",
            "theme_confidence",
            "explain_ko",
            "explain_en",
        },
        "min_rows": 0,
    },
}

LOGGER = logging.getLogger("run_theme_overlay_validation")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_step(step_name: str, command: list[str]) -> None:
    started_at = time.time()
    LOGGER.info("[validation] step_start name=%s command=%s", step_name, " ".join(command))
    subprocess.run(command, check=True)
    LOGGER.info("[validation] step_success name=%s elapsed_sec=%.2f", step_name, time.time() - started_at)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"code": str}, low_memory=False)


def _extract_metric_from_markdown(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(f"- {key}:"):
            return line.split(":", 1)[1].strip()
    return ""


def validate_outputs(min_mtime: float) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    acceptance_metrics: dict[str, str] = {}

    for path, spec in REQUIRED_OUTPUTS.items():
        exists = path.exists()
        updated_after_run = bool(exists and path.stat().st_mtime >= min_mtime)
        size = path.stat().st_size if exists else 0
        row: dict[str, Any] = {
            "path": str(path),
            "exists": exists,
            "updated_after_run": updated_after_run,
            "size": size,
            "kind": spec["kind"],
            "row_count": None,
            "column_check": True,
            "content_check": True,
        }
        if not exists:
            rows.append(row)
            continue

        if spec["kind"] == "csv":
            df = _load_csv(path)
            row["row_count"] = int(len(df))
            required_columns = set(spec.get("required_columns", set()))
            row["column_check"] = required_columns.issubset(set(df.columns))
            min_rows = int(spec.get("min_rows", 0))
            row["content_check"] = int(len(df)) >= min_rows
        else:
            text = path.read_text(encoding="utf-8")
            required_text = spec.get("required_text", [])
            row["content_check"] = all(token in text for token in required_text)
            if path.name == "theme_overlay_acceptance_summary.md":
                for key in [
                    "decision_status",
                    "latest_date",
                    "top20_churn_count",
                    "no_theme_displaced_count",
                    "near_top20_theme_entries",
                    "top1_theme_count",
                    "top2_theme_count",
                ]:
                    acceptance_metrics[key] = _extract_metric_from_markdown(text, key)

        rows.append(row)

    status_df = pd.DataFrame(rows)
    missing = status_df.loc[~status_df["exists"], "path"].tolist()
    stale = status_df.loc[status_df["exists"] & ~status_df["updated_after_run"], "path"].tolist()
    bad_schema = status_df.loc[status_df["exists"] & ~status_df["column_check"], "path"].tolist()
    bad_content = status_df.loc[status_df["exists"] & ~status_df["content_check"], "path"].tolist()
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")
    if stale:
        raise RuntimeError(f"Outputs not refreshed by current run: {stale}")
    if bad_schema:
        raise RuntimeError(f"CSV schema validation failed: {bad_schema}")
    if bad_content:
        raise RuntimeError(f"Output content validation failed: {bad_content}")
    return status_df, acceptance_metrics


def build_runtime_summary(status_df: pd.DataFrame, acceptance_metrics: dict[str, str]) -> list[str]:
    top20_df = _load_csv(DATA_DIR / "top20_before_after_compare_v3.csv")
    concentration_df = _load_csv(DATA_DIR / "theme_concentration_report.csv")
    near_lift_df = _load_csv(DATA_DIR / "near_top20_theme_lift_report.csv")
    acceptance_status = acceptance_metrics.get("decision_status", "UNKNOWN")
    churn_count = acceptance_metrics.get("top20_churn_count", "NA")
    no_theme_displaced = acceptance_metrics.get("no_theme_displaced_count", "NA")
    near_lift_count = acceptance_metrics.get("near_top20_theme_entries", str(len(near_lift_df)))
    top20_union_count = len(top20_df)
    top20_enter_count = int((~top20_df["in_base_top20"].fillna(False) & top20_df["in_v3_top20"].fillna(False)).sum())
    top20_exit_count = int((top20_df["in_base_top20"].fillna(False) & ~top20_df["in_v3_top20"].fillna(False)).sum())

    actual_theme_concentration = concentration_df[concentration_df["theme_label"].astype(str) != "(none)"].copy()
    if actual_theme_concentration.empty:
        top_theme_label = "(none)"
        top_theme_count = 0
        top_theme_share = 0.0
    else:
        top_theme_row = actual_theme_concentration.sort_values(
            ["stock_count", "top20_share", "theme_label"], ascending=[False, False, True]
        ).iloc[0]
        top_theme_label = str(top_theme_row["theme_label"])
        top_theme_count = int(top_theme_row["stock_count"])
        top_theme_share = float(top_theme_row["top20_share"])

    lines = [
        f"decision_status={acceptance_status}",
        f"top20_churn_count={churn_count}",
        f"no_theme_displaced_count={no_theme_displaced}",
        f"near_top20_theme_entries={near_lift_count}",
        f"top20_before_after_union_rows={top20_union_count}",
        f"top20_enter_count={top20_enter_count}",
        f"top20_exit_count={top20_exit_count}",
        f"top_theme_label={top_theme_label}",
        f"top_theme_count={top_theme_count}",
        f"top_theme_share={top_theme_share:.2%}",
        "summary_ko="
        + (
            f"판정은 {acceptance_status}이며 top20 churn={churn_count}, no-theme 이탈={no_theme_displaced}, "
            f"near-top20 신규 진입={near_lift_count}, 실제 테마 기준 상위 집중도는 {top_theme_label} {top_theme_count}종목이다."
        ),
        "summary_en="
        + (
            f"Decision={acceptance_status}; top20 churn={churn_count}, no-theme displacement={no_theme_displaced}, "
            f"near-top20 theme entries={near_lift_count}, and the largest actual theme concentration is "
            f"{top_theme_label} with {top_theme_count} names."
        ),
        "validated_outputs=" + str(status_df.to_dict(orient="records")),
    ]
    return lines


def main() -> None:
    setup_logging()
    started_at = time.time()
    print("run_order=" + str([name for name, _ in PIPELINE_STEPS]))
    for step_name, command in PIPELINE_STEPS:
        run_step(step_name, command)

    status_df, acceptance_metrics = validate_outputs(started_at)
    summary_lines = build_runtime_summary(status_df, acceptance_metrics)
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
