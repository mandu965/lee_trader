"""Persist daily theme shadow monitoring metrics into date-keyed history.

This script parses the latest theme overlay reports, extracts a compact set of
operational metrics, saves raw daily snapshots, and upserts one row per date
into a cumulative CSV history for shadow monitoring.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
HISTORY_CSV_PATH = DATA_DIR / "theme_shadow_monitor_history.csv"
SNAPSHOT_DIR = DATA_DIR / "history" / "theme_shadow"

ACCEPTANCE_REPORT_PATH = DATA_DIR / "theme_overlay_acceptance_report.md"
THEME_GUARD_REPORT_PATH = DATA_DIR / "ranking_builder_theme_guard_report.md"
THEME_LIFT_CSV_PATH = DATA_DIR / "theme_lift_analysis.csv"
PROMOTION_DECISION_PATH = DATA_DIR / "theme_promotion_decision.json"

HISTORY_COLUMNS = [
    "as_of_date",
    "acceptance_fail_count",
    "theme_concentration_status",
    "none_ratio",
    "near_top20_entry_quality_status",
    "coverage_ratio_guard",
    "theme_lift_status",
    "theme_lift_avg",
    "theme_lift_max",
    "daily_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist daily theme shadow metrics and evaluate promotion readiness.")
    parser.add_argument(
        "--evaluate-promotion",
        action="store_true",
        help="Evaluate promotion decision from existing theme shadow history only.",
    )
    return parser.parse_args()


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_regex(text: str, pattern: str, field_name: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    if not match:
        raise ValueError(f"Failed to parse field '{field_name}' with pattern: {pattern}")
    return match.group(1).strip()


def _extract_section_status(text: str, section_title: str, field_name: str) -> str:
    pattern = rf"##\s+{re.escape(section_title)}.*?(?:\r?\n)+-\s+status:\s*([A-Z]+)"
    return _extract_regex(text, pattern, field_name, flags=re.DOTALL)


def _parse_percent(raw: str, field_name: str) -> float:
    cleaned = str(raw).replace("%", "").strip()
    try:
        return float(cleaned) / 100.0
    except ValueError as exc:
        raise ValueError(f"Failed to parse percentage field '{field_name}': {raw}") from exc


def parse_acceptance_report(report_path: Path) -> dict[str, object]:
    """Parse the acceptance markdown report into core shadow metrics."""
    text = _read_text(report_path)
    try:
        as_of_date = _extract_regex(
            text,
            r"-\s+latest_ranking_date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            "latest_ranking_date",
        )
        metric_statuses_raw = _extract_regex(
            text,
            r"-\s+metric_statuses:\s*(\[[^\n]+\])",
            "metric_statuses",
        )
        acceptance_fail_count = len(re.findall(r"'FAIL'|\"FAIL\"", metric_statuses_raw))
        theme_concentration_status = _extract_section_status(
            text,
            "3. Theme Concentration",
            "theme_concentration_status",
        )
        none_ratio = _parse_percent(
            _extract_regex(text, r"-\s+max_share:\s*([0-9.]+%)", "max_share"),
            "max_share",
        )
        near_top20_entry_quality_status = _extract_section_status(
            text,
            "4. Near-Top20 Entry Quality",
            "near_top20_entry_quality_status",
        )
        theme_lift_status = _extract_section_status(
            text,
            "5. Theme Lift Effect",
            "theme_lift_status",
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Acceptance report parsing failed: {exc}") from exc

    return {
        "as_of_date": as_of_date,
        "acceptance_fail_count": acceptance_fail_count,
        "theme_concentration_status": theme_concentration_status,
        "none_ratio": none_ratio,
        "near_top20_entry_quality_status": near_top20_entry_quality_status,
        "theme_lift_status": theme_lift_status,
    }


def parse_theme_guard_report(report_path: Path) -> dict[str, object]:
    """Parse guard coverage from the theme guard markdown report."""
    text = _read_text(report_path)
    try:
        coverage_ratio_guard_raw = _extract_regex(
            text,
            r"-\s+coverage_ratio_guard:\s*([0-9.]+)",
            "coverage_ratio_guard",
        )
        coverage_ratio_guard = float(coverage_ratio_guard_raw)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Theme guard report parsing failed: {exc}") from exc

    return {
        "coverage_ratio_guard": coverage_ratio_guard,
    }


def parse_theme_lift_csv(csv_path: Path) -> dict[str, object]:
    """Parse aggregate theme lift statistics from the lift analysis CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Required file is missing: {csv_path}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Theme lift CSV parsing failed: {exc}") from exc

    if "score_delta_v3" not in df.columns:
        raise ValueError("Theme lift CSV parsing failed: missing required column 'score_delta_v3'")

    delta = pd.to_numeric(df["score_delta_v3"], errors="coerce").dropna()
    if delta.empty:
        return {
            "theme_lift_avg": 0.0,
            "theme_lift_max": 0.0,
        }

    return {
        "theme_lift_avg": float(delta.mean()),
        "theme_lift_max": float(delta.max()),
    }


def upsert_history(history_csv_path: Path, row_dict: dict[str, object], key_field: str = "as_of_date") -> None:
    """Update an existing date row or append a new one to the history CSV."""
    if history_csv_path.exists():
        try:
            history_df = pd.read_csv(history_csv_path, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Failed to read history CSV: {exc}") from exc
    else:
        history_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = pd.NA

    incoming_df = pd.DataFrame([row_dict], columns=HISTORY_COLUMNS)
    history_df[key_field] = history_df[key_field].astype(str)
    incoming_key = str(row_dict[key_field])
    history_df = history_df[history_df[key_field] != incoming_key].copy()
    if history_df.empty:
        history_df = incoming_df.copy()
    else:
        history_df = pd.concat([history_df, incoming_df], ignore_index=True)
    history_df = history_df.sort_values(by=key_field).reset_index(drop=True)
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")


def save_daily_snapshots(as_of_date: str, source_files: dict[str, Path]) -> None:
    """Copy raw source reports into the dated history snapshot folder."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    date_key = str(as_of_date).replace("-", "")

    snapshot_targets = {
        "acceptance_report": SNAPSHOT_DIR / f"{date_key}_acceptance_report.md",
        "theme_guard_report": SNAPSHOT_DIR / f"{date_key}_theme_guard_report.md",
        "theme_lift_analysis": SNAPSHOT_DIR / f"{date_key}_theme_lift_analysis.csv",
    }

    for key, source_path in source_files.items():
        if key not in snapshot_targets:
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Snapshot source file is missing: {source_path}")
        shutil.copy2(source_path, snapshot_targets[key])


def summarize_daily_status(row_dict: dict[str, object]) -> str:
    """Reduce the parsed row into PASS/WARN/FAIL for daily monitoring."""
    fail_count = int(row_dict.get("acceptance_fail_count", 0))
    concentration_status = str(row_dict.get("theme_concentration_status", "")).upper()
    entry_status = str(row_dict.get("near_top20_entry_quality_status", "")).upper()
    lift_status = str(row_dict.get("theme_lift_status", "")).upper()
    none_ratio = float(row_dict.get("none_ratio", 0.0))
    coverage = float(row_dict.get("coverage_ratio_guard", 0.0))

    if fail_count > 0 or "FAIL" in {concentration_status, entry_status, lift_status}:
        return "FAIL"
    if none_ratio > 0.80:
        return "WARN"
    if coverage < 0.50:
        return "WARN"
    if lift_status == "WARN":
        return "WARN"
    return "PASS"


def load_history(history_csv_path: Path) -> pd.DataFrame:
    """Load history CSV defensively and normalize the date column."""
    if not history_csv_path.exists():
        raise FileNotFoundError(f"Required file is missing: {history_csv_path}")

    try:
        history_df = pd.read_csv(history_csv_path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Failed to read history CSV: {exc}") from exc

    if history_df.empty:
        raise ValueError(f"History CSV is empty: {history_csv_path}")

    if "as_of_date" not in history_df.columns:
        raise ValueError("History CSV parsing failed: missing required column 'as_of_date'")

    history_df["as_of_date"] = history_df["as_of_date"].astype(str)
    history_df = history_df.sort_values(by="as_of_date").reset_index(drop=True)
    return history_df


def evaluate_promotion(history_df: pd.DataFrame, window_days: int = 5) -> dict[str, object]:
    """Evaluate promotion readiness from the recent shadow-monitor history window."""
    if history_df.empty:
        return {
            "decision": "HOLD",
            "window_days": window_days,
            "checked_dates": [],
            "reasons": ["history file is empty"],
            "summary": {
                "max_none_ratio": None,
                "min_coverage_ratio_guard": None,
                "fail_days": 0,
            },
        }

    recent_df = history_df.tail(window_days).copy()
    checked_dates = recent_df["as_of_date"].astype(str).tolist()
    reasons: list[str] = []

    def _numeric_series(column: str, default: float) -> pd.Series:
        if column not in recent_df.columns:
            return pd.Series([default] * len(recent_df), index=recent_df.index, dtype="float64")
        return pd.to_numeric(recent_df[column], errors="coerce").fillna(default)

    def _status_series(column: str, default: str = "") -> pd.Series:
        if column not in recent_df.columns:
            return pd.Series([default] * len(recent_df), index=recent_df.index, dtype="object")
        return recent_df[column].astype(str).str.upper().fillna(default)

    none_ratio_series = _numeric_series("none_ratio", 1.0)
    coverage_series = _numeric_series("coverage_ratio_guard", 0.0)
    fail_count_series = _numeric_series("acceptance_fail_count", 1.0)
    daily_status_series = _status_series("daily_status", "FAIL")
    concentration_series = _status_series("theme_concentration_status")
    entry_quality_series = _status_series("near_top20_entry_quality_status")
    lift_status_series = _status_series("theme_lift_status")

    fail_days = int(daily_status_series.eq("FAIL").sum())
    summary = {
        "max_none_ratio": float(none_ratio_series.max()) if not none_ratio_series.empty else None,
        "min_coverage_ratio_guard": float(coverage_series.min()) if not coverage_series.empty else None,
        "fail_days": fail_days,
    }

    if len(recent_df) < window_days:
        reasons.append(f"history has fewer than {window_days} valid business-day observations")
        return {
            "decision": "HOLD",
            "window_days": window_days,
            "checked_dates": checked_dates,
            "reasons": reasons,
            "summary": summary,
        }

    if daily_status_series.eq("FAIL").any():
        fail_dates = recent_df.loc[daily_status_series.eq("FAIL"), "as_of_date"].astype(str).tolist()
        reasons.append(f"daily_status reached FAIL on {', '.join(fail_dates)}")
        return {
            "decision": "ROLLBACK",
            "window_days": window_days,
            "checked_dates": checked_dates,
            "reasons": reasons,
            "summary": summary,
        }

    optional_fail_columns = {
        "stale_guard_status": "stale_guard_status reached FAIL",
        "explain_quality_status": "explain_quality_status reached FAIL",
    }
    for column, label in optional_fail_columns.items():
        if column not in recent_df.columns:
            continue
        status_series = _status_series(column)
        if status_series.eq("FAIL").any():
            fail_dates = recent_df.loc[status_series.eq("FAIL"), "as_of_date"].astype(str).tolist()
            reasons.append(f"{label} on {', '.join(fail_dates)}")
            return {
                "decision": "ROLLBACK",
                "window_days": window_days,
                "checked_dates": checked_dates,
                "reasons": reasons,
                "summary": summary,
            }
        if status_series.eq("WARN").any():
            warn_dates = recent_df.loc[status_series.eq("WARN"), "as_of_date"].astype(str).tolist()
            reasons.append(f"{column} was WARN on {', '.join(warn_dates)}")

    if none_ratio_series.gt(0.80).any():
        warn_dates = recent_df.loc[none_ratio_series.gt(0.80), "as_of_date"].astype(str).tolist()
        reasons.append(f"none_ratio exceeded 0.80 on {', '.join(warn_dates)}")
        return {
            "decision": "HOLD",
            "window_days": window_days,
            "checked_dates": checked_dates,
            "reasons": reasons,
            "summary": summary,
        }

    promote_reasons: list[str] = []
    if not fail_count_series.eq(0).all():
        promote_reasons.append("acceptance_fail_count was not zero for all checked dates")
    if not concentration_series.eq("PASS").all():
        promote_reasons.append("theme_concentration_status was not PASS for all checked dates")
    if not entry_quality_series.eq("PASS").all():
        promote_reasons.append("near_top20_entry_quality_status was not PASS for all checked dates")
    if not coverage_series.ge(0.50).all():
        promote_reasons.append("coverage_ratio_guard fell below 0.50 within the checked window")
    if lift_status_series.eq("FAIL").any():
        promote_reasons.append("theme_lift_status reached FAIL within the checked window")

    if promote_reasons:
        reasons.extend(promote_reasons)
        return {
            "decision": "HOLD",
            "window_days": window_days,
            "checked_dates": checked_dates,
            "reasons": reasons,
            "summary": summary,
        }

    reasons.append("all promotion checks passed for the recent 5 observations")
    return {
        "decision": "PROMOTE",
        "window_days": window_days,
        "checked_dates": checked_dates,
        "reasons": reasons,
        "summary": summary,
    }


def write_promotion_decision(output_path: Path, decision_payload: dict[str, object]) -> None:
    """Write the promotion decision payload as UTF-8 JSON."""
    output_path.write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """Parse reports, upsert history, save snapshots, and print a short summary."""
    args = parse_args()

    if args.evaluate_promotion:
        try:
            history_df = load_history(HISTORY_CSV_PATH)
            decision_payload = evaluate_promotion(history_df, window_days=5)
            write_promotion_decision(PROMOTION_DECISION_PATH, decision_payload)
        except FileNotFoundError as exc:
            print(f"FILE_ERROR: {exc}")
            return 1
        except ValueError as exc:
            print(f"PARSE_ERROR: {exc}")
            return 1
        except Exception as exc:
            print(f"WRITE_ERROR: {exc}")
            return 1

        print(f"DECISION={decision_payload['decision']}")
        print(f"CHECKED_DATES={decision_payload['checked_dates']}")
        for reason in decision_payload.get("reasons", []):
            print(f"REASON={reason}")
        return 0

    source_files = {
        "acceptance_report": ACCEPTANCE_REPORT_PATH,
        "theme_guard_report": THEME_GUARD_REPORT_PATH,
        "theme_lift_analysis": THEME_LIFT_CSV_PATH,
    }

    try:
        acceptance_metrics = parse_acceptance_report(ACCEPTANCE_REPORT_PATH)
        guard_metrics = parse_theme_guard_report(THEME_GUARD_REPORT_PATH)
        lift_metrics = parse_theme_lift_csv(THEME_LIFT_CSV_PATH)
    except FileNotFoundError as exc:
        print(f"FILE_ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"PARSE_ERROR: {exc}")
        return 1

    row_dict = {
        **acceptance_metrics,
        **guard_metrics,
        **lift_metrics,
    }
    row_dict["daily_status"] = summarize_daily_status(row_dict)

    try:
        upsert_history(HISTORY_CSV_PATH, row_dict, key_field="as_of_date")
        save_daily_snapshots(str(row_dict["as_of_date"]), source_files)
    except FileNotFoundError as exc:
        print(f"FILE_ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"WRITE_ERROR: {exc}")
        return 1

    print(f"[{row_dict['as_of_date']}]")
    print(f"FAIL={int(row_dict['acceptance_fail_count'])}")
    print(f"NONE_RATIO={float(row_dict['none_ratio']):.2f}")
    print(f"COVERAGE={float(row_dict['coverage_ratio_guard']):.2f}")
    print(f"LIFT_AVG={float(row_dict['theme_lift_avg']):.2f}")
    print(f"LIFT_MAX={float(row_dict['theme_lift_max']):.2f}")
    print(f"STATUS={row_dict['daily_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
