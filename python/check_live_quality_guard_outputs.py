from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from report_json import write_json_strict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_QUALITY_JSON = OUTPUT_DIR / "quality_risk_guard_live_review.json"
DEFAULT_CLOSED_JSON = OUTPUT_DIR / "live_closed_trade_report.json"
DEFAULT_LIVE_KPI_JSON = OUTPUT_DIR / "live_kpi_daily_report.json"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_quality_guard_output_check.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_quality_guard_output_check.md"

PROMOTION_STATUSES = {"KEEP_SHADOW", "REVIEW_READY", "PROMOTE_CANDIDATE", "REJECT"}
SAMPLE_STATUSES = {"INSUFFICIENT_SAMPLE", "MONITOR_ONLY", "ACTIONABLE", "DEGRADE_CONFIRMED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate live KPI / quality guard output artifacts.")
    parser.add_argument("--quality-json", type=Path, default=DEFAULT_QUALITY_JSON)
    parser.add_argument("--closed-json", type=Path, default=DEFAULT_CLOSED_JSON)
    parser.add_argument("--live-kpi-json", type=Path, default=DEFAULT_LIVE_KPI_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}, f"missing file: {resolved}"
    try:
        loaded = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"json parse failed: {resolved} ({exc})"
    if not isinstance(loaded, dict):
        return {}, f"json root is not object: {resolved}"
    return loaded, None


def _walk_bad_numbers(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_walk_bad_numbers(item, f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(_walk_bad_numbers(item, f"{path}[{idx}]"))
        return out
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return [f"non-finite number at {path}"]
    return []


def _require(report: dict[str, Any], key: str, issues: list[str], *, name: str) -> Any:
    if key not in report:
        issues.append(f"{name}: missing key {key}")
        return None
    return report.get(key)


def validate_quality(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    _require(report, "generated_at", issues, name="quality")
    _require(report, "overview", issues, name="quality")
    _require(report, "horizon_summary", issues, name="quality")
    _require(report, "closed_trade_summary", issues, name="quality")
    promotion_status = _require(report, "promotion_status", issues, name="quality")
    sample_status = _require(report, "sample_status", issues, name="quality")
    blockers = report.get("promotion_blockers")

    if promotion_status not in PROMOTION_STATUSES:
        issues.append(f"quality: invalid promotion_status {promotion_status}")
    if sample_status not in SAMPLE_STATUSES:
        issues.append(f"quality: invalid sample_status {sample_status}")
    if not isinstance(blockers, list):
        issues.append("quality: promotion_blockers must be a list")
    elif blockers and promotion_status == "PROMOTE_CANDIDATE":
        issues.append("quality: promotion_status is PROMOTE_CANDIDATE despite blockers")

    closed = report.get("closed_trade_summary")
    if not isinstance(closed, dict) or not closed.get("available"):
        issues.append("quality: closed_trade_summary is missing or unavailable")
    else:
        snapshot_fallback = int(closed.get("snapshot_fallback_count") or 0)
        if snapshot_fallback > 0 and not any("snapshot" in str(item).lower() for item in blockers or []):
            issues.append("quality: snapshot fallback exists but promotion_blockers does not mention it")
        closed_observed = int(closed.get("observed_count") or 0)
        if closed_observed < 30 and not any("Closed-trade observed_count is below 30" in str(item) for item in blockers or []):
            issues.append("quality: closed sample is below 30 but blocker is missing")

    horizon_summary = report.get("horizon_summary")
    if not isinstance(horizon_summary, list) or not horizon_summary:
        issues.append("quality: horizon_summary must be a non-empty list")
    else:
        horizons = {int(row.get("horizon")) for row in horizon_summary if row.get("horizon") is not None}
        if not {0, 1, 3, 5}.issubset(horizons):
            issues.append(f"quality: missing expected horizons {sorted({0, 1, 3, 5} - horizons)}")

    return issues


def validate_closed(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    overview = _require(report, "overview", issues, name="closed")
    _require(report, "recent_closed_trades", issues, name="closed")
    sample_status = _require(report, "sample_status", issues, name="closed")
    if sample_status not in SAMPLE_STATUSES:
        issues.append(f"closed: invalid sample_status {sample_status}")
    if isinstance(overview, dict):
        closed_count = int(overview.get("closed_trade_count") or 0)
        observed_count = int(overview.get("observed_count") or 0)
        unmatched_count = int(overview.get("unmatched_count") or 0)
        if observed_count > closed_count:
            issues.append("closed: observed_count cannot exceed closed_trade_count")
        if unmatched_count > 0 and observed_count == closed_count:
            issues.append("closed: unmatched_count exists but all rows are observed")
    else:
        issues.append("closed: overview must be object")
    return issues


def validate_live_kpi(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    _require(report, "overview", issues, name="live_kpi")
    _require(report, "horizon_summary", issues, name="live_kpi")
    sample_status = _require(report, "sample_status", issues, name="live_kpi")
    if sample_status not in SAMPLE_STATUSES:
        issues.append(f"live_kpi: invalid sample_status {sample_status}")
    return issues


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    quality, quality_error = _load_json(args.quality_json)
    closed, closed_error = _load_json(args.closed_json)
    live_kpi, live_kpi_error = _load_json(args.live_kpi_json)

    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    for name, payload, load_error, validator in (
        ("quality_risk_guard_live_review", quality, quality_error, validate_quality),
        ("live_closed_trade_report", closed, closed_error, validate_closed),
        ("live_kpi_daily_report", live_kpi, live_kpi_error, validate_live_kpi),
    ):
        item_issues: list[str] = []
        if load_error:
            item_issues.append(load_error)
        else:
            item_issues.extend(_walk_bad_numbers(payload))
            item_issues.extend(validator(payload))
        checks.append({"name": name, "status": "FAIL" if item_issues else "PASS", "issues": item_issues})
        issues.extend(item_issues)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "validation_status": "FAIL" if issues else "PASS",
        "issue_count": len(issues),
        "issues": issues,
        "checks": checks,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Quality Guard Output Check",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- validation_status: `{report['validation_status']}`",
        f"- issue_count: `{report['issue_count']}`",
        "",
        "## Issues",
        "",
    ]
    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("- None")
    lines.extend(["", "## Checks", "", "| artifact | status | issues |", "| --- | --- | ---: |"])
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {len(check['issues'])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = build_report(args)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    write_json_strict(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"live_quality_guard_output_check_json: {out_json}")
    print(f"live_quality_guard_output_check_md: {out_md}")
    print(f"validation_status: {report['validation_status']}")
    return 2 if report["validation_status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
