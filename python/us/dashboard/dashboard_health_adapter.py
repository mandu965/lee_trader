from __future__ import annotations

import json
from pathlib import Path

from python.us.dashboard.config import DashboardConfig
from python.us.dashboard.dashboard_markdown_renderer import PAPER_NOTICE


def _load_json(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _contains_notice(path: str | None) -> bool:
    if not path:
        return False
    file_path = Path(path)
    if not file_path.exists():
        return False
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return PAPER_NOTICE in text


def run_dashboard_health_adapter(
    *,
    trade_date: str,
    dashboard_result: dict[str, object],
    cfg: DashboardConfig | None = None,
) -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []

    json_path = dashboard_result.get("json_report_path")
    markdown_path = dashboard_result.get("markdown_report_path")
    latest_json_path = dashboard_result.get("latest_json_path")
    latest_markdown_path = dashboard_result.get("latest_markdown_path")

    payload = dashboard_result.get("payload")
    if not isinstance(payload, dict):
        payload = _load_json(str(json_path) if json_path else None)
    section_status_complete = False
    if isinstance(payload, dict):
        sections = [
            "daily_overview",
            "paper_portfolio_summary",
            "buy_decision_monitor",
            "sell_decision_monitor",
            "conflict_guard_monitor",
            "paper_performance_monitor",
            "benchmark_comparison",
            "risk_data_quality_monitor",
            "scheduler_health_monitor",
            "live_readiness_monitor",
        ]
        section_status_complete = all(isinstance(payload.get(section), dict) and payload.get(section, {}).get("status") for section in sections)

    checked_items = {
        "dashboard_json_report_exists": bool(json_path) and Path(str(json_path)).exists(),
        "dashboard_markdown_report_exists": bool(markdown_path) and Path(str(markdown_path)).exists(),
        "latest_dashboard_json_exists": bool(latest_json_path) and Path(str(latest_json_path)).exists(),
        "latest_dashboard_markdown_exists": bool(latest_markdown_path) and Path(str(latest_markdown_path)).exists(),
        "dashboard_payload_valid": isinstance(payload, dict),
        "paper_trading_only_true": isinstance(payload, dict) and bool((payload.get("meta") or {}).get("paper_trading_only")) is True,
        "paper_trading_notice_exists": _contains_notice(str(markdown_path) if markdown_path else None),
        "live_trading_enabled_false": isinstance(payload, dict) and bool((payload.get("meta") or {}).get("live_trading_enabled")) is False,
        "dashboard_generated_at_exists": isinstance(payload, dict) and bool((payload.get("meta") or {}).get("generated_at")),
        "dashboard_trade_date_matches": isinstance(payload, dict) and str((payload.get("meta") or {}).get("trade_date") or "") == str(trade_date),
        "dashboard_sections_have_status": section_status_complete,
    }

    require_json_report = True if cfg is None else cfg.require_json_report
    require_markdown_report = False if cfg is None else cfg.require_markdown_report

    if require_json_report and not checked_items["dashboard_json_report_exists"]:
        errors.append("DASHBOARD_JSON_REPORT_MISSING")
    if not checked_items["dashboard_payload_valid"]:
        errors.append("DASHBOARD_JSON_PAYLOAD_INVALID")
    if not checked_items["paper_trading_only_true"]:
        errors.append("DASHBOARD_PAPER_TRADING_ONLY_FLAG_INVALID")
    if not checked_items["live_trading_enabled_false"]:
        errors.append("DASHBOARD_LIVE_TRADING_FLAG_INVALID")
    if not checked_items["dashboard_generated_at_exists"]:
        errors.append("DASHBOARD_GENERATED_AT_MISSING")
    if not checked_items["dashboard_trade_date_matches"]:
        errors.append("DASHBOARD_TRADE_DATE_MISMATCH")

    if require_markdown_report and not checked_items["dashboard_markdown_report_exists"]:
        errors.append("DASHBOARD_MARKDOWN_REPORT_MISSING")
    elif not checked_items["dashboard_markdown_report_exists"]:
        warnings.append("DASHBOARD_MARKDOWN_REPORT_MISSING")
    if not checked_items["latest_dashboard_json_exists"]:
        warnings.append("LATEST_DASHBOARD_JSON_MISSING")
    if not checked_items["latest_dashboard_markdown_exists"]:
        warnings.append("LATEST_DASHBOARD_MARKDOWN_MISSING")
    if not checked_items["paper_trading_notice_exists"]:
        warnings.append("DASHBOARD_PAPER_TRADING_NOTICE_MISSING")
    if not checked_items["dashboard_sections_have_status"]:
        warnings.append("DASHBOARD_SECTION_STATUS_MISSING")

    if errors:
        health_status = "ERROR"
    elif warnings:
        health_status = "WARNING"
    else:
        health_status = "PASS"

    return {
        "dashboard_health_status": health_status,
        "checked_items": checked_items,
        "warnings": warnings,
        "errors": errors,
        "payload": payload,
    }
