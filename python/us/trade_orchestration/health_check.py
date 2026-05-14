from __future__ import annotations

from pathlib import Path

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_health_adapter import run_dashboard_health_adapter
from python.us.trade_orchestration.config import TradeOrchestrationConfig


def _report_paths(cfg: TradeOrchestrationConfig, trade_date: str) -> tuple[Path, Path]:
    return (
        cfg.report_output_dir / f"{trade_date}_integrated_trade_report.json",
        cfg.report_output_dir / f"{trade_date}_integrated_trade_report.md",
    )


def run_trade_health_check(
    cfg: TradeOrchestrationConfig,
    *,
    orchestration_result: dict[str, object],
    dashboard_result: dict[str, object] | None = None,
) -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []
    trade_date = str(orchestration_result.get("trade_date") or "")
    json_path, md_path = _report_paths(cfg, trade_date)
    integrated_report = orchestration_result.get("integrated_report") or {}
    buy_report = orchestration_result.get("buy_report") or {}
    sell_report = orchestration_result.get("sell_report") or {}
    final_action_summary = orchestration_result.get("final_action_summary") or {}

    checked_items = {
        "orchestration_log_exists": bool((orchestration_result.get("log_persistence") or {}).get("json_path")) or not orchestration_result.get("success") is False,
        "json_report_exists": json_path.exists() if trade_date else False,
        "markdown_report_exists": md_path.exists() if trade_date else False,
        "buy_summary_exists": bool(integrated_report.get("buy_summary")),
        "sell_summary_exists": bool(integrated_report.get("sell_summary")),
        "conflict_summary_exists": bool(integrated_report.get("conflict_summary") is not None),
        "final_action_summary_exists": bool(final_action_summary),
    }

    if cfg.health_check_fail_on_missing_report:
        if not checked_items["json_report_exists"]:
            errors.append("INTEGRATED_JSON_REPORT_MISSING")
        if "markdown" in cfg.report_formats and not checked_items["markdown_report_exists"]:
            errors.append("INTEGRATED_MARKDOWN_REPORT_MISSING")
    else:
        if not checked_items["json_report_exists"]:
            warnings.append("INTEGRATED_JSON_REPORT_MISSING")
        if "markdown" in cfg.report_formats and not checked_items["markdown_report_exists"]:
            warnings.append("INTEGRATED_MARKDOWN_REPORT_MISSING")

    if not checked_items["buy_summary_exists"]:
        errors.append("BUY_SUMMARY_MISSING")
    if not checked_items["sell_summary_exists"]:
        errors.append("SELL_SUMMARY_MISSING")
    if not checked_items["conflict_summary_exists"]:
        errors.append("CONFLICT_SUMMARY_MISSING")
    if not checked_items["final_action_summary_exists"]:
        errors.append("FINAL_ACTION_SUMMARY_MISSING")

    invalid_decisions = [
        str(item.get("symbol") or "UNKNOWN")
        for item in buy_report.get("candidates") or []
        if not item.get("allowed") and not (item.get("block_reasons") or [])
    ]
    if invalid_decisions:
        target = errors if cfg.health_check_fail_on_invalid_log else warnings
        target.append("INVALID_BUY_DECISION_LOG")

    review_required_count = int(sell_report.get("review_required", 0) or 0)
    if review_required_count > 0:
        warnings.append("REVIEW_REQUIRED_PRESENT")

    total_buy = int(buy_report.get("loaded_candidates", 0) or 0)
    missing_hits = 0
    for reason, count in (buy_report.get("block_summary") or {}).items():
        if "MISSING" in str(reason).upper():
            missing_hits += int(count or 0)
    data_missing_rate_pct = round((missing_hits / total_buy) * 100.0, 2) if total_buy > 0 else 0.0
    if data_missing_rate_pct > cfg.health_check_max_data_missing_rate_pct:
        warnings.append("DATA_MISSING_RATE_EXCEEDED")

    portfolio_inconsistent = str((orchestration_result.get("portfolio_state") or {}).get("status") or "").upper() == "PORTFOLIO_STATE_INCONSISTENT"
    if portfolio_inconsistent:
        errors.append("PORTFOLIO_STATE_INCONSISTENT")

    live_disabled_recorded = any(
        "LIVE_DISABLED_IN_SCHEDULER" in str(item)
        for item in [*warnings, *errors, *(orchestration_result.get("warnings") or []), *(orchestration_result.get("errors") or [])]
    )
    checked_items["live_disabled_recorded"] = live_disabled_recorded

    dashboard_health = {
        "enabled": False,
        "executed": False,
        "health_status": "NOT_AVAILABLE",
        "json_report_exists": False,
        "markdown_report_exists": False,
        "latest_json_exists": False,
        "latest_markdown_exists": False,
        "warnings": [],
        "errors": [],
    }
    if dashboard_result is not None:
        dashboard_cfg = load_dashboard_config()
        adapter = run_dashboard_health_adapter(
            trade_date=trade_date,
            dashboard_result=dashboard_result,
            cfg=dashboard_cfg,
        )
        dashboard_health = {
            "enabled": bool(dashboard_result.get("dashboard_enabled")),
            "executed": bool(dashboard_result.get("dashboard_executed")),
            "health_status": adapter.get("dashboard_health_status"),
            "json_report_exists": bool((adapter.get("checked_items") or {}).get("dashboard_json_report_exists")),
            "markdown_report_exists": bool((adapter.get("checked_items") or {}).get("dashboard_markdown_report_exists")),
            "latest_json_exists": bool((adapter.get("checked_items") or {}).get("latest_dashboard_json_exists")),
            "latest_markdown_exists": bool((adapter.get("checked_items") or {}).get("latest_dashboard_markdown_exists")),
            "warnings": list(adapter.get("warnings") or []),
            "errors": list(adapter.get("errors") or []),
        }
        checked_items["dashboard_json_report_exists"] = dashboard_health["json_report_exists"]
        checked_items["dashboard_markdown_report_exists"] = dashboard_health["markdown_report_exists"]
        checked_items["latest_dashboard_json_exists"] = dashboard_health["latest_json_exists"]
        checked_items["latest_dashboard_markdown_exists"] = dashboard_health["latest_markdown_exists"]
        warnings.extend(dashboard_health["warnings"])
        errors.extend(dashboard_health["errors"])

    health_status = "PASS" if not errors else "FAIL"
    return {
        "health_status": health_status,
        "checked_items": checked_items,
        "warnings": warnings,
        "errors": errors,
        "dashboard": dashboard_health,
        "data_missing_rate_pct": data_missing_rate_pct,
        "review_required_count": review_required_count,
        "pipeline_should_fail": bool(errors) and (cfg.health_check_fail_on_missing_report or cfg.health_check_fail_on_invalid_log or cfg.scheduler_fail_pipeline_on_error),
    }
