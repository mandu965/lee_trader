from __future__ import annotations

from datetime import datetime, timezone

from python.us.dashboard.config import DashboardConfig
from python.us.dashboard.dashboard_summary import (
    build_benchmark_comparison,
    build_buy_decision_monitor,
    build_conflict_guard_monitor,
    build_daily_overview,
    build_live_readiness_monitor,
    build_paper_performance_monitor,
    build_paper_portfolio_summary,
    build_risk_data_quality_monitor,
    build_scheduler_health_monitor,
    build_sell_decision_monitor,
)


def _mode_from_raw(raw_data: dict[str, object]) -> str:
    integrated = raw_data.get("integrated_report") or {}
    if isinstance(integrated, dict) and integrated.get("mode"):
        return str(integrated.get("mode"))
    logs = list(raw_data.get("orchestration_logs") or [])
    if logs and logs[0].get("mode"):
        return str(logs[0].get("mode"))
    return "UNKNOWN"


def build_dashboard_payload(raw_data: dict[str, object], cfg: DashboardConfig) -> dict[str, object]:
    daily_overview = build_daily_overview(raw_data)
    paper_portfolio_summary = build_paper_portfolio_summary(raw_data)
    buy_decision_monitor = build_buy_decision_monitor(raw_data) if cfg.include_buy_monitor else {"status": "NOT_AVAILABLE", "items": [], "warnings": [], "errors": []}
    sell_decision_monitor = build_sell_decision_monitor(raw_data) if cfg.include_sell_monitor else {"status": "NOT_AVAILABLE", "items": [], "warnings": [], "errors": []}
    conflict_guard_monitor = build_conflict_guard_monitor(raw_data) if cfg.include_conflict_monitor else {"status": "NOT_AVAILABLE", "items": [], "warnings": [], "errors": []}
    paper_performance_monitor = build_paper_performance_monitor(raw_data, cfg) if cfg.include_performance else {"status": "NOT_AVAILABLE", "warnings": [], "errors": []}
    benchmark_comparison = build_benchmark_comparison(raw_data, cfg) if cfg.include_performance else {"status": "NOT_AVAILABLE", "warnings": [], "errors": []}
    risk_data_quality_monitor = build_risk_data_quality_monitor(raw_data, cfg)
    scheduler_health_monitor = build_scheduler_health_monitor(raw_data) if cfg.include_health else {"status": "NOT_AVAILABLE", "warnings": [], "errors": []}
    live_readiness_monitor = build_live_readiness_monitor(raw_data) if cfg.include_readiness else {"status": "NOT_AVAILABLE", "warnings": [], "errors": []}

    warnings: list[str] = []
    errors: list[str] = []
    for section in (
        daily_overview,
        paper_portfolio_summary,
        buy_decision_monitor,
        sell_decision_monitor,
        conflict_guard_monitor,
        paper_performance_monitor,
        benchmark_comparison,
        risk_data_quality_monitor,
        scheduler_health_monitor,
        live_readiness_monitor,
    ):
        warnings.extend([str(item) for item in section.get("warnings") or []])
        errors.extend([str(item) for item in section.get("errors") or []])

    mode = _mode_from_raw(raw_data)
    return {
        "meta": {
            "report_type": "US_PAPER_TRADING_DASHBOARD",
            "trade_date": raw_data.get("trade_date"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_trading_only": True,
            "live_trading_enabled": False,
            "mode": mode,
            "dashboard_enabled": cfg.enabled,
            "missing_sources": list(raw_data.get("missing_sources") or []),
        },
        "daily_overview": daily_overview,
        "paper_portfolio_summary": paper_portfolio_summary,
        "buy_decision_monitor": buy_decision_monitor,
        "sell_decision_monitor": sell_decision_monitor,
        "conflict_guard_monitor": conflict_guard_monitor,
        "paper_performance_monitor": paper_performance_monitor,
        "benchmark_comparison": benchmark_comparison,
        "risk_data_quality_monitor": risk_data_quality_monitor,
        "scheduler_health_monitor": scheduler_health_monitor,
        "live_readiness_monitor": live_readiness_monitor,
        "warnings": warnings,
        "errors": errors,
    }
