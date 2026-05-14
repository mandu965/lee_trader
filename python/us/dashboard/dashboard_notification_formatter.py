from __future__ import annotations

import json

from python.us.dashboard.config import DashboardConfig


def _status_from_dashboard(payload: dict[str, object]) -> str:
    statuses = [
        str((payload.get(section) or {}).get("status") or "OK")
        for section in (
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
        )
    ]
    if "ERROR" in statuses:
        return "ERROR"
    if "DATA_MISSING" in statuses:
        return "WARNING"
    if "WARNING" in statuses:
        return "WARNING"
    return "OK"


def build_dashboard_notification_json_payload(
    dashboard_payload: dict[str, object],
    *,
    dashboard_health: dict[str, object] | None = None,
    cfg: DashboardConfig | None = None,
) -> dict[str, object]:
    daily = dashboard_payload.get("daily_overview") or {}
    risk = dashboard_payload.get("risk_data_quality_monitor") or {}
    scheduler = dashboard_payload.get("scheduler_health_monitor") or {}
    readiness = dashboard_payload.get("live_readiness_monitor") or {}
    sell = dashboard_payload.get("sell_decision_monitor") or {}
    buy = dashboard_payload.get("buy_decision_monitor") or {}
    health_status = (dashboard_health or {}).get("dashboard_health_status") or scheduler.get("health_check_status") or scheduler.get("status")
    include_warnings = True if cfg is None else cfg.notification_include_warnings
    include_readiness = True if cfg is None else cfg.notification_include_readiness
    include_top_symbols = True if cfg is None else cfg.notification_include_top_symbols
    max_symbols = 10 if cfg is None else cfg.notification_max_symbols

    top_symbols: list[str] = []
    if include_top_symbols:
        for row in list((buy.get("items") or []))[:max_symbols]:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                top_symbols.append(symbol)
        for row in list((sell.get("items") or []))[:max_symbols]:
            symbol = str(row.get("symbol") or "").upper()
            if symbol and symbol not in top_symbols:
                top_symbols.append(symbol)
        top_symbols = top_symbols[:max_symbols]

    payload = {
        "message_type": "US_PAPER_TRADING_DASHBOARD_SUMMARY",
        "trade_date": (dashboard_payload.get("meta") or {}).get("trade_date"),
        "mode": (dashboard_payload.get("meta") or {}).get("mode"),
        "status": _status_from_dashboard(dashboard_payload),
        "paper_trading_only": True,
        "live_orders_executed": False,
        "buy": {
            "candidates": daily.get("buy_candidates"),
            "final_allowed": daily.get("final_buy_allowed"),
            "conflict_blocked": daily.get("conflict_blocked_count"),
        },
        "sell": {
            "positions": sell.get("loaded_positions"),
            "sell_signals": daily.get("sell_signals"),
            "review_required": daily.get("review_required_count"),
        },
        "risk": {
            "data_missing_rate": risk.get("data_missing_rate"),
            "fail_safe_triggered": daily.get("fail_safe_triggered"),
            "top_warning_reason": daily.get("top_warning_reason"),
        },
        "health": {
            "scheduler_status": scheduler.get("health_check_status") or scheduler.get("status"),
            "dashboard_status": health_status,
        },
        "notice": "Paper Trading only. No live orders were executed.",
    }
    if include_readiness:
        payload["readiness"] = {
            "live_ready": readiness.get("live_ready"),
            "readiness_score": readiness.get("readiness_score"),
            "manual_approval_required": readiness.get("manual_approval_required"),
        }
    if include_warnings:
        payload["warnings"] = list(dashboard_payload.get("warnings") or [])[: max_symbols]
    if include_top_symbols:
        payload["top_symbols"] = top_symbols
    return payload


def render_dashboard_notification_text(
    dashboard_payload: dict[str, object],
    *,
    dashboard_health: dict[str, object] | None = None,
    cfg: DashboardConfig | None = None,
) -> str:
    payload = build_dashboard_notification_json_payload(dashboard_payload, dashboard_health=dashboard_health, cfg=cfg)
    lines = [
        "[US Paper Trading Dashboard]",
        "",
        f"date: {payload.get('trade_date')}",
        f"mode: {payload.get('mode')}",
        f"status: {payload.get('status')}",
        "",
        "BUY:",
        f"- candidates: {payload.get('buy', {}).get('candidates')}",
        f"- final allowed: {payload.get('buy', {}).get('final_allowed')}",
        f"- conflict blocked: {payload.get('buy', {}).get('conflict_blocked')}",
        "",
        "SELL:",
        f"- positions: {payload.get('sell', {}).get('positions')}",
        f"- sell signals: {payload.get('sell', {}).get('sell_signals')}",
        f"- review required: {payload.get('sell', {}).get('review_required')}",
        "",
        "Risk/Data:",
        f"- data missing rate: {payload.get('risk', {}).get('data_missing_rate')}%",
        f"- fail-safe: {'YES' if payload.get('risk', {}).get('fail_safe_triggered') else 'NO'}",
        f"- top warning: {payload.get('risk', {}).get('top_warning_reason')}",
        "",
        "Health:",
        f"- scheduler: {payload.get('health', {}).get('scheduler_status')}",
        f"- dashboard: {payload.get('health', {}).get('dashboard_status')}",
        "",
    ]
    if "readiness" in payload:
        lines.extend(
            [
                "Readiness:",
                f"- live_ready: {str(bool(payload.get('readiness', {}).get('live_ready'))).lower()}",
                f"- readiness_score: {payload.get('readiness', {}).get('readiness_score')}",
                f"- manual approval required: {str(bool(payload.get('readiness', {}).get('manual_approval_required'))).lower()}",
                "",
            ]
        )
    if payload.get("top_symbols"):
        lines.extend(
            [
                "Top Symbols:",
                f"- {', '.join(payload.get('top_symbols') or [])}",
                "",
            ]
        )
    if payload.get("warnings"):
        lines.extend(
            [
                "Warnings:",
                f"- {', '.join(str(item) for item in (payload.get('warnings') or []))}",
                "",
            ]
        )
    lines.extend(
        [
        "Notice:",
        payload.get("notice") or "Paper Trading only. No live orders were executed.",
        ]
    )
    return "\n".join(lines)


def render_dashboard_notification_json_text(
    dashboard_payload: dict[str, object],
    *,
    dashboard_health: dict[str, object] | None = None,
    cfg: DashboardConfig | None = None,
) -> str:
    return json.dumps(
        build_dashboard_notification_json_payload(dashboard_payload, dashboard_health=dashboard_health, cfg=cfg),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
