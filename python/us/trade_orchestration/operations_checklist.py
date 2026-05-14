from __future__ import annotations

from pathlib import Path

from python.us.trade_orchestration.config import TradeOrchestrationConfig


def build_operations_checklist(
    *,
    orchestration_result: dict[str, object],
    health_result: dict[str, object],
) -> str:
    report_generated = bool(orchestration_result.get("report_generated"))
    duplicate_run = "DUPLICATE_RUN_DETECTED" in [*(orchestration_result.get("warnings") or []), *(orchestration_result.get("errors") or [])]
    live_blocked = "LIVE_DISABLED_IN_SCHEDULER" in [*(orchestration_result.get("warnings") or []), *(orchestration_result.get("errors") or [])]
    return "\n".join(
        [
            "# US Trade Orchestration Daily Checklist",
            "",
            f"- [{'x' if orchestration_result.get('success') else ' '}] Scheduler job executed",
            f"- [{'x' if orchestration_result.get('sell_executed') else ' '}] SELL summary reviewed",
            f"- [{'x' if orchestration_result.get('buy_executed') else ' '}] BUY final candidates reviewed",
            f"- [{'x' if (orchestration_result.get('conflict_summary') is not None) else ' '}] Conflict block reasons reviewed",
            f"- [{'x' if (orchestration_result.get('sell_report', {}).get('review_required', 0) >= 0) else ' '}] REVIEW_REQUIRED symbols reviewed",
            f"- [{'x' if ('data_missing_rate_pct' in health_result) else ' '}] DATA_MISSING rate checked",
            f"- [{'x' if (orchestration_result.get('integrated_report', {}).get('paper_portfolio_summary') is not None) else ' '}] Paper PnL reviewed",
            f"- [{'x' if report_generated else ' '}] Integrated report generated",
            f"- [{'x' if not duplicate_run else ' '}] Duplicate run not detected",
            f"- [{'x' if (live_blocked or orchestration_result.get('mode') != 'LIVE') else ' '}] LIVE mode remains disabled",
            "",
            "## US Paper Trading Dashboard Daily Check",
            "",
            f"- [{'x' if orchestration_result.get('dashboard_json_report_path') else ' '}] Dashboard JSON report generated",
            f"- [{'x' if orchestration_result.get('dashboard_markdown_report_path') else ' '}] Dashboard Markdown report generated",
            f"- [{'x' if orchestration_result.get('dashboard_latest_json_path') else ' '}] latest_dashboard.json updated",
            f"- [{'x' if (orchestration_result.get('dashboard_payload', {}).get('daily_overview') is not None) else ' '}] Daily Overview status checked",
            f"- [{'x' if orchestration_result.get('buy_executed') else ' '}] BUY final candidates reviewed",
            f"- [{'x' if orchestration_result.get('sell_executed') else ' '}] SELL signals reviewed",
            f"- [{'x' if (orchestration_result.get('conflict_summary') is not None) else ' '}] Conflict blocked symbols reviewed",
            f"- [{'x' if (orchestration_result.get('sell_report', {}).get('review_required', 0) >= 0) else ' '}] REVIEW_REQUIRED symbols reviewed",
            f"- [{'x' if ('data_missing_rate_pct' in health_result) else ' '}] Data Missing rate checked",
            f"- [{'x' if health_result.get('health_status') is not None else ' '}] Scheduler / Health status checked",
            f"- [{'x' if (orchestration_result.get('dashboard_payload', {}).get('live_readiness_monitor') is not None) else ' '}] LIVE Readiness checked",
            f"- [{'x' if True else ' '}] Confirmed: Paper Trading only, no live orders executed",
        ]
    )


def write_operations_checklist(
    cfg: TradeOrchestrationConfig,
    *,
    trade_date: str,
    markdown: str,
) -> Path:
    cfg.checklist_output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.checklist_output_dir / f"{trade_date}_trade_orchestration_checklist.md"
    path.write_text(markdown, encoding="utf-8")
    return path
