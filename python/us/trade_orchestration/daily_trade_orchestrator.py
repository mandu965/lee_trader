from __future__ import annotations

from collections import Counter

from python.us.buy_automation.decision_engine import run_buy_automation
from python.us.sell_automation.sell_decision_engine import run_sell_automation
from python.us.trade_orchestration.config import TradeOrchestrationConfig, load_trade_orchestration_config
from python.us.trade_orchestration.integrated_logger import persist_integrated_logs
from python.us.trade_orchestration.integrated_report_generator import (
    build_integrated_report,
    write_integrated_report_json,
    write_integrated_report_markdown,
)
from python.us.trade_orchestration.portfolio_state_loader import load_portfolio_state


def run_trade_orchestration(
    *,
    trade_date: str | None = None,
    account_id: str = "US_TRADE_SHADOW",
    persist_logs: bool = True,
) -> dict[str, object]:
    cfg = load_trade_orchestration_config()
    warnings = list(cfg.warnings)
    if cfg.enabled and cfg.scheduler_enabled and str(__import__("os").environ.get("US_BUY_SCHEDULER_ENABLED", "0")).strip() in {"1", "true", "TRUE"}:
        warnings.append("SCHEDULER_CONFIGURATION_CONFLICT")

    result: dict[str, object] = {
        "trade_date": trade_date,
        "mode": cfg.mode,
        "enabled": cfg.enabled,
        "success": True,
        "error": None,
        "pipeline_should_fail": False,
        "sell_executed": False,
        "buy_executed": False,
        "report_generated": False,
        "fail_safe_triggered": False,
        "warnings": warnings,
    }

    try:
        sell_report = run_sell_automation(
            trade_date=trade_date,
            account_id=account_id,
            persist_logs=persist_logs,
        )
        result["sell_executed"] = True
        effective_trade_date = sell_report.get("trade_date")
        portfolio_state = load_portfolio_state(
            cfg,
            trade_date=__import__("datetime").date.fromisoformat(str(effective_trade_date)),
            account_id=account_id,
            sell_report=sell_report,
        )
        buy_report = run_buy_automation(
            trade_date=effective_trade_date,
            account_id=account_id,
            persist_logs=persist_logs,
            portfolio_state=portfolio_state,
        )
        result["buy_executed"] = True

        conflict_results = [item for item in buy_report.get("candidates") or [] if item.get("conflict_checked")]
        conflict_counter = Counter()
        for item in conflict_results:
            reasons = item.get("conflict_reasons") or []
            if reasons:
                conflict_counter["TOTAL_CONFLICT_BLOCKED"] += 1
            for reason in reasons:
                conflict_counter[str(reason)] += 1

        final_action_summary = Counter()
        for item in sell_report.get("decisions") or []:
            final_action_summary[str(item.get("decision") or "UNKNOWN")] += 1
        final_action_summary["BUY_ALLOWED"] = buy_report.get("allowed_candidates", 0)
        final_action_summary["BUY_BLOCKED"] = buy_report.get("blocked_candidates", 0)

        result.update(
            {
                "trade_date": effective_trade_date,
                "mode": cfg.mode,
                "sell_report": sell_report,
                "portfolio_state": portfolio_state,
                "buy_report": buy_report,
                "conflict_results": [
                    {
                        "symbol": item.get("symbol"),
                        "buy_allowed_after_conflict_check": item.get("buy_allowed_after_conflict_check"),
                        "conflict_reasons": item.get("conflict_reasons"),
                        "related_position_id": item.get("related_position_id"),
                        "sell_signal": item.get("related_sell_signal"),
                        "cooldown_until": item.get("cooldown_until"),
                    }
                    for item in conflict_results
                ],
                "conflict_summary": dict(sorted(conflict_counter.items())),
                "final_action_summary": dict(sorted(final_action_summary.items())),
            }
        )
        result["fail_safe_triggered"] = (
            str(portfolio_state.get("status") or "").upper() == "PORTFOLIO_STATE_INCONSISTENT"
            or bool(conflict_counter.get("PORTFOLIO_STATE_INCONSISTENT"))
            or bool(sell_report.get("review_required"))
        )

        integrated_report = None
        if cfg.report_enabled:
            integrated_report = build_integrated_report(result, cfg)
            if "json" in cfg.report_formats:
                write_integrated_report_json(integrated_report, cfg)
            if "markdown" in cfg.report_formats:
                write_integrated_report_markdown(integrated_report, cfg)
            result["report_generated"] = True
            result["integrated_report"] = integrated_report

        result["sell_summary"] = {
            "loaded_positions": sell_report.get("loaded_positions", 0),
            "sell_signals": sell_report.get("sell_signals", 0),
            "hold_positions": sell_report.get("hold_positions", 0),
            "review_required": sell_report.get("review_required", 0),
        }
        result["buy_summary"] = {
            "loaded_candidates": buy_report.get("loaded_candidates", 0),
            "allowed_before_conflict": buy_report.get("allowed_before_conflict", 0),
            "allowed_after_conflict": buy_report.get("allowed_after_conflict", 0),
            "conflict_blocked": buy_report.get("conflict_blocked_candidates", 0),
        }
        result["paper_orders"] = {
            "buy_orders": len(buy_report.get("paper_orders") or []),
            "sell_orders": len(sell_report.get("paper_sell_orders") or []),
        }
        if persist_logs:
            result["log_persistence"] = persist_integrated_logs(cfg=cfg, orchestration_result=result, integrated_report=integrated_report)
        return result
    except Exception as exc:
        result["success"] = False
        result["error"] = str(exc)
        result["pipeline_should_fail"] = cfg.fail_pipeline_on_error or cfg.scheduler_fail_pipeline_on_error
        return result
