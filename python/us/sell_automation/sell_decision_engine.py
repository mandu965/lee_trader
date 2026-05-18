from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import math
import uuid

from python.us.sell_automation.config import SellAutomationConfig, load_sell_automation_config
from python.us.sell_automation.paper_position_loader import load_paper_positions
from python.us.sell_automation.paper_sell_order import build_paper_sell_order
from python.us.sell_automation.sell_logger import persist_sell_automation_logs
from python.us.sell_automation.sell_rule_engine import evaluate_sell_rules
from python.us.us_config import parse_iso_date
from python.us.us_db import fetch_latest_daily_feature_snapshots


RULE_PRIORITY = [
    "DATA_QUALITY_CHECK",
    "RISK_OFF_MARKET_EXIT",
    "STOP_LOSS",
    "TRAILING_STOP",
    "MAX_HOLDING_DAYS",
    "RANK_SCORE_DETERIORATION",
    "BENCHMARK_UNDERPERFORMANCE",
    "TAKE_PROFIT",
    "HOLD",
]


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_context(cfg: SellAutomationConfig, trade_date: date) -> dict[str, object]:
    snapshots = fetch_latest_daily_feature_snapshots([cfg.benchmark_symbol], trade_date=trade_date)
    snapshot = snapshots.get(cfg.benchmark_symbol.upper(), {})
    return {
        "benchmark_symbol": cfg.benchmark_symbol,
        "benchmark_drawdown_pct": snapshot.get("ret_20d"),
        "benchmark_snapshot_trade_date": snapshot.get("trade_date"),
    }


def _whole_share_quantity(*, remaining_quantity: object, sell_ratio: float, full_exit: bool) -> int:
    qty = _safe_float(remaining_quantity) or 0.0
    if qty <= 0 or sell_ratio <= 0:
        return 0
    if full_exit:
        return max(0, int(round(qty)))
    partial_qty = int(math.floor(qty * sell_ratio))
    if partial_qty <= 0 and qty >= 1:
        return 1
    return max(0, partial_qty)


def _decision_for_rules(
    *,
    position: dict[str, object],
    rules: list[dict[str, object]],
    cfg: SellAutomationConfig,
) -> dict[str, object]:
    priority_map = {name: index for index, name in enumerate(RULE_PRIORITY)}
    actionable = sorted(
        [
            rule
            for rule in rules
            if rule.get("action") in {"REVIEW_REQUIRED", "FULL_SELL", "PARTIAL_SELL"}
            and rule.get("result") in {"FAIL", "UNKNOWN"}
        ],
        key=lambda item: priority_map.get(str(item.get("rule") or ""), 999),
    )
    selected_rule = actionable[0] if actionable else None

    decision = "HOLD"
    requested_sell_action = "NONE"
    sell_ratio = 0.0
    exit_reason = None
    review_required = False
    error_message = None

    if selected_rule is not None:
        requested_sell_action = str(selected_rule.get("action") or "NONE")
        exit_reason = selected_rule.get("reason")
        if requested_sell_action == "REVIEW_REQUIRED":
            decision = "REVIEW_REQUIRED"
            requested_sell_action = "NONE"
            review_required = True
            error_message = str(selected_rule.get("reason") or "SELL_DECISION_BLOCKED")
        elif requested_sell_action == "FULL_SELL":
            decision = "SELL"
            sell_ratio = 1.0
        elif requested_sell_action == "PARTIAL_SELL":
            decision = "PARTIAL_SELL"
            sell_ratio = cfg.partial_take_profit_ratio
    sell_quantity = _whole_share_quantity(
        remaining_quantity=position.get("remaining_quantity"),
        sell_ratio=sell_ratio,
        full_exit=decision == "SELL",
    )

    sell_action = requested_sell_action
    if not cfg.enabled and requested_sell_action != "NONE":
        sell_action = "DISABLED"
    elif cfg.mode == "SHADOW" and requested_sell_action != "NONE":
        sell_action = "SHADOW"
    elif cfg.mode == "LIVE" and requested_sell_action != "NONE":
        sell_action = "LIVE_NOT_IMPLEMENTED"
        review_required = True
        error_message = "LIVE_NOT_IMPLEMENTED"

    if decision == "HOLD":
        sell_action = "NONE"

    return {
        "sell_decision_id": f"USSELLDEC_{str(position.get('symbol') or 'UNKNOWN').upper()}_{uuid.uuid4().hex[:10].upper()}",
        "symbol": position.get("symbol"),
        "paper_position_id": position.get("paper_position_id"),
        "decision": decision,
        "sell_action": sell_action,
        "requested_sell_action": requested_sell_action,
        "sell_ratio": sell_ratio,
        "sell_quantity": sell_quantity,
        "exit_reason": exit_reason,
        "review_required": review_required,
        "mode": cfg.mode,
        "automation_enabled": cfg.enabled,
        "applied_rules": rules,
        "latest_price": position.get("latest_price"),
        "avg_entry_price": position.get("avg_entry_price"),
        "unrealized_pnl_pct": position.get("unrealized_pnl_pct"),
        "realized_paper_pnl": None,
        "error_message": error_message,
    }


def run_sell_automation(
    *,
    trade_date: str | None = None,
    account_id: str = "US_SELL_SHADOW",
    persist_logs: bool = True,
) -> dict[str, object]:
    cfg = load_sell_automation_config()
    requested_trade_date = parse_iso_date(trade_date, field_name="trade_date") if trade_date else None
    load_result = load_paper_positions(cfg, account_id=account_id, requested_trade_date=requested_trade_date)
    effective_trade_date = load_result.get("trade_date")
    positions = list(load_result.get("positions") or [])
    events = list(load_result.get("events") or [])

    market_context = _market_context(cfg, effective_trade_date) if isinstance(effective_trade_date, date) else {}
    decisions: list[dict[str, object]] = []
    paper_sell_orders: list[dict[str, object]] = []

    for position in positions:
        rules = evaluate_sell_rules(position, cfg, market_context=market_context)
        decision = _decision_for_rules(position=position, rules=rules, cfg=cfg)
        if cfg.mode == "PAPER" and cfg.enabled and decision.get("requested_sell_action") in {"FULL_SELL", "PARTIAL_SELL"}:
            paper_order = build_paper_sell_order(
                trade_date=str(effective_trade_date),
                decision=decision,
                position=position,
                mode=cfg.mode,
            )
            decision["realized_paper_pnl"] = paper_order.get("realized_paper_pnl")
            paper_sell_orders.append(paper_order)
        decisions.append(decision)

    counts = Counter(item.get("decision") or "UNKNOWN" for item in decisions)
    reason_summary = Counter()
    for item in decisions:
        if item.get("exit_reason"):
            reason_summary[str(item["exit_reason"])] += 1
        elif item.get("error_message"):
            reason_summary[str(item["error_message"])] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_date": str(effective_trade_date) if effective_trade_date else None,
        "account_id": account_id,
        "mode": cfg.mode,
        "automation_enabled": cfg.enabled,
        "benchmark_symbol": cfg.benchmark_symbol,
        "market_context": market_context,
        "config_snapshot": {
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "max_holding_days": cfg.max_holding_days,
            "rank_exit_threshold": cfg.rank_exit_threshold,
            "min_score_hold": cfg.min_score_hold,
            "min_prob_hold": cfg.min_prob_hold,
            "benchmark_underperform_pct": cfg.benchmark_underperform_pct,
            "market_drawdown_exit_pct": cfg.market_drawdown_exit_pct,
            "partial_take_profit_enabled": cfg.partial_take_profit_enabled,
            "partial_take_profit_ratio": cfg.partial_take_profit_ratio,
        },
        "config_warnings": list(cfg.warnings),
        "events": events,
        "loaded_positions": len(positions),
        "hold_positions": counts.get("HOLD", 0),
        "sell_signals": counts.get("SELL", 0),
        "partial_sell_signals": counts.get("PARTIAL_SELL", 0),
        "review_required": counts.get("REVIEW_REQUIRED", 0),
        "paper_sell_orders": paper_sell_orders,
        "positions": positions,
        "decisions": decisions,
        "reason_summary": dict(sorted(reason_summary.items())),
    }
    if persist_logs:
        report["log_persistence"] = persist_sell_automation_logs(cfg=cfg, report=report, account_id=account_id)
    return report
