from __future__ import annotations

from python.us.trade_orchestration.config import TradeOrchestrationConfig


def check_buy_conflict(
    candidate: dict[str, object],
    portfolio_state: dict[str, object],
    cfg: TradeOrchestrationConfig,
) -> dict[str, object]:
    symbol = str(candidate.get("symbol") or "").upper()
    open_position = (portfolio_state.get("open_position_map") or {}).get(symbol)
    sell_signal = (portfolio_state.get("sell_signal_map") or {}).get(symbol)
    review_signal = (portfolio_state.get("review_required_map") or {}).get(symbol)
    cooldown_entry = (portfolio_state.get("cooldown_map") or {}).get(symbol)
    paper_buy_today = symbol in set(portfolio_state.get("paper_buy_symbols_today") or [])
    inconsistent = str(portfolio_state.get("status") or "").upper() == "PORTFOLIO_STATE_INCONSISTENT"

    conflict_reasons: list[str] = []
    if inconsistent and cfg.conflict_failsafe:
        conflict_reasons.append("PORTFOLIO_STATE_INCONSISTENT")
    if cfg.block_buy_if_position_exists and open_position:
        conflict_reasons.append("OPEN_POSITION_EXISTS")
    if cfg.block_buy_if_sell_signal_exists and sell_signal:
        conflict_reasons.append("SELL_SIGNAL_EXISTS")
        if cfg.sell_priority_over_buy:
            conflict_reasons.append("SELL_PRIORITY_OVER_BUY")
    if cfg.block_buy_on_review_required and review_signal:
        conflict_reasons.append("REVIEW_REQUIRED_SYMBOL")
    if cooldown_entry:
        conflict_reasons.append("COOLDOWN_ACTIVE")
    if paper_buy_today:
        conflict_reasons.append("DUPLICATE_BUY")

    return {
        "symbol": symbol,
        "buy_allowed_after_conflict_check": len(conflict_reasons) == 0,
        "conflict_reasons": conflict_reasons,
        "related_position_id": open_position.get("paper_position_id") if isinstance(open_position, dict) else None,
        "sell_signal": sell_signal,
        "review_required_signal": review_signal,
        "cooldown_until": cooldown_entry.get("cooldown_until") if isinstance(cooldown_entry, dict) else None,
    }


def build_conflict_rule_rows(result: dict[str, object]) -> list[dict[str, object]]:
    symbol = result.get("symbol")
    if result.get("buy_allowed_after_conflict_check"):
        return [
            {
                "rule": "CONFLICT_GUARD",
                "result": "PASS",
                "reason_code": None,
                "value": symbol,
                "threshold": "NO_CONFLICT",
            }
        ]
    rows: list[dict[str, object]] = []
    for reason in result.get("conflict_reasons") or []:
        rows.append(
            {
                "rule": "CONFLICT_GUARD",
                "result": "FAIL",
                "reason_code": reason,
                "value": symbol,
                "threshold": "NO_CONFLICT",
            }
        )
    return rows
