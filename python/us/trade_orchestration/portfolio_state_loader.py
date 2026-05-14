from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from sqlalchemy import text

from python.us.sell_automation.config import load_sell_automation_config
from python.us.sell_automation.paper_position_loader import load_paper_positions
from python.us.trade_orchestration.config import TradeOrchestrationConfig
from python.us.us_db import get_us_engine, relation_exists


READ_SELL_ORDER_ROWS_SQL = text(
    """
    SELECT *
    FROM trade.us_paper_sell_order
    WHERE trade_date <= :trade_date
    ORDER BY trade_date DESC, created_at DESC, symbol
    """
)

READ_SELL_DECISION_ROWS_SQL = text(
    """
    SELECT DISTINCT ON (symbol)
        *
    FROM trade.us_sell_decision_log
    WHERE trade_date <= :trade_date
    ORDER BY symbol, trade_date DESC, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
    """
)

READ_BUY_DECISION_ROWS_SQL = text(
    """
    SELECT DISTINCT ON (symbol)
        *
    FROM trade.us_buy_decision_log
    WHERE trade_date <= :trade_date
    ORDER BY symbol, trade_date DESC, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
    """
)

READ_PAPER_BUY_ROWS_SQL = text(
    """
    SELECT *
    FROM trade.us_paper_order
    WHERE trade_date <= :trade_date
      AND side = 'BUY'
    ORDER BY trade_date DESC, created_at DESC, symbol
    """
)


def _ensure_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _load_optional_rows(sql: text, params: dict[str, object], relation_name: str) -> list[dict[str, object]]:
    if not relation_exists(relation_name):
        return []
    try:
        engine = get_us_engine()
        with engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [dict(row) for row in rows]
    except Exception:
        return []


def load_portfolio_state(
    cfg: TradeOrchestrationConfig,
    *,
    trade_date: date,
    account_id: str,
    sell_report: dict[str, object] | None = None,
    buy_report: dict[str, object] | None = None,
) -> dict[str, object]:
    data_quality_flags: list[str] = []
    events: list[dict[str, object]] = []

    if sell_report is not None:
        positions = list(sell_report.get("positions") or [])
        decisions = list(sell_report.get("decisions") or [])
    else:
        sell_cfg = load_sell_automation_config()
        loaded = load_paper_positions(sell_cfg, account_id=account_id, requested_trade_date=trade_date)
        positions = list(loaded.get("positions") or [])
        decisions = []
        events.extend(list(loaded.get("events") or []))

    sell_decision_rows = _load_optional_rows(READ_SELL_DECISION_ROWS_SQL, {"trade_date": trade_date}, "trade.us_sell_decision_log")
    buy_decision_rows = _load_optional_rows(READ_BUY_DECISION_ROWS_SQL, {"trade_date": trade_date}, "trade.us_buy_decision_log")
    paper_buy_order_rows = _load_optional_rows(READ_PAPER_BUY_ROWS_SQL, {"trade_date": trade_date}, "trade.us_paper_order")
    paper_sell_order_rows = _load_optional_rows(READ_SELL_ORDER_ROWS_SQL, {"trade_date": trade_date}, "trade.us_paper_sell_order")

    open_positions = []
    open_position_map: dict[str, dict[str, object]] = {}
    symbol_counter = Counter()
    for position in positions:
        symbol = str(position.get("symbol") or "").upper()
        if not symbol:
            continue
        symbol_counter[symbol] += 1
        open_positions.append(position)
        open_position_map[symbol] = position
    duplicate_symbols = sorted([symbol for symbol, count in symbol_counter.items() if count > 1])
    if duplicate_symbols:
        data_quality_flags.append("DUPLICATE_OPEN_POSITION_SYMBOL")

    current_decisions = decisions or sell_decision_rows
    sell_signals = []
    sell_signal_map: dict[str, dict[str, object]] = {}
    review_required_symbols: list[str] = []
    review_required_map: dict[str, dict[str, object]] = {}
    for row in current_decisions:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        decision = str(row.get("decision") or "").upper()
        if decision in {"SELL", "PARTIAL_SELL"}:
            item = {
                "symbol": symbol,
                "decision": decision,
                "exit_reason": row.get("exit_reason"),
                "paper_position_id": row.get("paper_position_id"),
            }
            sell_signals.append(item)
            sell_signal_map[symbol] = item
        if decision == "REVIEW_REQUIRED" or bool(row.get("review_required")):
            item = {
                "symbol": symbol,
                "decision": "REVIEW_REQUIRED",
                "exit_reason": row.get("exit_reason") or row.get("error_message"),
                "paper_position_id": row.get("paper_position_id"),
            }
            review_required_symbols.append(symbol)
            review_required_map[symbol] = item

    closed_positions = []
    cooldown_symbols: list[str] = []
    cooldown_map: dict[str, dict[str, object]] = {}
    for row in paper_sell_order_rows:
        symbol = str(row.get("symbol") or "").upper()
        trade_dt = _ensure_date(row.get("trade_date"))
        sell_action = str(row.get("sell_action") or "").upper()
        if not symbol or trade_dt is None:
            continue
        if sell_action != "FULL_SELL":
            continue
        cooldown_until = trade_dt + timedelta(days=cfg.block_buy_after_full_exit_days)
        item = {
            "symbol": symbol,
            "exit_trade_date": trade_dt.isoformat(),
            "exit_reason": row.get("exit_reason"),
            "cooldown_until": cooldown_until.isoformat(),
        }
        closed_positions.append(item)
        if trade_date <= cooldown_until:
            cooldown_symbols.append(symbol)
            cooldown_map[symbol] = item

    paper_buy_symbols_today = {
        str(row.get("symbol") or "").upper()
        for row in paper_buy_order_rows
        if _ensure_date(row.get("trade_date")) == trade_date
    }
    if buy_report is not None:
        for row in buy_report.get("paper_orders") or []:
            if str(row.get("trade_date") or "") == trade_date.isoformat():
                paper_buy_symbols_today.add(str(row.get("symbol") or "").upper())

    status = "OK"
    if data_quality_flags:
        status = "PORTFOLIO_STATE_INCONSISTENT"
        events.append({"severity": "ERROR", "reason_code": status, "detail": ", ".join(data_quality_flags)})

    return {
        "trade_date": trade_date.isoformat(),
        "status": status,
        "data_quality_flags": data_quality_flags,
        "events": events,
        "open_positions": open_positions,
        "open_position_map": open_position_map,
        "closed_positions": closed_positions,
        "sell_signals": sell_signals,
        "sell_signal_map": sell_signal_map,
        "review_required_symbols": sorted(set(review_required_symbols)),
        "review_required_map": review_required_map,
        "cooldown_symbols": sorted(set(cooldown_symbols)),
        "cooldown_map": cooldown_map,
        "latest_paper_buy_orders": paper_buy_order_rows,
        "latest_paper_sell_orders": paper_sell_order_rows,
        "latest_sell_decisions": current_decisions,
        "latest_buy_decisions": buy_decision_rows,
        "paper_buy_symbols_today": sorted(symbol for symbol in paper_buy_symbols_today if symbol),
    }
