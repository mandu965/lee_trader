from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import logging
import math
from pathlib import Path
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import (
    ensure_us_paper_trading_tables,
    fetch_mixed_price_rows_for_tickers_between,
    fetch_us_paper_account_rows,
    fetch_us_paper_fill_rows,
    fetch_us_paper_order_rows,
    fetch_us_paper_position_rows,
    get_us_engine,
)


LOGGER = logging.getLogger("us_paper_fill_sim")
EPSILON = 1e-9

UPSERT_PAPER_FILL_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_fill (
        paper_fill_id,
        paper_order_id,
        account_id,
        trade_date,
        symbol,
        side,
        filled_qty,
        filled_price,
        filled_amount,
        commission,
        slippage_amount,
        fill_status,
        created_at
    ) VALUES (
        :paper_fill_id,
        :paper_order_id,
        :account_id,
        :trade_date,
        :symbol,
        :side,
        :filled_qty,
        :filled_price,
        :filled_amount,
        :commission,
        :slippage_amount,
        :fill_status,
        now()
    )
    ON CONFLICT (paper_fill_id) DO NOTHING
    """
)

UPDATE_PAPER_ORDER_STATUS_SQL = text(
    """
    UPDATE paper.us_stock_paper_order
    SET status = :status,
        reason = :reason,
        reject_reason = :reject_reason,
        updated_at = now()
    WHERE paper_order_id = :paper_order_id
    """
)

UPSERT_PAPER_POSITION_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_position (
        account_id,
        symbol,
        qty,
        avg_price,
        cost_amount,
        last_price,
        market_value,
        unrealized_pnl,
        unrealized_pnl_pct,
        realized_pnl,
        last_trade_date,
        last_price_date,
        status,
        created_at,
        updated_at
    ) VALUES (
        :account_id,
        :symbol,
        :qty,
        :avg_price,
        :cost_amount,
        :last_price,
        :market_value,
        :unrealized_pnl,
        :unrealized_pnl_pct,
        :realized_pnl,
        :last_trade_date,
        :last_price_date,
        :status,
        now(),
        now()
    )
    ON CONFLICT (account_id, symbol) DO UPDATE SET
        qty = EXCLUDED.qty,
        avg_price = EXCLUDED.avg_price,
        cost_amount = EXCLUDED.cost_amount,
        last_price = EXCLUDED.last_price,
        market_value = EXCLUDED.market_value,
        unrealized_pnl = EXCLUDED.unrealized_pnl,
        unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
        realized_pnl = EXCLUDED.realized_pnl,
        last_trade_date = EXCLUDED.last_trade_date,
        last_price_date = EXCLUDED.last_price_date,
        status = EXCLUDED.status,
        updated_at = now()
    """
)

UPDATE_PAPER_ACCOUNT_SQL = text(
    """
    UPDATE paper.us_stock_paper_account
    SET cash_balance = :cash_balance,
        reserved_cash = :reserved_cash,
        market_value = :market_value,
        equity_value = :equity_value,
        realized_pnl = :realized_pnl,
        unrealized_pnl = :unrealized_pnl,
        total_pnl = :total_pnl,
        status = :status,
        updated_at = now()
    WHERE account_id = :account_id
    """
)

LOCK_PAPER_ORDER_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_order
    WHERE paper_order_id = :paper_order_id
    FOR UPDATE
    """
)

LOCK_PAPER_ACCOUNT_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_account
    WHERE account_id = :account_id
    FOR UPDATE
    """
)

READ_PAPER_FILL_EXISTS_SQL = text(
    """
    SELECT paper_fill_id
    FROM paper.us_stock_paper_fill
    WHERE paper_order_id = :paper_order_id
    LIMIT 1
    """
)


@dataclass(frozen=True)
class FillSimulationConfig:
    commission_per_trade: float
    slippage_bps: float
    real_order_blocked: bool
    log_level: str


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate paper-only fills for created US stock paper orders.")
    parser.add_argument("--as-of-date", required=True, help="Fill availability cutoff date. Format: YYYY-MM-DD.")
    parser.add_argument("--account-id", required=True, help="Paper account ID.")
    parser.add_argument("--side", choices=["BUY", "SELL", "ALL"], default="ALL")
    parser.add_argument("--order-id", default=None, help="Optional single paper_order_id filter.")
    parser.add_argument("--dry-run", action="store_true", help="Preview fills without DB writes.")
    return parser.parse_args()


def assert_paper_trading_only() -> None:
    cfg = load_us_paper_trading_config()
    if not cfg.real_order_blocked:
        raise RuntimeError("US_PAPER_REAL_ORDER_BLOCKED must be true for paper fill simulation.")
    LOGGER.info("[SAFETY] Paper fill simulation only. Real order APIs are blocked.")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _fmt_pct(value: object) -> str:
    num = _safe_float(value)
    if num is None:
        return "N/A"
    return f"{num * 100:.2f}%"


def _load_fill_config(account_id: str) -> FillSimulationConfig:
    cfg = load_us_paper_trading_config(account_id=account_id)
    return FillSimulationConfig(
        commission_per_trade=float(cfg.commission_per_trade),
        slippage_bps=float(cfg.slippage_bps),
        real_order_blocked=bool(cfg.real_order_blocked),
        log_level=str(cfg.log_level).upper() or "INFO",
    )


def _price_value(row: dict[str, object]) -> float | None:
    return _safe_float(row.get("adj_close_price")) or _safe_float(row.get("close_price")) or _safe_float(row.get("price"))


def _build_price_lookup(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        symbol = str(row.get("ticker") or "").upper()
        trade_date = row.get("trade_date")
        price = _price_value(row)
        if not symbol or not isinstance(trade_date, date) or price is None:
            continue
        output.setdefault(symbol, []).append({"trade_date": trade_date, "price": price})
    for symbol in output:
        output[symbol].sort(key=lambda item: item["trade_date"])
    return output


def _next_trade_date(symbol: str, trade_date: date, price_lookup: dict[str, list[dict[str, object]]]) -> date | None:
    for row in price_lookup.get(symbol.upper(), []):
        row_date = row.get("trade_date")
        if isinstance(row_date, date) and row_date > trade_date:
            return row_date
    return None


def _price_on_date(symbol: str, target_date: date | None, price_lookup: dict[str, list[dict[str, object]]]) -> float | None:
    if target_date is None:
        return None
    for row in price_lookup.get(symbol.upper(), []):
        if row.get("trade_date") == target_date:
            return _safe_float(row.get("price"))
    return None


def _latest_price_on_or_before(symbol: str, as_of_date: date, price_lookup: dict[str, list[dict[str, object]]]) -> tuple[float | None, date | None]:
    latest_price: float | None = None
    latest_date: date | None = None
    for row in price_lookup.get(symbol.upper(), []):
        row_date = row.get("trade_date")
        if not isinstance(row_date, date) or row_date > as_of_date:
            continue
        latest_price = _safe_float(row.get("price"))
        latest_date = row_date
    return latest_price, latest_date


def _build_fill_id(paper_order_id: str, fill_date: date) -> str:
    return f"USPF_{paper_order_id}_{fill_date:%Y%m%d}"


def _order_sort_key(row: dict[str, object]) -> tuple[date, object, str]:
    trade_date = row.get("trade_date")
    if not isinstance(trade_date, date):
        trade_date = date.min
    created_at = row.get("created_at")
    order_id = str(row.get("paper_order_id") or "")
    return trade_date, created_at, order_id


def _refresh_positions_for_mark_to_market(
    positions_by_symbol: dict[str, dict[str, object]],
    *,
    as_of_date: date,
    price_lookup: dict[str, list[dict[str, object]]],
) -> None:
    for symbol, row in positions_by_symbol.items():
        qty = _safe_float(row.get("qty")) or 0.0
        cost_amount = _safe_float(row.get("cost_amount")) or 0.0
        status = "OPEN" if qty > EPSILON else "CLOSED"
        last_price, last_price_date = _latest_price_on_or_before(symbol, as_of_date, price_lookup)
        if last_price is None:
            last_price = _safe_float(row.get("last_price")) or _safe_float(row.get("avg_price")) or 0.0
            last_price_date = row.get("last_price_date") if isinstance(row.get("last_price_date"), date) else None
        market_value = qty * last_price
        unrealized_pnl = market_value - cost_amount
        unrealized_pnl_pct = (unrealized_pnl / cost_amount) if cost_amount > EPSILON else None
        row["qty"] = round(qty, 6)
        row["cost_amount"] = round(cost_amount, 6)
        row["last_price"] = round(last_price, 6) if last_price is not None else None
        row["market_value"] = round(market_value, 6)
        row["unrealized_pnl"] = round(unrealized_pnl, 6)
        row["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 6) if unrealized_pnl_pct is not None else None
        row["status"] = status
        row["last_price_date"] = last_price_date
        if status == "CLOSED":
            row["market_value"] = 0.0
            row["unrealized_pnl"] = 0.0
            row["unrealized_pnl_pct"] = None
            row["last_price"] = 0.0 if last_price is None else row["last_price"]


def _refresh_account_state(
    account_row: dict[str, object],
    positions_by_symbol: dict[str, dict[str, object]],
    *,
    as_of_date: date,
    price_lookup: dict[str, list[dict[str, object]]],
) -> None:
    _refresh_positions_for_mark_to_market(positions_by_symbol, as_of_date=as_of_date, price_lookup=price_lookup)
    market_value = 0.0
    unrealized_pnl = 0.0
    for row in positions_by_symbol.values():
        if str(row.get("status") or "OPEN").upper() == "OPEN":
            market_value += _safe_float(row.get("market_value")) or 0.0
            unrealized_pnl += _safe_float(row.get("unrealized_pnl")) or 0.0
    realized_pnl = _safe_float(account_row.get("realized_pnl")) or 0.0
    cash_balance = _safe_float(account_row.get("cash_balance")) or 0.0
    account_row["market_value"] = round(market_value, 6)
    account_row["equity_value"] = round(cash_balance + market_value, 6)
    account_row["unrealized_pnl"] = round(unrealized_pnl, 6)
    account_row["total_pnl"] = round(realized_pnl + unrealized_pnl, 6)


def _build_position_row(
    *,
    account_id: str,
    symbol: str,
    qty: float,
    avg_price: float | None,
    cost_amount: float,
    realized_pnl: float,
    last_trade_date: date | None,
) -> dict[str, object]:
    return {
        "account_id": account_id,
        "symbol": symbol,
        "qty": round(max(0.0, qty), 6),
        "avg_price": round(avg_price, 6) if avg_price is not None else None,
        "cost_amount": round(max(0.0, cost_amount), 6),
        "last_price": 0.0,
        "market_value": 0.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": None,
        "realized_pnl": round(realized_pnl, 6),
        "last_trade_date": last_trade_date,
        "last_price_date": None,
        "status": "OPEN" if qty > EPSILON else "CLOSED",
    }


def _reject_decision(
    order_row: dict[str, object],
    *,
    reject_reason: str,
    message: str,
) -> dict[str, object]:
    return {
        "paper_order_id": order_row.get("paper_order_id"),
        "symbol": order_row.get("symbol"),
        "side": order_row.get("side"),
        "status": "REJECTED",
        "reject_reason": reject_reason,
        "reason": message,
    }


def simulate_paper_fills(
    *,
    as_of_date: date,
    account_row: dict[str, object],
    order_rows: list[dict[str, object]],
    position_rows: list[dict[str, object]],
    price_rows: list[dict[str, object]],
    cfg: FillSimulationConfig,
    side_option: str,
    existing_fill_rows: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]], dict[str, object]]:
    account_state = deepcopy(account_row)
    position_state = {
        str(row.get("symbol") or "").upper(): deepcopy(row)
        for row in position_rows
        if str(row.get("symbol") or "").strip()
    }
    existing_fill_order_ids = {str(row.get("paper_order_id") or "") for row in (existing_fill_rows or [])}
    price_lookup = _build_price_lookup(price_rows)
    orders = sorted(order_rows, key=_order_sort_key)
    decisions: list[dict[str, object]] = []
    counts = {
        "orders_read": len(order_rows),
        "fill_candidates": 0,
        "filled_count": 0,
        "rejected_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "buy_filled_count": 0,
        "sell_filled_count": 0,
        "total_buy_amount": 0.0,
        "total_sell_amount": 0.0,
        "commission_total": 0.0,
        "cash_balance_before": _safe_float(account_state.get("cash_balance")) or 0.0,
    }

    for order in orders:
        order_id = str(order.get("paper_order_id") or "")
        side = str(order.get("side") or "").upper()
        status = str(order.get("status") or "").upper()
        if side_option != "ALL" and side != side_option:
            counts["skipped_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": order.get("symbol"), "side": side, "status": "SKIPPED", "reason": "side_filter"})
            continue
        if status != "CREATED":
            counts["skipped_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": order.get("symbol"), "side": side, "status": "SKIPPED", "reason": "order_not_created"})
            continue
        if order_id in existing_fill_order_ids:
            counts["skipped_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": order.get("symbol"), "side": side, "status": "SKIPPED", "reason": "duplicate_fill"})
            continue

        symbol = str(order.get("symbol") or "").upper()
        trade_date = order.get("trade_date")
        if not isinstance(trade_date, date):
            counts["error_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": symbol, "side": side, "status": "ERROR", "reject_reason": "error", "reason": "Order trade_date is missing or invalid."})
            continue
        fill_date = _next_trade_date(symbol, trade_date, price_lookup)
        if fill_date is None or fill_date > as_of_date:
            counts["skipped_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": symbol, "side": side, "status": "SKIPPED", "reason": "not_fillable_yet"})
            continue

        counts["fill_candidates"] += 1
        order_type = str(order.get("order_type") or "MARKET").upper()
        if order_type != "MARKET":
            decision = _reject_decision(order, reject_reason="unsupported_order_type", message=f"Order type {order_type} is not supported in paper fill simulation.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue

        market_close = _price_on_date(symbol, fill_date, price_lookup)
        if market_close is None:
            decision = _reject_decision(order, reject_reason="missing_fill_price", message="Fill price is missing for the next trading day.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue
        if market_close <= 0:
            decision = _reject_decision(order, reject_reason="invalid_fill_price", message="Fill price must be positive.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue

        qty = _safe_float(order.get("order_qty")) or 0.0
        if qty <= EPSILON:
            decision = _reject_decision(order, reject_reason="qty_zero", message="Order quantity must be positive for paper fill simulation.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue

        slippage_rate = cfg.slippage_bps / 10000.0
        if side == "BUY":
            filled_price = market_close * (1.0 + slippage_rate)
        else:
            filled_price = market_close * (1.0 - slippage_rate)
        filled_amount = qty * filled_price
        slippage_amount = abs(filled_price - market_close) * qty
        commission = cfg.commission_per_trade

        account_cash = _safe_float(account_state.get("cash_balance")) or 0.0
        account_realized = _safe_float(account_state.get("realized_pnl")) or 0.0

        fill_row = {
            "paper_fill_id": _build_fill_id(order_id, fill_date),
            "paper_order_id": order_id,
            "account_id": str(order.get("account_id") or ""),
            "trade_date": fill_date,
            "symbol": symbol,
            "side": side,
            "filled_qty": round(qty, 6),
            "filled_price": round(filled_price, 6),
            "filled_amount": round(filled_amount, 6),
            "commission": round(commission, 6),
            "slippage_amount": round(slippage_amount, 6),
            "fill_status": "FILLED",
        }

        if side == "BUY":
            total_cash_needed = filled_amount + commission
            if account_cash + EPSILON < total_cash_needed:
                decision = _reject_decision(order, reject_reason="insufficient_cash_at_fill", message="Cash balance is insufficient at fill time.")
                counts["rejected_count"] += 1
                decisions.append(decision)
                continue

            existing = position_state.get(symbol)
            old_qty = _safe_float(existing.get("qty")) if existing else 0.0
            old_cost = _safe_float(existing.get("cost_amount")) if existing else 0.0
            old_realized = _safe_float(existing.get("realized_pnl")) if existing else 0.0
            new_qty = (old_qty or 0.0) + qty
            new_cost = (old_cost or 0.0) + filled_amount + commission
            new_avg = new_cost / new_qty if new_qty > EPSILON else None
            position_state[symbol] = _build_position_row(
                account_id=str(order.get("account_id") or ""),
                symbol=symbol,
                qty=new_qty,
                avg_price=new_avg,
                cost_amount=new_cost,
                realized_pnl=old_realized or 0.0,
                last_trade_date=fill_date,
            )
            account_state["cash_balance"] = round(account_cash - total_cash_needed, 6)
            _refresh_account_state(account_state, position_state, as_of_date=fill_date, price_lookup=price_lookup)
            if (_safe_float(account_state.get("cash_balance")) or 0.0) < -EPSILON:
                decision = _reject_decision(order, reject_reason="cash_negative_after_fill", message="Cash balance would become negative after buy fill.")
                counts["rejected_count"] += 1
                decisions.append(decision)
                continue

            counts["filled_count"] += 1
            counts["buy_filled_count"] += 1
            counts["total_buy_amount"] += filled_amount
            counts["commission_total"] += commission
            decisions.append(
                {
                    "paper_order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "status": "FILLED",
                    "reason": f"BUY filled on next trading day close. market_close={market_close:.6f}, slippage_bps={cfg.slippage_bps:.2f}.",
                    "reject_reason": None,
                    "fill_date": fill_date,
                    "market_close": round(market_close, 6),
                    "filled_price": round(filled_price, 6),
                    "filled_qty": round(qty, 6),
                    "filled_amount": round(filled_amount, 6),
                    "commission": round(commission, 6),
                    "fill_row": fill_row,
                    "account_after": deepcopy(account_state),
                    "positions_after": deepcopy(list(position_state.values())),
                }
            )
            continue

        if side != "SELL":
            counts["error_count"] += 1
            decisions.append({"paper_order_id": order_id, "symbol": symbol, "side": side, "status": "ERROR", "reject_reason": "error", "reason": f"Unsupported side {side}."})
            continue

        existing = position_state.get(symbol)
        if existing is None:
            decision = _reject_decision(order, reject_reason="position_not_found", message="Open position was not found for sell order.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue
        existing_qty = _safe_float(existing.get("qty")) or 0.0
        if existing_qty + EPSILON < qty:
            decision = _reject_decision(order, reject_reason="insufficient_position_qty", message="Open position quantity is smaller than sell quantity.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue

        avg_price = _safe_float(existing.get("avg_price")) or 0.0
        old_cost_amount = _safe_float(existing.get("cost_amount")) or 0.0
        old_realized = _safe_float(existing.get("realized_pnl")) or 0.0
        sell_cost_basis = avg_price * qty
        net_proceeds = filled_amount - commission
        realized_pnl = net_proceeds - sell_cost_basis
        remaining_qty = max(0.0, existing_qty - qty)
        remaining_cost_amount = max(0.0, old_cost_amount - sell_cost_basis)
        remaining_avg = (remaining_cost_amount / remaining_qty) if remaining_qty > EPSILON else None
        position_state[symbol] = _build_position_row(
            account_id=str(order.get("account_id") or ""),
            symbol=symbol,
            qty=remaining_qty,
            avg_price=remaining_avg,
            cost_amount=remaining_cost_amount,
            realized_pnl=(old_realized or 0.0) + realized_pnl,
            last_trade_date=fill_date,
        )
        account_state["cash_balance"] = round(account_cash + net_proceeds, 6)
        account_state["realized_pnl"] = round(account_realized + realized_pnl, 6)
        _refresh_account_state(account_state, position_state, as_of_date=fill_date, price_lookup=price_lookup)
        if (_safe_float(position_state[symbol].get("qty")) or 0.0) < -EPSILON:
            decision = _reject_decision(order, reject_reason="position_negative_after_fill", message="Position quantity would become negative after sell fill.")
            counts["rejected_count"] += 1
            decisions.append(decision)
            continue

        counts["filled_count"] += 1
        counts["sell_filled_count"] += 1
        counts["total_sell_amount"] += filled_amount
        counts["commission_total"] += commission
        decisions.append(
            {
                "paper_order_id": order_id,
                "symbol": symbol,
                "side": side,
                "status": "FILLED",
                "reason": f"SELL filled on next trading day close. market_close={market_close:.6f}, slippage_bps={cfg.slippage_bps:.2f}.",
                "reject_reason": None,
                "fill_date": fill_date,
                "market_close": round(market_close, 6),
                "filled_price": round(filled_price, 6),
                "filled_qty": round(qty, 6),
                "filled_amount": round(filled_amount, 6),
                "commission": round(commission, 6),
                "realized_pnl": round(realized_pnl, 6),
                "fill_row": fill_row,
                "account_after": deepcopy(account_state),
                "positions_after": deepcopy(list(position_state.values())),
            }
        )

    counts["cash_balance_after"] = _safe_float(account_state.get("cash_balance")) or 0.0
    counts["market_value_after"] = _safe_float(account_state.get("market_value")) or 0.0
    counts["equity_value_after"] = _safe_float(account_state.get("equity_value")) or 0.0
    counts["realized_pnl_after"] = _safe_float(account_state.get("realized_pnl")) or 0.0
    counts["unrealized_pnl_after"] = _safe_float(account_state.get("unrealized_pnl")) or 0.0
    return decisions, account_state, list(position_state.values()), counts


def validate_paper_account_integrity(account_id: str) -> list[str]:
    account_rows = fetch_us_paper_account_rows(account_id=account_id)
    if not account_rows:
        return [f"account_not_found:{account_id}"]
    account_row = account_rows[0]
    position_rows = fetch_us_paper_position_rows(account_id=account_id)
    order_rows = fetch_us_paper_order_rows(account_id=account_id)
    fill_rows = fetch_us_paper_fill_rows(account_id=account_id)

    issues: list[str] = []
    cash_balance = _safe_float(account_row.get("cash_balance")) or 0.0
    market_value = _safe_float(account_row.get("market_value")) or 0.0
    equity_value = _safe_float(account_row.get("equity_value")) or 0.0
    realized_pnl = _safe_float(account_row.get("realized_pnl")) or 0.0
    unrealized_pnl = _safe_float(account_row.get("unrealized_pnl")) or 0.0
    total_pnl = _safe_float(account_row.get("total_pnl")) or 0.0

    if cash_balance < -EPSILON:
        issues.append("cash_balance_negative")
    if abs(equity_value - (cash_balance + market_value)) > 1e-4:
        issues.append("equity_value_mismatch")
    if abs(total_pnl - (realized_pnl + unrealized_pnl)) > 1e-4:
        issues.append("total_pnl_mismatch")

    fills_by_order = {str(row.get("paper_order_id") or "") for row in fill_rows}
    for row in position_rows:
        status = str(row.get("status") or "").upper()
        qty = _safe_float(row.get("qty")) or 0.0
        if qty < -EPSILON:
            issues.append(f"negative_position_qty:{row.get('symbol')}")
        if status == "OPEN" and qty <= EPSILON:
            issues.append(f"open_position_zero_qty:{row.get('symbol')}")
        if status == "CLOSED" and abs(qty) > EPSILON:
            issues.append(f"closed_position_nonzero_qty:{row.get('symbol')}")
    for row in order_rows:
        if str(row.get("status") or "").upper() == "FILLED":
            order_id = str(row.get("paper_order_id") or "")
            if order_id not in fills_by_order:
                issues.append(f"filled_order_missing_fill:{order_id}")
    for row in fill_rows:
        filled_qty = _safe_float(row.get("filled_qty")) or 0.0
        filled_price = _safe_float(row.get("filled_price")) or 0.0
        filled_amount = _safe_float(row.get("filled_amount")) or 0.0
        if abs(filled_amount - (filled_qty * filled_price)) > 1e-4:
            issues.append(f"fill_amount_mismatch:{row.get('paper_fill_id')}")
    return issues


def _persist_decision(decision: dict[str, object]) -> bool:
    order_id = str(decision.get("paper_order_id") or "")
    account_after = decision.get("account_after")
    positions_after = decision.get("positions_after")
    engine = get_us_engine()
    with engine.begin() as conn:
        order_row = conn.execute(LOCK_PAPER_ORDER_SQL, {"paper_order_id": order_id}).mappings().first()
        if order_row is None or str(order_row.get("status") or "").upper() != "CREATED":
            return False
        existing_fill_id = conn.execute(READ_PAPER_FILL_EXISTS_SQL, {"paper_order_id": order_id}).scalar()
        if existing_fill_id is not None:
            return False
        if decision.get("status") == "FILLED":
            account_id = str(order_row.get("account_id") or "")
            account_row = conn.execute(LOCK_PAPER_ACCOUNT_SQL, {"account_id": account_id}).mappings().first()
            if account_row is None:
                raise RuntimeError(f"Paper account not found while applying fill: {account_id}")
            conn.execute(UPSERT_PAPER_FILL_SQL, decision["fill_row"])
            conn.execute(
                UPDATE_PAPER_ORDER_STATUS_SQL,
                {
                    "paper_order_id": order_id,
                    "status": "FILLED",
                    "reason": decision.get("reason"),
                    "reject_reason": None,
                },
            )
            for position_row in positions_after if isinstance(positions_after, list) else []:
                conn.execute(UPSERT_PAPER_POSITION_SQL, position_row)
            if not isinstance(account_after, dict):
                raise RuntimeError("account_after payload is missing for FILLED decision.")
            conn.execute(
                UPDATE_PAPER_ACCOUNT_SQL,
                {
                    "account_id": account_after.get("account_id"),
                    "cash_balance": account_after.get("cash_balance"),
                    "reserved_cash": account_after.get("reserved_cash"),
                    "market_value": account_after.get("market_value"),
                    "equity_value": account_after.get("equity_value"),
                    "realized_pnl": account_after.get("realized_pnl"),
                    "unrealized_pnl": account_after.get("unrealized_pnl"),
                    "total_pnl": account_after.get("total_pnl"),
                    "status": account_after.get("status"),
                },
            )
            return True

        conn.execute(
            UPDATE_PAPER_ORDER_STATUS_SQL,
            {
                "paper_order_id": order_id,
                "status": decision.get("status"),
                "reason": decision.get("reason"),
                "reject_reason": decision.get("reject_reason"),
            },
        )
        return True


def _print_dry_run(*, as_of_date: date, account_id: str, decisions: list[dict[str, object]], counts: dict[str, object]) -> None:
    print("[Paper Fill Dry Run]")
    print(f"As Of Date: {as_of_date.isoformat()}")
    print(f"Account: {account_id}")
    print("")
    print(f"Orders Read: {counts['orders_read']}")
    print(f"Fill Candidates: {counts['fill_candidates']}")
    print(f"Rejected Candidates: {counts['rejected_count']}")
    print(f"Skipped: {counts['skipped_count']}")

    fill_rows = [row for row in decisions if row.get("status") == "FILLED"]
    reject_rows = [row for row in decisions if row.get("status") == "REJECTED"]
    if fill_rows:
        print("")
        print("[Fill Preview]")
        print("Order ID | Side | Symbol | Fill Date | Market Close | Slippage Bps | Filled Price | Qty | Amount | Commission")
        for row in fill_rows[:10]:
            print(
                f"{row.get('paper_order_id')} | {row.get('side')} | {row.get('symbol')} | {row.get('fill_date')} | "
                f"{row.get('market_close')} | {counts.get('slippage_bps', '') or ''} | {row.get('filled_price')} | "
                f"{row.get('filled_qty')} | {row.get('filled_amount')} | {row.get('commission')}"
            )
    if reject_rows:
        print("")
        print("[Reject Preview]")
        print("Order ID | Symbol | Reason")
        for row in reject_rows[:10]:
            print(f"{row.get('paper_order_id')} | {row.get('symbol')} | {row.get('reject_reason')}")


def _print_summary(*, as_of_date: date, account_id: str, counts: dict[str, object]) -> None:
    print("[Paper Fill Summary]")
    print(f"as_of_date: {as_of_date.isoformat()}")
    print(f"account_id: {account_id}")
    print(f"orders_read: {counts['orders_read']}")
    print(f"fill_candidates: {counts['fill_candidates']}")
    print(f"filled_count: {counts['filled_count']}")
    print(f"rejected_count: {counts['rejected_count']}")
    print(f"error_count: {counts['error_count']}")
    print(f"skipped_count: {counts['skipped_count']}")
    print(f"buy_filled_count: {counts['buy_filled_count']}")
    print(f"sell_filled_count: {counts['sell_filled_count']}")
    print(f"total_buy_amount: {counts['total_buy_amount']:.6f}")
    print(f"total_sell_amount: {counts['total_sell_amount']:.6f}")
    print(f"commission_total: {counts['commission_total']:.6f}")
    print(f"cash_balance_before: {counts['cash_balance_before']:.6f}")
    print(f"cash_balance_after: {counts['cash_balance_after']:.6f}")
    print(f"market_value_after: {counts['market_value_after']:.6f}")
    print(f"equity_value_after: {counts['equity_value_after']:.6f}")
    print(f"realized_pnl_after: {counts['realized_pnl_after']:.6f}")
    print(f"unrealized_pnl_after: {counts['unrealized_pnl_after']:.6f}")


def main() -> int:
    args = parse_args()
    as_of_date = parse_iso_date(args.as_of_date, field_name="as_of_date")
    if as_of_date is None:
        raise ValueError("as_of_date is required.")

    cfg = _load_fill_config(args.account_id)
    setup_logging(cfg.log_level)
    assert_paper_trading_only()
    ensure_us_paper_trading_tables()

    account_rows = fetch_us_paper_account_rows(account_id=args.account_id)
    if not account_rows:
        print(f"paper account not found: {args.account_id}")
        return 1
    account_row = account_rows[0]
    if str(account_row.get("status") or "").upper() != "ACTIVE":
        print(f"paper account is not ACTIVE: {args.account_id}")
        return 1

    order_rows = fetch_us_paper_order_rows(
        paper_order_id=args.order_id,
        account_id=args.account_id,
        side=None if args.side == "ALL" else args.side,
        status="CREATED",
    )
    if not order_rows:
        print(f"no created paper orders found for {args.account_id}")
        return 0

    position_rows = fetch_us_paper_position_rows(account_id=args.account_id)
    fill_rows = fetch_us_paper_fill_rows(account_id=args.account_id)
    symbols = sorted(
        {
            str(row.get("symbol") or "").upper()
            for row in order_rows + position_rows
            if str(row.get("symbol") or "").strip()
        }
    )
    earliest_trade_date = min((row.get("trade_date") for row in order_rows if isinstance(row.get("trade_date"), date)), default=as_of_date)
    price_rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=symbols,
        start_date=earliest_trade_date,
        end_date=as_of_date,
    )

    decisions, account_after, positions_after, counts = simulate_paper_fills(
        as_of_date=as_of_date,
        account_row=account_row,
        order_rows=order_rows,
        position_rows=position_rows,
        price_rows=price_rows,
        cfg=cfg,
        side_option=args.side,
        existing_fill_rows=fill_rows,
    )
    counts["slippage_bps"] = cfg.slippage_bps

    if args.dry_run:
        _print_dry_run(as_of_date=as_of_date, account_id=args.account_id, decisions=decisions, counts=counts)
        _print_summary(as_of_date=as_of_date, account_id=args.account_id, counts=counts)
        return 0

    filled_or_rejected = [row for row in decisions if row.get("status") in {"FILLED", "REJECTED", "ERROR"}]
    for decision in filled_or_rejected:
        persisted = _persist_decision(decision)
        if not persisted:
            counts["skipped_count"] += 1

    integrity_issues = validate_paper_account_integrity(args.account_id)
    if integrity_issues:
        LOGGER.warning("[WARNING] integrity issues detected: %s", ", ".join(integrity_issues))

    _print_summary(as_of_date=as_of_date, account_id=args.account_id, counts=counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
