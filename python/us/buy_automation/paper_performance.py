from __future__ import annotations

from datetime import date
import os

from python.us.us_config import parse_iso_date
from python.us.us_db import fetch_price_history_for_tickers


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_map(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("ticker") or "").upper(), []).append(row)
    return grouped


def _latest_price_on_or_before(history: list[dict[str, object]], as_of_date: date) -> float | None:
    candidates = [row for row in history if row.get("trade_date") and row["trade_date"] <= as_of_date]
    if not candidates:
        return None
    latest = candidates[-1]
    return _safe_float(latest.get("adj_close_price")) or _safe_float(latest.get("close_price"))


def _price_on_or_before(history: list[dict[str, object]], target_date: date) -> float | None:
    candidates = [row for row in history if row.get("trade_date") and row["trade_date"] <= target_date]
    if not candidates:
        return None
    chosen = candidates[-1]
    return _safe_float(chosen.get("adj_close_price")) or _safe_float(chosen.get("close_price"))


def build_paper_performance(
    paper_orders: list[dict[str, object]],
    *,
    benchmark_symbol: str | None = None,
    as_of_date: date | None = None,
) -> dict[str, object]:
    benchmark = str(benchmark_symbol or os.environ.get("US_BUY_BENCHMARK_SYMBOL", "SPY")).upper()
    if not paper_orders:
        return {"benchmark_symbol": benchmark, "rows": [], "summary": {"count": 0, "price_data_missing_count": 0}}

    trade_dates = [parse_iso_date(str(row.get("trade_date")), field_name="trade_date") for row in paper_orders if row.get("trade_date")]
    valid_dates = [item for item in trade_dates if item is not None]
    effective_as_of = as_of_date or (max(valid_dates) if valid_dates else date.today())

    symbols = sorted({str(row.get("symbol") or "").upper() for row in paper_orders if row.get("symbol")} | {benchmark})
    try:
        history_rows = fetch_price_history_for_tickers(symbols, end_date=effective_as_of)
    except Exception:
        history_rows = []
    history_map = _history_map(history_rows)

    rows: list[dict[str, object]] = []
    price_data_missing_count = 0
    for order in paper_orders:
        trade_date_value = parse_iso_date(str(order.get("trade_date")), field_name="trade_date")
        symbol = str(order.get("symbol") or "").upper()
        qty = _safe_float(order.get("paper_order_qty"))
        buy_price = _safe_float(order.get("assumed_fill_price")) or _safe_float(order.get("paper_order_price"))
        invested_amount = _safe_float(order.get("paper_order_amount"))
        latest_price = _latest_price_on_or_before(history_map.get(symbol, []), effective_as_of) if trade_date_value else None
        benchmark_buy_price = _price_on_or_before(history_map.get(benchmark, []), trade_date_value) if trade_date_value else None
        benchmark_latest_price = _latest_price_on_or_before(history_map.get(benchmark, []), effective_as_of) if trade_date_value else None

        status = "OK"
        current_value = unrealized_pnl = unrealized_pnl_pct = benchmark_return_pct = excess_return_pct = None
        if not trade_date_value or qty is None or buy_price is None or invested_amount is None or latest_price is None:
            status = "PRICE_DATA_MISSING"
            price_data_missing_count += 1
        else:
            current_value = round(qty * latest_price, 6)
            unrealized_pnl = round(current_value - invested_amount, 6)
            unrealized_pnl_pct = round((unrealized_pnl / invested_amount), 6) if invested_amount else None
            if benchmark_buy_price and benchmark_latest_price and benchmark_buy_price > 0:
                benchmark_return_pct = round((benchmark_latest_price / benchmark_buy_price) - 1.0, 6)
                if unrealized_pnl_pct is not None:
                    excess_return_pct = round(unrealized_pnl_pct - benchmark_return_pct, 6)
            else:
                status = "PRICE_DATA_MISSING"
                price_data_missing_count += 1

        rows.append(
            {
                "paper_order_id": order.get("paper_order_id"),
                "symbol": symbol,
                "buy_trade_date": str(order.get("trade_date") or ""),
                "buy_price": buy_price,
                "latest_price": latest_price,
                "quantity": qty,
                "invested_amount": invested_amount,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "holding_days": (effective_as_of - trade_date_value).days if trade_date_value else None,
                "benchmark_symbol": benchmark,
                "benchmark_return_pct": benchmark_return_pct,
                "excess_return_pct": excess_return_pct,
                "status": status,
            }
        )

    return {
        "benchmark_symbol": benchmark,
        "as_of_date": effective_as_of.isoformat(),
        "rows": rows,
        "summary": {
            "count": len(rows),
            "price_data_missing_count": price_data_missing_count,
        },
    }
