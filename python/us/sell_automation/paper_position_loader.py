from __future__ import annotations

from collections import defaultdict
from datetime import date
import json

from sqlalchemy import text

import os

from python.us.sell_automation.config import SellAutomationConfig
from python.us.us_config import parse_iso_date
from python.us.us_db import (
    fetch_price_history_for_tickers,
    fetch_us_paper_fill_rows,
    fetch_us_paper_order_rows,
    fetch_us_paper_position_rows,
    get_us_engine,
    relation_exists,
)


LATEST_RANK_ROWS_SQL = text(
    """
    SELECT DISTINCT ON (symbol) *
    FROM recommend.us_stock_rank_daily
    WHERE symbol = ANY(:symbols)
      AND trade_date <= :trade_date
      AND source = :source
    ORDER BY symbol, trade_date DESC, rank_no ASC NULLS LAST
    """
)

LATEST_TRADE_DATE_SQL = text(
    """
    SELECT MAX(trade_date) AS max_trade_date
    FROM recommend.us_stock_rank_daily
    """
)

LATEST_PAPER_ORDER_DATE_SQL = text(
    """
    SELECT MAX(trade_date) AS max_trade_date
    FROM paper.us_stock_paper_order
    WHERE account_id = :account_id
    """
)

LATEST_TRADE_PAPER_ORDER_DATE_SQL = text(
    """
    SELECT MAX(trade_date) AS max_trade_date
    FROM trade.us_paper_order
    WHERE account_id = :account_id
    """
)

READ_TRADE_PAPER_BUY_ORDER_ROWS_SQL = text(
    """
    SELECT
        paper_order_id,
        trade_date,
        account_id,
        symbol,
        side,
        paper_order_qty AS order_qty,
        paper_order_price AS order_price,
        paper_order_amount,
        assumed_fill_price,
        assumed_fill_status,
        created_at,
        updated_at
    FROM trade.us_paper_order
    WHERE account_id = :account_id
      AND side = 'BUY'
    ORDER BY trade_date, created_at, symbol
    """
)

READ_TRADE_PAPER_SELL_ORDER_ROWS_SQL = text(
    """
    SELECT
        paper_sell_order_id,
        trade_date,
        paper_position_id,
        symbol,
        side,
        sell_action,
        sell_quantity,
        sell_price_ref,
        sell_amount,
        assumed_fill_status,
        created_at,
        updated_at
    FROM trade.us_paper_sell_order
    WHERE side = 'SELL'
    ORDER BY trade_date, created_at, symbol
    """
)

READ_TRADE_POSITION_SNAPSHOT_ROWS_SQL = text(
    """
    SELECT DISTINCT ON (symbol)
        paper_position_id,
        symbol,
        snapshot_date,
        remaining_quantity,
        latest_price,
        highest_price_since_entry,
        unrealized_pnl,
        unrealized_pnl_pct,
        holding_days,
        status,
        data_quality_flags
    FROM trade.us_paper_position_snapshot
    ORDER BY symbol, snapshot_date DESC, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
    """
)


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_value(row: dict[str, object]) -> float | None:
    return _safe_float(row.get("adj_close_price")) or _safe_float(row.get("close_price")) or _safe_float(row.get("price"))


def _history_map(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        symbol = str(row.get("ticker") or "").upper()
        trade_date = row.get("trade_date")
        price = _price_value(row)
        if not symbol or not isinstance(trade_date, date) or price is None:
            continue
        output[symbol].append({"trade_date": trade_date, "price": price})
    for symbol in output:
        output[symbol].sort(key=lambda item: item["trade_date"])
    return dict(output)


def _latest_price_on_or_before(history: list[dict[str, object]], target_date: date) -> float | None:
    latest = None
    for row in history:
        row_date = row.get("trade_date")
        if isinstance(row_date, date) and row_date <= target_date:
            latest = row
    return _safe_float(latest.get("price")) if latest else None


def _price_on_or_before(history: list[dict[str, object]], target_date: date) -> float | None:
    return _latest_price_on_or_before(history, target_date)


def _highest_price_since_entry(history: list[dict[str, object]], entry_date: date, target_date: date) -> float | None:
    prices = [
        _safe_float(row.get("price"))
        for row in history
        if isinstance(row.get("trade_date"), date) and entry_date <= row["trade_date"] <= target_date
    ]
    prices = [value for value in prices if value is not None]
    return max(prices) if prices else None


def _parse_score_detail(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_probability(rank_row: dict[str, object]) -> float | None:
    direct_keys = ("probability", "predicted_probability", "model_probability")
    for key in direct_keys:
        value = _safe_float(rank_row.get(key))
        if value is not None:
            return value
    detail = _parse_score_detail(rank_row.get("score_detail_json"))
    candidates = [
        detail.get("probability"),
        detail.get("model_probability"),
        detail.get("predicted_probability"),
        # ml_v1 uses prob_top20_20d / prob_top20_60d
        detail.get("prob_top20_20d"),
        detail.get("prob_top20_60d"),
        detail.get("meta", {}).get("probability") if isinstance(detail.get("meta"), dict) else None,
    ]
    for candidate in candidates:
        value = _safe_float(candidate)
        if value is not None:
            return value
    return None


def _paper_qty_for_order(order_row: dict[str, object], fill_map: dict[str, float]) -> float:
    order_id = str(order_row.get("paper_order_id") or "")
    fill_qty = fill_map.get(order_id)
    if fill_qty is not None:
        return max(0.0, fill_qty)
    for key in ("filled_qty", "order_qty", "paper_order_qty", "sell_quantity"):
        value = _safe_float(order_row.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _paper_price_for_order(order_row: dict[str, object]) -> float | None:
    for key in ("filled_price", "order_price", "paper_order_price", "assumed_fill_price", "sell_price_ref"):
        value = _safe_float(order_row.get(key))
        if value is not None:
            return value
    return None


def _effective_trade_date(account_id: str, requested_trade_date: date | None) -> date:
    if requested_trade_date is not None:
        return requested_trade_date
    try:
        engine = get_us_engine()
        with engine.connect() as conn:
            if relation_exists("recommend.us_stock_rank_daily"):
                value = conn.execute(LATEST_TRADE_DATE_SQL).scalar()
                if isinstance(value, date):
                    return value
            if relation_exists("trade.us_paper_order"):
                value = conn.execute(LATEST_TRADE_PAPER_ORDER_DATE_SQL, {"account_id": account_id}).scalar()
                if isinstance(value, date):
                    return value
            if relation_exists("paper.us_stock_paper_order"):
                value = conn.execute(LATEST_PAPER_ORDER_DATE_SQL, {"account_id": account_id}).scalar()
                if isinstance(value, date):
                    return value
    except Exception:
        pass
    return date.today()


def _load_latest_rank_map(symbols: list[str], trade_date: date, ranking_source: str = "rule_v1") -> dict[str, dict[str, object]]:
    if not symbols or not relation_exists("recommend.us_stock_rank_daily"):
        return {}
    try:
        engine = get_us_engine()
        with engine.connect() as conn:
            rows = conn.execute(LATEST_RANK_ROWS_SQL, {"symbols": symbols, "trade_date": trade_date, "source": ranking_source}).mappings().all()
        return {str(row.get("symbol") or "").upper(): dict(row) for row in rows}
    except Exception:
        return {}


def build_positions_from_orders(
    *,
    account_id: str,
    trade_date: date,
    buy_orders: list[dict[str, object]],
    sell_orders: list[dict[str, object]],
    fill_rows: list[dict[str, object]],
    existing_positions: list[dict[str, object]],
    latest_rank_map: dict[str, dict[str, object]],
    price_history_rows: list[dict[str, object]],
    benchmark_symbol: str,
) -> list[dict[str, object]]:
    fill_map: dict[str, float] = defaultdict(float)
    for row in fill_rows:
        order_id = str(row.get("paper_order_id") or "")
        qty = _safe_float(row.get("filled_qty"))
        if order_id and qty is not None:
            fill_map[order_id] += qty

    history_map = _history_map(price_history_rows)
    benchmark_history = history_map.get(benchmark_symbol.upper(), [])
    existing_map = {str(row.get("symbol") or "").upper(): row for row in existing_positions}
    grouped_buys: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_sells: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in buy_orders:
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            grouped_buys[symbol].append(row)
    for row in sell_orders:
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            grouped_sells[symbol].append(row)

    positions: list[dict[str, object]] = []
    for symbol, symbol_buys in grouped_buys.items():
        symbol_buys.sort(key=lambda item: item.get("trade_date") or date.min)
        total_buy_qty = 0.0
        total_buy_cost = 0.0
        data_quality_flags: list[str] = []
        entry_trade_date: date | None = None
        for order_row in symbol_buys:
            order_date = order_row.get("trade_date")
            qty = _paper_qty_for_order(order_row, fill_map)
            price = _paper_price_for_order(order_row)
            if entry_trade_date is None and isinstance(order_date, date):
                entry_trade_date = order_date
            if qty <= 0:
                continue
            total_buy_qty += qty
            if price is None:
                data_quality_flags.append("ENTRY_PRICE_MISSING")
            else:
                total_buy_cost += qty * price

        total_sell_qty = 0.0
        for order_row in grouped_sells.get(symbol, []):
            total_sell_qty += _paper_qty_for_order(order_row, fill_map)

        remaining_quantity = round(max(0.0, total_buy_qty - total_sell_qty), 6)
        if remaining_quantity <= 0:
            continue

        if entry_trade_date is None:
            data_quality_flags.append("ENTRY_TRADE_DATE_MISSING")
            entry_trade_date = trade_date

        avg_entry_price = round(total_buy_cost / total_buy_qty, 6) if total_buy_qty > 0 and total_buy_cost > 0 else None
        symbol_history = history_map.get(symbol, [])
        latest_price = _latest_price_on_or_before(symbol_history, trade_date)
        if latest_price is None:
            data_quality_flags.append("PRICE_DATA_MISSING")
        highest_price_since_entry = _highest_price_since_entry(symbol_history, entry_trade_date, trade_date)
        if highest_price_since_entry is None:
            data_quality_flags.append("HIGH_WATER_MARK_MISSING")

        unrealized_pnl = None
        unrealized_pnl_pct = None
        symbol_return_pct = None
        if latest_price is not None and avg_entry_price and avg_entry_price > 0:
            unrealized_pnl = round((latest_price - avg_entry_price) * remaining_quantity, 6)
            unrealized_pnl_pct = round((latest_price / avg_entry_price) - 1.0, 6)
            symbol_return_pct = unrealized_pnl_pct

        benchmark_entry_price = _price_on_or_before(benchmark_history, entry_trade_date)
        benchmark_latest_price = _latest_price_on_or_before(benchmark_history, trade_date)
        benchmark_return_pct = None
        if benchmark_entry_price and benchmark_entry_price > 0 and benchmark_latest_price is not None:
            benchmark_return_pct = round((benchmark_latest_price / benchmark_entry_price) - 1.0, 6)
        else:
            data_quality_flags.append("BENCHMARK_DATA_MISSING")

        rank_row = latest_rank_map.get(symbol, {})
        latest_probability = _extract_probability(rank_row) if rank_row else None
        if latest_probability is None:
            data_quality_flags.append("PROBABILITY_MISSING")

        existing_position = existing_map.get(symbol, {})
        paper_position_id = (
            existing_position.get("paper_position_id")
            or f"USPAPERPOS_{account_id}_{symbol}_{entry_trade_date.isoformat().replace('-', '')}"
        )

        status = "OPEN"
        if "PRICE_DATA_MISSING" in data_quality_flags:
            status = "PRICE_DATA_MISSING"
        elif "ENTRY_PRICE_MISSING" in data_quality_flags:
            status = "ENTRY_DATA_MISSING"

        positions.append(
            {
                "paper_position_id": paper_position_id,
                "account_id": account_id,
                "symbol": symbol,
                "entry_trade_date": entry_trade_date.isoformat(),
                "avg_entry_price": avg_entry_price,
                "quantity": round(total_buy_qty, 6),
                "remaining_quantity": remaining_quantity,
                "latest_price": latest_price,
                "highest_price_since_entry": highest_price_since_entry,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "holding_days": max(0, (trade_date - entry_trade_date).days),
                "latest_rank": rank_row.get("rank_no"),
                "latest_score": rank_row.get("total_score"),
                "latest_probability": latest_probability,
                "benchmark_return_pct": benchmark_return_pct,
                "symbol_return_pct": symbol_return_pct,
                "status": status,
                "data_quality_flags": sorted(set(data_quality_flags)),
                "price_source": "market.us_stock_daily_price",
                "ranking_source": rank_row.get("source"),
            }
        )

    positions.sort(key=lambda item: (str(item.get("symbol") or ""), str(item.get("entry_trade_date") or "")))
    return positions


def load_paper_positions(
    cfg: SellAutomationConfig,
    *,
    account_id: str,
    requested_trade_date: date | None = None,
) -> dict[str, object]:
    trade_date = _effective_trade_date(account_id, requested_trade_date)
    events: list[dict[str, object]] = []

    buy_orders: list[dict[str, object]] = []
    sell_orders: list[dict[str, object]] = []
    existing_positions: list[dict[str, object]] = []
    if relation_exists("trade.us_paper_order"):
        engine = get_us_engine()
        with engine.connect() as conn:
            buy_orders = [dict(row) for row in conn.execute(READ_TRADE_PAPER_BUY_ORDER_ROWS_SQL, {"account_id": account_id}).mappings().all()]
            if relation_exists("trade.us_paper_sell_order"):
                sell_orders = [dict(row) for row in conn.execute(READ_TRADE_PAPER_SELL_ORDER_ROWS_SQL).mappings().all()]
            if relation_exists("trade.us_paper_position_snapshot"):
                existing_positions = [dict(row) for row in conn.execute(READ_TRADE_POSITION_SNAPSHOT_ROWS_SQL).mappings().all()]
    else:
        buy_orders = [
            row
            for row in fetch_us_paper_order_rows(account_id=account_id, side="BUY")
            if str(row.get("status") or "").upper() in {"CREATED", "FILLED", "PENDING", "PARTIALLY_FILLED", ""}
        ]
        sell_orders = [
            row
            for row in fetch_us_paper_order_rows(account_id=account_id, side="SELL")
            if str(row.get("status") or "").upper() in {"CREATED", "FILLED", "PENDING", "PARTIALLY_FILLED", ""}
        ]
        existing_positions = fetch_us_paper_position_rows(account_id=account_id, status="OPEN")
    fill_rows = fetch_us_paper_fill_rows(account_id=account_id)

    symbols = sorted({str(row.get("symbol") or "").upper() for row in buy_orders if row.get("symbol")})
    if not symbols:
        events.append({"severity": "INFO", "reason_code": "NO_OPEN_PAPER_BUY_ORDER", "detail": "No paper BUY orders available for SELL evaluation."})
        return {
            "trade_date": trade_date,
            "benchmark_symbol": cfg.benchmark_symbol,
            "positions": [],
            "events": events,
        }

    ranking_source = str(os.environ.get("US_RANKING_DEFAULT_SOURCE", "rule_v1")).strip() or "rule_v1"
    latest_rank_map = _load_latest_rank_map(symbols, trade_date, ranking_source)
    price_history_rows = fetch_price_history_for_tickers(sorted(set(symbols) | {cfg.benchmark_symbol.upper()}), end_date=trade_date)
    positions = build_positions_from_orders(
        account_id=account_id,
        trade_date=trade_date,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        fill_rows=fill_rows,
        existing_positions=existing_positions,
        latest_rank_map=latest_rank_map,
        price_history_rows=price_history_rows,
        benchmark_symbol=cfg.benchmark_symbol,
    )
    if not positions:
        events.append({"severity": "WARNING", "reason_code": "NO_OPEN_POSITION", "detail": "Paper BUY orders exist, but no remaining open quantity was reconstructed."})
    return {
        "trade_date": trade_date,
        "benchmark_symbol": cfg.benchmark_symbol,
        "positions": positions,
        "events": events,
    }
