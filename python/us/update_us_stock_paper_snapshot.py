from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import logging
import math
from pathlib import Path
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.simulate_us_stock_paper_fills import assert_paper_trading_only, validate_paper_account_integrity
from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import (
    ensure_us_paper_trading_tables,
    fetch_mixed_price_rows_for_tickers_between,
    fetch_us_paper_account_rows,
    fetch_us_paper_account_snapshot_rows,
    fetch_us_paper_fill_rows,
    fetch_us_paper_position_rows,
    get_us_engine,
)


LOGGER = logging.getLogger("us_paper_snapshot")
EPSILON = 1e-9

UPSERT_PAPER_POSITION_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_position (
        account_id, symbol, qty, avg_price, cost_amount, last_price, market_value,
        unrealized_pnl, unrealized_pnl_pct, realized_pnl, last_trade_date, last_price_date,
        status, created_at, updated_at
    ) VALUES (
        :account_id, :symbol, :qty, :avg_price, :cost_amount, :last_price, :market_value,
        :unrealized_pnl, :unrealized_pnl_pct, :realized_pnl, :last_trade_date, :last_price_date,
        :status, now(), now()
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
    SET account_name = :account_name,
        base_currency = :base_currency,
        initial_cash = :initial_cash,
        cash_balance = :cash_balance,
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

UPSERT_PAPER_SNAPSHOT_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_account_snapshot (
        account_id, snapshot_date, cash_balance, reserved_cash, market_value, equity_value,
        realized_pnl, unrealized_pnl, total_pnl, total_pnl_pct, daily_return_pct,
        spy_return_pct, qqq_return_pct, excess_return_vs_spy, excess_return_vs_qqq,
        position_count, created_at
    ) VALUES (
        :account_id, :snapshot_date, :cash_balance, :reserved_cash, :market_value, :equity_value,
        :realized_pnl, :unrealized_pnl, :total_pnl, :total_pnl_pct, :daily_return_pct,
        :spy_return_pct, :qqq_return_pct, :excess_return_vs_spy, :excess_return_vs_qqq,
        :position_count, now()
    )
    ON CONFLICT (account_id, snapshot_date) DO UPDATE SET
        cash_balance = EXCLUDED.cash_balance,
        reserved_cash = EXCLUDED.reserved_cash,
        market_value = EXCLUDED.market_value,
        equity_value = EXCLUDED.equity_value,
        realized_pnl = EXCLUDED.realized_pnl,
        unrealized_pnl = EXCLUDED.unrealized_pnl,
        total_pnl = EXCLUDED.total_pnl,
        total_pnl_pct = EXCLUDED.total_pnl_pct,
        daily_return_pct = EXCLUDED.daily_return_pct,
        spy_return_pct = EXCLUDED.spy_return_pct,
        qqq_return_pct = EXCLUDED.qqq_return_pct,
        excess_return_vs_spy = EXCLUDED.excess_return_vs_spy,
        excess_return_vs_qqq = EXCLUDED.excess_return_vs_qqq,
        position_count = EXCLUDED.position_count
    """
)


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update US paper account daily snapshot and valuation.")
    parser.add_argument("--snapshot-date", required=True, help="Snapshot date. Format: YYYY-MM-DD.")
    parser.add_argument("--account-id", required=True, help="Paper account ID.")
    parser.add_argument("--dry-run", action="store_true", help="Preview valuation without DB writes.")
    parser.add_argument("--use-previous-close", action="store_true", help="Use the latest previous close if snapshot-date close is missing.")
    parser.add_argument("--skip-position-update", action="store_true", help="Do not persist per-position mark-to-market fields.")
    return parser.parse_args()


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


def _fmt_num(value: object) -> str:
    num = _safe_float(value)
    if num is None:
        return "N/A"
    return f"{num:,.2f}"


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


def _price_for_snapshot(
    symbol: str,
    snapshot_date: date,
    price_lookup: dict[str, list[dict[str, object]]],
    *,
    use_previous_close: bool,
) -> tuple[float | None, date | None]:
    prior_price: float | None = None
    prior_date: date | None = None
    for row in price_lookup.get(symbol.upper(), []):
        row_date = row.get("trade_date")
        price = _safe_float(row.get("price"))
        if not isinstance(row_date, date) or price is None or row_date > snapshot_date:
            continue
        if row_date == snapshot_date:
            return price, row_date
        prior_price = price
        prior_date = row_date
    if use_previous_close:
        return prior_price, prior_date
    return None, None


def _previous_snapshot(snapshot_rows: list[dict[str, object]], snapshot_date: date) -> dict[str, object] | None:
    for row in snapshot_rows:
        row_date = row.get("snapshot_date")
        if isinstance(row_date, date) and row_date < snapshot_date:
            return row
    return None


def _benchmark_start_date(snapshot_rows: list[dict[str, object]], fill_rows: list[dict[str, object]], snapshot_date: date) -> date:
    prior_dates = [row["snapshot_date"] for row in snapshot_rows if isinstance(row.get("snapshot_date"), date)]
    if prior_dates:
        return min(prior_dates)
    fill_dates = [row["trade_date"] for row in fill_rows if isinstance(row.get("trade_date"), date)]
    if fill_dates:
        return min(fill_dates)
    return snapshot_date


def evaluate_paper_snapshot(
    *,
    snapshot_date: date,
    account_row: dict[str, object],
    position_rows: list[dict[str, object]],
    snapshot_rows: list[dict[str, object]],
    fill_rows: list[dict[str, object]],
    price_rows: list[dict[str, object]],
    use_previous_close: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
    warnings: list[str] = []
    price_lookup = _build_price_lookup(price_rows)
    account = deepcopy(account_row)
    open_positions = []
    market_value_total = 0.0
    unrealized_total = 0.0

    for row in position_rows:
        position = deepcopy(row)
        symbol = str(position.get("symbol") or "").upper()
        qty = _safe_float(position.get("qty")) or 0.0
        status = str(position.get("status") or "").upper()
        if status != "OPEN":
            open_positions.append(position)
            continue
        if qty <= EPSILON:
            warnings.append(f"{symbol} position is OPEN but qty <= 0")
        last_price, last_price_date = _price_for_snapshot(symbol, snapshot_date, price_lookup, use_previous_close=use_previous_close)
        if last_price is None:
            warnings.append(f"{symbol} price is missing for {snapshot_date.isoformat()}")
            open_positions.append(position)
            continue
        cost_amount = _safe_float(position.get("cost_amount")) or 0.0
        market_value = qty * last_price
        unrealized_pnl = market_value - cost_amount
        unrealized_pnl_pct = (unrealized_pnl / cost_amount) if cost_amount > EPSILON else None
        position["last_price"] = round(last_price, 6)
        position["last_price_date"] = last_price_date
        position["market_value"] = round(market_value, 6)
        position["unrealized_pnl"] = round(unrealized_pnl, 6)
        position["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 6) if unrealized_pnl_pct is not None else None
        market_value_total += market_value
        unrealized_total += unrealized_pnl
        open_positions.append(position)

    cash_balance = _safe_float(account.get("cash_balance")) or 0.0
    reserved_cash = _safe_float(account.get("reserved_cash")) or 0.0
    realized_pnl = _safe_float(account.get("realized_pnl")) or 0.0
    initial_cash = _safe_float(account.get("initial_cash")) or 0.0
    equity_value = cash_balance + market_value_total
    total_pnl = realized_pnl + unrealized_total
    total_pnl_pct = (total_pnl / initial_cash) if initial_cash > EPSILON else None
    if initial_cash <= EPSILON:
        warnings.append("initial_cash is missing or zero; total_pnl_pct is null")

    prev_snapshot = _previous_snapshot(snapshot_rows, snapshot_date)
    prev_equity = _safe_float(prev_snapshot.get("equity_value")) if prev_snapshot else None
    daily_return_pct = ((equity_value - prev_equity) / prev_equity) if prev_equity and prev_equity > EPSILON else None

    benchmark_start = _benchmark_start_date(snapshot_rows, fill_rows, snapshot_date)
    spy_start, _ = _price_for_snapshot("SPY", benchmark_start, price_lookup, use_previous_close=True)
    spy_now, _ = _price_for_snapshot("SPY", snapshot_date, price_lookup, use_previous_close=use_previous_close)
    qqq_start, _ = _price_for_snapshot("QQQ", benchmark_start, price_lookup, use_previous_close=True)
    qqq_now, _ = _price_for_snapshot("QQQ", snapshot_date, price_lookup, use_previous_close=use_previous_close)
    spy_return_pct = ((spy_now - spy_start) / spy_start) if spy_start and spy_now and spy_start > EPSILON else None
    qqq_return_pct = ((qqq_now - qqq_start) / qqq_start) if qqq_start and qqq_now and qqq_start > EPSILON else None
    if spy_return_pct is None:
        warnings.append("SPY benchmark return is unavailable")
    if qqq_return_pct is None:
        warnings.append("QQQ benchmark return is unavailable")

    snapshot_row = {
        "account_id": account["account_id"],
        "snapshot_date": snapshot_date,
        "cash_balance": round(cash_balance, 6),
        "reserved_cash": round(reserved_cash, 6),
        "market_value": round(market_value_total, 6),
        "equity_value": round(equity_value, 6),
        "realized_pnl": round(realized_pnl, 6),
        "unrealized_pnl": round(unrealized_total, 6),
        "total_pnl": round(total_pnl, 6),
        "total_pnl_pct": round(total_pnl_pct, 6) if total_pnl_pct is not None else None,
        "daily_return_pct": round(daily_return_pct, 6) if daily_return_pct is not None else None,
        "spy_return_pct": round(spy_return_pct, 6) if spy_return_pct is not None else None,
        "qqq_return_pct": round(qqq_return_pct, 6) if qqq_return_pct is not None else None,
        "excess_return_vs_spy": round(total_pnl_pct - spy_return_pct, 6) if total_pnl_pct is not None and spy_return_pct is not None else None,
        "excess_return_vs_qqq": round(total_pnl_pct - qqq_return_pct, 6) if total_pnl_pct is not None and qqq_return_pct is not None else None,
        "position_count": sum(1 for row in open_positions if str(row.get("status") or "").upper() == "OPEN" and (_safe_float(row.get("qty")) or 0.0) > EPSILON),
    }

    account["market_value"] = snapshot_row["market_value"]
    account["equity_value"] = snapshot_row["equity_value"]
    account["unrealized_pnl"] = snapshot_row["unrealized_pnl"]
    account["total_pnl"] = snapshot_row["total_pnl"]
    return snapshot_row, open_positions, warnings


def _persist_snapshot(
    *,
    account_row: dict[str, object],
    position_rows: list[dict[str, object]],
    snapshot_row: dict[str, object],
    skip_position_update: bool,
) -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        if not skip_position_update:
            for row in position_rows:
                if str(row.get("status") or "").upper() != "OPEN":
                    continue
                conn.execute(UPSERT_PAPER_POSITION_SQL, row)
        conn.execute(UPDATE_PAPER_ACCOUNT_SQL, account_row)
        conn.execute(UPSERT_PAPER_SNAPSHOT_SQL, snapshot_row)


def _print_snapshot_summary(
    *,
    snapshot_date: date,
    account_row: dict[str, object],
    snapshot_row: dict[str, object],
    warnings: list[str],
    integrity_issues: list[str],
    dry_run: bool,
) -> None:
    title = "[Paper Snapshot Dry Run]" if dry_run else "[Paper Snapshot Summary]"
    print(title)
    print(f"snapshot_date: {snapshot_date.isoformat()}")
    print(f"account_id: {account_row['account_id']}")
    print(f"cash_balance: {_fmt_num(snapshot_row['cash_balance'])}")
    print(f"market_value: {_fmt_num(snapshot_row['market_value'])}")
    print(f"equity_value: {_fmt_num(snapshot_row['equity_value'])}")
    print(f"realized_pnl: {_fmt_num(snapshot_row['realized_pnl'])}")
    print(f"unrealized_pnl: {_fmt_num(snapshot_row['unrealized_pnl'])}")
    print(f"total_pnl: {_fmt_num(snapshot_row['total_pnl'])}")
    print(f"total_pnl_pct: {_fmt_pct(snapshot_row['total_pnl_pct'])}")
    print(f"daily_return_pct: {_fmt_pct(snapshot_row['daily_return_pct'])}")
    print(f"spy_return_pct: {_fmt_pct(snapshot_row['spy_return_pct'])}")
    print(f"qqq_return_pct: {_fmt_pct(snapshot_row['qqq_return_pct'])}")
    print(f"excess_return_vs_spy: {_fmt_pct(snapshot_row['excess_return_vs_spy'])}")
    print(f"excess_return_vs_qqq: {_fmt_pct(snapshot_row['excess_return_vs_qqq'])}")
    print(f"position_count: {snapshot_row['position_count']}")
    print(f"warnings: {len(warnings)}")
    print(f"integrity_issues: {len(integrity_issues)}")
    if warnings:
        print("")
        print("[Warnings]")
        for item in warnings:
            print(f"- {item}")
    if integrity_issues:
        print("")
        print("[Integrity Issues]")
        for item in integrity_issues:
            print(f"- {item}")


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    setup_logging(cfg.log_level)
    assert_paper_trading_only()
    LOGGER.info("[SAFETY] Paper trading evaluation only. Real order APIs are blocked.")

    snapshot_date = parse_iso_date(args.snapshot_date, field_name="snapshot_date")
    if snapshot_date is None:
        raise ValueError("snapshot_date is required.")

    ensure_us_paper_trading_tables()
    account_rows = fetch_us_paper_account_rows(account_id=args.account_id)
    if not account_rows:
        print(f"paper account not found: {args.account_id}")
        return 1
    account_row = deepcopy(account_rows[0])

    position_rows = fetch_us_paper_position_rows(account_id=args.account_id)
    snapshot_rows = fetch_us_paper_account_snapshot_rows(account_id=args.account_id)
    fill_rows = fetch_us_paper_fill_rows(account_id=args.account_id)
    symbols = sorted(
        {
            str(row.get("symbol") or "").upper()
            for row in position_rows
            if str(row.get("status") or "").upper() == "OPEN" and str(row.get("symbol") or "").strip()
        }
        | {"SPY", "QQQ"}
    )
    price_rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=symbols,
        start_date=snapshot_date - timedelta(days=30),
        end_date=snapshot_date,
    )

    snapshot_row, evaluated_positions, warnings = evaluate_paper_snapshot(
        snapshot_date=snapshot_date,
        account_row=account_row,
        position_rows=position_rows,
        snapshot_rows=snapshot_rows,
        fill_rows=fill_rows,
        price_rows=price_rows,
        use_previous_close=args.use_previous_close or cfg.use_previous_close_if_missing,
    )
    account_row["market_value"] = snapshot_row["market_value"]
    account_row["equity_value"] = snapshot_row["equity_value"]
    account_row["unrealized_pnl"] = snapshot_row["unrealized_pnl"]
    account_row["total_pnl"] = snapshot_row["total_pnl"]

    integrity_issues: list[str] = []
    if not args.dry_run:
        _persist_snapshot(
            account_row=account_row,
            position_rows=evaluated_positions,
            snapshot_row=snapshot_row,
            skip_position_update=args.skip_position_update,
        )
        integrity_issues = validate_paper_account_integrity(args.account_id)

    _print_snapshot_summary(
        snapshot_date=snapshot_date,
        account_row=account_row,
        snapshot_row=snapshot_row,
        warnings=warnings,
        integrity_issues=integrity_issues,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
