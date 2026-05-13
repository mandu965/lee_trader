from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.paper_rebalance import (
    PaperStrategyPolicy,
    build_order_id,
    load_policy,
    natural_order_key,
    order_price_lookup,
    order_reason_buy,
    order_reason_reject,
    order_reason_sell,
    position_value,
    round_qty,
    safe_float,
    sector_of,
)
from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import (
    ensure_us_paper_trading_tables,
    fetch_rank_component_rows_between,
    fetch_us_paper_account_rows,
    fetch_us_paper_order_rows,
    fetch_us_paper_position_rows,
    upsert_us_paper_order_rows,
)
from utils.paper_trading_safety import assert_paper_trading_only


LOGGER = logging.getLogger("us_paper_order_gen")


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate US stock paper-only orders from ranking snapshots.")
    parser.add_argument("--trade-date", required=True, help="Ranking trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--account-id", required=True, help="Paper account ID.")
    parser.add_argument("--side", choices=["BUY", "SELL", "ALL"], default="ALL")
    parser.add_argument("--replace-existing", action="store_true", help="Replace existing CREATED/REJECTED paper orders for the same natural key.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB writes.")
    return parser.parse_args()


def _build_rejected_order(
    *,
    account_id: str,
    trade_date: date,
    strategy_name: str,
    side: str,
    symbol: str,
    source_row: dict[str, object] | None,
    reject_reason: str,
    reason: str,
) -> dict[str, object]:
    return {
        "paper_order_id": build_order_id(account_id=account_id, trade_date=trade_date, strategy_name=strategy_name, side=side, symbol=symbol),
        "account_id": account_id,
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "order_type": "MARKET",
        "order_qty": 0.0,
        "order_price": None,
        "order_amount": 0.0,
        "limit_price": None,
        "source": "paper_rank_v1",
        "strategy_name": strategy_name,
        "rank_no": source_row.get("rank_no") if source_row else None,
        "recommend_grade": source_row.get("recommend_grade") if source_row else None,
        "total_score": source_row.get("total_score") if source_row else None,
        "status": "REJECTED",
        "reason": reason,
        "reject_reason": reject_reason,
    }


def _sell_detail_for_position(
    symbol: str,
    latest_rank: dict[str, object] | None,
    policy: PaperStrategyPolicy,
) -> str | None:
    if latest_rank is None:
        return "ranking row is missing"
    grade = str(latest_rank.get("recommend_grade") or "").upper()
    data_status = str(latest_rank.get("data_status") or "").upper()
    rank_no = int(latest_rank.get("rank_no") or 999999)
    if grade in set(policy.sell_grades) and policy.full_sell_on_grade_downgrade:
        return f"grade downgraded to {grade}"
    if rank_no > policy.max_rank_no and policy.full_sell_on_rank_exit:
        return f"exited Top{policy.max_rank_no}"
    if data_status in {"MISSING_PRICE_FEATURE", "ERROR"}:
        return f"data_status={data_status}"
    return None


def build_paper_orders(
    *,
    trade_date: date,
    account_row: dict[str, object],
    rank_rows: list[dict[str, object]],
    position_rows: list[dict[str, object]],
    existing_order_rows: list[dict[str, object]],
    policy: PaperStrategyPolicy,
    side_option: str,
    replace_existing: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    strategy_name = f"US_RANK_{policy.selection_rule}"
    rank_map = {str(row.get("symbol") or "").upper(): row for row in rank_rows}
    open_positions = [row for row in position_rows if str(row.get("status") or "OPEN").upper() == "OPEN"]
    position_map = {str(row.get("symbol") or "").upper(): row for row in open_positions}
    existing_map = {_natural_key: row for row in existing_order_rows for _natural_key in [natural_order_key(row)]}
    price_lookup = order_price_lookup(symbols=list(rank_map.keys()) + list(position_map.keys()), trade_date=trade_date)

    equity_value = safe_float(account_row.get("equity_value")) or safe_float(account_row.get("cash_balance")) or 0.0
    cash_balance = safe_float(account_row.get("cash_balance")) or 0.0
    reserved_cash = safe_float(account_row.get("reserved_cash")) or 0.0
    available_cash = cash_balance - reserved_cash
    min_cash_amount = equity_value * policy.min_cash_weight
    usable_cash = max(0.0, available_cash - min_cash_amount)
    target_position_value = equity_value * policy.max_position_weight

    sector_exposure: dict[str, float] = {}
    for row in open_positions:
        symbol = str(row.get("symbol") or "").upper()
        sector_exposure[sector_of(symbol, rank_map)] = sector_exposure.get(sector_of(symbol, rank_map), 0.0) + position_value(row)

    counts = {
        "rank_rows_read": len(rank_rows),
        "positions_read": len(position_rows),
        "existing_orders_read": len(existing_order_rows),
        "buy_candidates": 0,
        "sell_candidates": 0,
        "hold_candidates": 0,
        "target_candidates": 0,
        "orders_created": 0,
        "orders_rejected": 0,
        "orders_skipped": 0,
        "total_buy_amount": 0.0,
        "total_sell_amount": 0.0,
        "cash_balance": cash_balance,
        "reserved_cash": reserved_cash,
        "equity_value": equity_value,
    }
    orders: list[dict[str, object]] = []

    eligible_buy_rows = sorted(
        [
            row
            for row in rank_rows
            if str(row.get("recommend_grade") or "").upper() in set(policy.buy_grades)
            and int(row.get("rank_no") or 999999) <= policy.max_rank_no
            and row.get("total_score") is not None
            and str(row.get("data_status") or "").upper() in {"OK", "PARTIAL_DATA"}
            and not row.get("exclude_reason")
        ],
        key=lambda item: (
            int(item.get("rank_no") or 999999),
            -(safe_float(item.get("total_score")) or -999999.0),
            -(safe_float(item.get("momentum_score")) or -999999.0),
            -(safe_float(item.get("relative_strength_score")) or -999999.0),
            str(item.get("symbol") or ""),
        ),
    )
    counts["target_candidates"] = len(eligible_buy_rows)

    planned_sells: list[dict[str, object]] = []
    planned_sell_symbols: set[str] = set()
    projected_sell_cash = 0.0
    if side_option in {"SELL", "ALL"}:
        for row in open_positions:
            symbol = str(row.get("symbol") or "").upper()
            qty = safe_float(row.get("qty")) or 0.0
            if qty <= 0:
                continue
            detail = _sell_detail_for_position(symbol, rank_map.get(symbol), policy)
            if detail is None:
                counts["hold_candidates"] += 1
                continue
            counts["sell_candidates"] += 1
            planned_sell_symbols.add(symbol)
            price = price_lookup.get(symbol) or safe_float(row.get("last_price"))
            if price is not None and price > 0:
                projected_sell_cash += qty * price
            planned_sells.append({"symbol": symbol, "qty": qty, "detail": detail, "rank_row": rank_map.get(symbol), "price": price})

    if policy.sell_first and side_option in {"BUY", "ALL"}:
        usable_cash += projected_sell_cash

    planned_new_buys = 0
    if side_option in {"BUY", "ALL"}:
        for row in eligible_buy_rows:
            counts["buy_candidates"] += 1
            symbol = str(row.get("symbol") or "").upper()
            natural_key = (str(account_row.get("account_id") or ""), trade_date, symbol, "BUY", strategy_name)
            existing = existing_map.get(natural_key)
            if existing:
                existing_status = str(existing.get("status") or "").upper()
                if not replace_existing or existing_status in {"FILLED", "PARTIALLY_FILLED"}:
                    counts["orders_skipped"] += 1
                    continue
            if not policy.allow_rebuy_same_day and symbol in planned_sell_symbols:
                counts["orders_skipped"] += 1
                continue
            price = price_lookup.get(symbol)
            if price is None:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="missing_order_price", reason=order_reason_reject("missing order price")))
                counts["orders_rejected"] += 1
                continue
            if price <= 0:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="invalid_order_price", reason=order_reason_reject("invalid order price")))
                counts["orders_rejected"] += 1
                continue

            position = position_map.get(symbol)
            current_value = position_value(position) if position else 0.0
            current_weight = current_value / equity_value if equity_value > 0 else 0.0
            target_weight = policy.max_position_weight
            weight_diff = max(0.0, target_weight - current_weight)
            buy_amount = max(0.0, target_position_value - current_value)

            if weight_diff < policy.min_weight_diff:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="already_target_weight", reason=order_reason_reject("weight diff below min_rebalance_weight")))
                counts["orders_rejected"] += 1
                continue
            if buy_amount < policy.min_rebalance_amount:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="below_min_order_amount", reason=order_reason_reject("rebalance amount below minimum")))
                counts["orders_rejected"] += 1
                continue
            if len(open_positions) + planned_new_buys >= policy.max_positions and symbol not in position_map:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="max_positions_reached", reason=order_reason_reject("max positions reached")))
                counts["orders_rejected"] += 1
                continue
            if planned_new_buys >= policy.max_daily_new_buys and symbol not in position_map:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="max_positions_reached", reason=order_reason_reject("max daily new buys reached")))
                counts["orders_rejected"] += 1
                continue

            sector = sector_of(symbol, rank_map)
            if sector != "UNKNOWN":
                projected_sector = sector_exposure.get(sector, 0.0) + buy_amount
                if projected_sector > equity_value * policy.max_sector_weight:
                    orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="sector_weight_limit", reason=order_reason_reject("sector weight limit")))
                    counts["orders_rejected"] += 1
                    continue
            else:
                LOGGER.warning("[US_PAPER_ORDER] sector missing for %s. sector weight validation skipped.", symbol)

            buy_amount = min(buy_amount, usable_cash)
            if buy_amount < policy.min_rebalance_amount or buy_amount < policy.min_order_amount:
                reject_code = "insufficient_cash" if usable_cash < policy.min_order_amount else "below_min_order_amount"
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason=reject_code, reason=order_reason_reject("cash is below required rebalance amount")))
                counts["orders_rejected"] += 1
                continue

            qty = round_qty(buy_amount / price, allow_fractional=policy.allow_fractional_shares)
            if qty <= 0:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol, source_row=row, reject_reason="qty_zero", reason=order_reason_reject("calculated quantity is zero")))
                counts["orders_rejected"] += 1
                continue

            order_amount = round(qty * price, 6)
            usable_cash -= order_amount
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + order_amount
            if symbol not in position_map:
                planned_new_buys += 1
            counts["orders_created"] += 1
            counts["total_buy_amount"] += order_amount
            orders.append(
                {
                    "paper_order_id": build_order_id(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="BUY", symbol=symbol),
                    "account_id": str(account_row["account_id"]),
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "order_qty": qty,
                    "order_price": round(price, 6),
                    "order_amount": order_amount,
                    "limit_price": None,
                    "source": "paper_rank_v1",
                    "strategy_name": strategy_name,
                    "rank_no": row.get("rank_no"),
                    "recommend_grade": row.get("recommend_grade"),
                    "total_score": row.get("total_score"),
                    "status": "CREATED",
                    "reason": order_reason_buy(row, target_weight, current_weight, current_value),
                    "reject_reason": None,
                }
            )

    if side_option in {"SELL", "ALL"}:
        for plan in planned_sells:
            symbol = plan["symbol"]
            natural_key = (str(account_row.get("account_id") or ""), trade_date, symbol, "SELL", strategy_name)
            existing = existing_map.get(natural_key)
            if existing:
                existing_status = str(existing.get("status") or "").upper()
                if not replace_existing or existing_status in {"FILLED", "PARTIALLY_FILLED"}:
                    counts["orders_skipped"] += 1
                    continue
            price = plan["price"]
            if price is None:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="SELL", symbol=symbol, source_row=plan["rank_row"], reject_reason="missing_order_price", reason=order_reason_reject("missing order price")))
                counts["orders_rejected"] += 1
                continue
            if price <= 0:
                orders.append(_build_rejected_order(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="SELL", symbol=symbol, source_row=plan["rank_row"], reject_reason="invalid_order_price", reason=order_reason_reject("invalid order price")))
                counts["orders_rejected"] += 1
                continue
            qty = round(plan["qty"], 6)
            order_amount = round(qty * price, 6)
            counts["orders_created"] += 1
            counts["total_sell_amount"] += order_amount
            orders.append(
                {
                    "paper_order_id": build_order_id(account_id=str(account_row["account_id"]), trade_date=trade_date, strategy_name=strategy_name, side="SELL", symbol=symbol),
                    "account_id": str(account_row["account_id"]),
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "side": "SELL",
                    "order_type": "MARKET",
                    "order_qty": qty,
                    "order_price": round(price, 6),
                    "order_amount": order_amount,
                    "limit_price": None,
                    "source": "paper_rank_v1",
                    "strategy_name": strategy_name,
                    "rank_no": plan["rank_row"].get("rank_no") if plan["rank_row"] else None,
                    "recommend_grade": plan["rank_row"].get("recommend_grade") if plan["rank_row"] else None,
                    "total_score": plan["rank_row"].get("total_score") if plan["rank_row"] else None,
                    "status": "CREATED",
                    "reason": order_reason_sell(plan["rank_row"], plan["detail"]),
                    "reject_reason": None,
                }
            )

    return orders, counts


def _print_dry_run(*, trade_date: date, account_id: str, orders: list[dict[str, object]], counts: dict[str, object]) -> None:
    created = [row for row in orders if str(row.get("status") or "") == "CREATED"]
    rejected = [row for row in orders if str(row.get("status") or "") == "REJECTED"]
    print("[Paper Order Dry Run]")
    print(f"Trade Date: {trade_date.isoformat()}")
    print(f"Account: {account_id}")
    print("")
    print(f"BUY Candidates: {counts['buy_candidates']}")
    print(f"SELL Candidates: {counts['sell_candidates']}")
    print(f"Created Orders: {len(created)}")
    print(f"Rejected Orders: {len(rejected)}")
    print(f"Skipped Existing Orders: {counts['orders_skipped']}")
    print("")
    buy_preview = [row for row in created if str(row.get("side")) == "BUY"][:10]
    sell_preview = [row for row in created if str(row.get("side")) == "SELL"][:10]
    if buy_preview:
        print("[BUY Preview]")
        print("Symbol | Rank | Grade | Price | Qty | Amount | Reason")
        for row in buy_preview:
            print(f"{row['symbol']} | {row.get('rank_no')} | {row.get('recommend_grade')} | {row.get('order_price')} | {row.get('order_qty')} | {row.get('order_amount')} | {row.get('reason')}")
        print("")
    if sell_preview:
        print("[SELL Preview]")
        print("Symbol | Qty | Amount | Reason")
        for row in sell_preview:
            print(f"{row['symbol']} | {row.get('order_qty')} | {row.get('order_amount')} | {row.get('reason')}")
        print("")
    if rejected:
        print("[Rejected Preview]")
        print("Symbol | Side | RejectReason | Reason")
        for row in rejected[:10]:
            print(f"{row['symbol']} | {row.get('side')} | {row.get('reject_reason')} | {row.get('reason')}")


def _print_summary(*, trade_date: date, account_id: str, side_option: str, counts: dict[str, object]) -> None:
    print("[Paper Order Summary]")
    print(f"trade_date: {trade_date.isoformat()}")
    print(f"account_id: {account_id}")
    print(f"side option: {side_option}")
    print(f"rank rows read: {counts['rank_rows_read']}")
    print(f"positions read: {counts['positions_read']}")
    print(f"existing orders read: {counts['existing_orders_read']}")
    print(f"target candidates: {counts['target_candidates']}")
    print(f"hold candidates: {counts['hold_candidates']}")
    print(f"buy candidates: {counts['buy_candidates']}")
    print(f"sell candidates: {counts['sell_candidates']}")
    print(f"orders created: {counts['orders_created']}")
    print(f"orders rejected: {counts['orders_rejected']}")
    print(f"orders skipped: {counts['orders_skipped']}")
    print(f"total buy amount: {counts['total_buy_amount']:.2f}")
    print(f"total sell amount: {counts['total_sell_amount']:.2f}")
    print(f"cash balance: {counts['cash_balance']:.2f}")
    print(f"reserved cash: {counts['reserved_cash']:.2f}")
    print(f"equity value: {counts['equity_value']:.2f}")


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    setup_logging(cfg.log_level)
    assert_paper_trading_only(account_id=args.account_id, message="[SAFETY] Paper trading only. Real order APIs are blocked.")

    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    account_rows = fetch_us_paper_account_rows(account_id=args.account_id)
    if not account_rows:
        raise RuntimeError(f"Paper account not found: {args.account_id}")
    account_row = account_rows[0]
    if str(account_row.get("status") or "").upper() != "ACTIVE":
        raise RuntimeError(f"Paper account is not ACTIVE: {args.account_id}")

    policy = load_policy(args.account_id)
    rank_rows = fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source="rule_v1")
    position_rows = fetch_us_paper_position_rows(account_id=args.account_id, status="OPEN")
    existing_order_rows = fetch_us_paper_order_rows(account_id=args.account_id, trade_date=trade_date, strategy_name=f"US_RANK_{policy.selection_rule}")

    orders, counts = build_paper_orders(
        trade_date=trade_date,
        account_row=account_row,
        rank_rows=rank_rows,
        position_rows=position_rows,
        existing_order_rows=existing_order_rows,
        policy=policy,
        side_option=args.side,
        replace_existing=args.replace_existing,
    )

    if args.dry_run:
        _print_dry_run(trade_date=trade_date, account_id=args.account_id, orders=orders, counts=counts)
        _print_summary(trade_date=trade_date, account_id=args.account_id, side_option=args.side, counts=counts)
        return 0

    ensure_us_paper_trading_tables()
    upsert_us_paper_order_rows(orders)
    _print_summary(trade_date=trade_date, account_id=args.account_id, side_option=args.side, counts=counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
