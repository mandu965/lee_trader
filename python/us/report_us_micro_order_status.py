from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_db import fetch_us_micro_order_fill_rows
from utils.us_micro_order_request import list_micro_orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report synchronized US micro order status and recent fills.")
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--execution-mode", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list_micro_orders(account_id=args.account_id, trade_date=args.trade_date, execution_mode=args.execution_mode)
    summary: dict[str, int] = {}
    open_partial: list[dict] = []
    fills: list[dict] = []
    for row in rows:
        status = str(row.get("request_status") or "").upper()
        summary[status] = summary.get(status, 0) + 1
        if status in {"ORDER_OPEN", "ORDER_PARTIALLY_FILLED"}:
            open_partial.append(row)
        fills.extend(fetch_us_micro_order_fill_rows(micro_order_id=str(row.get("micro_order_id"))))

    print("[US Micro Order Status Report]")
    print("")
    print(f"Account: {args.account_id or 'ALL'}")
    print(f"Trade Date: {args.trade_date or 'ALL'}")
    print("")
    print("Status Summary:")
    for key in ["ORDER_OPEN", "ORDER_PARTIALLY_FILLED", "ORDER_FILLED", "ORDER_CANCELED", "ORDER_REJECTED", "ORDER_EXPIRED", "ORDER_UNKNOWN", "SYNC_ERROR"]:
        print(f"{key}: {summary.get(key, 0)}")
    print("")
    print("Open / Partial Orders:")
    print("Micro Order ID | Symbol | Side | Qty | Filled | Remaining | Status")
    for row in open_partial:
        print(
            f"{row.get('micro_order_id')} | {row.get('symbol')} | {row.get('side')} | {row.get('order_qty')} | "
            f"{row.get('filled_qty') or 0} | {row.get('remaining_qty') or '-'} | {row.get('request_status')}"
        )
    if not open_partial:
        print("None")
    print("")
    print("Recent Fills:")
    print("Fill ID | Symbol | Side | Qty | Price | Amount | Fill Time")
    for row in sorted(fills, key=lambda item: str(item.get("fill_time") or ""), reverse=True)[:20]:
        print(
            f"{row.get('micro_fill_id')} | {row.get('symbol')} | {row.get('side')} | {row.get('filled_qty')} | "
            f"{row.get('filled_price')} | {row.get('filled_amount_usd')} | {row.get('fill_time')}"
        )
    if not fills:
        print("None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
