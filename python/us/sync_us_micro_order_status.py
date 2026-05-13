from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_order_sync import sync_micro_order_fills, sync_micro_order_status, sync_micro_orders_by_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync US micro order status and optional fill rows.")
    parser.add_argument("--micro-order-id", default=None)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--status", default="ACCEPTED,LIVE_ACCEPTED,ORDER_OPEN,ORDER_PARTIALLY_FILLED")
    parser.add_argument("--include-fills", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _render_single(result: dict[str, object]) -> str:
    row = result.get("micro_order") if isinstance(result.get("micro_order"), dict) else result
    return "\n".join(
        [
            "[US Micro Order Sync]",
            f"Micro Order ID: {row.get('micro_order_id')}",
            f"Execution Mode: {row.get('execution_mode')}",
            f"Status: {row.get('request_status')}",
            f"Broker Order ID: {row.get('broker_order_id') or '-'}",
            f"Last Broker Status: {row.get('last_broker_status') or '-'}",
            f"Sync Status: {row.get('sync_status') or '-'}",
        ]
    )


def main() -> int:
    args = parse_args()
    try:
        if args.micro_order_id:
            result = sync_micro_order_status(args.micro_order_id, dry_run=args.dry_run)
            print(_render_single(result["micro_order"] if "micro_order" in result and isinstance(result["micro_order"], dict) else result))
            if args.include_fills:
                fill_result = sync_micro_order_fills(args.micro_order_id, dry_run=args.dry_run)
                count = len(fill_result.get("fills") or fill_result.get("insert_result", {}).get("rows", []))
                print(f"Fills Synced: {count}")
            return 0

        statuses = [item.strip().upper() for item in str(args.status or "").split(",") if item.strip()]
        results = sync_micro_orders_by_status(
            account_id=args.account_id,
            trade_date=args.trade_date,
            statuses=statuses,
            dry_run=args.dry_run,
            include_fills=args.include_fills,
        )
        print("[US Micro Order Sync Batch]")
        print(f"Result Count: {len(results)}")
        return 0
    except Exception as exc:
        print("[US Micro Order Sync]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
