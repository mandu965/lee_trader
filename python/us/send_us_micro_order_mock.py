from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_live_safety import SAFETY_MESSAGE, assert_us_micro_mock_only
from utils.us_micro_order_request import list_micro_orders, send_micro_order_via_mock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send US micro orders through the mock order client only.")
    parser.add_argument("--micro-order-id", default=None)
    parser.add_argument("--status", default="READY_TO_SEND")
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--created-by", default="SYSTEM")
    return parser.parse_args()


def _render(row: dict[str, object]) -> str:
    lines = [
        "[US Micro Order Mock Send]",
        f"Micro Order ID: {row.get('micro_order_id')}",
        f"Status: {row.get('request_status')}",
        f"Symbol: {row.get('symbol')}",
        f"Side: {row.get('side')}",
        f"Broker Order ID: {row.get('broker_order_id') or '-'}",
        f"Reject Reason Code: {row.get('reject_reason_code') or '-'}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    assert_us_micro_mock_only(SAFETY_MESSAGE)
    if args.micro_order_id:
        rows = [{"micro_order_id": args.micro_order_id}]
    else:
        rows = list_micro_orders(
            trade_date=args.trade_date,
            account_id=args.account_id,
            status=args.status,
            execution_mode="MOCK",
        )
    if not rows:
        print("[US Micro Order Mock Send]")
        print("No micro orders matched the request.")
        return 0
    for row in rows:
        result = send_micro_order_via_mock(
            str(row.get("micro_order_id")),
            created_by=args.created_by,
            dry_run=args.dry_run,
        )
        if args.dry_run and "micro_order" in result:
            result_row = result["micro_order"]
        else:
            result_row = result
        print(_render(result_row))
        if len(rows) > 1:
            print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
