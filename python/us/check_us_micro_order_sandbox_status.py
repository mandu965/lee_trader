from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_live_safety import SAFETY_MESSAGE, assert_us_micro_mock_only
from utils.us_micro_order_request import check_micro_order_sandbox_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check sandbox order status for a US micro order.")
    parser.add_argument("--micro-order-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--created-by", default="SYSTEM")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_us_micro_mock_only(SAFETY_MESSAGE)
    result = check_micro_order_sandbox_status(
        args.micro_order_id,
        created_by=args.created_by,
        dry_run=args.dry_run,
    )
    row = result["micro_order"] if args.dry_run and "micro_order" in result else result
    print("[US Micro Order Sandbox Status]")
    print(f"Micro Order ID: {row.get('micro_order_id')}")
    print(f"Status: {row.get('request_status')}")
    print(f"Broker Order ID: {row.get('broker_order_id') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
