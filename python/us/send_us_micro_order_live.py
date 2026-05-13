from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_order_safety import LIVE_FINAL_CONFIRMATION_TEXT, LIVE_SAFETY_MESSAGE
from utils.us_micro_order_request import send_micro_order_via_live


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attempt a tightly gated Micro Live order send.")
    parser.add_argument("--micro-order-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-live-order", default=None)
    parser.add_argument("--created-by", default="SYSTEM")
    return parser.parse_args()


def _render(result: dict[str, object]) -> str:
    row = result.get("micro_order") if isinstance(result.get("micro_order"), dict) else result
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    lines = [
        "[US Micro Live Send]",
        f"Micro Order ID: {row.get('micro_order_id')}",
        f"Status: {row.get('request_status')}",
        f"Execution Mode: {row.get('execution_mode')}",
        f"Symbol: {row.get('symbol')}",
        f"Side: {row.get('side')}",
        f"Broker Order ID: {row.get('broker_order_id') or '-'}",
        f"Reject Reason Code: {row.get('reject_reason_code') or validation.get('reason_code') or '-'}",
    ]
    if validation:
        lines.append(f"Validation Status: {validation.get('status')}")
        lines.append(f"Validation Detail: {validation.get('reason_detail')}")
    if result.get("dry_run"):
        lines.append("Dry Run: true")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    print(LIVE_SAFETY_MESSAGE)
    if not args.dry_run and args.confirm_live_order != LIVE_FINAL_CONFIRMATION_TEXT:
        print("[US Micro Live Send]")
        print("Live send blocked: final confirmation text is missing or invalid.")
        return 0
    try:
        result = send_micro_order_via_live(
            args.micro_order_id,
            created_by=args.created_by,
            dry_run=args.dry_run,
            confirm_live_order=args.confirm_live_order,
        )
    except ValueError as exc:
        print("[US Micro Live Send]")
        print(str(exc))
        return 1
    print(_render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
