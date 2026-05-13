from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_order_safety import LIVE_SAFETY_MESSAGE
from utils.us_micro_live_safety import SAFETY_MESSAGE, assert_us_micro_mock_only
from utils.us_micro_order_request import create_micro_order_from_approval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a US micro order request from an approved approval row.")
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--execution-mode", choices=["MOCK", "SANDBOX", "LIVE"], default="MOCK")
    parser.add_argument("--created-by", default="SYSTEM")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--allow-live-create", action="store_true")
    return parser.parse_args()


def _render(result: dict[str, object]) -> str:
    row = result.get("micro_order") if isinstance(result.get("micro_order"), dict) else result
    lines = [
        "[US Micro Order Request]",
        "Phase 7-3 keeps Micro Live as a tightly gated one-off flow.",
        "Micro Live validates real-order connectivity, not automated profit generation.",
        "LIMIT orders only. Manual approval and final confirmation remain required for LIVE.",
        "",
        f"Approval ID: {row.get('approval_id')}",
        f"Micro Order ID: {row.get('micro_order_id')}",
        f"Execution Mode: {row.get('execution_mode')}",
        f"Status: {row.get('request_status')}",
        f"Symbol: {row.get('symbol')}",
        f"Side: {row.get('side')}",
        f"Amount USD: {row.get('order_amount_usd')}",
        f"Reject Reason Code: {row.get('reject_reason_code') or '-'}",
    ]
    if result.get("dry_run"):
        lines.append("Dry Run: true")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.execution_mode == "LIVE":
        print(LIVE_SAFETY_MESSAGE)
    else:
        assert_us_micro_mock_only(SAFETY_MESSAGE)
    result = create_micro_order_from_approval(
        args.approval_id,
        execution_mode=args.execution_mode,
        created_by=args.created_by,
        dry_run=args.dry_run,
        replace_existing=args.replace_existing,
        allow_live_create=args.allow_live_create,
    )
    print(_render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
