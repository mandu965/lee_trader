from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_order_approval import (
    approve_order_approval,
    expire_order_approvals,
    get_order_approval,
    get_order_approval_events,
    list_order_approvals,
    reject_order_approval,
)
from utils.us_live_trading_safety import assert_us_live_pre_trade_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage US live order-approval requests without creating any real order.")
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--approval-id", default=None)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reject", action="store_true")
    parser.add_argument("--expire-pending", action="store_true")
    parser.add_argument("--status", choices=["PENDING", "APPROVED", "REJECTED", "EXPIRED", "CANCELED", "ERROR"], default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--approved-by", default=None)
    parser.add_argument("--rejected-by", default=None)
    parser.add_argument("--reason", default=None)
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    return parser.parse_args()


def _render_summary(rows: list[dict[str, object]]) -> str:
    lines = ["[US Live Order Approval Status]", ""]
    lines.append(f"Pending Approvals: {sum(1 for row in rows if str(row.get('approval_status') or '').upper() == 'PENDING')}")
    lines.append("")
    if rows:
        lines.append("Approval ID | Trade Date | Symbol | Side | Amount | Grade | Rank | Expires At | Status")
        for row in rows:
            lines.append(
                f"{row.get('approval_id')} | {row.get('trade_date')} | {row.get('symbol')} | {row.get('side')} | "
                f"{row.get('requested_order_amount_usd')} | {row.get('recommend_grade') or ''} | {row.get('rank_no') or ''} | "
                f"{row.get('expires_at') or ''} | {row.get('approval_status')}"
            )
    else:
        lines.append("No approval rows found.")
    lines.extend(["", "[Safety]", "Approval does not create real orders.", "Real order APIs were not called."])
    return "\n".join(lines)


def _render_detail(row: dict[str, object]) -> str:
    events = get_order_approval_events(str(row.get("approval_id")))
    lines = [
        "[Approval Detail]",
        f"Approval ID: {row.get('approval_id')}",
        f"Status: {row.get('approval_status')}",
        f"Trade Date: {row.get('trade_date')}",
        f"Account: {row.get('account_id')}",
        f"Symbol: {row.get('symbol')}",
        f"Side: {row.get('side')}",
        f"Amount: {row.get('requested_order_amount_usd')} USD",
        f"Order Type: {row.get('requested_order_type')}",
        f"Limit Price: {row.get('requested_limit_price')}",
        "",
        f"Pre-Check Decision: {row.get('precheck_decision')}",
        "Reason Codes:",
        f"{row.get('precheck_reason_codes')}",
        "",
        "Pre-Check Summary:",
        f"{row.get('precheck_summary') or ''}",
        "",
        f"Requested By: {row.get('requested_by')}",
        f"Requested At: {row.get('requested_at')}",
        f"Expires At: {row.get('expires_at')}",
        "",
        "Events:",
    ]
    for event in events:
        lines.append(
            f"- {event.get('event_type')} {event.get('before_status')} -> {event.get('after_status')} "
            f"by {event.get('performed_by') or ''} at {event.get('created_at')}"
        )
    lines.extend(["", "Safety:", "- No real order was created.", "- Approval and order execution remain separated."])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    assert_us_live_pre_trade_only(
        policy_id=args.policy_id,
        message="[SAFETY] Approval management only. No real orders are created.",
    )
    if args.expire_pending:
        result = expire_order_approvals()
        print("[US Live Order Approval]")
        print(f"Expired Count: {result['expired_count']}")
        print("No real order was created.")
        return 0
    if args.approve:
        if not args.approval_id:
            raise ValueError("--approval-id is required for --approve.")
        row = approve_order_approval(args.approval_id, approved_by=args.approved_by or "", approval_reason=args.reason or "")
        print(_render_detail(row))
        return 0
    if args.reject:
        if not args.approval_id:
            raise ValueError("--approval-id is required for --reject.")
        row = reject_order_approval(args.approval_id, rejected_by=args.rejected_by or "", reject_reason=args.reason or "")
        print(_render_detail(row))
        return 0
    if args.approval_id:
        print(_render_detail(get_order_approval(args.approval_id)))
        return 0
    rows = list_order_approvals(status=args.status, trade_date=args.trade_date, account_id=args.account_id)
    output = _render_summary(rows)
    if args.format == "markdown":
        output_dir = Path(__file__).resolve().parents[2] / "outputs" / "us_stock_live_risk"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "us_live_order_approval_status.md"
        path.write_text(output.replace("[US Live Order Approval Status]", "# US Live Order Approval Status"), encoding="utf-8")
        print(path)
        return 0
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
