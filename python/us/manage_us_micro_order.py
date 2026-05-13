from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_order_request import (
    build_micro_order_report,
    cancel_micro_order,
    get_micro_order,
    get_micro_order_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List, inspect, and cancel US micro order requests.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--micro-order-id", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    parser.add_argument("--execution-mode", default=None)
    parser.add_argument("--created-by", default="OPERATOR")
    return parser.parse_args()


def _render_list(report: dict[str, object], *, trade_date: str | None, execution_mode: str | None) -> str:
    rows = report["rows"]
    summary = report["summary"]
    execution_summary = report["execution_summary"]
    sandbox_config = report["sandbox_config"]
    sandbox_events = report["sandbox_events"]
    live_config = report["live_config"]
    live_events = report["live_events"]
    live_rows = report["live_rows"]

    lines = [
        "[US Micro Order Report]",
        f"Trade Date: {trade_date or 'ALL'}",
        f"Execution Mode: {execution_mode or 'ALL'}",
        "",
        "Status Summary:",
    ]
    ordered_statuses = [
        "CREATED",
        "READY_TO_SEND",
        "SENT",
        "ACCEPTED",
        "REJECTED",
        "FAILED",
        "CANCELED",
        "PRECHECK_FAILED",
        "APPROVAL_INVALID",
        "LIVE_BLOCKED",
        "LIVE_READY",
        "LIVE_CONFIRMATION_REQUIRED",
        "LIVE_SENT",
        "LIVE_ACCEPTED",
        "LIVE_REJECTED",
        "LIVE_FAILED",
        "LIVE_CANCELED",
        "ORDER_OPEN",
        "ORDER_PARTIALLY_FILLED",
        "ORDER_FILLED",
        "ORDER_CANCELED",
        "ORDER_REJECTED",
        "ORDER_EXPIRED",
        "ORDER_UNKNOWN",
        "SYNC_ERROR",
        "ERROR",
    ]
    for key in ordered_statuses:
        lines.append(f"{key}: {summary.get(key, 0)}")

    lines.extend(
        [
            "",
            "Recent Orders:",
            "Micro Order ID | Mode | Symbol | Side | Amount | Status | BrokerOrderId",
        ]
    )
    for row in rows:
        lines.append(
            f"{row.get('micro_order_id')} | {row.get('execution_mode')} | {row.get('symbol')} | {row.get('side')} | "
            f"{row.get('order_amount_usd')} | {row.get('request_status')} | {row.get('broker_order_id') or '-'}"
        )
    if not rows:
        lines.append("No micro order rows found.")

    lines.extend(
        [
            "",
            "[US Sandbox Micro Order Report]",
            "",
            f"Execution Mode: {execution_mode or 'ALL'}",
            f"Sandbox Enabled: {sandbox_config.get('allow_sandbox')}",
            f"Sandbox Broker: {sandbox_config.get('sandbox_broker_name')}",
            f"Live Allowed: {sandbox_config.get('allow_live')}",
            f"Real Order Blocked: {sandbox_config.get('real_order_blocked')}",
            "",
            "Recent Sandbox Events:",
            "Event Type | Micro Order ID | Status | Reason",
        ]
    )
    for event in sandbox_events:
        lines.append(
            f"{event.get('event_type')} | {event.get('micro_order_id')} | {event.get('after_status') or '-'} | "
            f"{event.get('reason_code') or event.get('reason_detail') or '-'}"
        )
    if not sandbox_events:
        lines.append("No sandbox events found.")

    lines.extend(
        [
            "",
            "[US Micro Live Order Report]",
            "",
            "Execution Mode Summary:",
            f"MOCK: {execution_summary.get('MOCK', 0)}",
            f"SANDBOX: {execution_summary.get('SANDBOX', 0)}",
            f"LIVE: {execution_summary.get('LIVE', 0)}",
            "",
            "Live Safety:",
            f"US_MICRO_ALLOW_LIVE: {live_config.get('micro_allow_live')}",
            f"US_MICRO_REAL_ORDER_BLOCKED: {live_config.get('micro_real_order_blocked')}",
            f"US_LIVE_ORDER_ENABLED: {live_config.get('live_order_enabled')}",
            f"Final Confirmation Required: {live_config.get('require_final_confirmation')}",
            "",
            "Live Orders:",
            "Micro Order ID | Symbol | Side | Amount | Status | Reason",
        ]
    )
    for row in live_rows:
        lines.append(
            f"{row.get('micro_order_id')} | {row.get('symbol')} | {row.get('side')} | {row.get('order_amount_usd')} | "
            f"{row.get('request_status')} | {row.get('reject_reason_code') or '-'}"
        )
    if not live_rows:
        lines.append("No live micro orders found.")

    lines.extend(
        [
            "",
            "Recent Live Events:",
            "Event Type | Micro Order ID | Status | Reason",
        ]
    )
    for event in live_events:
        lines.append(
            f"{event.get('event_type')} | {event.get('micro_order_id')} | {event.get('after_status') or '-'} | "
            f"{event.get('reason_code') or event.get('reason_detail') or '-'}"
        )
    if not live_events:
        lines.append("No live events found.")
    return "\n".join(lines)


def _render_detail(row: dict[str, object]) -> str:
    events = get_micro_order_events(str(row.get("micro_order_id")))
    lines = [
        "[US Micro Order Detail]",
        "Phase 7-3 supports Mock, Sandbox, and gated Micro Live review.",
        "Micro Live validates connectivity under manual approval, LIMIT-only rules, and kill-switch protection.",
        "LIVE_ACCEPTED means broker request acceptance only. It does not mean a fill.",
        "",
        f"Micro Order ID: {row.get('micro_order_id')}",
        f"Approval ID: {row.get('approval_id')}",
        f"Status: {row.get('request_status')}",
        f"Execution Mode: {row.get('execution_mode')}",
        f"Trade Date: {row.get('trade_date')}",
        f"Account: {row.get('account_id')}",
        f"Symbol: {row.get('symbol')}",
        f"Side: {row.get('side')}",
        f"Amount USD: {row.get('order_amount_usd')}",
        f"Order Type: {row.get('order_type')}",
        f"Broker Order ID: {row.get('broker_order_id') or '-'}",
        f"Last Broker Status: {row.get('last_broker_status') or '-'}",
        f"Last Sync At: {row.get('last_sync_at') or '-'}",
        f"Filled Qty: {row.get('filled_qty') or '-'}",
        f"Remaining Qty: {row.get('remaining_qty') or '-'}",
        f"Avg Filled Price: {row.get('avg_filled_price') or '-'}",
        f"Filled Amount USD: {row.get('filled_amount_usd') or '-'}",
        f"Sync Status: {row.get('sync_status') or '-'}",
        f"Sync Error: {row.get('sync_error') or '-'}",
        f"Reject Reason: {row.get('reject_reason_code') or '-'} / {row.get('reject_reason_detail') or '-'}",
        "",
        "Events:",
    ]
    for event in events:
        lines.append(
            f"- {event.get('event_type')} {event.get('before_status') or '-'} -> {event.get('after_status') or '-'} "
            f"source={event.get('event_source') or '-'} at {event.get('created_at')}"
        )
    if not events:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.cancel:
        if not args.micro_order_id:
            raise ValueError("--micro-order-id is required for --cancel.")
        if not args.reason:
            raise ValueError("--reason is required for --cancel.")
        row = cancel_micro_order(args.micro_order_id, reason=args.reason, created_by=args.created_by)
        print(_render_detail(row))
        return 0
    if args.micro_order_id:
        print(_render_detail(get_micro_order(args.micro_order_id)))
        return 0
    report = build_micro_order_report(
        trade_date=args.trade_date,
        account_id=args.account_id,
        status=args.status,
        execution_mode=args.execution_mode,
    )
    output = _render_list(report, trade_date=args.trade_date, execution_mode=args.execution_mode)
    if args.format == "markdown":
        output_dir = Path(__file__).resolve().parents[2] / "outputs" / "us_stock_micro_live"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "us_micro_order_report.md"
        path.write_text(output.replace("[US Micro Order Report]", "# US Micro Order Report"), encoding="utf-8")
        print(path)
        return 0
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
