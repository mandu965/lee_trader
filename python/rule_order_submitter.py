from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kis_client import KISClient
from kis_live_account import order_cash
from rule_account_guard import assert_order_allowed
from rule_execution_simulator import (
    append_jsonl,
    build_aborted_results,
    load_calendar,
    load_json,
    render_reconciliation,
    validate_previous_execution_completed,
    validate_trading_session,
)
from rule_market_open_snapshot import check_market_data_available, resolve_rule_account_env
from rule_paper_state_manager import default_state
from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"

DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_RESULTS = OUTPUT_DIR / "rule_execution_results.json"
DEFAULT_RECON_MD = OUTPUT_DIR / "rule_execution_reconciliation_report.md"
DEFAULT_CALENDAR = ROOT / "config" / "trading_calendar_kr.json"
DEFAULT_EXECUTION_HISTORY = OUTPUT_DIR / "rule_execution_history.jsonl"
DEFAULT_MARKET_SNAPSHOT = OUTPUT_DIR / "rule_market_open_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit RULE pilot/live orders using KIS.")
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out-results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-reconciliation-md", type=Path, default=DEFAULT_RECON_MD)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--out-execution-history-jsonl", type=Path, default=DEFAULT_EXECUTION_HISTORY)
    parser.add_argument("--out-market-snapshot-json", type=Path, default=DEFAULT_MARKET_SNAPSHOT)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _append_reason(existing: str | None, *reasons: str | None) -> str:
    parts = [part for part in str(existing or "").split(";") if part and part != "none"]
    for reason in reasons:
        text = str(reason or "").strip()
        if text and text != "none":
            parts.append(text)
    unique = list(dict.fromkeys(parts))
    return ";".join(unique) if unique else "none"


def _ord_dvsn(item: dict[str, Any]) -> str:
    if str(item.get("order_type") or "").strip().lower() == "market":
        return "01"
    return "00"


def _ord_unpr(item: dict[str, Any], ord_dvsn: str) -> str:
    if ord_dvsn == "01":
        return "0"
    price = item.get("limit_price") or item.get("original_limit_price") or item.get("expected_execution_price")
    numeric = _float(price)
    if numeric is None or numeric <= 0:
        raise ValueError("limit_price_missing")
    return str(int(numeric))


def _build_market_snapshot(symbols: list[str], out_path: Path) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": [str(symbol).zfill(6) for symbol in symbols if str(symbol).strip()],
        **check_market_data_available(symbols),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _live_order_context(item: dict[str, Any], market_row: dict[str, Any] | None, preview: dict[str, Any]) -> dict[str, Any]:
    actual_open_gap = _float((market_row or {}).get("actual_open_gap"))
    actual_gap_reason = None
    if actual_open_gap is None and str(item.get("side") or "").upper() == "BUY":
        actual_gap_reason = "actual_open_gap_unavailable"
    elif actual_open_gap is not None and actual_open_gap > 0.05:
        actual_gap_reason = "actual_open_gap_gt_5pct"
    elif actual_open_gap is not None and actual_open_gap < -0.04:
        actual_gap_reason = "actual_open_gap_lt_minus_4pct"

    base_gap_reason = item.get("gap_risk_reason")
    gap_reason = actual_gap_reason or base_gap_reason
    return {
        "account_id": preview.get("account_id"),
        "strategy_id": preview.get("strategy_id"),
        "engine_type": preview.get("engine_type"),
        "run_mode": preview.get("run_mode"),
        "side": item.get("side"),
        "order_qty": item.get("order_qty"),
        "order_amount": item.get("order_amount"),
        "signal_strength": item.get("signal_strength"),
        "market_defensive_mode": item.get("market_defensive_mode"),
        "gap_risk_blocked": gap_reason not in {None, "", "none"},
        "gap_risk_reason": gap_reason,
        "trading_value_pass": item.get("trading_value_block_reason") in {None, "", "none"},
        "trading_value_block_reason": item.get("trading_value_block_reason"),
        "sector_limit_pass": item.get("sector_limit_pass", True),
        "cooldown_pass": item.get("cooldown_pass", True),
        "cash_limit_pass": item.get("cash_limit_pass", True),
    }


def _submit_items(preview: dict[str, Any], market_snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    account = resolve_rule_account_env()
    client = KISClient.from_env()
    client.issue_access_token()

    snapshot_map = market_snapshot.get("snapshots") or {}
    results: list[dict[str, Any]] = []
    submitted_count = 0
    failed_count = 0
    skipped_count = 0
    submitted_buy_amount = 0.0
    submitted_sell_amount = 0.0

    for item in preview.get("items") or []:
        row = dict(item)
        row["execution_checked_at"] = datetime.now().isoformat(timespec="seconds")
        row["submitted_at"] = None
        row["raw_response"] = None
        row["broker_order_id"] = None
        row["broker_org_order_id"] = None
        row["filled_qty"] = 0
        row["filled_amount"] = 0.0
        row["avg_fill_price"] = None
        row["unfilled_qty"] = int(float(row.get("order_qty") or 0))

        side = str(row.get("side") or "NONE").upper()
        if side == "NONE":
            row["order_status"] = "planned"
            row["reconciliation_status"] = "skipped_no_order_action"
            results.append(row)
            continue

        market_row = snapshot_map.get(str(row.get("code") or row.get("symbol") or "").zfill(6))
        row["actual_open_gap"] = _float((market_row or {}).get("actual_open_gap"))
        row["market_data_available"] = bool((market_row or {}).get("market_data_available"))
        row["api_health_status"] = market_snapshot.get("api_health_status")
        row["api_failure_reason"] = market_snapshot.get("api_failure_reason")
        if market_row and market_row.get("open_price") is not None:
            row["expected_execution_price"] = market_row.get("open_price")

        order_allowed, guard_reasons = assert_order_allowed(_live_order_context(row, market_row, preview))
        if not order_allowed:
            row["order_allowed"] = False
            row["order_block_reason"] = _append_reason(row.get("order_block_reason"), *guard_reasons)
            row["order_status"] = "blocked"
            row["reconciliation_status"] = "blocked_before_submit"
            skipped_count += 1
            results.append(row)
            continue

        try:
            ord_dvsn = _ord_dvsn(row)
            ord_unpr = _ord_unpr(row, ord_dvsn)
            response_df = order_cash(
                client,
                account,
                side=side.lower(),
                pdno=str(row.get("code") or row.get("symbol") or "").zfill(6),
                ord_dvsn=ord_dvsn,
                ord_qty=str(int(float(row.get("order_qty") or 0))),
                ord_unpr=ord_unpr,
            )
            response_row = response_df.iloc[0].to_dict() if not response_df.empty else {}
            row["submitted_at"] = datetime.now().isoformat(timespec="seconds")
            row["raw_response"] = response_row
            row["broker_order_id"] = response_row.get("ODNO") or response_row.get("odno")
            row["broker_org_order_id"] = response_row.get("KRX_FWDG_ORD_ORGNO") or response_row.get("krx_fwdg_ord_orgno")
            row["order_status"] = "submitted"
            row["reconciliation_status"] = "submitted_pending_fill_check"
            submitted_count += 1
            order_amount = float(row.get("order_amount") or 0.0)
            if side == "BUY":
                submitted_buy_amount += order_amount
            elif side == "SELL":
                submitted_sell_amount += order_amount
        except Exception as exc:
            row["submitted_at"] = datetime.now().isoformat(timespec="seconds")
            row["order_block_reason"] = _append_reason(row.get("order_block_reason"), "order_submit_failed", str(exc))
            row["order_status"] = "failed"
            row["reconciliation_status"] = "submit_failed"
            failed_count += 1
        results.append(row)

    return results, {
        "requested_count": len(results),
        "request_count": len(results),
        "submitted_count": submitted_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "filled_count": 0,
        "partial_filled_count": 0,
        "unfilled_count": 0,
        "canceled_count": 0,
        "simulated_filled_count": 0,
        "simulated_unfilled_count": 0,
        "buy_submitted_amount": submitted_buy_amount,
        "sell_submitted_amount": submitted_sell_amount,
    }


def main() -> None:
    args = parse_args()
    preview = load_json(args.order_preview_json, {})
    state = default_state()

    out_results = resolve(args.out_results_json)
    out_recon = resolve(args.out_reconciliation_md)
    out_execution_history = resolve(args.out_execution_history_jsonl)
    out_market_snapshot = resolve(args.out_market_snapshot_json)
    calendar = load_calendar(args.calendar_json)
    session_status = validate_trading_session(datetime.now(), str(preview.get("as_of_date") or ""), calendar)
    reconciliation_status = validate_previous_execution_completed(out_results, out_recon, str(preview.get("as_of_date") or ""))

    if not preview:
        results = build_aborted_results(
            {},
            state,
            "order_preview_missing",
            {"trading_day_valid": None, "trading_day_reason": "preview_missing"},
            reconciliation_status,
        )
    elif str(preview.get("run_mode") or "").lower() not in {"pilot", "live"}:
        results = build_aborted_results(
            preview,
            state,
            "order_submitter_requires_pilot_or_live",
            session_status,
            reconciliation_status,
        )
    elif not session_status.get("session_valid"):
        results = build_aborted_results(
            preview,
            state,
            str(session_status.get("session_reason") or "session_invalid"),
            session_status,
            reconciliation_status,
        )
    elif reconciliation_status.get("new_orders_blocked_by_reconciliation"):
        results = build_aborted_results(
            preview,
            state,
            str(reconciliation_status.get("reconciliation_block_reason") or "reconciliation_guard_blocked"),
            session_status,
            reconciliation_status,
        )
    else:
        buy_symbols = [
            str(item.get("code") or item.get("symbol") or "").zfill(6)
            for item in preview.get("items") or []
            if str(item.get("side") or "").upper() == "BUY"
        ]
        market_snapshot = _build_market_snapshot(buy_symbols, out_market_snapshot)
        items, summary = _submit_items(preview, market_snapshot)
        results = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "as_of_date": preview.get("as_of_date"),
            "account_id": preview.get("account_id"),
            "strategy_id": preview.get("strategy_id"),
            "engine_type": preview.get("engine_type"),
            "run_mode": preview.get("run_mode"),
            "paper_only": False,
            "trading_day_valid": session_status.get("trading_day_valid"),
            "trading_day_reason": session_status.get("trading_day_reason"),
            "market_data_available": market_snapshot.get("market_data_available"),
            "api_health_status": market_snapshot.get("api_health_status"),
            "api_failure_reason": market_snapshot.get("api_failure_reason"),
            "order_run_aborted": False,
            "order_run_abort_reason": None,
            "previous_reconciliation_found": reconciliation_status.get("previous_reconciliation_found"),
            "previous_reconciliation_status": reconciliation_status.get("previous_reconciliation_status"),
            "new_orders_blocked_by_reconciliation": reconciliation_status.get("new_orders_blocked_by_reconciliation"),
            "reconciliation_block_reason": reconciliation_status.get("reconciliation_block_reason"),
            "items": items,
            "summary": summary,
        }

    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_recon.write_text(render_reconciliation(results, state), encoding="utf-8")
    append_jsonl(out_execution_history, results)
    print(f"saved {out_results}")
    print(f"saved {out_recon}")
    print(f"appended {out_execution_history}")


if __name__ == "__main__":
    main()
