from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kis_client import KISClient
from kis_live_account import inquire_daily_ccld, inquire_psbl_rvsecncl, order_rvsecncl
from rule_execution_simulator import append_jsonl, default_state, load_json, render_reconciliation
from rule_live_account_snapshot import build_rule_live_account_state
from rule_market_open_snapshot import resolve_rule_account_env
from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"

DEFAULT_RESULTS = OUTPUT_DIR / "rule_execution_results.json"
DEFAULT_RECON_MD = OUTPUT_DIR / "rule_execution_reconciliation_report.md"
DEFAULT_EXECUTION_HISTORY = OUTPUT_DIR / "rule_execution_history.jsonl"
DEFAULT_FILL_SYNC = OUTPUT_DIR / "rule_execution_fill_sync.json"
DEFAULT_LIVE_STATE = OUTPUT_DIR / "rule_account_live_state.json"
DEFAULT_PARTIAL_FILL_QUEUE = OUTPUT_DIR / "rule_partial_fill_queue.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync RULE pilot/live fill results from KIS daily order history.")
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-reconciliation-md", type=Path, default=DEFAULT_RECON_MD)
    parser.add_argument("--out-execution-history-jsonl", type=Path, default=DEFAULT_EXECUTION_HISTORY)
    parser.add_argument("--out-fill-sync-json", type=Path, default=DEFAULT_FILL_SYNC)
    parser.add_argument("--out-live-state-json", type=Path, default=DEFAULT_LIVE_STATE)
    parser.add_argument("--out-partial-fill-queue-json", type=Path, default=DEFAULT_PARTIAL_FILL_QUEUE)
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD or YYYYMMDD. Defaults to submitted_at date or today.")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD or YYYYMMDD. Defaults to submitted_at date or today.")
    return parser.parse_args()


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10].replace("-", "")
    return text.replace("-", "")[:8]


def _num(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _broker_order_id_from_row(row: pd.Series) -> str:
    for key in ("odno", "ODNO", "orgn_odno"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _raw_rows_by_order(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if frame.empty:
        return grouped
    for raw in frame.where(pd.notna(frame), None).to_dict(orient="records"):
        row = pd.Series(raw)
        order_id = _broker_order_id_from_row(row)
        if not order_id:
            continue
        grouped.setdefault(order_id, []).append(raw)
    return grouped


def _status_from_order_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "filled_qty": 0,
            "unfilled_qty": None,
            "avg_fill_price": None,
            "filled_amount": 0.0,
            "fee": None,
            "tax": None,
            "order_status": "unfilled",
            "reconciliation_status": "unfilled_after_fill_check",
            "filled_at": None,
            "raw_fill_response": None,
        }

    frame = pd.DataFrame(rows)
    ord_qty = _num(frame["ord_qty"].iloc[0]) if "ord_qty" in frame.columns else None
    rmn_qty = _num(frame["rmn_qty"].iloc[0]) if "rmn_qty" in frame.columns else None
    tot_ccld_qty = _num(frame["tot_ccld_qty"].iloc[0]) if "tot_ccld_qty" in frame.columns else None
    filled_qty = tot_ccld_qty
    if filled_qty is None and ord_qty is not None and rmn_qty is not None:
        filled_qty = max(ord_qty - rmn_qty, 0.0)
    if filled_qty is None and "ccld_qty" in frame.columns:
        filled_qty = float(sum(_num(value) or 0.0 for value in frame["ccld_qty"]))
    filled_qty = max(float(filled_qty or 0.0), 0.0)

    total_amount = _num(frame["tot_ccld_amt"].iloc[0]) if "tot_ccld_amt" in frame.columns else None
    if total_amount is None and "ccld_qty" in frame.columns and "ccld_unpr" in frame.columns:
        total_amount = float(
            sum((_num(qty) or 0.0) * (_num(price) or 0.0) for qty, price in zip(frame["ccld_qty"], frame["ccld_unpr"]))
        )

    avg_fill_price = _num(frame["avg_prvs"].iloc[0]) if "avg_prvs" in frame.columns else None
    if avg_fill_price is None and "avg_pric" in frame.columns:
        avg_fill_price = _num(frame["avg_pric"].iloc[0])
    if avg_fill_price is None and total_amount is not None and filled_qty > 0:
        avg_fill_price = total_amount / filled_qty

    unfilled_qty = None
    if ord_qty is not None:
        unfilled_qty = max(ord_qty - filled_qty, 0.0)

    filled_at = None
    date_value = _date_text(frame["ord_dt"].iloc[0]) if "ord_dt" in frame.columns else ""
    for key in ("ccld_tmd", "ord_tmd"):
        if key in frame.columns:
            time_value = str(frame[key].iloc[0] or "").strip()
            if date_value and time_value:
                try:
                    filled_at = datetime.strptime(date_value + time_value.zfill(6)[:6], "%Y%m%d%H%M%S").isoformat(timespec="seconds")
                    break
                except ValueError:
                    pass

    if filled_qty <= 0:
        status = "unfilled"
        recon = "unfilled_after_fill_check"
    elif unfilled_qty and unfilled_qty > 0:
        status = "partial_filled"
        recon = "partial_filled_after_fill_check"
    else:
        status = "filled"
        recon = "filled_after_fill_check"

    return {
        "filled_qty": int(filled_qty),
        "unfilled_qty": int(unfilled_qty) if unfilled_qty is not None else None,
        "avg_fill_price": float(avg_fill_price) if avg_fill_price is not None else None,
        "filled_amount": float(total_amount if total_amount is not None else (filled_qty * (avg_fill_price or 0.0))),
        "fee": _num(frame["ord_tlex"].iloc[0]) if "ord_tlex" in frame.columns else None,
        "tax": _num(frame["stax"].iloc[0]) if "stax" in frame.columns else None,
        "order_status": status,
        "reconciliation_status": recon,
        "filled_at": filled_at,
        "raw_fill_response": rows,
    }


def _resolve_query_dates(results: dict[str, Any], start_date: str, end_date: str) -> tuple[str, str]:
    if start_date or end_date:
        resolved_end = _date_text(end_date) or _date_text(start_date) or datetime.now().strftime("%Y%m%d")
        resolved_start = _date_text(start_date) or resolved_end
        return resolved_start, resolved_end

    submitted_dates = sorted(
        {
            _date_text(item.get("submitted_at"))
            for item in results.get("items") or []
            if str(item.get("submitted_at") or "").strip()
        }
    )
    if submitted_dates:
        return submitted_dates[0], submitted_dates[-1]
    today = datetime.now().strftime("%Y%m%d")
    return today, today


def _write_skip(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _flag(name: str, default: str = "0") -> bool:
    return str(__import__("os").getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _cancel_unfilled_order(
    *,
    client: KISClient,
    account: Any,
    item: dict[str, Any],
    cancelable_rows: pd.DataFrame,
) -> dict[str, Any]:
    order_id = str(item.get("broker_order_id") or "").strip()
    if not order_id:
        return {"cancel_request_status": "not_applicable", "cancel_request_reason": "missing_broker_order_id"}
    row_match = cancelable_rows.loc[cancelable_rows.get("odno", pd.Series(dtype=str)).astype(str).str.strip() == order_id].copy()
    cancel_qty = int(float(item.get("unfilled_qty") or 0))
    if cancel_qty <= 0:
        return {"cancel_request_status": "not_applicable", "cancel_request_reason": "no_unfilled_qty"}
    if row_match.empty:
        return {"cancel_request_status": "not_found", "cancel_request_reason": "cancelable_order_row_missing"}
    cancel_row = row_match.iloc[0]
    psbl_qty = int(float(_num(cancel_row.get("psbl_qty")) or 0))
    if psbl_qty <= 0:
        return {"cancel_request_status": "not_applicable", "cancel_request_reason": "psbl_qty_zero"}
    cancel_qty = min(cancel_qty, psbl_qty)
    response_df = order_rvsecncl(
        client,
        account,
        krx_fwdg_ord_orgno=str(item.get("broker_org_order_id") or cancel_row.get("ord_gno_brno") or "").strip(),
        orgn_odno=order_id,
        ord_dvsn=str(item.get("ord_dvsn") or cancel_row.get("ord_dvsn_cd") or "01").strip(),
        rvse_cncl_dvsn_cd="02",
        ord_qty=str(cancel_qty),
        ord_unpr=str(int(float(item.get("limit_price") or item.get("original_limit_price") or 0))),
        qty_all_ord_yn="Y",
        excg_id_dvsn_cd=str(cancel_row.get("excg_id_dvsn_cd") or "KRX").strip() or "KRX",
    )
    response_row = response_df.iloc[0].to_dict() if not response_df.empty else {}
    return {
        "cancel_requested_at": datetime.now().isoformat(timespec="seconds"),
        "cancel_request_status": "submitted",
        "cancel_request_reason": None,
        "cancel_broker_order_id": response_row.get("ODNO") or response_row.get("odno"),
        "cancel_raw_response": response_row,
    }


def main() -> None:
    args = parse_args()
    in_results = load_json(args.results_json, {})
    out_results = resolve(args.out_results_json)
    out_recon = resolve(args.out_reconciliation_md)
    out_history = resolve(args.out_execution_history_jsonl)
    out_fill_sync = resolve(args.out_fill_sync_json)
    out_live_state = resolve(args.out_live_state_json)
    out_partial_fill_queue = resolve(args.out_partial_fill_queue_json)
    state = default_state()
    start_date, end_date = _resolve_query_dates(in_results, args.start_date, args.end_date)

    if not in_results:
        _write_skip(
            out_fill_sync,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "start_date": start_date,
                "end_date": end_date,
                "skip_reason": "execution_results_missing",
            },
        )
        print(f"saved {out_fill_sync}")
        return

    run_mode = str(in_results.get("run_mode") or "").lower()
    if run_mode not in {"pilot", "live"} or bool(in_results.get("order_run_aborted")):
        _write_skip(
            out_fill_sync,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "as_of_date": in_results.get("as_of_date"),
                "run_mode": in_results.get("run_mode"),
                "start_date": start_date,
                "end_date": end_date,
                "skip_reason": (
                    str(in_results.get("order_run_abort_reason") or "execution_aborted")
                    if bool(in_results.get("order_run_aborted"))
                    else "fill_sync_requires_pilot_or_live"
                ),
            },
        )
        print(f"saved {out_fill_sync}")
        return

    submitted_order_ids = {
        str(item.get("broker_order_id") or "").strip()
        for item in in_results.get("items") or []
        if str(item.get("order_status") or "").strip() == "submitted" and str(item.get("broker_order_id") or "").strip()
    }
    if not submitted_order_ids:
        _write_skip(
            out_fill_sync,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "as_of_date": in_results.get("as_of_date"),
                "run_mode": in_results.get("run_mode"),
                "start_date": start_date,
                "end_date": end_date,
                "skip_reason": "no_submitted_orders_to_sync",
            },
        )
        print(f"saved {out_fill_sync}")
        return

    client = KISClient.from_rule_env()
    client.issue_access_token()
    account = resolve_rule_account_env()
    frame, _ = inquire_daily_ccld(client, account, start_date=start_date, end_date=end_date)
    cancelable_rows = inquire_psbl_rvsecncl(client, account) if _flag("RULE_CANCEL_UNFILLED_ENABLED", "0") else pd.DataFrame()
    rows_by_order = _raw_rows_by_order(frame)

    updated_items: list[dict[str, Any]] = []
    sync_items: list[dict[str, Any]] = []
    counts = {
        "submitted_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "filled_count": 0,
        "partial_filled_count": 0,
        "unfilled_count": 0,
        "canceled_count": 0,
    }
    buy_filled_amount = 0.0
    sell_filled_amount = 0.0

    for item in in_results.get("items") or []:
        row = dict(item)
        order_id = str(row.get("broker_order_id") or "").strip()
        if row.get("order_status") == "submitted" and order_id:
            row.update(_status_from_order_rows(rows_by_order.get(order_id, [])))
            row["fill_sync_checked_at"] = datetime.now().isoformat(timespec="seconds")
            if row.get("order_status") in {"unfilled", "partial_filled"} and _flag("RULE_CANCEL_UNFILLED_ENABLED", "0"):
                try:
                    cancel_meta = _cancel_unfilled_order(
                        client=client,
                        account=account,
                        item=row,
                        cancelable_rows=cancelable_rows,
                    )
                    row.update(cancel_meta)
                    if row.get("order_status") == "unfilled" and row.get("cancel_request_status") == "submitted":
                        row["order_status"] = "canceled"
                        row["reconciliation_status"] = "canceled_after_fill_check"
                    elif row.get("order_status") == "partial_filled" and row.get("cancel_request_status") == "submitted":
                        row["reconciliation_status"] = "partial_filled_cancel_requested"
                except Exception as exc:
                    row["cancel_request_status"] = "failed"
                    row["cancel_request_reason"] = str(exc)
            sync_items.append(
                {
                    "broker_order_id": order_id,
                    "code": row.get("code"),
                    "side": row.get("side"),
                    "order_status": row.get("order_status"),
                    "filled_qty": row.get("filled_qty"),
                    "unfilled_qty": row.get("unfilled_qty"),
                    "avg_fill_price": row.get("avg_fill_price"),
                    "filled_amount": row.get("filled_amount"),
                    "cancel_request_status": row.get("cancel_request_status"),
                }
            )

        status = str(row.get("order_status") or "")
        if status == "submitted":
            counts["submitted_count"] += 1
        elif status == "failed":
            counts["failed_count"] += 1
        elif status == "filled":
            counts["filled_count"] += 1
        elif status == "partial_filled":
            counts["partial_filled_count"] += 1
        elif status == "unfilled":
            counts["unfilled_count"] += 1
        elif status == "canceled":
            counts["canceled_count"] += 1
        elif status in {"blocked", "planned"}:
            counts["skipped_count"] += 1
        amount = float(row.get("filled_amount") or 0.0)
        if str(row.get("side") or "").upper() == "BUY":
            buy_filled_amount += amount
        elif str(row.get("side") or "").upper() == "SELL":
            sell_filled_amount += amount
        updated_items.append(row)

    partial_fill_items = [
        {
            "queued_at": datetime.now().isoformat(timespec="seconds"),
            "code": item.get("code"),
            "side": item.get("side"),
            "filled_qty": item.get("filled_qty"),
            "unfilled_qty": item.get("unfilled_qty"),
            "order_qty": item.get("order_qty"),
            "avg_fill_price": item.get("avg_fill_price"),
            "broker_order_id": item.get("broker_order_id"),
        }
        for item in updated_items
        if str(item.get("side") or "").upper() == "BUY"
        and str(item.get("order_status") or "") == "partial_filled"
        and str(item.get("reconciliation_status") or "") != "partial_filled_cancel_requested"
    ]

    updated_results = dict(in_results)
    updated_results["generated_at"] = datetime.now().isoformat(timespec="seconds")
    updated_results["items"] = updated_items
    summary = dict(updated_results.get("summary") or {})
    summary["requested_count"] = len(updated_items)
    summary["request_count"] = len(updated_items)
    summary["submitted_count"] = counts["submitted_count"]
    summary["failed_count"] = counts["failed_count"]
    summary["skipped_count"] = counts["skipped_count"]
    summary["filled_count"] = counts["filled_count"]
    summary["partial_filled_count"] = counts["partial_filled_count"]
    summary["unfilled_count"] = counts["unfilled_count"]
    summary["canceled_count"] = counts["canceled_count"]
    summary["buy_filled_amount"] = buy_filled_amount
    summary["sell_filled_amount"] = sell_filled_amount
    updated_results["summary"] = summary

    fill_sync_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": updated_results.get("as_of_date"),
        "run_mode": updated_results.get("run_mode"),
        "start_date": start_date,
        "end_date": end_date,
        "api_row_count": int(len(frame)),
        "submitted_order_count": len(submitted_order_ids),
        **counts,
        "cancel_enabled": _flag("RULE_CANCEL_UNFILLED_ENABLED", "0"),
        "items": sync_items,
    }

    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_fill_sync.parent.mkdir(parents=True, exist_ok=True)
    out_live_state.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(updated_results, ensure_ascii=False, indent=2), encoding="utf-8")
    live_state, _ = build_rule_live_account_state(
        as_of_date=str(updated_results.get("as_of_date") or ""),
        execution_results_json=out_results,
    )
    out_recon.write_text(render_reconciliation(updated_results, state), encoding="utf-8")
    out_fill_sync.write_text(json.dumps(fill_sync_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_live_state.write_text(json.dumps(live_state, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(out_history, updated_results)
    out_partial_fill_queue.parent.mkdir(parents=True, exist_ok=True)
    out_partial_fill_queue.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "as_of_date": updated_results.get("as_of_date"),
                "run_mode": updated_results.get("run_mode"),
                "items": partial_fill_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out_results}")
    print(f"saved {out_recon}")
    print(f"saved {out_fill_sync}")
    print(f"saved {out_live_state}")
    print(f"saved {out_partial_fill_queue}")
    print(f"appended {out_history}")


if __name__ == "__main__":
    main()
