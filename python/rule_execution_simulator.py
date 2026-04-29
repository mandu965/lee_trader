from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from rule_market_open_snapshot import check_market_data_available
from rule_paper_state_manager import default_state
from rule_signal_builder import ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_STATE = OUTPUT_DIR / "rule_account_paper_state.json"
DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_RESULTS = OUTPUT_DIR / "rule_execution_results.json"
DEFAULT_RECON_MD = OUTPUT_DIR / "rule_execution_reconciliation_report.md"
DEFAULT_CALENDAR = ROOT / "config" / "trading_calendar_kr.json"
DEFAULT_EXECUTION_HISTORY = OUTPUT_DIR / "rule_execution_history.jsonl"
DEFAULT_STATE_HISTORY = OUTPUT_DIR / "rule_paper_state_history.jsonl"
DEFAULT_MARKET_SNAPSHOT = OUTPUT_DIR / "rule_market_open_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate RULE paper before-open execution and reconciliation.")
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--out-results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-reconciliation-md", type=Path, default=DEFAULT_RECON_MD)
    parser.add_argument("--out-state-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--out-execution-history-jsonl", type=Path, default=DEFAULT_EXECUTION_HISTORY)
    parser.add_argument("--out-state-history-jsonl", type=Path, default=DEFAULT_STATE_HISTORY)
    parser.add_argument("--out-market-snapshot-json", type=Path, default=DEFAULT_MARKET_SNAPSHOT)
    return parser.parse_args()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def load_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule signals not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["next_open", "actual_open_gap", "expected_gap", "open_gap", "close", "expected_entry_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)


def load_calendar(path: Path) -> dict[str, Any]:
    raw = load_json(path, {})
    closed_dates = {
        str(value).strip()
        for value in raw.get("closed_dates", [])
        if str(value).strip()
    }
    open_dates = {
        str(value).strip()
        for value in raw.get("open_dates", [])
        if str(value).strip()
    }
    return {
        "name": raw.get("name") or "KRX",
        "closed_dates": closed_dates,
        "open_dates": open_dates,
    }


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_hhmm(value: str, fallback: str) -> time:
    text = str(value or fallback).strip() or fallback
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError:
        return datetime.strptime(fallback, "%H:%M").time()


def is_trading_day(target: date, calendar_source: dict[str, Any]) -> tuple[bool, str]:
    target_text = target.isoformat()
    if target_text in calendar_source.get("open_dates", set()):
        return True, "forced_open_date"
    if target.weekday() >= 5:
        return False, "weekend"
    if target_text in calendar_source.get("closed_dates", set()):
        return False, "calendar_closed_date"
    return True, "regular_trading_day"


def get_next_trading_day(target: date, calendar_source: dict[str, Any]) -> date:
    current = target + timedelta(days=1)
    while True:
        valid, _ = is_trading_day(current, calendar_source)
        if valid:
            return current
        current += timedelta(days=1)


def validate_trading_session(now: datetime, preview_as_of_date: str | None, calendar_source: dict[str, Any]) -> dict[str, Any]:
    today = now.date()
    valid, reason = is_trading_day(today, calendar_source)
    if not valid:
        return {
            "trading_day_valid": False,
            "trading_day_reason": reason,
            "session_valid": False,
            "session_reason": reason,
        }

    preview_date = _parse_iso_date(preview_as_of_date)
    if preview_date is not None:
        expected = get_next_trading_day(preview_date, calendar_source)
        if today != expected:
            return {
                "trading_day_valid": True,
                "trading_day_reason": reason,
                "session_valid": False,
                "session_reason": f"execution_date_mismatch_expected_{expected.isoformat()}",
            }

    session_start = _parse_hhmm(os.getenv("RULE_BEFORE_OPEN_START_TIME", "08:55"), "08:55")
    session_end = _parse_hhmm(os.getenv("RULE_BEFORE_OPEN_END_TIME", "09:30"), "09:30")
    if not (session_start <= now.time() <= session_end):
        return {
            "trading_day_valid": True,
            "trading_day_reason": reason,
            "session_valid": False,
            "session_reason": f"outside_before_open_window_{session_start.strftime('%H:%M')}_{session_end.strftime('%H:%M')}",
        }

    return {
        "trading_day_valid": True,
        "trading_day_reason": reason,
        "session_valid": True,
        "session_reason": "session_window_ok",
    }


def validate_previous_execution_completed(results_path: Path, recon_path: Path, current_as_of_date: str | None) -> dict[str, Any]:
    existing_results = load_json(results_path, {})
    existing_recon_exists = resolve(recon_path).exists()
    if not existing_results:
        return {
            "previous_reconciliation_found": False,
            "previous_reconciliation_status": "bootstrap_no_previous_execution",
            "new_orders_blocked_by_reconciliation": False,
            "reconciliation_block_reason": None,
        }

    previous_as_of_date = str(existing_results.get("as_of_date") or "")
    if current_as_of_date and previous_as_of_date == current_as_of_date:
        return {
            "previous_reconciliation_found": existing_recon_exists,
            "previous_reconciliation_status": "same_day_rerun_allowed",
            "new_orders_blocked_by_reconciliation": False,
            "reconciliation_block_reason": None,
        }

    if not existing_recon_exists:
        return {
            "previous_reconciliation_found": False,
            "previous_reconciliation_status": "missing_reconciliation_report",
            "new_orders_blocked_by_reconciliation": True,
            "reconciliation_block_reason": "previous_reconciliation_report_missing",
        }

    if bool(existing_results.get("order_run_aborted")):
        return {
            "previous_reconciliation_found": True,
            "previous_reconciliation_status": "previous_execution_aborted",
            "new_orders_blocked_by_reconciliation": True,
            "reconciliation_block_reason": "previous_execution_aborted",
        }

    return {
        "previous_reconciliation_found": True,
        "previous_reconciliation_status": "success",
        "new_orders_blocked_by_reconciliation": False,
        "reconciliation_block_reason": None,
    }


def build_signal_map(signals: pd.DataFrame, as_of_date: str) -> dict[str, dict[str, Any]]:
    target = pd.to_datetime(as_of_date, errors="coerce")
    day = signals.loc[signals["date"] == target].copy()
    return {str(row["code"]).zfill(6): row.to_dict() for _, row in day.iterrows()}


def execution_price(
    signal_row: dict[str, Any],
    preview_item: dict[str, Any],
    market_snapshot: dict[str, Any] | None,
) -> tuple[float | None, float | None, bool, str]:
    if market_snapshot:
        open_price = _float(market_snapshot.get("open_price"))
        actual_gap = _float(market_snapshot.get("actual_open_gap"))
        if open_price and open_price > 0:
            return open_price, actual_gap, bool(market_snapshot.get("market_data_available")), "kis_open_snapshot"
    next_open = _float(signal_row.get("next_open"))
    actual_gap = (
        _float(signal_row.get("actual_open_gap"))
        or _float(signal_row.get("expected_gap"))
        or _float(signal_row.get("open_gap"))
    )
    if next_open and next_open > 0:
        return next_open, actual_gap, True, "signal_next_open"
    expected_execution_price = _float(preview_item.get("expected_execution_price"))
    if expected_execution_price and expected_execution_price > 0:
        return expected_execution_price, actual_gap, actual_gap is not None, "signal_expected_entry_price"
    return expected_execution_price, actual_gap, False, "unavailable"


def evaluate_preview_item(item: dict[str, Any], signal_row: dict[str, Any], market_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(item)
    side = str(row.get("side") or "NONE").upper()
    qty = int(float(row.get("order_qty") or 0))
    exec_price, actual_gap, market_data_available, market_data_source = execution_price(signal_row, row, market_snapshot)
    row["expected_execution_price"] = exec_price
    row["actual_open_gap"] = actual_gap
    row["market_data_available"] = market_data_available
    row["market_data_source"] = market_data_source
    if market_snapshot:
        row["api_health_status"] = "ok"
        row["api_failure_reason"] = None
    elif market_data_available:
        row["api_health_status"] = "signal_fallback"
        row["api_failure_reason"] = None
    else:
        row["api_health_status"] = "csv_fallback"
        row["api_failure_reason"] = "rule_market_open_snapshot_unavailable"
    row["execution_checked_at"] = datetime.now().isoformat(timespec="seconds")
    row["filled_qty"] = 0
    row["unfilled_qty"] = max(qty, 0)
    row["filled_amount"] = 0.0
    row["avg_fill_price"] = None
    row["order_run_aborted"] = False
    row["order_run_abort_reason"] = None

    if side == "NONE":
        row["order_status"] = "planned"
        row["reconciliation_status"] = "skipped_no_order_action"
        return row

    block_reasons: list[str] = []
    existing_block_reason = str(row.get("order_block_reason") or "none").strip()
    existing_reasons = [
        reason
        for reason in existing_block_reason.split(";")
        if reason and reason not in {"none", "paper_mode_no_order_submission"}
    ]
    block_reasons.extend(existing_reasons)
    if side == "BUY":
        if not market_data_available:
            block_reasons.append("actual_open_gap_unavailable")
        if actual_gap is not None and actual_gap > 0.05:
            block_reasons.append("actual_open_gap_gt_5pct")
        if actual_gap is not None and actual_gap < -0.04:
            block_reasons.append("actual_open_gap_lt_minus_4pct")

    if block_reasons:
        existing = str(row.get("order_block_reason") or "").strip()
        existing_parts = [part for part in existing.split(";") if part and part != "none"]
        combined = existing_parts + block_reasons
        row["order_block_reason"] = ";".join(dict.fromkeys(combined)) if combined else "none"
        row["order_status"] = "simulated_unfilled"
        row["reconciliation_status"] = "blocked_before_open"
        return row

    if exec_price and qty > 0:
        row["filled_qty"] = qty
        row["unfilled_qty"] = 0
        row["filled_amount"] = qty * exec_price
        row["avg_fill_price"] = exec_price
        row["order_status"] = "simulated_filled"
        row["reconciliation_status"] = "ready_to_apply"
        return row

    row["order_status"] = "simulated_unfilled"
    row["reconciliation_status"] = "invalid_size_or_price"
    return row


def apply_filled_orders_to_state(state: dict[str, Any], executed_items: list[dict[str, Any]], signal_map: dict[str, dict[str, Any]], as_of_date: str) -> dict[str, Any]:
    positions = {str(row.get("code") or "").zfill(6): dict(row) for row in state.get("positions") or []}
    cash = float(state.get("cash") or 0.0)
    recent_trades = list(state.get("recent_trades") or [])
    applied_ids = set(state.get("last_applied_order_ids") or [])

    for item in executed_items:
        if item.get("order_status") != "simulated_filled":
            continue
        order_id = str(item.get("order_id") or "")
        code = str(item.get("code") or item.get("symbol") or "").zfill(6)
        side = str(item.get("side") or "").upper()
        qty = int(float(item.get("filled_qty") or 0))
        price = _float(item.get("avg_fill_price")) or 0.0
        if not order_id or not code or qty <= 0 or price <= 0 or order_id in applied_ids:
            continue
        signal = signal_map.get(code, {})
        last_price = _float(signal.get("close")) or price

        if side == "BUY":
            filled_amount = qty * price
            if cash < filled_amount:
                continue
            current = positions.get(code)
            if current:
                old_qty = int(float(current.get("qty") or 0))
                total_qty = old_qty + qty
                avg_price = ((float(current.get("entry_price") or 0.0) * old_qty) + filled_amount) / total_qty
                current["qty"] = total_qty
                current["entry_price"] = avg_price
                current["last_price"] = last_price
                current["market_value"] = total_qty * last_price
                current["amount"] = current["market_value"]
            else:
                positions[code] = {
                    "code": code,
                    "name": signal.get("name") or item.get("name"),
                    "sector": signal.get("sector"),
                    "qty": qty,
                    "entry_price": price,
                    "last_price": last_price,
                    "market_value": qty * last_price,
                    "amount": qty * last_price,
                    "weight": 0.0,
                }
            cash -= filled_amount
        elif side == "SELL" and code in positions:
            current = positions[code]
            old_qty = int(float(current.get("qty") or 0))
            sell_qty = min(qty, old_qty)
            if sell_qty <= 0:
                continue
            proceeds = sell_qty * price
            current["qty"] = old_qty - sell_qty
            current["last_price"] = last_price
            current["market_value"] = current["qty"] * last_price
            current["amount"] = current["market_value"]
            cash += proceeds
            if current["qty"] <= 0:
                positions.pop(code, None)
        recent_trades.append(
            {
                "date": as_of_date,
                "code": code,
                "name": item.get("name"),
                "side": side,
                "qty": qty,
                "price": price,
                "amount": qty * price,
            }
        )
        applied_ids.add(order_id)

    position_rows = list(positions.values())
    total_market_value = sum(float(row.get("market_value") or 0.0) for row in position_rows)
    total_equity = cash + total_market_value
    for row in position_rows:
        mv = float(row.get("market_value") or 0.0)
        row["weight"] = (mv / total_equity) if total_equity > 0 else 0.0

    cooldown_codes = sorted(
        {
            str(row.get("code") or "").zfill(6)
            for row in recent_trades[-200:]
            if str(row.get("side") or "").upper() == "SELL"
        }
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "total_equity": total_equity,
        "cash": cash,
        "positions": sorted(position_rows, key=lambda row: float(row.get("weight") or 0.0), reverse=True),
        "recent_trades": recent_trades[-200:],
        "cooldown_codes": cooldown_codes,
        "last_applied_order_ids": sorted(applied_ids),
        "simulated_orders": [
            {
                "order_id": row.get("order_id"),
                "code": row.get("code"),
                "side": row.get("side"),
                "qty": row.get("filled_qty"),
                "price": row.get("avg_fill_price"),
            }
            for row in executed_items
            if row.get("order_status") == "simulated_filled"
        ],
    }


def build_results(preview: dict[str, Any], executed_items: list[dict[str, Any]], updated_state: dict[str, Any]) -> dict[str, Any]:
    requested = len(executed_items)
    filled = sum(1 for row in executed_items if row.get("order_status") == "simulated_filled")
    unfilled = sum(1 for row in executed_items if row.get("order_status") == "simulated_unfilled")
    buy_amount = sum(float(row.get("filled_amount") or 0.0) for row in executed_items if row.get("side") == "BUY")
    sell_amount = sum(float(row.get("filled_amount") or 0.0) for row in executed_items if row.get("side") == "SELL")
    market_data_available = all(
        bool(row.get("market_data_available"))
        for row in executed_items
        if str(row.get("side") or "").upper() == "BUY"
    ) if any(str(row.get("side") or "").upper() == "BUY" for row in executed_items) else True

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": preview.get("as_of_date"),
        "account_id": preview.get("account_id"),
        "strategy_id": preview.get("strategy_id"),
        "engine_type": preview.get("engine_type"),
        "run_mode": preview.get("run_mode"),
        "paper_only": True,
        "trading_day_valid": None,
        "trading_day_reason": "not_checked_in_paper_execution",
        "market_data_available": market_data_available,
        "api_health_status": "ok" if market_data_available else "market_data_unavailable",
        "api_failure_reason": None,
        "order_run_aborted": False,
        "order_run_abort_reason": None,
        "items": executed_items,
        "summary": {
            "requested_count": requested,
            "submitted_count": 0,
            "filled_count": 0,
            "partial_filled_count": 0,
            "unfilled_count": 0,
            "canceled_count": 0,
            "failed_count": 0,
            "simulated_filled_count": filled,
            "simulated_unfilled_count": unfilled,
            "buy_filled_amount": buy_amount,
            "sell_filled_amount": sell_amount,
            "cash_after": float(updated_state.get("cash") or 0.0),
            "total_equity_after": float(updated_state.get("total_equity") or 0.0),
            "position_count_after": len(updated_state.get("positions") or []),
        },
    }


def build_aborted_results(
    preview: dict[str, Any],
    state: dict[str, Any],
    reason: str,
    session_status: dict[str, Any],
    reconciliation_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": preview.get("as_of_date"),
        "account_id": preview.get("account_id") or "RULE_ACCOUNT_01",
        "strategy_id": preview.get("strategy_id") or "RULE_TREND_LIQUIDITY_V1",
        "engine_type": preview.get("engine_type") or "rule_based",
        "run_mode": preview.get("run_mode") or "paper",
        "paper_only": True,
        "trading_day_valid": session_status.get("trading_day_valid"),
        "trading_day_reason": session_status.get("trading_day_reason"),
        "market_data_available": False,
        "api_health_status": "not_called_paper_execution",
        "api_failure_reason": None,
        "order_run_aborted": True,
        "order_run_abort_reason": reason,
        "previous_reconciliation_found": reconciliation_status.get("previous_reconciliation_found"),
        "previous_reconciliation_status": reconciliation_status.get("previous_reconciliation_status"),
        "new_orders_blocked_by_reconciliation": reconciliation_status.get("new_orders_blocked_by_reconciliation"),
        "reconciliation_block_reason": reconciliation_status.get("reconciliation_block_reason"),
        "items": [],
        "summary": {
            "requested_count": 0,
            "submitted_count": 0,
            "filled_count": 0,
            "partial_filled_count": 0,
            "unfilled_count": 0,
            "canceled_count": 0,
            "failed_count": 0,
            "simulated_filled_count": 0,
            "simulated_unfilled_count": 0,
            "buy_filled_amount": 0.0,
            "sell_filled_amount": 0.0,
            "cash_after": float(state.get("cash") or 0.0),
            "total_equity_after": float(state.get("total_equity") or 0.0),
            "position_count_after": len(state.get("positions") or []),
        },
    }


def build_state_history_entry(state: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    positions = state.get("positions") or []
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": state.get("as_of_date"),
        "run_mode": results.get("run_mode"),
        "order_run_aborted": bool(results.get("order_run_aborted")),
        "order_run_abort_reason": results.get("order_run_abort_reason"),
        "position_count": len(positions),
        "cash": float(state.get("cash") or 0.0),
        "total_equity": float(state.get("total_equity") or 0.0),
        "cooldown_codes": list(state.get("cooldown_codes") or []),
        "positions": [
            {
                "code": row.get("code"),
                "name": row.get("name"),
                "qty": int(float(row.get("qty") or 0)),
                "weight": float(row.get("weight") or 0.0),
                "market_value": float(row.get("market_value") or 0.0),
            }
            for row in positions
        ],
    }


def render_reconciliation(results: dict[str, Any], state: dict[str, Any]) -> str:
    summary = results.get("summary") or {}
    positions = state.get("positions") or []
    max_positions_violation = len(positions) > 5
    cash = float(state.get("cash") or 0.0)
    total_equity = float(state.get("total_equity") or 0.0)
    min_cash_weight_violation = (cash / total_equity) < 0.2 if total_equity > 0 else False

    lines = [
        "# RULE Execution Reconciliation Report",
        "",
        f"- generated_at: `{results.get('generated_at')}`",
        f"- as_of_date: `{results.get('as_of_date')}`",
        f"- run_mode: `{results.get('run_mode')}`",
        f"- trading_day_valid: `{results.get('trading_day_valid')}`",
        f"- trading_day_reason: `{results.get('trading_day_reason')}`",
        f"- market_data_available: `{results.get('market_data_available')}`",
        f"- api_health_status: `{results.get('api_health_status')}`",
        f"- api_failure_reason: `{results.get('api_failure_reason')}`",
        f"- order_run_aborted: `{results.get('order_run_aborted')}`",
        f"- order_run_abort_reason: `{results.get('order_run_abort_reason')}`",
        f"- previous_reconciliation_status: `{results.get('previous_reconciliation_status')}`",
        f"- reconciliation_block_reason: `{results.get('reconciliation_block_reason')}`",
        "",
        "## Summary",
        "",
        f"- requested_count: `{summary.get('requested_count', 0)}`",
        f"- submitted_count: `{summary.get('submitted_count', 0)}`",
        f"- failed_count: `{summary.get('failed_count', 0)}`",
        f"- skipped_count: `{summary.get('skipped_count', 0)}`",
        f"- filled_count: `{summary.get('filled_count', 0)}`",
        f"- partial_filled_count: `{summary.get('partial_filled_count', 0)}`",
        f"- unfilled_count: `{summary.get('unfilled_count', 0)}`",
        f"- canceled_count: `{summary.get('canceled_count', 0)}`",
        f"- simulated_filled_count: `{summary.get('simulated_filled_count', 0)}`",
        f"- simulated_unfilled_count: `{summary.get('simulated_unfilled_count', 0)}`",
        f"- buy_filled_amount: `{float(summary.get('buy_filled_amount') or 0.0):,.0f}`",
        f"- sell_filled_amount: `{float(summary.get('sell_filled_amount') or 0.0):,.0f}`",
        f"- cash_after: `{float(summary.get('cash_after') or 0.0):,.0f}`",
        f"- total_equity_after: `{float(summary.get('total_equity_after') or 0.0):,.0f}`",
        "",
        "## Guard Check",
        "",
        f"- max_positions_violation: `{max_positions_violation}`",
        f"- min_cash_weight_violation: `{min_cash_weight_violation}`",
        "",
        "## Orders",
        "",
        "| order_id | code | side | status | filled_qty | avg_fill_price | block_reason |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in results.get("items") or []:
        lines.append(
            "| {order_id} | {code} | {side} | {status} | {qty} | {price} | {reason} |".format(
                order_id=row.get("order_id") or "",
                code=row.get("code") or "",
                side=row.get("side") or "",
                status=row.get("order_status") or "",
                qty=int(float(row.get("filled_qty") or 0)),
                price=f"{float(row.get('avg_fill_price') or 0.0):,.0f}" if row.get("avg_fill_price") is not None else "-",
                reason=row.get("order_block_reason") or "-",
            )
        )
    lines.extend(["", "## Positions After", ""])
    if not positions:
        lines.append("_No open positions._")
    else:
        lines.extend(["| code | name | qty | entry_price | last_price | market_value | weight |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in positions:
            lines.append(
                "| {code} | {name} | {qty} | {entry:.0f} | {last:.0f} | {mv:.0f} | {wt:.2%} |".format(
                    code=row.get("code") or "",
                    name=row.get("name") or "",
                    qty=int(float(row.get("qty") or 0)),
                    entry=float(row.get("entry_price") or 0.0),
                    last=float(row.get("last_price") or 0.0),
                    mv=float(row.get("market_value") or 0.0),
                    wt=float(row.get("weight") or 0.0),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    preview = load_json(args.order_preview_json, {})
    state = load_json(args.state_json, default_state())

    out_results = resolve(args.out_results_json)
    out_recon = resolve(args.out_reconciliation_md)
    out_state = resolve(args.out_state_json)
    out_execution_history = resolve(args.out_execution_history_jsonl)
    out_state_history = resolve(args.out_state_history_jsonl)
    out_market_snapshot = resolve(args.out_market_snapshot_json)
    calendar = load_calendar(args.calendar_json)
    session_status = validate_trading_session(datetime.now(), str(preview.get("as_of_date") or ""), calendar)
    reconciliation_status = validate_previous_execution_completed(out_results, out_recon, str(preview.get("as_of_date") or ""))

    if not preview:
        results = build_aborted_results(
            {},
            state,
            "order_preview_missing",
            {
                "trading_day_valid": None,
                "trading_day_reason": "preview_missing",
            },
            reconciliation_status,
        )
        updated_state = state
    elif not session_status.get("session_valid"):
        results = build_aborted_results(
            preview,
            state,
            str(session_status.get("session_reason") or "session_invalid"),
            session_status,
            reconciliation_status,
        )
        updated_state = state
    elif str(preview.get("run_mode") or "paper").lower() != "paper":
        results = build_aborted_results(
            preview,
            state,
            "execution_simulator_supports_paper_only",
            session_status,
            reconciliation_status,
        )
        updated_state = state
    elif reconciliation_status.get("new_orders_blocked_by_reconciliation"):
        results = build_aborted_results(
            preview,
            state,
            str(reconciliation_status.get("reconciliation_block_reason") or "reconciliation_guard_blocked"),
            session_status,
            reconciliation_status,
        )
        updated_state = state
    else:
        signals = load_signals(args.signals_csv)
        signal_map = build_signal_map(signals, str(preview.get("as_of_date") or ""))
        symbols = [str(item.get("code") or item.get("symbol") or "").zfill(6) for item in preview.get("items") or [] if str(item.get("side") or "").upper() == "BUY"]
        market_snapshot_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "symbols": symbols,
            "market_data_available": False,
            "api_health_status": "not_called_paper_execution",
            "api_failure_reason": None,
            "snapshots": {},
            "failures": [],
        }
        if symbols:
            market_snapshot_payload.update(check_market_data_available(symbols))
        out_market_snapshot.parent.mkdir(parents=True, exist_ok=True)
        out_market_snapshot.write_text(json.dumps(market_snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot_map = market_snapshot_payload.get("snapshots") or {}
        executed_items = [
            evaluate_preview_item(
                item,
                signal_map.get(str(item.get("code") or "").zfill(6), {}),
                snapshot_map.get(str(item.get("code") or "").zfill(6)),
            )
            for item in preview.get("items") or []
        ]
        updated_state = apply_filled_orders_to_state(state, executed_items, signal_map, str(preview.get("as_of_date") or ""))
        results = build_results(preview, executed_items, updated_state)
        results["trading_day_valid"] = session_status.get("trading_day_valid")
        results["trading_day_reason"] = session_status.get("trading_day_reason")
        results["previous_reconciliation_found"] = reconciliation_status.get("previous_reconciliation_found")
        results["previous_reconciliation_status"] = reconciliation_status.get("previous_reconciliation_status")
        results["new_orders_blocked_by_reconciliation"] = reconciliation_status.get("new_orders_blocked_by_reconciliation")
        results["reconciliation_block_reason"] = reconciliation_status.get("reconciliation_block_reason")
        buy_rows = [row for row in executed_items if str(row.get("side") or "").upper() == "BUY"]
        fallback_ready = bool(buy_rows) and all(bool(row.get("market_data_available")) for row in buy_rows)
        results["market_data_available"] = market_snapshot_payload.get("market_data_available") or fallback_ready
        if market_snapshot_payload.get("market_data_available"):
            results["api_health_status"] = market_snapshot_payload.get("api_health_status")
            results["api_failure_reason"] = market_snapshot_payload.get("api_failure_reason")
        elif fallback_ready:
            results["api_health_status"] = "signal_fallback"
            results["api_failure_reason"] = None
        else:
            results["api_health_status"] = market_snapshot_payload.get("api_health_status")
            results["api_failure_reason"] = market_snapshot_payload.get("api_failure_reason")

    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_state.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_recon.write_text(render_reconciliation(results, updated_state), encoding="utf-8")
    out_state.write_text(json.dumps(updated_state, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(out_execution_history, results)
    append_jsonl(out_state_history, build_state_history_entry(updated_state, results))
    print(f"saved {out_results}")
    print(f"saved {out_recon}")
    print(f"saved {out_state}")
    print(f"appended {out_execution_history}")
    print(f"appended {out_state_history}")


if __name__ == "__main__":
    main()
