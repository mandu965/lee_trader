from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path

from python.us.us_db import (
    ensure_us_micro_live_tables,
    fetch_us_micro_order_fill_rows,
    fetch_us_micro_order_request_rows,
    insert_us_micro_reconciliation_event_log_rows,
    upsert_us_micro_reconciliation_result_rows,
)
from utils.us_live_kill_switch import activate_kill_switch
from utils.us_live_order_safety import mask_sensitive_payload
from utils.us_live_order_client import UsLiveOrderClient
from utils.us_live_account_client import UsLiveAccountClient
from utils.us_mock_account_client import UsMockAccountClient
from utils.us_mock_order_client import UsMockOrderClient
from utils.us_order_status_mapper import map_broker_order_status, normalize_fill_payload
from utils.us_sandbox_account_client import UsSandboxAccountClient
from utils.us_sandbox_order_client import UsSandboxOrderClient


RECON_TYPES = {"ORDER_STATUS", "FILL", "POSITION", "CASH", "ACCOUNT_EQUITY", "SUMMARY"}
RECON_STATUSES = {"MATCH", "MISMATCH", "MISSING_INTERNAL", "MISSING_BROKER", "UNKNOWN", "ERROR"}
SEVERITIES = {"INFO", "WARNING", "ERROR", "CRITICAL"}


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _safe_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _safe_float(value: object) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_text(payload: object | None) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return json.dumps(mask_sensitive_payload(payload), ensure_ascii=False, indent=2, default=str)


def _tolerance_qty() -> float:
    return _safe_float(_text("US_MICRO_RECON_TOLERANCE_QTY", "0.000001")) or 0.000001


def _tolerance_amount() -> float:
    return _safe_float(_text("US_MICRO_RECON_TOLERANCE_AMOUNT_USD", "1.00")) or 1.0


def _tolerance_cash() -> float:
    return _safe_float(_text("US_MICRO_RECON_TOLERANCE_CASH_USD", "1.00")) or 1.0


def _select_order_client(execution_mode: str):
    mode = _safe_str(execution_mode).upper()
    if mode == "MOCK":
        return UsMockOrderClient()
    if mode == "SANDBOX":
        return UsSandboxOrderClient()
    if mode == "LIVE":
        if not _flag("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", "false"):
            raise RuntimeError("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY must be true for LIVE reconciliation.")
        if not _flag("US_MICRO_RECON_REAL_ORDER_BLOCKED", "true"):
            raise RuntimeError("US_MICRO_RECON_REAL_ORDER_BLOCKED must remain true for LIVE reconciliation.")
        return UsLiveOrderClient()
    raise ValueError(f"Unsupported execution_mode: {mode}")


def _select_account_client(execution_mode: str):
    mode = _safe_str(execution_mode).upper()
    if mode == "MOCK":
        return UsMockAccountClient()
    if mode == "SANDBOX":
        return UsSandboxAccountClient()
    if mode == "LIVE":
        return UsLiveAccountClient()
    raise ValueError(f"Unsupported execution_mode: {mode}")


def _assert_reconciliation_enabled(execution_mode: str) -> None:
    if not _flag("US_MICRO_RECON_ENABLED", "false"):
        raise RuntimeError("US_MICRO_RECON_ENABLED must be true before reconciliation can run.")
    mode = _safe_str(execution_mode).upper()
    if mode == "LIVE":
        if not _flag("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", "false"):
            raise RuntimeError("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY must be true for LIVE reconciliation.")
        if not _flag("US_MICRO_RECON_REAL_ORDER_BLOCKED", "true"):
            raise RuntimeError("US_MICRO_RECON_REAL_ORDER_BLOCKED must remain true for LIVE reconciliation.")


def _build_recon_run_id(account_id: str, recon_date: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"USRECON_{account_id}_{recon_date.replace('-', '')}_{timestamp}"


def _build_recon_id(recon_run_id: str, recon_type: str, token: str, seq: int) -> str:
    return f"USRECONITEM_{recon_run_id}_{recon_type}_{token}_{seq}"


def _build_event_row(
    *,
    recon_run_id: str,
    event_type: str,
    account_id: str,
    execution_mode: str,
    message: str,
    severity: str,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return {
        "event_id": f"USRECONLOG_{timestamp}",
        "recon_run_id": recon_run_id,
        "event_type": event_type,
        "account_id": account_id,
        "execution_mode": execution_mode,
        "message": message,
        "severity": severity,
        "created_at": datetime.now(timezone.utc),
    }


def _build_summary_result(
    *,
    recon_run_id: str,
    recon_date: date,
    account_id: str,
    execution_mode: str,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "recon_id": _build_recon_id(recon_run_id, "SUMMARY", "RUN", 1),
        "recon_run_id": recon_run_id,
        "recon_date": recon_date,
        "account_id": account_id,
        "execution_mode": execution_mode,
        "recon_type": "SUMMARY",
        "symbol": None,
        "micro_order_id": None,
        "broker_order_id": None,
        "internal_qty": None,
        "broker_qty": None,
        "qty_diff": None,
        "internal_amount_usd": None,
        "broker_amount_usd": None,
        "amount_diff_usd": None,
        "internal_cash_usd": None,
        "broker_cash_usd": None,
        "cash_diff_usd": None,
        "internal_status": None,
        "broker_status": None,
        "recon_status": "MATCH" if not summary.get("critical_count") and not summary.get("error_count") and not summary.get("mismatch_count") else "MISMATCH",
        "severity": "CRITICAL" if summary.get("critical_count") else ("ERROR" if summary.get("error_count") else "INFO"),
        "reason_code": "reconciliation_summary",
        "reason_detail": (
            f"match={summary.get('match_count', 0)}, mismatch={summary.get('mismatch_count', 0)}, "
            f"missing_internal={summary.get('missing_internal_count', 0)}, missing_broker={summary.get('missing_broker_count', 0)}, "
            f"error={summary.get('error_count', 0)}, critical={summary.get('critical_count', 0)}"
        ),
        "raw_internal_payload": None,
        "raw_broker_payload": _json_text(summary),
        "created_at": datetime.now(timezone.utc),
    }


def _severity_for_order_status(internal_status: str, broker_status: str, recon_status: str, reason_code: str) -> str:
    if recon_status in {"ERROR", "MISSING_BROKER"}:
        return "ERROR"
    if recon_status == "MATCH":
        return "INFO"
    if reason_code == "order_status_mismatch" and "ORDER_FILLED" in {internal_status, broker_status}:
        return "CRITICAL"
    return "WARNING"


def _severity_for_fill(reason_code: str, recon_status: str) -> str:
    if recon_status == "MATCH":
        return "INFO"
    if reason_code in {"fill_qty_mismatch", "fill_amount_mismatch"}:
        return "CRITICAL"
    if recon_status == "ERROR":
        return "ERROR"
    return "WARNING"


def _severity_for_position(recon_status: str) -> str:
    if recon_status == "MATCH":
        return "INFO"
    if recon_status in {"MISMATCH", "MISSING_INTERNAL", "MISSING_BROKER"}:
        return "CRITICAL"
    return "ERROR"


def _severity_for_cash(recon_status: str, reason_code: str) -> str:
    if recon_status == "MATCH":
        return "INFO"
    if reason_code == "internal_cash_unavailable":
        return "WARNING"
    if recon_status == "UNKNOWN":
        return "WARNING"
    if recon_status == "ERROR":
        return "ERROR"
    return "CRITICAL"


def build_internal_expected_positions(
    account_id: str,
    recon_date: str,
    execution_mode: str | None = None,
) -> dict[str, dict[str, object]]:
    recon_date_value = date.fromisoformat(str(recon_date)[:10])
    orders = fetch_us_micro_order_request_rows(
        account_id=account_id,
        trade_date=recon_date_value,
        execution_mode=_safe_str(execution_mode).upper() if execution_mode else None,
    )
    fills: list[dict[str, object]] = []
    for order in orders:
        fills.extend(fetch_us_micro_order_fill_rows(micro_order_id=_safe_str(order.get("micro_order_id"))))
    positions: dict[str, dict[str, object]] = {}
    for fill in fills:
        symbol = _safe_str(fill.get("symbol")).upper()
        if not symbol:
            continue
        qty = _safe_float(fill.get("filled_qty")) or 0.0
        amount = _safe_float(fill.get("filled_amount_usd")) or 0.0
        side = _safe_str(fill.get("side")).upper()
        sign = 1.0 if side == "BUY" else -1.0
        bucket = positions.setdefault(
            symbol,
            {
                "symbol": symbol,
                "internal_qty": 0.0,
                "internal_amount_usd": 0.0,
                "fills": [],
            },
        )
        bucket["internal_qty"] = round(float(bucket["internal_qty"]) + (sign * qty), 6)
        bucket["internal_amount_usd"] = round(float(bucket["internal_amount_usd"]) + (sign * amount), 6)
        bucket["fills"].append(fill)
    return positions


def fetch_broker_positions(account_id: str, execution_mode: str) -> list[dict]:
    client = _select_account_client(execution_mode)
    return client.get_positions(account_id)


def compare_order_statuses(internal_orders: list[dict], broker_orders: list[dict]) -> list[dict]:
    broker_map = {_safe_str(item.get("micro_order_id")): item for item in broker_orders}
    results: list[dict[str, object]] = []
    for row in internal_orders:
        micro_order_id = _safe_str(row.get("micro_order_id"))
        broker_row = broker_map.get(micro_order_id)
        internal_status = _safe_str(row.get("request_status"), "UNKNOWN").upper()
        if not _safe_str(row.get("broker_order_id")):
            reason_code = "broker_order_missing"
            recon_status = "MISSING_BROKER"
            broker_status = ""
        elif not broker_row:
            reason_code = "order_status_query_failed"
            recon_status = "ERROR"
            broker_status = ""
        else:
            broker_status = _safe_str(broker_row.get("broker_status"), "UNKNOWN").upper()
            if broker_status == "ORDER_UNKNOWN":
                reason_code = "broker_status_unknown"
                recon_status = "UNKNOWN"
            elif broker_status == internal_status:
                reason_code = None
                recon_status = "MATCH"
            else:
                reason_code = "order_status_mismatch"
                recon_status = "MISMATCH"
        results.append(
            {
                "recon_type": "ORDER_STATUS",
                "symbol": row.get("symbol"),
                "micro_order_id": micro_order_id,
                "broker_order_id": row.get("broker_order_id"),
                "internal_status": internal_status,
                "broker_status": broker_status or None,
                "recon_status": recon_status,
                "severity": _severity_for_order_status(internal_status, broker_status, recon_status, reason_code or ""),
                "reason_code": reason_code,
                "reason_detail": None if recon_status == "MATCH" else f"internal={internal_status}, broker={broker_status or 'N/A'}",
                "raw_internal_payload": _json_text(row),
                "raw_broker_payload": _json_text(broker_row),
            }
        )
    return results


def compare_fills(internal_fills: list[dict], broker_fills: list[dict]) -> list[dict]:
    results: list[dict[str, object]] = []
    internal_map = {_safe_str(item.get("broker_fill_id") or item.get("micro_fill_id")): item for item in internal_fills}
    broker_map = {_safe_str(item.get("broker_fill_id")): item for item in broker_fills if _safe_str(item.get("broker_fill_id"))}
    keys = sorted(set(internal_map) | set(broker_map))
    for key in keys:
        internal = internal_map.get(key)
        broker = broker_map.get(key)
        internal_qty = _safe_float((internal or {}).get("filled_qty"))
        broker_qty = _safe_float((broker or {}).get("filled_qty"))
        internal_amount = _safe_float((internal or {}).get("filled_amount_usd"))
        broker_amount = _safe_float((broker or {}).get("filled_amount_usd"))
        qty_diff = None if internal_qty is None or broker_qty is None else round(internal_qty - broker_qty, 6)
        amount_diff = None if internal_amount is None or broker_amount is None else round(internal_amount - broker_amount, 6)
        reason_code = None
        recon_status = "MATCH"
        if internal is None:
            recon_status = "MISSING_INTERNAL"
            reason_code = "fill_missing_internal"
        elif broker is None:
            recon_status = "MISSING_BROKER"
            reason_code = "fill_missing_broker"
        elif qty_diff is not None and abs(qty_diff) > _tolerance_qty():
            recon_status = "MISMATCH"
            reason_code = "fill_qty_mismatch"
        elif amount_diff is not None and abs(amount_diff) > _tolerance_amount():
            recon_status = "MISMATCH"
            reason_code = "fill_amount_mismatch"
        else:
            internal_price = _safe_float((internal or {}).get("filled_price"))
            broker_price = _safe_float((broker or {}).get("filled_price"))
            if internal_price is not None and broker_price is not None and abs(internal_price - broker_price) > _tolerance_amount():
                recon_status = "MISMATCH"
                reason_code = "fill_price_mismatch"
        sample = internal or broker or {}
        results.append(
            {
                "recon_type": "FILL",
                "symbol": sample.get("symbol"),
                "micro_order_id": sample.get("micro_order_id"),
                "broker_order_id": sample.get("broker_order_id"),
                "internal_qty": internal_qty,
                "broker_qty": broker_qty,
                "qty_diff": qty_diff,
                "internal_amount_usd": internal_amount,
                "broker_amount_usd": broker_amount,
                "amount_diff_usd": amount_diff,
                "recon_status": recon_status,
                "severity": _severity_for_fill(reason_code or "", recon_status),
                "reason_code": reason_code,
                "reason_detail": None if recon_status == "MATCH" else f"fill_key={key}",
                "raw_internal_payload": _json_text(internal),
                "raw_broker_payload": _json_text(broker),
            }
        )
    return results


def compare_positions(internal_positions: dict, broker_positions: list[dict]) -> list[dict]:
    results: list[dict[str, object]] = []
    broker_map = {_safe_str(item.get("symbol")).upper(): item for item in broker_positions if _safe_str(item.get("symbol"))}
    symbols = sorted(set(internal_positions) | set(broker_map))
    for symbol in symbols:
        internal = internal_positions.get(symbol)
        broker = broker_map.get(symbol)
        internal_qty = _safe_float((internal or {}).get("internal_qty"))
        broker_qty = _safe_float((broker or {}).get("qty"))
        qty_diff = None if internal_qty is None or broker_qty is None else round(internal_qty - broker_qty, 6)
        if internal is None:
            recon_status = "MISSING_INTERNAL"
            reason_code = "unexpected_broker_position"
        elif broker is None:
            recon_status = "MISSING_BROKER"
            reason_code = "position_missing_broker"
        elif qty_diff is not None and abs(qty_diff) > _tolerance_qty():
            recon_status = "MISMATCH"
            reason_code = "position_qty_mismatch"
        else:
            recon_status = "MATCH"
            reason_code = None
        results.append(
            {
                "recon_type": "POSITION",
                "symbol": symbol,
                "micro_order_id": None,
                "broker_order_id": None,
                "internal_qty": internal_qty,
                "broker_qty": broker_qty,
                "qty_diff": qty_diff,
                "internal_amount_usd": _safe_float((internal or {}).get("internal_amount_usd")),
                "broker_amount_usd": _safe_float((broker or {}).get("market_value")),
                "amount_diff_usd": None,
                "recon_status": recon_status,
                "severity": _severity_for_position(recon_status),
                "reason_code": reason_code,
                "reason_detail": None if recon_status == "MATCH" else f"symbol={symbol}",
                "raw_internal_payload": _json_text(internal),
                "raw_broker_payload": _json_text(broker),
            }
        )
    return results


def compare_cash(internal_cash: dict, broker_cash: dict) -> list[dict]:
    internal_cash_usd = _safe_float(internal_cash.get("cash_balance") if isinstance(internal_cash, dict) else None)
    broker_cash_usd = _safe_float(broker_cash.get("cash_balance") if isinstance(broker_cash, dict) else None)
    cash_diff = None if internal_cash_usd is None or broker_cash_usd is None else round(internal_cash_usd - broker_cash_usd, 6)
    reason_code = None
    recon_status = "MATCH"
    if broker_cash_usd is None:
        recon_status = "ERROR"
        reason_code = "cash_query_failed"
    elif internal_cash_usd is None:
        recon_status = "UNKNOWN"
        reason_code = "internal_cash_unavailable"
    elif abs(cash_diff or 0.0) > _tolerance_cash():
        recon_status = "MISMATCH"
        reason_code = "cash_mismatch"
    return [
        {
            "recon_type": "CASH",
            "symbol": None,
            "micro_order_id": None,
            "broker_order_id": None,
            "internal_cash_usd": internal_cash_usd,
            "broker_cash_usd": broker_cash_usd,
            "cash_diff_usd": cash_diff,
            "recon_status": recon_status,
            "severity": _severity_for_cash(recon_status, reason_code or ""),
            "reason_code": reason_code,
            "reason_detail": None if recon_status == "MATCH" else "cash balance comparison",
            "raw_internal_payload": _json_text(internal_cash),
            "raw_broker_payload": _json_text(broker_cash),
        }
    ]


def summarize_reconciliation(results: list[dict]) -> dict:
    summary = {
        "match_count": 0,
        "mismatch_count": 0,
        "missing_internal_count": 0,
        "missing_broker_count": 0,
        "unknown_count": 0,
        "error_count": 0,
        "critical_count": 0,
        "warning_count": 0,
        "info_count": 0,
        "result_count": len(results),
        "kill_switch_recommended": False,
        "kill_switch_triggered": False,
    }
    for row in results:
        status = _safe_str(row.get("recon_status")).upper()
        severity = _safe_str(row.get("severity")).upper()
        if status == "MATCH":
            summary["match_count"] += 1
        elif status == "MISMATCH":
            summary["mismatch_count"] += 1
        elif status == "MISSING_INTERNAL":
            summary["missing_internal_count"] += 1
        elif status == "MISSING_BROKER":
            summary["missing_broker_count"] += 1
        elif status == "UNKNOWN":
            summary["unknown_count"] += 1
        elif status == "ERROR":
            summary["error_count"] += 1
        if severity == "CRITICAL":
            summary["critical_count"] += 1
        elif severity == "WARNING":
            summary["warning_count"] += 1
        elif severity == "INFO":
            summary["info_count"] += 1
    summary["kill_switch_recommended"] = summary["critical_count"] > 0
    return summary


def _fetch_internal_orders(account_id: str, recon_date: date, execution_mode: str) -> list[dict[str, object]]:
    return fetch_us_micro_order_request_rows(
        account_id=account_id,
        trade_date=recon_date,
        execution_mode=_safe_str(execution_mode).upper(),
    )


def _fetch_internal_fills(internal_orders: list[dict[str, object]]) -> list[dict[str, object]]:
    fills: list[dict[str, object]] = []
    for row in internal_orders:
        fills.extend(fetch_us_micro_order_fill_rows(micro_order_id=_safe_str(row.get("micro_order_id"))))
    return fills


def _fetch_broker_order_snapshots(internal_orders: list[dict[str, object]], execution_mode: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    client = _select_order_client(execution_mode)
    statuses: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    for row in internal_orders:
        micro_order_id = _safe_str(row.get("micro_order_id"))
        broker_order_id = _safe_str(row.get("broker_order_id"))
        if not broker_order_id:
            continue
        status_response = client.get_order_status(broker_order_id)
        mapped_status = map_broker_order_status(
            _safe_str(row.get("broker_name"), _safe_str(execution_mode)),
            _safe_str(status_response.get("status"), "UNKNOWN"),
            status_response,
        )
        statuses.append(
            {
                "micro_order_id": micro_order_id,
                "broker_order_id": broker_order_id,
                "broker_status_raw": status_response.get("status"),
                "broker_status": mapped_status,
                "success": status_response.get("success"),
                "raw_payload": status_response,
            }
        )
        raw_fills = client.get_order_fills(broker_order_id)
        for item in raw_fills:
            normalized = normalize_fill_payload(_safe_str(row.get("broker_name"), _safe_str(execution_mode)), item)
            normalized["micro_order_id"] = micro_order_id
            normalized["broker_order_id"] = broker_order_id
            normalized["symbol"] = row.get("symbol")
            fills.append(normalized)
    return statuses, fills


def _build_internal_cash_reference() -> dict[str, object]:
    value = _safe_float(_text("US_MICRO_RECON_INTERNAL_CASH_USD"))
    return {"cash_balance": value} if value is not None else {}


def _results_with_ids(
    results: list[dict[str, object]],
    *,
    recon_run_id: str,
    recon_date: date,
    account_id: str,
    execution_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, result in enumerate(results, start=1):
        token = _safe_str(result.get("symbol") or result.get("micro_order_id") or result.get("broker_order_id") or "ITEM").replace(" ", "_")
        rows.append(
            {
                "recon_id": _build_recon_id(recon_run_id, _safe_str(result.get("recon_type"), "SUMMARY"), token, idx),
                "recon_run_id": recon_run_id,
                "recon_date": recon_date,
                "account_id": account_id,
                "execution_mode": execution_mode,
                "recon_type": result.get("recon_type"),
                "symbol": result.get("symbol"),
                "micro_order_id": result.get("micro_order_id"),
                "broker_order_id": result.get("broker_order_id"),
                "internal_qty": result.get("internal_qty"),
                "broker_qty": result.get("broker_qty"),
                "qty_diff": result.get("qty_diff"),
                "internal_amount_usd": result.get("internal_amount_usd"),
                "broker_amount_usd": result.get("broker_amount_usd"),
                "amount_diff_usd": result.get("amount_diff_usd"),
                "internal_cash_usd": result.get("internal_cash_usd"),
                "broker_cash_usd": result.get("broker_cash_usd"),
                "cash_diff_usd": result.get("cash_diff_usd"),
                "internal_status": result.get("internal_status"),
                "broker_status": result.get("broker_status"),
                "recon_status": result.get("recon_status"),
                "severity": result.get("severity"),
                "reason_code": result.get("reason_code"),
                "reason_detail": result.get("reason_detail"),
                "raw_internal_payload": result.get("raw_internal_payload"),
                "raw_broker_payload": result.get("raw_broker_payload"),
                "created_at": datetime.now(timezone.utc),
            }
        )
    return rows


def _kill_switch_reason(results: list[dict[str, object]]) -> tuple[str, str]:
    for row in results:
        if _safe_str(row.get("severity")).upper() != "CRITICAL":
            continue
        recon_type = _safe_str(row.get("recon_type")).upper()
        symbol = _safe_str(row.get("symbol"))
        if recon_type == "POSITION":
            return "reconciliation_position_mismatch", f"Critical position mismatch detected{f' for {symbol}' if symbol else ''}."
        if recon_type == "CASH":
            return "reconciliation_cash_mismatch", "Critical cash mismatch detected."
        if recon_type == "FILL":
            return "reconciliation_fill_mismatch", f"Critical fill mismatch detected{f' for {symbol}' if symbol else ''}."
        if recon_type == "ORDER_STATUS":
            return "reconciliation_order_status_mismatch", f"Critical order-status mismatch detected{f' for {symbol}' if symbol else ''}."
    return "reconciliation_critical", "Critical reconciliation mismatch detected."


def run_micro_reconciliation(
    account_id: str,
    recon_date: str,
    execution_mode: str = "MOCK",
    include_orders: bool = True,
    include_fills: bool = True,
    include_positions: bool = True,
    include_cash: bool = False,
    dry_run: bool = False,
    trigger_kill_on_critical: bool = False,
) -> dict:
    execution_mode = _safe_str(execution_mode, "MOCK").upper()
    _assert_reconciliation_enabled(execution_mode)
    ensure_us_micro_live_tables()
    recon_date_value = date.fromisoformat(str(recon_date)[:10])
    recon_run_id = _build_recon_run_id(account_id, str(recon_date))
    event_rows = [
        _build_event_row(
            recon_run_id=recon_run_id,
            event_type="RECON_START",
            account_id=account_id,
            execution_mode=execution_mode,
            message=f"Reconciliation started for account={account_id}, recon_date={recon_date_value.isoformat()}",
            severity="INFO",
        )
    ]
    try:
        internal_orders = _fetch_internal_orders(account_id, recon_date_value, execution_mode)
        internal_fills = _fetch_internal_fills(internal_orders)
        internal_positions = build_internal_expected_positions(account_id, recon_date_value.isoformat(), execution_mode=execution_mode)
        broker_order_statuses, broker_fills = _fetch_broker_order_snapshots(internal_orders, execution_mode)
        account_client = _select_account_client(execution_mode)
        broker_positions = account_client.get_positions(account_id) if include_positions else []
        broker_snapshot = account_client.get_account_snapshot(account_id) if include_cash else {}
        broker_cash = account_client.get_cash_balance(account_id) if include_cash else {}
        internal_cash = _build_internal_cash_reference() if include_cash else {}

        raw_results: list[dict[str, object]] = []
        if include_orders:
            raw_results.extend(compare_order_statuses(internal_orders, broker_order_statuses))
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="ORDER_STATUS_CHECK", account_id=account_id, execution_mode=execution_mode, message=f"Checked {len(internal_orders)} internal orders.", severity="INFO"))
        if include_fills:
            raw_results.extend(compare_fills(internal_fills, broker_fills))
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="FILL_CHECK", account_id=account_id, execution_mode=execution_mode, message=f"Checked {len(internal_fills)} internal fills against {len(broker_fills)} broker fills.", severity="INFO"))
        if include_positions:
            raw_results.extend(compare_positions(internal_positions, broker_positions))
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="POSITION_CHECK", account_id=account_id, execution_mode=execution_mode, message=f"Checked {len(internal_positions)} internal positions against {len(broker_positions)} broker positions.", severity="INFO"))
        if include_cash:
            raw_results.extend(compare_cash(internal_cash, broker_cash))
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="CASH_CHECK", account_id=account_id, execution_mode=execution_mode, message="Checked cash balance consistency.", severity="INFO"))
            if broker_snapshot and _safe_float(broker_snapshot.get("equity_value")) is not None:
                raw_results.append(
                    {
                        "recon_type": "ACCOUNT_EQUITY",
                        "symbol": None,
                        "micro_order_id": None,
                        "broker_order_id": None,
                        "internal_amount_usd": None,
                        "broker_amount_usd": _safe_float(broker_snapshot.get("equity_value")),
                        "amount_diff_usd": None,
                        "recon_status": "UNKNOWN",
                        "severity": "WARNING",
                        "reason_code": "account_snapshot_missing",
                        "reason_detail": "Internal account-equity reference is not available in Phase 7-5.",
                        "raw_internal_payload": None,
                        "raw_broker_payload": _json_text(broker_snapshot),
                    }
                )

        summary = summarize_reconciliation(raw_results)
        if summary["kill_switch_recommended"]:
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="KILL_SWITCH_RECOMMENDED", account_id=account_id, execution_mode=execution_mode, message="Critical reconciliation mismatch detected. Kill Switch review required.", severity="CRITICAL"))
        if summary["kill_switch_recommended"] and trigger_kill_on_critical and _flag("US_MICRO_RECON_TRIGGER_KILL_ON_CRITICAL", "true"):
            reason_code, reason_detail = _kill_switch_reason(raw_results)
            if not dry_run:
                activate_kill_switch(
                    scope="ACCOUNT",
                    target_value=account_id,
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    performed_by="SYSTEM",
                    trigger_source="MICRO_RECONCILIATION",
                    trigger_ref_id=recon_run_id,
                )
            summary["kill_switch_triggered"] = True
            event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="KILL_SWITCH_TRIGGERED", account_id=account_id, execution_mode=execution_mode, message=f"Kill Switch {'would be triggered' if dry_run else 'triggered'} for account={account_id}.", severity="CRITICAL"))

        results = _results_with_ids(raw_results, recon_run_id=recon_run_id, recon_date=recon_date_value, account_id=account_id, execution_mode=execution_mode)
        results.append(_build_summary_result(recon_run_id=recon_run_id, recon_date=recon_date_value, account_id=account_id, execution_mode=execution_mode, summary=summary))
        event_rows.append(_build_event_row(recon_run_id=recon_run_id, event_type="RECON_COMPLETE", account_id=account_id, execution_mode=execution_mode, message=f"Reconciliation completed with {summary['critical_count']} critical and {summary['mismatch_count']} mismatch results.", severity="INFO" if not summary["critical_count"] else "CRITICAL"))

        if not dry_run:
            upsert_us_micro_reconciliation_result_rows(results)
            insert_us_micro_reconciliation_event_log_rows(event_rows)
        return {
            "recon_run_id": recon_run_id,
            "account_id": account_id,
            "recon_date": recon_date_value.isoformat(),
            "execution_mode": execution_mode,
            "results": results,
            "events": event_rows,
            "summary": summary,
            "dry_run": dry_run,
            "tolerance": {
                "qty": _tolerance_qty(),
                "amount_usd": _tolerance_amount(),
                "cash_usd": _tolerance_cash(),
            },
        }
    except Exception as exc:
        error_event = _build_event_row(
            recon_run_id=recon_run_id,
            event_type="RECON_ERROR",
            account_id=account_id,
            execution_mode=execution_mode,
            message=str(exc),
            severity="ERROR",
        )
        if not dry_run:
            insert_us_micro_reconciliation_event_log_rows(event_rows + [error_event])
        raise


def render_reconciliation_console(report: dict[str, object]) -> str:
    summary = report.get("summary") or {}
    results = report.get("results") or []
    position_rows = [row for row in results if _safe_str(row.get("recon_type")).upper() == "POSITION"]
    lines = [
        "[US Micro Live Reconciliation]",
        f"Run ID: {report.get('recon_run_id')}",
        f"Account: {report.get('account_id')}",
        f"Date: {report.get('recon_date')}",
        f"Execution Mode: {report.get('execution_mode')}",
        "",
        "Summary:",
        f"MATCH: {summary.get('match_count', 0)}",
        f"MISMATCH: {summary.get('mismatch_count', 0)}",
        f"MISSING_INTERNAL: {summary.get('missing_internal_count', 0)}",
        f"MISSING_BROKER: {summary.get('missing_broker_count', 0)}",
        f"ERROR: {summary.get('error_count', 0)}",
        f"CRITICAL: {summary.get('critical_count', 0)}",
        "",
        "[Position Check]",
        "Symbol | Internal Qty | Broker Qty | Diff | Status | Severity | Reason",
    ]
    for row in position_rows:
        lines.append(
            f"{row.get('symbol') or '-'} | {row.get('internal_qty') if row.get('internal_qty') is not None else '-'} | "
            f"{row.get('broker_qty') if row.get('broker_qty') is not None else '-'} | "
            f"{row.get('qty_diff') if row.get('qty_diff') is not None else '-'} | "
            f"{row.get('recon_status')} | {row.get('severity')} | {row.get('reason_code') or ''}"
        )
    if not position_rows:
        lines.append("None")
    lines.extend(
        [
            "",
            "[Action]",
            f"Kill Switch Recommended: {'YES' if summary.get('kill_switch_recommended') else 'NO'}",
            f"Auto Triggered: {'YES' if summary.get('kill_switch_triggered') else 'NO'}",
        ]
    )
    return "\n".join(lines)


def render_reconciliation_markdown(report: dict[str, object]) -> str:
    summary = report.get("summary") or {}
    results = [row for row in (report.get("results") or []) if _safe_str(row.get("recon_type")).upper() != "SUMMARY"]
    lines = [
        "# US Micro Live Reconciliation",
        "",
        f"- Run ID: `{report.get('recon_run_id')}`",
        f"- Account: `{report.get('account_id')}`",
        f"- Date: `{report.get('recon_date')}`",
        f"- Execution Mode: `{report.get('execution_mode')}`",
        f"- Dry Run: `{report.get('dry_run')}`",
        "",
        "## Summary",
        "",
        f"- MATCH: `{summary.get('match_count', 0)}`",
        f"- MISMATCH: `{summary.get('mismatch_count', 0)}`",
        f"- MISSING_INTERNAL: `{summary.get('missing_internal_count', 0)}`",
        f"- MISSING_BROKER: `{summary.get('missing_broker_count', 0)}`",
        f"- UNKNOWN: `{summary.get('unknown_count', 0)}`",
        f"- ERROR: `{summary.get('error_count', 0)}`",
        f"- CRITICAL: `{summary.get('critical_count', 0)}`",
        "",
        "## Results",
        "",
        "| Type | Symbol | Internal Qty | Broker Qty | Cash Diff | Status | Severity | Reason |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row.get('recon_type') or '-'} | {row.get('symbol') or '-'} | "
            f"{row.get('internal_qty') if row.get('internal_qty') is not None else '-'} | "
            f"{row.get('broker_qty') if row.get('broker_qty') is not None else '-'} | "
            f"{row.get('cash_diff_usd') if row.get('cash_diff_usd') is not None else '-'} | "
            f"{row.get('recon_status') or '-'} | {row.get('severity') or '-'} | {row.get('reason_code') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Action",
            "",
            f"- Kill Switch Recommended: `{'YES' if summary.get('kill_switch_recommended') else 'NO'}`",
            f"- Auto Triggered: `{'YES' if summary.get('kill_switch_triggered') else 'NO'}`",
        ]
    )
    return "\n".join(lines)


def write_reconciliation_markdown(report: dict[str, object]) -> Path:
    output_dir = Path(_text("US_MICRO_RECON_REPORT_OUTPUT_DIR", "output/us_stock_micro_live"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"recon_{report.get('account_id')}_{report.get('recon_date')}.md"
    path.write_text(render_reconciliation_markdown(report), encoding="utf-8")
    return path
