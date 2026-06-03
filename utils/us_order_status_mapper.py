from __future__ import annotations

from datetime import date, datetime


STANDARD_ORDER_STATUSES = {
    "ORDER_OPEN",
    "ORDER_PARTIALLY_FILLED",
    "ORDER_FILLED",
    "ORDER_CANCELED",
    "ORDER_REJECTED",
    "ORDER_EXPIRED",
    "ORDER_UNKNOWN",
    "SYNC_ERROR",
}


def _safe_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def map_broker_order_status(broker_name: str, raw_status: str, raw_payload: dict | None = None) -> str:
    broker = _safe_str(broker_name, "UNKNOWN").upper()
    status = _safe_str(raw_status, "UNKNOWN").strip().lower()
    payload = raw_payload or {}

    if status in {"new", "accepted", "pending_new", "open", "working", "submitted"}:
        return "ORDER_OPEN"
    if status in {"partially_filled", "partial_fill", "partial-filled"}:
        return "ORDER_PARTIALLY_FILLED"
    if status in {"filled", "fill", "done_for_day"}:
        filled_qty = _safe_float(payload.get("filled_qty"))
        order_qty = _safe_float(payload.get("order_qty")) or _safe_float(payload.get("qty"))
        if filled_qty is not None and order_qty is not None and filled_qty < order_qty:
            return "ORDER_PARTIALLY_FILLED"
        return "ORDER_FILLED"
    if status in {"canceled", "cancelled"}:
        return "ORDER_CANCELED"
    if status in {"rejected"}:
        return "ORDER_REJECTED"
    if status in {"expired"}:
        return "ORDER_EXPIRED"
    if status in {"failed", "error"}:
        return "SYNC_ERROR"

    # Placeholder broker-specific branch point.
    if broker in {"ALPACA", "IBKR", "PAPER", "MOCK_BROKER", "MOCK", "SANDBOX", "NONE"}:
        return "ORDER_UNKNOWN"
    return "ORDER_UNKNOWN"


def is_terminal_order_status(status: str) -> bool:
    return _safe_str(status).upper() in {
        "ORDER_FILLED",
        "ORDER_CANCELED",
        "ORDER_REJECTED",
        "ORDER_EXPIRED",
        "SYNC_ERROR",
    }


def is_fill_status(status: str) -> bool:
    return _safe_str(status).upper() in {"ORDER_PARTIALLY_FILLED", "ORDER_FILLED"}


def normalize_fill_payload(broker_name: str, raw_fill: dict) -> dict:
    broker = _safe_str(broker_name, "UNKNOWN").upper()
    fill_time = raw_fill.get("fill_time") or raw_fill.get("filled_at") or raw_fill.get("timestamp")
    if isinstance(fill_time, str) and fill_time.endswith("Z"):
        fill_time = fill_time.replace("Z", "+00:00")
    parsed_time = None
    if isinstance(fill_time, datetime):
        parsed_time = fill_time
    elif isinstance(fill_time, str) and fill_time:
        try:
            parsed_time = datetime.fromisoformat(fill_time)
        except ValueError:
            parsed_time = None
    filled_qty = _safe_float(raw_fill.get("filled_qty") or raw_fill.get("qty"))
    filled_price = _safe_float(raw_fill.get("filled_price") or raw_fill.get("price"))
    filled_amount = _safe_float(raw_fill.get("filled_amount_usd") or raw_fill.get("amount"))
    if filled_amount is None and filled_qty is not None and filled_price is not None:
        filled_amount = round(filled_qty * filled_price, 6)
    return {
        "broker_name": broker,
        "broker_fill_id": raw_fill.get("broker_fill_id") or raw_fill.get("fill_id") or raw_fill.get("id"),
        "filled_qty": filled_qty,
        "filled_price": filled_price,
        "filled_amount_usd": filled_amount,
        "commission_usd": _safe_float(raw_fill.get("commission_usd") or raw_fill.get("commission")) or 0,
        "fee_usd": _safe_float(raw_fill.get("fee_usd") or raw_fill.get("fee")) or 0,
        "fill_time": parsed_time,
        "fill_date": parsed_time.date() if isinstance(parsed_time, datetime) else (date.fromisoformat(fill_time[:10]) if isinstance(fill_time, str) and len(fill_time) >= 10 else None),
        "liquidity_flag": raw_fill.get("liquidity_flag"),
        "raw_fill_payload": raw_fill,
    }
