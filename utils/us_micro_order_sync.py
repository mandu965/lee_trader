from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os

from python.us.us_db import (
    ensure_us_micro_live_tables,
    fetch_us_micro_order_fill_rows,
    upsert_us_micro_order_fill_rows,
    upsert_us_micro_order_request_rows,
)
from utils.us_live_order_client import UsLiveOrderClient
from utils.us_live_order_safety import mask_sensitive_payload
from utils.us_micro_order_request import (
    get_micro_order,
    list_micro_orders,
    update_micro_order_status,
)
from utils.us_mock_order_client import UsMockOrderClient
from utils.us_order_status_mapper import (
    is_fill_status,
    map_broker_order_status,
    normalize_fill_payload,
)
from utils.us_sandbox_order_client import UsSandboxOrderClient


SYNCABLE_STATUSES = {
    "ACCEPTED",
    "LIVE_ACCEPTED",
    "ORDER_OPEN",
    "ORDER_PARTIALLY_FILLED",
    "LIVE_SENT",
    "SENT",
}


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


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


def _json_text(payload: object | None) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _select_order_client(micro_order: dict[str, object]):
    mode = _safe_str(micro_order.get("execution_mode")).upper()
    if mode == "MOCK":
        return UsMockOrderClient()
    if mode == "SANDBOX":
        return UsSandboxOrderClient()
    if mode == "LIVE":
        if not _flag("US_MICRO_ALLOW_LIVE_STATUS_QUERY", "false"):
            raise RuntimeError("US_MICRO_ALLOW_LIVE_STATUS_QUERY must be true for Live status sync.")
        if _flag("US_MICRO_SYNC_REAL_ORDER_BLOCKED", "true"):
            raise RuntimeError("US_MICRO_SYNC_REAL_ORDER_BLOCKED must be false for Live status sync.")
        return UsLiveOrderClient()
    raise ValueError(f"Unsupported execution_mode for sync: {mode}")


def _assert_sync_enabled(micro_order: dict[str, object]) -> None:
    if not _flag("US_MICRO_ORDER_SYNC_ENABLED", "false"):
        raise RuntimeError("US_MICRO_ORDER_SYNC_ENABLED must be true before broker status sync can run.")
    mode = _safe_str(micro_order.get("execution_mode")).upper()
    if mode == "LIVE":
        if not _flag("US_MICRO_ALLOW_LIVE_STATUS_QUERY", "false"):
            raise RuntimeError("US_MICRO_ALLOW_LIVE_STATUS_QUERY must be true for Live status sync.")
        if _flag("US_MICRO_SYNC_REAL_ORDER_BLOCKED", "true"):
            raise RuntimeError("US_MICRO_SYNC_REAL_ORDER_BLOCKED must be false for Live status sync.")


def _build_fill_id(micro_order_id: str, normalized_fill: dict[str, object]) -> str:
    broker_fill_id = _safe_str(normalized_fill.get("broker_fill_id"))
    if broker_fill_id:
        return f"USFILL_{broker_fill_id}"
    digest_source = "|".join(
        [
            micro_order_id,
            _safe_str(normalized_fill.get("fill_time")),
            _safe_str(normalized_fill.get("filled_qty")),
            _safe_str(normalized_fill.get("filled_price")),
        ]
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"USFILL_{digest}"


def update_micro_order_from_broker_status(
    micro_order: dict[str, object],
    broker_status_response: dict,
    dry_run: bool = False,
) -> dict:
    raw_status = _safe_str(broker_status_response.get("status"), "UNKNOWN")
    internal_status = map_broker_order_status(
        _safe_str(micro_order.get("broker_name"), _safe_str(micro_order.get("execution_mode"))),
        raw_status,
        broker_status_response,
    )
    updated = dict(micro_order)
    updated["request_status"] = internal_status
    updated["last_broker_status"] = raw_status
    updated["last_sync_at"] = datetime.now(timezone.utc)
    updated["sync_status"] = "OK" if internal_status != "ORDER_UNKNOWN" else "UNKNOWN"
    updated["sync_error"] = None
    updated["response_payload"] = _json_text(mask_sensitive_payload(broker_status_response))
    updated["filled_qty"] = broker_status_response.get("filled_qty") if broker_status_response.get("filled_qty") is not None else updated.get("filled_qty")
    updated["avg_filled_price"] = broker_status_response.get("filled_price") if broker_status_response.get("filled_price") is not None else updated.get("avg_filled_price")
    if updated.get("filled_qty") is not None and updated.get("avg_filled_price") is not None:
        updated["filled_amount_usd"] = round(float(updated["filled_qty"]) * float(updated["avg_filled_price"]), 6)
    if updated.get("order_qty") is not None and updated.get("filled_qty") is not None:
        updated["remaining_qty"] = round(float(updated["order_qty"]) - float(updated["filled_qty"]), 6)
    if dry_run:
        return {
            "micro_order": updated,
            "broker_status_response": mask_sensitive_payload(broker_status_response),
            "mapped_status": internal_status,
            "dry_run": True,
        }
    upsert_us_micro_order_request_rows([updated])
    return update_micro_order_status(
        _safe_str(updated.get("micro_order_id")),
        internal_status,
        reason_code="order_status_synced" if internal_status != "ORDER_UNKNOWN" else "order_status_unknown",
        reason_detail=f"Broker status {raw_status} mapped to {internal_status}.",
        response_payload=mask_sensitive_payload(broker_status_response),
        event_type="ORDER_STATUS_SYNCED" if internal_status != "SYNC_ERROR" else "SYNC_ERROR",
        event_source="BROKER_SYNC",
        created_by="SYSTEM",
    )


def insert_micro_order_fills(
    micro_order: dict,
    normalized_fills: list[dict],
    dry_run: bool = False,
) -> dict:
    existing = fetch_us_micro_order_fill_rows(micro_order_id=_safe_str(micro_order.get("micro_order_id")))
    existing_ids = {str(row.get("micro_fill_id")) for row in existing}
    rows: list[dict[str, object]] = []
    inserted = 0
    for fill in normalized_fills:
        row = {
            "micro_fill_id": _build_fill_id(_safe_str(micro_order.get("micro_order_id")), fill),
            "micro_order_id": _safe_str(micro_order.get("micro_order_id")),
            "broker_order_id": micro_order.get("broker_order_id"),
            "broker_fill_id": fill.get("broker_fill_id"),
            "account_id": micro_order.get("account_id"),
            "symbol": micro_order.get("symbol"),
            "side": micro_order.get("side"),
            "filled_qty": fill.get("filled_qty"),
            "filled_price": fill.get("filled_price"),
            "filled_amount_usd": fill.get("filled_amount_usd"),
            "commission_usd": fill.get("commission_usd"),
            "fee_usd": fill.get("fee_usd"),
            "fill_time": fill.get("fill_time"),
            "fill_date": fill.get("fill_date"),
            "liquidity_flag": fill.get("liquidity_flag"),
            "raw_fill_payload": _json_text(mask_sensitive_payload(fill.get("raw_fill_payload") or fill)),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        if row["micro_fill_id"] in existing_ids:
            continue
        existing_ids.add(str(row["micro_fill_id"]))
        rows.append(row)
        inserted += 1
    if dry_run:
        return {"inserted_count": inserted, "rows": rows, "dry_run": True}
    upsert_us_micro_order_fill_rows(rows)
    return {"inserted_count": inserted, "rows": rows, "dry_run": False}


def sync_micro_order_fills(micro_order_id: str, dry_run: bool = False) -> dict:
    ensure_us_micro_live_tables()
    micro_order = get_micro_order(micro_order_id)
    _assert_sync_enabled(micro_order)
    broker_order_id = _safe_str(micro_order.get("broker_order_id"))
    if not broker_order_id:
        raise ValueError("broker_order_id is required to sync fills.")
    client = _select_order_client(micro_order)
    raw_fills = client.get_order_fills(broker_order_id)
    normalized = [
        normalize_fill_payload(_safe_str(micro_order.get("broker_name"), _safe_str(micro_order.get("execution_mode"))), item)
        for item in raw_fills
    ]
    result = insert_micro_order_fills(micro_order, normalized, dry_run=dry_run)
    if dry_run:
        return {"micro_order": micro_order, "fills": normalized, "insert_result": result, "dry_run": True}
    update_micro_order_status(
        micro_order_id,
        _safe_str(get_micro_order(micro_order_id).get("request_status")),
        reason_code="fills_synced",
        reason_detail=f"Synced {result['inserted_count']} fill rows.",
        response_payload={"fills": [mask_sensitive_payload(item) for item in normalized]},
        event_type="ORDER_FILL_SYNCED",
        event_source="BROKER_SYNC",
        created_by="SYSTEM",
    )
    return {"micro_order": get_micro_order(micro_order_id), "fills": normalized, "insert_result": result}


def sync_micro_order_status(micro_order_id: str, dry_run: bool = False) -> dict:
    ensure_us_micro_live_tables()
    micro_order = get_micro_order(micro_order_id)
    _assert_sync_enabled(micro_order)
    broker_order_id = _safe_str(micro_order.get("broker_order_id"))
    if not broker_order_id:
        raise ValueError("broker_order_id is required to sync order status.")
    client = _select_order_client(micro_order)
    response = client.get_order_status(broker_order_id)
    result = update_micro_order_from_broker_status(micro_order, response, dry_run=dry_run)
    mapped_status = _safe_str(result.get("mapped_status") if dry_run else result.get("request_status"))
    include_fills = _flag("US_MICRO_SYNC_INCLUDE_FILLS", "true")
    if not dry_run and include_fills and is_fill_status(mapped_status):
        fill_result = sync_micro_order_fills(micro_order_id, dry_run=False)
        result = {"micro_order": get_micro_order(micro_order_id), "status_result": result, "fill_result": fill_result}
    return result


def sync_micro_orders_by_status(
    account_id: str | None = None,
    trade_date: str | None = None,
    statuses: list[str] | None = None,
    dry_run: bool = False,
    include_fills: bool = False,
) -> list[dict]:
    ensure_us_micro_live_tables()
    target_statuses = statuses or sorted(SYNCABLE_STATUSES)
    rows: list[dict] = []
    for status in target_statuses:
        rows.extend(list_micro_orders(account_id=account_id, trade_date=trade_date, status=status))
    seen: set[str] = set()
    results: list[dict] = []
    for row in rows:
        micro_order_id = _safe_str(row.get("micro_order_id"))
        if micro_order_id in seen:
            continue
        seen.add(micro_order_id)
        result = sync_micro_order_status(micro_order_id, dry_run=dry_run)
        if include_fills and not dry_run:
            current = get_micro_order(micro_order_id)
            if is_fill_status(_safe_str(current.get("request_status"))):
                result = {
                    "status_result": result,
                    "fill_result": sync_micro_order_fills(micro_order_id, dry_run=False),
                }
        results.append(result)
    return results
