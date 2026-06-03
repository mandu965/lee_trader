from __future__ import annotations

"""Phase 7-3 micro-order orchestration for Mock, Sandbox, and gated Live review.

Micro Live exists to validate real-order connectivity under strict controls.
It is not full auto-trading, and Live execution remains blocked by default.
LIVE_ACCEPTED does not mean a fill or completed real-account execution.
"""

from datetime import date, datetime, timezone
import json
import os
from typing import Any

from python.us.us_db import (
    ensure_us_micro_live_tables,
    fetch_us_micro_order_event_log_rows,
    fetch_us_micro_order_request_rows,
    insert_us_micro_order_event_log_rows,
    upsert_us_micro_order_request_rows,
)
from utils.us_live_order_approval import get_order_approval, validate_approval_for_candidate
from utils.us_live_order_client import UsLiveOrderClient
from utils.us_live_order_safety import (
    LIVE_FINAL_CONFIRMATION_TEXT,
    build_live_order_payload,
    live_gate_snapshot,
    mask_sensitive_payload,
    validate_live_order_execution_allowed,
)
from utils.us_live_pre_trade_check import UsLiveOrderCandidate, run_us_live_pre_trade_check
from utils.us_micro_live_safety import (
    assert_sandbox_enabled,
    assert_us_micro_mock_only,
    sandbox_config_snapshot,
)
from utils.us_mock_order_client import UsMockOrderClient
from utils.us_sandbox_order_client import UsSandboxOrderClient


MICRO_ORDER_STATUSES = {
    "CREATED",
    "PRECHECK_FAILED",
    "APPROVAL_INVALID",
    "READY_TO_SEND",
    "SENT",
    "ACCEPTED",
    "REJECTED",
    "FAILED",
    "CANCELED",
    "ERROR",
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
}
MOCK_SANDBOX_SENDABLE_STATUSES = {"CREATED", "READY_TO_SEND"}
LIVE_SENDABLE_STATUSES = {"CREATED", "LIVE_BLOCKED", "LIVE_READY", "LIVE_CONFIRMATION_REQUIRED"}
CANCELABLE_SANDBOX_STATUSES = {"SENT", "ACCEPTED"}
VALID_EXECUTION_MODES = {"MOCK", "SANDBOX", "LIVE"}


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


def _trade_date_value(value: object) -> date | None:
    text = _safe_str(value)
    if not text:
        return None
    return date.fromisoformat(text[:10])


def _json_text(payload: object | None) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _json_obj(payload: object | None) -> dict[str, object]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        return data if isinstance(data, dict) else {"value": data}
    return {"value": payload}


def _reason_codes_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return json.dumps([], ensure_ascii=False)


def _parse_reason_codes(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]
    return []


def _approval_row_to_candidate(row: dict[str, object]) -> UsLiveOrderCandidate:
    return UsLiveOrderCandidate(
        trade_date=_safe_str(row.get("trade_date")),
        account_id=_safe_str(row.get("account_id")),
        policy_id=_safe_str(row.get("policy_id")),
        symbol=_safe_str(row.get("symbol")).upper(),
        side=_safe_str(row.get("side")).upper(),
        requested_order_amount_usd=_safe_float(row.get("requested_order_amount_usd")),
        requested_qty=_safe_float(row.get("requested_qty")),
        requested_order_type=_safe_str(row.get("requested_order_type"), "LIMIT").upper(),
        requested_limit_price=_safe_float(row.get("requested_limit_price")),
        candidate_source=_safe_str(row.get("candidate_source"), "APPROVAL"),
        strategy_name=_safe_str(row.get("strategy_name")) or None,
        rank_no=int(row.get("rank_no")) if row.get("rank_no") is not None else None,
        recommend_grade=_safe_str(row.get("recommend_grade")) or None,
        total_score=_safe_float(row.get("total_score")),
        reason=_safe_str(row.get("precheck_summary")) or None,
    )


def _micro_row_to_candidate(row: dict[str, object]) -> UsLiveOrderCandidate:
    return UsLiveOrderCandidate(
        trade_date=_safe_str(row.get("trade_date")),
        account_id=_safe_str(row.get("account_id")),
        policy_id=_safe_str(row.get("policy_id")),
        symbol=_safe_str(row.get("symbol")).upper(),
        side=_safe_str(row.get("side")).upper(),
        requested_order_amount_usd=_safe_float(row.get("order_amount_usd")),
        requested_qty=_safe_float(row.get("order_qty")),
        requested_order_type=_safe_str(row.get("order_type"), "LIMIT").upper(),
        requested_limit_price=_safe_float(row.get("limit_price")),
        candidate_source=_safe_str(row.get("candidate_source"), "MICRO"),
        strategy_name=_safe_str(row.get("strategy_name")) or None,
        rank_no=int(row.get("rank_no")) if row.get("rank_no") is not None else None,
        recommend_grade=_safe_str(row.get("recommend_grade")) or None,
        total_score=_safe_float(row.get("total_score")),
        reason=_safe_str(row.get("precheck_summary")) or None,
    )


def build_micro_order_id(approval_row: dict[str, object], now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    trade_date = _safe_str(approval_row.get("trade_date")).replace("-", "")
    account_id = _safe_str(approval_row.get("account_id")).upper()
    symbol = _safe_str(approval_row.get("symbol")).upper()
    side = _safe_str(approval_row.get("side")).upper()
    return f"USMO_{trade_date}_{account_id}_{symbol}_{side}_{now.strftime('%Y%m%d%H%M%S%f')}"


def _build_event_row(
    *,
    micro_order_id: str,
    event_type: str,
    before_status: str | None,
    after_status: str | None,
    event_source: str,
    reason_code: str | None,
    reason_detail: str | None,
    request_payload: object | None,
    response_payload: object | None,
    created_by: str,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return {
        "event_id": f"USMOEVT_{micro_order_id}_{event_type}_{timestamp}",
        "micro_order_id": micro_order_id,
        "event_type": event_type,
        "before_status": before_status,
        "after_status": after_status,
        "event_source": event_source,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "request_payload": _json_text(mask_sensitive_payload(_json_obj(request_payload))),
        "response_payload": _json_text(mask_sensitive_payload(_json_obj(response_payload))),
        "created_by": created_by,
    }


def _status_event_type(status: str, execution_mode: str) -> str:
    mode = execution_mode.upper()
    if mode == "SANDBOX":
        return {
            "PRECHECK_FAILED": "PRECHECK_FAIL",
            "APPROVAL_INVALID": "APPROVAL_INVALID",
            "READY_TO_SEND": "APPROVAL_VALID",
            "SENT": "SEND_SANDBOX",
            "ACCEPTED": "SANDBOX_ACCEPTED",
            "REJECTED": "SANDBOX_REJECTED",
            "FAILED": "SANDBOX_FAILED",
            "CANCELED": "SANDBOX_CANCELED",
            "ERROR": "ERROR",
        }.get(status, "ERROR")
    if mode == "LIVE":
        return {
            "LIVE_BLOCKED": "LIVE_BLOCKED",
            "LIVE_READY": "LIVE_READY",
            "LIVE_CONFIRMATION_REQUIRED": "LIVE_CONFIRMATION_REQUIRED",
            "LIVE_SENT": "SEND_LIVE",
            "LIVE_ACCEPTED": "LIVE_ACCEPTED",
            "LIVE_REJECTED": "LIVE_REJECTED",
            "LIVE_FAILED": "LIVE_FAILED",
            "LIVE_CANCELED": "LIVE_CANCELED",
            "APPROVAL_INVALID": "APPROVAL_INVALID",
            "PRECHECK_FAILED": "LIVE_SAFETY_CHECK_FAILED",
            "ERROR": "LIVE_SAFETY_CHECK_FAILED",
            "ORDER_OPEN": "ORDER_STATUS_SYNCED",
            "ORDER_PARTIALLY_FILLED": "ORDER_STATUS_SYNCED",
            "ORDER_FILLED": "ORDER_STATUS_SYNCED",
            "ORDER_CANCELED": "ORDER_STATUS_SYNCED",
            "ORDER_REJECTED": "ORDER_STATUS_SYNCED",
            "ORDER_EXPIRED": "ORDER_STATUS_SYNCED",
            "ORDER_UNKNOWN": "ORDER_STATUS_SYNCED",
            "SYNC_ERROR": "LIVE_SAFETY_CHECK_FAILED",
        }.get(status, "LIVE_SAFETY_CHECK_FAILED")
    return {
        "PRECHECK_FAILED": "PRECHECK_FAIL",
        "APPROVAL_INVALID": "APPROVAL_INVALID",
        "READY_TO_SEND": "APPROVAL_VALID",
        "SENT": "SEND_MOCK",
        "ACCEPTED": "ACCEPTED",
        "REJECTED": "REJECTED",
        "FAILED": "FAILED",
        "CANCELED": "CANCELED",
        "ERROR": "ERROR",
        "ORDER_OPEN": "ORDER_STATUS_SYNCED",
        "ORDER_PARTIALLY_FILLED": "ORDER_STATUS_SYNCED",
        "ORDER_FILLED": "ORDER_STATUS_SYNCED",
        "ORDER_CANCELED": "ORDER_STATUS_SYNCED",
        "ORDER_REJECTED": "ORDER_STATUS_SYNCED",
        "ORDER_EXPIRED": "ORDER_STATUS_SYNCED",
        "ORDER_UNKNOWN": "ORDER_STATUS_SYNCED",
        "SYNC_ERROR": "ERROR",
    }.get(status, "ERROR")


def _status_event_source(status: str, execution_mode: str) -> str:
    mode = execution_mode.upper()
    if mode == "SANDBOX" and status in {"SENT", "ACCEPTED", "REJECTED", "FAILED", "CANCELED"}:
        return "SANDBOX_BROKER"
    if mode == "LIVE" and status in {"LIVE_SENT", "LIVE_ACCEPTED", "LIVE_REJECTED", "LIVE_FAILED", "LIVE_CANCELED"}:
        return "LIVE_BROKER"
    if mode == "LIVE" and status in {"LIVE_BLOCKED", "LIVE_READY", "LIVE_CONFIRMATION_REQUIRED"}:
        return "LIVE_GATE"
    if status in {"ORDER_OPEN", "ORDER_PARTIALLY_FILLED", "ORDER_FILLED", "ORDER_CANCELED", "ORDER_REJECTED", "ORDER_EXPIRED", "ORDER_UNKNOWN", "SYNC_ERROR"}:
        return "BROKER_SYNC"
    if status == "PRECHECK_FAILED":
        return "PRE_TRADE_CHECK"
    if status == "APPROVAL_INVALID":
        return "APPROVAL"
    if status == "CANCELED":
        return "OPERATOR"
    if mode == "MOCK" and status in {"SENT", "ACCEPTED", "REJECTED", "FAILED"}:
        return "MOCK_BROKER"
    return "SYSTEM"


def _build_request_payload(approval_row: dict[str, object], precheck_result, *, execution_mode: str) -> dict[str, object]:
    return {
        "phase": "7-3",
        "note": "Micro Live exists to validate order-connectivity safety under strict gates. It is not full auto-trading.",
        "execution_mode": execution_mode,
        "approval": {
            "approval_id": approval_row.get("approval_id"),
            "approval_status": approval_row.get("approval_status"),
            "approved_by": approval_row.get("approved_by"),
            "approved_at": approval_row.get("approved_at"),
            "expires_at": approval_row.get("expires_at"),
        },
        "precheck": {
            "decision": precheck_result.decision,
            "reason_codes": list(precheck_result.reason_codes),
            "reason_details": list(precheck_result.reason_details),
            "check_results": dict(precheck_result.check_results),
        },
    }


def _resolve_execution_mode(requested: str | None) -> str:
    mode = _safe_str(requested or os.environ.get("US_MICRO_EXECUTION_MODE"), "MOCK").upper()
    if mode not in VALID_EXECUTION_MODES:
        raise ValueError(f"Unsupported execution_mode: {mode}")
    return mode


def _resolve_broker_name(execution_mode: str) -> str:
    mode = execution_mode.upper()
    if mode == "MOCK":
        return "MOCK_BROKER"
    if mode == "SANDBOX":
        return _safe_str(os.environ.get("US_SANDBOX_BROKER_NAME"), "NONE").upper()
    if mode == "LIVE":
        return _safe_str(os.environ.get("US_LIVE_BROKER"), "NONE").upper()
    return "NONE"


def validate_micro_order_request(micro_order: dict[str, object]) -> list[str]:
    issues: list[str] = []
    required_text_fields = ["micro_order_id", "policy_id", "account_id", "symbol", "side", "order_type", "execution_mode", "request_status"]
    for field in required_text_fields:
        if not _safe_str(micro_order.get(field)):
            issues.append(f"{field}_missing")
    mode = _safe_str(micro_order.get("execution_mode")).upper()
    if mode not in VALID_EXECUTION_MODES:
        issues.append("execution_mode_invalid")
    if _safe_str(micro_order.get("request_status")).upper() not in MICRO_ORDER_STATUSES:
        issues.append("request_status_invalid")
    order_qty = _safe_float(micro_order.get("order_qty"))
    order_amount = _safe_float(micro_order.get("order_amount_usd"))
    if order_qty is not None and order_qty <= 0:
        issues.append("order_qty_non_positive")
    if order_amount is not None and order_amount <= 0:
        issues.append("order_amount_non_positive")
    limit_price = _safe_float(micro_order.get("limit_price"))
    if limit_price is not None and limit_price <= 0:
        issues.append("limit_price_non_positive")
    if mode == "LIVE" and _safe_str(micro_order.get("order_type"), "LIMIT").upper() != "LIMIT":
        issues.append("market_order_not_allowed")
    return issues


def _sandbox_limits_ok(row: dict[str, object]) -> tuple[bool, str | None, str | None]:
    order_amount = _safe_float(row.get("order_amount_usd")) or 0.0
    max_amount = _safe_float(os.environ.get("US_SANDBOX_MAX_ORDER_AMOUNT_USD", "50")) or 50.0
    if order_amount > max_amount:
        return False, "sandbox_order_amount_exceeded", f"order_amount_usd={order_amount} exceeds US_SANDBOX_MAX_ORDER_AMOUNT_USD={max_amount}"
    order_type = _safe_str(row.get("order_type"), "LIMIT").upper()
    if order_type == "MARKET" and not _flag("US_SANDBOX_ALLOW_MARKET_ORDER", "false"):
        return False, "sandbox_market_order_blocked", "US_SANDBOX_ALLOW_MARKET_ORDER=false"
    if order_type not in {"LIMIT", "MARKET"}:
        return False, "sandbox_order_type_invalid", f"Unsupported order_type={order_type}"
    return True, None, None


def _live_create_validation(row: dict[str, object], *, allow_live_create: bool) -> tuple[str | None, str | None]:
    if not allow_live_create:
        return "live_mode_not_allowed", "--allow-live-create is required for execution_mode=LIVE."
    if not _flag("US_LIVE_REQUIRE_MANUAL_APPROVAL", "true"):
        return "approval_invalid", "US_LIVE_REQUIRE_MANUAL_APPROVAL must remain true for Micro Live."
    if _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"):
        return "real_order_blocked", "US_MICRO_REAL_ORDER_BLOCKED=true keeps Live creation in blocked mode."
    if not _flag("US_MICRO_ALLOW_LIVE", "false"):
        return "live_mode_not_allowed", "US_MICRO_ALLOW_LIVE=true is required before Live-create can proceed."
    if _safe_str(row.get("order_type"), "LIMIT").upper() != "LIMIT":
        return "market_order_not_allowed", "Micro Live only allows LIMIT orders."
    order_amount = _safe_float(row.get("order_amount_usd")) or 0.0
    max_amount = _safe_float(os.environ.get("US_LIVE_MAX_ORDER_AMOUNT_USD", "50")) or 50.0
    if order_amount > max_amount:
        return "order_amount_exceeded", f"order_amount_usd={order_amount} exceeds US_LIVE_MAX_ORDER_AMOUNT_USD={max_amount}"
    side = _safe_str(row.get("side")).upper()
    if side == "SELL":
        return "live_sell_disabled", "SELL automation stays disabled in Phase 7-3."
    return None, None


def _create_micro_order_row(
    approval_row: dict[str, object],
    *,
    micro_order_id: str,
    execution_mode: str,
    created_by: str,
    precheck_result,
    request_status: str,
    response_payload: object | None = None,
    reject_reason_code: str | None = None,
    reject_reason_detail: str | None = None,
) -> dict[str, object]:
    default_order_type = "LIMIT"
    if execution_mode == "SANDBOX":
        default_order_type = _safe_str(os.environ.get("US_SANDBOX_DEFAULT_ORDER_TYPE"), "LIMIT").upper()
    elif execution_mode == "LIVE":
        default_order_type = _safe_str(os.environ.get("US_LIVE_DEFAULT_ORDER_TYPE"), "LIMIT").upper()
    return {
        "micro_order_id": micro_order_id,
        "approval_id": approval_row.get("approval_id"),
        "policy_id": approval_row.get("policy_id"),
        "account_id": approval_row.get("account_id"),
        "trade_date": _trade_date_value(approval_row.get("trade_date")),
        "symbol": _safe_str(approval_row.get("symbol")).upper(),
        "side": _safe_str(approval_row.get("side")).upper(),
        "order_type": _safe_str(approval_row.get("requested_order_type"), default_order_type).upper(),
        "limit_price": _safe_float(approval_row.get("requested_limit_price")),
        "order_qty": _safe_float(approval_row.get("requested_qty")),
        "order_amount_usd": _safe_float(approval_row.get("requested_order_amount_usd")),
        "candidate_source": approval_row.get("candidate_source"),
        "strategy_name": approval_row.get("strategy_name"),
        "rank_no": approval_row.get("rank_no"),
        "recommend_grade": approval_row.get("recommend_grade"),
        "total_score": approval_row.get("total_score"),
        "precheck_decision": precheck_result.decision,
        "precheck_reason_codes": _reason_codes_text(precheck_result.reason_codes),
        "precheck_summary": "; ".join(precheck_result.reason_details or []),
        "execution_mode": execution_mode,
        "broker_name": _resolve_broker_name(execution_mode),
        "request_status": request_status,
        "request_payload": _json_text(mask_sensitive_payload(_build_request_payload(approval_row, precheck_result, execution_mode=execution_mode))),
        "response_payload": _json_text(mask_sensitive_payload(response_payload)),
        "broker_order_id": None,
        "last_broker_status": None,
        "last_sync_at": None,
        "filled_qty": None,
        "remaining_qty": None,
        "avg_filled_price": None,
        "filled_amount_usd": None,
        "sync_status": None,
        "sync_error": None,
        "reject_reason_code": reject_reason_code,
        "reject_reason_detail": reject_reason_detail,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def get_micro_order(micro_order_id: str) -> dict[str, object]:
    rows = fetch_us_micro_order_request_rows(micro_order_id=micro_order_id)
    if not rows:
        raise ValueError(f"Micro order not found: {micro_order_id}")
    return rows[0]


def get_micro_order_events(micro_order_id: str) -> list[dict[str, object]]:
    return fetch_us_micro_order_event_log_rows(micro_order_id=micro_order_id)


def list_micro_orders(
    *,
    trade_date: str | None = None,
    account_id: str | None = None,
    status: str | None = None,
    execution_mode: str | None = None,
    approval_id: str | None = None,
) -> list[dict[str, object]]:
    return fetch_us_micro_order_request_rows(
        trade_date=_trade_date_value(trade_date),
        account_id=account_id,
        status=status.upper() if status else None,
        execution_mode=execution_mode.upper() if execution_mode else None,
        approval_id=approval_id,
    )


def update_micro_order_status(
    micro_order_id: str,
    new_status: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    response_payload: dict[str, object] | None = None,
    *,
    event_type: str | None = None,
    event_source: str | None = None,
    created_by: str = "SYSTEM",
    request_payload: dict[str, object] | None = None,
    broker_order_id: str | None = None,
) -> dict[str, object]:
    ensure_us_micro_live_tables()
    row = get_micro_order(micro_order_id)
    target_status = _safe_str(new_status).upper()
    if target_status not in MICRO_ORDER_STATUSES:
        raise ValueError(f"Unsupported micro order status: {new_status}")
    before_status = _safe_str(row.get("request_status")).upper()
    updated = dict(row)
    updated.setdefault("last_broker_status", None)
    updated.setdefault("last_sync_at", None)
    updated.setdefault("filled_qty", None)
    updated.setdefault("remaining_qty", None)
    updated.setdefault("avg_filled_price", None)
    updated.setdefault("filled_amount_usd", None)
    updated.setdefault("sync_status", None)
    updated.setdefault("sync_error", None)
    updated["request_status"] = target_status
    updated["updated_at"] = datetime.now(timezone.utc)
    if reason_code is not None:
        updated["reject_reason_code"] = reason_code
    if reason_detail is not None:
        updated["reject_reason_detail"] = reason_detail
    if request_payload is not None:
        updated["request_payload"] = _json_text(mask_sensitive_payload(request_payload))
    if response_payload is not None:
        updated["response_payload"] = _json_text(mask_sensitive_payload(response_payload))
    if broker_order_id is not None:
        updated["broker_order_id"] = broker_order_id
    elif isinstance(response_payload, dict) and response_payload.get("broker_order_id"):
        updated["broker_order_id"] = response_payload.get("broker_order_id")
    upsert_us_micro_order_request_rows([updated])
    execution_mode = _safe_str(updated.get("execution_mode"), "MOCK")
    insert_us_micro_order_event_log_rows(
        [
            _build_event_row(
                micro_order_id=micro_order_id,
                event_type=event_type or _status_event_type(target_status, execution_mode),
                before_status=before_status,
                after_status=target_status,
                event_source=event_source or _status_event_source(target_status, execution_mode),
                reason_code=reason_code,
                reason_detail=reason_detail,
                request_payload=request_payload,
                response_payload=response_payload,
                created_by=created_by,
            )
        ]
    )
    return get_micro_order(micro_order_id)


def _persist_initial_micro_order(row: dict[str, object], created_by: str) -> None:
    upsert_us_micro_order_request_rows([row])
    insert_us_micro_order_event_log_rows(
        [
            _build_event_row(
                micro_order_id=_safe_str(row.get("micro_order_id")),
                event_type="CREATE_REQUEST",
                before_status=None,
                after_status="CREATED",
                event_source="SYSTEM",
                reason_code=None,
                reason_detail="Micro order request created.",
                request_payload=_json_obj(row.get("request_payload")),
                response_payload=_json_obj(row.get("response_payload")),
                created_by=created_by,
            )
        ]
    )


def _precheck_failure_status(precheck_result) -> str:
    if precheck_result.decision == "BLOCK":
        return "PRECHECK_FAILED"
    if precheck_result.decision == "ERROR":
        return "ERROR"
    return "READY_TO_SEND"


def _cancel_replaceable_micro_orders(existing_rows: list[dict[str, object]], created_by: str) -> None:
    replaceable = {
        "CREATED",
        "READY_TO_SEND",
        "LIVE_BLOCKED",
        "LIVE_READY",
        "LIVE_CONFIRMATION_REQUIRED",
    }
    for existing in existing_rows:
        existing_status = _safe_str(existing.get("request_status")).upper()
        if existing_status in replaceable:
            update_micro_order_status(
                _safe_str(existing.get("micro_order_id")),
                "CANCELED",
                reason_code="replaced_existing_micro_order",
                reason_detail="Replaced by newer micro order request.",
                created_by=created_by,
            )


def create_micro_order_from_approval(
    approval_id: str,
    execution_mode: str = "MOCK",
    created_by: str = "SYSTEM",
    dry_run: bool = False,
    replace_existing: bool = False,
    allow_live_create: bool = False,
) -> dict[str, object]:
    mode = _resolve_execution_mode(execution_mode)
    if mode in {"MOCK", "SANDBOX"}:
        assert_us_micro_mock_only()
    ensure_us_micro_live_tables()
    existing_rows = fetch_us_micro_order_request_rows(approval_id=approval_id, execution_mode=mode)
    if existing_rows and not replace_existing:
        return existing_rows[0]

    approval_row = get_order_approval(approval_id)
    candidate = _approval_row_to_candidate(approval_row)
    approval_validation = validate_approval_for_candidate(candidate, approval_id)
    precheck_result = run_us_live_pre_trade_check(candidate, write_block_log=not dry_run, log_requires_approval=True)
    micro_order_id = build_micro_order_id(approval_row)
    initial_row = _create_micro_order_row(
        approval_row,
        micro_order_id=micro_order_id,
        execution_mode=mode,
        created_by=created_by,
        precheck_result=precheck_result,
        request_status="CREATED",
        response_payload={
            "approval_validation": approval_validation,
            "precheck_decision": precheck_result.decision,
            "precheck_reason_codes": list(precheck_result.reason_codes),
        },
    )
    validation_errors = validate_micro_order_request(initial_row)
    live_create_code: str | None = None
    live_create_detail: str | None = None

    if mode == "SANDBOX":
        if not _flag("US_MICRO_ALLOW_SANDBOX", "false"):
            validation_errors.append("sandbox_mode_not_allowed")
        limits_ok, limit_code, limit_detail = _sandbox_limits_ok(initial_row)
        if not limits_ok:
            validation_errors.append(limit_code or "sandbox_limit_invalid")
            initial_row["reject_reason_detail"] = limit_detail

    if mode == "LIVE":
        live_create_code, live_create_detail = _live_create_validation(initial_row, allow_live_create=allow_live_create)

    preview = {
        "micro_order": initial_row,
        "approval_validation": approval_validation,
        "precheck_result": {
            "decision": precheck_result.decision,
            "reason_codes": list(precheck_result.reason_codes),
            "reason_details": list(precheck_result.reason_details),
            "check_results": dict(precheck_result.check_results),
        },
        "dry_run": dry_run,
    }
    if dry_run:
        if validation_errors:
            preview["micro_order"]["request_status"] = "ERROR"
        elif precheck_result.decision in {"BLOCK", "ERROR"}:
            preview["micro_order"]["request_status"] = _precheck_failure_status(precheck_result)
        elif not approval_validation.get("valid"):
            preview["micro_order"]["request_status"] = "APPROVAL_INVALID"
        elif mode == "LIVE" and live_create_code:
            preview["micro_order"]["request_status"] = "LIVE_BLOCKED"
            preview["micro_order"]["reject_reason_code"] = live_create_code
            preview["micro_order"]["reject_reason_detail"] = live_create_detail
        elif mode == "LIVE":
            preview["micro_order"]["request_status"] = "LIVE_CONFIRMATION_REQUIRED"
        else:
            preview["micro_order"]["request_status"] = "READY_TO_SEND"
        return preview

    _cancel_replaceable_micro_orders(existing_rows, created_by)
    _persist_initial_micro_order(initial_row, created_by)
    if validation_errors:
        return update_micro_order_status(
            micro_order_id,
            "ERROR",
            reason_code="micro_order_validation_failed",
            reason_detail="; ".join(validation_errors),
            response_payload={"validation_errors": validation_errors},
            created_by=created_by,
        )
    if precheck_result.decision in {"BLOCK", "ERROR"}:
        return update_micro_order_status(
            micro_order_id,
            _precheck_failure_status(precheck_result),
            reason_code="precheck_blocked" if precheck_result.decision == "BLOCK" else "precheck_error",
            reason_detail="; ".join(precheck_result.reason_details or ["Pre-trade check failed."]),
            response_payload={
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
            },
            event_type="PRECHECK_FAIL" if precheck_result.decision == "BLOCK" else "ERROR",
            event_source="PRE_TRADE_CHECK",
            created_by=created_by,
        )
    insert_us_micro_order_event_log_rows(
        [
            _build_event_row(
                micro_order_id=micro_order_id,
                event_type="PRECHECK_PASS",
                before_status="CREATED",
                after_status="CREATED",
                event_source="PRE_TRADE_CHECK",
                reason_code=precheck_result.decision.lower(),
                reason_detail="Pre-trade check re-run passed for micro order creation.",
                request_payload=None,
                response_payload={
                    "decision": precheck_result.decision,
                    "reason_codes": list(precheck_result.reason_codes),
                    "reason_details": list(precheck_result.reason_details),
                },
                created_by=created_by,
            )
        ]
    )
    if not approval_validation.get("valid"):
        return update_micro_order_status(
            micro_order_id,
            "APPROVAL_INVALID",
            reason_code=_safe_str(approval_validation.get("reason_code"), "approval_invalid"),
            reason_detail=_safe_str(approval_validation.get("detail"), "Approval validation failed."),
            response_payload=approval_validation,
            event_type="APPROVAL_INVALID",
            event_source="APPROVAL",
            created_by=created_by,
        )
    if mode == "LIVE" and live_create_code:
        return update_micro_order_status(
            micro_order_id,
            "LIVE_BLOCKED",
            reason_code=live_create_code,
            reason_detail=live_create_detail,
            response_payload={"live_gate": live_gate_snapshot()},
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            created_by=created_by,
        )
    final_status = "LIVE_CONFIRMATION_REQUIRED" if mode == "LIVE" else "READY_TO_SEND"
    final_reason_code = "live_confirmation_required" if mode == "LIVE" else "approval_valid"
    final_reason_detail = (
        "Approval and pre-trade checks passed. Final confirmation is required before any Live send attempt."
        if mode == "LIVE"
        else "Approval is valid and pre-trade check passed."
    )
    return update_micro_order_status(
        micro_order_id,
        final_status,
        reason_code=final_reason_code,
        reason_detail=final_reason_detail,
        response_payload=approval_validation,
        event_type="LIVE_CONFIRMATION_REQUIRED" if mode == "LIVE" else "APPROVAL_VALID",
        event_source="LIVE_GATE" if mode == "LIVE" else "APPROVAL",
        created_by=created_by,
    )


def _prepare_send(
    micro_order_id: str,
    *,
    execution_mode: str,
    dry_run: bool,
    allowed_statuses: set[str],
) -> tuple[dict[str, object], dict[str, object], Any, dict[str, object]]:
    ensure_us_micro_live_tables()
    row = get_micro_order(micro_order_id)
    current_mode = _safe_str(row.get("execution_mode")).upper()
    if current_mode != execution_mode.upper():
        raise ValueError(f"Micro order execution_mode mismatch. expected={execution_mode} current={current_mode}")
    current_status = _safe_str(row.get("request_status")).upper()
    if current_status not in allowed_statuses:
        raise ValueError(f"Micro order is not sendable. current_status={current_status}")
    candidate = _micro_row_to_candidate(row)
    approval_validation = validate_approval_for_candidate(candidate, _safe_str(row.get("approval_id")) or None)
    precheck_result = run_us_live_pre_trade_check(candidate, write_block_log=not dry_run, log_requires_approval=True)
    send_payload = {
        "micro_order_id": micro_order_id,
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "order_type": row.get("order_type"),
        "limit_price": row.get("limit_price"),
        "order_qty": row.get("order_qty"),
        "order_amount_usd": row.get("order_amount_usd"),
        "execution_mode": row.get("execution_mode"),
        "broker_name": row.get("broker_name"),
    }
    return row, approval_validation, precheck_result, send_payload


def _handle_pre_send_failure(
    micro_order_id: str,
    *,
    precheck_result,
    approval_validation: dict[str, object],
    created_by: str,
) -> dict[str, object] | None:
    if precheck_result.decision == "BLOCK":
        return update_micro_order_status(
            micro_order_id,
            "PRECHECK_FAILED",
            reason_code="precheck_failed",
            reason_detail="; ".join(precheck_result.reason_details or ["Pre-trade check blocked before send."]),
            response_payload={
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
            },
            event_type="PRECHECK_FAIL",
            event_source="PRE_TRADE_CHECK",
            created_by=created_by,
        )
    if precheck_result.decision == "ERROR":
        return update_micro_order_status(
            micro_order_id,
            "ERROR",
            reason_code="precheck_error",
            reason_detail="; ".join(precheck_result.reason_details or ["Pre-trade check errored before send."]),
            response_payload={
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
            },
            event_type="ERROR",
            event_source="PRE_TRADE_CHECK",
            created_by=created_by,
        )
    if not approval_validation.get("valid"):
        return update_micro_order_status(
            micro_order_id,
            "APPROVAL_INVALID",
            reason_code=_safe_str(approval_validation.get("reason_code"), "approval_invalid"),
            reason_detail=_safe_str(approval_validation.get("detail"), "Approval validation failed."),
            response_payload=approval_validation,
            event_type="APPROVAL_INVALID",
            event_source="APPROVAL",
            created_by=created_by,
        )
    return None


def send_micro_order_via_mock(
    micro_order_id: str,
    *,
    created_by: str = "SYSTEM",
    dry_run: bool = False,
) -> dict[str, object]:
    assert_us_micro_mock_only()
    row, approval_validation, precheck_result, send_payload = _prepare_send(
        micro_order_id,
        execution_mode="MOCK",
        dry_run=dry_run,
        allowed_statuses=MOCK_SANDBOX_SENDABLE_STATUSES,
    )
    if dry_run:
        return {
            "micro_order": row,
            "approval_validation": approval_validation,
            "precheck_result": {
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
                "check_results": dict(precheck_result.check_results),
            },
            "send_payload": send_payload,
            "dry_run": True,
        }
    failed = _handle_pre_send_failure(micro_order_id, precheck_result=precheck_result, approval_validation=approval_validation, created_by=created_by)
    if failed:
        return failed
    if _safe_str(row.get("request_status")).upper() == "CREATED":
        update_micro_order_status(
            micro_order_id,
            "READY_TO_SEND",
            reason_code="precheck_passed",
            reason_detail="Pre-trade check re-run passed before mock submit.",
            response_payload={
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
            },
            event_type="PRECHECK_PASS",
            event_source="PRE_TRADE_CHECK",
            created_by=created_by,
        )
    update_micro_order_status(
        micro_order_id,
        "SENT",
        reason_code="send_mock",
        reason_detail="Mock order submission requested.",
        request_payload=send_payload,
        event_type="SEND_MOCK",
        event_source="MOCK_BROKER",
        created_by=created_by,
    )
    response = UsMockOrderClient().submit_order(send_payload)
    status = _safe_str(response.get("status"), "ERROR").upper()
    if status not in {"ACCEPTED", "REJECTED", "FAILED"}:
        status = "ERROR"
    return update_micro_order_status(
        micro_order_id,
        status,
        reason_code=_safe_str(response.get("error_code")) or ("mock_accepted" if status == "ACCEPTED" else "mock_response"),
        reason_detail=_safe_str(response.get("message")),
        response_payload=response,
        event_type=_status_event_type(status, "MOCK"),
        event_source="MOCK_BROKER",
        created_by=created_by,
        broker_order_id=_safe_str(response.get("broker_order_id")) or None,
    )


def send_micro_order_via_sandbox(
    micro_order_id: str,
    *,
    created_by: str = "SYSTEM",
    dry_run: bool = False,
    force_status_check_before_send: bool = False,
) -> dict[str, object]:
    assert_sandbox_enabled()
    row, approval_validation, precheck_result, send_payload = _prepare_send(
        micro_order_id,
        execution_mode="SANDBOX",
        dry_run=dry_run,
        allowed_statuses=MOCK_SANDBOX_SENDABLE_STATUSES,
    )
    limits_ok, limit_code, limit_detail = _sandbox_limits_ok(row)
    if dry_run:
        return {
            "micro_order": row,
            "approval_validation": approval_validation,
            "precheck_result": {
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
                "check_results": dict(precheck_result.check_results),
            },
            "send_payload": send_payload,
            "sandbox_config": sandbox_config_snapshot(),
            "force_status_check_before_send": force_status_check_before_send,
            "dry_run": True,
        }
    failed = _handle_pre_send_failure(micro_order_id, precheck_result=precheck_result, approval_validation=approval_validation, created_by=created_by)
    if failed:
        return failed
    if not limits_ok:
        return update_micro_order_status(
            micro_order_id,
            "FAILED",
            reason_code=limit_code,
            reason_detail=limit_detail,
            response_payload={"sandbox_config": sandbox_config_snapshot()},
            event_type="SANDBOX_FAILED",
            event_source="SANDBOX_BROKER",
            created_by=created_by,
        )
    if _safe_str(row.get("request_status")).upper() == "CREATED":
        update_micro_order_status(
            micro_order_id,
            "READY_TO_SEND",
            reason_code="precheck_passed",
            reason_detail="Pre-trade check re-run passed before sandbox submit.",
            response_payload={
                "decision": precheck_result.decision,
                "reason_codes": list(precheck_result.reason_codes),
                "reason_details": list(precheck_result.reason_details),
            },
            event_type="PRECHECK_PASS",
            event_source="PRE_TRADE_CHECK",
            created_by=created_by,
        )
    if force_status_check_before_send and _safe_str(row.get("broker_order_id")):
        check_micro_order_sandbox_status(micro_order_id, created_by=created_by)
    update_micro_order_status(
        micro_order_id,
        "SENT",
        reason_code="send_sandbox",
        reason_detail="Sandbox order submission requested.",
        request_payload=send_payload,
        event_type="SEND_SANDBOX",
        event_source="SANDBOX_BROKER",
        created_by=created_by,
    )
    response = UsSandboxOrderClient().submit_order(send_payload)
    status = _safe_str(response.get("status"), "ERROR").upper()
    if status not in {"ACCEPTED", "REJECTED", "FAILED"}:
        status = "ERROR"
    return update_micro_order_status(
        micro_order_id,
        status,
        reason_code=_safe_str(response.get("error_code")) or ("sandbox_accepted" if status == "ACCEPTED" else "sandbox_response"),
        reason_detail=_safe_str(response.get("message")),
        response_payload=response,
        event_type=_status_event_type(status, "SANDBOX"),
        event_source="SANDBOX_BROKER",
        created_by=created_by,
        broker_order_id=_safe_str(response.get("broker_order_id")) or None,
    )


def check_micro_order_sandbox_status(
    micro_order_id: str,
    *,
    created_by: str = "SYSTEM",
    dry_run: bool = False,
) -> dict[str, object]:
    assert_sandbox_enabled()
    row = get_micro_order(micro_order_id)
    if _safe_str(row.get("execution_mode")).upper() != "SANDBOX":
        raise ValueError("Sandbox status check requires execution_mode=SANDBOX.")
    broker_order_id = _safe_str(row.get("broker_order_id"))
    if not broker_order_id:
        raise ValueError("Sandbox status check requires broker_order_id.")
    if dry_run:
        return {"micro_order": row, "broker_order_id": broker_order_id, "dry_run": True}
    response = UsSandboxOrderClient().get_order_status(broker_order_id)
    status = _safe_str(response.get("status"), _safe_str(row.get("request_status"), "ERROR")).upper()
    return update_micro_order_status(
        micro_order_id,
        status if status in MICRO_ORDER_STATUSES else _safe_str(row.get("request_status"), "ERROR").upper(),
        reason_code=_safe_str(response.get("error_code")) or "sandbox_status_check",
        reason_detail=_safe_str(response.get("message")),
        response_payload=response,
        event_type="SANDBOX_STATUS_CHECK",
        event_source="SANDBOX_BROKER",
        created_by=created_by,
        broker_order_id=broker_order_id,
    )


def cancel_micro_order_sandbox(
    micro_order_id: str,
    *,
    reason: str,
    created_by: str = "OPERATOR",
    dry_run: bool = False,
) -> dict[str, object]:
    assert_sandbox_enabled()
    row = get_micro_order(micro_order_id)
    if _safe_str(row.get("execution_mode")).upper() != "SANDBOX":
        raise ValueError("Sandbox cancel requires execution_mode=SANDBOX.")
    current_status = _safe_str(row.get("request_status")).upper()
    if current_status not in CANCELABLE_SANDBOX_STATUSES:
        raise ValueError(f"Sandbox micro order cannot be canceled in status={current_status}")
    broker_order_id = _safe_str(row.get("broker_order_id"))
    if not broker_order_id:
        raise ValueError("Sandbox cancel requires broker_order_id.")
    if dry_run:
        preview = dict(row)
        preview["request_status"] = "CANCELED"
        preview["reject_reason_detail"] = reason
        return preview
    response = UsSandboxOrderClient().cancel_order(broker_order_id)
    status = _safe_str(response.get("status"), "FAILED").upper()
    if status == "CANCELED":
        return update_micro_order_status(
            micro_order_id,
            "CANCELED",
            reason_code="sandbox_canceled",
            reason_detail=reason or _safe_str(response.get("message")),
            response_payload=response,
            event_type="SANDBOX_CANCELED",
            event_source="SANDBOX_BROKER",
            created_by=created_by,
            broker_order_id=broker_order_id,
        )
    return update_micro_order_status(
        micro_order_id,
        "FAILED",
        reason_code=_safe_str(response.get("error_code")) or "sandbox_cancel_failed",
        reason_detail=_safe_str(response.get("message")) or reason,
        response_payload=response,
        event_type="SANDBOX_CANCEL_FAILED",
        event_source="SANDBOX_BROKER",
        created_by=created_by,
        broker_order_id=broker_order_id,
    )


def send_micro_order_via_live(
    micro_order_id: str,
    *,
    created_by: str = "SYSTEM",
    dry_run: bool = False,
    confirm_live_order: str | None = None,
) -> dict[str, object]:
    ensure_us_micro_live_tables()
    row = get_micro_order(micro_order_id)
    if _safe_str(row.get("execution_mode")).upper() != "LIVE":
        raise ValueError("Live send requires execution_mode=LIVE.")
    current_status = _safe_str(row.get("request_status")).upper()
    if current_status not in LIVE_SENDABLE_STATUSES:
        raise ValueError(f"Live micro order is not sendable. current_status={current_status}")

    final_confirm = _safe_str(confirm_live_order) == LIVE_FINAL_CONFIRMATION_TEXT
    validation = validate_live_order_execution_allowed(micro_order_id, final_confirm=final_confirm)
    preview = {
        "micro_order": row,
        "validation": mask_sensitive_payload(validation),
        "send_payload": validation.get("masked_payload") or mask_sensitive_payload(build_live_order_payload(row)),
        "dry_run": dry_run,
    }
    if dry_run:
        return preview
    if not validation.get("allowed"):
        return update_micro_order_status(
            micro_order_id,
            _safe_str(validation.get("status"), "LIVE_BLOCKED"),
            reason_code=_safe_str(validation.get("reason_code"), "live_mode_not_allowed"),
            reason_detail=_safe_str(validation.get("reason_detail"), "Live order was blocked by safety policy."),
            response_payload=mask_sensitive_payload(validation),
            request_payload=validation.get("masked_payload") if isinstance(validation.get("masked_payload"), dict) else None,
            event_type=_safe_str(validation.get("event_type"), "LIVE_SAFETY_CHECK_FAILED"),
            event_source=_safe_str(validation.get("event_source"), "LIVE_GATE"),
            created_by=created_by,
        )
    update_micro_order_status(
        micro_order_id,
        "LIVE_READY",
        reason_code="live_ready",
        reason_detail="All Live safety checks passed; submitting placeholder live request.",
        request_payload=validation.get("masked_payload") if isinstance(validation.get("masked_payload"), dict) else None,
        response_payload=mask_sensitive_payload(validation),
        event_type="LIVE_READY",
        event_source="LIVE_GATE",
        created_by=created_by,
    )
    update_micro_order_status(
        micro_order_id,
        "LIVE_SENT",
        reason_code="send_live",
        reason_detail="Live order submission requested.",
        request_payload=validation.get("masked_payload") if isinstance(validation.get("masked_payload"), dict) else None,
        event_type="SEND_LIVE",
        event_source="LIVE_BROKER",
        created_by=created_by,
    )
    response = UsLiveOrderClient().submit_order(validation.get("payload") if isinstance(validation.get("payload"), dict) else build_live_order_payload(row))
    status = _safe_str(response.get("status"), "FAILED").upper()
    mapped_status = {
        "ACCEPTED": "LIVE_ACCEPTED",
        "REJECTED": "LIVE_REJECTED",
        "FAILED": "LIVE_FAILED",
        "CANCELED": "LIVE_CANCELED",
    }.get(status, "LIVE_FAILED")
    return update_micro_order_status(
        micro_order_id,
        mapped_status,
        reason_code=_safe_str(response.get("error_code")) or mapped_status.lower(),
        reason_detail=_safe_str(response.get("message")),
        response_payload=mask_sensitive_payload(response),
        event_type=_status_event_type(mapped_status, "LIVE"),
        event_source="LIVE_BROKER",
        created_by=created_by,
        broker_order_id=_safe_str(response.get("broker_order_id")) or None,
    )


def cancel_micro_order(
    micro_order_id: str,
    *,
    reason: str,
    created_by: str = "OPERATOR",
    dry_run: bool = False,
) -> dict[str, object]:
    row = get_micro_order(micro_order_id)
    mode = _safe_str(row.get("execution_mode")).upper()
    current_status = _safe_str(row.get("request_status")).upper()
    if mode == "LIVE":
        raise ValueError("Phase 7-3 does not support Live cancel from manage_us_micro_order.py.")
    if current_status not in {"CREATED", "READY_TO_SEND"}:
        raise ValueError(f"Micro order cannot be canceled in status={current_status}")
    if dry_run:
        preview = dict(row)
        preview["request_status"] = "CANCELED"
        preview["reject_reason_detail"] = reason
        return preview
    return update_micro_order_status(
        micro_order_id,
        "CANCELED",
        reason_code="operator_canceled",
        reason_detail=reason,
        event_type="CANCELED",
        event_source="OPERATOR",
        created_by=created_by,
    )


def build_micro_order_report(
    *,
    trade_date: str | None = None,
    account_id: str | None = None,
    status: str | None = None,
    execution_mode: str | None = None,
) -> dict[str, object]:
    rows = list_micro_orders(
        trade_date=trade_date,
        account_id=account_id,
        status=status,
        execution_mode=execution_mode,
    )
    summary: dict[str, int] = {name: 0 for name in sorted(MICRO_ORDER_STATUSES)}
    execution_summary: dict[str, int] = {"MOCK": 0, "SANDBOX": 0, "LIVE": 0}
    for row in rows:
        summary[_safe_str(row.get("request_status")).upper()] = summary.get(_safe_str(row.get("request_status")).upper(), 0) + 1
        mode = _safe_str(row.get("execution_mode")).upper()
        execution_summary[mode] = execution_summary.get(mode, 0) + 1

    sandbox_rows = [row for row in rows if _safe_str(row.get("execution_mode")).upper() == "SANDBOX"]
    sandbox_events: list[dict[str, object]] = []
    for row in sandbox_rows[-10:]:
        sandbox_events.extend(get_micro_order_events(_safe_str(row.get("micro_order_id"))))
    sandbox_events = [event for event in sandbox_events if "SANDBOX" in _safe_str(event.get("event_type")).upper()][:20]

    live_rows = [row for row in rows if _safe_str(row.get("execution_mode")).upper() == "LIVE"]
    live_events: list[dict[str, object]] = []
    for row in live_rows[-10:]:
        live_events.extend(get_micro_order_events(_safe_str(row.get("micro_order_id"))))
    live_events = [event for event in live_events if "LIVE" in _safe_str(event.get("event_type")).upper()][:20]

    return {
        "rows": rows,
        "summary": summary,
        "execution_summary": execution_summary,
        "sandbox_config": sandbox_config_snapshot(),
        "sandbox_events": sandbox_events,
        "live_config": live_gate_snapshot(),
        "live_events": live_events,
        "live_rows": live_rows,
    }
