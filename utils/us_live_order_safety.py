from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from python.us.us_db import fetch_us_micro_order_request_rows
from utils.us_live_order_approval import validate_approval_for_candidate
from utils.us_live_pre_trade_check import UsLiveOrderCandidate, run_us_live_pre_trade_check


LIVE_FINAL_CONFIRMATION_TEXT = "I_UNDERSTAND_THIS_IS_A_REAL_ORDER"
LIVE_SAFETY_MESSAGE = "[SAFETY] Micro Live real-order path is blocked by default and requires explicit gates plus final confirmation."


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


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


def live_gate_snapshot() -> dict[str, object]:
    return {
        "micro_allow_live": _flag("US_MICRO_ALLOW_LIVE", "false"),
        "micro_real_order_blocked": _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"),
        "live_trading_enabled": _flag("US_LIVE_TRADING_ENABLED", "false"),
        "live_order_enabled": _flag("US_LIVE_ORDER_ENABLED", "false"),
        "live_buy_enabled": _flag("US_LIVE_BUY_ENABLED", "false"),
        "live_sell_enabled": _flag("US_LIVE_SELL_ENABLED", "false"),
        "require_manual_approval": _flag("US_LIVE_REQUIRE_MANUAL_APPROVAL", "true"),
        "require_final_confirmation": _flag("US_LIVE_REQUIRE_FINAL_CONFIRMATION", "true"),
        "live_broker": _text("US_LIVE_BROKER", "none").upper() or "NONE",
        "live_default_order_type": _text("US_LIVE_DEFAULT_ORDER_TYPE", "LIMIT").upper() or "LIMIT",
        "live_allow_market_order": _flag("US_LIVE_ALLOW_MARKET_ORDER", "false"),
        "max_order_amount_usd": _safe_float(_text("US_LIVE_MAX_ORDER_AMOUNT_USD", "50")),
        "max_daily_order_count": _safe_float(_text("US_LIVE_MAX_DAILY_ORDER_COUNT", "3")),
        "max_daily_new_buys": _safe_float(_text("US_LIVE_MAX_DAILY_NEW_BUYS", "1")),
    }


def assert_live_order_gate_closed_by_default() -> None:
    snapshot = live_gate_snapshot()
    if snapshot["micro_allow_live"]:
        raise RuntimeError("US_MICRO_ALLOW_LIVE must remain false in the safe default configuration.")
    if not snapshot["micro_real_order_blocked"]:
        raise RuntimeError("US_MICRO_REAL_ORDER_BLOCKED must remain true in the safe default configuration.")
    if snapshot["live_trading_enabled"] or snapshot["live_order_enabled"]:
        raise RuntimeError("US_LIVE_TRADING_ENABLED and US_LIVE_ORDER_ENABLED must remain false in the safe default configuration.")
    if snapshot["live_buy_enabled"] or snapshot["live_sell_enabled"]:
        raise RuntimeError("US_LIVE_BUY_ENABLED and US_LIVE_SELL_ENABLED must remain false in the safe default configuration.")
    if snapshot["live_allow_market_order"]:
        raise RuntimeError("US_LIVE_ALLOW_MARKET_ORDER must remain false in the safe default configuration.")


def mask_sensitive_payload(payload: Any) -> Any:
    sensitive_keys = {
        "api_key",
        "api_secret",
        "access_token",
        "authorization",
        "auth_token",
        "account_number",
        "account_no",
        "account_num",
    }
    if isinstance(payload, dict):
        masked: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key).strip().lower()
            if key_text in sensitive_keys or "secret" in key_text or "token" in key_text or "authorization" in key_text:
                masked[key] = "***MASKED***"
            else:
                masked[key] = mask_sensitive_payload(value)
        return masked
    if isinstance(payload, list):
        return [mask_sensitive_payload(item) for item in payload]
    return payload


def build_live_order_payload(micro_order: dict[str, object]) -> dict[str, object]:
    return {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "micro_order_id": micro_order.get("micro_order_id"),
        "approval_id": micro_order.get("approval_id"),
        "trade_date": _safe_str(micro_order.get("trade_date")),
        "policy_id": micro_order.get("policy_id"),
        "account_id": micro_order.get("account_id"),
        "symbol": _safe_str(micro_order.get("symbol")).upper(),
        "side": _safe_str(micro_order.get("side")).upper(),
        "order_type": _safe_str(micro_order.get("order_type"), "LIMIT").upper(),
        "limit_price": _safe_float(micro_order.get("limit_price")),
        "order_qty": _safe_float(micro_order.get("order_qty")),
        "order_amount_usd": _safe_float(micro_order.get("order_amount_usd")),
        "execution_mode": "LIVE",
        "broker_name": _safe_str(micro_order.get("broker_name") or _text("US_LIVE_BROKER", "none")).upper(),
    }


def _validation_result(
    *,
    allowed: bool,
    status: str,
    reason_code: str,
    reason_detail: str,
    event_type: str,
    event_source: str,
    micro_order: dict[str, object] | None,
    approval_validation: dict[str, object] | None = None,
    precheck_result: dict[str, object] | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "allowed": allowed,
        "status": status,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "event_type": event_type,
        "event_source": event_source,
        "micro_order": micro_order,
        "approval_validation": approval_validation or {},
        "precheck_result": precheck_result or {},
        "payload": payload or {},
        "masked_payload": mask_sensitive_payload(payload or {}),
        "gate": live_gate_snapshot(),
    }


def validate_live_order_execution_allowed(micro_order_id: str, final_confirm: bool = False) -> dict[str, object]:
    rows = fetch_us_micro_order_request_rows(micro_order_id=micro_order_id)
    if not rows:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="approval_missing",
            reason_detail=f"Micro order not found: {micro_order_id}",
            event_type="LIVE_SAFETY_CHECK_FAILED",
            event_source="SYSTEM",
            micro_order=None,
        )
    row = rows[0]
    if _safe_str(row.get("execution_mode")).upper() != "LIVE":
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_mode_not_allowed",
            reason_detail="execution_mode must be LIVE for a Live send attempt.",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
        )
    if not _safe_str(row.get("approval_id")):
        return _validation_result(
            allowed=False,
            status="APPROVAL_INVALID",
            reason_code="approval_missing",
            reason_detail="approval_id is required for Live execution.",
            event_type="APPROVAL_INVALID",
            event_source="APPROVAL",
            micro_order=row,
        )

    candidate = _micro_row_to_candidate(row)
    approval_validation = validate_approval_for_candidate(candidate, _safe_str(row.get("approval_id")))
    if not approval_validation.get("valid"):
        return _validation_result(
            allowed=False,
            status="APPROVAL_INVALID",
            reason_code=_safe_str(approval_validation.get("reason_code"), "approval_invalid"),
            reason_detail=_safe_str(approval_validation.get("detail"), "Approval validation failed."),
            event_type="APPROVAL_INVALID",
            event_source="APPROVAL",
            micro_order=row,
            approval_validation=approval_validation,
        )

    precheck = run_us_live_pre_trade_check(candidate, write_block_log=False, log_requires_approval=True)
    precheck_payload = {
        "decision": precheck.decision,
        "reason_codes": list(precheck.reason_codes),
        "reason_details": list(precheck.reason_details),
        "check_results": dict(precheck.check_results),
    }
    if precheck.decision in {"BLOCK", "ERROR"}:
        source = "KILL_SWITCH" if any("kill_switch" in code for code in precheck.reason_codes) else "PRE_TRADE_CHECK"
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code=precheck.reason_codes[0] if precheck.reason_codes else "precheck_failed",
            reason_detail="; ".join(precheck.reason_details or ["Pre-trade check blocked Live send."]),
            event_type="LIVE_SAFETY_CHECK_FAILED",
            event_source=source,
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    gate = live_gate_snapshot()
    if not gate["micro_allow_live"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_mode_not_allowed",
            reason_detail="US_MICRO_ALLOW_LIVE=true is required.",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )
    if gate["micro_real_order_blocked"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="real_order_blocked",
            reason_detail="US_MICRO_REAL_ORDER_BLOCKED=true keeps Live sends blocked.",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )
    if not gate["live_trading_enabled"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_trading_disabled",
            reason_detail="US_LIVE_TRADING_ENABLED=false",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )
    if not gate["live_order_enabled"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_order_disabled",
            reason_detail="US_LIVE_ORDER_ENABLED=false",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    side = _safe_str(row.get("side")).upper()
    if side == "BUY" and not gate["live_buy_enabled"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_buy_disabled",
            reason_detail="US_LIVE_BUY_ENABLED=false",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )
    if side == "SELL" and not gate["live_sell_enabled"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="live_sell_disabled",
            reason_detail="US_LIVE_SELL_ENABLED=false",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    order_type = _safe_str(row.get("order_type"), "LIMIT").upper()
    if order_type != "LIMIT" or gate["live_allow_market_order"]:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="market_order_not_allowed",
            reason_detail="Micro Live only allows LIMIT orders.",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    order_amount = _safe_float(row.get("order_amount_usd")) or 0.0
    max_amount = _safe_float(gate.get("max_order_amount_usd")) or 50.0
    if order_amount > max_amount:
        return _validation_result(
            allowed=False,
            status="LIVE_BLOCKED",
            reason_code="order_amount_exceeded",
            reason_detail=f"order_amount_usd={order_amount} exceeds US_LIVE_MAX_ORDER_AMOUNT_USD={max_amount}",
            event_type="LIVE_BLOCKED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    broker_name = _safe_str(row.get("broker_name") or gate.get("live_broker")).upper()
    if broker_name in {"", "NONE"}:
        return _validation_result(
            allowed=False,
            status="LIVE_FAILED",
            reason_code="broker_not_configured",
            reason_detail="US_LIVE_BROKER is not configured.",
            event_type="LIVE_FAILED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
        )

    payload = build_live_order_payload(row)
    required_fields = ["micro_order_id", "approval_id", "symbol", "side", "order_type", "limit_price", "order_amount_usd", "broker_name"]
    missing_fields = [field for field in required_fields if payload.get(field) in {None, ""}]
    if missing_fields:
        return _validation_result(
            allowed=False,
            status="LIVE_FAILED",
            reason_code="live_client_not_configured",
            reason_detail=f"Live payload is missing required fields: {', '.join(missing_fields)}",
            event_type="LIVE_FAILED",
            event_source="LIVE_GATE",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
            payload=payload,
        )

    if gate["require_final_confirmation"] and not final_confirm:
        return _validation_result(
            allowed=False,
            status="LIVE_CONFIRMATION_REQUIRED",
            reason_code="final_confirmation_missing",
            reason_detail=f"Real Live sends require --confirm-live-order {LIVE_FINAL_CONFIRMATION_TEXT}",
            event_type="LIVE_CONFIRMATION_REQUIRED",
            event_source="OPERATOR",
            micro_order=row,
            approval_validation=approval_validation,
            precheck_result=precheck_payload,
            payload=payload,
        )

    return _validation_result(
        allowed=True,
        status="LIVE_READY",
        reason_code="live_ready",
        reason_detail="All Live safety checks passed for this one-off submission attempt.",
        event_type="LIVE_READY",
        event_source="LIVE_GATE",
        micro_order=row,
        approval_validation=approval_validation,
        precheck_result=precheck_payload,
        payload=payload,
    )
