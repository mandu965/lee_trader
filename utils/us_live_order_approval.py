from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json

from python.us.us_db import (
    ensure_us_live_risk_tables,
    fetch_us_live_order_approval_event_log_rows,
    fetch_us_live_order_approval_rows,
    insert_us_live_order_approval_event_log_rows,
    upsert_us_live_order_approval_rows,
)
from utils.us_live_risk_policy import load_us_live_risk_policy


FINAL_APPROVAL_STATUSES = {"APPROVED", "REJECTED", "EXPIRED", "CANCELED", "ERROR"}


def _safe_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _serialize_reason_codes(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return json.dumps([], ensure_ascii=False)


def _trade_date_value(value: object) -> date | None:
    text = _safe_str(value)
    if not text:
        return None
    return date.fromisoformat(text[:10])


def _as_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = _safe_str(value)
    if not text:
        return None
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _candidate_identity(candidate) -> tuple[object, ...]:
    return (
        _safe_str(getattr(candidate, "trade_date", None)),
        _safe_str(getattr(candidate, "account_id", None)),
        _safe_str(getattr(candidate, "symbol", None)).upper(),
        _safe_str(getattr(candidate, "side", None)).upper(),
        _safe_str(getattr(candidate, "candidate_source", None)),
        _safe_str(getattr(candidate, "strategy_name", None)),
        getattr(candidate, "requested_order_amount_usd", None),
        _safe_str(getattr(candidate, "requested_order_type", None)).upper(),
    )


def _row_identity(row: dict[str, object]) -> tuple[object, ...]:
    return (
        _safe_str(row.get("trade_date")),
        _safe_str(row.get("account_id")),
        _safe_str(row.get("symbol")).upper(),
        _safe_str(row.get("side")).upper(),
        _safe_str(row.get("candidate_source")),
        _safe_str(row.get("strategy_name")),
        row.get("requested_order_amount_usd"),
        _safe_str(row.get("requested_order_type")).upper(),
    )


def _build_event_row(
    *,
    approval_id: str,
    event_type: str,
    before_status: str | None,
    after_status: str | None,
    reason_code: str | None,
    reason_detail: str | None,
    performed_by: str | None,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return {
        "event_id": f"USAPPEVT_{approval_id}_{event_type}_{timestamp}",
        "approval_id": approval_id,
        "event_type": event_type,
        "before_status": before_status,
        "after_status": after_status,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "performed_by": performed_by,
    }


def build_approval_id(candidate, now=None) -> str:
    now = now or datetime.now(timezone.utc)
    trade_date = _safe_str(getattr(candidate, "trade_date", None)).replace("-", "")
    account_id = _safe_str(getattr(candidate, "account_id", None)).upper()
    symbol = _safe_str(getattr(candidate, "symbol", None)).upper()
    side = _safe_str(getattr(candidate, "side", None)).upper()
    return f"USAPP_{trade_date}_{account_id}_{symbol}_{side}_{now.strftime('%Y%m%d%H%M%S')}"


def _build_approval_row(
    candidate,
    precheck_result,
    *,
    approval_id: str,
    requested_by: str,
    expires_at,
) -> dict[str, object]:
    return {
        "approval_id": approval_id,
        "trade_date": _trade_date_value(getattr(candidate, "trade_date", None)),
        "policy_id": getattr(candidate, "policy_id", None),
        "account_id": getattr(candidate, "account_id", None),
        "symbol": _safe_str(getattr(candidate, "symbol", None)).upper(),
        "side": _safe_str(getattr(candidate, "side", None)).upper(),
        "candidate_source": getattr(candidate, "candidate_source", None),
        "strategy_name": getattr(candidate, "strategy_name", None),
        "rank_no": getattr(candidate, "rank_no", None),
        "recommend_grade": getattr(candidate, "recommend_grade", None),
        "total_score": getattr(candidate, "total_score", None),
        "requested_order_type": _safe_str(getattr(candidate, "requested_order_type", None)).upper(),
        "requested_limit_price": getattr(candidate, "requested_limit_price", None),
        "requested_qty": getattr(candidate, "requested_qty", None),
        "requested_order_amount_usd": getattr(candidate, "requested_order_amount_usd", None),
        "precheck_decision": getattr(precheck_result, "decision", None),
        "precheck_reason_codes": _serialize_reason_codes(getattr(precheck_result, "reason_codes", [])),
        "precheck_summary": "; ".join(getattr(precheck_result, "reason_details", []) or []),
        "approval_status": "PENDING",
        "requested_by": requested_by,
        "requested_at": datetime.now(timezone.utc),
        "approved_by": None,
        "approved_at": None,
        "approval_reason": None,
        "rejected_by": None,
        "rejected_at": None,
        "reject_reason": None,
        "expired_at": None,
        "expires_at": expires_at,
    }


def _find_existing_pending(candidate) -> list[dict[str, object]]:
    rows = fetch_us_live_order_approval_rows(
        trade_date=_trade_date_value(getattr(candidate, "trade_date", None)),
        account_id=getattr(candidate, "account_id", None),
        status="PENDING",
    )
    target_identity = _candidate_identity(candidate)
    return [row for row in rows if _row_identity(row) == target_identity]


def _cancel_existing_pending(row: dict[str, object], *, performed_by: str, reason: str) -> None:
    before_status = _safe_str(row.get("approval_status")).upper()
    if before_status != "PENDING":
        return
    row = dict(row)
    row["approval_status"] = "CANCELED"
    row["reject_reason"] = reason
    row["rejected_by"] = performed_by
    row["rejected_at"] = datetime.now(timezone.utc)
    upsert_us_live_order_approval_rows([row])
    insert_us_live_order_approval_event_log_rows(
        [
            _build_event_row(
                approval_id=_safe_str(row.get("approval_id")),
                event_type="CANCEL",
                before_status=before_status,
                after_status="CANCELED",
                reason_code="replaced_pending_request",
                reason_detail=reason,
                performed_by=performed_by,
            )
        ]
    )


def create_order_approval_request(
    candidate,
    precheck_result,
    requested_by: str = "SYSTEM",
    expires_minutes: int | None = None,
    replace_existing: bool = False,
) -> dict[str, object]:
    decision = _safe_str(getattr(precheck_result, "decision", None)).upper()
    if decision in {"BLOCK", "ERROR"}:
        raise ValueError("Approval request is not created for BLOCK or ERROR pre-check results.")
    ensure_us_live_risk_tables()
    existing_pending = _find_existing_pending(candidate)
    if existing_pending and not replace_existing:
        return existing_pending[0]
    for row in existing_pending:
        _cancel_existing_pending(
            row,
            performed_by=requested_by,
            reason="Replaced by newer approval request.",
        )
    if expires_minutes is None:
        policy = load_us_live_risk_policy(getattr(candidate, "policy_id", None))
        approval_policy = policy.get("approval", {}) if isinstance(policy.get("approval"), dict) else {}
        expires_minutes = int(approval_policy.get("approval_expires_minutes", 30) or 30)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=int(expires_minutes))
    approval_id = build_approval_id(candidate)
    row = _build_approval_row(
        candidate,
        precheck_result,
        approval_id=approval_id,
        requested_by=requested_by,
        expires_at=expires_at,
    )
    upsert_us_live_order_approval_rows([row])
    insert_us_live_order_approval_event_log_rows(
        [
            _build_event_row(
                approval_id=approval_id,
                event_type="REQUEST",
                before_status=None,
                after_status="PENDING",
                reason_code=decision.lower(),
                reason_detail=row["precheck_summary"] or f"Pre-trade decision={decision}",
                performed_by=requested_by,
            )
        ]
    )
    return get_order_approval(approval_id)


def get_order_approval(approval_id: str) -> dict:
    rows = fetch_us_live_order_approval_rows(approval_id=approval_id)
    if not rows:
        raise ValueError(f"Approval request not found: {approval_id}")
    return rows[0]


def list_order_approvals(
    status: str | None = None,
    trade_date: str | None = None,
    account_id: str | None = None,
) -> list[dict]:
    return fetch_us_live_order_approval_rows(
        trade_date=_trade_date_value(trade_date),
        account_id=account_id,
        status=status.upper() if status else None,
    )


def approve_order_approval(
    approval_id: str,
    approved_by: str,
    approval_reason: str,
) -> dict:
    approved_by = _safe_str(approved_by)
    approval_reason = _safe_str(approval_reason)
    if not approved_by:
        raise ValueError("approved_by is required.")
    if not approval_reason:
        raise ValueError("approval_reason is required.")
    row = get_order_approval(approval_id)
    before_status = _safe_str(row.get("approval_status")).upper()
    if before_status != "PENDING":
        raise ValueError(f"Only PENDING approvals can be approved. current_status={before_status}")
    row = dict(row)
    row["approval_status"] = "APPROVED"
    row["approved_by"] = approved_by
    row["approved_at"] = datetime.now(timezone.utc)
    row["approval_reason"] = approval_reason
    upsert_us_live_order_approval_rows([row])
    insert_us_live_order_approval_event_log_rows(
        [
            _build_event_row(
                approval_id=approval_id,
                event_type="APPROVE",
                before_status=before_status,
                after_status="APPROVED",
                reason_code="manual_approved",
                reason_detail=approval_reason,
                performed_by=approved_by,
            )
        ]
    )
    return get_order_approval(approval_id)


def reject_order_approval(
    approval_id: str,
    rejected_by: str,
    reject_reason: str,
) -> dict:
    rejected_by = _safe_str(rejected_by)
    reject_reason = _safe_str(reject_reason)
    if not rejected_by:
        raise ValueError("rejected_by is required.")
    if not reject_reason:
        raise ValueError("reject_reason is required.")
    row = get_order_approval(approval_id)
    before_status = _safe_str(row.get("approval_status")).upper()
    if before_status != "PENDING":
        raise ValueError(f"Only PENDING approvals can be rejected. current_status={before_status}")
    row = dict(row)
    row["approval_status"] = "REJECTED"
    row["rejected_by"] = rejected_by
    row["rejected_at"] = datetime.now(timezone.utc)
    row["reject_reason"] = reject_reason
    upsert_us_live_order_approval_rows([row])
    insert_us_live_order_approval_event_log_rows(
        [
            _build_event_row(
                approval_id=approval_id,
                event_type="REJECT",
                before_status=before_status,
                after_status="REJECTED",
                reason_code="manual_rejected",
                reason_detail=reject_reason,
                performed_by=rejected_by,
            )
        ]
    )
    return get_order_approval(approval_id)


def expire_order_approvals(as_of_time=None) -> dict:
    as_of_time = as_of_time or datetime.now(timezone.utc)
    pending_rows = fetch_us_live_order_approval_rows(status="PENDING")
    expired_count = 0
    for row in pending_rows:
        expires_at = _as_utc_datetime(row.get("expires_at"))
        if not expires_at or expires_at > as_of_time:
            continue
        before_status = _safe_str(row.get("approval_status")).upper()
        updated = dict(row)
        updated["approval_status"] = "EXPIRED"
        updated["expired_at"] = as_of_time
        upsert_us_live_order_approval_rows([updated])
        insert_us_live_order_approval_event_log_rows(
            [
                _build_event_row(
                    approval_id=_safe_str(row.get("approval_id")),
                    event_type="EXPIRE",
                    before_status=before_status,
                    after_status="EXPIRED",
                    reason_code="approval_expired",
                    reason_detail="Pending approval expired.",
                    performed_by="SYSTEM",
                )
            ]
        )
        expired_count += 1
    return {"expired_count": expired_count, "as_of_time": as_of_time}


def validate_approval_for_candidate(candidate, approval_id: str | None = None) -> dict:
    row = get_order_approval(approval_id) if approval_id else None
    if row is None:
        approved_rows = [
            item for item in list_order_approvals(
                trade_date=getattr(candidate, "trade_date", None),
                account_id=getattr(candidate, "account_id", None),
                status="APPROVED",
            )
            if _row_identity(item) == _candidate_identity(candidate)
        ]
        if not approved_rows:
            return {"valid": False, "reason_code": "approval_missing", "detail": "No matching approved approval request found."}
        row = approved_rows[0]
    status = _safe_str(row.get("approval_status")).upper()
    if status != "APPROVED":
        return {"valid": False, "reason_code": "approval_not_approved", "detail": f"approval_status={status}"}
    expires_at = _as_utc_datetime(row.get("expires_at"))
    if expires_at and expires_at <= datetime.now(timezone.utc):
        return {"valid": False, "reason_code": "approval_expired", "detail": "Approval has expired."}
    if _row_identity(row) != _candidate_identity(candidate):
        return {"valid": False, "reason_code": "approval_candidate_mismatch", "detail": "Candidate fields do not match approval request."}
    return {
        "valid": True,
        "reason_code": "approved",
        "detail": "Approval is present, but Pre-Trade Check must be re-run before any live-order review.",
        "approval_id": row.get("approval_id"),
    }


def get_order_approval_events(approval_id: str) -> list[dict]:
    return fetch_us_live_order_approval_event_log_rows(approval_id=approval_id)
