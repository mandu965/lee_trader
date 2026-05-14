from __future__ import annotations

from python.us.notification.config import NotificationConfig


SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
DEFAULT_DATA_MISSING_WARNING_PCT = 5.0
DEFAULT_DATA_MISSING_CRITICAL_PCT = 20.0


def _raise(current: str, candidate: str) -> str:
    return candidate if SEVERITY_ORDER[candidate] > SEVERITY_ORDER[current] else current


def determine_notification_severity(
    payload: dict[str, object] | None,
    *,
    validation_result: dict[str, object],
    cfg: NotificationConfig,
) -> dict[str, object]:
    severity = "INFO"
    reasons: list[str] = []
    payload = payload or {}

    for error in validation_result.get("errors") or []:
        code = str(error)
        if code in {"PAPER_TRADING_ONLY_FALSE", "LIVE_ORDERS_EXECUTED_TRUE"}:
            severity = _raise(severity, "CRITICAL")
            reasons.append(code)
        else:
            severity = _raise(severity, "ERROR")
            reasons.append(code)

    if payload.get("paper_trading_only") is False:
        severity = _raise(severity, "CRITICAL")
        reasons.append("PAPER_TRADING_ONLY_FALSE")
    if payload.get("live_orders_executed") is True:
        severity = _raise(severity, "CRITICAL")
        reasons.append("LIVE_ORDERS_EXECUTED_TRUE")
    if payload.get("live_trading_enabled") is True:
        severity = _raise(severity, "CRITICAL")
        reasons.append("LIVE_TRADING_ENABLED_TRUE")

    if str(payload.get("status") or "").upper() == "ERROR":
        severity = _raise(severity, "ERROR")
        reasons.append("DASHBOARD_STATUS_ERROR")

    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    if str((health or {}).get("scheduler_status") or "").upper() == "ERROR":
        severity = _raise(severity, "ERROR")
        reasons.append("SCHEDULER_STATUS_ERROR")
    if str((health or {}).get("dashboard_status") or "").upper() == "ERROR":
        severity = _raise(severity, "ERROR")
        reasons.append("DASHBOARD_HEALTH_ERROR")

    sell = payload.get("sell") if isinstance(payload.get("sell"), dict) else {}
    review_required = int((sell or {}).get("review_required") or 0)
    if review_required > 0:
        severity = _raise(severity, "WARNING")
        reasons.append("REVIEW_REQUIRED_EXISTS")

    risk = payload.get("risk") if isinstance(payload.get("risk"), dict) else {}
    try:
        data_missing_rate = float((risk or {}).get("data_missing_rate") or 0.0)
    except (TypeError, ValueError):
        data_missing_rate = 0.0
    critical_threshold = float(getattr(cfg, "data_missing_critical_pct", DEFAULT_DATA_MISSING_CRITICAL_PCT))
    warning_threshold = float(getattr(cfg, "data_missing_warning_pct", DEFAULT_DATA_MISSING_WARNING_PCT))
    if data_missing_rate > critical_threshold:
        severity = _raise(severity, "ERROR")
        reasons.append("DATA_MISSING_RATE_CRITICAL")
    elif data_missing_rate > warning_threshold:
        severity = _raise(severity, "WARNING")
        reasons.append("DATA_MISSING_RATE_WARNING")

    buy = payload.get("buy") if isinstance(payload.get("buy"), dict) else {}
    conflict_blocked = int((buy or {}).get("conflict_blocked") or 0)
    if conflict_blocked >= 2:
        severity = _raise(severity, "WARNING")
        reasons.append("CONFLICT_BLOCKED_ELEVATED")

    for warning in validation_result.get("warnings") or []:
        code = str(warning)
        if code == "PAPER_TRADING_NOTICE_MISSING":
            severity = _raise(severity, "WARNING")
            reasons.append(code)
        elif code == "PAYLOAD_SENSITIVE_FIELD_FOUND":
            severity = _raise(severity, "WARNING")
            reasons.append(code)

    if not reasons:
        reasons.append("NORMAL_OPERATION")

    # Keep unique order.
    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    return {"severity": severity, "reasons": unique_reasons}
