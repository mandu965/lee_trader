from __future__ import annotations

import json
from pathlib import Path

from python.us.notification.config import NotificationConfig


REQUIRED_FIELDS = (
    "message_type",
    "trade_date",
    "generated_at",
    "mode",
    "status",
    "paper_trading_only",
    "live_orders_executed",
    "notice",
)
SENSITIVE_KEY_PARTS = ("api_key", "secret", "token", "webhook", "account_number", "broker_account", "password")


def _collect_sensitive_paths(value: object, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            current = f"{prefix}.{key_text}" if prefix else key_text
            lower_key = key_text.lower()
            if any(part in lower_key for part in SENSITIVE_KEY_PARTS):
                paths.append(current)
            paths.extend(_collect_sensitive_paths(nested, prefix=current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            current = f"{prefix}[{index}]"
            paths.extend(_collect_sensitive_paths(item, prefix=current))
    return paths


def _redact_sensitive_fields(value: object) -> object:
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, nested in value.items():
            lower_key = str(key).lower()
            if any(part in lower_key for part in SENSITIVE_KEY_PARTS):
                redacted[str(key)] = "REDACTED"
            else:
                redacted[str(key)] = _redact_sensitive_fields(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value


def _notification_path(cfg: NotificationConfig, trade_date: str | None) -> Path:
    if trade_date:
        return cfg.payload_dir / f"{trade_date}_notification.json"
    return cfg.payload_dir / "latest_notification.json"


def load_notification_payload(cfg: NotificationConfig, *, trade_date: str | None = None) -> dict[str, object]:
    path = _notification_path(cfg, trade_date)
    warnings: list[str] = []
    errors: list[str] = []
    severity_hint = "INFO"
    payload: dict[str, object] | None = None
    redacted_fields: list[str] = []

    if not path.exists():
        return {
            "payload": None,
            "valid": False,
            "warnings": warnings,
            "errors": ["PAYLOAD_MISSING"],
            "severity_hint": "ERROR",
            "source_path": str(path),
            "redacted_fields": redacted_fields,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "payload": None,
            "valid": False,
            "warnings": warnings,
            "errors": ["PAYLOAD_INVALID"],
            "severity_hint": "ERROR",
            "source_path": str(path),
            "redacted_fields": redacted_fields,
        }

    missing_required = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_required:
        errors.append("PAYLOAD_INVALID")
        warnings.append(f"MISSING_FIELDS:{','.join(missing_required)}")
        severity_hint = "ERROR"

    if payload.get("paper_trading_only") is not True:
        errors.append("PAPER_TRADING_ONLY_FALSE")
        severity_hint = "CRITICAL"

    if payload.get("live_orders_executed") is not False:
        errors.append("LIVE_ORDERS_EXECUTED_TRUE")
        severity_hint = "CRITICAL"

    notice = str(payload.get("notice") or "")
    if "Paper Trading" not in notice:
        warnings.append("PAPER_TRADING_NOTICE_MISSING")
        if severity_hint == "INFO":
            severity_hint = "WARNING"

    redacted_fields = _collect_sensitive_paths(payload)
    if redacted_fields:
        warnings.append("PAYLOAD_SENSITIVE_FIELD_FOUND")
        if cfg.redact_sensitive_fields:
            payload = _redact_sensitive_fields(payload)  # type: ignore[assignment]

    return {
        "payload": payload,
        "valid": not errors,
        "warnings": warnings,
        "errors": errors,
        "severity_hint": severity_hint,
        "source_path": str(path),
        "redacted_fields": redacted_fields,
    }
