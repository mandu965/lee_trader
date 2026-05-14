from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ALLOWED_CHANNELS = {
    "FILE",
    "CONSOLE",
    "EMAIL_DRY_RUN",
    "SLACK_DRY_RUN",
    "EMAIL_LIVE",
    "SLACK_LIVE",
}
SAFE_CHANNELS = {"FILE", "CONSOLE", "EMAIL_DRY_RUN", "SLACK_DRY_RUN"}
ALLOWED_MODES = {"DISABLED", "DRY_RUN", "MANUAL_APPROVAL", "LIVE"}


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _raw(name: str, default: str | None = None) -> str:
    return str(os.environ.get(name, default or "")).strip()


def _safe_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(_raw(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _resolve_dir(root_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    return path if path.is_absolute() else root_dir / path


def _parse_channels(raw_value: str, warnings: list[str]) -> tuple[str, ...]:
    values = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
    if not values:
        warnings.append("US_NOTIFICATION_CHANNELS invalid; fallback=FILE,CONSOLE")
        return ("CONSOLE", "FILE")
    normalized: list[str] = []
    for value in values:
        if value not in ALLOWED_CHANNELS:
            warnings.append(f"US_NOTIFICATION_CHANNELS unsupported channel ignored: {value}")
            continue
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        warnings.append("US_NOTIFICATION_CHANNELS resolved empty; fallback=FILE,CONSOLE")
        return ("CONSOLE", "FILE")
    return tuple(normalized)


@dataclass(frozen=True)
class NotificationConfig:
    root_dir: Path
    enabled: bool
    mode: str
    channels: tuple[str, ...]
    require_manual_approval: bool
    fail_pipeline_on_error: bool
    file_enabled: bool
    console_enabled: bool
    email_dry_run_enabled: bool
    email_recipients: tuple[str, ...]
    email_subject_prefix: str
    slack_dry_run_enabled: bool
    slack_channel: str
    slack_username: str
    email_live_enabled: bool
    slack_live_enabled: bool
    include_paper_trading_notice: bool
    include_live_disabled_notice: bool
    max_symbols: int
    redact_sensitive_fields: bool
    output_dir: Path
    payload_dir: Path
    approvals_dir: Path
    logs_dir: Path
    warnings: tuple[str, ...]


def load_notification_config() -> NotificationConfig:
    root_dir = Path(__file__).resolve().parents[3]
    warnings: list[str] = []
    mode = (_raw("US_NOTIFICATION_ADAPTER_MODE", "DRY_RUN") or "DRY_RUN").upper()
    if mode not in ALLOWED_MODES:
        warnings.append(f"US_NOTIFICATION_ADAPTER_MODE invalid: {mode}; fallback=DRY_RUN")
        mode = "DRY_RUN"

    output_dir = _resolve_dir(root_dir, "reports/lee_trader_us/notification")
    payload_dir = _resolve_dir(root_dir, "reports/lee_trader_us/dashboard/notifications")
    approvals_dir = output_dir / "approvals"
    logs_dir = output_dir / "logs"

    channels = _parse_channels(_raw("US_NOTIFICATION_CHANNELS", "FILE,CONSOLE") or "FILE,CONSOLE", warnings)
    recipients = tuple(item.strip() for item in _raw("US_NOTIFICATION_EMAIL_RECIPIENTS", "").split(",") if item.strip())

    output_dir.mkdir(parents=True, exist_ok=True)
    approvals_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return NotificationConfig(
        root_dir=root_dir,
        enabled=_flag("US_NOTIFICATION_ADAPTER_ENABLED", "0"),
        mode=mode,
        channels=channels,
        require_manual_approval=_flag("US_NOTIFICATION_REQUIRE_MANUAL_APPROVAL", "1"),
        fail_pipeline_on_error=_flag("US_NOTIFICATION_FAIL_PIPELINE_ON_ERROR", "0"),
        file_enabled=_flag("US_NOTIFICATION_FILE_ENABLED", "1"),
        console_enabled=_flag("US_NOTIFICATION_CONSOLE_ENABLED", "1"),
        email_dry_run_enabled=_flag("US_NOTIFICATION_EMAIL_DRY_RUN_ENABLED", "0"),
        email_recipients=recipients,
        email_subject_prefix=_raw("US_NOTIFICATION_EMAIL_SUBJECT_PREFIX", "[US Paper Trading]") or "[US Paper Trading]",
        slack_dry_run_enabled=_flag("US_NOTIFICATION_SLACK_DRY_RUN_ENABLED", "0"),
        slack_channel=_raw("US_NOTIFICATION_SLACK_CHANNEL", ""),
        slack_username=_raw("US_NOTIFICATION_SLACK_USERNAME", "LeeTraderBot") or "LeeTraderBot",
        email_live_enabled=_flag("US_NOTIFICATION_EMAIL_LIVE_ENABLED", "0"),
        slack_live_enabled=_flag("US_NOTIFICATION_SLACK_LIVE_ENABLED", "0"),
        include_paper_trading_notice=_flag("US_NOTIFICATION_INCLUDE_PAPER_TRADING_NOTICE", "1"),
        include_live_disabled_notice=_flag("US_NOTIFICATION_INCLUDE_LIVE_DISABLED_NOTICE", "1"),
        max_symbols=_safe_int("US_NOTIFICATION_MAX_SYMBOLS", 10, minimum=1),
        redact_sensitive_fields=_flag("US_NOTIFICATION_REDACT_SENSITIVE_FIELDS", "1"),
        output_dir=output_dir,
        payload_dir=payload_dir,
        approvals_dir=approvals_dir,
        logs_dir=logs_dir,
        warnings=tuple(warnings),
    )
