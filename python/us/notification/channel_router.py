from __future__ import annotations

from python.us.notification.config import NotificationConfig
from python.us.notification.console_adapter import render_console_notification, run_console_adapter
from python.us.notification.email_dry_run_adapter import run_email_dry_run_adapter
from python.us.notification.file_adapter import write_notification_adapter_files
from python.us.notification.manual_approval import build_manual_approval
from python.us.notification.notification_logger import (
    log_notification_approval,
    log_notification_delivery,
    log_notification_event,
)
from python.us.notification.severity_policy import determine_notification_severity
from python.us.notification.slack_dry_run_adapter import run_slack_dry_run_adapter


def run_notification_channels(
    cfg: NotificationConfig,
    *,
    payload_result: dict[str, object],
    channels_override: tuple[str, ...] | None = None,
    emit_console: bool = True,
) -> dict[str, object]:
    payload = payload_result.get("payload")
    if not isinstance(payload, dict):
        return {
            "notification_executed": False,
            "mode": cfg.mode,
            "severity": "ERROR",
            "channels": {},
            "live_channels_blocked": [],
            "pipeline_should_fail": cfg.fail_pipeline_on_error,
            "approval_required": False,
            "approval_status": None,
            "warnings": list(payload_result.get("warnings") or []),
            "errors": list(payload_result.get("errors") or ["PAYLOAD_MISSING"]),
        }

    if cfg.mode == "LIVE":
        return {
            "notification_executed": False,
            "mode": cfg.mode,
            "severity": "ERROR",
            "channels": {},
            "live_channels_blocked": [{"channel": "LIVE", "status": "BLOCKED", "reason": "LIVE_NOTIFICATION_NOT_IMPLEMENTED"}],
            "pipeline_should_fail": cfg.fail_pipeline_on_error,
            "approval_required": False,
            "approval_status": None,
            "warnings": list(payload_result.get("warnings") or []),
            "errors": ["LIVE_NOTIFICATION_NOT_IMPLEMENTED"],
            "payload": payload,
        }

    severity_result = determine_notification_severity(payload, validation_result=payload_result, cfg=cfg)
    severity = str(severity_result.get("severity") or "INFO")
    channels = channels_override or cfg.channels
    trade_date = str(payload.get("trade_date") or "unknown")
    approval_record = build_manual_approval(cfg, trade_date=trade_date, severity=severity, payload=payload)
    log_notification_event(
        cfg,
        notification_event_id=str(approval_record["notification_event_id"]),
        trade_date=trade_date,
        payload=payload,
        severity=severity,
        mode=cfg.mode,
        approval_required=bool(approval_record.get("approval_required")),
        approval_status=approval_record.get("approval_status"),
    )
    if approval_record.get("approval_required"):
        log_notification_approval(cfg, approval_record)

    warnings = list(payload_result.get("warnings") or [])
    warnings.extend(str(item) for item in severity_result.get("reasons") or [] if str(item).endswith("_WARNING"))
    errors = list(payload_result.get("errors") or [])
    channel_results: dict[str, dict[str, object]] = {}
    live_channels_blocked: list[dict[str, object]] = []

    text_summary = render_console_notification(payload, severity=severity)
    file_requested = "FILE" in channels
    execution_channels = tuple(channel for channel in channels if channel != "FILE")

    for channel in execution_channels:
        if channel == "CONSOLE":
            if not cfg.console_enabled:
                channel_results[channel] = {"channel": channel, "status": "SKIPPED", "reason": "CHANNEL_DISABLED"}
            else:
                channel_results[channel] = run_console_adapter(payload, severity=severity, emit_console=emit_console)
        elif channel == "EMAIL_DRY_RUN":
            if not cfg.email_dry_run_enabled:
                channel_results[channel] = {"channel": channel, "status": "SKIPPED", "reason": "CHANNEL_DISABLED"}
            else:
                channel_results[channel] = run_email_dry_run_adapter(cfg, payload, severity=severity)
        elif channel == "SLACK_DRY_RUN":
            if not cfg.slack_dry_run_enabled:
                channel_results[channel] = {"channel": channel, "status": "SKIPPED", "reason": "CHANNEL_DISABLED"}
            else:
                channel_results[channel] = run_slack_dry_run_adapter(cfg, payload, severity=severity)
        elif channel in {"EMAIL_LIVE", "SLACK_LIVE"}:
            blocked = {"channel": channel, "status": "BLOCKED", "reason": "LIVE_NOTIFICATION_NOT_IMPLEMENTED"}
            live_channels_blocked.append(blocked)
            channel_results[channel] = blocked
            errors.append("LIVE_NOTIFICATION_NOT_IMPLEMENTED")
        else:
            channel_results[channel] = {"channel": channel, "status": "SKIPPED", "reason": "CHANNEL_UNSUPPORTED"}

        warnings.extend(str(item) for item in channel_results[channel].get("warnings") or [])
        errors.extend(str(item) for item in channel_results[channel].get("errors") or [])
        log_notification_delivery(
            cfg,
            notification_event_id=str(approval_record["notification_event_id"]),
            trade_date=trade_date,
            channel_result=channel_results[channel],
            severity=severity,
            mode=cfg.mode,
            approval_required=bool(approval_record.get("approval_required")),
            approval_status=approval_record.get("approval_status"),
            payload=payload,
        )

    if file_requested:
        if not cfg.file_enabled:
            channel_results["FILE"] = {"channel": "FILE", "status": "SKIPPED", "reason": "CHANNEL_DISABLED"}
        else:
            try:
                channel_results["FILE"] = write_notification_adapter_files(
                    cfg,
                    trade_date=trade_date,
                    text_summary=text_summary,
                    payload=payload,
                    severity=severity,
                    channel_results=channel_results,
                    approval_record=approval_record,
                    warnings=warnings,
                    errors=errors,
                )
            except OSError:
                channel_results["FILE"] = {"channel": "FILE", "status": "ERROR", "errors": ["FILE_WRITE_FAILED"]}
        warnings.extend(str(item) for item in channel_results["FILE"].get("warnings") or [])
        errors.extend(str(item) for item in channel_results["FILE"].get("errors") or [])
        log_notification_delivery(
            cfg,
            notification_event_id=str(approval_record["notification_event_id"]),
            trade_date=trade_date,
            channel_result=channel_results["FILE"],
            severity=severity,
            mode=cfg.mode,
            approval_required=bool(approval_record.get("approval_required")),
            approval_status=approval_record.get("approval_status"),
            payload=payload,
        )

    if cfg.email_live_enabled and "EMAIL_LIVE" not in channel_results:
        live_channels_blocked.append({"channel": "EMAIL_LIVE", "status": "BLOCKED", "reason": "LIVE_NOTIFICATION_NOT_IMPLEMENTED"})
        errors.append("LIVE_NOTIFICATION_NOT_IMPLEMENTED")
    if cfg.slack_live_enabled and "SLACK_LIVE" not in channel_results:
        live_channels_blocked.append({"channel": "SLACK_LIVE", "status": "BLOCKED", "reason": "LIVE_NOTIFICATION_NOT_IMPLEMENTED"})
        errors.append("LIVE_NOTIFICATION_NOT_IMPLEMENTED")

    return {
        "notification_executed": True,
        "mode": cfg.mode,
        "severity": severity,
        "severity_reasons": list(severity_result.get("reasons") or []),
        "channels": channel_results,
        "live_channels_blocked": live_channels_blocked,
        "notification_event_id": approval_record.get("notification_event_id"),
        "approval_required": bool(approval_record.get("approval_required")),
        "approval_status": approval_record.get("approval_status"),
        "approval_path": approval_record.get("approval_path"),
        "pipeline_should_fail": bool(errors) and cfg.fail_pipeline_on_error,
        "warnings": warnings,
        "errors": errors,
        "payload": payload,
    }
