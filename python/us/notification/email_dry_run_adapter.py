from __future__ import annotations

from python.us.notification.config import NotificationConfig
from python.us.notification.console_adapter import render_console_notification


def run_email_dry_run_adapter(cfg: NotificationConfig, payload: dict[str, object], *, severity: str) -> dict[str, object]:
    warnings: list[str] = []
    if not cfg.email_recipients:
        warnings.append("EMAIL_RECIPIENT_MISSING")
    subject = f"{cfg.email_subject_prefix}[{severity}] {payload.get('trade_date')} Dashboard Summary"
    plain_text = render_console_notification(payload, severity=severity)
    markdown_body = plain_text.replace("\n", "  \n")
    links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
    attachment_candidates = [value for value in [links.get("dashboard_json"), links.get("dashboard_markdown")] if value]
    return {
        "channel": "EMAIL_DRY_RUN",
        "status": "SUCCESS",
        "dry_run": True,
        "recipients": list(cfg.email_recipients),
        "subject": subject,
        "plain_text_body": plain_text,
        "markdown_body": markdown_body,
        "attachment_candidates": attachment_candidates,
        "warnings": warnings,
    }
