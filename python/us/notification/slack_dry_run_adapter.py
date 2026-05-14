from __future__ import annotations

from python.us.notification.config import NotificationConfig


EMOJI = {"INFO": ":information_source:", "WARNING": ":warning:", "ERROR": ":x:", "CRITICAL": ":rotating_light:"}


def run_slack_dry_run_adapter(cfg: NotificationConfig, payload: dict[str, object], *, severity: str) -> dict[str, object]:
    warnings: list[str] = []
    if not cfg.slack_channel:
        warnings.append("SLACK_CHANNEL_MISSING")
    buy = payload.get("buy") if isinstance(payload.get("buy"), dict) else {}
    sell = payload.get("sell") if isinstance(payload.get("sell"), dict) else {}
    text = (
        f"{EMOJI.get(severity, ':information_source:')} US Paper Trading Dashboard - {severity}\n"
        f"Date: {payload.get('trade_date')}\n"
        f"BUY allowed: {buy.get('final_allowed')} / SELL signals: {sell.get('sell_signals')} / "
        f"Review required: {sell.get('review_required')}\n"
        f"{payload.get('notice') or 'Paper Trading only. No live orders were executed.'}"
    )
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*US Paper Trading Dashboard* - {severity}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
    ]
    return {
        "channel": "SLACK_DRY_RUN",
        "status": "SUCCESS",
        "dry_run": True,
        "slack_channel": cfg.slack_channel,
        "username": cfg.slack_username,
        "text": text,
        "blocks": blocks,
        "warnings": warnings,
    }
