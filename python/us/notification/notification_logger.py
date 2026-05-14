from __future__ import annotations

from datetime import datetime, timezone
import json

from python.us.notification.config import NotificationConfig


def _append_jsonl(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def log_notification_event(
    cfg: NotificationConfig,
    *,
    notification_event_id: str,
    trade_date: str,
    payload: dict[str, object],
    severity: str,
    mode: str,
    approval_required: bool,
    approval_status: str | None,
) -> None:
    record = {
        "notification_event_id": notification_event_id,
        "trade_date": trade_date,
        "message_type": payload.get("message_type"),
        "severity": severity,
        "mode": mode,
        "paper_trading_only": payload.get("paper_trading_only"),
        "live_orders_executed": payload.get("live_orders_executed"),
        "approval_required": approval_required,
        "approval_status": approval_status,
        "payload_json": payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(cfg.logs_dir / "notification_event_log.jsonl", record)


def log_notification_delivery(
    cfg: NotificationConfig,
    *,
    notification_event_id: str,
    trade_date: str,
    channel_result: dict[str, object],
    severity: str,
    mode: str,
    approval_required: bool,
    approval_status: str | None,
    payload: dict[str, object],
) -> None:
    record = {
        "notification_event_id": notification_event_id,
        "trade_date": trade_date,
        "message_type": payload.get("message_type"),
        "severity": severity,
        "mode": mode,
        "paper_trading_only": payload.get("paper_trading_only"),
        "live_orders_executed": payload.get("live_orders_executed"),
        "channel": channel_result.get("channel"),
        "delivery_mode": mode,
        "approval_required": approval_required,
        "approval_status": approval_status,
        "delivery_status": channel_result.get("status"),
        "payload_json": channel_result,
        "message_text": channel_result.get("text") or channel_result.get("plain_text_body"),
        "error_message": ",".join(str(item) for item in channel_result.get("errors") or []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(cfg.logs_dir / "notification_delivery_log.jsonl", record)


def log_notification_approval(cfg: NotificationConfig, approval_record: dict[str, object]) -> None:
    record = dict(approval_record)
    record["created_at"] = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    _append_jsonl(cfg.logs_dir / "notification_approval_log.jsonl", record)
