from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from python.us.notification.config import NotificationConfig


def build_manual_approval(cfg: NotificationConfig, *, trade_date: str, severity: str, payload: dict[str, object]) -> dict[str, object]:
    notification_event_id = f"USNOTI_{trade_date.replace('-', '')}_{uuid.uuid4().hex[:10].upper()}"
    approval_required = cfg.require_manual_approval or cfg.mode == "MANUAL_APPROVAL"
    approval_status = "PENDING" if approval_required else None
    record = {
        "notification_event_id": notification_event_id,
        "trade_date": trade_date,
        "approval_required": approval_required,
        "approval_status": approval_status,
        "severity": severity,
        "message_type": payload.get("message_type"),
        "mode": cfg.mode,
        "note": "Notification delivery approval only. This is not live trading approval.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths: dict[str, str] = {}
    if approval_required:
        cfg.approvals_dir.mkdir(parents=True, exist_ok=True)
        date_path = cfg.approvals_dir / f"{trade_date}_approval_pending.json"
        latest_path = cfg.approvals_dir / "latest_approval_pending.json"
        text = json.dumps(record, ensure_ascii=False, indent=2, default=str)
        date_path.write_text(text, encoding="utf-8")
        latest_path.write_text(text, encoding="utf-8")
        paths = {"approval_path": str(date_path), "latest_approval_path": str(latest_path)}
    record.update(paths)
    return record
