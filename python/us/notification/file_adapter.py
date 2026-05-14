from __future__ import annotations

import json

from python.us.notification.config import NotificationConfig


def write_notification_adapter_files(
    cfg: NotificationConfig,
    *,
    trade_date: str,
    text_summary: str,
    payload: dict[str, object],
    severity: str,
    channel_results: dict[str, object],
    approval_record: dict[str, object],
    warnings: list[str],
    errors: list[str],
) -> dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    text_path = cfg.output_dir / f"{trade_date}_notification_adapter.txt"
    json_path = cfg.output_dir / f"{trade_date}_notification_adapter.json"
    latest_text_path = cfg.output_dir / "latest_notification_adapter.txt"
    latest_json_path = cfg.output_dir / "latest_notification_adapter.json"

    payload_wrapper = {
        "trade_date": trade_date,
        "severity": severity,
        "payload": payload,
        "channel_results": channel_results,
        "approval": approval_record,
        "warnings": warnings,
        "errors": errors,
    }
    text_path.write_text(text_summary, encoding="utf-8")
    latest_text_path.write_text(text_summary, encoding="utf-8")
    json_text = json.dumps(payload_wrapper, ensure_ascii=False, indent=2, default=str)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")
    return {
        "channel": "FILE",
        "status": "SUCCESS",
        "path": str(json_path),
        "text_path": str(text_path),
        "latest_path": str(latest_json_path),
        "latest_text_path": str(latest_text_path),
    }
