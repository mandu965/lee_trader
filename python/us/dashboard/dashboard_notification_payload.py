from __future__ import annotations

import json
from pathlib import Path

from python.us.dashboard.config import DashboardConfig


def write_dashboard_notification_payloads(
    cfg: DashboardConfig,
    *,
    trade_date: str,
    text_payload: str,
    json_payload: dict[str, object],
    formats: tuple[str, ...] | None = None,
) -> dict[str, str]:
    output_dir = cfg.output_dir / "notifications"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_formats = formats or cfg.notification_formats

    text_path = output_dir / f"{trade_date}_notification.txt"
    json_path = output_dir / f"{trade_date}_notification.json"
    latest_text_path = output_dir / "latest_notification.txt"
    latest_json_path = output_dir / "latest_notification.json"
    result: dict[str, str] = {}

    if "text" in selected_formats:
        text_path.write_text(text_payload, encoding="utf-8")
        latest_text_path.write_text(text_payload, encoding="utf-8")
        result["text_path"] = str(text_path)
        result["latest_text_path"] = str(latest_text_path)

    if "json" in selected_formats:
        json_text = json.dumps(json_payload, ensure_ascii=False, indent=2, default=str)
        json_path.write_text(json_text, encoding="utf-8")
        latest_json_path.write_text(json_text, encoding="utf-8")
        result["json_path"] = str(json_path)
        result["latest_json_path"] = str(latest_json_path)

    return result
