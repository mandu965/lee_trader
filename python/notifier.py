from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any
from urllib import error, request


LEVEL_INFO = "INFO"
LEVEL_WARNING = "WARNING"
LEVEL_CRITICAL = "CRITICAL"
VALID_LEVELS = {LEVEL_INFO, LEVEL_WARNING, LEVEL_CRITICAL}


def _normalize_level(level: str) -> str:
    value = str(level or LEVEL_INFO).strip().upper() or LEVEL_INFO
    return value if value in VALID_LEVELS else LEVEL_INFO


def _render_details(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    lines: list[str] = []
    for key, value in details.items():
        if value is None:
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _render_message(level: str, title: str, message: str, details: dict[str, Any] | None = None) -> str:
    lines = [
        f"[{level}] {title}",
        str(message or "").strip() or "-",
        f"- occurred_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    detail_text = _render_details(details)
    if detail_text:
        lines.append(detail_text)
    return "\n".join(lines)


def notify(level: str, title: str, message: str, details: dict[str, Any] | None = None) -> bool:
    normalized_level = _normalize_level(level)
    rendered = _render_message(normalized_level, title, message, details)
    webhook_url = str(os.environ.get("SLACK_WEBHOOK_URL", "")).strip()

    try:
        if not webhook_url:
            logging.warning("[ALERT-FALLBACK]\n%s", rendered)
            return True

        payload = json.dumps({"text": rendered}, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with request.urlopen(req, timeout=10) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            if int(status_code) >= 400:
                raise RuntimeError(f"slack webhook responded with status={status_code}")
        return True
    except (error.URLError, error.HTTPError, TimeoutError, RuntimeError, ValueError):
        logging.warning("Notifier delivery failed", exc_info=True)
    except Exception:
        logging.warning("Notifier crashed unexpectedly", exc_info=True)

    try:
        logging.warning("[ALERT-DROPPED]\n%s", rendered)
    except Exception:
        pass
    return False


def notify_info(title: str, message: str, details: dict[str, Any] | None = None) -> bool:
    return notify(LEVEL_INFO, title, message, details)


def notify_warning(title: str, message: str, details: dict[str, Any] | None = None) -> bool:
    return notify(LEVEL_WARNING, title, message, details)


def notify_critical(title: str, message: str, details: dict[str, Any] | None = None) -> bool:
    return notify(LEVEL_CRITICAL, title, message, details)
