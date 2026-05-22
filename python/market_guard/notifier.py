"""알람 발송 — 텔레그램 우선, 없으면 로그로 대체."""
from __future__ import annotations

import logging
import urllib.request
import urllib.parse
import json

LOGGER = logging.getLogger("market_guard.notifier")


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        LOGGER.warning("텔레그램 발송 실패: %s", exc)
        return False


def send_alert(
    *,
    level: str,
    summary: str,
    conditions: list[str],
    kr_activated: bool,
    us_activated: bool,
    dry_run: bool,
    telegram_token: str,
    telegram_chat_id: str,
) -> None:
    emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "NONE": "✅"}.get(level, "ℹ️")
    dry_tag = "[DRY_RUN] " if dry_run else ""

    lines = [
        f"{emoji} *MarketGuard {dry_tag}{level}*",
        f"`{summary}`",
    ]
    if conditions:
        lines.append("*트리거 조건:*")
        for c in conditions:
            lines.append(f"• {c}")

    if level == "CRITICAL":
        lines.append("")
        lines.append(f"KR kill switch: {'활성화 ✅' if kr_activated else '스킵'}")
        lines.append(f"US kill switch: {'활성화 ✅' if us_activated else '스킵'}")
        if not dry_run:
            lines.append("")
            lines.append("⚡ KR AI 신규 매수 차단됨. 포지션 수동 확인 요망.")

    message = "\n".join(lines)

    # 텔레그램 발송 시도
    sent = _send_telegram(telegram_token, telegram_chat_id, message)

    # 항상 로그에도 기록
    log_fn = LOGGER.critical if level == "CRITICAL" else (LOGGER.warning if level == "WARNING" else LOGGER.info)
    log_fn("ALERT | level=%s telegram_sent=%s | %s", level, sent, summary)
    if not sent:
        LOGGER.info("텔레그램 미설정 또는 발송 실패 — 로그에만 기록됨")


def send_recovery(
    *,
    summary: str,
    dry_run: bool,
    telegram_token: str,
    telegram_chat_id: str,
) -> None:
    dry_tag = "[DRY_RUN] " if dry_run else ""
    message = f"✅ *MarketGuard {dry_tag}RECOVERY*\n`{summary}`\nKill switch 해제됨."
    sent = _send_telegram(telegram_token, telegram_chat_id, message)
    LOGGER.info("RECOVERY | telegram_sent=%s | %s", sent, summary)
