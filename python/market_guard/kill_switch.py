"""Kill switch 활성화/해제.

KR: data/market_guard_kill.json 파일 생성/삭제
US: us_live_kill_switch DB 테이블 (activate_kill_switch 함수 재사용)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("market_guard.kill_switch")

US_KILL_SWITCH_ID = "MARKET_GUARD_AUTO"


def activate_kr(guard_file_path: str, reason: str, dry_run: bool) -> bool:
    """KR kill switch 파일 생성."""
    path = Path(guard_file_path)
    payload = {
        "active": True,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    if dry_run:
        LOGGER.info("[DRY_RUN] KR kill switch 파일 생성 스킵: %s | reason=%s", path, reason)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.warning("KR_KILL_SWITCH ACTIVATED | file=%s reason=%s", path, reason)
    return True


def deactivate_kr(guard_file_path: str, dry_run: bool) -> bool:
    """KR kill switch 파일 삭제."""
    path = Path(guard_file_path)
    if not path.exists():
        return False
    if dry_run:
        LOGGER.info("[DRY_RUN] KR kill switch 파일 삭제 스킵: %s", path)
        return False

    path.unlink(missing_ok=True)
    LOGGER.info("KR_KILL_SWITCH DEACTIVATED | file=%s", path)
    return True


def is_kr_active(guard_file_path: str) -> bool:
    return Path(guard_file_path).exists()


def activate_us(account_id: str, reason: str, dry_run: bool) -> bool:
    """US kill switch DB 활성화."""
    if not account_id:
        LOGGER.info("US_ACCOUNT_ID 미설정, US kill switch 스킵")
        return False
    if dry_run:
        LOGGER.info("[DRY_RUN] US kill switch 활성화 스킵: account=%s reason=%s", account_id, reason)
        return False

    try:
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).resolve().parents[2] / "python"))
        from python.utils.us_live_kill_switch import activate_kill_switch
        activate_kill_switch(
            kill_switch_id=US_KILL_SWITCH_ID,
            scope="ACCOUNT",
            target_value=account_id,
            reason=reason,
        )
        LOGGER.warning("US_KILL_SWITCH ACTIVATED | account=%s reason=%s", account_id, reason)
        return True
    except Exception as exc:
        LOGGER.error("US kill switch 활성화 실패 (무시하고 계속): %s", exc)
        return False


def deactivate_us(account_id: str, dry_run: bool) -> bool:
    """US kill switch DB 해제."""
    if not account_id:
        return False
    if dry_run:
        LOGGER.info("[DRY_RUN] US kill switch 해제 스킵: account=%s", account_id)
        return False

    try:
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).resolve().parents[2] / "python"))
        from python.utils.us_live_kill_switch import clear_kill_switch
        clear_kill_switch(kill_switch_id=US_KILL_SWITCH_ID)
        LOGGER.info("US_KILL_SWITCH DEACTIVATED | account=%s", account_id)
        return True
    except Exception as exc:
        LOGGER.error("US kill switch 해제 실패 (무시하고 계속): %s", exc)
        return False
