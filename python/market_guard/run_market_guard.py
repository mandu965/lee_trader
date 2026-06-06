"""시장 급락 감시 서비스 — 1회 실행형.

스케줄러에서 하루 2회 호출:
  09:10 (오전 BUY 전 점검) / 15:40 (장마감 후 다음 거래일 보호)

감지 조건 충족 시:
  - KR: data/market_guard_kill.json 파일 생성 → common_live_risk_guard.py가 읽어 매수 차단
  - US: activate_kill_switch() 호출 → US AI 매수 차단
  - 텔레그램 알람 발송

조건 해제 시:
  - 파일 삭제 + US kill switch 해제 + 알람
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from market_guard.config import load_config
from market_guard.detector import data_quality_to_dict, evaluate, evaluate_data_quality, get_snapshot
from market_guard.kill_switch import activate_kr, deactivate_kr, is_kr_active, activate_us, deactivate_us
from market_guard.notifier import send_alert, send_recovery


def setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def main() -> int:
    cfg = load_config()
    setup_logging(cfg.log_path)
    logger = logging.getLogger("market_guard.main")

    logger.info("=" * 60)
    logger.info("MARKET_GUARD START | enabled=%s dry_run=%s", cfg.enabled, cfg.dry_run)

    if not cfg.enabled:
        logger.info("MARKET_GUARD_ENABLED=0 → 종료")
        return 0

    # 1. 시장 스냅샷 수집
    try:
        snapshot = get_snapshot()
    except Exception as exc:
        logger.error("시장 데이터 수집 실패 → 종료: %s", exc)
        return 1

    # 2. 감지 평가
    result = evaluate(
        snapshot,
        kospi_1d_threshold=cfg.kospi_1d_threshold,
        kospi_5d_threshold=cfg.kospi_5d_threshold,
        kospi_10d_threshold=cfg.kospi_10d_threshold,
        recovery_1d_threshold=cfg.recovery_1d_threshold,
        recovery_5d_threshold=cfg.recovery_5d_threshold,
    )
    data_quality = evaluate_data_quality(snapshot)
    logger.info(
        "DATA_QUALITY | status=%s can_activate=%s warnings=%s",
        data_quality.status,
        data_quality.can_activate,
        data_quality.warnings,
    )

    currently_active = is_kr_active(cfg.guard_file_path)

    # 3. CRITICAL → 차단 활성화
    if result.alert_level == "CRITICAL":
        guard_reason = "; ".join(result.triggered_conditions)
        if not data_quality.can_activate:
            logger.warning(
                "CRITICAL detected but kill switch activation skipped by data quality guard: %s",
                "; ".join(data_quality.warnings),
            )
            kr_ok = False
            us_ok = False
        else:
            kr_ok = activate_kr(cfg.guard_file_path, reason=guard_reason, dry_run=cfg.dry_run)
            us_ok = activate_us(cfg.us_account_id, reason=guard_reason, dry_run=cfg.dry_run)
        send_alert(
            level="CRITICAL",
            summary=result.summary,
            conditions=[
                *result.triggered_conditions,
                *([f"DATA_QUALITY={data_quality.status}: " + "; ".join(data_quality.warnings)] if data_quality.warnings else []),
            ],
            kr_activated=kr_ok,
            us_activated=us_ok,
            dry_run=cfg.dry_run,
            telegram_token=cfg.telegram_token,
            telegram_chat_id=cfg.telegram_chat_id,
        )

    # 4. WARNING → 알람만
    elif result.alert_level == "WARNING":
        send_alert(
            level="WARNING",
            summary=result.summary,
            conditions=result.triggered_conditions,
            kr_activated=False,
            us_activated=False,
            dry_run=cfg.dry_run,
            telegram_token=cfg.telegram_token,
            telegram_chat_id=cfg.telegram_chat_id,
        )

    # 5. NONE + 이전에 활성화돼 있었으면 → 해제
    elif result.alert_level == "NONE" and currently_active:
        deactivate_kr(cfg.guard_file_path, dry_run=cfg.dry_run)
        deactivate_us(cfg.us_account_id, dry_run=cfg.dry_run)
        send_recovery(
            summary=result.summary,
            dry_run=cfg.dry_run,
            telegram_token=cfg.telegram_token,
            telegram_chat_id=cfg.telegram_chat_id,
        )

    # 6. 상태 기록 (outputs/)
    status_path = Path("outputs/market_guard_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({
            "trade_date": str(snapshot.trade_date),
            "kospi_close": snapshot.kospi_close,
            "ret_1d": snapshot.ret_1d,
            "ret_5d": snapshot.ret_5d,
            "ret_10d": snapshot.ret_10d,
            "alert_level": result.alert_level,
            "triggered_conditions": result.triggered_conditions,
            "kill_switch_active": is_kr_active(cfg.guard_file_path),
            "dry_run": cfg.dry_run,
            "data_quality": data_quality_to_dict(data_quality),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("MARKET_GUARD END | level=%s", result.alert_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
