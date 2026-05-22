from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class MarketGuardConfig:
    enabled: bool
    dry_run: bool                    # True: 감지만, 실제 차단 안 함
    kospi_1d_threshold: float        # 당일 하락 임계 (기본 -3%)
    kospi_5d_threshold: float        # 5일 누적 하락 임계 (기본 -7%)
    kospi_10d_threshold: float       # 10일 누적 하락 임계 (경고, 기본 -10%)
    recovery_1d_threshold: float     # 해제 기준: 당일 반등 (기본 +1%)
    recovery_5d_threshold: float     # 해제 기준: 5일 누적 (기본 -3% 이내)
    guard_file_path: str             # KR kill switch 파일 경로
    telegram_token: str
    telegram_chat_id: str
    log_path: str
    # US kill switch 계정 (비면 US kill switch 비활성)
    us_account_id: str


def load_config() -> MarketGuardConfig:
    return MarketGuardConfig(
        enabled=_flag("MARKET_GUARD_ENABLED", "0"),
        dry_run=_flag("MARKET_GUARD_DRY_RUN", "1"),
        kospi_1d_threshold=_float("MARKET_GUARD_KOSPI_1D_THRESHOLD", -0.03),
        kospi_5d_threshold=_float("MARKET_GUARD_KOSPI_5D_THRESHOLD", -0.07),
        kospi_10d_threshold=_float("MARKET_GUARD_KOSPI_10D_THRESHOLD", -0.10),
        recovery_1d_threshold=_float("MARKET_GUARD_RECOVERY_1D_THRESHOLD", 0.01),
        recovery_5d_threshold=_float("MARKET_GUARD_RECOVERY_5D_THRESHOLD", -0.03),
        guard_file_path=_str("MARKET_GUARD_FILE_PATH", "data/market_guard_kill.json"),
        telegram_token=_str("MARKET_GUARD_TELEGRAM_TOKEN", ""),
        telegram_chat_id=_str("MARKET_GUARD_TELEGRAM_CHAT_ID", ""),
        log_path=_str("MARKET_GUARD_LOG_PATH", "logs/market_guard.log"),
        us_account_id=_str("MARKET_GUARD_US_ACCOUNT_ID", "US_TRADE_PAPER"),
    )
