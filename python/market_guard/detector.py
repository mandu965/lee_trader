from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

LOGGER = logging.getLogger("market_guard.detector")

KOSPI_TICKER = "^KS11"   # yfinance KOSPI 종합지수


@dataclass
class MarketSnapshot:
    trade_date: date
    kospi_close: float
    ret_1d: float | None    # 당일 수익률
    ret_5d: float | None    # 5영업일 누적
    ret_10d: float | None   # 10영업일 누적


@dataclass
class DetectionResult:
    snapshot: MarketSnapshot
    alert_level: str        # "NONE" | "WARNING" | "CRITICAL"
    triggered_conditions: list[str]
    is_recovery: bool       # CRITICAL → NONE 해제 조건 충족 여부
    summary: str


def _fetch_kospi(lookback_days: int = 30) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed") from exc

    raw = yf.download(KOSPI_TICKER, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("KOSPI 가격 데이터를 가져올 수 없습니다.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    return raw["close"].dropna()


def _pct(series: pd.Series, n: int) -> float | None:
    if len(series) < n + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-n - 1]) - 1)


def get_snapshot() -> MarketSnapshot:
    closes = _fetch_kospi(lookback_days=30)
    today_close = float(closes.iloc[-1])
    trade_date = closes.index[-1].date()

    ret_1d = _pct(closes, 1)
    ret_5d = _pct(closes, 5)
    ret_10d = _pct(closes, 10)

    LOGGER.info(
        "MARKET_SNAPSHOT | date=%s KOSPI=%.2f 1d=%s 5d=%s 10d=%s",
        trade_date, today_close,
        f"{ret_1d:.2%}" if ret_1d is not None else "N/A",
        f"{ret_5d:.2%}" if ret_5d is not None else "N/A",
        f"{ret_10d:.2%}" if ret_10d is not None else "N/A",
    )
    return MarketSnapshot(
        trade_date=trade_date,
        kospi_close=today_close,
        ret_1d=ret_1d,
        ret_5d=ret_5d,
        ret_10d=ret_10d,
    )


def evaluate(
    snapshot: MarketSnapshot,
    *,
    kospi_1d_threshold: float,
    kospi_5d_threshold: float,
    kospi_10d_threshold: float,
    recovery_1d_threshold: float,
    recovery_5d_threshold: float,
) -> DetectionResult:
    triggered: list[str] = []

    # 강한 조건 (CRITICAL)
    if snapshot.ret_1d is not None and snapshot.ret_1d <= kospi_1d_threshold:
        triggered.append(f"KOSPI_1D={snapshot.ret_1d:.2%} <= {kospi_1d_threshold:.2%}")

    if snapshot.ret_5d is not None and snapshot.ret_5d <= kospi_5d_threshold:
        triggered.append(f"KOSPI_5D={snapshot.ret_5d:.2%} <= {kospi_5d_threshold:.2%}")

    # 경고 조건 (WARNING)
    warning_conditions: list[str] = []
    if snapshot.ret_10d is not None and snapshot.ret_10d <= kospi_10d_threshold:
        warning_conditions.append(f"KOSPI_10D={snapshot.ret_10d:.2%} <= {kospi_10d_threshold:.2%}")

    if triggered:
        alert_level = "CRITICAL"
    elif warning_conditions:
        alert_level = "WARNING"
        triggered = warning_conditions
    else:
        alert_level = "NONE"

    # 해제 조건 (CRITICAL에서만 의미 있음)
    is_recovery = (
        alert_level == "NONE"
        and (snapshot.ret_1d or 0) >= recovery_1d_threshold
        and (snapshot.ret_5d or -1) >= recovery_5d_threshold
    )

    summary = (
        f"KOSPI {snapshot.kospi_close:,.0f} | "
        f"1d={snapshot.ret_1d:.2%} 5d={snapshot.ret_5d:.2%} | "
        f"level={alert_level}"
        if snapshot.ret_1d is not None and snapshot.ret_5d is not None
        else f"KOSPI {snapshot.kospi_close:,.0f} | level={alert_level}"
    )

    LOGGER.info("DETECTION | level=%s conditions=%s", alert_level, triggered)
    return DetectionResult(
        snapshot=snapshot,
        alert_level=alert_level,
        triggered_conditions=triggered,
        is_recovery=is_recovery,
        summary=summary,
    )
