from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

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
    row_count: int = 0


@dataclass
class DetectionResult:
    snapshot: MarketSnapshot
    alert_level: str        # "NONE" | "WARNING" | "CRITICAL"
    triggered_conditions: list[str]
    is_recovery: bool       # CRITICAL → NONE 해제 조건 충족 여부
    summary: str


@dataclass
class DataQualityResult:
    status: str              # "OK" | "WARNING" | "BLOCK"
    can_activate: bool
    source: str
    latest_trade_date: str
    internal_latest_date: str | None
    row_count: int
    warnings: list[str]


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
        row_count=int(len(closes)),
    )


def _latest_internal_market_date(path: Path) -> date | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=lambda col: col in {"date", "asof_date"})
    except Exception as exc:
        LOGGER.warning("market_status read failed for data quality check: %s", exc)
        return None
    if frame.empty:
        return None
    date_columns = [col for col in ("date", "asof_date") if col in frame.columns]
    if not date_columns:
        return None
    values = []
    for col in date_columns:
        parsed = pd.to_datetime(frame[col], errors="coerce")
        values.extend(parsed.dropna().dt.date.tolist())
    return max(values) if values else None


def _market_status_close(path: Path, trade_date: date) -> float | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns or "kospi_close" not in frame.columns:
        return None
    parsed = pd.to_datetime(frame["date"], errors="coerce").dt.date
    matched = frame.loc[parsed == trade_date, "kospi_close"]
    if matched.empty:
        return None
    numeric = pd.to_numeric(matched.iloc[-1], errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def evaluate_data_quality(
    snapshot: MarketSnapshot,
    *,
    market_status_csv: str | Path = "data/market_status.csv",
    min_row_count: int = 11,
    max_abs_daily_return: float = 0.20,
    close_mismatch_warn_pct: float = 0.02,
    close_mismatch_block_pct: float = 0.05,
) -> DataQualityResult:
    """Check whether the market snapshot is reliable enough to activate kill switch."""
    market_status_path = Path(market_status_csv)
    warnings: list[str] = []
    block_reasons: list[str] = []

    if snapshot.row_count < min_row_count:
        block_reasons.append(f"insufficient_rows:{snapshot.row_count}<{min_row_count}")
    if snapshot.kospi_close <= 0:
        block_reasons.append("invalid_kospi_close")
    if snapshot.ret_1d is not None and abs(snapshot.ret_1d) > max_abs_daily_return:
        block_reasons.append(f"daily_return_outlier:{snapshot.ret_1d:.2%}")

    internal_latest_date = _latest_internal_market_date(market_status_path)
    if internal_latest_date is None:
        warnings.append("internal_market_status_date_unavailable")
    elif snapshot.trade_date < internal_latest_date:
        block_reasons.append(f"source_stale:{snapshot.trade_date.isoformat()}<internal:{internal_latest_date.isoformat()}")
    elif snapshot.trade_date > internal_latest_date:
        warnings.append(f"internal_market_status_lag:{internal_latest_date.isoformat()}<source:{snapshot.trade_date.isoformat()}")

    internal_close = _market_status_close(market_status_path, snapshot.trade_date)
    if internal_close is not None and internal_close > 0 and snapshot.kospi_close > 0:
        mismatch = abs(snapshot.kospi_close / internal_close - 1.0)
        if mismatch > close_mismatch_block_pct:
            block_reasons.append(f"close_mismatch:{mismatch:.2%}")
        elif mismatch > close_mismatch_warn_pct:
            warnings.append(f"close_mismatch:{mismatch:.2%}")

    status = "BLOCK" if block_reasons else "WARNING" if warnings else "OK"
    return DataQualityResult(
        status=status,
        can_activate=not block_reasons,
        source=f"yfinance:{KOSPI_TICKER}",
        latest_trade_date=snapshot.trade_date.isoformat(),
        internal_latest_date=internal_latest_date.isoformat() if internal_latest_date else None,
        row_count=snapshot.row_count,
        warnings=[*block_reasons, *warnings],
    )


def data_quality_to_dict(result: DataQualityResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "can_activate": result.can_activate,
        "source": result.source,
        "latest_trade_date": result.latest_trade_date,
        "internal_latest_date": result.internal_latest_date,
        "row_count": result.row_count,
        "warnings": result.warnings,
    }


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
