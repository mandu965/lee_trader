"""
compute_us_macro_feature_daily.py

Phase 2 - US Macro Overlay: Compute macro features from market.us_etf_daily_price
and store results in signal.us_macro_feature_daily.

⚠ SHADOW MODE ONLY (Phase 1~2):
   This script reads US price data and writes macro features.
   It does NOT touch any Korean auto-trading logic, orders, or rankings.

Usage:
    python compute_us_macro_feature_daily.py
    python compute_us_macro_feature_daily.py --date 2026-05-07
    python compute_us_macro_feature_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine

LOGGER = logging.getLogger("compute_us_macro_feature_daily")

# ── Ticker roles ───────────────────────────────────────────────────────────────

SECTOR_TICKERS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
}

REQUIRED_TICKERS = ["SPY", "QQQ"]  # Missing these → DATA_INCOMPLETE

KR_CALENDAR_PATH = Path(os.environ.get(
    "RULE_TRADING_CALENDAR_PATH", "config/trading_calendar_kr.json"
))

# ── Korean trading calendar ────────────────────────────────────────────────────

def _load_kr_closed_dates() -> set[date]:
    try:
        data = json.loads(KR_CALENDAR_PATH.read_text(encoding="utf-8"))
        return {date.fromisoformat(d) for d in data.get("closed_dates", [])}
    except Exception:
        return set()


def next_kr_trading_day(from_date: date, closed: set[date]) -> date:
    """Return the next KRX open day strictly after from_date."""
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5 or d in closed:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


# ── DB helpers ─────────────────────────────────────────────────────────────────

_READ_PRICES_SQL = text("""
SELECT trade_date, ticker, close
FROM market.us_etf_daily_price
WHERE trade_date BETWEEN :start_date AND :end_date
ORDER BY trade_date
""")

_UPSERT_FEATURE_SQL = text("""
INSERT INTO signal.us_macro_feature_daily (
    us_trade_date, kr_apply_date,
    spy_ret_1d, spy_ret_5d,
    qqq_ret_1d, qqq_ret_5d,
    semiconductor_ret_1d, semiconductor_ret_5d,
    vix_ret_1d, dxy_ret_1d, tnx_ret_1d,
    sector_breadth,
    top_sector, bottom_sector, sector_strength_rank,
    risk_on_flag, risk_off_flag,
    vix_spike_flag, semiconductor_strength_flag,
    macro_status, macro_summary,
    data_source, missing_tickers,
    updated_at
) VALUES (
    :us_trade_date, :kr_apply_date,
    :spy_ret_1d, :spy_ret_5d,
    :qqq_ret_1d, :qqq_ret_5d,
    :semiconductor_ret_1d, :semiconductor_ret_5d,
    :vix_ret_1d, :dxy_ret_1d, :tnx_ret_1d,
    :sector_breadth,
    :top_sector, :bottom_sector, :sector_strength_rank,
    :risk_on_flag, :risk_off_flag,
    :vix_spike_flag, :semiconductor_strength_flag,
    :macro_status, :macro_summary,
    :data_source, :missing_tickers,
    now()
)
ON CONFLICT (us_trade_date) DO UPDATE SET
    kr_apply_date             = EXCLUDED.kr_apply_date,
    spy_ret_1d                = EXCLUDED.spy_ret_1d,
    spy_ret_5d                = EXCLUDED.spy_ret_5d,
    qqq_ret_1d                = EXCLUDED.qqq_ret_1d,
    qqq_ret_5d                = EXCLUDED.qqq_ret_5d,
    semiconductor_ret_1d      = EXCLUDED.semiconductor_ret_1d,
    semiconductor_ret_5d      = EXCLUDED.semiconductor_ret_5d,
    vix_ret_1d                = EXCLUDED.vix_ret_1d,
    dxy_ret_1d                = EXCLUDED.dxy_ret_1d,
    tnx_ret_1d                = EXCLUDED.tnx_ret_1d,
    sector_breadth            = EXCLUDED.sector_breadth,
    top_sector                = EXCLUDED.top_sector,
    bottom_sector             = EXCLUDED.bottom_sector,
    sector_strength_rank      = EXCLUDED.sector_strength_rank,
    risk_on_flag              = EXCLUDED.risk_on_flag,
    risk_off_flag             = EXCLUDED.risk_off_flag,
    vix_spike_flag            = EXCLUDED.vix_spike_flag,
    semiconductor_strength_flag = EXCLUDED.semiconductor_strength_flag,
    macro_status              = EXCLUDED.macro_status,
    macro_summary             = EXCLUDED.macro_summary,
    data_source               = EXCLUDED.data_source,
    missing_tickers           = EXCLUDED.missing_tickers,
    updated_at                = now()
""")


def _read_prices(engine: Any, start_date: date, end_date: date) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(_READ_PRICES_SQL, {"start_date": start_date, "end_date": end_date})
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


# ── Return calculations ────────────────────────────────────────────────────────

def _ret(series: pd.Series, n: int) -> float | None:
    """n-period return using last available close vs n rows prior."""
    s = series.dropna()
    if len(s) < n + 1:
        return None
    cur = float(s.iloc[-1])
    prev = float(s.iloc[-(n + 1)])
    if prev == 0:
        return None
    return round((cur / prev) - 1, 6)


def _get_ticker_series(pivot: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker not in pivot.columns:
        return pd.Series(dtype=float)
    return pivot[ticker].dropna()


# ── Config from env ────────────────────────────────────────────────────────────

class _Cfg:
    def __init__(self) -> None:
        self.risk_off_qqq_ret     = float(os.environ.get("US_MACRO_RISK_OFF_QQQ_RET", "-0.015"))
        self.risk_off_spy_ret     = float(os.environ.get("US_MACRO_RISK_OFF_SPY_RET", "-0.012"))
        self.vix_spike_ret        = float(os.environ.get("US_MACRO_VIX_SPIKE_RET", "0.10"))
        self.semi_positive_ret    = float(os.environ.get("US_MACRO_SEMI_POSITIVE_RET", "0.01"))
        self.semi_negative_ret    = float(os.environ.get("US_MACRO_SEMI_NEGATIVE_RET", "-0.01"))
        self.breadth_risk_off     = float(os.environ.get("US_MACRO_SECTOR_BREADTH_RISK_OFF_RATIO", "0.3"))
        self.breadth_risk_on      = float(os.environ.get("US_MACRO_SECTOR_BREADTH_RISK_ON_RATIO", "0.6"))
        self.data_source          = os.environ.get("US_MACRO_DATA_SOURCE", "yfinance")
        self.stale_days_limit     = int(os.environ.get("US_MACRO_STALE_DAYS_LIMIT", "3"))


# ── Feature computation ────────────────────────────────────────────────────────

def compute_features(
    trade_date: date,
    pivot: pd.DataFrame,
    cfg: _Cfg,
    kr_closed: set[date],
) -> dict[str, Any]:
    """Compute all macro features for a single US trade date."""

    missing: list[str] = []

    def get_series(ticker: str) -> pd.Series:
        s = _get_ticker_series(pivot, ticker)
        if s.empty:
            missing.append(ticker)
        return s

    spy = get_series("SPY")
    qqq = get_series("QQQ")
    smh = get_series("SMH")
    vix = get_series("^VIX")
    dxy = get_series("DX-Y.NYB")
    tnx = get_series("^TNX")

    spy_1d = _ret(spy, 1)
    spy_5d = _ret(spy, 5)
    qqq_1d = _ret(qqq, 1)
    qqq_5d = _ret(qqq, 5)
    semi_1d = _ret(smh, 1)
    semi_5d = _ret(smh, 5)
    vix_1d = _ret(vix, 1)
    dxy_1d = _ret(dxy, 1)
    tnx_1d = _ret(tnx, 1)

    # Sector breadth: ratio of sectors with positive 1d return
    sector_rets: dict[str, float | None] = {}
    for t, name in SECTOR_TICKERS.items():
        s = _get_ticker_series(pivot, t)
        sector_rets[name] = _ret(s, 1)

    valid_sector_rets = {k: v for k, v in sector_rets.items() if v is not None}
    if valid_sector_rets:
        positive_count = sum(1 for v in valid_sector_rets.values() if v > 0)
        sector_breadth = round(positive_count / len(valid_sector_rets), 4)
        ranked = sorted(valid_sector_rets.items(), key=lambda x: x[1], reverse=True)
        top_sector = ranked[0][0] if ranked else None
        bottom_sector = ranked[-1][0] if ranked else None
        sector_strength_rank = json.dumps(
            [{"sector": k, "ret_1d": round(v, 6)} for k, v in ranked]
        )
    else:
        sector_breadth = None
        top_sector = None
        bottom_sector = None
        sector_strength_rank = None

    # Flags
    data_complete = all(t not in missing for t in REQUIRED_TICKERS)

    risk_off_flag: bool | None = None
    risk_on_flag: bool | None = None
    vix_spike_flag: bool | None = None
    semi_strength_flag: bool | None = None

    if data_complete:
        qqq_risk_off = qqq_1d is not None and qqq_1d < cfg.risk_off_qqq_ret
        spy_risk_off = spy_1d is not None and spy_1d < cfg.risk_off_spy_ret
        breadth_risk_off = sector_breadth is not None and sector_breadth < cfg.breadth_risk_off

        risk_off_flag = qqq_risk_off or spy_risk_off or breadth_risk_off

        breadth_risk_on = sector_breadth is not None and sector_breadth >= cfg.breadth_risk_on
        risk_on_flag = (not risk_off_flag) and breadth_risk_on and (spy_1d or 0) >= 0 and (qqq_1d or 0) >= 0

        vix_spike_flag = vix_1d is not None and vix_1d > cfg.vix_spike_ret

        semi_strength_flag = (
            semi_1d is not None
            and (semi_1d > cfg.semi_positive_ret or semi_1d < cfg.semi_negative_ret)
        )

    # Status
    if not data_complete or missing:
        macro_status = "DATA_INCOMPLETE"
    elif risk_off_flag:
        macro_status = "RISK_OFF"
    elif risk_on_flag:
        macro_status = "RISK_ON"
    else:
        macro_status = "NEUTRAL"

    # Human-readable summary
    parts: list[str] = [f"US {trade_date} → status={macro_status}"]
    if spy_1d is not None:
        parts.append(f"SPY={spy_1d:+.2%}")
    if qqq_1d is not None:
        parts.append(f"QQQ={qqq_1d:+.2%}")
    if vix_1d is not None:
        parts.append(f"VIX={vix_1d:+.2%}")
    if sector_breadth is not None:
        parts.append(f"breadth={sector_breadth:.0%}")
    if top_sector:
        parts.append(f"top={top_sector}")
    if missing:
        parts.append(f"missing={missing}")
    macro_summary = " | ".join(parts)

    kr_apply = next_kr_trading_day(trade_date, kr_closed)

    return {
        "us_trade_date":             trade_date,
        "kr_apply_date":             kr_apply,
        "spy_ret_1d":                spy_1d,
        "spy_ret_5d":                spy_5d,
        "qqq_ret_1d":                qqq_1d,
        "qqq_ret_5d":                qqq_5d,
        "semiconductor_ret_1d":      semi_1d,
        "semiconductor_ret_5d":      semi_5d,
        "vix_ret_1d":                vix_1d,
        "dxy_ret_1d":                dxy_1d,
        "tnx_ret_1d":                tnx_1d,
        "sector_breadth":            sector_breadth,
        "top_sector":                top_sector,
        "bottom_sector":             bottom_sector,
        "sector_strength_rank":      sector_strength_rank,
        "risk_on_flag":              risk_on_flag,
        "risk_off_flag":             risk_off_flag,
        "vix_spike_flag":            vix_spike_flag,
        "semiconductor_strength_flag": semi_strength_flag,
        "macro_status":              macro_status,
        "macro_summary":             macro_summary,
        "data_source":               cfg.data_source,
        "missing_tickers":           missing if missing else None,
    }


# ── Orchestration ──────────────────────────────────────────────────────────────

def run(
    target_date: date,
    engine: Any,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compute and store macro features for target_date.

    Reads the previous 10 trading days of price data so 5d returns are computable.
    Returns a result dict with keys: us_trade_date, macro_status, missing_tickers.
    """
    cfg = _Cfg()
    kr_closed = _load_kr_closed_dates()

    # Fetch enough history for 5d returns (10 calendar days → ~7 trading days)
    lookback_start = target_date - timedelta(days=20)
    df = _read_prices(engine, lookback_start, target_date)

    if df.empty:
        LOGGER.error("No price data found for %s (lookback from %s).", target_date, lookback_start)
        return {"us_trade_date": target_date, "macro_status": "DATA_INCOMPLETE", "missing_tickers": ["ALL"]}

    # Check data freshness: most recent date in DB
    latest_date = df["trade_date"].max()
    stale_days = (target_date - latest_date).days
    if stale_days > cfg.stale_days_limit:
        LOGGER.warning(
            "Price data is stale: latest=%s, target=%s, gap=%d days (limit=%d).",
            latest_date, target_date, stale_days, cfg.stale_days_limit,
        )

    # Pivot to (date × ticker) with close prices
    pivot = df.pivot(index="trade_date", columns="ticker", values="close").sort_index()

    # Use the most recent date available for feature computation
    features = compute_features(latest_date, pivot, cfg, kr_closed)

    LOGGER.info(
        "Computed features for %s: status=%s | missing=%s",
        features["us_trade_date"],
        features["macro_status"],
        features.get("missing_tickers"),
    )
    LOGGER.info("Summary: %s", features.get("macro_summary"))

    if dry_run:
        LOGGER.info("DRY RUN - feature row not written to DB.")
        return features

    with engine.begin() as conn:
        conn.execute(_UPSERT_FEATURE_SQL, features)
    LOGGER.info("Upserted macro feature row for %s.", features["us_trade_date"])

    return features


# ── CLI entry point ────────────────────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Compute US macro features → signal.us_macro_feature_daily"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="US trade date to compute features for (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features but do not write to DB.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    enabled = os.environ.get("US_MACRO_ENABLED", "1").strip()
    if enabled not in ("1", "true", "yes", "y"):
        LOGGER.info("US_MACRO_ENABLED is not set - skipping feature computation.")
        return

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    engine = get_engine()

    result = run(target_date=target_date, engine=engine, dry_run=args.dry_run)

    status = result.get("macro_status", "UNKNOWN")
    missing = result.get("missing_tickers") or []
    LOGGER.info("Done | status=%s | missing_tickers=%s", status, missing)


if __name__ == "__main__":
    main()
