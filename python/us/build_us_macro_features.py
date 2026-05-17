"""Build US macro context features (VIX, SPY, QQQ) into feature.us_macro_daily.

Runs daily after market close. Fetches VIX/SPY/QQQ price history via yfinance
and computes regime-level features used as per-stock context in ML training.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import logging
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_db import get_us_engine


LOGGER = logging.getLogger("us_macro_feature")
LOOKBACK_DAYS = 520  # enough for 252-day calculations
MIN_HISTORY_ROWS = 60


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US macro daily features (VIX/SPY/QQQ).")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: today)")
    return parser.parse_args()


def _fetch_series(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index().dropna()


def _build_macro_frame(
    vix: pd.Series,
    spy: pd.Series,
    qqq: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame({"vix": vix, "spy": spy, "qqq": qqq}).dropna(how="all")
    frame.index = pd.to_datetime(frame.index)

    frame["vix_ret_20d"] = frame["vix"] / frame["vix"].shift(20) - 1.0
    frame["spy_ret_20d"] = frame["spy"] / frame["spy"].shift(20) - 1.0
    frame["qqq_ret_20d"] = frame["qqq"] / frame["qqq"].shift(20) - 1.0

    spy_ma200 = frame["spy"].rolling(window=200, min_periods=120).mean()
    frame["spy_above_ma200"] = frame["spy"] > spy_ma200

    # Regime: Bull (SPY above MA200 + positive 20d), Bear, Neutral
    def _regime(row: pd.Series) -> str:
        if pd.isna(row.get("spy_above_ma200")) or pd.isna(row.get("spy_ret_20d")):
            return "neutral"
        if row["spy_above_ma200"] and (row["spy_ret_20d"] or 0.0) >= 0.0:
            return "bull"
        if not row["spy_above_ma200"] and (row["spy_ret_20d"] or 0.0) < 0.0:
            return "bear"
        return "neutral"

    frame["market_regime"] = frame.apply(_regime, axis=1)
    return frame


def _ensure_table_exists() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feature.us_macro_daily (
                trade_date DATE NOT NULL,
                vix_close NUMERIC,
                vix_ret_20d NUMERIC,
                spy_ret_20d NUMERIC,
                spy_above_ma200 BOOLEAN,
                qqq_ret_20d NUMERIC,
                market_regime VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (trade_date)
            )
        """))


def _upsert_macro_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = text("""
        INSERT INTO feature.us_macro_daily (
            trade_date, vix_close, vix_ret_20d,
            spy_ret_20d, spy_above_ma200, qqq_ret_20d,
            market_regime, updated_at
        ) VALUES (
            :trade_date, :vix_close, :vix_ret_20d,
            :spy_ret_20d, :spy_above_ma200, :qqq_ret_20d,
            :market_regime, now()
        )
        ON CONFLICT (trade_date) DO UPDATE SET
            vix_close = EXCLUDED.vix_close,
            vix_ret_20d = EXCLUDED.vix_ret_20d,
            spy_ret_20d = EXCLUDED.spy_ret_20d,
            spy_above_ma200 = EXCLUDED.spy_above_ma200,
            qqq_ret_20d = EXCLUDED.qqq_ret_20d,
            market_regime = EXCLUDED.market_regime,
            updated_at = now()
    """)
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def build_us_macro_features(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    _ensure_table_exists()
    end = end_date or date.today()
    fetch_start = (end - timedelta(days=LOOKBACK_DAYS)).isoformat()
    fetch_end = (end + timedelta(days=1)).isoformat()

    LOGGER.info("[US_MACRO] Fetching VIX/SPY/QQQ start=%s end=%s", fetch_start, fetch_end)
    vix = _fetch_series("^VIX", fetch_start, fetch_end)
    spy = _fetch_series("SPY", fetch_start, fetch_end)
    qqq = _fetch_series("QQQ", fetch_start, fetch_end)

    if len(spy) < MIN_HISTORY_ROWS:
        LOGGER.warning("[US_MACRO] Insufficient SPY history rows=%s", len(spy))
        return 0

    frame = _build_macro_frame(vix, spy, qqq)

    # Filter to requested date window
    if start_date is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end_date)]

    # Drop rows without enough lookback (no 20d calculation yet)
    frame = frame.loc[frame["spy_ret_20d"].notna()].copy()

    rows = []
    for ts, row in frame.iterrows():
        rows.append({
            "trade_date": ts.date(),
            "vix_close": _safe_float(row.get("vix")),
            "vix_ret_20d": _safe_float(row.get("vix_ret_20d")),
            "spy_ret_20d": _safe_float(row.get("spy_ret_20d")),
            "spy_above_ma200": None if pd.isna(row.get("spy_above_ma200")) else bool(row["spy_above_ma200"]),
            "qqq_ret_20d": _safe_float(row.get("qqq_ret_20d")),
            "market_regime": str(row.get("market_regime") or "neutral"),
        })

    saved = _upsert_macro_rows(rows)
    LOGGER.info("[US_MACRO] Done upserted=%s rows", saved)
    return saved


def main() -> None:
    setup_logging()
    args = parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_MACRO] DB connection failed: {exc}") from exc

    build_us_macro_features(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
