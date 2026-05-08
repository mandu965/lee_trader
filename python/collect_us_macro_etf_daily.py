"""
collect_us_macro_etf_daily.py

Phase 1 - US Macro Overlay: Collect US ETF / index daily OHLCV data.

Data is stored in market.us_etf_daily_price (isolated from Korean stock tables).
Source adapter pattern allows swapping yfinance → Polygon / Alpaca later.

Usage:
    python collect_us_macro_etf_daily.py
    python collect_us_macro_etf_daily.py --date 2026-05-07
    python collect_us_macro_etf_daily.py --lookback-days 30
    python collect_us_macro_etf_daily.py --dry-run
"""

from __future__ import annotations

import abc
import argparse
import logging
import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine

LOGGER = logging.getLogger("collect_us_macro_etf_daily")

# ── Ticker configuration ───────────────────────────────────────────────────────

US_TICKER_LABELS: dict[str, str] = {
    "SPY":       "S&P 500 ETF",
    "QQQ":       "NASDAQ 100 ETF",
    "DIA":       "Dow Jones ETF",
    "XLK":       "Technology Sector",
    "XLF":       "Financials Sector",
    "XLE":       "Energy Sector",
    "XLV":       "Healthcare Sector",
    "XLI":       "Industrials Sector",
    "XLY":       "Consumer Discretionary Sector",
    "XLP":       "Consumer Staples Sector",
    "SMH":       "Semiconductor ETF",
    "^VIX":      "VIX Volatility Index",
    "DX-Y.NYB":  "US Dollar Index",
    "^TNX":      "10-Year Treasury Yield",
}

DEFAULT_TICKERS = list(US_TICKER_LABELS.keys())

# ── Source adapters ────────────────────────────────────────────────────────────

class BaseUSMacroAdapter(abc.ABC):
    """Abstract adapter for US market data sources.

    Implement this interface to swap in Polygon, Alpaca, or other providers.
    """

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Return DataFrame with columns: trade_date, open, high, low, close, volume, adj_close.

        trade_date must be a Python date. Returns empty DataFrame on failure.
        """


class YFinanceAdapter(BaseUSMacroAdapter):
    """Fetch daily OHLCV via yfinance."""

    def fetch_ohlcv(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        try:
            import yfinance as yf  # lazy import — not required at module load
        except ImportError:
            raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

        try:
            raw = yf.download(
                ticker,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
            )
        except Exception as exc:
            LOGGER.warning("[%s] yfinance download failed: %s", ticker, exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # yfinance may return MultiIndex columns for a single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        rename_map = {
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        raw = raw.rename(columns=rename_map)

        needed = [c for c in rename_map.values() if c in raw.columns]
        raw = raw[needed].copy()

        if "trade_date" in raw.columns:
            raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date

        # Drop rows with no close price
        raw = raw.dropna(subset=["close"])
        return raw


def _get_adapter(source: str) -> BaseUSMacroAdapter:
    source = source.lower().strip()
    if source == "yfinance":
        return YFinanceAdapter()
    raise ValueError(
        f"Unknown US_MACRO_DATA_SOURCE '{source}'. "
        "Supported: yfinance. (Polygon/Alpaca adapters: implement BaseUSMacroAdapter)"
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

_UPSERT_SQL = text("""
INSERT INTO market.us_etf_daily_price
    (trade_date, ticker, ticker_label, open, high, low, close, volume, adj_close,
     data_source, updated_at)
VALUES
    (:trade_date, :ticker, :ticker_label, :open, :high, :low, :close, :volume, :adj_close,
     :data_source, now())
ON CONFLICT (trade_date, ticker) DO UPDATE SET
    ticker_label = EXCLUDED.ticker_label,
    open         = EXCLUDED.open,
    high         = EXCLUDED.high,
    low          = EXCLUDED.low,
    close        = EXCLUDED.close,
    volume       = EXCLUDED.volume,
    adj_close    = EXCLUDED.adj_close,
    data_source  = EXCLUDED.data_source,
    updated_at   = now()
""")


def _upsert_rows(engine: Any, rows: list[dict]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, rows)
    return len(rows)


# ── Main collection logic ──────────────────────────────────────────────────────

def collect(
    tickers: list[str],
    start_date: date,
    end_date: date,
    adapter: BaseUSMacroAdapter,
    engine: Any,
    dry_run: bool = False,
    data_source: str = "yfinance",
) -> dict[str, Any]:
    """Collect OHLCV for each ticker and upsert to market.us_etf_daily_price.

    Returns a summary dict with keys: success, failed, rows_upserted.
    """
    success: list[str] = []
    failed: list[str] = []
    total_rows = 0

    for ticker in tickers:
        label = US_TICKER_LABELS.get(ticker, ticker)
        LOGGER.info("[%s] Fetching %s to %s", ticker, start_date, end_date)
        try:
            df = adapter.fetch_ohlcv(ticker, start_date, end_date)
        except Exception as exc:
            LOGGER.error("[%s] Adapter error: %s", ticker, exc)
            failed.append(ticker)
            continue

        if df.empty:
            LOGGER.warning("[%s] No data returned for date range.", ticker)
            failed.append(ticker)
            continue

        rows = []
        for _, row in df.iterrows():
            rows.append({
                "trade_date":   row["trade_date"],
                "ticker":       ticker,
                "ticker_label": label,
                "open":         _safe_float(row.get("open")),
                "high":         _safe_float(row.get("high")),
                "low":          _safe_float(row.get("low")),
                "close":        _safe_float(row["close"]),
                "volume":       _safe_int(row.get("volume")),
                "adj_close":    _safe_float(row.get("adj_close")),
                "data_source":  data_source,
            })

        if dry_run:
            LOGGER.info("[%s] DRY RUN - would upsert %d rows.", ticker, len(rows))
            success.append(ticker)
            continue

        try:
            n = _upsert_rows(engine, rows)
            LOGGER.info("[%s] Upserted %d rows.", ticker, n)
            total_rows += n
            success.append(ticker)
        except Exception as exc:
            LOGGER.error("[%s] DB upsert failed: %s", ticker, exc)
            failed.append(ticker)

    summary = {
        "success":       success,
        "failed":        failed,
        "rows_upserted": total_rows,
        "start_date":    start_date,
        "end_date":      end_date,
    }

    if failed:
        LOGGER.warning("Collection finished. Failed tickers: %s", failed)
    else:
        LOGGER.info("Collection finished. All %d tickers succeeded.", len(success))

    return summary


# ── Utilities ──────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if f != f else f  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    try:
        f = float(val)
        if f != f:
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _resolve_tickers() -> list[str]:
    env_val = os.environ.get("US_MACRO_TICKERS", "").strip()
    if env_val:
        return [t.strip() for t in env_val.split(",") if t.strip()]
    return DEFAULT_TICKERS


# ── CLI entry point ────────────────────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Collect US ETF/index daily OHLCV → market.us_etf_daily_price"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target end date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=10,
        help="Calendar days of history to collect. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to DB.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Data source adapter. Default: US_MACRO_DATA_SOURCE env var or 'yfinance'.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    enabled = os.environ.get("US_MACRO_ENABLED", "1").strip()
    if enabled not in ("1", "true", "yes", "y"):
        LOGGER.info("US_MACRO_ENABLED is not set - skipping collection.")
        return

    end_date = date.fromisoformat(args.date) if args.date else date.today()
    start_date = end_date - timedelta(days=args.lookback_days)

    source = args.source or os.environ.get("US_MACRO_DATA_SOURCE", "yfinance")
    tickers = _resolve_tickers()
    adapter = _get_adapter(source)
    engine = get_engine()

    LOGGER.info(
        "Starting US macro ETF collection: %s → %s | tickers=%d | source=%s | dry_run=%s",
        start_date, end_date, len(tickers), source, args.dry_run,
    )

    summary = collect(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        adapter=adapter,
        engine=engine,
        dry_run=args.dry_run,
        data_source=source,
    )

    LOGGER.info(
        "Summary | success=%d | failed=%d | rows_upserted=%d",
        len(summary["success"]),
        len(summary["failed"]),
        summary["rows_upserted"],
    )
    if summary["failed"]:
        LOGGER.warning("Failed tickers (will cause DATA_INCOMPLETE if required): %s", summary["failed"])


if __name__ == "__main__":
    main()
