"""
backfill_us_macro_features.py

Phase 4 보조 스크립트: 과거 기간의 US macro feature를 소급 계산하여
signal.us_macro_feature_daily에 저장합니다.

compute_us_macro_feature_daily.py의 compute_features 함수를 재사용합니다.

Usage:
    python backfill_us_macro_features.py
    python backfill_us_macro_features.py --start-date 2025-04-01 --end-date 2026-05-08
    python backfill_us_macro_features.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
from db import get_engine
from compute_us_macro_feature_daily import (
    compute_features,
    _load_kr_closed_dates,
    _Cfg,
    _UPSERT_FEATURE_SQL,
)

LOGGER = logging.getLogger("backfill_us_macro_features")

_READ_ALL_PRICES_SQL = text("""
SELECT trade_date, ticker, close
FROM market.us_etf_daily_price
WHERE trade_date BETWEEN :start_date AND :end_date
ORDER BY trade_date, ticker
""")

_GET_EXISTING_DATES_SQL = text("""
SELECT us_trade_date FROM signal.us_macro_feature_daily
WHERE us_trade_date BETWEEN :start_date AND :end_date
""")


def _us_trading_days(df_prices: pd.DataFrame) -> list[date]:
    """Return sorted list of unique US trade dates from price data."""
    return sorted(df_prices["trade_date"].unique().tolist())


def run_backfill(
    start_date: date,
    end_date: date,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> int:
    """Compute and upsert macro features for each US trading day in range.

    Returns the number of rows written.
    """
    engine = get_engine()
    cfg = _Cfg()
    kr_closed = _load_kr_closed_dates()

    # Load ETF price data (need 20-day lookback for 5d returns)
    lookback_start = start_date - timedelta(days=30)
    with engine.connect() as conn:
        rows = conn.execute(
            _READ_ALL_PRICES_SQL, {"start_date": lookback_start, "end_date": end_date}
        )
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())

    if df.empty:
        LOGGER.error("No ETF price data found for %s ~ %s. Run collect_us_macro_etf_daily.py first.", start_date, end_date)
        return 0

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Pivot: rows=date, cols=ticker
    pivot = df.pivot(index="trade_date", columns="ticker", values="close").sort_index()

    # Determine target US trade dates
    all_dates = _us_trading_days(df[df["trade_date"] >= start_date])
    LOGGER.info("Found %d US trading days in %s ~ %s", len(all_dates), start_date, end_date)

    # Skip already computed dates
    existing: set[date] = set()
    if skip_existing:
        with engine.connect() as conn:
            result = conn.execute(
                _GET_EXISTING_DATES_SQL, {"start_date": start_date, "end_date": end_date}
            )
            existing = {row[0] for row in result}
        LOGGER.info("Skipping %d dates already in signal.us_macro_feature_daily", len(existing))

    target_dates = [d for d in all_dates if d not in existing]
    LOGGER.info("Computing features for %d dates", len(target_dates))

    if not target_dates:
        LOGGER.info("Nothing to do.")
        return 0

    written = 0
    for trade_date in target_dates:
        # Use price data up to and including trade_date
        pivot_up_to = pivot[pivot.index <= trade_date]
        features = compute_features(trade_date, pivot_up_to, cfg, kr_closed)

        LOGGER.info(
            "  %s → status=%s | SPY=%s QQQ=%s VIX=%s",
            trade_date,
            features["macro_status"],
            f"{features.get('spy_ret_1d', 0):+.2%}" if features.get("spy_ret_1d") is not None else "n/a",
            f"{features.get('qqq_ret_1d', 0):+.2%}" if features.get("qqq_ret_1d") is not None else "n/a",
            f"{features.get('vix_ret_1d', 0):+.2%}" if features.get("vix_ret_1d") is not None else "n/a",
        )

        if not dry_run:
            with engine.begin() as conn:
                conn.execute(_UPSERT_FEATURE_SQL, features)
            written += 1

    LOGGER.info(
        "Backfill %s | %d rows written (%s)",
        "DRY RUN" if dry_run else "done",
        written,
        "nothing committed" if dry_run else "committed",
    )
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill signal.us_macro_feature_daily for a historical date range")
    p.add_argument("--start-date", default=None, help="Start US trade date (YYYY-MM-DD). Default: 1 year ago")
    p.add_argument("--end-date", default=None, help="End US trade date (YYYY-MM-DD). Default: today")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if date already exists")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = date.fromisoformat(args.start_date) if args.start_date else (today - timedelta(days=365))

    LOGGER.info("Backfill range: %s ~ %s (dry_run=%s)", start_date, end_date, args.dry_run)
    n = run_backfill(start_date, end_date, dry_run=args.dry_run, skip_existing=not args.overwrite)
    LOGGER.info("Done. Rows written: %d", n)


if __name__ == "__main__":
    main()
