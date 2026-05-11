from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_stock_config, parse_iso_date
from python.us.us_db import (
    fetch_active_tickers,
    fetch_anomaly_stats,
    fetch_orphan_tickers,
    fetch_price_stats,
    fetch_universe_counts,
    get_us_engine,
    upsert_quality_report,
)


LOGGER = logging.getLogger("us_quality")
SUPPORTED_SOURCE = "yfinance"
FAILED_THRESHOLD = 0.10
WARN_THRESHOLD = 0.05
LONG_STALE_DAYS = 30
SUMMARY_LIMIT = 10


@dataclass(frozen=True)
class ValidationResult:
    active_ticker_count: int
    total_ticker_count: int
    price_ok_count: int
    price_missing_count: int
    stale_ticker_count: int
    failed_ticker_count: int
    anomaly_count: int
    orphan_ticker_count: int
    quality_status: str
    missing_tickers: list[str]
    stale_tickers: list[str]
    failed_tickers: list[str]
    orphan_tickers: list[str]
    summary: str


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_stock_config()
    parser = argparse.ArgumentParser(description="Validate US stock OHLCV data quality.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag. Default: env US_STOCK_UNIVERSE.")
    parser.add_argument("--as-of-date", default=None, help="Validation date. Format: YYYY-MM-DD.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed ticker lists.")
    return parser.parse_args()


def _format_ticker_sample(tickers: list[str], *, limit: int = SUMMARY_LIMIT) -> str:
    if not tickers:
        return "-"
    head = tickers[:limit]
    extra = len(tickers) - len(head)
    if extra > 0:
        return f"{', '.join(head)} (+{extra} more)"
    return ", ".join(head)


def _build_summary(
    *,
    as_of_date: date,
    stale_days_limit: int,
    missing_tickers: list[str],
    stale_tickers: list[str],
    failed_tickers: list[str],
    orphan_tickers: list[str],
    anomaly_count: int,
) -> str:
    # Stale is based on calendar-day difference, which can overstate issues around US weekends/holidays.
    parts = [
        f"as_of_date={as_of_date.isoformat()}",
        f"stale_days_limit={stale_days_limit}",
        f"missing={len(missing_tickers)} [{_format_ticker_sample(missing_tickers)}]",
        f"stale={len(stale_tickers)} [{_format_ticker_sample(stale_tickers)}]",
        f"failed={len(failed_tickers)} [{_format_ticker_sample(failed_tickers)}]",
        f"orphan={len(orphan_tickers)} [{_format_ticker_sample(orphan_tickers)}]",
        f"anomaly_rows={anomaly_count}",
        "stale_check_note=calendar_day_based",
    ]
    return " | ".join(parts)


def _decide_quality_status(
    *,
    active_ticker_count: int,
    affected_ticker_count: int,
    has_any_price_data: bool,
) -> str:
    if active_ticker_count <= 0:
        return "FAILED"
    if not has_any_price_data:
        return "FAILED"

    affected_ratio = affected_ticker_count / active_ticker_count
    if affected_ratio >= FAILED_THRESHOLD:
        return "FAILED"
    if affected_ratio >= WARN_THRESHOLD:
        return "WARN"
    return "OK"


def validate_price_data(*, universe_tag: str, as_of_date: date, verbose: bool) -> ValidationResult:
    cfg = load_us_stock_config()
    stale_days_limit = int(cfg.stale_days_limit)
    if stale_days_limit < 0:
        raise ValueError(f"Invalid US_STOCK_STALE_DAYS_LIMIT: {stale_days_limit}. Expected >= 0.")

    counts = fetch_universe_counts(universe_tag)
    active_tickers = fetch_active_tickers(universe_tag)
    active_ticker_count = len(active_tickers)
    total_ticker_count = counts["total_ticker_count"]

    LOGGER.info("[US_QUALITY] Start validation universe=%s", universe_tag)
    LOGGER.info("[US_QUALITY] active_tickers=%s", active_ticker_count)

    if active_ticker_count == 0:
        LOGGER.info("[US_QUALITY] No active tickers found. universe=%s", universe_tag)
        return ValidationResult(
            active_ticker_count=0,
            total_ticker_count=total_ticker_count,
            price_ok_count=0,
            price_missing_count=0,
            stale_ticker_count=0,
            failed_ticker_count=0,
            anomaly_count=0,
            orphan_ticker_count=0,
            quality_status="FAILED",
            missing_tickers=[],
            stale_tickers=[],
            failed_tickers=[],
            orphan_tickers=[],
            summary=f"as_of_date={as_of_date.isoformat()} | no_active_tickers=true",
        )

    price_stats = fetch_price_stats(active_tickers, as_of_date=as_of_date, data_source=cfg.data_source)
    anomaly_stats = fetch_anomaly_stats(active_tickers, as_of_date=as_of_date, data_source=cfg.data_source)
    orphan_tickers = fetch_orphan_tickers(universe_tag=universe_tag, as_of_date=as_of_date, data_source=cfg.data_source)

    missing_tickers: list[str] = []
    stale_tickers: list[str] = []
    failed_tickers: list[str] = []
    ok_tickers: list[str] = []
    anomaly_count = 0

    for ticker in active_tickers:
        stat = price_stats.get(ticker)
        anomaly = anomaly_stats.get(ticker)
        if anomaly is not None:
            anomaly_count += int(anomaly["anomaly_count"] or 0)
            failed_tickers.append(ticker)

        if stat is None or int(stat["row_count"] or 0) <= 0:
            missing_tickers.append(ticker)
            continue

        last_trade_date = stat["last_trade_date"]
        if last_trade_date is None:
            missing_tickers.append(ticker)
            continue

        stale_days = (as_of_date - last_trade_date).days
        if stale_days > stale_days_limit:
            stale_tickers.append(ticker)
            continue

        if anomaly is not None:
            continue

        ok_tickers.append(ticker)

    has_any_price_data = bool(price_stats)
    affected_tickers = set(missing_tickers) | set(stale_tickers) | set(failed_tickers)
    quality_status = _decide_quality_status(
        active_ticker_count=active_ticker_count,
        affected_ticker_count=len(affected_tickers),
        has_any_price_data=has_any_price_data,
    )

    summary = _build_summary(
        as_of_date=as_of_date,
        stale_days_limit=stale_days_limit,
        missing_tickers=missing_tickers,
        stale_tickers=stale_tickers,
        failed_tickers=failed_tickers,
        orphan_tickers=orphan_tickers,
        anomaly_count=anomaly_count,
    )

    LOGGER.info(
        "[US_QUALITY] price_ok=%s missing=%s stale=%s failed=%s",
        len(ok_tickers),
        len(missing_tickers),
        len(stale_tickers),
        len(failed_tickers),
    )
    LOGGER.info("[US_QUALITY] anomaly_count=%s", anomaly_count)
    LOGGER.info("[US_QUALITY] orphan_tickers=%s", len(orphan_tickers))
    LOGGER.info("[US_QUALITY] status=%s", quality_status)

    if verbose:
        LOGGER.info("[US_QUALITY] stale tickers: %s", _format_ticker_sample(stale_tickers, limit=200))
        LOGGER.info("[US_QUALITY] missing tickers: %s", _format_ticker_sample(missing_tickers, limit=200))
        LOGGER.info("[US_QUALITY] failed tickers: %s", _format_ticker_sample(failed_tickers, limit=200))
        LOGGER.info("[US_QUALITY] orphan tickers: %s", _format_ticker_sample(orphan_tickers, limit=200))
    else:
        if stale_tickers:
            LOGGER.info("[US_QUALITY] stale tickers: %s", _format_ticker_sample(stale_tickers))
        if missing_tickers:
            LOGGER.info("[US_QUALITY] missing tickers: %s", _format_ticker_sample(missing_tickers))

    long_stale = [
        ticker
        for ticker in stale_tickers
        if (as_of_date - price_stats[ticker]["last_trade_date"]).days >= LONG_STALE_DAYS
    ]
    if long_stale:
        LOGGER.info("[US_QUALITY] long stale tickers(>=30d): %s", _format_ticker_sample(long_stale))

    return ValidationResult(
        active_ticker_count=active_ticker_count,
        total_ticker_count=total_ticker_count,
        price_ok_count=len(ok_tickers),
        price_missing_count=len(missing_tickers),
        stale_ticker_count=len(stale_tickers),
        failed_ticker_count=len(failed_tickers),
        anomaly_count=anomaly_count,
        orphan_ticker_count=len(orphan_tickers),
        quality_status=quality_status,
        missing_tickers=missing_tickers,
        stale_tickers=stale_tickers,
        failed_tickers=failed_tickers,
        orphan_tickers=orphan_tickers,
        summary=summary,
    )


def validate_and_save_quality_report(*, universe_tag: str, as_of_date: date, verbose: bool) -> ValidationResult:
    result = validate_price_data(universe_tag=universe_tag, as_of_date=as_of_date, verbose=verbose)
    upsert_quality_report(
        {
            "check_date": as_of_date,
            "universe_tag": universe_tag,
            "total_ticker_count": result.total_ticker_count,
            "active_ticker_count": result.active_ticker_count,
            "price_ok_count": result.price_ok_count,
            "price_missing_count": result.price_missing_count,
            "stale_ticker_count": result.stale_ticker_count,
            "failed_ticker_count": result.failed_ticker_count,
            "quality_status": result.quality_status,
            "summary": result.summary,
        }
    )
    LOGGER.info("[US_QUALITY] Saved quality report check_date=%s universe=%s", as_of_date.isoformat(), universe_tag)
    LOGGER.info("[US_QUALITY] Completed")
    return result


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_us_stock_config()
    universe_tag = str(args.universe or cfg.universe).strip().upper() or "NASDAQ100"
    as_of_date = parse_iso_date(args.as_of_date, field_name="as_of_date") or date.today()

    if cfg.data_source != SUPPORTED_SOURCE:
        raise SystemExit(
            f"[US_QUALITY] Unsupported US_STOCK_DATA_SOURCE='{cfg.data_source}'. Only '{SUPPORTED_SOURCE}' is supported."
        )

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_QUALITY] DB connection failed: {exc}") from exc

    validate_and_save_quality_report(universe_tag=universe_tag, as_of_date=as_of_date, verbose=bool(args.verbose))


if __name__ == "__main__":
    main()
