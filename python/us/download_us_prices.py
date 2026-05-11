from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import USStockConfig, load_us_stock_config, parse_iso_date, resolve_price_window
from python.us.us_db import (
    fetch_active_tickers,
    fetch_last_trade_dates,
    get_us_engine,
    insert_collect_log_rows,
    upsert_price_rows,
)


LOGGER = logging.getLogger("us_price")
SUPPORTED_SOURCE = "yfinance"
RUN_STAGE = "us_price_collect"


@dataclass(frozen=True)
class CollectWindow:
    start_date: date
    end_date: date
    mode: str


@dataclass(frozen=True)
class PriceCollectResult:
    universe_tag: str
    mode: str
    active_ticker_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    total_rows: int
    start_date: date | None
    end_date: date


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_stock_config()
    parser = argparse.ArgumentParser(description="Collect US stock daily OHLCV into market.us_stock_daily_price.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag. Default: env US_STOCK_UNIVERSE.")
    parser.add_argument("--backfill", action="store_true", help="Collect historical data using backfill years.")
    parser.add_argument("--incremental", action="store_true", help="Collect from each ticker's latest saved date + 1.")
    parser.add_argument("--start-date", default=None, help="Explicit start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Explicit end date. Format: YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for testing.")
    return parser.parse_args()


def _validate_mode(args: argparse.Namespace) -> str:
    if args.backfill and args.incremental:
        raise ValueError("Choose only one mode: --backfill or --incremental.")
    if args.incremental:
        if args.start_date or args.end_date:
            raise ValueError("--incremental cannot be combined with --start-date or --end-date.")
        return "incremental"
    if args.backfill:
        return "backfill"
    if args.start_date or args.end_date:
        return "date_range"
    return "date_range"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(num):
        return None
    return num


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(num):
        return None
    return int(num)


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date", "open_price", "high_price", "low_price", "close_price", "adj_close_price", "volume"])

    local = frame.copy()
    if isinstance(local.columns, pd.MultiIndex):
        local.columns = local.columns.get_level_values(0)
    local = local.reset_index()
    rename_map = {
        "Date": "trade_date",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Adj Close": "adj_close_price",
        "Volume": "volume",
    }
    local = local.rename(columns=rename_map)
    for required in ["trade_date", "open_price", "high_price", "low_price", "close_price", "adj_close_price", "volume"]:
        if required not in local.columns:
            local[required] = pd.NA
    local["trade_date"] = pd.to_datetime(local["trade_date"], errors="coerce").dt.date
    local = local.dropna(subset=["trade_date", "close_price"]).copy()
    return local[["trade_date", "open_price", "high_price", "low_price", "close_price", "adj_close_price", "volume"]]


def _fetch_batch_yfinance(tickers: list[str], start_date: date, end_date: date) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    if not tickers:
        return {}

    batch_text = " ".join(tickers)
    raw = yf.download(
        tickers=batch_text,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )

    if raw is None or raw.empty:
        return {ticker: pd.DataFrame() for ticker in tickers}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        top_level = list(raw.columns.get_level_values(0).unique())
        second_level = list(raw.columns.get_level_values(1).unique())

        if set(tickers).issubset(set(top_level)):
            for ticker in tickers:
                result[ticker] = _normalize_download_frame(raw[ticker])
            return result

        if set(tickers).issubset(set(second_level)):
            for ticker in tickers:
                subset = raw.xs(ticker, axis=1, level=1)
                result[ticker] = _normalize_download_frame(subset)
            return result

    if len(tickers) == 1:
        result[tickers[0]] = _normalize_download_frame(raw)
        return result

    return {ticker: pd.DataFrame() for ticker in tickers}


def _fetch_single_yfinance(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    return _fetch_batch_yfinance([ticker], start_date, end_date).get(ticker, pd.DataFrame())


def _build_price_rows(ticker: str, frame: pd.DataFrame, data_source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "trade_date": row["trade_date"],
                "ticker": ticker,
                "open_price": _safe_float(row.get("open_price")),
                "high_price": _safe_float(row.get("high_price")),
                "low_price": _safe_float(row.get("low_price")),
                "close_price": _safe_float(row.get("close_price")),
                "adj_close_price": _safe_float(row.get("adj_close_price")),
                "volume": _safe_int(row.get("volume")),
                "data_source": data_source,
            }
        )
    return rows


def _log_row(
    *,
    collect_date: date,
    ticker: str,
    universe_tag: str,
    data_source: str,
    status: str,
    row_count: int,
    start_date: date,
    end_date: date,
    error_message: str | None = None,
) -> dict[str, object]:
    return {
        "collect_date": collect_date,
        "ticker": ticker,
        "universe_tag": universe_tag,
        "data_source": data_source,
        "status": status,
        "row_count": int(row_count),
        "start_date": start_date,
        "end_date": end_date,
        "error_message": error_message,
        "run_stage": RUN_STAGE,
    }


def _resolve_global_window(cfg: USStockConfig, args: argparse.Namespace, mode: str) -> CollectWindow:
    if mode == "incremental":
        end_date = parse_iso_date(args.end_date, field_name="end_date") or parse_iso_date(
            cfg.price_end_date, field_name="US_STOCK_PRICE_END_DATE"
        ) or date.today()
        return CollectWindow(start_date=end_date, end_date=end_date, mode=mode)
    start_date, end_date = resolve_price_window(
        cfg=cfg,
        cli_start_date=args.start_date,
        cli_end_date=args.end_date,
        backfill=(mode == "backfill"),
    )
    return CollectWindow(start_date=start_date, end_date=end_date, mode=mode)


def run_non_incremental(
    *,
    tickers: list[str],
    universe_tag: str,
    data_source: str,
    start_date: date,
    end_date: date,
    batch_size: int,
    sleep_sec: float,
) -> None:
    success = 0
    failed = 0
    skipped = 0
    total_rows = 0
    collect_date = date.today()

    for batch in _chunked(tickers, max(1, batch_size)):
        try:
            batch_frames = _fetch_batch_yfinance(batch, start_date, end_date)
        except Exception as exc:
            batch_frames = {}
            LOGGER.warning("[US_PRICE] Batch fallback triggered tickers=%s error=%s", ",".join(batch), exc)
            for ticker in batch:
                try:
                    batch_frames[ticker] = _fetch_single_yfinance(ticker, start_date, end_date)
                except Exception as inner_exc:
                    batch_frames[ticker] = pd.DataFrame()
                    failed += 1
                    insert_collect_log_rows(
                        [
                            _log_row(
                                collect_date=collect_date,
                                ticker=ticker,
                                universe_tag=universe_tag,
                                data_source=data_source,
                                status="FAILED",
                                row_count=0,
                                start_date=start_date,
                                end_date=end_date,
                                error_message=str(inner_exc),
                            )
                        ]
                    )
                    LOGGER.info("[US_PRICE] %s FAILED error=%s", ticker, inner_exc)

        for ticker in batch:
            if ticker not in batch_frames:
                continue
            frame = batch_frames[ticker]
            try:
                if frame.empty:
                    skipped += 1
                    insert_collect_log_rows(
                        [
                            _log_row(
                                collect_date=collect_date,
                                ticker=ticker,
                                universe_tag=universe_tag,
                                data_source=data_source,
                                status="SKIPPED",
                                row_count=0,
                                start_date=start_date,
                                end_date=end_date,
                                error_message="No data returned from yfinance.",
                            )
                        ]
                    )
                    LOGGER.info("[US_PRICE] %s SKIPPED rows=0", ticker)
                    continue

                row_count = upsert_price_rows(_build_price_rows(ticker, frame, data_source))
                insert_collect_log_rows(
                    [
                        _log_row(
                            collect_date=collect_date,
                            ticker=ticker,
                            universe_tag=universe_tag,
                            data_source=data_source,
                            status="SUCCESS",
                            row_count=row_count,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    ]
                )
                success += 1
                total_rows += row_count
                LOGGER.info("[US_PRICE] %s SUCCESS rows=%d", ticker, row_count)
            except Exception as exc:
                failed += 1
                insert_collect_log_rows(
                    [
                        _log_row(
                            collect_date=collect_date,
                            ticker=ticker,
                            universe_tag=universe_tag,
                            data_source=data_source,
                            status="FAILED",
                            row_count=0,
                            start_date=start_date,
                            end_date=end_date,
                            error_message=str(exc),
                        )
                    ]
                )
                LOGGER.info("[US_PRICE] %s FAILED error=%s", ticker, exc)
        time.sleep(max(0.0, sleep_sec))

    LOGGER.info("[US_PRICE] Finished success=%d failed=%d skipped=%d total_rows=%d", success, failed, skipped, total_rows)
    return PriceCollectResult(
        universe_tag=universe_tag,
        mode="date_range",
        active_ticker_count=len(tickers),
        success_count=success,
        failed_count=failed,
        skipped_count=skipped,
        total_rows=total_rows,
        start_date=start_date,
        end_date=end_date,
    )


def run_incremental(
    *,
    cfg: USStockConfig,
    tickers: list[str],
    universe_tag: str,
    data_source: str,
    end_date: date,
    sleep_sec: float,
) -> None:
    collect_date = date.today()
    last_dates = fetch_last_trade_dates(tickers)
    success = 0
    failed = 0
    skipped = 0
    total_rows = 0

    for ticker in tickers:
        try:
            existing_last_date = last_dates.get(ticker)
            if existing_last_date is not None:
                start_date = existing_last_date + timedelta(days=1)
            else:
                fallback_start, _ = resolve_price_window(cfg=cfg, backfill=True)
                start_date = fallback_start

            if start_date > end_date:
                skipped += 1
                insert_collect_log_rows(
                    [
                        _log_row(
                            collect_date=collect_date,
                            ticker=ticker,
                            universe_tag=universe_tag,
                            data_source=data_source,
                            status="SKIPPED",
                            row_count=0,
                            start_date=start_date,
                            end_date=end_date,
                            error_message="No new date range to collect.",
                        )
                    ]
                )
                LOGGER.info("[US_PRICE] %s SKIPPED rows=0", ticker)
                continue

            frame = _fetch_single_yfinance(ticker, start_date, end_date)
            if frame.empty:
                skipped += 1
                insert_collect_log_rows(
                    [
                        _log_row(
                            collect_date=collect_date,
                            ticker=ticker,
                            universe_tag=universe_tag,
                            data_source=data_source,
                            status="SKIPPED",
                            row_count=0,
                            start_date=start_date,
                            end_date=end_date,
                            error_message="No data returned from yfinance.",
                        )
                    ]
                )
                LOGGER.info("[US_PRICE] %s SKIPPED rows=0", ticker)
                time.sleep(max(0.0, sleep_sec))
                continue

            row_count = upsert_price_rows(_build_price_rows(ticker, frame, data_source))
            insert_collect_log_rows(
                [
                    _log_row(
                        collect_date=collect_date,
                        ticker=ticker,
                        universe_tag=universe_tag,
                        data_source=data_source,
                        status="SUCCESS",
                        row_count=row_count,
                        start_date=start_date,
                        end_date=end_date,
                    )
                ]
            )
            success += 1
            total_rows += row_count
            LOGGER.info("[US_PRICE] %s SUCCESS rows=%d", ticker, row_count)
        except Exception as exc:
            failed += 1
            insert_collect_log_rows(
                [
                    _log_row(
                        collect_date=collect_date,
                        ticker=ticker,
                        universe_tag=universe_tag,
                        data_source=data_source,
                        status="FAILED",
                        row_count=0,
                        start_date=last_dates.get(ticker, end_date),
                        end_date=end_date,
                        error_message=str(exc),
                    )
                ]
            )
            LOGGER.info("[US_PRICE] %s FAILED error=%s", ticker, exc)
        time.sleep(max(0.0, sleep_sec))

    LOGGER.info("[US_PRICE] Finished success=%d failed=%d skipped=%d total_rows=%d", success, failed, skipped, total_rows)
    return PriceCollectResult(
        universe_tag=universe_tag,
        mode="incremental",
        active_ticker_count=len(tickers),
        success_count=success,
        failed_count=failed,
        skipped_count=skipped,
        total_rows=total_rows,
        start_date=None,
        end_date=end_date,
    )


def collect_us_prices(
    *,
    universe_tag: str,
    mode: str,
    limit: int | None = None,
    start_date_text: str | None = None,
    end_date_text: str | None = None,
) -> PriceCollectResult:
    cfg = load_us_stock_config()

    if str(cfg.data_source).strip().lower() != SUPPORTED_SOURCE:
        raise ValueError(f"Unsupported US_STOCK_DATA_SOURCE='{cfg.data_source}'. Only yfinance is supported in Phase 1-3.")

    namespace = argparse.Namespace(
        universe=universe_tag,
        backfill=(mode == "backfill"),
        incremental=(mode == "incremental"),
        start_date=start_date_text,
        end_date=end_date_text,
        limit=limit,
    )
    normalized_mode = _validate_mode(namespace)
    window = _resolve_global_window(cfg, namespace, normalized_mode)

    LOGGER.info("[US_PRICE] Start collect universe=%s source=%s mode=%s", universe_tag, cfg.data_source, normalized_mode)

    tickers = fetch_active_tickers(universe_tag)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        tickers = tickers[:limit]

    if not tickers:
        LOGGER.info("[US_PRICE] Active tickers=0")
        LOGGER.info("[US_PRICE] No active tickers found for universe=%s. Exiting.", universe_tag)
        return PriceCollectResult(
            universe_tag=universe_tag,
            mode=normalized_mode,
            active_ticker_count=0,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            total_rows=0,
            start_date=window.start_date if normalized_mode != "incremental" else None,
            end_date=window.end_date,
        )

    LOGGER.info("[US_PRICE] Active tickers=%d", len(tickers))
    if normalized_mode != "incremental":
        LOGGER.info("[US_PRICE] Period start=%s end=%s", window.start_date.isoformat(), window.end_date.isoformat())
        result = run_non_incremental(
            tickers=tickers,
            universe_tag=universe_tag,
            data_source=cfg.data_source,
            start_date=window.start_date,
            end_date=window.end_date,
            batch_size=cfg.batch_size,
            sleep_sec=cfg.request_sleep_sec,
        )
        return PriceCollectResult(
            universe_tag=result.universe_tag,
            mode=normalized_mode,
            active_ticker_count=result.active_ticker_count,
            success_count=result.success_count,
            failed_count=result.failed_count,
            skipped_count=result.skipped_count,
            total_rows=result.total_rows,
            start_date=window.start_date,
            end_date=window.end_date,
        )

    LOGGER.info("[US_PRICE] Period end=%s incremental_per_ticker=true", window.end_date.isoformat())
    return run_incremental(
        cfg=cfg,
        tickers=tickers,
        universe_tag=universe_tag,
        data_source=cfg.data_source,
        end_date=window.end_date,
        sleep_sec=cfg.request_sleep_sec,
    )


def main() -> None:
    setup_logging()
    cfg = load_us_stock_config()
    args = parse_args()

    if str(cfg.data_source).strip().lower() != SUPPORTED_SOURCE:
        raise SystemExit(f"Unsupported US_STOCK_DATA_SOURCE='{cfg.data_source}'. Only yfinance is supported in Phase 1-3.")

    universe_tag = str(args.universe or cfg.universe).strip().upper() or cfg.universe
    mode = _validate_mode(args)

    try:
        get_us_engine().connect().close()
    except Exception as exc:
        raise SystemExit(f"DB connection failed: {exc}") from exc

    try:
        collect_us_prices(
            universe_tag=universe_tag,
            mode=mode,
            limit=args.limit,
            start_date_text=args.start_date,
            end_date_text=args.end_date,
        )
    except KeyboardInterrupt:
        LOGGER.info("[US_PRICE] Interrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
