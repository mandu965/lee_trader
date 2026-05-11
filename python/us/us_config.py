from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import os
from pathlib import Path


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class USStockConfig:
    enabled: bool
    universe: str
    data_source: str
    root_dir: Path
    data_dir: Path
    us_data_dir: Path
    default_universe_csv: Path
    price_backfill_years: int
    price_start_date: str
    price_end_date: str
    stale_days_limit: int
    batch_size: int
    request_sleep_sec: float
    paper_trading_enabled: bool
    live_trading_enabled: bool
    live_broker: str


@dataclass(frozen=True)
class USFinancialCollectorConfig:
    enabled: bool
    source: str
    universe: str
    period_types: tuple[str, ...]
    lookback_years: int
    max_tickers_per_run: int
    sleep_sec: float
    retry_count: int
    retry_sleep_sec: float
    fail_fast: bool
    write_mode: str
    log_level: str


def load_us_stock_config() -> USStockConfig:
    """
    Load Project C Phase 1 environment configuration.

    This module does not implement data collection or trading logic.
    """
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data"
    us_data_dir = data_dir / "us"
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    return USStockConfig(
        enabled=_flag("US_STOCK_ENABLED", "0"),
        universe=universe,
        data_source=str(os.environ.get("US_STOCK_DATA_SOURCE", "yfinance")).strip() or "yfinance",
        root_dir=root_dir,
        data_dir=data_dir,
        us_data_dir=us_data_dir,
        default_universe_csv=us_data_dir / f"{universe.lower()}_universe.csv",
        price_backfill_years=int(os.environ.get("US_STOCK_PRICE_BACKFILL_YEARS", "5")),
        price_start_date=str(os.environ.get("US_STOCK_PRICE_START_DATE", "")).strip(),
        price_end_date=str(os.environ.get("US_STOCK_PRICE_END_DATE", "")).strip(),
        stale_days_limit=int(os.environ.get("US_STOCK_STALE_DAYS_LIMIT", "3")),
        batch_size=int(os.environ.get("US_STOCK_BATCH_SIZE", "20")),
        request_sleep_sec=float(os.environ.get("US_STOCK_REQUEST_SLEEP_SEC", "1")),
        paper_trading_enabled=_flag("US_PAPER_TRADING_ENABLED", "0"),
        live_trading_enabled=_flag("US_LIVE_TRADING_ENABLED", "0"),
        live_broker=str(os.environ.get("US_LIVE_BROKER", "none")).strip() or "none",
    )


def _split_csv(value: str | None) -> tuple[str, ...]:
    parts = [part.strip() for part in str(value or "").split(",")]
    return tuple(part for part in parts if part)


def load_us_financial_collector_config() -> USFinancialCollectorConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    period_types = tuple(name.lower() for name in _split_csv(os.environ.get("US_FINANCIAL_PERIOD_TYPES", "annual,quarterly")))
    return USFinancialCollectorConfig(
        enabled=_flag("US_FINANCIAL_COLLECT_ENABLED", "0"),
        source=str(os.environ.get("US_FINANCIAL_SOURCE", "yfinance")).strip() or "yfinance",
        universe=universe,
        period_types=period_types or ("annual", "quarterly"),
        lookback_years=int(os.environ.get("US_FINANCIAL_LOOKBACK_YEARS", "5")),
        max_tickers_per_run=int(os.environ.get("US_FINANCIAL_MAX_TICKERS_PER_RUN", "100")),
        sleep_sec=float(os.environ.get("US_FINANCIAL_SLEEP_SEC", "1.0")),
        retry_count=int(os.environ.get("US_FINANCIAL_RETRY_COUNT", "3")),
        retry_sleep_sec=float(os.environ.get("US_FINANCIAL_RETRY_SLEEP_SEC", "5")),
        fail_fast=_flag("US_FINANCIAL_FAIL_FAST", "0"),
        write_mode=str(os.environ.get("US_FINANCIAL_WRITE_MODE", "upsert")).strip().lower() or "upsert",
        log_level=str(os.environ.get("US_FINANCIAL_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def resolve_universe_csv_path(universe_tag: str, override: str | None = None) -> Path:
    cfg = load_us_stock_config()
    if override:
        path = Path(override)
        return path if path.is_absolute() else cfg.root_dir / path
    return cfg.us_data_dir / f"{str(universe_tag).strip().lower()}_universe.csv"


def parse_iso_date(value: str | None, *, field_name: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: '{text}'. Expected YYYY-MM-DD.") from exc


def resolve_price_window(
    *,
    cfg: USStockConfig,
    cli_start_date: str | None = None,
    cli_end_date: str | None = None,
    backfill: bool = False,
    today: date | None = None,
) -> tuple[date, date]:
    base_today = today or date.today()
    start_date = parse_iso_date(cli_start_date, field_name="start_date") or parse_iso_date(
        cfg.price_start_date, field_name="US_STOCK_PRICE_START_DATE"
    )
    end_date = parse_iso_date(cli_end_date, field_name="end_date") or parse_iso_date(
        cfg.price_end_date, field_name="US_STOCK_PRICE_END_DATE"
    )
    if end_date is None:
        end_date = base_today

    if start_date is None:
        if backfill:
            start_date = end_date - timedelta(days=max(1, cfg.price_backfill_years) * 366)
        else:
            start_date = end_date

    if start_date > end_date:
        raise ValueError(
            f"Invalid date window: start_date {start_date.isoformat()} is after end_date {end_date.isoformat()}."
        )
    return start_date, end_date
