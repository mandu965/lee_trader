from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import USFinancialCollectorConfig, load_us_financial_collector_config
from python.us.us_db import (
    fetch_active_tickers,
    get_us_engine,
    upsert_financial_rows,
)


LOGGER = logging.getLogger("us_financial")
SUPPORTED_SOURCE = "yfinance"
SUPPORTED_PERIOD_TYPES = {"annual", "quarterly"}

YF_FINANCIAL_STATEMENT_MAP = {
    "Total Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "EBITDA": "ebitda",
}

YF_BALANCE_SHEET_MAP = {
    "Total Assets": "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity": "total_equity",
    "Total Equity Gross Minority Interest": "total_equity",
}

YF_CASHFLOW_MAP = {
    "Operating Cash Flow": "operating_cash_flow",
    "Cash Flow From Continuing Operating Activities": "operating_cash_flow",
    "Investing Cash Flow": "investing_cash_flow",
    "Cash Flow From Continuing Investing Activities": "investing_cash_flow",
    "Financing Cash Flow": "financing_cash_flow",
    "Cash Flow From Continuing Financing Activities": "financing_cash_flow",
    "Free Cash Flow": "free_cash_flow",
}

YF_FINANCIAL_METRIC_FRAME_MAP = {
    "Diluted EPS": "eps",
    "Basic EPS": "eps",
}

YF_INFO_MAP = {
    "trailingEps": "eps",
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "sharesOutstanding": "shares_outstanding",
    "marketCap": "market_cap",
    "trailingPE": "per",
    "priceToBook": "pbr",
    "priceToSalesTrailing12Months": "psr",
    "enterpriseToEbitda": "ev_ebitda",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "dividendYield": "dividend_yield",
}


@dataclass(frozen=True)
class FinancialCollectResult:
    processed_ticker_count: int
    success_ticker_count: int
    failed_ticker_count: int
    skipped_ticker_count: int
    statement_row_count: int
    metric_row_count: int
    failed_tickers: list[str]


@dataclass(frozen=True)
class TickerCollectResult:
    ticker: str
    statement_rows: int
    metric_rows: int
    skipped_rows: int
    status: str
    error_message: str | None = None


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_financial_collector_config()
    parser = argparse.ArgumentParser(description="Collect US stock financial raw data from yfinance.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag used to load active tickers.")
    parser.add_argument("--ticker", action="append", default=None, help="Collect a specific ticker. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Override per-run ticker limit.")
    parser.add_argument("--period-types", default=None, help="Override period types. Example: annual,quarterly")
    return parser.parse_args()


def _normalize_period_types(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    out = tuple(part for part in parts if part)
    if not out:
        raise ValueError("No period types configured.")
    unsupported = [name for name in out if name not in SUPPORTED_PERIOD_TYPES]
    if unsupported:
        raise ValueError(f"Unsupported period types: {unsupported}. Supported: annual, quarterly.")
    return out


def _safe_number(value: Any, *, field_name: str, ticker: str) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("[US_FINANCIAL] %s warning field=%s invalid_number=%r", ticker, field_name, value)
        return None


def _safe_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _frame_value(frame: pd.DataFrame | None, field_name: str, fiscal_date: date, *, ticker: str) -> float | None:
    if frame is None or frame.empty:
        return None
    target_col = None
    for column in frame.columns:
        column_date = _safe_date(column)
        if column_date == fiscal_date:
            target_col = column
            break
    if target_col is None or field_name not in frame.index:
        return None
    return _safe_number(frame.at[field_name, target_col], field_name=field_name, ticker=ticker)


def _collect_period_dates(frames: list[pd.DataFrame | None], *, lookback_start: date) -> list[date]:
    dates: set[date] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            fiscal_date = _safe_date(column)
            if fiscal_date is not None and fiscal_date >= lookback_start:
                dates.add(fiscal_date)
    return sorted(dates)


def _extract_info_metrics(info: dict[str, Any], *, ticker: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for yf_name, field_name in YF_INFO_MAP.items():
        out[field_name] = _safe_number(info.get(yf_name), field_name=yf_name, ticker=ticker)
    currency = str(info.get("financialCurrency") or info.get("currency") or "").strip()
    out["currency"] = currency or None
    return out


def _build_rows_for_period(
    *,
    ticker: str,
    period_type: str,
    financials: pd.DataFrame | None,
    balance_sheet: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    info_metrics: dict[str, object],
    source: str,
    collected_at: datetime,
    lookback_start: date,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    statement_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    skipped_rows = 0
    period_dates = _collect_period_dates([financials, balance_sheet, cashflow], lookback_start=lookback_start)

    for fiscal_date in period_dates:
        base = {
            "ticker": ticker,
            "market": "US",
            "period_type": period_type,
            "fiscal_date": fiscal_date,
            "reported_date": None,
            "currency": info_metrics.get("currency"),
            "source": source,
            "source_updated_at": None,
            "collected_at": collected_at,
        }
        if not base["ticker"] or base["fiscal_date"] is None:
            skipped_rows += 1
            continue

        statement_row = dict(base)
        for yf_name, internal_name in YF_FINANCIAL_STATEMENT_MAP.items():
            statement_row[internal_name] = _frame_value(financials, yf_name, fiscal_date, ticker=ticker)
        for yf_name, internal_name in YF_BALANCE_SHEET_MAP.items():
            if statement_row.get(internal_name) is None:
                statement_row[internal_name] = _frame_value(balance_sheet, yf_name, fiscal_date, ticker=ticker)
        for yf_name, internal_name in YF_CASHFLOW_MAP.items():
            if statement_row.get(internal_name) is None:
                statement_row[internal_name] = _frame_value(cashflow, yf_name, fiscal_date, ticker=ticker)

        metric_row = dict(base)
        for yf_name, internal_name in YF_FINANCIAL_METRIC_FRAME_MAP.items():
            if metric_row.get(internal_name) is None:
                metric_row[internal_name] = _frame_value(financials, yf_name, fiscal_date, ticker=ticker)
        for field_name in [
            "eps",
            "roe",
            "roa",
            "shares_outstanding",
            "market_cap",
            "per",
            "pbr",
            "psr",
            "ev_ebitda",
            "debt_to_equity",
            "current_ratio",
            "dividend_yield",
        ]:
            if metric_row.get(field_name) is None:
                metric_row[field_name] = info_metrics.get(field_name)

        statement_rows.append(statement_row)
        metric_rows.append(metric_row)

    return statement_rows, metric_rows, skipped_rows


def _collect_ticker_frames(ticker_obj: Any, period_type: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if period_type == "annual":
        return ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
    return ticker_obj.quarterly_financials, ticker_obj.quarterly_balance_sheet, ticker_obj.quarterly_cashflow


def resolve_target_tickers(*, universe_tag: str, limit: int | None, explicit_tickers: list[str] | None) -> list[str]:
    if explicit_tickers:
        return [str(ticker).strip().upper() for ticker in explicit_tickers if str(ticker).strip()]
    tickers = fetch_active_tickers(universe_tag)
    if not tickers:
        raise RuntimeError(
            f"No active tickers found in market.us_stock_universe for universe={universe_tag}. "
            "Apply the US universe migration and load the universe first."
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        tickers = tickers[:limit]
    return tickers


def collect_single_ticker_financials(
    *,
    ticker: str,
    cfg: USFinancialCollectorConfig,
    ticker_factory,
    row_writer,
) -> TickerCollectResult:
    collected_at = datetime.now(timezone.utc)
    lookback_start = (collected_at.date() - timedelta(days=max(1, cfg.lookback_years) * 366))
    ticker_obj = ticker_factory(ticker)
    info_metrics = _extract_info_metrics(getattr(ticker_obj, "info", {}) or {}, ticker=ticker)

    statement_rows_all: list[dict[str, object]] = []
    metric_rows_all: list[dict[str, object]] = []
    skipped_rows = 0

    for period_type in cfg.period_types:
        financials, balance_sheet, cashflow = _collect_ticker_frames(ticker_obj, period_type)
        statement_rows, metric_rows, skipped = _build_rows_for_period(
            ticker=ticker,
            period_type=period_type,
            financials=financials,
            balance_sheet=balance_sheet,
            cashflow=cashflow,
            info_metrics=info_metrics,
            source=cfg.source,
            collected_at=collected_at,
            lookback_start=lookback_start,
        )
        LOGGER.info(
            "[US_FINANCIAL] %s period=%s statement_rows=%s metric_rows=%s skipped=%s",
            ticker,
            period_type,
            len(statement_rows),
            len(metric_rows),
            skipped,
        )
        statement_rows_all.extend(statement_rows)
        metric_rows_all.extend(metric_rows)
        skipped_rows += skipped

    if not statement_rows_all and not metric_rows_all:
        return TickerCollectResult(
            ticker=ticker,
            statement_rows=0,
            metric_rows=0,
            skipped_rows=skipped_rows,
            status="SKIPPED",
            error_message="No financial rows available from yfinance.",
        )

    row_writer(statement_rows_all, metric_rows_all)
    return TickerCollectResult(
        ticker=ticker,
        statement_rows=len(statement_rows_all),
        metric_rows=len(metric_rows_all),
        skipped_rows=skipped_rows,
        status="SUCCESS",
    )


def collect_us_financials(
    *,
    cfg: USFinancialCollectorConfig,
    universe_tag: str,
    explicit_tickers: list[str] | None = None,
    limit: int | None = None,
    ticker_factory=None,
    row_writer=upsert_financial_rows,
    sleep_fn=time.sleep,
) -> FinancialCollectResult:
    if not cfg.enabled:
        LOGGER.info("[US_FINANCIAL] US_FINANCIAL_COLLECT_ENABLED=0. Skip collector.")
        return FinancialCollectResult(0, 0, 0, 0, 0, 0, [])

    if str(cfg.source).strip().lower() != SUPPORTED_SOURCE:
        raise ValueError(f"Unsupported US_FINANCIAL_SOURCE='{cfg.source}'. Only '{SUPPORTED_SOURCE}' is supported.")
    if cfg.write_mode != "upsert":
        raise ValueError(f"Unsupported US_FINANCIAL_WRITE_MODE='{cfg.write_mode}'. Only 'upsert' is supported.")
    _normalize_period_types(cfg.period_types)

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(f"DB connection failed: {exc}") from exc

    target_limit = limit if limit is not None else cfg.max_tickers_per_run
    tickers = resolve_target_tickers(universe_tag=universe_tag, limit=target_limit, explicit_tickers=explicit_tickers)
    LOGGER.info(
        "[US_FINANCIAL] Start collect source=%s universe=%s period_types=%s tickers=%s",
        cfg.source,
        universe_tag,
        ",".join(cfg.period_types),
        len(tickers),
    )
    LOGGER.info(
        "[US_FINANCIAL] Config lookback_years=%s max_tickers_per_run=%s retry_count=%s sleep_sec=%s fail_fast=%s write_mode=%s",
        cfg.lookback_years,
        cfg.max_tickers_per_run,
        cfg.retry_count,
        cfg.sleep_sec,
        int(cfg.fail_fast),
        cfg.write_mode,
    )

    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker

    success = 0
    failed = 0
    skipped = 0
    statement_rows = 0
    metric_rows = 0
    failed_tickers: list[str] = []

    for ticker in tickers:
        LOGGER.info("[US_FINANCIAL] %s start", ticker)
        last_error: Exception | None = None
        for attempt in range(1, max(1, cfg.retry_count) + 1):
            try:
                result = collect_single_ticker_financials(
                    ticker=ticker,
                    cfg=cfg,
                    ticker_factory=ticker_factory,
                    row_writer=row_writer,
                )
                if result.status == "SUCCESS":
                    success += 1
                    statement_rows += result.statement_rows
                    metric_rows += result.metric_rows
                    LOGGER.info(
                        "[US_FINANCIAL] %s success statement_rows=%s metric_rows=%s skipped_rows=%s",
                        ticker,
                        result.statement_rows,
                        result.metric_rows,
                        result.skipped_rows,
                    )
                else:
                    skipped += 1
                    LOGGER.info("[US_FINANCIAL] %s skipped reason=%s", ticker, result.error_message)
                    last_error = None
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning("[US_FINANCIAL] %s failed attempt=%s error=%s", ticker, attempt, exc)
                if attempt < max(1, cfg.retry_count):
                    sleep_fn(max(0.0, cfg.retry_sleep_sec))
                else:
                    failed += 1
                    failed_tickers.append(ticker)
                    if cfg.fail_fast:
                        raise
        sleep_fn(max(0.0, cfg.sleep_sec))

        if last_error is not None and cfg.fail_fast:
            break

    LOGGER.info(
        "[US_FINANCIAL] Finished success=%s failed=%s skipped=%s statement_rows=%s metric_rows=%s",
        success,
        failed,
        skipped,
        statement_rows,
        metric_rows,
    )
    if failed_tickers:
        LOGGER.info("[US_FINANCIAL] failed_tickers=%s", ",".join(failed_tickers))

    return FinancialCollectResult(
        processed_ticker_count=len(tickers),
        success_ticker_count=success,
        failed_ticker_count=failed,
        skipped_ticker_count=skipped,
        statement_row_count=statement_rows,
        metric_row_count=metric_rows,
        failed_tickers=failed_tickers,
    )


def main() -> None:
    args = parse_args()
    cfg = load_us_financial_collector_config()
    setup_logging(cfg.log_level)
    period_types = _normalize_period_types(args.period_types or cfg.period_types)
    runtime_cfg = USFinancialCollectorConfig(
        enabled=cfg.enabled,
        source=cfg.source,
        universe=str(args.universe or cfg.universe).strip().upper() or cfg.universe,
        period_types=period_types,
        lookback_years=cfg.lookback_years,
        max_tickers_per_run=cfg.max_tickers_per_run,
        sleep_sec=cfg.sleep_sec,
        retry_count=cfg.retry_count,
        retry_sleep_sec=cfg.retry_sleep_sec,
        fail_fast=cfg.fail_fast,
        write_mode=cfg.write_mode,
        log_level=cfg.log_level,
    )
    tickers = [str(name).strip().upper() for name in (args.ticker or []) if str(name).strip()] or None

    try:
        collect_us_financials(
            cfg=runtime_cfg,
            universe_tag=runtime_cfg.universe,
            explicit_tickers=tickers,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        LOGGER.info("[US_FINANCIAL] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_FINANCIAL] Failed error=%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
