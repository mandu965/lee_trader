from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

import pandas as pd

import python.us.collect_us_financials_yfinance as collector_module
from python.us.collect_us_financials_yfinance import (
    FinancialCollectResult,
    USFinancialCollectorConfig,
    _extract_info_metrics,
    _normalize_period_types,
    _safe_number,
    collect_single_ticker_financials,
    collect_us_financials,
)


class DummyTicker:
    def __init__(self) -> None:
        annual_cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
        quarterly_cols = [pd.Timestamp("2026-03-31")]
        self.financials = pd.DataFrame(
            {
                annual_cols[0]: {
                    "Total Revenue": 1000,
                    "Gross Profit": 400,
                    "Operating Income": 200,
                    "Net Income": 150,
                    "EBITDA": 250,
                    "Diluted EPS": 2.5,
                },
                annual_cols[1]: {
                    "Total Revenue": 900,
                    "Gross Profit": 350,
                    "Operating Income": 180,
                    "Net Income": 130,
                    "EBITDA": 220,
                    "Diluted EPS": 2.1,
                },
            }
        )
        self.quarterly_financials = pd.DataFrame(
            {
                quarterly_cols[0]: {
                    "Total Revenue": 260,
                    "Gross Profit": 110,
                    "Operating Income": 60,
                    "Net Income": 45,
                    "EBITDA": 75,
                    "Diluted EPS": 0.7,
                }
            }
        )
        self.balance_sheet = pd.DataFrame(
            {
                annual_cols[0]: {
                    "Total Assets": 5000,
                    "Total Liabilities Net Minority Interest": 2000,
                    "Stockholders Equity": 3000,
                },
                annual_cols[1]: {
                    "Total Assets": 4500,
                    "Total Liabilities Net Minority Interest": 1800,
                    "Stockholders Equity": 2700,
                },
            }
        )
        self.quarterly_balance_sheet = pd.DataFrame(
            {
                quarterly_cols[0]: {
                    "Total Assets": 5100,
                    "Total Liabilities Net Minority Interest": 2050,
                    "Stockholders Equity": 3050,
                }
            }
        )
        self.cashflow = pd.DataFrame(
            {
                annual_cols[0]: {
                    "Operating Cash Flow": 300,
                    "Investing Cash Flow": -120,
                    "Financing Cash Flow": -80,
                    "Free Cash Flow": 180,
                },
                annual_cols[1]: {
                    "Operating Cash Flow": 260,
                    "Investing Cash Flow": -100,
                    "Financing Cash Flow": -70,
                    "Free Cash Flow": 160,
                },
            }
        )
        self.quarterly_cashflow = pd.DataFrame(
            {
                quarterly_cols[0]: {
                    "Operating Cash Flow": 70,
                    "Investing Cash Flow": -30,
                    "Financing Cash Flow": -15,
                    "Free Cash Flow": 40,
                }
            }
        )
        self.info = {
            "financialCurrency": "USD",
            "marketCap": 1000000,
            "priceToBook": 4.2,
            "trailingPE": 20.5,
            "returnOnEquity": 0.18,
            "returnOnAssets": 0.09,
            "debtToEquity": 0.55,
            "currentRatio": 1.7,
            "dividendYield": 0.012,
            "sharesOutstanding": 500000,
        }


def make_config(**overrides) -> USFinancialCollectorConfig:
    values = {
        "enabled": True,
        "source": "yfinance",
        "universe": "NASDAQ100",
        "period_types": ("annual", "quarterly"),
        "lookback_years": 5,
        "max_tickers_per_run": 100,
        "sleep_sec": 0.0,
        "retry_count": 2,
        "retry_sleep_sec": 0.0,
        "fail_fast": False,
        "write_mode": "upsert",
        "log_level": "INFO",
    }
    values.update(overrides)
    return USFinancialCollectorConfig(**values)


class USFinancialCollectorTests(unittest.TestCase):
    def _patch_db_engine(self):
        class _DummyConn:
            def execute(self, _stmt):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _DummyEngine:
            def connect(self):
                return _DummyConn()

        return patch.object(collector_module, "get_us_engine", return_value=_DummyEngine())

    def test_disabled_returns_skipped_result(self) -> None:
        cfg = make_config(enabled=False)
        called = {"factory": 0}

        def ticker_factory(_ticker: str):
            called["factory"] += 1
            return DummyTicker()

        result = collect_us_financials(
            cfg=cfg,
            universe_tag="NASDAQ100",
            explicit_tickers=["AAPL"],
            ticker_factory=ticker_factory,
            sleep_fn=lambda _sec: None,
        )
        self.assertIsInstance(result, FinancialCollectResult)
        self.assertEqual(result.processed_ticker_count, 0)
        self.assertEqual(called["factory"], 0)

    def test_normalize_period_types_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_period_types("annual,monthly")

    def test_safe_number_returns_none_for_invalid_values(self) -> None:
        with self.assertLogs("us_financial", level="WARNING") as logs:
            value = _safe_number("bad-number", field_name="marketCap", ticker="AAPL")
        self.assertIsNone(value)
        self.assertTrue(any("invalid_number" in line for line in logs.output))

    def test_extract_info_metrics_handles_missing_values(self) -> None:
        metrics = _extract_info_metrics({"financialCurrency": "USD", "marketCap": None}, ticker="AAPL")
        self.assertEqual(metrics["currency"], "USD")
        self.assertIsNone(metrics["market_cap"])
        self.assertIsNone(metrics["per"])

    def test_collect_single_ticker_maps_annual_and_quarterly_rows(self) -> None:
        stmt_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        result = collect_single_ticker_financials(
            ticker="AAPL",
            cfg=make_config(),
            ticker_factory=lambda _ticker: DummyTicker(),
            row_writer=lambda s_rows, m_rows: (stmt_rows.extend(s_rows), metric_rows.extend(m_rows)),
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(len(stmt_rows), 3)
        self.assertEqual(len(metric_rows), 3)
        annual_row = next(row for row in stmt_rows if row["period_type"] == "annual" and row["fiscal_date"] == date(2025, 12, 31))
        self.assertEqual(annual_row["revenue"], 1000.0)
        self.assertEqual(annual_row["total_assets"], 5000.0)
        metric_row = next(row for row in metric_rows if row["period_type"] == "annual" and row["fiscal_date"] == date(2025, 12, 31))
        self.assertEqual(metric_row["per"], 20.5)
        self.assertEqual(metric_row["pbr"], 4.2)

    def test_missing_fields_do_not_fail_collection(self) -> None:
        ticker = DummyTicker()
        ticker.cashflow = pd.DataFrame()
        stmt_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        result = collect_single_ticker_financials(
            ticker="AAPL",
            cfg=make_config(period_types=("annual",)),
            ticker_factory=lambda _ticker: ticker,
            row_writer=lambda s_rows, m_rows: (stmt_rows.extend(s_rows), metric_rows.extend(m_rows)),
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNone(stmt_rows[0]["free_cash_flow"])

    def test_fail_fast_false_continues_after_ticker_failure(self) -> None:
        def ticker_factory(ticker: str):
            if ticker == "FAIL":
                raise RuntimeError("boom")
            return DummyTicker()

        with self._patch_db_engine():
            result = collect_us_financials(
                cfg=make_config(retry_count=1, fail_fast=False),
                universe_tag="NASDAQ100",
                explicit_tickers=["FAIL", "AAPL"],
                ticker_factory=ticker_factory,
                row_writer=lambda s_rows, m_rows: (len(s_rows), len(m_rows)),
                sleep_fn=lambda _sec: None,
            )
        self.assertEqual(result.failed_ticker_count, 1)
        self.assertEqual(result.success_ticker_count, 1)
        self.assertEqual(result.failed_tickers, ["FAIL"])

    def test_fail_fast_true_stops_on_first_failure(self) -> None:
        def ticker_factory(ticker: str):
            if ticker == "FAIL":
                raise RuntimeError("boom")
            return DummyTicker()

        with self._patch_db_engine():
            with self.assertRaises(RuntimeError):
                collect_us_financials(
                    cfg=make_config(retry_count=1, fail_fast=True),
                    universe_tag="NASDAQ100",
                    explicit_tickers=["FAIL", "AAPL"],
                    ticker_factory=ticker_factory,
                    row_writer=lambda s_rows, m_rows: (len(s_rows), len(m_rows)),
                    sleep_fn=lambda _sec: None,
                )


if __name__ == "__main__":
    unittest.main()
