from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from python.us.build_us_financial_features import (
    USFinancialFeatureConfig,
    _safe_ratio,
    build_financial_feature_frame,
    build_us_financial_features,
)


def make_config(**overrides) -> USFinancialFeatureConfig:
    values = {
        "enabled": True,
        "source_statement_table": "raw.us_stock_financial_statement",
        "source_metric_table": "raw.us_stock_financial_metric",
        "target_table": "feature.us_stock_financial_feature",
        "period_types": ("annual", "quarterly"),
        "lookback_years": 5,
        "write_mode": "upsert",
        "log_level": "INFO",
        "universe": "NASDAQ100",
    }
    values.update(overrides)
    return USFinancialFeatureConfig(**values)


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "market": "US",
                "period_type": "annual",
                "fiscal_date": date(2024, 12, 31),
                "source": "yfinance",
                "revenue": 100.0,
                "gross_profit": 40.0,
                "operating_income": 20.0,
                "net_income": 15.0,
                "ebitda": 25.0,
                "eps": 2.0,
                "total_assets": 300.0,
                "total_liabilities": 120.0,
                "total_equity": 180.0,
                "operating_cash_flow": 30.0,
                "free_cash_flow": 18.0,
                "shares_outstanding": 1000.0,
                "market_cap": 10000.0,
                "roe": 0.12,
                "roa": 0.05,
                "debt_to_equity": None,
                "current_ratio": 1.5,
                "per": 20.0,
                "pbr": 4.0,
                "psr": 5.0,
                "ev_ebitda": 12.0,
                "dividend_yield": 0.01,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "AAPL",
                "market": "US",
                "period_type": "annual",
                "fiscal_date": date(2025, 12, 31),
                "source": "yfinance",
                "revenue": 120.0,
                "gross_profit": 50.0,
                "operating_income": 22.0,
                "net_income": 18.0,
                "ebitda": 28.0,
                "eps": 2.4,
                "total_assets": 320.0,
                "total_liabilities": 128.0,
                "total_equity": 192.0,
                "operating_cash_flow": 36.0,
                "free_cash_flow": 24.0,
                "shares_outstanding": 1000.0,
                "market_cap": 11000.0,
                "roe": 0.13,
                "roa": 0.055,
                "debt_to_equity": None,
                "current_ratio": 1.6,
                "per": 18.0,
                "pbr": 3.8,
                "psr": 4.8,
                "ev_ebitda": 11.0,
                "dividend_yield": 0.011,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_type": "quarterly",
                "fiscal_date": date(2025, 3, 31),
                "source": "yfinance",
                "revenue": 50.0,
                "gross_profit": 20.0,
                "operating_income": 10.0,
                "net_income": 8.0,
                "ebitda": 12.0,
                "eps": 1.0,
                "total_assets": 150.0,
                "total_liabilities": 60.0,
                "total_equity": 90.0,
                "operating_cash_flow": 12.0,
                "free_cash_flow": 9.0,
                "shares_outstanding": 2000.0,
                "market_cap": 20000.0,
                "roe": 0.10,
                "roa": 0.04,
                "debt_to_equity": 0.6,
                "current_ratio": 1.4,
                "per": 25.0,
                "pbr": 5.0,
                "psr": 6.0,
                "ev_ebitda": 15.0,
                "dividend_yield": 0.008,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_type": "quarterly",
                "fiscal_date": date(2025, 6, 30),
                "source": "yfinance",
                "revenue": 55.0,
                "gross_profit": 21.0,
                "operating_income": 11.0,
                "net_income": 8.5,
                "ebitda": 13.0,
                "eps": 1.1,
                "total_assets": 152.0,
                "total_liabilities": 62.0,
                "total_equity": 90.0,
                "operating_cash_flow": 12.5,
                "free_cash_flow": 9.3,
                "shares_outstanding": 2000.0,
                "market_cap": 20200.0,
                "roe": 0.11,
                "roa": 0.041,
                "debt_to_equity": 0.62,
                "current_ratio": 1.45,
                "per": 24.0,
                "pbr": 4.9,
                "psr": 5.8,
                "ev_ebitda": 14.5,
                "dividend_yield": 0.008,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_type": "quarterly",
                "fiscal_date": date(2025, 9, 30),
                "source": "yfinance",
                "revenue": 58.0,
                "gross_profit": 23.0,
                "operating_income": 11.4,
                "net_income": 9.0,
                "ebitda": 13.5,
                "eps": 1.2,
                "total_assets": 154.0,
                "total_liabilities": 63.0,
                "total_equity": 91.0,
                "operating_cash_flow": 12.8,
                "free_cash_flow": 9.5,
                "shares_outstanding": 2000.0,
                "market_cap": 20500.0,
                "roe": 0.12,
                "roa": 0.042,
                "debt_to_equity": 0.63,
                "current_ratio": 1.46,
                "per": 23.0,
                "pbr": 4.8,
                "psr": 5.7,
                "ev_ebitda": 14.0,
                "dividend_yield": 0.008,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_type": "quarterly",
                "fiscal_date": date(2025, 12, 31),
                "source": "yfinance",
                "revenue": 60.0,
                "gross_profit": 24.0,
                "operating_income": 12.0,
                "net_income": 9.5,
                "ebitda": 14.0,
                "eps": 1.25,
                "total_assets": 156.0,
                "total_liabilities": 64.0,
                "total_equity": 92.0,
                "operating_cash_flow": 13.0,
                "free_cash_flow": 10.0,
                "shares_outstanding": 2000.0,
                "market_cap": 20800.0,
                "roe": 0.13,
                "roa": 0.043,
                "debt_to_equity": 0.64,
                "current_ratio": 1.5,
                "per": 22.0,
                "pbr": 4.7,
                "psr": 5.5,
                "ev_ebitda": 13.8,
                "dividend_yield": 0.008,
                "collected_at": "2026-05-12T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_type": "quarterly",
                "fiscal_date": date(2026, 3, 31),
                "source": "yfinance",
                "revenue": 65.0,
                "gross_profit": 26.0,
                "operating_income": 13.0,
                "net_income": 10.5,
                "ebitda": 15.0,
                "eps": 1.35,
                "total_assets": 158.0,
                "total_liabilities": 65.0,
                "total_equity": 93.0,
                "operating_cash_flow": 14.0,
                "free_cash_flow": 11.0,
                "shares_outstanding": 2000.0,
                "market_cap": 21000.0,
                "roe": 0.14,
                "roa": 0.044,
                "debt_to_equity": 0.65,
                "current_ratio": 1.55,
                "per": 21.0,
                "pbr": 4.6,
                "psr": 5.2,
                "ev_ebitda": 13.0,
                "dividend_yield": 0.008,
                "collected_at": "2026-05-12T00:00:00Z",
            },
        ]
    )


class USFinancialFeatureBuilderTests(unittest.TestCase):
    def test_revenue_growth_yoy_for_annual(self) -> None:
        built = build_financial_feature_frame(sample_raw_frame())
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertAlmostEqual(row["revenue_growth_yoy"], 0.2)

    def test_revenue_growth_qoq_for_quarterly(self) -> None:
        built = build_financial_feature_frame(sample_raw_frame())
        row = built[(built["ticker"] == "MSFT") & (built["fiscal_date"] == date(2025, 6, 30))].iloc[0]
        self.assertAlmostEqual(row["revenue_growth_qoq"], 0.1)

    def test_denominator_zero_returns_null(self) -> None:
        self.assertIsNone(_safe_ratio(10, 0))

    def test_nan_inf_values_become_null(self) -> None:
        self.assertIsNone(_safe_ratio(float("inf"), 10))
        self.assertIsNone(_safe_ratio(10, float("nan")))

    def test_margin_calculation(self) -> None:
        built = build_financial_feature_frame(sample_raw_frame())
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertAlmostEqual(row["gross_margin"], 50.0 / 120.0)
        self.assertAlmostEqual(row["net_margin"], 18.0 / 120.0)

    def test_debt_ratio_calculation(self) -> None:
        built = build_financial_feature_frame(sample_raw_frame())
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertAlmostEqual(row["debt_ratio"], 128.0 / 320.0)

    def test_score_skeleton_calculation(self) -> None:
        built = build_financial_feature_frame(sample_raw_frame())
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertIsNotNone(row["financial_quality_score"])
        self.assertIsNotNone(row["financial_growth_score"])
        self.assertIsNotNone(row["financial_value_score"])

    def test_enabled_false_skips_build(self) -> None:
        result = build_us_financial_features(
            cfg=make_config(enabled=False),
            universe_tag="NASDAQ100",
        )
        self.assertEqual(result.processed_ticker_count, 0)
        self.assertEqual(result.built_row_count, 0)

    def test_missing_raw_fields_do_not_fail(self) -> None:
        frame = sample_raw_frame().copy()
        frame["gross_profit"] = pd.NA
        built = build_financial_feature_frame(frame)
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertIsNone(row["gross_margin"])

    def test_duplicate_keys_are_overwritten_by_latest_value(self) -> None:
        frame = sample_raw_frame().copy()
        duplicate = frame.iloc[[1]].copy()
        duplicate["revenue"] = 130.0
        duplicate["collected_at"] = "2026-05-12T01:00:00Z"
        merged = pd.concat([frame, duplicate], ignore_index=True)
        built = build_financial_feature_frame(
            merged.sort_values(["ticker", "period_type", "fiscal_date", "collected_at"]).drop_duplicates(
                subset=["ticker", "period_type", "fiscal_date", "source"], keep="last"
            )
        )
        row = built[(built["ticker"] == "AAPL") & (built["fiscal_date"] == date(2025, 12, 31))].iloc[0]
        self.assertAlmostEqual(row["revenue"], 130.0)


if __name__ == "__main__":
    unittest.main()
