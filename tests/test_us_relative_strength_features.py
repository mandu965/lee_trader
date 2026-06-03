from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from python.us.build_us_relative_strength_features import (
    USRelativeStrengthConfig,
    build_us_relative_strength_features,
    prepare_relative_strength_frame,
)


def make_config(**overrides) -> USRelativeStrengthConfig:
    values = {
        "enabled": True,
        "source_table": "market.us_stock_daily_price",
        "target_table": "feature.us_stock_relative_strength_daily",
        "benchmarks": ("SPY", "QQQ"),
        "windows": (5, 20, 60, 120, 252),
        "price_column": "auto",
        "write_mode": "upsert",
        "log_level": "INFO",
        "universe": "NASDAQ100",
    }
    values.update(overrides)
    return USRelativeStrengthConfig(**values)


def make_price_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = date(2025, 1, 2)
    tickers = {
        "AAPL": 100.0,
        "SPY": 200.0,
        "QQQ": 300.0,
    }
    for offset in range(260):
        trade_date = start + timedelta(days=offset + (offset // 5) * 2)
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": "AAPL",
                "open_price": None,
                "high_price": None,
                "low_price": None,
                "close_price": None,
                "adj_close_price": tickers["AAPL"] + offset * 1.5,
                "volume": None,
            }
        )
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": "SPY",
                "open_price": None,
                "high_price": None,
                "low_price": None,
                "close_price": None,
                "adj_close_price": tickers["SPY"] + offset * 1.0,
                "volume": None,
            }
        )
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": "QQQ",
                "open_price": None,
                "high_price": None,
                "low_price": None,
                "close_price": None,
                "adj_close_price": tickers["QQQ"] + offset * 0.5,
                "volume": None,
            }
        )
    return rows


class RelativeStrengthBuilderTests(unittest.TestCase):
    def test_ret_5d_and_ret_20d_are_computed_by_trading_day_shift(self) -> None:
        prepared, _ = prepare_relative_strength_frame(make_price_rows(), windows=(5, 20))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[20]
        expected_5d = ((100.0 + 20 * 1.5) / (100.0 + 15 * 1.5)) - 1.0
        expected_20d = ((100.0 + 20 * 1.5) / 100.0) - 1.0
        self.assertAlmostEqual(row["ret_5d"], expected_5d)
        self.assertAlmostEqual(row["ret_20d"], expected_20d)

    def test_calendar_day_gap_is_not_used(self) -> None:
        prepared, _ = prepare_relative_strength_frame(make_price_rows(), windows=(5,))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[6]
        self.assertIsNotNone(row["ret_5d"])

    def test_spy_and_qqq_benchmark_returns_and_relative_strength(self) -> None:
        written: list[dict[str, object]] = []
        result = build_us_relative_strength_features(
            cfg=make_config(windows=(5,)),
            universe_tag="NASDAQ100",
            explicit_tickers=["AAPL"],
            price_fetcher=lambda tickers: make_price_rows(),
            row_writer=lambda rows: written.extend(rows) or len(rows),
        )
        self.assertGreater(result.built_rows, 0)
        row = next(item for item in written if item["trade_date"] == written[-1]["trade_date"])
        self.assertAlmostEqual(row["rs_spy_5d"], row["ret_5d"] - row["spy_ret_5d"])
        self.assertAlmostEqual(row["rs_qqq_5d"], row["ret_5d"] - row["qqq_ret_5d"])

    def test_benchmark_missing_results_in_null_relative_strength(self) -> None:
        rows = [row for row in make_price_rows() if row["ticker"] != "QQQ"]
        written: list[dict[str, object]] = []
        build_us_relative_strength_features(
            cfg=make_config(windows=(5,)),
            universe_tag="NASDAQ100",
            explicit_tickers=["AAPL"],
            price_fetcher=lambda tickers: rows,
            row_writer=lambda items: written.extend(items) or len(items),
        )
        last = written[-1]
        self.assertIsNone(last["qqq_ret_5d"])
        self.assertIsNone(last["rs_qqq_5d"])

    def test_zero_or_missing_past_price_returns_null(self) -> None:
        rows = make_price_rows()
        rows[0]["adj_close_price"] = 0.0
        prepared, _ = prepare_relative_strength_frame(rows, windows=(5,))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[5]
        self.assertTrue(pd.isna(row["ret_5d"]))

    def test_enabled_false_skips_execution(self) -> None:
        result = build_us_relative_strength_features(
            cfg=make_config(enabled=False),
            universe_tag="NASDAQ100",
        )
        self.assertEqual(result.built_rows, 0)
        self.assertEqual(result.source_price_rows, 0)


if __name__ == "__main__":
    unittest.main()
