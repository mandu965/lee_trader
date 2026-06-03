from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from python.us.build_us_stock_labels import USLabelConfig, add_label_columns, build_us_stock_labels, prepare_label_frame


def make_config(**overrides) -> USLabelConfig:
    values = {
        "enabled": True,
        "source_price_table": "market.us_stock_daily_price",
        "target_table": "label.us_stock_label_daily",
        "price_column": "auto",
        "windows": (5, 20, 60),
        "top_percentile": 0.20,
        "min_universe_size": 2,
        "exclude_benchmarks": ("SPY", "QQQ"),
        "write_mode": "upsert",
        "log_level": "INFO",
        "universe": "NASDAQ100",
    }
    values.update(overrides)
    return USLabelConfig(**values)


def make_price_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = date(2025, 1, 2)
    slopes = {"AAPL": 2.0, "MSFT": 1.0, "SPY": 0.8, "QQQ": 0.9}
    bases = {"AAPL": 100.0, "MSFT": 120.0, "SPY": 200.0, "QQQ": 300.0}
    for offset in range(80):
        trade_date = start + timedelta(days=offset + (offset // 5) * 2)
        for ticker, slope in slopes.items():
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "close_price": None,
                    "adj_close_price": bases[ticker] + slope * offset,
                }
            )
    return rows


class USStockLabelBuilderTests(unittest.TestCase):
    def test_future_ret_5d_and_20d(self) -> None:
        prepared, _ = prepare_label_frame(make_price_rows(), windows=(5, 20))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[0]
        self.assertAlmostEqual(row["future_ret_5d"], (110.0 / 100.0) - 1.0)
        self.assertAlmostEqual(row["future_ret_20d"], (140.0 / 100.0) - 1.0)

    def test_trading_day_shift_not_calendar_day(self) -> None:
        prepared, _ = prepare_label_frame(make_price_rows(), windows=(5,))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[0]
        self.assertIsNotNone(row["future_ret_5d"])

    def test_close_today_zero_returns_null(self) -> None:
        rows = make_price_rows()
        rows[0]["adj_close_price"] = 0.0
        prepared, _ = prepare_label_frame(rows, windows=(5,))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[0]
        self.assertTrue(pd.isna(row["future_ret_5d"]))

    def test_future_price_missing_returns_null(self) -> None:
        rows = make_price_rows()
        for item in rows:
            if item["ticker"] == "AAPL" and item["trade_date"] == sorted({r["trade_date"] for r in rows})[5]:
                item["adj_close_price"] = None
                break
        prepared, _ = prepare_label_frame(rows, windows=(5,))
        row = prepared[(prepared["ticker"] == "AAPL")].iloc[0]
        self.assertTrue(pd.isna(row["future_ret_5d"]))

    def test_positive_and_top20_labels(self) -> None:
        prepared, _ = prepare_label_frame(make_price_rows(), windows=(20, 60))
        labeled = add_label_columns(
            prepared,
            windows=(20, 60),
            top_percentile=0.20,
            min_universe_size=2,
            exclude_benchmarks=("SPY", "QQQ"),
        )
        aapl = labeled[labeled["ticker"] == "AAPL"].iloc[0]
        msft = labeled[labeled["ticker"] == "MSFT"].iloc[0]
        self.assertEqual(aapl["label_positive_20d"], 1)
        self.assertEqual(msft["label_positive_20d"], 1)
        self.assertEqual(aapl["label_top20_20d"], 1)
        self.assertEqual(msft["label_top20_20d"], 0)

    def test_universe_too_small_makes_top20_null(self) -> None:
        prepared, _ = prepare_label_frame(make_price_rows(), windows=(20,))
        labeled = add_label_columns(
            prepared,
            windows=(20,),
            top_percentile=0.20,
            min_universe_size=10,
            exclude_benchmarks=("SPY", "QQQ"),
        )
        row = labeled[labeled["ticker"] == "AAPL"].iloc[0]
        self.assertTrue(pd.isna(row["label_top20_20d"]))

    def test_benchmarks_excluded_from_top20(self) -> None:
        prepared, _ = prepare_label_frame(make_price_rows(), windows=(20,))
        labeled = add_label_columns(
            prepared,
            windows=(20,),
            top_percentile=0.20,
            min_universe_size=2,
            exclude_benchmarks=("SPY", "QQQ"),
        )
        spy = labeled[labeled["ticker"] == "SPY"].iloc[0]
        self.assertTrue(pd.isna(spy["label_top20_20d"]))

    def test_enabled_false_skips(self) -> None:
        result = build_us_stock_labels(cfg=make_config(enabled=False), universe_tag="NASDAQ100")
        self.assertEqual(result.label_rows, 0)


if __name__ == "__main__":
    unittest.main()
