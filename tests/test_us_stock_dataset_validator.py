from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from python.us.validate_us_stock_ml_dataset import USDatasetValidationConfig, validate_us_stock_ml_dataset


def make_config(report_path: Path, **overrides) -> USDatasetValidationConfig:
    values = {
        "enabled": True,
        "feature_table": "feature.us_stock_feature_daily",
        "financial_feature_table": "feature.us_stock_financial_feature",
        "relative_strength_table": "feature.us_stock_relative_strength_daily",
        "label_table": "label.us_stock_label_daily",
        "report_path": report_path,
        "log_level": "INFO",
        "universe": "NASDAQ100",
    }
    values.update(overrides)
    return USDatasetValidationConfig(**values)


class USDatasetValidatorTests(unittest.TestCase):
    def test_join_counts_distribution_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            result = validate_us_stock_ml_dataset(
                cfg=make_config(report_path),
                universe_tag="NASDAQ100",
                ticker_loader=lambda universe: ["AAPL", "MSFT"],
                feature_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "trade_date": "2025-01-02", "ret_5d": 0.1, "ret_20d": 0.2, "ret_60d": 0.3},
                    {"ticker": "MSFT", "trade_date": "2025-01-02", "ret_5d": 0.05, "ret_20d": 0.1, "ret_60d": 0.15},
                ],
                rs_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price", "rs_spy_20d": 0.05, "rs_spy_60d": 0.07, "rs_qqq_20d": 0.04, "rs_qqq_60d": 0.06},
                    {"ticker": "MSFT", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price", "rs_spy_20d": 0.01, "rs_spy_60d": 0.02, "rs_qqq_20d": 0.01, "rs_qqq_60d": 0.02},
                ],
                label_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price", "future_ret_20d": 0.2, "future_ret_60d": 0.3, "label_positive_20d": 1, "label_positive_60d": 1, "label_top20_20d": 1, "label_top20_60d": 1},
                    {"ticker": "MSFT", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price", "future_ret_20d": 0.1, "future_ret_60d": 0.2, "label_positive_20d": 1, "label_positive_60d": 1, "label_top20_20d": 0, "label_top20_60d": 0},
                ],
                financial_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "fiscal_date": "2025-12-31"},
                ],
            )
            self.assertEqual(result.joined_row_count, 2)
            self.assertEqual(result.duplicate_key_count, 0)
            self.assertTrue(report_path.exists())

    def test_duplicate_detection_and_leakage_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            result = validate_us_stock_ml_dataset(
                cfg=make_config(report_path),
                universe_tag="NASDAQ100",
                ticker_loader=lambda universe: ["AAPL"],
                feature_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "trade_date": "2025-01-02"},
                    {"ticker": "AAPL", "trade_date": "2025-01-02"},
                ],
                rs_fetcher=lambda tickers: [],
                label_fetcher=lambda tickers: [
                    {"ticker": "AAPL", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price"},
                    {"ticker": "AAPL", "trade_date": "2025-01-02", "source": "market.us_stock_daily_price"},
                ],
                financial_fetcher=lambda tickers: [],
            )
            self.assertGreater(result.duplicate_key_count, 0)
            self.assertTrue(any("reported_date" in note or "fiscal_date" in note for note in result.leakage_risk_notes))

    def test_enabled_false_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            result = validate_us_stock_ml_dataset(cfg=make_config(report_path, enabled=False), universe_tag="NASDAQ100")
            self.assertEqual(result.joined_row_count, 0)


if __name__ == "__main__":
    unittest.main()
