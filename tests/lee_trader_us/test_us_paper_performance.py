from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from python.us.buy_automation.paper_performance import build_paper_performance


class BuyPaperPerformanceTests(unittest.TestCase):
    @patch("python.us.buy_automation.paper_performance.fetch_price_history_for_tickers")
    def test_price_missing_sets_status(self, history_mock) -> None:
        history_mock.return_value = []
        result = build_paper_performance(
            [{"paper_order_id": "P1", "trade_date": "2026-05-14", "symbol": "AAPL", "paper_order_qty": 1.0, "paper_order_price": 100.0, "paper_order_amount": 100.0}],
            as_of_date=date(2026, 5, 16),
        )
        self.assertEqual(result["rows"][0]["status"], "PRICE_DATA_MISSING")

    @patch("python.us.buy_automation.paper_performance.fetch_price_history_for_tickers")
    def test_pnl_and_benchmark_return_calculated(self, history_mock) -> None:
        history_mock.return_value = [
            {"ticker": "AAPL", "trade_date": date(2026, 5, 14), "close_price": 100.0, "adj_close_price": 100.0},
            {"ticker": "AAPL", "trade_date": date(2026, 5, 16), "close_price": 110.0, "adj_close_price": 110.0},
            {"ticker": "SPY", "trade_date": date(2026, 5, 14), "close_price": 500.0, "adj_close_price": 500.0},
            {"ticker": "SPY", "trade_date": date(2026, 5, 16), "close_price": 510.0, "adj_close_price": 510.0},
        ]
        result = build_paper_performance(
            [{"paper_order_id": "P1", "trade_date": "2026-05-14", "symbol": "AAPL", "paper_order_qty": 2.0, "paper_order_price": 100.0, "paper_order_amount": 200.0}],
            as_of_date=date(2026, 5, 16),
        )
        row = result["rows"][0]
        self.assertEqual(row["status"], "OK")
        self.assertEqual(row["current_value"], 220.0)
        self.assertEqual(row["unrealized_pnl"], 20.0)


if __name__ == "__main__":
    unittest.main()
