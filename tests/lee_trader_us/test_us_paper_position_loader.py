from __future__ import annotations

from datetime import date
import unittest

from python.us.sell_automation.paper_position_loader import build_positions_from_orders


class SellPaperPositionLoaderTests(unittest.TestCase):
    def test_price_missing_marks_position_reviewable(self) -> None:
        positions = build_positions_from_orders(
            account_id="US_TEST",
            trade_date=date(2026, 5, 14),
            buy_orders=[
                {
                    "paper_order_id": "BUY1",
                    "symbol": "AAPL",
                    "trade_date": date(2026, 5, 10),
                    "order_qty": 2.0,
                    "order_price": 50.0,
                    "status": "FILLED",
                }
            ],
            sell_orders=[],
            fill_rows=[],
            existing_positions=[],
            latest_rank_map={"AAPL": {"symbol": "AAPL", "rank_no": 10, "total_score": 75.0, "source": "rule_v1"}},
            price_history_rows=[
                {"ticker": "SPY", "trade_date": date(2026, 5, 10), "close_price": 100.0},
                {"ticker": "SPY", "trade_date": date(2026, 5, 14), "close_price": 102.0},
            ],
            benchmark_symbol="SPY",
        )
        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertEqual(position["symbol"], "AAPL")
        self.assertEqual(position["remaining_quantity"], 2.0)
        self.assertEqual(position["status"], "PRICE_DATA_MISSING")
        self.assertIn("PRICE_DATA_MISSING", position["data_quality_flags"])


if __name__ == "__main__":
    unittest.main()
