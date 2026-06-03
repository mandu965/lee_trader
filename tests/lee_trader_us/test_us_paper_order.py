from __future__ import annotations

import unittest

from python.us.buy_automation.paper_order import build_paper_order


class BuyAutomationPaperOrderTests(unittest.TestCase):
    def test_builds_internal_paper_order(self) -> None:
        order = build_paper_order(
            trade_date="2026-05-14",
            symbol="AAPL",
            allocated_amount_usd=100.0,
            reference_price=50.0,
            mode="PAPER",
        )
        self.assertEqual(order["symbol"], "AAPL")
        self.assertEqual(order["side"], "BUY")
        self.assertEqual(order["paper_order_qty"], 2)
        self.assertEqual(order["paper_order_amount"], 100.0)
        self.assertEqual(order["assumed_fill_status"], "ASSUMED_FILLED")

    def test_rounds_down_to_whole_shares(self) -> None:
        order = build_paper_order(
            trade_date="2026-05-14",
            symbol="NVDA",
            allocated_amount_usd=5000.0,
            reference_price=225.320007,
            mode="PAPER",
        )
        self.assertEqual(order["paper_order_qty"], 22)
        self.assertAlmostEqual(order["paper_order_amount"], 4957.040154, places=6)


if __name__ == "__main__":
    unittest.main()
