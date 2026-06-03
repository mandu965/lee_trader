from __future__ import annotations

import unittest

from python.us.sell_automation.paper_sell_order import build_paper_sell_order


class SellAutomationPaperSellOrderTests(unittest.TestCase):
    def test_builds_internal_paper_sell_order(self) -> None:
        order = build_paper_sell_order(
            trade_date="2026-05-14",
            decision={
                "sell_decision_id": "DEC1",
                "requested_sell_action": "FULL_SELL",
                "sell_action": "FULL_SELL",
                "sell_quantity": 2.0,
                "exit_reason": "STOP_LOSS",
            },
            position={
                "paper_position_id": "POS1",
                "symbol": "AAPL",
                "avg_entry_price": 50.0,
                "latest_price": 40.0,
            },
            mode="PAPER",
        )
        self.assertEqual(order["symbol"], "AAPL")
        self.assertEqual(order["side"], "SELL")
        self.assertEqual(order["sell_quantity"], 2)
        self.assertEqual(order["assumed_fill_status"], "ASSUMED_FILLED")


if __name__ == "__main__":
    unittest.main()
