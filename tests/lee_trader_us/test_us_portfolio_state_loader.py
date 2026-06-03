from __future__ import annotations

from datetime import date
import os
import unittest
from unittest.mock import patch

from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.trade_orchestration.portfolio_state_loader import load_portfolio_state


class PortfolioStateLoaderTests(unittest.TestCase):
    @patch("python.us.trade_orchestration.portfolio_state_loader._load_optional_rows")
    @patch.dict(os.environ, {"US_TRADE_BLOCK_BUY_AFTER_FULL_EXIT_DAYS": "10"}, clear=False)
    def test_builds_cooldown_and_review_sets(self, load_rows_mock) -> None:
        load_rows_mock.side_effect = [
            [],
            [],
            [{"trade_date": date(2026, 5, 14), "symbol": "AAPL"}],
            [{"trade_date": date(2026, 5, 10), "symbol": "MSFT", "sell_action": "FULL_SELL", "exit_reason": "TAKE_PROFIT"}],
        ]
        cfg = load_trade_orchestration_config()
        state = load_portfolio_state(
            cfg,
            trade_date=date(2026, 5, 14),
            account_id="US_TEST",
            sell_report={
                "positions": [{"symbol": "AAPL", "paper_position_id": "POS1", "status": "OPEN"}],
                "decisions": [
                    {"symbol": "NVDA", "decision": "SELL", "exit_reason": "STOP_LOSS"},
                    {"symbol": "TSLA", "decision": "REVIEW_REQUIRED", "exit_reason": "PRICE_DATA_MISSING", "review_required": True},
                ],
            },
        )
        self.assertEqual(state["status"], "OK")
        self.assertIn("AAPL", state["open_position_map"])
        self.assertIn("NVDA", state["sell_signal_map"])
        self.assertIn("TSLA", state["review_required_symbols"])
        self.assertIn("MSFT", state["cooldown_symbols"])
        self.assertIn("AAPL", state["paper_buy_symbols_today"])


if __name__ == "__main__":
    unittest.main()
