from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from python.us.sell_automation.sell_decision_engine import run_sell_automation


class SellDecisionEngineTests(unittest.TestCase):
    @patch("python.us.sell_automation.sell_decision_engine.fetch_latest_daily_feature_snapshots")
    @patch("python.us.sell_automation.sell_decision_engine.load_paper_positions")
    @patch.dict(
        os.environ,
        {
            "US_SELL_AUTOMATION_MODE": "LIVE",
            "US_SELL_AUTOMATION_ENABLED": "1",
        },
        clear=False,
    )
    def test_live_mode_never_creates_paper_sell_order(self, load_positions_mock, feature_mock) -> None:
        feature_mock.return_value = {"SPY": {"ret_20d": -0.01, "trade_date": date(2026, 5, 14)}}
        load_positions_mock.return_value = {
            "trade_date": date(2026, 5, 14),
            "events": [],
            "positions": [
                {
                    "paper_position_id": "POS1",
                    "symbol": "AAPL",
                    "avg_entry_price": 100.0,
                    "remaining_quantity": 1.0,
                    "latest_price": 90.0,
                    "highest_price_since_entry": 105.0,
                    "unrealized_pnl_pct": -0.10,
                    "holding_days": 5,
                    "latest_rank": 10,
                    "latest_score": 80.0,
                    "latest_probability": 0.7,
                    "benchmark_return_pct": -0.01,
                    "symbol_return_pct": -0.10,
                    "data_quality_flags": [],
                }
            ],
        }
        report = run_sell_automation(account_id="US_TEST", persist_logs=False)
        self.assertEqual(report["mode"], "LIVE")
        self.assertEqual(len(report["paper_sell_orders"]), 0)
        self.assertEqual(report["decisions"][0]["sell_action"], "LIVE_NOT_IMPLEMENTED")
        self.assertTrue(report["decisions"][0]["review_required"])

    @patch("python.us.sell_automation.sell_decision_engine.fetch_latest_daily_feature_snapshots")
    @patch("python.us.sell_automation.sell_decision_engine.load_paper_positions")
    @patch.dict(
        os.environ,
        {
            "US_SELL_AUTOMATION_MODE": "PAPER",
            "US_SELL_AUTOMATION_ENABLED": "0",
        },
        clear=False,
    )
    def test_disabled_mode_keeps_sell_action_disabled(self, load_positions_mock, feature_mock) -> None:
        feature_mock.return_value = {"SPY": {"ret_20d": -0.01, "trade_date": date(2026, 5, 14)}}
        load_positions_mock.return_value = {
            "trade_date": date(2026, 5, 14),
            "events": [],
            "positions": [
                {
                    "paper_position_id": "POS1",
                    "symbol": "AAPL",
                    "avg_entry_price": 100.0,
                    "remaining_quantity": 2.0,
                    "latest_price": 120.0,
                    "highest_price_since_entry": 121.0,
                    "unrealized_pnl_pct": 0.20,
                    "holding_days": 5,
                    "latest_rank": 10,
                    "latest_score": 80.0,
                    "latest_probability": 0.7,
                    "benchmark_return_pct": 0.01,
                    "symbol_return_pct": 0.20,
                    "data_quality_flags": [],
                }
            ],
        }
        report = run_sell_automation(account_id="US_TEST", persist_logs=False)
        self.assertEqual(report["mode"], "PAPER")
        self.assertEqual(report["decisions"][0]["decision"], "SELL")
        self.assertEqual(report["decisions"][0]["sell_action"], "DISABLED")
        self.assertEqual(len(report["paper_sell_orders"]), 0)
        self.assertTrue(report["decisions"][0]["applied_rules"])


if __name__ == "__main__":
    unittest.main()
