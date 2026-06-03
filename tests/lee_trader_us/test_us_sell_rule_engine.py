from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.sell_automation.config import load_sell_automation_config
from python.us.sell_automation.sell_rule_engine import evaluate_sell_rules


class SellRuleEngineTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=False)
    def test_price_missing_returns_review_required_rule(self) -> None:
        cfg = load_sell_automation_config()
        rules = evaluate_sell_rules(
            {
                "symbol": "AAPL",
                "avg_entry_price": 100.0,
                "remaining_quantity": 1.0,
                "latest_price": None,
                "highest_price_since_entry": 110.0,
                "unrealized_pnl_pct": None,
                "holding_days": 5,
                "latest_rank": 10,
                "latest_score": 80.0,
                "latest_probability": 0.7,
                "benchmark_return_pct": 0.02,
                "symbol_return_pct": 0.01,
                "data_quality_flags": ["PRICE_DATA_MISSING"],
            },
            cfg,
            market_context={"benchmark_drawdown_pct": -0.01},
        )
        self.assertEqual(rules[0]["rule"], "DATA_QUALITY_CHECK")
        self.assertEqual(rules[0]["action"], "REVIEW_REQUIRED")
        self.assertEqual(rules[0]["result"], "FAIL")

    @patch.dict(os.environ, {}, clear=False)
    def test_stop_loss_triggers_full_sell(self) -> None:
        cfg = load_sell_automation_config()
        rules = evaluate_sell_rules(
            {
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
            },
            cfg,
            market_context={"benchmark_drawdown_pct": -0.01},
        )
        stop_loss_rule = next(rule for rule in rules if rule["rule"] == "STOP_LOSS")
        self.assertEqual(stop_loss_rule["result"], "FAIL")
        self.assertEqual(stop_loss_rule["action"], "FULL_SELL")


if __name__ == "__main__":
    unittest.main()
