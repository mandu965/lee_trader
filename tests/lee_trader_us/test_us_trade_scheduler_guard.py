from __future__ import annotations

from datetime import date
import os
import unittest
from unittest.mock import patch

from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.trade_orchestration.scheduler_guard import evaluate_scheduler_guard


class TradeSchedulerGuardTests(unittest.TestCase):
    @patch("python.us.trade_orchestration.scheduler_guard._latest_ranking_trade_date")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "0",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_scheduler_disabled_blocks_run(self, ranking_mock) -> None:
        ranking_mock.return_value = date(2026, 5, 14)
        cfg = load_trade_orchestration_config()
        result = evaluate_scheduler_guard(cfg, requested_trade_date=date(2026, 5, 14))
        self.assertFalse(result["can_run"])
        self.assertIn("SCHEDULER_DISABLED", result["errors"])

    @patch("python.us.trade_orchestration.scheduler_guard._latest_ranking_trade_date")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "LIVE",
            "US_TRADE_SCHEDULER_ALLOW_LIVE": "0",
        },
        clear=False,
    )
    def test_live_mode_is_blocked(self, ranking_mock) -> None:
        ranking_mock.return_value = date(2026, 5, 14)
        cfg = load_trade_orchestration_config()
        result = evaluate_scheduler_guard(cfg, requested_trade_date=date(2026, 5, 14))
        self.assertFalse(result["can_run"])
        self.assertIn("LIVE_DISABLED_IN_SCHEDULER", result["errors"])

    @patch("python.us.trade_orchestration.scheduler_guard._latest_ranking_trade_date")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "SHADOW",
            "US_BUY_SCHEDULER_ENABLED": "1",
        },
        clear=False,
    )
    def test_warns_when_buy_only_scheduler_enabled(self, ranking_mock) -> None:
        ranking_mock.return_value = date(2026, 5, 14)
        cfg = load_trade_orchestration_config()
        result = evaluate_scheduler_guard(cfg, requested_trade_date=date(2026, 5, 14))
        self.assertIn("BUY_ONLY_SCHEDULER_ENABLED", result["warnings"])


if __name__ == "__main__":
    unittest.main()
