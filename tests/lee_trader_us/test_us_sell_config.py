from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.sell_automation.config import load_sell_automation_config


class SellAutomationConfigTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_SELL_AUTOMATION_MODE": "INVALID",
            "US_SELL_STOP_LOSS_PCT": "-8",
            "US_SELL_TAKE_PROFIT_PCT": "15",
            "US_SELL_PARTIAL_TAKE_PROFIT_RATIO": "50",
        },
        clear=False,
    )
    def test_config_uses_safe_fallbacks(self) -> None:
        cfg = load_sell_automation_config()
        self.assertEqual(cfg.mode, "SHADOW")
        self.assertEqual(cfg.stop_loss_pct, -0.08)
        self.assertEqual(cfg.take_profit_pct, 0.15)
        self.assertEqual(cfg.partial_take_profit_ratio, 0.5)
        self.assertTrue(any("fallback=SHADOW" in warning for warning in cfg.warnings))

    @patch.dict(
        os.environ,
        {
            "US_SELL_AUTOMATION_ENABLED": "0",
            "US_SELL_REQUIRE_BENCHMARK_STRENGTH": "1",
        },
        clear=False,
    )
    def test_disabled_flag_and_benchmark_requirement_loaded(self) -> None:
        cfg = load_sell_automation_config()
        self.assertFalse(cfg.enabled)
        self.assertTrue(cfg.require_benchmark_strength)


if __name__ == "__main__":
    unittest.main()
