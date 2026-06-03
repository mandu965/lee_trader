from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.buy_automation.config import load_buy_automation_config


class BuyAutomationConfigTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_BUY_AUTOMATION_MODE": "INVALID",
            "US_BUY_MIN_SCORE": "0.70",
            "US_BUY_MAX_GAP_UP_PCT": "5",
        },
        clear=False,
    )
    def test_config_uses_safe_fallbacks(self) -> None:
        cfg = load_buy_automation_config()
        self.assertEqual(cfg.mode, "SHADOW")
        self.assertEqual(cfg.min_score, 70.0)
        self.assertEqual(cfg.max_gap_up_pct, 0.05)
        self.assertTrue(any("fallback=SHADOW" in warning for warning in cfg.warnings))

    @patch.dict(
        os.environ,
        {
            "US_BUY_AUTOMATION_ENABLED": "0",
            "US_BUY_MIN_PROB": "0.60",
        },
        clear=False,
    )
    def test_disabled_flag_and_probability_loaded(self) -> None:
        cfg = load_buy_automation_config()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.min_prob, 0.60)


if __name__ == "__main__":
    unittest.main()
