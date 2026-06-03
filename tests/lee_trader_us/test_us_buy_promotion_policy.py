from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.buy_automation.promotion_policy import load_live_promotion_policy


class BuyPromotionPolicyTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_BUY_MIN_WIN_RATE_PCT": "50",
            "US_BUY_MAX_DRAWDOWN_PCT": "15",
            "US_BUY_REQUIRE_MANUAL_APPROVAL": "1",
        },
        clear=False,
    )
    def test_policy_converts_percent_values(self) -> None:
        policy = load_live_promotion_policy()
        self.assertEqual(policy.min_win_rate_pct, 0.5)
        self.assertEqual(policy.max_drawdown_pct, 0.15)
        self.assertTrue(policy.require_manual_approval)


if __name__ == "__main__":
    unittest.main()
