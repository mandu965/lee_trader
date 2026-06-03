from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.us_live_risk_policy import collect_us_live_risk_policy_issues, load_us_live_risk_policy


class USLiveRiskPolicyTests(unittest.TestCase):
    def test_load_policy_uses_safe_defaults(self) -> None:
        policy = load_us_live_risk_policy("US_LIVE_RULE_V1")
        safety = policy["safety"]
        self.assertFalse(safety["live_trading_enabled"])
        self.assertFalse(safety["live_order_enabled"])
        self.assertTrue(safety["require_manual_approval"])
        self.assertTrue(safety["real_order_blocked"])

    def test_validation_flags_unsafe_policy(self) -> None:
        policy = load_us_live_risk_policy("US_LIVE_RULE_V1")
        policy["safety"]["live_order_enabled"] = True
        policy["order"]["allow_market_order"] = True
        issues = collect_us_live_risk_policy_issues(policy)
        codes = {item.code for item in issues}
        self.assertIn("live_order_enabled_not_false", codes)
        self.assertIn("market_order_enabled", codes)


if __name__ == "__main__":
    unittest.main()
