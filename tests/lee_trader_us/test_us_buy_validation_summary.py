from __future__ import annotations

import unittest

from python.us.buy_automation.validation_summary import summarize_validation


class BuyValidationSummaryTests(unittest.TestCase):
    def test_invalid_decision_log_is_flagged(self) -> None:
        summary = summarize_validation(
            [
                {
                    "symbol": "AAPL",
                    "allowed": False,
                    "block_reasons": [],
                    "applied_rules": [{"rule": "PRICE_RANGE", "result": "PASS"}],
                }
            ]
        )
        self.assertIn("INVALID_DECISION_LOG", summary["block_counts"])
        self.assertIn("AAPL", summary["invalid_decision_logs"])

    def test_block_reason_counts_and_rule_summary(self) -> None:
        summary = summarize_validation(
            [
                {
                    "symbol": "AAPL",
                    "allowed": False,
                    "block_reasons": ["DATA_MISSING", "AUTOMATION_DISABLED"],
                    "applied_rules": [{"rule": "PRICE_RANGE", "result": "PASS"}, {"rule": "FINANCIAL_DATA", "result": "FAIL"}],
                },
                {
                    "symbol": "MSFT",
                    "allowed": True,
                    "block_reasons": [],
                    "applied_rules": [{"rule": "PRICE_RANGE", "result": "PASS"}],
                },
            ]
        )
        self.assertEqual(summary["block_counts"]["DATA_MISSING"], 1)
        self.assertEqual(summary["automation_disabled_count"], 1)
        self.assertEqual(summary["rule_summary"]["PRICE_RANGE"]["PASS"], 2)


if __name__ == "__main__":
    unittest.main()
