from __future__ import annotations

import unittest

from python.us.buy_automation.notification_formatter import (
    format_notification_detail,
    format_notification_summary,
)


class BuyNotificationFormatterTests(unittest.TestCase):
    def test_summary_and_detail_are_strings_only(self) -> None:
        report = {
            "trade_date": "2026-05-14",
            "mode": "SHADOW",
            "loaded_candidates": 5,
            "allowed_candidates": 0,
            "blocked_candidates": 5,
            "paper_orders": [],
            "validation_summary": {
                "block_counts": {"DATA_MISSING": 3},
                "fail_safe_block_count": 3,
                "data_missing_symbols": ["AAPL"],
            },
            "candidates": [{"symbol": "AAPL", "allowed": False, "block_reasons": ["DATA_MISSING"]}],
            "paper_performance": {"rows": []},
        }
        summary = format_notification_summary(report)
        detail = format_notification_detail(report)
        self.assertIn("[US BUY Automation Summary]", summary)
        self.assertIn("top_block_reason: DATA_MISSING", summary)
        self.assertIn("[Blocked Candidates]", detail)


if __name__ == "__main__":
    unittest.main()
