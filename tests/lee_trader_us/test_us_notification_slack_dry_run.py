from __future__ import annotations

from dataclasses import replace
import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.slack_dry_run_adapter import run_slack_dry_run_adapter


class NotificationSlackDryRunTests(unittest.TestCase):
    def test_slack_dry_run_missing_channel_warns(self) -> None:
        cfg = replace(load_notification_config(), slack_channel="")
        result = run_slack_dry_run_adapter(
            cfg,
            {
                "trade_date": "2026-05-15",
                "notice": "Paper Trading only. No live orders were executed.",
                "buy": {"final_allowed": 1},
                "sell": {"sell_signals": 1, "review_required": 1},
            },
            severity="WARNING",
        )
        self.assertTrue(result["dry_run"])
        self.assertIn("SLACK_CHANNEL_MISSING", result["warnings"])
        self.assertIn("Paper Trading only", result["text"])


if __name__ == "__main__":
    unittest.main()
