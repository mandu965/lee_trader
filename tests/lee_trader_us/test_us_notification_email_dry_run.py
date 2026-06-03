from __future__ import annotations

from dataclasses import replace
import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.email_dry_run_adapter import run_email_dry_run_adapter


class NotificationEmailDryRunTests(unittest.TestCase):
    def test_email_dry_run_does_not_require_recipients(self) -> None:
        cfg = replace(load_notification_config(), email_recipients=())
        result = run_email_dry_run_adapter(
            cfg,
            {
                "trade_date": "2026-05-15",
                "mode": "SHADOW",
                "status": "WARNING",
                "buy": {"candidates": 5, "final_allowed": 1, "conflict_blocked": 2},
                "sell": {"positions": 4, "sell_signals": 1, "review_required": 1},
                "risk": {"data_missing_rate": 8.5, "fail_safe_triggered": True, "top_warning_reason": "DATA_MISSING"},
                "health": {"scheduler_status": "PASS", "dashboard_status": "PASS"},
                "readiness": {"live_ready": False, "readiness_score": 62, "manual_approval_required": True},
                "notice": "Paper Trading only. No live orders were executed.",
            },
            severity="WARNING",
        )
        self.assertTrue(result["dry_run"])
        self.assertIn("EMAIL_RECIPIENT_MISSING", result["warnings"])
        self.assertIn("[WARNING]", result["subject"])


if __name__ == "__main__":
    unittest.main()
