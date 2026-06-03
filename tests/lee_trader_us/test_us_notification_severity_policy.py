from __future__ import annotations

import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.severity_policy import determine_notification_severity


class NotificationSeverityPolicyTests(unittest.TestCase):
    def test_live_orders_executed_true_is_critical(self) -> None:
        cfg = load_notification_config()
        result = determine_notification_severity(
            {
                "status": "OK",
                "paper_trading_only": True,
                "live_orders_executed": True,
                "sell": {"review_required": 0},
                "risk": {"data_missing_rate": 0},
                "health": {"scheduler_status": "PASS", "dashboard_status": "PASS"},
            },
            validation_result={"warnings": [], "errors": []},
            cfg=cfg,
        )
        self.assertEqual(result["severity"], "CRITICAL")
        self.assertIn("LIVE_ORDERS_EXECUTED_TRUE", result["reasons"])


if __name__ == "__main__":
    unittest.main()
