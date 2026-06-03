from __future__ import annotations

import unittest

from python.us.dashboard.dashboard_notification_formatter import (
    build_dashboard_notification_json_payload,
    render_dashboard_notification_text,
)


class DashboardNotificationFormatterTests(unittest.TestCase):
    def test_notification_contains_paper_only_notice(self) -> None:
        payload = {
            "meta": {"trade_date": "2026-05-15", "mode": "SHADOW"},
            "daily_overview": {
                "status": "WARNING",
                "buy_candidates": 5,
                "final_buy_allowed": 1,
                "sell_signals": 1,
                "review_required_count": 1,
                "conflict_blocked_count": 2,
                "fail_safe_triggered": True,
                "top_warning_reason": "DATA_MISSING",
            },
            "buy_decision_monitor": {"items": [{"symbol": "AAPL"}]},
            "sell_decision_monitor": {"loaded_positions": 4, "items": [{"symbol": "NVDA"}]},
            "risk_data_quality_monitor": {"data_missing_rate": 8.5},
            "scheduler_health_monitor": {"health_check_status": "PASS", "status": "PASS"},
            "live_readiness_monitor": {"live_ready": False, "readiness_score": 62, "manual_approval_required": True},
            "warnings": [],
            "errors": [],
        }
        text = render_dashboard_notification_text(payload, dashboard_health={"dashboard_health_status": "PASS"})
        json_payload = build_dashboard_notification_json_payload(payload, dashboard_health={"dashboard_health_status": "PASS"})
        self.assertIn("Paper Trading only", text)
        self.assertFalse(json_payload["live_orders_executed"])


if __name__ == "__main__":
    unittest.main()
