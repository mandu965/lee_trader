from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest

from python.us.notification.channel_router import run_notification_channels
from python.us.notification.config import load_notification_config


def _payload_result() -> dict[str, object]:
    return {
        "payload": {
            "message_type": "US_PAPER_TRADING_DASHBOARD_SUMMARY",
            "trade_date": "2026-05-15",
            "generated_at": "2026-05-15T00:00:00+00:00",
            "mode": "SHADOW",
            "status": "WARNING",
            "paper_trading_only": True,
            "live_orders_executed": False,
            "buy": {"candidates": 5, "final_allowed": 1, "conflict_blocked": 2},
            "sell": {"positions": 4, "sell_signals": 1, "review_required": 1},
            "risk": {"data_missing_rate": 8.5, "fail_safe_triggered": True, "top_warning_reason": "DATA_MISSING"},
            "health": {"scheduler_status": "PASS", "dashboard_status": "PASS"},
            "readiness": {"live_ready": False, "readiness_score": 62, "manual_approval_required": True},
            "notice": "Paper Trading only. No live orders were executed.",
        },
        "valid": True,
        "warnings": [],
        "errors": [],
    }


class NotificationChannelRouterTests(unittest.TestCase):
    def test_live_channel_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = load_notification_config()
            cfg = replace(
                base,
                output_dir=base.root_dir / tmpdir,
                approvals_dir=(base.root_dir / tmpdir) / "approvals",
                logs_dir=(base.root_dir / tmpdir) / "logs",
                channels=("EMAIL_LIVE",),
            )
            result = run_notification_channels(cfg, payload_result=_payload_result(), emit_console=False)
            self.assertEqual(result["channels"]["EMAIL_LIVE"]["status"], "BLOCKED")
            self.assertIn("LIVE_NOTIFICATION_NOT_IMPLEMENTED", result["errors"])


if __name__ == "__main__":
    unittest.main()
