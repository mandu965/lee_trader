from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.manual_approval import build_manual_approval


class NotificationManualApprovalTests(unittest.TestCase):
    def test_manual_approval_creates_pending_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = load_notification_config()
            approvals_dir = (base.root_dir / tmpdir) / "approvals"
            cfg = replace(base, approvals_dir=approvals_dir, require_manual_approval=True, mode="MANUAL_APPROVAL")
            result = build_manual_approval(
                cfg,
                trade_date="2026-05-15",
                severity="WARNING",
                payload={"message_type": "US_PAPER_TRADING_DASHBOARD_SUMMARY"},
            )
            self.assertEqual(result["approval_status"], "PENDING")
            self.assertTrue((approvals_dir / "2026-05-15_approval_pending.json").exists())


if __name__ == "__main__":
    unittest.main()
