from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.us_micro_live_operations import (
    derive_health_status,
    generate_action_required,
    write_operations_csv,
)


class USMicroLiveOperationsTests(unittest.TestCase):
    def _base_report(self) -> dict[str, object]:
        return {
            "trade_date": "2026-05-16",
            "account_id": "US_LIVE_TEST",
            "kill_switch": {"active_count": 0, "active_rows": [], "rows": []},
            "precheck": {"ERROR": 0},
            "approvals": {"pending": 1, "expired": 1, "approved": 0, "rejected": 0, "rows": []},
            "orders": {
                "counts": {"ORDER_UNKNOWN": 1, "SYNC_ERROR": 0, "ORDER_PARTIALLY_FILLED": 0, "ORDER_REJECTED": 0},
                "rows": [],
            },
            "reconciliation": {"mismatch": 0, "critical": 1, "rows": []},
            "daily_risk_usage": {"row": {}, "total_order_count": 0, "failed_order_count": 0, "blocked_order_count": 0, "new_buy_count": 0},
            "block_logs": {"rows": []},
        }

    def test_generate_actions_and_health(self) -> None:
        report = self._base_report()
        actions = generate_action_required(report)
        report["actions"] = actions
        health = derive_health_status(report)
        self.assertTrue(any(item["reason_code"] == "approval_pending" for item in actions))
        self.assertTrue(any(item["reason_code"] == "approval_expired" for item in actions))
        self.assertTrue(any(item["reason_code"] == "order_unknown_exists" for item in actions))
        self.assertTrue(any(item["reason_code"] == "reconciliation_critical" for item in actions))
        self.assertEqual(health["status"], "CRITICAL")

    def test_write_operations_csv_creates_files(self) -> None:
        report = {
            "trade_date": "2026-05-16",
            "account_id": "US_LIVE_TEST",
            "orders": {"rows": [{"trade_date": "2026-05-16", "account_id": "US_LIVE_TEST", "micro_order_id": "USMO_1"}]},
            "approvals": {"rows": [{"approval_id": "USAPP_1", "trade_date": "2026-05-16", "account_id": "US_LIVE_TEST"}]},
            "block_logs": {"rows": [{"trade_date": "2026-05-16", "account_id": "US_LIVE_TEST"}]},
            "reconciliation": {"rows": [{"recon_date": "2026-05-16", "account_id": "US_LIVE_TEST"}]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            files = write_operations_csv(report, output_dir=tmpdir)
            self.assertEqual(len(files), 4)
            for path in files:
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
