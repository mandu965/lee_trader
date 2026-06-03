from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_notification_payload import write_dashboard_notification_payloads


class DashboardNotificationPayloadTests(unittest.TestCase):
    def test_notification_payload_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_DASHBOARD_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_dashboard_config()
                paths = write_dashboard_notification_payloads(
                    cfg,
                    trade_date="2026-05-15",
                    text_payload="hello",
                    json_payload={"trade_date": "2026-05-15", "live_orders_executed": False},
                )
                self.assertTrue(Path(paths["text_path"]).exists())
                self.assertTrue(Path(paths["json_path"]).exists())
                self.assertTrue(Path(paths["latest_text_path"]).exists())
                self.assertTrue(Path(paths["latest_json_path"]).exists())


if __name__ == "__main__":
    unittest.main()
