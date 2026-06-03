from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.file_adapter import write_notification_adapter_files


class NotificationFileAdapterTests(unittest.TestCase):
    def test_file_adapter_creates_date_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = load_notification_config()
            cfg = replace(base, output_dir=base.root_dir / tmpdir)
            result = write_notification_adapter_files(
                cfg,
                trade_date="2026-05-15",
                text_summary="hello",
                payload={"message_type": "X"},
                severity="INFO",
                channel_results={},
                approval_record={},
                warnings=[],
                errors=[],
            )
            self.assertEqual(result["status"], "SUCCESS")
            self.assertTrue((cfg.output_dir / "2026-05-15_notification_adapter.json").exists())
            self.assertTrue((cfg.output_dir / "latest_notification_adapter.json").exists())


if __name__ == "__main__":
    unittest.main()
