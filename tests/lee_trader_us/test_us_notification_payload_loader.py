from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest

from python.us.notification.config import load_notification_config
from python.us.notification.notification_payload_loader import load_notification_payload


class NotificationPayloadLoaderTests(unittest.TestCase):
    def test_missing_payload_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = replace(load_notification_config(), payload_dir=load_notification_config().root_dir / "does_not_exist")
            result = load_notification_payload(cfg, trade_date="2026-05-15")
            self.assertFalse(result["valid"])
            self.assertIn("PAYLOAD_MISSING", result["errors"])

    def test_paper_trading_only_false_is_critical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = replace(load_notification_config(), payload_dir=replace(load_notification_config()).root_dir / tmpdir)
            path = cfg.payload_dir / "2026-05-15_notification.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "message_type": "US_PAPER_TRADING_DASHBOARD_SUMMARY",
                        "trade_date": "2026-05-15",
                        "generated_at": "2026-05-15T00:00:00+00:00",
                        "mode": "SHADOW",
                        "status": "WARNING",
                        "paper_trading_only": False,
                        "live_orders_executed": False,
                        "notice": "Paper Trading only. No live orders were executed.",
                    }
                ),
                encoding="utf-8",
            )
            result = load_notification_payload(cfg, trade_date="2026-05-15")
            self.assertFalse(result["valid"])
            self.assertEqual(result["severity_hint"], "CRITICAL")
            self.assertIn("PAPER_TRADING_ONLY_FALSE", result["errors"])


if __name__ == "__main__":
    unittest.main()
