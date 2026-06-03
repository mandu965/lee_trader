from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from python.us.notification.config import load_notification_config


class NotificationConfigTests(unittest.TestCase):
    def test_invalid_mode_and_channels_fall_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as _tmpdir:
            with patch.dict(
                os.environ,
                {
                    "US_NOTIFICATION_ADAPTER_MODE": "bad",
                    "US_NOTIFICATION_CHANNELS": "bad_channel",
                },
                clear=False,
            ):
                cfg = load_notification_config()
                self.assertEqual(cfg.mode, "DRY_RUN")
                self.assertEqual(cfg.channels, ("CONSOLE", "FILE"))
                self.assertTrue(cfg.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
