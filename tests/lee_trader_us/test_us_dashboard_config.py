from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config


class DashboardConfigTests(unittest.TestCase):
    def test_invalid_formats_fall_back_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "US_DASHBOARD_OUTPUT_DIR": tmpdir,
                    "US_DASHBOARD_FORMAT": "",
                    "US_DASHBOARD_DEFAULT_LOOKBACK_DAYS": "bad",
                },
                clear=False,
            ):
                cfg = load_dashboard_config()
                self.assertEqual(cfg.formats, ("json", "markdown"))
                self.assertEqual(cfg.default_lookback_days, 60)
                self.assertTrue(cfg.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
