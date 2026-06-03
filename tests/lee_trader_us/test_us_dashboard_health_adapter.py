from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_health_adapter import run_dashboard_health_adapter


class DashboardHealthAdapterTests(unittest.TestCase):
    def test_missing_json_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_DASHBOARD_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_dashboard_config()
                result = run_dashboard_health_adapter(
                    trade_date="2026-05-15",
                    dashboard_result={},
                    cfg=cfg,
                )
                self.assertEqual(result["dashboard_health_status"], "ERROR")
                self.assertIn("DASHBOARD_JSON_REPORT_MISSING", result["errors"])


if __name__ == "__main__":
    unittest.main()
