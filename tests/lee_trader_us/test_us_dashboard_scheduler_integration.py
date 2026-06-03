from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.dashboard.scheduler_integration import run_dashboard_scheduler_integration


class DashboardSchedulerIntegrationTests(unittest.TestCase):
    @patch.dict(os.environ, {"US_DASHBOARD_ENABLED": "0"}, clear=False)
    def test_disabled_dashboard_is_skipped(self) -> None:
        result = run_dashboard_scheduler_integration(trade_date="2026-05-15", force=False)
        self.assertFalse(result["dashboard_executed"])
        self.assertTrue(result["success"])

    @patch("python.us.dashboard.scheduler_integration.load_dashboard_raw_data", side_effect=RuntimeError("boom"))
    @patch.dict(
        os.environ,
        {
            "US_DASHBOARD_ENABLED": "1",
            "US_DASHBOARD_FAIL_PIPELINE_ON_ERROR": "0",
        },
        clear=False,
    )
    def test_dashboard_failure_is_isolated_by_default(self, _) -> None:
        result = run_dashboard_scheduler_integration(trade_date="2026-05-15", force=False)
        self.assertFalse(result["success"])
        self.assertFalse(result["pipeline_should_fail"])
        self.assertTrue(any("DASHBOARD_REPORT_GENERATION_FAILED" in item for item in result["errors"]))


if __name__ == "__main__":
    unittest.main()
