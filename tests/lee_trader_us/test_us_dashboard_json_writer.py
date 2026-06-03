from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_json_writer import write_dashboard_outputs


class DashboardJsonWriterTests(unittest.TestCase):
    def test_json_and_latest_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_DASHBOARD_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_dashboard_config()
                paths = write_dashboard_outputs(
                    {
                        "meta": {"trade_date": "2026-05-14"},
                        "daily_overview": {"status": "OK"},
                        "paper_portfolio_summary": {"status": "OK"},
                        "buy_decision_monitor": {"status": "OK", "items": []},
                        "sell_decision_monitor": {"status": "OK", "items": []},
                        "conflict_guard_monitor": {"status": "OK", "items": []},
                        "paper_performance_monitor": {"status": "OK"},
                        "benchmark_comparison": {"status": "OK"},
                        "risk_data_quality_monitor": {"status": "OK"},
                        "scheduler_health_monitor": {"status": "OK"},
                        "live_readiness_monitor": {"status": "OK", "live_transition_note": "x"},
                        "warnings": [],
                        "errors": [],
                    },
                    cfg,
                )
                self.assertTrue(Path(paths["json"]).exists())
                self.assertTrue(Path(paths["latest_json"]).exists())
                self.assertTrue(Path(paths["markdown"]).exists())
                self.assertTrue(Path(paths["latest_markdown"]).exists())


if __name__ == "__main__":
    unittest.main()
