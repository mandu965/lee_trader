from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_report_generator import build_dashboard_payload


class DashboardReportGeneratorTests(unittest.TestCase):
    def test_payload_contains_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_DASHBOARD_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_dashboard_config()
                payload = build_dashboard_payload(
                    {
                        "trade_date": "2026-05-14",
                        "loaded_at": "2026-05-14T20:00:00",
                        "integrated_report": {"mode": "SHADOW", "success": True, "buy_summary": {}, "sell_summary": {}, "conflict_summary": {}},
                        "orchestration_logs": [{"mode": "SHADOW", "success": True}],
                        "buy_decisions": [],
                        "sell_decisions": [],
                        "conflicts": [],
                        "paper_buy_orders": [],
                        "paper_sell_orders": [],
                        "paper_positions": [],
                        "paper_position_snapshots": [],
                        "scheduler_run_logs": [],
                        "scheduler_health_rows": [],
                        "readiness": None,
                        "missing_sources": [],
                        "load_warnings": [],
                    },
                    cfg,
                )
                self.assertEqual(payload["meta"]["report_type"], "US_PAPER_TRADING_DASHBOARD")
                self.assertIn("daily_overview", payload)
                self.assertIn("live_readiness_monitor", payload)


if __name__ == "__main__":
    unittest.main()
