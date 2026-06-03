from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_summary import build_daily_overview, build_live_readiness_monitor


class DashboardSummaryTests(unittest.TestCase):
    def test_daily_overview_returns_data_missing_without_sources(self) -> None:
        raw_data = {
            "trade_date": "2026-05-14",
            "loaded_at": "2026-05-14T20:00:00",
            "integrated_report": None,
            "buy_decisions": [],
            "sell_decisions": [],
            "conflicts": [],
            "paper_buy_orders": [],
            "paper_sell_orders": [],
            "orchestration_logs": [],
            "missing_sources": ["integrated_daily_report"],
            "load_warnings": [],
        }
        overview = build_daily_overview(raw_data)
        self.assertEqual(overview["status"], "ERROR")
        self.assertFalse(overview["orchestration_executed"])

    def test_live_readiness_note_is_explicit(self) -> None:
        section = build_live_readiness_monitor(
            {
                "readiness": {
                    "live_ready": True,
                    "readiness_score": 100,
                    "manual_approval_required": True,
                    "reasons": [],
                    "promotion_policy": {
                        "min_shadow_days": 20,
                        "min_paper_days": 60,
                        "min_paper_orders": 20,
                        "min_win_rate_pct": 0.5,
                        "max_drawdown_pct": 0.15,
                        "min_excess_return_pct": 0.0,
                        "max_data_missing_rate_pct": 0.05,
                        "min_scheduler_success_rate_pct": 0.95,
                    },
                    "operational_stability": {
                        "shadow_days": 30,
                        "paper_days": 90,
                        "scheduler_success_rate": 1.0,
                        "data_missing_rate": 0.01,
                    },
                    "paper_performance_summary": {
                        "paper_order_count": 25,
                        "win_rate": 0.6,
                        "max_drawdown_pct": 0.1,
                        "excess_return_pct": 0.02,
                    },
                }
            }
        )
        self.assertIn("자동 실매매 전환", section["live_transition_note"])


if __name__ == "__main__":
    unittest.main()
