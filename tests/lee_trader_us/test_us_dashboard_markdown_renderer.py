from __future__ import annotations

import unittest

from python.us.dashboard.dashboard_markdown_renderer import PAPER_NOTICE, render_dashboard_markdown


class DashboardMarkdownRendererTests(unittest.TestCase):
    def test_markdown_includes_paper_notice(self) -> None:
        markdown = render_dashboard_markdown(
            {
                "meta": {"trade_date": "2026-05-14", "generated_at": "x", "mode": "SHADOW", "paper_trading_only": True, "live_trading_enabled": False},
                "daily_overview": {"status": "OK"},
                "paper_portfolio_summary": {"status": "OK"},
                "buy_decision_monitor": {"status": "DATA_MISSING", "items": []},
                "sell_decision_monitor": {"status": "DATA_MISSING", "items": []},
                "conflict_guard_monitor": {"status": "DATA_MISSING", "items": []},
                "paper_performance_monitor": {"status": "DATA_MISSING"},
                "benchmark_comparison": {"status": "DATA_MISSING"},
                "risk_data_quality_monitor": {"status": "WARNING"},
                "scheduler_health_monitor": {"status": "DATA_MISSING"},
                "live_readiness_monitor": {"status": "DATA_MISSING", "live_transition_note": "note"},
                "warnings": [],
                "errors": [],
            }
        )
        self.assertIn(PAPER_NOTICE, markdown)
        self.assertIn("# US Paper Trading Dashboard", markdown)


if __name__ == "__main__":
    unittest.main()
