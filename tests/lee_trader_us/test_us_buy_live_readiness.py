from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from python.us.buy_automation.live_readiness_evaluator import evaluate_live_readiness


class BuyLiveReadinessTests(unittest.TestCase):
    def test_not_ready_when_paper_days_below_minimum(self) -> None:
        with patch("python.us.buy_automation.live_readiness_evaluator.load_buy_automation_run_logs", return_value=[]), patch(
            "python.us.buy_automation.live_readiness_evaluator.load_scheduler_job_logs", return_value=[]
        ), patch(
            "python.us.buy_automation.live_readiness_evaluator.build_paper_backtest_summary",
            return_value={
                "paper_order_count": 0,
                "positive_excess_return": False,
                "benchmark_data_missing": True,
                "max_drawdown_pct": None,
                "win_rate": None,
                "data_missing_rate": None,
            },
        ):
            result = evaluate_live_readiness(days=60)
        self.assertFalse(result["live_ready"])
        self.assertIn("PAPER_DAYS_BELOW_MINIMUM", result["reasons"])

    def test_manual_approval_is_always_required(self) -> None:
        logs = [{"trade_date": "2026-05-01", "mode": "SHADOW", "candidates": []}] * 25 + [{"trade_date": "2026-05-01", "mode": "PAPER", "candidates": []}] * 65
        scheduler_logs = [{"enabled": True, "success": True, "report_executed": True, "error": "LIVE_DISABLED_IN_SCHEDULER"}] * 100
        with patch("python.us.buy_automation.live_readiness_evaluator.load_buy_automation_run_logs", return_value=logs), patch(
            "python.us.buy_automation.live_readiness_evaluator.load_scheduler_job_logs", return_value=scheduler_logs
        ), patch(
            "python.us.buy_automation.live_readiness_evaluator.build_paper_backtest_summary",
            return_value={
                "paper_order_count": 30,
                "positive_excess_return": True,
                "benchmark_data_missing": False,
                "max_drawdown_pct": 0.05,
                "win_rate": 0.7,
                "data_missing_rate": 0.01,
                "excess_return_pct": 0.03,
            },
        ), patch(
            "python.us.buy_automation.live_readiness_evaluator._candidate_data_missing_rate",
            return_value=0.01,
        ), patch(
            "python.us.buy_automation.live_readiness_evaluator._aggregate_validation",
            return_value={"fail_safe_block_count": 1, "invalid_decision_logs": []},
        ):
            result = evaluate_live_readiness(days=60)
        self.assertTrue(result["manual_approval_required"])


if __name__ == "__main__":
    unittest.main()
