from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.trade_orchestration.scheduler_job import run_trade_scheduler_job


class TradeSchedulerJobTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "0",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
        },
        clear=False,
    )
    def test_disabled_scheduler_does_not_run(self) -> None:
        result = run_trade_scheduler_job(emit_console=False)
        self.assertFalse(result["guard_passed"])
        self.assertFalse(result["success"])
        self.assertIn("SCHEDULER_DISABLED", result["errors"])

    @patch("python.us.trade_orchestration.scheduler_job.evaluate_scheduler_guard")
    @patch("python.us.trade_orchestration.scheduler_job.acquire_run_lock")
    @patch("python.us.trade_orchestration.scheduler_job.run_trade_orchestration")
    @patch("python.us.trade_orchestration.scheduler_job.run_trade_health_check")
    @patch("python.us.trade_orchestration.scheduler_job.write_operations_checklist")
    @patch("python.us.trade_orchestration.scheduler_job.build_operations_checklist")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_orchestration_exception_is_isolated_by_default(
        self,
        checklist_mock,
        write_checklist_mock,
        health_mock,
        orchestrator_mock,
        lock_mock,
        guard_mock,
    ) -> None:
        guard_mock.return_value = {
            "can_run": True,
            "mode": "SHADOW",
            "warnings": [],
            "errors": [],
            "pipeline_should_fail": False,
            "ranking_trade_date": "2026-05-14",
        }
        lock_mock.return_value = {"lock_acquired": True}
        orchestrator_mock.side_effect = RuntimeError("boom")
        result = run_trade_scheduler_job(trade_date="2026-05-14", emit_console=False)
        self.assertFalse(result["success"])
        self.assertFalse(result["pipeline_should_fail"])
        self.assertTrue(any("TRADE_SCHEDULER_JOB_ERROR" in item for item in result["errors"]))

    @patch("python.us.trade_orchestration.scheduler_job.evaluate_scheduler_guard")
    @patch("python.us.trade_orchestration.scheduler_job.acquire_run_lock")
    @patch("python.us.trade_orchestration.scheduler_job.run_trade_orchestration")
    @patch("python.us.trade_orchestration.scheduler_job.run_trade_health_check")
    @patch("python.us.trade_orchestration.scheduler_job.write_operations_checklist")
    @patch("python.us.trade_orchestration.scheduler_job.build_operations_checklist")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_scheduler_returns_summary_structure(
        self,
        checklist_mock,
        write_checklist_mock,
        health_mock,
        orchestrator_mock,
        lock_mock,
        guard_mock,
    ) -> None:
        guard_mock.return_value = {
            "can_run": True,
            "mode": "SHADOW",
            "warnings": [],
            "errors": [],
            "pipeline_should_fail": False,
            "ranking_trade_date": "2026-05-14",
        }
        lock_mock.return_value = {"lock_acquired": True}
        orchestrator_mock.return_value = {
            "success": True,
            "report_generated": True,
            "warnings": [],
            "sell_summary": {"loaded_positions": 4, "sell_signals": 1, "hold_positions": 2, "review_required": 1},
            "buy_summary": {"loaded_candidates": 5, "allowed_before_conflict": 2, "conflict_blocked": 1, "allowed_after_conflict": 1},
            "conflict_summary": {"OPEN_POSITION_EXISTS": 1, "SELL_SIGNAL_EXISTS": 0, "COOLDOWN_ACTIVE": 0},
            "trade_date": "2026-05-14",
        }
        health_mock.return_value = {"health_status": "PASS", "warnings": [], "errors": [], "pipeline_should_fail": False}
        checklist_mock.return_value = "# checklist"
        result = run_trade_scheduler_job(trade_date="2026-05-14", emit_console=False)
        self.assertTrue(result["success"])
        self.assertTrue(result["guard_passed"])
        self.assertTrue(result["lock_acquired"])
        self.assertTrue(result["orchestration_executed"])
        self.assertTrue(result["health_check_passed"])
        self.assertEqual(result["summary"]["buy_candidates"], 5)


if __name__ == "__main__":
    unittest.main()
