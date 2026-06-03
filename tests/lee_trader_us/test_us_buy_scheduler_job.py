from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.buy_automation.scheduler_job import run_buy_scheduler_job


class BuySchedulerJobTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "0",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_disabled_scheduler_skips_execution(self) -> None:
        result = run_buy_scheduler_job(emit_console=False)
        self.assertFalse(result["enabled"])
        self.assertFalse(result["automation_executed"])
        self.assertFalse(result["report_executed"])
        self.assertEqual(result["error"], "SCHEDULER_DISABLED")

    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_TRADE_SCHEDULER_ENABLED": "1",
            "US_TRADE_SCHEDULER_RUN_ORCHESTRATION": "1",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_scheduler_detects_orchestration_conflict(self) -> None:
        result = run_buy_scheduler_job(emit_console=False)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "SCHEDULER_CONFIGURATION_CONFLICT")

    @patch("python.us.buy_automation.scheduler_job.write_buy_report_markdown")
    @patch("python.us.buy_automation.scheduler_job.write_buy_report_json")
    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_BUY_SCHEDULER_RUN_REPORT": "1",
            "US_BUY_AUTOMATION_MODE": "LIVE",
            "US_BUY_AUTOMATION_ENABLED": "1",
        },
        clear=False,
    )
    def test_live_mode_is_blocked_in_scheduler(self, write_json_mock, write_md_mock) -> None:
        result = run_buy_scheduler_job(emit_console=False)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["mode"], "LIVE")
        self.assertFalse(result["automation_executed"])
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "LIVE_DISABLED_IN_SCHEDULER")
        self.assertTrue(result["report_executed"])
        write_json_mock.assert_called_once()
        write_md_mock.assert_called_once()

    @patch("python.us.buy_automation.scheduler_job.run_buy_automation", side_effect=RuntimeError("boom"))
    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR": "0",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_error_is_isolated_by_default(self, _) -> None:
        result = run_buy_scheduler_job(emit_console=False)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "US_BUY_AUTOMATION_ERROR")
        self.assertFalse(result["pipeline_should_fail"])

    @patch("python.us.buy_automation.scheduler_job.run_buy_automation", side_effect=RuntimeError("boom"))
    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR": "1",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_error_can_fail_pipeline_when_enabled(self, _) -> None:
        with self.assertRaises(RuntimeError):
            run_buy_scheduler_job(emit_console=False)

    @patch("python.us.buy_automation.scheduler_job.write_buy_report_markdown")
    @patch("python.us.buy_automation.scheduler_job.write_buy_report_json")
    @patch("python.us.buy_automation.scheduler_job.finalize_buy_report")
    @patch("python.us.buy_automation.scheduler_job.run_buy_automation")
    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_BUY_SCHEDULER_RUN_AUTOMATION": "1",
            "US_BUY_SCHEDULER_RUN_REPORT": "1",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_scheduler_returns_summary_structure(
        self,
        run_automation_mock,
        finalize_mock,
        write_json_mock,
        write_md_mock,
    ) -> None:
        run_automation_mock.return_value = {
            "loaded_candidates": 5,
            "allowed_candidates": 1,
            "blocked_candidates": 4,
            "paper_orders": [],
            "block_summary": {"DATA_MISSING": 4},
            "events": [],
        }
        finalize_mock.return_value = {"trade_date": "2026-05-14"}

        result = run_buy_scheduler_job(trade_date="2026-05-14", emit_console=False)

        self.assertTrue(result["success"])
        self.assertTrue(result["automation_executed"])
        self.assertTrue(result["report_executed"])
        self.assertEqual(result["summary"]["loaded_candidates"], 5)
        self.assertEqual(result["summary"]["allowed_candidates"], 1)
        self.assertEqual(result["summary"]["blocked_candidates"], 4)
        self.assertIn("paper_orders", result["summary"])
        write_json_mock.assert_called_once()
        write_md_mock.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "US_BUY_SCHEDULER_ENABLED": "1",
            "US_BUY_SCHEDULER_RUN_AUTOMATION": "0",
            "US_BUY_SCHEDULER_RUN_REPORT": "0",
            "US_BUY_AUTOMATION_MODE": "SHADOW",
        },
        clear=False,
    )
    def test_scheduler_can_skip_automation_and_report(self) -> None:
        result = run_buy_scheduler_job(emit_console=False)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["success"])
        self.assertFalse(result["automation_executed"])
        self.assertFalse(result["report_executed"])
        self.assertEqual(result["summary"]["loaded_candidates"], 0)
        self.assertEqual(result["summary"]["allowed_candidates"], 0)
        self.assertEqual(result["summary"]["blocked_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
