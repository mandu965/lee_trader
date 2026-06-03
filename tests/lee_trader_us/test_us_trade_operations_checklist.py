from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python.us.trade_orchestration.config import TradeOrchestrationConfig
from python.us.trade_orchestration.operations_checklist import build_operations_checklist, write_operations_checklist


class TradeOperationsChecklistTests(unittest.TestCase):
    def _cfg(self, tmpdir: str) -> TradeOrchestrationConfig:
        return TradeOrchestrationConfig(
            root_dir=Path(tmpdir),
            mode="SHADOW",
            enabled=False,
            block_buy_if_position_exists=True,
            block_buy_if_sell_signal_exists=True,
            block_buy_after_full_exit_days=10,
            block_buy_on_review_required=True,
            sell_priority_over_buy=True,
            conflict_failsafe=True,
            report_enabled=True,
            report_output_dir=Path(tmpdir),
            report_formats=("json", "markdown"),
            fail_pipeline_on_error=False,
            scheduler_enabled=False,
            scheduler_run_orchestration=True,
            scheduler_run_health_check=True,
            scheduler_run_report=True,
            scheduler_fail_pipeline_on_error=False,
            scheduler_allow_live=False,
            scheduler_prevent_duplicate_run=True,
            scheduler_lock_ttl_seconds=1800,
            scheduler_max_runtime_seconds=600,
            disable_buy_only_scheduler_when_orchestration=True,
            warn_if_buy_only_scheduler_enabled=True,
            health_check_enabled=True,
            health_check_fail_on_missing_report=False,
            health_check_fail_on_invalid_log=False,
            health_check_max_data_missing_rate_pct=20.0,
            lock_dir=Path(tmpdir) / "lock",
            checklist_output_dir=Path(tmpdir),
            warnings=(),
        )

    def test_checklist_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown = build_operations_checklist(
                orchestration_result={
                    "success": True,
                    "sell_executed": True,
                    "buy_executed": True,
                    "conflict_summary": {},
                    "sell_report": {"review_required": 0},
                    "integrated_report": {"paper_portfolio_summary": {}},
                    "report_generated": True,
                    "mode": "SHADOW",
                    "warnings": [],
                    "errors": [],
                },
                health_result={"data_missing_rate_pct": 0},
            )
            self.assertIn("# US Trade Orchestration Daily Checklist", markdown)
            cfg = self._cfg(tmpdir)
            path = write_operations_checklist(cfg, trade_date="2026-05-14", markdown=markdown)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
