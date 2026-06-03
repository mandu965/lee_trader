from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python.us.trade_orchestration.config import TradeOrchestrationConfig
from python.us.trade_orchestration.integrated_report_generator import (
    build_integrated_report,
    render_integrated_report_markdown,
    write_integrated_report_json,
    write_integrated_report_markdown,
)


class IntegratedReportGeneratorTests(unittest.TestCase):
    def _cfg(self, output_dir: str) -> TradeOrchestrationConfig:
        return TradeOrchestrationConfig(
            root_dir=Path(output_dir),
            mode="SHADOW",
            enabled=False,
            block_buy_if_position_exists=True,
            block_buy_if_sell_signal_exists=True,
            block_buy_after_full_exit_days=10,
            block_buy_on_review_required=True,
            sell_priority_over_buy=True,
            conflict_failsafe=True,
            report_enabled=True,
            report_output_dir=Path(output_dir),
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
            lock_dir=Path(output_dir) / "lock",
            checklist_output_dir=Path(output_dir),
            warnings=(),
        )

    def test_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._cfg(tmpdir)
            report = build_integrated_report(
                {
                    "trade_date": "2026-05-14",
                    "mode": "SHADOW",
                    "enabled": False,
                    "success": True,
                    "fail_safe_triggered": False,
                    "warnings": [],
                    "sell_report": {"loaded_positions": 1, "hold_positions": 1, "sell_signals": 0, "partial_sell_signals": 0, "review_required": 0, "reason_summary": {}, "decisions": []},
                    "buy_report": {"loaded_candidates": 1, "allowed_before_conflict": 1, "allowed_after_conflict": 1, "allowed_candidates": 0, "blocked_candidates": 1, "block_summary": {}, "candidates": []},
                    "portfolio_state": {"open_positions": [], "open_position_map": {}},
                    "conflict_summary": {"OPEN_POSITION_EXISTS": 1},
                    "conflict_results": [],
                },
                cfg,
            )
            json_path = write_integrated_report_json(report, cfg)
            md_path = write_integrated_report_markdown(report, cfg)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("# US Integrated Trade Report", render_integrated_report_markdown(report))


if __name__ == "__main__":
    unittest.main()
