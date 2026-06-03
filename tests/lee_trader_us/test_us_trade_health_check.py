from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.trade_orchestration.health_check import run_trade_health_check


class TradeHealthCheckTests(unittest.TestCase):
    def test_missing_report_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_TRADE_REPORT_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_trade_orchestration_config()
                result = run_trade_health_check(
                    cfg,
                    orchestration_result={
                        "trade_date": "2026-05-14",
                        "integrated_report": {"buy_summary": {}, "sell_summary": {}, "conflict_summary": {}},
                        "buy_report": {"loaded_candidates": 0, "candidates": [], "block_summary": {}},
                        "sell_report": {"review_required": 0},
                        "final_action_summary": {"BUY_BLOCKED": 0},
                        "portfolio_state": {"status": "OK"},
                    },
                )
                self.assertIn("INTEGRATED_JSON_REPORT_MISSING", result["warnings"])

    def test_invalid_buy_decision_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_TRADE_REPORT_OUTPUT_DIR": tmpdir}, clear=False):
                cfg = load_trade_orchestration_config()
                Path(tmpdir, "2026-05-14_integrated_trade_report.json").write_text("{}", encoding="utf-8")
                Path(tmpdir, "2026-05-14_integrated_trade_report.md").write_text("# x", encoding="utf-8")
                result = run_trade_health_check(
                    cfg,
                    orchestration_result={
                        "trade_date": "2026-05-14",
                        "integrated_report": {"buy_summary": {"x": 1}, "sell_summary": {"x": 1}, "conflict_summary": {}},
                        "buy_report": {"loaded_candidates": 1, "candidates": [{"symbol": "AAPL", "allowed": False, "block_reasons": []}], "block_summary": {}},
                        "sell_report": {"review_required": 0},
                        "final_action_summary": {"BUY_BLOCKED": 1},
                        "portfolio_state": {"status": "OK"},
                    },
                )
                self.assertIn("INVALID_BUY_DECISION_LOG", result["warnings"])


if __name__ == "__main__":
    unittest.main()
