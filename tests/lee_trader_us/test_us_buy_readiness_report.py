from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.buy_automation.run_us_buy_readiness import main


class BuyReadinessReportTests(unittest.TestCase):
    @patch("python.us.buy_automation.run_us_buy_readiness.build_paper_backtest_summary")
    @patch("python.us.buy_automation.run_us_buy_readiness.evaluate_live_readiness")
    @patch("sys.argv", ["run_us_buy_readiness", "--format", "json"])
    def test_json_report_is_generated(self, evaluate_mock, summary_mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluate_mock.return_value = {
                "benchmark_symbol": "SPY",
                "evaluation_period_days": 60,
                "live_ready": False,
                "decision": "NOT_READY",
                "readiness_score": 42,
                "manual_approval_required": True,
                "reasons": ["PAPER_DAYS_BELOW_MINIMUM"],
                "paper_performance_summary": {
                    "paper_order_count": 0,
                    "total_return_pct": None,
                    "benchmark_return_pct": None,
                    "excess_return_pct": None,
                    "win_rate": None,
                    "max_drawdown_pct": None,
                    "data_missing_rate": None,
                },
                "operational_stability": {"scheduler_success_rate": None},
                "evaluation_generated_at": "2026-05-14T00:00:00+00:00",
            }
            summary_mock.return_value = {"period_label": "60", "paper_order_count": 0, "status": "NO_PAPER_ORDERS", "total_return_pct": None, "excess_return_pct": None}
            with patch("python.us.buy_automation.run_us_buy_readiness.load_live_promotion_policy") as policy_mock:
                policy_mock.return_value.output_dir = Path(tmpdir)
                policy_mock.return_value.compare_qqq = True
                exit_code = main()
            self.assertEqual(exit_code, 0)
            files = list(Path(tmpdir).glob("*_live_readiness.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "NOT_READY")


if __name__ == "__main__":
    unittest.main()
