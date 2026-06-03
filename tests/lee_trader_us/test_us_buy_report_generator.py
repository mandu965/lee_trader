from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from python.us.buy_automation.report_generator import (
    finalize_buy_report,
    load_buy_automation_run_log,
    render_buy_report_markdown,
    write_buy_report_json,
    write_buy_report_markdown,
)


class BuyReportGeneratorTests(unittest.TestCase):
    def _raw_payload(self) -> dict[str, object]:
        return {
            "trade_date": "2026-05-14",
            "mode": "SHADOW",
            "enabled": False,
            "loaded_candidates": 1,
            "ranking_source": "rule_v1",
            "config_snapshot": {"max_daily_amount_usd": 100.0, "max_daily_symbols": 1, "min_score": 70.0},
            "events": [],
            "paper_orders": [],
            "candidates": [
                {
                    "candidate_id": "C1",
                    "decision_id": "D1",
                    "guard_log_id": "G1",
                    "symbol": "AAPL",
                    "rank": 1,
                    "score": 60.0,
                    "probability": None,
                    "allowed": False,
                    "block_reasons": ["AUTOMATION_DISABLED", "DATA_MISSING"],
                    "applied_rules": [{"rule": "PRICE_RANGE", "result": "PASS"}],
                    "allocated_amount_usd": 0.0,
                }
            ],
        }

    def test_load_and_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buy_automation_2026-05-14_SHADOW_1.json"
            path.write_text(json.dumps(self._raw_payload()), encoding="utf-8")
            raw = load_buy_automation_run_log(trade_date="2026-05-14", input_dir=tmpdir)
            report = finalize_buy_report(raw)
            json_path = write_buy_report_json(report, output_dir=tmpdir)
            md_path = write_buy_report_markdown(report, output_dir=tmpdir)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("# US BUY Automation Report", render_buy_report_markdown(report))


if __name__ == "__main__":
    unittest.main()
