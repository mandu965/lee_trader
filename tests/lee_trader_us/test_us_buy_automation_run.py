from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.buy_automation.decision_engine import run_buy_automation


class BuyAutomationRunTests(unittest.TestCase):
    @patch("python.us.buy_automation.decision_engine.persist_buy_automation_logs")
    @patch("python.us.buy_automation.decision_engine.load_buy_candidates")
    @patch("python.us.buy_automation.decision_engine.list_active_kill_switches")
    @patch.dict(
        os.environ,
        {
            "US_BUY_AUTOMATION_MODE": "LIVE",
            "US_BUY_AUTOMATION_ENABLED": "1",
            "US_BUY_MIN_PROB": "0",
        },
        clear=False,
    )
    def test_live_mode_never_creates_paper_order(self, kill_switch_mock, load_candidates_mock, persist_logs_mock) -> None:
        kill_switch_mock.return_value = []
        persist_logs_mock.return_value = {}
        load_candidates_mock.return_value = {
            "trade_date": __import__("datetime").date(2026, 5, 14),
            "events": [],
            "candidates": [
                {
                    "symbol": "AAPL",
                    "rank": 1,
                    "score": 85.0,
                    "probability": 0.9,
                    "reference_price": 100.0,
                    "recommend_grade": "BUY",
                    "financial_feature": {"financial_quality_score": 50},
                    "financial_quality_score": 50,
                    "relative_strength": {"rs_spy_20d": 0.1},
                    "gap_up_pct": 0.01,
                    "intraday_change_pct": 0.01,
                    "volatility_20d": 0.02,
                    "data_status": "OK",
                    "company_name": "Apple",
                    "sector": "Technology",
                    "score_detail_json": "{}",
                }
            ],
        }
        report = run_buy_automation(account_id="US_TEST", persist_logs=False)
        self.assertEqual(report["mode"], "LIVE")
        self.assertEqual(len(report["paper_orders"]), 0)
        self.assertEqual(report["allowed_candidates"], 0)
        self.assertIn("LIVE_NOT_IMPLEMENTED", report["candidates"][0]["block_reasons"])


if __name__ == "__main__":
    unittest.main()
