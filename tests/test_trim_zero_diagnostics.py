from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import build_trim_zero_diagnostics


class TrimZeroDiagnosticsTests(unittest.TestCase):
    def test_target_weight_equal_to_current_weight_is_diagnosed_as_no_sell_needed(self) -> None:
        order_preview = {
            "run_id": "RUN1",
            "generated_at": "2026-06-05 18:29:25",
            "asof_date": "2026-06-05",
            "items": [
                {
                    "code": "071050",
                    "name": "한국금융지주",
                    "intent_type": "TRIM",
                    "blocked_reason": "trim_ratio_zero",
                    "final_request_qty": 0,
                    "target_weight": 0.0723057315125391,
                    "executable_now": False,
                }
            ],
        }
        trade_intents = {
            "run_id": "RUN1",
            "generated_at": "2026-06-05 18:29:16",
            "asof_date": "2026-06-05",
            "intents": [
                {
                    "code": "071050",
                    "name": "한국금융지주",
                    "intent_type": "TRIM",
                    "target_weight": 0.0723057315125391,
                    "reason": "out of top10 but within hold range top20; live grade C",
                    "ranking_rank": 11,
                }
            ],
        }
        holdings = pd.DataFrame(
            [
                {
                    "code": "071050",
                    "name": "한국금융지주",
                    "qty": 1,
                    "current_price": 246500,
                    "eval_amount": 246500,
                    "weight": 0.0723057315125391,
                }
            ]
        )

        payload = build_trim_zero_diagnostics.build_diagnostics(
            order_preview=order_preview,
            trade_intents=trade_intents,
            holdings=holdings,
            generated_at=datetime(2026, 6, 6, 10, 0, 0),
        )

        self.assertEqual(payload["summary"]["trim_ratio_zero_count"], 1)
        self.assertEqual(payload["summary"]["executable_trim_count"], 0)
        row = payload["items"][0]
        self.assertEqual(row["diagnosis_code"], "target_weight_at_or_above_current_weight")
        self.assertEqual(row["expected_sell_qty_from_weights"], 0)
        self.assertAlmostEqual(row["trim_ratio"], 0.0)

    def test_positive_trim_ratio_is_diagnosed_as_executable_quantity(self) -> None:
        order_preview = {
            "run_id": "RUN2",
            "asof_date": "2026-06-05",
            "items": [
                {
                    "code": "005930",
                    "name": "삼성전자",
                    "intent_type": "TRIM",
                    "final_request_qty": 2,
                    "target_weight": 0.05,
                    "executable_now": True,
                }
            ],
        }
        trade_intents = {"run_id": "RUN2", "intents": []}
        holdings = pd.DataFrame(
            [
                {
                    "code": "005930",
                    "name": "삼성전자",
                    "qty": 10,
                    "eval_amount": 1000000,
                    "weight": 0.10,
                }
            ]
        )

        payload = build_trim_zero_diagnostics.build_diagnostics(
            order_preview=order_preview,
            trade_intents=trade_intents,
            holdings=holdings,
            generated_at=datetime(2026, 6, 6, 10, 0, 0),
        )

        row = payload["items"][0]
        self.assertEqual(row["diagnosis_code"], "trim_has_executable_quantity")
        self.assertEqual(row["expected_sell_qty_from_weights"], 5)
        self.assertAlmostEqual(row["trim_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
