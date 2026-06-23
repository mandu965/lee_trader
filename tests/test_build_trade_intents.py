from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import build_trade_intents as intents


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        max_position_weight=0.24,
        pilot_limited_position_cap=0.10,
    )


def _ai_context() -> dict[str, dict[str, object]]:
    return {
        "316140": {
            "selected_for_ai_top5": True,
            "entry_quality_status": "WATCH",
            "ai_filtered_rank": 1,
            "original_final_rank": 1,
            "original_final_score": 72.8,
        }
    }


class BuildTradeIntentsTests(unittest.TestCase):
    def test_ai_top5_does_not_revive_ineligible_cooldown_candidate(self) -> None:
        strategy = SimpleNamespace(
            buy_ready_queue=pd.DataFrame(
                [
                    {
                        "code": "316140",
                        "name": "우리금융지주",
                        "entry_eligible": False,
                        "entry_note": "re-entry cooldown 10d remaining",
                        "target_weight": pd.NA,
                    }
                ]
            ),
            held_codes=set(),
        )

        rows = intents.build_ai_buy_intents(
            args=_args(),
            strategy=strategy,
            asof_date=pd.Timestamp("2026-06-22"),
            run_id="TEST",
            gate_status="PILOT",
            ranking_context={"316140": {"name": "우리금융지주"}},
            ai_filtered_context=_ai_context(),
            existing_codes=set(),
        )

        self.assertEqual(rows, [])

    def test_ai_top5_keeps_eligible_candidate(self) -> None:
        strategy = SimpleNamespace(
            buy_ready_queue=pd.DataFrame(
                [
                    {
                        "code": "316140",
                        "name": "우리금융지주",
                        "entry_eligible": True,
                        "entry_note": "meets entry criteria",
                        "target_weight": 0.08,
                    }
                ]
            ),
            held_codes=set(),
        )

        rows = intents.build_ai_buy_intents(
            args=_args(),
            strategy=strategy,
            asof_date=pd.Timestamp("2026-06-22"),
            run_id="TEST",
            gate_status="PILOT",
            ranking_context={"316140": {"name": "우리금융지주"}},
            ai_filtered_context=_ai_context(),
            existing_codes=set(),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["code"], "316140")
        self.assertTrue(rows[0]["executable"])
        self.assertEqual(rows[0]["target_weight"], 0.08)


if __name__ == "__main__":
    unittest.main()
