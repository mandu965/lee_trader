from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import submit_live_orders


class SubmitLiveOrdersFreshnessTests(unittest.TestCase):
    def _market_status_csv(self, directory: Path, date_text: str) -> Path:
        path = directory / "market_status.csv"
        path.write_text(f"date,market_up\n{date_text},true\n", encoding="utf-8")
        return path

    def test_signal_freshness_passes_when_asof_matches_latest_available_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-04")
            ranking = pd.DataFrame([{"code": "005930", "date": "2026-06-04", "close": 1000}])

            result = submit_live_orders.build_signal_freshness(
                intents_payload={"asof_date": "2026-06-04"},
                ranking=ranking,
                market_status_csv=market_csv,
                now=datetime(2026, 6, 5, 9, 30),
            )

        self.assertTrue(result["is_current"])
        self.assertFalse(result["block_buy"])
        self.assertEqual(result["reason"], "signal_data_current")

    def test_signal_freshness_blocks_buy_when_asof_is_older_than_latest_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-04")
            ranking = pd.DataFrame([{"code": "005930", "date": "2026-06-04", "close": 1000}])

            result = submit_live_orders.build_signal_freshness(
                intents_payload={"asof_date": "2026-06-02"},
                ranking=ranking,
                market_status_csv=market_csv,
                now=datetime(2026, 6, 5, 9, 30),
            )

        self.assertFalse(result["is_current"])
        self.assertTrue(result["block_buy"])
        self.assertEqual(result["reason"], "signal_data_stale")
        self.assertEqual(result["stale_days"], 2)

    def test_stale_buy_intent_is_blocked_in_preview_without_broker_context(self) -> None:
        intents_payload = {
            "run_id": "RUN1",
            "asof_date": "2026-06-02",
            "gate_status": "PILOT",
            "intents": [
                {
                    "intent_id": "20260602:BUY:005930",
                    "code": "005930",
                    "name": "삼성전자",
                    "intent_type": "BUY",
                    "target_weight": 0.1,
                    "executable": True,
                    "policy_status": "WATCH",
                }
            ],
        }
        ranking = pd.DataFrame([{"code": "005930", "date": "2026-06-04", "name": "삼성전자", "close": 1000, "buy_rank": 1}])

        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-04")
            with patch.object(submit_live_orders, "MARKET_STATUS_CSV", market_csv):
                with patch.dict(os.environ, {"AUTO_TRADE_BLOCK_BUY_ON_STALE_SIGNAL": "1"}, clear=False):
                    payload = submit_live_orders.build_order_requests(
                        intents_payload=intents_payload,
                        holdings=pd.DataFrame(),
                        ranking=ranking,
                        ord_dvsn="01",
                    )

        self.assertEqual(payload["signal_freshness"]["reason"], "signal_data_stale")
        self.assertEqual(payload["items"][0]["blocked_reason"], "signal_data_stale")
        self.assertFalse(payload["items"][0]["executable_now"])


if __name__ == "__main__":
    unittest.main()
