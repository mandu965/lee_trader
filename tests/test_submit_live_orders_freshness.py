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
import sync_auxiliary_payloads
import trading_diagnostics


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
                with patch.object(
                    submit_live_orders,
                    "build_market_context",
                    return_value={"market_status": "OPEN", "market_status_ko": "\uc7a5\uc911", "is_trading_day": True},
                ):
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

    def test_calendar_observed_holiday_is_market_holiday(self) -> None:
        context = trading_diagnostics.build_market_context(
            datetime(2026, 3, 2, 10, 0, tzinfo=trading_diagnostics.KST)
        )

        self.assertEqual(context["market_status"], "HOLIDAY")
        self.assertFalse(context["is_trading_day"])

    def test_regular_trading_session_is_open(self) -> None:
        context = trading_diagnostics.build_market_context(
            datetime(2026, 6, 15, 10, 0, tzinfo=trading_diagnostics.KST)
        )

        self.assertEqual(context["market_status"], "OPEN")
        self.assertTrue(context["is_trading_day"])

    def test_market_date_guard_blocks_preview_before_broker_context(self) -> None:
        intents_payload = {
            "run_id": "RUN2",
            "asof_date": "2026-06-12",
            "gate_status": "PILOT",
            "intents": [
                {
                    "intent_id": "20260612:BUY:005930",
                    "code": "005930",
                    "name": "?쇱꽦?꾩옄",
                    "intent_type": "BUY",
                    "target_weight": 0.1,
                    "executable": True,
                    "policy_status": "WATCH",
                }
            ],
        }
        ranking = pd.DataFrame([{"code": "005930", "date": "2026-06-12", "name": "?쇱꽦?꾩옄", "close": 1000, "buy_rank": 1}])
        market_context = {
            "market_status": "HOLIDAY",
            "market_status_ko": "\ud734\uc7a5",
            "is_trading_day": False,
            "kst_date": "2026-06-14",
            "kst_time": "18:30:00",
            "timezone": "Asia/Seoul",
        }

        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-12")
            with patch.object(submit_live_orders, "MARKET_STATUS_CSV", market_csv):
                with patch.object(submit_live_orders, "build_market_context", return_value=market_context):
                    with patch.object(submit_live_orders.KISClient, "from_env", side_effect=AssertionError("broker must not be called")):
                        payload = submit_live_orders.build_order_requests(
                            intents_payload=intents_payload,
                            holdings=pd.DataFrame(),
                            ranking=ranking,
                            ord_dvsn="01",
                        )

        self.assertTrue(payload["market_order_guard"]["active"])
        self.assertEqual(payload["items"][0]["blocked_reason"], "market_date_guard")
        self.assertFalse(payload["items"][0]["executable_now"])
        self.assertEqual(payload["summary"]["submit_allowed_count"], 0)

    def test_csv_payload_reader_preserves_six_digit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdings.csv"
            path.write_text("code,name,qty\n083650,비에이치아이,1\n10950,S-Oil,2\n", encoding="utf-8-sig")

            rows = sync_auxiliary_payloads.read_csv_rows(path)

        self.assertEqual(rows[0]["code"], "083650")
        self.assertEqual(rows[1]["code"], "010950")


if __name__ == "__main__":
    unittest.main()
