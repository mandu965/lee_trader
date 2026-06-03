from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.us_micro_order_sync import insert_micro_order_fills, update_micro_order_from_broker_status


class USMicroOrderSyncTests(unittest.TestCase):
    def _micro_order(self) -> dict[str, object]:
        return {
            "micro_order_id": "USMO_X",
            "approval_id": "USAPP_X",
            "policy_id": "US_LIVE_RULE_V1",
            "account_id": "US_LIVE_TEST",
            "trade_date": "2026-05-15",
            "symbol": "NVDA",
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 900.0,
            "order_qty": 1.0,
            "order_amount_usd": 50.0,
            "execution_mode": "MOCK",
            "broker_name": "MOCK_BROKER",
            "broker_order_id": "MOCK_NVDA_20260515231000",
            "request_status": "ACCEPTED",
        }

    def test_update_micro_order_from_broker_status_dry_run_maps_partial_fill(self) -> None:
        result = update_micro_order_from_broker_status(
            self._micro_order(),
            {"status": "partially_filled", "filled_qty": 0.4, "filled_price": 900.0},
            dry_run=True,
        )
        self.assertEqual(result["mapped_status"], "ORDER_PARTIALLY_FILLED")
        self.assertEqual(result["micro_order"]["remaining_qty"], 0.6)

    @patch("utils.us_micro_order_sync.fetch_us_micro_order_fill_rows")
    def test_insert_micro_order_fills_dry_run_deduplicates(self, fetch_mock) -> None:
        fetch_mock.return_value = [{"micro_fill_id": "USFILL_BROKER_FILL_1"}]
        result = insert_micro_order_fills(
            self._micro_order(),
            [
                {
                    "broker_fill_id": "BROKER_FILL_1",
                    "filled_qty": 0.4,
                    "filled_price": 900.0,
                    "filled_amount_usd": 360.0,
                    "commission_usd": 0,
                    "fee_usd": 0,
                    "fill_time": None,
                    "fill_date": None,
                    "liquidity_flag": None,
                    "raw_fill_payload": {},
                }
            ],
            dry_run=True,
        )
        self.assertEqual(result["inserted_count"], 0)


if __name__ == "__main__":
    unittest.main()
