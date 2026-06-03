from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils.us_mock_order_client import UsMockOrderClient


class USMockOrderClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = UsMockOrderClient()
        self.request = {
            "symbol": "NVDA",
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 900.0,
            "order_qty": 1.0,
            "order_amount_usd": 50.0,
        }

    @patch.dict(os.environ, {}, clear=False)
    def test_submit_order_accepts_by_default(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertTrue(str(result["broker_order_id"]).startswith("MOCK_NVDA_"))

    @patch.dict(os.environ, {"US_MICRO_MOCK_FORCE_REJECT": "true"}, clear=False)
    def test_submit_order_supports_forced_reject(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["error_code"], "MOCK_REJECTED")

    @patch.dict(os.environ, {"US_MICRO_MOCK_FORCE_FAIL": "true"}, clear=False)
    def test_submit_order_supports_forced_fail(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "MOCK_FAILED")

    @patch.dict(os.environ, {}, clear=False)
    def test_submit_order_rejects_invalid_amount(self) -> None:
        bad = dict(self.request)
        bad["order_amount_usd"] = 0
        result = self.client.submit_order(bad)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("non_positive_order_amount_usd", result["reason_codes"])

    @patch.dict(os.environ, {"US_MICRO_MOCK_FILLED_QTY": "1", "US_MICRO_MOCK_FILLED_PRICE": "100"}, clear=False)
    def test_get_order_fills_returns_placeholder_fill(self) -> None:
        fills = self.client.get_order_fills("MOCK_NVDA_20260515231000")
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["filled_qty"], 1.0)
        self.assertEqual(fills[0]["filled_price"], 100.0)


if __name__ == "__main__":
    unittest.main()
