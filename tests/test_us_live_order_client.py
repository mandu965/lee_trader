from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils.us_live_order_client import UsLiveOrderClient


class USLiveOrderClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = UsLiveOrderClient()
        self.request = {
            "symbol": "NVDA",
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 900.0,
            "order_qty": 1.0,
            "order_amount_usd": 50.0,
        }

    @patch.dict(os.environ, {}, clear=False)
    def test_submit_order_fails_when_live_gate_closed(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "LIVE_CLIENT_NOT_ENABLED")

    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_LIVE": "true",
            "US_MICRO_REAL_ORDER_BLOCKED": "false",
            "US_LIVE_TRADING_ENABLED": "true",
            "US_LIVE_ORDER_ENABLED": "true",
            "US_LIVE_BUY_ENABLED": "true",
            "US_LIVE_BROKER": "ALPACA",
        },
        clear=False,
    )
    def test_submit_order_fails_with_placeholder_adapter(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "LIVE_CLIENT_NOT_IMPLEMENTED")


if __name__ == "__main__":
    unittest.main()
