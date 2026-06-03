from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils.us_micro_live_safety import resolve_micro_execution_mode, validate_us_micro_sandbox_config
from utils.us_sandbox_order_client import UsSandboxOrderClient


class USSandboxOrderClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = UsSandboxOrderClient()
        self.request = {
            "symbol": "NVDA",
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 900.0,
            "order_qty": 1.0,
            "order_amount_usd": 50.0,
        }

    @patch.dict(os.environ, {}, clear=False)
    def test_submit_order_fails_when_not_configured(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "SANDBOX_NOT_CONFIGURED")

    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_SANDBOX": "true",
            "US_SANDBOX_ORDER_ENABLED": "true",
            "US_SANDBOX_BROKER_NAME": "PAPER",
            "US_SANDBOX_BASE_URL": "https://sandbox.example.test",
            "US_SANDBOX_API_KEY": "key",
            "US_SANDBOX_API_SECRET": "secret",
        },
        clear=False,
    )
    def test_submit_order_accepts_with_placeholder_configuration(self) -> None:
        result = self.client.submit_order(dict(self.request))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertTrue(str(result["broker_order_id"]).startswith("SANDBOX_NVDA_"))

    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_SANDBOX": "true",
            "US_SANDBOX_ORDER_ENABLED": "true",
            "US_SANDBOX_BROKER_NAME": "PAPER",
            "US_SANDBOX_BASE_URL": "https://sandbox.example.test",
            "US_SANDBOX_API_KEY": "key",
            "US_SANDBOX_API_SECRET": "secret",
            "US_SANDBOX_MOCK_FILLED_QTY": "0.5",
            "US_SANDBOX_MOCK_FILLED_PRICE": "200",
        },
        clear=False,
    )
    def test_get_order_fills_returns_placeholder_fill(self) -> None:
        fills = self.client.get_order_fills("SANDBOX_NVDA_20260515231000")
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["filled_qty"], 0.5)
        self.assertEqual(fills[0]["filled_price"], 200.0)

    def test_resolve_micro_execution_mode_blocks_live(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_micro_execution_mode("LIVE")

    @patch.dict(os.environ, {}, clear=False)
    def test_validate_sandbox_config_safe_default_warning(self) -> None:
        result = validate_us_micro_sandbox_config()
        self.assertEqual(result["result"], "WARNING")
        self.assertIn("Sandbox broker is not configured.", result["warnings"])

    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_LIVE": "true",
            "US_SANDBOX_MAX_ORDER_AMOUNT_USD": "55",
            "US_SANDBOX_ALLOW_MARKET_ORDER": "true",
        },
        clear=False,
    )
    def test_validate_sandbox_config_detects_unsafe_values(self) -> None:
        result = validate_us_micro_sandbox_config()
        self.assertEqual(result["result"], "ERROR")
        self.assertIn("US_MICRO_ALLOW_LIVE must remain false.", result["errors"])
        self.assertIn("US_SANDBOX_MAX_ORDER_AMOUNT_USD must be <= 50.", result["errors"])
        self.assertIn("US_SANDBOX_ALLOW_MARKET_ORDER must remain false.", result["errors"])


if __name__ == "__main__":
    unittest.main()
