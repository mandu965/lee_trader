from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils.us_micro_reconciliation import (
    compare_fills,
    compare_positions,
    run_micro_reconciliation,
)


class USMicroReconciliationTests(unittest.TestCase):
    def test_compare_positions_detects_critical_mismatch(self) -> None:
        results = compare_positions(
            {"NVDA": {"symbol": "NVDA", "internal_qty": 1.0, "internal_amount_usd": 100.0, "fills": []}},
            [{"symbol": "NVDA", "qty": 0.0, "market_value": 0.0}],
        )
        self.assertEqual(results[0]["recon_status"], "MISMATCH")
        self.assertEqual(results[0]["severity"], "CRITICAL")
        self.assertEqual(results[0]["reason_code"], "position_qty_mismatch")

    def test_compare_fills_detects_amount_mismatch(self) -> None:
        results = compare_fills(
            [
                {
                    "broker_fill_id": "FILL_1",
                    "micro_fill_id": "USFILL_1",
                    "micro_order_id": "USMO_1",
                    "broker_order_id": "BROKER_1",
                    "symbol": "NVDA",
                    "filled_qty": 1.0,
                    "filled_price": 100.0,
                    "filled_amount_usd": 100.0,
                }
            ],
            [
                {
                    "broker_fill_id": "FILL_1",
                    "micro_order_id": "USMO_1",
                    "broker_order_id": "BROKER_1",
                    "symbol": "NVDA",
                    "filled_qty": 1.0,
                    "filled_price": 102.0,
                    "filled_amount_usd": 102.0,
                }
            ],
        )
        self.assertEqual(results[0]["recon_status"], "MISMATCH")
        self.assertEqual(results[0]["reason_code"], "fill_amount_mismatch")
        self.assertEqual(results[0]["severity"], "CRITICAL")

    @patch("utils.us_micro_reconciliation.ensure_us_micro_live_tables")
    @patch("utils.us_micro_reconciliation.fetch_us_micro_order_request_rows")
    @patch("utils.us_micro_reconciliation.fetch_us_micro_order_fill_rows")
    @patch("utils.us_micro_reconciliation.UsMockOrderClient")
    @patch("utils.us_micro_reconciliation.UsMockAccountClient")
    def test_run_micro_reconciliation_dry_run_recommends_kill_switch(
        self,
        account_client_mock,
        order_client_mock,
        fill_rows_mock,
        order_rows_mock,
        ensure_mock,
    ) -> None:
        previous_enabled = os.environ.get("US_MICRO_RECON_ENABLED")
        try:
            os.environ["US_MICRO_RECON_ENABLED"] = "true"
            order_rows_mock.return_value = [
                {
                    "micro_order_id": "USMO_1",
                    "broker_order_id": "BROKER_1",
                    "account_id": "US_LIVE_TEST",
                    "trade_date": "2026-05-16",
                    "symbol": "NVDA",
                    "side": "BUY",
                    "request_status": "ORDER_FILLED",
                    "execution_mode": "MOCK",
                    "broker_name": "MOCK_BROKER",
                }
            ]
            fill_rows_mock.return_value = [
                {
                    "micro_fill_id": "USFILL_1",
                    "micro_order_id": "USMO_1",
                    "broker_order_id": "BROKER_1",
                    "broker_fill_id": "FILL_1",
                    "symbol": "NVDA",
                    "side": "BUY",
                    "filled_qty": 1.0,
                    "filled_price": 100.0,
                    "filled_amount_usd": 100.0,
                }
            ]
            order_client = order_client_mock.return_value
            order_client.get_order_status.return_value = {"success": True, "status": "filled", "filled_qty": 1.0, "filled_price": 100.0}
            order_client.get_order_fills.return_value = [
                {"broker_fill_id": "FILL_1", "filled_qty": 1.0, "filled_price": 100.0, "filled_amount_usd": 100.0, "fill_time": "2026-05-16T13:30:00Z"}
            ]
            account_client = account_client_mock.return_value
            account_client.get_positions.return_value = [{"symbol": "NVDA", "qty": 0.0, "market_value": 0.0}]
            account_client.get_account_snapshot.return_value = {"cash_balance": 900.0, "equity_value": 900.0}
            account_client.get_cash_balance.return_value = {"cash_balance": 900.0}

            report = run_micro_reconciliation(
                account_id="US_LIVE_TEST",
                recon_date="2026-05-16",
                execution_mode="MOCK",
                include_orders=True,
                include_fills=True,
                include_positions=True,
                include_cash=False,
                dry_run=True,
            )
            self.assertTrue(report["summary"]["kill_switch_recommended"])
            self.assertGreater(report["summary"]["critical_count"], 0)
            self.assertTrue(report["dry_run"])
        finally:
            if previous_enabled is None:
                os.environ.pop("US_MICRO_RECON_ENABLED", None)
            else:
                os.environ["US_MICRO_RECON_ENABLED"] = previous_enabled

    def test_run_micro_reconciliation_blocks_live_by_default(self) -> None:
        previous_enabled = os.environ.get("US_MICRO_RECON_ENABLED")
        previous_live = os.environ.get("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY")
        try:
            os.environ["US_MICRO_RECON_ENABLED"] = "true"
            os.environ.pop("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", None)
            with self.assertRaises(RuntimeError):
                run_micro_reconciliation(
                    account_id="US_LIVE_TEST",
                    recon_date="2026-05-16",
                    execution_mode="LIVE",
                    dry_run=True,
                )
        finally:
            if previous_enabled is None:
                os.environ.pop("US_MICRO_RECON_ENABLED", None)
            else:
                os.environ["US_MICRO_RECON_ENABLED"] = previous_enabled
            if previous_live is None:
                os.environ.pop("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", None)
            else:
                os.environ["US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY"] = previous_live


if __name__ == "__main__":
    unittest.main()
