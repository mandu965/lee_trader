from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from python.us.validate_us_stock_paper_trading import collect_paper_validation


class USPaperValidationTests(unittest.TestCase):
    @patch("python.us.validate_us_stock_paper_trading.fetch_us_paper_account_rows")
    @patch("python.us.validate_us_stock_paper_trading.fetch_us_paper_position_rows")
    @patch("python.us.validate_us_stock_paper_trading.fetch_us_paper_order_rows")
    @patch("python.us.validate_us_stock_paper_trading.fetch_us_paper_fill_rows")
    @patch("python.us.validate_us_stock_paper_trading.fetch_us_paper_account_snapshot_rows")
    @patch("python.us.validate_us_stock_paper_trading.fetch_rank_component_rows_between")
    @patch("python.us.validate_us_stock_paper_trading.validate_paper_account_integrity")
    def test_collect_validation_reports_warning_and_error_counts(
        self,
        integrity_mock,
        rank_mock,
        snapshot_mock,
        fill_mock,
        order_mock,
        position_mock,
        account_mock,
    ) -> None:
        account_mock.return_value = [{
            "account_id": "US_PAPER_RULE_V1",
            "status": "ACTIVE",
            "cash_balance": 1000.0,
            "market_value": 500.0,
            "equity_value": 1500.0,
            "realized_pnl": 10.0,
            "unrealized_pnl": 5.0,
            "total_pnl": 15.0,
        }]
        position_mock.return_value = [{
            "account_id": "US_PAPER_RULE_V1",
            "symbol": "NVDA",
            "qty": 1.0,
            "last_price": 120.0,
            "market_value": 120.0,
            "cost_amount": 100.0,
            "unrealized_pnl": 20.0,
            "unrealized_pnl_pct": 0.2,
            "status": "OPEN",
        }]
        order_mock.return_value = [{
            "paper_order_id": "USPO_1",
            "trade_date": date(2026, 5, 10),
            "status": "REJECTED",
            "reject_reason": "",
        }]
        fill_mock.return_value = []
        snapshot_mock.return_value = [{
            "account_id": "US_PAPER_RULE_V1",
            "snapshot_date": date(2026, 5, 13),
            "cash_balance": 1000.0,
            "market_value": 500.0,
            "equity_value": 1500.0,
            "daily_return_pct": None,
        }]
        rank_mock.return_value = [{
            "trade_date": date(2026, 5, 13),
            "symbol": "NVDA",
            "sector": "Technology",
        }]
        integrity_mock.return_value = ["equity_value_mismatch"]

        report = collect_paper_validation("US_PAPER_RULE_V1", snapshot_date=date(2026, 5, 13))
        self.assertGreaterEqual(report["warnings"], 1)
        self.assertGreaterEqual(report["errors"], 1)
        codes = {item.code for item in report["issues"]}
        self.assertIn("equity_value_mismatch", codes)
        self.assertIn("rejected_order_missing_reason", codes)


if __name__ == "__main__":
    unittest.main()
