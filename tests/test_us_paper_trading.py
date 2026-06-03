from __future__ import annotations

from pathlib import Path
import unittest

from python.us.init_us_stock_paper_account import build_account_row
from python.us.us_config import load_us_paper_trading_config


class USPaperTradingTests(unittest.TestCase):
    def test_paper_config_defaults(self) -> None:
        cfg = load_us_paper_trading_config("US_PAPER_RULE_V1")
        self.assertEqual(cfg.account_id, "US_PAPER_RULE_V1")
        self.assertEqual(cfg.base_currency, "USD")
        self.assertTrue(cfg.real_order_blocked)
        self.assertTrue(str(cfg.config_path).endswith(str(Path("config") / "us_stock_paper_trading.yaml")))

    def test_build_account_row_sets_cash_balance(self) -> None:
        row = build_account_row(account_id="US_PAPER_RULE_V1", initial_cash=250000)
        self.assertEqual(row["account_id"], "US_PAPER_RULE_V1")
        self.assertEqual(row["initial_cash"], 250000.0)
        self.assertEqual(row["cash_balance"], 250000.0)
        self.assertEqual(row["equity_value"], 250000.0)
        self.assertEqual(row["status"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()
