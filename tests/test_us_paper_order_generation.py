from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from python.us.generate_us_stock_paper_orders import build_paper_orders
from python.us.paper_rebalance import PaperStrategyPolicy


class USPaperOrderGenerationTests(unittest.TestCase):
    def make_policy(self) -> PaperStrategyPolicy:
        return PaperStrategyPolicy(
            selection_rule="TOP20",
            buy_grades=("STRONG_BUY", "BUY"),
            sell_grades=("HOLD", "EXCLUDE"),
            max_rank_no=20,
            max_positions=20,
            max_position_weight=0.10,
            max_sector_weight=0.30,
            min_cash_weight=0.05,
            max_daily_new_buys=5,
            allow_fractional_shares=True,
            min_order_amount=100.0,
            source="recommend.us_stock_rank_daily",
            sell_first=True,
            allow_rebuy_same_day=False,
            min_rebalance_amount=100.0,
            min_weight_diff=0.02,
            full_sell_on_rank_exit=True,
            full_sell_on_grade_downgrade=True,
            rebalance_frequency="DAILY",
        )

    @patch("python.us.generate_us_stock_paper_orders.order_price_lookup")
    def test_generates_buy_order_for_rank_candidate(self, price_lookup_mock) -> None:
        price_lookup_mock.return_value = {"NVDA": 100.0}
        orders, counts = build_paper_orders(
            trade_date=date(2026, 5, 12),
            account_row={"account_id": "US_PAPER_RULE_V1", "equity_value": 100000.0, "cash_balance": 100000.0, "reserved_cash": 0.0},
            rank_rows=[{"symbol": "NVDA", "rank_no": 3, "recommend_grade": "BUY", "total_score": 76.5, "data_status": "OK", "exclude_reason": None, "momentum_score": 20.0, "relative_strength_score": 15.0, "sector": "Technology"}],
            position_rows=[],
            existing_order_rows=[],
            policy=self.make_policy(),
            side_option="BUY",
            replace_existing=False,
        )
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["status"], "CREATED")
        self.assertEqual(orders[0]["side"], "BUY")
        self.assertGreater(float(orders[0]["order_qty"]), 0)
        self.assertEqual(counts["orders_created"], 1)

    @patch("python.us.generate_us_stock_paper_orders.order_price_lookup")
    def test_rejects_buy_when_cash_below_min_amount(self, price_lookup_mock) -> None:
        price_lookup_mock.return_value = {"NVDA": 100.0}
        policy = self.make_policy()
        orders, counts = build_paper_orders(
            trade_date=date(2026, 5, 12),
            account_row={"account_id": "US_PAPER_RULE_V1", "equity_value": 100000.0, "cash_balance": 100.0, "reserved_cash": 0.0},
            rank_rows=[{"symbol": "NVDA", "rank_no": 3, "recommend_grade": "BUY", "total_score": 76.5, "data_status": "OK", "exclude_reason": None, "momentum_score": 20.0, "relative_strength_score": 15.0, "sector": "Technology"}],
            position_rows=[],
            existing_order_rows=[],
            policy=policy,
            side_option="BUY",
            replace_existing=False,
        )
        self.assertEqual(orders[0]["status"], "REJECTED")
        self.assertEqual(orders[0]["reject_reason"], "insufficient_cash")
        self.assertEqual(counts["orders_rejected"], 1)

    @patch("python.us.generate_us_stock_paper_orders.order_price_lookup")
    def test_generates_sell_when_rank_missing(self, price_lookup_mock) -> None:
        price_lookup_mock.return_value = {"AAPL": 150.0}
        orders, counts = build_paper_orders(
            trade_date=date(2026, 5, 12),
            account_row={"account_id": "US_PAPER_RULE_V1", "equity_value": 100000.0, "cash_balance": 100000.0, "reserved_cash": 0.0},
            rank_rows=[],
            position_rows=[{"symbol": "AAPL", "qty": 5.0, "last_price": 149.0, "status": "OPEN"}],
            existing_order_rows=[],
            policy=self.make_policy(),
            side_option="SELL",
            replace_existing=False,
        )
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["status"], "CREATED")
        self.assertEqual(orders[0]["side"], "SELL")
        self.assertEqual(counts["sell_candidates"], 1)

    @patch("python.us.generate_us_stock_paper_orders.order_price_lookup")
    def test_skips_when_existing_order_present(self, price_lookup_mock) -> None:
        price_lookup_mock.return_value = {"NVDA": 100.0}
        orders, counts = build_paper_orders(
            trade_date=date(2026, 5, 12),
            account_row={"account_id": "US_PAPER_RULE_V1", "equity_value": 100000.0, "cash_balance": 100000.0, "reserved_cash": 0.0},
            rank_rows=[{"symbol": "NVDA", "rank_no": 3, "recommend_grade": "BUY", "total_score": 76.5, "data_status": "OK", "exclude_reason": None, "momentum_score": 20.0, "relative_strength_score": 15.0, "sector": "Technology"}],
            position_rows=[],
            existing_order_rows=[{"account_id": "US_PAPER_RULE_V1", "trade_date": date(2026, 5, 12), "symbol": "NVDA", "side": "BUY", "strategy_name": "US_RANK_TOP20", "status": "CREATED"}],
            policy=self.make_policy(),
            side_option="BUY",
            replace_existing=False,
        )
        self.assertEqual(orders, [])
        self.assertEqual(counts["orders_skipped"], 1)

    @patch("python.us.generate_us_stock_paper_orders.order_price_lookup")
    def test_prevents_same_day_rebuy_after_sell_plan(self, price_lookup_mock) -> None:
        price_lookup_mock.return_value = {"NVDA": 100.0}
        orders, _ = build_paper_orders(
            trade_date=date(2026, 5, 12),
            account_row={"account_id": "US_PAPER_RULE_V1", "equity_value": 100000.0, "cash_balance": 100000.0, "reserved_cash": 0.0},
            rank_rows=[{"symbol": "NVDA", "rank_no": 25, "recommend_grade": "BUY", "total_score": 76.5, "data_status": "OK", "exclude_reason": None, "momentum_score": 20.0, "relative_strength_score": 15.0, "sector": "Technology"}],
            position_rows=[{"symbol": "NVDA", "qty": 1.0, "last_price": 99.0, "status": "OPEN"}],
            existing_order_rows=[],
            policy=self.make_policy(),
            side_option="ALL",
            replace_existing=False,
        )
        self.assertTrue(any(row["side"] == "SELL" for row in orders))
        self.assertFalse(any(row["side"] == "BUY" for row in orders))


if __name__ == "__main__":
    unittest.main()
