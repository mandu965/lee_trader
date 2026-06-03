from __future__ import annotations

from datetime import date
import unittest

from python.us.simulate_us_stock_paper_fills import FillSimulationConfig, simulate_paper_fills


class USPaperFillSimulationTests(unittest.TestCase):
    def make_cfg(self) -> FillSimulationConfig:
        return FillSimulationConfig(
            commission_per_trade=1.0,
            slippage_bps=5.0,
            real_order_blocked=True,
            log_level="INFO",
        )

    def test_buy_fill_reduces_cash_and_opens_position(self) -> None:
        decisions, account_after, positions_after, counts = simulate_paper_fills(
            as_of_date=date(2026, 5, 13),
            account_row={"account_id": "US_PAPER_RULE_V1", "cash_balance": 100000.0, "reserved_cash": 0.0, "market_value": 0.0, "equity_value": 100000.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0, "status": "ACTIVE"},
            order_rows=[{"paper_order_id": "USPO_1", "account_id": "US_PAPER_RULE_V1", "trade_date": date(2026, 5, 12), "symbol": "NVDA", "side": "BUY", "order_type": "MARKET", "order_qty": 10.0, "status": "CREATED"}],
            position_rows=[],
            price_rows=[
                {"ticker": "NVDA", "trade_date": date(2026, 5, 12), "close_price": 95.0, "adj_close_price": 95.0},
                {"ticker": "NVDA", "trade_date": date(2026, 5, 13), "close_price": 100.0, "adj_close_price": 100.0},
            ],
            cfg=self.make_cfg(),
            side_option="ALL",
            existing_fill_rows=[],
        )
        self.assertEqual(counts["filled_count"], 1)
        self.assertEqual(decisions[0]["status"], "FILLED")
        self.assertAlmostEqual(float(decisions[0]["filled_price"]), 100.05, places=6)
        self.assertAlmostEqual(float(account_after["cash_balance"]), 100000.0 - 1000.5 - 1.0, places=6)
        self.assertEqual(len([row for row in positions_after if str(row.get("status")) == "OPEN"]), 1)
        self.assertAlmostEqual(float(positions_after[0]["qty"]), 10.0, places=6)

    def test_sell_fill_realizes_pnl_and_closes_position(self) -> None:
        decisions, account_after, positions_after, counts = simulate_paper_fills(
            as_of_date=date(2026, 5, 13),
            account_row={"account_id": "US_PAPER_RULE_V1", "cash_balance": 1000.0, "reserved_cash": 0.0, "market_value": 1000.0, "equity_value": 2000.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0, "status": "ACTIVE"},
            order_rows=[{"paper_order_id": "USPO_2", "account_id": "US_PAPER_RULE_V1", "trade_date": date(2026, 5, 12), "symbol": "AAPL", "side": "SELL", "order_type": "MARKET", "order_qty": 5.0, "status": "CREATED"}],
            position_rows=[{"account_id": "US_PAPER_RULE_V1", "symbol": "AAPL", "qty": 5.0, "avg_price": 100.0, "cost_amount": 500.0, "last_price": 100.0, "market_value": 500.0, "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0, "realized_pnl": 0.0, "status": "OPEN"}],
            price_rows=[
                {"ticker": "AAPL", "trade_date": date(2026, 5, 12), "close_price": 99.0, "adj_close_price": 99.0},
                {"ticker": "AAPL", "trade_date": date(2026, 5, 13), "close_price": 120.0, "adj_close_price": 120.0},
            ],
            cfg=self.make_cfg(),
            side_option="ALL",
            existing_fill_rows=[],
        )
        self.assertEqual(counts["filled_count"], 1)
        self.assertEqual(decisions[0]["status"], "FILLED")
        self.assertAlmostEqual(float(decisions[0]["filled_price"]), 119.94, places=6)
        self.assertGreater(float(account_after["realized_pnl"]), 0.0)
        closed_positions = [row for row in positions_after if str(row.get("symbol")) == "AAPL"]
        self.assertEqual(len(closed_positions), 1)
        self.assertEqual(str(closed_positions[0]["status"]), "CLOSED")
        self.assertAlmostEqual(float(closed_positions[0]["qty"]), 0.0, places=6)

    def test_buy_rejected_when_cash_insufficient(self) -> None:
        decisions, _account_after, _positions_after, counts = simulate_paper_fills(
            as_of_date=date(2026, 5, 13),
            account_row={"account_id": "US_PAPER_RULE_V1", "cash_balance": 10.0, "reserved_cash": 0.0, "market_value": 0.0, "equity_value": 10.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0, "status": "ACTIVE"},
            order_rows=[{"paper_order_id": "USPO_3", "account_id": "US_PAPER_RULE_V1", "trade_date": date(2026, 5, 12), "symbol": "NVDA", "side": "BUY", "order_type": "MARKET", "order_qty": 1.0, "status": "CREATED"}],
            position_rows=[],
            price_rows=[
                {"ticker": "NVDA", "trade_date": date(2026, 5, 13), "close_price": 100.0, "adj_close_price": 100.0},
            ],
            cfg=self.make_cfg(),
            side_option="ALL",
            existing_fill_rows=[],
        )
        self.assertEqual(counts["rejected_count"], 1)
        self.assertEqual(decisions[0]["status"], "REJECTED")
        self.assertEqual(decisions[0]["reject_reason"], "insufficient_cash_at_fill")

    def test_sell_rejected_when_position_qty_insufficient(self) -> None:
        decisions, _account_after, _positions_after, counts = simulate_paper_fills(
            as_of_date=date(2026, 5, 13),
            account_row={"account_id": "US_PAPER_RULE_V1", "cash_balance": 1000.0, "reserved_cash": 0.0, "market_value": 500.0, "equity_value": 1500.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0, "status": "ACTIVE"},
            order_rows=[{"paper_order_id": "USPO_4", "account_id": "US_PAPER_RULE_V1", "trade_date": date(2026, 5, 12), "symbol": "AAPL", "side": "SELL", "order_type": "MARKET", "order_qty": 10.0, "status": "CREATED"}],
            position_rows=[{"account_id": "US_PAPER_RULE_V1", "symbol": "AAPL", "qty": 5.0, "avg_price": 100.0, "cost_amount": 500.0, "last_price": 100.0, "market_value": 500.0, "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0, "realized_pnl": 0.0, "status": "OPEN"}],
            price_rows=[
                {"ticker": "AAPL", "trade_date": date(2026, 5, 13), "close_price": 120.0, "adj_close_price": 120.0},
            ],
            cfg=self.make_cfg(),
            side_option="ALL",
            existing_fill_rows=[],
        )
        self.assertEqual(counts["rejected_count"], 1)
        self.assertEqual(decisions[0]["status"], "REJECTED")
        self.assertEqual(decisions[0]["reject_reason"], "insufficient_position_qty")


if __name__ == "__main__":
    unittest.main()
