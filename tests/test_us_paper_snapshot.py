from __future__ import annotations

from datetime import date
import unittest

from python.us.update_us_stock_paper_snapshot import evaluate_paper_snapshot


class USPaperSnapshotTests(unittest.TestCase):
    def test_evaluate_snapshot_marks_position_and_account(self) -> None:
        snapshot_row, positions, warnings = evaluate_paper_snapshot(
            snapshot_date=date(2026, 5, 14),
            account_row={
                "account_id": "US_PAPER_RULE_V1",
                "initial_cash": 100000.0,
                "cash_balance": 95000.0,
                "reserved_cash": 0.0,
                "realized_pnl": 50.0,
                "market_value": 0.0,
                "equity_value": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "status": "ACTIVE",
            },
            position_rows=[
                {
                    "account_id": "US_PAPER_RULE_V1",
                    "symbol": "NVDA",
                    "qty": 10.0,
                    "avg_price": 100.0,
                    "cost_amount": 1000.0,
                    "status": "OPEN",
                }
            ],
            snapshot_rows=[{"account_id": "US_PAPER_RULE_V1", "snapshot_date": date(2026, 5, 13), "equity_value": 95900.0}],
            fill_rows=[{"trade_date": date(2026, 5, 8)}],
            price_rows=[
                {"ticker": "NVDA", "trade_date": date(2026, 5, 14), "close_price": 120.0, "adj_close_price": 120.0},
                {"ticker": "SPY", "trade_date": date(2026, 5, 8), "close_price": 500.0, "adj_close_price": 500.0},
                {"ticker": "SPY", "trade_date": date(2026, 5, 14), "close_price": 510.0, "adj_close_price": 510.0},
                {"ticker": "QQQ", "trade_date": date(2026, 5, 8), "close_price": 400.0, "adj_close_price": 400.0},
                {"ticker": "QQQ", "trade_date": date(2026, 5, 14), "close_price": 412.0, "adj_close_price": 412.0},
            ],
            use_previous_close=False,
        )
        self.assertEqual(warnings, [])
        self.assertAlmostEqual(float(positions[0]["market_value"]), 1200.0, places=6)
        self.assertAlmostEqual(float(positions[0]["unrealized_pnl"]), 200.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["market_value"]), 1200.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["equity_value"]), 96200.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["total_pnl"]), 250.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["daily_return_pct"]), (96200.0 - 95900.0) / 95900.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["spy_return_pct"]), 0.02, places=6)
        self.assertAlmostEqual(float(snapshot_row["qqq_return_pct"]), 0.03, places=6)

    def test_evaluate_snapshot_uses_previous_close_when_enabled(self) -> None:
        snapshot_row, positions, warnings = evaluate_paper_snapshot(
            snapshot_date=date(2026, 5, 14),
            account_row={
                "account_id": "US_PAPER_RULE_V1",
                "initial_cash": 100000.0,
                "cash_balance": 100000.0,
                "reserved_cash": 0.0,
                "realized_pnl": 0.0,
                "market_value": 0.0,
                "equity_value": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "status": "ACTIVE",
            },
            position_rows=[
                {
                    "account_id": "US_PAPER_RULE_V1",
                    "symbol": "AAPL",
                    "qty": 5.0,
                    "avg_price": 100.0,
                    "cost_amount": 500.0,
                    "status": "OPEN",
                }
            ],
            snapshot_rows=[],
            fill_rows=[],
            price_rows=[
                {"ticker": "AAPL", "trade_date": date(2026, 5, 13), "close_price": 110.0, "adj_close_price": 110.0},
            ],
            use_previous_close=True,
        )
        self.assertEqual(warnings[:2], ["SPY benchmark return is unavailable", "QQQ benchmark return is unavailable"])
        self.assertAlmostEqual(float(positions[0]["last_price"]), 110.0, places=6)
        self.assertAlmostEqual(float(snapshot_row["market_value"]), 550.0, places=6)


if __name__ == "__main__":
    unittest.main()
