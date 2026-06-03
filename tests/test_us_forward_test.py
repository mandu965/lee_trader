from __future__ import annotations

from datetime import date
import unittest

from python.us.forward_test_us_stock import (
    apply_entry_updates,
    apply_exit_updates,
    build_console_report,
    build_forward_summary_rows,
    next_trade_date,
    target_exit_date,
)


class USForwardTestTests(unittest.TestCase):
    def test_next_trade_date_uses_next_session(self) -> None:
        calendar = [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)]
        self.assertEqual(next_trade_date(date(2026, 5, 11), calendar), date(2026, 5, 12))

    def test_target_exit_date_adds_holding_sessions(self) -> None:
        calendar = [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14), date(2026, 5, 15)]
        self.assertEqual(target_exit_date(date(2026, 5, 12), 2, calendar), date(2026, 5, 14))

    def test_entry_and_exit_status_flow(self) -> None:
        rows = [
            {
                "forward_test_id": "FWD1",
                "trade_date": date(2026, 5, 11),
                "symbol": "AAPL",
                "holding_days": 2,
                "strategy_name": "US_RANK_TOP5",
                "status": "PENDING_ENTRY",
                "data_status": "OK",
                "entry_date": date(2026, 5, 12),
                "entry_price": None,
                "target_exit_date": None,
            }
        ]
        price_lookup = {
            "AAPL": [
                {"trade_date": date(2026, 5, 12), "price": 100.0},
                {"trade_date": date(2026, 5, 13), "price": 101.0},
                {"trade_date": date(2026, 5, 14), "price": 103.0},
            ],
            "SPY": [
                {"trade_date": date(2026, 5, 12), "price": 500.0},
                {"trade_date": date(2026, 5, 13), "price": 501.0},
                {"trade_date": date(2026, 5, 14), "price": 505.0},
            ],
            "QQQ": [
                {"trade_date": date(2026, 5, 12), "price": 400.0},
                {"trade_date": date(2026, 5, 13), "price": 402.0},
                {"trade_date": date(2026, 5, 14), "price": 403.0},
            ],
        }
        calendar = [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14)]
        active = apply_entry_updates(rows, as_of_date=date(2026, 5, 12), price_lookup=price_lookup, market_calendar=calendar)
        self.assertEqual(active[0]["status"], "ACTIVE")
        self.assertEqual(active[0]["entry_price"], 100.0)
        completed = apply_exit_updates(active, as_of_date=date(2026, 5, 14), price_lookup=price_lookup, market_calendar=calendar)
        self.assertEqual(completed[0]["status"], "COMPLETED")
        self.assertAlmostEqual(completed[0]["return_pct"], 0.03, places=6)

    def test_summary_uses_completed_rows_only(self) -> None:
        rows = [
            {"forward_test_id": "FWD1", "trade_date": date(2026, 5, 11), "strategy_name": "US_RANK_TOP5", "holding_days": 5, "status": "COMPLETED", "return_pct": 0.02, "spy_return_pct": 0.01, "qqq_return_pct": 0.015, "excess_return_vs_spy": 0.01, "excess_return_vs_qqq": 0.005, "win_flag": 1, "win_vs_spy_flag": 1, "win_vs_qqq_flag": 1, "symbol": "AAPL"},
            {"forward_test_id": "FWD1", "trade_date": date(2026, 5, 11), "strategy_name": "US_RANK_TOP5", "holding_days": 5, "status": "ACTIVE", "return_pct": None, "symbol": "MSFT"},
        ]
        summary = build_forward_summary_rows(rows, forward_test_id="FWD1")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["completed_count"], 1)
        self.assertAlmostEqual(summary[0]["avg_return_pct"], 0.02, places=6)

    def test_console_report_mentions_no_completed_rows(self) -> None:
        rendered = build_console_report(
            forward_test_id="FWD1",
            detail_rows=[{"status": "PENDING_ENTRY"}],
            summary_rows=[{"strategy_name": "US_RANK_TOP5", "holding_days": 5, "completed_count": 0}],
        )
        self.assertIn("No completed forward-test rows yet.", rendered)


if __name__ == "__main__":
    unittest.main()
