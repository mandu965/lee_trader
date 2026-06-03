from __future__ import annotations

from datetime import date
import unittest

from python.us.backtest_us_stock_rank_strategy import (
    StrategySpec,
    build_backtest_id,
    build_summary_row,
    resolve_forward_window,
    resolve_strategy_specs,
    select_strategy_rows,
)


class USRankBacktestTests(unittest.TestCase):
    def test_build_backtest_id_uses_source_and_holding_days(self) -> None:
        value = build_backtest_id(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 5, 12),
            holding_days=[5, 20, 60],
            source="rule_v1",
        )
        self.assertEqual(value, "US_RANK_RULE_V1_20260101_20260512_HD5_20_60")

    def test_resolve_forward_window_uses_next_trading_day_entry(self) -> None:
        rows = [
            {"trade_date": date(2026, 5, 12), "price": 100.0},
            {"trade_date": date(2026, 5, 13), "price": 101.0},
            {"trade_date": date(2026, 5, 14), "price": 103.0},
            {"trade_date": date(2026, 5, 15), "price": 106.0},
        ]
        window = resolve_forward_window(rows, trade_date=date(2026, 5, 12), holding_days=2)
        self.assertEqual(window.entry_date, date(2026, 5, 13))
        self.assertEqual(window.exit_date, date(2026, 5, 15))
        self.assertEqual(window.data_status, "OK")

    def test_resolve_forward_window_flags_insufficient_forward_data(self) -> None:
        rows = [
            {"trade_date": date(2026, 5, 13), "price": 101.0},
            {"trade_date": date(2026, 5, 14), "price": 103.0},
        ]
        window = resolve_forward_window(rows, trade_date=date(2026, 5, 12), holding_days=5)
        self.assertEqual(window.data_status, "NOT_ENOUGH_FORWARD_DATA")

    def test_select_strategy_rows_filters_exclude(self) -> None:
        spec = StrategySpec("US_RANK_TOP5", "rank_no <= 5", lambda row: int(row["rank_no"]) <= 5 and str(row["recommend_grade"]) != "EXCLUDE")
        rows = [
            {"symbol": "AAA", "rank_no": 1, "recommend_grade": "BUY"},
            {"symbol": "BBB", "rank_no": 2, "recommend_grade": "EXCLUDE"},
            {"symbol": "CCC", "rank_no": 6, "recommend_grade": "BUY"},
        ]
        selected = select_strategy_rows(rows, spec)
        self.assertEqual([row["symbol"] for row in selected], ["AAA"])

    def test_resolve_strategy_specs_accepts_alias(self) -> None:
        specs = resolve_strategy_specs(custom_top_n=20, strategy_filter="BUY_OR_BETTER")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].strategy_name, "US_RANK_BUY_OR_BETTER")

    def test_build_summary_row_aggregates_returns(self) -> None:
        spec = StrategySpec("US_RANK_TOP20", "rank_no <= 20", lambda row: True)
        row = build_summary_row(
            backtest_id="BT1",
            trade_date=date(2026, 5, 12),
            holding_days=5,
            spec=spec,
            rows=[
                {"symbol": "AAA", "return_pct": 0.10, "spy_return_pct": 0.02, "qqq_return_pct": 0.03, "universe_avg_return_pct": 0.04, "excess_return_vs_spy": 0.08, "excess_return_vs_qqq": 0.07, "excess_return_vs_universe": 0.06, "win_flag": 1, "win_vs_spy_flag": 1, "win_vs_qqq_flag": 1, "win_vs_universe_flag": 1, "data_status": "OK"},
                {"symbol": "BBB", "return_pct": -0.05, "spy_return_pct": 0.02, "qqq_return_pct": 0.03, "universe_avg_return_pct": 0.04, "excess_return_vs_spy": -0.07, "excess_return_vs_qqq": -0.08, "excess_return_vs_universe": -0.09, "win_flag": 0, "win_vs_spy_flag": 0, "win_vs_qqq_flag": 0, "win_vs_universe_flag": 0, "data_status": "OK"},
            ],
        )
        self.assertEqual(row["selected_count"], 2)
        self.assertAlmostEqual(float(row["avg_return_pct"]), 0.025, places=6)
        self.assertAlmostEqual(float(row["win_rate"]), 0.5, places=6)
        self.assertEqual(row["best_symbol"], "AAA")
        self.assertEqual(row["worst_symbol"], "BBB")


if __name__ == "__main__":
    unittest.main()
