from __future__ import annotations

from datetime import date
import tempfile
import unittest
from pathlib import Path

from python.us.report_us_stock_rank_backtest import (
    _aggregate_strategy_summary,
    _build_data_quality_summary,
    _build_interpretation_lines,
    build_console_report,
    build_markdown_report,
    build_symbol_console_report,
)
from python.us.us_config import USBacktestReportConfig


def make_cfg() -> USBacktestReportConfig:
    return USBacktestReportConfig(
        output_dir=Path("outputs/us_stock_backtest"),
        default_format="console",
        recent_days=10,
        best_worst_limit=10,
        min_test_days_warning=30,
        missing_rate_warning=0.1,
        log_level="INFO",
    )


def make_summary_row(**overrides) -> dict[str, object]:
    row = {
        "backtest_id": "US_RANK_RULE_V1_TEST",
        "trade_date": date(2026, 5, 11),
        "strategy_name": "US_RANK_TOP20",
        "selection_rule": "rank_no <= 20",
        "holding_days": 20,
        "selected_count": 20,
        "avg_return_pct": 0.032,
        "median_return_pct": 0.028,
        "win_rate": 0.58,
        "avg_spy_return_pct": 0.021,
        "avg_qqq_return_pct": 0.024,
        "avg_universe_return_pct": 0.019,
        "avg_excess_return_vs_spy": 0.011,
        "avg_excess_return_vs_qqq": 0.008,
        "avg_excess_return_vs_universe": 0.013,
        "win_rate_vs_spy": 0.55,
        "win_rate_vs_qqq": 0.53,
        "win_rate_vs_universe": 0.57,
        "best_symbol": "NVDA",
        "best_return_pct": 0.12,
        "worst_symbol": "XYZ",
        "worst_return_pct": -0.08,
        "data_status": "OK",
    }
    row.update(overrides)
    return row


def make_result_row(**overrides) -> dict[str, object]:
    row = {
        "backtest_id": "US_RANK_RULE_V1_TEST",
        "trade_date": date(2026, 5, 11),
        "symbol": "NVDA",
        "strategy_name": "US_RANK_TOP20",
        "selection_rule": "rank_no <= 20",
        "holding_days": 20,
        "rank_no": 1,
        "recommend_grade": "BUY",
        "total_score": 78.5,
        "entry_date": date(2026, 5, 12),
        "entry_price": 100.0,
        "exit_date": date(2026, 6, 9),
        "exit_price": 110.0,
        "return_pct": 0.10,
        "excess_return_vs_spy": 0.03,
        "excess_return_vs_qqq": 0.02,
        "excess_return_vs_universe": 0.04,
        "data_status": "OK",
    }
    row.update(overrides)
    return row


class USBacktestReportTests(unittest.TestCase):
    def test_aggregate_strategy_summary(self) -> None:
        rows = [
            make_summary_row(trade_date=date(2026, 5, 11), avg_return_pct=0.03, avg_excess_return_vs_spy=0.01),
            make_summary_row(trade_date=date(2026, 5, 12), avg_return_pct=0.05, avg_excess_return_vs_spy=0.02),
        ]
        aggregate = _aggregate_strategy_summary(rows)
        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(float(aggregate[0]["avg_return_pct"]), 0.04, places=6)
        self.assertEqual(aggregate[0]["test_days"], 2)

    def test_quality_summary_counts_missing_status(self) -> None:
        quality = _build_data_quality_summary(
            [make_summary_row()],
            [make_result_row(), make_result_row(symbol="AAPL", data_status="NOT_ENOUGH_FORWARD_DATA")],
        )
        self.assertEqual(quality["total_result_rows"], 2)
        self.assertEqual(quality["not_enough_forward_data_rows"], 1)

    def test_console_report_contains_best_candidate(self) -> None:
        aggregate = _aggregate_strategy_summary([make_summary_row()])
        quality = _build_data_quality_summary([make_summary_row()], [make_result_row()])
        rendered = build_console_report(
            backtest_id="US_RANK_RULE_V1_TEST",
            summary_rows=[make_summary_row()],
            aggregate_rows=aggregate,
            quality=quality,
            interpretation_lines=["SPY 대비 초과성과가 관찰됩니다."],
        )
        self.assertIn("[Best Candidate]", rendered)
        self.assertIn("US_RANK_TOP20", rendered)

    def test_markdown_report_contains_sections(self) -> None:
        aggregate = _aggregate_strategy_summary([make_summary_row()])
        quality = _build_data_quality_summary([make_summary_row()], [make_result_row()])
        rendered = build_markdown_report(
            backtest_id="US_RANK_RULE_V1_TEST",
            summary_rows=[make_summary_row()],
            aggregate_rows=aggregate,
            result_rows=[make_result_row()],
            quality=quality,
            interpretation_lines=["SPY 대비 초과성과가 관찰됩니다."],
            cfg=make_cfg(),
        )
        self.assertIn("# 미국주식 랭킹 백테스트 리포트", rendered)
        self.assertIn("## 2. 전략별 성과 요약", rendered)
        self.assertIn("## 8. 데이터 품질 및 누락 현황", rendered)

    def test_symbol_console_report_contains_hd_rows(self) -> None:
        rendered = build_symbol_console_report(
            backtest_id="US_RANK_RULE_V1_TEST",
            symbol="NVDA",
            result_rows=[make_result_row(), make_result_row(holding_days=5, return_pct=0.02)],
        )
        self.assertIn("[Symbol Backtest Detail]", rendered)
        self.assertIn("NVDA", rendered)
        self.assertIn("20", rendered)

    def test_interpretation_warns_for_short_test(self) -> None:
        aggregate = _aggregate_strategy_summary([make_summary_row()])
        quality = _build_data_quality_summary([make_summary_row()], [make_result_row()])
        lines = _build_interpretation_lines(aggregate, quality=quality, cfg=make_cfg())
        self.assertTrue(any("테스트 일수" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
