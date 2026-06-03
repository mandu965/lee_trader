from __future__ import annotations

from datetime import date
import json
import tempfile
import unittest
from pathlib import Path

from python.us.report_us_stock_top_rank import (
    build_console_report,
    build_detail_console_report,
    build_excluded_console_report,
    build_markdown_report,
    build_summary_text,
    write_csv,
)
from python.us.validate_us_stock_rank_daily import build_validation_summary_text, summarize_validation
from python.us.us_config import USRuleRankingConfig


def make_row(**overrides) -> dict[str, object]:
    detail = {
        "meta": {
            "data_status": "OK",
            "reason_category": "MOMENTUM_LEADER",
            "reason_tags": ["strong_momentum", "qqq_outperform", "expensive_valuation"],
            "grade_rationale": "Total score cleared the BUY cutoff (70) but stayed below STRONG_BUY.",
        },
        "momentum": {"score": 23.0, "max_score": 25},
        "relative_strength": {"score": 18.0, "max_score": 20},
        "fundamental": {"score": 14.0, "max_score": 20},
        "growth": {"score": 13.0, "max_score": 15},
        "valuation": {"score": 4.0, "max_score": 10},
        "risk": {"score": -3.0, "min_score": -10},
    }
    row = {
        "trade_date": date(2026, 5, 12),
        "rank_no": 1,
        "symbol": "NVDA",
        "company_name": "NVIDIA Corp",
        "sector": "Technology",
        "industry": "Semiconductors",
        "recommend_grade": "BUY",
        "total_score": 78.5,
        "momentum_score": 23.0,
        "relative_strength_score": 18.0,
        "fundamental_score": 14.0,
        "growth_score": 13.0,
        "valuation_score": 4.0,
        "risk_score": -3.0,
        "feature_quality_score": 80.0,
        "reason_summary": "Momentum and relative strength are strong, but valuation still looks demanding.",
        "score_detail_json": json.dumps(detail),
        "source": "rule_v1",
        "data_status": "OK",
        "exclude_reason": None,
        "is_etf": False,
    }
    row.update(overrides)
    return row


def make_stats() -> dict[str, object]:
    return {
        "trade_date": date(2026, 5, 12),
        "total_ranked": 520,
        "eligible_count": 430,
        "strong_buy_count": 3,
        "buy_count": 18,
        "watch_count": 71,
        "hold_count": 122,
        "exclude_count": 90,
        "avg_total_score": 54.8,
        "max_total_score": 78.5,
        "min_total_score": 0.0,
        "avg_feature_quality_score": 63.2,
        "avg_momentum_score": 12.1,
        "avg_relative_strength_score": 9.4,
        "avg_fundamental_score": 10.5,
        "avg_risk_score": -1.2,
    }


class USRankReportTests(unittest.TestCase):
    def make_cfg(self) -> USRuleRankingConfig:
        return USRuleRankingConfig(
            enabled=True,
            source="rule_v1",
            min_feature_quality_score=40.0,
            apply_fundamental_quality_to_etf=False,
            volatility_20d_threshold=0.05,
            volatility_60d_threshold=0.04,
            return_20d_overheat_threshold=0.25,
            strong_buy_score=80.0,
            buy_score=70.0,
            watch_score=60.0,
            hold_score=50.0,
            log_level="INFO",
        )

    def test_console_report_contains_summary(self) -> None:
        rendered = build_console_report(
            trade_date=date(2026, 5, 12),
            rows=[make_row()],
            stats=make_stats(),
            top_n=20,
        )
        self.assertIn("[US Stock Top 20 Ranking]", rendered)
        self.assertIn("[Summary]", rendered)
        self.assertIn("NVDA", rendered)
        self.assertIn("MOMENTUM_LEADER", rendered)

    def test_markdown_report_contains_sections(self) -> None:
        rendered = build_markdown_report(
            trade_date=date(2026, 5, 12),
            rows=[make_row()],
            stats=make_stats(),
            top_n=20,
            grade=None,
        )
        self.assertIn("# US Stock Top 20 Report", rendered)
        self.assertIn("## Recommendation Reasons", rendered)
        self.assertIn("NVIDIA Corp", rendered)
        self.assertIn("reason_tags", rendered)

    def test_detail_console_report_uses_score_detail_json(self) -> None:
        rendered = build_detail_console_report(
            trade_date=date(2026, 5, 12),
            row=make_row(),
        )
        self.assertIn("[US Stock Ranking Detail]", rendered)
        self.assertIn("Momentum: 23.0 / 25", rendered)
        self.assertIn("Relative Strength: 18.0 / 20", rendered)
        self.assertIn("Risk: -3.0 / -10~0", rendered)
        self.assertIn("Category: MOMENTUM_LEADER", rendered)
        self.assertIn("qqq_outperform", rendered)

    def test_summary_text_renders_expected_fields(self) -> None:
        rendered = build_summary_text(make_stats())
        self.assertIn("Eligible: 430", rendered)
        self.assertIn("Avg Feature Quality: 63.2", rendered)

    def test_write_csv_uses_utf8_and_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "report.csv"
            write_csv(path, [make_row()])
            text_value = path.read_text(encoding="utf-8")
        self.assertIn("trade_date,rank_no,symbol", text_value)
        self.assertIn("NVDA", text_value)
        self.assertIn("reason_category", text_value)

    def test_validation_summary_text_renders(self) -> None:
        summary = summarize_validation([make_row()], cfg=self.make_cfg())
        rendered = build_validation_summary_text(summary)
        self.assertIn("[Validation Summary]", rendered)
        self.assertIn("Total Checked: 1", rendered)

    def test_excluded_console_report_renders(self) -> None:
        rendered = build_excluded_console_report(
            trade_date=date(2026, 5, 12),
            rows=[make_row(recommend_grade="EXCLUDE", exclude_reason="Total score below HOLD threshold 50.", data_status="EXCLUDED")],
            limit_n=50,
        )
        self.assertIn("[US Stock Excluded List]", rendered)
        self.assertIn("Exclude Reason", rendered)


if __name__ == "__main__":
    unittest.main()
