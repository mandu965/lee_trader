from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from python.us.experiment_us_stock_rule_weights import (
    WeightConfig,
    _assign_summary_ranks,
    _build_experiment_rank_row,
    _score_contributions,
    default_weight_configs,
)
from python.us.report_us_stock_rule_weight_experiment import (
    _promote_status,
    build_console_report,
)
from python.us.us_config import USRuleRankingConfig


def make_rank_cfg() -> USRuleRankingConfig:
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


class USWeightExperimentTests(unittest.TestCase):
    def test_baseline_contribution_matches_original_total(self) -> None:
        baseline = next(item for item in default_weight_configs() if item.weight_config_id == "RULE_V1_BASELINE")
        source_row = {
            "momentum_score": 25.0,
            "relative_strength_score": 20.0,
            "fundamental_score": 20.0,
            "growth_score": 15.0,
            "valuation_score": 10.0,
            "risk_score": -10.0,
        }
        contributions, total = _score_contributions(source_row, baseline)
        self.assertAlmostEqual(total, 80.0, places=6)
        self.assertAlmostEqual(contributions["momentum_score"], 25.0, places=6)
        self.assertAlmostEqual(contributions["risk_score"], -10.0, places=6)

    def test_build_experiment_rank_row_regrades_score(self) -> None:
        baseline = WeightConfig("RULE_V1_TEST", "test", 30, 25, 15, 15, 5, 10)
        row = _build_experiment_rank_row(
            experiment_id="EXP1",
            weight_config=baseline,
            source_row={
                "trade_date": date(2026, 5, 11),
                "symbol": "NVDA",
                "total_score": 78.5,
                "momentum_score": 20.0,
                "relative_strength_score": 16.0,
                "fundamental_score": 18.0,
                "growth_score": 10.0,
                "valuation_score": 8.0,
                "risk_score": -1.0,
                "data_status": "OK",
                "exclude_reason": None,
                "reason_summary": "baseline",
                "score_detail_json": "{}",
                "is_etf": False,
                "source": "rule_v1",
            },
            cfg=make_rank_cfg(),
        )
        self.assertEqual(row["symbol"], "NVDA")
        self.assertIn("weight_config_id", row["_detail"]["meta"])
        self.assertIn(row["recommend_grade"], {"BUY", "STRONG_BUY", "WATCH", "HOLD", "EXCLUDE"})

    def test_assign_summary_ranks_prefers_higher_excess(self) -> None:
        rows = [
            {
                "weight_config_id": "A",
                "strategy_name": "US_RANK_TOP20",
                "holding_days": 20,
                "avg_excess_return_vs_spy": 0.02,
                "avg_excess_return_vs_qqq": 0.01,
                "win_rate_vs_spy": 0.55,
                "win_rate_vs_qqq": 0.53,
                "avg_return_bear": -0.01,
                "avg_return_high_vol": -0.02,
            },
            {
                "weight_config_id": "B",
                "strategy_name": "US_RANK_TOP20",
                "holding_days": 20,
                "avg_excess_return_vs_spy": 0.01,
                "avg_excess_return_vs_qqq": 0.00,
                "win_rate_vs_spy": 0.50,
                "win_rate_vs_qqq": 0.49,
                "avg_return_bear": -0.03,
                "avg_return_high_vol": -0.04,
            },
        ]
        _assign_summary_ranks(rows)
        rank_map = {row["weight_config_id"]: row["score_rank"] for row in rows}
        self.assertLess(rank_map["A"], rank_map["B"])

    def test_promote_status_detects_candidate(self) -> None:
        baseline = {
            "avg_excess_return_vs_spy": 0.01,
            "win_rate_vs_spy": 0.50,
            "avg_return_bear": -0.03,
            "avg_return_high_vol": -0.02,
        }
        candidate = {
            "test_days": 40,
            "avg_excess_return_vs_spy": 0.02,
            "win_rate_vs_spy": 0.55,
            "avg_return_bear": -0.02,
            "avg_return_high_vol": -0.01,
        }
        self.assertEqual(_promote_status(candidate, baseline, 30), "PROMOTE_CANDIDATE")

    def test_console_report_contains_best_candidate_section(self) -> None:
        rendered = build_console_report(
            experiment_id="EXP1",
            target_rows=[
                {
                    "weight_config_id": "RULE_V1_BASELINE",
                    "avg_return_pct": 0.02,
                    "avg_excess_return_vs_spy": 0.01,
                    "avg_excess_return_vs_qqq": 0.0,
                    "win_rate_vs_spy": 0.5,
                    "win_rate_vs_qqq": 0.49,
                    "avg_return_bear": -0.03,
                    "avg_return_high_vol": -0.02,
                    "score_rank": 12,
                },
                {
                    "weight_config_id": "RULE_V1_QUALITY_PLUS",
                    "avg_return_pct": 0.021,
                    "avg_excess_return_vs_spy": 0.012,
                    "avg_excess_return_vs_qqq": 0.001,
                    "win_rate_vs_spy": 0.52,
                    "win_rate_vs_qqq": 0.50,
                    "avg_return_bear": -0.02,
                    "avg_return_high_vol": -0.01,
                    "score_rank": 8,
                    "test_days": 40,
                },
            ],
            baseline_row={
                "avg_excess_return_vs_spy": 0.01,
                "win_rate_vs_spy": 0.5,
                "avg_return_bear": -0.03,
                "avg_return_high_vol": -0.02,
            },
            baseline_id="RULE_V1_BASELINE",
            period=(date(2026, 1, 1), date(2026, 5, 12)),
            min_test_days=30,
        )
        self.assertIn("[Best Candidate]", rendered)
        self.assertIn("RULE_V1_QUALITY_PLUS", rendered)


if __name__ == "__main__":
    unittest.main()
