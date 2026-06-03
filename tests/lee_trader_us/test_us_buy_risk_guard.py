from __future__ import annotations

import unittest

from python.us.buy_automation.config import BuyAutomationConfig
from python.us.buy_automation.risk_guard import evaluate_candidate


def _config() -> BuyAutomationConfig:
    from pathlib import Path

    return BuyAutomationConfig(
        root_dir=Path("."),
        output_dir=Path("."),
        mode="SHADOW",
        enabled=True,
        top_n=5,
        min_grade="BUY",
        min_score=70.0,
        min_prob=0.6,
        max_daily_symbols=1,
        max_daily_amount_usd=100.0,
        max_per_symbol_amount_usd=100.0,
        min_price=5.0,
        max_price=500.0,
        max_gap_up_pct=0.05,
        max_intraday_change_pct=0.08,
        max_volatility_pct=0.10,
        require_financial_data=True,
        require_benchmark_strength=True,
        cooldown_days=10,
        failsafe_on_data_error=True,
        block_on_kill_switch=True,
        ranking_source="rule_v1",
        live_implemented=False,
        warnings=(),
    )


class BuyAutomationRiskGuardTests(unittest.TestCase):
    def test_blocks_when_required_data_missing(self) -> None:
        candidate = {
            "symbol": "AAPL",
            "rank": 1,
            "score": 82.0,
            "probability": None,
            "reference_price": 150.0,
            "recommend_grade": "BUY",
            "financial_feature": {},
            "relative_strength": {},
            "gap_up_pct": None,
            "intraday_change_pct": None,
            "volatility_20d": None,
            "data_status": "OK",
        }
        result = evaluate_candidate(candidate, _config(), selected_count=0, selected_amount_usd=0.0)
        self.assertFalse(result["allowed"])
        self.assertIn("PROBABILITY_MISSING", result["block_reasons"])
        self.assertIn("FINANCIAL_DATA_MISSING", result["block_reasons"])
        self.assertIn("BENCHMARK_STRENGTH_MISSING", result["block_reasons"])

    def test_blocks_when_kill_switch_active(self) -> None:
        candidate = {
            "symbol": "NVDA",
            "rank": 1,
            "score": 90.0,
            "probability": 0.8,
            "reference_price": 100.0,
            "recommend_grade": "BUY",
            "financial_feature": {"financial_quality_score": 50},
            "financial_quality_score": 50,
            "relative_strength": {"rs_spy_20d": 0.1},
            "gap_up_pct": 0.01,
            "intraday_change_pct": 0.01,
            "volatility_20d": 0.02,
            "data_status": "OK",
        }
        result = evaluate_candidate(candidate, _config(), selected_count=0, selected_amount_usd=0.0, kill_switch_active=True)
        self.assertFalse(result["allowed"])
        self.assertIn("KILL_SWITCH_ACTIVE", result["block_reasons"])


if __name__ == "__main__":
    unittest.main()
