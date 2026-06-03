from __future__ import annotations

from datetime import date
import json
import unittest

from python.us.calculate_us_stock_rule_scores import (
    _normalize_debt_to_equity,
    _normalize_ratio,
    _rank_rows,
    calculate_fundamental_score,
    calculate_growth_score,
    calculate_momentum_score,
    calculate_relative_strength_score,
    calculate_risk_score,
    calculate_total_score,
    calculate_valuation_score,
)
from python.us.validate_us_stock_rank_daily import validate_rank_result
from python.us.us_config import USRuleRankingConfig


def make_config(**overrides) -> USRuleRankingConfig:
    values = {
        "enabled": True,
        "source": "rule_v1",
        "min_feature_quality_score": 40.0,
        "apply_fundamental_quality_to_etf": False,
        "volatility_20d_threshold": 0.05,
        "volatility_60d_threshold": 0.04,
        "return_20d_overheat_threshold": 0.25,
        "strong_buy_score": 80.0,
        "buy_score": 70.0,
        "watch_score": 60.0,
        "hold_score": 50.0,
        "log_level": "INFO",
    }
    values.update(overrides)
    return USRuleRankingConfig(**values)


def make_row(**overrides) -> dict[str, object]:
    row = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Hardware",
        "universe_group": "NASDAQ100,SP500",
        "is_active": True,
        "is_etf": False,
        "is_leveraged": False,
        "is_inverse": False,
        "exclude_reason": None,
        "close": 185.2,
        "price_trade_date": date(2026, 5, 12),
        "price_row_count": 250,
        "ret_20d": 0.08,
        "ret_60d": 0.14,
        "ret_120d": 0.22,
        "volatility_20d": 0.02,
        "volatility_60d": 0.03,
        "ma_20": 178.4,
        "ma_60": 170.1,
        "price_above_ma20_flag": "Y",
        "price_above_ma60_flag": "Y",
        "avg_volume": 10000000.0,
        "rs_spy_20d": 0.03,
        "rs_spy_60d": 0.04,
        "rs_qqq_20d": -0.01,
        "rs_qqq_60d": 0.02,
        "roe": 0.28,
        "operating_margin": 0.31,
        "profit_margin": 0.23,
        "debt_to_equity": 0.42,
        "revenue_growth": 0.16,
        "earnings_growth": 0.21,
        "trailing_pe": 32.4,
        "forward_pe": 27.1,
        "price_to_book": 8.5,
        "market_cap": 1000000000000.0,
        "feature_quality_score": 75.0,
    }
    row.update(overrides)
    return row


class USRuleRankingTests(unittest.TestCase):
    def test_normalize_ratio_percent_style(self) -> None:
        self.assertAlmostEqual(_normalize_ratio(15), 0.15)
        self.assertAlmostEqual(_normalize_ratio(0.15), 0.15)

    def test_normalize_debt_to_equity_percent_style(self) -> None:
        self.assertAlmostEqual(_normalize_debt_to_equity(42), 0.42)
        self.assertAlmostEqual(_normalize_debt_to_equity(0.42), 0.42)

    def test_momentum_score_caps_at_25(self) -> None:
        score, detail = calculate_momentum_score(make_row())
        self.assertEqual(score, 25.0)
        self.assertEqual(detail["max_score"], 25)

    def test_relative_strength_score_partial_positive(self) -> None:
        score, _ = calculate_relative_strength_score(make_row())
        self.assertEqual(score, 15.0)

    def test_fundamental_score_matches_rule(self) -> None:
        score, _ = calculate_fundamental_score(make_row())
        self.assertEqual(score, 20.0)

    def test_growth_score_matches_rule(self) -> None:
        score, _ = calculate_growth_score(make_row())
        self.assertEqual(score, 13.0)

    def test_valuation_score_zero_for_expensive_name(self) -> None:
        score, _ = calculate_valuation_score(make_row())
        self.assertEqual(score, 0.0)

    def test_risk_score_penalty_applies(self) -> None:
        score, detail = calculate_risk_score(
            make_row(volatility_20d=0.06, volatility_60d=0.05, ret_20d=0.30, feature_quality_score=30.0),
            cfg=make_config(),
        )
        self.assertEqual(score, -10.0)
        self.assertTrue(detail["reasons"])

    def test_total_score_and_grade(self) -> None:
        result = calculate_total_score(
            make_row(rs_qqq_20d=0.01, forward_pe=20.0, price_to_book=4.0),
            trade_date=date(2026, 5, 12),
            cfg=make_config(),
        )
        self.assertEqual(result["recommend_grade"], "STRONG_BUY")
        self.assertGreaterEqual(result["total_score"], 80.0)
        payload = json.loads(result["score_detail_json"])
        self.assertEqual(payload["meta"]["data_status"], "OK")
        self.assertEqual(payload["meta"]["reason_category"], "MOMENTUM_LEADER")
        self.assertIn("strong_momentum", payload["meta"]["reason_tags"])
        self.assertTrue(result["reason_summary"])

    def test_missing_price_forces_exclude(self) -> None:
        result = calculate_total_score(
            make_row(close=None, price_trade_date=date(2026, 5, 11)),
            trade_date=date(2026, 5, 12),
            cfg=make_config(),
        )
        self.assertEqual(result["recommend_grade"], "EXCLUDE")
        self.assertEqual(result["data_status"], "MISSING_PRICE_FEATURE")
        self.assertTrue(result["exclude_reason"])

    def test_validate_rank_result_flags_missing_exclude_reason(self) -> None:
        result = calculate_total_score(
            make_row(rs_qqq_20d=0.01, forward_pe=20.0, price_to_book=4.0),
            trade_date=date(2026, 5, 12),
            cfg=make_config(),
        )
        result["recommend_grade"] = "EXCLUDE"
        result["exclude_reason"] = None
        messages = validate_rank_result(result, cfg=make_config())
        self.assertTrue(any("EXCLUDE row missing exclude_reason" in message for message in messages))

    def test_rank_sort_uses_total_then_tiebreakers(self) -> None:
        rows = [
            {"symbol": "BBB", "total_score": 80.0, "momentum_score": 20.0, "relative_strength_score": 10.0, "fundamental_score": 10.0},
            {"symbol": "AAA", "total_score": 80.0, "momentum_score": 21.0, "relative_strength_score": 9.0, "fundamental_score": 10.0},
        ]
        ranked = _rank_rows(rows)
        self.assertEqual(ranked[0]["symbol"], "AAA")
        self.assertEqual(ranked[0]["rank_no"], 1)


if __name__ == "__main__":
    unittest.main()
