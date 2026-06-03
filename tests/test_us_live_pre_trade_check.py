from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from utils.us_live_pre_trade_check import UsLiveOrderCandidate, run_us_live_pre_trade_check


class USLivePreTradeCheckTests(unittest.TestCase):
    @patch("utils.us_live_pre_trade_check.fetch_rank_component_rows_between")
    @patch("utils.us_live_pre_trade_check.fetch_meta_us_universe_rows")
    @patch("utils.us_live_pre_trade_check.fetch_latest_daily_feature_snapshots")
    @patch("utils.us_live_pre_trade_check.fetch_market_regime_rows_between")
    @patch("utils.us_live_pre_trade_check.fetch_us_live_daily_risk_usage_rows")
    @patch("utils.us_live_pre_trade_check.fetch_mixed_price_rows_for_tickers_between")
    @patch("utils.us_live_pre_trade_check.check_kill_switch_for_order_candidate")
    def test_default_policy_blocks_live_candidate(
        self,
        kill_check_mock,
        price_mock,
        usage_mock,
        regime_mock,
        feature_mock,
        universe_mock,
        rank_mock,
    ) -> None:
        rank_mock.return_value = [{
            "trade_date": date(2026, 5, 15),
            "symbol": "NVDA",
            "rank_no": 1,
            "recommend_grade": "BUY",
            "total_score": 80.0,
            "data_status": "OK",
            "exclude_reason": None,
        }]
        universe_mock.return_value = [{
            "symbol": "NVDA",
            "is_active": True,
            "is_etf": False,
            "is_leveraged": False,
            "is_inverse": False,
            "currency": "USD",
            "sector": "Technology",
        }]
        feature_mock.return_value = {"NVDA": {"volatility_20d": 0.02}}
        regime_mock.return_value = [{"trade_date": date(2026, 5, 15), "market_regime": "BULL_LOW_VOL", "spy_daily_ret_1d": 0.0, "qqq_daily_ret_1d": 0.0}]
        usage_mock.return_value = [{"trade_date": date(2026, 5, 15), "policy_id": "US_LIVE_RULE_V1", "account_id": "US_LIVE_TEST", "total_order_count": 0, "buy_amount_usd": 0, "sell_amount_usd": 0, "new_buy_count": 0, "failed_order_count": 0}]
        kill_check_mock.return_value = {"active": False, "matches": [], "reason_codes": [], "reason_details": []}
        price_mock.return_value = [
            {"trade_date": date(2026, 5, 14), "ticker": "NVDA", "close_price": 100.0},
            {"trade_date": date(2026, 5, 15), "ticker": "NVDA", "close_price": 101.0},
        ]
        candidate = UsLiveOrderCandidate(
            trade_date="2026-05-15",
            account_id="US_LIVE_TEST",
            policy_id="US_LIVE_RULE_V1",
            symbol="NVDA",
            side="BUY",
            requested_order_amount_usd=50.0,
            requested_qty=None,
            requested_order_type="LIMIT",
            requested_limit_price=100.0,
            candidate_source="MANUAL",
            strategy_name=None,
            rank_no=1,
            recommend_grade="BUY",
            total_score=80.0,
            reason=None,
        )
        result = run_us_live_pre_trade_check(candidate, write_block_log=False)
        self.assertEqual(result.decision, "BLOCK")
        self.assertIn("live_disabled", result.reason_codes)
        self.assertIn("manual_approval_required", result.reason_codes)

    @patch("utils.us_live_pre_trade_check.fetch_rank_component_rows_between")
    @patch("utils.us_live_pre_trade_check.fetch_meta_us_universe_rows")
    @patch("utils.us_live_pre_trade_check.fetch_latest_daily_feature_snapshots")
    @patch("utils.us_live_pre_trade_check.fetch_market_regime_rows_between")
    @patch("utils.us_live_pre_trade_check.fetch_us_live_daily_risk_usage_rows")
    @patch("utils.us_live_pre_trade_check.fetch_mixed_price_rows_for_tickers_between")
    @patch("utils.us_live_pre_trade_check.load_us_live_risk_policy")
    @patch("utils.us_live_pre_trade_check.check_kill_switch_for_order_candidate")
    def test_kill_switch_blocks_when_live_flags_enabled(
        self,
        kill_check_mock,
        policy_mock,
        price_mock,
        usage_mock,
        regime_mock,
        feature_mock,
        universe_mock,
        rank_mock,
    ) -> None:
        policy_mock.return_value = {
            "policy_id": "US_LIVE_RULE_V1",
            "safety": {
                "live_trading_enabled": True,
                "live_order_enabled": True,
                "buy_enabled": True,
                "sell_enabled": True,
                "require_manual_approval": False,
                "real_order_blocked": True,
            },
            "strategy": {"buy_grades": ["BUY", "STRONG_BUY"], "sell_grades": ["HOLD", "EXCLUDE"], "max_rank_no": 20},
            "order": {"max_order_amount_usd": 50, "min_order_amount_usd": 10, "max_daily_order_count": 3, "max_daily_buy_amount_usd": 100, "max_daily_sell_amount_usd": 500, "max_daily_new_buys": 1, "max_order_retry": 1},
            "market": {"block_bear_high_vol_regime": True, "block_buy_on_spy_drop_pct": -0.02, "block_buy_on_qqq_drop_pct": -0.025, "block_buy_on_symbol_gap_up_pct": 0.05, "block_buy_on_symbol_gap_down_pct": -0.05, "max_symbol_volatility_20d": 0.05},
            "instrument": {"block_leveraged_etf": True, "block_inverse_etf": True, "allow_etf": True},
            "time": {"regular_session_only": False},
        }
        rank_mock.return_value = [{"trade_date": date(2026, 5, 15), "symbol": "NVDA", "rank_no": 1, "recommend_grade": "BUY", "total_score": 80.0, "data_status": "OK", "exclude_reason": None}]
        universe_mock.return_value = [{"symbol": "NVDA", "is_active": True, "is_etf": False, "is_leveraged": False, "is_inverse": False, "currency": "USD", "sector": "Technology"}]
        feature_mock.return_value = {"NVDA": {"volatility_20d": 0.02}}
        regime_mock.return_value = [{"trade_date": date(2026, 5, 15), "market_regime": "BULL_LOW_VOL", "spy_daily_ret_1d": 0.0, "qqq_daily_ret_1d": 0.0}]
        usage_mock.return_value = [{"trade_date": date(2026, 5, 15), "policy_id": "US_LIVE_RULE_V1", "account_id": "US_LIVE_TEST", "total_order_count": 0, "buy_amount_usd": 0, "sell_amount_usd": 0, "new_buy_count": 0, "failed_order_count": 0}]
        kill_check_mock.return_value = {
            "active": True,
            "matches": [{"kill_switch_id": "US_LIVE_GLOBAL_KILL", "scope": "GLOBAL", "target_value": "ALL"}],
            "reason_codes": ["global_kill_switch_active"],
            "reason_details": ["US_LIVE_GLOBAL_KILL is active"],
        }
        price_mock.return_value = [{"trade_date": date(2026, 5, 15), "ticker": "NVDA", "close_price": 101.0}]
        candidate = UsLiveOrderCandidate(
            trade_date="2026-05-15",
            account_id="US_LIVE_TEST",
            policy_id="US_LIVE_RULE_V1",
            symbol="NVDA",
            side="BUY",
            requested_order_amount_usd=50.0,
            requested_qty=None,
            requested_order_type="LIMIT",
            requested_limit_price=100.0,
            candidate_source="MANUAL",
            strategy_name=None,
            rank_no=1,
            recommend_grade="BUY",
            total_score=80.0,
            reason=None,
        )
        result = run_us_live_pre_trade_check(candidate, write_block_log=False)
        self.assertEqual(result.decision, "BLOCK")
        self.assertIn("global_kill_switch_active", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
