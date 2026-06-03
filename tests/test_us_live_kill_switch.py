from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from utils.us_live_kill_switch import (
    activate_kill_switch,
    build_kill_switch_id,
    check_kill_switch_for_order_candidate,
    clear_kill_switch,
    evaluate_kill_switch_triggers,
)
from utils.us_live_pre_trade_check import UsLiveOrderCandidate


class USLiveKillSwitchTests(unittest.TestCase):
    def test_build_kill_switch_id_normalizes_scope_and_target(self) -> None:
        self.assertEqual(build_kill_switch_id("GLOBAL"), "US_LIVE_GLOBAL_KILL")
        self.assertEqual(build_kill_switch_id("BUY"), "US_LIVE_BUY_KILL")
        self.assertEqual(build_kill_switch_id("SYMBOL", "nvda"), "US_LIVE_SYMBOL_NVDA_KILL")
        self.assertEqual(build_kill_switch_id("SECTOR", "Information Technology"), "US_LIVE_SECTOR_INFORMATION_TECHNOLOGY_KILL")

    @patch("utils.us_live_kill_switch.ensure_us_live_risk_tables")
    @patch("utils.us_live_kill_switch.insert_us_live_kill_switch_event_log_rows")
    @patch("utils.us_live_kill_switch.upsert_us_live_kill_switch_rows")
    @patch("utils.us_live_kill_switch.get_kill_switch_status")
    def test_activate_kill_switch_logs_event(
        self,
        status_mock,
        upsert_mock,
        event_mock,
        ensure_mock,
    ) -> None:
        status_mock.side_effect = [
            {"kill_switch_id": "US_LIVE_GLOBAL_KILL", "scope": "GLOBAL", "target_value": "ALL", "is_active": False, "activated_at": None, "activated_by": None},
            {"kill_switch_id": "US_LIVE_GLOBAL_KILL", "scope": "GLOBAL", "target_value": "ALL", "is_active": True},
        ]
        row = activate_kill_switch(
            "GLOBAL",
            None,
            reason_code="manual_stop",
            reason_detail="operator stop",
            performed_by="lee",
        )
        self.assertTrue(row["is_active"])
        upsert_mock.assert_called_once()
        event_mock.assert_called_once()

    @patch("utils.us_live_kill_switch.ensure_us_live_risk_tables")
    @patch("utils.us_live_kill_switch.insert_us_live_kill_switch_event_log_rows")
    @patch("utils.us_live_kill_switch.upsert_us_live_kill_switch_rows")
    @patch("utils.us_live_kill_switch.get_kill_switch_status")
    def test_clear_kill_switch_requires_reason_and_operator(
        self,
        status_mock,
        upsert_mock,
        event_mock,
        ensure_mock,
    ) -> None:
        status_mock.side_effect = [
            {"kill_switch_id": "US_LIVE_SYMBOL_NVDA_KILL", "scope": "SYMBOL", "target_value": "NVDA", "is_active": True, "reason_code": "data_error", "reason_detail": "bad data"},
            {"kill_switch_id": "US_LIVE_SYMBOL_NVDA_KILL", "scope": "SYMBOL", "target_value": "NVDA", "is_active": False},
        ]
        row = clear_kill_switch("SYMBOL", "NVDA", clear_reason="verified", performed_by="lee")
        self.assertFalse(row["is_active"])
        upsert_mock.assert_called_once()
        event_mock.assert_called_once()

    @patch("utils.us_live_kill_switch.fetch_meta_us_universe_rows")
    @patch("utils.us_live_kill_switch.list_active_kill_switches")
    def test_check_candidate_matches_symbol_scope(self, active_mock, universe_mock) -> None:
        universe_mock.return_value = [{"symbol": "NVDA", "sector": "Technology"}]
        active_mock.return_value = [
            {"kill_switch_id": "US_LIVE_SYMBOL_NVDA_KILL", "scope": "SYMBOL", "target_value": "NVDA", "is_active": True},
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
        matched = check_kill_switch_for_order_candidate(candidate)
        self.assertTrue(matched["active"])
        self.assertIn("symbol_kill_switch_active", matched["reason_codes"])

    @patch("utils.us_live_kill_switch.fetch_market_regime_rows_between")
    @patch("utils.us_live_kill_switch.fetch_us_live_order_block_log_rows")
    @patch("utils.us_live_kill_switch.fetch_us_live_daily_risk_usage_rows")
    @patch("utils.us_live_kill_switch.load_us_live_risk_policy")
    def test_evaluate_triggers_detects_failure_limit(
        self,
        policy_mock,
        usage_mock,
        block_mock,
        regime_mock,
    ) -> None:
        policy_mock.return_value = {
            "policy_id": "US_LIVE_RULE_V1",
            "order": {"max_daily_order_failures": 3, "max_daily_order_count": 3},
            "market": {"block_bear_high_vol_regime": True},
        }
        usage_mock.return_value = [{
            "trade_date": date(2026, 5, 15),
            "policy_id": "US_LIVE_RULE_V1",
            "account_id": "US_LIVE_TEST",
            "failed_order_count": 3,
            "blocked_order_count": 0,
        }]
        block_mock.return_value = []
        regime_mock.return_value = []
        triggers = evaluate_kill_switch_triggers("2026-05-15", "US_LIVE_TEST", "US_LIVE_RULE_V1")
        self.assertEqual(triggers[0]["reason_code"], "order_failure_limit")


if __name__ == "__main__":
    unittest.main()
