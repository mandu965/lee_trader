from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from utils.us_live_pre_trade_check import UsLivePreTradeCheckResult
from utils.us_live_order_safety import (
    LIVE_FINAL_CONFIRMATION_TEXT,
    assert_live_order_gate_closed_by_default,
    mask_sensitive_payload,
    validate_live_order_execution_allowed,
)


class USLiveOrderSafetyTests(unittest.TestCase):
    def _micro_order_row(self) -> dict[str, object]:
        return {
            "micro_order_id": "USMO_X",
            "approval_id": "USAPP_X",
            "policy_id": "US_LIVE_RULE_V1",
            "account_id": "US_LIVE_TEST",
            "trade_date": "2026-05-15",
            "symbol": "NVDA",
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 900.0,
            "order_qty": 1.0,
            "order_amount_usd": 50.0,
            "candidate_source": "MANUAL",
            "strategy_name": "rule_v1",
            "rank_no": 1,
            "recommend_grade": "BUY",
            "total_score": 81.5,
            "precheck_summary": "",
            "execution_mode": "LIVE",
            "request_status": "LIVE_CONFIRMATION_REQUIRED",
            "broker_name": "ALPACA",
        }

    def _precheck(self, decision: str = "REQUIRE_APPROVAL") -> UsLivePreTradeCheckResult:
        return UsLivePreTradeCheckResult(
            decision=decision,
            symbol="NVDA",
            side="BUY",
            reason_codes=["manual_approval_required"] if decision != "ALLOW" else [],
            reason_details=["Manual approval is required by policy."] if decision != "ALLOW" else [],
            severity="WARNING" if decision != "ALLOW" else "INFO",
            check_results={"APPROVAL": decision},
            requires_manual_approval=decision == "REQUIRE_APPROVAL",
            blocked=decision in {"BLOCK", "ERROR"},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @patch.dict(os.environ, {}, clear=False)
    def test_assert_live_order_gate_closed_by_default_passes(self) -> None:
        assert_live_order_gate_closed_by_default()

    def test_mask_sensitive_payload_hides_secrets(self) -> None:
        payload = {
            "api_key": "secret-key",
            "nested": {"authorization": "Bearer abc"},
            "safe": "ok",
        }
        masked = mask_sensitive_payload(payload)
        self.assertEqual(masked["api_key"], "***MASKED***")
        self.assertEqual(masked["nested"]["authorization"], "***MASKED***")
        self.assertEqual(masked["safe"], "ok")

    @patch("utils.us_live_order_safety.run_us_live_pre_trade_check")
    @patch("utils.us_live_order_safety.validate_approval_for_candidate")
    @patch("utils.us_live_order_safety.fetch_us_micro_order_request_rows")
    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_LIVE": "true",
            "US_MICRO_REAL_ORDER_BLOCKED": "false",
            "US_LIVE_TRADING_ENABLED": "true",
            "US_LIVE_ORDER_ENABLED": "true",
            "US_LIVE_BUY_ENABLED": "true",
            "US_LIVE_SELL_ENABLED": "false",
            "US_LIVE_REQUIRE_MANUAL_APPROVAL": "true",
            "US_LIVE_REQUIRE_FINAL_CONFIRMATION": "true",
            "US_LIVE_ALLOW_MARKET_ORDER": "false",
            "US_LIVE_MAX_ORDER_AMOUNT_USD": "50",
        },
        clear=False,
    )
    def test_validate_live_order_requires_final_confirmation(
        self,
        fetch_mock,
        approval_mock,
        precheck_mock,
    ) -> None:
        fetch_mock.return_value = [self._micro_order_row()]
        approval_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok", "approval_id": "USAPP_X"}
        precheck_mock.return_value = self._precheck("REQUIRE_APPROVAL")
        result = validate_live_order_execution_allowed("USMO_X", final_confirm=False)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["status"], "LIVE_CONFIRMATION_REQUIRED")
        self.assertEqual(result["reason_code"], "final_confirmation_missing")

    @patch("utils.us_live_order_safety.run_us_live_pre_trade_check")
    @patch("utils.us_live_order_safety.validate_approval_for_candidate")
    @patch("utils.us_live_order_safety.fetch_us_micro_order_request_rows")
    @patch.dict(
        os.environ,
        {
            "US_MICRO_ALLOW_LIVE": "true",
            "US_MICRO_REAL_ORDER_BLOCKED": "false",
            "US_LIVE_TRADING_ENABLED": "true",
            "US_LIVE_ORDER_ENABLED": "true",
            "US_LIVE_BUY_ENABLED": "true",
            "US_LIVE_SELL_ENABLED": "false",
            "US_LIVE_REQUIRE_MANUAL_APPROVAL": "true",
            "US_LIVE_REQUIRE_FINAL_CONFIRMATION": "true",
            "US_LIVE_ALLOW_MARKET_ORDER": "false",
            "US_LIVE_MAX_ORDER_AMOUNT_USD": "50",
        },
        clear=False,
    )
    def test_validate_live_order_returns_live_ready_with_confirmation(
        self,
        fetch_mock,
        approval_mock,
        precheck_mock,
    ) -> None:
        row = self._micro_order_row()
        row["approval_expires_at"] = datetime.now(timezone.utc) + timedelta(minutes=10)
        fetch_mock.return_value = [row]
        approval_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok", "approval_id": "USAPP_X"}
        precheck_mock.return_value = self._precheck("REQUIRE_APPROVAL")
        result = validate_live_order_execution_allowed("USMO_X", final_confirm=True)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["status"], "LIVE_READY")
        self.assertEqual(result["payload"]["order_type"], "LIMIT")
        self.assertEqual(LIVE_FINAL_CONFIRMATION_TEXT, "I_UNDERSTAND_THIS_IS_A_REAL_ORDER")


if __name__ == "__main__":
    unittest.main()
