from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from utils.us_live_pre_trade_check import UsLivePreTradeCheckResult
from utils.us_micro_order_request import create_micro_order_from_approval, send_micro_order_via_mock


class USMicroOrderRequestTests(unittest.TestCase):
    def _approval_row(self) -> dict[str, object]:
        return {
            "approval_id": "USAPP_20260515_US_LIVE_TEST_NVDA_BUY_20260515230000",
            "approval_status": "APPROVED",
            "trade_date": "2026-05-15",
            "policy_id": "US_LIVE_RULE_V1",
            "account_id": "US_LIVE_TEST",
            "symbol": "NVDA",
            "side": "BUY",
            "candidate_source": "MANUAL",
            "strategy_name": "rule_v1",
            "rank_no": 1,
            "recommend_grade": "BUY",
            "total_score": 81.5,
            "requested_order_type": "LIMIT",
            "requested_limit_price": 900.0,
            "requested_qty": 1.0,
            "requested_order_amount_usd": 50.0,
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
            "approved_by": "lee",
            "approved_at": datetime.now(timezone.utc),
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

    @patch("utils.us_micro_order_request.fetch_us_micro_order_request_rows")
    @patch("utils.us_micro_order_request.run_us_live_pre_trade_check")
    @patch("utils.us_micro_order_request.validate_approval_for_candidate")
    @patch("utils.us_micro_order_request.get_order_approval")
    @patch("utils.us_micro_order_request.ensure_us_micro_live_tables")
    @patch("utils.us_micro_order_request.assert_us_micro_mock_only")
    def test_create_micro_order_dry_run_ready_to_send(
        self,
        _safety,
        _ensure,
        get_approval_mock,
        validate_mock,
        precheck_mock,
        fetch_mock,
    ) -> None:
        fetch_mock.return_value = []
        get_approval_mock.return_value = self._approval_row()
        validate_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok"}
        precheck_mock.return_value = self._precheck("REQUIRE_APPROVAL")
        result = create_micro_order_from_approval("USAPP_X", dry_run=True)
        self.assertEqual(result["micro_order"]["request_status"], "READY_TO_SEND")
        self.assertEqual(result["micro_order"]["execution_mode"], "MOCK")

    @patch("utils.us_micro_order_request.fetch_us_micro_order_request_rows")
    @patch("utils.us_micro_order_request.run_us_live_pre_trade_check")
    @patch("utils.us_micro_order_request.validate_approval_for_candidate")
    @patch("utils.us_micro_order_request.get_order_approval")
    @patch("utils.us_micro_order_request.ensure_us_micro_live_tables")
    @patch("utils.us_micro_order_request.assert_us_micro_mock_only")
    def test_create_micro_order_dry_run_precheck_failed(
        self,
        _safety,
        _ensure,
        get_approval_mock,
        validate_mock,
        precheck_mock,
        fetch_mock,
    ) -> None:
        fetch_mock.return_value = []
        get_approval_mock.return_value = self._approval_row()
        validate_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok"}
        precheck_mock.return_value = self._precheck("BLOCK")
        result = create_micro_order_from_approval("USAPP_X", dry_run=True)
        self.assertEqual(result["micro_order"]["request_status"], "PRECHECK_FAILED")

    @patch("utils.us_micro_order_request.fetch_us_micro_order_request_rows")
    @patch("utils.us_micro_order_request.run_us_live_pre_trade_check")
    @patch("utils.us_micro_order_request.validate_approval_for_candidate")
    @patch("utils.us_micro_order_request.get_order_approval")
    @patch("utils.us_micro_order_request.ensure_us_micro_live_tables")
    def test_create_live_micro_order_dry_run_blocked_without_allow_live_create(
        self,
        _ensure,
        get_approval_mock,
        validate_mock,
        precheck_mock,
        fetch_mock,
    ) -> None:
        fetch_mock.return_value = []
        get_approval_mock.return_value = self._approval_row()
        validate_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok"}
        precheck_mock.return_value = self._precheck("REQUIRE_APPROVAL")
        result = create_micro_order_from_approval("USAPP_X", execution_mode="LIVE", dry_run=True)
        self.assertEqual(result["micro_order"]["request_status"], "LIVE_BLOCKED")
        self.assertEqual(result["micro_order"]["reject_reason_code"], "live_mode_not_allowed")

    @patch("utils.us_micro_order_request.update_micro_order_status")
    @patch("utils.us_micro_order_request.UsMockOrderClient")
    @patch("utils.us_micro_order_request.run_us_live_pre_trade_check")
    @patch("utils.us_micro_order_request.validate_approval_for_candidate")
    @patch("utils.us_micro_order_request.get_micro_order")
    @patch("utils.us_micro_order_request.ensure_us_micro_live_tables")
    @patch("utils.us_micro_order_request.assert_us_micro_mock_only")
    def test_send_micro_order_mock_accepts(
        self,
        _safety,
        _ensure,
        get_micro_mock,
        validate_mock,
        precheck_mock,
        client_cls_mock,
        update_mock,
    ) -> None:
        get_micro_mock.return_value = {
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
            "execution_mode": "MOCK",
            "request_status": "READY_TO_SEND",
        }
        validate_mock.return_value = {"valid": True, "reason_code": "approved", "detail": "ok"}
        precheck_mock.return_value = self._precheck("REQUIRE_APPROVAL")
        client = client_cls_mock.return_value
        client.submit_order.return_value = {
            "success": True,
            "broker_order_id": "MOCK_NVDA_20260515231000",
            "status": "ACCEPTED",
            "message": "Mock order accepted",
        }
        update_mock.side_effect = [
            {"micro_order_id": "USMO_X", "request_status": "SENT"},
            {"micro_order_id": "USMO_X", "request_status": "ACCEPTED", "broker_order_id": "MOCK_NVDA_20260515231000"},
        ]
        result = send_micro_order_via_mock("USMO_X")
        self.assertEqual(result["request_status"], "ACCEPTED")
        self.assertEqual(update_mock.call_args_list[-1].args[1], "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
