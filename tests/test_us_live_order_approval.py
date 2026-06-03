from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from utils.us_live_order_approval import (
    approve_order_approval,
    create_order_approval_request,
    expire_order_approvals,
    reject_order_approval,
    validate_approval_for_candidate,
)
from utils.us_live_pre_trade_check import UsLiveOrderCandidate, UsLivePreTradeCheckResult


class USLiveOrderApprovalTests(unittest.TestCase):
    def _candidate(self) -> UsLiveOrderCandidate:
        return UsLiveOrderCandidate(
            trade_date="2026-05-15",
            account_id="US_LIVE_TEST",
            policy_id="US_LIVE_RULE_V1",
            symbol="NVDA",
            side="BUY",
            requested_order_amount_usd=50.0,
            requested_qty=None,
            requested_order_type="LIMIT",
            requested_limit_price=900.0,
            candidate_source="MANUAL",
            strategy_name=None,
            rank_no=3,
            recommend_grade="BUY",
            total_score=80.0,
            reason=None,
        )

    def _result(self, decision: str = "REQUIRE_APPROVAL") -> UsLivePreTradeCheckResult:
        return UsLivePreTradeCheckResult(
            decision=decision,
            symbol="NVDA",
            side="BUY",
            reason_codes=["manual_approval_required"],
            reason_details=["Manual approval is required by policy."],
            severity="WARNING",
            check_results={"APPROVAL": "REQUIRE_APPROVAL"},
            requires_manual_approval=True,
            blocked=False,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @patch("utils.us_live_order_approval.get_order_approval")
    @patch("utils.us_live_order_approval.insert_us_live_order_approval_event_log_rows")
    @patch("utils.us_live_order_approval.upsert_us_live_order_approval_rows")
    @patch("utils.us_live_order_approval._find_existing_pending")
    @patch("utils.us_live_order_approval.ensure_us_live_risk_tables")
    def test_create_request_for_require_approval(
        self,
        ensure_mock,
        existing_mock,
        upsert_mock,
        event_mock,
        get_mock,
    ) -> None:
        existing_mock.return_value = []
        get_mock.return_value = {"approval_id": "USAPP_X", "approval_status": "PENDING"}
        row = create_order_approval_request(self._candidate(), self._result(), requested_by="SYSTEM", expires_minutes=30)
        self.assertEqual(row["approval_status"], "PENDING")
        upsert_mock.assert_called_once()
        event_mock.assert_called_once()

    @patch("utils.us_live_order_approval.get_order_approval")
    @patch("utils.us_live_order_approval.insert_us_live_order_approval_event_log_rows")
    @patch("utils.us_live_order_approval.upsert_us_live_order_approval_rows")
    @patch("utils.us_live_order_approval._find_existing_pending")
    @patch("utils.us_live_order_approval.ensure_us_live_risk_tables")
    def test_create_request_supports_zero_minute_expiry(
        self,
        ensure_mock,
        existing_mock,
        upsert_mock,
        event_mock,
        get_mock,
    ) -> None:
        existing_mock.return_value = []
        get_mock.return_value = {"approval_id": "USAPP_X", "approval_status": "PENDING"}
        create_order_approval_request(self._candidate(), self._result(), requested_by="SYSTEM", expires_minutes=0)
        upsert_row = upsert_mock.call_args.args[0][0]
        self.assertIsNotNone(upsert_row["expires_at"])

    def test_block_result_does_not_create_request(self) -> None:
        with self.assertRaises(ValueError):
            create_order_approval_request(self._candidate(), self._result("BLOCK"))

    @patch("utils.us_live_order_approval.get_order_approval")
    @patch("utils.us_live_order_approval.insert_us_live_order_approval_event_log_rows")
    @patch("utils.us_live_order_approval.upsert_us_live_order_approval_rows")
    def test_approve_changes_status(self, upsert_mock, event_mock, get_mock) -> None:
        get_mock.side_effect = [
            {"approval_id": "USAPP_X", "approval_status": "PENDING"},
            {"approval_id": "USAPP_X", "approval_status": "APPROVED"},
        ]
        row = approve_order_approval("USAPP_X", approved_by="lee", approval_reason="ok")
        self.assertEqual(row["approval_status"], "APPROVED")

    @patch("utils.us_live_order_approval.get_order_approval")
    @patch("utils.us_live_order_approval.insert_us_live_order_approval_event_log_rows")
    @patch("utils.us_live_order_approval.upsert_us_live_order_approval_rows")
    def test_reject_changes_status(self, upsert_mock, event_mock, get_mock) -> None:
        get_mock.side_effect = [
            {"approval_id": "USAPP_X", "approval_status": "PENDING"},
            {"approval_id": "USAPP_X", "approval_status": "REJECTED"},
        ]
        row = reject_order_approval("USAPP_X", rejected_by="lee", reject_reason="no")
        self.assertEqual(row["approval_status"], "REJECTED")

    @patch("utils.us_live_order_approval.insert_us_live_order_approval_event_log_rows")
    @patch("utils.us_live_order_approval.upsert_us_live_order_approval_rows")
    @patch("utils.us_live_order_approval.fetch_us_live_order_approval_rows")
    def test_expire_pending(self, fetch_mock, upsert_mock, event_mock) -> None:
        fetch_mock.return_value = [
            {"approval_id": "USAPP_X", "approval_status": "PENDING", "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1)}
        ]
        result = expire_order_approvals()
        self.assertEqual(result["expired_count"], 1)

    @patch("utils.us_live_order_approval.get_order_approval")
    def test_validate_approval_for_candidate(self, get_mock) -> None:
        get_mock.return_value = {
            "approval_id": "USAPP_X",
            "approval_status": "APPROVED",
            "trade_date": "2026-05-15",
            "account_id": "US_LIVE_TEST",
            "symbol": "NVDA",
            "side": "BUY",
            "candidate_source": "MANUAL",
            "strategy_name": "",
            "requested_order_amount_usd": 50.0,
            "requested_order_type": "LIMIT",
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        }
        result = validate_approval_for_candidate(self._candidate(), "USAPP_X")
        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
