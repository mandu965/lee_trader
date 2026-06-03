from __future__ import annotations


class UsBrokerAccountClient:
    """Account/position lookup adapter for Phase 7-5 reconciliation."""

    def get_account_snapshot(self, account_id: str) -> dict:
        raise NotImplementedError

    def get_positions(self, account_id: str) -> list[dict]:
        raise NotImplementedError

    def get_cash_balance(self, account_id: str) -> dict:
        raise NotImplementedError
