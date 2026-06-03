from __future__ import annotations

import os

from utils.us_broker_account_interface import UsBrokerAccountClient


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


class UsLiveAccountClient(UsBrokerAccountClient):
    """Live account lookup remains blocked by default in Phase 7-5."""

    def _assert_allowed(self) -> None:
        if not _flag("US_MICRO_RECON_ENABLED", "false"):
            raise RuntimeError("US_MICRO_RECON_ENABLED must be true for Live reconciliation.")
        if not _flag("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", "false"):
            raise RuntimeError("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY must be true for Live account queries.")
        if not _flag("US_MICRO_RECON_REAL_ORDER_BLOCKED", "true"):
            raise RuntimeError("US_MICRO_RECON_REAL_ORDER_BLOCKED must remain true for Live reconciliation.")

    def get_account_snapshot(self, account_id: str) -> dict:
        self._assert_allowed()
        raise RuntimeError("Live account snapshot integration is not implemented in Phase 7-5.")

    def get_positions(self, account_id: str) -> list[dict]:
        self._assert_allowed()
        raise RuntimeError("Live position integration is not implemented in Phase 7-5.")

    def get_cash_balance(self, account_id: str) -> dict:
        self._assert_allowed()
        raise RuntimeError("Live cash-balance integration is not implemented in Phase 7-5.")
