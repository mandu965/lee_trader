from __future__ import annotations


class UsOrderClient:
    """Phase 7-1 is a verification stage before any real order API integration."""

    def submit_order(self, order_request: dict) -> dict:
        raise NotImplementedError

    def cancel_order(self, broker_order_id: str) -> dict:
        raise NotImplementedError

    def get_order_status(self, broker_order_id: str) -> dict:
        raise NotImplementedError

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        raise NotImplementedError
