from __future__ import annotations

from brokers.us.base import UsOrderClient


class AlpacaLiveOrderClient(UsOrderClient):
    def submit_order(self, order_request: dict) -> dict:
        raise RuntimeError("Live Alpaca order client is not implemented in Phase 7-3.")

    def cancel_order(self, broker_order_id: str) -> dict:
        raise RuntimeError("Live Alpaca order client is not implemented in Phase 7-3.")

    def get_order_status(self, broker_order_id: str) -> dict:
        raise RuntimeError("Live Alpaca order client is not implemented in Phase 7-3.")

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        raise RuntimeError("Live Alpaca order client is not implemented in Phase 7-3.")
