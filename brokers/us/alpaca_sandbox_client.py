from __future__ import annotations

from brokers.us.base import UsOrderClient


class AlpacaSandboxOrderClient(UsOrderClient):
    def submit_order(self, order_request: dict) -> dict:
        raise RuntimeError("Alpaca sandbox adapter is not implemented in this project baseline.")

    def cancel_order(self, broker_order_id: str) -> dict:
        raise RuntimeError("Alpaca sandbox adapter is not implemented in this project baseline.")

    def get_order_status(self, broker_order_id: str) -> dict:
        raise RuntimeError("Alpaca sandbox adapter is not implemented in this project baseline.")

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        raise RuntimeError("Alpaca sandbox adapter is not implemented in this project baseline.")
