from __future__ import annotations

import os

from brokers.us.alpaca_live_client import AlpacaLiveOrderClient
from brokers.us.ibkr_live_client import IbkrLiveOrderClient
from brokers.us.kis_overseas_client import KisOverseasOrderClient
from utils.us_order_client_interface import UsOrderClient


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


class UsLiveOrderClient(UsOrderClient):
    """Phase 7-3 keeps Live broker integration behind a placeholder safety adapter."""

    def _gate_open(self, order_request: dict | None = None) -> tuple[bool, str, str]:
        side = str((order_request or {}).get("side") or "").strip().upper()
        if not _flag("US_MICRO_ALLOW_LIVE", "false"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_MICRO_ALLOW_LIVE=false"
        if _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_MICRO_REAL_ORDER_BLOCKED=true"
        if not _flag("US_LIVE_TRADING_ENABLED", "false"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_LIVE_TRADING_ENABLED=false"
        if not _flag("US_LIVE_ORDER_ENABLED", "false"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_LIVE_ORDER_ENABLED=false"
        if side == "BUY" and not _flag("US_LIVE_BUY_ENABLED", "false"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_LIVE_BUY_ENABLED=false"
        if side == "SELL" and not _flag("US_LIVE_SELL_ENABLED", "false"):
            return False, "LIVE_CLIENT_NOT_ENABLED", "US_LIVE_SELL_ENABLED=false"
        return True, "", ""

    def _resolve_adapter(self) -> tuple[UsOrderClient | None, str | None, str | None]:
        broker_name = _text("US_LIVE_BROKER", "none").upper() or "NONE"
        if broker_name in {"", "NONE"}:
            return None, "broker_not_configured", "US_LIVE_BROKER is not configured."
        if broker_name == "ALPACA":
            return AlpacaLiveOrderClient(), None, None
        if broker_name == "IBKR":
            return IbkrLiveOrderClient(), None, None
        if broker_name == "KIS_OVERSEAS":
            return KisOverseasOrderClient(), None, None
        return None, "live_client_not_configured", f"Unsupported Live broker: {broker_name}"

    def _failed_response(self, *, error_code: str, message: str, broker_order_id: str | None = None) -> dict:
        return {
            "success": False,
            "broker_order_id": broker_order_id,
            "status": "FAILED",
            "error_code": error_code,
            "message": message,
        }

    def submit_order(self, order_request: dict) -> dict:
        gate_open, error_code, message = self._gate_open(order_request)
        if not gate_open:
            return self._failed_response(error_code=error_code, message=message)
        adapter, adapter_error, adapter_message = self._resolve_adapter()
        if adapter is None:
            return self._failed_response(error_code=str(adapter_error), message=str(adapter_message))
        try:
            return adapter.submit_order(order_request)
        except Exception as exc:
            return self._failed_response(error_code="LIVE_CLIENT_NOT_IMPLEMENTED", message=str(exc))

    def cancel_order(self, broker_order_id: str) -> dict:
        gate_open, error_code, message = self._gate_open()
        if not gate_open:
            return self._failed_response(error_code=error_code, message=message, broker_order_id=broker_order_id)
        adapter, adapter_error, adapter_message = self._resolve_adapter()
        if adapter is None:
            return self._failed_response(error_code=str(adapter_error), message=str(adapter_message), broker_order_id=broker_order_id)
        try:
            return adapter.cancel_order(broker_order_id)
        except Exception as exc:
            return self._failed_response(error_code="LIVE_CLIENT_NOT_IMPLEMENTED", message=str(exc), broker_order_id=broker_order_id)

    def get_order_status(self, broker_order_id: str) -> dict:
        gate_open, error_code, message = self._gate_open()
        if not gate_open:
            return self._failed_response(error_code=error_code, message=message, broker_order_id=broker_order_id)
        adapter, adapter_error, adapter_message = self._resolve_adapter()
        if adapter is None:
            return self._failed_response(error_code=str(adapter_error), message=str(adapter_message), broker_order_id=broker_order_id)
        try:
            return adapter.get_order_status(broker_order_id)
        except Exception as exc:
            return self._failed_response(error_code="LIVE_CLIENT_NOT_IMPLEMENTED", message=str(exc), broker_order_id=broker_order_id)

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        gate_open, _, _ = self._gate_open()
        if not gate_open:
            return []
        adapter, _, _ = self._resolve_adapter()
        if adapter is None:
            return []
        try:
            return adapter.get_order_fills(broker_order_id)
        except Exception:
            return []
