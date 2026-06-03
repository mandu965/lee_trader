from __future__ import annotations

import os
from datetime import datetime, timezone

from utils.us_order_client_interface import UsOrderClient


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


class UsSandboxOrderClient(UsOrderClient):
    """SANDBOX is not a live-account order and remains isolated from real-order execution."""

    def _configured(self) -> tuple[bool, str]:
        broker_name = _text("US_SANDBOX_BROKER_NAME", "NONE").upper() or "NONE"
        if broker_name == "NONE":
            return False, "SANDBOX_NOT_CONFIGURED"
        if not _flag("US_MICRO_ALLOW_SANDBOX", "false"):
            return False, "SANDBOX_NOT_ALLOWED"
        if not _flag("US_SANDBOX_ORDER_ENABLED", "false"):
            return False, "SANDBOX_ORDER_DISABLED"
        if not _text("US_SANDBOX_BASE_URL"):
            return False, "SANDBOX_BASE_URL_MISSING"
        if not _text("US_SANDBOX_API_KEY"):
            return False, "SANDBOX_API_KEY_MISSING"
        if not _text("US_SANDBOX_API_SECRET"):
            return False, "SANDBOX_API_SECRET_MISSING"
        return True, ""

    def submit_order(self, order_request: dict) -> dict:
        ok, reason = self._configured()
        if not ok:
            return {
                "success": False,
                "broker_order_id": None,
                "status": "FAILED",
                "error_code": reason,
                "message": "Sandbox broker is not configured",
                "raw_response": {},
            }
        symbol = str(order_request.get("symbol") or "").strip().upper()
        return {
            "success": True,
            "broker_order_id": f"SANDBOX_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "status": "ACCEPTED",
            "message": "Sandbox order accepted",
            "raw_response": {"placeholder": True, "note": "Broker-specific sandbox integration remains pending."},
        }

    def cancel_order(self, broker_order_id: str) -> dict:
        ok, reason = self._configured()
        if not ok:
            return {
                "success": False,
                "broker_order_id": broker_order_id,
                "status": "FAILED",
                "error_code": reason,
                "message": "Sandbox broker is not configured",
                "raw_response": {},
            }
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": "CANCELED",
            "message": "Sandbox order canceled",
            "raw_response": {"placeholder": True},
        }

    def get_order_status(self, broker_order_id: str) -> dict:
        ok, reason = self._configured()
        if not ok:
            return {
                "success": False,
                "broker_order_id": broker_order_id,
                "status": "FAILED",
                "error_code": reason,
                "message": "Sandbox broker is not configured",
                "raw_response": {},
            }
        status = _text("US_SANDBOX_MOCK_STATUS", "ACCEPTED").upper() or "ACCEPTED"
        filled_qty = float(_text("US_SANDBOX_MOCK_FILLED_QTY", "0") or 0)
        filled_price = _text("US_SANDBOX_MOCK_FILLED_PRICE")
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": status,
            "message": "Sandbox order status retrieved",
            "filled_qty": filled_qty,
            "filled_price": float(filled_price) if filled_price else None,
            "raw_response": {"placeholder": True, "state": "working"},
        }

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        ok, _ = self._configured()
        if not ok:
            return []
        filled_qty = float(_text("US_SANDBOX_MOCK_FILLED_QTY", "0") or 0)
        if filled_qty <= 0:
            return []
        filled_price = float(_text("US_SANDBOX_MOCK_FILLED_PRICE", "100") or 100)
        fill_time = _text("US_SANDBOX_MOCK_FILL_TIME", "2026-05-15T15:30:00Z")
        return [
            {
                "broker_fill_id": f"SANDBOX_FILL_{broker_order_id}",
                "filled_qty": filled_qty,
                "filled_price": filled_price,
                "filled_amount_usd": round(filled_qty * filled_price, 6),
                "commission_usd": 0,
                "fee_usd": 0,
                "fill_time": fill_time,
                "raw_fill_payload": {"placeholder": True},
            }
        ]
