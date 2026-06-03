from __future__ import annotations

import os
from datetime import datetime, timezone

from utils.us_order_client_interface import UsOrderClient


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _symbol_set(name: str) -> set[str]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return set()
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


class UsMockOrderClient(UsOrderClient):
    """Phase 7-1 uses mock responses only and does not call any external broker API."""

    def submit_order(self, order_request: dict) -> dict:
        symbol = str(order_request.get("symbol") or "").strip().upper()
        order_type = str(order_request.get("order_type") or "").strip().upper()
        order_qty = order_request.get("order_qty")
        order_amount = order_request.get("order_amount_usd")
        limit_price = order_request.get("limit_price")

        if _flag("US_MICRO_MOCK_FORCE_FAIL", "false") or symbol in _symbol_set("US_MICRO_MOCK_FAIL_SYMBOLS"):
            return {
                "success": False,
                "broker_order_id": None,
                "status": "FAILED",
                "error_code": "MOCK_FAILED",
                "message": "Mock failure triggered by configured scenario",
                "filled_qty": 0,
                "filled_price": None,
            }

        reject_reasons: list[str] = []
        if _flag("US_MICRO_MOCK_FORCE_REJECT", "false") or symbol in _symbol_set("US_MICRO_MOCK_REJECT_SYMBOLS"):
            reject_reasons.append("configured_reject_scenario")
        if order_type not in {"LIMIT", "MARKET"}:
            reject_reasons.append("unsupported_order_type")
        try:
            if order_qty is not None and float(order_qty) <= 0:
                reject_reasons.append("non_positive_order_qty")
        except (TypeError, ValueError):
            reject_reasons.append("invalid_order_qty")
        try:
            if order_amount is not None and float(order_amount) <= 0:
                reject_reasons.append("non_positive_order_amount_usd")
        except (TypeError, ValueError):
            reject_reasons.append("invalid_order_amount_usd")
        try:
            if limit_price is not None and float(limit_price) <= 0:
                reject_reasons.append("non_positive_limit_price")
        except (TypeError, ValueError):
            reject_reasons.append("invalid_limit_price")

        if reject_reasons:
            return {
                "success": False,
                "broker_order_id": None,
                "status": "REJECTED",
                "error_code": "MOCK_REJECTED",
                "message": "Mock rejected by configured scenario",
                "reason_codes": reject_reasons,
                "filled_qty": 0,
                "filled_price": None,
            }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return {
            "success": True,
            "broker_order_id": f"MOCK_{symbol}_{timestamp}",
            "status": "ACCEPTED",
            "message": "Mock order accepted",
            "filled_qty": 0,
            "filled_price": None,
        }

    def cancel_order(self, broker_order_id: str) -> dict:
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": "CANCELED",
            "message": "Mock order canceled",
        }

    def get_order_status(self, broker_order_id: str) -> dict:
        status = str(os.environ.get("US_MICRO_MOCK_STATUS", "ACCEPTED")).strip().upper() or "ACCEPTED"
        filled_qty = float(os.environ.get("US_MICRO_MOCK_FILLED_QTY", "0") or 0)
        filled_price = os.environ.get("US_MICRO_MOCK_FILLED_PRICE")
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": status,
            "message": "Mock order status lookup",
            "filled_qty": filled_qty,
            "filled_price": float(filled_price) if filled_price not in {None, ""} else None,
        }

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        filled_qty = float(os.environ.get("US_MICRO_MOCK_FILLED_QTY", "0") or 0)
        if filled_qty <= 0:
            return []
        filled_price = float(os.environ.get("US_MICRO_MOCK_FILLED_PRICE", "100") or 100)
        fill_time = os.environ.get("US_MICRO_MOCK_FILL_TIME", "2026-05-15T15:30:00Z")
        filled_amount = round(filled_qty * filled_price, 6)
        return [
            {
                "broker_fill_id": f"MOCK_FILL_{broker_order_id}",
                "filled_qty": filled_qty,
                "filled_price": filled_price,
                "filled_amount_usd": filled_amount,
                "commission_usd": 0,
                "fee_usd": 0,
                "fill_time": fill_time,
            }
        ]
