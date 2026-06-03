from __future__ import annotations

import json
import os

from utils.us_broker_account_interface import UsBrokerAccountClient


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _json_env(name: str, default):
    raw = _text(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


class UsSandboxAccountClient(UsBrokerAccountClient):
    def _configured(self) -> tuple[bool, str]:
        broker_name = _text("US_SANDBOX_BROKER_NAME", "NONE").upper() or "NONE"
        if broker_name == "NONE":
            return False, "SANDBOX_NOT_CONFIGURED"
        if not _flag("US_MICRO_ALLOW_SANDBOX", "false"):
            return False, "SANDBOX_NOT_ALLOWED"
        if not _flag("US_SANDBOX_ORDER_ENABLED", "false"):
            return False, "SANDBOX_ORDER_DISABLED"
        return True, ""

    def get_account_snapshot(self, account_id: str) -> dict:
        ok, reason = self._configured()
        configured = _json_env("US_SANDBOX_ACCOUNT_SNAPSHOT_JSON", {})
        if not ok:
            return {"success": False, "account_id": account_id, "error_code": reason, "raw_payload": {}}
        cash = _safe_float(configured.get("cash_balance") if isinstance(configured, dict) else None, _safe_float(os.environ.get("US_SANDBOX_MOCK_CASH_USD"), 1000.0))
        market_value = _safe_float(configured.get("market_value") if isinstance(configured, dict) else None, _safe_float(os.environ.get("US_SANDBOX_MOCK_MARKET_VALUE_USD"), 0.0))
        equity_value = _safe_float(configured.get("equity_value") if isinstance(configured, dict) else None, None)
        if equity_value is None:
            equity_value = round((cash or 0.0) + (market_value or 0.0), 6)
        return {
            "success": True,
            "account_id": account_id,
            "cash_balance": cash,
            "market_value": market_value,
            "equity_value": equity_value,
            "raw_payload": configured if isinstance(configured, dict) else {"placeholder": True},
        }

    def get_positions(self, account_id: str) -> list[dict]:
        ok, _ = self._configured()
        if not ok:
            return []
        positions = _json_env("US_SANDBOX_POSITIONS_JSON", [])
        if not isinstance(positions, list):
            return []
        rows: list[dict] = []
        for item in positions:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "account_id": account_id,
                    "symbol": str(item.get("symbol") or "").strip().upper(),
                    "qty": _safe_float(item.get("qty")),
                    "market_value": _safe_float(item.get("market_value")),
                    "avg_price": _safe_float(item.get("avg_price")),
                    "raw_payload": item,
                }
            )
        return rows

    def get_cash_balance(self, account_id: str) -> dict:
        snapshot = self.get_account_snapshot(account_id)
        return {
            "account_id": account_id,
            "cash_balance": snapshot.get("cash_balance"),
            "currency": "USD",
            "success": snapshot.get("success", False),
            "raw_payload": snapshot.get("raw_payload") or {},
        }
