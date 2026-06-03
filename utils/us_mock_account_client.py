from __future__ import annotations

import json
import os

from utils.us_broker_account_interface import UsBrokerAccountClient


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_env(name: str, default):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


class UsMockAccountClient(UsBrokerAccountClient):
    """Mock account state for reconciliation-only verification."""

    def get_account_snapshot(self, account_id: str) -> dict:
        configured = _json_env("US_MICRO_MOCK_ACCOUNT_SNAPSHOT_JSON", {})
        cash = _safe_float(configured.get("cash_balance") if isinstance(configured, dict) else None, _safe_float(os.environ.get("US_MICRO_MOCK_CASH_USD"), 1000.0))
        market_value = _safe_float(configured.get("market_value") if isinstance(configured, dict) else None, _safe_float(os.environ.get("US_MICRO_MOCK_MARKET_VALUE_USD"), 0.0))
        equity_value = _safe_float(configured.get("equity_value") if isinstance(configured, dict) else None, None)
        if equity_value is None:
            equity_value = round((cash or 0.0) + (market_value or 0.0), 6)
        return {
            "account_id": account_id,
            "cash_balance": cash,
            "market_value": market_value,
            "equity_value": equity_value,
            "buying_power": _safe_float(configured.get("buying_power") if isinstance(configured, dict) else None, cash),
            "raw_payload": configured if isinstance(configured, dict) else {},
        }

    def get_positions(self, account_id: str) -> list[dict]:
        positions = _json_env("US_MICRO_MOCK_POSITIONS_JSON", [])
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
            "raw_payload": snapshot.get("raw_payload") or {},
        }
