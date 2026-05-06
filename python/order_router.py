"""
order_router.py

Order routing for paper-first trading orchestration.
Important safety rule:
- paper mode logs only
- live mode does not call any broker API in this implementation
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OrderRouteResult:
    """Result returned by the order router."""

    status: str
    message: str
    order: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return asdict(self)


class OrderRouter:
    """Router that logs paper orders and blocks live API routing."""

    def __init__(self, *, mode: str, enable_order_execution: bool, require_manual_approval: bool, output_dir: Path) -> None:
        self.mode = str(mode or "paper").lower()
        self.enable_order_execution = bool(enable_order_execution)
        self.require_manual_approval = bool(require_manual_approval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.orders_log_path = self.output_dir / "orders_log.csv"
        self.trades_log_path = self.output_dir / "trades_log.csv"

    def _append_csv(self, path: Path, row: dict[str, Any]) -> None:
        """Append one dict row to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _manual_approval(self, order: dict[str, Any]) -> bool:
        """Ask for manual approval when enabled."""
        if not self.require_manual_approval:
            return True
        side = str(order.get("side") or "").upper()
        symbol = str(order.get("name") or order.get("symbol") or "")
        quantity = order.get("quantity") or "?"
        try:
            answer = input(f"{side} {symbol} {quantity}주 진행하시겠습니까? (y/n) ").strip().lower()
        except EOFError:
            return False
        return answer == "y"

    def send_order(self, order: dict[str, Any]) -> OrderRouteResult:
        """
        Route an order.

        Safety behavior:
        - paper mode: log-only success
        - live mode: blocked in this implementation
        """
        if not self._manual_approval(order):
            return OrderRouteResult(
                status="rejected",
                message="manual approval rejected",
                order=order,
            )

        if self.mode == "paper" or not self.enable_order_execution:
            return OrderRouteResult(
                status="paper_logged",
                message="order logged in paper mode only",
                order=order,
            )

        return OrderRouteResult(
            status="blocked",
            message="live broker API routing is intentionally not implemented",
            order=order,
        )

    def log_order(self, order: dict[str, Any], result: OrderRouteResult) -> None:
        """Write routed order result to orders/trades logs."""
        log_row = {
            "date": order.get("trade_date"),
            "symbol": order.get("symbol"),
            "name": order.get("name"),
            "side": order.get("side"),
            "quantity": order.get("quantity"),
            "planned_time": order.get("planned_time"),
            "strategy": order.get("strategy"),
            "reason": order.get("reason"),
            "status": result.status,
            "message": result.message,
        }
        self._append_csv(self.orders_log_path, log_row)
        if result.status in {"paper_logged"}:
            self._append_csv(self.trades_log_path, log_row)
