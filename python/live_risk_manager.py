"""
live_risk_manager.py

Risk checks for paper/live trading orchestration.
This module does not place orders. It only validates whether an order should
be blocked before routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RiskDecision:
    """Structured result of one risk validation."""

    allowed: bool
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision."""
        return asdict(self)


@dataclass
class LiveRiskManager:
    """Portfolio/order risk gate for paper-first live trading flows."""

    max_daily_loss: float
    max_position_per_stock: float
    max_total_positions: int
    initial_cash: float
    risk_events: list[dict[str, Any]] = field(default_factory=list)

    def _record_event(self, trade_date: str, symbol: str, reason: str, blocked_order: dict[str, Any], detail: str = "") -> None:
        """Append one blocked-order event."""
        self.risk_events.append({
            "date": trade_date,
            "symbol": symbol,
            "reason": reason,
            "blocked_order": blocked_order,
            "detail": detail,
        })

    def check_daily_loss(self, current_pnl: float) -> RiskDecision:
        """Block new orders if daily loss breaches the configured threshold."""
        limit_amount = float(self.initial_cash) * float(self.max_daily_loss)
        if current_pnl < 0 and abs(float(current_pnl)) > limit_amount:
            return RiskDecision(
                allowed=False,
                reason="daily_loss_limit_exceeded",
                detail=f"current_pnl={current_pnl:.2f}, limit_amount={limit_amount:.2f}",
            )
        return RiskDecision(allowed=True, reason="ok")

    def check_position_limit(self, symbol: str, current_positions: dict[str, Any], total_value: float) -> RiskDecision:
        """Block BUY orders when one symbol already exceeds configured weight."""
        if symbol not in current_positions:
            return RiskDecision(allowed=True, reason="ok")
        position = current_positions.get(symbol, {})
        quantity = float(position.get("quantity") or 0.0)
        last_price = float(position.get("last_price") or 0.0)
        if total_value <= 0:
            return RiskDecision(allowed=True, reason="ok")
        weight = quantity * last_price / total_value
        if weight > float(self.max_position_per_stock):
            return RiskDecision(
                allowed=False,
                reason="position_limit_exceeded",
                detail=f"symbol={symbol}, current_weight={weight:.4f}, max={self.max_position_per_stock:.4f}",
            )
        return RiskDecision(allowed=True, reason="ok")

    def check_total_positions(self, current_positions: dict[str, Any]) -> RiskDecision:
        """Block new BUY orders if total position count already meets the cap."""
        total_positions = len(current_positions)
        if total_positions >= int(self.max_total_positions):
            return RiskDecision(
                allowed=False,
                reason="total_positions_limit_exceeded",
                detail=f"total_positions={total_positions}, max={self.max_total_positions}",
            )
        return RiskDecision(allowed=True, reason="ok")

    def validate_order(self, order: dict[str, Any], portfolio_state: dict[str, Any]) -> RiskDecision:
        """Run all pre-routing risk checks for one order."""
        side = str(order.get("side") or "").upper()
        symbol = str(order.get("symbol") or "")
        trade_date = str(order.get("trade_date") or "")
        current_positions = portfolio_state.get("positions", {}) or {}
        current_total_value = float(portfolio_state.get("total_value") or portfolio_state.get("cash") or self.initial_cash)
        current_daily_pnl = float(portfolio_state.get("daily_pnl") or 0.0)

        daily_loss_decision = self.check_daily_loss(current_daily_pnl)
        if not daily_loss_decision.allowed:
            self._record_event(trade_date, symbol, daily_loss_decision.reason, order, daily_loss_decision.detail)
            return daily_loss_decision

        if side == "BUY":
            total_positions_decision = self.check_total_positions(current_positions)
            if not total_positions_decision.allowed:
                self._record_event(trade_date, symbol, total_positions_decision.reason, order, total_positions_decision.detail)
                return total_positions_decision

            position_limit_decision = self.check_position_limit(symbol, current_positions, current_total_value)
            if not position_limit_decision.allowed:
                self._record_event(trade_date, symbol, position_limit_decision.reason, order, position_limit_decision.detail)
                return position_limit_decision

        return RiskDecision(allowed=True, reason="ok")
