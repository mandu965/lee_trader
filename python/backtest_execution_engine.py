"""
backtest_execution_engine.py

This file is an execution engine for walk-forward backtests.
It must never be connected to live brokerage orders or real-money execution.
The current implementation is a conservative daily-bar approximation engine.
If intraday data is added later, buy_time/sell_time execution can be extended.
To prevent look-ahead bias, this engine never uses price data after trade_date.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


def _as_timestamp(value: Any) -> pd.Timestamp:
    """Convert a date-like input into a normalized pandas Timestamp."""
    return pd.Timestamp(value).normalize()


def _normalize_symbol(value: Any) -> str:
    """Normalize ticker values into zero-padded six-digit strings."""
    return str(value or "").strip().replace(".0", "").zfill(6)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert to float and guard against NaN/invalid values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert to int with a safe fallback."""
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return default
    return numeric


def _pick_price_row(prices_df: pd.DataFrame, trade_date: pd.Timestamp, symbol: str) -> pd.Series | None:
    """Return the same-day price row only; future rows are never considered."""
    if prices_df.empty:
        return None
    work = prices_df.copy()
    if "date" not in work.columns:
        raise ValueError("prices_df must contain a date column")
    if "symbol" not in work.columns:
        if "code" not in work.columns:
            raise ValueError("prices_df must contain either symbol or code")
        work["symbol"] = work["code"].map(_normalize_symbol)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    day = work.loc[
        (work["date"] == trade_date) & (work["symbol"].astype(str).map(_normalize_symbol) == symbol)
    ]
    if day.empty:
        return None
    return day.iloc[-1]


def _first_valid_price(row: pd.Series, columns: list[str]) -> float | None:
    """Return the first valid numeric price from the provided column priority."""
    for column in columns:
        if column not in row.index:
            continue
        value = _safe_float(row.get(column))
        if value is not None and value > 0:
            return value
    return None


@dataclass
class BacktestExecutionConfig:
    """Configuration for daily-bar execution approximation."""

    initial_cash: float = 10_000_000.0
    buy_fee_rate: float = 0.00015
    sell_fee_rate: float = 0.00015
    tax_rate: float = 0.0018
    slippage_rate: float = 0.001
    max_position_count: int = 10
    position_size_mode: str = "equal_weight"
    buy_price_mode: str = "next_open_or_close"
    sell_price_mode: str = "close"
    stop_loss_pct: float | None = None
    trailing_stop_pct: float | None = None
    max_holding_days: int | None = None


@dataclass
class Position:
    """Current position state for one symbol."""

    symbol: str
    name: str
    quantity: int
    avg_price: float
    invested_amount: float
    last_price: float
    entry_date: str | None = None
    highest_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the position."""
        return asdict(self)


@dataclass
class TradeResult:
    """Executed trade result."""

    trade_date: str
    symbol: str
    name: str
    side: str
    planned_time: str
    planned_price: float | None
    executed_price: float | None
    quantity: int
    amount: float
    fee: float
    tax: float
    slippage: float
    reason: str
    strategy: str
    cost_basis_amount: float | None = None
    realized_pnl: float | None = None
    realized_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trade result."""
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    """End-of-day portfolio snapshot."""

    trade_date: str
    cash: float
    equity: float
    position_value: float
    total_value: float
    daily_return: float
    cumulative_return: float
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the portfolio snapshot."""
        return asdict(self)


def get_price_for_execution(
    prices_df: pd.DataFrame,
    trade_date: pd.Timestamp | str,
    symbol: str,
    side: str,
    price_mode: str,
) -> float | None:
    """
    Return the same-day execution reference price for one symbol.

    Look-ahead guard:
    - Only the price row on trade_date is used.
    - No D+1 or later data is inspected.
    - Labels/future returns are never consulted here.
    """
    normalized_date = _as_timestamp(trade_date)
    normalized_symbol = _normalize_symbol(symbol)
    row = _pick_price_row(prices_df, normalized_date, normalized_symbol)
    if row is None:
        return None

    side_upper = str(side or "").upper()
    mode = str(price_mode or "").lower()

    if side_upper == "BUY":
        if mode == "next_open_or_close":
            return _first_valid_price(row, ["open", "adj_open", "close", "adj_close"])
        if mode == "close":
            return _first_valid_price(row, ["close", "adj_close"])
        if mode == "open":
            return _first_valid_price(row, ["open", "adj_open"])
        return _first_valid_price(row, ["open", "adj_open", "close", "adj_close"])

    if side_upper == "SELL":
        if mode == "close":
            return _first_valid_price(row, ["close", "adj_close"])
        if mode == "adjusted_close":
            return _first_valid_price(row, ["adj_close", "close"])
        if mode == "open":
            return _first_valid_price(row, ["open", "adj_open"])
        return _first_valid_price(row, ["close", "adj_close", "open", "adj_open"])

    return None


def apply_slippage(price: float | None, side: str, slippage_rate: float) -> float | None:
    """Apply conservative slippage to the reference price."""
    if price is None or price <= 0:
        return None
    side_upper = str(side or "").upper()
    if side_upper == "BUY":
        return price * (1.0 + slippage_rate)
    if side_upper == "SELL":
        return price * (1.0 - slippage_rate)
    return price


def calculate_costs(amount: float, side: str, config: BacktestExecutionConfig) -> tuple[float, float]:
    """Calculate fee and tax by side."""
    gross_amount = max(_safe_float(amount, 0.0) or 0.0, 0.0)
    side_upper = str(side or "").upper()
    if side_upper == "BUY":
        return gross_amount * config.buy_fee_rate, 0.0
    if side_upper == "SELL":
        return gross_amount * config.sell_fee_rate, gross_amount * config.tax_rate
    return 0.0, 0.0


def value_positions(
    positions: dict[str, Position],
    prices_df: pd.DataFrame,
    trade_date: pd.Timestamp | str,
    skipped: list[dict[str, Any]] | None = None,
) -> float:
    """
    Mark open positions to same-day close.

    If close is unavailable, keep the prior last_price.
    This function does not use future prices.
    """
    normalized_date = _as_timestamp(trade_date)
    total_value = 0.0
    for symbol, position in positions.items():
        row = _pick_price_row(prices_df, normalized_date, symbol)
        mark_price = None
        if row is not None:
            mark_price = _first_valid_price(row, ["close", "adj_close", "open", "adj_open"])
        if mark_price is None:
            mark_price = _safe_float(position.last_price)
            if skipped is not None:
                skipped.append({
                    "trade_date": normalized_date.strftime("%Y-%m-%d"),
                    "reason": "position_mark_price_missing",
                    "detail": f"Using last_price fallback for symbol={symbol}",
                })
        if mark_price is None or mark_price <= 0:
            if skipped is not None:
                skipped.append({
                    "trade_date": normalized_date.strftime("%Y-%m-%d"),
                    "reason": "position_unvaluable",
                    "detail": f"Could not value symbol={symbol}",
                })
            continue
        position.last_price = float(mark_price)
        total_value += position.quantity * position.last_price
    return float(total_value)


def create_portfolio_snapshot(
    trade_date: pd.Timestamp | str,
    cash: float,
    position_value: float,
    previous_total_value: float | None,
    initial_cash: float,
    strategy: str,
) -> PortfolioSnapshot:
    """Build one end-of-day portfolio snapshot row."""
    normalized_date = _as_timestamp(trade_date)
    cash_value = max(_safe_float(cash, 0.0) or 0.0, 0.0)
    position_value = max(_safe_float(position_value, 0.0) or 0.0, 0.0)
    total_value = cash_value + position_value
    if previous_total_value is None or previous_total_value <= 0:
        daily_return = 0.0
    else:
        daily_return = total_value / previous_total_value - 1.0
    cumulative_return = 0.0 if initial_cash <= 0 else total_value / initial_cash - 1.0
    return PortfolioSnapshot(
        trade_date=normalized_date.strftime("%Y-%m-%d"),
        cash=round(cash_value, 2),
        equity=round(total_value, 2),
        position_value=round(position_value, 2),
        total_value=round(total_value, 2),
        daily_return=float(daily_return),
        cumulative_return=float(cumulative_return),
        strategy=str(strategy),
    )


@dataclass
class ExecutionEngine:
    """Stateful daily-bar execution engine for walk-forward backtests."""

    config: BacktestExecutionConfig = field(default_factory=BacktestExecutionConfig)
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    trades: list[TradeResult] = field(default_factory=list, init=False)
    portfolio_snapshots: list[PortfolioSnapshot] = field(default_factory=list, init=False)
    skipped: list[dict[str, Any]] = field(default_factory=list, init=False)
    previous_total_value: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize engine state from config."""
        self.cash = float(self.config.initial_cash)

    def _current_total_value(self, trade_date: pd.Timestamp, prices_df: pd.DataFrame) -> float:
        """Return cash plus marked position value on the current trade date."""
        position_value = value_positions(self.positions, prices_df, trade_date)
        return float(self.cash + position_value)

    def _target_buy_amount(self, trade_date: pd.Timestamp, prices_df: pd.DataFrame, order: dict[str, Any]) -> float:
        """
        Determine the gross capital target for a buy order.

        Default sizing is equal-weight using current total account value.
        If the order already includes a positive amount, that explicit plan wins.
        """
        explicit_amount = _safe_float(order.get("amount"))
        if explicit_amount is not None and explicit_amount > 0:
            return float(explicit_amount)
        if self.config.position_size_mode != "equal_weight":
            raise ValueError(f"Unsupported position_size_mode: {self.config.position_size_mode}")
        total_value = self._current_total_value(trade_date, prices_df)
        return max(total_value / max(self.config.max_position_count, 1), 0.0)

    def execute_buy_order(
        self,
        order: dict[str, Any],
        prices_df: pd.DataFrame,
        trade_date: pd.Timestamp | str,
        strategy: str,
    ) -> TradeResult | None:
        """
        Execute one buy order under cash and sizing constraints.

        Duplicate buys for already-held symbols are currently skipped.
        TODO: add controlled averaging/re-entry rules if needed later.
        """
        normalized_date = _as_timestamp(trade_date)
        symbol = _normalize_symbol(order.get("symbol"))
        name = str(order.get("name") or symbol)

        if not symbol:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "buy_missing_symbol",
                "detail": "BUY order missing symbol",
            })
            return None
        if symbol in self.positions:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "buy_duplicate_position",
                "detail": f"Symbol already held: {symbol}",
            })
            return None
        if len(self.positions) >= self.config.max_position_count:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "max_position_count_reached",
                "detail": f"Cannot add {symbol}; max_position_count={self.config.max_position_count}",
            })
            return None

        planned_price = get_price_for_execution(
            prices_df,
            normalized_date,
            symbol,
            "BUY",
            self.config.buy_price_mode,
        )
        if planned_price is None:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "buy_price_missing",
                "detail": f"No same-day BUY price for symbol={symbol}",
            })
            return None

        executed_price = apply_slippage(planned_price, "BUY", self.config.slippage_rate)
        if executed_price is None or executed_price <= 0:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "buy_executed_price_invalid",
                "detail": f"Invalid executed BUY price for symbol={symbol}",
            })
            return None

        requested_quantity = _safe_int(order.get("quantity"), 0)
        target_amount = self._target_buy_amount(normalized_date, prices_df, order)
        max_qty_by_target = int(target_amount // executed_price) if target_amount > 0 else 0

        if requested_quantity > 0:
            quantity = min(requested_quantity, max_qty_by_target or requested_quantity)
        else:
            quantity = max_qty_by_target

        while quantity > 0:
            gross_amount = executed_price * quantity
            fee, tax = calculate_costs(gross_amount, "BUY", self.config)
            total_cost = gross_amount + fee + tax
            if total_cost <= self.cash + 1e-9:
                break
            quantity -= 1

        if quantity <= 0:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "buy_cash_insufficient",
                "detail": f"Insufficient cash to buy at least 1 share of {symbol}",
            })
            return None

        gross_amount = executed_price * quantity
        fee, tax = calculate_costs(gross_amount, "BUY", self.config)
        total_cost = gross_amount + fee + tax
        slippage_cost = max((executed_price - planned_price) * quantity, 0.0)

        self.cash -= total_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            name=name,
            quantity=quantity,
            avg_price=float(executed_price),
            invested_amount=float(gross_amount),
            last_price=float(executed_price),
            entry_date=normalized_date.strftime("%Y-%m-%d"),
            highest_price=float(executed_price),
        )

        trade = TradeResult(
            trade_date=normalized_date.strftime("%Y-%m-%d"),
            symbol=symbol,
            name=name,
            side="BUY",
            planned_time=str(order.get("planned_time") or ""),
            planned_price=float(planned_price),
            executed_price=float(executed_price),
            quantity=int(quantity),
            amount=round(gross_amount, 2),
            fee=round(fee, 2),
            tax=round(tax, 2),
            slippage=round(slippage_cost, 2),
            reason=str(order.get("reason") or "buy_order"),
            strategy=str(strategy),
            cost_basis_amount=round(gross_amount, 2),
            realized_pnl=None,
            realized_return=None,
        )
        self.trades.append(trade)
        return trade

    def execute_sell_order(
        self,
        order: dict[str, Any],
        prices_df: pd.DataFrame,
        trade_date: pd.Timestamp | str,
        strategy: str,
    ) -> TradeResult | None:
        """Execute one sell order against an existing position."""
        normalized_date = _as_timestamp(trade_date)
        symbol = _normalize_symbol(order.get("symbol"))
        if symbol not in self.positions:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "sell_position_missing",
                "detail": f"No held position to sell: {symbol}",
            })
            return None

        position = self.positions[symbol]
        planned_price = get_price_for_execution(
            prices_df,
            normalized_date,
            symbol,
            "SELL",
            self.config.sell_price_mode,
        )
        if planned_price is None:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "sell_price_missing",
                "detail": f"No same-day SELL price for symbol={symbol}",
            })
            return None

        executed_price = apply_slippage(planned_price, "SELL", self.config.slippage_rate)
        if executed_price is None or executed_price <= 0:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "sell_executed_price_invalid",
                "detail": f"Invalid executed SELL price for symbol={symbol}",
            })
            return None

        requested_quantity = _safe_int(order.get("quantity"), 0)
        quantity = position.quantity if requested_quantity <= 0 else min(requested_quantity, position.quantity)
        if quantity <= 0:
            self.skipped.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "sell_quantity_invalid",
                "detail": f"Computed sell quantity <= 0 for {symbol}",
            })
            return None

        gross_amount = executed_price * quantity
        fee, tax = calculate_costs(gross_amount, "SELL", self.config)
        net_proceeds = gross_amount - fee - tax
        slippage_cost = max((planned_price - executed_price) * quantity, 0.0)
        cost_basis_amount = position.avg_price * quantity
        realized_pnl = net_proceeds - cost_basis_amount
        realized_return = None
        if cost_basis_amount > 0:
            realized_return = realized_pnl / cost_basis_amount

        self.cash += net_proceeds
        remaining_quantity = position.quantity - quantity
        if remaining_quantity <= 0:
            self.positions.pop(symbol, None)
        else:
            position.quantity = remaining_quantity
            position.invested_amount = position.avg_price * remaining_quantity
            position.last_price = float(executed_price)

        trade = TradeResult(
            trade_date=normalized_date.strftime("%Y-%m-%d"),
            symbol=symbol,
            name=position.name,
            side="SELL",
            planned_time=str(order.get("planned_time") or ""),
            planned_price=float(planned_price),
            executed_price=float(executed_price),
            quantity=int(quantity),
            amount=round(gross_amount, 2),
            fee=round(fee, 2),
            tax=round(tax, 2),
            slippage=round(slippage_cost, 2),
            reason=str(order.get("reason") or "sell_order"),
            strategy=str(strategy),
            cost_basis_amount=round(cost_basis_amount, 2),
            realized_pnl=round(realized_pnl, 2),
            realized_return=float(realized_return) if realized_return is not None else None,
        )
        self.trades.append(trade)
        return trade

    def execute_daily_orders(
        self,
        trade_date: pd.Timestamp | str,
        orders: list[dict[str, Any]],
        prices_df: pd.DataFrame,
        strategy: str,
    ) -> list[TradeResult]:
        """
        Execute one day's order list.

        Current ordering rule is BUY first, SELL later, matching the requested
        skeleton contract. TODO: if same-day sell-then-buy rotation is needed,
        add explicit sequencing rules or separate session buckets.
        """
        normalized_date = _as_timestamp(trade_date)
        day_orders = []
        for order in orders or []:
            if _as_timestamp(order.get("trade_date")) != normalized_date:
                continue
            day_orders.append(dict(order))

        side_priority = {"BUY": 0, "SELL": 1}
        day_orders.sort(
            key=lambda item: (
                side_priority.get(str(item.get("side") or "").upper(), 9),
                str(item.get("planned_time") or ""),
                _normalize_symbol(item.get("symbol")),
            )
        )

        executed_today: list[TradeResult] = []
        for order in day_orders:
            try:
                side = str(order.get("side") or "").upper()
                if side == "BUY":
                    result = self.execute_buy_order(order, prices_df, normalized_date, strategy)
                elif side == "SELL":
                    result = self.execute_sell_order(order, prices_df, normalized_date, strategy)
                else:
                    self.skipped.append({
                        "trade_date": normalized_date.strftime("%Y-%m-%d"),
                        "reason": "unknown_order_side",
                        "detail": f"Unsupported side={order.get('side')}",
                    })
                    result = None
                if result is not None:
                    executed_today.append(result)
            except Exception as exc:
                self.skipped.append({
                    "trade_date": normalized_date.strftime("%Y-%m-%d"),
                    "reason": "order_execution_exception",
                    "detail": f"{_normalize_symbol(order.get('symbol'))}: {exc}",
                })
        return executed_today

    def create_snapshot(
        self,
        trade_date: pd.Timestamp | str,
        prices_df: pd.DataFrame,
        strategy: str,
    ) -> PortfolioSnapshot:
        """Create and store the end-of-day portfolio snapshot."""
        normalized_date = _as_timestamp(trade_date)
        position_value = value_positions(self.positions, prices_df, normalized_date, self.skipped)
        snapshot = create_portfolio_snapshot(
            trade_date=normalized_date,
            cash=self.cash,
            position_value=position_value,
            previous_total_value=self.previous_total_value,
            initial_cash=self.config.initial_cash,
            strategy=strategy,
        )
        self.previous_total_value = snapshot.total_value
        self.portfolio_snapshots.append(snapshot)
        return snapshot

    def build_risk_exit_orders(
        self,
        trade_date: pd.Timestamp | str,
        prices_df: pd.DataFrame,
        strategy: str,
    ) -> list[dict[str, Any]]:
        """
        Build same-day close-based risk exit orders.

        Look-ahead guard:
        - only same-day price rows are used
        - no future rows or labels are referenced
        """
        normalized_date = _as_timestamp(trade_date)
        orders: list[dict[str, Any]] = []

        for symbol, position in list(self.positions.items()):
            row = _pick_price_row(prices_df, normalized_date, symbol)
            close_price = None
            if row is not None:
                close_price = _first_valid_price(row, ["close", "adj_close", "open", "adj_open"])
            if close_price is None or close_price <= 0:
                continue

            if position.highest_price is None:
                position.highest_price = close_price
            else:
                position.highest_price = max(float(position.highest_price), float(close_price))

            entry_price = _safe_float(position.avg_price, 0.0) or 0.0
            holding_days = 0
            if position.entry_date:
                holding_days = max((normalized_date - _as_timestamp(position.entry_date)).days, 0)
            current_return = close_price / entry_price - 1.0 if entry_price > 0 else 0.0
            trailing_drawdown = 0.0
            if position.highest_price and position.highest_price > 0:
                trailing_drawdown = close_price / float(position.highest_price) - 1.0

            reason = None
            if self.config.stop_loss_pct is not None and current_return <= -float(self.config.stop_loss_pct):
                reason = "stop_loss"
            elif (
                self.config.trailing_stop_pct is not None
                and trailing_drawdown <= -float(self.config.trailing_stop_pct)
                and current_return > 0
            ):
                reason = "trailing_stop_exit"
            elif self.config.max_holding_days is not None and holding_days >= int(self.config.max_holding_days):
                reason = "max_holding_days_exit"

            if reason is None:
                continue

            orders.append({
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": position.name,
                "side": "SELL",
                "planned_time": "15:10",
                "planned_price": float(close_price),
                "quantity": int(position.quantity),
                "strategy": str(strategy),
                "reason": reason,
            })
        return orders

    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of engine state."""
        return {
            "cash": round(self.cash, 2),
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "trades": [trade.to_dict() for trade in self.trades],
            "portfolio_snapshots": [snapshot.to_dict() for snapshot in self.portfolio_snapshots],
            "skipped": list(self.skipped),
        }


def execute_daily_orders(
    engine: ExecutionEngine,
    trade_date: pd.Timestamp | str,
    orders: list[dict[str, Any]],
    prices_df: pd.DataFrame,
    strategy: str,
) -> list[TradeResult]:
    """Module-level wrapper for the engine execute_daily_orders method."""
    return engine.execute_daily_orders(trade_date, orders, prices_df, strategy)
