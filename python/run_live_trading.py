"""
run_live_trading.py

Paper-first trading runner with mandatory pre-order risk checks.

Safety rules in this implementation:
- no broker API calls
- paper mode only produces logs
- live mode is loaded but actual broker routing remains blocked
- risk manager validation always runs before routing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_execution_engine import BacktestExecutionConfig, ExecutionEngine
from live_risk_manager import LiveRiskManager
from order_router import OrderRouter
from run_forward_test import (
    DEFAULT_CONFIGS_JSON,
    build_rebalance_orders,
    load_forward_configs,
    load_rule_candidates_for_day,
)
from walk_forward_backtest import DEFAULT_FEATURES_CSV, DEFAULT_PRICES_CSV, DEFAULT_RULE_SIGNALS_CSV, _resolve_path, load_features, load_prices, load_rule_signals


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_CONFIG_JSON = ROOT / "configs" / "live_trading_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "live_trading"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run paper-first live trading orchestration.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=10_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live-config-json", type=Path, default=DEFAULT_LIVE_CONFIG_JSON)
    parser.add_argument("--forward-configs-json", type=Path, default=DEFAULT_CONFIGS_JSON)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES_CSV)
    parser.add_argument("--rule-signals-csv", type=Path, default=DEFAULT_RULE_SIGNALS_CSV)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object file."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"config file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {resolved}")
    return payload


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one row to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_portfolio_state(engine: ExecutionEngine, trade_date: pd.Timestamp) -> dict[str, Any]:
    """Construct a minimal portfolio state object for risk validation."""
    position_value = 0.0
    for position in engine.positions.values():
        last_price = float(position.last_price or 0.0)
        quantity = int(position.quantity or 0)
        position_value += last_price * quantity
    total_value = float(engine.cash) + float(position_value)
    daily_pnl = 0.0
    if engine.portfolio_snapshots:
        last_total = float(engine.portfolio_snapshots[-1].total_value)
        daily_pnl = total_value - last_total
    return {
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "cash": float(engine.cash),
        "positions": {symbol: position.to_dict() for symbol, position in engine.positions.items()},
        "position_value": float(position_value),
        "total_value": float(total_value),
        "daily_pnl": float(daily_pnl),
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    live_config = _load_json(args.live_config_json)
    profiles = load_forward_configs(args.forward_configs_json)
    allowed_profiles = {str(item).strip() for item in live_config.get("allowed_strategies", [])}
    profiles = [profile for profile in profiles if profile.profile in allowed_profiles]
    if not profiles:
        raise ValueError("No allowed forward profiles matched live_trading_config.json")

    features_df = load_features(_resolve_path(args.features_csv))
    prices_df = load_prices(_resolve_path(args.prices_csv))
    rule_signals_df = load_rule_signals(_resolve_path(args.rule_signals_csv))
    prices_df["open"] = pd.to_numeric(prices_df.get("adj_open"), errors="coerce")
    prices_df["close"] = pd.to_numeric(prices_df.get("adj_close"), errors="coerce")

    latest_trade_date = pd.Timestamp(prices_df["date"].max()).normalize()
    if start_date > latest_trade_date:
        raise ValueError(f"start-date {start_date.date()} is after latest available price date {latest_trade_date.date()}")

    risk_manager = LiveRiskManager(
        max_daily_loss=float(live_config.get("max_daily_loss", 0.03)),
        max_position_per_stock=float(live_config.get("max_position_per_stock", 0.1)),
        max_total_positions=int(live_config.get("max_total_positions", 10)),
        initial_cash=float(live_config.get("initial_cash", args.initial_cash)),
    )
    router = OrderRouter(
        mode=str(live_config.get("mode", "paper")),
        enable_order_execution=bool(live_config.get("enable_order_execution", False)),
        require_manual_approval=bool(live_config.get("require_manual_approval", True)),
        output_dir=output_dir,
    )

    orders_log_path = output_dir / "orders_log.csv"
    trades_log_path = output_dir / "trades_log.csv"
    risk_events_path = output_dir / "risk_events.csv"

    profile_values: dict[str, float] = {}
    for profile in profiles:
        engine = ExecutionEngine(
            BacktestExecutionConfig(
                initial_cash=float(args.initial_cash),
                max_position_count=int(live_config.get("max_total_positions", args.top_n)),
                stop_loss_pct=profile.stop_loss,
                trailing_stop_pct=profile.trailing_stop,
                max_holding_days=profile.max_holding_days,
            )
        )

        try:
            candidates_df, candidate_source = load_rule_candidates_for_day(
                latest_trade_date,
                features_df,
                rule_signals_df,
            )
            day_orders, day_skips = build_rebalance_orders(
                latest_trade_date,
                candidates_df,
                prices_df,
                engine.get_state()["positions"],
                top_n=int(args.top_n),
            )
            engine.skipped.extend(day_skips)
            risk_exit_orders = engine.build_risk_exit_orders(
                trade_date=latest_trade_date,
                prices_df=prices_df,
                strategy=profile.strategy,
            )
            if risk_exit_orders:
                day_orders.extend(risk_exit_orders)

            portfolio_state = _build_portfolio_state(engine, latest_trade_date)
            approved_orders: list[dict[str, Any]] = []
            for order in day_orders:
                decision = risk_manager.validate_order(order, portfolio_state)
                if not decision.allowed:
                    _append_csv(risk_events_path, {
                        "date": latest_trade_date.strftime("%Y-%m-%d"),
                        "symbol": order.get("symbol"),
                        "reason": decision.reason,
                        "blocked_order": json.dumps(order, ensure_ascii=False),
                    })
                    continue
                route_result = router.send_order(order)
                router.log_order(order, route_result)
                if route_result.status not in {"paper_logged"}:
                    _append_csv(risk_events_path, {
                        "date": latest_trade_date.strftime("%Y-%m-%d"),
                        "symbol": order.get("symbol"),
                        "reason": route_result.status,
                        "blocked_order": json.dumps(order, ensure_ascii=False),
                    })
                    continue
                approved_orders.append(order)

            executed = engine.execute_daily_orders(
                trade_date=latest_trade_date,
                orders=approved_orders,
                prices_df=prices_df,
                strategy=profile.strategy,
            )
            snapshot = engine.create_snapshot(
                trade_date=latest_trade_date,
                prices_df=prices_df,
                strategy=profile.strategy,
            )
            profile_values[profile.profile] = float(snapshot.total_value)

            for trade in executed:
                _append_csv(trades_log_path, {
                    "date": trade.trade_date,
                    "profile": profile.profile,
                    "symbol": trade.symbol,
                    "name": trade.name,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "amount": trade.amount,
                    "reason": trade.reason,
                    "status": "executed_in_paper_engine",
                })
            for order in approved_orders:
                _append_csv(orders_log_path, {
                    "date": order.get("trade_date"),
                    "profile": profile.profile,
                    "symbol": order.get("symbol"),
                    "name": order.get("name"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "reason": order.get("reason"),
                    "candidate_source": candidate_source,
                    "status": "approved",
                })
        except Exception as exc:
            _append_csv(risk_events_path, {
                "date": latest_trade_date.strftime("%Y-%m-%d"),
                "symbol": "",
                "reason": "profile_exception",
                "blocked_order": json.dumps({"profile": profile.profile, "error": str(exc)}, ensure_ascii=False),
            })
            profile_values[profile.profile] = float(args.initial_cash)

    status_path = output_dir / "live_trading_status.json"
    status_payload = {
        "mode": live_config.get("mode", "paper"),
        "enable_order_execution": bool(live_config.get("enable_order_execution", False)),
        "require_manual_approval": bool(live_config.get("require_manual_approval", True)),
        "trade_date": latest_trade_date.strftime("%Y-%m-%d"),
        "profiles": profile_values,
        "todo": [
            "real broker API integration",
            "automatic scheduling",
            "web UI integration",
            "alerting integration",
        ],
    }
    status_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    values_line = ", ".join(f"{name}: {value:.2f}" for name, value in profile_values.items())
    print(f"[{latest_trade_date.strftime('%Y-%m-%d')}] {values_line}")


if __name__ == "__main__":
    main()
