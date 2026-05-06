"""
run_forward_test.py

Paper forward-test runner for selected strategy configurations.
This script simulates future-facing portfolio operation using the existing
daily-bar execution engine. It never sends orders to a broker and must never
be connected to live trading APIs.

Design constraints:
- forward test only
- labels.csv is not used
- no price data after trade_date is used
- each profile runs in its own isolated paper portfolio

Current TODO scope:
- real-time data refresh
- scheduler integration
- live brokerage integration
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_execution_engine import BacktestExecutionConfig, ExecutionEngine
from walk_forward_backtest import (
    DEFAULT_FEATURES_CSV,
    DEFAULT_PRICES_CSV,
    DEFAULT_RULE_SIGNALS_CSV,
    _resolve_path,
    load_features,
    load_prices,
    load_rule_signals,
    load_trading_days,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_JSON = ROOT / "configs" / "forward_test_configs.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forward_test"


@dataclass
class ForwardProfileConfig:
    """One forward-test profile configuration."""

    profile: str
    strategy: str
    stop_loss: float | None
    trailing_stop: float | None
    max_holding_days: int | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run paper forward test portfolios.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=10_000_000.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["batch", "daily"], default="batch")
    parser.add_argument("--configs-json", type=Path, default=DEFAULT_CONFIGS_JSON)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES_CSV)
    parser.add_argument("--rule-signals-csv", type=Path, default=DEFAULT_RULE_SIGNALS_CSV)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_forward_configs(path: Path) -> list[ForwardProfileConfig]:
    """Load profile configurations from JSON."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"forward test config file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("forward test config JSON must be a non-empty list")

    profiles: list[ForwardProfileConfig] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("forward test config entries must be objects")
        profile = str(item.get("profile") or "").strip()
        strategy = str(item.get("strategy") or "").strip().lower()
        if not profile:
            raise ValueError("forward test config entry missing profile")
        if strategy != "rule":
            raise ValueError(f"unsupported forward test strategy: {strategy}")
        profiles.append(
            ForwardProfileConfig(
                profile=profile,
                strategy=strategy,
                stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
                trailing_stop=float(item["trailing_stop"]) if item.get("trailing_stop") is not None else None,
                max_holding_days=int(item["max_holding_days"]) if item.get("max_holding_days") is not None else None,
            )
        )
    return profiles


def _compute_available_trading_days(
    prices_df: pd.DataFrame,
    start_date: pd.Timestamp,
    mode: str,
) -> list[pd.Timestamp]:
    """Resolve the trading days to process for batch or daily mode."""
    latest_available_date = pd.Timestamp(prices_df["date"].max()).normalize()
    trading_days = load_trading_days(prices_df, start_date, latest_available_date)
    if mode == "daily":
        return [trading_days[-1]] if trading_days else []
    return trading_days


def _derive_rule_candidates_from_features(prior_features_day: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple rule candidates from prior-day features only.

    This is a forward-test fallback when same-day rule_signals history is not
    available. It uses only the latest prior feature snapshot.
    """
    if prior_features_day.empty:
        return pd.DataFrame()

    day = prior_features_day.copy()
    for col in [
        "close",
        "ma_20",
        "ma_60",
        "ret_5d",
        "mom_20",
        "rsi_14",
        "volume_ratio_20d",
        "liquidity_score",
        "quality_score",
    ]:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors="coerce")

    conditions = pd.Series(True, index=day.index)
    if {"close", "ma_20"}.issubset(day.columns):
        conditions &= day["close"] > day["ma_20"]
    if {"ma_20", "ma_60"}.issubset(day.columns):
        conditions &= day["ma_20"] >= day["ma_60"]
    if "ret_5d" in day.columns:
        conditions &= day["ret_5d"] > 0
    if "mom_20" in day.columns:
        conditions &= day["mom_20"] > 0
    if "rsi_14" in day.columns:
        conditions &= day["rsi_14"].between(45, 75, inclusive="both")
    if "volume_ratio_20d" in day.columns:
        conditions &= day["volume_ratio_20d"] >= 1.2

    filtered = day.loc[conditions.fillna(False)].copy()
    if filtered.empty:
        return pd.DataFrame()

    liquidity = pd.to_numeric(filtered.get("liquidity_score"), errors="coerce").fillna(0.0)
    quality = pd.to_numeric(filtered.get("quality_score"), errors="coerce").fillna(0.0)
    mom = pd.to_numeric(filtered.get("mom_20"), errors="coerce").fillna(0.0)
    filtered["candidate_score"] = liquidity * 0.5 + quality * 0.3 + mom.rank(pct=True).fillna(0.0) * 100.0 * 0.2
    filtered["candidate_kind"] = "feature_fallback"
    filtered["candidate_date"] = filtered["date"]
    return filtered.sort_values(["candidate_score", "symbol"], ascending=[False, True]).reset_index(drop=True)


def load_rule_candidates_for_day(
    trade_date: pd.Timestamp,
    features_df: pd.DataFrame,
    rule_signals_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Load rule candidates for trade_date using only information available before D.

    Priority:
    1. latest prior rule_signals day < D
    2. latest prior features day < D with feature-based fallback scoring
    """
    prior_rule = rule_signals_df.loc[rule_signals_df["date"] < trade_date].copy()
    if not prior_rule.empty:
        latest_rule_date = prior_rule["date"].max()
        rule_day = prior_rule.loc[prior_rule["date"] == latest_rule_date].copy()
        if "strong_entry_signal" in rule_day.columns:
            rule_day = rule_day.loc[rule_day["strong_entry_signal"].fillna(False)]
        elif "entry_signal" in rule_day.columns:
            rule_day = rule_day.loc[rule_day["entry_signal"].fillna(False)]
        if not rule_day.empty:
            rule_day["candidate_score"] = pd.to_numeric(rule_day.get("rule_score_v2", rule_day.get("rule_score")), errors="coerce")
            rule_day["candidate_kind"] = "rule_signal"
            rule_day["candidate_date"] = latest_rule_date
            rule_day.attrs["candidate_debug"] = f"rule:{latest_rule_date.date()}"
            return rule_day.sort_values(["candidate_score", "symbol"], ascending=[False, True]).reset_index(drop=True), "rule_signal"

    prior_features = features_df.loc[features_df["date"] < trade_date].copy()
    if prior_features.empty:
        return pd.DataFrame(), "no_prior_features"
    latest_feature_date = prior_features["date"].max()
    feature_day = prior_features.loc[prior_features["date"] == latest_feature_date].copy()
    fallback = _derive_rule_candidates_from_features(feature_day)
    if fallback.empty:
        return pd.DataFrame(), "no_feature_candidates"
    fallback.attrs["candidate_debug"] = f"feature_fallback:{latest_feature_date.date()}"
    return fallback, "feature_fallback"


def build_rebalance_orders(
    trade_date: pd.Timestamp,
    candidates_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    current_positions: dict[str, Any],
    *,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build daily rebalance orders for forward test."""
    orders: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    if candidates_df.empty:
        return orders, skips

    day_prices = prices_df.loc[prices_df["date"] == trade_date].copy()
    if day_prices.empty:
        return orders, [{
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "reason": "missing_day_prices",
            "detail": f"No prices for trade_date={trade_date.date()}",
        }]

    price_map = day_prices.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol").to_dict(orient="index")
    top_candidates = candidates_df.head(top_n).copy()
    target_symbols = set(top_candidates["symbol"].astype(str).tolist())
    held_symbols = set(map(str, current_positions.keys()))

    for held_symbol in sorted(held_symbols - target_symbols):
        held = current_positions.get(held_symbol, {})
        orders.append({
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "symbol": held_symbol,
            "name": held.get("name") or held_symbol,
            "side": "SELL",
            "planned_time": "15:10",
            "strategy": "rule",
            "reason": "rebalance_exit",
        })

    for row in top_candidates.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        if symbol in held_symbols:
            continue
        px = price_map.get(symbol)
        if not px or pd.isna(px.get("adj_open")):
            skips.append({
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "reason": "missing_symbol_price",
                "detail": f"Missing adj_open for symbol={symbol}",
            })
            continue
        orders.append({
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "name": getattr(row, "name", symbol),
            "side": "BUY",
            "planned_time": "09:30",
            "planned_price": float(px["adj_open"]),
            "strategy": "rule",
            "reason": "top_n_candidate",
        })
    return orders, skips


def _compute_profile_summary(engine: ExecutionEngine) -> dict[str, Any]:
    """Compute forward summary metrics from one profile engine state."""
    snapshots = [snapshot.to_dict() for snapshot in engine.portfolio_snapshots]
    trades = [trade.to_dict() for trade in engine.trades]
    current_value = float(engine.cash)
    running_days = len(snapshots)
    last_update = None
    total_return = 0.0
    mdd = 0.0
    win_rate = 0.0

    if snapshots:
        snap_df = pd.DataFrame(snapshots)
        snap_df["total_value"] = pd.to_numeric(snap_df["total_value"], errors="coerce")
        snap_df["trade_date"] = pd.to_datetime(snap_df["trade_date"], errors="coerce")
        snap_df = snap_df.dropna(subset=["trade_date", "total_value"]).copy()
        if not snap_df.empty:
            current_value = float(snap_df["total_value"].iloc[-1])
            last_update = snap_df["trade_date"].iloc[-1].strftime("%Y-%m-%d")
            initial_cash = float(engine.config.initial_cash)
            if initial_cash > 0:
                total_return = current_value / initial_cash - 1.0
            running_peak = snap_df["total_value"].cummax()
            drawdown = snap_df["total_value"] / running_peak - 1.0
            if not drawdown.empty:
                mdd = float(drawdown.min())

    if trades:
        trade_df = pd.DataFrame(trades)
        sells = trade_df.loc[trade_df["side"].astype(str).str.upper() == "SELL"].copy()
        if not sells.empty and "realized_return" in sells.columns:
            sells["realized_return"] = pd.to_numeric(sells["realized_return"], errors="coerce")
            valid = sells.dropna(subset=["realized_return"])
            if not valid.empty:
                win_rate = float((valid["realized_return"] > 0).mean())

    return {
        "current_value": round(current_value, 2),
        "total_return": float(total_return),
        "running_days": int(running_days),
        "last_update": last_update,
        "win_rate": float(win_rate),
        "MDD": float(mdd),
    }


def _save_profile_outputs(profile_dir: Path, engine: ExecutionEngine) -> None:
    """Persist one profile's paper trading outputs."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame([trade.to_dict() for trade in engine.trades])
    portfolio_df = pd.DataFrame([snapshot.to_dict() for snapshot in engine.portfolio_snapshots])
    skipped_df = pd.DataFrame(engine.skipped)

    trades_df.to_csv(profile_dir / "trades.csv", index=False, encoding="utf-8-sig")
    portfolio_df.to_csv(profile_dir / "portfolio.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(profile_dir / "skipped_days.csv", index=False, encoding="utf-8-sig")
    (profile_dir / "state.json").write_text(
        json.dumps(engine.get_state(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    output_dir = _resolve_path(args.output_dir)

    profiles = load_forward_configs(args.configs_json)
    features_df = load_features(_resolve_path(args.features_csv))
    prices_df = load_prices(_resolve_path(args.prices_csv))
    rule_signals_df = load_rule_signals(_resolve_path(args.rule_signals_csv))
    prices_df["open"] = pd.to_numeric(prices_df.get("adj_open"), errors="coerce")
    prices_df["close"] = pd.to_numeric(prices_df.get("adj_close"), errors="coerce")

    trading_days = _compute_available_trading_days(prices_df, start_date, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    engines: dict[str, ExecutionEngine] = {}
    for profile in profiles:
        execution_config = BacktestExecutionConfig(
            initial_cash=float(args.initial_cash),
            max_position_count=int(args.top_n),
            stop_loss_pct=profile.stop_loss,
            trailing_stop_pct=profile.trailing_stop,
            max_holding_days=profile.max_holding_days,
        )
        engines[profile.profile] = ExecutionEngine(execution_config)

    if not trading_days:
        forward_summary = {
            "mode": args.mode,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "last_available_data_date": pd.Timestamp(prices_df["date"].max()).strftime("%Y-%m-%d"),
            "profiles": {
                profile.profile: {
                    **asdict(profile),
                    "current_value": float(args.initial_cash),
                    "total_return": 0.0,
                    "running_days": 0,
                    "last_update": None,
                    "win_rate": 0.0,
                    "MDD": 0.0,
                    "status": "no_trading_days",
                }
                for profile in profiles
            },
        }
        (output_dir / "forward_summary.json").write_text(
            json.dumps(forward_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    for trade_date in trading_days:
        for profile in profiles:
            engine = engines[profile.profile]
            try:
                candidates_df, candidate_source = load_rule_candidates_for_day(
                    trade_date,
                    features_df,
                    rule_signals_df,
                )
                day_orders, day_skips = build_rebalance_orders(
                    trade_date,
                    candidates_df,
                    prices_df,
                    engine.get_state()["positions"],
                    top_n=int(args.top_n),
                )
                engine.skipped.extend(day_skips)

                risk_exit_orders = engine.build_risk_exit_orders(
                    trade_date=trade_date,
                    prices_df=prices_df,
                    strategy=profile.strategy,
                )
                if risk_exit_orders:
                    existing_sell_symbols = {
                        str(order.get("symbol"))
                        for order in day_orders
                        if str(order.get("side") or "").upper() == "SELL"
                    }
                    for exit_order in risk_exit_orders:
                        if str(exit_order.get("symbol")) not in existing_sell_symbols:
                            day_orders.append(exit_order)

                if not day_orders:
                    engine.skipped.append({
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "reason": "no_orders",
                        "detail": f"profile={profile.profile}, candidate_source={candidate_source}",
                    })

                engine.execute_daily_orders(
                    trade_date=trade_date,
                    orders=day_orders,
                    prices_df=prices_df,
                    strategy=profile.strategy,
                )
                snapshot = engine.create_snapshot(
                    trade_date=trade_date,
                    prices_df=prices_df,
                    strategy=profile.strategy,
                )
                print(f"[{trade_date.strftime('%Y-%m-%d')}] {profile.profile}: {snapshot.total_value}")
            except Exception as exc:
                engine.skipped.append({
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "reason": "profile_exception",
                    "detail": f"profile={profile.profile}: {exc}",
                })

    summary_profiles: dict[str, Any] = {}
    for profile in profiles:
        engine = engines[profile.profile]
        profile_dir = output_dir / profile.profile
        _save_profile_outputs(profile_dir, engine)
        summary_profiles[profile.profile] = {
            **asdict(profile),
            **_compute_profile_summary(engine),
            "status": "completed",
        }

    forward_summary = {
        "mode": args.mode,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "processed_trading_days": [day.strftime("%Y-%m-%d") for day in trading_days],
        "profiles": summary_profiles,
        "todo": [
            "real-time data integration",
            "automatic scheduling",
            "live account integration",
        ],
    }
    (output_dir / "forward_summary.json").write_text(
        json.dumps(forward_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
