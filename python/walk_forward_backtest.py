"""
walk_forward_backtest.py

Initial skeleton for a point-in-time walk-forward backtest over roughly three
years of data (2023-04-14 to 2026-05-04).

Look-ahead protection is the primary design rule:
- When evaluating trade date D, features and labels must be restricted to rows
  strictly earlier than D.
- Reused recommendation snapshots must also come from dates strictly earlier
  than D.
- Daily prices may be used up to D, but no D+1 or later data is ever touched.
- Future labels/realized returns are never consumed for same-day decisions.

Scope of this first version:
- Load historical datasets
- Build trading days
- Run a day-by-day walk-forward loop
- Reuse existing historical recommendation artifacts when available
- Emit planned order events and backtest output files

Explicitly out of scope for this file:
- Real execution engine
- Fill simulation
- Slippage
- Fees
- Taxes

Those are intentionally deferred to python/backtest_execution_engine.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from backtest_execution_engine import BacktestExecutionConfig, ExecutionEngine


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "backtest"

DEFAULT_FEATURES_CSV = DATA_DIR / "features.csv"
DEFAULT_LABELS_CSV = DATA_DIR / "labels.csv"
DEFAULT_PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
DEFAULT_RULE_SIGNALS_CSV = DATA_DIR / "rule_signals.csv"
DEFAULT_AI_SNAPSHOT_CSV = DATA_DIR / "ranking_snapshot_archive.csv"
DEFAULT_AI_FILTERED_CSV = DATA_DIR / "ai_filtered_top_candidates.csv"


def setup_logging() -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


@dataclass
class BacktestConfig:
    """Runtime configuration for the walk-forward skeleton."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    strategy: str
    initial_cash: float
    top_n: int
    buy_time: str
    sell_time: str
    output_dir: Path
    stop_loss: float | None
    trailing_stop: float | None
    max_holding_days: int | None
    features_csv: Path
    labels_csv: Path
    prices_csv: Path
    rule_signals_csv: Path
    ai_snapshot_csv: Path
    ai_filtered_csv: Path


def parse_args() -> BacktestConfig:
    """Parse CLI arguments into a typed config."""
    parser = argparse.ArgumentParser(
        description="Run a point-in-time walk-forward backtest skeleton."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, choices=["ai", "rule", "hybrid"])
    parser.add_argument("--initial-cash", type=float, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--buy-time", type=str, default="09:30")
    parser.add_argument("--sell-time", type=str, default="15:10")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--trailing-stop", type=float, default=0.04)
    parser.add_argument("--max-holding-days", type=int, default=10)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES_CSV)
    parser.add_argument("--rule-signals-csv", type=Path, default=DEFAULT_RULE_SIGNALS_CSV)
    parser.add_argument("--ai-snapshot-csv", type=Path, default=DEFAULT_AI_SNAPSHOT_CSV)
    parser.add_argument("--ai-filtered-csv", type=Path, default=DEFAULT_AI_FILTERED_CSV)
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise ValueError("--end-date must be >= --start-date")
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be > 0")

    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy=str(args.strategy),
        initial_cash=float(args.initial_cash),
        top_n=int(args.top_n),
        buy_time=str(args.buy_time),
        sell_time=str(args.sell_time),
        output_dir=_resolve_path(args.output_dir),
        stop_loss=float(args.stop_loss) if args.stop_loss is not None else None,
        trailing_stop=float(args.trailing_stop) if args.trailing_stop is not None else None,
        max_holding_days=int(args.max_holding_days) if args.max_holding_days is not None else None,
        features_csv=_resolve_path(args.features_csv),
        labels_csv=_resolve_path(args.labels_csv),
        prices_csv=_resolve_path(args.prices_csv),
        rule_signals_csv=_resolve_path(args.rule_signals_csv),
        ai_snapshot_csv=_resolve_path(args.ai_snapshot_csv),
        ai_filtered_csv=_resolve_path(args.ai_filtered_csv),
    )


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path if path.is_absolute() else ROOT / path


def _validate_file_exists(path: Path, label: str) -> None:
    """Raise a clear error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _normalize_symbol(series: pd.Series) -> pd.Series:
    """Normalize stock symbols/codes into zero-padded strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )


def _get_symbol_column(df: pd.DataFrame) -> str:
    """Return the symbol column name supported by the input file."""
    for candidate in ("symbol", "code"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Input CSV must contain either 'symbol' or 'code'")


def _finalize_base_frame(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Apply shared normalization for date/symbol keyed datasets."""
    if "date" not in df.columns:
        raise ValueError(f"{label} missing required column: date")
    symbol_col = _get_symbol_column(df)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = _normalize_symbol(work[symbol_col])
    work = work.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return work


def load_features(path: Path) -> pd.DataFrame:
    """Load features.csv with date normalization, symbol sorting, and basic NA cleanup."""
    _validate_file_exists(path, "features csv")
    df = pd.read_csv(path, dtype={"code": str, "symbol": str}, low_memory=False)
    return _finalize_base_frame(df, label="features csv")


def load_labels(path: Path) -> pd.DataFrame:
    """Load labels.csv with date normalization, symbol sorting, and basic NA cleanup."""
    _validate_file_exists(path, "labels csv")
    df = pd.read_csv(path, dtype={"code": str, "symbol": str}, low_memory=False)
    return _finalize_base_frame(df, label="labels csv")


def load_prices(path: Path) -> pd.DataFrame:
    """Load adjusted daily prices with normalized symbol/date and numeric price fields."""
    _validate_file_exists(path, "prices csv")
    df = pd.read_csv(path, dtype={"code": str, "symbol": str}, low_memory=False)
    work = _finalize_base_frame(df, label="prices csv")
    for col in ("adj_open", "adj_high", "adj_low", "adj_close", "volume"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["adj_open", "adj_close"])
    return work.reset_index(drop=True)


def load_rule_signals(path: Path) -> pd.DataFrame:
    """Load historical rule signals for candidate reuse."""
    if not path.exists():
        logging.warning("rule signals not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"code": str, "symbol": str}, low_memory=False)
    work = _finalize_base_frame(df, label="rule signals csv")
    for col in ("entry_signal", "strong_entry_signal", "market_entry_allowed"):
        if col in work.columns:
            work[col] = work[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    for col in ("rule_score_v2", "rule_score", "liquidity_score"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def load_ai_snapshots(snapshot_path: Path, filtered_path: Path) -> pd.DataFrame:
    """Load reusable AI recommendation history from available historical artifacts."""
    frames: list[pd.DataFrame] = []

    if snapshot_path.exists():
        snap = pd.read_csv(snapshot_path, dtype={"code": str, "symbol": str}, low_memory=False)
        if "asof_date" not in snap.columns:
            raise ValueError(f"AI snapshot csv missing required column: asof_date ({snapshot_path})")
        snap = snap.rename(columns={"asof_date": "date"})
        snap = _finalize_base_frame(snap, label="ai snapshot csv")
        snap["candidate_source"] = "ranking_snapshot_archive"
        if "rank" in snap.columns:
            snap["rank"] = pd.to_numeric(snap["rank"], errors="coerce")
        if "final_score" in snap.columns:
            snap["final_score"] = pd.to_numeric(snap["final_score"], errors="coerce")
        frames.append(snap)

    if filtered_path.exists():
        latest = pd.read_csv(filtered_path, dtype={"code": str, "symbol": str}, low_memory=False)
        latest = _finalize_base_frame(latest, label="ai filtered csv")
        latest["candidate_source"] = "ai_filtered_top_candidates"
        if "ai_filtered_rank" in latest.columns:
            latest["rank"] = pd.to_numeric(latest["ai_filtered_rank"], errors="coerce")
        elif "rank_final" in latest.columns:
            latest["rank"] = pd.to_numeric(latest["rank_final"], errors="coerce")
        elif "live_rank" in latest.columns:
            latest["rank"] = pd.to_numeric(latest["live_rank"], errors="coerce")
        if "final_score" in latest.columns:
            latest["final_score"] = pd.to_numeric(latest["final_score"], errors="coerce")
        frames.append(latest)

    if not frames:
        logging.warning("No AI candidate artifact found: %s or %s", snapshot_path, filtered_path)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    return combined


def load_trading_days(prices_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Build the ordered trading day list from prices_daily_adjusted within the requested range."""
    days = (
        prices_df.loc[prices_df["date"].between(start_date, end_date), "date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [pd.Timestamp(day).normalize() for day in days]


def load_candidates_for_day(
    trade_date: pd.Timestamp,
    strategy: str,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    rule_signals_df: pd.DataFrame,
    ai_candidates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Load reusable candidates for a single trading day using point-in-time rules.

    Look-ahead guard:
    - features are restricted to date < trade_date
    - labels are restricted to date < trade_date
    - candidate artifacts must come from a date < trade_date
    - future labels are never used directly for ranking decisions
    """
    prior_features = features_df.loc[features_df["date"] < trade_date]
    prior_labels = labels_df.loc[labels_df["date"] < trade_date]
    if prior_features.empty:
        return pd.DataFrame(), {
            "reason": "no_prior_features",
            "detail": f"No features before {trade_date.date()}",
        }
    if prior_labels.empty:
        return pd.DataFrame(), {
            "reason": "no_prior_labels",
            "detail": f"No labels before {trade_date.date()}",
        }

    frames: list[pd.DataFrame] = []
    debug_bits: list[str] = []

    if strategy in {"rule", "hybrid"}:
        prior_rule = rule_signals_df.loc[rule_signals_df["date"] < trade_date].copy()
        if not prior_rule.empty:
            latest_rule_date = prior_rule["date"].max()
            rule_day = prior_rule.loc[prior_rule["date"] == latest_rule_date].copy()
            if "strong_entry_signal" in rule_day.columns:
                rule_day = rule_day.loc[rule_day["strong_entry_signal"].fillna(False)]
            elif "entry_signal" in rule_day.columns:
                rule_day = rule_day.loc[rule_day["entry_signal"].fillna(False)]
            if not rule_day.empty:
                rule_day["candidate_kind"] = "rule"
                rule_day["candidate_date"] = latest_rule_date
                if "rule_score_v2" in rule_day.columns:
                    rule_day["candidate_score"] = pd.to_numeric(rule_day["rule_score_v2"], errors="coerce")
                else:
                    rule_day["candidate_score"] = pd.to_numeric(rule_day.get("rule_score"), errors="coerce")
                frames.append(rule_day)
                debug_bits.append(f"rule:{latest_rule_date.date()}")

    if strategy in {"ai", "hybrid"}:
        prior_ai = ai_candidates_df.loc[ai_candidates_df["date"] < trade_date].copy()
        if not prior_ai.empty:
            latest_ai_date = prior_ai["date"].max()
            ai_day = prior_ai.loc[prior_ai["date"] == latest_ai_date].copy()
            ai_day["candidate_kind"] = "ai"
            ai_day["candidate_date"] = latest_ai_date
            ai_day["candidate_score"] = pd.to_numeric(ai_day.get("final_score"), errors="coerce")
            frames.append(ai_day)
            debug_bits.append(f"ai:{latest_ai_date.date()}")

    if not frames:
        return pd.DataFrame(), {
            "reason": "candidate_source_unavailable",
            "detail": (
                "No reusable historical recommendation source exists before "
                f"{trade_date.date()} for strategy={strategy}. TODO: add point-in-time candidate generation."
            ),
        }

    if strategy == "hybrid":
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged["candidate_score"] = pd.to_numeric(merged["candidate_score"], errors="coerce")
        merged["candidate_score"] = merged["candidate_score"].fillna(0.0)
        combined = (
            merged.groupby("symbol", as_index=False)
            .agg(
                candidate_score=("candidate_score", "mean"),
                candidate_kind=("candidate_kind", lambda x: "+".join(sorted(set(map(str, x))))),
                candidate_date=("candidate_date", "max"),
                name=("name", "first") if "name" in merged.columns else ("symbol", "first"),
            )
        )
    else:
        combined = frames[0].copy()

    combined["symbol"] = _normalize_symbol(combined["symbol"])
    combined["candidate_score"] = pd.to_numeric(combined.get("candidate_score"), errors="coerce")
    combined = combined.dropna(subset=["symbol"]).sort_values(
        by=["candidate_score", "symbol"],
        ascending=[False, True],
    )
    combined = combined.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    combined.attrs["candidate_debug"] = ", ".join(debug_bits)
    return combined, None


def build_daily_orders(
    trade_date: pd.Timestamp,
    candidates_df: pd.DataFrame,
    prices_until_day_df: pd.DataFrame,
    config: BacktestConfig,
    current_positions: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build planned order events for one trading day.

    This function only builds planned rebalance orders.
    Actual fills, fees, tax, and slippage are delegated to
    backtest_execution_engine.py.

    Look-ahead guard:
    - buy planning uses the same-day adjusted open only
    - sell planning uses only membership changes versus the current held set
    - no D+1 or later data is referenced here
    """
    if candidates_df.empty:
        return [], []

    day_prices = prices_until_day_df.loc[prices_until_day_df["date"] == trade_date].copy()
    if day_prices.empty:
        return [], [{
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "reason": "missing_day_prices",
            "detail": f"No prices available for {trade_date.date()}",
        }]

    price_map = (
        day_prices.drop_duplicates(subset=["symbol"], keep="last")
        .set_index("symbol")
        .to_dict(orient="index")
    )

    orders: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    top_candidates = candidates_df.head(config.top_n).copy()
    target_symbols = set(top_candidates["symbol"].astype(str).tolist())
    held_symbols = set(map(str, current_positions.keys()))

    # Full liquidation for names that fell out of the current top-N rule set.
    for held_symbol in sorted(held_symbols - target_symbols):
        held_state = current_positions.get(held_symbol, {})
        orders.append({
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "symbol": held_symbol,
            "name": held_state.get("name") or held_symbol,
            "side": "SELL",
            "planned_time": config.sell_time,
            "planned_price": None,
            "strategy": str(config.strategy),
            "reason": "rebalance_exit",
        })

    for row in top_candidates.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        px = price_map.get(symbol)
        if not px or pd.isna(px.get("adj_open")):
            skips.append({
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "reason": "missing_symbol_price",
                "detail": f"Missing adj_open for symbol={symbol}",
            })
            continue
        if symbol in held_symbols:
            continue

        planned_buy_price = float(px["adj_open"])
        strategy_name = str(config.strategy)

        orders.append({
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "name": getattr(row, "name", symbol),
            "strategy": strategy_name,
            "side": "BUY",
            "planned_time": config.buy_time,
            "planned_price": round(planned_buy_price, 6),
            "reason": "top_n_candidate",
        })

    return orders, skips


def _calculate_summary_metrics(
    initial_cash: float,
    final_cash: float,
    daily_portfolio_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Compute simple backtest summary metrics from snapshots and executed trades."""
    final_total_value = initial_cash
    cagr = 0.0
    mdd = 0.0
    if daily_portfolio_rows:
        portfolio_df = pd.DataFrame(daily_portfolio_rows)
        portfolio_df["total_value"] = pd.to_numeric(portfolio_df["total_value"], errors="coerce")
        portfolio_df = portfolio_df.dropna(subset=["total_value"]).copy()
        if not portfolio_df.empty:
            final_total_value = float(portfolio_df["total_value"].iloc[-1])
            day_count = max((end_date - start_date).days, 1)
            years = day_count / 365.25
            if initial_cash > 0 and final_total_value > 0 and years > 0:
                cagr = (final_total_value / initial_cash) ** (1.0 / years) - 1.0
            running_peak = portfolio_df["total_value"].cummax()
            drawdown = portfolio_df["total_value"] / running_peak - 1.0
            if not drawdown.empty:
                mdd = float(drawdown.min())

    total_return = 0.0 if initial_cash <= 0 else final_total_value / initial_cash - 1.0
    trades_df = pd.DataFrame(trade_rows)
    total_trades = int(len(trades_df))
    win_rate = 0.0
    if not trades_df.empty and "side" in trades_df.columns:
        sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy()
        if not sells.empty and "realized_return" in sells.columns:
            sells["realized_return"] = pd.to_numeric(sells["realized_return"], errors="coerce")
            valid_sells = sells.dropna(subset=["realized_return"])
            if not valid_sells.empty:
                win_rate = float((valid_sells["realized_return"] > 0).mean())

    return {
        "initial_cash": round(float(initial_cash), 2),
        "final_cash": round(float(final_cash), 2),
        "final_total_value": round(float(final_total_value), 2),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "total_trades": total_trades,
        "win_rate": float(win_rate),
    }


def save_backtest_outputs(
    config: BacktestConfig,
    summary: dict[str, Any],
    daily_portfolio_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    """Persist summary JSON and tabular CSV outputs under the requested directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = config.output_dir / "backtest_summary.json"
    daily_path = config.output_dir / "daily_portfolio.csv"
    trades_path = config.output_dir / "trades.csv"
    skipped_path = config.output_dir / "skipped_days.csv"

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    pd.DataFrame(
        daily_portfolio_rows,
        columns=[
            "trade_date",
            "cash",
            "equity",
            "position_value",
            "total_value",
            "daily_return",
            "cumulative_return",
            "strategy",
            "planned_order_count",
            "executed_trade_count",
            "candidate_count",
            "candidate_debug",
        ],
    ).to_csv(daily_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        trade_rows,
        columns=[
            "trade_date",
            "symbol",
            "side",
            "planned_time",
            "planned_price",
            "executed_price",
            "quantity",
            "amount",
            "fee",
            "tax",
            "slippage",
            "reason",
            "strategy",
            "cost_basis_amount",
            "realized_pnl",
            "realized_return",
        ],
    ).to_csv(trades_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        skipped_rows,
        columns=["trade_date", "reason", "detail"],
    ).to_csv(skipped_path, index=False, encoding="utf-8-sig")


def run_walk_forward_backtest(config: BacktestConfig) -> dict[str, Any]:
    """Execute the day-by-day walk-forward skeleton and return in-memory results."""
    features_df = load_features(config.features_csv)
    labels_df = load_labels(config.labels_csv)
    prices_df = load_prices(config.prices_csv)
    rule_signals_df = load_rule_signals(config.rule_signals_csv)
    ai_candidates_df = load_ai_snapshots(config.ai_snapshot_csv, config.ai_filtered_csv)
    prices_df["open"] = pd.to_numeric(prices_df.get("adj_open"), errors="coerce")
    prices_df["close"] = pd.to_numeric(prices_df.get("adj_close"), errors="coerce")

    trading_days = load_trading_days(prices_df, config.start_date, config.end_date)
    if not trading_days:
        raise ValueError("No trading days found in the requested date range")

    execution_config = BacktestExecutionConfig(
        initial_cash=config.initial_cash,
        max_position_count=config.top_n,
        stop_loss_pct=config.stop_loss,
        trailing_stop_pct=config.trailing_stop,
        max_holding_days=config.max_holding_days,
    )
    engine = ExecutionEngine(execution_config)
    daily_portfolio_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    logging.info(
        "Starting walk-forward loop: strategy=%s trading_days=%s range=%s~%s",
        config.strategy,
        len(trading_days),
        config.start_date.date(),
        config.end_date.date(),
    )

    for trade_date in trading_days:
        logging.info("Processing trade_date=%s", trade_date.date())
        try:
            prices_until_day = prices_df.loc[prices_df["date"] <= trade_date].copy()
            effective_strategy = config.strategy
            candidate_strategy = config.strategy
            forced_skip: dict[str, Any] | None = None

            if config.strategy == "ai":
                forced_skip = {
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "reason": "AI_DATA_MISSING",
                    "detail": "AI historical recommendation data is not available for point-in-time backtesting.",
                }
            elif config.strategy == "hybrid":
                effective_strategy = "rule"
                candidate_strategy = "rule"
                skipped_rows.append({
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "reason": "RULE_FALLBACK_USED",
                    "detail": "Hybrid strategy fell back to rule because AI historical data is unavailable.",
                })

            candidates_df = pd.DataFrame()
            if forced_skip is None:
                candidates_df, skip_info = load_candidates_for_day(
                    trade_date,
                    candidate_strategy,
                    features_df,
                    labels_df,
                    rule_signals_df=rule_signals_df,
                    ai_candidates_df=ai_candidates_df,
                )
                if skip_info is not None:
                    skipped_rows.append({
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "reason": skip_info["reason"],
                        "detail": skip_info["detail"],
                    })
            else:
                skipped_rows.append(forced_skip)

            day_orders: list[dict[str, Any]] = []
            if not candidates_df.empty:
                day_orders, symbol_skips = build_daily_orders(
                    trade_date,
                    candidates_df,
                    prices_until_day,
                    config,
                    current_positions=engine.get_state()["positions"],
                )
                skipped_rows.extend(symbol_skips)
            else:
                skipped_rows.append({
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "reason": "no_candidates",
                    "detail": f"No candidate rows selected for strategy={candidate_strategy}",
                })

            if not day_orders:
                skipped_rows.append({
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "reason": "no_orders",
                    "detail": f"No orders generated for strategy={effective_strategy}",
                })

            risk_exit_orders = engine.build_risk_exit_orders(
                trade_date=trade_date,
                prices_df=prices_df,
                strategy=effective_strategy,
            )
            if risk_exit_orders:
                existing_sell_symbols = {
                    str(order.get("symbol"))
                    for order in day_orders
                    if str(order.get("side") or "").upper() == "SELL"
                }
                for exit_order in risk_exit_orders:
                    if str(exit_order.get("symbol")) in existing_sell_symbols:
                        continue
                    day_orders.append(exit_order)

            skipped_before = len(engine.skipped)
            trades = engine.execute_daily_orders(
                trade_date=trade_date,
                orders=day_orders,
                prices_df=prices_df,
                strategy=effective_strategy,
            )
            snapshot = engine.create_snapshot(
                trade_date=trade_date,
                prices_df=prices_df,
                strategy=effective_strategy,
            )
            if len(engine.skipped) > skipped_before:
                skipped_rows.extend(engine.skipped[skipped_before:])

            candidate_debug = candidates_df.attrs.get("candidate_debug", "") if not candidates_df.empty else ""
            daily_row = snapshot.to_dict()
            daily_row["planned_order_count"] = len(day_orders)
            daily_row["executed_trade_count"] = len(trades)
            daily_row["candidate_count"] = int(min(len(candidates_df), config.top_n)) if not candidates_df.empty else 0
            daily_row["candidate_debug"] = candidate_debug

            print(f"[{trade_date.strftime('%Y-%m-%d')}] trades={len(trades)} cash={round(engine.cash, 2)}")
            daily_portfolio_rows.append(daily_row)
        except Exception as exc:
            logging.exception("Error on trade_date=%s", trade_date.date())
            skipped_rows.append({
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "reason": "exception",
                "detail": str(exc),
            })

    state = engine.get_state()
    trade_rows = state["trades"]
    summary_metrics = _calculate_summary_metrics(
        initial_cash=config.initial_cash,
        final_cash=state["cash"],
        daily_portfolio_rows=daily_portfolio_rows,
        trade_rows=trade_rows,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    summary = {
        "config": {
            **asdict(config),
            "start_date": config.start_date.strftime("%Y-%m-%d"),
            "end_date": config.end_date.strftime("%Y-%m-%d"),
            "output_dir": str(config.output_dir),
            "features_csv": str(config.features_csv),
            "labels_csv": str(config.labels_csv),
            "prices_csv": str(config.prices_csv),
            "rule_signals_csv": str(config.rule_signals_csv),
            "ai_snapshot_csv": str(config.ai_snapshot_csv),
            "ai_filtered_csv": str(config.ai_filtered_csv),
        },
        "data_ranges": {
            "features": {
                "min_date": features_df["date"].min().strftime("%Y-%m-%d"),
                "max_date": features_df["date"].max().strftime("%Y-%m-%d"),
                "rows": int(len(features_df)),
            },
            "labels": {
                "min_date": labels_df["date"].min().strftime("%Y-%m-%d"),
                "max_date": labels_df["date"].max().strftime("%Y-%m-%d"),
                "rows": int(len(labels_df)),
            },
            "prices": {
                "min_date": prices_df["date"].min().strftime("%Y-%m-%d"),
                "max_date": prices_df["date"].max().strftime("%Y-%m-%d"),
                "rows": int(len(prices_df)),
            },
        },
        "trading_days": {
            "count": len(trading_days),
            "first": trading_days[0].strftime("%Y-%m-%d"),
            "last": trading_days[-1].strftime("%Y-%m-%d"),
        },
        "baseline_reference": "rule_portfolio_backtest_report.json",
        "execution_config": asdict(execution_config),
        "performance": {
            **summary_metrics,
            "final_positions": state["positions"],
        },
        "results": {
            "planned_trade_rows": int(sum(row.get("planned_order_count", 0) for row in daily_portfolio_rows)),
            "executed_trade_rows": int(len(trade_rows)),
            "daily_portfolio_rows": int(len(daily_portfolio_rows)),
            "skipped_rows": int(len(skipped_rows)),
            "execution_engine_implemented": True,
            "notes": [
                "This version executes daily-bar approximated fills through backtest_execution_engine.py.",
                "Rule strategy is the primary supported path for historical runs.",
                "AI strategy is skipped because point-in-time AI history is unavailable.",
                "Future labels are not used in decision logic.",
            ],
        },
    }

    return {
        "summary": summary,
        "daily_portfolio_rows": daily_portfolio_rows,
        "trade_rows": trade_rows,
        "skipped_rows": skipped_rows,
    }


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    config = parse_args()
    result = run_walk_forward_backtest(config)
    save_backtest_outputs(
        config,
        result["summary"],
        result["daily_portfolio_rows"],
        result["trade_rows"],
        result["skipped_rows"],
    )
    logging.info("Backtest outputs written to %s", config.output_dir)


if __name__ == "__main__":
    main()
