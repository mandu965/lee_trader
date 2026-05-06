"""
analyze_backtest_results.py

Analysis-only utility for walk-forward backtest outputs.
This script reads existing backtest result files and generates diagnostic
reports for strategy improvement work. It never mutates the original
backtest outputs and must not be connected to live trading code.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "backtest"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "backtest_analysis"
WEAK_MARKET_START = pd.Timestamp("2023-04-14")
WEAK_MARKET_END = pd.Timestamp("2024-12-31")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze backtest result files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve paths relative to project root."""
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    """Recursively sanitize values before JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _validate_required_inputs(input_dir: Path) -> None:
    """Ensure required input files exist."""
    required = [
        input_dir / "backtest_summary.json",
        input_dir / "daily_portfolio.csv",
        input_dir / "trades.csv",
        input_dir / "skipped_days.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required backtest inputs:\n" + "\n".join(missing))


def load_summary(path: Path) -> dict[str, Any]:
    """Load backtest_summary.json."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_daily_portfolio(path: Path) -> pd.DataFrame:
    """Load daily portfolio history with normalized types."""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    for col in [
        "cash",
        "equity",
        "position_value",
        "total_value",
        "daily_return",
        "cumulative_return",
        "planned_order_count",
        "executed_trade_count",
        "candidate_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def load_trades(path: Path) -> pd.DataFrame:
    """Load trades.csv with normalized numeric/date fields."""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    if "reason" in df.columns:
        df["reason"] = df["reason"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")
    if "name" not in df.columns:
        df["name"] = df.get("symbol", "").astype(str)
    for col in [
        "planned_price",
        "executed_price",
        "quantity",
        "amount",
        "fee",
        "tax",
        "slippage",
        "cost_basis_amount",
        "realized_pnl",
        "realized_return",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["trade_date"]).sort_values(["trade_date", "symbol", "side"]).reset_index(drop=True)


def load_skipped(path: Path) -> pd.DataFrame:
    """Load skipped day log."""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    return df


def attach_holding_days(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Attach FIFO-based holding days to SELL rows when BUY legs exist."""
    if trades_df.empty:
        trades_df["holding_days"] = pd.Series(dtype="float64")
        return trades_df

    work = trades_df.copy()
    work["holding_days"] = pd.NA
    buy_queues: dict[str, deque[dict[str, Any]]] = {}

    for idx, row in work.iterrows():
        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "").upper()
        qty = _safe_int(row.get("quantity"), 0)
        if qty <= 0:
            continue

        queue = buy_queues.setdefault(symbol, deque())
        if side == "BUY":
            queue.append({
                "trade_date": row["trade_date"],
                "quantity": qty,
            })
            continue

        if side != "SELL":
            continue

        remaining = qty
        matched_days: list[float] = []
        while remaining > 0 and queue:
            head = queue[0]
            matched = min(remaining, int(head["quantity"]))
            holding_days = max(int((row["trade_date"] - head["trade_date"]).days), 0)
            matched_days.extend([float(holding_days)] * matched)
            head["quantity"] -= matched
            remaining -= matched
            if head["quantity"] <= 0:
                queue.popleft()
        if matched_days:
            work.at[idx, "holding_days"] = sum(matched_days) / len(matched_days)

    return work


def compute_sharpe(daily_portfolio_df: pd.DataFrame) -> float:
    """Compute simple annualized Sharpe from daily returns."""
    if daily_portfolio_df.empty or "daily_return" not in daily_portfolio_df.columns:
        return 0.0
    returns = pd.to_numeric(daily_portfolio_df["daily_return"], errors="coerce").dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(252.0))


def compute_mdd(daily_portfolio_df: pd.DataFrame) -> float:
    """Compute max drawdown from total_value."""
    if daily_portfolio_df.empty or "total_value" not in daily_portfolio_df.columns:
        return 0.0
    values = pd.to_numeric(daily_portfolio_df["total_value"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    return float(drawdown.min())


def compute_total_return(initial_cash: float, final_total_value: float) -> float:
    """Compute cumulative return."""
    if initial_cash <= 0:
        return 0.0
    return float(final_total_value / initial_cash - 1.0)


def compute_cagr(initial_cash: float, final_total_value: float, daily_portfolio_df: pd.DataFrame) -> float:
    """Compute CAGR from date span."""
    if initial_cash <= 0 or final_total_value <= 0 or daily_portfolio_df.empty:
        return 0.0
    start_date = daily_portfolio_df["trade_date"].min()
    end_date = daily_portfolio_df["trade_date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0
    years = max((end_date - start_date).days, 1) / 365.25
    if years <= 0:
        return 0.0
    return float((final_total_value / initial_cash) ** (1.0 / years) - 1.0)


def compute_trade_metrics(trades_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute trade-level summary metrics from SELL rows."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
        }

    sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy()
    if sells.empty:
        return {
            "total_trades": int(len(trades_df)),
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
        }

    sell_returns = pd.to_numeric(sells.get("realized_return"), errors="coerce").dropna()
    sell_pnl = pd.to_numeric(sells.get("realized_pnl"), errors="coerce").dropna()

    wins = sell_returns[sell_returns > 0]
    losses = sell_returns[sell_returns < 0]
    pnl_wins = sell_pnl[sell_pnl > 0]
    pnl_losses = sell_pnl[sell_pnl < 0]

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    profit_factor = float(pnl_wins.sum() / abs(pnl_losses.sum())) if not pnl_losses.empty and abs(float(pnl_losses.sum())) > 0 else 0.0

    return {
        "total_trades": int(len(sells)),
        "win_rate": float((sell_returns > 0).mean()) if not sell_returns.empty else 0.0,
        "avg_return": float(sell_returns.mean()) if not sell_returns.empty else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
    }


def build_period_return_table(
    daily_portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Build yearly or monthly return summary table."""
    if daily_portfolio_df.empty:
        if freq == "Y":
            return pd.DataFrame(columns=["year", "start_value", "end_value", "return_pct", "max_drawdown", "trade_count"])
        return pd.DataFrame(columns=["year_month", "start_value", "end_value", "return_pct", "max_drawdown", "trade_count"])

    work = daily_portfolio_df.copy()
    if freq == "Y":
        work["period_key"] = work["trade_date"].dt.strftime("%Y")
        key_col = "year"
    else:
        work["period_key"] = work["trade_date"].dt.strftime("%Y-%m")
        key_col = "year_month"

    trade_counts = pd.DataFrame(columns=["period_key", "trade_count"])
    if not trades_df.empty:
        trade_work = trades_df.copy()
        trade_work["period_key"] = trade_work["trade_date"].dt.strftime("%Y" if freq == "Y" else "%Y-%m")
        trade_counts = trade_work.groupby("period_key", as_index=False).size().rename(columns={"size": "trade_count"})

    rows: list[dict[str, Any]] = []
    for period_key, group in work.groupby("period_key", sort=True):
        values = pd.to_numeric(group["total_value"], errors="coerce").dropna()
        if values.empty:
            continue
        running_peak = values.cummax()
        drawdown = values / running_peak - 1.0
        trade_count = 0
        match = trade_counts.loc[trade_counts["period_key"] == period_key]
        if not match.empty:
            trade_count = _safe_int(match["trade_count"].iloc[0], 0)
        rows.append({
            key_col: period_key,
            "start_value": float(values.iloc[0]),
            "end_value": float(values.iloc[-1]),
            "return_pct": float(values.iloc[-1] / values.iloc[0] - 1.0) if float(values.iloc[0]) != 0 else 0.0,
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "trade_count": int(trade_count),
        })
    return pd.DataFrame(rows)


def build_symbol_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build symbol-level realized performance table."""
    columns = [
        "symbol",
        "name",
        "trade_count",
        "win_count",
        "loss_count",
        "win_rate",
        "realized_pnl",
        "avg_return",
        "avg_holding_days",
        "total_fee",
        "total_tax",
        "total_slippage",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for symbol, group in sells.groupby("symbol", sort=True):
        returns = pd.to_numeric(group["realized_return"], errors="coerce")
        pnl = pd.to_numeric(group["realized_pnl"], errors="coerce")
        holding = pd.to_numeric(group.get("holding_days"), errors="coerce")
        rows.append({
            "symbol": symbol,
            "name": str(group["name"].dropna().iloc[0]) if "name" in group.columns and not group["name"].dropna().empty else symbol,
            "trade_count": int(len(group)),
            "win_count": int((returns > 0).sum()),
            "loss_count": int((returns < 0).sum()),
            "win_rate": float((returns > 0).mean()) if returns.notna().any() else 0.0,
            "realized_pnl": float(pnl.sum()) if pnl.notna().any() else 0.0,
            "avg_return": float(returns.mean()) if returns.notna().any() else 0.0,
            "avg_holding_days": float(holding.mean()) if holding.notna().any() else 0.0,
            "total_fee": float(pd.to_numeric(group["fee"], errors="coerce").fillna(0.0).sum()),
            "total_tax": float(pd.to_numeric(group["tax"], errors="coerce").fillna(0.0).sum()),
            "total_slippage": float(pd.to_numeric(group["slippage"], errors="coerce").fillna(0.0).sum()),
        })
    return pd.DataFrame(rows).sort_values(["realized_pnl", "trade_count"], ascending=[True, False]).reset_index(drop=True)


def build_exit_reason_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build exit-reason level realized performance table."""
    columns = ["reason", "trade_count", "win_rate", "realized_pnl", "avg_return", "avg_win", "avg_loss", "payoff_ratio"]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame(columns=columns)
    sells["reason"] = sells["reason"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")

    rows: list[dict[str, Any]] = []
    for reason, group in sells.groupby("reason", sort=True):
        returns = pd.to_numeric(group["realized_return"], errors="coerce").dropna()
        pnl = pd.to_numeric(group["realized_pnl"], errors="coerce").dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
        rows.append({
            "reason": reason or "UNKNOWN",
            "trade_count": int(len(group)),
            "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
            "realized_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
            "avg_return": float(returns.mean()) if not returns.empty else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
        })
    return pd.DataFrame(rows).sort_values(["realized_pnl", "trade_count"], ascending=[True, False]).reset_index(drop=True)


def build_drawdown_periods(daily_portfolio_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Extract top drawdown periods from total_value history."""
    columns = ["start_date", "trough_date", "recovery_date", "drawdown_pct", "duration_days", "recovered"]
    if daily_portfolio_df.empty:
        return pd.DataFrame(columns=columns)

    df = daily_portfolio_df[["trade_date", "total_value"]].copy()
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    df = df.dropna(subset=["trade_date", "total_value"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=columns)

    peak_value = None
    peak_date = None
    in_dd = False
    trough_value = None
    trough_date = None
    periods: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        date = pd.Timestamp(row.trade_date)
        value = float(row.total_value)

        if peak_value is None or value >= peak_value:
            if in_dd and peak_date is not None and trough_value is not None and trough_date is not None:
                periods.append({
                    "start_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": date,
                    "drawdown_pct": float(trough_value / peak_value - 1.0),
                    "duration_days": int((date - peak_date).days),
                    "recovered": True,
                })
                in_dd = False
                trough_value = None
                trough_date = None
            peak_value = value
            peak_date = date
            continue

        current_drawdown = value / peak_value - 1.0
        if not in_dd:
            in_dd = True
            trough_value = value
            trough_date = date
        elif current_drawdown < float(trough_value / peak_value - 1.0):
            trough_value = value
            trough_date = date

    if in_dd and peak_date is not None and trough_value is not None and trough_date is not None:
        periods.append({
            "start_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": pd.NaT,
            "drawdown_pct": float(trough_value / peak_value - 1.0),
            "duration_days": int((df["trade_date"].iloc[-1] - peak_date).days),
            "recovered": False,
        })

    out = pd.DataFrame(periods)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.sort_values("drawdown_pct", ascending=True).head(limit).reset_index(drop=True)
    for col in ["start_date", "trough_date", "recovery_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def build_stop_loss_analysis(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-trade stop-loss analysis table."""
    columns = ["trade_date", "symbol", "name", "realized_pnl", "return_pct", "holding_days", "amount", "fee", "tax", "slippage"]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy()
    mask = sells["reason"].fillna("").astype(str).str.contains("stop_loss", case=False, na=False)
    stop_df = sells.loc[mask].copy()
    if stop_df.empty:
        return pd.DataFrame(columns=columns)

    out = stop_df.rename(columns={"realized_return": "return_pct"})[
        ["trade_date", "symbol", "name", "realized_pnl", "return_pct", "holding_days", "amount", "fee", "tax", "slippage"]
    ].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values(["trade_date", "realized_pnl"], ascending=[True, True]).reset_index(drop=True)


def build_weak_market_analysis(
    daily_portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row weak market regime summary for 2023-04-14 to 2024-12-31."""
    columns = [
        "period_start",
        "period_end",
        "return_pct",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "avg_return",
        "stop_loss_ratio",
        "top_loss_symbol_1",
        "top_loss_symbol_2",
        "top_loss_symbol_3",
        "top_loss_reason_1",
        "top_loss_reason_2",
        "top_loss_reason_3",
    ]
    if daily_portfolio_df.empty:
        return pd.DataFrame(columns=columns)

    daily_slice = daily_portfolio_df.loc[
        daily_portfolio_df["trade_date"].between(WEAK_MARKET_START, WEAK_MARKET_END)
    ].copy()
    if daily_slice.empty:
        return pd.DataFrame(columns=columns)

    values = pd.to_numeric(daily_slice["total_value"], errors="coerce").dropna()
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0 if not values.empty else pd.Series(dtype="float64")

    sells = pd.DataFrame()
    if not trades_df.empty:
        sells = trades_df.loc[
            (trades_df["side"].astype(str).str.upper() == "SELL")
            & (trades_df["trade_date"].between(WEAK_MARKET_START, WEAK_MARKET_END))
        ].copy()

    returns = pd.to_numeric(sells.get("realized_return"), errors="coerce").dropna() if not sells.empty else pd.Series(dtype="float64")
    stop_loss_ratio = 0.0
    if not sells.empty:
        stop_loss_ratio = float(sells["reason"].fillna("").astype(str).str.contains("stop_loss", case=False, na=False).mean())

    top_symbols: list[str] = []
    top_reasons: list[str] = []
    if not sells.empty:
        symbol_loss = (
            sells.groupby("symbol", as_index=False)["realized_pnl"]
            .sum()
            .sort_values("realized_pnl", ascending=True)
            .head(3)
        )
        reason_loss = (
            sells.groupby("reason", as_index=False)["realized_pnl"]
            .sum()
            .sort_values("realized_pnl", ascending=True)
            .head(3)
        )
        top_symbols = [str(v) for v in symbol_loss["symbol"].tolist()]
        top_reasons = [str(v) for v in reason_loss["reason"].tolist()]

    row = {
        "period_start": WEAK_MARKET_START.strftime("%Y-%m-%d"),
        "period_end": WEAK_MARKET_END.strftime("%Y-%m-%d"),
        "return_pct": float(values.iloc[-1] / values.iloc[0] - 1.0) if len(values) >= 2 and float(values.iloc[0]) != 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "trade_count": int(len(sells)),
        "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "avg_return": float(returns.mean()) if not returns.empty else 0.0,
        "stop_loss_ratio": stop_loss_ratio,
        "top_loss_symbol_1": top_symbols[0] if len(top_symbols) > 0 else "",
        "top_loss_symbol_2": top_symbols[1] if len(top_symbols) > 1 else "",
        "top_loss_symbol_3": top_symbols[2] if len(top_symbols) > 2 else "",
        "top_loss_reason_1": top_reasons[0] if len(top_reasons) > 0 else "",
        "top_loss_reason_2": top_reasons[1] if len(top_reasons) > 1 else "",
        "top_loss_reason_3": top_reasons[2] if len(top_reasons) > 2 else "",
    }
    return pd.DataFrame([row], columns=columns)


def build_summary(
    backtest_summary: dict[str, Any],
    daily_portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    stop_loss_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build top-level analysis summary JSON."""
    summary_perf = backtest_summary.get("performance", {}) if isinstance(backtest_summary, dict) else {}
    initial_cash = _safe_float(summary_perf.get("initial_cash"), 0.0)
    if initial_cash <= 0:
        initial_cash = _safe_float(backtest_summary.get("config", {}).get("initial_cash"), 0.0)
    final_total_value = _safe_float(summary_perf.get("final_total_value"), 0.0)
    if final_total_value <= 0 and not daily_portfolio_df.empty:
        final_total_value = _safe_float(daily_portfolio_df["total_value"].iloc[-1], 0.0)

    trade_metrics = compute_trade_metrics(trades_df)
    mdd = compute_mdd(daily_portfolio_df)
    sharpe = compute_sharpe(daily_portfolio_df)
    cagr = compute_cagr(initial_cash, final_total_value, daily_portfolio_df)
    total_return = compute_total_return(initial_cash, final_total_value)

    stop_loss_total = 0.0
    stop_loss_avg_loss = 0.0
    if not stop_loss_df.empty:
        stop_loss_total = float(pd.to_numeric(stop_loss_df["realized_pnl"], errors="coerce").fillna(0.0).sum())
        stop_loss_returns = pd.to_numeric(stop_loss_df["return_pct"], errors="coerce").dropna()
        stop_loss_avg_loss = float(stop_loss_returns.mean()) if not stop_loss_returns.empty else 0.0

    return {
        "initial_cash": initial_cash,
        "final_total_value": final_total_value,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "total_trades": trade_metrics["total_trades"],
        "win_rate": trade_metrics["win_rate"],
        "avg_return": trade_metrics["avg_return"],
        "avg_win": trade_metrics["avg_win"],
        "avg_loss": trade_metrics["avg_loss"],
        "payoff_ratio": trade_metrics["payoff_ratio"],
        "profit_factor": trade_metrics["profit_factor"],
        "stop_loss_trade_count": int(len(stop_loss_df)),
        "stop_loss_avg_loss": stop_loss_avg_loss,
        "stop_loss_total_pnl": stop_loss_total,
        "baseline_reference": backtest_summary.get("baseline_reference"),
    }


def _fmt_pct(value: Any) -> str:
    """Format return values as percentage strings."""
    numeric = _safe_float(value, 0.0)
    return f"{numeric * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    """Format numeric values for markdown."""
    numeric = _safe_float(value, 0.0)
    return f"{numeric:,.2f}"


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_none_"
    work = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit is not None:
        work = work.head(limit).copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].dt.strftime("%Y-%m-%d")
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


def build_analysis_report(
    analysis_summary: dict[str, Any],
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    exit_reason_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    stop_loss_df: pd.DataFrame,
    weak_market_df: pd.DataFrame,
) -> str:
    """Build markdown analysis report."""
    stop_loss_impact = analysis_summary.get("stop_loss_total_pnl", 0.0)
    stop_loss_avg_loss = analysis_summary.get("stop_loss_avg_loss", 0.0)
    observations: list[str] = []

    if not exit_reason_df.empty:
        worst_reason = exit_reason_df.sort_values("realized_pnl", ascending=True).iloc[0]
        observations.append(
            f"- 가장 손익이 나쁜 종료 사유는 `{worst_reason['reason']}` 이며 realized_pnl={_fmt_num(worst_reason['realized_pnl'])} 입니다."
        )
    if not symbol_df.empty:
        worst_symbol = symbol_df.sort_values("realized_pnl", ascending=True).iloc[0]
        observations.append(
            f"- 손실 기여가 가장 큰 종목은 `{worst_symbol['symbol']}` 이며 realized_pnl={_fmt_num(worst_symbol['realized_pnl'])} 입니다."
        )
    if not drawdown_df.empty:
        worst_dd = drawdown_df.iloc[0]
        observations.append(
            f"- 최대 낙폭 구간은 `{worst_dd['start_date']}` 시작, `{worst_dd['trough_date']}` 저점, drawdown={_fmt_pct(worst_dd['drawdown_pct'])} 입니다."
        )
    if stop_loss_df.empty:
        observations.append("- `stop_loss` 종료 거래가 없어 별도 손절 분석 표는 비어 있습니다.")

    experiments = [
        "- stop_loss 조건별 성과 비교",
        "- 2023~2024 약세장 sector 필터 강화",
        "- 거래대금 필터 강화",
        "- gap 상승/하락 필터 추가",
        "- cooldown 적용 여부 비교",
        "- trailing_stop_reduce 이후 잔여 포지션 성과 분리",
    ]

    lines = [
        "# Backtest Analysis Report",
        "",
        "## 1. Summary",
        f"- initial_cash: `{_fmt_num(analysis_summary.get('initial_cash'))}`",
        f"- final_total_value: `{_fmt_num(analysis_summary.get('final_total_value'))}`",
        f"- total_return: `{_fmt_pct(analysis_summary.get('total_return'))}`",
        f"- CAGR: `{_fmt_pct(analysis_summary.get('cagr'))}`",
        f"- MDD: `{_fmt_pct(analysis_summary.get('mdd'))}`",
        f"- Sharpe: `{analysis_summary.get('sharpe', 0.0):.4f}`",
        f"- total_trades: `{analysis_summary.get('total_trades', 0)}`",
        f"- win_rate: `{_fmt_pct(analysis_summary.get('win_rate'))}`",
        f"- avg_return: `{_fmt_pct(analysis_summary.get('avg_return'))}`",
        f"- avg_win: `{_fmt_pct(analysis_summary.get('avg_win'))}`",
        f"- avg_loss: `{_fmt_pct(analysis_summary.get('avg_loss'))}`",
        f"- payoff_ratio: `{analysis_summary.get('payoff_ratio', 0.0):.4f}`",
        f"- profit_factor: `{analysis_summary.get('profit_factor', 0.0):.4f}`",
        "",
        "## 2. Yearly Performance",
        _markdown_table(yearly_df, ["year", "start_value", "end_value", "return_pct", "max_drawdown", "trade_count"]),
        "",
        "## 3. Monthly Performance",
        _markdown_table(monthly_df, ["year_month", "start_value", "end_value", "return_pct", "max_drawdown", "trade_count"], limit=24),
        "",
        "## 4. Symbol Performance",
        _markdown_table(symbol_df, ["symbol", "name", "trade_count", "win_rate", "realized_pnl", "avg_return", "avg_holding_days"], limit=15),
        "",
        "## 5. Exit Reason Performance",
        _markdown_table(exit_reason_df, ["reason", "trade_count", "win_rate", "realized_pnl", "avg_return", "payoff_ratio"]),
        "",
        "## 6. Drawdown Analysis",
        _markdown_table(drawdown_df, ["start_date", "trough_date", "recovery_date", "drawdown_pct", "duration_days", "recovered"]),
        "",
        "## 7. Stop Loss Analysis",
        f"- stop_loss 거래 수: `{analysis_summary.get('stop_loss_trade_count', 0)}`",
        f"- stop_loss 평균 손실률: `{_fmt_pct(stop_loss_avg_loss)}`",
        f"- stop_loss 총 손실금: `{_fmt_num(stop_loss_impact)}`",
        f"- stop_loss 손익 영향: `{_fmt_num(stop_loss_impact)}`",
        _markdown_table(stop_loss_df, ["trade_date", "symbol", "name", "realized_pnl", "return_pct", "holding_days"], limit=20),
        "",
        "## 8. Weak Market Analysis: 2023~2024",
        _markdown_table(
            weak_market_df,
            [
                "period_start",
                "period_end",
                "return_pct",
                "max_drawdown",
                "trade_count",
                "win_rate",
                "avg_return",
                "stop_loss_ratio",
                "top_loss_symbol_1",
                "top_loss_reason_1",
            ],
        ),
        "",
        "## 9. Observations",
        *(observations or ["- 추가 관찰 포인트가 아직 충분히 쌓이지 않았습니다."]),
        "",
        "## 10. Recommended Next Experiments",
        *experiments,
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    output_dir: Path,
    analysis_summary: dict[str, Any],
    report_md: str,
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    exit_reason_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    stop_loss_df: pd.DataFrame,
    weak_market_df: pd.DataFrame,
) -> None:
    """Persist all analysis outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(_json_safe(analysis_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "analysis_report.md").write_text(report_md, encoding="utf-8")
    yearly_df.to_csv(output_dir / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(output_dir / "monthly_returns.csv", index=False, encoding="utf-8-sig")
    symbol_df.to_csv(output_dir / "symbol_performance.csv", index=False, encoding="utf-8-sig")
    exit_reason_df.to_csv(output_dir / "exit_reason_performance.csv", index=False, encoding="utf-8-sig")
    drawdown_df.to_csv(output_dir / "drawdown_periods.csv", index=False, encoding="utf-8-sig")
    stop_loss_df.to_csv(output_dir / "stop_loss_analysis.csv", index=False, encoding="utf-8-sig")
    weak_market_df.to_csv(output_dir / "weak_market_analysis.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    _validate_required_inputs(input_dir)

    backtest_summary = load_summary(input_dir / "backtest_summary.json")
    daily_portfolio_df = load_daily_portfolio(input_dir / "daily_portfolio.csv")
    trades_df = attach_holding_days(load_trades(input_dir / "trades.csv"))
    skipped_df = load_skipped(input_dir / "skipped_days.csv")
    _ = skipped_df  # reserved for future report extensions

    yearly_df = build_period_return_table(daily_portfolio_df, trades_df, freq="Y")
    monthly_df = build_period_return_table(daily_portfolio_df, trades_df, freq="M")
    symbol_df = build_symbol_performance(trades_df)
    exit_reason_df = build_exit_reason_performance(trades_df)
    drawdown_df = build_drawdown_periods(daily_portfolio_df, limit=10)
    stop_loss_df = build_stop_loss_analysis(trades_df)
    weak_market_df = build_weak_market_analysis(daily_portfolio_df, trades_df)
    analysis_summary = build_summary(backtest_summary, daily_portfolio_df, trades_df, stop_loss_df)
    report_md = build_analysis_report(
        analysis_summary,
        yearly_df,
        monthly_df,
        symbol_df,
        exit_reason_df,
        drawdown_df,
        stop_loss_df,
        weak_market_df,
    )
    save_outputs(
        output_dir,
        analysis_summary,
        report_md,
        yearly_df,
        monthly_df,
        symbol_df,
        exit_reason_df,
        drawdown_df,
        stop_loss_df,
        weak_market_df,
    )


if __name__ == "__main__":
    main()
