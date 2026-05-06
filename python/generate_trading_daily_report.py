"""
generate_trading_daily_report.py

Generate daily monitoring reports for forward test, paper trading, and
live trading outputs. This script is read-only: it never places orders,
modifies configs, or touches live APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "daily_reports"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate daily trading report.")
    parser.add_argument("--date", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--mode", choices=["paper", "live", "forward"], default="paper")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve path relative to repo root."""
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _json_safe(value: Any) -> Any:
    """Recursively sanitize values for JSON output."""
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
    if isinstance(value, float) and (pd.isna(value) or value == float("inf") or value == float("-inf")):
        return None
    return value


def _load_json_if_exists(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Load JSON if present."""
    if not path.exists():
        return None, False
    return json.loads(path.read_text(encoding="utf-8")), True


def _load_csv_if_exists(path: Path) -> tuple[pd.DataFrame, bool]:
    """Load CSV if present."""
    if not path.exists():
        return pd.DataFrame(), False
    df = pd.read_csv(path, low_memory=False)
    return df, True


def _find_selection_dir(input_dir: Path) -> Path | None:
    """Locate selection output directory if present."""
    direct = input_dir / "backtest_experiments" / "selection"
    if direct.exists():
        return direct
    smoke = input_dir / "backtest_experiments_smoke" / "selection"
    if smoke.exists():
        return smoke
    return None


def _find_forward_root(input_dir: Path) -> Path | None:
    """Locate forward test root if present."""
    direct = input_dir / "forward_test"
    if direct.exists():
        return direct
    smoke_batch = input_dir / "forward_test_smoke_batch"
    if smoke_batch.exists():
        return smoke_batch
    smoke_daily = input_dir / "forward_test_smoke_daily"
    if smoke_daily.exists():
        return smoke_daily
    return None


def _find_live_root(input_dir: Path) -> Path | None:
    """Locate live trading root if present."""
    direct = input_dir / "live_trading"
    if direct.exists():
        return direct
    for name in ["live_trading_smoke3", "live_trading_smoke2", "live_trading_smoke"]:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    return None


def _normalize_trade_dates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize one date column if present."""
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df


def _profile_forward_stats(profile_dir: Path, report_date: pd.Timestamp) -> dict[str, Any]:
    """Compute one profile's forward/paper portfolio stats."""
    portfolio_df, portfolio_exists = _load_csv_if_exists(profile_dir / "portfolio.csv")
    trades_df, trades_exists = _load_csv_if_exists(profile_dir / "trades.csv")
    skipped_df, skipped_exists = _load_csv_if_exists(profile_dir / "skipped_days.csv")
    state_json, state_exists = _load_json_if_exists(profile_dir / "state.json")

    if portfolio_exists:
        portfolio_df = _normalize_trade_dates(portfolio_df, "trade_date")
        for col in ["cash", "equity", "position_value", "total_value", "daily_return", "cumulative_return"]:
            if col in portfolio_df.columns:
                portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors="coerce")

    if trades_exists:
        trades_df = _normalize_trade_dates(trades_df, "trade_date")
        for col in ["realized_pnl", "realized_return"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

    today_portfolio = portfolio_df.loc[portfolio_df["trade_date"] == report_date].copy() if portfolio_exists and "trade_date" in portfolio_df.columns else pd.DataFrame()
    latest_portfolio = today_portfolio.iloc[-1] if not today_portfolio.empty else (portfolio_df.iloc[-1] if portfolio_exists and not portfolio_df.empty else None)
    current_value = _safe_float(latest_portfolio["total_value"]) if latest_portfolio is not None and "total_value" in latest_portfolio else None
    daily_return = _safe_float(latest_portfolio["daily_return"]) if latest_portfolio is not None and "daily_return" in latest_portfolio else None
    cumulative_return = _safe_float(latest_portfolio["cumulative_return"]) if latest_portfolio is not None and "cumulative_return" in latest_portfolio else None

    mdd = None
    if portfolio_exists and not portfolio_df.empty and "total_value" in portfolio_df.columns:
        values = pd.to_numeric(portfolio_df["total_value"], errors="coerce").dropna()
        if not values.empty:
            running_peak = values.cummax()
            drawdown = values / running_peak - 1.0
            mdd = float(drawdown.min())

    trade_count = 0
    win_rate = None
    realized_pnl = 0.0
    if trades_exists and not trades_df.empty:
        sells = trades_df.loc[trades_df["side"].astype(str).str.upper() == "SELL"].copy() if "side" in trades_df.columns else pd.DataFrame()
        trade_count = int(len(sells))
        if not sells.empty and "realized_return" in sells.columns:
            valid_returns = pd.to_numeric(sells["realized_return"], errors="coerce").dropna()
            if not valid_returns.empty:
                win_rate = float((valid_returns > 0).mean())
        if "realized_pnl" in sells.columns:
            realized_pnl = float(pd.to_numeric(sells["realized_pnl"], errors="coerce").fillna(0.0).sum())

    open_positions = 0
    unrealized_pnl = None
    if state_exists and isinstance(state_json, dict):
        positions = state_json.get("positions", {}) or {}
        open_positions = len(positions)
        unrealized_pnl = 0.0
        for position in positions.values():
            qty = _safe_float(position.get("quantity"))
            last_price = _safe_float(position.get("last_price"))
            avg_price = _safe_float(position.get("avg_price"))
            unrealized_pnl += qty * (last_price - avg_price)

    blocked_orders = 0
    if skipped_exists and not skipped_df.empty and "trade_date" in skipped_df.columns:
        skipped_df = _normalize_trade_dates(skipped_df, "trade_date")
        blocked_orders = int(len(skipped_df.loc[skipped_df["trade_date"] == report_date]))

    return {
        "current_value": current_value,
        "daily_return": daily_return,
        "cumulative_return": cumulative_return,
        "MDD": mdd,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "open_positions": open_positions,
        "blocked_orders": blocked_orders,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "portfolio_missing": not portfolio_exists,
        "trades_missing": not trades_exists,
        "state_missing": not state_exists,
    }


def _load_selected_configs(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load selected configs table if present."""
    selection_dir = _find_selection_dir(input_dir)
    missing: list[str] = []
    if selection_dir is None:
        return pd.DataFrame(), ["selected_configs"]
    selected_df, exists = _load_csv_if_exists(selection_dir / "selected_configs.csv")
    if not exists:
        missing.append("selected_configs.csv")
    return selected_df, missing


def _build_backtest_expectation_map(selected_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Map selected profile -> expected backtest metrics."""
    if selected_df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in selected_df.iterrows():
        out[str(row.get("profile"))] = {
            "CAGR": _safe_float(row.get("CAGR")),
            "MDD": _safe_float(row.get("MDD")),
            "Sharpe": _safe_float(row.get("Sharpe")),
            "win_rate": _safe_float(row.get("win_rate")),
        }
    return out


def _generate_observations(summary: dict[str, Any], profiles: dict[str, dict[str, Any]], comparisons: dict[str, Any], risk_events_df: pd.DataFrame) -> list[str]:
    """Generate simple automatic observations."""
    notes: list[str] = []
    if int(summary.get("executed_trades", 0)) == 0:
        notes.append("No trades today")
    if _safe_float(summary.get("daily_return")) <= -0.02:
        notes.append("Daily loss exceeded warning level")
    if not risk_events_df.empty:
        notes.append("Risk manager blocked orders")
    for profile, comparison in comparisons.items():
        if comparison.get("status") == "underperforming":
            notes.append(f"{profile} forward return is below backtest expectation")
        elif comparison.get("status") == "insufficient_sample":
            notes.append(f"{profile} comparison is insufficient sample")
    if not notes:
        notes.append("No major anomalies detected from the available files")
    return notes


def _fmt_pct(value: Any) -> str:
    """Format decimal return-like values as percent."""
    if value is None or pd.isna(value):
        return "missing"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    """Format numeric value."""
    if value is None or pd.isna(value):
        return "missing"
    return f"{float(value):,.2f}"


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """Render compact markdown table."""
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


def build_daily_report(report_date: pd.Timestamp, mode: str, input_dir: Path) -> tuple[dict[str, Any], str]:
    """Build both JSON payload and markdown report."""
    forward_root = _find_forward_root(input_dir)
    live_root = _find_live_root(input_dir)
    selected_df, missing_selected = _load_selected_configs(input_dir)
    expectation_map = _build_backtest_expectation_map(selected_df)

    missing_files: list[str] = list(missing_selected)
    profiles_monitored: list[str] = []
    profile_stats: dict[str, dict[str, Any]] = {}

    if forward_root is not None:
        for profile_dir in sorted([p for p in forward_root.iterdir() if p.is_dir()]):
            profiles_monitored.append(profile_dir.name)
            profile_stats[profile_dir.name] = _profile_forward_stats(profile_dir, report_date)
    else:
        missing_files.append("forward_test_root")

    live_orders_df, live_orders_exists = _load_csv_if_exists(live_root / "orders_log.csv") if live_root else (pd.DataFrame(), False)
    live_trades_df, live_trades_exists = _load_csv_if_exists(live_root / "trades_log.csv") if live_root else (pd.DataFrame(), False)
    risk_events_df, risk_events_exists = _load_csv_if_exists(live_root / "risk_events.csv") if live_root else (pd.DataFrame(), False)
    live_status_json, live_status_exists = _load_json_if_exists(live_root / "live_trading_status.json") if live_root else (None, False)

    if not live_orders_exists:
        missing_files.append("orders_log.csv")
    if not live_trades_exists:
        missing_files.append("trades_log.csv")
    if not risk_events_exists:
        missing_files.append("risk_events.csv")
    if not live_status_exists:
        missing_files.append("live_trading_status.json")

    if live_orders_exists:
        live_orders_df = _normalize_trade_dates(live_orders_df.rename(columns={"date": "trade_date"}), "trade_date")
    if live_trades_exists:
        live_trades_df = _normalize_trade_dates(live_trades_df.rename(columns={"date": "trade_date"}), "trade_date")
    if risk_events_exists:
        risk_events_df = _normalize_trade_dates(risk_events_df.rename(columns={"date": "trade_date"}), "trade_date")

    total_orders = int(len(live_orders_df.loc[live_orders_df["trade_date"] == report_date])) if live_orders_exists and "trade_date" in live_orders_df.columns else 0
    executed_trades = int(len(live_trades_df.loc[live_trades_df["trade_date"] == report_date])) if live_trades_exists and "trade_date" in live_trades_df.columns else 0
    blocked_orders = int(len(risk_events_df.loc[risk_events_df["trade_date"] == report_date])) if risk_events_exists and "trade_date" in risk_events_df.columns else 0
    skipped_orders = int(sum(_safe_float(item.get("blocked_orders"), 0.0) for item in profile_stats.values()))
    realized_pnl = float(sum(_safe_float(item.get("realized_pnl"), 0.0) for item in profile_stats.values()))
    unrealized_pnl = float(sum(_safe_float(item.get("unrealized_pnl"), 0.0) for item in profile_stats.values() if item.get("unrealized_pnl") is not None))
    total_value = float(sum(_safe_float(item.get("current_value"), 0.0) for item in profile_stats.values()))
    daily_return_candidates = [item.get("daily_return") for item in profile_stats.values() if item.get("daily_return") is not None]
    daily_return = float(sum(map(float, daily_return_candidates)) / len(daily_return_candidates)) if daily_return_candidates else None

    risk_reason_summary = pd.DataFrame()
    risk_symbol_summary = pd.DataFrame()
    if risk_events_exists and not risk_events_df.empty:
        risk_reason_summary = (
            risk_events_df.groupby("reason", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        if "symbol" in risk_events_df.columns:
            risk_symbol_summary = (
                risk_events_df.groupby("symbol", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )

    comparisons: dict[str, Any] = {}
    for profile, stats in profile_stats.items():
        expected = expectation_map.get(profile)
        if expected is None:
            comparisons[profile] = {"status": "missing_expected_backtest"}
            continue
        running_days = int(stats.get("trade_count") or 0)
        if running_days < 5:
            comparisons[profile] = {"status": "insufficient_sample"}
            continue
        comparisons[profile] = {
            "status": "underperforming" if _safe_float(stats.get("win_rate"), 0.0) < _safe_float(expected.get("win_rate"), 0.0) else "ok",
            "expected_CAGR": expected.get("CAGR"),
            "expected_MDD": expected.get("MDD"),
            "expected_Sharpe": expected.get("Sharpe"),
            "expected_win_rate": expected.get("win_rate"),
            "current_cumulative_return": stats.get("cumulative_return"),
        }

    summary = {
        "date": report_date.strftime("%Y-%m-%d"),
        "mode": mode,
        "profiles_monitored": profiles_monitored,
        "total_orders": total_orders,
        "executed_trades": executed_trades,
        "skipped_orders": skipped_orders,
        "blocked_orders": blocked_orders,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_value": total_value,
        "daily_return": daily_return,
        "missing_files": missing_files,
    }

    observations = _generate_observations(summary, profile_stats, comparisons, risk_events_df if risk_events_exists else pd.DataFrame())
    payload = {
        "summary": summary,
        "profiles": profile_stats,
        "risk_reason_summary": risk_reason_summary.to_dict(orient="records"),
        "risk_symbol_summary": risk_symbol_summary.to_dict(orient="records"),
        "comparisons": comparisons,
        "observations": observations,
    }

    profile_rows = []
    for profile, stats in profile_stats.items():
        profile_rows.append({
            "profile": profile,
            "current_value": _fmt_num(stats.get("current_value")),
            "daily_return": _fmt_pct(stats.get("daily_return")),
            "cumulative_return": _fmt_pct(stats.get("cumulative_return")),
            "MDD": _fmt_pct(stats.get("MDD")),
            "trade_count": stats.get("trade_count"),
            "win_rate": _fmt_pct(stats.get("win_rate")),
            "open_positions": stats.get("open_positions"),
            "blocked_orders": stats.get("blocked_orders"),
        })
    profile_df = pd.DataFrame(profile_rows)

    lines = [
        f"# Daily Trading Report: {report_date.strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        f"- date: `{summary['date']}`",
        f"- mode: `{summary['mode']}`",
        f"- profiles monitored: `{', '.join(profiles_monitored) if profiles_monitored else 'missing'}`",
        f"- total orders: `{summary['total_orders']}`",
        f"- executed trades: `{summary['executed_trades']}`",
        f"- skipped orders: `{summary['skipped_orders']}`",
        f"- blocked orders: `{summary['blocked_orders']}`",
        f"- realized pnl: `{_fmt_num(summary['realized_pnl'])}`",
        f"- unrealized pnl: `{_fmt_num(summary['unrealized_pnl'])}`",
        f"- total value: `{_fmt_num(summary['total_value'])}`",
        f"- daily return: `{_fmt_pct(summary['daily_return'])}`",
        f"- missing files: `{', '.join(summary['missing_files']) if summary['missing_files'] else 'none'}`",
        "",
        "## Profile별 현황",
        _markdown_table(profile_df, ["profile", "current_value", "daily_return", "cumulative_return", "MDD", "trade_count", "win_rate", "open_positions", "blocked_orders"]),
        "",
        "## Risk Events",
        "### By Reason",
        _markdown_table(risk_reason_summary, ["reason", "count"]),
        "",
        "### By Symbol",
        _markdown_table(risk_symbol_summary, ["symbol", "count"]),
        "",
        "## Trading Events",
        "### Trades",
        _markdown_table(live_trades_df if live_trades_exists else pd.DataFrame(), ["trade_date", "profile", "symbol", "side", "quantity", "reason", "status"], limit=20),
        "",
        "### Orders",
        _markdown_table(live_orders_df if live_orders_exists else pd.DataFrame(), ["trade_date", "profile", "symbol", "side", "quantity", "reason", "status"], limit=20),
        "",
        "### Skips",
        _markdown_table(risk_events_df if risk_events_exists else pd.DataFrame(), ["trade_date", "symbol", "reason", "blocked_order"], limit=20),
        "",
        "## Backtest vs Forward/Live Comparison",
        _markdown_table(pd.DataFrame([{"profile": k, **v} for k, v in comparisons.items()]), ["profile", "status", "expected_CAGR", "expected_MDD", "expected_Sharpe", "expected_win_rate", "current_cumulative_return"]),
        "",
        "## Observations",
        *[f"- {item}" for item in observations],
        "",
    ]
    return payload, "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    report_date = pd.Timestamp(args.date).normalize()
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, markdown = build_daily_report(report_date, args.mode, input_dir)
    json_path = output_dir / f"daily_report_{report_date.strftime('%Y-%m-%d')}.json"
    md_path = output_dir / f"daily_report_{report_date.strftime('%Y-%m-%d')}.md"

    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
