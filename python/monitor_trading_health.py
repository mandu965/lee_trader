"""
monitor_trading_health.py

Daily health checks for forward/paper/live trading artifacts.
This script is read-only and never places orders or changes configs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "health_checks"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run trading health checks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve path relative to repo root."""
    return path if path.is_absolute() else ROOT / path


def _json_safe(value: Any) -> Any:
    """Sanitize objects for JSON serialization."""
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce to finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _load_json_if_exists(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Load JSON if present."""
    if not path.exists():
        return None, False
    return json.loads(path.read_text(encoding="utf-8")), True


def _load_csv_if_exists(path: Path) -> tuple[pd.DataFrame, bool]:
    """Load CSV if present."""
    if not path.exists():
        return pd.DataFrame(), False
    return pd.read_csv(path, low_memory=False), True


def _find_forward_root(input_dir: Path) -> Path | None:
    """Locate forward output root."""
    for name in ["forward_test", "forward_test_smoke_batch", "forward_test_smoke_daily"]:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    return None


def _find_live_root(input_dir: Path) -> Path | None:
    """Locate live output root."""
    for name in ["live_trading", "live_trading_smoke3", "live_trading_smoke2", "live_trading_smoke"]:
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    return None


def _find_selection_dir(input_dir: Path) -> Path | None:
    """Locate config selection output directory."""
    for path in [
        input_dir / "backtest_experiments" / "selection",
        input_dir / "backtest_experiments_smoke" / "selection",
    ]:
        if path.exists():
            return path
    return None


def _normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalize one date column."""
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def run_health_checks(input_dir: Path) -> dict[str, Any]:
    """Run health checks and return structured result."""
    checked_at = pd.Timestamp.now()
    warnings: list[str] = []
    criticals: list[str] = []
    missing_files: list[str] = []

    forward_root = _find_forward_root(input_dir)
    live_root = _find_live_root(input_dir)
    selection_dir = _find_selection_dir(input_dir)

    selected_profiles: set[str] = set()
    if selection_dir is not None:
        selected_df, exists = _load_csv_if_exists(selection_dir / "selected_configs.csv")
        if exists and not selected_df.empty and "profile" in selected_df.columns:
            selected_profiles = set(selected_df["profile"].astype(str))
        else:
            missing_files.append("selected_configs.csv")
    else:
        missing_files.append("selection_dir")

    current_date = None
    if live_root is not None:
        live_status, status_exists = _load_json_if_exists(live_root / "live_trading_status.json")
        orders_df, orders_exists = _load_csv_if_exists(live_root / "orders_log.csv")
        trades_df, trades_exists = _load_csv_if_exists(live_root / "trades_log.csv")
        risk_df, risk_exists = _load_csv_if_exists(live_root / "risk_events.csv")
        if not status_exists:
            missing_files.append("live_trading_status.json")
        if not orders_exists:
            missing_files.append("orders_log.csv")
        if not trades_exists:
            missing_files.append("trades_log.csv")
        if not risk_exists:
            missing_files.append("risk_events.csv")

        if status_exists and isinstance(live_status, dict):
            current_date = pd.to_datetime(live_status.get("trade_date"), errors="coerce")

        if orders_exists:
            orders_df = _normalize_date_col(orders_df.rename(columns={"date": "trade_date"}), "trade_date")
            required_order_cols = {"trade_date", "profile", "symbol", "side", "status"}
            missing_order_cols = sorted(required_order_cols - set(orders_df.columns))
            if missing_order_cols:
                criticals.append("orders_log missing columns: " + ", ".join(missing_order_cols))

            if not orders_df.empty and "profile" in orders_df.columns and selected_profiles:
                used_profiles = set(orders_df["profile"].dropna().astype(str))
                unexpected = sorted(used_profiles - selected_profiles)
                if unexpected:
                    criticals.append("allowed_strategies violation: " + ", ".join(unexpected))

            if not orders_df.empty and trades_exists and risk_exists:
                orders_count = len(orders_df)
                trades_count = len(trades_df)
                risk_count = len(risk_df)
                if orders_count > 0 and trades_count == 0 and risk_count == 0:
                    warnings.append("orders generated but no trades or risk events recorded")

            if not orders_df.empty and {"profile", "symbol", "side"}.issubset(orders_df.columns):
                dup_buy = (
                    orders_df.loc[orders_df["side"].astype(str).str.upper() == "BUY"]
                    .groupby(["profile", "symbol"])
                    .size()
                    .reset_index(name="count")
                )
                if not dup_buy.empty and int(dup_buy["count"].max()) >= 3:
                    warnings.append("duplicate buy concentration detected")

        if risk_exists:
            risk_df = _normalize_date_col(risk_df.rename(columns={"date": "trade_date"}), "trade_date")
            if current_date is not None:
                today_risk_count = int(len(risk_df.loc[risk_df["trade_date"] == current_date.normalize()]))
            else:
                today_risk_count = int(len(risk_df))
            if today_risk_count >= 10:
                criticals.append(f"risk events too high: {today_risk_count}")
            elif today_risk_count >= 5:
                warnings.append(f"risk events elevated: {today_risk_count}")
    else:
        missing_files.append("live_trading_root")

    if forward_root is not None:
        for profile_dir in [p for p in forward_root.iterdir() if p.is_dir()]:
            portfolio_df, portfolio_exists = _load_csv_if_exists(profile_dir / "portfolio.csv")
            trades_df, trades_exists = _load_csv_if_exists(profile_dir / "trades.csv")
            skipped_df, skipped_exists = _load_csv_if_exists(profile_dir / "skipped_days.csv")
            state_json, state_exists = _load_json_if_exists(profile_dir / "state.json")

            if not portfolio_exists:
                missing_files.append(f"{profile_dir.name}/portfolio.csv")
                continue

            portfolio_df = _normalize_date_col(portfolio_df, "trade_date")
            if portfolio_df["trade_date"].isna().any():
                warnings.append(f"{profile_dir.name} portfolio.csv has missing trade_date rows")
            if "total_value" not in portfolio_df.columns:
                criticals.append(f"{profile_dir.name} portfolio.csv missing total_value")
            else:
                portfolio_df["total_value"] = pd.to_numeric(portfolio_df["total_value"], errors="coerce")
                if portfolio_df["total_value"].isna().any() or (portfolio_df["total_value"] <= 0).any():
                    criticals.append(f"{profile_dir.name} total_value contains NaN or non-positive values")
                returns = pd.to_numeric(portfolio_df.get("daily_return"), errors="coerce")
                if not returns.dropna().empty:
                    latest_daily_return = float(returns.dropna().iloc[-1])
                    if latest_daily_return <= -0.03:
                        criticals.append(f"{profile_dir.name} daily_return critical: {latest_daily_return:.4f}")
                    elif latest_daily_return <= -0.02:
                        warnings.append(f"{profile_dir.name} daily_return warning: {latest_daily_return:.4f}")
                values = portfolio_df["total_value"].dropna()
                if not values.empty:
                    running_peak = values.cummax()
                    drawdown = values / running_peak - 1.0
                    current_mdd = abs(float(drawdown.min()))
                    if current_mdd >= 0.15:
                        criticals.append(f"{profile_dir.name} current_mdd critical: {current_mdd:.4f}")
                    elif current_mdd >= 0.10:
                        warnings.append(f"{profile_dir.name} current_mdd warning: {current_mdd:.4f}")

            if not trades_exists:
                missing_files.append(f"{profile_dir.name}/trades.csv")
            else:
                required_trade_cols = {"trade_date", "symbol", "side"}
                missing_trade_cols = sorted(required_trade_cols - set(trades_df.columns))
                if missing_trade_cols:
                    criticals.append(f"{profile_dir.name} trades.csv missing columns: {', '.join(missing_trade_cols)}")

            if skipped_exists and not skipped_df.empty:
                reason_col = "reason" if "reason" in skipped_df.columns else None
                if reason_col:
                    missing_price_count = int(skipped_df[reason_col].astype(str).str.contains("missing", case=False, na=False).sum())
                    if missing_price_count >= 5:
                        warnings.append(f"{profile_dir.name} missing-price style skips elevated: {missing_price_count}")

            if state_exists and isinstance(state_json, dict):
                positions = state_json.get("positions", {}) or {}
                if len(positions) > 10:
                    criticals.append(f"{profile_dir.name} max_total_positions exceeded: {len(positions)}")
    else:
        missing_files.append("forward_test_root")

    status = "OK"
    if criticals:
        status = "CRITICAL"
    elif warnings:
        status = "WARNING"

    recommended_action = []
    if status == "CRITICAL":
        recommended_action.append("Pause new order routing and inspect the latest profile outputs immediately.")
    elif status == "WARNING":
        recommended_action.append("Review the affected profile logs before the next trading session.")
    else:
        recommended_action.append("No urgent operational intervention required from the available files.")

    recommended_action.extend([
        "Compare current forward/live behavior with selected backtest baselines.",
        "Escalate repeated risk events before enabling any real broker routing.",
    ])

    return {
        "status": status,
        "checked_at": checked_at,
        "warnings": warnings,
        "criticals": criticals,
        "missing_files": missing_files,
        "recommended_action": recommended_action,
    }


def build_health_markdown(payload: dict[str, Any]) -> str:
    """Build markdown health report."""
    lines = [
        f"# Trading Health Check: {pd.Timestamp(payload['checked_at']).strftime('%Y-%m-%d')}",
        "",
        f"- status: `{payload['status']}`",
        f"- checked_at: `{pd.Timestamp(payload['checked_at']).isoformat()}`",
        "",
        "## Warnings",
    ]
    if payload["warnings"]:
        lines.extend(f"- {item}" for item in payload["warnings"])
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Criticals",
    ])
    if payload["criticals"]:
        lines.extend(f"- {item}" for item in payload["criticals"])
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Missing Files",
    ])
    if payload["missing_files"]:
        lines.extend(f"- {item}" for item in payload["missing_files"])
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Recommended Action",
    ])
    lines.extend(f"- {item}" for item in payload["recommended_action"])
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_health_checks(input_dir)
    check_date = pd.Timestamp(payload["checked_at"]).strftime("%Y-%m-%d")
    json_path = output_dir / f"health_check_{check_date}.json"
    md_path = output_dir / f"health_check_{check_date}.md"

    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_health_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
