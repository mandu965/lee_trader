from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

from python.us.buy_automation.paper_performance import build_paper_performance
from python.us.buy_automation.performance_metrics import (
    build_compounded_equity_curve,
    calculate_excess_return,
    calculate_max_drawdown,
    summarize_returns,
)
from python.us.us_config import parse_iso_date


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def log_input_dir() -> Path:
    raw = str(os.environ.get("US_BUY_LOG_INPUT_DIR", "output/us_stock_buy_automation")).strip() or "output/us_stock_buy_automation"
    path = Path(raw)
    return path if path.is_absolute() else _root_dir() / path


def load_buy_automation_run_logs(*, input_dir: str | None = None) -> list[dict[str, object]]:
    directory = Path(input_dir) if input_dir else log_input_dir()
    if not directory.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("buy_automation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_source_json_path"] = str(path)
        rows.append(payload)
    return rows


def load_scheduler_job_logs(*, input_dir: str | None = None) -> list[dict[str, object]]:
    directory = Path(input_dir) if input_dir else log_input_dir()
    if not directory.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("scheduler_job_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_source_json_path"] = str(path)
        rows.append(payload)
    return rows


def _dedupe_paper_orders(logs: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[str, dict[str, object]] = {}
    for payload in logs:
        for row in payload.get("paper_orders") or []:
            if not isinstance(row, dict):
                continue
            key = str(row.get("paper_order_id") or f"{payload.get('trade_date')}|{payload.get('mode')}|{row.get('symbol')}|{row.get('side')}")
            merged = dict(row)
            merged.setdefault("trade_date", payload.get("trade_date"))
            merged.setdefault("automation_mode", payload.get("mode"))
            by_key[key] = merged
    return list(by_key.values())


def _select_trade_dates(orders: list[dict[str, object]], days: int | None) -> set[date]:
    dates = sorted(
        {
            parsed
            for parsed in (
                parse_iso_date(str(item.get("trade_date")), field_name="trade_date") for item in orders if item.get("trade_date")
            )
            if parsed is not None
        }
    )
    if days is None or len(dates) <= days:
        return set(dates)
    return set(dates[-days:])


def _weighted_average(values: list[float | None], weights: list[float | None]) -> float | None:
    pairs = [(value, weight) for value, weight in zip(values, weights, strict=False) if value is not None and weight is not None and weight > 0]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def build_paper_backtest_summary(
    *,
    days: int | None = None,
    benchmark_symbol: str = "SPY",
    compare_qqq: bool = True,
    input_dir: str | None = None,
    as_of_date: date | None = None,
) -> dict[str, object]:
    raw_logs = [row for row in load_buy_automation_run_logs(input_dir=input_dir) if str(row.get("mode") or "").upper() == "PAPER"]
    all_orders = _dedupe_paper_orders(raw_logs)
    selected_dates = _select_trade_dates(all_orders, days)
    filtered_orders = []
    for order in all_orders:
        trade_date_value = parse_iso_date(str(order.get("trade_date")), field_name="trade_date") if order.get("trade_date") else None
        if trade_date_value is None:
            continue
        if not selected_dates or trade_date_value in selected_dates:
            filtered_orders.append(order)

    label = "ALL" if days is None else str(days)
    if not filtered_orders:
        return {
            "period_label": label,
            "days": days,
            "status": "NO_PAPER_ORDERS",
            "paper_order_count": 0,
            "unique_symbol_count": 0,
            "excluded_order_count": 0,
            "data_missing_rate": None,
            "benchmark_symbol": benchmark_symbol.upper(),
            "benchmark_data_missing": True,
            "compare_qqq": compare_qqq,
            "qqq_comparison": None,
        }

    primary = build_paper_performance(filtered_orders, benchmark_symbol=benchmark_symbol, as_of_date=as_of_date)
    valid_rows = [row for row in primary.get("rows", []) if row.get("status") == "OK"]
    excluded_rows = [row for row in primary.get("rows", []) if row.get("status") != "OK"]

    invested_amounts = [float(row.get("invested_amount") or 0.0) for row in valid_rows]
    current_values = [float(row.get("current_value") or 0.0) for row in valid_rows]
    returns = [row.get("unrealized_pnl_pct") for row in valid_rows]
    benchmark_returns = [row.get("benchmark_return_pct") for row in valid_rows]
    holding_days = [float(row.get("holding_days")) for row in valid_rows if row.get("holding_days") is not None]

    invested_total = sum(invested_amounts)
    current_value_total = sum(current_values)
    total_return_pct = ((current_value_total / invested_total) - 1.0) if invested_total > 0 else None
    benchmark_return_pct = _weighted_average(benchmark_returns, invested_amounts)
    excess_return_pct = calculate_excess_return(total_return_pct, benchmark_return_pct)
    return_summary = summarize_returns(returns)
    compounded_curve = build_compounded_equity_curve(returns)
    max_drawdown_pct = calculate_max_drawdown(compounded_curve)
    benchmark_data_missing = bool(valid_rows) and benchmark_return_pct is None
    data_missing_rate = (len(excluded_rows) / len(primary.get("rows", []))) if primary.get("rows") else None

    qqq_comparison = None
    if compare_qqq:
        qqq = build_paper_performance(filtered_orders, benchmark_symbol="QQQ", as_of_date=as_of_date)
        qqq_valid = [row for row in qqq.get("rows", []) if row.get("status") == "OK"]
        qqq_benchmark_return = _weighted_average(
            [row.get("benchmark_return_pct") for row in qqq_valid],
            [float(row.get("invested_amount") or 0.0) for row in qqq_valid],
        )
        qqq_comparison = {
            "benchmark_symbol": "QQQ",
            "benchmark_return_pct": qqq_benchmark_return,
            "excess_return_pct": calculate_excess_return(total_return_pct, qqq_benchmark_return),
            "benchmark_data_missing": bool(qqq_valid) and qqq_benchmark_return is None,
        }

    return {
        "period_label": label,
        "days": days,
        "status": "BENCHMARK_DATA_MISSING" if benchmark_data_missing else "OK",
        "paper_order_count": len(filtered_orders),
        "unique_symbol_count": len({str(row.get("symbol") or "").upper() for row in filtered_orders if row.get("symbol")}),
        "invested_amount_total": round(invested_total, 6),
        "current_value_total": round(current_value_total, 6),
        "realized_basis": "UNREALIZED_ONLY",
        "total_return_pct": total_return_pct,
        "avg_return_pct": return_summary.get("avg_return_pct"),
        "median_return_pct": return_summary.get("median_return_pct"),
        "win_rate": return_summary.get("win_rate"),
        "loss_rate": return_summary.get("loss_rate"),
        "best_trade_return_pct": return_summary.get("best_trade_return_pct"),
        "worst_trade_return_pct": return_summary.get("worst_trade_return_pct"),
        "max_drawdown_pct": max_drawdown_pct,
        "avg_holding_days": (sum(holding_days) / len(holding_days)) if holding_days else None,
        "benchmark_symbol": str(primary.get("benchmark_symbol") or benchmark_symbol).upper(),
        "benchmark_return_pct": benchmark_return_pct,
        "excess_return_pct": excess_return_pct,
        "positive_excess_return": excess_return_pct is not None and excess_return_pct > 0,
        "benchmark_data_missing": benchmark_data_missing,
        "excluded_order_count": len(excluded_rows),
        "excluded_order_ids": [row.get("paper_order_id") for row in excluded_rows],
        "data_missing_rate": data_missing_rate,
        "compare_qqq": compare_qqq,
        "qqq_comparison": qqq_comparison,
        "rows": primary.get("rows", []),
        "as_of_date": primary.get("as_of_date"),
    }
