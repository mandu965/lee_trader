from __future__ import annotations

import math
from statistics import median


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def calculate_return_pct(entry_price: object, latest_price: object) -> float | None:
    entry = _safe_float(entry_price)
    latest = _safe_float(latest_price)
    if entry is None or latest is None or entry <= 0:
        return None
    return round((latest / entry) - 1.0, 12)


def calculate_win_rate(returns: list[object]) -> float | None:
    values = [_safe_float(value) for value in returns]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    wins = sum(1 for value in clean if value > 0)
    return wins / len(clean)


def calculate_loss_rate(returns: list[object]) -> float | None:
    values = [_safe_float(value) for value in returns]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    losses = sum(1 for value in clean if value < 0)
    return losses / len(clean)


def calculate_max_drawdown(equity_curve: list[object]) -> float | None:
    values = [_safe_float(value) for value in equity_curve]
    clean = [value for value in values if value is not None and value > 0]
    if not clean:
        return None
    peak = clean[0]
    max_drawdown = 0.0
    for value in clean:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def calculate_excess_return(strategy_return: object, benchmark_return: object) -> float | None:
    strategy = _safe_float(strategy_return)
    benchmark = _safe_float(benchmark_return)
    if strategy is None or benchmark is None:
        return None
    return round(strategy - benchmark, 12)


def summarize_returns(returns: list[object]) -> dict[str, float | int | None]:
    values = [_safe_float(value) for value in returns]
    clean = [value for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "avg_return_pct": None,
            "median_return_pct": None,
            "best_trade_return_pct": None,
            "worst_trade_return_pct": None,
            "win_rate": None,
            "loss_rate": None,
        }
    return {
        "count": len(clean),
        "avg_return_pct": sum(clean) / len(clean),
        "median_return_pct": median(clean),
        "best_trade_return_pct": max(clean),
        "worst_trade_return_pct": min(clean),
        "win_rate": calculate_win_rate(clean),
        "loss_rate": calculate_loss_rate(clean),
    }


def build_compounded_equity_curve(returns: list[object], *, initial_equity: float = 1.0) -> list[float]:
    equity = initial_equity
    curve: list[float] = []
    for value in returns:
        number = _safe_float(value)
        if number is None:
            continue
        equity *= 1.0 + number
        curve.append(equity)
    return curve
