from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _raw(name: str, default: str | None = None) -> str:
    return str(os.environ.get(name, default or "")).strip()


def _safe_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(_raw(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _safe_float(name: str, default: float) -> float:
    try:
        return float(_raw(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _pct(name: str, default: float, *, allow_negative: bool = False) -> tuple[float, str | None]:
    value = _safe_float(name, default)
    warning = None
    if not allow_negative and value < 0:
        value = default
        warning = f"{name} invalid negative value; fallback={default}"
    if abs(value) > 1.0:
        value = value / 100.0
        warning = f"{name} interpreted as percentage and converted to decimal"
    return value, warning


def _ratio(name: str, default: float) -> tuple[float, str | None]:
    value = _safe_float(name, default)
    if value > 1.0:
        return max(0.0, min(1.0, value / 100.0)), f"{name} interpreted as percentage and converted to decimal"
    return max(0.0, min(1.0, value)), None


@dataclass(frozen=True)
class SellAutomationConfig:
    root_dir: Path
    output_dir: Path
    mode: str
    enabled: bool
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_holding_days: int
    rank_exit_threshold: int
    min_score_hold: float
    min_prob_hold: float
    require_benchmark_strength: bool
    benchmark_symbol: str
    benchmark_underperform_pct: float
    risk_off_exit_enabled: bool
    market_drawdown_exit_pct: float
    partial_take_profit_enabled: bool
    partial_take_profit_ratio: float
    failsafe_on_data_error: bool
    live_implemented: bool
    warnings: tuple[str, ...]


def load_sell_automation_config() -> SellAutomationConfig:
    root_dir = Path(__file__).resolve().parents[3]
    output_dir_raw = _raw("US_SELL_REPORT_OUTPUT_DIR", "output/us_stock_sell_automation")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    warnings: list[str] = []
    stop_loss_pct, stop_loss_warning = _pct("US_SELL_STOP_LOSS_PCT", -0.08, allow_negative=True)
    take_profit_pct, take_profit_warning = _pct("US_SELL_TAKE_PROFIT_PCT", 0.15)
    trailing_stop_pct, trailing_stop_warning = _pct("US_SELL_TRAILING_STOP_PCT", 0.10)
    min_score_hold, score_warning = _ratio("US_SELL_MIN_SCORE_HOLD", 0.50)
    min_prob_hold, prob_warning = _ratio("US_SELL_MIN_PROB_HOLD", 0.50)
    benchmark_underperform_pct, benchmark_warning = _pct(
        "US_SELL_BENCHMARK_UNDERPERFORM_PCT",
        -0.05,
        allow_negative=True,
    )
    market_drawdown_exit_pct, market_warning = _pct(
        "US_SELL_MARKET_DRAWDOWN_EXIT_PCT",
        -0.05,
        allow_negative=True,
    )
    partial_take_profit_ratio, partial_warning = _ratio("US_SELL_PARTIAL_TAKE_PROFIT_RATIO", 0.5)
    for warning in (
        stop_loss_warning,
        take_profit_warning,
        trailing_stop_warning,
        score_warning,
        prob_warning,
        benchmark_warning,
        market_warning,
        partial_warning,
    ):
        if warning:
            warnings.append(warning)

    mode = (_raw("US_SELL_AUTOMATION_MODE", "SHADOW").upper() or "SHADOW").strip()
    if mode not in {"SHADOW", "PAPER", "LIVE"}:
        warnings.append(f"US_SELL_AUTOMATION_MODE invalid: {mode}; fallback=SHADOW")
        mode = "SHADOW"

    if stop_loss_pct > 0:
        stop_loss_pct = -abs(stop_loss_pct)
    if benchmark_underperform_pct > 0:
        benchmark_underperform_pct = -abs(benchmark_underperform_pct)
    if market_drawdown_exit_pct > 0:
        market_drawdown_exit_pct = -abs(market_drawdown_exit_pct)

    return SellAutomationConfig(
        root_dir=root_dir,
        output_dir=output_dir,
        mode=mode,
        enabled=_flag("US_SELL_AUTOMATION_ENABLED", "0"),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=max(0.0, take_profit_pct),
        trailing_stop_pct=max(0.0, trailing_stop_pct),
        max_holding_days=_safe_int("US_SELL_MAX_HOLDING_DAYS", 60, minimum=1),
        rank_exit_threshold=_safe_int("US_SELL_RANK_EXIT_THRESHOLD", 30, minimum=1),
        min_score_hold=min_score_hold,
        min_prob_hold=min_prob_hold,
        require_benchmark_strength=_flag("US_SELL_REQUIRE_BENCHMARK_STRENGTH", "1"),
        benchmark_symbol=_raw("US_SELL_BENCHMARK_SYMBOL", "SPY").upper() or "SPY",
        benchmark_underperform_pct=benchmark_underperform_pct,
        risk_off_exit_enabled=_flag("US_SELL_RISK_OFF_EXIT_ENABLED", "1"),
        market_drawdown_exit_pct=market_drawdown_exit_pct,
        partial_take_profit_enabled=_flag("US_SELL_PARTIAL_TAKE_PROFIT_ENABLED", "0"),
        partial_take_profit_ratio=partial_take_profit_ratio,
        failsafe_on_data_error=_flag("US_SELL_FAILSAFE_ON_DATA_ERROR", "1"),
        live_implemented=False,
        warnings=tuple(warnings),
    )
