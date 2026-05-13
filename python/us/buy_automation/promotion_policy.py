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


def _safe_float(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(_raw(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _pct(name: str, default: float) -> float:
    value = _safe_float(name, default, minimum=0.0)
    return value / 100.0 if value > 1.0 else value


@dataclass(frozen=True)
class LivePromotionPolicy:
    readiness_enabled: bool
    output_dir: Path
    benchmark_symbol: str
    compare_qqq: bool
    min_excess_return_pct: float
    min_shadow_days: int
    min_paper_days: int
    min_paper_orders: int
    min_win_rate_pct: float
    max_drawdown_pct: float
    max_data_missing_rate_pct: float
    min_scheduler_success_rate_pct: float
    require_positive_excess_return: bool
    require_manual_approval: bool
    scheduler_run_readiness: bool


def load_live_promotion_policy() -> LivePromotionPolicy:
    root_dir = Path(__file__).resolve().parents[3]
    output_dir_raw = _raw("US_BUY_READINESS_REPORT_OUTPUT_DIR", "reports/lee_trader_us/buy_automation/readiness")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    return LivePromotionPolicy(
        readiness_enabled=_flag("US_BUY_READINESS_ENABLED", "1"),
        output_dir=output_dir,
        benchmark_symbol=_raw("US_BUY_READINESS_BENCHMARK_SYMBOL", "SPY").upper() or "SPY",
        compare_qqq=_flag("US_BUY_READINESS_COMPARE_QQQ", "1"),
        min_excess_return_pct=_pct("US_BUY_READINESS_MIN_EXCESS_RETURN_PCT", 0.0),
        min_shadow_days=_safe_int("US_BUY_MIN_SHADOW_DAYS", 20, minimum=0),
        min_paper_days=_safe_int("US_BUY_MIN_PAPER_DAYS", 60, minimum=0),
        min_paper_orders=_safe_int("US_BUY_MIN_PAPER_ORDERS", 20, minimum=0),
        min_win_rate_pct=_pct("US_BUY_MIN_WIN_RATE_PCT", 50.0),
        max_drawdown_pct=_pct("US_BUY_MAX_DRAWDOWN_PCT", 15.0),
        max_data_missing_rate_pct=_pct("US_BUY_MAX_DATA_MISSING_RATE_PCT", 5.0),
        min_scheduler_success_rate_pct=_pct("US_BUY_MIN_SCHEDULER_SUCCESS_RATE_PCT", 95.0),
        require_positive_excess_return=_flag("US_BUY_REQUIRE_POSITIVE_EXCESS_RETURN", "1"),
        require_manual_approval=_flag("US_BUY_REQUIRE_MANUAL_APPROVAL", "1"),
        scheduler_run_readiness=_flag("US_BUY_SCHEDULER_RUN_READINESS", "0"),
    )
