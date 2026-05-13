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


def _score_threshold() -> tuple[float, str | None]:
    raw = _raw("US_BUY_MIN_SCORE") or _raw("US_BUY_MIN_TOTAL_SCORE", "70")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 70.0, "US_BUY_MIN_SCORE invalid; fallback=70"
    if 0.0 <= value <= 1.0:
        return value * 100.0, "US_BUY_MIN_SCORE interpreted as ratio and converted to 0-100 scale"
    return max(0.0, value), None


def _probability_threshold() -> tuple[float, str | None]:
    raw = _raw("US_BUY_MIN_PROB") or _raw("US_BUY_MIN_PROBABILITY", "0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, "US_BUY_MIN_PROB invalid; fallback=0"
    value = min(max(0.0, value), 1.0)
    return value, None


def _pct(name: str, default: float) -> tuple[float, str | None]:
    value = _safe_float(name, default, minimum=0.0)
    if value > 1.0:
        return value / 100.0, f"{name} interpreted as percentage and converted to decimal"
    return value, None


@dataclass(frozen=True)
class BuyAutomationConfig:
    root_dir: Path
    output_dir: Path
    mode: str
    enabled: bool
    top_n: int
    min_grade: str
    min_score: float
    min_prob: float
    max_daily_symbols: int
    max_daily_amount_usd: float
    max_per_symbol_amount_usd: float
    min_price: float
    max_price: float
    max_gap_up_pct: float
    max_intraday_change_pct: float
    max_volatility_pct: float
    require_financial_data: bool
    require_benchmark_strength: bool
    cooldown_days: int
    failsafe_on_data_error: bool
    block_on_kill_switch: bool
    ranking_source: str
    live_implemented: bool
    warnings: tuple[str, ...]


def load_buy_automation_config() -> BuyAutomationConfig:
    root_dir = Path(__file__).resolve().parents[3]
    output_dir_raw = _raw("US_BUY_REPORT_OUTPUT_DIR", "output/us_stock_buy_automation")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    warnings: list[str] = []
    min_score, score_warning = _score_threshold()
    min_prob, prob_warning = _probability_threshold()
    max_gap_up_pct, gap_warning = _pct("US_BUY_MAX_GAP_UP_PCT", 0.05)
    max_intraday_change_pct, intraday_warning = _pct("US_BUY_MAX_INTRADAY_CHANGE_PCT", 0.08)
    max_volatility_pct, volatility_warning = _pct("US_BUY_MAX_VOLATILITY_PCT", 0.10)
    for warning in (score_warning, prob_warning, gap_warning, intraday_warning, volatility_warning):
        if warning:
            warnings.append(warning)

    mode = (_raw("US_BUY_AUTOMATION_MODE", "SHADOW").upper() or "SHADOW").strip()
    if mode not in {"SHADOW", "PAPER", "LIVE"}:
        warnings.append(f"US_BUY_AUTOMATION_MODE invalid: {mode}; fallback=SHADOW")
        mode = "SHADOW"

    return BuyAutomationConfig(
        root_dir=root_dir,
        output_dir=output_dir,
        mode=mode,
        enabled=_flag("US_BUY_AUTOMATION_ENABLED", "0"),
        top_n=_safe_int("US_BUY_TOP_N", 5, minimum=1),
        min_grade=_raw("US_BUY_MIN_GRADE", "BUY").upper() or "BUY",
        min_score=min_score,
        min_prob=min_prob,
        max_daily_symbols=_safe_int("US_BUY_MAX_DAILY_SYMBOLS", 1, minimum=1),
        max_daily_amount_usd=_safe_float("US_BUY_MAX_DAILY_AMOUNT_USD", 100.0, minimum=0.0),
        max_per_symbol_amount_usd=_safe_float("US_BUY_MAX_PER_SYMBOL_AMOUNT_USD", 100.0, minimum=0.0),
        min_price=_safe_float("US_BUY_MIN_PRICE", 5.0, minimum=0.0),
        max_price=_safe_float("US_BUY_MAX_PRICE", 500.0, minimum=0.0),
        max_gap_up_pct=max_gap_up_pct,
        max_intraday_change_pct=max_intraday_change_pct,
        max_volatility_pct=max_volatility_pct,
        require_financial_data=_flag("US_BUY_REQUIRE_FINANCIAL_DATA", "1"),
        require_benchmark_strength=_flag("US_BUY_REQUIRE_BENCHMARK_STRENGTH", "1")
        or _flag("US_BUY_REQUIRE_RELATIVE_STRENGTH", "1"),
        cooldown_days=_safe_int("US_BUY_COOLDOWN_DAYS", 10, minimum=0),
        failsafe_on_data_error=_flag("US_BUY_FAILSAFE_ON_DATA_ERROR", "1"),
        block_on_kill_switch=_flag("US_BUY_BLOCK_ON_KILL_SWITCH", "1"),
        ranking_source=_raw("US_RANKING_DEFAULT_SOURCE", "rule_v1") or "rule_v1",
        live_implemented=False,
        warnings=tuple(warnings),
    )
