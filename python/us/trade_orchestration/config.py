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


@dataclass(frozen=True)
class TradeOrchestrationConfig:
    root_dir: Path
    mode: str
    enabled: bool
    block_buy_if_position_exists: bool
    block_buy_if_sell_signal_exists: bool
    block_buy_after_full_exit_days: int
    block_buy_on_review_required: bool
    sell_priority_over_buy: bool
    conflict_failsafe: bool
    report_enabled: bool
    report_output_dir: Path
    report_formats: tuple[str, ...]
    fail_pipeline_on_error: bool
    scheduler_enabled: bool
    scheduler_run_orchestration: bool
    scheduler_run_dashboard: bool
    scheduler_run_notification_adapter: bool
    scheduler_run_health_check: bool
    scheduler_run_report: bool
    scheduler_fail_pipeline_on_error: bool
    scheduler_allow_live: bool
    scheduler_prevent_duplicate_run: bool
    scheduler_lock_ttl_seconds: int
    scheduler_max_runtime_seconds: int
    disable_buy_only_scheduler_when_orchestration: bool
    warn_if_buy_only_scheduler_enabled: bool
    health_check_enabled: bool
    health_check_fail_on_missing_report: bool
    health_check_fail_on_invalid_log: bool
    health_check_max_data_missing_rate_pct: float
    lock_dir: Path
    checklist_output_dir: Path
    warnings: tuple[str, ...]


def load_trade_orchestration_config() -> TradeOrchestrationConfig:
    root_dir = Path(__file__).resolve().parents[3]
    warnings: list[str] = []

    mode = (_raw("US_TRADE_ORCHESTRATION_MODE", "SHADOW").upper() or "SHADOW").strip()
    if mode not in {"SHADOW", "PAPER", "LIVE"}:
        warnings.append(f"US_TRADE_ORCHESTRATION_MODE invalid: {mode}; fallback=SHADOW")
        mode = "SHADOW"

    report_dir_raw = _raw("US_TRADE_REPORT_OUTPUT_DIR", "reports/lee_trader_us/trade_orchestration")
    report_output_dir = Path(report_dir_raw)
    if not report_output_dir.is_absolute():
        report_output_dir = root_dir / report_output_dir

    raw_formats = _raw("US_TRADE_REPORT_FORMAT", "json,markdown") or "json,markdown"
    report_formats = tuple(sorted({item.strip().lower() for item in raw_formats.split(",") if item.strip()}))
    if not report_formats:
        report_formats = ("json", "markdown")
        warnings.append("US_TRADE_REPORT_FORMAT invalid; fallback=json,markdown")

    lock_dir_raw = _raw("US_TRADE_LOCK_DIR", "tmp/lee_trader_us/trade_orchestration")
    lock_dir = Path(lock_dir_raw)
    if not lock_dir.is_absolute():
        lock_dir = root_dir / lock_dir

    checklist_output_dir_raw = _raw("US_TRADE_CHECKLIST_OUTPUT_DIR", str(report_output_dir))
    checklist_output_dir = Path(checklist_output_dir_raw)
    if not checklist_output_dir.is_absolute():
        checklist_output_dir = root_dir / checklist_output_dir

    return TradeOrchestrationConfig(
        root_dir=root_dir,
        mode=mode,
        enabled=_flag("US_TRADE_ORCHESTRATION_ENABLED", "0"),
        block_buy_if_position_exists=_flag("US_TRADE_BLOCK_BUY_IF_POSITION_EXISTS", "1"),
        block_buy_if_sell_signal_exists=_flag("US_TRADE_BLOCK_BUY_IF_SELL_SIGNAL_EXISTS", "1"),
        block_buy_after_full_exit_days=_safe_int("US_TRADE_BLOCK_BUY_AFTER_FULL_EXIT_DAYS", 10, minimum=0),
        block_buy_on_review_required=_flag("US_TRADE_BLOCK_BUY_ON_REVIEW_REQUIRED", "1"),
        sell_priority_over_buy=_flag("US_TRADE_SELL_PRIORITY_OVER_BUY", "1"),
        conflict_failsafe=_flag("US_TRADE_CONFLICT_FAILSAFE", "1"),
        report_enabled=_flag("US_TRADE_REPORT_ENABLED", "1"),
        report_output_dir=report_output_dir,
        report_formats=report_formats,
        fail_pipeline_on_error=_flag("US_TRADE_FAIL_PIPELINE_ON_ERROR", "0"),
        scheduler_enabled=_flag("US_TRADE_SCHEDULER_ENABLED", "0"),
        scheduler_run_orchestration=_flag("US_TRADE_SCHEDULER_RUN_ORCHESTRATION", "1"),
        scheduler_run_dashboard=_flag("US_TRADE_SCHEDULER_RUN_DASHBOARD", "0"),
        scheduler_run_notification_adapter=_flag("US_TRADE_SCHEDULER_RUN_NOTIFICATION_ADAPTER", "0"),
        scheduler_run_health_check=_flag("US_TRADE_SCHEDULER_RUN_HEALTH_CHECK", "1"),
        scheduler_run_report=_flag("US_TRADE_SCHEDULER_RUN_REPORT", "1"),
        scheduler_fail_pipeline_on_error=_flag("US_TRADE_SCHEDULER_FAIL_PIPELINE_ON_ERROR", "0"),
        scheduler_allow_live=_flag("US_TRADE_SCHEDULER_ALLOW_LIVE", "0"),
        scheduler_prevent_duplicate_run=_flag("US_TRADE_SCHEDULER_PREVENT_DUPLICATE_RUN", "1"),
        scheduler_lock_ttl_seconds=_safe_int("US_TRADE_SCHEDULER_LOCK_TTL_SECONDS", 1800, minimum=1),
        scheduler_max_runtime_seconds=_safe_int("US_TRADE_SCHEDULER_MAX_RUNTIME_SECONDS", 600, minimum=1),
        disable_buy_only_scheduler_when_orchestration=_flag("US_TRADE_DISABLE_BUY_ONLY_SCHEDULER_WHEN_ORCHESTRATION", "1"),
        warn_if_buy_only_scheduler_enabled=_flag("US_TRADE_WARN_IF_BUY_ONLY_SCHEDULER_ENABLED", "1"),
        health_check_enabled=_flag("US_TRADE_HEALTH_CHECK_ENABLED", "1"),
        health_check_fail_on_missing_report=_flag("US_TRADE_HEALTH_CHECK_FAIL_ON_MISSING_REPORT", "0"),
        health_check_fail_on_invalid_log=_flag("US_TRADE_HEALTH_CHECK_FAIL_ON_INVALID_LOG", "0"),
        health_check_max_data_missing_rate_pct=float(_safe_int("US_TRADE_HEALTH_CHECK_MAX_DATA_MISSING_RATE_PCT", 20, minimum=0)),
        lock_dir=lock_dir,
        checklist_output_dir=checklist_output_dir,
        warnings=tuple(warnings),
    )
