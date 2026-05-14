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
class DashboardConfig:
    root_dir: Path
    enabled: bool
    output_dir: Path
    formats: tuple[str, ...]
    include_buy_monitor: bool
    include_sell_monitor: bool
    include_conflict_monitor: bool
    include_performance: bool
    include_health: bool
    include_readiness: bool
    default_lookback_days: int
    data_missing_warning_pct: float
    data_missing_critical_pct: float
    fail_pipeline_on_error: bool
    require_json_report: bool
    require_markdown_report: bool
    notification_enabled: bool
    notification_formats: tuple[str, ...]
    notification_include_warnings: bool
    notification_include_top_symbols: bool
    notification_include_readiness: bool
    notification_max_symbols: int
    trade_report_dir: Path
    buy_output_dir: Path
    sell_output_dir: Path
    readiness_output_dir: Path
    warnings: tuple[str, ...]


def _resolve_dir(root_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    return path if path.is_absolute() else root_dir / path


def load_dashboard_config() -> DashboardConfig:
    root_dir = Path(__file__).resolve().parents[3]
    warnings: list[str] = []

    output_dir = _resolve_dir(root_dir, _raw("US_DASHBOARD_OUTPUT_DIR", "reports/lee_trader_us/dashboard") or "reports/lee_trader_us/dashboard")
    raw_formats = _raw("US_DASHBOARD_FORMAT", "json,markdown") or "json,markdown"
    formats = tuple(sorted({item.strip().lower() for item in raw_formats.split(",") if item.strip()}))
    if not formats:
        formats = ("json", "markdown")
        warnings.append("US_DASHBOARD_FORMAT invalid; fallback=json,markdown")

    trade_report_dir = _resolve_dir(root_dir, _raw("US_TRADE_REPORT_OUTPUT_DIR", "reports/lee_trader_us/trade_orchestration") or "reports/lee_trader_us/trade_orchestration")
    buy_output_dir = _resolve_dir(root_dir, _raw("US_BUY_LOG_INPUT_DIR", "output/us_stock_buy_automation") or "output/us_stock_buy_automation")
    sell_output_dir = _resolve_dir(root_dir, _raw("US_SELL_REPORT_OUTPUT_DIR", "output/us_stock_sell_automation") or "output/us_stock_sell_automation")
    readiness_output_dir = _resolve_dir(
        root_dir,
        _raw("US_BUY_READINESS_REPORT_OUTPUT_DIR", "reports/lee_trader_us/buy_automation/readiness") or "reports/lee_trader_us/buy_automation/readiness",
    )
    raw_notification_formats = _raw("US_DASHBOARD_NOTIFICATION_FORMAT", "text,json") or "text,json"
    notification_formats = tuple(sorted({item.strip().lower() for item in raw_notification_formats.split(",") if item.strip()}))
    if not notification_formats:
        notification_formats = ("json", "text")
        warnings.append("US_DASHBOARD_NOTIFICATION_FORMAT invalid; fallback=text,json")

    output_dir.mkdir(parents=True, exist_ok=True)

    return DashboardConfig(
        root_dir=root_dir,
        enabled=_flag("US_DASHBOARD_ENABLED", "0"),
        output_dir=output_dir,
        formats=formats,
        include_buy_monitor=_flag("US_DASHBOARD_INCLUDE_BUY_MONITOR", "1"),
        include_sell_monitor=_flag("US_DASHBOARD_INCLUDE_SELL_MONITOR", "1"),
        include_conflict_monitor=_flag("US_DASHBOARD_INCLUDE_CONFLICT_MONITOR", "1"),
        include_performance=_flag("US_DASHBOARD_INCLUDE_PERFORMANCE", "1"),
        include_health=_flag("US_DASHBOARD_INCLUDE_HEALTH", "1"),
        include_readiness=_flag("US_DASHBOARD_INCLUDE_READINESS", "1"),
        default_lookback_days=_safe_int("US_DASHBOARD_DEFAULT_LOOKBACK_DAYS", 60, minimum=1),
        data_missing_warning_pct=float(_safe_int("US_DASHBOARD_DATA_MISSING_WARNING_PCT", 5, minimum=0)),
        data_missing_critical_pct=float(_safe_int("US_DASHBOARD_DATA_MISSING_CRITICAL_PCT", 20, minimum=0)),
        fail_pipeline_on_error=_flag("US_DASHBOARD_FAIL_PIPELINE_ON_ERROR", "0"),
        require_json_report=_flag("US_DASHBOARD_REQUIRE_JSON_REPORT", "1"),
        require_markdown_report=_flag("US_DASHBOARD_REQUIRE_MARKDOWN_REPORT", "0"),
        notification_enabled=_flag("US_DASHBOARD_NOTIFICATION_ENABLED", "0"),
        notification_formats=notification_formats,
        notification_include_warnings=_flag("US_DASHBOARD_NOTIFICATION_INCLUDE_WARNINGS", "1"),
        notification_include_top_symbols=_flag("US_DASHBOARD_NOTIFICATION_INCLUDE_TOP_SYMBOLS", "1"),
        notification_include_readiness=_flag("US_DASHBOARD_NOTIFICATION_INCLUDE_READINESS", "1"),
        notification_max_symbols=_safe_int("US_DASHBOARD_NOTIFICATION_MAX_SYMBOLS", 10, minimum=1),
        trade_report_dir=trade_report_dir,
        buy_output_dir=buy_output_dir,
        sell_output_dir=sell_output_dir,
        readiness_output_dir=readiness_output_dir,
        warnings=tuple(warnings),
    )
