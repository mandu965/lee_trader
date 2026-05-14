from __future__ import annotations

from datetime import date
from pathlib import Path

from sqlalchemy import text

from python.us.trade_orchestration.config import TradeOrchestrationConfig
from python.us.trade_orchestration.run_lock import inspect_run_lock
from python.us.us_db import get_us_engine, relation_exists


LATEST_RANKING_SQL = text(
    """
    SELECT MAX(trade_date) AS trade_date
    FROM recommend.us_stock_rank_daily
    WHERE (:requested_trade_date IS NULL OR trade_date <= :requested_trade_date)
    """
)


def _latest_ranking_trade_date(requested_trade_date: date | None) -> date | None:
    if not relation_exists("recommend.us_stock_rank_daily"):
        return None
    engine = get_us_engine()
    with engine.connect() as conn:
        value = conn.execute(LATEST_RANKING_SQL, {"requested_trade_date": requested_trade_date}).scalar()
    return value if isinstance(value, date) else None


def evaluate_scheduler_guard(
    cfg: TradeOrchestrationConfig,
    *,
    requested_trade_date: date | None = None,
) -> dict[str, object]:
    warnings = list(cfg.warnings)
    errors: list[str] = []

    if not cfg.scheduler_enabled:
        errors.append("SCHEDULER_DISABLED")
    if not cfg.enabled:
        errors.append("ORCHESTRATION_DISABLED")
    if cfg.mode == "LIVE" and not cfg.scheduler_allow_live:
        errors.append("LIVE_DISABLED_IN_SCHEDULER")

    if cfg.warn_if_buy_only_scheduler_enabled and str(__import__("os").environ.get("US_BUY_SCHEDULER_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        warnings.append("BUY_ONLY_SCHEDULER_ENABLED")
    if cfg.disable_buy_only_scheduler_when_orchestration and str(__import__("os").environ.get("US_BUY_SCHEDULER_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        warnings.append("BUY_ONLY_SCHEDULER_WILL_BE_DISABLED")

    ranking_trade_date = _latest_ranking_trade_date(requested_trade_date)
    if ranking_trade_date is None:
        errors.append("RANKING_DATA_MISSING")

    try:
        from python.us.buy_automation.decision_engine import run_buy_automation as _  # noqa: F401
        from python.us.sell_automation.sell_decision_engine import run_sell_automation as _  # noqa: F401
    except Exception as exc:
        errors.append(f"MODULE_IMPORT_FAILED:{exc}")

    try:
        cfg.report_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"REPORT_OUTPUT_DIR_UNAVAILABLE:{exc}")

    lock_state = inspect_run_lock(cfg, trade_date=(requested_trade_date or ranking_trade_date or date.today()).isoformat())
    if cfg.scheduler_prevent_duplicate_run and lock_state.get("lock_exists") and not lock_state.get("stale"):
        errors.append("DUPLICATE_RUN_DETECTED")

    return {
        "can_run": len(errors) == 0,
        "mode": cfg.mode,
        "warnings": warnings,
        "errors": errors,
        "pipeline_should_fail": cfg.scheduler_fail_pipeline_on_error,
        "ranking_trade_date": ranking_trade_date.isoformat() if ranking_trade_date else None,
        "lock_state": lock_state,
    }
