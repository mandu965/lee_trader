from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.buy_automation.config import load_buy_automation_config
from python.us.buy_automation.decision_engine import run_buy_automation
from python.us.buy_automation.report_generator import (
    finalize_buy_report,
    write_buy_report_json,
    write_buy_report_markdown,
)


LOGGER = logging.getLogger("us_buy_scheduler_job")


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
class BuySchedulerConfig:
    enabled: bool
    run_automation: bool
    run_report: bool
    fail_pipeline_on_error: bool
    allow_live: bool
    trade_date: str | None
    max_runtime_seconds: int
    log_level: str


def load_buy_scheduler_config() -> BuySchedulerConfig:
    return BuySchedulerConfig(
        enabled=_flag("US_BUY_SCHEDULER_ENABLED", "0"),
        run_automation=_flag("US_BUY_SCHEDULER_RUN_AUTOMATION", "1"),
        run_report=_flag("US_BUY_SCHEDULER_RUN_REPORT", "1"),
        fail_pipeline_on_error=_flag("US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR", "0"),
        allow_live=_flag("US_BUY_SCHEDULER_ALLOW_LIVE", "0"),
        trade_date=_raw("US_BUY_SCHEDULER_TRADE_DATE") or None,
        max_runtime_seconds=_safe_int("US_BUY_SCHEDULER_MAX_RUNTIME_SECONDS", 300, minimum=1),
        log_level=_raw("US_BUY_SCHEDULER_LOG_LEVEL", "INFO").upper() or "INFO",
    )


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def _scheduler_artifact_dir() -> Path:
    root_dir = Path(__file__).resolve().parents[3]
    raw = _raw("US_BUY_LOG_INPUT_DIR", "output/us_stock_buy_automation") or "output/us_stock_buy_automation"
    path = Path(raw)
    return path if path.is_absolute() else root_dir / path


def _persist_scheduler_result(result: dict[str, object], *, trade_date: str | None) -> None:
    directory = _scheduler_artifact_dir()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_trade_date = str(trade_date or "unknown")
    safe_mode = str(result.get("mode") or "UNKNOWN").upper()
    path = directory / f"scheduler_job_{safe_trade_date}_{safe_mode}_{timestamp}.json"
    payload = dict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _top_block_reason(report: dict[str, object] | None) -> str | None:
    if not isinstance(report, dict):
        return None
    summary = report.get("block_summary") or {}
    if not isinstance(summary, dict) or not summary:
        return None
    return max(summary.items(), key=lambda item: int(item[1] or 0))[0]


def _render_job_console(result: dict[str, object]) -> str:
    lines = [
        "[US BUY Scheduler Job]",
        f"enabled={str(result.get('enabled')).lower()}",
        f"mode={result.get('mode')}",
        f"automation_executed={str(result.get('automation_executed')).lower()}",
        f"report_executed={str(result.get('report_executed')).lower()}",
        f"success={str(result.get('success')).lower()}",
    ]
    if result.get("error"):
        lines.append(f"error={result.get('error')}")
        lines.append(f"pipeline_should_fail={str(result.get('pipeline_should_fail')).lower()}")
    summary = result.get("summary") or {}
    if summary:
        lines.extend(
            [
                "",
                f"candidates={summary.get('loaded_candidates', 0)}",
                f"allowed={summary.get('allowed_candidates', 0)}",
                f"blocked={summary.get('blocked_candidates', 0)}",
                f"paper_orders={summary.get('paper_orders', 0)}",
                f"top_block_reason={summary.get('top_block_reason') or '-'}",
                f"fail_safe={str(summary.get('fail_safe', False)).lower()}",
            ]
        )
    return "\n".join(lines)


def run_buy_scheduler_job(*, trade_date: str | None = None, emit_console: bool = True) -> dict[str, object]:
    scheduler_cfg = load_buy_scheduler_config()
    buy_cfg = load_buy_automation_config()
    effective_trade_date = trade_date or scheduler_cfg.trade_date

    result: dict[str, object] = {
        "job": "us_buy_automation",
        "enabled": scheduler_cfg.enabled,
        "mode": buy_cfg.mode,
        "automation_executed": False,
        "report_executed": False,
        "success": True,
        "error": None,
        "pipeline_should_fail": False,
        "summary": {},
    }

    if not scheduler_cfg.enabled:
        result["error"] = "SCHEDULER_DISABLED"
        _persist_scheduler_result(result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_job_console(result))
        return result

    if _flag("US_TRADE_SCHEDULER_ENABLED", "0") and _flag("US_TRADE_SCHEDULER_RUN_ORCHESTRATION", "1"):
        result["success"] = False
        result["error"] = "SCHEDULER_CONFIGURATION_CONFLICT"
        result["pipeline_should_fail"] = False
        _persist_scheduler_result(result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_job_console(result))
        return result

    if buy_cfg.mode == "LIVE":
        result["success"] = False
        result["error"] = "LIVE_DISABLED_IN_SCHEDULER"
        result["pipeline_should_fail"] = False
        if scheduler_cfg.run_report:
            live_report = {
                "report_generated_at": None,
                "trade_date": effective_trade_date,
                "mode": "LIVE",
                "automation_enabled": buy_cfg.enabled,
                "loaded_candidates": 0,
                "allowed_candidates": 0,
                "blocked_candidates": 0,
                "paper_order_count": 0,
                "validation_summary": {"block_counts": {"LIVE_DISABLED_IN_SCHEDULER": 1}, "rule_summary": {}},
                "paper_orders": [],
                "paper_performance": {"rows": [], "summary": {"count": 0, "price_data_missing_count": 0}, "benchmark_symbol": _raw("US_BUY_BENCHMARK_SYMBOL", "SPY")},
                "fail_safe_triggered": False,
                "live_transition_readiness": "NO",
                "events": [{"severity": "ERROR", "reason_code": "LIVE_DISABLED_IN_SCHEDULER", "detail": "Scheduler blocks LIVE execution."}],
                "final_candidates": [],
                "blocked_candidates_detail": [],
                "candidates": [],
                "data_missing": False,
                "amount_limit_usage_pct": None,
                "symbol_limit_usage_pct": None,
                "used_amount_usd": 0.0,
                "used_symbol_count": 0,
                "notification_summary": "",
                "notification_detail": "",
            }
            write_buy_report_json(live_report)
            write_buy_report_markdown(live_report)
            result["report_executed"] = True
        _persist_scheduler_result(result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_job_console(result))
        return result

    started_at = time.perf_counter()
    try:
        automation_report: dict[str, object] | None = None
        if scheduler_cfg.run_automation:
            automation_report = run_buy_automation(
                trade_date=effective_trade_date,
                account_id="US_BUY_SCHEDULER",
                persist_logs=True,
            )
            result["automation_executed"] = True

            events = automation_report.get("events") or []
            if events and automation_report.get("loaded_candidates", 0) == 0:
                reason_codes = {str(item.get("reason_code") or "") for item in events if isinstance(item, dict)}
                if "RANKING_DATA_MISSING" in reason_codes:
                    result["error"] = "SOURCE_DATA_MISSING"
                elif "TOP_N_EMPTY" in reason_codes:
                    result["error"] = "NO_CANDIDATE"

        if scheduler_cfg.run_report:
            if automation_report is None:
                automation_report = run_buy_automation(
                    trade_date=effective_trade_date,
                    account_id="US_BUY_SCHEDULER",
                    persist_logs=True,
                )
                result["automation_executed"] = True
            final_report = finalize_buy_report(automation_report)
            write_buy_report_json(final_report)
            write_buy_report_markdown(final_report)
            result["report_executed"] = True

        elapsed = time.perf_counter() - started_at
        if elapsed > scheduler_cfg.max_runtime_seconds:
            result["success"] = False
            result["error"] = "SCHEDULER_TIMEOUT"
        summary_source = automation_report or {}
        result["summary"] = {
            "loaded_candidates": summary_source.get("loaded_candidates", 0),
            "allowed_candidates": summary_source.get("allowed_candidates", 0),
            "blocked_candidates": summary_source.get("blocked_candidates", 0),
            "paper_orders": len(summary_source.get("paper_orders", [])),
            "top_block_reason": _top_block_reason(summary_source),
            "fail_safe": any(
                "FAILSAFE" in str(reason).upper() or "MISSING" in str(reason).upper()
                for reason in (summary_source.get("block_summary") or {}).keys()
            ),
        }
        _persist_scheduler_result(result, trade_date=effective_trade_date or summary_source.get("trade_date"))
        if emit_console:
            print(_render_job_console(result))
        return result
    except Exception as exc:
        result["success"] = False
        result["error"] = "US_BUY_AUTOMATION_ERROR"
        result["error_detail"] = str(exc)
        result["pipeline_should_fail"] = scheduler_cfg.fail_pipeline_on_error
        _persist_scheduler_result(result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_job_console(result))
        if scheduler_cfg.fail_pipeline_on_error:
            raise
        return result


def main() -> int:
    cfg = load_buy_scheduler_config()
    setup_logging(cfg.log_level)
    try:
        result = run_buy_scheduler_job()
    except Exception as exc:
        LOGGER.info("[US_BUY_SCHEDULER] failed error=%s", exc)
        return 1
    if result.get("success") is False and result.get("pipeline_should_fail"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
