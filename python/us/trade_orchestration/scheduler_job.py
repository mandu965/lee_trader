from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.trade_orchestration.config import TradeOrchestrationConfig, load_trade_orchestration_config
from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_notification_formatter import (
    build_dashboard_notification_json_payload,
    render_dashboard_notification_text,
)
from python.us.dashboard.dashboard_notification_payload import write_dashboard_notification_payloads
from python.us.dashboard.scheduler_integration import run_dashboard_scheduler_integration
from python.us.notification.run_us_notification_adapter import run_notification_adapter
from python.us.trade_orchestration.daily_trade_orchestrator import run_trade_orchestration
from python.us.trade_orchestration.health_check import run_trade_health_check
from python.us.trade_orchestration.operations_checklist import build_operations_checklist, write_operations_checklist
from python.us.trade_orchestration.run_lock import acquire_run_lock, release_run_lock
from python.us.trade_orchestration.scheduler_guard import evaluate_scheduler_guard


LOGGER = logging.getLogger("us_trade_scheduler_job")


@dataclass(frozen=True)
class TradeSchedulerSummary:
    sell_signals: int
    buy_candidates: int
    conflict_blocked: int
    final_buy_candidates: int


def setup_logging(level_name: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def _artifact_dir(cfg: TradeOrchestrationConfig) -> Path:
    return cfg.report_output_dir


def _persist_scheduler_result(cfg: TradeOrchestrationConfig, result: dict[str, object], *, trade_date: str | None) -> None:
    directory = _artifact_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_trade_date = str(trade_date or "unknown")
    safe_mode = str(result.get("mode") or "UNKNOWN").upper()
    path = directory / f"trade_scheduler_job_{safe_trade_date}_{safe_mode}_{timestamp}.json"
    payload = dict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _render_console(result: dict[str, object]) -> str:
    lines = [
        "[US TRADE SCHEDULER JOB]",
        f"enabled={str(result.get('enabled')).lower()}",
        f"mode={result.get('mode')}",
        f"guard_passed={str(result.get('guard_passed')).lower()}",
        f"lock_acquired={str(result.get('lock_acquired')).lower()}",
        f"orchestration_executed={str(result.get('orchestration_executed')).lower()}",
        f"report_generated={str(result.get('report_generated')).lower()}",
        f"dashboard_executed={str(result.get('dashboard_executed')).lower()}",
        f"dashboard_success={str(result.get('dashboard_success')).lower()}",
        f"health_check_passed={str(result.get('health_check_passed')).lower()}",
        f"success={str(result.get('success')).lower()}",
    ]
    if result.get("summary"):
        summary = result["summary"]
        lines.extend(
            [
                "",
                "SELL:",
                f"positions={summary.get('loaded_positions', 0)}",
                f"sell={summary.get('sell_signals', 0)}",
                f"hold={summary.get('hold_positions', 0)}",
                f"review_required={summary.get('review_required', 0)}",
                "",
                "BUY:",
                f"candidates={summary.get('buy_candidates', 0)}",
                f"allowed_before_conflict={summary.get('allowed_before_conflict', 0)}",
                f"conflict_blocked={summary.get('conflict_blocked', 0)}",
                f"allowed_after_conflict={summary.get('final_buy_candidates', 0)}",
                "",
                "CONFLICT:",
                f"OPEN_POSITION_EXISTS={summary.get('OPEN_POSITION_EXISTS', 0)}",
                f"SELL_SIGNAL_EXISTS={summary.get('SELL_SIGNAL_EXISTS', 0)}",
                f"COOLDOWN_ACTIVE={summary.get('COOLDOWN_ACTIVE', 0)}",
                "",
                f"pipeline_should_fail={str(result.get('pipeline_should_fail')).lower()}",
            ]
        )
    if result.get("errors"):
        lines.append(f"errors={','.join(result.get('errors') or [])}")
    return "\n".join(lines)


def run_trade_scheduler_job(*, trade_date: str | None = None, emit_console: bool = True) -> dict[str, object]:
    cfg = load_trade_orchestration_config()
    requested_trade_date = date.fromisoformat(trade_date) if trade_date else None
    result: dict[str, object] = {
        "job": "us_trade_orchestration",
        "enabled": cfg.scheduler_enabled,
        "mode": cfg.mode,
        "guard_passed": False,
        "lock_acquired": False,
        "orchestration_executed": False,
        "report_generated": False,
        "dashboard_executed": False,
        "dashboard_success": True,
        "dashboard_json_report_path": None,
        "dashboard_markdown_report_path": None,
        "dashboard_latest_json_path": None,
        "dashboard_latest_markdown_path": None,
        "dashboard_errors": [],
        "notification_payload_generated": False,
        "notification_payload_path": None,
        "notification_adapter_executed": False,
        "notification_adapter_success": True,
        "notification_adapter_errors": [],
        "health_check_passed": False,
        "success": True,
        "pipeline_should_fail": False,
        "summary": {},
        "warnings": [],
        "errors": [],
    }

    guard = evaluate_scheduler_guard(cfg, requested_trade_date=requested_trade_date)
    result["warnings"].extend(guard.get("warnings") or [])
    result["errors"].extend(guard.get("errors") or [])
    result["guard_passed"] = bool(guard.get("can_run"))
    effective_trade_date = trade_date or guard.get("ranking_trade_date") or (requested_trade_date.isoformat() if requested_trade_date else date.today().isoformat())

    if not guard.get("can_run"):
        result["success"] = False
        result["pipeline_should_fail"] = bool(guard.get("pipeline_should_fail"))
        _persist_scheduler_result(cfg, result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_console(result))
        return result

    lock_result = {"lock_acquired": True, "lock_path": None, "stale_lock_removed": False}
    if cfg.scheduler_prevent_duplicate_run:
        lock_result = acquire_run_lock(cfg, trade_date=effective_trade_date, owner="scheduler_job")
        result["lock_acquired"] = bool(lock_result.get("lock_acquired"))
        if not result["lock_acquired"]:
            result["success"] = False
            result["errors"].append(str(lock_result.get("reason") or "DUPLICATE_RUN_DETECTED"))
            result["pipeline_should_fail"] = False
            _persist_scheduler_result(cfg, result, trade_date=effective_trade_date)
            if emit_console:
                print(_render_console(result))
            return result
    else:
        result["lock_acquired"] = True

    started_at = time.perf_counter()
    health_result: dict[str, object] | None = None
    dashboard_result: dict[str, object] | None = None
    try:
        orchestration_result = run_trade_orchestration(
            trade_date=effective_trade_date,
            account_id="US_TRADE_SCHEDULER",
            persist_logs=True,
        )
        orchestration_result.setdefault("dashboard_payload", None)
        result["orchestration_executed"] = True
        result["report_generated"] = bool(orchestration_result.get("report_generated"))
        result["warnings"].extend(orchestration_result.get("warnings") or [])
        if not orchestration_result.get("success"):
            result["success"] = False
            if orchestration_result.get("error"):
                result["errors"].append(str(orchestration_result.get("error")))

        if cfg.scheduler_run_dashboard:
            dashboard_result = run_dashboard_scheduler_integration(
                trade_date=str(orchestration_result.get("trade_date") or effective_trade_date),
                force=True,
            )
            result["dashboard_executed"] = bool(dashboard_result.get("dashboard_executed"))
            result["dashboard_success"] = bool(dashboard_result.get("success"))
            result["dashboard_json_report_path"] = dashboard_result.get("json_report_path")
            result["dashboard_markdown_report_path"] = dashboard_result.get("markdown_report_path")
            result["dashboard_latest_json_path"] = dashboard_result.get("latest_json_path")
            result["dashboard_latest_markdown_path"] = dashboard_result.get("latest_markdown_path")
            result["dashboard_errors"] = list(dashboard_result.get("errors") or [])
            result["warnings"].extend(dashboard_result.get("warnings") or [])
            if dashboard_result.get("errors"):
                result["errors"].extend(str(item) for item in dashboard_result.get("errors") or [])
            if isinstance(dashboard_result.get("payload"), dict):
                dashboard_result["payload"].setdefault("meta", {})
                dashboard_result["payload"]["meta"]["json_report_path"] = dashboard_result.get("json_report_path")
                dashboard_result["payload"]["meta"]["markdown_report_path"] = dashboard_result.get("markdown_report_path")
                dashboard_result["payload"]["meta"]["latest_json_path"] = dashboard_result.get("latest_json_path")
                dashboard_result["payload"]["meta"]["latest_markdown_path"] = dashboard_result.get("latest_markdown_path")
            orchestration_result["dashboard_payload"] = dashboard_result.get("payload")
            orchestration_result["dashboard_json_report_path"] = dashboard_result.get("json_report_path")
            orchestration_result["dashboard_markdown_report_path"] = dashboard_result.get("markdown_report_path")
            orchestration_result["dashboard_latest_json_path"] = dashboard_result.get("latest_json_path")
            orchestration_result["dashboard_latest_markdown_path"] = dashboard_result.get("latest_markdown_path")
            if dashboard_result.get("pipeline_should_fail"):
                result["pipeline_should_fail"] = True

        if cfg.scheduler_run_health_check and cfg.health_check_enabled:
            health_result = run_trade_health_check(cfg, orchestration_result=orchestration_result, dashboard_result=dashboard_result)
            result["health_check_passed"] = health_result.get("health_status") == "PASS"
            result["warnings"].extend(health_result.get("warnings") or [])
            result["errors"].extend(health_result.get("errors") or [])
            if health_result.get("pipeline_should_fail"):
                result["pipeline_should_fail"] = True
        else:
            result["health_check_passed"] = True

        if cfg.scheduler_run_report:
            checklist = build_operations_checklist(orchestration_result=orchestration_result, health_result=health_result or {})
            write_operations_checklist(cfg, trade_date=str(orchestration_result.get("trade_date") or effective_trade_date), markdown=checklist)

        dashboard_cfg = load_dashboard_config()
        if (dashboard_cfg.notification_enabled or cfg.scheduler_run_notification_adapter) and dashboard_result is not None and isinstance(dashboard_result.get("payload"), dict):
            notification_json = build_dashboard_notification_json_payload(
                dashboard_result["payload"],
                dashboard_health=(health_result or {}).get("dashboard"),
                cfg=dashboard_cfg,
            )
            notification_text = render_dashboard_notification_text(
                dashboard_result["payload"],
                dashboard_health=(health_result or {}).get("dashboard"),
                cfg=dashboard_cfg,
            )
            notification_paths = write_dashboard_notification_payloads(
                dashboard_cfg,
                trade_date=str(orchestration_result.get("trade_date") or effective_trade_date),
                text_payload=notification_text,
                json_payload=notification_json,
                formats=dashboard_cfg.notification_formats,
            )
            result["notification_payload_generated"] = True
            result["notification_payload_path"] = notification_paths.get("json_path") or notification_paths.get("text_path")
        if cfg.scheduler_run_notification_adapter:
            notification_adapter_result = run_notification_adapter(
                trade_date=str(orchestration_result.get("trade_date") or effective_trade_date),
                force=True,
                emit_console=False,
            )
            result["notification_adapter_executed"] = bool(notification_adapter_result.get("notification_executed"))
            result["notification_adapter_success"] = not bool(notification_adapter_result.get("pipeline_should_fail")) and not bool(notification_adapter_result.get("errors"))
            result["notification_adapter_errors"] = list(notification_adapter_result.get("errors") or [])
            result["warnings"].extend(notification_adapter_result.get("warnings") or [])
            if notification_adapter_result.get("errors"):
                result["errors"].extend(str(item) for item in notification_adapter_result.get("errors") or [])
            if notification_adapter_result.get("pipeline_should_fail"):
                result["pipeline_should_fail"] = True

        elapsed = time.perf_counter() - started_at
        if elapsed > cfg.scheduler_max_runtime_seconds:
            result["success"] = False
            result["errors"].append("SCHEDULER_TIMEOUT")

        conflict_summary = orchestration_result.get("conflict_summary") or {}
        result["summary"] = {
            "loaded_positions": orchestration_result.get("sell_summary", {}).get("loaded_positions", 0),
            "sell_signals": orchestration_result.get("sell_summary", {}).get("sell_signals", 0),
            "hold_positions": orchestration_result.get("sell_summary", {}).get("hold_positions", 0),
            "review_required": orchestration_result.get("sell_summary", {}).get("review_required", 0),
            "buy_candidates": orchestration_result.get("buy_summary", {}).get("loaded_candidates", 0),
            "allowed_before_conflict": orchestration_result.get("buy_summary", {}).get("allowed_before_conflict", 0),
            "conflict_blocked": orchestration_result.get("buy_summary", {}).get("conflict_blocked", 0),
            "final_buy_candidates": orchestration_result.get("buy_summary", {}).get("allowed_after_conflict", 0),
            "OPEN_POSITION_EXISTS": conflict_summary.get("OPEN_POSITION_EXISTS", 0),
            "SELL_SIGNAL_EXISTS": conflict_summary.get("SELL_SIGNAL_EXISTS", 0),
            "COOLDOWN_ACTIVE": conflict_summary.get("COOLDOWN_ACTIVE", 0),
        }
        result["success"] = (
            result["success"]
            and orchestration_result.get("success", False)
            and (result["health_check_passed"] or not cfg.scheduler_run_health_check)
            and (result["dashboard_success"] or not cfg.scheduler_run_dashboard)
        )
        if orchestration_result.get("pipeline_should_fail"):
            result["pipeline_should_fail"] = True
        _persist_scheduler_result(cfg, result, trade_date=str(orchestration_result.get("trade_date") or effective_trade_date))
        if emit_console:
            print(_render_console(result))
        return result
    except Exception as exc:
        result["success"] = False
        result["errors"].append(f"TRADE_SCHEDULER_JOB_ERROR:{exc}")
        result["pipeline_should_fail"] = cfg.scheduler_fail_pipeline_on_error
        _persist_scheduler_result(cfg, result, trade_date=effective_trade_date)
        if emit_console:
            print(_render_console(result))
        if cfg.scheduler_fail_pipeline_on_error:
            raise
        return result
    finally:
        if cfg.scheduler_prevent_duplicate_run:
            release_result = release_run_lock(cfg, trade_date=effective_trade_date)
            warning = release_result.get("warning")
            if warning:
                result.setdefault("warnings", []).append(str(warning))


def main() -> int:
    setup_logging()
    result = run_trade_scheduler_job()
    if result.get("success") is False and result.get("pipeline_should_fail"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
