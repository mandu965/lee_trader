from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from payload_store import upsert_json_payload
from validate_runtime_assets import validate_runtime_assets

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_TIME = "18:10"
DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEZONE = "Asia/Seoul"
DEFAULT_SKIP_CATCHUP = True
DEFAULT_RUN_POLICY = "always"
DEFAULT_PRIMARY_MAX_AGE_HOURS = 20
DEFAULT_SCHEDULER_MODE = "internal_service"
DEFAULT_INTERVAL_MINUTES = 0
OPTIONAL_POST_SYNC_STEPS = {
    "build_live_kpi_daily_report",
    "build_live_closed_trade_report",
    "build_quality_risk_guard_live_review",
    "check_live_quality_guard_outputs",
}
STRUCTURED_ERROR_PREFIXES = (
    "[LIVE_KPI_REPORT_ERROR]",
    "[STEP_ERROR]",
)
DEFAULT_FAIL_ON_RUNTIME_ASSET_ISSUES = True
DEFAULT_MULTI_SLOT_SUCCESS_POLICY = "each_slot"


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)


def _should_sync_web_display() -> bool:
    return str(os.environ.get("SCHEDULER_SYNC_WEB_DISPLAY", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _close_batch_command(*, skip_web_sync: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "python" / "run_manual_close_batch.py"),
        "--skip-build",
        "--skip-node-api",
    ]
    if skip_web_sync or not _should_sync_web_display():
        command.append("--skip-web-sync")
    return command


def _auto_buy_refresh_command() -> list[str]:
    command = [sys.executable, str(ROOT / "python" / "run_operational_refresh.py"), "--with-live-account", "--skip-live-preview"]
    if str(os.environ.get("AUTO_TRADE_SKIP_THEME_SHADOW", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        command.append("--skip-theme-shadow")
    if str(os.environ.get("AUTO_TRADE_SKIP_PAPER_TRADING", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        command.append("--skip-paper-trading")
    if str(os.environ.get("AUTO_TRADE_SKIP_PAPER_TRADING_DB", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        command.append("--skip-paper-trading-db")
    return command


def _auto_buy_submit_command() -> list[str]:
    command = [sys.executable, str(ROOT / "python" / "submit_live_orders.py")]
    if str(os.environ.get("AUTO_TRADE_EXECUTE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        confirm_text = str(os.environ.get("AUTO_TRADE_CONFIRM_TEXT", "")).strip()
        if confirm_text != "LIVE_ORDER":
            raise ValueError("AUTO_TRADE_CONFIRM_TEXT must be LIVE_ORDER when AUTO_TRADE_EXECUTE is enabled")
        command.extend(["--execute", "--confirm-text", confirm_text])
        if str(os.environ.get("AUTO_TRADE_ALLOW_BUY", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            command.append("--allow-buy")
        if str(os.environ.get("AUTO_TRADE_FORCE_RESUBMIT", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            command.append("--force-resubmit")
    return command


def _live_account_sync_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "sync_live_account_holdings.py")]


def _live_order_fills_sync_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "sync_live_order_fills.py")]


def _live_trade_consistency_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_live_trade_consistency_report.py")]


def _live_trade_review_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_live_trade_review.py")]


def _live_trade_review_summary_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_live_trade_review_summary.py")]


def _live_kpi_daily_report_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_live_kpi_daily_report.py")]


def _quality_risk_guard_live_review_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_quality_risk_guard_live_review.py")]


def _live_closed_trade_report_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "build_live_closed_trade_report.py")]


def _live_quality_guard_output_check_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "check_live_quality_guard_outputs.py")]


def _rule_after_close_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "run_rule_after_close_cycle.py")]


def _rule_before_open_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "run_rule_before_open_cycle.py")]


def _rule_after_open_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "run_rule_after_open_cycle.py")]


def _us_macro_collect_command() -> list[str]:
    lookback = str(os.environ.get("US_MACRO_SCHEDULER_LOOKBACK_DAYS", "2")).strip() or "2"
    return [
        sys.executable,
        str(ROOT / "python" / "run_us_macro_overlay_scheduler.py"),
        "--lookback-days", lookback,
    ]


def _us_macro_shadow_command() -> list[str]:
    return [sys.executable, str(ROOT / "python" / "run_us_macro_overlay_shadow.py")]


def _resolve_run_steps() -> list[tuple[str, list[str]]]:
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    if command_set == "intraday":
        return [
            ("run_intraday_refresh", [sys.executable, str(ROOT / "python" / "run_intraday_refresh.py")]),
        ]
    if command_set == "auto_buy":
        return [
            ("run_operational_refresh", _auto_buy_refresh_command()),
            ("submit_live_orders", _auto_buy_submit_command()),
        ]
    if command_set == "live_sync":
        return [
            ("sync_live_account_holdings", _live_account_sync_command()),
            ("sync_live_order_fills", _live_order_fills_sync_command()),
            ("sync_rule_live_account_snapshot", [sys.executable, str(ROOT / "python" / "rule_live_account_snapshot.py")]),
            ("build_live_trade_consistency_report", _live_trade_consistency_command()),
            ("build_live_trade_review", _live_trade_review_command()),
            ("build_live_trade_review_summary", _live_trade_review_summary_command()),
            ("build_live_kpi_daily_report", _live_kpi_daily_report_command()),
            ("build_live_closed_trade_report", _live_closed_trade_report_command()),
            ("build_quality_risk_guard_live_review", _quality_risk_guard_live_review_command()),
            ("check_live_quality_guard_outputs", _live_quality_guard_output_check_command()),
        ]
    if command_set == "rule_after_close":
        return [
            ("run_manual_close_batch", _close_batch_command(skip_web_sync=True)),
            ("run_rule_after_close_cycle", _rule_after_close_command()),
        ]
    if command_set == "rule_before_open":
        return [("run_rule_before_open_cycle", _rule_before_open_command())]
    if command_set == "rule_after_open":
        return [("run_rule_after_open_cycle", _rule_after_open_command())]
    if command_set == "us_macro":
        return [
            ("run_us_macro_collect", _us_macro_collect_command()),
        ]
    if command_set == "us_macro_shadow":
        return [
            ("run_us_macro_shadow", _us_macro_shadow_command()),
        ]
    return [("run_manual_close_batch", _close_batch_command())]


def _resolve_post_sync_steps() -> list[tuple[str, list[str]]]:
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    if command_set == "auto_buy":
        steps: list[tuple[str, list[str]]] = [
            ("sync_live_account_holdings", _live_account_sync_command()),
            ("sync_live_order_fills", _live_order_fills_sync_command()),
            ("build_live_trade_consistency_report", _live_trade_consistency_command()),
            ("build_live_trade_review", _live_trade_review_command()),
            ("build_live_trade_review_summary", _live_trade_review_summary_command()),
            ("build_live_kpi_daily_report", _live_kpi_daily_report_command()),
            ("build_live_closed_trade_report", _live_closed_trade_report_command()),
            ("build_quality_risk_guard_live_review", _quality_risk_guard_live_review_command()),
            ("check_live_quality_guard_outputs", _live_quality_guard_output_check_command()),
        ]
        if not _should_sync_web_display():
            return steps
        if not str(os.environ.get("WEB_DATABASE_URL", "")).strip():
            logging.info("Skip web display sync: WEB_DATABASE_URL not set")
            return steps
        steps.append(
            (
                "sync_web_display_data",
                [
                    sys.executable,
                    str(ROOT / "python" / "sync_web_display_data.py"),
                    "--skip-core",
                    "--skip-paper-trading",
                    "--skip-trades",
                ],
            )
        )
        return steps
    if command_set == "close":
        return []
    if command_set == "rule_after_close":
        if not _should_sync_web_display():
            return []
        if not str(os.environ.get("WEB_DATABASE_URL", "")).strip():
            logging.info("Skip web display sync: WEB_DATABASE_URL not set")
            return []
        return [("sync_web_display_data", [sys.executable, str(ROOT / "python" / "sync_web_display_data.py")])]
    if command_set in {"rule_before_open", "rule_after_open"}:
        return []
    if not _should_sync_web_display():
        return []
    if not str(os.environ.get("WEB_DATABASE_URL", "")).strip():
        logging.info("Skip web display sync: WEB_DATABASE_URL not set")
        return []

    command = [sys.executable, str(ROOT / "python" / "sync_web_display_data.py")]
    if command_set in {"intraday", "live_sync"}:
        command.extend(["--skip-core", "--skip-paper-trading", "--skip-trades"])
    return [("sync_web_display_data", command)]


def _scheduler_mode() -> str:
    return str(os.environ.get("SCHEDULER_MODE", DEFAULT_SCHEDULER_MODE)).strip() or DEFAULT_SCHEDULER_MODE


def ensure_dirs() -> None:
    _status_path().parent.mkdir(parents=True, exist_ok=True)
    _log_path().parent.mkdir(parents=True, exist_ok=True)


def _resolve_runtime_path(raw_value: str, default_path: Path) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        return default_path
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _status_path() -> Path:
    return _resolve_runtime_path(
        os.environ.get("SCHEDULER_STATUS_PATH", ""),
        DEFAULT_OUTPUTS_DIR / "auto_ops_scheduler_status.json",
    )


def _log_path() -> Path:
    return _resolve_runtime_path(
        os.environ.get("SCHEDULER_LOG_PATH", ""),
        DEFAULT_LOG_DIR / "auto_ops_scheduler.log",
    )


def _primary_status_path() -> Path:
    return _resolve_runtime_path(
        os.environ.get("SCHEDULER_PRIMARY_STATUS_PATH", ""),
        _status_path(),
    )


def setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(_log_path(), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _load_status() -> dict[str, object]:
    status_path = _status_path()
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Failed to load scheduler status file", exc_info=True)
        return {}


def _write_status(payload: dict[str, object]) -> None:
    ensure_dirs()
    status_path = _status_path()
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload_key = None
    if status_path.name == "auto_ops_scheduler_status.json":
        payload_key = "auto_ops_scheduler_status"
    elif status_path.name == "auto_ops_recovery_scheduler_status.json":
        payload_key = "auto_ops_recovery_scheduler_status"
    try:
        if payload_key:
            upsert_json_payload(
                payload_key,
                payload,
                asof_date=payload.get("last_success_date"),
                generated_at=payload.get("last_success_at") or payload.get("last_attempt_at"),
                source_path=status_path,
            )
        elif status_path.name == "auto_ops_auto_buy_scheduler_status.json":
            upsert_json_payload(
                "auto_ops_auto_buy_scheduler_status",
                payload,
                asof_date=payload.get("last_success_date"),
                generated_at=payload.get("last_success_at") or payload.get("last_attempt_at"),
                source_path=status_path,
            )
        elif status_path.name == "auto_ops_live_account_sync_scheduler_status.json":
            upsert_json_payload(
                "auto_ops_live_account_sync_scheduler_status",
                payload,
                asof_date=payload.get("last_success_date"),
                generated_at=payload.get("last_success_at") or payload.get("last_attempt_at"),
                source_path=status_path,
            )
    except Exception:
        logging.warning("Scheduler status file was written, but DB payload status sync failed", exc_info=True)


def _now(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def _parse_daily_time(raw: str) -> tuple[int, int]:
    text = (raw or DEFAULT_TIME).strip()
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception as exc:  # pragma: no cover - config guard
        raise ValueError(f"Invalid SCHEDULER_DAILY_TIME: {text}") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid SCHEDULER_DAILY_TIME: {text}")
    return hour, minute


def _parse_daily_times(raw: str) -> list[tuple[int, int]]:
    text = str(raw or "").strip()
    if not text:
        return []
    parsed = {_parse_daily_time(part.strip()) for part in text.split(",") if part.strip()}
    return sorted(parsed)


def _parse_interval_minutes() -> int:
    raw_minutes = str(os.environ.get("SCHEDULER_INTERVAL_MINUTES", "")).strip()
    raw_hours = str(os.environ.get("SCHEDULER_INTERVAL_HOURS", "")).strip()
    if raw_minutes:
        return max(0, int(raw_minutes))
    if raw_hours:
        return max(0, int(raw_hours) * 60)
    return DEFAULT_INTERVAL_MINUTES


def _skip_after_failure_until_next_day() -> bool:
    return str(os.environ.get("SCHEDULER_SKIP_AFTER_FAILURE_UNTIL_NEXT_DAY", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _multi_slot_success_policy() -> str:
    value = str(os.environ.get("SCHEDULER_MULTI_SLOT_SUCCESS_POLICY", DEFAULT_MULTI_SLOT_SUCCESS_POLICY)).strip().lower()
    if value in {"once_per_day", "each_slot"}:
        return value
    return DEFAULT_MULTI_SLOT_SUCCESS_POLICY


def _should_run_today(now: datetime, scheduled_hour: int, scheduled_minute: int, status: dict[str, object]) -> bool:
    last_success_date = str(status.get("last_success_date") or "").strip()
    bootstrap_skip_date = str(status.get("bootstrap_skip_until_date") or "").strip()
    policy_skip_date = str(status.get("last_policy_skip_date") or "").strip()
    failure_skip_date = str(status.get("last_failure_skip_date") or "").strip()
    today = now.strftime("%Y-%m-%d")
    if last_success_date == today:
        return False
    if bootstrap_skip_date == today:
        return False
    if policy_skip_date == today:
        return False
    if failure_skip_date == today:
        return False
    return (now.hour, now.minute) >= (scheduled_hour, scheduled_minute)


def _should_run_daily_slots(now: datetime, schedule_slots: list[tuple[int, int]], status: dict[str, object]) -> bool:
    if not schedule_slots:
        return False
    bootstrap_skip_date = str(status.get("bootstrap_skip_until_date") or "").strip()
    policy_skip_date = str(status.get("last_policy_skip_date") or "").strip()
    failure_skip_date = str(status.get("last_failure_skip_date") or "").strip()
    today = now.strftime("%Y-%m-%d")
    if bootstrap_skip_date == today:
        return False
    if policy_skip_date == today:
        return False
    if failure_skip_date == today:
        return False

    eligible_slots = [
        f"{today} {hour:02d}:{minute:02d}"
        for hour, minute in schedule_slots
        if (now.hour, now.minute) >= (hour, minute)
    ]
    if not eligible_slots:
        return False

    if _multi_slot_success_policy() == "once_per_day":
        last_success_date = str(status.get("last_success_date") or "").strip()
        if last_success_date == today:
            return False

    last_attempt_slot = str(status.get("last_attempt_schedule_slot") or "").strip()
    if last_attempt_slot == eligible_slots[-1]:
        return False

    last_success_slot = str(status.get("last_success_schedule_slot") or "").strip()
    return last_success_slot != eligible_slots[-1]


def _should_run_interval(now: datetime, interval_minutes: int, status: dict[str, object]) -> bool:
    if interval_minutes <= 0:
        return False
    last_success_at = _parse_last_success_at(status)
    if last_success_at is None:
        return True
    last_success = last_success_at.astimezone(now.tzinfo) if last_success_at.tzinfo else last_success_at.replace(tzinfo=now.tzinfo)
    return now >= last_success + timedelta(minutes=max(1, interval_minutes))


def _parse_last_success_at(status: dict[str, object]) -> datetime | None:
    raw = str(status.get("last_success_at") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _load_primary_status() -> dict[str, object]:
    primary_path = _primary_status_path()
    if not primary_path.exists():
        return {}
    try:
        return json.loads(primary_path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Failed to load primary scheduler status file: %s", primary_path, exc_info=True)
        return {}


def _evaluate_run_policy(now: datetime, policy: str) -> tuple[bool, str]:
    normalized_policy = (policy or DEFAULT_RUN_POLICY).strip().lower() or DEFAULT_RUN_POLICY
    if normalized_policy == "always":
        return True, ""
    if normalized_policy != "if_primary_stale":
        logging.warning("Unknown SCHEDULER_RUN_POLICY='%s' -> fallback to 'always'", policy)
        return True, ""

    primary_status = _load_primary_status()
    primary_last_success_at = _parse_last_success_at(primary_status)
    primary_max_age_hours = int(os.environ.get("SCHEDULER_PRIMARY_MAX_AGE_HOURS", str(DEFAULT_PRIMARY_MAX_AGE_HOURS)))
    if primary_last_success_at is None:
        return True, "Primary scheduler has no recorded successful run."

    age = now - primary_last_success_at.astimezone(now.tzinfo) if primary_last_success_at.tzinfo else now - primary_last_success_at.replace(tzinfo=now.tzinfo)
    if age <= timedelta(hours=max(1, primary_max_age_hours)):
        return False, (
            f"Primary scheduler is healthy; last_success_at={primary_last_success_at.isoformat()} "
            f"within {primary_max_age_hours}h window."
        )
    return True, (
        f"Primary scheduler is stale; last_success_at={primary_last_success_at.isoformat()} "
        f"older than {primary_max_age_hours}h window."
    )


def _should_fail_on_runtime_asset_issues() -> bool:
    return str(
        os.environ.get(
            "SCHEDULER_FAIL_ON_RUNTIME_ASSET_ISSUES",
            "1" if DEFAULT_FAIL_ON_RUNTIME_ASSET_ISSUES else "0",
        )
    ).strip().lower() in {"1", "true", "yes", "on"}


class SchedulerStepFailure(RuntimeError):
    def __init__(self, details: dict[str, object]):
        self.details = details
        super().__init__(str(details.get("error_message") or details.get("step_name") or "scheduler step failed"))


def _script_name(command: list[str]) -> str:
    for part in reversed(command):
        text = str(part)
        if text.endswith(".py"):
            return Path(text).name
    return Path(str(command[0])).name if command else "unknown"


def _tail_text(value: str, limit: int = 600) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if len(text) <= limit else text[-limit:]


def _parse_structured_error(stderr: str) -> dict[str, object]:
    for raw_line in reversed(str(stderr or "").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        for prefix in STRUCTURED_ERROR_PREFIXES:
            if line.startswith(prefix):
                try:
                    parsed = json.loads(line[len(prefix):].strip())
                except Exception:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}


def _infer_failure_hint(script_name: str, stderr: str, stdout: str) -> str:
    combined = "\n".join(part for part in (stderr, stdout) if part).lower()
    if "analytics view sql not found" in combined or "missing a required sql/report file" in combined:
        return "Runtime image is missing postgres/analytics_live_trade_views.sql. Rebuild the image after fixing .dockerignore."
    if "connection refused" in combined or "could not connect" in combined:
        return "DB connectivity failed. Verify DATABASE_URL/Postgres availability."
    if "does not exist" in combined and "column" in combined:
        return "A required DB column is missing. Apply the matching schema/view migration."
    if "does not exist" in combined:
        return "A required DB table/view is missing. Apply the matching schema/view migration."
    return f"Check {script_name} stderr/stdout for the failing stage."


def _build_failure_details(name: str, command: list[str], returncode: int, stdout: str, stderr: str) -> dict[str, object]:
    structured = _parse_structured_error(stderr)
    script_name = str(structured.get("script_name") or _script_name(command))
    error_message = str(
        structured.get("error_message")
        or _tail_text(stderr.splitlines()[-1] if str(stderr or "").splitlines() else "")
        or _tail_text(stdout.splitlines()[-1] if str(stdout or "").splitlines() else "")
        or f"{script_name} exited with code {returncode}"
    )
    hint = str(structured.get("hint") or _infer_failure_hint(script_name, stderr, stdout))
    exception_type = str(structured.get("exception_type") or "CalledProcessError")
    occurred_at = str(structured.get("occurred_at") or datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).isoformat())
    return {
        "step_name": str(structured.get("step_name") or name),
        "script_name": script_name,
        "exit_code": int(structured.get("exit_code") or returncode),
        "exception_type": exception_type,
        "error_message": error_message,
        "hint": hint,
        "occurred_at": occurred_at,
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
        "command": [str(part) for part in command],
    }


def _run_step(name: str, command: list[str]) -> None:
    logging.info("START %s", name)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout.strip():
            logging.error("STDOUT %s\n%s", name, completed.stdout.rstrip())
        if completed.stderr.strip():
            logging.error("STDERR %s\n%s", name, completed.stderr.rstrip())
        raise SchedulerStepFailure(
            _build_failure_details(name, command, completed.returncode, completed.stdout, completed.stderr)
        )
    logging.info("OK %s", name)


def run_daily_cycle(now: datetime, tz_name: str, status: dict[str, object]) -> dict[str, object]:
    run_steps = _resolve_run_steps()
    post_sync_steps = _resolve_post_sync_steps()
    started_at = now.isoformat()
    payload = {
        **status,
        "timezone": tz_name,
        "scheduler_mode": _scheduler_mode(),
        "status": "running",
        "last_attempt_at": started_at,
        "last_error": "",
        "last_failure_details": {},
        "last_warning_details": {},
        "post_sync_warnings": [],
    }
    _write_status(payload)

    try:
        for name, command in run_steps:
            _run_step(name, command)
        finished_at = _now(ZoneInfo(tz_name)).isoformat()
        payload.update(
            {
                "status": "idle",
                "last_success_at": finished_at,
                "last_success_date": finished_at[:10],
                "last_success_schedule_slot": str(status.get("pending_schedule_slot") or ""),
                "last_attempt_schedule_slot": str(status.get("pending_schedule_slot") or ""),
                "last_completed_step": run_steps[-1][0],
                "last_error": "",
                "status_note": "",
                "last_failure_skip_date": "",
                "last_failure_schedule_slot": "",
                "last_failure_details": {},
                "last_warning_details": {},
                "post_sync_warnings": [],
                "pending_schedule_slot": "",
            }
        )
        _write_status(payload)
        warning_details: list[dict[str, object]] = []
        for name, command in post_sync_steps:
            try:
                _run_step(name, command)
            except SchedulerStepFailure as exc:
                details = exc.details
                if name not in OPTIONAL_POST_SYNC_STEPS:
                    raise
                warning_details.append(details)
                logging.warning("Optional post-sync step failed: %s", details.get("error_message"))
        if warning_details:
            latest_warning = warning_details[-1]
            payload.update(
                {
                    "status": "warning",
                    "status_note": f"Post-sync warning: {latest_warning.get('step_name')} - {latest_warning.get('error_message')}",
                    "last_warning_at": latest_warning.get("occurred_at"),
                    "last_warning_details": latest_warning,
                    "post_sync_warnings": warning_details,
                }
            )
        else:
            payload.update(
                {
                    "status": "idle",
                    "status_note": "",
                    "last_warning_at": "",
                    "last_warning_details": {},
                    "post_sync_warnings": [],
                }
            )
        logging.info("DONE daily cycle completed")
    except SchedulerStepFailure as exc:
        finished_at = _now(ZoneInfo(tz_name)).isoformat()
        details = dict(exc.details)
        details["occurred_at"] = details.get("occurred_at") or finished_at
        payload.update(
            {
                "status": "error",
                "last_failure_at": finished_at,
                "last_attempt_schedule_slot": str(status.get("pending_schedule_slot") or ""),
                "last_failure_schedule_slot": str(status.get("pending_schedule_slot") or ""),
                "last_completed_step": "",
                "last_error": f"{details.get('script_name')} failed (exit={details.get('exit_code')})",
                "last_failure_details": details,
                "status_note": str(details.get("hint") or ""),
                "pending_schedule_slot": "",
            }
        )
        if _skip_after_failure_until_next_day():
            payload["last_failure_skip_date"] = finished_at[:10]
            payload["status_note"] = (
                f"{details.get('hint') or 'Last run failed.'} Skipping further runs until next day."
            )
        logging.exception("Daily cycle failed")
    _write_status(payload)
    return payload


def main() -> int:
    setup_logging()

    tz_name = os.environ.get("SCHEDULER_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    tz = ZoneInfo(tz_name)
    scheduled_hour, scheduled_minute = _parse_daily_time(os.environ.get("SCHEDULER_DAILY_TIME", DEFAULT_TIME))
    scheduled_times = _parse_daily_times(os.environ.get("SCHEDULER_DAILY_TIMES", ""))
    interval_minutes = _parse_interval_minutes()
    poll_seconds = int(os.environ.get("SCHEDULER_POLL_SECONDS", str(DEFAULT_POLL_SECONDS)))
    run_policy = os.environ.get("SCHEDULER_RUN_POLICY", DEFAULT_RUN_POLICY).strip() or DEFAULT_RUN_POLICY
    multi_slot_success_policy = _multi_slot_success_policy()
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    skip_catchup = str(os.environ.get("SCHEDULER_SKIP_CATCHUP_ON_START", "1" if DEFAULT_SKIP_CATCHUP else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    logging.info(
        "Scheduler started timezone=%s daily_time=%02d:%02d daily_times=%s interval_minutes=%d poll_seconds=%d skip_catchup=%s run_policy=%s multi_slot_success_policy=%s command_set=%s status_path=%s",
        tz_name,
        scheduled_hour,
        scheduled_minute,
        ",".join(f"{hour:02d}:{minute:02d}" for hour, minute in scheduled_times) or "-",
        interval_minutes,
        poll_seconds,
        skip_catchup,
        run_policy,
        multi_slot_success_policy,
        command_set,
        _status_path(),
    )

    status = _load_status()
    status.setdefault("timezone", tz_name)
    status.setdefault("scheduler_mode", _scheduler_mode())
    status.setdefault("status", "idle")
    configured_daily_time = (
        ",".join(f"{hour:02d}:{minute:02d}" for hour, minute in scheduled_times)
        if scheduled_times
        else f"{scheduled_hour:02d}:{scheduled_minute:02d}"
    )
    status["configured_daily_time"] = configured_daily_time
    status["configured_interval_minutes"] = interval_minutes
    status["skip_catchup_on_start"] = skip_catchup
    status["run_policy"] = run_policy
    status["multi_slot_success_policy"] = multi_slot_success_policy
    status["command_set"] = command_set
    status.setdefault("pending_schedule_slot", "")
    status.setdefault("last_attempt_schedule_slot", "")
    status.setdefault("last_failure_schedule_slot", "")
    status["status_path"] = str(_status_path())
    status["log_path"] = str(_log_path())
    runtime_asset_issues = validate_runtime_assets(command_set)
    status["runtime_asset_issues"] = runtime_asset_issues
    if runtime_asset_issues:
        for issue in runtime_asset_issues:
            logging.error("Runtime asset validation failed: %s", issue)
        if _should_fail_on_runtime_asset_issues():
            failed_at = _now(tz).isoformat()
            details = {
                "step_name": "runtime_asset_validation",
                "script_name": Path(__file__).name,
                "exit_code": 1,
                "exception_type": "RuntimeAssetValidationError",
                "error_message": runtime_asset_issues[0],
                "hint": "Rebuild/redeploy the runtime image after restoring the missing file.",
                "occurred_at": failed_at,
            }
            status.update(
                {
                    "status": "error",
                    "last_attempt_at": failed_at,
                    "last_failure_at": failed_at,
                    "last_error": "runtime_asset_validation failed (exit=1)",
                    "last_failure_details": details,
                    "status_note": details["hint"],
                }
            )
            _write_status(status)
            return 1
    status.setdefault("bootstrap_skip_until_date", "")
    if (
        interval_minutes <= 0
        and (
        skip_catchup
        and not str(status.get("last_success_date") or "").strip()
        and not str(status.get("bootstrap_skip_until_date") or "").strip()
        )
    ):
        now = _now(tz)
        first_scheduled_hour, first_scheduled_minute = (
            scheduled_times[0] if scheduled_times else (scheduled_hour, scheduled_minute)
        )
        if (now.hour, now.minute) >= (first_scheduled_hour, first_scheduled_minute):
            status["bootstrap_skip_until_date"] = now.strftime("%Y-%m-%d")
            status["status_note"] = "Started after scheduled time; first catch-up run skipped until next day."
        else:
            status["status_note"] = "Waiting for first scheduled run."
    _write_status(status)

    while True:
        now = _now(tz)
        should_run = (
            _should_run_interval(now, interval_minutes, status)
            if interval_minutes > 0
            else (
                _should_run_daily_slots(now, scheduled_times, status)
                if scheduled_times
                else _should_run_today(now, scheduled_hour, scheduled_minute, status)
            )
        )
        if should_run:
            if interval_minutes > 0:
                status["pending_schedule_slot"] = ""
            elif scheduled_times:
                eligible_slots = [
                    f"{now.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}"
                    for hour, minute in scheduled_times
                    if (now.hour, now.minute) >= (hour, minute)
                ]
                status["pending_schedule_slot"] = eligible_slots[-1] if eligible_slots else ""
            else:
                status["pending_schedule_slot"] = f"{now.strftime('%Y-%m-%d')} {scheduled_hour:02d}:{scheduled_minute:02d}"
            can_run, reason = _evaluate_run_policy(now, run_policy)
            status["last_policy_check_at"] = now.isoformat()
            status["last_policy_reason"] = reason
            if can_run:
                status["last_policy_skip_date"] = ""
                status = run_daily_cycle(now, tz_name, status)
            else:
                status["status"] = "idle"
                status["status_note"] = reason
                status["last_skipped_at"] = now.isoformat()
                status["last_policy_skip_date"] = now.strftime("%Y-%m-%d")
                _write_status(status)
        time.sleep(max(5, poll_seconds))


if __name__ == "__main__":
    _load_env()
    raise SystemExit(main())
