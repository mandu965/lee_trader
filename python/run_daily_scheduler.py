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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_TIME = "16:00"
DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEZONE = "Asia/Seoul"
DEFAULT_SKIP_CATCHUP = True
DEFAULT_RUN_POLICY = "always"
DEFAULT_PRIMARY_MAX_AGE_HOURS = 20
DEFAULT_SCHEDULER_MODE = "internal_service"

def _resolve_run_steps() -> list[tuple[str, list[str]]]:
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    if command_set == "intraday":
        return [
            ("run_intraday_refresh", [sys.executable, str(ROOT / "python" / "run_intraday_refresh.py")]),
        ]
    return [
        ("run_pipeline", [sys.executable, str(ROOT / "python" / "run_pipeline.py")]),
        ("run_operational_refresh", [sys.executable, str(ROOT / "python" / "run_operational_refresh.py")]),
    ]


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
    if payload_key:
        upsert_json_payload(
            payload_key,
            payload,
            asof_date=payload.get("last_success_date"),
            generated_at=payload.get("last_success_at") or payload.get("last_attempt_at"),
            source_path=status_path,
        )


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


def _should_run_today(now: datetime, scheduled_hour: int, scheduled_minute: int, status: dict[str, object]) -> bool:
    last_success_date = str(status.get("last_success_date") or "").strip()
    bootstrap_skip_date = str(status.get("bootstrap_skip_until_date") or "").strip()
    policy_skip_date = str(status.get("last_policy_skip_date") or "").strip()
    today = now.strftime("%Y-%m-%d")
    if last_success_date == today:
        return False
    if bootstrap_skip_date == today:
        return False
    if policy_skip_date == today:
        return False
    return (now.hour, now.minute) >= (scheduled_hour, scheduled_minute)


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


def _run_step(name: str, command: list[str]) -> None:
    logging.info("START %s", name)
    subprocess.run(command, cwd=ROOT, check=True)
    logging.info("OK %s", name)


def run_daily_cycle(now: datetime, tz_name: str, status: dict[str, object]) -> dict[str, object]:
    run_steps = _resolve_run_steps()
    started_at = now.isoformat()
    payload = {
        **status,
        "timezone": tz_name,
        "scheduler_mode": _scheduler_mode(),
        "status": "running",
        "last_attempt_at": started_at,
        "last_error": "",
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
                "last_completed_step": run_steps[-1][0],
                "last_error": "",
            }
        )
        logging.info("DONE daily cycle completed")
    except subprocess.CalledProcessError as exc:
        finished_at = _now(ZoneInfo(tz_name)).isoformat()
        payload.update(
            {
                "status": "error",
                "last_failure_at": finished_at,
                "last_completed_step": "",
                "last_error": f"{exc.cmd} (exit={exc.returncode})",
            }
        )
        logging.exception("Daily cycle failed")
    _write_status(payload)
    return payload


def main() -> int:
    setup_logging()

    tz_name = os.environ.get("SCHEDULER_TIMEZONE", DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    tz = ZoneInfo(tz_name)
    scheduled_hour, scheduled_minute = _parse_daily_time(os.environ.get("SCHEDULER_DAILY_TIME", DEFAULT_TIME))
    poll_seconds = int(os.environ.get("SCHEDULER_POLL_SECONDS", str(DEFAULT_POLL_SECONDS)))
    run_policy = os.environ.get("SCHEDULER_RUN_POLICY", DEFAULT_RUN_POLICY).strip() or DEFAULT_RUN_POLICY
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    skip_catchup = str(os.environ.get("SCHEDULER_SKIP_CATCHUP_ON_START", "1" if DEFAULT_SKIP_CATCHUP else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    logging.info(
        "Scheduler started timezone=%s daily_time=%02d:%02d poll_seconds=%d skip_catchup=%s run_policy=%s command_set=%s status_path=%s",
        tz_name,
        scheduled_hour,
        scheduled_minute,
        poll_seconds,
        skip_catchup,
        run_policy,
        command_set,
        _status_path(),
    )

    status = _load_status()
    status.setdefault("timezone", tz_name)
    status.setdefault("scheduler_mode", _scheduler_mode())
    status.setdefault("status", "idle")
    status["configured_daily_time"] = f"{scheduled_hour:02d}:{scheduled_minute:02d}"
    status["skip_catchup_on_start"] = skip_catchup
    status["run_policy"] = run_policy
    status["command_set"] = command_set
    status["status_path"] = str(_status_path())
    status["log_path"] = str(_log_path())
    status.setdefault("bootstrap_skip_until_date", "")
    if (
        skip_catchup
        and not str(status.get("last_success_date") or "").strip()
        and not str(status.get("bootstrap_skip_until_date") or "").strip()
    ):
        now = _now(tz)
        if (now.hour, now.minute) >= (scheduled_hour, scheduled_minute):
            status["bootstrap_skip_until_date"] = now.strftime("%Y-%m-%d")
            status["status_note"] = "Started after scheduled time; first catch-up run skipped until next day."
        else:
            status["status_note"] = "Waiting for first scheduled run."
    _write_status(status)

    while True:
        now = _now(tz)
        if _should_run_today(now, scheduled_hour, scheduled_minute, status):
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
    raise SystemExit(main())
