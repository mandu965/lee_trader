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
DEFAULT_TIME = "18:10"
DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEZONE = "Asia/Seoul"
DEFAULT_SKIP_CATCHUP = True
DEFAULT_RUN_POLICY = "always"
DEFAULT_PRIMARY_MAX_AGE_HOURS = 20
DEFAULT_SCHEDULER_MODE = "internal_service"
DEFAULT_INTERVAL_MINUTES = 0


def _should_sync_web_display() -> bool:
    return str(os.environ.get("SCHEDULER_SYNC_WEB_DISPLAY", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _close_batch_command() -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "python" / "run_manual_close_batch.py"),
        "--skip-build",
        "--skip-node-api",
    ]
    if not _should_sync_web_display():
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
            ("build_live_trade_consistency_report", _live_trade_consistency_command()),
            ("build_live_trade_review", _live_trade_review_command()),
            ("build_live_trade_review_summary", _live_trade_review_summary_command()),
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


def _run_step(name: str, command: list[str]) -> None:
    logging.info("START %s", name)
    subprocess.run(command, cwd=ROOT, check=True)
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
                "last_completed_step": run_steps[-1][0],
                "last_error": "",
            }
        )
        _write_status(payload)
        for name, command in post_sync_steps:
            _run_step(name, command)
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
        if _skip_after_failure_until_next_day():
            payload["last_failure_skip_date"] = finished_at[:10]
            payload["status_note"] = "Last run failed; skipping further runs until next day."
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
    command_set = str(os.environ.get("SCHEDULER_COMMAND_SET", "close")).strip().lower() or "close"
    skip_catchup = str(os.environ.get("SCHEDULER_SKIP_CATCHUP_ON_START", "1" if DEFAULT_SKIP_CATCHUP else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    logging.info(
        "Scheduler started timezone=%s daily_time=%02d:%02d daily_times=%s interval_minutes=%d poll_seconds=%d skip_catchup=%s run_policy=%s command_set=%s status_path=%s",
        tz_name,
        scheduled_hour,
        scheduled_minute,
        ",".join(f"{hour:02d}:{minute:02d}" for hour, minute in scheduled_times) or "-",
        interval_minutes,
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
    configured_daily_time = (
        ",".join(f"{hour:02d}:{minute:02d}" for hour, minute in scheduled_times)
        if scheduled_times
        else f"{scheduled_hour:02d}:{scheduled_minute:02d}"
    )
    status["configured_daily_time"] = configured_daily_time
    status["configured_interval_minutes"] = interval_minutes
    status["skip_catchup_on_start"] = skip_catchup
    status["run_policy"] = run_policy
    status["command_set"] = command_set
    status["status_path"] = str(_status_path())
    status["log_path"] = str(_log_path())
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
    raise SystemExit(main())
