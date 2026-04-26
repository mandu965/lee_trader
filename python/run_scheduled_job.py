from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_daily_scheduler import _load_status, _write_status, run_daily_cycle, setup_logging


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEZONE = "Asia/Seoul"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single scheduled Lee Trader job and update scheduler status payloads.")
    parser.add_argument("--command-set", choices=["close", "intraday"], required=True)
    parser.add_argument("--daily-time", default=None, help="Display-only scheduled clock time recorded in status payloads.")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--trigger-source", default="github_actions")
    return parser.parse_args()


def _resolve_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path
    return path if path.is_absolute() else ROOT / path


def _default_daily_time(command_set: str) -> str:
    return "12:00" if command_set == "intraday" else "18:10"


def _default_status_path(command_set: str) -> Path:
    if command_set == "intraday":
        return ROOT / "outputs" / "auto_ops_recovery_scheduler_status.json"
    return ROOT / "outputs" / "auto_ops_scheduler_status.json"


def _default_log_path(command_set: str) -> Path:
    if command_set == "intraday":
        return ROOT / "logs" / "auto_ops_recovery_scheduler.log"
    return ROOT / "logs" / "auto_ops_scheduler.log"


def main() -> int:
    args = parse_args()
    daily_time = (args.daily_time or _default_daily_time(args.command_set)).strip()
    status_path = _resolve_path(args.status_path, _default_status_path(args.command_set))
    log_path = _resolve_path(args.log_path, _default_log_path(args.command_set))

    os.environ["SCHEDULER_COMMAND_SET"] = args.command_set
    os.environ["SCHEDULER_DAILY_TIME"] = daily_time
    os.environ["SCHEDULER_TIMEZONE"] = args.timezone
    os.environ["SCHEDULER_STATUS_PATH"] = str(status_path)
    os.environ["SCHEDULER_LOG_PATH"] = str(log_path)
    os.environ["SCHEDULER_MODE"] = args.trigger_source

    setup_logging()
    tz = ZoneInfo(args.timezone)
    now = datetime.now(tz)

    status = _load_status()
    status.update(
        {
            "timezone": args.timezone,
            "scheduler_mode": args.trigger_source,
            "status": "idle",
            "configured_daily_time": daily_time,
            "skip_catchup_on_start": False,
            "run_policy": "external_schedule",
            "command_set": args.command_set,
            "status_path": str(status_path),
            "log_path": str(log_path),
            "status_note": f"Triggered by {args.trigger_source}.",
            "bootstrap_skip_until_date": "",
            "last_policy_reason": "",
            "last_policy_skip_date": "",
            "last_policy_check_at": now.isoformat(),
        }
    )
    _write_status(status)
    run_daily_cycle(now, args.timezone, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
