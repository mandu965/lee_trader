"""
run_trading_pipeline.py

Operational pipeline runner for the local trading workflow.
This script orchestrates existing research/paper-monitoring steps only:

1. backtest result check
2. forward test run
3. daily report generation
4. health check run

It does not call any broker API and does not modify strategy logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pipeline"
OUTPUTS_DIR = ROOT / "outputs"


@dataclass
class StepResult:
    """One pipeline step result."""

    name: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the trading operations pipeline.")
    parser.add_argument("--mode", choices=["paper", "live"], required=True)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--skip-backtest-check", action="store_true")
    return parser.parse_args()


def _append_log(path: Path, text: str) -> None:
    """Append text to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip())
        handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _run_subprocess(command: list[str], log_path: Path, error_log_path: Path, step_name: str) -> StepResult:
    """Run one subprocess and write stdout/stderr to logs."""
    with log_path.open("w", encoding="utf-8") as out_handle, error_log_path.open("a", encoding="utf-8") as err_handle:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=out_handle,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.stderr:
            err_handle.write(f"[{step_name}] exit_code={result.returncode}\n")
            err_handle.write(result.stderr)
            if not result.stderr.endswith("\n"):
                err_handle.write("\n")
    if result.returncode == 0:
        return StepResult(name=step_name, status="success", message="completed")
    return StepResult(name=step_name, status="failed", message=f"exit_code={result.returncode}")


def run_backtest_check(skip_check: bool, pipeline_log: Path, error_log: Path) -> StepResult:
    """Validate that baseline backtest outputs exist and log key metrics."""
    if skip_check:
        _append_log(pipeline_log, "[backtest_check] skipped by CLI flag")
        return StepResult(name="backtest_check", status="skipped", message="skip-backtest-check enabled")

    summary_path = OUTPUTS_DIR / "backtest" / "backtest_summary.json"
    if not summary_path.exists():
        message = f"missing baseline backtest summary: {summary_path}"
        _append_log(error_log, f"[backtest_check] {message}")
        return StepResult(name="backtest_check", status="failed", message=message)

    summary = _load_json(summary_path)
    performance = summary.get("performance", {})
    total_return = performance.get("total_return")
    cagr = performance.get("cagr")
    mdd = performance.get("mdd")
    _append_log(
        pipeline_log,
        (
            "[backtest_check] "
            f"total_return={total_return} "
            f"CAGR={cagr} "
            f"MDD={mdd}"
        ),
    )
    return StepResult(name="backtest_check", status="success", message="baseline summary loaded")


def read_forward_summary() -> dict[str, Any] | None:
    """Read forward summary if present."""
    path = OUTPUTS_DIR / "forward_test" / "forward_summary.json"
    if not path.exists():
        return None
    return _load_json(path)


def read_daily_report(report_date: pd.Timestamp) -> dict[str, Any] | None:
    """Read daily report JSON if present."""
    path = OUTPUTS_DIR / "daily_reports" / f"daily_report_{report_date.strftime('%Y-%m-%d')}.json"
    if not path.exists():
        return None
    return _load_json(path)


def read_health_check(check_date: pd.Timestamp) -> dict[str, Any] | None:
    """Read health check JSON if present."""
    path = OUTPUTS_DIR / "health_checks" / f"health_check_{check_date.strftime('%Y-%m-%d')}.json"
    if not path.exists():
        return None
    return _load_json(path)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_log = LOG_DIR / "pipeline.log"
    error_log = LOG_DIR / "error.log"
    forward_log = LOG_DIR / "forward_test.log"
    report_log = LOG_DIR / "report.log"
    health_log = LOG_DIR / "health.log"

    start_date = pd.Timestamp(args.start_date).normalize()
    today = pd.Timestamp.today().normalize()
    step_results: list[StepResult] = []

    _append_log(
        pipeline_log,
        f"[pipeline] start mode={args.mode} start_date={start_date.strftime('%Y-%m-%d')} checked_at={today.strftime('%Y-%m-%d')}",
    )

    step_results.append(run_backtest_check(args.skip_backtest_check, pipeline_log, error_log))

    forward_cmd = [
        sys.executable,
        "python/run_forward_test.py",
        "--start-date",
        start_date.strftime("%Y-%m-%d"),
        "--initial-cash",
        "10000000",
        "--output-dir",
        "outputs/forward_test",
        "--mode",
        "daily",
    ]
    forward_result = _run_subprocess(forward_cmd, forward_log, error_log, "forward_test")
    step_results.append(forward_result)
    _append_log(pipeline_log, f"[forward_test] status={forward_result.status} message={forward_result.message}")

    report_cmd = [
        sys.executable,
        "python/generate_trading_daily_report.py",
        "--date",
        today.strftime("%Y-%m-%d"),
        "--mode",
        args.mode,
        "--input-dir",
        "outputs",
        "--output-dir",
        "outputs/daily_reports",
    ]
    report_result = _run_subprocess(report_cmd, report_log, error_log, "daily_report")
    step_results.append(report_result)
    _append_log(pipeline_log, f"[daily_report] status={report_result.status} message={report_result.message}")

    health_cmd = [
        sys.executable,
        "python/monitor_trading_health.py",
        "--input-dir",
        "outputs",
        "--output-dir",
        "outputs/health_checks",
    ]
    health_result = _run_subprocess(health_cmd, health_log, error_log, "health_check")
    step_results.append(health_result)
    _append_log(pipeline_log, f"[health_check] status={health_result.status} message={health_result.message}")

    forward_summary = read_forward_summary()
    daily_report = read_daily_report(today)
    health_payload = read_health_check(today)

    forward_status = forward_result.status
    daily_return = None
    total_value = None
    if daily_report and isinstance(daily_report, dict):
        summary = daily_report.get("summary", {})
        daily_return = summary.get("daily_return")
        total_value = summary.get("total_value")
    elif forward_summary and isinstance(forward_summary, dict):
        profiles = forward_summary.get("profiles", {})
        if isinstance(profiles, dict) and profiles:
            total_value = sum(float(item.get("current_value", 0.0)) for item in profiles.values())

    health_status = health_payload.get("status") if isinstance(health_payload, dict) else "missing"

    print("[PIPELINE RESULT]")
    print(f"- forward test status: {forward_status}")
    print(f"- daily return: {daily_return}")
    print(f"- total value: {total_value}")
    print(f"- health status: {health_status}")

    _append_log(
        pipeline_log,
        (
            "[pipeline_result] "
            f"forward_status={forward_status} "
            f"daily_return={daily_return} "
            f"total_value={total_value} "
            f"health_status={health_status}"
        ),
    )


if __name__ == "__main__":
    main()
