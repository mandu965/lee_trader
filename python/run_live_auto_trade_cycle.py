from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from notifier import notify_critical


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OPS_REFRESH_SCRIPT = ROOT / "python" / "run_operational_refresh.py"
SUBMIT_LIVE_ORDERS_SCRIPT = ROOT / "python" / "submit_live_orders.py"
SYNC_WEB_DISPLAY_SCRIPT = ROOT / "python" / "sync_web_display_data.py"
SYNC_LIVE_ACCOUNT_HOLDINGS_SCRIPT = ROOT / "python" / "sync_live_account_holdings.py"
SYNC_LIVE_ORDER_FILLS_SCRIPT = ROOT / "python" / "sync_live_order_fills.py"
BUILD_LIVE_TRADE_CONSISTENCY_SCRIPT = ROOT / "python" / "build_live_trade_consistency_report.py"
BUILD_LIVE_TRADE_REVIEW_SCRIPT = ROOT / "python" / "build_live_trade_review.py"
BUILD_LIVE_TRADE_REVIEW_SUMMARY_SCRIPT = ROOT / "python" / "build_live_trade_review_summary.py"
BUILD_LIVE_KPI_DAILY_REPORT_SCRIPT = ROOT / "python" / "build_live_kpi_daily_report.py"
BUILD_QUALITY_RISK_GUARD_LIVE_REVIEW_SCRIPT = ROOT / "python" / "build_quality_risk_guard_live_review.py"
BUILD_LIVE_CLOSED_TRADE_REPORT_SCRIPT = ROOT / "python" / "build_live_closed_trade_report.py"
CHECK_LIVE_QUALITY_GUARD_OUTPUTS_SCRIPT = ROOT / "python" / "check_live_quality_guard_outputs.py"
UPDATE_AI_POSITION_STATE_SCRIPT = ROOT / "python" / "update_ai_position_state.py"
OPTIONAL_POST_SYNC_STEPS = {
    "build_live_kpi_daily_report",
    "build_live_closed_trade_report",
    "build_quality_risk_guard_live_review",
    "check_live_quality_guard_outputs",
}


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a guarded live auto-trade cycle from refresh to submit.")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip run_operational_refresh.py and only build/submit orders.")
    parser.add_argument("--execute", action="store_true", help="Actually submit guarded orders.")
    parser.add_argument("--allow-buy", action="store_true", help="Allow BUY submissions during execute.")
    parser.add_argument("--force-resubmit", action="store_true", help="Ignore previous successful request ids.")
    return parser.parse_args()


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def _run_step(name: str, command: list[str]) -> None:
    print(f"[START] {name}")
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
    except subprocess.CalledProcessError as exc:
        if str(exc.output or "").strip():
            print(str(exc.output).rstrip(), file=sys.stderr)
        if str(exc.stderr or "").strip():
            print(str(exc.stderr).rstrip(), file=sys.stderr)
        if name in {"submit_live_orders", "sync_live_order_fills"}:
            _notify_step_failure(name, command, exc)
        raise
    print(f"[OK] {name}")


def _notify_step_failure(name: str, command: list[str], exc: subprocess.CalledProcessError) -> None:
    try:
        message = (
            "Live order submission step failed. Check KIS order submission path and execution artifacts."
            if name == "submit_live_orders"
            else "Live order fill synchronization failed. Check KIS fill inquiry and DB sync path."
        )
        notify_critical(
            f"Lee Trader live auto-trade failure: {name}",
            message,
            {
                "step": name,
                "returncode": exc.returncode,
                "command": " ".join(str(part) for part in command),
            },
        )
    except Exception:
        pass


def _refresh_command() -> list[str]:
    command = [PYTHON, str(OPS_REFRESH_SCRIPT), "--with-live-account", "--skip-live-preview"]
    if _env_flag("AUTO_TRADE_SKIP_THEME_SHADOW", True):
        command.append("--skip-theme-shadow")
    if _env_flag("AUTO_TRADE_SKIP_PAPER_TRADING", True):
        command.append("--skip-paper-trading")
    if _env_flag("AUTO_TRADE_SKIP_PAPER_TRADING_DB", True):
        command.append("--skip-paper-trading-db")
    return command


def _submit_command(*, execute: bool, allow_buy: bool, force_resubmit: bool) -> list[str]:
    command = [PYTHON, str(SUBMIT_LIVE_ORDERS_SCRIPT)]
    if not execute:
        return command

    confirm_text = str(os.environ.get("AUTO_TRADE_CONFIRM_TEXT", "")).strip()
    if confirm_text != "LIVE_ORDER":
        raise ValueError("AUTO_TRADE_CONFIRM_TEXT must be LIVE_ORDER when execute is enabled")

    command.extend(["--execute", "--confirm-text", confirm_text])
    if allow_buy:
        command.append("--allow-buy")
    if force_resubmit:
        command.append("--force-resubmit")
    return command


def main() -> int:
    _load_env()
    args = parse_args()
    execute = bool(args.execute or _env_flag("AUTO_TRADE_EXECUTE", False))
    allow_buy = bool(args.allow_buy or _env_flag("AUTO_TRADE_ALLOW_BUY", False))
    force_resubmit = bool(args.force_resubmit or _env_flag("AUTO_TRADE_FORCE_RESUBMIT", False))

    if not args.skip_refresh:
        _run_step("run_operational_refresh", _refresh_command())

    _run_step(
        "submit_live_orders",
        _submit_command(execute=execute, allow_buy=allow_buy, force_resubmit=force_resubmit),
    )

    _run_step("sync_live_account_holdings", [PYTHON, str(SYNC_LIVE_ACCOUNT_HOLDINGS_SCRIPT)])
    _run_step("update_ai_position_state", [PYTHON, str(UPDATE_AI_POSITION_STATE_SCRIPT)])

    _run_step("sync_live_order_fills", [PYTHON, str(SYNC_LIVE_ORDER_FILLS_SCRIPT)])
    _run_step("build_live_trade_consistency_report", [PYTHON, str(BUILD_LIVE_TRADE_CONSISTENCY_SCRIPT)])
    _run_step("build_live_trade_review", [PYTHON, str(BUILD_LIVE_TRADE_REVIEW_SCRIPT)])
    _run_step("build_live_trade_review_summary", [PYTHON, str(BUILD_LIVE_TRADE_REVIEW_SUMMARY_SCRIPT)])
    warnings: list[dict[str, str]] = []
    for name, command in [
        ("build_live_kpi_daily_report", [PYTHON, str(BUILD_LIVE_KPI_DAILY_REPORT_SCRIPT)]),
        ("build_live_closed_trade_report", [PYTHON, str(BUILD_LIVE_CLOSED_TRADE_REPORT_SCRIPT)]),
        ("build_quality_risk_guard_live_review", [PYTHON, str(BUILD_QUALITY_RISK_GUARD_LIVE_REVIEW_SCRIPT)]),
        ("check_live_quality_guard_outputs", [PYTHON, str(CHECK_LIVE_QUALITY_GUARD_OUTPUTS_SCRIPT)]),
    ]:
        try:
            _run_step(name, command)
        except subprocess.CalledProcessError as exc:
            if name not in OPTIONAL_POST_SYNC_STEPS:
                raise
            warning = {
                "step_name": name,
                "script_name": Path(str(command[-1])).name,
                "exit_code": str(exc.returncode),
                "error_message": (str(exc.stderr or "").strip().splitlines() or [f"{name} failed"])[-1],
            }
            warnings.append(warning)
            print(f"[WARN] {json.dumps(warning, ensure_ascii=False)}")
    if str(os.environ.get("WEB_DATABASE_URL", "")).strip():
        _run_step(
            "sync_web_display_data",
            [
                PYTHON,
                str(SYNC_WEB_DISPLAY_SCRIPT),
                "--skip-core",
                "--skip-paper-trading",
                "--skip-trades",
            ],
        )
    if warnings:
        print(f"[DONE] live auto trade cycle completed with post-sync warnings={len(warnings)}")
        return 0
    print(
        f"[DONE] live auto trade cycle completed execute={'Y' if execute else 'N'} "
        f"allow_buy={'Y' if allow_buy else 'N'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
