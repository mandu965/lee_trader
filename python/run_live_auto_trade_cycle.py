from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OPS_REFRESH_SCRIPT = ROOT / "python" / "run_operational_refresh.py"
SUBMIT_LIVE_ORDERS_SCRIPT = ROOT / "python" / "submit_live_orders.py"
SYNC_WEB_DISPLAY_SCRIPT = ROOT / "python" / "sync_web_display_data.py"
SYNC_LIVE_ACCOUNT_HOLDINGS_SCRIPT = ROOT / "python" / "sync_live_account_holdings.py"
SYNC_LIVE_ORDER_FILLS_SCRIPT = ROOT / "python" / "sync_live_order_fills.py"


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
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[OK] {name}")


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
    _run_step("sync_live_order_fills", [PYTHON, str(SYNC_LIVE_ORDER_FILLS_SCRIPT)])
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
    print(
        f"[DONE] live auto trade cycle completed execute={'Y' if execute else 'N'} "
        f"allow_buy={'Y' if allow_buy else 'N'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
