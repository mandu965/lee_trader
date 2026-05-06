from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(name: str, command: list[str]) -> None:
    print(f"[START] {name}")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[OK] {name}")


def main() -> int:
    run_mode = str(os.environ.get("RULE_TRADING_RUN_MODE", "paper")).strip().lower() or "paper"
    state_json = (
        ROOT / "outputs" / "rule_account_paper_state.json"
        if run_mode == "paper"
        else ROOT / "outputs" / "rule_account_live_state.json"
    )
    _run("rule_signal_builder", [PYTHON, str(ROOT / "python" / "rule_signal_builder.py")])
    _run("rule_backtest", [PYTHON, str(ROOT / "python" / "rule_backtest.py")])
    _run(
        "rule_portfolio_manager",
        [
            PYTHON,
            str(ROOT / "python" / "rule_portfolio_manager.py"),
            "--run-mode",
            run_mode,
            "--state-json",
            str(state_json),
        ],
    )
    _run(
        "rule_order_preview_builder",
        [PYTHON, str(ROOT / "python" / "rule_order_preview_builder.py"), "--run-mode", run_mode],
    )
    _run("rule_daily_report", [PYTHON, str(ROOT / "python" / "rule_daily_report.py")])
    if str(os.environ.get("RULE_BLOCK_REASON_REPORT_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        _run("rule_block_reason_report", [PYTHON, str(ROOT / "python" / "rule_block_reason_report.py")])
    print(f"[DONE] rule after-close cycle completed run_mode={run_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
