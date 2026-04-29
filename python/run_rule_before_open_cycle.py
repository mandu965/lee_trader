from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    run_mode = str(os.environ.get("RULE_TRADING_RUN_MODE", "paper")).strip().lower() or "paper"
    target = "rule_execution_simulator.py" if run_mode == "paper" else "rule_order_submitter.py"
    subprocess.run([PYTHON, str(ROOT / "python" / target)], cwd=ROOT, check=True)
    print(f"[DONE] rule before-open cycle completed run_mode={run_mode} target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
