from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    run_mode = str(os.environ.get("RULE_TRADING_RUN_MODE", "paper")).strip().lower() or "paper"
    if run_mode == "paper":
        print("[SKIP] rule after-open cycle does not rerun paper execution")
        return 0
    target = "rule_order_fill_sync.py"
    subprocess.run([PYTHON, str(ROOT / "python" / target)], cwd=ROOT, check=True)
    print(f"[DONE] rule after-open cycle completed run_mode={run_mode} target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
