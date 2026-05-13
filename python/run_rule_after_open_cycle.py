from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
_RESULTS_PATH = ROOT / "outputs" / "rule_execution_results.json"


def _should_retry_gap_blocked() -> bool:
    """Return True when before_open was blocked solely due to gap unavailability.

    Conditions:
    - market_data_available is False (open prices were not available)
    - At least one BUY item is blocked by actual_open_gap_unavailable
    - No BUY items were actually submitted (prevents duplicate orders)
    """
    try:
        data = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("market_data_available"):
        return False
    items = data.get("items") or []
    buy_items = [item for item in items if str(item.get("side") or "").upper() == "BUY"]
    if not buy_items:
        return False
    any_submitted = any(
        str(item.get("order_status") or "") in {"submitted", "filled", "partial_fill"}
        for item in buy_items
    )
    if any_submitted:
        return False
    gap_blocked_count = sum(
        1 for item in buy_items
        if "actual_open_gap_unavailable" in str(item.get("order_block_reason") or "")
    )
    return gap_blocked_count > 0


def main() -> int:
    run_mode = str(os.environ.get("RULE_TRADING_RUN_MODE", "paper")).strip().lower() or "paper"
    if run_mode == "paper":
        print("[SKIP] rule after-open cycle does not rerun paper execution")
        return 0

    if _should_retry_gap_blocked():
        print("[INFO] before-open blocked by actual_open_gap_unavailable — retrying with fresh market snapshot")
        subprocess.run([PYTHON, str(ROOT / "python" / "rule_order_submitter.py")], cwd=ROOT, check=True)
        print("[DONE] gap-retry order submission completed")

    target = "rule_order_fill_sync.py"
    subprocess.run([PYTHON, str(ROOT / "python" / target)], cwd=ROOT, check=True)
    print(f"[DONE] rule after-open cycle completed run_mode={run_mode} target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
