from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python.us.run_us_live_pre_trade_check import main


if __name__ == "__main__":
    raise SystemExit(main())
