from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python.us.plan_us_stock_paper_rebalance import main


if __name__ == "__main__":
    raise SystemExit(main())
