from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.sell_automation.run_us_sell_automation import main


if __name__ == "__main__":
    raise SystemExit(main())
