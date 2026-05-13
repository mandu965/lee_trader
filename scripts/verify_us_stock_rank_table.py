from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.us.verify_us_stock_rank_table import main


if __name__ == "__main__":
    main()
