from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.us.trade_orchestration.run_us_trade_orchestration import main


if __name__ == "__main__":
    raise SystemExit(main())
