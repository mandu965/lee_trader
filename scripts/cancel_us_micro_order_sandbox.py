from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.us.cancel_us_micro_order_sandbox import main


if __name__ == "__main__":
    raise SystemExit(main())
