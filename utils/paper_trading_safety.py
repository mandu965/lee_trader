from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python.us.us_config import load_us_paper_trading_config


def assert_paper_trading_only(*, account_id: str | None = None, message: str | None = None) -> None:
    cfg = load_us_paper_trading_config(account_id=account_id)
    if not cfg.real_order_blocked:
        raise RuntimeError("US_PAPER_REAL_ORDER_BLOCKED must be true for paper trading scripts.")
    if message:
        print(message)
