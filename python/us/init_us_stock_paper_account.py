from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_paper_trading_config
from python.us.us_db import (
    ensure_us_paper_trading_tables,
    fetch_us_paper_account_rows,
    reset_us_paper_account,
    upsert_us_paper_account_rows,
)


LOGGER = logging.getLogger("us_paper_account_init")
REAL_ORDER_BLOCKED = True


def assert_paper_only() -> None:
    if not REAL_ORDER_BLOCKED:
        raise RuntimeError("Paper trading must not call real order APIs.")


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a US stock paper-trading account.")
    parser.add_argument("--account-id", required=True, help="Paper account ID.")
    parser.add_argument("--initial-cash", type=float, default=None, help="Override initial cash.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without DB writes.")
    parser.add_argument("--reset", action="store_true", help="Reset existing paper account data before initialization.")
    return parser.parse_args()


def build_account_row(*, account_id: str, initial_cash: float | None) -> dict[str, object]:
    cfg = load_us_paper_trading_config(account_id=account_id)
    cash = float(initial_cash if initial_cash is not None else cfg.initial_cash)
    return {
        "account_id": account_id,
        "account_name": cfg.account_name,
        "base_currency": cfg.base_currency,
        "initial_cash": round(cash, 6),
        "cash_balance": round(cash, 6),
        "reserved_cash": 0.0,
        "market_value": 0.0,
        "equity_value": round(cash, 6),
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "status": "ACTIVE",
    }


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    setup_logging(cfg.log_level)
    assert_paper_only()

    row = build_account_row(account_id=args.account_id, initial_cash=args.initial_cash)
    existing = fetch_us_paper_account_rows(account_id=args.account_id)

    LOGGER.info("[US_PAPER_INIT] account_id=%s dry_run=%s reset=%s", args.account_id, args.dry_run, args.reset)
    LOGGER.info("[US_PAPER_INIT] initial_cash=%.2f base_currency=%s config=%s", row["initial_cash"], row["base_currency"], cfg.config_path)

    if existing and not args.reset:
        LOGGER.info("[US_PAPER_INIT] Existing account found. Initialization skipped because --reset was not provided.")
        return 0

    if args.reset:
        LOGGER.warning("[US_PAPER_INIT] --reset requested. This clears only paper.us_stock_* rows for account_id=%s.", args.account_id)

    if args.dry_run:
        LOGGER.info("[US_PAPER_INIT] Dry-run complete. No DB writes were performed.")
        return 0

    ensure_us_paper_trading_tables()
    if args.reset and existing:
        reset_us_paper_account(args.account_id)
    upsert_us_paper_account_rows([row])
    LOGGER.info("[US_PAPER_INIT] Paper account initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
