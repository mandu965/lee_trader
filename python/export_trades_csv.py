from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRADES_CSV = DATA_DIR / "trades.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public.trades into data/trades.csv.")
    parser.add_argument("--output", type=Path, default=TRADES_CSV)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    args = parse_args()
    setup_logging()
    engine = get_engine()
    query = text(
        """
        SELECT trade_id, date, side, code, name, market, sector, qty, price, amount, fee, memo, created_at
        FROM trades
        ORDER BY date ASC, trade_id ASC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    logging.info("Exported trades rows=%d output=%s", len(df), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
