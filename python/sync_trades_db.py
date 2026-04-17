from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine, replace_table_rows_pg


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRADES_CSV = DATA_DIR / "trades.csv"

TRADE_COLUMNS = [
    "trade_id",
    "date",
    "side",
    "code",
    "name",
    "market",
    "sector",
    "qty",
    "price",
    "amount",
    "fee",
    "memo",
    "created_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync trades.csv into public.trades for web display.")
    parser.add_argument("--csv", type=Path, default=TRADES_CSV)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_tables() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id    BIGINT PRIMARY KEY,
                    date        DATE NOT NULL,
                    side        TEXT NOT NULL,
                    code        TEXT NOT NULL,
                    name        TEXT,
                    market      TEXT,
                    sector      TEXT,
                    qty         NUMERIC,
                    price       NUMERIC,
                    amount      NUMERIC,
                    fee         NUMERIC,
                    memo        TEXT,
                    created_at  TIMESTAMPTZ
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_code_date ON trades(code, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)"))


def read_csv(path: Path) -> pd.DataFrame:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    out = df.copy()
    for col in TRADE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["trade_id"] = pd.to_numeric(out["trade_id"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["side"] = out["side"].astype(str).str.strip().str.upper()
    out["code"] = out["code"].astype(str).str.zfill(6)
    for col in ["qty", "price", "amount", "fee"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["name", "market", "sector", "memo"]:
        out[col] = out[col].astype("string")
    out = out.dropna(subset=["trade_id", "date", "side", "code"]).drop_duplicates(subset=["trade_id"], keep="last")
    return out.loc[:, TRADE_COLUMNS].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    setup_logging()
    ensure_tables()
    df = normalize(read_csv(args.csv))
    replace_table_rows_pg("trades", df, columns=TRADE_COLUMNS)
    logging.info("Synced trades rows=%d csv=%s", len(df), args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
