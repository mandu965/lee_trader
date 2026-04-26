from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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

AUDIT_COLUMNS = [
    "audit_id",
    "trade_id",
    "action",
    "trade_snapshot",
    "actor",
    "reason",
    "created_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync web public.trades into the local DB.")
    parser.add_argument("--source-url", default="", help="Source web DB URL. Defaults to WEB_DATABASE_URL.")
    parser.add_argument(
        "--target-url",
        default="",
        help="Target local DB URL. Defaults to LOCAL_DATABASE_URL, then DATABASE_URL.",
    )
    parser.add_argument("--skip-audit-log", action="store_true", help="Do not sync public.trade_audit_log.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not refresh data/trades.csv.")
    parser.add_argument("--csv", type=Path, default=TRADES_CSV, help="CSV mirror path for synced trades.")
    parser.add_argument(
        "--allow-empty-source",
        action="store_true",
        help="Allow replacing local trades with an empty web source.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def resolve_urls(args: argparse.Namespace) -> tuple[str, str]:
    load_dotenv(ROOT / ".env", override=False)
    source_url = str(args.source_url or os.environ.get("WEB_DATABASE_URL", "")).strip()
    target_url = str(
        args.target_url
        or os.environ.get("LOCAL_DATABASE_URL", "")
        or os.environ.get("DATABASE_URL", "")
    ).strip()
    if not source_url:
        raise RuntimeError("WEB_DATABASE_URL is required for web -> local trades sync.")
    if not target_url:
        raise RuntimeError("LOCAL_DATABASE_URL or DATABASE_URL is required for local target sync.")
    if source_url == target_url:
        raise RuntimeError("Source and target DB URLs are identical; refusing to sync.")
    return source_url, target_url


def table_exists(conn, qualified_name: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": qualified_name}).scalar())


def ensure_local_tables(target_engine, include_audit_log: bool) -> None:
    with target_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.trades (
                    trade_id    BIGSERIAL PRIMARY KEY,
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
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_code_date ON public.trades(code, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_date ON public.trades(date)"))
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF pg_get_serial_sequence('public.trades', 'trade_id') IS NULL THEN
                        CREATE SEQUENCE IF NOT EXISTS public.trades_trade_id_seq;
                        ALTER TABLE public.trades
                            ALTER COLUMN trade_id SET DEFAULT nextval('public.trades_trade_id_seq');
                        ALTER SEQUENCE public.trades_trade_id_seq OWNED BY public.trades.trade_id;
                    END IF;
                END $$;
                """
            )
        )
        if include_audit_log:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS public.trade_audit_log (
                        audit_id       BIGSERIAL PRIMARY KEY,
                        trade_id       BIGINT,
                        action         TEXT NOT NULL,
                        trade_snapshot JSONB,
                        actor          TEXT,
                        reason         TEXT,
                        created_at     TIMESTAMPTZ DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_trade_audit_log_trade_id
                    ON public.trade_audit_log(trade_id, created_at DESC)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF pg_get_serial_sequence('public.trade_audit_log', 'audit_id') IS NULL THEN
                            CREATE SEQUENCE IF NOT EXISTS public.trade_audit_log_audit_id_seq;
                            ALTER TABLE public.trade_audit_log
                                ALTER COLUMN audit_id SET DEFAULT nextval('public.trade_audit_log_audit_id_seq');
                            ALTER SEQUENCE public.trade_audit_log_audit_id_seq
                                OWNED BY public.trade_audit_log.audit_id;
                        END IF;
                    END $$;
                    """
                )
            )


def read_trades(source_engine) -> pd.DataFrame:
    query = text(
        """
        SELECT trade_id, date, side, code, name, market, sector, qty, price, amount, fee, memo, created_at
        FROM public.trades
        ORDER BY date ASC, trade_id ASC
        """
    )
    with source_engine.connect() as conn:
        if not table_exists(conn, "public.trades"):
            raise RuntimeError("Source table public.trades does not exist.")
        return pd.read_sql_query(query, conn)


def read_audit_log(source_engine) -> pd.DataFrame:
    query = text(
        """
        SELECT audit_id, trade_id, action, trade_snapshot, actor, reason, created_at
        FROM public.trade_audit_log
        ORDER BY audit_id ASC
        """
    )
    with source_engine.connect() as conn:
        if not table_exists(conn, "public.trade_audit_log"):
            logging.info("Skip trade audit log sync: source table not found")
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        return pd.read_sql_query(query, conn)


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    out = df.copy()
    for col in TRADE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["trade_id"] = pd.to_numeric(out["trade_id"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["side"] = out["side"].astype(str).str.strip().str.upper()
    out["code"] = out["code"].astype(str).str.zfill(6)
    for col in ["qty", "price", "amount", "fee"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out = out.dropna(subset=["trade_id", "date", "side", "code"]).drop_duplicates(subset=["trade_id"], keep="last")
    return out.loc[:, TRADE_COLUMNS].reset_index(drop=True)


def normalize_audit_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    out = df.copy()
    for col in AUDIT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["audit_id"] = pd.to_numeric(out["audit_id"], errors="coerce").astype("Int64")
    out["trade_id"] = pd.to_numeric(out["trade_id"], errors="coerce").astype("Int64")
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out = out.dropna(subset=["audit_id", "action"]).drop_duplicates(subset=["audit_id"], keep="last")
    return out.loc[:, AUDIT_COLUMNS].reset_index(drop=True)


def record_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.where(pd.notna(df), None).to_dict(orient="records"):
        normalized = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            normalized[key] = value
        records.append(normalized)
    return records


def audit_record_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = record_values(df)
    for row in records:
        snapshot = row.get("trade_snapshot")
        if snapshot is None:
            continue
        if isinstance(snapshot, str):
            row["trade_snapshot"] = snapshot
        else:
            row["trade_snapshot"] = json.dumps(snapshot, ensure_ascii=False, default=str)
    return records


def refresh_sequence(conn, table_name: str, id_column: str) -> None:
    conn.execute(
        text(
            """
            SELECT setval(seq_name, next_value, false)
            FROM (
                SELECT
                    pg_get_serial_sequence(:table_name, :id_column)::regclass AS seq_name,
                    GREATEST(COALESCE((SELECT MAX(%s) FROM %s), 0) + 1, 1) AS next_value
            ) s
            WHERE seq_name IS NOT NULL
            """
            % (id_column, table_name)
        ),
        {"table_name": table_name, "id_column": id_column},
    )


def replace_local_rows(target_engine, trades: pd.DataFrame, audit_log: pd.DataFrame | None) -> None:
    with target_engine.begin() as conn:
        if audit_log is not None:
            conn.execute(text("DELETE FROM public.trade_audit_log"))
        conn.execute(text("DELETE FROM public.trades"))
        trade_records = record_values(trades)
        if trade_records:
            conn.execute(
                text(
                    """
                    INSERT INTO public.trades (
                        trade_id, date, side, code, name, market, sector, qty, price, amount, fee, memo, created_at
                    ) VALUES (
                        :trade_id, :date, :side, :code, :name, :market, :sector, :qty, :price, :amount, :fee, :memo,
                        :created_at
                    )
                    """
                ),
                trade_records,
            )
        refresh_sequence(conn, "public.trades", "trade_id")
        if audit_log is not None:
            audit_records = audit_record_values(audit_log)
            if audit_records:
                conn.execute(
                    text(
                        """
                        INSERT INTO public.trade_audit_log (
                            audit_id, trade_id, action, trade_snapshot, actor, reason, created_at
                        ) VALUES (
                            :audit_id, :trade_id, :action, CAST(:trade_snapshot AS jsonb), :actor, :reason, :created_at
                        )
                        """
                    ),
                    audit_records,
                )
            refresh_sequence(conn, "public.trade_audit_log", "audit_id")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    output = path if path.is_absolute() else ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_df = df.copy()
    csv_df["date"] = pd.to_datetime(csv_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    csv_df["created_at"] = pd.to_datetime(csv_df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    csv_df.to_csv(output, index=False, encoding="utf-8-sig")
    logging.info("Refreshed trades CSV rows=%d output=%s", len(csv_df), output)


def main() -> int:
    args = parse_args()
    setup_logging()
    source_url, target_url = resolve_urls(args)
    source_engine = create_engine(source_url, future=True)
    target_engine = create_engine(target_url, future=True)
    try:
        trades = normalize_trades(read_trades(source_engine))
        if trades.empty and not args.allow_empty_source:
            raise RuntimeError("Source public.trades is empty; use --allow-empty-source to replace local rows.")
        audit_log = None if args.skip_audit_log else normalize_audit_log(read_audit_log(source_engine))
        ensure_local_tables(target_engine, include_audit_log=audit_log is not None)
        replace_local_rows(target_engine, trades, audit_log)
        if not args.skip_csv:
            write_csv(trades, args.csv)
        logging.info(
            "Synced web trades into local DB trades=%d audit_log=%s",
            len(trades),
            "skipped" if audit_log is None else len(audit_log),
        )
        return 0
    finally:
        source_engine.dispose()
        target_engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
