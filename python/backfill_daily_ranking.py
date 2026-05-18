"""
Backfill daily_ranking table from archived ranking snapshot CSVs.
data/history/ranking/YYYYMMDD_ranking_final.csv → daily_ranking (accumulate, skip existing dates)
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "data" / "history" / "ranking"

sys.path.insert(0, str(ROOT / "python"))
from db import accumulate_date_rows_pg, raw_psycopg2_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_existing_dates() -> set[str]:
    conn = raw_psycopg2_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT date::text FROM daily_ranking")
            return {row[0][:10] for row in cur.fetchall()}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_integer_columns() -> set[str]:
    conn = raw_psycopg2_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'daily_ranking'
                  AND data_type IN ('integer', 'bigint', 'smallint')
            """)
            return {row[0] for row in cur.fetchall()}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_table_columns() -> list[str]:
    conn = raw_psycopg2_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'daily_ranking'
                ORDER BY ordinal_position
            """)
            return [row[0] for row in cur.fetchall()]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    files = sorted(ARCHIVE_DIR.glob("*_ranking_final.csv"))
    if not files:
        logging.error("No archive files found in %s", ARCHIVE_DIR)
        return

    existing_dates = get_existing_dates()
    table_columns = set(get_table_columns())
    logging.info("daily_ranking 기존 날짜: %s", sorted(existing_dates))
    logging.info("아카이브 파일 수: %d", len(files))

    skipped, loaded, errors = 0, 0, 0

    for f in files:
        m = re.match(r"(\d{4})(\d{2})(\d{2})_ranking_final\.csv", f.name)
        if not m:
            continue
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        if date_str in existing_dates:
            logging.info("SKIP %s (already in DB)", date_str)
            skipped += 1
            continue

        try:
            df = pd.read_csv(f, low_memory=False)
            if df.empty:
                logging.warning("EMPTY %s", f.name)
                continue

            # date 컬럼 보정
            if "date" not in df.columns:
                df["date"] = date_str
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            if "code" in df.columns:
                df["code"] = df["code"].astype(str).str.zfill(6)

            # DB에 있는 컬럼만 사용
            use_cols = [c for c in df.columns if c in table_columns]
            df = df[use_cols].drop_duplicates(subset=["date", "code"], keep="last")

            # float으로 저장된 integer 컬럼 변환 (e.g. 3.0 → 3)
            int_cols = get_integer_columns()
            for col in use_cols:
                if col in int_cols and col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

            accumulate_date_rows_pg("daily_ranking", df, columns=use_cols, date_col="date")
            logging.info("LOADED %s  (%d rows)", date_str, len(df))
            loaded += 1

        except Exception as e:
            logging.error("ERROR %s: %s", f.name, e)
            errors += 1

    logging.info("=== 완료: loaded=%d  skipped=%d  errors=%d ===", loaded, skipped, errors)


if __name__ == "__main__":
    main()
