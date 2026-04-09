import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
INPUT = DATA_DIR / "prices_daily_clean.csv"
OUTPUT = DATA_DIR / "prices_daily_adjusted.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg, replace_table_rows_sqlite
except Exception:
    get_engine = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None

PRICES_ADJUSTED_DB_COLUMNS = ["date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
FACT_PRICE_DAILY_DB_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "value",
    "market_cap",
    "listed_shares",
]
PRICE_PK = ["date", "code"]

def detect_split_ratios(df):
    """
    액면분할/병합 이벤트를 감지하여 누적 보정 계수를 계산한다.
    기준:
      - 이전 close 대비 다음 close가 일정 배수로 점프했을 때
      - ratio > 1.5 or ratio < 0.7 (대략 30% 이상 단절)
    """
    df = df.sort_values("date").copy()
    df["ratio"] = df["close"] / df["close"].shift(1)

    # 분할/병합 이벤트로 판단되는 구간 탐지
    events = df[(df["ratio"] > 1.5) | (df["ratio"] < 0.7)].copy()

    # 누적 조정 계수
    df["adj_factor"] = 1.0
    cumulative = 1.0

    for idx, row in events.iterrows():
        ratio = row["ratio"]
        # ratio > 1 → 액면병합 (가격이 급등)
        # ratio < 1 → 액면분할 (가격이 급락)
        cumulative *= ratio
        df.loc[df.index >= idx, "adj_factor"] = cumulative

    return df


def apply_adjustment(df):
    """
    조정계수 적용하여 adjusted_close 생성
    """
    df["adj_close"] = df["close"] / df["adj_factor"]
    df["adj_open"]  = df["open"]  / df["adj_factor"]
    df["adj_high"]  = df["high"]  / df["adj_factor"]
    df["adj_low"]   = df["low"]   / df["adj_factor"]
    return df


def main():
    df = pd.read_csv(INPUT, dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"])

    out_list = []

    for code, g in df.groupby("code"):
        g = g.sort_values("date").copy()

        g = detect_split_ratios(g)
        g = apply_adjustment(g)

        out_list.append(g)

    final = pd.concat(out_list).reset_index(drop=True)

    # CSV 출력(조정가)
    csv_cols = ["date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    csv_out = final[csv_cols].copy()
    csv_out["date"] = csv_out["date"].dt.strftime("%Y-%m-%d")
    csv_out = csv_out[PRICES_ADJUSTED_DB_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(csv_out, PRICE_PK, "prices_adjusted")
    csv_out.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Adjusted prices saved: {OUTPUT}, rows={len(csv_out)}")

    # fact_price_daily 적재용 (raw + adj_close)
    fact_cols = ["date", "code", "open", "high", "low", "close", "adj_close", "volume"]
    fact_df = final[fact_cols].copy()
    fact_df["date"] = fact_df["date"].dt.strftime("%Y-%m-%d")
    fact_df["value"] = pd.NA
    fact_df["market_cap"] = pd.NA
    fact_df["listed_shares"] = pd.NA
    fact_df = fact_df[FACT_PRICE_DAILY_DB_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(fact_df, PRICE_PK, "fact_price_daily")

    # Replace table rows while preserving schema and indexes.
    try:
        if replace_table_rows_pg:
            replace_table_rows_pg("prices_adjusted", csv_out, columns=PRICES_ADJUSTED_DB_COLUMNS)
            replace_table_rows_pg("fact_price_daily", fact_df, columns=FACT_PRICE_DAILY_DB_COLUMNS)
            print(f"Adjusted prices replaced in Postgres, rows={len(csv_out)}")
            print(f"fact_price_daily replaced in Postgres, rows={len(fact_df)}")
            return
    except Exception as e:
        print(f"[WARN] Postgres row replace failed, fallback to sqlite: {e}")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_adjusted (
                date      DATE NOT NULL,
                code      TEXT NOT NULL,
                adj_open  REAL,
                adj_high  REAL,
                adj_low   REAL,
                adj_close REAL,
                volume    REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_price_daily (
                date          DATE NOT NULL,
                code          TEXT NOT NULL,
                open          REAL,
                high          REAL,
                low           REAL,
                close         REAL,
                adj_close     REAL,
                volume        REAL,
                value         REAL,
                market_cap    REAL,
                listed_shares REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "prices_adjusted", csv_out)
            replace_table_rows_sqlite(conn, "fact_price_daily", fact_df)
        conn.commit()
        print(f"Adjusted prices saved to sqlite DB: {DB_PATH}, rows={len(csv_out)}")
        print(f"fact_price_daily saved to sqlite DB: {DB_PATH}, rows={len(fact_df)}")
    except Exception as e:
        print(f"[ERROR] Failed to save adjusted prices to sqlite DB: {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
