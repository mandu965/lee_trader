import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sqlalchemy import text

DATA_DIR = Path("data")
INPUT = DATA_DIR / "prices_daily_clean.csv"
OUTPUT = DATA_DIR / "prices_daily_adjusted.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg, replace_table_rows_sqlite, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None
    def use_sqlite_fallback_writes() -> bool:
        return False

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
BASE_PRICE_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume"]


def _load_existing_fact_price_daily() -> pd.DataFrame:
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(
                    text("SELECT date, code, open, high, low, close, volume FROM fact_price_daily"),
                    conn,
                    dtype={"code": str},
                )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["code"] = df["code"].astype(str).str.zfill(6)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["date", "code", "close"])
                return df[BASE_PRICE_COLUMNS]
        except Exception:
            pass

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql(
                    "SELECT date, code, open, high, low, close, volume FROM fact_price_daily",
                    conn,
                    dtype={"code": str},
                )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["code"] = df["code"].astype(str).str.zfill(6)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["date", "code", "close"])
                return df[BASE_PRICE_COLUMNS]
        except Exception:
            pass

    return pd.DataFrame(columns=BASE_PRICE_COLUMNS)


def _load_existing_adjusted_csv() -> pd.DataFrame:
    adjusted_cols = ["date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    if not OUTPUT.exists():
        return pd.DataFrame(columns=BASE_PRICE_COLUMNS)

    try:
        df = pd.read_csv(OUTPUT, dtype={"code": str}, usecols=lambda col: col in adjusted_cols)
        if df.empty:
            return pd.DataFrame(columns=BASE_PRICE_COLUMNS)

        rename_map = {
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
        }
        df = df.rename(columns=rename_map)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["code"] = df["code"].astype(str).str.zfill(6)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "code", "close"])
        return df[BASE_PRICE_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=BASE_PRICE_COLUMNS)


def _choose_existing_history() -> pd.DataFrame:
    db_df = _load_existing_fact_price_daily()
    adjusted_df = _load_existing_adjusted_csv()

    if db_df.empty and adjusted_df.empty:
        return pd.DataFrame(columns=BASE_PRICE_COLUMNS)
    if db_df.empty:
        print(f"Using adjusted CSV history fallback: rows={len(adjusted_df)}")
        return adjusted_df
    if adjusted_df.empty:
        print(f"Using DB fact_price_daily history: rows={len(db_df)}")
        return db_df

    db_min = db_df["date"].min()
    adj_min = adjusted_df["date"].min()
    if len(db_df) < len(adjusted_df):
        print(
            "DB fact_price_daily history shorter than adjusted CSV "
            f"(db_rows={len(db_df)}, csv_rows={len(adjusted_df)}); using adjusted CSV fallback"
        )
        return adjusted_df
    if pd.notna(db_min) and pd.notna(adj_min) and db_min > adj_min:
        print(
            "DB fact_price_daily starts later than adjusted CSV "
            f"(db_min={db_min.date()}, csv_min={adj_min.date()}); using adjusted CSV fallback"
        )
        return adjusted_df

    print(f"Using DB fact_price_daily history: rows={len(db_df)}")
    return db_df


def _merge_price_history(current_df: pd.DataFrame) -> pd.DataFrame:
    current = current_df[BASE_PRICE_COLUMNS].copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    current["code"] = current["code"].astype(str).str.zfill(6)
    for col in ["open", "high", "low", "close", "volume"]:
        current[col] = pd.to_numeric(current[col], errors="coerce")
    current = current.dropna(subset=["date", "code", "close"])

    existing = _choose_existing_history()
    if existing.empty:
        return current.sort_values(["code", "date"]).reset_index(drop=True)

    merged = pd.concat([existing, current], ignore_index=True)
    merged = merged.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    print(
        f"Merged price history for adjustment: existing_rows={len(existing)} current_rows={len(current)} merged_rows={len(merged)}"
    )
    return merged
<<<<<<< HEAD

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


=======

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


>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
def main():
    df = pd.read_csv(INPUT, dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = _merge_price_history(df)

    out_list = []
<<<<<<< HEAD

    for code, g in df.groupby("code"):
        g = g.sort_values("date").copy()

        g = detect_split_ratios(g)
        g = apply_adjustment(g)

        out_list.append(g)

    final = pd.concat(out_list).reset_index(drop=True)

=======

    for code, g in df.groupby("code"):
        g = g.sort_values("date").copy()

        g = detect_split_ratios(g)
        g = apply_adjustment(g)

        out_list.append(g)

    final = pd.concat(out_list).reset_index(drop=True)

>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
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

    if not use_sqlite_fallback_writes():
        print("[INFO] Skipping sqlite fallback for adjusted prices (USE_SQLITE_FALLBACK_WRITES=0)")
        return

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
<<<<<<< HEAD


if __name__ == "__main__":
    main()
=======


if __name__ == "__main__":
    main()
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
