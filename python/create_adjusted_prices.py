import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
INPUT = DATA_DIR / "prices_daily_clean.csv"
OUTPUT = DATA_DIR / "prices_daily_adjusted.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

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

    # 출력 컬럼 정리
    cols = ["date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    final = final[cols]

    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Adjusted prices saved: {OUTPUT}, rows={len(final)}")

    # DB upsert
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
        records = final.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices_adjusted
            (date, code, adj_open, adj_high, adj_low, adj_close, volume)
            VALUES (:date, :code, :adj_open, :adj_high, :adj_low, :adj_close, :volume)
            """,
            records,
        )
        conn.commit()
        print(f"Adjusted prices saved to DB: {DB_PATH}, rows={len(final)}")
    except Exception as e:
        print(f"[ERROR] Failed to save adjusted prices to DB: {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
