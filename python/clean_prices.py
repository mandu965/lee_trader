import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"
DB_PATH = DATA_DIR / "lee_trader.db"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV.resolve()}")
    df = pd.read_csv(RAW_CSV, dtype={"code": str})
    # 표준 컬럼 기대: date, code, open, high, low, close, volume
    expected = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw csv: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 날짜 파싱 및 정렬
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])

    # 숫자형 변환
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 기본 결측/이상치 처리
    # 음수/0 가격 제거, 볼륨 음수 제거
    df.loc[df["open"] <= 0, "open"] = np.nan
    df.loc[df["high"] <= 0, "high"] = np.nan
    df.loc[df["low"] <= 0, "low"] = np.nan
    df.loc[df["close"] <= 0, "close"] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan

    # 고가/저가 불일치 보정: low <= open/close <= high가 아니면 NaN
    mask_bad_range = (df["high"] < df["low"]) | (df["open"] > df["high"]) | (df["open"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
    df.loc[mask_bad_range, ["open", "high", "low", "close"]] = np.nan

    # 코드/날짜 기준 중복 제거(마지막 값 유지)
    df = df.drop_duplicates(subset=["code", "date"], keep="last")

    # 전후 보간: 개별 종목 단위로 가격 보간(극단적으로 비현실적일 수 있으므로 과도한 연속 NaN은 남김)
    def _interpolate_group(g: pd.DataFrame) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume"]:
            g[col] = g[col].interpolate(method="linear", limit=2, limit_direction="both")
        return g

    df = df.groupby("code", group_keys=False).apply(_interpolate_group)

    # 여전히 결측이 남아 있는 행 제거(모델/지표 계산을 위해 필수)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # 최종 정렬
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def save_clean(df: pd.DataFrame):
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved clean prices: {CLEAN_CSV.resolve()} (rows={len(df_out)})")

    # DB upsert
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_clean (
                date    DATE NOT NULL,
                code    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        records = df_out.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices_clean
            (date, code, open, high, low, close, volume)
            VALUES (:date, :code, :open, :high, :low, :close, :volume)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved clean prices to DB: %s (rows=%d)", DB_PATH.resolve(), len(df_out))
    except Exception:
        logging.exception("Failed to save clean prices to DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main():
    setup_logging()
    ensure_data_dir()
    logging.info("Loading raw prices...")
    raw = load_raw()
    logging.info(f"Raw rows: {len(raw)}")
    logging.info("Cleaning...")
    cleaned = clean(raw)
    logging.info(f"Clean rows: {len(cleaned)}")
    save_clean(cleaned)


if __name__ == "__main__":
    main()
