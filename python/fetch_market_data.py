"""
fetch_market_data.py

Fetch KOSPI index data and foreign investor flows to build a simple
"market regime" flag (market_up).

market_up is True if ALL of the following hold on the latest trading day:

- kospi_close > kospi_ma20
- volatility_5d < 0.03  (5-day std of daily returns < 3%)
- foreign_net_5d > 0    (sum of last 5 days foreign net buying > 0)

Results are saved to data/market_status.csv with columns:

date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pykrx import stock
from db import raw_psycopg2_conn

DATA_DIR = Path("data")
OUT_CSV = DATA_DIR / "market_status.csv"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )


def _to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def fetch_kospi_ohlcv(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch KOSPI index OHLCV (ticker: 1001) between start and end."""
    logging.info("Fetching KOSPI index OHLCV from %s to %s", start.date(), end.date())
    df = stock.get_index_ohlcv(_to_yyyymmdd(start), _to_yyyymmdd(end), "1001")
    if df is None or df.empty:
        raise RuntimeError("No KOSPI index data returned from pykrx.get_index_ohlcv")
    df = df.copy()
    # index: usually '날짜'
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.rename(
        columns={
            "종가": "close",
        },
        inplace=True,
    )
    return df


def fetch_kospi_foreign_flow(start: datetime, end: datetime) -> pd.Series:
    """
    Fetch market-wide trading value by investor for KOSPI and
    return a Series of daily foreign net buying (외국인합계).
    """
    logging.info(
        "Fetching KOSPI foreign trading value by date from %s to %s",
        start.date(),
        end.date(),
    )
    df = stock.get_market_trading_value_by_date(
        _to_yyyymmdd(start),
        _to_yyyymmdd(end),
        "KOSPI",
    )
    if df is None or df.empty:
        raise RuntimeError(
            "No KOSPI trading value data returned from "
            "pykrx.get_market_trading_value_by_date"
        )
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "외국인합계" not in df.columns:
        raise RuntimeError(
            "Expected column '외국인합계' not found in trading value data"
        )

    foreign = df["외국인합계"].astype(float)
    return foreign


def build_market_status() -> pd.DataFrame:
    """
    Build market_status dataframe with last ~90 calendar days
    and compute market_up on each trading date.
    """
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    today = datetime.today()
    start = today - timedelta(days=90)

    # --- KOSPI index ---
    idx_df = fetch_kospi_ohlcv(start, today)

    close = idx_df["close"].astype(float)

    # 20-day moving average of close
    ma20 = close.rolling(window=20, min_periods=5).mean()

    # 5-day volatility of daily returns
    returns = close.pct_change()
    vol_5d = returns.rolling(window=5, min_periods=3).std()

    # --- Foreign flow ---
    foreign_series = fetch_kospi_foreign_flow(start, today)
    # align index with idx_df
    foreign_series = foreign_series.reindex(idx_df.index).fillna(0.0)
    foreign_5d = foreign_series.rolling(window=5, min_periods=3).sum()

    # --- assemble ---
    df = pd.DataFrame(
        {
            "kospi_close": close,
            "kospi_ma20": ma20,
            "volatility_5d": vol_5d,
            "foreign_net_5d": foreign_5d,
        },
        index=idx_df.index,
    ).dropna(subset=["kospi_close"])

    # Market regime condition
    market_up = (
        (df["kospi_close"] > df["kospi_ma20"])
        & (df["volatility_5d"] < 0.03)
        & (df["foreign_net_5d"] > 0)
    )
    df["market_up"] = market_up.astype(bool)

    # ✅ 여기서 인덱스를 그대로 date로 복사 (index 이름이 뭐든 상관없게)
    df["date"] = df.index
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 마지막으로 index 제거하고 깔끔하게
    df = df.reset_index(drop=True)

    return df


def save_market_status(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved market status: %s (rows=%d)", OUT_CSV.resolve(), len(df))


def save_market_status_db(df: pd.DataFrame) -> None:
    """
    Persist market_status to Postgres (upsert by date) using DATABASE_URL.
    """
    import psycopg2
    from psycopg2.extras import execute_batch

    conn = None
    try:
        conn = raw_psycopg2_conn()
    except Exception:
        logging.exception("Failed to connect to Postgres (DATABASE_URL)")
        return

    try:
        with conn, conn.cursor() as cur:
            # Ensure table/index exist (idempotent)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS market_status (
                    date            DATE PRIMARY KEY,
                    kospi_close     NUMERIC,
                    kospi_ma20      NUMERIC,
                    volatility_5d   NUMERIC,
                    foreign_net_5d  NUMERIC,
                    market_up       BOOLEAN
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_status_date_desc ON market_status(date DESC);"
            )

            records = df.to_dict(orient="records")
            sql = """
                INSERT INTO market_status
                (date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up)
                VALUES (%(date)s, %(kospi_close)s, %(kospi_ma20)s, %(volatility_5d)s, %(foreign_net_5d)s, %(market_up)s)
                ON CONFLICT (date) DO UPDATE SET
                    kospi_close = EXCLUDED.kospi_close,
                    kospi_ma20 = EXCLUDED.kospi_ma20,
                    volatility_5d = EXCLUDED.volatility_5d,
                    foreign_net_5d = EXCLUDED.foreign_net_5d,
                    market_up = EXCLUDED.market_up
            """
            execute_batch(cur, sql, records, page_size=200)
            logging.info("Saved market status to Postgres (rows=%d)", len(records))
    except Exception:
        logging.exception("Failed to save market status to Postgres")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    try:
        df = build_market_status()
    except Exception:
        logging.exception("Failed to build market_status.csv")
        raise
    save_market_status(df)
    save_market_status_db(df)


if __name__ == "__main__":
    main()
