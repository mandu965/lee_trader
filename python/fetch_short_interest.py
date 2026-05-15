"""
공매도 잔고 수집 스크립트 — C-1

pykrx를 사용하여 종목별 공매도 잔고 비율(short interest ratio)을 일별로 수집한다.

수집 데이터:
  - 종목별 공매도 잔고 주수 / 금액
  - 상장주식수
  - 공매도 잔고 비율 (short_ratio, %) = 공매도잔고 / 상장주식수 × 100

활용:
  - feature_builder.py merge_short_interest() → short_ratio, short_ratio_5d_chg, short_ratio_20d_avg
  - risk_penalty 보조 입력 (高 short_ratio = 하방 압력 신호)
  - 공매도 급증 구간 = fundamental_decline_flag 보완 지표

출력:
  - data/short_interest.csv  (로컬 캐시)
  - DB 테이블 short_interest_daily (Postgres 우선, SQLite fallback)

환경변수:
  - SHORT_INTEREST_YEARS_BACK: 수집 기간 (기본 2)
  - SHORT_INTEREST_SLEEP_SEC: 날짜 간 슬립 (기본 1.5)

선행 조건:
  - pip install pykrx

사용법:
  python fetch_short_interest.py --start 20250101 --end 20260515
  python fetch_short_interest.py --start 20250101 --end 20260515 --market KOSPI KOSDAQ
  python fetch_short_interest.py --start 20250101 --end 20260515 --codes 005930,000660
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

DATA_DIR = Path("data")
SHORT_CSV = DATA_DIR / "short_interest.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

DB_TABLE = "short_interest_daily"

DB_COLUMNS = [
    "date",
    "code",
    "short_volume",
    "short_amount",
    "listed_shares",
    "short_ratio",
    "market",
]

DEFAULT_SLEEP_SEC = float(os.getenv("SHORT_INTEREST_SLEEP_SEC", "1.5"))
DEFAULT_YEARS_BACK = int(os.getenv("SHORT_INTEREST_YEARS_BACK", "2"))

try:
    from db import get_engine, replace_table_rows_pg, replace_table_rows_sqlite, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None

    def use_sqlite_fallback_writes() -> bool:
        return False


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            if value.startswith('"'):
                end = value.find('"', 1)
                value = value[1:end] if end != -1 else value[1:]
            elif value.startswith("'"):
                end = value.find("'", 1)
                value = value[1:end] if end != -1 else value[1:]
            else:
                value = re.sub(r"\s+#.*$", "", value).strip()
            if key not in os.environ:
                os.environ[key] = value


def iter_business_dates(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    result = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:  # 월~금
            result.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return result


def get_universe_codes() -> Optional[List[str]]:
    if not UNIVERSE_CSV.exists():
        return None
    try:
        df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
        if "code" not in df.columns:
            return None
        codes = df["code"].dropna().astype(str).str.zfill(6).unique().tolist()
        return sorted(codes) if codes else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 기존 캐시 로드
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    if not SHORT_CSV.exists():
        return pd.DataFrame(columns=DB_COLUMNS)
    try:
        df = pd.read_csv(SHORT_CSV, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        logging.warning("Failed to read existing short_interest.csv")
        return pd.DataFrame(columns=DB_COLUMNS)


def covered_dates(existing: pd.DataFrame) -> set:
    if existing.empty or "date" not in existing.columns:
        return set()
    return set(existing["date"].dropna().unique())


# ---------------------------------------------------------------------------
# pykrx 공매도 잔고 수집
# ---------------------------------------------------------------------------

def fetch_short_balance_day(
    date_str: str,
    markets: List[str],
    target_codes: Optional[set],
) -> pd.DataFrame:
    """특정 날짜의 전 종목 공매도 잔고 조회 (pykrx).

    pykrx.stock.get_shorting_balance_by_ticker() 컬럼명은 버전에 따라 다를 수 있어
    공통 후보명으로 처리한다.
    """
    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        raise RuntimeError("pykrx not installed. pip install pykrx")

    frames = []
    for market in markets:
        try:
            df = pykrx_stock.get_shorting_balance_by_ticker(date_str, market=market)
            if df is None or df.empty:
                continue

            df = df.reset_index()

            # 티커 컬럼 표준화
            ticker_col = None
            for c in ["티커", "종목코드", "Ticker", "ticker", "code"]:
                if c in df.columns:
                    ticker_col = c
                    break
            if ticker_col is None:
                logging.warning("pykrx: ticker column not found for date=%s market=%s cols=%s", date_str, market, df.columns.tolist())
                continue
            df = df.rename(columns={ticker_col: "code"})
            df["code"] = df["code"].astype(str).str.zfill(6)

            # 공매도 잔고 주수 컬럼
            vol_col = None
            for c in ["공매도잔고", "공매도 잔고", "ShortVolume", "short_volume"]:
                if c in df.columns:
                    vol_col = c
                    break

            # 공매도 잔고 금액 컬럼
            amt_col = None
            for c in ["공매도잔고금액", "공매도 잔고금액", "ShortAmount", "short_amount"]:
                if c in df.columns:
                    amt_col = c
                    break

            # 상장주식수 컬럼
            listed_col = None
            for c in ["상장주식수", "ListedShares", "listed_shares"]:
                if c in df.columns:
                    listed_col = c
                    break

            # 잔고 비율 컬럼
            ratio_col = None
            for c in ["공매도잔고비율", "공매도 잔고비율", "ShortRatio", "short_ratio"]:
                if c in df.columns:
                    ratio_col = c
                    break

            out = pd.DataFrame()
            out["code"] = df["code"]
            out["short_volume"]  = pd.to_numeric(df[vol_col],    errors="coerce") if vol_col    else None
            out["short_amount"]  = pd.to_numeric(df[amt_col],    errors="coerce") if amt_col    else None
            out["listed_shares"] = pd.to_numeric(df[listed_col], errors="coerce") if listed_col else None
            out["short_ratio"]   = pd.to_numeric(df[ratio_col],  errors="coerce") if ratio_col  else None

            # ratio 직접 계산 (컬럼이 없거나 NaN인 경우)
            if "short_ratio" not in out.columns or out["short_ratio"].isna().all():
                if out["short_volume"].notna().any() and out["listed_shares"].notna().any():
                    out["short_ratio"] = (out["short_volume"] / out["listed_shares"] * 100).round(4)

            out["market"] = market
            out["date"]   = pd.to_datetime(date_str, format="%Y%m%d").strftime("%Y-%m-%d")

            if target_codes is not None:
                out = out[out["code"].isin(target_codes)]

            frames.append(out)

        except Exception as e:
            logging.warning("pykrx short balance fetch failed: date=%s market=%s error=%s", date_str, market, e)

    if not frames:
        return pd.DataFrame(columns=DB_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["code", "date"], keep="first")
    return result[DB_COLUMNS]


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def save_short_interest(df: pd.DataFrame) -> None:
    if df.empty:
        logging.warning("No short interest data to save")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in DB_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[DB_COLUMNS]
    out = out.sort_values(["code", "date"]).reset_index(drop=True)

    out.to_csv(SHORT_CSV, index=False, encoding="utf-8")
    logging.info("Saved CSV: %s (rows=%d)", SHORT_CSV.resolve(), len(out))

    # Postgres 저장
    try:
        if replace_table_rows_pg is not None and get_engine is not None:
            from sqlalchemy import text
            eng = get_engine()
            with eng.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS short_interest_daily (
                        date          DATE NOT NULL,
                        code          TEXT NOT NULL,
                        short_volume  BIGINT,
                        short_amount  BIGINT,
                        listed_shares BIGINT,
                        short_ratio   DOUBLE PRECISION,
                        market        TEXT,
                        created_at    TIMESTAMP DEFAULT now(),
                        updated_at    TIMESTAMP DEFAULT now(),
                        PRIMARY KEY (date, code)
                    )
                """))
                # migration
                for col_sql in [
                    "ALTER TABLE short_interest_daily ADD COLUMN IF NOT EXISTS short_volume BIGINT",
                    "ALTER TABLE short_interest_daily ADD COLUMN IF NOT EXISTS short_amount BIGINT",
                    "ALTER TABLE short_interest_daily ADD COLUMN IF NOT EXISTS listed_shares BIGINT",
                    "ALTER TABLE short_interest_daily ADD COLUMN IF NOT EXISTS short_ratio DOUBLE PRECISION",
                    "ALTER TABLE short_interest_daily ADD COLUMN IF NOT EXISTS market TEXT",
                ]:
                    conn.execute(text(col_sql))
            replace_table_rows_pg(DB_TABLE, out, columns=DB_COLUMNS)
            logging.info("Saved to Postgres: %s (rows=%d)", DB_TABLE, len(out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to SQLite")

    if not use_sqlite_fallback_writes():
        logging.info("SQLite fallback disabled")
        return

    _save_sqlite(out)


def _save_sqlite(df: pd.DataFrame) -> None:
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS short_interest_daily (
                date          TEXT NOT NULL,
                code          TEXT NOT NULL,
                short_volume  INTEGER,
                short_amount  INTEGER,
                listed_shares INTEGER,
                short_ratio   REAL,
                market        TEXT,
                PRIMARY KEY (date, code)
            )
        """)
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(short_interest_daily)").fetchall()}
        for col, col_type in [
            ("short_volume", "INTEGER"), ("short_amount", "INTEGER"),
            ("listed_shares", "INTEGER"), ("short_ratio", "REAL"), ("market", "TEXT"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE short_interest_daily ADD COLUMN {col} {col_type}")

        placeholders = ", ".join(f":{c}" for c in DB_COLUMNS)
        col_names = ", ".join(DB_COLUMNS)
        records = df[DB_COLUMNS].where(pd.notnull(df[DB_COLUMNS]), None).to_dict(orient="records")
        conn.executemany(
            f"INSERT OR REPLACE INTO short_interest_daily ({col_names}) VALUES ({placeholders})",
            records,
        )
        conn.commit()
        logging.info("Saved to SQLite: short_interest_daily (rows=%d)", len(df))
    except Exception:
        logging.exception("SQLite save failed")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# 메인 수집 루프
# ---------------------------------------------------------------------------

def collect(
    dates: List[str],
    markets: List[str],
    target_codes: Optional[set],
    sleep_sec: float,
    force_refresh: bool,
) -> pd.DataFrame:
    existing = load_existing()
    already = covered_dates(existing) if not force_refresh else set()

    to_fetch = [d for d in dates if d[:4] + "-" + d[4:6] + "-" + d[6:8] not in already]
    logging.info(
        "Short interest fetch plan: total_dates=%d, skip=%d, fetch=%d",
        len(dates), len(dates) - len(to_fetch), len(to_fetch),
    )

    new_frames = []
    for i, date_str in enumerate(to_fetch, 1):
        df_day = fetch_short_balance_day(date_str, markets, target_codes)
        if not df_day.empty:
            new_frames.append(df_day)
            logging.info("[%d/%d] %s: %d rows", i, len(to_fetch), date_str, len(df_day))
        else:
            logging.debug("[%d/%d] %s: no data (휴장일 가능성)", i, len(to_fetch), date_str)
        time.sleep(sleep_sec)

    if not new_frames:
        logging.info("No new data fetched — reusing existing cache")
        return existing

    new_df = pd.concat(new_frames, ignore_index=True)
    if existing.empty:
        merged = new_df
    else:
        merged = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["date", "code"], keep="last")
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    today = datetime.today()
    start_default = (today.replace(year=today.year - DEFAULT_YEARS_BACK)).strftime("%Y%m%d")
    end_default = today.strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="공매도 잔고 비율 수집 (pykrx)")
    p.add_argument("--start", default=start_default, help="시작일 YYYYMMDD")
    p.add_argument("--end",   default=end_default,   help="종료일 YYYYMMDD")
    p.add_argument(
        "--market", nargs="+", default=["KOSPI", "KOSDAQ"],
        choices=["KOSPI", "KOSDAQ", "KONEX"],
        help="대상 시장 (기본: KOSPI KOSDAQ)",
    )
    p.add_argument("--codes", default="", help="쉼표 구분 종목코드 (미지정 시 universe.csv 사용)")
    p.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC, help="날짜 간 슬립(초)")
    p.add_argument("--force-refresh", action="store_true", help="기존 캐시 무시하고 전체 재수집")
    return p.parse_args()


def main() -> None:
    setup_logging()
    _load_env()
    args = parse_args()

    dates = iter_business_dates(args.start, args.end)
    logging.info("Date range: %s ~ %s (%d business days)", args.start, args.end, len(dates))

    # 종목 필터
    target_codes: Optional[set] = None
    if args.codes:
        target_codes = {c.strip().zfill(6) for c in args.codes.split(",") if c.strip()}
        logging.info("Filtering to %d specified codes", len(target_codes))
    else:
        uni_codes = get_universe_codes()
        if uni_codes:
            target_codes = set(uni_codes)
            logging.info("Filtering to universe: %d codes", len(target_codes))
        else:
            logging.info("No universe filter — collecting all tickers")

    result = collect(
        dates=dates,
        markets=args.market,
        target_codes=target_codes,
        sleep_sec=args.sleep_sec,
        force_refresh=args.force_refresh,
    )

    if result.empty:
        logging.warning("No data collected. 종료.")
        return

    save_short_interest(result)
    logging.info("완료: short_interest_daily rows=%d", len(result))


if __name__ == "__main__":
    main()
