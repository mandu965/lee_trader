"""
fetch_market_data.py

Fetch KOSPI index data and foreign investor flows to build a simple
"market regime" flag (market_up).

market_up is True if ALL of the following hold on the latest trading day:

- kospi_close > kospi_ma20
- volatility_5d < 0.03  (5-day std of daily returns < 3%)
- foreign_net_5d > 0    (sum of last 5 days foreign net buying > 0)
                         ※ 외국인 순매수: 네이버 금융 시총 Top50 외국인비율
                            일별 변화량(합산)으로 대체
                            (KRX API 차단 환경 대응)

Results are saved to data/market_status.csv with columns:
date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up

──────────────────────────────────────────────────────────────────
변경 이력 (원본 대비)
──────────────────────────────────────────────────────────────────
1. SSL Inspection 환경 대응
   - urllib3.disable_warnings() 추가
   - SSL_VERIFY = False 상수로 전체 requests 호출에 적용

2. pykrx 의존 제거
   - fetch_kospi_ohlcv()       : pykrx → yfinance (^KS11 티커)
   - fetch_kospi_foreign_flow(): pykrx → 네이버 금융 시총 Top50
                                  외국인비율 일별 변화량 proxy 사용
   - _resolve_last_trading_day(): pykrx → yfinance

3. FinanceDataReader(fdr) 의존 → soft-import 유지 (있으면 사용)

4. from db import raw_psycopg2_conn → soft-import 수정
   (db.py 없는 환경에서도 CSV 저장까지는 정상 동작)

5. Postgres 연결/저장 실패 시 logging.exception → logging.warning
   (traceback 없이 한 줄 경고만 출력)

6. yfinance → 네이버 금융 KOSPI 지수 크롤링 1순위로 변경
   - _naver_fetch_kospi_ohlcv(): 네이버 금융 KOSPI 일별 시세 크롤링 추가
   - fetch_kospi_ohlcv(): 네이버 1순위 → yfinance ^KS11 2순위 → proxy 3순위
   - _resolve_last_trading_day(): 네이버 1순위 → yfinance 2순위 → 시스템날짜 3순위
   - yfinance → soft-import 변경 (없어도 동작)
   (Yahoo Finance 차단 환경의 Docker 컨테이너에서도 정상 동작)
"""
import logging
import os
import time
import urllib3
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# soft-import: yfinance (있으면 보조 수단으로 활용)
try:
    import yfinance as yf
except Exception:
    yf = None

# ─────────────────────────────────────────────────────────────────────────────
#  SSL Inspection(회사 보안장비) 환경 대응
# ─────────────────────────────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SSL_VERIFY = False   # 일반 인터넷 환경이면 True 로 변경

# ─────────────────────────────────────────────────────────────────────────────
#  soft-import: db 모듈 (없어도 CSV 저장까지 정상 동작)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from db import raw_psycopg2_conn as _raw_psycopg2_conn
except Exception:
    _raw_psycopg2_conn = None

try:
    from db import replace_table_rows_pg as _replace_table_rows_pg
except Exception:
    _replace_table_rows_pg = None

# soft-import: FinanceDataReader (있으면 보조 수단으로 활용)
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

# ─────────────────────────────────────────────────────────────────────────────
#  상수
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_CSV  = DATA_DIR / "market_status.csv"

_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
_REQUEST_DELAY = 0.3


# ─────────────────────────────────────────────────────────────────────────────
#  공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )


def _to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _env_market_date() -> datetime | None:
    value = os.environ.get("MARKET_DATE", "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    logging.warning("Invalid MARKET_DATE format: %s (expected YYYY-MM-DD or YYYYMMDD)", value)
    return None


def _naver_get(url: str, **kwargs) -> requests.Response:
    """SSL verify=False + 공통 헤더를 적용한 네이버 금융 GET 래퍼."""
    kwargs.setdefault("headers", _NAVER_HEADERS)
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("verify", SSL_VERIFY)
    return requests.get(url, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  네이버 금융 KOSPI 지수 일별 시세 크롤러 (핵심 데이터 소스)
# ─────────────────────────────────────────────────────────────────────────────
def _naver_fetch_kospi_ohlcv(pages: int = 10) -> pd.DataFrame:
    """
    네이버 금융 KOSPI 지수 일별 시세 페이지를 크롤링하여 DataFrame 반환.
    컬럼: date(DatetimeIndex), close(float), volume(float)
    페이지당 약 6행, pages=10 → 약 60 거래일치
    """
    rows = []
    for page in range(1, pages + 1):
        url = (
            f"https://finance.naver.com/sise/sise_index_day.naver"
            f"?code=KOSPI&page={page}"
        )
        try:
            resp = _naver_get(url)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text), encoding="euc-kr")
            if not tables:
                continue
            df_page = tables[0].dropna(how="all")
            rows.append(df_page)
            time.sleep(0.2)
        except Exception as exc:
            logging.warning("_naver_fetch_kospi_ohlcv page=%d failed: %s", page, exc)
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True).dropna(subset=["날짜"])
    df["date"]   = pd.to_datetime(df["날짜"], format="%Y.%m.%d", errors="coerce")
    df["close"]  = pd.to_numeric(df["체결가"].astype(str).str.replace(",", ""), errors="coerce")
    df["volume"] = pd.to_numeric(
        df["거래량(천주)"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")[["close", "volume"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  _resolve_last_trading_day — 네이버 금융 1순위, yfinance 2순위
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_last_trading_day(max_years_back: int = 5) -> datetime:
    """
    가장 최근 거래일을 반환한다.
    1순위: 네이버 금융 KOSPI 지수 시세 (SSL inspection 환경에서도 동작)
    2순위: yfinance ^KS11
    3순위: 시스템 날짜
    """
    # 1순위: 네이버 금융
    try:
        df = _naver_fetch_kospi_ohlcv(pages=1)
        if df is not None and not df.empty:
            last_day = datetime.combine(df.index.max().date(), datetime.min.time())
            logging.info("Resolved last trading day: %s (via Naver Finance)", last_day.date())
            return last_day
    except Exception as exc:
        logging.warning("Naver _resolve_last_trading_day failed: %s", exc)

    # 2순위: yfinance
    if yf is not None:
        base  = datetime.today()
        start = base - timedelta(days=365 * max_years_back)
        try:
            df = yf.download(
                "^KS11",
                start=start.strftime("%Y-%m-%d"),
                end=base.strftime("%Y-%m-%d"),
                progress=False,
            )
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                last_day = datetime.combine(df.index.max().date(), datetime.min.time())
                logging.info("Resolved last trading day: %s (via yfinance)", last_day.date())
                return last_day
        except Exception as exc:
            logging.warning("yfinance _resolve_last_trading_day failed: %s", exc)

    # 3순위: 시스템 날짜
    base = datetime.today()
    logging.warning("Falling back to system date: %s", base.date())
    return base


def _resolve_last_completed_trading_day(max_years_back: int = 5) -> datetime:
    latest_visible = _resolve_last_trading_day(max_years_back=max_years_back)
    today = datetime.today().date()
    if latest_visible.date() < today:
        return latest_visible

    try:
        df = _naver_fetch_kospi_ohlcv(pages=2)
        if df is not None and not df.empty:
            prior_dates = sorted({idx.date() for idx in pd.to_datetime(df.index) if idx.date() < today})
            if prior_dates:
                resolved = datetime.combine(prior_dates[-1], datetime.min.time())
                logging.info("Resolved last completed trading day: %s (via Naver Finance prior row)", resolved.date())
                return resolved
    except Exception as exc:
        logging.warning("Naver prior-row lookup failed: %s", exc)

    if yf is not None:
        try:
            base = datetime.today()
            start = base - timedelta(days=14)
            df = yf.download(
                "^KS11",
                start=start.strftime("%Y-%m-%d"),
                end=base.strftime("%Y-%m-%d"),
                progress=False,
            )
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                prior_dates = sorted({idx.date() for idx in df.index if idx.date() < today})
                if prior_dates:
                    resolved = datetime.combine(prior_dates[-1], datetime.min.time())
                    logging.info("Resolved last completed trading day: %s (via yfinance prior row)", resolved.date())
                    return resolved
        except Exception as exc:
            logging.warning("yfinance prior-row lookup failed: %s", exc)

    resolved = latest_visible - timedelta(days=1)
    logging.warning("Falling back to previous calendar day for completed trading day: %s", resolved.date())
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
#  fetch_kospi_ohlcv — 네이버 금융 1순위, yfinance 2순위 (pykrx 완전 제거)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_kospi_ohlcv(start: datetime, end: datetime) -> pd.DataFrame:
    """
    KOSPI 지수 OHLCV를 반환한다. 반환 DataFrame 컬럼에 'close' 가 포함된다.

    데이터 소스 우선순위:
    1. 네이버 금융 KOSPI 지수 일별 시세 (SSL inspection 환경에서도 동작)
    2. yfinance ^KS11
    3. yfinance KODEX 200 ETF (069500.KS) proxy
    """
    logging.info("Fetching KOSPI OHLCV from %s to %s", start.date(), end.date())

    # ── 1순위: 네이버 금융 KOSPI 지수 크롤링 ──────────────────────────────
    try:
        df = _naver_fetch_kospi_ohlcv(pages=10)
        if df is not None and not df.empty:
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            if not df.empty:
                logging.info("KOSPI OHLCV fetched: %d rows (Naver Finance)", len(df))
                return df
    except Exception as exc:
        logging.warning("Naver fetch_kospi_ohlcv failed: %s", exc)

    # ── 2순위: yfinance ^KS11 ─────────────────────────────────────────────
    if yf is not None:
        end_dl = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df = yf.download(
                "^KS11",
                start=start.strftime("%Y-%m-%d"),
                end=end_dl,
                progress=False,
            )
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.sort_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                if "close" not in df.columns and "Close" in df.columns:
                    df.rename(columns={"Close": "close"}, inplace=True)
                logging.info("KOSPI OHLCV fetched: %d rows (yfinance ^KS11)", len(df))
                return df
        except Exception as exc:
            logging.warning("yfinance ^KS11 failed: %s", exc)

        # ── 3순위: yfinance KODEX 200 ETF proxy ──────────────────────────
        proxy = os.environ.get("KOSPI_PROXY_TICKER", "069500.KS").strip()
        logging.warning("Trying proxy ticker %s for KOSPI index", proxy)
        try:
            proxy_df = yf.download(
                proxy,
                start=start.strftime("%Y-%m-%d"),
                end=end_dl,
                progress=False,
            )
            if proxy_df is not None and not proxy_df.empty:
                proxy_df.index = pd.to_datetime(proxy_df.index).tz_localize(None)
                proxy_df = proxy_df.sort_index()
                if isinstance(proxy_df.columns, pd.MultiIndex):
                    proxy_df.columns = [col[0].lower() for col in proxy_df.columns]
                else:
                    proxy_df.columns = [c.lower() for c in proxy_df.columns]
                if "close" not in proxy_df.columns and "Close" in proxy_df.columns:
                    proxy_df.rename(columns={"Close": "close"}, inplace=True)
                logging.info("KOSPI OHLCV fetched: %d rows (yfinance proxy %s)", len(proxy_df), proxy)
                return proxy_df
        except Exception as exc:
            logging.warning("Proxy ticker %s failed: %s", proxy, exc)

    raise RuntimeError(
        "No KOSPI index data available. "
        "Naver Finance, yfinance ^KS11, and proxy ticker all failed."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  fetch_kospi_foreign_flow — 네이버 금융 외국인비율 변화 proxy (pykrx 제거)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_naver_foreign_ratio_today() -> float:
    """
    네이버 금융 KOSPI 시총 순위 1페이지(50종목)의 외국인비율(%) 평균을 반환한다.
    """
    url = "https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1"
    try:
        resp = _naver_get(url)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.select_one("table.type_2")
        if table is None:
            return 0.0
        ratios = []
        for row in table.select("tbody tr"):
            cols = row.select("td")
            if len(cols) < 9:
                continue
            try:
                ratio = float(cols[8].text.strip().replace(",", ""))
                ratios.append(ratio)
            except ValueError:
                continue
        return float(np.mean(ratios)) if ratios else 0.0
    except Exception as exc:
        logging.warning("_fetch_naver_foreign_ratio_today failed: %s", exc)
        return 0.0


def fetch_kospi_foreign_flow(start: datetime, end: datetime) -> pd.Series:
    """
    외국인 순매수 Series(일별)를 반환한다.
    yfinance ^KS11 Volume × (오늘 외국인비율 proxy)로 추정.
    """
    logging.info(
        "Fetching KOSPI foreign flow proxy from %s to %s",
        start.date(), end.date(),
    )

    try:
        ohlcv  = fetch_kospi_ohlcv(start, end)
        volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else pd.Series(dtype=float)
    except Exception as exc:
        logging.warning("fetch_kospi_foreign_flow: OHLCV 조회 실패 → 0 series 반환: %s", exc)
        return pd.Series(dtype=float, name="foreign_net")

    if volume.empty:
        return pd.Series(dtype=float, name="foreign_net")

    foreign_ratio = _fetch_naver_foreign_ratio_today() / 100.0
    logging.info("Foreign ratio proxy (today): %.2f%%", foreign_ratio * 100)

    foreign_vol = volume * foreign_ratio
    foreign_net = foreign_vol.diff().fillna(0.0)
    foreign_net.name = "foreign_net"

    logging.info("Foreign flow proxy computed: %d rows", len(foreign_net))
    return foreign_net


# ─────────────────────────────────────────────────────────────────────────────
#  build_market_status
# ─────────────────────────────────────────────────────────────────────────────
def build_market_status() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    env_date = _env_market_date()
    if env_date:
        today = env_date
        logging.info("Using MARKET_DATE override: %s", today.date())
    else:
        today = _resolve_last_completed_trading_day()
        logging.info("Using last completed trading day: %s", today.date())

    start = today - timedelta(days=90)

    idx_df = fetch_kospi_ohlcv(start, today)
    close  = idx_df["close"].astype(float)

    ma20   = close.rolling(window=20, min_periods=5).mean()
    vol_5d = close.pct_change().rolling(window=5, min_periods=3).std()

    foreign_series = fetch_kospi_foreign_flow(start, today)
    foreign_series = foreign_series.reindex(idx_df.index).fillna(0.0)
    foreign_5d     = foreign_series.rolling(window=5, min_periods=3).sum()

    df = pd.DataFrame(
        {
            "kospi_close":    close,
            "kospi_ma20":     ma20,
            "volatility_5d":  vol_5d,
            "foreign_net_5d": foreign_5d,
        },
        index=idx_df.index,
    ).dropna(subset=["kospi_close"])

    df["market_up"] = (
        (df["kospi_close"] > df["kospi_ma20"])
        & (df["volatility_5d"] < 0.03)
        & (df["foreign_net_5d"] > 0)
    ).astype(bool)

    df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  저장 / 로드
# ─────────────────────────────────────────────────────────────────────────────
def save_market_status(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved market status: %s (rows=%d)", OUT_CSV.resolve(), len(df))


def load_cached_market_status() -> pd.DataFrame | None:
    if not OUT_CSV.exists():
        return None
    try:
        return pd.read_csv(OUT_CSV)
    except Exception:
        logging.exception("Failed to load cached market_status.csv")
        return None


def save_market_status_db(df: pd.DataFrame) -> None:
    """
    Postgres(DATABASE_URL)에 market_status를 upsert한다.
    db.py / psycopg2 가 없는 환경에서는 경고만 출력하고 건너뛴다.
    """
    if _replace_table_rows_pg is not None:
        try:
            out = df.copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            out = out[["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d", "market_up"]]
            _replace_table_rows_pg("market_status", out, columns=list(out.columns))
            logging.info("Replaced market_status rows in Postgres (rows=%d)", len(out))
            return
        except Exception as exc:
            logging.warning("replace_table_rows_pg failed for market_status, fallback to upsert: %s", exc)

    if _raw_psycopg2_conn is None:
        logging.warning(
            "raw_psycopg2_conn 을 import할 수 없습니다 (db.py 미존재 또는 psycopg2 미설치). "
            "Postgres 저장을 건너뜁니다."
        )
        return

    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError:
        logging.warning("psycopg2 가 설치되어 있지 않습니다. Postgres 저장을 건너뜁니다.")
        return

    conn = None
    try:
        conn = _raw_psycopg2_conn()
    except Exception as exc:
        logging.warning(
            "Postgres 연결 실패 (서버가 실행 중이지 않거나 DATABASE_URL 미설정). "
            "DB 저장을 건너뜁니다. 원인: %s", exc
        )
        return

    try:
        with conn, conn.cursor() as cur:
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
                "CREATE INDEX IF NOT EXISTS idx_market_status_date_desc "
                "ON market_status(date DESC);"
            )
            records = df.to_dict(orient="records")
            sql = """
                INSERT INTO market_status
                (date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up)
                VALUES (%(date)s, %(kospi_close)s, %(kospi_ma20)s,
                        %(volatility_5d)s, %(foreign_net_5d)s, %(market_up)s)
                ON CONFLICT (date) DO UPDATE SET
                    kospi_close    = EXCLUDED.kospi_close,
                    kospi_ma20     = EXCLUDED.kospi_ma20,
                    volatility_5d  = EXCLUDED.volatility_5d,
                    foreign_net_5d = EXCLUDED.foreign_net_5d,
                    market_up      = EXCLUDED.market_up
            """
            execute_batch(cur, sql, records, page_size=200)
            logging.info("Saved market status to Postgres (rows=%d)", len(records))
    except Exception as exc:
        logging.warning("Postgres 저장 실패, CSV는 정상 저장됨. 원인: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    try:
        df = build_market_status()
    except Exception:
        logging.exception("Failed to build market_status")
        cached = load_cached_market_status()
        if cached is None:
            logging.warning("No market_status data available; skipping write")
            return
        logging.warning("Using cached market_status from %s", OUT_CSV.resolve())
        df = cached

    save_market_status(df)
    save_market_status_db(df)


if __name__ == "__main__":
    main()
