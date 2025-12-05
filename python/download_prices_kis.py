import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import sqlite3

# Optional fallback data source (pykrx) if KIS is not configured
try:
    from pykrx import stock as pykrx_stock
except Exception:
    pykrx_stock = None

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
DB_PATH = DATA_DIR / "lee_trader.db"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_demo_prices(symbols=None, days=240, seed=42) -> pd.DataFrame:
    """
    Generate simple synthetic OHLCV daily data for multiple symbols.
    This acts as a fallback when KIS credentials or network are unavailable.
    """
    if symbols is None:
        # 대표 종목 3개 (삼성전자/하이닉스/네이버)
        symbols = ["005930", "000660", "035420"]

    rng = np.random.default_rng(seed)
    end_date = datetime.today()
    # 최근 영업일 기준으로 240개 정도의 일자를 생성 (주말 제외)
    dates = []
    d = end_date
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d.replace(hour=0, minute=0, second=0, microsecond=0))
        d -= timedelta(days=1)
    dates = sorted(dates)

    rows = []
    for code in symbols:
        # 기초가격을 종목별로 다르게
        base_price = rng.uniform(40000, 90000)
        price = base_price
        for dt in dates:
            # 간단한 랜덤 워크 + 약간의 모멘텀/노이즈
            ret = rng.normal(0, 0.01)
            price = max(1000, price * (1 + ret))
            # OHLCV 구성
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = (high + low) / 2 * (1 + rng.normal(0, 0.001))
            close = price
            volume = int(rng.uniform(1e5, 5e6))
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": int(round(open_)),
                    "high": int(round(high)),
                    "low": int(round(low)),
                    "close": int(round(close)),
                    "volume": volume,
                }
            )
    df = pd.DataFrame(rows)
    return df


def _env_symbols() -> List[str]:
    # 1) universe.csv 우선
    try:
        uni_path = DATA_DIR / "universe.csv"
        if uni_path.exists():
            dfu = pd.read_csv(uni_path, dtype={"code": str})
            codes = [str(c).strip() for c in dfu["code"].dropna().tolist() if str(c).strip()]
            if codes:
                return codes
    except Exception as e:
        logging.warning(f"Failed to load universe.csv, fallback to .env SYMBOLS: {e}")

    # 2) .env SYMBOLS
    val = os.getenv("SYMBOLS")
    if not val:
        return ["005930", "000660", "035420"]
    syms = [s.strip() for s in val.split(",") if s.strip()]
    return syms or ["005930", "000660", "035420"]


def _kis_get_token(base_url: str, app_key: str, app_secret: str) -> Optional[str]:
    """
    POST /oauth2/tokenP
    Body(JSON):
      {
        "grant_type": "client_credentials",
        "appkey": "...",
        "appsecret": "..."
      }
    """
    url = base_url.rstrip("/") + "/oauth2/tokenP"
    try:
        res = requests.post(
            url,
            json={
                "grant_type": "client_credentials",
                "appkey": app_key,
                "appsecret": app_secret,
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if res.status_code != 200:
            logging.warning(f"KIS tokenP failed: {res.status_code} {res.text}")
            return None
        data = res.json()
        access_token = data.get("access_token")
        if not access_token:
            logging.warning(f"KIS tokenP response missing access_token: {data}")
            return None
        return access_token
    except Exception as e:
        logging.warning(f"KIS tokenP exception: {e}")
        return None


def _kis_fetch_daily_prices(
    base_url: str,
    app_key: str,
    app_secret: str,
    access_token: str,
    code: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> Optional[pd.DataFrame]:
    """
    GET /uapi/domestic-stock/v1/quotations/inquire-daily-price
    Query params (예시):
      - FID_COND_MRKT_DIV_CODE=J
      - FID_INPUT_ISCD={code}
      - FID_INPUT_DATE_1={start}
      - FID_INPUT_DATE_2={end}
      - FID_PERIOD_DIV_CODE=D
      - FID_ORG_ADJ_PRC=0
    Headers:
      - authorization: Bearer {access_token}
      - appkey, appsecret
      - tr_id: FHKST03010100 (일별시세)
    """
    url = base_url.rstrip("/") + "/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": app_key,
        "appsecret": app_secret,
        # 모의투자 서버에서도 동일 코드 사용 가능(공식 문서 참고)
        "tr_id": "FHKST03010100",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
        if res.status_code != 200:
            logging.warning(f"KIS daily price failed({code}): {res.status_code} {res.text}")
            return None
        data = res.json()
        # 응답 구조에서 일자별 배열을 찾음(보통 'output2' 사용)
        arr = data.get("output2") or data.get("output") or []
        if not isinstance(arr, list) or not arr:
            logging.warning(f"KIS daily price empty({code}): {data}")
            return None

        rows = []
        for it in arr:
            # KIS 필드명 예: stck_bsop_date(YYYYMMDD), stck_oprc, stck_hgpr, stck_lwpr, stck_clpr, acml_vol
            ymd = str(it.get("stck_bsop_date") or "")
            if len(ymd) == 8:
                date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
            else:
                # fallback
                try:
                    date_str = datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")
                except Exception:
                    continue
            try:
                open_ = int(float(it.get("stck_oprc", 0)))
                high = int(float(it.get("stck_hgpr", 0)))
                low = int(float(it.get("stck_lwpr", 0)))
                close = int(float(it.get("stck_clpr", 0)))
                vol = int(float(it.get("acml_vol", 0)))
            except Exception:
                continue

            rows.append(
                {
                    "date": date_str,
                    "code": code,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
        if not rows:
            return None
        # API는 역순(최근→과거)일 가능성이 높음 → 정렬
        df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
        return df
    except Exception as e:
        logging.warning(f"KIS daily price exception({code}): {e}")
        return None


def try_kis_download() -> Optional[pd.DataFrame]:
    """
    실제 KIS 연동:
      - .env: KIS_BASE_URL, KIS_APP_KEY, KIS_APP_SECRET 필요
      - tokenP로 access_token 획득 후 일별시세 조회
      - 실패 시 None 반환(상위에서 다른 대안 또는 데모로 대체)
    """
    base_url = os.getenv("KIS_BASE_URL")
    app_key = os.getenv("KIS_APP_KEY")
    app_secret = os.getenv("KIS_APP_SECRET")

    required = [base_url, app_key, app_secret]
    if not all(required):
        logging.info("KIS env missing or incomplete -> skip KIS")
        return None

    token = _kis_get_token(base_url, app_key, app_secret)
    if not token:
        return None

    # 최근 240 영업일 정도를 목표로 대략 365일 전부터 오늘까지 요청
    end = datetime.today()
    # 3년치 수집 (평가/학습 데이터 확보용)
    start = end - timedelta(days=365 * 3)
    start_ymd = start.strftime("%Y%m%d")
    end_ymd = end.strftime("%Y%m%d")

    symbols = _env_symbols()
    frames: List[pd.DataFrame] = []
    for code in symbols:
        df_code = _kis_fetch_daily_prices(
            base_url=base_url,
            app_key=app_key,
            app_secret=app_secret,
            access_token=token,
            code=code,
            start_yyyymmdd=start_ymd,
            end_yyyymmdd=end_ymd,
        )
        if df_code is not None and not df_code.empty:
            frames.append(df_code)

    if not frames:
        logging.warning("KIS daily prices fetched nothing")
        return None

    out = pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    return out


def try_pykrx_download() -> Optional[pd.DataFrame]:
    """
    pykrx를 이용한 일별 시세 수집(OHLCV).
    KIS 자격증명이 없거나 실패한 경우의 현실 데이터 대안.
    """
    if pykrx_stock is None:
        logging.info("pykrx not available -> skip pykrx fallback")
        return None
    try:
        end = datetime.today()
        # 3년치 수집
        start = end - timedelta(days=365 * 3)
        start_ymd = start.strftime("%Y%m%d")
        end_ymd = end.strftime("%Y%m%d")

        symbols = _env_symbols()
        frames: List[pd.DataFrame] = []
        for code in symbols:
            try:
                df = pykrx_stock.get_market_ohlcv_by_date(start_ymd, end_ymd, code)
                if df is None or df.empty:
                    continue
                df = df.reset_index().rename(
                    columns={
                        "날짜": "date",
                        "시가": "open",
                        "고가": "high",
                        "저가": "low",
                        "종가": "close",
                        "거래량": "volume",
                    }
                )
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["code"] = code
                df = df[["date", "code", "open", "high", "low", "close", "volume"]]
                frames.append(df)
            except Exception as e:
                logging.warning(f"pykrx fetch error({code}): {e}")
                continue
        if not frames:
            logging.warning("pykrx daily prices fetched nothing")
            return None
        out = pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
        return out
    except Exception as e:
        logging.warning(f"pykrx fallback exception: {e}")
        return None


def main():
    setup_logging()
    ensure_data_dir()

    # 1) 실제 KIS 다운로드 시도
    df = try_kis_download()

    # 2) KIS 실패 시 pykrx 대안 시도
    if df is None:
        logging.info("KIS unavailable -> trying pykrx fallback...")
        df = try_pykrx_download()

    # 3) 둘 다 실패 시 데모 데이터 생성
    demo_mode = False
    if df is None:
        logging.info("Generating demo prices...")
        df = generate_demo_prices()
        demo_mode = True

    # 3) 저장
    df = df.sort_values(["code", "date"])
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    # 데모/라이브 모드 마커 처리(서버 배지 표시에 사용)
    try:
        marker = DATA_DIR / ".demo"
        if demo_mode:
            marker.write_text("demo", encoding="utf-8")
        else:
            if marker.exists():
                marker.unlink()
    except Exception as e:
        logging.warning(f"Failed to update demo marker: {e}")
    logging.info(f"Saved raw prices: {RAW_CSV.resolve()} (rows={len(df)})")

    # DB upsert
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_raw (
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
        records = df.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices_raw
            (date, code, open, high, low, close, volume)
            VALUES (:date, :code, :open, :high, :low, :close, :volume)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved raw prices to DB: %s (rows=%d)", DB_PATH.resolve(), len(df))
    except Exception:
        logging.exception("Failed to save raw prices to DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
