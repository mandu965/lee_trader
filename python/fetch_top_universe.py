import logging
import os
import sqlite3
import time
import urllib3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

# ─────────────────────────────────────────────────────────────────────────────
#  SSL Inspection(회사 보안장비) 환경 대응
#  - 모든 requests 호출에 verify=False 적용
#  - InsecureRequestWarning 경고 메시지 억제
# ─────────────────────────────────────────────────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# NAVER_FALLBACK_VERIFY_SSL=0 일 때만 False (SSL 인스펙션 차단 환경 대응)
SSL_VERIFY = os.environ.get("NAVER_FALLBACK_VERIFY_SSL", "1").strip() != "0"

# ─────────────────────────────────────────────────────────────────────────────
#  경로 상수
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
UNIVERSE_CSV = DATA_DIR / "universe.csv"
SECTORS_CSV  = DATA_DIR / "sectors.csv"
DB_PATH      = DATA_DIR / "lee_trader.db"
STOCKS_STORE_COLUMNS = ["code", "name", "market", "sector", "listed_at", "delisted_at"]
STOCKS_PK = ["code"]

try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg, replace_table_rows_sqlite, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None
    def use_sqlite_fallback_writes() -> bool:
        return False


def _upsert_stocks_pg(df: pd.DataFrame) -> None:
    if not get_engine:
        raise RuntimeError("Postgres engine unavailable")
    out = df.copy()
    for col in ["listed_at", "delisted_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype(object)
            out.loc[out[col].isna(), col] = None
    records = out.astype(object).where(pd.notna(out), None).to_dict(orient="records")
    stmt = text(
        """
        INSERT INTO stocks (code, name, market, sector, listed_at, delisted_at)
        VALUES (:code, :name, :market, :sector, :listed_at, :delisted_at)
        ON CONFLICT (code) DO UPDATE SET
            name = EXCLUDED.name,
            market = EXCLUDED.market,
            sector = EXCLUDED.sector,
            listed_at = EXCLUDED.listed_at,
            delisted_at = EXCLUDED.delisted_at
        """
    )
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(stmt, records)

# pykrx — 설치돼 있으면 쓰고, 없어도 동작하도록 soft-import
try:
    from pykrx import stock as _pykrx_stock
except Exception:
    _pykrx_stock = None

# FinanceDataReader — soft-import (사용 불가 환경에서도 동작)
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# ─────────────────────────────────────────────────────────────────────────────
#  네이버 금융 크롤링 공통 설정
# ─────────────────────────────────────────────────────────────────────────────
_NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
ETF_NAME_TOKENS = (
    "etf",
    "etn",
    "kodex",
    "tiger",
    "kosef",
    "kindex",
    "koact",
    "rise",
    "ace",
    "plus",
    "sol",
    "timefolio",
    "hanaro",
    "arirang",
    "fofocus",
)
_MARKET_SOSOK = {"KOSPI": 0, "KOSDAQ": 1}
_REQUEST_DELAY = 0.3   # 서버 부하 방지용 요청 간격(초)
_MAX_PAGES     = 5     # 페이지당 50종목 × 5페이지 = 최대 250종목


# ─────────────────────────────────────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _requests_get(url: str, **kwargs) -> requests.Response:
    """SSL verify=False 를 공통 적용한 requests.get 래퍼."""
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("verify", SSL_VERIFY)
    kwargs.setdefault("headers", _NAVER_HEADERS)
    return requests.get(url, **kwargs)


def _requests_post(url: str, **kwargs) -> requests.Response:
    """SSL verify=False 를 공통 적용한 requests.post 래퍼."""
    kwargs.setdefault("timeout", 15)
    kwargs.setdefault("verify", SSL_VERIFY)
    return requests.post(url, **kwargs)


def _parse_mcap(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _parse_market_date_override() -> Optional[str]:
    raw = str(os.environ.get("MARKET_DATE", "")).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    logging.warning("Invalid MARKET_DATE format in fetch_top_universe: %s", raw)
    return None


def _is_etf_like_name(name: object) -> bool:
    text = str(name or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in ETF_NAME_TOKENS)


def _filter_common_equities(df: pd.DataFrame, *, market: str, stage: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "market"])

    out = df.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    if "market" not in out.columns:
        out["market"] = market
    out["market"] = out["market"].astype(str).str.upper().str.strip()
    if "name" not in out.columns:
        out["name"] = ""
    out["name"] = out["name"].fillna("").astype(str).str.strip()

    before = len(out)
    mask = out["name"].map(_is_etf_like_name)
    removed = int(mask.sum())
    if removed:
        sample = out.loc[mask, "name"].head(10).tolist()
        logging.info("[%s] filtered ETF/ETN-like rows: removed=%d sample=%s", stage, removed, sample)
    out = out.loc[~mask].copy()
    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)
    logging.info("[%s] equity-only rows=%d (before=%d)", stage, len(out), before)
    return out[["code", "name", "market"]]


# ─────────────────────────────────────────────────────────────────────────────
#  네이버 금융 — 시총 순위 크롤링 (pykrx 대체)
# ─────────────────────────────────────────────────────────────────────────────
def _naver_parse_page(soup: BeautifulSoup) -> list[dict]:
    """네이버 금융 시총 순위 한 페이지를 파싱해 종목 리스트를 반환한다."""
    table = soup.select_one("table.type_2")
    if table is None:
        return []

    records = []
    for row in table.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 11:
            continue
        name_tag = row.select_one("td a")
        if not name_tag:
            continue

        href  = name_tag.get("href", "")
        code  = href.split("code=")[-1] if "code=" in href else ""
        texts = [c.text.strip() for c in cols]

        records.append({
            "code":        code,
            "name":        name_tag.text.strip(),
            "현재가":       texts[2],
            "등락률":       texts[4],
            "시가총액(억원)": texts[6],
            "상장주식수":    texts[7],
            "외국인비율":    texts[8],
            "거래량":       texts[9],
            "PER":         texts[10],
            "_mcap_raw":   texts[6],
        })
    return records


def _naver_fetch_market_top(market: str, top_n: int) -> pd.DataFrame:
    """
    네이버 금융 시총 순위에서 market(KOSPI/KOSDAQ) 상위 top_n 종목을 반환.
    반환 컬럼: code, name, market
    """
    sosok = _MARKET_SOSOK.get(market.upper())
    if sosok is None:
        logging.warning("지원하지 않는 market: %s", market)
        return pd.DataFrame(columns=["code", "name", "market"])

    all_records: list[dict] = []
    filtered_count = 0

    for page in range(1, _MAX_PAGES + 1):
        url = (
            f"https://finance.naver.com/sise/sise_market_sum.naver"
            f"?sosok={sosok}&page={page}"
        )
        try:
            resp = _requests_get(url)
            resp.raise_for_status()
        except requests.RequestException as e:
            logging.warning("[%s] page %d 요청 실패: %s", market, page, e)
            break

        soup    = BeautifulSoup(resp.text, "html.parser")
        records = _naver_parse_page(soup)

        if not records:
            logging.debug("[%s] page %d 데이터 없음 → 수집 종료", market, page)
            break

        all_records.extend(records)
        logging.info("[%s] page %d: %d개 수집 (누적: %d개)", market, page, len(records), len(all_records))

        preview = pd.DataFrame(all_records)
        if not preview.empty:
            preview["market"] = market.upper()
            filtered_preview = _filter_common_equities(
                preview[["code", "name", "market"]],
                market=market.upper(),
                stage=f"naver_{market.upper()}_preview",
            )
            filtered_count = len(filtered_preview)
            logging.info("[%s] non-ETF progress: %d/%d", market, filtered_count, top_n)

        if filtered_count >= top_n:
            break

        time.sleep(_REQUEST_DELAY)

    if not all_records:
        logging.warning("[%s] 네이버 금융에서 데이터를 수집하지 못했습니다.", market)
        return pd.DataFrame(columns=["code", "name", "market"])

    df = pd.DataFrame(all_records)
    df["market"] = market.upper()
    df["code"]   = df["code"].astype(str).str.zfill(6)
    df = _filter_common_equities(df[["code", "name", "market"]], market=market.upper(), stage=f"naver_{market.upper()}")
    if len(df) < top_n:
        logging.warning(
            "[%s] unable to fill non-ETF target=%d within max_pages=%d (fetched=%d)",
            market,
            top_n,
            _MAX_PAGES,
            len(df),
        )
    return df.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  네이버 금융 — 종목별 업종(sector) 크롤링
# ─────────────────────────────────────────────────────────────────────────────
def _naver_fetch_sector(code: str) -> str:
    """네이버 금융 종목 상세 페이지에서 업종명을 추출한다."""
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        resp = _requests_get(url, timeout=8)
        if resp.status_code != 200 or not resp.text:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "sise_group" in href and "type=upjong" in href:
                sec = a.get_text(strip=True)
                if sec:
                    return sec
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  KIS (한국투자증권) API 관련 함수 — 기존 코드 유지, SSL 패치만 적용
# ─────────────────────────────────────────────────────────────────────────────
def _env_use_kis() -> bool:
    return os.environ.get("USE_KIS_UNIVERSE", "1").strip().lower() in ("1", "true", "yes", "y")


def _kis_get_token(base_url: str, app_key: str, app_secret: str) -> Optional[str]:
    url = base_url.rstrip("/") + "/oauth2/tokenP"
    try:
        res = _requests_post(
            url,
            json={
                "grant_type": "client_credentials",
                "appkey":     app_key,
                "appsecret":  app_secret,
            },
            headers={"Content-Type": "application/json"},
        )
        if res.status_code != 200:
            logging.warning("KIS tokenP failed: %s %s", res.status_code, res.text)
            return None
        data = res.json()
        access_token = data.get("access_token")
        if not access_token:
            logging.warning("KIS tokenP response missing access_token: %s", data)
            return None
        return access_token
    except Exception as e:
        logging.warning("KIS tokenP exception: %s", e)
        return None


def _kis_inquire_price(
    base_url: str,
    app_key: str,
    app_secret: str,
    access_token: str,
    code: str,
) -> Optional[dict]:
    """GET /uapi/domestic-stock/v1/quotations/inquire-price — output dict 반환."""
    url = base_url.rstrip("/") + "/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "Content-Type":  "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey":        app_key,
        "appsecret":     app_secret,
        "tr_id":         "FHKST01010100",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD":         code,
    }
    try:
        res = requests.get(url, headers=headers, params=params,
                           timeout=15, verify=SSL_VERIFY)
        if res.status_code != 200:
            logging.warning("KIS inquire-price failed(%s): %s %s", code, res.status_code, res.text)
            return None
        data = res.json()
        out = data.get("output") or data.get("output1") or {}
        if not isinstance(out, dict) or not out:
            logging.warning("KIS inquire-price empty(%s): %s", code, data)
            return None
        return out
    except Exception as e:
        logging.warning("KIS inquire-price exception(%s): %s", code, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  DB / CSV 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _load_universe_from_db() -> pd.DataFrame:
    """DB(Postgres → sqlite) 에서 universe 로드."""
    if get_engine:
        try:
            eng = get_engine()
            df  = pd.read_sql("SELECT code, name, market, sector FROM stocks", con=eng)
            if not df.empty:
                logging.info("Loaded universe from Postgres stocks (rows=%d)", len(df))
                return df
        except Exception:
            logging.warning("Failed to load universe from Postgres stocks", exc_info=True)

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql("SELECT code, name, market, sector FROM stocks", con=conn)
            if not df.empty:
                logging.info("Loaded universe from sqlite stocks (rows=%d)", len(df))
                return df
        except Exception:
            logging.warning("Failed to load universe from sqlite stocks", exc_info=True)

    return pd.DataFrame()


def _get_pg_table_columns(table: str) -> list[str]:
    if not get_engine:
        return []
    try:
        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table
                    ORDER BY ordinal_position
                    """
                ),
                {"table": table},
            ).fetchall()
        return [row[0] for row in rows]
    except Exception:
        logging.exception("Failed to inspect Postgres columns for %s", table)
        return []


def _get_sqlite_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        logging.exception("Failed to inspect sqlite columns for %s", table)
        return []
    return [row[1] for row in rows]


def _prepare_stocks_rows(df: pd.DataFrame, actual_columns: list[str]) -> pd.DataFrame:
    use_columns = [col for col in STOCKS_STORE_COLUMNS if col in actual_columns]
    out = df.copy()
    for col in use_columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[use_columns]


def _load_universe_candidates() -> pd.DataFrame:
    """
    전체 종목 코드 목록 구성 우선순위:
      1) 네이버 금융 시총 순위 (KOSPI + KOSDAQ 각 250개)  ← FDR 대체
      2) DB fallback
      3) universe.csv fallback
    반환 컬럼: code, name, market
    """
    # 1) 네이버 금융 크롤링 (FDR 대체)
    try:
        frames = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            df = _naver_fetch_market_top(mkt, top_n=250)
            if df is not None and not df.empty:
                frames.append(df)
        if frames:
            meta = pd.concat(frames, ignore_index=True)
            meta = meta.drop_duplicates(subset=["code"])
            meta = _filter_common_equities(meta, market="ALL", stage="candidate_naver")
            logging.info("Candidate universe from Naver Finance (rows=%d)", len(meta))
            return meta
    except Exception:
        logging.warning("Naver Finance candidate fetch failed", exc_info=True)

    # 2) DB fallback
    df = _load_universe_from_db()
    if not df.empty:
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        if "market" in df.columns:
            df["market"] = df["market"].astype(str).str.upper().str.strip()
        return _filter_common_equities(df, market="ALL", stage="candidate_db")

    # 3) universe.csv fallback
    if UNIVERSE_CSV.exists():
        try:
            df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
            if "code" in df.columns:
                df["code"] = df["code"].astype(str).str.zfill(6)
            if "market" in df.columns:
                df["market"] = df["market"].astype(str).str.upper().str.strip()
            return _filter_common_equities(df, market="ALL", stage="candidate_csv")
        except Exception:
            logging.warning("Failed to load universe.csv for candidate universe", exc_info=True)

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  last_trading_date — pykrx 대체 (네이버 금융 기준)
# ─────────────────────────────────────────────────────────────────────────────
def last_trading_date(max_back_days: int = 10) -> str:
    """
    네이버 금융 시총 순위 페이지에서 데이터가 존재하는지 확인해
    오늘 날짜를 거래일로 간주하여 반환한다.
    (네이버 금융은 항상 최신 거래일 기준 실시간 데이터를 제공하므로
     별도 날짜 탐색 없이 오늘 날짜를 사용해도 무방하다.)
    """
    today = datetime.today()
    url   = "https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1"
    for i in range(max_back_days + 1):
        dt  = today - timedelta(days=i)
        ymd = dt.strftime("%Y%m%d")
        try:
            resp = _requests_get(url)
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, "html.parser")
            table = soup.select_one("table.type_2")
            if table and table.select("tbody tr td a"):
                logging.info("last_trading_date → %s (네이버 금융 기준)", ymd)
                return ymd
        except Exception as e:
            logging.warning("last_trading_date 시도 실패 (%s): %s", ymd, e)
        time.sleep(0.2)

    return today.strftime("%Y%m%d")


# ─────────────────────────────────────────────────────────────────────────────
#  top_by_market — pykrx 대체 (네이버 금융 기준)
# ─────────────────────────────────────────────────────────────────────────────
def top_by_market(ymd: str, market: str, top_n: int) -> pd.DataFrame:
    """
    market: 'KOSPI' or 'KOSDAQ'
    반환: DataFrame with columns ['code', 'name', 'market']

    원본은 pykrx 기반이었으나 KRX API 차단 및 pykrx 내부 버그로 인해
    네이버 금융 시총 순위 크롤링으로 대체.
    (ymd 파라미터는 인터페이스 호환성 유지를 위해 보존하나 실제로는 미사용)
    """
    logging.info("[top_by_market] %s (기준일: %s, top_n=%d) 수집 시작", market, ymd, top_n)
    df = _naver_fetch_market_top(market, top_n=top_n)

    if df.empty:
        logging.warning("[top_by_market] %s: 네이버 금융에서 데이터를 가져오지 못했습니다.", market)
    else:
        if len(df) < top_n:
            logging.warning("[top_by_market] %s: requested=%d fetched=%d", market, top_n, len(df))
        logging.info("[top_by_market] %s: %d개 종목 수집 완료", market, len(df))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  top_by_market_kis — KIS API 기반 (기존 로직 유지, SSL 패치만 적용)
# ─────────────────────────────────────────────────────────────────────────────
def top_by_market_kis(market: str, top_n: int) -> pd.DataFrame:
    base_url   = os.getenv("KIS_BASE_URL")
    app_key    = os.getenv("KIS_APP_KEY")
    app_secret = os.getenv("KIS_APP_SECRET")

    if not base_url or not app_key or not app_secret:
        logging.warning("KIS env missing or incomplete -> skip KIS top_by_market")
        return pd.DataFrame(columns=["code", "name", "market"])

    candidates = _load_universe_candidates()
    if candidates.empty:
        logging.warning("No candidate universe available for KIS top_by_market")
        return pd.DataFrame(columns=["code", "name", "market"])

    mkt = market.upper().strip()
    if "market" in candidates.columns:
        cand = candidates[candidates["market"].astype(str).str.upper() == mkt].copy()
    else:
        cand = candidates.copy()

    if cand.empty:
        logging.warning("Candidate universe empty for market %s", mkt)
        return pd.DataFrame(columns=["code", "name", "market"])

    max_codes = os.getenv("KIS_UNIVERSE_MAX_CODES")
    if max_codes:
        try:
            cap = int(max_codes)
            if cap > 0:
                cand = cand.head(cap)
        except Exception:
            pass

    token = _kis_get_token(base_url, app_key, app_secret)
    if not token:
        return pd.DataFrame(columns=["code", "name", "market"])

    delay = 0.0
    try:
        delay = float(os.getenv("KIS_REQUEST_DELAY", "0"))
    except Exception:
        delay = 0.0

    rows = []
    for _, row in cand.iterrows():
        code = str(row.get("code", "")).zfill(6)
        if not code:
            continue
        out = _kis_inquire_price(base_url, app_key, app_secret, token, code)
        if not out:
            continue
        mcap = None
        for key in ("hts_avls", "stck_avls", "mkt_cap", "market_cap", "mktcap"):
            if key in out:
                mcap = _parse_mcap(out.get(key))
                if mcap:
                    break
        if not mcap:
            continue
        name = row.get("name") or out.get("prdt_name") or out.get("hts_kor_isnm") or ""
        rows.append({"code": code, "name": name, "market": mkt, "mcap": mcap})
        if delay > 0:
            time.sleep(delay)

    if not rows:
        logging.warning("KIS top_by_market produced no rows for %s", mkt)
        return pd.DataFrame(columns=["code", "name", "market"])

    df = pd.DataFrame(rows).sort_values("mcap", ascending=False).head(top_n)
    df = _filter_common_equities(df[["code", "name", "market"]], market=mkt, stage=f"kis_{mkt}")
    return df.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
#  업종(sector) 분류 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _classify_sector_by_name(name: str) -> str:
    """종목명 키워드 기반 대분류 업종 추정 (최후 fallback)."""
    n = (name or "").lower()
    if any(k in n for k in ["반도체", "하이닉스", "리노공", "테크윙", "하나마이크론",
                             "동진쎄미", "유진테크", "파두", "isc", "psk", "코미코",
                             "솔브레인", "주성엔지니어링"]):
        return "반도체"
    if any(k in n for k in ["전자", "elec", "lg이노텍", "삼성전기"]):
        return "전자/부품"
    if any(k in n for k in ["배터리", "에너지솔루션", "sdi", "엘앤에프",
                             "포스코퓨처엠", "엔켐", "레이크머티리얼즈"]):
        return "2차전지"
    if any(k in n for k in ["naver", "카카오", "카카오페이", "카카오뱅크",
                             "cj enm", "jyp", "와이지엔터", "스튜디오드래곤",
                             "넥슨게임즈", "위메이드", "카카오게임즈"]):
        return "인터넷/플랫폼·콘텐츠"
    if any(k in n for k in ["바이오", "제약", "셀트리온", "씨젠", "휴젤", "메지온",
                             "에스티팜", "엘앤씨바이오", "큐리언트", "알테오젠", "네이처셀"]):
        return "바이오/제약"
    if any(k in n for k in ["s-oil", "정유", "화학", "lg화학",
                             "포스코인터내셔널", "현대오일", "이노베이션"]):
        return "정유/화학"
    if any(k in n for k in ["현대차", "기아", "모비스", "오토에버", "글로비스", "한진칼"]):
        return "자동차/모빌리티"
    if any(k in n for k in ["조선", "현대미포", "마린솔루션", "ocean", "hmm"]):
        return "조선/해양·해운"
    if any(k in n for k in ["두산", "hd현대중공업", "한화오션", "한화에어로",
                             "lignex1", "한국항공우주", "로보틱스"]):
        return "기계/중공업·방산"
    if any(k in n for k in ["금융", "은행", "증권", "보험", "지주", "kb",
                             "신한", "하나금융", "bnk", "키움증권", "nh투자"]):
        return "금융"
    if any(k in n for k in ["통신", "sk텔레콤", "kt", "lg유플러스"]):
        return "통신"
    if any(k in n for k in ["아모레", "gs", "cj", "코웨이", "삼양식품", "맥쿼리인프라"]):
        return "소비재/유통"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  공통 초기화
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if load_dotenv:
        load_dotenv()
    setup_logging()
    ensure_data_dir()

    use_kis = _env_use_kis()
    market_date_override = _parse_market_date_override()
    if market_date_override:
        ymd = market_date_override
        logging.info("Using MARKET_DATE override for universe: %s", ymd)
    elif use_kis:
        ymd = datetime.today().strftime("%Y%m%d")
        logging.info("Using KIS universe mode (date=%s)", ymd)
    else:
        ymd = last_trading_date()
        logging.info("Using trading date: %s", ymd)

    # ── KOSPI Top100 ──────────────────────────────────────────────────────────
    try:
        if use_kis:
            kospi_top = top_by_market_kis("KOSPI", top_n=100)
        else:
            kospi_top = top_by_market(ymd, "KOSPI", top_n=100)
        logging.info("KOSPI top fetched: %d", len(kospi_top))
    except Exception as e:
        logging.exception("Failed to fetch KOSPI top: %s", e)
        kospi_top = pd.DataFrame(columns=["code", "name", "market"])

    # ── KOSDAQ Top100 ─────────────────────────────────────────────────────────
    try:
        if use_kis:
            kosdaq_top = top_by_market_kis("KOSDAQ", top_n=100)
        else:
            kosdaq_top = top_by_market(ymd, "KOSDAQ", top_n=100)
        logging.info("KOSDAQ top fetched: %d", len(kosdaq_top))
    except Exception as e:
        logging.exception("Failed to fetch KOSDAQ top: %s", e)
        kosdaq_top = pd.DataFrame(columns=["code", "name", "market"])

    # ── 합치기 & 중복 제거 ───────────────────────────────────────────────────
    uni = pd.concat([kospi_top, kosdaq_top], ignore_index=True)
    uni = uni.drop_duplicates(subset=["code"]).reset_index(drop=True)

    used_db_fallback = False
    if uni.empty:
        logging.error("Universe fetch returned empty; attempting DB fallback")
        uni = _load_universe_from_db()
        if uni.empty:
            raise RuntimeError("Universe fetch failed and DB fallback unavailable")
        used_db_fallback = True

    if "code" in uni.columns:
        uni["code"] = uni["code"].astype(str).str.zfill(6)
    if "market" in uni.columns:
        uni["market"] = uni["market"].astype(str).str.upper().str.strip()
    else:
        uni["market"] = ""
    uni = _filter_common_equities(uni, market="ALL", stage="final_universe")

    # ── sector 처리 ──────────────────────────────────────────────────────────
    if not used_db_fallback:
        old_sector_map: dict = {}
        try:
            if UNIVERSE_CSV.exists():
                old = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
                old["code"] = old["code"].astype(str).str.zfill(6)
                if "sector" in old.columns:
                    old_sector_map = dict(
                        zip(old["code"], old["sector"].fillna("").astype(str))
                    )
        except Exception:
            pass

        if SECTORS_CSV.exists():
            try:
                s = pd.read_csv(SECTORS_CSV, dtype={"code": str})
                s["code"] = s["code"].astype(str).str.zfill(6)
                if "sector" in s.columns:
                    uni = uni.merge(s[["code", "sector"]], on="code", how="left")
                else:
                    uni["sector"] = ""
            except Exception:
                uni["sector"] = ""
        else:
            if "sector" not in uni.columns:
                uni["sector"] = ""
            else:
                uni["sector"] = uni["sector"].fillna("").astype(str)
    else:
        if "sector" not in uni.columns:
            uni["sector"] = ""
        else:
            uni["sector"] = uni["sector"].fillna("").astype(str)

    # ── FDR 메타 병합 (사용 가능한 경우만) ───────────────────────────────────
    if fdr is not None:
        try:
            metas = []
            for market_id in ["KRX", "KOSPI", "KOSDAQ"]:
                try:
                    m = fdr.StockListing(market_id)
                    if m is not None and not m.empty:
                        m["__market_id__"] = market_id
                        metas.append(m)
                except Exception:
                    continue
            if metas:
                meta = pd.concat(metas, ignore_index=True)
                code_col = next(
                    (c for c in ["Code", "Symbol", "종목코드", "Ticker"] if c in meta.columns),
                    None
                )
                sector_col = next(
                    (c for c in meta.columns if "sector" in str(c).lower()),
                    None
                )
                if sector_col is None:
                    sector_col = next(
                        (c for c in meta.columns
                         if any(k in str(c).lower() for k in ["industry", "업종", "섹터", "산업"])),
                        None
                    )
                if code_col and sector_col:
                    meta = (
                        meta[[code_col, sector_col]]
                        .rename(columns={code_col: "code", sector_col: "sector_fdr"})
                    )
                    meta["code"] = meta["code"].astype(str).str.zfill(6)
                    uni = uni.merge(meta, on="code", how="left")
                    if "sector" not in uni.columns:
                        uni["sector"] = ""
                    mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
                    uni.loc[mask, "sector"] = uni.loc[mask, "sector_fdr"].fillna("").astype(str)
                    if "sector_fdr" in uni.columns:
                        uni.drop(columns=["sector_fdr"], inplace=True)
        except Exception:
            pass

    # ── 네이버 금융 업종 크롤링 fallback (빈 sector 보강, 최대 50개) ──────────
    try:
        mask          = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
        missing_codes = uni.loc[mask, "code"].astype(str).str.zfill(6).unique().tolist()[:50]
        if missing_codes:
            found: dict = {}
            for code in missing_codes:
                sec = _naver_fetch_sector(code)
                if sec:
                    found[code] = sec
                time.sleep(0.15)
            if found:
                mdf = pd.DataFrame(list(found.items()), columns=["code", "sector_nav"])
                uni = uni.merge(mdf, on="code", how="left")
                mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
                uni.loc[mask, "sector"] = uni.loc[mask, "sector_nav"].fillna("")
                if "sector_nav" in uni.columns:
                    uni.drop(columns=["sector_nav"], inplace=True)
                try:
                    if SECTORS_CSV.exists():
                        s = pd.read_csv(SECTORS_CSV, dtype={"code": str})
                        s["code"] = s["code"].astype(str).str.zfill(6)
                        s_map = dict(zip(s["code"], s.get("sector", pd.Series(dtype=str)).fillna("")))
                        s_map.update(found)
                        pd.DataFrame(list(s_map.items()), columns=["code", "sector"]).to_csv(
                            SECTORS_CSV, index=False, encoding="utf-8"
                        )
                    else:
                        pd.DataFrame(list(found.items()), columns=["code", "sector"]).to_csv(
                            SECTORS_CSV, index=False, encoding="utf-8"
                        )
                except Exception:
                    pass
    except Exception:
        pass

    # ── 종목명 키워드 기반 sector 추정 (최후 fallback) ────────────────────────
    try:
        mask_kw = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
        if mask_kw.any():
            uni.loc[mask_kw, "sector"] = (
                uni.loc[mask_kw, "name"]
                .apply(_classify_sector_by_name)
                .fillna("")
            )
    except Exception:
        pass

    # ── 이전 universe.csv 의 sector 보존 ─────────────────────────────────────
    if "sector" not in uni.columns:
        uni["sector"] = ""
    try:
        if old_sector_map:
            mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
            uni.loc[mask, "sector"] = uni.loc[mask, "code"].map(old_sector_map).fillna("")
        uni["sector"] = uni["sector"].fillna("").astype(str)
    except Exception:
        pass

    # ── 저장 컬럼 순서 고정 ───────────────────────────────────────────────────
    cols = [c for c in ["code", "name", "market", "sector"] if c in uni.columns]
    uni  = uni[cols]
    for col in STOCKS_STORE_COLUMNS:
        if col not in uni.columns:
            uni[col] = pd.NA
    uni = uni[STOCKS_STORE_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(uni, STOCKS_PK, "stocks")

    # ── sectors_template.csv 생성 (sectors.csv 미존재 시) ────────────────────
    try:
        template_path = DATA_DIR / "sectors_template.csv"
        if not SECTORS_CSV.exists():
            tmp = uni[["code"]].copy()
            tmp["sector"] = ""
            tmp.to_csv(template_path, index=False, encoding="utf-8")
    except Exception:
        pass

    # ── universe.csv 저장 ─────────────────────────────────────────────────────
    uni.to_csv(UNIVERSE_CSV, index=False, encoding="utf-8")
    logging.info("Saved universe: %s (rows=%d)", UNIVERSE_CSV.resolve(), len(uni))

    # ── DB upsert (Postgres → sqlite fallback) ────────────────────────────────
    try:
        if replace_table_rows_pg:
            pg_columns = _get_pg_table_columns("stocks")
            db_out = _prepare_stocks_rows(uni, pg_columns or STOCKS_STORE_COLUMNS)
            if ensure_unique_keys:
                ensure_unique_keys(db_out, STOCKS_PK, "stocks")
            _upsert_stocks_pg(db_out)
            logging.info("Upserted stocks rows in Postgres (rows=%d)", len(db_out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to sqlite")

    if not use_sqlite_fallback_writes():
        logging.info("Skipping sqlite fallback for stocks (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                code        TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                market      TEXT,
                sector      TEXT,
                listed_at   DATE,
                delisted_at DATE
            );
            """
        )
        sqlite_columns = _get_sqlite_table_columns(conn, "stocks")
        db_out = _prepare_stocks_rows(uni, sqlite_columns or STOCKS_STORE_COLUMNS)
        if ensure_unique_keys:
            ensure_unique_keys(db_out, STOCKS_PK, "stocks")
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "stocks", db_out)
        conn.commit()
        logging.info("Saved universe to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(uni))
    except Exception:
        logging.exception("Failed to save universe to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
