"""
DART 분기 재무데이터 수집 스크립트 — Financial Momentum Phase 1

수집 대상:
  - 1분기보고서 (Q1, reprt_code=11011)
  - 반기보고서   (Q2, reprt_code=11012)
  - 3분기보고서  (Q3, reprt_code=11013)
  - 사업보고서   (Q4, reprt_code=11014)

주의:
  - DART API는 분기 누적값을 반환한다 (Q1=1~3월, Q2=1~6월, Q3=1~9월, Q4=1~12월).
  - 진짜 분기값(true_*) 계산은 Phase 2 feature builder에서 수행한다.
  - disclosed_at: 법정 공시 기한 기준 추정값. 실제 공시일과 다를 수 있으므로
    백테스트에서 보수적으로 사용해야 한다.

출력:
  - data/dart/financial_quarterly.csv  (로컬 캐시)
  - DB 테이블 financial_statement_quarterly (Postgres 우선, SQLite fallback)
"""

import argparse
import logging
import os
import sqlite3
import time
import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from db import get_engine, replace_table_rows_pg, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    replace_table_rows_pg = None

    def use_sqlite_fallback_writes() -> bool:
        return False


# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "dart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

QUARTERLY_CSV = CACHE_DIR / "financial_quarterly.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

DB_TABLE = "financial_statement_quarterly"

# ---------------------------------------------------------------------------
# DART API 설정
# ---------------------------------------------------------------------------
DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DART_FNLTT_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

# reprt_code → (quarter 라벨, 분기 종료 월일, 법정 공시 기한 오프셋 일수)
QUARTER_SPECS: Dict[str, Tuple[str, str, int]] = {
    "11011": ("Q1", "-03-31", 45),   # 1분기: 3/31 기준, ~5/15 공시
    "11012": ("Q2", "-06-30", 45),   # 반기:  6/30 기준, ~8/14 공시
    "11013": ("Q3", "-09-30", 45),   # 3분기: 9/30 기준, ~11/14 공시
    "11014": ("Q4", "-12-31", 90),   # 연간: 12/31 기준, ~3/31 공시
}

# 연결(CFS) 우선, 별도(OFS) fallback
FS_ORDER = ["CFS", "OFS"]

# 계정명 후보 (한국어 기준)
ACC_CANDIDATES = {
    "revenue":    ["매출액", "영업수익", "매출"],
    "op_income":  ["영업이익"],
    "net_income": ["당기순이익"],
    "assets":     ["자산총계"],
    "equity":     ["자본총계"],
    "liabilities":["부채총계"],
    "ocf":        ["영업활동으로인한현금흐름", "영업활동현금흐름", "영업활동으로 인한 현금흐름"],
}

# DB 컬럼 순서
DB_COLUMNS = [
    "stock_code", "fiscal_year", "quarter", "report_code",
    "source_report_date", "disclosed_at",
    "revenue", "op_income", "net_income",
    "assets", "liabilities", "equity", "ocf",
    "op_margin", "debt_ratio", "net_margin", "ocf_to_assets",
    "fs_div",
]

DEFAULT_YEARS_BACK = 5
DEFAULT_SLEEP_SEC = 0.2
DEFAULT_LOG_EVERY = 20
DEFAULT_CHECKPOINT_EVERY = 100


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("DART_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DART_API_KEY not set in .env")
    return key


def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    params: Dict,
    expect_json: bool = True,
    retries: int = 3,
    sleep_sec: float = 0.6,
):
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json() if expect_json else r.content
            logging.warning("HTTP %s url=%s", r.status_code, url)
        except Exception as e:
            logging.warning("request error (attempt %d/%d): %s", i + 1, retries, e)
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def normalize_account_name(s: str) -> str:
    return (s or "").replace(" ", "").replace("　", "")


def parse_amount(x) -> Optional[float]:
    s = str(x or "").strip().replace(",", "")
    if s in ("", "-", "NaN", "nan", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def value_by_candidates(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    if df.empty:
        return None
    m = df.copy()
    m["acc"] = m["account_nm"].astype(str).map(normalize_account_name)
    for c in candidates:
        key = normalize_account_name(c)
        hit = m[m["acc"].str.contains(key, na=False)]
        if not hit.empty:
            vals = hit["thstrm_amount"].map(parse_amount).dropna()
            if not vals.empty:
                return float(vals.iloc[0])
    return None


# ---------------------------------------------------------------------------
# corp_code 매핑
# ---------------------------------------------------------------------------

def download_corp_codes(api_key: str, session: requests.Session) -> Path:
    cache_xml = CACHE_DIR / "corp_codes.xml"
    if (
        cache_xml.exists()
        and cache_xml.stat().st_size > 0
        and (time.time() - cache_xml.stat().st_mtime) < 86400 * 7
    ):
        logging.info("Using cached corp_codes.xml")
        return cache_xml

    logging.info("Downloading corpCode.zip from DART...")
    content = request_with_retry(
        session, DART_CORP_CODE_URL, {"crtfc_key": api_key}, expect_json=False
    )
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if not xml_name:
            raise RuntimeError("corp_codes zip has no XML")
        xml_bytes = zf.open(xml_name).read()
    cache_xml.write_bytes(xml_bytes)
    return cache_xml


def parse_corp_codes(xml_path: Path) -> Dict[str, str]:
    """stock_code → corp_code 매핑 반환"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mapping: Dict[str, str] = {}
    for el in root.findall(".//list"):
        corp_code = (el.findtext("corp_code") or "").strip()
        stock_code = (el.findtext("stock_code") or "").strip().zfill(6)
        if corp_code and stock_code and stock_code != "000000":
            mapping[stock_code] = corp_code
    return mapping


# ---------------------------------------------------------------------------
# 대상 종목 코드 로드
# ---------------------------------------------------------------------------

def get_target_codes() -> List[str]:
    for csv_path in (UNIVERSE_CSV, FEATURES_CSV):
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"code": str})
            if "code" in df.columns:
                codes = sorted(df["code"].dropna().astype(str).str.zfill(6).unique())
                if codes:
                    logging.info("Target codes from %s: %d", csv_path.name, len(codes))
                    return codes
    raise RuntimeError("No target codes found in universe.csv or features.csv")


# ---------------------------------------------------------------------------
# 기존 캐시 로드
# ---------------------------------------------------------------------------

def load_existing_cache() -> pd.DataFrame:
    if not QUARTERLY_CSV.exists():
        return pd.DataFrame(columns=DB_COLUMNS)
    try:
        df = pd.read_csv(QUARTERLY_CSV, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        return df
    except Exception:
        logging.warning("Failed to read existing quarterly cache")
        return pd.DataFrame(columns=DB_COLUMNS)


# ---------------------------------------------------------------------------
# DART 분기 재무 조회
# ---------------------------------------------------------------------------

def estimate_disclosed_at(fiscal_year: int, quarter: str, offset_days: int) -> str:
    """법정 기한 기준 추정 공시일. 실제 공시일이 아님을 주의."""
    month_day = {
        "Q1": (3, 31),
        "Q2": (6, 30),
        "Q3": (9, 30),
        "Q4": (12, 31),
    }[quarter]
    year = fiscal_year if quarter != "Q4" else fiscal_year + 1
    fiscal_end = date(year if quarter != "Q4" else fiscal_year, month_day[0], month_day[1])
    # Q4는 다음 해 공시
    if quarter == "Q4":
        fiscal_end = date(fiscal_year, 12, 31)
        disclosed = date(fiscal_year + 1, 3, 31)
    else:
        disclosed = fiscal_end + timedelta(days=offset_days)
    return disclosed.strftime("%Y-%m-%d")


def fetch_quarter_financials(
    api_key: str,
    corp_code: str,
    fiscal_year: int,
    reprt_code: str,
    session: requests.Session,
) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """단일 분기 재무 조회. (결과 dict, fs_div, status) 반환"""
    for fs_div in FS_ORDER:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(fiscal_year),
            "reprt_code": reprt_code,
            "fs_div": fs_div,
        }
        try:
            data = request_with_retry(session, DART_FNLTT_URL, params)
        except Exception as e:
            logging.debug("request failed corp=%s year=%s reprt=%s: %s", corp_code, fiscal_year, reprt_code, e)
            continue

        if not isinstance(data, dict) or data.get("status") != "000":
            continue

        ls = data.get("list") or []
        df = pd.DataFrame(ls)
        if df.empty:
            continue

        fin = {
            "revenue":    value_by_candidates(df, ACC_CANDIDATES["revenue"]),
            "op_income":  value_by_candidates(df, ACC_CANDIDATES["op_income"]),
            "net_income": value_by_candidates(df, ACC_CANDIDATES["net_income"]),
            "assets":     value_by_candidates(df, ACC_CANDIDATES["assets"]),
            "equity":     value_by_candidates(df, ACC_CANDIDATES["equity"]),
            "liabilities":value_by_candidates(df, ACC_CANDIDATES["liabilities"]),
            "ocf":        value_by_candidates(df, ACC_CANDIDATES["ocf"]),
        }
        return fin, fs_div, "000"

    return None, None, "NOT_FOUND"


# ---------------------------------------------------------------------------
# 파생 비율 계산
# ---------------------------------------------------------------------------

def compute_ratios(row: Dict) -> Dict:
    revenue    = row.get("revenue")
    op_income  = row.get("op_income")
    net_income = row.get("net_income")
    assets     = row.get("assets")
    equity     = row.get("equity")
    liabilities= row.get("liabilities")
    ocf        = row.get("ocf")

    def safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    return {
        "op_margin":     safe_div(op_income, revenue),
        "debt_ratio":    safe_div(liabilities, equity),
        "net_margin":    safe_div(net_income, revenue),
        "ocf_to_assets": safe_div(ocf, assets),
    }


# ---------------------------------------------------------------------------
# 수집 메인 로직
# ---------------------------------------------------------------------------

def build_fetch_plan(
    codes: List[str],
    existing: pd.DataFrame,
    years_back: int,
    refresh_recent_years: int,
) -> Dict[Tuple[str, int, str], str]:
    """(stock_code, fiscal_year, quarter) → reprt_code 형태의 수집 계획 반환"""
    today = date.today()
    end_year = today.year
    start_year = end_year - (years_back - 1)

    covered: set = set()
    if not existing.empty and {"stock_code", "fiscal_year", "quarter"}.issubset(existing.columns):
        for _, r in existing.iterrows():
            covered.add((str(r["stock_code"]).zfill(6), int(r["fiscal_year"]), str(r["quarter"])))

    plan: Dict[Tuple[str, int, str], str] = {}
    for code in codes:
        for year in range(start_year, end_year + 1):
            for reprt_code, (quarter, _, _) in QUARTER_SPECS.items():
                key = (code, year, quarter)
                # Q4는 다음 해에야 공시되므로 현재 연도 Q4는 스킵
                if quarter == "Q4" and year == end_year:
                    continue
                # refresh_recent_years 강제 재수집
                if refresh_recent_years > 0 and year >= end_year - refresh_recent_years:
                    plan[key] = reprt_code
                    continue
                if key not in covered:
                    plan[key] = reprt_code

    return plan


def collect_quarterly(
    api_key: str,
    codes: List[str],
    years_back: int = DEFAULT_YEARS_BACK,
    refresh_recent_years: int = 0,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    log_every: int = DEFAULT_LOG_EVERY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> pd.DataFrame:

    session = build_http_session()
    xml_path = download_corp_codes(api_key, session)
    code_map = parse_corp_codes(xml_path)

    existing = load_existing_cache()
    plan = build_fetch_plan(codes, existing, years_back, refresh_recent_years)

    logging.info(
        "Quarterly fetch plan: codes=%d, planned=%d, existing_rows=%d",
        len(codes), len(plan), len(existing),
    )

    rows: List[Dict] = []
    total = len(plan)
    ok = skip = fail = 0
    start_ts = time.perf_counter()

    for idx, ((stock_code, fiscal_year, quarter), reprt_code) in enumerate(plan.items(), 1):
        corp_code = code_map.get(stock_code)
        if not corp_code:
            skip += 1
            continue

        _, _, offset_days = QUARTER_SPECS[reprt_code]
        fin, fs_div, status = fetch_quarter_financials(
            api_key, corp_code, fiscal_year, reprt_code, session
        )

        if fin is not None:
            ratios = compute_ratios(fin)
            source_report_date = f"{fiscal_year}{QUARTER_SPECS[reprt_code][1]}"
            # Q4는 다음 해 공시
            disclosed_at = estimate_disclosed_at(fiscal_year, quarter, offset_days)

            rows.append({
                "stock_code":        stock_code,
                "fiscal_year":       fiscal_year,
                "quarter":           quarter,
                "report_code":       reprt_code,
                "source_report_date":source_report_date,
                "disclosed_at":      disclosed_at,
                "revenue":           fin.get("revenue"),
                "op_income":         fin.get("op_income"),
                "net_income":        fin.get("net_income"),
                "assets":            fin.get("assets"),
                "liabilities":       fin.get("liabilities"),
                "equity":            fin.get("equity"),
                "ocf":               fin.get("ocf"),
                "op_margin":         ratios.get("op_margin"),
                "debt_ratio":        ratios.get("debt_ratio"),
                "net_margin":        ratios.get("net_margin"),
                "ocf_to_assets":     ratios.get("ocf_to_assets"),
                "fs_div":            fs_div,
            })
            ok += 1
        else:
            fail += 1

        if idx % max(1, log_every) == 0 or idx == total:
            elapsed = time.perf_counter() - start_ts
            eta = (elapsed / idx) * (total - idx)
            m, s = divmod(int(eta), 60)
            logging.info(
                "[DART-Q] %d/%d (%.1f%%) code=%s year=%s q=%s ok=%d skip=%d fail=%d ETA=%02d:%02d",
                idx, total, idx / total * 100,
                stock_code, fiscal_year, quarter, ok, skip, fail, m, s,
            )

        if checkpoint_every > 0 and rows and idx % checkpoint_every == 0:
            merged = _merge_with_existing(existing, rows)
            save_quarterly(merged)
            logging.info("Checkpoint saved: rows=%d", len(merged))

        time.sleep(sleep_sec)

    elapsed = time.perf_counter() - start_ts
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logging.info(
        "Done: total=%d ok=%d skip=%d fail=%d elapsed=%02d:%02d:%02d",
        total, ok, skip, fail, h, m, s,
    )

    if not rows:
        if not existing.empty:
            logging.info("No new rows fetched — reusing existing cache")
            return existing
        raise RuntimeError("No quarterly financial rows fetched from DART")

    return _merge_with_existing(existing, rows)


def _merge_with_existing(existing: pd.DataFrame, new_rows: List[Dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df.drop_duplicates(subset=["stock_code", "fiscal_year", "quarter"], keep="last")
    merged = (
        pd.concat([existing, new_df], ignore_index=True)
        .drop_duplicates(subset=["stock_code", "fiscal_year", "quarter"], keep="last")
        .sort_values(["stock_code", "fiscal_year", "quarter"])
        .reset_index(drop=True)
    )
    return merged


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def save_quarterly(df: pd.DataFrame) -> None:
    out = df.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    for col in DB_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[DB_COLUMNS]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(QUARTERLY_CSV, index=False, encoding="utf-8")
    logging.info("Saved quarterly CSV: %s (rows=%d)", QUARTERLY_CSV.resolve(), len(out))

    # Postgres 저장
    try:
        if replace_table_rows_pg and get_engine:
            eng = get_engine()
            with eng.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS financial_statement_quarterly (
                        stock_code          TEXT NOT NULL,
                        fiscal_year         INTEGER NOT NULL,
                        quarter             TEXT NOT NULL,
                        report_code         TEXT,
                        source_report_date  DATE,
                        disclosed_at        DATE,
                        revenue             DOUBLE PRECISION,
                        op_income           DOUBLE PRECISION,
                        net_income          DOUBLE PRECISION,
                        assets              DOUBLE PRECISION,
                        liabilities         DOUBLE PRECISION,
                        equity              DOUBLE PRECISION,
                        ocf                 DOUBLE PRECISION,
                        op_margin           DOUBLE PRECISION,
                        debt_ratio          DOUBLE PRECISION,
                        net_margin          DOUBLE PRECISION,
                        ocf_to_assets       DOUBLE PRECISION,
                        fs_div              TEXT,
                        created_at          TIMESTAMP DEFAULT now(),
                        updated_at          TIMESTAMP DEFAULT now(),
                        PRIMARY KEY (stock_code, fiscal_year, quarter)
                    )
                """))
            replace_table_rows_pg(DB_TABLE, out, columns=DB_COLUMNS)
            logging.info("Saved to Postgres: %s (rows=%d)", DB_TABLE, len(out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to SQLite")

    if not use_sqlite_fallback_writes():
        logging.info("SQLite fallback disabled (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    _save_sqlite(out)


def _save_sqlite(df: pd.DataFrame) -> None:
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_statement_quarterly (
                stock_code          TEXT NOT NULL,
                fiscal_year         INTEGER NOT NULL,
                quarter             TEXT NOT NULL,
                report_code         TEXT,
                source_report_date  TEXT,
                disclosed_at        TEXT,
                revenue             REAL,
                op_income           REAL,
                net_income          REAL,
                assets              REAL,
                liabilities         REAL,
                equity              REAL,
                ocf                 REAL,
                op_margin           REAL,
                debt_ratio          REAL,
                net_margin          REAL,
                ocf_to_assets       REAL,
                fs_div              TEXT,
                PRIMARY KEY (stock_code, fiscal_year, quarter)
            )
        """)
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(financial_statement_quarterly)").fetchall()}
        new_cols = {
            "report_code": "TEXT", "source_report_date": "TEXT", "disclosed_at": "TEXT",
            "revenue": "REAL", "op_income": "REAL", "net_income": "REAL",
            "assets": "REAL", "liabilities": "REAL", "equity": "REAL", "ocf": "REAL",
            "op_margin": "REAL", "debt_ratio": "REAL", "net_margin": "REAL",
            "ocf_to_assets": "REAL", "fs_div": "TEXT",
        }
        for col, col_type in new_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE financial_statement_quarterly ADD COLUMN {col} {col_type}")

        placeholders = ", ".join(f":{c}" for c in DB_COLUMNS)
        col_names = ", ".join(DB_COLUMNS)
        conn.executemany(
            f"INSERT OR REPLACE INTO financial_statement_quarterly ({col_names}) VALUES ({placeholders})",
            df.to_dict(orient="records"),
        )
        conn.commit()
        logging.info("Saved to SQLite: financial_statement_quarterly (rows=%d)", len(df))
    except Exception:
        logging.exception("SQLite save failed")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DART 분기 재무데이터 수집")
    p.add_argument("--years-back", type=int,
                   default=int(os.getenv("DART_QUARTERLY_YEARS_BACK", str(DEFAULT_YEARS_BACK))),
                   help="수집 기간(연도 수, 기본: 5)")
    p.add_argument("--refresh-recent-years", type=int,
                   default=int(os.getenv("DART_QUARTERLY_REFRESH_RECENT_YEARS", "0")),
                   help="최근 N년 강제 재수집")
    p.add_argument("--sleep-sec", type=float,
                   default=float(os.getenv("DART_QUARTERLY_SLEEP_SEC", str(DEFAULT_SLEEP_SEC))),
                   help="API 호출 간 슬립(초, 기본: 0.2)")
    p.add_argument("--log-every", type=int,
                   default=int(os.getenv("DART_QUARTERLY_LOG_EVERY", str(DEFAULT_LOG_EVERY))),
                   help="로그 출력 주기")
    p.add_argument("--checkpoint-every", type=int,
                   default=int(os.getenv("DART_QUARTERLY_CHECKPOINT_EVERY", str(DEFAULT_CHECKPOINT_EVERY))),
                   help="중간 저장 주기(0=비활성)")
    p.add_argument("--codes", type=str, default="",
                   help="쉼표 구분 종목코드 직접 지정 (미지정 시 universe.csv 사용)")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    api_key = load_api_key()

    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        logging.info("Using codes from argument: %d", len(codes))
    else:
        codes = get_target_codes()

    df = collect_quarterly(
        api_key=api_key,
        codes=codes,
        years_back=args.years_back,
        refresh_recent_years=args.refresh_recent_years,
        sleep_sec=args.sleep_sec,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )
    save_quarterly(df)
    logging.info("완료: financial_statement_quarterly rows=%d", len(df))


if __name__ == "__main__":
    main()
