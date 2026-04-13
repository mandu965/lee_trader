import argparse
import csv
import io
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os
import sqlite3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg
except Exception:
    ensure_unique_keys = None
    get_engine = None
    replace_table_rows_pg = None

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "dart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FUND_OUT = DATA_DIR / "fundamentals.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
FUNDAMENTALS_PK = ["date", "code"]
FUNDAMENTALS_DB_COLUMNS = ["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"]

DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DART_FNLTT_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

# 보고서 코드: 11011(1Q), 11012(반기), 11013(3Q), 11014(사업보고서, 연말)
REPRT_CODE_ANNUAL = "11014"  # 사업보고서(연말)만 사용하여 연간 값으로 일관성 유지

# prefer 연결(CFS), fallback 별도(OFS)
FS_ORDER = ["CFS", "OFS"]

# 강건한 계정명 후보(한국어명 기준)
ACC_CANDIDATES = {
    "revenue": ["매출액", "영업수익", "매출"],
    "op_income": ["영업이익"],
    "net_income": ["당기순이익"],
    "assets": ["자산총계"],
    "equity": ["자본총계"],
    "liabilities": ["부채총계"],
    "ocf": ["영업활동으로인한현금흐름", "영업활동현금흐름", "영업활동으로 인한 현금흐름"],
}

# 수집 기간(연도)
DEFAULT_YEARS_BACK = 7  # 최근 7년(필요시 조정)


DEFAULT_REFRESH_RECENT_YEARS = 0
DEFAULT_SLEEP_SEC = 0.15
DEFAULT_LOG_EVERY = 20
DEFAULT_CHECKPOINT_EVERY = 100


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DART_API_KEY not found in .env. Please set DART_API_KEY=...")
    return api_key


def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    params: Dict[str, str],
    expect_json: bool = True,
    retries: int = 3,
    sleep_sec: float = 0.6,
):
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json() if expect_json else r.content
            logging.warning("HTTP %s for %s params=%s", r.status_code, url, params)
        except Exception as e:
            logging.warning("request error: %s (attempt %d/%d)", e, i + 1, retries)
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed request after {retries} retries: {url} params={params}")


def download_and_cache_corp_codes(api_key: str, session: requests.Session) -> Path:
    cache_xml = CACHE_DIR / "corp_codes.xml"
    if (
        cache_xml.exists()
        and cache_xml.stat().st_size > 0
        and (time.time() - cache_xml.stat().st_mtime) < 86400 * 7
    ):
        logging.info("Using cached corp_codes.xml: %s", cache_xml.resolve())
        return cache_xml

    logging.info("Downloading corpCode.zip from DART...")
    content = request_with_retry(session, DART_CORP_CODE_URL, params={"crtfc_key": api_key}, expect_json=False)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if not xml_name:
            raise RuntimeError("corpCode zip did not contain an XML file.")
        with zf.open(xml_name) as f:
            xml_bytes = f.read()
    cache_xml.write_bytes(xml_bytes)
    logging.info("Saved corp codes XML: %s", cache_xml.resolve())
    return cache_xml


def parse_corp_codes(xml_path: Path) -> pd.DataFrame:
    # 반환: columns = corp_code, stock_code, corp_name
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for el in root.findall(".//list"):
        corp_code = (el.findtext("corp_code") or "").strip()
        stock_code = (el.findtext("stock_code") or "").strip()  # 6자리 종목코드(없을 수도 있음)
        corp_name = (el.findtext("corp_name") or "").strip()
        if corp_code:
            rows.append({"corp_code": corp_code, "stock_code": stock_code, "corp_name": corp_name})
    df = pd.DataFrame(rows)
    return df


def get_target_codes() -> List[str]:
    # 우선 universe.csv의 code 사용. 없으면 features.csv에서 고유 code 목록 사용.
    if UNIVERSE_CSV.exists():
        uni = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
        if "code" in uni.columns:
            codes = sorted(pd.Series(uni["code"].dropna().astype(str).str.zfill(6)).unique())
            if codes:
                return codes
    if FEATURES_CSV.exists():
        feats = pd.read_csv(FEATURES_CSV, dtype={"code": str})
        if "code" in feats.columns:
            codes = sorted(pd.Series(feats["code"].dropna().astype(str).str.zfill(6)).unique())
            if codes:
                return codes
    raise RuntimeError("No target codes found in universe.csv or features.csv")


def normalize_account_name(s: str) -> str:
    return (s or "").replace(" ", "").replace("\u3000", "")  # remove spaces and ideographic space


def parse_amount(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
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
            # thstrm_amount(당기) 우선
            vals = hit["thstrm_amount"].map(parse_amount).dropna()
            if not vals.empty:
                return float(vals.iloc[0])
    return None


def fetch_annual_financials(
    api_key: str, corp_code: str, year: int, session: requests.Session
) -> Tuple[Optional[Dict[str, Optional[float]]], Optional[str], Optional[str]]:
    """연간 재무 조회. (결과 dict, 사용된 fs_div, status) 반환"""
    last_status: Optional[str] = None
    for fs_div in FS_ORDER:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": REPRT_CODE_ANNUAL,
            "fs_div": fs_div,
        }
        try:
            data = request_with_retry(session, DART_FNLTT_URL, params=params, expect_json=True)
        except Exception as e:
            logging.debug("request failed corp=%s year=%s fs=%s err=%s", corp_code, year, fs_div, e)
            last_status = "REQ_ERR"
            continue
        if not isinstance(data, dict):
            last_status = "RESP_NODICT"
            continue
        status = data.get("status")
        last_status = status or "UNK"
        if status != "000":
            # 013: 조회 데이터 없음 등
            continue
        ls = data.get("list") or []
        df = pd.DataFrame(ls)
        if df.empty:
            last_status = "EMPTY"
            continue

        fin = {
            "revenue": value_by_candidates(df, ACC_CANDIDATES["revenue"]),
            "op_income": value_by_candidates(df, ACC_CANDIDATES["op_income"]),
            "net_income": value_by_candidates(df, ACC_CANDIDATES["net_income"]),
            "assets": value_by_candidates(df, ACC_CANDIDATES["assets"]),
            "equity": value_by_candidates(df, ACC_CANDIDATES["equity"]),
            "liabilities": value_by_candidates(df, ACC_CANDIDATES["liabilities"]),
            "ocf": value_by_candidates(df, ACC_CANDIDATES["ocf"]),
        }
        return fin, fs_div, last_status
    return None, None, last_status


def load_existing_fundamentals() -> pd.DataFrame:
    if not FUND_OUT.exists():
        return pd.DataFrame(columns=["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"])
    try:
        existing = pd.read_csv(FUND_OUT, dtype={"code": str})
    except Exception:
        logging.warning("Failed to read existing fundamentals cache: %s", FUND_OUT.resolve())
        return pd.DataFrame(columns=["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"])
    if existing.empty or "code" not in existing.columns or "date" not in existing.columns:
        return pd.DataFrame(columns=["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"])
    existing = existing.copy()
    existing["code"] = existing["code"].astype(str).str.zfill(6)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    existing = existing.dropna(subset=["date", "code"])
    keep_cols = ["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"]
    return existing.loc[:, [c for c in keep_cols if c in existing.columns]].copy()


def finalize_fundamentals_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=FUNDAMENTALS_DB_COLUMNS)

    df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
    for c in ["revenue", "op_income", "net_income", "assets", "equity", "liabilities", "ocf"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["equity_avg"] = (g["equity"].shift(1) + g["equity"]) / 2.0
        g["assets_avg"] = (g["assets"].shift(1) + g["assets"]) / 2.0
        g["equity_basis"] = np.where(g["equity_avg"].abs() > 1e-9, g["equity_avg"], g["equity"])
        g["assets_basis"] = np.where(g["assets_avg"].abs() > 1e-9, g["assets_avg"], g["assets"])
        g["roe"] = np.where((g["equity_basis"].abs() > 1e-9), g["net_income"] / g["equity_basis"], np.nan)
        g["op_margin"] = np.where((g["revenue"].abs() > 1e-9), g["op_income"] / g["revenue"], np.nan)
        g["debt_ratio"] = np.where((g["equity"].abs() > 1e-9), g["liabilities"] / g["equity"], np.nan)
        g["ocf_to_assets"] = np.where((g["assets_basis"].abs() > 1e-9), g["ocf"] / g["assets_basis"], np.nan)
        g["net_margin"] = np.where((g["revenue"].abs() > 1e-9), g["net_income"] / g["revenue"], np.nan)
        return g

    df = df.groupby("code", group_keys=False).apply(per_code)
    return df[FUNDAMENTALS_DB_COLUMNS].copy()


def merge_fundamentals_with_existing(
    existing_out: pd.DataFrame,
    fetched_rows: List[Dict[str, object]],
    fetch_plan: Dict[str, int],
) -> pd.DataFrame:
    if not fetched_rows:
        if existing_out.empty:
            return pd.DataFrame(columns=FUNDAMENTALS_DB_COLUMNS)
        return existing_out.sort_values(["code", "date"]).reset_index(drop=True)

    out = finalize_fundamentals_rows(fetched_rows)
    if existing_out.empty:
        return out.sort_values(["code", "date"]).reset_index(drop=True)

    refresh_frames = []
    for code, fetch_start_year in fetch_plan.items():
        refresh_frames.append((code, pd.Timestamp(f"{fetch_start_year}-01-01")))
    retained = existing_out.copy()
    for code, cutoff_date in refresh_frames:
        retained = retained.loc[~((retained["code"] == code) & (retained["date"] >= cutoff_date))]
    merged = (
        pd.concat([retained, out], ignore_index=True)
        .drop_duplicates(subset=["date", "code"], keep="last")
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    return merged


def build_fundamentals_from_dart(
    api_key: str,
    codes: List[str],
    years_back: int = DEFAULT_YEARS_BACK,
    refresh_recent_years: int = 0,
    sleep_sec: float = 0.15,
    log_every: int = 20,
    checkpoint_every: int = 0,
    raw_log: bool = False,
) -> pd.DataFrame:
    session = build_http_session()
    xml_path = download_and_cache_corp_codes(api_key, session)
    corp_df = parse_corp_codes(xml_path)
    corp_df["stock_code"] = corp_df["stock_code"].astype(str).str.zfill(6)
    code_map = dict(zip(corp_df["stock_code"], corp_df["corp_code"]))

    today = datetime.today()
    end_year = today.year
    start_year = end_year - (years_back - 1)
    existing_out = load_existing_fundamentals()
    fetch_plan: Dict[str, int] = {}
    cached_code_count = 0
    for code in codes:
        covered_years: set[int] = set()
        if not existing_out.empty:
            covered_years = set(
                existing_out.loc[existing_out["code"] == code, "date"].dt.year.dropna().astype(int).tolist()
            )
        missing_years = [year for year in range(start_year, end_year + 1) if year not in covered_years]
        forced_refresh_start_year = None
        if refresh_recent_years > 0:
            forced_refresh_start_year = max(start_year, end_year - refresh_recent_years)
        if not missing_years:
            if forced_refresh_start_year is None:
                cached_code_count += 1
                continue
            fetch_plan[code] = forced_refresh_start_year
            continue
        planned_start_year = max(start_year, min(missing_years) - 1)
        if forced_refresh_start_year is not None:
            planned_start_year = min(planned_start_year, forced_refresh_start_year)
        fetch_plan[code] = planned_start_year

    unmapped_codes: List[str] = []
    rows: List[Dict[str, object]] = []

    # 진행률/ETA 집계
    total_iters = sum((end_year - fetch_start_year + 1) for fetch_start_year in fetch_plan.values())
    iter_idx = 0
    ok_count = 0
    empty_count = 0
    fail_count = 0
    cached_hits = 1  # corp_codes.xml 캐시 활용 지표
    api_calls = 0
    total_ms = 0.0
    start_ts = time.perf_counter()
    checkpoint_count = 0
    logging.info(
        "Fundamentals fetch plan: target_codes=%d cached_codes=%d fetch_codes=%d planned_calls=%d window=%d-%d",
        len(codes),
        cached_code_count,
        len(fetch_plan),
        total_iters,
        start_year,
        end_year,
    )

    # raw 로그 설정
    csv_path: Optional[Path] = None
    csv_writer: Optional[csv.writer] = None
    csv_file = None
    if raw_log:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = CACHE_DIR / f"fetch_log_{ts}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["ts", "code", "corp_code", "year", "fs_used", "status", "duration_ms", "ok"]
        )

    for code in codes:
        fetch_start_year = fetch_plan.get(code)
        if fetch_start_year is None:
            continue
        corp_code = code_map.get(code)
        if not corp_code:
            unmapped_codes.append(code)
            logging.info("No corp_code for stock %s (skip)", code)
            continue

        for y in range(fetch_start_year, end_year + 1):
            t0 = time.perf_counter()
            fin, fs_used, status = fetch_annual_financials(api_key, corp_code, y, session)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += dt_ms
            api_calls += 1
            iter_idx += 1

            ok = fin is not None
            if ok:
                ok_count += 1
                rows.append({"date": pd.Timestamp(f"{y}-12-31"), "code": code, **fin})
            else:
                # status가 EMPTY, 013 등일 수 있음
                if status in {"EMPTY", "013"}:
                    empty_count += 1
                else:
                    fail_count += 1

            # 진행 로그(주기)
            if (iter_idx % max(1, log_every)) == 0 or iter_idx == total_iters:
                avg_ms = total_ms / max(1, api_calls)
                remain = max(0, total_iters - iter_idx)
                eta_sec = (avg_ms / 1000.0) * remain
                m, s = divmod(int(eta_sec), 60)
                h, m = divmod(m, 60)
                pct = (iter_idx / total_iters) * 100.0
                logging.info(
                    "[DART] %d/%d (%.1f%%) code=%s year=%s fs=%s dt=%.0fms avg=%.0fms ETA=%02d:%02d:%02d ok=%d empty=%d fail=%d unmapped=%d cache_xml=on",
                    iter_idx,
                    total_iters,
                    pct,
                    code,
                    y,
                    fs_used or "-",
                    dt_ms,
                    avg_ms,
                    h,
                    m,
                    s,
                    ok_count,
                    empty_count,
                    fail_count,
                    len(unmapped_codes),
                )

            if checkpoint_every > 0 and rows and (iter_idx % checkpoint_every) == 0:
                checkpoint_df = merge_fundamentals_with_existing(existing_out, rows, fetch_plan)
                save_fundamentals(checkpoint_df)
                checkpoint_count += 1
                logging.info(
                    "Checkpoint saved at %d/%d calls (rows=%d checkpoints=%d)",
                    iter_idx,
                    total_iters,
                    len(checkpoint_df),
                    checkpoint_count,
                )

            # raw 로그 저장
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        code,
                        corp_code,
                        y,
                        fs_used or "",
                        status or "",
                        f"{dt_ms:.0f}",
                        1 if ok else 0,
                    ]
                )

            # 요청 간 슬립(레이트리밋 완화)
            time.sleep(sleep_sec)

    if csv_file is not None:
        csv_file.close()
        logging.info("Saved fetch raw log: %s", csv_path.resolve())

    elapsed = time.perf_counter() - start_ts
    # format elapsed as h:m:s to avoid missing-argument logging errors
    total_sec = int(elapsed)
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    logging.info(
        "Summary: codes=%d years=%d total=%d calls=%d ok=%d empty=%d fail=%d unmapped=%d checkpoints=%d elapsed=%02d:%02d:%02d",
        len(codes),
        years_back,
        total_iters,
        api_calls,
        ok_count,
        empty_count,
        fail_count,
        len(unmapped_codes),
        checkpoint_count,
        h,
        m,
        s,
    )

    if not rows:
        if not existing_out.empty:
            logging.info("No missing fundamentals to fetch. Reusing existing fundamentals cache.")
            return existing_out.sort_values(["code", "date"]).reset_index(drop=True)
        raise RuntimeError("No financial rows fetched from DART.")

    return merge_fundamentals_with_existing(existing_out, rows, fetch_plan)


def save_fundamentals(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["code"] = out["code"].astype(str).str.zfill(6)
    out = out[FUNDAMENTALS_DB_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(out, FUNDAMENTALS_PK, "fundamentals")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FUND_OUT, index=False, encoding="utf-8")
    logging.info("Saved fundamentals: %s (rows=%d)", FUND_OUT.resolve(), len(out))

    try:
        if replace_table_rows_pg and get_engine:
            replace_table_rows_pg("fundamentals", out, columns=FUNDAMENTALS_DB_COLUMNS)
            logging.info("Replaced fundamentals rows in Postgres (rows=%d)", len(out))
            return
    except Exception:
        logging.exception("Failed to save fundamentals to Postgres, fallback to sqlite")

    # SQLite fallback
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals (
                date           DATE NOT NULL,
                code           TEXT NOT NULL,
                roe            REAL,
                op_margin      REAL,
                debt_ratio     REAL,
                ocf_to_assets  REAL,
                net_margin     REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        records = out.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO fundamentals
            (date, code, roe, op_margin, debt_ratio, ocf_to_assets, net_margin)
            VALUES (:date, :code, :roe, :op_margin, :debt_ratio, :ocf_to_assets, :net_margin)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved fundamentals to DB: %s (rows=%d)", DB_PATH.resolve(), len(out))
    except Exception:
        logging.exception("Failed to save fundamentals to DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch fundamentals via OpenDART and build fundamentals.csv")
    p.add_argument(
        "--years-back",
        type=int,
        default=int(os.getenv("DART_YEARS_BACK", str(DEFAULT_YEARS_BACK))),
        help="Number of years to fetch (default: 7)",
    )
    p.add_argument(
        "--refresh-recent-years",
        type=int,
        default=int(os.getenv("DART_REFRESH_RECENT_YEARS", str(DEFAULT_REFRESH_RECENT_YEARS))),
        help="Force refetch for the most recent N years window (plus prior-year anchor if needed).",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=float(os.getenv("DART_SLEEP_SEC", str(DEFAULT_SLEEP_SEC))),
        help="Sleep seconds between calls (default: 0.15)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=int(os.getenv("DART_LOG_EVERY", str(DEFAULT_LOG_EVERY))),
        help="Log progress every N calls (default: 20)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.getenv("DART_CHECKPOINT_EVERY", str(DEFAULT_CHECKPOINT_EVERY))),
        help="Persist intermediate fundamentals every N DART calls; set 0 to disable (default: 100)",
    )
    p.add_argument("--raw-log", action="store_true", help="Write raw fetch log CSV under data/dart")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    api_key = load_api_key()
    codes = get_target_codes()
    logging.info("Target codes: %d", len(codes))
    df = build_fundamentals_from_dart(
        api_key,
        codes,
        years_back=args.years_back,
        refresh_recent_years=args.refresh_recent_years,
        sleep_sec=args.sleep_sec,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        raw_log=args.raw_log,
    )
    save_fundamentals(df)
    logging.info("Done.")


if __name__ == "__main__":
    main()
