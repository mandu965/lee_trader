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

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "dart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FUND_OUT = DATA_DIR / "fundamentals.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

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


def request_with_retry(
    url: str,
    params: Dict[str, str],
    expect_json: bool = True,
    retries: int = 3,
    sleep_sec: float = 0.6,
):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json() if expect_json else r.content
            logging.warning("HTTP %s for %s params=%s", r.status_code, url, params)
        except Exception as e:
            logging.warning("request error: %s (attempt %d/%d)", e, i + 1, retries)
        time.sleep(sleep_sec)
    raise RuntimeError(f"Failed request after {retries} retries: {url} params={params}")


def download_and_cache_corp_codes(api_key: str) -> Path:
    cache_xml = CACHE_DIR / "corp_codes.xml"
    if (
        cache_xml.exists()
        and cache_xml.stat().st_size > 0
        and (time.time() - cache_xml.stat().st_mtime) < 86400 * 7
    ):
        logging.info("Using cached corp_codes.xml: %s", cache_xml.resolve())
        return cache_xml

    logging.info("Downloading corpCode.zip from DART...")
    content = request_with_retry(DART_CORP_CODE_URL, params={"crtfc_key": api_key}, expect_json=False)
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
    api_key: str, corp_code: str, year: int
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
            data = request_with_retry(DART_FNLTT_URL, params=params, expect_json=True)
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


def build_fundamentals_from_dart(
    api_key: str,
    codes: List[str],
    years_back: int = DEFAULT_YEARS_BACK,
    sleep_sec: float = 0.15,
    log_every: int = 20,
    raw_log: bool = False,
) -> pd.DataFrame:
    xml_path = download_and_cache_corp_codes(api_key)
    corp_df = parse_corp_codes(xml_path)
    corp_df["stock_code"] = corp_df["stock_code"].astype(str).str.zfill(6)
    code_map = dict(zip(corp_df["stock_code"], corp_df["corp_code"]))

    today = datetime.today()
    end_year = today.year
    start_year = end_year - (years_back - 1)

    unmapped_codes: List[str] = []
    rows: List[Dict[str, object]] = []

    # 진행률/ETA 집계
    total_iters = len(codes) * years_back
    iter_idx = 0
    ok_count = 0
    empty_count = 0
    fail_count = 0
    cached_hits = 1  # corp_codes.xml 캐시 활용 지표
    api_calls = 0
    total_ms = 0.0
    start_ts = time.perf_counter()

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
        corp_code = code_map.get(code)
        if not corp_code:
            unmapped_codes.append(code)
            logging.info("No corp_code for stock %s (skip)", code)
            # 해당 코드에 대해 years_back 만큼 iter 증가 처리
            iter_idx += years_back
            continue

        for y in range(start_year, end_year + 1):
            t0 = time.perf_counter()
            fin, fs_used, status = fetch_annual_financials(api_key, corp_code, y)
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
        "Summary: codes=%d years=%d total=%d calls=%d ok=%d empty=%d fail=%d unmapped=%d elapsed=%02d:%02d:%02d",
        len(codes),
        years_back,
        total_iters,
        api_calls,
        ok_count,
        empty_count,
        fail_count,
        len(unmapped_codes),
        h,
        m,
        s,
    )

    if not rows:
        raise RuntimeError("No financial rows fetched from DART.")

    df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
    # 숫자형 변환
    for c in ["revenue", "op_income", "net_income", "assets", "equity", "liabilities", "ocf"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 전년도 대비 평균자본(ROE), 평균자산(OCF/Assets) 계산
    def per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["equity_avg"] = (g["equity"].shift(1) + g["equity"]) / 2.0
        g["assets_avg"] = (g["assets"].shift(1) + g["assets"]) / 2.0
        # 비율 계산(0 나눗셈 방지)
        g["roe"] = np.where((g["equity_avg"].abs() > 1e-9), g["net_income"] / g["equity_avg"], np.nan)
        g["op_margin"] = np.where((g["revenue"].abs() > 1e-9), g["op_income"] / g["revenue"], np.nan)
        g["debt_ratio"] = np.where((g["equity"].abs() > 1e-9), g["liabilities"] / g["equity"], np.nan)
        g["ocf_to_assets"] = np.where((g["assets_avg"].abs() > 1e-9), g["ocf"] / g["assets_avg"], np.nan)
        g["net_margin"] = np.where((g["revenue"].abs() > 1e-9), g["net_income"] / g["revenue"], np.nan)
        return g

    df = df.groupby("code", group_keys=False).apply(per_code)
    # 필요 컬럼만
    out_cols = ["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"]
    out = df[out_cols].copy()
    return out


def save_fundamentals(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FUND_OUT, index=False, encoding="utf-8")
    logging.info("Saved fundamentals: %s (rows=%d)", FUND_OUT.resolve(), len(out))

    # DB upsert
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
    p.add_argument("--years-back", type=int, default=DEFAULT_YEARS_BACK, help="Number of years to fetch (default: 7)")
    p.add_argument("--sleep-sec", type=float, default=0.15, help="Sleep seconds between calls (default: 0.15)")
    p.add_argument("--log-every", type=int, default=20, help="Log progress every N calls (default: 20)")
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
        sleep_sec=args.sleep_sec,
        log_every=args.log_every,
        raw_log=args.raw_log,
    )
    save_fundamentals(df)
    logging.info("Done.")


if __name__ == "__main__":
    main()
