"""
midcap_backfill_dart.py  —  Phase 1: 중형주 DART 재무 + 재무모멘텀 피처 백필 (clean-room)

운영 DART 수집기는 save_*()가 CSV+PG+sqlite 다중 싱크에 써서 격리 사고 위험이 크다.
→ 여기서는 **순수 계산 함수만 재사용**하고 I/O는 전부 연구 파일로 직접 처리한다.

재사용(순수 함수):
  fetch_financials_dart_quarterly:  load_api_key / build_http_session /
    download_corp_codes / parse_corp_codes / build_fetch_plan /
    fetch_quarter_financials / compute_ratios / estimate_disclosed_at /
    QUARTER_SPECS / DB_COLUMNS
  build_financial_momentum_features: build_features(df)  (de-accum→YoY→QoQ→phase→scores)

출력(격리): data/research_midcap/financial_quarterly_midcap.csv
            data/research_midcap/financial_momentum_midcap.csv
원칙: 운영 financial_statement_quarterly / financial_momentum_quarterly (CSV·DB) 미접촉.

사용법: python python/research/midcap_backfill_dart.py [--years-back 5]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import sys
import time
from pathlib import Path

import pandas as pd

socket.setdefaulttimeout(20)  # DART API hang 방지

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))  # 운영 모듈 import 경로

RESEARCH_DIR = ROOT / "data" / "research_midcap"
UNI = RESEARCH_DIR / "universe_midcap.csv"
QUARTERLY_OUT = RESEARCH_DIR / "financial_quarterly_midcap.csv"
MOMENTUM_OUT = RESEARCH_DIR / "financial_momentum_midcap.csv"


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if v.startswith(('"', "'")):
            q = v[0]; end = v.find(q, 1); v = v[1:end] if end != -1 else v[1:]
        else:
            v = re.sub(r"\s+#.*$", "", v).strip()
        if k and k not in os.environ:
            os.environ[k] = v


def load_existing() -> pd.DataFrame:
    if QUARTERLY_OUT.exists():
        try:
            return pd.read_csv(QUARTERLY_OUT, dtype={"stock_code": str})
        except Exception:
            pass
    return pd.DataFrame()


def main() -> int:
    setup_logging()
    load_env()

    import fetch_financials_dart_quarterly as fq
    import build_financial_momentum_features as bm
    # 모멘텀 빌더가 sector 예외 판정 시 읽는 universe.csv → 연구 유니버스로 격리
    # (sector 컬럼 없으면 예외 없음으로 graceful degrade)
    bm.UNIVERSE_CSV = UNI
    fq.get_engine = None  # 방어적: DB 훅 차단 (save_* 미호출이지만 안전)

    api_key = fq.load_api_key()
    session = fq.build_http_session()
    xml_path = fq.download_corp_codes(api_key, session)
    code_map = fq.parse_corp_codes(xml_path)
    logging.info("corp_code map: %d", len(code_map))

    uni = pd.read_csv(UNI, dtype={"code": str})
    codes = sorted(uni["code"].str.zfill(6).unique())

    ap = argparse.ArgumentParser()
    ap.add_argument("--years-back", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="검증용: 앞 N종목만")
    args = ap.parse_args()
    if args.limit:
        codes = codes[:args.limit]
        logging.info("검증 모드: %d종목으로 제한", len(codes))

    existing = load_existing()
    plan = fq.build_fetch_plan(codes, existing, args.years_back, refresh_recent_years=0)
    logging.info("DART fetch plan: codes=%d planned=%d existing_rows=%d", len(codes), len(plan), len(existing))

    rows: list[dict] = []
    ok = skip = fail = 0
    total = len(plan)
    for idx, ((stock_code, fy, q), reprt) in enumerate(plan.items(), 1):
        corp = code_map.get(stock_code)
        if not corp:
            skip += 1
            continue
        try:
            fin, fs_div, status = fq.fetch_quarter_financials(api_key, corp, fy, reprt, session)
        except Exception as e:
            logging.warning("fetch 실패 code=%s fy=%s q=%s: %s", stock_code, fy, q, e)
            fin = None
        if fin is not None:
            ratios = fq.compute_ratios(fin)
            _, src_suffix, offset_days = fq.QUARTER_SPECS[reprt]
            rows.append({
                "stock_code": stock_code, "fiscal_year": fy, "quarter": q, "report_code": reprt,
                "source_report_date": f"{fy}{src_suffix}",
                "disclosed_at": fq.estimate_disclosed_at(fy, q, offset_days),
                "revenue": fin.get("revenue"), "op_income": fin.get("op_income"),
                "net_income": fin.get("net_income"), "assets": fin.get("assets"),
                "liabilities": fin.get("liabilities"), "equity": fin.get("equity"),
                "ocf": fin.get("ocf"),
                "op_margin": ratios.get("op_margin"), "debt_ratio": ratios.get("debt_ratio"),
                "net_margin": ratios.get("net_margin"), "ocf_to_assets": ratios.get("ocf_to_assets"),
                "fs_div": fs_div,
            })
            ok += 1
        else:
            fail += 1
        if idx % 100 == 0 or idx == total:
            logging.info("[DART-Q] %d/%d ok=%d skip=%d fail=%d", idx, total, ok, skip, fail)
        time.sleep(args.sleep)

    # 분기재무 병합 저장 (기존 + 신규)
    new_df = pd.DataFrame(rows)
    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    if merged.empty:
        logging.error("수집된 분기재무 없음")
        return 1
    merged = (merged.drop_duplicates(subset=["stock_code", "fiscal_year", "quarter"], keep="last")
                    .sort_values(["stock_code", "fiscal_year", "quarter"]).reset_index(drop=True))
    for col in fq.DB_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    merged[fq.DB_COLUMNS].to_csv(QUARTERLY_OUT, index=False, encoding="utf-8-sig")
    logging.info("분기재무 저장: %s (rows=%d)", QUARTERLY_OUT.name, len(merged))

    # 모멘텀 피처 산출 (순수 변환) → 격리 저장
    mom = bm.build_features(merged.copy())
    mom.to_csv(MOMENTUM_OUT, index=False, encoding="utf-8-sig")
    logging.info("모멘텀 피처 저장: %s (rows=%d, codes=%d)", MOMENTUM_OUT.name, len(mom), mom["stock_code"].nunique())

    print("\n" + "=" * 56)
    print(f"  분기재무 : {len(merged):,}행 → {QUARTERLY_OUT.name}")
    print(f"  모멘텀   : {len(mom):,}행 / {mom['stock_code'].nunique()}종목 → {MOMENTUM_OUT.name}")
    print(f"  fetch ok/skip/fail: {ok}/{skip}/{fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
