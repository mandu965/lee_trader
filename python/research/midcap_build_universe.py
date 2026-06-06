"""
midcap_build_universe.py  —  중형주(101~200위) 연구 유니버스 CSV 생성

Phase 0 스냅샷(phase0_universe_snapshot.csv)에서 tier=midcap_101_200 행을 추출하여
연구 전용 유니버스 파일을 만든다. 저유동 종목은 drop하지 않고 flag만 단다
(1일 스냅샷 노이즈 회피 — 실제 하한은 Phase 2에서 다일 ADV로 적용).

산출: data/research_midcap/universe_midcap.csv
  컬럼: code, name, market, mcap_rank, turnover_eok, low_liquidity_flag,
        already_in_features
원칙: 운영 universe.csv 미수정. 라이브 무영향.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path("data/research_midcap")
SNAPSHOT = RESEARCH_DIR / "phase0_universe_snapshot.csv"
OUT = RESEARCH_DIR / "universe_midcap.csv"
LOW_LIQ_THRESHOLD_KRW = 3_000_000_000  # ₩30억/일 미만 = 저유동 flag


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_existing_feature_codes() -> set[str]:
    url = os.environ.get("DATABASE_URL")
    if not url:
        return set()
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(url)
        with eng.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT code FROM public.features")).fetchall()
        return {str(r[0]).zfill(6) for r in rows}
    except Exception as e:
        logging.warning("기존 feature 코드 조회 실패: %s", e)
        return set()


def main() -> int:
    setup_logging()
    if not SNAPSHOT.exists():
        logging.error("스냅샷 없음: %s (Phase 0 먼저 실행)", SNAPSHOT)
        return 1

    df = pd.read_csv(SNAPSHOT, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    mid = df[df["tier"] == "midcap_101_200"].copy()
    if mid.empty:
        logging.error("midcap_101_200 행이 없음")
        return 1

    existing = load_existing_feature_codes()

    out = pd.DataFrame({
        "code": mid["code"],
        "name": mid["name"],
        "market": mid["market"],
        "mcap_rank": mid["mcap_rank"].astype(int),
        "turnover_eok": (mid["turnover_krw"] / 1e8).round(1),
        "low_liquidity_flag": (mid["turnover_krw"] < LOW_LIQ_THRESHOLD_KRW),
        "already_in_features": mid["code"].isin(existing),
    }).sort_values(["market", "mcap_rank"]).reset_index(drop=True)

    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    n = len(out)
    n_low = int(out["low_liquidity_flag"].sum())
    n_have = int(out["already_in_features"].sum())
    logging.info("연구 유니버스 생성: %s", OUT.resolve())
    print(f"  총 종목수             : {n}")
    print(f"  저유동 flag (<30억)   : {n_low}")
    print(f"  이미 features 보유    : {n_have}")
    print(f"  신규 백필 필요        : {n - n_have}")
    print(f"  시장별: KOSPI={int((out.market=='KOSPI').sum())}  KOSDAQ={int((out.market=='KOSDAQ').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
