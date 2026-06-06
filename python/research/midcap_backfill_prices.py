"""
midcap_backfill_prices.py  —  Phase 1: 중형주 가격(OHLCV) 백필

연구 유니버스(universe_midcap.csv) 전 종목의 일봉을 pykrx로 수집한다.
종목당 1회 호출(get_market_ohlcv, from~to)이라 빠르다.
운영 prices_daily_raw 스키마(date,code,open,high,low,close,volume,...)에 맞춰
저장하여 추후 feature_builder 재사용을 가능케 한다.

원칙: 운영 KIS 인증 미사용(KRX 공개데이터), 출력은 data/research_midcap/ 격리.
"""
from __future__ import annotations

import argparse
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import requests as _rq
from pykrx import stock

# pykrx/KRX 호출 hang 방지: requests에 기본 (connect,read) 타임아웃 주입
_orig_request = _rq.sessions.Session.request
def _request_with_timeout(self, *a, **k):
    if k.get("timeout") is None:
        k["timeout"] = (10, 15)
    return _orig_request(self, *a, **k)
_rq.sessions.Session.request = _request_with_timeout

RESEARCH_DIR = Path("data/research_midcap")
UNI = RESEARCH_DIR / "universe_midcap.csv"
OUT = RESEARCH_DIR / "prices_midcap_raw.csv"
FROM, TO = "20220101", "20260530"   # 120d/52w 룩백 여유 포함
SLEEP = 0.4


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_COLMAP = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}


def fetch_one(code: str) -> pd.DataFrame | None:
    try:
        df = stock.get_market_ohlcv(FROM, TO, code)
    except Exception as e:
        logging.warning("OHLCV 실패 code=%s err=%s", code, e)
        return None
    if df is None or df.empty:
        return None
    df = df.reset_index().rename(columns=_COLMAP)
    # 날짜 컬럼 표준화 (pykrx index명: '날짜')
    date_col = "날짜" if "날짜" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = str(code).zfill(6)
    keep = ["date", "code", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep]
    # 거래대금 = 종가 × 거래량 (참고용)
    df["turnover"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
    df = df[df["volume"].fillna(0) > 0]  # 거래정지일 제거
    return df


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(UNI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    uni_path, out_path = Path(args.universe), Path(args.out)
    if not uni_path.exists():
        logging.error("유니버스 없음: %s", uni_path)
        return 1
    uni = pd.read_csv(uni_path, dtype={"code": str})
    uni["code"] = uni["code"].str.zfill(6)
    codes = uni["code"].tolist()
    logging.info("가격 백필 시작: %d종목 · %s~%s", len(codes), FROM, TO)

    frames, ok, fail = [], 0, 0
    for i, code in enumerate(codes, 1):
        df = fetch_one(code)
        if df is not None and not df.empty:
            df["asset_type"] = "stock"
            df["universe_source"] = "research_midcap"
            frames.append(df)
            ok += 1
        else:
            fail += 1
        if i % 25 == 0:
            logging.info("진행 %d/%d (ok=%d fail=%d)", i, len(codes), ok, fail)
        time.sleep(SLEEP)

    if not frames:
        logging.error("수집 결과 없음")
        return 1
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values(["code", "date"]).reset_index(drop=True)
    allp.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 56)
    print(f"  저장: {out_path.resolve()}")
    print(f"  종목 ok/fail   : {ok}/{fail}")
    print(f"  총 행수        : {len(allp):,}")
    print(f"  기간           : {allp['date'].min()} ~ {allp['date'].max()}")
    print(f"  종목당 평균 행수: {len(allp)/max(ok,1):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
