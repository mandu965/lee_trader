"""
midcap_validate_source.py  —  Phase 1 사전 검증: pykrx가 중형주 백필에 충분한가?

연구 유니버스에서 표본 종목을 골라 (1)OHLCV+거래대금 (2)투자자별 수급
(3)공매도 시계열을 실제로 받아본다. 운영 KIS 인증 미사용(KRX 공개데이터).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pykrx import stock

UNI = Path("data/research_midcap/universe_midcap.csv")
FROM, TO = "20260401", "20260530"


def reconf():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def pick_samples() -> list[tuple[str, str, str]]:
    df = pd.read_csv(UNI, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    picks = []
    for mkt in ("KOSPI", "KOSDAQ"):
        sub = df[df.market == mkt]
        if not sub.empty:
            picks.append((sub.iloc[0]["code"], sub.iloc[0]["name"], f"{mkt} 상단"))
    low = df[df.low_liquidity_flag == True]
    if not low.empty:
        picks.append((low.iloc[-1]["code"], low.iloc[-1]["name"], "저유동 flag"))
    return picks


def main() -> int:
    reconf()
    samples = pick_samples()
    print(f"표본 {len(samples)}종목 · 기간 {FROM}~{TO}\n" + "=" * 60)
    for code, name, tag in samples:
        print(f"\n[{code} {name}] ({tag})")
        # 1) OHLCV + 거래대금
        try:
            ohlcv = stock.get_market_ohlcv(FROM, TO, code)
            cols = list(ohlcv.columns)
            print(f"  OHLCV   : rows={len(ohlcv)} cols={cols}")
        except Exception as e:
            print(f"  OHLCV   : FAIL {e}")
        # 2) 투자자별 거래대금(수급) — 외국인/기관 순매수 산출 가능 여부
        try:
            inv = stock.get_market_trading_value_by_date(FROM, TO, code)
            print(f"  수급    : rows={len(inv)} cols={list(inv.columns)}")
        except Exception as e:
            print(f"  수급    : FAIL {e}")
        # 3) 공매도
        try:
            shrt = stock.get_shorting_volume_by_date(FROM, TO, code)
            print(f"  공매도  : rows={len(shrt)} cols={list(shrt.columns)}")
        except Exception as e:
            print(f"  공매도  : FAIL {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
