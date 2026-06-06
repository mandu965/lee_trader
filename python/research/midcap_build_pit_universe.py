"""
midcap_build_pit_universe.py  —  Phase 3: PIT 유니버스 종목 리스트(784) 생성

pit_universe_membership.csv(과거 분기별 101~200위) ∪ universe_midcap.csv(현재 200)
→ Phase 3 학습/평가 대상 종목 전체. 생존편향 제거의 토대.

출력: data/research_midcap/universe_pit.csv
  컬럼: code, market, n_quarters_in_band(과거 밴드 등장 분기수), in_current(현재 200 포함)
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / "data" / "research_midcap"
PIT = RD / "pit_universe_membership.csv"
CUR = RD / "universe_midcap.csv"
OUT = RD / "universe_pit.csv"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pit = pd.read_csv(PIT, dtype={"code": str}); pit["code"] = pit["code"].str.zfill(6)
    cur = pd.read_csv(CUR, dtype={"code": str}); cur["code"] = cur["code"].str.zfill(6)

    # 종목별 시장(최빈)·등장 분기수
    agg = (pit.groupby("code")
              .agg(market=("market", lambda s: s.mode().iloc[0]),
                   n_quarters_in_band=("date", "nunique"))
              .reset_index())
    cur_codes = set(cur["code"])
    # 현재 200 중 PIT에 없던 코드도 포함(합집합)
    extra = cur[~cur["code"].isin(set(agg["code"]))][["code", "market"]].copy()
    extra["n_quarters_in_band"] = 0
    allu = pd.concat([agg, extra], ignore_index=True).drop_duplicates("code")
    allu["in_current"] = allu["code"].isin(cur_codes)
    allu = allu.sort_values(["market", "code"]).reset_index(drop=True)
    allu.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"  universe_pit.csv: {len(allu)}종목")
    print(f"    현재 200 포함: {int(allu.in_current.sum())}  / 신규(dropout): {int((~allu.in_current).sum())}")
    print(f"    시장별: KOSPI={int((allu.market=='KOSPI').sum())} KOSDAQ={int((allu.market=='KOSDAQ').sum())}")
    print(f"    저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
