"""
midcap_build_features.py  —  Phase 1 마무리 / Phase 2 입력: 중형주 43피처 빌드

운영 feature_builder의 순수/경로기반 함수를 격리 재사용한다.
  - 가격/기술 24피처 : fb.build_features(prices)  (그대로 재사용)
  - 재무 fin + YoY    : fb.merge_financial_momentum  (FIN_MOMENTUM_CSV 경로 override)
  - 공매도 short 3종  : fb.merge_short_interest       (SHORT_INTEREST_CSV 경로 override)
  - 수급 flow 4종     : 직접 재구현(연구 flow CSV는 이미 wide; 운영 롤링 수식과 동일)
  - 미싱(quality_score 4 + roe_yoy + sector_rel_momentum_20d): NaN
       → LightGBM 네이티브 NaN 처리. Phase 2 top-100 비교 시 동일 중립으로 공정 비교.

출력(격리): data/research_midcap/features_midcap.csv
원칙: 운영 data/features.csv·DB 미접촉 (fb.get_engine=None, save_features 미호출).
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

RESEARCH_DIR = ROOT / "data" / "research_midcap"
PRICES = RESEARCH_DIR / "prices_midcap_raw.csv"
FLOW = RESEARCH_DIR / "flow_midcap_raw.csv"
SHORT = RESEARCH_DIR / "short_midcap_raw.csv"
FIN_MOM = RESEARCH_DIR / "financial_momentum_midcap.csv"
OUT = RESEARCH_DIR / "features_midcap.csv"

MISSING_NAN = [
    "quality_score", "quality_factor_count", "quality_missing_ratio",
    "quality_score_confidence", "roe_yoy", "sector_rel_momentum_20d",
]


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def merge_flow_research(feat: pd.DataFrame) -> pd.DataFrame:
    """연구 flow CSV(date,code,foreign_net,inst_net) → 5d/20d 롤링합계 (운영 merge_flow와 동일 수식)."""
    if not FLOW.exists():
        logging.warning("flow CSV 없음 → flow 피처 NaN")
        for c in ["flow_foreign_net_5d", "flow_foreign_net_20d", "flow_inst_net_5d", "flow_inst_net_20d"]:
            feat[c] = np.nan
        return feat
    f = pd.read_csv(FLOW, dtype={"code": str})
    f["code"] = f["code"].str.zfill(6)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    for c in ("foreign_net", "inst_net"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    parts = []
    for _, g in f.groupby("code", sort=False):
        g = g.sort_values("date").copy()
        g["flow_foreign_net_5d"] = g["foreign_net"].rolling(5, min_periods=3).sum()
        g["flow_foreign_net_20d"] = g["foreign_net"].rolling(20, min_periods=10).sum()
        g["flow_inst_net_5d"] = g["inst_net"].rolling(5, min_periods=3).sum()
        g["flow_inst_net_20d"] = g["inst_net"].rolling(20, min_periods=10).sum()
        parts.append(g[["date", "code", "flow_foreign_net_5d", "flow_foreign_net_20d",
                        "flow_inst_net_5d", "flow_inst_net_20d"]])
    flow_feat = pd.concat(parts, ignore_index=True)
    return feat.merge(flow_feat, on=["date", "code"], how="left")


def main() -> int:
    setup_logging()
    import feature_builder as fb
    # 경로/엔진 override (운영 데이터·DB 미접촉, 연구 CSV만 사용)
    fb.SHORT_INTEREST_CSV = SHORT
    fb.FIN_MOMENTUM_CSV = FIN_MOM
    fb.get_engine = None  # merge_flow가 DB 안 보고 skip하도록 (flow는 직접 처리)

    prices = pd.read_csv(PRICES, dtype={"code": str})
    prices["code"] = prices["code"].str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    logging.info("prices: rows=%d codes=%d %s~%s", len(prices), prices.code.nunique(),
                 prices.date.min().date(), prices.date.max().date())

    feat = fb.build_features(prices)                  # 24 가격/기술 피처
    feat["date"] = pd.to_datetime(feat["date"])
    logging.info("build_features: rows=%d", len(feat))

    feat = merge_flow_research(feat)                  # 4 flow
    feat = fb.merge_financial_momentum(feat)          # fin_* + revenue/op YoY
    feat = fb.merge_short_interest(feat)              # short 3종

    for c in MISSING_NAN:                             # 미싱 6개 NaN
        if c not in feat.columns:
            feat[c] = np.nan

    # 모델 입력 43피처 존재 확인
    model_feats = [
        "close","ret_1d","ret_5d","ret_10d","mom_20","ret_60d","ret_120d","high_52w_ratio",
        "ma_5","ma_20","ma_60","close_over_ma20","vol_20","vol_60","rsi_14","volume",
        "volume_ratio_5d","volume_ratio_20d","value_ratio_5d","value_ratio_20d","volume_score",
        "liquidity_score","vol_ma_20","vol_ratio_20","quality_score","quality_factor_count",
        "quality_missing_ratio","quality_score_confidence","flow_foreign_net_5d","flow_foreign_net_20d",
        "flow_inst_net_5d","flow_inst_net_20d","revenue_growth_yoy","op_income_growth_yoy","roe_yoy",
        "sector_rel_momentum_20d","fin_momentum_score","fin_risk_score","fin_turnaround_score",
        "fin_hard_risk","short_ratio","short_ratio_5d_chg","short_ratio_20d_avg",
    ]
    missing = [c for c in model_feats if c not in feat.columns]
    if missing:
        logging.error("모델 피처 누락: %s", missing)
        return 1

    feat = feat.sort_values(["code", "date"]).reset_index(drop=True)
    out_cols = ["date", "code"] + [c for c in model_feats if c != "close"] + ["close"]
    # 중복 제거 후 저장
    out_cols = list(dict.fromkeys(out_cols))
    feat["date"] = feat["date"].dt.strftime("%Y-%m-%d")
    feat[out_cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    # 커버리지 리포트
    cov = {c: round(feat[c].notna().mean() * 100, 1) for c in model_feats}
    print("\n" + "=" * 60)
    print(f"  features_midcap.csv: rows={len(feat):,} codes={feat.code.nunique()}")
    print(f"  날짜범위: {feat.date.min()} ~ {feat.date.max()}")
    print("  주요 피처 non-null %:")
    for grp, cols in [("flow", ["flow_foreign_net_20d", "flow_inst_net_20d"]),
                      ("short", ["short_ratio", "short_ratio_20d_avg"]),
                      ("fin", ["fin_momentum_score", "revenue_growth_yoy"]),
                      ("price", ["ret_60d", "high_52w_ratio", "volume_score"]),
                      ("missing(NaN예상)", ["quality_score", "roe_yoy", "sector_rel_momentum_20d"])]:
        print(f"    {grp:18}: " + ", ".join(f"{c}={cov[c]}%" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
