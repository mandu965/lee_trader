"""
midcap_phase3_dedicated.py  —  Phase 3: 중형주 전용 모델 (walk-forward, PIT 유니버스)

생존편향·in-sample 누수를 둘 다 제거한 핵심 테스트:
  - 유니버스 : point-in-time (각 시점의 시총 101~200위, pit_universe_membership.csv 분기 스냅샷 forward-fill)
  - 모델     : 중형주로 직접 학습한 LightGBM (price-only 24피처, 1차) — target = log 60d 수익
  - 검증     : walk-forward (분기별 재학습, 라벨창 끝나기 전 데이터만 학습 → 누수 0)
  - 비교군   : naive 모멘텀(ret_120d) 랭킹 (ML이 단순 모멘텀을 이기나)

출력: data/research_midcap/phase3_walkforward.csv + 콘솔 요약
원칙: 읽기 전용, 운영 무접촉, 연구 디렉터리에만 저장.

사용법: python python/research/midcap_phase3_dedicated.py --prices data/research_midcap/prices_pit_raw.csv
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
RD = ROOT / "data" / "research_midcap"

H = 60                 # 예측/보유 지평(거래일)
GAP = 5                # 라벨창과 학습 사이 purge 버퍼
PRICE_FEATS = [
    "ret_5d","ret_10d","mom_20","ret_60d","ret_120d","high_52w_ratio",
    "close_over_ma20","vol_20","vol_60","rsi_14","volume_ratio_5d","volume_ratio_20d",
    "value_ratio_5d","value_ratio_20d","volume_score","liquidity_score","vol_ratio_20",
]
TOPN_LIST = [10, 20]
ADV_FLOOR = 3_000_000_000
POSITION_KRW = 10_000_000
OOS_START = "2024-07-01"


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def slippage(adv20):
    if not adv20 or adv20 <= 0 or np.isnan(adv20):
        return 0.02
    return float(np.clip(0.001 + 0.1 * (POSITION_KRW / adv20), 0.001, 0.02))


def round_trip(adv20):
    return 0.00015 * 2 + 0.0018 + slippage(adv20) * 2


def main() -> int:
    setup_logging()
    import feature_builder as fb
    from lightgbm import LGBMRegressor
    fb.get_engine = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(RD / "prices_pit_raw.csv"))
    args = ap.parse_args()

    # ---- 가격 패널 + 피처 + 라벨 ----
    prices = pd.read_csv(args.prices, dtype={"code": str})
    prices["code"] = prices["code"].str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    for c in ("close", "volume"):
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = prices.dropna(subset=["date", "close"]).sort_values(["code", "date"])
    logging.info("prices: rows=%d codes=%d %s~%s", len(prices), prices.code.nunique(),
                 prices.date.min().date(), prices.date.max().date())

    feat = fb.build_features(prices)        # 24 가격/기술 피처
    feat["date"] = pd.to_datetime(feat["date"])

    # 라벨: 60거래일 forward log수익 + ADV20
    panel = prices.copy()
    panel["turnover"] = panel["close"] * panel["volume"]
    panel["adv20"] = panel.groupby("code")["turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["fwd_close"] = panel.groupby("code")["close"].shift(-H)
    panel["label_log60"] = np.log(panel["fwd_close"] / panel["close"])
    feat = feat.merge(panel[["date", "code", "adv20", "label_log60", "close"]], on=["date", "code"], how="left")

    # ---- PIT 멤버십 (분기 스냅샷 → forward-fill) ----
    mem = pd.read_csv(RD / "pit_universe_membership.csv", dtype={"code": str})
    mem["code"] = mem["code"].str.zfill(6); mem["date"] = pd.to_datetime(mem["date"])
    snap_dates = np.array(sorted(mem["date"].unique()))
    mem_by_snap = {d: set(mem.loc[mem["date"] == d, "code"]) for d in snap_dates}

    def universe_asof(d):
        prior = snap_dates[snap_dates <= np.datetime64(d)]
        return mem_by_snap[prior[-1]] if len(prior) else set()

    all_dates = np.array(sorted(feat["date"].unique()))
    def dminus(d, n):
        i = np.searchsorted(all_dates, np.datetime64(d)); j = i - n
        return pd.Timestamp(all_dates[j]) if j >= 0 else None
    def dplus(d, n):
        i = np.searchsorted(all_dates, np.datetime64(d)); j = i + n
        return pd.Timestamp(all_dates[j]) if j < len(all_dates) else None

    # ---- 워크포워드: 분기별 재학습 ----
    feat = feat.sort_values("date")
    oos = feat[feat["date"] >= pd.to_datetime(OOS_START)]
    eval_quarters = pd.to_datetime(sorted(oos["date"].dt.to_period("Q").dt.start_time.unique()))
    close_tbl = {c: g.set_index("date")["close"] for c, g in panel.groupby("code")}

    preds = []  # 누적 OOS 예측
    for q_start in eval_quarters:
        # 학습: 라벨창(+H)이 q_start 이전에 끝난 데이터만 (purge GAP)
        train_cut = dminus(q_start, H + GAP)
        if train_cut is None:
            continue
        tr = feat[(feat["date"] <= train_cut) & feat["label_log60"].notna()].dropna(subset=PRICE_FEATS, how="all")
        tr = tr.dropna(subset=["label_log60"])
        if len(tr) < 2000:
            continue
        model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(tr[PRICE_FEATS], tr["label_log60"])
        q_end = q_start + pd.offsets.QuarterEnd(0)
        qd = oos[(oos["date"] >= q_start) & (oos["date"] <= q_end)].copy()
        if qd.empty:
            continue
        qd["pred"] = model.predict(qd[PRICE_FEATS])
        preds.append(qd[["date", "code", "pred", "ret_120d", "adv20", "label_log60"]])
        logging.info("  WF quarter %s: train=%d eval=%d", q_start.date(), len(tr), len(qd))

    if not preds:
        logging.error("워크포워드 결과 없음")
        return 1
    P = pd.concat(preds, ignore_index=True)

    # ---- 평가: PIT 멤버십 + ADV 필터 적용 ----
    def realized(code, d):
        s = close_tbl.get(code)
        if s is None: return np.nan
        p0 = s.get(d); pn = s.get(dplus(d, H))
        return (pn / p0 - 1) if (p0 and pn and p0 > 0) else np.nan

    # rank-IC (주간 그리드)
    P["in_uni"] = [c in universe_asof(d) for c, d in zip(P["code"], P["date"])]
    P = P[P["in_uni"] & (P["adv20"] >= ADV_FLOOR)].copy()
    grid = np.array(sorted(P["date"].unique()))[::5]
    ic_ml, ic_mom = [], []
    for d in grid:
        day = P[P["date"] == d].copy()
        if len(day) < 10: continue
        day["fwd"] = [realized(c, pd.Timestamp(d)) for c in day["code"]]
        day = day.dropna(subset=["fwd"])
        if len(day) >= 10:
            ic_ml.append(day["pred"].corr(day["fwd"], method="spearman"))
            ic_mom.append(day["ret_120d"].corr(day["fwd"], method="spearman"))

    # 포트폴리오: 비중첩 60일 리밸런스
    rebal = np.array(sorted(P["date"].unique()))[::H]
    rows = []
    for d in rebal:
        dd = pd.Timestamp(d)
        day = P[P["date"] == dd].copy()
        if len(day) < 20: continue
        day["fwd"] = [realized(c, dd) for c in day["code"]]
        day = day.dropna(subset=["fwd"])
        if len(day) < 20: continue
        bench = day["fwd"].mean()
        row = {"date": dd.date(), "n": len(day), "bench": bench}
        for N in TOPN_LIST:
            top = day.nlargest(N, "pred")
            net = (top["fwd"] - top["adv20"].apply(round_trip)).mean()
            row[f"top{N}_net"] = net
            row[f"top{N}_alpha"] = net - bench
            row[f"top{N}_hit"] = (top["fwd"] - top["adv20"].apply(round_trip) > 0).mean()
        rows.append(row)
    pf = pd.DataFrame(rows)
    pf.to_csv(RD / "phase3_walkforward.csv", index=False, encoding="utf-8-sig")

    ic_ml_m = np.nanmean(ic_ml) if ic_ml else np.nan
    ic_ml_ir = (np.nanmean(ic_ml) / np.nanstd(ic_ml)) if len(ic_ml) > 1 else np.nan
    print("\n" + "=" * 64)
    print("  Phase 3 — 중형주 전용모델 walk-forward (PIT 유니버스, price-only)")
    print("=" * 64)
    print(f"  OOS 기간: {OOS_START}~ , 분기 재학습 {len(eval_quarters)}회")
    print(f"  rank-IC(ML pred) : mean={ic_ml_m:.4f}  IR={ic_ml_ir:.2f}  (n={len(ic_ml)})")
    print(f"  rank-IC(naive ret_120d): mean={np.nanmean(ic_mom):.4f}  (ML이 이겨야 의미)")
    if not pf.empty:
        print(f"  리밸런스 기간: {len(pf)}  / PIT 벤치 60d 평균: {pf['bench'].mean()*100:.2f}%")
        for N in TOPN_LIST:
            print(f"  top{N:2d}: 60d net={pf[f'top{N}_net'].mean()*100:6.2f}%  "
                  f"alpha={pf[f'top{N}_alpha'].mean()*100:6.2f}%p  hit={pf[f'top{N}_hit'].mean()*100:3.0f}%  "
                  f"누적={(1+pf[f'top{N}_net']).prod()-1:7.1%}")
    print(f"\n  저장: phase3_walkforward.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
