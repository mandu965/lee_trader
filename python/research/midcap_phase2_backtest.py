"""
midcap_phase2_backtest.py  —  Phase 2: 섀도 백테스트 (모델 전이 테스트)

기존 top-100 모델(model.pkl)을 그대로 적용해, 중형주 랭킹이 비용 차감 후에도
수익이 나는지("엣지가 전이되는가") 검증한다. 동일 harness를 top-100에도 적용해
공정 비교(미싱 6피처를 양쪽 동일 중립 처리)한다.

신호: 예측 target_log_60d(기대 60일 로그수익) [primary], target_60d_top20 확률 [cross-check]
분석:
  (1) rank-IC : 주간 그리드에서 spearman(신호, 실현 60일 수익) → 신호 품질
  (2) 포트폴리오 : 비중첩 60일 리밸런스 top-N 비용차감 수익 + 유니버스 벤치마크 대비 알파
비용: 수수료 0.015%×2 + 세금 0.18% + 슬리피지×2 (ADV 참여율 연동)
유동성: ADV20 < ₩30억 종목 제외 (양 유니버스 동일 적용)

원칙: 읽기 전용(features/prices/model). 운영 무영향. 출력은 data/research_midcap/.

사용법:
  python python/research/midcap_phase2_backtest.py \
      --features data/research_midcap/features_midcap.csv \
      --prices   data/research_midcap/prices_midcap_raw.csv --label midcap
  python python/research/midcap_phase2_backtest.py \
      --features data/features.csv --prices data/prices_daily_adjusted.csv --label top100
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))  # model.pkl unpickle(calibrated_classifier 등)
OUT_DIR = ROOT / "data" / "research_midcap"

H = 60               # 보유/예측 지평 (거래일)
REBAL = 60           # 비중첩 리밸런스 간격
IC_GRID = 5          # rank-IC 평가 그리드(거래일)
TOPN_LIST = [10, 20]
ADV_FLOOR = 3_000_000_000   # ₩30억/일 미만 제외
POSITION_KRW = 10_000_000   # 1종목당 주문금액(슬리피지 참여율용)

MODEL_FEATS = [
    "close","ret_1d","ret_5d","ret_10d","mom_20","ret_60d","ret_120d","high_52w_ratio",
    "ma_5","ma_20","ma_60","close_over_ma20","vol_20","vol_60","rsi_14","volume",
    "volume_ratio_5d","volume_ratio_20d","value_ratio_5d","value_ratio_20d","volume_score",
    "liquidity_score","vol_ma_20","vol_ratio_20","quality_score","quality_factor_count",
    "quality_missing_ratio","quality_score_confidence","flow_foreign_net_5d","flow_foreign_net_20d",
    "flow_inst_net_5d","flow_inst_net_20d","revenue_growth_yoy","op_income_growth_yoy","roe_yoy",
    "sector_rel_momentum_20d","fin_momentum_score","fin_risk_score","fin_turnaround_score",
    "fin_hard_risk","short_ratio","short_ratio_5d_chg","short_ratio_20d_avg",
]
# 공정 비교: 미드캡에서 결측인 6피처는 top-100에서도 중립(NaN) 처리
NEUTRALIZE = ["quality_score","quality_factor_count","quality_missing_ratio",
              "quality_score_confidence","roe_yoy","sector_rel_momentum_20d"]


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_model(path: str | None = None):
    import pickle, lightgbm  # noqa
    mp = Path(path) if path else (ROOT / "data" / "model.pkl")
    pack = pickle.load(open(mp, "rb"))
    return pack


def predict_signals(pack, feat: pd.DataFrame) -> pd.DataFrame:
    X = feat[MODEL_FEATS].apply(pd.to_numeric, errors="coerce")
    out = feat[["date", "code"]].copy()
    reg = pack["reg_models"]["target_log_60d"]
    out["exp_ret60"] = reg.predict(X)
    cls = pack["cls_models"].get("target_60d_top20")
    if cls is not None:
        try:
            out["prob_top20"] = cls.predict_proba(X)[:, 1]
        except Exception:
            out["prob_top20"] = cls.predict(X)
    return out


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """code×date 종가/ADV20 패널 + 거래일 인덱스."""
    p = prices.copy()
    # 수정주가 파일(adj_close)·원천(close) 양쪽 지원
    rename = {"adj_close": "close", "adj_open": "open", "adj_high": "high", "adj_low": "low"}
    p = p.rename(columns={k: v for k, v in rename.items() if k in p.columns and v not in p.columns})
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["code"] = p["code"].astype(str).str.zfill(6)
    for c in ("close", "volume"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["date", "close"]).sort_values(["code", "date"])
    p["turnover"] = p["close"] * p["volume"]
    p["adv20"] = p.groupby("code")["turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    return p


def fwd_return(panel: pd.DataFrame, trade_dates: list) -> dict:
    """각 (code) 거래일 인덱스 기반 H일 전방수익 lookup 테이블."""
    # code별 date→close 시리즈
    tbl = {}
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        tbl[code] = g.set_index("date")["close"]
    return tbl


def slippage_rate(position_krw: float, adv20: float) -> float:
    if not adv20 or adv20 <= 0 or np.isnan(adv20):
        return 0.02
    part = position_krw / adv20
    return float(np.clip(0.001 + 0.1 * part, 0.001, 0.02))


def round_trip_cost(adv20: float) -> float:
    slip = slippage_rate(POSITION_KRW, adv20)
    return 0.00015 * 2 + 0.0018 + slip * 2  # 매수수수료+매도수수료+세금+슬리피지왕복


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--start", default="2023-06-01", help="백테스트 시작(피처 워밍업 이후)")
    ap.add_argument("--model", default=None, help="모델 pkl 경로 (기본 운영 model.pkl)")
    args = ap.parse_args()

    pack = load_model(args.model)
    feat = pd.read_csv(args.features, dtype={"code": str})
    feat["code"] = feat["code"].str.zfill(6)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    for c in MODEL_FEATS:
        if c not in feat.columns:
            feat[c] = np.nan
    feat[NEUTRALIZE] = np.nan  # 공정 비교: 6피처 중립
    feat = feat[feat["date"] >= pd.to_datetime(args.start)].dropna(subset=["date"])
    logging.info("[%s] features rows=%d codes=%d %s~%s", args.label, len(feat),
                 feat.code.nunique(), feat.date.min().date(), feat.date.max().date())

    sig = predict_signals(pack, feat)

    panel = build_panel(pd.read_csv(args.prices, dtype={"code": str}))
    close_tbl = {c: g.set_index("date")["close"] for c, g in panel.groupby("code")}
    adv_tbl = {c: g.set_index("date")["adv20"] for c, g in panel.groupby("code")}
    all_dates = np.array(sorted(panel["date"].unique()))

    def date_after(d, n):
        idx = np.searchsorted(all_dates, np.datetime64(d))
        j = idx + n
        return pd.Timestamp(all_dates[j]) if j < len(all_dates) else None

    sig = sig.merge(panel[["date", "code", "adv20"]], on=["date", "code"], how="left")

    # ---- (1) rank-IC : 주간 그리드 ----
    sig_dates = np.array(sorted(sig["date"].unique()))
    grid = sig_dates[::IC_GRID]
    ics_ret, ics_prob = [], []
    for d in grid:
        dd = pd.Timestamp(d)
        dfu = date_after(dd, H)
        if dfu is None:
            continue
        day = sig[sig["date"] == dd].copy()
        day = day[(day["adv20"] >= ADV_FLOOR)]
        if len(day) < 10:
            continue
        fwd = []
        for _, r in day.iterrows():
            s = close_tbl.get(r["code"])
            if s is None:
                fwd.append(np.nan); continue
            p0 = s.get(dd);
            # d+H 종가(가장 가까운 거래일)
            pn = s.reindex([dfu]).iloc[0] if dfu in s.index else np.nan
            fwd.append((pn / p0 - 1) if (p0 and pn and p0 > 0) else np.nan)
        day["fwd"] = fwd
        day = day.dropna(subset=["fwd", "exp_ret60"])
        if len(day) >= 10:
            ics_ret.append(day["exp_ret60"].corr(day["fwd"], method="spearman"))
            if "prob_top20" in day:
                ics_prob.append(day["prob_top20"].corr(day["fwd"], method="spearman"))

    ic_ret = np.nanmean(ics_ret) if ics_ret else np.nan
    ic_ret_ir = (np.nanmean(ics_ret) / np.nanstd(ics_ret)) if len(ics_ret) > 1 else np.nan
    ic_prob = np.nanmean(ics_prob) if ics_prob else np.nan

    # ---- (2) 포트폴리오 : 비중첩 60일 리밸런스 ----
    rebal = sig_dates[::REBAL]
    rows = []
    for d in rebal:
        dd = pd.Timestamp(d)
        dfu = date_after(dd, H)
        if dfu is None:
            continue
        day = sig[sig["date"] == dd].copy()
        day = day[day["adv20"] >= ADV_FLOOR].dropna(subset=["exp_ret60"])
        if len(day) < 20:
            continue
        # 실현수익 + 비용
        recs = []
        for _, r in day.iterrows():
            s = close_tbl.get(r["code"]);
            if s is None: continue
            p0 = s.get(dd); pn = s.get(dfu)
            if not (p0 and pn and p0 > 0): continue
            gross = pn / p0 - 1
            net = gross - round_trip_cost(r["adv20"])
            recs.append({"code": r["code"], "exp_ret60": r["exp_ret60"], "gross": gross, "net": net})
        rd = pd.DataFrame(recs)
        if len(rd) < 20: continue
        bench = rd["gross"].mean()
        rd = rd.sort_values("exp_ret60", ascending=False)
        row = {"date": dd.date(), "n": len(rd), "bench_gross": bench}
        for N in TOPN_LIST:
            top = rd.head(N)
            row[f"top{N}_net"] = top["net"].mean()
            row[f"top{N}_gross"] = top["gross"].mean()
            row[f"top{N}_alpha"] = top["net"].mean() - bench
            row[f"top{N}_hit"] = (top["net"] > 0).mean()
        rows.append(row)

    pf = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT_DIR / f"phase2_portfolio_{args.label}.csv", index=False, encoding="utf-8-sig")

    # ---- 요약 ----
    print("\n" + "=" * 64)
    print(f"  Phase 2 결과 — [{args.label}]  (지평 {H}d, 비중첩 리밸런스, ADV≥30억)")
    print("=" * 64)
    print(f"  rank-IC(exp_ret60): mean={ic_ret:.4f}  IR={ic_ret_ir:.2f}  (n_grid={len(ics_ret)})")
    print(f"  rank-IC(prob_top20): mean={ic_prob:.4f}")
    if not pf.empty:
        n_periods = len(pf)
        ann = 252 / H
        print(f"  리밸런스 기간 수: {n_periods}  (각 {H}거래일 비중첩)")
        print(f"  유니버스 벤치마크 60d 평균(gross): {pf['bench_gross'].mean()*100:.2f}%")
        for N in TOPN_LIST:
            net = pf[f"top{N}_net"].mean(); gross = pf[f"top{N}_gross"].mean()
            alpha = pf[f"top{N}_alpha"].mean(); hit = pf[f"top{N}_hit"].mean()
            # 비중첩 복리 누적
            cum = (1 + pf[f"top{N}_net"]).prod() - 1
            print(f"  top{N:2d}: 60d net평균={net*100:6.2f}%  gross={gross*100:6.2f}%  "
                  f"alpha={alpha*100:6.2f}%p  hit={hit*100:4.0f}%  누적net={cum*100:7.1f}%  연환산net≈{net*ann*100:6.1f}%")
    else:
        print("  포트폴리오 기간 부족(데이터 워밍업/범위 확인)")
    print(f"\n  저장: phase2_portfolio_{args.label}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
