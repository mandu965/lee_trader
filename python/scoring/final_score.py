"""KR AI 종합 점수 v9_flow 산출 — 운영 중인 최종 점수 공식 모듈.

[Stage 1] 점수 산출 체인의 핵심 모듈. ranking_builder.py가 호출하여 ranking_final.csv 생성.

final_score 공식:
    final_score = w_ret  * ret_score
                + w_prob * prob_score
                + w_tech * tech_score
                + w_qual * qual_score
                + w_flow * flow_score
                - w_risk * risk_penalty

각 component는 universe(204종목) 내 percentile rank (0~100):
    ret_score:     모델 예측 수익률(pred_return_60d 등) 기반
    prob_score:    prob_top20_60d (LightGBM classifier, OOF calibrated since 2026-05-29)
    tech_score:    기술적 지표
    qual_score:    재무 품질
    flow_score:    수급(외국인·기관 순매수) — v9_flow에서 추가됨
    risk_penalty:  pred_mdd_60d 등 위험 예측 기반

레짐별 가중치 (Bull/Neutral/Defensive):
    Bull       : ret=0.33, prob=0.24, tech=0.23, qual=0.08, flow=0.12
    Neutral    : ret=0.28, prob=0.23, tech=0.21, qual=0.18, flow=0.10
    Defensive  : ret=0.23, prob=0.19, tech=0.16, qual=0.34, flow=0.08

운영 위치:
    production_config.py의 _SCORE_FORMULA_VERSION_DEFAULT = "ranking_builder_v9_flow"
    ranking_builder.py가 이 모듈의 함수를 호출하여 ranking_final.csv 생성.
    이후 build_ai_entry_quality_score(Stage 2) → build_ai_filtered_top_candidates(Stage 3) 순으로 이어짐.

설계 이력:
- 2026-05-14: v9_flow 적용 — flow_score 도입으로 수급 신호가 종합 점수에 직접 반영.
  이전 버전(v8 이하)에서 누락되었던 한국 시장 특유의 외국인·기관 수급 모멘텀 보완.
- 2026-05-23: pred_return_30d 학습 타겟 추가, prob_top20_90d 제거.
- 2026-05-29: prob_top20_60d 분류기 calibration 누설 수정 — OOF 방식 적용.

설계 메모 (2026-05-29 추가):
- 레짐별 가중치 수치의 출처는 추적 불가능. 백테스트 검증 없이 초기 결정 후 유지.
- 운영 데이터 누적 후 정량적 검증 권장 (2026-07~08 60d 만기 도달 이후).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Production operating weight profile:
# positive axes = ret_score, prob_score, tech_score, qual_score, flow_score
# deduction axis = risk_penalty
@dataclass(frozen=True)
class WeightProfile:
    profile: str
    ret: float
    prob: float
    tech: float
    qual: float
    flow: float
    risk_penalty: float


# v9_flow: flow_score component added (2026-05-14)
# Existing positive-axis weights reduced proportionally to accommodate flow=0.08~0.12.
BULL_WEIGHT_PROFILE = WeightProfile(
    profile="bull_service_growth_lead",
    ret=0.33,
    prob=0.24,
    tech=0.23,
    qual=0.08,
    flow=0.12,
    risk_penalty=0.40,
)

NEUTRAL_WEIGHT_PROFILE = WeightProfile(
    profile="neutral_service_balanced",
    ret=0.28,
    prob=0.23,
    tech=0.21,
    qual=0.18,
    flow=0.10,
    risk_penalty=0.65,
)

DEFENSIVE_WEIGHT_PROFILE = WeightProfile(
    profile="defensive_service_carry",
    ret=0.23,
    prob=0.19,
    tech=0.16,
    qual=0.34,
    flow=0.08,
    risk_penalty=0.80,
)

FINAL_SCORE_REQUIRED_INPUTS = [
    "date",
    "code",
    "pred_return_60d",
    "pred_return_90d",
    "prob_top20_60d",
    "pred_mdd_60d",
    "pred_mdd_90d",
]

FINAL_SCORE_OPTIONAL_INPUTS = [
    "pred_return_30d",
    "pred_mdd_30d",
    "prob_top20_30d",
    "flow_foreign_net_5d",
    "flow_inst_net_5d",
    # prob_top20_90d is retained as an optional stored / research signal.
    # The operating prob_score policy uses prob_top20_60d only.
    "prob_top20_90d",
    "quality_score",
    "composite",
    "score_score",
    "vol_20",
    "vol_60",
    "vol_ma_20",
    "volume",
    "mom_20",
    "close_over_ma20",
    "rsi_14",
    "vol_ratio_20",
    "ma_5",
    "ma_20",
    "ma_60",
    "ret_5d",
    "ret_10d",
    "close",
    "per",
    "pe",
    "pbr",
    "pb",
    "psr",
    "ps",
    "ev_ebitda",
    "ev_to_ebitda",
    "price_to_book",
    "price_to_sales",
    "price_to_earnings",
    "earnings_yield",
    "book_to_price",
    "free_cash_flow_yield",
    "dividend_yield",
    "regime",
]


def _series_like(df: pd.DataFrame, value, *, dtype="float64") -> pd.Series:
    return pd.Series(value, index=df.index, dtype=dtype)


def clip01(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).clip(lower=lower, upper=upper)


def percentile_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return _series_like(df, np.nan)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True, method="average") * 100.0

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def percentile01_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return _series_like(df, np.nan)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True, method="average")

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def winsorize_by_date(df: pd.DataFrame, col: str, *, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if col not in df.columns:
        return _series_like(df, np.nan)

    def _clip(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        valid = s.dropna()
        if valid.empty:
            return s
        lo = valid.quantile(lower_q)
        hi = valid.quantile(upper_q)
        return s.clip(lower=lo, upper=hi)

    return df.groupby("date", group_keys=False)[col].transform(_clip)


def average_available(parts: list[pd.Series], index: pd.Index, *, fill_value: float = np.nan) -> pd.Series:
    valid_parts = [pd.to_numeric(part, errors="coerce") for part in parts if part is not None]
    if not valid_parts:
        return pd.Series(fill_value, index=index, dtype=float)
    frame = pd.concat(valid_parts, axis=1)
    out = frame.mean(axis=1, skipna=True)
    if pd.isna(fill_value):
        return out
    return out.fillna(fill_value)


def _is_usable_legacy_tech_source(base: pd.DataFrame, col: str) -> bool:
    if col not in base.columns:
        return False
    series = pd.to_numeric(base[col], errors="coerce")
    nonnull_ratio = float(series.notna().mean()) if len(series) else 0.0
    unique_count = int(series.dropna().nunique())
    usable = nonnull_ratio >= 0.80 and unique_count >= 10
    logging.info(
        "Legacy tech source check: col=%s nonnull_ratio=%.3f unique_count=%d usable=%s",
        col,
        nonnull_ratio,
        unique_count,
        usable,
    )
    return usable


def compute_feature_based_tech_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    index = base.index

    for col in ["close", "ma_5", "ma_20", "ma_60", "rsi_14", "vol_ratio_20", "vol_ma_20", "volume"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if "rsi_14" in base.columns:
        base["rsi_14"] = base["rsi_14"].clip(lower=0.0, upper=100.0)
    if "vol_ratio_20" in base.columns:
        base["vol_ratio_20"] = base["vol_ratio_20"].clip(lower=0.0, upper=3.0)

    trend_position = None
    if "close_over_ma20" in base.columns:
        base["close_over_ma20_win"] = winsorize_by_date(base, "close_over_ma20")
        trend_position = percentile_by_date(base, "close_over_ma20_win")

    ma_ready = all(col in base.columns for col in ["close", "ma_5", "ma_20", "ma_60"])
    if ma_ready:
        cond_100 = (base["close"] >= base["ma_5"]) & (base["ma_5"] >= base["ma_20"]) & (base["ma_20"] >= base["ma_60"])
        cond_75 = (base["close"] >= base["ma_5"]) & (base["ma_5"] >= base["ma_20"])
        cond_55 = base["close"] >= base["ma_20"]
        cond_35 = (base["close"] < base["ma_20"]) & (base["ma_20"] >= base["ma_60"])
        base["tech_trend_alignment"] = np.select(
            [cond_100, cond_75, cond_55, cond_35],
            [100.0, 75.0, 55.0, 35.0],
            default=15.0,
        )
    else:
        base["tech_trend_alignment"] = 50.0

    base["tech_trend_score"] = average_available([trend_position, base["tech_trend_alignment"]], index, fill_value=50.0)

    momentum_parts = []
    for src_col, out_col in [("ret_5d", "tech_ret_5d_win"), ("ret_10d", "tech_ret_10d_win"), ("mom_20", "tech_mom_20_win")]:
        if src_col in base.columns:
            base[out_col] = winsorize_by_date(base, src_col)
            momentum_parts.append(percentile_by_date(base, out_col))
    if "rsi_14" in base.columns:
        base["tech_rsi_score"] = (100.0 - ((base["rsi_14"] - 60.0).abs() * 2.0)).clip(lower=0.0, upper=100.0)
    else:
        base["tech_rsi_score"] = 50.0
    momentum_parts.append(base["tech_rsi_score"])
    base["tech_momentum_score"] = average_available(momentum_parts, index, fill_value=50.0)

    stability_parts = []
    if "vol_20" in base.columns:
        base["tech_vol_20_win"] = winsorize_by_date(base, "vol_20")
        stability_parts.append(100.0 - percentile_by_date(base, "tech_vol_20_win"))
    if "vol_60" in base.columns:
        base["tech_vol_60_win"] = winsorize_by_date(base, "vol_60")
        stability_parts.append(100.0 - percentile_by_date(base, "tech_vol_60_win"))
    base["tech_stability_score"] = average_available(stability_parts, index, fill_value=50.0)

    volume_parts = []
    if "vol_ma_20" in base.columns:
        base["tech_vol_ma_20_win"] = winsorize_by_date(base, "vol_ma_20")
        volume_parts.append(percentile_by_date(base, "tech_vol_ma_20_win"))
    if "vol_ratio_20" in base.columns:
        volume_parts.append(percentile_by_date(base, "vol_ratio_20"))
    if "volume" in base.columns:
        base["tech_volume_win"] = winsorize_by_date(base, "volume")
        volume_parts.append(percentile_by_date(base, "tech_volume_win"))
    base["tech_volume_score"] = average_available(volume_parts, index, fill_value=50.0)

    if "vol_ma_20" in base.columns:
        base["tech_liquidity_pct"] = percentile_by_date(base, "tech_vol_ma_20_win" if "tech_vol_ma_20_win" in base.columns else "vol_ma_20")
    elif "volume" in base.columns:
        base["tech_liquidity_pct"] = percentile_by_date(base, "tech_volume_win" if "tech_volume_win" in base.columns else "volume")
    else:
        base["tech_liquidity_pct"] = 50.0

    base["tech_liquidity_guard"] = np.select(
        [base["tech_liquidity_pct"] < 10.0, base["tech_liquidity_pct"] < 20.0],
        [0.72, 0.90],
        default=1.00,
    )

    base["tech_score"] = (
        0.30 * base["tech_trend_score"].fillna(50.0)
        + 0.35 * base["tech_momentum_score"].fillna(50.0)
        + 0.20 * base["tech_stability_score"].fillna(50.0)
        + 0.15 * base["tech_volume_score"].fillna(50.0)
    )
    base["tech_score"] = (
        base["tech_score"].fillna(50.0) * pd.to_numeric(base["tech_liquidity_guard"], errors="coerce").fillna(1.0)
    ).clip(lower=0.0, upper=100.0)
    base["tech_source"] = "feature_v1"
    return base


def compute_tech_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    if _is_usable_legacy_tech_source(base, "composite"):
        base["tech_score"] = percentile_by_date(base, "composite")
        base["tech_source"] = "scores_final.composite"
    elif _is_usable_legacy_tech_source(base, "score_score"):
        base["tech_score"] = percentile_by_date(base, "score_score")
        base["tech_source"] = "scores_final.score_score"
    else:
        base = compute_feature_based_tech_score(base)
        return base

    for col in ["tech_trend_score", "tech_momentum_score", "tech_stability_score", "tech_volume_score"]:
        if col not in base.columns:
            base[col] = np.nan
    if "tech_liquidity_guard" not in base.columns:
        base["tech_liquidity_guard"] = 1.0
    return base


def compute_ret_and_pred_scores(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    pred_60 = None
    pred_90 = None

    if "pred_return_60d" in base.columns:
        base["pred_score_60"] = percentile_by_date(base, "pred_return_60d")
        pred_60 = base["pred_score_60"]
        base["ret_rank_60"] = percentile01_by_date(base, "pred_return_60d")
        base["pred_return_60d_pct01"] = base["ret_rank_60"]
    else:
        base["ret_rank_60"] = np.nan
        base["pred_return_60d_pct01"] = np.nan
        base["pred_score_60"] = np.nan

    if "pred_return_90d" in base.columns:
        base["pred_score_90"] = percentile_by_date(base, "pred_return_90d")
        pred_90 = base["pred_score_90"]
        base["ret_rank_90"] = percentile01_by_date(base, "pred_return_90d")
        base["pred_return_90d_pct01"] = base["ret_rank_90"]
    else:
        base["ret_rank_90"] = np.nan
        base["pred_return_90d_pct01"] = np.nan
        base["pred_score_90"] = np.nan

    base["ret_score"] = 100.0 * (0.7 * base["ret_rank_60"].fillna(0.0) + 0.3 * base["ret_rank_90"].fillna(0.0))
    base["ret_score_v11"] = base["ret_score"]

    if (pred_60 is not None) and (pred_90 is not None):
        base["pred_score"] = 0.6 * pred_60 + 0.4 * pred_90
    elif pred_60 is not None:
        base["pred_score"] = pred_60
    elif pred_90 is not None:
        base["pred_score"] = pred_90
    else:
        base["pred_score"] = np.nan

    return base


def compute_prob_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    Operating probability score policy.

    - Production prob_score uses prob_top20_60d only.
    - prob_top20_90d may be present as an optional stored / research signal,
      but it is not blended into the operating final_score probability axis.
    - prob_score_raw keeps the absolute 60d probability conversion, while
      prob_score keeps the same-date relative operating rank.
    """
    base = base.copy()
    if "prob_top20_60d" in base.columns:
        prob = pd.to_numeric(base["prob_top20_60d"], errors="coerce")
        base["prob_score_missing"] = prob.isna()
        base["prob_score_raw"] = clip01(prob.fillna(0.0) * 100.0, 0.0, 100.0)
        base["prob_rank_pct"] = percentile01_by_date(base.assign(prob_top20_60d=prob), "prob_top20_60d")
        base["prob_rank_pct"] = base.groupby("date", group_keys=False)["prob_rank_pct"].transform(lambda s: s.fillna(s.median()))
        base["prob_rank_pct"] = pd.to_numeric(base["prob_rank_pct"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
        base["prob_score"] = (base["prob_rank_pct"] * 100.0).clip(lower=0.0, upper=100.0)
    else:
        base["prob_score_missing"] = True
        base["prob_score_raw"] = np.nan
        base["prob_rank_pct"] = 0.5
        base["prob_score"] = 50.0
    return base


def compute_qual_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    if "quality_score" in base.columns:
        base["quality_score_raw"] = pd.to_numeric(base.get("quality_score"), errors="coerce")
        base["quality_score_rank"] = percentile_by_date(base, "quality_score")

        confidence = (
            pd.to_numeric(base.get("quality_score_confidence"), errors="coerce").clip(lower=0.0, upper=100.0)
            if "quality_score_confidence" in base.columns
            else _series_like(base, 100.0)
        )
        missing_ratio = (
            pd.to_numeric(base.get("quality_missing_ratio"), errors="coerce").clip(lower=0.0, upper=1.0)
            if "quality_missing_ratio" in base.columns
            else _series_like(base, 0.0)
        )

        # Separate the quality signal itself from its reliability.
        # - confidence below 60 and missing_ratio above 0.6 are treated as materially weak coverage
        # - even weak coverage retains some signal, but not at full production strength
        confidence_multiplier = (0.35 + 0.65 * (confidence / 100.0)).clip(lower=0.35, upper=1.0)
        missing_multiplier = (1.0 - 0.55 * missing_ratio).clip(lower=0.45, upper=1.0)
        base["quality_score_confidence_multiplier"] = confidence_multiplier
        base["quality_score_missing_multiplier"] = missing_multiplier
        base["quality_score_effective"] = (
            pd.to_numeric(base["quality_score_rank"], errors="coerce")
            * confidence_multiplier.fillna(0.675)
            * missing_multiplier.fillna(0.725)
        )
        base["qual_score"] = pd.to_numeric(base["quality_score_effective"], errors="coerce")
    else:
        base["quality_score_raw"] = np.nan
        base["quality_score_rank"] = np.nan
        base["quality_score_confidence_multiplier"] = np.nan
        base["quality_score_missing_multiplier"] = np.nan
        base["quality_score_effective"] = np.nan
        base["qual_score"] = np.nan
    return base


def compute_flow_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    flow_score = 0.6 * percentile(flow_foreign_net_5d) + 0.4 * percentile(flow_inst_net_5d)
    Range 0-100; NaN when both inputs are absent.
    """
    base = base.copy()
    has_foreign = "flow_foreign_net_5d" in base.columns
    has_inst = "flow_inst_net_5d" in base.columns
    if has_foreign and has_inst:
        foreign_pct = percentile_by_date(base, "flow_foreign_net_5d")
        inst_pct = percentile_by_date(base, "flow_inst_net_5d")
        base["flow_score"] = (0.6 * foreign_pct + 0.4 * inst_pct).clip(lower=0.0, upper=100.0)
    elif has_foreign:
        base["flow_score"] = percentile_by_date(base, "flow_foreign_net_5d").clip(lower=0.0, upper=100.0)
    elif has_inst:
        base["flow_score"] = percentile_by_date(base, "flow_inst_net_5d").clip(lower=0.0, upper=100.0)
    else:
        base["flow_score"] = np.nan
    return base


def compute_safety_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    safety_parts = []
    if "vol_20" in base.columns:
        base["vol_20_pct"] = percentile_by_date(base, "vol_20")
        safety_parts.append(100.0 - base["vol_20_pct"])
    if "vol_60" in base.columns:
        base["vol_60_pct"] = percentile_by_date(base, "vol_60")
        safety_parts.append(100.0 - base["vol_60_pct"])

    if safety_parts:
        base["safety_score"] = sum(safety_parts) / len(safety_parts)
    else:
        base["safety_score"] = np.nan
    return base


def compute_liquidity_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    if "vol_ma_20" in base.columns:
        base["liquidity_score"] = percentile_by_date(base, "vol_ma_20")
    elif "volume" in base.columns:
        base["liquidity_score"] = percentile_by_date(base, "volume")
    else:
        base["liquidity_score"] = np.nan
    return base


def compute_valuation_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    valuation_candidates: list[pd.Series] = []
    lower_is_better = [
        "per",
        "pe",
        "pbr",
        "pb",
        "psr",
        "ps",
        "ev_ebitda",
        "ev_to_ebitda",
        "price_to_book",
        "price_to_sales",
        "price_to_earnings",
    ]
    higher_is_better = [
        "earnings_yield",
        "book_to_price",
        "free_cash_flow_yield",
        "dividend_yield",
    ]

    for col in lower_is_better:
        if col in base.columns:
            valuation_candidates.append(100.0 - percentile_by_date(base, col))
    for col in higher_is_better:
        if col in base.columns:
            valuation_candidates.append(percentile_by_date(base, col))

    base["valuation_score"] = sum(valuation_candidates) / len(valuation_candidates) if valuation_candidates else 50.0
    return base


def attach_operational_score_aliases(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["return_score"] = pd.to_numeric(base.get("ret_score"), errors="coerce")
    base["probability_score"] = pd.to_numeric(base.get("prob_score"), errors="coerce")
    base["technical_score"] = pd.to_numeric(base.get("tech_score"), errors="coerce")
    if "quality_score" in base.columns:
        base["quality_score"] = pd.to_numeric(base.get("quality_score"), errors="coerce")
    else:
        base["quality_score"] = pd.to_numeric(base.get("qual_score"), errors="coerce")
    base["valuation_score"] = pd.to_numeric(base.get("valuation_score"), errors="coerce")
    return base


def compute_component_scores(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base = compute_tech_score(base)
    base = compute_ret_and_pred_scores(base)
    base = compute_prob_score(base)
    base = compute_qual_score(base)
    base = compute_flow_score(base)
    base = compute_safety_score(base)
    base = compute_liquidity_score(base)
    base = compute_valuation_score(base)
    base = attach_operational_score_aliases(base)
    return base


def baseline_risk_penalty_from_mix(mix_like, penalty_cap: float = 18.0) -> pd.Series:
    mix = pd.to_numeric(pd.Series(mix_like), errors="coerce").fillna(0.0).abs()
    penalty = np.select(
        [mix <= 0.10, mix <= 0.15, mix <= 0.20],
        [0.0, (mix - 0.10) * 40.0, 2.0 + (mix - 0.15) * 80.0],
        default=6.0 + (mix - 0.20) * 120.0,
    )
    return pd.to_numeric(pd.Series(penalty, index=mix.index), errors="coerce").fillna(0.0).clip(lower=0.0, upper=penalty_cap)


def compute_risk_penalty(base: pd.DataFrame, *, penalty_cap: float = 18.0) -> pd.DataFrame:
    base = base.copy()
    if "pred_mdd_60d" in base.columns or "pred_mdd_90d" in base.columns:
        mdd_60 = pd.to_numeric(base.get("pred_mdd_60d"), errors="coerce").abs()
        mdd_90 = pd.to_numeric(base.get("pred_mdd_90d"), errors="coerce").abs()
        base["pred_mdd_mix"] = 0.6 * mdd_60.fillna(0.0) + 0.4 * mdd_90.fillna(0.0)
        base["risk_penalty"] = baseline_risk_penalty_from_mix(base["pred_mdd_mix"], penalty_cap=penalty_cap)
    else:
        base["pred_mdd_mix"] = np.nan
        base["risk_penalty"] = 0.0
    return base


def ensure_regime_column(
    df: pd.DataFrame,
    *,
    log_distribution: bool = False,
    log_prefix: str = "regime",
) -> pd.DataFrame:
    df = df.copy()
    regime = df.get("regime")
    if regime is None:
        df["regime"] = "defensive"
    else:
        regime_clean = regime.astype(str).str.strip().str.lower()
        df["regime"] = np.select(
            [regime_clean.eq("bull"), regime_clean.eq("neutral")],
            ["bull", "neutral"],
            default="defensive",
        )
    if "regime_reason" not in df.columns:
        df["regime_reason"] = "fallback_defensive"

    if log_distribution:
        regime_counts = df["regime"].fillna("NA").value_counts(dropna=False).to_dict()
        logging.info("%s regime distribution: %s", log_prefix, regime_counts)
    return df


def detect_market_regime(df: pd.DataFrame, market_info: dict, market_history: pd.DataFrame | None = None) -> tuple[str, str]:
    conditions: list[tuple[str, bool | None]] = []
    missing_inputs: list[str] = []

    close = pd.to_numeric(market_info.get("kospi_close"), errors="coerce")
    ma20 = pd.to_numeric(market_info.get("kospi_ma20"), errors="coerce")
    if pd.notna(close) and pd.notna(ma20):
        conditions.append(("close_gt_ma20", bool(close > ma20)))
    else:
        missing_inputs.append("close_gt_ma20")

    ma60 = pd.Series(dtype=float)
    recent_20d_return = pd.NA
    if market_history is not None and not market_history.empty and "kospi_close" in market_history.columns:
        hist = market_history.copy()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist["kospi_close"] = pd.to_numeric(hist["kospi_close"], errors="coerce")
        hist = hist.sort_values("date")
        hist["kospi_ma60"] = hist["kospi_close"].rolling(60, min_periods=20).mean()
        hist["recent_20d_return"] = hist["kospi_close"] / hist["kospi_close"].shift(20) - 1.0
        last = hist.iloc[-1]
        ma60 = pd.to_numeric(last.get("kospi_ma60"), errors="coerce")
        recent_20d_return = pd.to_numeric(last.get("recent_20d_return"), errors="coerce")

    if pd.notna(ma20) and pd.notna(ma60):
        conditions.append(("ma20_gt_ma60", bool(ma20 > ma60)))
    else:
        missing_inputs.append("ma20_gt_ma60")

    if pd.notna(recent_20d_return):
        conditions.append(("recent_20d_return_gt_0.03", bool(recent_20d_return > 0.03)))
    else:
        missing_inputs.append("recent_20d_return_gt_0.03")

    breadth_20d = pd.NA
    if "close_over_ma20" in df.columns:
        close_over_ma20 = pd.to_numeric(df.get("close_over_ma20"), errors="coerce")
        if close_over_ma20.notna().any():
            breadth_20d = float(close_over_ma20.gt(0).mean())
    if pd.notna(breadth_20d):
        conditions.append(("breadth_20d_gt_0.55", bool(breadth_20d > 0.55)))
    else:
        missing_inputs.append("breadth_20d_gt_0.55")

    volatility_5d = pd.to_numeric(market_info.get("volatility_5d"), errors="coerce")
    if pd.notna(volatility_5d):
        volatility_risk_flag = bool(volatility_5d > 0.025)
        conditions.append(("volatility_risk_flag_false", not volatility_risk_flag))
    else:
        volatility_risk_flag = None
        missing_inputs.append("volatility_risk_flag_false")

    true_conditions = [name for name, value in conditions if value is True]
    true_count = len(true_conditions)
    if true_count >= 4:
        regime = "bull"
    elif true_count >= 2:
        regime = "neutral"
    else:
        regime = "defensive"

    reason_parts = [f"met_conditions={','.join(true_conditions) if true_conditions else 'none'}", f"true_count={true_count}"]
    if pd.notna(breadth_20d):
        reason_parts.append(f"breadth_20d={breadth_20d:.3f}")
    if pd.notna(recent_20d_return):
        reason_parts.append(f"recent_20d_return={float(recent_20d_return):.4f}")
    if pd.notna(ma60):
        reason_parts.append(f"ma60={float(ma60):.2f}")
    if volatility_risk_flag is not None:
        reason_parts.append(f"volatility_risk_flag={volatility_risk_flag}")
    if missing_inputs:
        reason_parts.append(f"missing_inputs={','.join(missing_inputs)}")

    return regime, "; ".join(reason_parts)


def attach_market_columns(
    df: pd.DataFrame,
    market_up: bool,
    market_info: dict,
    market_history: pd.DataFrame | None = None,
    *,
    log_distribution: bool = True,
    log_prefix: str = "_attach_market_columns",
) -> pd.DataFrame:
    df = df.copy()
    df["market_up"] = market_up
    df["market_status_date"] = market_info.get("date")
    df["market_kospi_close"] = pd.to_numeric(market_info.get("kospi_close"), errors="coerce")
    df["market_kospi_ma20"] = pd.to_numeric(market_info.get("kospi_ma20"), errors="coerce")
    df["market_vol_5d"] = pd.to_numeric(market_info.get("volatility_5d"), errors="coerce")
    df["market_foreign_5d"] = pd.to_numeric(market_info.get("foreign_net_5d"), errors="coerce")
    regime, regime_reason = detect_market_regime(df, market_info, market_history)
    df["regime"] = regime
    df["regime_reason"] = regime_reason
    return ensure_regime_column(df, log_distribution=log_distribution, log_prefix=log_prefix)


def attach_market_columns_by_date(df: pd.DataFrame, market_status_history: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if market_status_history is None or market_status_history.empty or "date" not in market_status_history.columns:
        return ensure_regime_column(out)

    hist = market_status_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    if hist.empty:
        return ensure_regime_column(out)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    date_frames = []
    for date_value, date_df in out.groupby("date", dropna=False):
        date_block = date_df.copy()
        if pd.isna(date_value):
            date_frames.append(ensure_regime_column(date_block))
            continue
        hist_slice = hist.loc[hist["date"] <= date_value].copy()
        if hist_slice.empty:
            date_frames.append(ensure_regime_column(date_block))
            continue
        last = hist_slice.iloc[-1]
        market_up_raw = last.get("market_up")
        if isinstance(market_up_raw, bool):
            market_up = market_up_raw
        else:
            market_up = str(market_up_raw).strip().lower() in {"true", "1", "t", "y", "yes"}
        market_info = {
            "date": last.get("date"),
            "kospi_close": last.get("kospi_close"),
            "kospi_ma20": last.get("kospi_ma20"),
            "volatility_5d": last.get("volatility_5d"),
            "foreign_net_5d": last.get("foreign_net_5d"),
        }
        date_frames.append(
            attach_market_columns(
                date_block,
                market_up=market_up,
                market_info=market_info,
                market_history=hist_slice,
                log_distribution=False,
                log_prefix="attach_market_columns_by_date",
            )
        )

    result = pd.concat(date_frames).sort_index()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return ensure_regime_column(result)


def resolve_core_weight_profile(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    regime = base.get("regime", _series_like(base, "defensive", dtype="object")).astype(str).str.lower()
    bull_mask = regime.eq("bull")
    neutral_mask = regime.eq("neutral")

    base["weight_profile"] = np.select(
        [bull_mask, neutral_mask],
        [BULL_WEIGHT_PROFILE.profile, NEUTRAL_WEIGHT_PROFILE.profile],
        default=DEFENSIVE_WEIGHT_PROFILE.profile,
    )
    base["w_tech"] = np.select([bull_mask, neutral_mask], [BULL_WEIGHT_PROFILE.tech, NEUTRAL_WEIGHT_PROFILE.tech], default=DEFENSIVE_WEIGHT_PROFILE.tech)
    base["w_tech_base"] = base["w_tech"]
    base["w_ret"] = np.select([bull_mask, neutral_mask], [BULL_WEIGHT_PROFILE.ret, NEUTRAL_WEIGHT_PROFILE.ret], default=DEFENSIVE_WEIGHT_PROFILE.ret)
    base["w_ret_base"] = base["w_ret"]
    base["w_prob"] = np.select([bull_mask, neutral_mask], [BULL_WEIGHT_PROFILE.prob, NEUTRAL_WEIGHT_PROFILE.prob], default=DEFENSIVE_WEIGHT_PROFILE.prob)
    base["w_prob_base"] = base["w_prob"]
    base["w_qual"] = np.select([bull_mask, neutral_mask], [BULL_WEIGHT_PROFILE.qual, NEUTRAL_WEIGHT_PROFILE.qual], default=DEFENSIVE_WEIGHT_PROFILE.qual)
    base["w_qual_base"] = base["w_qual"]
    base["w_flow"] = np.select([bull_mask, neutral_mask], [BULL_WEIGHT_PROFILE.flow, NEUTRAL_WEIGHT_PROFILE.flow], default=DEFENSIVE_WEIGHT_PROFILE.flow)
    base["w_flow_base"] = base["w_flow"]
    base["w_risk_penalty"] = np.select(
        [bull_mask, neutral_mask],
        [BULL_WEIGHT_PROFILE.risk_penalty, NEUTRAL_WEIGHT_PROFILE.risk_penalty],
        default=DEFENSIVE_WEIGHT_PROFILE.risk_penalty,
    )
    # safety/liquidity remain diagnostic-only operating columns and are not
    # direct positive axes in the production final_score formula.
    base["w_safety"] = 0.0
    base["w_liquidity"] = 0.0
    return base


def attach_component_integrity_flags(base: pd.DataFrame, core_component_columns: list[str]) -> pd.DataFrame:
    base = base.copy()
    for col in core_component_columns:
        missing_col = f"{col}_missing"
        if col == "prob_score" and "prob_score_missing" in base.columns:
            base[missing_col] = base["prob_score_missing"].fillna(True).astype(bool)
        else:
            source = base[col] if col in base.columns else _series_like(base, np.nan)
            base[missing_col] = pd.to_numeric(source, errors="coerce").isna()

    fallback_map = {
        "ret_score_fallback_used": "ret_score_missing",
        "prob_score_fallback_used": "prob_score_missing",
        "qual_score_fallback_used": "qual_score_missing",
        "tech_score_fallback_used": "tech_score_missing",
        "safety_score_fallback_used": "safety_score_missing",
        "liquidity_score_fallback_used": "liquidity_score_missing",
    }
    for fallback_col, missing_col in fallback_map.items():
        base[fallback_col] = base.get(missing_col, False)

    fallback_cols = list(fallback_map.keys())
    base["fallback_count"] = (
        pd.DataFrame({col: base[col].fillna(False).astype(bool) for col in fallback_cols}, index=base.index).sum(axis=1).astype(int)
    )
    return base


def compute_score_explain(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["contrib_tech"] = pd.to_numeric(base["w_tech"], errors="coerce").fillna(0.0) * pd.to_numeric(base["tech_score"], errors="coerce").fillna(0.0)
    base["contrib_ret"] = pd.to_numeric(base["w_ret"], errors="coerce").fillna(0.0) * pd.to_numeric(base["ret_score"], errors="coerce").fillna(0.0)
    base["contrib_prob"] = pd.to_numeric(base["w_prob"], errors="coerce").fillna(0.0) * pd.to_numeric(base["prob_score"], errors="coerce").fillna(0.0)
    base["contrib_qual"] = pd.to_numeric(base["w_qual"], errors="coerce").fillna(0.0) * pd.to_numeric(base["qual_score"], errors="coerce").fillna(0.0)
    base["contrib_flow"] = pd.to_numeric(base.get("w_flow"), errors="coerce").fillna(0.0) * pd.to_numeric(base.get("flow_score"), errors="coerce").fillna(50.0)
    # valuation_score is retained as a compatibility / diagnostic column, but
    # it is not part of the production operating final_score formula.
    base["contrib_valuation"] = 0.0
    base["contrib_theme"] = 0.0
    base["contrib_safety"] = 0.0
    base["contrib_liquidity"] = 0.0
    base["contrib_penalty"] = -pd.to_numeric(base["w_risk_penalty"], errors="coerce").fillna(0.0) * pd.to_numeric(base["risk_penalty"], errors="coerce").fillna(0.0)

    base["score_contribution_ret"] = base["contrib_ret"]
    base["score_contribution_prob"] = base["contrib_prob"]
    base["score_contribution_tech"] = base["contrib_tech"]
    base["score_contribution_qual"] = base["contrib_qual"]
    base["score_contribution_flow"] = base["contrib_flow"]
    base["score_contribution_safety"] = base["contrib_safety"]
    base["score_contribution_liquidity"] = base["contrib_liquidity"]
    base["score_contribution_theme"] = base["contrib_theme"]
    base["score_contribution_risk"] = base["contrib_penalty"]
    base["final_score_raw"] = (
        base["contrib_tech"].fillna(0.0)
        + base["contrib_ret"].fillna(0.0)
        + base["contrib_prob"].fillna(0.0)
        + base["contrib_qual"].fillna(0.0)
        + base["contrib_flow"].fillna(0.0)
    )
    return base


def apply_baseline_final_score(
    base: pd.DataFrame,
    *,
    fill_score_columns: bool = True,
    include_explain: bool = True,
) -> pd.DataFrame:
    """
    Apply the production operating final_score formula (v9_flow).

    final_score =
        w_ret   * ret_score
      + w_prob  * prob_score
      + w_tech  * tech_score
      + w_qual  * qual_score
      + w_flow  * flow_score
      - w_risk_penalty * risk_penalty

    flow_score missing → filled with 50.0 (neutral percentile) so that stocks
    without flow data are not penalized relative to the universe.
    valuation_score, safety_score, and liquidity_score remain diagnostic-only.
    """
    out = base.copy()
    out = compute_risk_penalty(out)
    out = ensure_regime_column(out, log_distribution=True, log_prefix="apply_baseline_final_score")
    out = resolve_core_weight_profile(out)

    score_cols = ["tech_score", "ret_score", "prob_score", "qual_score"]
    if fill_score_columns:
        for col in score_cols:
            source = out[col] if col in out.columns else _series_like(out, np.nan)
            out[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    # flow_score: missing data → neutral 50.0, not 0.0, to avoid penalizing stocks with no flow
    if "flow_score" not in out.columns:
        out["flow_score"] = 50.0
    else:
        out["flow_score"] = pd.to_numeric(out["flow_score"], errors="coerce").fillna(50.0)
    if "valuation_score" in out.columns:
        out["valuation_score"] = pd.to_numeric(out.get("valuation_score"), errors="coerce")
    out["risk_penalty"] = pd.to_numeric(out.get("risk_penalty"), errors="coerce").fillna(0.0)

    out["final_score_before_theme"] = (
        out["w_ret_base"] * out["ret_score"]
        + out["w_prob_base"] * out["prob_score"]
        + out["w_tech_base"] * out["tech_score"]
        + out["w_qual_base"] * out["qual_score"]
        + out["w_flow_base"] * out["flow_score"]
        - out["w_risk_penalty"] * out["risk_penalty"]
    )
    out["final_score"] = pd.to_numeric(out["final_score_before_theme"], errors="coerce").clip(lower=0.0, upper=100.0)
    out["final_score_before_theme"] = out["final_score"]
    if include_explain:
        out = compute_score_explain(out)
    return out
