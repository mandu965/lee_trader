import pandas as pd
import numpy as np


def clip(x, lo, hi):
    """Safely clip numeric array/Series to [lo, hi]."""
    return np.clip(x, lo, hi)


def _z_to_score(series: pd.Series) -> pd.Series:
    """Z-score to 0~100 scale with center 50, step 10σ, clipped to [0,100]."""
    mean = series.mean()
    std = series.std(ddof=0)
    std = std if std and std > 0 else 1e-6
    z = (series - mean) / std
    return clip(50 + 10 * z, 0, 100)


def compute_ret_score(df: pd.DataFrame, alpha: float = 3.0) -> pd.Series:
    """
    기대수익 기반 점수.
    r_comb = 0.6*pred_return_60d + 0.4*pred_return_90d
    r_adj = r_comb / (1 + alpha*abs(pred_mdd_comb))
    """
    pred_mdd_comb = 0.6 * df["pred_mdd_60d"] + 0.4 * df["pred_mdd_90d"]
    r_comb = 0.6 * df["pred_return_60d"] + 0.4 * df["pred_return_90d"]
    r_adj = r_comb / (1 + alpha * pred_mdd_comb.abs())
    return _z_to_score(r_adj.fillna(0))


def compute_prob_score(df: pd.DataFrame) -> pd.Series:
    """
    상위 20% 확률 점수 (0~100).
    p_comb = 0.7*prob_top20_60d + 0.3*prob_top20_90d
    """
    p_comb = 0.7 * df["prob_top20_60d"] + 0.3 * df["prob_top20_90d"]
    return clip(100 * p_comb.fillna(0), 0, 100)


def compute_qual_score(df: pd.DataFrame) -> pd.Series:
    """펀더멘털 퀄리티 점수: quality_score를 유니버스 z-score로 0~100."""
    return _z_to_score(df["quality_score"].fillna(0))


def compute_tech_score(
    df: pd.DataFrame,
    mom_col: str = "mom_60d",
    rsi_col: str = "rsi_14",
    turnover_col: str = "turnover_20d",
) -> pd.Series:
    """
    기술적 흐름 점수.
    tech = 0.5*score_mom + 0.3*score_rsi + 0.2*score_turnover (각각 z-score 0~100)
    """
    score_mom = _z_to_score(df[mom_col].fillna(0))
    score_rsi = _z_to_score(df[rsi_col].fillna(0))
    score_turnover = _z_to_score(df[turnover_col].fillna(0))
    tech = 0.5 * score_mom + 0.3 * score_rsi + 0.2 * score_turnover
    return clip(tech, 0, 100)


def compute_risk_penalty(df: pd.DataFrame, beta: float = 0.7, mdd_ref: float = 0.3) -> pd.Series:
    """
    리스크 패널티 (0.5~1.0).
    risk_mdd = |pred_mdd_comb| / mdd_ref (clipped 0~1)
    risk_vol = vol_20d min-max 정규화 (0~1)
    risk_level = 0.7*risk_mdd + 0.3*risk_vol
    risk_penalty = 1 - beta * risk_level, clipped to [0.5, 1.0]
    """
    pred_mdd_comb = 0.6 * df["pred_mdd_60d"] + 0.4 * df["pred_mdd_90d"]
    risk_mdd = clip(pred_mdd_comb.abs() / mdd_ref, 0, 1)

    vol = df["vol_20d"].fillna(df["vol_20d"].median()).fillna(0)
    vol_min = vol.min()
    vol_max = vol.max()
    vol_range = vol_max - vol_min
    if vol_range <= 0:
        risk_vol = pd.Series(0, index=df.index)
    else:
        risk_vol = (vol - vol_min) / vol_range
        risk_vol = clip(risk_vol, 0, 1)

    risk_level = 0.7 * risk_mdd + 0.3 * risk_vol
    penalty = 1 - beta * risk_level
    return clip(penalty, 0.5, 1.0)


def compute_market_regime_score(market_row: dict | pd.Series) -> float:
    """
    시장 레짐 보정 (0.7~1.1).
    base: bull=1.0, sideways=0.9, bear=0.8
    외국인 수급 5d >0 +0.05, <0 -0.05
    """
    regime = str(market_row.get("market_regime", "")).lower()
    foreign_5d = market_row.get("market_foreign_5d", 0) or 0

    if regime == "bull":
        base = 1.0
    elif regime == "sideways":
        base = 0.9
    elif regime == "bear":
        base = 0.8
    else:
        base = 0.9  # default neutral

    adj = 0.05 if foreign_5d > 0 else (-0.05 if foreign_5d < 0 else 0.0)
    return float(clip(base + adj, 0.7, 1.1))


def compute_final_score_v5(
    df: pd.DataFrame,
    market_row: dict | pd.Series,
    pred_score_default: float = 60.0,
    regime_bonus_scale: float = 20.0,
) -> pd.DataFrame:
    """
    최종 V5 점수 계산 후 컬럼 추가.
    추가 컬럼: ret_score, prob_score, qual_score, tech_score,
               pred_score, risk_penalty, market_regime_score, final_score_v5
    """
    df = df.copy()

    df["ret_score"] = compute_ret_score(df)
    df["prob_score"] = compute_prob_score(df)
    df["qual_score"] = compute_qual_score(df)
    df["tech_score"] = compute_tech_score(df)
    df["pred_score"] = pred_score_default  # 현재는 고정값

    df["risk_penalty"] = compute_risk_penalty(df)
    df["market_regime_score"] = compute_market_regime_score(market_row)

    # Balanced weights: RET 0.28, PROB 0.25, QUAL 0.20, TECH 0.17, MODEL 0.10
    w_ret, w_prob, w_qual, w_tech, w_pred = 0.28, 0.25, 0.20, 0.17, 0.10
    base_score = (
        w_ret * df["ret_score"]
        + w_prob * df["prob_score"]
        + w_qual * df["qual_score"]
        + w_tech * df["tech_score"]
        + w_pred * df["pred_score"]
    )
    risk_adjusted = base_score * df["risk_penalty"]
    # Apply market regime as additive bonus/penalty instead of global multiplier.
    # Neutral regime is 0.9; scale to +/- points with regime_bonus_scale.
    regime_bonus = (df["market_regime_score"] - 0.9) * regime_bonus_scale
    final_score = risk_adjusted + regime_bonus
    df["final_score_v5"] = clip(final_score, 0, 100)

    fill_cols = [
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "pred_score",
        "risk_penalty",
        "market_regime_score",
        "final_score_v5",
    ]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df
