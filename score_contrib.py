from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ScoreWeights:
    """
    final_score를 구성하는 항목별 가중치.

    ⚠ 실제 프로젝트에서 사용하는 값으로 교체해서 쓰면 됨.
    예)
      w_ret   = 0.4
      w_prob  = 0.3
      w_qual  = 0.1
      w_tech  = 0.2
      w_risk  = 1.0
      w_mkt   = 1.0
    """

    w_ret: float = 1.0
    w_prob: float = 1.0
    w_qual: float = 1.0
    w_tech: float = 1.0
    # 페널티들은 보통 "점수를 깎는" 개념이라 부호를 -로 적용
    w_risk: float = 1.0
    w_mkt: float = 1.0


def add_score_contributions(
    df: pd.DataFrame,
    weights: Optional[ScoreWeights] = None,
) -> pd.DataFrame:
    """
    각 종목별 final_score를 구성하는 항목별 '기여 점수'와 '기여 비율(%)'을 계산해서
    다음 컬럼을 추가한다.

    - score_return, score_prob, score_qual, score_tech, score_risk, score_market
    - contrib_return, contrib_prob, contrib_qual, contrib_tech, contrib_risk, contrib_market

    contrib_* 는 각 항목이 final_score에 기여한 비율(%)로, 합이 100% 근처가 되며
    음수면 점수를 깎아먹는 비중을 의미한다.

    사용 예시:
        from score_contrib import ScoreWeights, add_score_contributions

        weights = ScoreWeights(
            w_ret=0.4, w_prob=0.3, w_qual=0.1, w_tech=0.2,
            w_risk=1.0, w_mkt=1.0,
        )
        df_with_contrib = add_score_contributions(df, weights)
    """
    if weights is None:
        weights = ScoreWeights()

    df = df.copy()

    # market_penalty 컬럼이 없으면 0으로 취급
    if "market_penalty" not in df.columns:
        df["market_penalty"] = 0.0

    # 1) 항목별 '기여 점수' 계산 (부호 포함)
    df["score_return"] = df["ret_score"] * weights.w_ret
    df["score_prob"] = df["prob_score"] * weights.w_prob
    df["score_qual"] = df["qual_score"] * weights.w_qual
    df["score_tech"] = df["tech_score"] * weights.w_tech

    # penalty 계열은 "점수에서 빼는" 개념이라 - 부호를 붙여서 기여값을 만든다.
    df["score_risk"] = -df["risk_penalty"] * weights.w_risk
    df["score_market"] = -df["market_penalty"] * weights.w_mkt

    component_cols = [
        "score_return",
        "score_prob",
        "score_qual",
        "score_tech",
        "score_risk",
        "score_market",
    ]

    # 2) 항목별 기여점수의 합 (이걸 기준으로 % 계산)
    df["score_components_sum"] = df[component_cols].sum(axis=1)

    # 0 나누기 방지용
    safe_denominator = df["score_components_sum"].replace(0, 1e-9)

    # 3) 항목별 기여 비율(%) 계산
    df["contrib_return"] = df["score_return"] / safe_denominator * 100.0
    df["contrib_prob"] = df["score_prob"] / safe_denominator * 100.0
    df["contrib_qual"] = df["score_qual"] / safe_denominator * 100.0
    df["contrib_tech"] = df["score_tech"] / safe_denominator * 100.0
    df["contrib_risk"] = df["score_risk"] / safe_denominator * 100.0
    df["contrib_market"] = df["score_market"] / safe_denominator * 100.0

    return df
