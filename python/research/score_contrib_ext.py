# python/research/score_contrib_ext.py
import pandas as pd
from .config import ScoreWeights


def compute_score_contrib(
    df: pd.DataFrame,
    weights: ScoreWeights,
) -> pd.DataFrame:
    """
    각 score 항목별 절대 점수 + final_score 대비 비율(%) 계산
    """
    d = df.copy()

    d["score_ret"]   = weights.w_ret   * d["ret_score"]
    d["score_prob"]  = weights.w_prob  * d["prob_score"]
    d["score_qual"]  = weights.w_qual  * d["qual_score"]
    d["score_tech"]  = weights.w_tech  * d["tech_score"]
    d["score_risk"]  = -weights.w_risk * d["risk_penalty"]

    d["score_total"] = (
        d["score_ret"] + d["score_prob"] +
        d["score_qual"] + d["score_tech"] +
        d["score_risk"]
    )

    for col in ["score_ret", "score_prob", "score_qual",
                "score_tech", "score_risk"]:
        contrib_col = col.replace("score_", "contrib_")
        d[contrib_col] = d[col] / d["score_total"]

    return d
