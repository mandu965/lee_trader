# python/research/scoring.py
import pandas as pd
try:
    from .config import BacktestConfig
except ImportError:
    from config import BacktestConfig


def apply_score_weights(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    df: ret_score, prob_score, qual_score, tech_score, risk_penalty 포함
    """
    w = config.weights
    df = df.copy()

    df["final_score_custom"] = (
        w.w_ret   * df["ret_score"]
        + w.w_prob  * df["prob_score"]
        + w.w_qual  * df["qual_score"]
        + w.w_tech  * df["tech_score"]
        - w.w_risk  * df["risk_penalty"]
    )

    return df
