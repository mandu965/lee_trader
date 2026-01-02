# python/research/metrics.py
import numpy as np
import pandas as pd


# === prediction-level ===

def prediction_regression_metrics(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str,
) -> dict:
    sub = df[[pred_col, true_col]].dropna().copy()
    if sub.empty:
        return {}

    e = sub[pred_col] - sub[true_col]
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    corr = sub[[pred_col, true_col]].corr().iloc[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "n": len(sub),
    }


def quantile_analysis(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str = "realized_return",
    q: int = 5,
) -> pd.Series | None:
    """
    기존 score_backtest_from_labels.py의 quantile 분석 로직 이관
    """
    sub = df[[score_col, ret_col]].dropna().copy()
    if sub.empty:
        return None

    sub["quantile"] = pd.qcut(sub[score_col], q, labels=False) + 1
    return sub.groupby("quantile")[ret_col].mean()


# === strategy-level summary ===

def summarize_strategy_performance(portfolio: pd.DataFrame) -> dict:
    r = portfolio["portfolio_return"].dropna()
    if r.empty:
        return {}

    avg_ret = r.mean()
    hit_rate = (r > 0).mean()
    std_ret = r.std(ddof=1)
    sharpe_like = avg_ret / std_ret if std_ret > 0 else None

    return {
        "avg_return": avg_ret,
        "hit_rate": hit_rate,
        "sharpe_like": sharpe_like,
        "num_periods": len(r),
    }
