# python/research/outcomes.py
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
try:
    from .config import BacktestConfig
except ImportError:
    from config import BacktestConfig


def compute_realized_outcomes(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Add realized_return based on horizon_days forward price move.
    Assumes prices contain date, code, close.
    """
    horizon_days = config.horizon_days
    df = preds.copy()

    df["date_h"] = df["date"].apply(
        lambda d: (datetime.combine(d, datetime.min.time()) + timedelta(days=horizon_days)).date()
    )

    price_t = prices.rename(columns={"close": "price_t"})
    df = df.merge(
        price_t[["date", "code", "price_t"]],
        on=["date", "code"],
        how="left",
    )

    price_th = prices.rename(columns={"date": "date_h", "close": "price_th"})
    df = df.merge(
        price_th[["date_h", "code", "price_th"]],
        on=["date_h", "code"],
        how="left",
    )

    df["realized_return"] = np.where(
        (df["price_t"] > 0) & (df["price_th"].notna()),
        (df["price_th"] - df["price_t"]) / df["price_t"],
        np.nan,
    )

    # placeholder for realized turnover at prediction level (strategy-level turnover computed separately)
    df["turnover"] = 0.0

    df = df.dropna(subset=["price_t", "price_th", "realized_return"])
    return df
