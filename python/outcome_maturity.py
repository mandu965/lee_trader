"""
Common maturity rules for walk-forward backtest outcomes.

Why this module exists
----------------------
`label_builder.py` creates forward labels by shifting *actual trading rows*:

    target_{h}d = close[t+h] / close[t] - 1

This means outcome maturity cannot be judged by plain calendar days alone.
For a prediction made on `as_of_date`, the outcome is mature only when the
same stock has at least `horizon_days` future trading rows in the price data.

Common rule
-----------
Row-level maturity:
  - Input: prediction row (`code`, `as_of_date`), `horizon_days`,
    and actual price availability by (`code`, `date`).
  - A row is mature when:
      future_trading_days_available >= horizon_days
  - `future_trading_days_available` is computed from the stock's real price
    history, not from calendar arithmetic.
  - `last_available_price_date` is tracked per stock and attached to each row.

Run-level maturity:
  - A run is `OUTCOME_NOT_MATURE` if zero prediction rows are mature.
  - A run is `OUTCOME_PARTIAL` if only some prediction rows are mature or
    if matured rows exist but fewer outcomes were written than expected.
  - A run is `OUTCOME_READY` if all prediction rows are mature and the number
    of saved outcomes covers all mature prediction rows.
  - A run is `OUTCOME_EMPTY` if the run has no prediction rows at all.

Shared run_status rule:
  - `unmatured`: maturity_ratio == 0.0
  - `partial`: 0.0 < maturity_ratio < 0.8
  - `matured`: maturity_ratio >= 0.8
  - Only `matured` runs should be used for official performance comparison.
  - `partial` runs are reference-only.
  - `unmatured` runs must be excluded from performance comparison with reason.
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from db import get_engine
except Exception:
    get_engine = None


PRICE_CSV_FALLBACK = Path("data/prices_daily_adjusted.csv")


def load_price_availability(
    *,
    prices_csv: Path = PRICE_CSV_FALLBACK,
) -> pd.DataFrame:
    """
    Load unique (`code`, `date`) pairs that represent actual tradeable price rows.

    Preference order:
    1. Postgres table `fact_price_daily`
    2. CSV fallback `data/prices_daily_adjusted.csv`
    """
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(
                    "SELECT date, code FROM fact_price_daily",
                    conn,
                    parse_dates=["date"],
                    dtype={"code": str},
                )
            if not df.empty:
                df["code"] = df["code"].astype(str).str.zfill(6)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                return df.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
        except Exception:
            pass

    if not prices_csv.exists():
        raise FileNotFoundError(f"Price availability source not found: {prices_csv}")

    df = pd.read_csv(prices_csv, usecols=["date", "code"], dtype={"code": str}, parse_dates=["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)


def load_price_history(
    *,
    prices_csv: Path = PRICE_CSV_FALLBACK,
) -> pd.DataFrame:
    """
    Load price rows used for maturity and realized outcome calculation.

    Returned columns:
    - code
    - date
    - close
    """
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(
                    "SELECT date, code, COALESCE(adj_close, close) AS close FROM fact_price_daily",
                    conn,
                    parse_dates=["date"],
                    dtype={"code": str},
                )
            if not df.empty:
                df["code"] = df["code"].astype(str).str.zfill(6)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna(subset=["close"])
                return df.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
        except Exception:
            pass

    if not prices_csv.exists():
        raise FileNotFoundError(f"Price history source not found: {prices_csv}")

    df = pd.read_csv(prices_csv, dtype={"code": str})
    if "adj_close" in df.columns:
        close_col = "adj_close"
    elif "close" in df.columns:
        close_col = "close"
    else:
        raise ValueError(f"{prices_csv} must contain 'adj_close' or 'close'")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df[["code", "date", "close"]].drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)


def build_price_reference(price_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-stock trading calendar reference.

    Columns returned:
    - code
    - date
    - trade_index: zero-based index of each available trading row per stock
    - max_trade_index: last trade index per stock
    - last_available_price_date: latest price date per stock
    """
    if price_dates.empty:
        return pd.DataFrame(
            columns=["code", "date", "trade_index", "max_trade_index", "last_available_price_date"]
        )

    ref = price_dates.copy()
    ref["date"] = pd.to_datetime(ref["date"]).dt.normalize()
    ref["code"] = ref["code"].astype(str).str.zfill(6)
    ref = ref.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    ref["trade_index"] = ref.groupby("code").cumcount()
    ref["max_trade_index"] = ref.groupby("code")["trade_index"].transform("max")
    ref["last_available_price_date"] = ref.groupby("code")["date"].transform("max")
    return ref


def _compute_forward_mdd(close_values: np.ndarray, horizon_days: int) -> np.ndarray:
    out = np.full(len(close_values), np.nan, dtype=float)
    for i in range(len(close_values)):
        j = i + horizon_days
        if j >= len(close_values):
            continue
        window = close_values[i : j + 1]
        run_max = np.maximum.accumulate(window)
        drawdown = window / run_max - 1.0
        out[i] = float(drawdown.min())
    return out


def attach_forward_outcomes(
    price_history: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Compute realized forward return and MDD directly from actual price history.

    The logic matches the label-building convention:
    - realized_return = close[t+h] / close[t] - 1
    - realized_mdd = min drawdown over the inclusive window [t, t+h]
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if price_history.empty:
        return pd.DataFrame(columns=["code", "date", "realized_return", "realized_mdd"])

    frames = []
    for code, group in price_history.sort_values(["code", "date"]).groupby("code", sort=False):
        g = group.copy()
        close_vals = g["close"].to_numpy(dtype=float)
        future_close = g["close"].shift(-horizon_days).to_numpy(dtype=float)
        g["realized_return"] = future_close / close_vals - 1.0
        g["realized_mdd"] = _compute_forward_mdd(close_vals, horizon_days)
        frames.append(g[["code", "date", "realized_return", "realized_mdd"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "date", "realized_return", "realized_mdd"]
    )


def evaluate_prediction_maturity_rows(
    predictions: pd.DataFrame,
    *,
    price_reference: pd.DataFrame,
    horizon_days: int,
    as_of_col: str = "as_of_date",
    code_col: str = "code",
) -> pd.DataFrame:
    """
    Evaluate maturity for each prediction row using real trading-date coverage.

    Interpretation:
    - `available_future_trading_days` counts how many future price rows remain
      for the same stock after `as_of_date`.
    - `is_matured` becomes True when that count is at least `horizon_days`.
    - If the prediction date itself is not found in the stock's price series,
      the row is marked `PRICE_DATE_MISSING`.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")

    out = predictions.copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                code_col,
                as_of_col,
                "last_available_price_date",
                "available_future_trading_days",
                "required_future_trading_days",
                "is_matured",
                "maturity_status",
            ]
        )

    out[code_col] = out[code_col].astype(str).str.zfill(6)
    out[as_of_col] = pd.to_datetime(out[as_of_col]).dt.normalize()

    merged = out.merge(
        price_reference,
        left_on=[code_col, as_of_col],
        right_on=["code", "date"],
        how="left",
    )
    merged["required_future_trading_days"] = int(horizon_days)
    merged["available_future_trading_days"] = merged["max_trade_index"] - merged["trade_index"]
    merged["available_future_trading_days"] = merged["available_future_trading_days"].fillna(-1).astype(int)

    missing_price_mask = merged["trade_index"].isna()
    merged["is_matured"] = (~missing_price_mask) & (
        merged["available_future_trading_days"] >= merged["required_future_trading_days"]
    )
    merged["maturity_status"] = "UNMATURED"
    merged.loc[merged["is_matured"], "maturity_status"] = "MATURED"
    merged.loc[missing_price_mask, "maturity_status"] = "PRICE_DATE_MISSING"

    return merged[
        list(predictions.columns)
        + [
            "last_available_price_date",
            "available_future_trading_days",
            "required_future_trading_days",
            "is_matured",
            "maturity_status",
        ]
    ]


def summarize_run_maturity(
    maturity_rows: pd.DataFrame,
    *,
    run_id_col: str = "run_id",
    as_of_col: str = "as_of_date",
    outcome_rows_by_run: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Summarize row-level maturity into run-level maturity.

    Output columns:
    - run_id
    - prediction_rows
    - matured_prediction_rows
    - unmatured_prediction_rows
    - matured_prediction_dates
    - unmatured_prediction_dates
    - expected_outcome_rows
    - actual_outcome_rows
    - run_outcome_status
    """
    if maturity_rows.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "prediction_rows",
                "matured_prediction_rows",
                "unmatured_prediction_rows",
                "matured_prediction_dates",
                "unmatured_prediction_dates",
                "expected_outcome_rows",
                "actual_outcome_rows",
                "run_outcome_status",
            ]
        )

    df = maturity_rows.copy()
    df[run_id_col] = pd.to_numeric(df[run_id_col], errors="raise").astype(int)
    df[as_of_col] = pd.to_datetime(df[as_of_col]).dt.normalize()

    rows = []
    for run_id, group in df.groupby(run_id_col):
        prediction_rows = int(len(group))
        matured_mask = group["is_matured"].fillna(False)
        matured_prediction_rows = int(matured_mask.sum())
        unmatured_prediction_rows = int((~matured_mask).sum())
        matured_prediction_dates = int(group.loc[matured_mask, as_of_col].nunique())
        unmatured_prediction_dates = int(group.loc[~matured_mask, as_of_col].nunique())
        expected_outcome_rows = matured_prediction_rows
        actual_outcome_rows = int((outcome_rows_by_run or {}).get(int(run_id), 0))

        if prediction_rows == 0:
            run_outcome_status = "OUTCOME_EMPTY"
        elif matured_prediction_rows == 0:
            run_outcome_status = "OUTCOME_NOT_MATURE"
        elif actual_outcome_rows < expected_outcome_rows:
            run_outcome_status = "OUTCOME_PARTIAL"
        elif matured_prediction_rows < prediction_rows:
            run_outcome_status = "OUTCOME_PARTIAL"
        else:
            run_outcome_status = "OUTCOME_READY"

        rows.append(
            {
                "run_id": int(run_id),
                "prediction_rows": prediction_rows,
                "matured_prediction_rows": matured_prediction_rows,
                "unmatured_prediction_rows": unmatured_prediction_rows,
                "matured_prediction_dates": matured_prediction_dates,
                "unmatured_prediction_dates": unmatured_prediction_dates,
                "expected_outcome_rows": expected_outcome_rows,
                "actual_outcome_rows": actual_outcome_rows,
                "run_outcome_status": run_outcome_status,
            }
        )

    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def classify_run_status(maturity_ratio: float) -> tuple[str, str]:
    """
    Shared run-status classifier for coverage and reporting.

    Returns:
    - run_status: one of `unmatured`, `partial`, `matured`
    - run_status_reason: short explanation for logging/reporting
    """
    if pd.isna(maturity_ratio) or maturity_ratio <= 0.0:
        return "unmatured", "no outcome-evaluable rows yet"
    if maturity_ratio < 0.8:
        return "partial", "outcome coverage is incomplete; reference-only"
    return "matured", "outcome coverage is sufficient for official comparison"
