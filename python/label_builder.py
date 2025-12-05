import logging
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
ADJ_CSV = DATA_DIR / "prices_daily_adjusted.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

HORIZONS: List[int] = [30, 60, 90]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_adjusted_prices() -> pd.DataFrame:
    if not ADJ_CSV.exists():
        raise FileNotFoundError(
            f"Adjusted price CSV not found: {ADJ_CSV.resolve()}\n"
            "Generate prices_daily_adjusted.csv before running label_builder."
        )
    logging.info("Loading adjusted prices: %s", ADJ_CSV.resolve())
    df = pd.read_csv(ADJ_CSV, dtype={"code": str})
    expected = {"date", "code", "adj_close"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in adjusted csv: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["close"])
    logging.info("Adjusted price rows after cleaning: %d", len(df))
    return df[["date", "code", "close"]]


def _compute_forward_mdd(close_values: np.ndarray, horizon: int) -> np.ndarray:
    """
    Compute forward max drawdown over [i, i+horizon].
      window = close[i : i+horizon]
      run_max = cumulative max over window
      dd = window / run_max - 1
      mdd = dd.min()
    """
    n = len(close_values)
    out = np.full(n, np.nan, dtype=float)
    if horizon <= 0:
        return out

    for i in range(n):
        j = i + horizon
        if j >= n:
            continue
        window = close_values[i : j + 1]
        run_max = np.maximum.accumulate(window)
        dd = window / run_max - 1.0
        out[i] = dd.min()
    return out


def build_labels_multi(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Build forward returns (simple/log), MDD, realized return, and top20 flags for each horizon.
      target_{h}d       = (close[t+h] / close[t]) - 1
      target_log_{h}d   = log(close[t+h] / close[t])
      target_mdd_{h}d   = min(drawdown over window)
      realized_return_{h}d = target_{h}d
      target_{h}d_top20 = 1 if target_{h}d >= 80th percentile for that date else 0
    """
    logging.info("Building labels for horizons: %s", horizons)
    df = df.sort_values(["code", "date"]).copy()
    out = df[["date", "code", "close"]].copy()

    for h in horizons:
        out[f"target_{h}d"] = np.nan
        out[f"target_log_{h}d"] = np.nan
        out[f"target_mdd_{h}d"] = np.nan
        out[f"realized_return_{h}d"] = np.nan

    for code, g in df.groupby("code"):
        g = g.sort_values("date").copy()
        idx = g.index.to_numpy()
        close_vals = g["close"].to_numpy()

        for h in horizons:
            future_close = g["close"].shift(-h).to_numpy()
            simple_ret = (future_close / close_vals) - 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret = np.log(future_close / close_vals)
            mdd_arr = _compute_forward_mdd(close_vals, horizon=h)

            out.loc[idx, f"target_{h}d"] = simple_ret
            out.loc[idx, f"target_log_{h}d"] = log_ret
            out.loc[idx, f"target_mdd_{h}d"] = mdd_arr
            out.loc[idx, f"realized_return_{h}d"] = simple_ret

    out = out.drop(columns=["close"])
    cond = np.zeros(len(out), dtype=bool)
    for h in horizons:
        col = f"target_{h}d"
        if col in out.columns:
            cond |= out[col].notna().values
    out = out[cond].copy()
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    logging.info("Label rows after horizon filtering: %d", len(out))
    return out


def add_top20_flags(labels: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = labels.copy()
    for h in horizons:
        base = f"target_{h}d"
        if base not in out.columns:
            continue
        top_col = f"{base}_top20"
        out[top_col] = (
            out.groupby("date")[base]
            .transform(lambda s: (s >= s.quantile(0.8)).astype(int))
            .fillna(0)
            .astype(int)
        )
    return out


def summarize_labels(labels: pd.DataFrame, horizons: List[int]) -> None:
    for h in horizons:
        for base in [f"target_{h}d", f"target_log_{h}d", f"target_mdd_{h}d"]:
            if base not in labels.columns:
                continue
            series = labels[base].dropna()
            if series.empty:
                logging.warning("%s: no data", base)
                continue
            desc = series.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            logging.info(
                "%s summary:\n"
                "  count=%d\n"
                "  mean=%.4f, std=%.4f\n"
                "  min=%.4f, 1%%=%.4f, median=%.4f, 95%%=%.4f, 99%%=%.4f, max=%.4f",
                base,
                int(desc["count"]),
                desc["mean"],
                desc["std"],
                desc["min"],
                desc["1%"],
                desc["50%"],
                desc["95%"],
                desc["99%"],
                desc["max"],
            )


def save_labels_csv(labels: pd.DataFrame) -> None:
    out = labels.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(LABELS_CSV, index=False, encoding="utf-8")
    logging.info("Saved labels: %s (rows=%d)", LABELS_CSV.resolve(), len(out))


def save_labels_db(labels: pd.DataFrame) -> None:
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        required_cols = [
            "target_30d",
            "target_60d",
            "target_90d",
            "target_log_30d",
            "target_log_60d",
            "target_log_90d",
            "target_mdd_30d",
            "target_mdd_60d",
            "target_mdd_90d",
            "target_30d_top20",
            "target_60d_top20",
            "target_90d_top20",
            "realized_return_30d",
            "realized_return_60d",
            "realized_return_90d",
        ]
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                date                DATE NOT NULL,
                code                TEXT NOT NULL,
                target_30d          REAL,
                target_60d          REAL,
                target_90d          REAL,
                target_log_30d      REAL,
                target_log_60d      REAL,
                target_log_90d      REAL,
                target_mdd_30d      REAL,
                target_mdd_60d      REAL,
                target_mdd_90d      REAL,
                target_30d_top20    REAL,
                target_60d_top20    REAL,
                target_90d_top20    REAL,
                realized_return_30d REAL,
                realized_return_60d REAL,
                realized_return_90d REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        # Ensure columns exist (legacy table may lack new horizons)
        cur = conn.execute("PRAGMA table_info(labels);")
        existing = {row[1] for row in cur.fetchall()}
        for col in required_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE labels ADD COLUMN {col} REAL;")
        labels_copy = labels.copy()
        labels_copy["date"] = pd.to_datetime(labels_copy["date"]).dt.strftime("%Y-%m-%d")
        records = labels_copy.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO labels
            (date, code,
             target_30d, target_60d, target_90d,
             target_log_30d, target_log_60d, target_log_90d,
             target_mdd_30d, target_mdd_60d, target_mdd_90d,
             target_30d_top20, target_60d_top20, target_90d_top20,
             realized_return_30d, realized_return_60d, realized_return_90d)
            VALUES (:date, :code,
                    :target_30d, :target_60d, :target_90d,
                    :target_log_30d, :target_log_60d, :target_log_90d,
                    :target_mdd_30d, :target_mdd_60d, :target_mdd_90d,
                    :target_30d_top20, :target_60d_top20, :target_90d_top20,
                    :realized_return_30d, :realized_return_60d, :realized_return_90d)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved labels to DB: %s (rows=%d)", DB_PATH.resolve(), len(labels))
    except Exception:
        logging.exception("Failed to save labels to DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    ensure_data_dir()
    prices_df = load_adjusted_prices()
    labels_df = build_labels_multi(prices_df, horizons=HORIZONS)
    labels_df = add_top20_flags(labels_df, horizons=HORIZONS)
    summarize_labels(labels_df, horizons=HORIZONS)
    for h in HORIZONS:
        top_col = f"target_{h}d_top20"
        if top_col in labels_df.columns:
            logging.info("%s positive rate=%.3f", top_col, labels_df[top_col].mean())
    save_labels_csv(labels_df)
    save_labels_db(labels_df)


if __name__ == "__main__":
    main()
