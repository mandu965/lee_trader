import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from db import get_engine, copy_df
except Exception:
    get_engine = None
    copy_df = None

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lee_trader.db"
CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"
ADJ_CSV = DATA_DIR / "prices_daily_adjusted.csv"
FEATURE_CSV = DATA_DIR / "features.csv"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_prices() -> pd.DataFrame:
    """
    ???? ????: DB fact_price_daily -> adjusted CSV -> clean CSV.
    fact_price_daily? ??? adj_close? close? ??.
    """
    # DB ??? ??, ?? ????? adjusted CSV? ??
    # 1) DB fact_price_daily (Postgres ??, fallback sqlite)
    try:
        df_db = None
        if get_engine:
            eng = get_engine()
            df_db = pd.read_sql(
                "SELECT date, code, open, high, low, close, adj_close, volume FROM fact_price_daily",
                eng,
                dtype={"code": str},
            )
        elif DB_PATH.exists():
            with sqlite3.connect(DB_PATH) as conn:
                df_db = pd.read_sql(
                    "SELECT date, code, open, high, low, close, adj_close, volume FROM fact_price_daily",
                    conn,
                    dtype={"code": str},
                )

        if df_db is not None and not df_db.empty:
            df_db["code"] = df_db["code"].astype(str).str.zfill(6)
            df_db["date"] = pd.to_datetime(df_db["date"])
            if "adj_close" in df_db.columns:
                df_db["close"] = pd.to_numeric(df_db["adj_close"], errors="coerce").fillna(df_db["close"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df_db.columns:
                    df_db[col] = pd.to_numeric(df_db[col], errors="coerce")
            df_db = df_db.dropna(subset=["close"])
            df_db = df_db.sort_values(["code", "date"]).reset_index(drop=True)
            fallback_to_adjusted = False
            if ADJ_CSV.exists():
                try:
                    adj_codes = pd.read_csv(ADJ_CSV, dtype={"code": str})["code"].astype(str).str.zfill(6).unique()
                    db_codes = df_db["code"].unique()
                    missing_codes = set(adj_codes) - set(db_codes)
                    if missing_codes:
                        sample = sorted(list(missing_codes))[:5]
                        logging.warning(
                            "DB fact_price_daily missing %d codes vs adjusted CSV (sample=%s); fallback to adjusted CSV",
                            len(missing_codes),
                            sample,
                        )
                        fallback_to_adjusted = True
                except Exception:
                    logging.exception("code coverage check failed; keeping DB data")
            if not fallback_to_adjusted:
                return df_db[["date", "code", "open", "high", "low", "close", "volume"]]
    except Exception:
        logging.exception("Failed to load fact_price_daily from DB; falling back to CSV")

    # 2) Adjusted CSV
    if ADJ_CSV.exists():
        df = pd.read_csv(ADJ_CSV, dtype={"code": str})
        expected = {"date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"}
        missing = expected.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in adjusted csv: {missing}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        df["open"] = pd.to_numeric(df["adj_open"], errors="coerce")
        df["high"] = pd.to_numeric(df["adj_high"], errors="coerce")
        df["low"] = pd.to_numeric(df["adj_low"], errors="coerce")
        df["close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        return df[["date", "code", "open", "high", "low", "close", "volume"]]

    # 3) Clean CSV fallback
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV.resolve()}")
    df = pd.read_csv(CLEAN_CSV, dtype={"code": str})
    expected = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in clean csv: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0).clip(0, 100)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        g["ret_1d"] = g["close"].pct_change(1)
        g["mom_20"] = g["close"].pct_change(20)
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_10d"] = g["close"].pct_change(10)

        g["ma_5"] = g["close"].rolling(5, min_periods=5).mean()
        g["ma_20"] = g["close"].rolling(20, min_periods=20).mean()
        g["ma_60"] = g["close"].rolling(60, min_periods=25).mean()
        g["close_over_ma20"] = g["close"] / g["ma_20"] - 1

        g["vol_20"] = g["ret_1d"].rolling(20, min_periods=10).std()
        g["vol_60"] = g["ret_1d"].rolling(60, min_periods=20).std()

        g["vol_ma_20"] = g["volume"].rolling(20, min_periods=10).mean()
        g["vol_ratio_20"] = g["volume"] / g["vol_ma_20"]

        g["rsi_14"] = compute_rsi(g["close"], window=14)

        keep = [
            "date",
            "code",
            "close",
            "ret_1d",
            "ret_5d",
            "ret_10d",
            "mom_20",
            "ma_5",
            "ma_20",
            "ma_60",
            "close_over_ma20",
            "vol_20",
            "vol_60",
            "rsi_14",
            "volume",
            "vol_ma_20",
            "vol_ratio_20",
        ]
        return g[keep]

    feat = df.groupby("code", group_keys=False).apply(_per_code)
    feat = feat.sort_values(["code", "date"]).reset_index(drop=True)
    return feat


def merge_quality(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge forward-filled quality_score from data/quality.csv onto daily features by (code, date).
    For each (code, date) take the latest quality.date that is <= feature.date (asof join).
    """
    try:
        qual_path = DATA_DIR / "quality.csv"
        if not qual_path.exists():
            logging.info("quality.csv not found – skipping quality merge")
            return feat_df

        q = pd.read_csv(qual_path, dtype={"code": str})
        required = {"date", "code", "quality_score"}
        if not required.issubset(q.columns):
            logging.warning("quality.csv missing required columns %s – skipping", required)
            return feat_df

        q["date"] = pd.to_datetime(q["date"], errors="coerce")
        q = q.sort_values(["code", "date"]).reset_index(drop=True)

        left = feat_df.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            q[["date", "code", "quality_score"]].sort_values("date"),
            on="date",
            by="code",
            direction="backward",
            allow_exact_matches=True,
        )
        logging.info(
            "Merged quality_score: rows=%d, NaN ratio=%.3f",
            len(merged),
            merged["quality_score"].isna().mean() if "quality_score" in merged.columns else 1.0,
        )
        return merged
    except Exception as e:
        logging.exception("quality merge failed: %s", e)
        return feat_df


def save_features(df_feat: pd.DataFrame):
    out = df_feat.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FEATURE_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved features: {FEATURE_CSV.resolve()} (rows={len(out)})")

    # Save to DB (prefer Postgres via copy_expert, then SQLAlchemy)
    try:
        if copy_df:
            try:
                copy_df("features", out, columns=list(out.columns), truncate=True)
                logging.info("Saved features to Postgres via copy_expert (rows=%d)", len(out))
                return
            except Exception:
                logging.exception("copy_expert failed, trying SQLAlchemy")
        if get_engine:
            eng = get_engine()
            out.to_sql("features", eng, if_exists="replace", index=False, method="multi")
            logging.info("Saved features to Postgres via SQLAlchemy (rows=%d)", len(out))
            return
    except Exception:
        logging.exception("SQLAlchemy save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        out.to_sql("features", conn, if_exists="replace", index=False)
        conn.commit()
        logging.info("Saved features to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(out))
    except Exception:
        logging.exception("Failed to save features to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main():
    setup_logging()
    ensure_data_dir()
    logging.info("Loading prices (DB fact_price_daily preferred)...")
    clean_df = load_prices()
    logging.info("Clean rows: %d", len(clean_df))

    logging.info("Building features...")
    feat_df = build_features(clean_df)
    logging.info("Feature rows: %d", len(feat_df))

    feat_df = merge_quality(feat_df)
    logging.info(
        "After quality merge: rows=%d, quality_score NaN ratio=%.3f",
        len(feat_df),
        feat_df.get("quality_score").isna().mean() if "quality_score" in feat_df.columns else 1.0,
    )

    save_features(feat_df)


if __name__ == "__main__":
    main()
