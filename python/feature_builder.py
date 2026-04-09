import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
try:
    from db import (
        copy_df,
        ensure_unique_keys,
        get_engine,
        replace_table_rows_pg,
        replace_table_rows_sqlite,
    )
except Exception:
    get_engine = None
    copy_df = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lee_trader.db"
CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"
ADJ_CSV = DATA_DIR / "prices_daily_adjusted.csv"
FEATURE_CSV = DATA_DIR / "features.csv"
FEATURES_DB_COLUMNS = [
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
    "volume_ratio_5d",
    "volume_ratio_20d",
    "value_ratio_5d",
    "value_ratio_20d",
    "volume_score",
    "liquidity_score",
    "vol_ma_20",
    "vol_ratio_20",
    "quality_score",
    "quality_factor_count",
    "quality_missing_ratio",
    "quality_score_confidence",
]
FEATURES_PK = ["date", "code"]


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
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    warmup_mask = avg_gain.isna() | avg_loss.isna()
    both_flat = (~warmup_mask) & avg_gain.eq(0) & avg_loss.eq(0)
    only_gain = (~warmup_mask) & avg_loss.eq(0) & avg_gain.gt(0)
    only_loss = (~warmup_mask) & avg_gain.eq(0) & avg_loss.gt(0)

    rsi = rsi.mask(both_flat, 50.0)
    rsi = rsi.mask(only_gain, 100.0)
    rsi = rsi.mask(only_loss, 0.0)
    rsi = rsi.mask(warmup_mask)
    return rsi.clip(0, 100)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return num / den


def _clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.clip(lower=lower, upper=upper)


def _winsorize_by_date(df: pd.DataFrame, col: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _clip(s: pd.Series) -> pd.Series:
        vals = pd.to_numeric(s, errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            return vals
        lo = valid.quantile(lower_q)
        hi = valid.quantile(upper_q)
        return vals.clip(lower=lo, upper=hi)

    return df.groupby("date", group_keys=False)[col].transform(_clip)


def _percentile_score_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _rank(s: pd.Series) -> pd.Series:
        vals = pd.to_numeric(s, errors="coerce")
        return vals.rank(pct=True, ascending=True, method="average") * 100.0

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        g["ret_1d"] = g["close"].pct_change(1)
        g["mom_20"] = g["close"].pct_change(20)
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_10d"] = g["close"].pct_change(10)

        g["ma_5"] = g["close"].rolling(5, min_periods=5).mean()
        g["ma_20"] = g["close"].rolling(20, min_periods=20).mean()
        g["ma_60"] = g["close"].rolling(60, min_periods=60).mean()
        g["close_over_ma20"] = g["close"] / g["ma_20"] - 1

        g["vol_20"] = g["ret_1d"].rolling(20, min_periods=10).std()
        g["vol_60"] = g["ret_1d"].rolling(60, min_periods=20).std()

        g["vol_ma_20"] = g["volume"].rolling(20, min_periods=10).mean()
        g["vol_ratio_20"] = g["volume"] / g["vol_ma_20"]
        g["vol_ma_5"] = g["volume"].rolling(5, min_periods=3).mean()

        g["value_traded"] = pd.to_numeric(g["close"], errors="coerce") * pd.to_numeric(g["volume"], errors="coerce")
        g["value_ma_5"] = g["value_traded"].rolling(5, min_periods=3).mean()
        g["value_ma_20"] = g["value_traded"].rolling(20, min_periods=10).mean()

        g["volume_ratio_5d"] = _safe_ratio(g["volume"], g["vol_ma_5"])
        g["volume_ratio_20d"] = _safe_ratio(g["volume"], g["vol_ma_20"])
        g["value_ratio_5d"] = _safe_ratio(g["value_traded"], g["value_ma_5"])
        g["value_ratio_20d"] = _safe_ratio(g["value_traded"], g["value_ma_20"])

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
            "volume_ratio_5d",
            "volume_ratio_20d",
            "value_ratio_5d",
            "value_ratio_20d",
            "vol_ma_20",
            "vol_ratio_20",
        ]
        return g[keep]

    feat = df.groupby("code", group_keys=False).apply(_per_code)
    feat = feat.sort_values(["code", "date"]).reset_index(drop=True)

    for col in ["volume_ratio_5d", "volume_ratio_20d", "value_ratio_5d", "value_ratio_20d"]:
        feat[col] = _clip_series(feat[col], 0.0, 5.0)
        feat[col] = _winsorize_by_date(feat, col, lower_q=0.01, upper_q=0.99)
        feat[col] = _clip_series(feat[col], 0.0, 5.0)

    feat["volume_level_win"] = _winsorize_by_date(feat, "volume", lower_q=0.01, upper_q=0.99)
    feat["vol_ma_20_win"] = _winsorize_by_date(feat, "vol_ma_20", lower_q=0.01, upper_q=0.99)
    feat["volume_ratio_5d_win"] = _winsorize_by_date(feat, "volume_ratio_5d", lower_q=0.01, upper_q=0.99)
    feat["volume_ratio_20d_win"] = _winsorize_by_date(feat, "volume_ratio_20d", lower_q=0.01, upper_q=0.99)
    feat["value_ratio_5d_win"] = _winsorize_by_date(feat, "value_ratio_5d", lower_q=0.01, upper_q=0.99)
    feat["value_ratio_20d_win"] = _winsorize_by_date(feat, "value_ratio_20d", lower_q=0.01, upper_q=0.99)

    feat["volume_score"] = (
        0.40 * _percentile_score_by_date(feat, "volume_ratio_5d_win").fillna(50.0)
        + 0.40 * _percentile_score_by_date(feat, "volume_ratio_20d_win").fillna(50.0)
        + 0.20 * _percentile_score_by_date(feat, "volume_level_win").fillna(50.0)
    ).clip(lower=0.0, upper=100.0)

    feat["liquidity_score"] = (
        0.35 * _percentile_score_by_date(feat, "vol_ma_20_win").fillna(50.0)
        + 0.25 * _percentile_score_by_date(feat, "volume_level_win").fillna(50.0)
        + 0.20 * _percentile_score_by_date(feat, "value_ratio_20d_win").fillna(50.0)
        + 0.20 * _percentile_score_by_date(feat, "value_ratio_5d_win").fillna(50.0)
    ).clip(lower=0.0, upper=100.0)

    drop_cols = [
        "volume_level_win",
        "vol_ma_20_win",
        "volume_ratio_5d_win",
        "volume_ratio_20d_win",
        "value_ratio_5d_win",
        "value_ratio_20d_win",
    ]
    feat = feat.drop(columns=[col for col in drop_cols if col in feat.columns])
    return feat


def merge_quality(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge forward-filled quality fields from data/quality.csv onto daily features by (code, date).
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
        quality_cols = [
            col
            for col in ["quality_score", "quality_factor_count", "quality_missing_ratio", "quality_score_confidence"]
            if col in q.columns
        ]

        left = feat_df.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            q[["date", "code", *quality_cols]].sort_values("date"),
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
    if "quality_factor_count" in out.columns:
        out["quality_factor_count"] = pd.to_numeric(out["quality_factor_count"], errors="coerce").round().astype("Int64")
    for col in FEATURES_DB_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[FEATURES_DB_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(out, FEATURES_PK, "features")
    out.to_csv(FEATURE_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved features: {FEATURE_CSV.resolve()} (rows={len(out)})")

    # Save to DB while preserving schema and indexes.
    try:
        if replace_table_rows_pg:
            try:
                if get_engine:
                    eng = get_engine()
                    with eng.begin() as conn_pg:
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS quality_factor_count INTEGER"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS quality_missing_ratio DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS quality_score_confidence DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_ratio_5d DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_ratio_20d DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS value_ratio_5d DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS value_ratio_20d DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_score DOUBLE PRECISION"))
                        conn_pg.execute(text("ALTER TABLE features ADD COLUMN IF NOT EXISTS liquidity_score DOUBLE PRECISION"))
                replace_table_rows_pg("features", out, columns=FEATURES_DB_COLUMNS)
                logging.info("Replaced features rows in Postgres (rows=%d)", len(out))
                return
            except Exception:
                logging.exception("Postgres row replace failed, fallback to sqlite")
    except Exception:
        logging.exception("Postgres save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                date             DATE NOT NULL,
                code             TEXT NOT NULL,
                close            REAL,
                ret_1d           REAL,
                ret_5d           REAL,
                ret_10d          REAL,
                mom_20           REAL,
                ma_5             REAL,
                ma_20            REAL,
                ma_60            REAL,
                close_over_ma20  REAL,
                vol_20           REAL,
                vol_60           REAL,
                rsi_14           REAL,
                volume           REAL,
                volume_ratio_5d  REAL,
                volume_ratio_20d REAL,
                value_ratio_5d   REAL,
                value_ratio_20d  REAL,
                volume_score     REAL,
                liquidity_score  REAL,
                vol_ma_20        REAL,
                vol_ratio_20     REAL,
                quality_score    REAL,
                quality_factor_count INTEGER,
                quality_missing_ratio REAL,
                quality_score_confidence REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()}
        for col, col_type in [
            ("quality_factor_count", "INTEGER"),
            ("quality_missing_ratio", "REAL"),
            ("quality_score_confidence", "REAL"),
            ("volume_ratio_5d", "REAL"),
            ("volume_ratio_20d", "REAL"),
            ("value_ratio_5d", "REAL"),
            ("value_ratio_20d", "REAL"),
            ("volume_score", "REAL"),
            ("liquidity_score", "REAL"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type}")
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "features", out)
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
