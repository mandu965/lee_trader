"""
CSV -> SQLite migration helper.
- Creates data/lee_trader.db
- Creates tables: stocks, daily_ranking, daily_scores, labels, backtest_trades
- Loads data from existing CSV outputs.

Run once after CSVs are generated:
    python migrate_to_sqlite.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "lee_trader.db"

UNIVERSE_CSV = DATA_DIR / "universe.csv"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
SCORES_CSV = DATA_DIR / "scores_final.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
BACKTEST_CSV = DATA_DIR / "analysis" / "score_backtest_from_labels.csv"
MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
PRICES_RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
PRICES_CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"
PRICES_ADJ_CSV = DATA_DIR / "prices_daily_adjusted.csv"
FUND_CSV = DATA_DIR / "fundamentals.csv"
QUALITY_CSV = DATA_DIR / "quality.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
TRADES_CSV = DATA_DIR / "trades.csv"
# NOTE: trades are managed in DB only; CSV import is disabled by default.


# ------------------------- DB helpers ------------------------- #
def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def truncate_table(conn: sqlite3.Connection, table: str) -> None:
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute(f"DELETE FROM {table};")
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass


def reset_tables(conn: sqlite3.Connection) -> None:
    """Drop known tables so schema changes apply cleanly."""
    tables = [
        "daily_ranking",
        "daily_scores",
        "labels",
        "backtest_trades",
        "market_status",
        "prices_adjusted",
        "prices_clean",
        "prices_raw",
        "fact_price_daily",
        "fundamentals",
        "quality",
        "features",
        "predictions",
        # "trades",  # keep trades; managed separately
        "stocks",
    ]
    conn.execute("PRAGMA foreign_keys = OFF;")
    for tbl in tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {tbl};")
        except Exception:
            continue
    conn.execute("PRAGMA foreign_keys = ON;")


def create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            code        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            market      TEXT,
            sector      TEXT,
            listed_at   DATE,
            delisted_at DATE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_ranking (
            date                 DATE NOT NULL,
            code                 TEXT NOT NULL,
            close                REAL,
            pred_return_60d      REAL,
            pred_return_90d      REAL,
            pred_mdd_60d         REAL,
            pred_mdd_90d         REAL,
            prob_top20_60d       REAL,
            prob_top20_90d       REAL,
            score                REAL,
            score_score          REAL,
            composite            REAL,
            quality_score        REAL,
            name                 TEXT,
            market               TEXT,
            sector               TEXT,
            tech_score           REAL,
            pred_score           REAL,
            ret_score            REAL,
            prob_score           REAL,
            qual_score           REAL,
            safety_score         REAL,
            liquidity_score      REAL,
            final_score          REAL,
            risk_penalty         REAL,
            market_up            INTEGER,
            market_status_date   DATE,
            market_kospi_close   REAL,
            market_kospi_ma20    REAL,
            market_vol_5d        REAL,
            market_foreign_5d    REAL,
            generated_at         TEXT,
            model_version        TEXT,
            PRIMARY KEY (date, code),
            FOREIGN KEY (code) REFERENCES stocks(code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_scores (
            date       DATE NOT NULL,
            code       TEXT NOT NULL,
            score      REAL,
            composite  REAL,
            PRIMARY KEY (date, code),
            FOREIGN KEY (code) REFERENCES stocks(code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            date               DATE NOT NULL,
            code               TEXT NOT NULL,
            target_60d         REAL,
            target_90d         REAL,
            target_log_60d     REAL,
            target_log_90d     REAL,
            target_mdd_60d     REAL,
            target_mdd_90d     REAL,
            target_60d_top20   REAL,
            target_90d_top20   REAL,
            realized_price_60d REAL,
            realized_price_90d REAL,
            PRIMARY KEY (date, code),
            FOREIGN KEY (code) REFERENCES stocks(code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date       DATE NOT NULL,
            strategy         TEXT NOT NULL,
            code             TEXT NOT NULL,
            final_score      REAL,
            pred_return_60d  REAL,
            realized_return  REAL NOT NULL,
            created_at       TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (code) REFERENCES stocks(code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_status (
            date             DATE PRIMARY KEY,
            kospi_close      REAL,
            kospi_ma20       REAL,
            volatility_5d    REAL,
            foreign_net_5d   REAL,
            market_up        INTEGER
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_raw (
            date    DATE NOT NULL,
            code    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_clean (
            date    DATE NOT NULL,
            code    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_adjusted (
            date      DATE NOT NULL,
            code      TEXT NOT NULL,
            adj_open  REAL,
            adj_high  REAL,
            adj_low   REAL,
            adj_close REAL,
            volume    REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_price_daily (
            date          DATE NOT NULL,
            code          TEXT NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            adj_close     REAL,
            volume        REAL,
            value         REAL,
            market_cap    REAL,
            listed_shares REAL,
            PRIMARY KEY (date, code),
            FOREIGN KEY (code) REFERENCES stocks(code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals (
            date           DATE NOT NULL,
            code           TEXT NOT NULL,
            roe            REAL,
            op_margin      REAL,
            debt_ratio     REAL,
            ocf_to_assets  REAL,
            net_margin     REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quality (
            date          DATE NOT NULL,
            code          TEXT NOT NULL,
            quality_score REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            date            DATE NOT NULL,
            code            TEXT NOT NULL,
            close           REAL,
            ret_1d          REAL,
            ret_5d          REAL,
            ret_10d         REAL,
            mom_20          REAL,
            ma_5            REAL,
            ma_20           REAL,
            ma_60           REAL,
            close_over_ma20 REAL,
            vol_20          REAL,
            vol_60          REAL,
            rsi_14          REAL,
            volume          REAL,
            vol_ma_20       REAL,
            vol_ratio_20    REAL,
            quality_score   REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            date             DATE NOT NULL,
            code             TEXT NOT NULL,
            pred_return_60d  REAL,
            pred_return_90d  REAL,
            pred_mdd_60d     REAL,
            pred_mdd_90d     REAL,
            prob_top20_60d   REAL,
            prob_top20_90d   REAL,
            score            REAL,
            PRIMARY KEY (date, code)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            date        DATE NOT NULL,
            side        TEXT NOT NULL,
            code        TEXT NOT NULL,
            name        TEXT,
            market      TEXT,
            sector      TEXT,
            qty         REAL,
            price       REAL,
            amount      REAL,
            fee         REAL,
            memo        TEXT,
            created_at  TEXT
        );
        """
    )

    conn.commit()


# ------------------------- loaders ------------------------- #
def normalize_code(series: pd.Series) -> pd.Series:
    """Normalize ticker codes to 6-digit strings."""
    return series.astype(str).str.zfill(6)


def _safe_read_csv(path: Path, dtype: Optional[dict] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] failed to read {path}: {exc}")
        return None


def load_stock_master() -> pd.DataFrame:
    """
    Build a stock master from any CSVs available (universe, ranking, scores, labels, backtest).
    Ensures every code referenced anywhere exists in stocks to satisfy FK constraints.
    """
    frames = []

    # universe.csv (preferred for name/market/sector)
    uni = _safe_read_csv(UNIVERSE_CSV, dtype={"code": str})
    if uni is not None:
        code_col = next((c for c in uni.columns if c.lower() == "code"), None)
        name_col = next((c for c in uni.columns if c.lower() in ("name", "stock_name", "종목명")), None)
        market_col = next((c for c in uni.columns if c.lower() in ("market", "시장구분")), None)
        sector_col = next((c for c in uni.columns if c.lower() in ("sector", "업종")), None)
        if code_col and name_col:
            f = pd.DataFrame()
            f["code"] = normalize_code(uni[code_col])
            f["name"] = uni[name_col].astype(str)
            f["market"] = uni[market_col].astype(str) if market_col else ""
            f["sector"] = uni[sector_col].astype(str) if sector_col else ""
            frames.append(f)

    # ranking_final.csv (also has names/market/sector)
    rank = _safe_read_csv(RANKING_CSV, dtype={"code": str})
    if rank is not None and "code" in rank.columns:
        f = pd.DataFrame()
        f["code"] = normalize_code(rank["code"])
        f["name"] = rank["name"].astype(str) if "name" in rank.columns else ""
        f["market"] = rank["market"].astype(str) if "market" in rank.columns else ""
        f["sector"] = rank["sector"].astype(str) if "sector" in rank.columns else ""
        frames.append(f)

    # scores_final.csv (code only)
    scores = _safe_read_csv(SCORES_CSV, dtype={"code": str})
    if scores is not None and "code" in scores.columns:
        f = pd.DataFrame()
        f["code"] = normalize_code(scores["code"])
        f["name"] = ""
        f["market"] = ""
        f["sector"] = ""
        frames.append(f)

    # labels.csv (code only)
    labels = _safe_read_csv(LABELS_CSV, dtype={"code": str})
    if labels is not None and "code" in labels.columns:
        f = pd.DataFrame()
        f["code"] = normalize_code(labels["code"])
        f["name"] = ""
        f["market"] = ""
        f["sector"] = ""
        frames.append(f)

    # backtest csv (code only)
    back = _safe_read_csv(BACKTEST_CSV, dtype={"code": str})
    if back is not None and "code" in back.columns:
        f = pd.DataFrame()
        f["code"] = normalize_code(back["code"])
        f["name"] = ""
        f["market"] = ""
        f["sector"] = ""
        frames.append(f)

    if not frames:
        raise FileNotFoundError("No CSVs found to build stock master.")

    master = pd.concat(frames, ignore_index=True)
    # prefer non-empty name/market/sector from earlier frames (universe, ranking)
    def _first_non_empty(series: pd.Series) -> str:
        for v in series:
            if isinstance(v, str) and v.strip():
                return v
        return ""

    master = (
        master.groupby("code")
        .apply(
            lambda g: pd.Series(
                {
                    "name": _first_non_empty(g["name"]),
                    "market": _first_non_empty(g["market"]),
                    "sector": _first_non_empty(g["sector"]),
                }
            )
        )
        .reset_index()
    )
    master["listed_at"] = pd.NaT
    master["delisted_at"] = pd.NaT
    return master


# ------------------------- migration steps ------------------------- #
def migrate_stocks(conn: sqlite3.Connection) -> None:
    print("[INFO] migrating stocks ...")
    stocks = load_stock_master()
    truncate_table(conn, "stocks")
    stocks.to_sql("stocks", conn, if_exists="append", index=False)
    print(f"[INFO] stocks rows inserted: {len(stocks)}")


def migrate_daily_ranking(conn: sqlite3.Connection) -> None:
    if not RANKING_CSV.exists():
        print(f"[WARN] {RANKING_CSV} not found. skip daily_ranking.")
        return

    print("[INFO] migrating daily_ranking ...")
    df = pd.read_csv(RANKING_CSV, dtype={"code": str})
    if "date" not in df.columns:
        raise ValueError("ranking_final.csv missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])

    numeric_cols = [
        "close",
        "pred_return_60d",
        "pred_return_90d",
        "pred_mdd_60d",
        "pred_mdd_90d",
        "prob_top20_60d",
        "prob_top20_90d",
        "score",
        "score_score",
        "composite",
        "quality_score",
        "tech_score",
        "pred_score",
        "ret_score",
        "prob_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
        "final_score",
        "risk_penalty",
        "market_up",
        "market_kospi_close",
        "market_kospi_ma20",
        "market_vol_5d",
        "market_foreign_5d",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "market_status_date" in df.columns:
        df["market_status_date"] = pd.to_datetime(df["market_status_date"]).dt.strftime("%Y-%m-%d")
    else:
        df["market_status_date"] = None

    if "generated_at" not in df.columns:
        df["generated_at"] = None
    if "model_version" not in df.columns:
        df["model_version"] = None

    keep_cols = [
        "date",
        "code",
        "close",
        "pred_return_60d",
        "pred_return_90d",
        "pred_mdd_60d",
        "pred_mdd_90d",
        "prob_top20_60d",
        "prob_top20_90d",
        "score",
        "score_score",
        "composite",
        "quality_score",
        "name",
        "market",
        "sector",
        "tech_score",
        "pred_score",
        "ret_score",
        "prob_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
        "final_score",
        "risk_penalty",
        "market_up",
        "market_status_date",
        "market_kospi_close",
        "market_kospi_ma20",
        "market_vol_5d",
        "market_foreign_5d",
        "generated_at",
        "model_version",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    df = df[keep_cols]
    truncate_table(conn, "daily_ranking")
    df.to_sql("daily_ranking", conn, if_exists="append", index=False)
    print(f"[INFO] daily_ranking rows inserted: {len(df)}")


def migrate_daily_scores(conn: sqlite3.Connection) -> None:
    if not SCORES_CSV.exists():
        print(f"[WARN] {SCORES_CSV} not found. skip daily_scores.")
        return

    print("[INFO] migrating daily_scores ...")
    df = pd.read_csv(SCORES_CSV, dtype={"code": str})
    if "date" not in df.columns:
        raise ValueError("scores_final.csv missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])

    for c in ["score", "composite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None

    df = df[["date", "code", "score", "composite"]]
    truncate_table(conn, "daily_scores")
    df.to_sql("daily_scores", conn, if_exists="append", index=False)
    print(f"[INFO] daily_scores rows inserted: {len(df)}")


def migrate_labels(conn: sqlite3.Connection) -> None:
    if not LABELS_CSV.exists():
        print(f"[WARN] {LABELS_CSV} not found. skip labels.")
        return

    print("[INFO] migrating labels ...")
    df = pd.read_csv(LABELS_CSV, dtype={"code": str})
    if "date" not in df.columns:
        raise ValueError("labels.csv missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])

    # Known label columns
    def pick(colnames: list[str], candidates: list[str]) -> Optional[str]:
        for c in colnames:
            if c.lower() in candidates:
                return c
        return None

    cols = [c for c in df.columns]
    target_60_col = pick(cols, ["target_60d", "realized_return_60d", "y_return_60d", "target_return_60d"])
    target_90_col = pick(cols, ["target_90d", "realized_return_90d", "y_return_90d", "target_return_90d"])
    target_log_60_col = pick(cols, ["target_log_60d"])
    target_log_90_col = pick(cols, ["target_log_90d"])
    target_mdd_60_col = pick(cols, ["target_mdd_60d"])
    target_mdd_90_col = pick(cols, ["target_mdd_90d"])
    top20_60_col = pick(cols, ["target_60d_top20"])
    top20_90_col = pick(cols, ["target_90d_top20"])

    if target_60_col is None:
        print("[WARN] labels.csv missing target_60d-like column; inserting NULL.")

    out = pd.DataFrame()
    out["date"] = df["date"]
    out["code"] = df["code"]
    out["target_60d"] = pd.to_numeric(df[target_60_col], errors="coerce") if target_60_col else None
    out["target_90d"] = pd.to_numeric(df[target_90_col], errors="coerce") if target_90_col else None
    out["target_log_60d"] = pd.to_numeric(df[target_log_60_col], errors="coerce") if target_log_60_col else None
    out["target_log_90d"] = pd.to_numeric(df[target_log_90_col], errors="coerce") if target_log_90_col else None
    out["target_mdd_60d"] = pd.to_numeric(df[target_mdd_60_col], errors="coerce") if target_mdd_60_col else None
    out["target_mdd_90d"] = pd.to_numeric(df[target_mdd_90_col], errors="coerce") if target_mdd_90_col else None
    out["target_60d_top20"] = pd.to_numeric(df[top20_60_col], errors="coerce") if top20_60_col else None
    out["target_90d_top20"] = pd.to_numeric(df[top20_90_col], errors="coerce") if top20_90_col else None

    truncate_table(conn, "labels")
    out.to_sql("labels", conn, if_exists="append", index=False)
    print(f"[INFO] labels rows inserted: {len(out)}")


def migrate_backtest(conn: sqlite3.Connection) -> None:
    if not BACKTEST_CSV.exists():
        print(f"[WARN] {BACKTEST_CSV} not found. skip backtest_trades.")
        return

    print("[INFO] migrating backtest_trades ...")
    df = pd.read_csv(BACKTEST_CSV, dtype={"code": str})

    if "trade_date" not in df.columns:
        raise ValueError("score_backtest_from_labels.csv missing 'trade_date' column.")

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])

    if "strategy" not in df.columns:
        df["strategy"] = "score"

    for c in ["final_score", "pred_return_60d", "realized_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None

    out = df[["trade_date", "strategy", "code", "final_score", "pred_return_60d", "realized_return"]].copy()
    truncate_table(conn, "backtest_trades")
    out.to_sql("backtest_trades", conn, if_exists="append", index=False)
    print(f"[INFO] backtest_trades rows inserted: {len(out)}")


def migrate_market_status(conn: sqlite3.Connection) -> None:
    if not MARKET_STATUS_CSV.exists():
        print(f"[WARN] {MARKET_STATUS_CSV} not found. skip market_status.")
        return
    print("[INFO] migrating market_status ...")
    df = pd.read_csv(MARKET_STATUS_CSV)
    if "date" not in df.columns:
        raise ValueError("market_status.csv missing 'date' column.")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    if "market_up" in df.columns:
        df["market_up"] = df["market_up"].astype(int)
    else:
        df["market_up"] = None
    df = df[
        [
            "date",
            "kospi_close",
            "kospi_ma20",
            "volatility_5d",
            "foreign_net_5d",
            "market_up",
        ]
    ]
    truncate_table(conn, "market_status")
    df.to_sql("market_status", conn, if_exists="append", index=False)
    print(f"[INFO] market_status rows inserted: {len(df)}")


def _prep_price(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("price CSV missing date/code columns.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    df = df[["date", "code"] + cols]
    return df


def migrate_prices_raw(conn: sqlite3.Connection) -> None:
    if not PRICES_RAW_CSV.exists():
        print(f"[WARN] {PRICES_RAW_CSV} not found. skip prices_raw.")
        return
    print("[INFO] migrating prices_raw ...")
    df = pd.read_csv(PRICES_RAW_CSV, dtype={"code": str})
    df = _prep_price(df, ["open", "high", "low", "close", "volume"])
    truncate_table(conn, "prices_raw")
    df.to_sql("prices_raw", conn, if_exists="append", index=False)
    print(f"[INFO] prices_raw rows inserted: {len(df)}")


def migrate_prices_clean(conn: sqlite3.Connection) -> None:
    if not PRICES_CLEAN_CSV.exists():
        print(f"[WARN] {PRICES_CLEAN_CSV} not found. skip prices_clean.")
        return
    print("[INFO] migrating prices_clean ...")
    df = pd.read_csv(PRICES_CLEAN_CSV, dtype={"code": str})
    df = _prep_price(df, ["open", "high", "low", "close", "volume"])
    truncate_table(conn, "prices_clean")
    df.to_sql("prices_clean", conn, if_exists="append", index=False)
    print(f"[INFO] prices_clean rows inserted: {len(df)}")


def migrate_prices_adjusted(conn: sqlite3.Connection) -> None:
    if not PRICES_ADJ_CSV.exists():
        print(f"[WARN] {PRICES_ADJ_CSV} not found. skip prices_adjusted.")
        return
    print("[INFO] migrating prices_adjusted ...")
    df = pd.read_csv(PRICES_ADJ_CSV, dtype={"code": str})
    df = _prep_price(df, ["adj_open", "adj_high", "adj_low", "adj_close", "volume"])
    truncate_table(conn, "prices_adjusted")
    df.to_sql("prices_adjusted", conn, if_exists="append", index=False)
    print(f"[INFO] prices_adjusted rows inserted: {len(df)}")


def migrate_fact_price_daily(conn: sqlite3.Connection) -> None:
    """
    Populate fact_price_daily (adjusted 기준). Falls back to raw if adjusted is missing.
    Columns: date, code, open, high, low, close, adj_close, volume, value, market_cap, listed_shares
    """
    src = None
    use_adjusted = PRICES_ADJ_CSV.exists()
    if use_adjusted:
        print("[INFO] migrating fact_price_daily from prices_daily_adjusted.csv ...")
        src = pd.read_csv(PRICES_ADJ_CSV, dtype={"code": str})
        src = _prep_price(src, ["adj_open", "adj_high", "adj_low", "adj_close", "volume"])
        src.rename(
            columns={
                "adj_open": "open",
                "adj_high": "high",
                "adj_low": "low",
            },
            inplace=True,
        )
        src["close"] = src["adj_close"]
    elif PRICES_RAW_CSV.exists():
        print("[INFO] migrating fact_price_daily from prices_daily_raw.csv (fallback, unadjusted close) ...")
        src = pd.read_csv(PRICES_RAW_CSV, dtype={"code": str})
        src = _prep_price(src, ["open", "high", "low", "close", "volume"])
        src["adj_close"] = src["close"]
    else:
        print("[WARN] No price CSV found for fact_price_daily.")
        return

    for col in ["value", "market_cap", "listed_shares"]:
        src[col] = None

    truncate_table(conn, "fact_price_daily")
    src = src[
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "value",
            "market_cap",
            "listed_shares",
        ]
    ]
    src.to_sql("fact_price_daily", conn, if_exists="append", index=False)
    print(f"[INFO] fact_price_daily rows inserted: {len(src)}")


def migrate_fundamentals(conn: sqlite3.Connection) -> None:
    if not FUND_CSV.exists():
        print(f"[WARN] {FUND_CSV} not found. skip fundamentals.")
        return
    print("[INFO] migrating fundamentals ...")
    df = pd.read_csv(FUND_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("fundamentals.csv missing 'date' or 'code'")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])
    for c in ["roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    df = df[["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"]]
    truncate_table(conn, "fundamentals")
    df.to_sql("fundamentals", conn, if_exists="append", index=False)
    print(f"[INFO] fundamentals rows inserted: {len(df)}")


def migrate_quality(conn: sqlite3.Connection) -> None:
    if not QUALITY_CSV.exists():
        print(f"[WARN] {QUALITY_CSV} not found. skip quality.")
        return
    print("[INFO] migrating quality ...")
    df = pd.read_csv(QUALITY_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("quality.csv missing 'date' or 'code'")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])
    if "quality_score" in df.columns:
        df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    else:
        df["quality_score"] = None
    df = df[["date", "code", "quality_score"]]
    truncate_table(conn, "quality")
    df.to_sql("quality", conn, if_exists="append", index=False)
    print(f"[INFO] quality rows inserted: {len(df)}")


def migrate_features(conn: sqlite3.Connection) -> None:
    if not FEATURES_CSV.exists():
        print(f"[WARN] {FEATURES_CSV} not found. skip features.")
        return
    print("[INFO] migrating features ...")
    df = pd.read_csv(FEATURES_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv missing 'date' or 'code'")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])
    num_cols = [
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
        "quality_score",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    keep = ["date", "code"] + num_cols
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = df[keep]
    truncate_table(conn, "features")
    df.to_sql("features", conn, if_exists="append", index=False)
    print(f"[INFO] features rows inserted: {len(df)}")


def migrate_predictions(conn: sqlite3.Connection) -> None:
    if not PREDICTIONS_CSV.exists():
        print(f"[WARN] {PREDICTIONS_CSV} not found. skip predictions.")
        return
    print("[INFO] migrating predictions ...")
    df = pd.read_csv(PREDICTIONS_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("predictions.csv missing 'date' or 'code'")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code(df["code"])
    cols = [
        "pred_return_60d",
        "pred_return_90d",
        "pred_mdd_60d",
        "pred_mdd_90d",
        "prob_top20_60d",
        "prob_top20_90d",
        "score",
    ]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    df = df[["date", "code"] + cols]
    truncate_table(conn, "predictions")
    df.to_sql("predictions", conn, if_exists="append", index=False)
    print(f"[INFO] predictions rows inserted: {len(df)}")


def migrate_trades(conn: sqlite3.Connection) -> None:
    # Disabled: trades are managed in DB only. If CSV import is needed, call this manually.
    print("[INFO] migrate_trades skipped (DB-managed only).")


def main() -> None:
    print(f"[INFO] DB path: {DB_PATH}")
    conn = connect_db()
    try:
        reset_tables(conn)
        create_tables(conn)
        migrate_stocks(conn)
        migrate_daily_ranking(conn)
        migrate_daily_scores(conn)
        migrate_labels(conn)
        migrate_backtest(conn)
        migrate_market_status(conn)
        migrate_prices_raw(conn)
        migrate_prices_clean(conn)
        migrate_prices_adjusted(conn)
        migrate_fact_price_daily(conn)
        migrate_fundamentals(conn)
        migrate_quality(conn)
        migrate_features(conn)
        migrate_predictions(conn)
        # trades are DB-managed; skip CSV import
    finally:
        conn.close()
    print("[INFO] migration done.")


if __name__ == "__main__":
    main()
