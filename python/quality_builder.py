import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg, replace_table_rows_sqlite, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None
    def use_sqlite_fallback_writes() -> bool:
        return False


DATA_DIR = Path("data")
FUND_CSV = DATA_DIR / "fundamentals.csv"
OUT_CSV = DATA_DIR / "quality.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
QUALITY_PK = ["date", "code"]

# Factor configuration is explicit so directionality and operating weights are stable.
QUALITY_FACTOR_SPECS = {
    "roe": {
        "aliases": ["roe", "ROE", "return_on_equity"],
        "weight": 0.25,
        "direction": 1.0,  # high is good
    },
    "op_margin": {
        "aliases": ["op_margin", "operating_margin", "oper_margin", "OPE_MARGIN"],
        "weight": 0.20,
        "direction": 1.0,  # high is good
    },
    "net_margin": {
        "aliases": ["net_margin", "net_profit_margin", "profit_margin", "netincome_margin"],
        "weight": 0.20,
        "direction": 1.0,  # high is good
    },
    "debt_ratio": {
        "aliases": ["debt_ratio", "debt", "liabilities_ratio", "debt_to_equity"],
        "weight": 0.15,
        "direction": -1.0,  # low is good
    },
    "ocf_to_assets": {
        "aliases": ["ocf_to_assets", "ocf_assets", "cashflow_to_assets", "operating_cf_to_assets"],
        "weight": 0.20,
        "direction": 1.0,  # high is good
    },
}
FACTOR_COLUMNS = list(QUALITY_FACTOR_SPECS.keys())
YOY_GROWTH_COLUMNS = ["revenue_growth_yoy", "op_income_growth_yoy", "roe_yoy"]
QUALITY_OUTPUT_COLUMNS = [
    "date",
    "code",
    *FACTOR_COLUMNS,
    *YOY_GROWTH_COLUMNS,
    "quality_raw_score",
    "quality_score",
    "quality_factor_count",
    "quality_missing_ratio",
    "quality_score_confidence",
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _pick_first_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _apply_limited_factor_backfill(df: pd.DataFrame, factors: list[str], limit: int = 1) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty or not factors:
        return df.copy(), {}

    out = df.copy()
    filled_counts: dict[str, int] = {}
    for factor in factors:
        if factor not in out.columns:
            continue
        before_missing = pd.to_numeric(out[factor], errors="coerce").isna()
        out[factor] = out.groupby("code", group_keys=False)[factor].ffill(limit=limit)
        after_missing = pd.to_numeric(out[factor], errors="coerce").isna()
        filled_counts[factor] = int((before_missing & ~after_missing).sum())
    return out, filled_counts


def _winsorize_by_date(series: pd.Series, dates: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "value": pd.to_numeric(series, errors="coerce")})

    def _clip(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if valid.empty:
            return values
        lo = valid.quantile(lower_q)
        hi = valid.quantile(upper_q)
        return values.clip(lower=lo, upper=hi)

    return frame.groupby("date")["value"].transform(_clip)


def _median_fill_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "value": pd.to_numeric(series, errors="coerce")})
    date_median = frame.groupby("date")["value"].transform("median")
    filled = frame["value"].fillna(date_median)
    return filled.fillna(0.0)


def _robust_zscore_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "value": pd.to_numeric(series, errors="coerce")})

    def _transform(values: pd.Series) -> pd.Series:
        values = values.astype(float)
        median = values.median(skipna=True)
        mad = (values - median).abs().median(skipna=True)
        if pd.isna(mad) or mad == 0:
            mean = values.mean(skipna=True)
            std = values.std(ddof=0, skipna=True)
            if pd.isna(std) or std == 0:
                return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
            return (values - mean) / std
        scale = 1.4826 * mad
        return (values - median) / scale

    return frame.groupby("date")["value"].transform(_transform)


def _percentile_rank_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "value": pd.to_numeric(series, errors="coerce")})

    def _rank(values: pd.Series) -> pd.Series:
        if values.notna().sum() == 0:
            return pd.Series(np.nan, index=values.index, dtype=float)
        return values.rank(pct=True, method="average") * 100.0

    return frame.groupby("date")["value"].transform(_rank)


def build_yoy_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """코드별 연간 YoY 성장률 계산. fundamentals는 연 1행(date=연말)이 전제."""
    out = df.sort_values(["code", "date"]).copy()

    yoy_map = [
        ("revenue",   "revenue_growth_yoy"),
        ("op_income", "op_income_growth_yoy"),
        ("roe",       "roe_yoy"),
    ]
    for src_col, new_col in yoy_map:
        if src_col in out.columns:
            raw = out.groupby("code", group_keys=False)[src_col].pct_change(1)
            # 분모 0 또는 부호 역전 시 inf/-inf 발생 → 극단값 clip
            out[new_col] = raw.replace([np.inf, -np.inf], np.nan).clip(lower=-10.0, upper=10.0)
        else:
            out[new_col] = np.nan

    return out


def load_fundamentals() -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    if not FUND_CSV.exists():
        logging.warning("fundamentals.csv not found (%s) -> skipping quality build", FUND_CSV.resolve())
        return pd.DataFrame(columns=["date", "code"]), {}, FACTOR_COLUMNS.copy()

    df = pd.read_csv(FUND_CSV, dtype={"code": str})
    # revenue / op_income 컬럼이 없는 구버전 fundamentals.csv 호환
    for col in ("revenue", "op_income"):
        if col not in df.columns:
            df[col] = np.nan
    for col in ["date", "code"]:
        if col not in df.columns:
            raise ValueError(f"fundamentals.csv missing required '{col}' column")

    mapped: dict[str, str] = {}
    missing: list[str] = []
    for factor, spec in QUALITY_FACTOR_SPECS.items():
        source_col = _pick_first_column(df, spec["aliases"])
        if source_col is None:
            missing.append(factor)
            logging.warning("quality factor column missing: factor=%s aliases=%s", factor, spec["aliases"])
            continue
        mapped[factor] = source_col

    extra_cols = [c for c in ("revenue", "op_income") if c in df.columns]
    keep_cols = ["date", "code", *mapped.values(), *extra_cols]
    work = df[keep_cols].copy()
    rename_map = {source: factor for factor, source in mapped.items()}
    work = work.rename(columns=rename_map)
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
    work = _coerce_numeric(work, list(mapped.keys()))
    work, filled_counts = _apply_limited_factor_backfill(work, list(mapped.keys()), limit=1)

    logging.info(
        "Loaded fundamentals: rows=%d dates=%d mapped_factors=%s missing_factors=%s backfilled=%s",
        len(work),
        work["date"].nunique(),
        mapped,
        missing,
        filled_counts,
    )
    return work, mapped, missing


def build_quality(df: pd.DataFrame, mapped: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS)

    active_factors = [factor for factor in FACTOR_COLUMNS if factor in mapped]
    if not active_factors:
        logging.warning("No mapped quality factors available -> returning empty quality output")
        return pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS)

    out = df[["date", "code", *active_factors]].copy()
    original_missing = out[active_factors].isna()
    out["quality_factor_count"] = (~original_missing).sum(axis=1).astype(int)
    out["quality_missing_ratio"] = 1.0 - (out["quality_factor_count"] / float(len(active_factors)))

    available_weight = np.zeros(len(out), dtype=float)
    for factor in active_factors:
        factor_weight = float(QUALITY_FACTOR_SPECS[factor]["weight"])
        available_weight += (~original_missing[factor]).astype(float).to_numpy() * factor_weight
    active_weight_sum = sum(QUALITY_FACTOR_SPECS[factor]["weight"] for factor in active_factors)
    weighted_confidence = pd.Series((available_weight / float(active_weight_sum)) * 100.0, index=out.index, dtype=float)

    has_op_margin = (~original_missing["op_margin"]) if "op_margin" in original_missing.columns else pd.Series(False, index=out.index)
    has_debt_ratio = (~original_missing["debt_ratio"]) if "debt_ratio" in original_missing.columns else pd.Series(False, index=out.index)
    has_net_margin = (~original_missing["net_margin"]) if "net_margin" in original_missing.columns else pd.Series(False, index=out.index)
    has_roe = (~original_missing["roe"]) if "roe" in original_missing.columns else pd.Series(False, index=out.index)
    has_ocf = (~original_missing["ocf_to_assets"]) if "ocf_to_assets" in original_missing.columns else pd.Series(False, index=out.index)

    core_pair = has_op_margin & has_debt_ratio
    core_trio = core_pair & (has_net_margin | has_roe | has_ocf)
    full_house = core_pair & has_net_margin & has_roe & has_ocf

    confidence = weighted_confidence.copy()
    confidence = confidence.where(~core_pair, confidence.clip(lower=60.0))
    confidence = confidence.where(~core_trio, confidence.clip(lower=75.0))
    confidence = confidence.where(~full_house, confidence.clip(lower=95.0))
    out["quality_score_confidence"] = confidence.clip(lower=0.0, upper=100.0)

    z_cols: list[str] = []
    for factor in active_factors:
        spec = QUALITY_FACTOR_SPECS[factor]
        # Cross-sectional winsorization by date avoids temporal leakage and reduces outlier impact.
        winsorized = _winsorize_by_date(out[factor], out["date"])
        filled = _median_fill_by_date(winsorized, out["date"])
        z = _robust_zscore_by_date(filled, out["date"]) * float(spec["direction"])
        z = pd.to_numeric(z, errors="coerce").clip(lower=-4.0, upper=4.0)
        z_col = f"{factor}_z"
        out[z_col] = z.fillna(0.0)
        z_cols.append(z_col)

    normalized_weights = {
        factor: QUALITY_FACTOR_SPECS[factor]["weight"] / active_weight_sum for factor in active_factors
    }
    raw_score = np.zeros(len(out), dtype=float)
    for factor in active_factors:
        raw_score += normalized_weights[factor] * out[f"{factor}_z"].to_numpy(dtype=float)
    out["quality_raw_score"] = raw_score

    # Percentile rank is used instead of min-max because it is more stable under outliers
    # while still spreading the cross-sectional quality score across the full 0~100 operating range.
    out["quality_score"] = _percentile_rank_by_date(out["quality_raw_score"], out["date"]).clip(lower=0.0, upper=100.0)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for factor in FACTOR_COLUMNS:
        if factor not in out.columns:
            out[factor] = np.nan

    # YoY 성장률: build_yoy_growth_features()가 먼저 실행된 df에서 전달됨
    for yoy_col in YOY_GROWTH_COLUMNS:
        if yoy_col in df.columns:
            out[yoy_col] = df[yoy_col].values
        else:
            out[yoy_col] = np.nan

    logging.info(
        "Built quality score: rows=%d active_factors=%s score_mean=%.4f confidence_mean=%.4f",
        len(out),
        active_factors,
        float(pd.to_numeric(out["quality_score"], errors="coerce").mean()),
        float(pd.to_numeric(out["quality_score_confidence"], errors="coerce").mean()),
    )
    return out[QUALITY_OUTPUT_COLUMNS].copy()


def _get_pg_columns(table: str) -> list[str]:
    if not get_engine:
        return []
    try:
        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table
                    ORDER BY ordinal_position
                    """
                ),
                {"table": table},
            ).fetchall()
        return [row[0] for row in rows]
    except Exception:
        logging.exception("Failed to inspect Postgres columns for %s", table)
        return []


def _get_sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        logging.exception("Failed to inspect sqlite columns for %s", table)
        return []
    return [row[1] for row in rows]


def _prepare_rows(df: pd.DataFrame, actual_columns: list[str]) -> pd.DataFrame:
    use_columns = [col for col in QUALITY_OUTPUT_COLUMNS if col in actual_columns]
    out = df.copy()
    for col in use_columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[use_columns]


def _ensure_pg_quality_columns() -> None:
    if not get_engine:
        return
    column_defs = {
        "roe": "DOUBLE PRECISION",
        "op_margin": "DOUBLE PRECISION",
        "net_margin": "DOUBLE PRECISION",
        "debt_ratio": "DOUBLE PRECISION",
        "ocf_to_assets": "DOUBLE PRECISION",
        "revenue_growth_yoy": "DOUBLE PRECISION",
        "op_income_growth_yoy": "DOUBLE PRECISION",
        "roe_yoy": "DOUBLE PRECISION",
        "quality_raw_score": "DOUBLE PRECISION",
        "quality_factor_count": "INTEGER",
        "quality_missing_ratio": "DOUBLE PRECISION",
        "quality_score_confidence": "DOUBLE PRECISION",
    }
    eng = get_engine()
    with eng.begin() as conn:
        for col, col_type in column_defs.items():
            conn.execute(text(f"ALTER TABLE quality ADD COLUMN IF NOT EXISTS {col} {col_type}"))


def _ensure_sqlite_quality_columns(conn: sqlite3.Connection) -> None:
    existing = set(_get_sqlite_columns(conn, "quality"))
    column_defs = {
        "roe": "REAL",
        "op_margin": "REAL",
        "net_margin": "REAL",
        "debt_ratio": "REAL",
        "ocf_to_assets": "REAL",
        "revenue_growth_yoy": "REAL",
        "op_income_growth_yoy": "REAL",
        "roe_yoy": "REAL",
        "quality_raw_score": "REAL",
        "quality_factor_count": "INTEGER",
        "quality_missing_ratio": "REAL",
        "quality_score_confidence": "REAL",
    }
    for col, col_type in column_defs.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE quality ADD COLUMN {col} {col_type}")


def save_quality(df: pd.DataFrame) -> None:
    ensure_data_dir()
    out = df.copy()
    for col in QUALITY_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[QUALITY_OUTPUT_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(out, QUALITY_PK, "quality")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved quality: %s (rows=%d)", OUT_CSV.resolve(), len(out))

    try:
        if replace_table_rows_pg:
            _ensure_pg_quality_columns()
            pg_columns = _get_pg_columns("quality")
            db_out = _prepare_rows(out, pg_columns or QUALITY_OUTPUT_COLUMNS)
            if ensure_unique_keys:
                ensure_unique_keys(db_out, QUALITY_PK, "quality")
            replace_table_rows_pg("quality", db_out, columns=list(db_out.columns))
            logging.info("Replaced quality rows in Postgres (rows=%d)", len(db_out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to sqlite")

    if not use_sqlite_fallback_writes():
        logging.info("Skipping sqlite fallback for quality (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quality (
                date                     DATE NOT NULL,
                code                     TEXT NOT NULL,
                roe                      REAL,
                op_margin                REAL,
                net_margin               REAL,
                debt_ratio               REAL,
                ocf_to_assets            REAL,
                revenue_growth_yoy       REAL,
                op_income_growth_yoy     REAL,
                roe_yoy                  REAL,
                quality_raw_score        REAL,
                quality_score            REAL,
                quality_factor_count     INTEGER,
                quality_missing_ratio    REAL,
                quality_score_confidence REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        _ensure_sqlite_quality_columns(conn)
        sqlite_columns = _get_sqlite_columns(conn, "quality")
        db_out = _prepare_rows(out, sqlite_columns or QUALITY_OUTPUT_COLUMNS)
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "quality", db_out)
        conn.commit()
        logging.info("Saved quality to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(db_out))
    except Exception:
        logging.exception("Failed to save quality to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    ensure_data_dir()
    fund, mapped, _missing = load_fundamentals()
    if fund.empty:
        logging.warning("No fundamentals data -> quality.csv will be empty with headers only")
        qual = pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS)
    else:
        fund = build_yoy_growth_features(fund)
        qual = build_quality(fund, mapped)
    save_quality(qual)


if __name__ == "__main__":
    main()
