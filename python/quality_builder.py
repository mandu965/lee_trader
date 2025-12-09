import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sqlite3
try:
    from db import get_engine
except Exception:
    get_engine = None

DATA_DIR = Path("data")
FUND_CSV = DATA_DIR / "fundamentals.csv"
OUT_CSV = DATA_DIR / "quality.csv"
DB_PATH = DATA_DIR / "lee_trader.db"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    m = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - m) / sd


def load_fundamentals() -> pd.DataFrame:
    if not FUND_CSV.exists():
        logging.warning("fundamentals.csv not found (%s) -> skipping quality build", FUND_CSV.resolve())
        return pd.DataFrame(columns=["date", "code"])

    df = pd.read_csv(FUND_CSV, dtype={"code": str})
    # 기본 컬럼 정규화
    required_like = ["date", "code"]
    for c in required_like:
        if c not in df.columns:
            raise ValueError(f"fundamentals.csv missing required '{c}' column")

    # 컬럼 매핑(유연하게): 사용자가 제공한 이름이 다를 수 있으므로 후보 키를 정의
    # 사용 후보 -> 표준 컬럼명 매핑
    candidates: Dict[str, List[str]] = {
        "roe": ["roe", "ROE", "return_on_equity"],
        "op_margin": ["op_margin", "operating_margin", "OPE_MARGIN"],
        "debt_ratio": ["debt_ratio", "debt", "liabilities_ratio"],
        "ocf_to_assets": ["ocf_to_assets", "ocf_assets", "cashflow_to_assets", "ocf"],
        "net_margin": ["net_margin", "netprofit_margin", "NPM"],
    }

    def pick_first(d: pd.DataFrame, keys: List[str]) -> str:
        for k in keys:
            if k in d.columns:
                return k
        return ""

    col_map: Dict[str, str] = {}
    for std_name, keys in candidates.items():
        col = pick_first(df, keys)
        if col:
            col_map[std_name] = col

    if not col_map:
        logging.warning("fundamentals.csv has no known financial columns -> skipping quality build")
        return pd.DataFrame(columns=["date", "code"])

    # 표준 명칭으로 복사(존재하는 컬럼만)
    work = df[["date", "code"] + list(col_map.values())].copy()
    # rename to std names
    inv_map = {v: k for k, v in col_map.items()}
    work = work.rename(columns=inv_map)

    # 타입/정렬
    work["code"] = work["code"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values(["code", "date"]).reset_index(drop=True)

    num_cols = list(inv_map.values())
    work = _coerce_numeric(work, num_cols)

    logging.info(
        "Loaded fundamentals: rows=%d, cols=%s (mapped=%s)",
        len(work),
        list(work.columns),
        col_map,
    )
    return work


def build_quality(df: pd.DataFrame) -> pd.DataFrame:
    # z-score 계산(존재하는 컬럼만)
    z = {}
    if "roe" in df.columns:
        z["z_roe"] = _zscore(df["roe"])
    if "op_margin" in df.columns:
        z["z_op"] = _zscore(df["op_margin"])
    if "net_margin" in df.columns:
        z["z_net"] = _zscore(df["net_margin"])
    if "debt_ratio" in df.columns:
        # 부채비율은 낮을수록 우수 -> 음의 방향
        z["z_debt_inv"] = -_zscore(df["debt_ratio"])

    # 가용 z-score 가중합
    # 기본 가중치
    weights = {
        "z_roe": 0.35,
        "z_op": 0.25,
        "z_net": 0.20,
        "z_debt_inv": 0.20,
    }
    # 실제 사용 가능한 키만 추림
    use_keys = [k for k in weights.keys() if k in z]
    if not use_keys:
        logging.warning("No valid financial columns available to build quality_score -> returning empty")
        return pd.DataFrame(columns=["date", "code", "quality_score"])

    # 가중치 재정규화(가용 키만 합=1)
    w_sum = sum(weights[k] for k in use_keys)
    w = {k: weights[k] / w_sum for k in use_keys}

    out = df[["date", "code"]].copy()
    for k in z:
        out[k] = z[k]

    # weighted sum
    score = np.zeros(len(out), dtype=float)
    for k in use_keys:
        score += w[k] * out[k].fillna(0.0).astype(float).values
    # clip to [-3,3] to reduce extremes
    score = np.clip(score, -3.0, 3.0)

    out["quality_score"] = score
    return out[["date", "code", "quality_score"]].copy()


def save_quality(df: pd.DataFrame) -> None:
    ensure_data_dir()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved quality: %s (rows=%d)", OUT_CSV.resolve(), len(out))

    # DB upsert (prefer Postgres via SQLAlchemy)
    try:
        if get_engine:
            eng = get_engine()
            out.to_sql("quality", eng, if_exists="replace", index=False)
            logging.info("Saved quality to Postgres via SQLAlchemy (rows=%d)", len(out))
            return
    except Exception:
        logging.exception("SQLAlchemy save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        out.to_sql("quality", conn, if_exists="replace", index=False)
        conn.commit()
        logging.info("Saved quality to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(out))
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
    fund = load_fundamentals()
    if fund.empty:
        logging.warning("No fundamentals data -> quality.csv will be empty with headers only")
        qual = pd.DataFrame(columns=["date", "code", "quality_score"])
    else:
        qual = build_quality(fund)
    save_quality(qual)


if __name__ == "__main__":
    main()
