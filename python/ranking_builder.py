"""
ranking_builder.py

Build the final per-stock ranking table from:

- data/predictions.csv   (model outputs)
- data/scores_final.csv  (technical composite score)
- data/features.csv      (price/indicators + quality_score)
- data/universe.csv      (code, name, market, sector)
- data/market_status.csv (KOSPI regime info)

항상 종목 랭킹은 만들고,
market_status.csv 에서 읽은 시장 상태(시장 상승/하락 및 지표)를
각 row에 meta 컬럼으로 붙여준다.

추가되는 컬럼:
- tech_score, pred_score, prob_score, qual_score, final_score
- market_up               (bool)
- market_status_date      (str)
- market_kospi_close      (float)
- market_kospi_ma20       (float)
- market_vol_5d           (float)
- market_foreign_5d       (float)
- generated_at            (str)
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
from scoring import compute_final_score_v5
try:
    from db import get_engine
except Exception:
    get_engine = None

DATA_DIR = Path("data")

PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
SCORES_CSV = DATA_DIR / "scores_final.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"

OUT_CSV = DATA_DIR / "ranking_final.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

# -------------------------------
# V2 Scoring Weights
# -------------------------------
# pred_score : 예측 수익률 기반 점수 (return_score)
# prob_score : 상위 20% 안에 들 확률 점수
# tech_score : 기술적 패턴 점수 (차트/모멘텀)
# safety_score : 변동성 낮을수록 높은 점수
# qual_score : 재무 퀄리티 점수
# liquidity_score : 거래량(유동성) 점수

WEIGHT_TECH = 0.15         # 기술적 패턴
WEIGHT_PRED = 0.30         # 예측 수익(핵심)
WEIGHT_PROB = 0.25         # 상위20% 확률
WEIGHT_SAFETY = 0.15       # 리스크(변동성) 낮을수록 +
WEIGHT_QUAL = 0.10         # 재무 퀄리티
WEIGHT_LIQUIDITY = 0.05    # 유동성

# Risk penalty 설정: -15%까지는 감점 없음, 그 아래로는 점점 감점
RISK_MDD_THRESHOLD = 0.15   # 15% drawdown까지는 허용
RISK_PENALTY_SCALE = 100.0  # penalty_raw(0~0.3 정도)를 점수 스케일로 맞춰주기


# Risk penalty 설정: -15%까지는 감점 없음, 그 아래로는 점점 감점
RISK_MDD_THRESHOLD = 0.15   # 15% drawdown까지는 허용
RISK_PENALTY_SCALE = 100.0  # penalty_raw(0~0.3 정도)를 점수 스케일로 맞춰주기

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def _load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input CSV not found: {path}")
        logging.warning("Optional input CSV not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logging.info("Loaded %s (rows=%d)", path, len(df))
    return df


def _clip01(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.astype(float).clip(lower=lower, upper=upper)


def _percentile_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Compute 0~100 percentile (rank) of `col` within each date group.
    Higher values -> higher percentile.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True) * 100.0

    ranked = df.groupby("date", group_keys=False)[col].transform(_rank)
    return ranked


def _percentile01_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Compute 0~1 percentile (rank) of `col` within each date group.
    Higher values -> higher percentile.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True)

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _load_market_status():
    """
    market_status.csv 에서 최신 시장 상태와 지표들을 읽어온다.

    return:
        market_up (bool),
        info (dict) – {
            "date": str,
            "kospi_close": float,
            "kospi_ma20": float,
            "volatility_5d": float,
            "foreign_net_5d": float,
        }
    """
    if not MARKET_STATUS_CSV.exists():
        logging.warning("market_status.csv not found; default market_up=True")
        return True, {}

    try:
        df = pd.read_csv(MARKET_STATUS_CSV)
    except Exception:
        logging.exception("Failed to read market_status.csv; default market_up=True")
        return True, {}

    if df.empty or "market_up" not in df.columns:
        logging.warning("market_status.csv empty or missing market_up; default True")
        return True, {}

    last = df.iloc[-1]

    raw = last["market_up"]
    if isinstance(raw, bool):
        market_up = raw
    else:
        v = str(raw).strip().lower()
        market_up = v in {"true", "1", "t", "y", "yes"}

    info = {}
    for col in ["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d"]:
        if col in last.index:
            info[col] = last[col]

    logging.info(
        "Loaded market status: market_up=%s, info=%s",
        market_up,
        {k: info.get(k) for k in ["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d"]},
    )
    return market_up, info


def build_ranking() -> pd.DataFrame:
    # ---------------------------------------------
    # 1. 원본 CSV 로드
    # ---------------------------------------------
    preds = _load_csv(PREDICTIONS_CSV, required=True)
    scores = _load_csv(SCORES_CSV, required=False)
    feats = _load_csv(FEATURES_CSV, required=True)
    universe = _load_csv(UNIVERSE_CSV, required=False)

    # 날짜 포맷 통일
    preds = _normalize_date(preds)
    scores = _normalize_date(scores)
    feats = _normalize_date(feats)

    # 기본 sanity check
    for df, name in [
        (preds, "predictions"),
        (feats, "features"),
    ]:
        if df.empty:
            raise RuntimeError(f"{name} is empty – cannot build ranking.")

    # scores_final 없으면 tech_score=0 기본값으로 대체
    if scores.empty:
        logging.warning("scores_final.csv missing/empty -> tech_score set to 0")
        scores = preds[["date", "code"]].copy()
        scores["score"] = 0.0

    # code를 모두 문자열/6자리로 통일
    for df in [preds, scores, feats, universe]:
        if df is not None and not df.empty and "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)

    # ---------------------------------------------
    # 2. 병합 (predictions 기준)
    # ---------------------------------------------
    base = preds.merge(
        scores,
        on=["date", "code"],
        how="left",
        suffixes=("", "_score"),
    )

    # features에서 필요한 컬럼만 사용 (close, quality_score, 변동성/유동성 등)
    feat_cols = ["date", "code", "close"]

    # 재무 퀄리티
    if "quality_score" in feats.columns:
        feat_cols.append("quality_score")

    # 변동성 / 유동성 관련 피처 (있을 때만 사용)
    for col in ["vol_20", "vol_60", "vol_ma_20", "volume", "mom_60d", "rsi_14", "turnover_20d", "vol_20d"]:
        if col in feats.columns:
            feat_cols.append(col)

    base = base.merge(
        feats[feat_cols],
        on=["date", "code"],
        how="left",
        suffixes=("", "_feat"),
    )


    logging.info(
        "Base merged shape (preds + scores + features): %s",
        base.shape,
    )

    # universe에서 name, market, sector, etc. 붙이기 (선택)
    if universe is not None and not universe.empty and "code" in universe.columns:
        base = base.merge(
            universe,
            on="code",
            how="left",
            suffixes=("", "_univ"),
        )
        logging.info("After universe merge shape: %s", base.shape)

    if base.empty:
        raise RuntimeError(
            "No rows after merging predictions/scores/features – cannot build ranking."
        )
    # ---------------------------------------------
    # 3. scoring (scoring.py based)
    # ---------------------------------------------
    for col in ["mom_60d", "rsi_14", "turnover_20d", "vol_20"]:
        if col not in base.columns:
            base[col] = 0.0
    if "vol_20d" not in base.columns and "vol_20" in base.columns:
        base["vol_20d"] = base["vol_20"]


    base["pred_return_60d_pct01"] = _percentile01_by_date(base, "pred_return_60d") if "pred_return_60d" in base.columns else np.nan
    base["pred_return_90d_pct01"] = _percentile01_by_date(base, "pred_return_90d") if "pred_return_90d" in base.columns else np.nan
    base["ret_score_v11"] = 100.0 * (
        0.7 * base["pred_return_60d_pct01"].fillna(0)
        + 0.3 * base["pred_return_90d_pct01"].fillna(0)
    )

    market_up, mkt_info = _load_market_status()
    market_row = {
        "market_regime": "bull" if market_up else "bear",
        "foreign_5d": mkt_info.get("foreign_net_5d", 0),
        "market_foreign_5d": mkt_info.get("foreign_net_5d", 0),
    }
    base = compute_final_score_v5(base, market_row)
    base["final_score"] = base["final_score_v5"]

    # Rescale final_score to date-wise percentile (0~100) for more meaningful ranking spread.
    base["final_score_raw"] = base["final_score"]
    base["final_score"] = _percentile_by_date(base, "final_score_raw").fillna(0)

    # ---------------------------------------------
    # 6. 정렬 (최신 날짜 + 높은 점수 순)
    # ---------------------------------------------
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(
        ["date", "final_score"],
        ascending=[False, False],
    )
    base["date"] = base["date"].dt.strftime("%Y-%m-%d")

    # ---------------------------------------------
    # 7. 시장 상태 메타 정보 붙이기
    # ---------------------------------------------
    market_up, mkt_info = _load_market_status()
    base["market_up"] = market_up
    base["market_status_date"] = mkt_info.get("date")

    # 수치형으로 변환
    base["market_kospi_close"] = pd.to_numeric(
        mkt_info.get("kospi_close"),
        errors="coerce",
    )
    base["market_kospi_ma20"] = pd.to_numeric(
        mkt_info.get("kospi_ma20"),
        errors="coerce",
    )
    base["market_vol_5d"] = pd.to_numeric(
        mkt_info.get("volatility_5d"),
        errors="coerce",
    )
    base["market_foreign_5d"] = pd.to_numeric(
        mkt_info.get("foreign_net_5d"),
        errors="coerce",
    )

    # 생성 시각
    base["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return base



def save_ranking(df: pd.DataFrame) -> None:
    ensure_data_dir()
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
    # ensure model_version exists (even as None) for DB binding
    if "model_version" not in df_out.columns:
        df_out["model_version"] = None
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved ranking: %s (rows=%d)", OUT_CSV.resolve(), len(df_out))

    # DB upsert (prefer Postgres via SQLAlchemy)
    try:
        if get_engine:
            eng = get_engine()
            df_out.to_sql("daily_ranking", eng, if_exists="replace", index=False)
            logging.info("Saved ranking to Postgres via SQLAlchemy (rows=%d)", len(df_out))
            return
    except Exception:
        logging.exception("SQLAlchemy save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        df_out.to_sql("daily_ranking", conn, if_exists="replace", index=False)
        conn.commit()
        logging.info("Saved ranking to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(df_out))
    except Exception:
        logging.exception("Failed to save ranking to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    ranking = build_ranking()
    save_ranking(ranking)


if __name__ == "__main__":
    main()
