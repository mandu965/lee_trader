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
    scores = _load_csv(SCORES_CSV, required=True)
    feats = _load_csv(FEATURES_CSV, required=True)
    universe = _load_csv(UNIVERSE_CSV, required=False)

    # 날짜 포맷 통일
    preds = _normalize_date(preds)
    scores = _normalize_date(scores)
    feats = _normalize_date(feats)

    # 기본 sanity check
    for df, name in [
        (preds, "predictions"),
        (scores, "scores_final"),
        (feats, "features"),
    ]:
        if df.empty:
            raise RuntimeError(f"{name} is empty – cannot build ranking.")

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
    for col in ["vol_20", "vol_60", "vol_ma_20", "volume"]:
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
    # 3. 점수 계산 (V2)
    # ---------------------------------------------

    # 3-1) tech_score: scores_final.score를 0~100으로 clip
    # if "score" in base.columns:
    #     base["tech_score"] = _clip01(base["score"].fillna(0.0), 0.0, 100.0)
    # else:
    #     logging.warning("'score' column not found; tech_score will be NaN.")
    #     base["tech_score"] = np.nan

    # 3-1) tech_score: scores_final.csv에서 온 기술 점수 사용
    #   - score_score: 과거 scoring.py에서 만든 기술 점수
    #   - composite:   추가로 만든 종합 기술 점수라면 이쪽을 우선 사용해도 됨

    if "composite" in base.columns:
        # composite이 더 종합적인 기술점수라면 이걸 쓰자
        base["tech_score"] = _percentile_by_date(base, "composite")
    elif "score_score" in base.columns:
        # 아니면 score_score를 날짜별 percentile로 변환 (0~100)
        base["tech_score"] = _percentile_by_date(base, "score_score")
    else:
        logging.warning(
            "No 'composite' or 'score_score' column found; tech_score will be NaN."
        )
        base["tech_score"] = np.nan
    

    # 3-2) pred_score (return_score):
    #   pred_return_60d와 pred_return_90d 둘 다 있으면 0.6 : 0.4 가중 평균
    pred_60 = None
    pred_90 = None

    if "pred_return_60d" in base.columns:
        base["pred_score_60"] = _percentile_by_date(base, "pred_return_60d")
        pred_60 = base["pred_score_60"]
    if "pred_return_90d" in base.columns:
        base["pred_score_90"] = _percentile_by_date(base, "pred_return_90d")
        pred_90 = base["pred_score_90"]

    if (pred_60 is not None) and (pred_90 is not None):
        base["pred_score"] = 0.6 * pred_60 + 0.4 * pred_90
    elif pred_60 is not None:
        base["pred_score"] = pred_60
    elif pred_90 is not None:
        base["pred_score"] = pred_90
    else:
        logging.warning(
            "No 'pred_return_60d' or 'pred_return_90d' columns; pred_score will be NaN."
        )
        base["pred_score"] = np.nan

    # 🔥 Node /api/top20 에서 사용하는 이름(ret_score)은 pred_score와 동일하게 유지
    base["ret_score"] = base["pred_score"]

    # 3-3) prob_score: prob_top20_60d * 100  (분류 모델 확률 활용)
    if "prob_top20_60d" in base.columns:
        base["prob_score"] = _clip01(
            base["prob_top20_60d"].fillna(0.0) * 100.0,
            0.0,
            100.0,
        )
    else:
        logging.warning("'prob_top20_60d' column not found; prob_score will be NaN.")
        base["prob_score"] = np.nan

    # 3-4) qual_score: quality_score의 날짜별 percentile (0~100)
    if "quality_score" in base.columns:
        base["qual_score"] = _percentile_by_date(base, "quality_score")
    else:
        logging.warning("'quality_score' column not found; qual_score will be NaN.")
        base["qual_score"] = np.nan

    # 3-5) safety_score: 변동성(vol_20, vol_60)이 낮을수록 높은 점수
    safety_parts = []

    if "vol_20" in base.columns:
        base["vol_20_pct"] = _percentile_by_date(base, "vol_20")
        # 변동성 낮을수록 좋으므로 100 - percentile
        safety_parts.append(100.0 - base["vol_20_pct"])

    if "vol_60" in base.columns:
        base["vol_60_pct"] = _percentile_by_date(base, "vol_60")
        safety_parts.append(100.0 - base["vol_60_pct"])

    if safety_parts:
        # 여러 개가 있으면 단순 평균 (0~100)
        base["safety_score"] = sum(safety_parts) / len(safety_parts)
    else:
        logging.info("No vol_20 / vol_60 columns; safety_score will be NaN.")
        base["safety_score"] = np.nan

    # 3-6) liquidity_score: 최근 20일 평균 거래량 기준 (vol_ma_20 우선)
    if "vol_ma_20" in base.columns:
        base["liquidity_score"] = _percentile_by_date(base, "vol_ma_20")
    elif "volume" in base.columns:
        base["liquidity_score"] = _percentile_by_date(base, "volume")
    else:
        logging.info(
            "No vol_ma_20 / volume columns; liquidity_score will be NaN."
        )
        base["liquidity_score"] = np.nan

    # NaN component scores -> 0 (점수 계산에서 결측치는 0점 처리)
    for col in [
        "tech_score",
        "pred_score",
        "prob_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
    ]:
        base[col] = base[col].fillna(0.0)


    # ---------------------------------------------
    # 4. 기본 종합 점수 (회귀 + 분류 + 기술 + 퀄리티 + 리스크 + 유동성)
    # ---------------------------------------------
    base["final_score"] = (
        WEIGHT_TECH * base["tech_score"]
        + WEIGHT_PRED * base["pred_score"]
        + WEIGHT_PROB * base["prob_score"]
        + WEIGHT_QUAL * base["qual_score"]
        + WEIGHT_SAFETY * base["safety_score"]
        + WEIGHT_LIQUIDITY * base["liquidity_score"]
    )


    # ---------------------------------------------
    # 5. 리스크(예측 MDD) 기반 감점 적용
    #    pred_mdd_60d가 클수록(낙폭이 깊을수록) final_score를 깎음
    # ---------------------------------------------
    if "pred_mdd_60d" in base.columns:
        # pred_mdd_60d: 음수(예: -0.25 = -25% 최대 낙폭 예상)
        dd = pd.to_numeric(base["pred_mdd_60d"], errors="coerce")

        # threshold(예: 0.15 = -15%)까지는 감점 없음,
        # 그 아래부터 penalty_raw 증가
        #   penalty_raw = max(0, -dd - RISK_MDD_THRESHOLD)
        penalty_raw = (-dd) - RISK_MDD_THRESHOLD
        penalty_raw = penalty_raw.clip(lower=0)  # 음수는 0으로

        # 스케일(예: 100 * 0.3 = 30점 감점 등) 곱해서 최종 감점값 계산
        base["risk_penalty"] = penalty_raw * RISK_PENALTY_SCALE

        # final_score에서 감점 적용
        base["final_score"] = base["final_score"] - base["risk_penalty"]
    else:
        # pred_mdd_60d가 없으면 감점 없이 0
        base["risk_penalty"] = 0.0

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

    # DB upsert
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
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
                PRIMARY KEY (date, code)
            );
            """
        )
        records = df_out.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_ranking
            (date, code, close, pred_return_60d, pred_return_90d, pred_mdd_60d, pred_mdd_90d,
             prob_top20_60d, prob_top20_90d, score, score_score, composite, quality_score,
             name, market, sector, tech_score, pred_score, ret_score, prob_score, qual_score,
             safety_score, liquidity_score, final_score, risk_penalty, market_up,
             market_status_date, market_kospi_close, market_kospi_ma20, market_vol_5d, market_foreign_5d,
             generated_at, model_version)
            VALUES (:date, :code, :close, :pred_return_60d, :pred_return_90d, :pred_mdd_60d, :pred_mdd_90d,
                    :prob_top20_60d, :prob_top20_90d, :score, :score_score, :composite, :quality_score,
                    :name, :market, :sector, :tech_score, :pred_score, :ret_score, :prob_score, :qual_score,
                    :safety_score, :liquidity_score, :final_score, :risk_penalty, :market_up,
                    :market_status_date, :market_kospi_close, :market_kospi_ma20, :market_vol_5d, :market_foreign_5d,
                    :generated_at, :model_version)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved ranking to DB: %s (rows=%d)", DB_PATH.resolve(), len(df_out))
    except Exception:
        logging.exception("Failed to save ranking to DB")
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
