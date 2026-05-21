import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import sqlite3
try:
    from db import (
        copy_df,
        ensure_unique_keys,
        get_engine,
        raw_psycopg2_conn,
        replace_table_rows_pg,
        replace_table_rows_sqlite,
        use_sqlite_fallback_writes,
    )
except Exception:
    get_engine = None
    copy_df = None
    ensure_unique_keys = None
    raw_psycopg2_conn = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None
    def use_sqlite_fallback_writes() -> bool:
        return False

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
SCORES_CSV = DATA_DIR / "scores_final.csv"
MODEL_PKL = DATA_DIR / "model.pkl"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
PREDICTIONS_DB_COLUMNS = [
    "date",
    "code",
    "pred_return_30d",
    "pred_return_60d",
    "pred_return_90d",
    "pred_mdd_30d",
    "pred_mdd_60d",
    "pred_mdd_90d",
    "prob_top20_60d",
    "prob_top20_90d",
    "score",
]
PREDICTIONS_PK = ["date", "code"]

# 회귀 타깃 이름과 horizon 매핑 (log-return / MDD)
REG_LOG_TARGETS = {
    "target_log_30d": ("pred_return_30d", 30),
    "target_log_60d": ("pred_return_60d", 60),
    "target_log_90d": ("pred_return_90d", 90),
}
REG_MDD_TARGETS = {
    "target_mdd_30d": "pred_mdd_30d",
    "target_mdd_60d": "pred_mdd_60d",
    "target_mdd_90d": "pred_mdd_90d",
}

# 분류 타깃(Top20) -> 확률 컬럼명 매핑
# prob_top20_90d 는 점수에 반영되지 않으므로 학습/예측 대상에서 제외 (저장은 유지)
CLS_TARGET_PROB_COL = {
    "target_60d_top20": "prob_top20_60d",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_model() -> Dict[str, Any]:
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PKL.resolve()}")
    with open(MODEL_PKL, "rb") as f:
        pack = pickle.load(f)

    if not isinstance(pack, dict):
        raise ValueError("model.pkl should contain a dict pack")

    features = pack.get("features")
    reg_models = pack.get("reg_models", {})
    cls_models = pack.get("cls_models", {})
    reg_targets = pack.get("reg_targets", list(reg_models.keys()))
    cls_targets = pack.get("cls_targets", list(cls_models.keys()))

    if not features or not isinstance(features, (list, tuple)):
        raise ValueError("model pack must contain 'features' list")

    logging.info(
        "Loaded model pack: %d reg targets, %d cls targets, %d features",
        len(reg_targets),
        len(cls_targets),
        len(features),
    )

    return {
        "features": list(features),
        "reg_models": reg_models,
        "cls_models": cls_models,
        "reg_targets": list(reg_targets),
        "cls_targets": list(cls_targets),
    }


def load_features_latest(feature_cols) -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"features.csv not found at {FEATURES_CSV.resolve()}")
    df = pd.read_csv(FEATURES_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must contain 'date' and 'code' columns.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 종목별 최신 row만 사용 (예측 기준 시점)
    latest = df.groupby("code", as_index=False).tail(1).copy()
    latest = latest.reset_index(drop=True)

    # 운영 예측은 latest snapshot 단일 기준일만 사용한다. 일부 종목의 최신
    # feature 행이 과거 날짜에 멈춰 있으면 stale prediction이 ranking까지
    # 전파되므로 최신 전체 기준일보다 오래된 종목은 제외한다.
    latest_date = latest["date"].max()
    stale_mask = latest["date"].lt(latest_date)
    stale_count = int(stale_mask.sum())
    if stale_count > 0:
        stale_codes = latest.loc[stale_mask, "code"].astype(str).head(10).tolist()
        logging.warning(
            "Dropping stale feature rows before prediction: latest_date=%s stale_rows=%d sample_codes=%s",
            latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "NA",
            stale_count,
            ",".join(stale_codes) if stale_codes else "NA",
        )
        latest = latest.loc[~stale_mask].copy()
        latest = latest.reset_index(drop=True)

    # 필요한 feature 컬럼만 유지
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing feature columns in features.csv: {missing}")

    logging.info(
        "Using latest features per code on latest snapshot date=%s: %d rows",
        latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "NA",
        len(latest),
    )
    return latest


def predict_all(model_pack: Dict[str, Any], feats_latest: pd.DataFrame) -> pd.DataFrame:
    feature_cols = model_pack["features"]
    reg_models: Dict[str, Any] = model_pack["reg_models"]
    cls_models: Dict[str, Any] = model_pack["cls_models"]
    reg_targets = model_pack["reg_targets"]
    cls_targets = model_pack["cls_targets"]

    X = feats_latest[feature_cols].copy()
    codes = feats_latest["code"].astype(str).values
    dates = feats_latest["date"].dt.strftime("%Y-%m-%d").values

    out = pd.DataFrame({"date": dates, "code": codes})

    # 회귀: log-return + MDD
    for target in reg_targets:
        model = reg_models.get(target)
        if model is None:
            continue
        logging.info("Predicting regression target: %s", target)
        pred = model.predict(X)

        # log-return -> 일반 수익률로 변환
        if target in REG_LOG_TARGETS:
            col_name, _h = REG_LOG_TARGETS[target]
            # exp(log_r) - 1 with safety
            pred_ret = np.exp(np.clip(pred, -5.0, 5.0)) - 1.0
            out[col_name] = pred_ret
        elif target in REG_MDD_TARGETS:
            col_name = REG_MDD_TARGETS[target]
            out[col_name] = pred
        else:
            # 기타 회귀 타깃은 그대로 저장 (디버깅용)
            out[target] = pred

    # 분류: Top20 확률
    for target in cls_targets:
        model = cls_models.get(target)
        if model is None:
            continue
        col_name = CLS_TARGET_PROB_COL.get(target)
        if not col_name:
            continue

        logging.info("Predicting classifier target: %s", target)
        proba = model.predict_proba(X)[:, 1]
        out[col_name] = proba

    return out


def _ensure_predictions_pg_columns() -> None:
    """Postgres predictions 테이블에 누락된 컬럼을 DOUBLE PRECISION으로 자동 추가."""
    if raw_psycopg2_conn is None:
        return
    numeric_cols = [c for c in PREDICTIONS_DB_COLUMNS if c not in ("date", "code")]
    try:
        conn = raw_psycopg2_conn()
        try:
            with conn, conn.cursor() as cur:
                for col in numeric_cols:
                    cur.execute(
                        f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"
                    )
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        logging.warning("Could not ensure predictions pg columns (table may not exist yet)")


def save_predictions(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in PREDICTIONS_DB_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PREDICTIONS_DB_COLUMNS]
    if ensure_unique_keys:
        ensure_unique_keys(out, PREDICTIONS_PK, "predictions")

    out.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    logging.info("Saved predictions: %s (rows=%d)", PREDICTIONS_CSV.resolve(), len(out))

    # Save to DB while preserving schema and indexes.
    try:
        if replace_table_rows_pg:
            try:
                _ensure_predictions_pg_columns()
                replace_table_rows_pg("predictions", out, columns=PREDICTIONS_DB_COLUMNS)
                logging.info("Replaced predictions rows in Postgres (rows=%d)", len(out))
                return
            except Exception:
                logging.exception("Postgres row replace failed, fallback to sqlite")
    except Exception:
        logging.exception("Postgres save failed, fallback to sqlite")

    if not use_sqlite_fallback_writes():
        logging.info("Skipping sqlite fallback for predictions (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                date             DATE NOT NULL,
                code             TEXT NOT NULL,
                pred_return_30d  REAL,
                pred_return_60d  REAL,
                pred_return_90d  REAL,
                pred_mdd_30d     REAL,
                pred_mdd_60d     REAL,
                pred_mdd_90d     REAL,
                prob_top20_60d   REAL,
                prob_top20_90d   REAL,
                score            REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "predictions", out)
        conn.commit()
        logging.info("Saved predictions to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(out))
    except Exception:
        logging.exception("Failed to save predictions to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    logging.info("Loading model...")
    pack = load_model()

    logging.info("Loading latest features...")
    feats_latest = load_features_latest(pack["features"])

    logging.info("Predicting...")
    preds = predict_all(pack, feats_latest)

    save_predictions(preds)


if __name__ == "__main__":
    main()
