import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import sqlite3
try:
    from db import get_engine, copy_df
except Exception:
    get_engine = None
    copy_df = None

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
SCORES_CSV = DATA_DIR / "scores_final.csv"
MODEL_PKL = DATA_DIR / "model.pkl"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
DB_PATH = DATA_DIR / "lee_trader.db"

# 회귀 타깃 이름과 horizon 매핑 (log-return / MDD)
REG_LOG_TARGETS = {
    "target_log_60d": ("pred_return_60d", 60),
    "target_log_90d": ("pred_return_90d", 90),
}
REG_MDD_TARGETS = {
    "target_mdd_60d": "pred_mdd_60d",
    "target_mdd_90d": "pred_mdd_90d",
}

# 분류 타깃(Top20) -> 확률 컬럼명 매핑
CLS_TARGET_PROB_COL = {
    "target_60d_top20": "prob_top20_60d",
    "target_90d_top20": "prob_top20_90d",
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

    # 필요한 feature 컬럼만 유지
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing feature columns in features.csv: {missing}")

    logging.info("Using latest features per code: %d rows", len(latest))
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

    # /api/stocks 에서 사용하는 score 컬럼 (간단히 60d 예측 수익률 기반)
    if "pred_return_60d" in out.columns:
        out["score"] = out["pred_return_60d"] * 100.0

    return out


def save_predictions(df: pd.DataFrame) -> None:
    df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    logging.info("Saved predictions: %s (rows=%d)", PREDICTIONS_CSV.resolve(), len(df))

    # Save to DB (prefer Postgres via SQLAlchemy)
    try:
        if copy_df:
            try:
                copy_df("predictions", df, columns=list(df.columns), truncate=True)
                logging.info("Saved predictions to Postgres via copy_expert (rows=%d)", len(df))
                return
            except Exception:
                logging.exception("copy_expert failed, trying SQLAlchemy")
        if get_engine:
            eng = get_engine()
            df.to_sql("predictions", eng, if_exists="replace", index=False, method="multi")
            logging.info("Saved predictions to Postgres via SQLAlchemy (rows=%d)", len(df))
            return
    except Exception:
        logging.exception("SQLAlchemy save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        df.to_sql("predictions", conn, if_exists="replace", index=False)
        conn.commit()
        logging.info("Saved predictions to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(df))
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
