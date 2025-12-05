import logging
import pickle
import json          # ⬅ 이거 추가
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor, LGBMClassifier

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
MODEL_PKL = DATA_DIR / "model.pkl"

# (추가) 튜닝된 LightGBM 파라미터 JSON 경로
MODELS_DIR = Path("models")
LGBM_REG_PARAMS_JSON = MODELS_DIR / "lgbm_reg_params.json"

# 회귀 타깃: 로그 수익률 + MDD (옵션 B)
REG_TARGETS = [
    "target_log_60d",
    "target_log_90d",
    "target_mdd_60d",
    "target_mdd_90d",
]

# 분류 타깃: Top20 여부 (기존 구조 유지)
CLS_TARGETS = [
    "target_60d_top20",
    "target_90d_top20",
]

N_SPLITS = 3  # TimeSeriesSplit fold 수 (너무 크지 않게)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def load_tuned_reg_params() -> Dict[str, float]:
    """
    model_train_optuna.py가 저장한 JSON이 있으면 읽어서 반환.
    없거나 실패하면 빈 dict 반환(기본 파라미터 사용).
    """
    try:
        if not LGBM_REG_PARAMS_JSON.exists():
            return {}
        data = json.loads(LGBM_REG_PARAMS_JSON.read_text(encoding="utf-8"))
        params = data.get("params", {})
        if not isinstance(params, dict):
            return {}
        logging.info("Loaded tuned LGBM reg params from %s", LGBM_REG_PARAMS_JSON)
        return params
    except Exception as e:
        logging.warning("Failed to load tuned reg params: %s", e)
        return {}


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"features.csv not found at {FEATURES_CSV.resolve()}")
    df = pd.read_csv(FEATURES_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must contain 'date' and 'code' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    logging.info("Loaded features.csv: %s (rows=%d)", FEATURES_CSV, len(df))
    return df


def load_labels() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV.resolve()}")
    df = pd.read_csv(LABELS_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("labels.csv must contain 'date' and 'code' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    logging.info("Loaded labels.csv: %s (rows=%d)", LABELS_CSV, len(df))
    return df


def make_merged() -> pd.DataFrame:
    feats = load_features()
    labels = load_labels()

    merged = pd.merge(
        feats,
        labels,
        on=["date", "code"],
        how="inner",
        suffixes=("", "_y"),
    )
    logging.info("Merged features + labels shape: %s", merged.shape)

    # feature 컬럼 선정: date, code, target 계열, *_top20 제외
    exclude_cols = {"date", "code"}
    exclude_cols.update(REG_TARGETS)
    exclude_cols.update(CLS_TARGETS)

    feature_cols = [
        c
        for c in merged.columns
        if c not in exclude_cols
        and not c.endswith("_top20")
        and not c.startswith("target_")
        and not c.startswith("realized_return_")  # exclude label-derived realized returns
    ]

    logging.info("Using %d feature columns: %s", len(feature_cols), feature_cols)
    return merged, feature_cols


def time_series_folds(dates: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    dates 배열(중복 허용)에 대해 date 레벨에서 TimeSeriesSplit 수행.
    """
    uniq = np.array(sorted(pd.Series(dates).dropna().unique()))
    if uniq.size < 10:
        # 너무 적으면 CV 생략
        return []
    eff = min(n_splits, max(2, uniq.size - 1))
    tscv = TimeSeriesSplit(n_splits=eff)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in tscv.split(uniq):
        tr_dates = uniq[tr_idx]
        va_dates = uniq[va_idx]
        folds.append((tr_dates, va_dates))
    return folds


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ---------------------------------------------------------------------
# Train regression models (log-return + MDD)
# ---------------------------------------------------------------------


def train_regressors(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, LGBMRegressor]:
    reg_models: Dict[str, LGBMRegressor] = {}

    for target in REG_TARGETS:
        if target not in df.columns:
            logging.warning("Regression target %s not found in merged data; skipping.", target)
            continue

        df_t = df[df[target].notna()].copy()
        if df_t.empty:
            logging.warning("No rows for regression target %s; skipping.", target)
            continue

        X = df_t[feature_cols]
        y = df_t[target].astype(float)
        dates = df_t["date"].values

        logging.info("Training regressor for %s (rows=%d)", target, len(df_t))

        # 기본 파라미터 (과적합 방지용으로 비교적 보수적)
        # --- 기본 파라미터 (지금 쓰던 값) ---
        base_params = dict(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=42,
            n_jobs=-1,
        )

        # --- Optuna 튜닝 결과가 있으면 덮어쓰기 ---
        tuned = load_tuned_reg_params()
        if tuned:
            base_params.update(tuned)

        reg = LGBMRegressor(**base_params)

        folds = time_series_folds(dates, N_SPLITS)
        if not folds:
            # CV 생략, 전체로 학습
            reg.fit(X, y)
            reg_models[target] = reg
            logging.info("  [%s] trained on full data (no CV).", target)
        else:
            # 간단한 time-series CV (조기 종료는 사용 X; 평균 성능 모니터)
            rmses: List[float] = []
            maes: List[float] = []
            for i, (tr_dates, va_dates) in enumerate(folds, start=1):
                tr_mask = np.isin(dates, tr_dates)
                va_mask = np.isin(dates, va_dates)
                X_tr, y_tr = X[tr_mask], y[tr_mask]
                X_va, y_va = X[va_mask], y[va_mask]
                if len(X_va) == 0 or len(X_tr) == 0:
                    continue

                #reg_i = reg.__class__(**reg.get_params())
                reg_i = LGBMRegressor(**reg.get_params())
                reg_i.fit(X_tr, y_tr)
                pred_va = reg_i.predict(X_va)
                fold_rmse = rmse(y_va, pred_va)
                fold_mae = mae(y_va, pred_va)
                rmses.append(fold_rmse)
                maes.append(fold_mae)
                logging.info(
                    "  [%s][fold %d/%d] RMSE=%.4f MAE=%.4f (n_tr=%d, n_va=%d)",
                    target,
                    i,
                    len(folds),
                    fold_rmse,
                    fold_mae,
                    len(X_tr),
                    len(X_va),
                )

            # 전체 데이터로 최종 재학습
            reg.fit(X, y)
            reg_models[target] = reg
            if rmses:
                logging.info(
                    "  [%s] CV RMSE=%.4f±%.4f, MAE=%.4f±%.4f",
                    target,
                    float(np.mean(rmses)),
                    float(np.std(rmses)),
                    float(np.mean(maes)),
                    float(np.std(maes)),
                )

    return reg_models


# ---------------------------------------------------------------------
# Train classification models (Top20 여부)
# ---------------------------------------------------------------------


def train_classifiers(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, LGBMClassifier]:
    cls_models: Dict[str, LGBMClassifier] = {}

    for target in CLS_TARGETS:
        if target not in df.columns:
            logging.warning("Classification target %s not found; skipping.", target)
            continue

        df_t = df[df[target].notna()].copy()
        if df_t.empty:
            logging.warning("No rows for classification target %s; skipping.", target)
            continue

        X = df_t[feature_cols]
        y = df_t[target].astype(int)
        dates = df_t["date"].values

        # sanity check: 양성 비율이 너무 낮으면 경고
        pos_ratio = y.mean()
        logging.info(
            "Training classifier for %s (rows=%d, positive_rate=%.3f)",
            target,
            len(df_t),
            float(pos_ratio),
        )
        if pos_ratio < 0.01:
            logging.warning("  [%s] positive rate is very low; classifier may be unstable.", target)

        cls = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            n_jobs=-1,
        )

        folds = time_series_folds(dates, N_SPLITS)
        if not folds:
            cls.fit(X, y)
            cls_models[target] = cls
            logging.info("  [%s] classifier trained on full data (no CV).", target)
        else:
            aucs: List[float] = []
            losses: List[float] = []
            for i, (tr_dates, va_dates) in enumerate(folds, start=1):
                tr_mask = np.isin(dates, tr_dates)
                va_mask = np.isin(dates, va_dates)
                X_tr, y_tr = X[tr_mask], y[tr_mask]
                X_va, y_va = X[va_mask], y[va_mask]
                if len(X_va) == 0 or len(X_tr) == 0:
                    continue

                cls_i = cls.__class__(**cls.get_params())
                cls_i.fit(X_tr, y_tr)
                proba_va = cls_i.predict_proba(X_va)[:, 1]
                auc = roc_auc_score(y_va, proba_va)
                loss = log_loss(y_va, proba_va, labels=[0, 1])
                aucs.append(float(auc))
                losses.append(float(loss))
                logging.info(
                    "  [%s][fold %d/%d] AUC=%.4f logloss=%.4f (n_tr=%d, n_va=%d)",
                    target,
                    i,
                    len(folds),
                    auc,
                    loss,
                    len(X_tr),
                    len(X_va),
                )

            cls.fit(X, y)
            cls_models[target] = cls
            if aucs:
                logging.info(
                    "  [%s] CV AUC=%.4f±%.4f, logloss=%.4f±%.4f",
                    target,
                    float(np.mean(aucs)),
                    float(np.std(aucs)),
                    float(np.mean(losses)),
                    float(np.std(losses)),
                )

    return cls_models


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    setup_logging()
    df, feature_cols = make_merged()

    logging.info("Start training regressors (log-return + MDD)...")
    reg_models = train_regressors(df, feature_cols)

    logging.info("Start training classifiers (Top20 flags)...")
    cls_models = train_classifiers(df, feature_cols)

    pack = {
        "features": feature_cols,
        "reg_models": reg_models,
        "cls_models": cls_models,
        "reg_targets": list(reg_models.keys()),
        "cls_targets": list(cls_models.keys()),
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pack, f)

    logging.info(
        "Saved model package to %s (reg_targets=%s, cls_targets=%s)",
        MODEL_PKL.resolve(),
        list(reg_models.keys()),
        list(cls_models.keys()),
    )


if __name__ == "__main__":
    main()
