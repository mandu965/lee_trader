# python/model_train_optuna.py

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor


# 경로 설정
DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_JSON = MODELS_DIR / "lgbm_reg_params.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_training_data(target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    features.csv + labels.csv에서 target_col 기준으로 학습용 데이터 생성.
    - features: date, code, (나머지 numeric 컬럼들 = feature)
    - labels: date, code, target_col
    """
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"{FEATURES_CSV} not found")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"{LABELS_CSV} not found")

    feat = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)

    # feature 컬럼: date/code 제외한 숫자형 컬럼만 사용
    exclude = {"date", "code"}
    feature_cols: List[str] = []
    for c in feat.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(feat[c]):
            feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in features.csv")

    logging.info("Selected %d feature columns: %s", len(feature_cols), feature_cols)

    # date, code 기준으로 라벨 join
    if target_col not in labels.columns:
        raise RuntimeError(f"{target_col} not found in labels.csv")

    df = pd.merge(
        feat,
        labels[["date", "code", target_col]].copy(),
        on=["date", "code"],
        how="inner",
        validate="one_to_one",
    )

    # feature + target NaN 제거
    df = df.dropna(subset=feature_cols + [target_col]).copy()

    logging.info(
        "Training data for %s: rows=%d, features=%d",
        target_col,
        len(df),
        len(feature_cols),
    )
    return df, feature_cols


def make_time_splits(dates: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    date(문자열) 배열을 받아서, 시간 순서를 유지하는 CV split 생성.
    - unique한 날짜를 정렬해서 앞부분은 train, 뒷부분은 val로 사용.
    """
    uniq = np.unique(dates)
    uniq_sorted = np.sort(uniq)
    n = len(uniq_sorted)
    if n < n_splits + 1:
        # 날짜가 너무 적으면 한두 개만 split
        n_splits = max(1, n - 1)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_size = n // (n_splits + 1) if (n_splits + 1) > 0 else 1
    fold_size = max(fold_size, 1)

    for i in range(1, n_splits + 1):
        val_start = i * fold_size
        if val_start >= n:
            break
        val_end = (i + 1) * fold_size if i < n_splits else n

        train_dates = uniq_sorted[:val_start]
        val_dates = uniq_sorted[val_start:val_end]
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue
        splits.append((train_dates, val_dates))

    return splits


def make_param_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    LightGBM 회귀용 하이퍼파라미터 탐색 공간 정의.
    """
    params: Dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    # 고정 옵션
    params.update(
        dict(
            objective="regression",
            random_state=42,
            n_jobs=-1,
        )
    )
    return params


def objective_factory(train_df: pd.DataFrame, feature_cols: List[str], target_col: str):
    """
    Optuna에서 쓸 objective 함수 생성.
    - 타임 시리즈 CV로 각 fold의 RMSE 평균을 최소화.
    """
    dates = train_df["date"].values
    splits = make_time_splits(dates, n_splits=5)
    if not splits:
        raise RuntimeError("Not enough unique dates to build CV splits.")

    def objective(trial: optuna.Trial) -> float:
        params = make_param_space(trial)
        rmse_list: List[float] = []

        for fold_idx, (tr_dates, va_dates) in enumerate(splits, start=1):
            tr = train_df[train_df["date"].isin(tr_dates)]
            va = train_df[train_df["date"].isin(va_dates)]

            tr = tr.dropna(subset=feature_cols + [target_col])
            va = va.dropna(subset=feature_cols + [target_col])

            if tr.empty or va.empty:
                continue

            X_tr = tr[feature_cols].values
            y_tr = tr[target_col].values
            X_va = va[feature_cols].values
            y_va = va[target_col].values

            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)

            pred = model.predict(X_va)
            rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
            rmse_list.append(rmse)

        if not rmse_list:
            raise optuna.TrialPruned("No valid folds in this trial.")

        score = float(np.mean(rmse_list))
        trial.set_user_attr("rmse_per_fold", rmse_list)
        return score

    return objective


def save_best_params(best_params: Dict[str, Any], target_col: str) -> None:
    payload = {
        "target": target_col,
        "params": best_params,
    }
    PARAMS_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float),
        encoding="utf-8",
    )
    logging.info("Saved best params to %s", PARAMS_JSON.resolve())


def main() -> None:
    setup_logging()
    target_col = "target_60d"  # 주력 튜닝 타깃

    logging.info("Loading training data...")
    train_df, feature_cols = load_training_data(target_col)

    logging.info("Starting Optuna study for %s", target_col)
    study = optuna.create_study(
        direction="minimize",
        study_name=f"lgbm_reg_{target_col}",
    )
    objective = objective_factory(train_df, feature_cols, target_col)
    study.optimize(objective, n_trials=40, show_progress_bar=False)

    logging.info("Best RMSE: %.6f", study.best_value)
    logging.info("Best params: %s", study.best_params)

    save_best_params(study.best_params, target_col)


if __name__ == "__main__":
    main()
