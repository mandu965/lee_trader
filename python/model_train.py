import argparse
import logging
import pickle
import os
from datetime import datetime
import json  # 튜닝 파라미터 로딩용
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from calibrated_classifier import BinaryIsotonicCalibratedClassifier
except Exception:  # pragma: no cover - package import path
    from .calibrated_classifier import BinaryIsotonicCalibratedClassifier
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older scikit-learn
    FrozenEstimator = None

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
MODEL_PKL = DATA_DIR / "model.pkl"
MODEL_FEATURE_IMPORTANCE_DIR = DATA_DIR / "model_feature_importance"

# (추가) 튜닝된 LightGBM 파라미터 JSON 경로
MODELS_DIR = Path("models")
LGBM_REG_PARAMS_JSON = MODELS_DIR / "lgbm_reg_params.json"

# 회귀 horizon: log-return + MDD 예측
# 30d: AI_MAX_HOLDING_DAYS 정합을 위해 추가 (target_log_30d / target_mdd_30d)
DEFAULT_HORIZONS = [30, 60, 90]

# 분류 horizon: Top20 확률 예측
# prob_top20_90d 는 final_score.py 에서 점수에 반영되지 않으므로 60d만 학습
DEFAULT_CLS_HORIZONS = [60]

N_SPLITS = 3  # TimeSeriesSplit fold 수 (너무 크지 않게)
DEFAULT_SAMPLE_WEIGHT_HALFLIFE_YEARS = 3.0  # 3년 반감기: 최근 데이터에 더 높은 가중치


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def compute_sample_weights(dates: np.ndarray, halflife_years: float) -> np.ndarray | None:
    """지수 감쇠 샘플 가중치. halflife_years <= 0 이면 None 반환(비활성).

    오래된 데이터일수록 낮은 가중치를 부여해 최근 시장 환경에 더 민감하게 학습한다.
    가중치는 평균=1로 정규화되어 유효 샘플 수 해석이 직관적이다.
    """
    if halflife_years <= 0:
        return None
    dts = pd.to_datetime(dates)
    days_ago = (dts.max() - dts).days.to_numpy().astype(float)
    decay_rate = np.log(2.0) / (halflife_years * 365.0)
    weights = np.exp(-decay_rate * days_ago)
    return weights / weights.mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM models and package into model.pkl")
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Regression horizons for log-return + MDD (e.g., 60 90). Defaults to 60 90.",
    )
    p.add_argument(
        "--cls-horizons",
        type=int,
        nargs="+",
        default=None,
        dest="cls_horizons",
        help=(
            "Classification horizons for Top20 classifiers. "
            "Defaults to DEFAULT_CLS_HORIZONS=[60]. "
            "Pass '60 90' to restore both classifiers."
        ),
    )
    p.add_argument(
        "--output-pkl",
        type=Path,
        default=MODEL_PKL,
        help="Output model package path. Defaults to data/model.pkl",
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=FEATURES_CSV,
        help="Features CSV path. Defaults to data/features.csv",
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=LABELS_CSV,
        help="Labels CSV path. Defaults to data/labels.csv",
    )
    p.add_argument(
        "--train-end-date",
        type=str,
        help="Use only rows with date <= train_end_date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--model-version",
        type=str,
        default=os.environ.get("MODEL_VERSION", "v1"),
        help="Model version metadata stored in the model pack.",
    )
    p.add_argument(
        "--sample-weight-halflife",
        type=float,
        default=DEFAULT_SAMPLE_WEIGHT_HALFLIFE_YEARS,
        dest="sample_weight_halflife",
        help=(
            "Exponential decay halflife in years for sample weights. "
            "0 disables weighting. Default=%(default).1f."
        ),
    )
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        dest="no_calibrate",
        help="Disable isotonic calibration for classifiers (enabled by default).",
    )
    return p.parse_args()


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


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found at {path.resolve()}")
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must contain 'date' and 'code' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    logging.info("Loaded features.csv: %s (rows=%d)", path, len(df))
    return df


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"labels.csv not found at {path.resolve()}")
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("labels.csv must contain 'date' and 'code' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    logging.info("Loaded labels.csv: %s (rows=%d)", path, len(df))
    return df


def build_targets(reg_horizons: List[int], cls_horizons: List[int] | None = None) -> Tuple[List[str], List[str]]:
    """
    reg_horizons: horizons for log-return + MDD regressors (e.g., [60, 90])
    cls_horizons: horizons for Top20 classifiers (e.g., [60])
                  defaults to reg_horizons when None
    returns: (reg_targets, cls_targets)
    """
    if cls_horizons is None:
        cls_horizons = reg_horizons
    reg_targets = []
    cls_targets = []
    for h in sorted({int(h) for h in reg_horizons}):
        reg_targets.append(f"target_log_{h}d")
        reg_targets.append(f"target_mdd_{h}d")
    for h in sorted({int(h) for h in cls_horizons}):
        cls_targets.append(f"target_{h}d_top20")
    return reg_targets, cls_targets


def make_merged(reg_targets: List[str], cls_targets: List[str], features_path: Path, labels_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    feats = load_features(features_path)
    labels = load_labels(labels_path)

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
    exclude_cols.update(reg_targets)
    exclude_cols.update(cls_targets)

    feature_cols = [
        c
        for c in merged.columns
        if c not in exclude_cols
        and not c.endswith("_top20")
        and not c.startswith("target_")
        and not c.startswith("realized_return_")  # exclude label-derived realized returns
        and pd.api.types.is_numeric_dtype(merged[c])  # LightGBM: int/float/bool only
    ]
    non_numeric = [c for c in merged.columns if c not in exclude_cols and not c.endswith("_top20") and not c.startswith("target_") and not c.startswith("realized_return_") and not pd.api.types.is_numeric_dtype(merged[c])]
    if non_numeric:
        logging.info("Excluded non-numeric feature columns: %s", non_numeric)

    logging.info("Using %d feature columns: %s", len(feature_cols), feature_cols)
    return merged, feature_cols


def apply_train_end_date(df: pd.DataFrame, train_end_date: str | None) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if not train_end_date:
        return df, None
    cutoff = pd.to_datetime(train_end_date).normalize()
    filtered = df[df["date"] <= cutoff].copy()
    logging.info(
        "Applied train_end_date=%s -> rows=%d (from %d)",
        cutoff.date(),
        len(filtered),
        len(df),
    )
    if filtered.empty:
        raise ValueError(f"No training rows remain after applying train_end_date={cutoff.date()}")
    return filtered, cutoff


def time_series_folds(dates: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    dates 배열(중복 허용)에 대해 date 레벨에서 TimeSeriesSplit 수행.
    """
    uniq = np.array(
        sorted(pd.Series(pd.to_datetime(dates)).dropna().unique()),
        dtype="datetime64[ns]",
    )
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


def _unwrap_feature_importance_model(model: object) -> object:
    estimator = getattr(model, "estimator", None)
    if estimator is not None:
        return estimator
    base_estimator = getattr(model, "base_estimator", None)
    if base_estimator is not None:
        return base_estimator
    return model


def _build_feature_importance_frame(
    model: object,
    feature_cols: List[str],
    *,
    target: str,
    model_group: str,
    model_version: str,
    trained_at: str,
    train_end_date: str | None,
) -> pd.DataFrame:
    base_model = _unwrap_feature_importance_model(model)
    split_values = getattr(base_model, "feature_importances_", None)
    booster = getattr(base_model, "booster_", None)
    if split_values is None and booster is None:
        return pd.DataFrame()

    if split_values is None:
        split_values = booster.feature_importance(importance_type="split")
    gain_values = None
    if booster is not None:
        gain_values = booster.feature_importance(importance_type="gain")

    split_arr = np.asarray(split_values, dtype=float)
    if split_arr.shape[0] != len(feature_cols):
        logging.warning(
            "Skipping feature importance export for %s: feature length mismatch (%d != %d)",
            target,
            split_arr.shape[0],
            len(feature_cols),
        )
        return pd.DataFrame()
    gain_arr = np.asarray(gain_values, dtype=float) if gain_values is not None else np.full(len(feature_cols), np.nan)

    frame = pd.DataFrame(
        {
            "target": target,
            "model_group": model_group,
            "feature": feature_cols,
            "importance_split": split_arr,
            "importance_gain": gain_arr,
            "model_version": model_version,
            "trained_at": trained_at,
            "train_end_date": train_end_date or "",
        }
    )
    split_total = float(frame["importance_split"].sum())
    gain_non_null = frame["importance_gain"].dropna()
    gain_total = float(gain_non_null.sum()) if not gain_non_null.empty else 0.0
    frame["importance_split_pct"] = frame["importance_split"] / split_total if split_total > 0 else 0.0
    frame["importance_gain_pct"] = frame["importance_gain"] / gain_total if gain_total > 0 else np.nan
    frame = frame.sort_values(["importance_gain", "importance_split"], ascending=False, na_position="last").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def export_feature_importance_reports(
    *,
    reg_models: Dict[str, object],
    cls_models: Dict[str, object],
    feature_cols: List[str],
    output_dir: Path,
    model_version: str,
    trained_at: str,
    train_end_date: str | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for model_group, models in (("regression", reg_models), ("classification", cls_models)):
        for target, model in models.items():
            frame = _build_feature_importance_frame(
                model,
                feature_cols,
                target=target,
                model_group=model_group,
                model_version=model_version,
                trained_at=trained_at,
                train_end_date=train_end_date,
            )
            if frame.empty:
                continue
            path = output_dir / f"{target}_feature_importance.csv"
            frame.to_csv(path, index=False, encoding="utf-8")
            written_paths.append(path)
            frames.append(frame)

    if not frames:
        logging.warning("No feature importance reports were generated.")
        return written_paths

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "feature_importance_all_targets.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")
    written_paths.append(combined_path)

    summary = (
        combined.assign(
            split_pct_filled=combined["importance_split_pct"].fillna(0.0),
            gain_pct_filled=combined["importance_gain_pct"].fillna(0.0),
        )
        .groupby("feature", as_index=False)
        .agg(
            target_count=("target", "nunique"),
            mean_split_pct=("split_pct_filled", "mean"),
            mean_gain_pct=("gain_pct_filled", "mean"),
            total_split=("importance_split", "sum"),
            total_gain=("importance_gain", "sum"),
        )
    )
    summary["composite_score"] = (summary["mean_split_pct"] + summary["mean_gain_pct"]) / 2.0
    summary = summary.sort_values(["composite_score", "target_count", "total_gain"], ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary_path = output_dir / "feature_importance_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    written_paths.append(summary_path)

    logging.info("Saved %d feature importance report files under %s", len(written_paths), output_dir.resolve())
    return written_paths


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ---------------------------------------------------------------------
# Train regression models (log-return + MDD)
# ---------------------------------------------------------------------


def train_regressors(
    df: pd.DataFrame,
    feature_cols: List[str],
    reg_targets: List[str],
    *,
    halflife_years: float = 0.0,
) -> Dict[str, LGBMRegressor]:
    reg_models: Dict[str, LGBMRegressor] = {}

    for target in reg_targets:
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
        sample_weight = compute_sample_weights(dates, halflife_years)
        if sample_weight is not None:
            logging.info(
                "  [%s] sample_weight enabled (halflife_years=%.1f, min=%.3f, max=%.3f)",
                target, halflife_years, float(sample_weight.min()), float(sample_weight.max()),
            )

        logging.info("Training regressor for %s (rows=%d)", target, len(df_t))

        # 기본 파라미터 (과적합 방지용으로 비교적 보수적)
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

        # Optuna 튜닝 결과가 있으면 덮어쓰기
        tuned = load_tuned_reg_params()
        if tuned:
            base_params.update(tuned)

        reg = LGBMRegressor(**base_params)

        folds = time_series_folds(dates, N_SPLITS)
        if not folds:
            reg.fit(X, y, sample_weight=sample_weight)
            reg_models[target] = reg
            logging.info("  [%s] trained on full data (no CV).", target)
        else:
            rmses: List[float] = []
            maes: List[float] = []
            for i, (tr_dates, va_dates) in enumerate(folds, start=1):
                tr_mask = np.isin(dates, tr_dates)
                va_mask = np.isin(dates, va_dates)
                X_tr, y_tr = X[tr_mask], y[tr_mask]
                X_va, y_va = X[va_mask], y[va_mask]
                if len(X_va) == 0 or len(X_tr) == 0:
                    continue

                reg_i = LGBMRegressor(**reg.get_params())
                sw_tr = sample_weight[tr_mask] if sample_weight is not None else None
                reg_i.fit(X_tr, y_tr, sample_weight=sw_tr)
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

            reg.fit(X, y, sample_weight=sample_weight)
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


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: List[str],
    cls_targets: List[str],
    *,
    halflife_years: float = 0.0,
    calibrate: bool = True,
) -> Dict[str, object]:
    cls_models: Dict[str, object] = {}

    for target in cls_targets:
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
        sample_weight = compute_sample_weights(dates, halflife_years)
        if sample_weight is not None:
            logging.info(
                "  [%s] sample_weight enabled (halflife_years=%.1f, min=%.3f, max=%.3f)",
                target, halflife_years, float(sample_weight.min()), float(sample_weight.max()),
            )

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

        last_va_mask: np.ndarray | None = None
        folds = time_series_folds(dates, N_SPLITS)
        if not folds:
            cls.fit(X, y, sample_weight=sample_weight)
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
                sw_tr = sample_weight[tr_mask] if sample_weight is not None else None
                cls_i.fit(X_tr, y_tr, sample_weight=sw_tr)
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
                last_va_mask = va_mask

            cls.fit(X, y, sample_weight=sample_weight)
            if calibrate and last_va_mask is not None and int(last_va_mask.sum()) >= 20:
                X_cal = X[last_va_mask]
                y_cal = y[last_va_mask]
                if FrozenEstimator is not None:
                    cal = CalibratedClassifierCV(FrozenEstimator(cls), method="isotonic", cv=None)
                else:
                    cal = CalibratedClassifierCV(cls, method="isotonic", cv="prefit")
                cal.fit(X_cal, y_cal)
                calibrated_list = getattr(cal, "calibrated_classifiers_", None) or []
                calibrators = getattr(calibrated_list[0], "calibrators", []) if calibrated_list else []
                if len(calibrators) != 1:
                    raise ValueError(
                        f"Expected exactly one binary isotonic calibrator for {target}, got {len(calibrators)}"
                    )
                cls_models[target] = BinaryIsotonicCalibratedClassifier(
                    estimator=cls,
                    calibrator=calibrators[0],
                )
                logging.info(
                    "  [%s] isotonic calibration applied (last-fold n_cal=%d)",
                    target, int(last_va_mask.sum()),
                )
            else:
                if calibrate:
                    logging.warning(
                        "  [%s] calibration skipped (last-fold n=%d)",
                        target, int(last_va_mask.sum()) if last_va_mask is not None else 0,
                    )
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
    args = parse_args()
    cls_horizons = args.cls_horizons if args.cls_horizons is not None else DEFAULT_CLS_HORIZONS
    reg_targets, cls_targets = build_targets(args.horizons, cls_horizons)

    logging.info("Training horizons: %s", reg_targets)
    df, feature_cols = make_merged(reg_targets, cls_targets, args.features_csv, args.labels_csv)
    df, cutoff = apply_train_end_date(df, args.train_end_date)

    logging.info("Start training regressors (log-return + MDD)...")
    reg_models = train_regressors(
        df, feature_cols, reg_targets,
        halflife_years=args.sample_weight_halflife,
    )

    logging.info("Start training classifiers (Top20 flags)...")
    cls_models = train_classifiers(
        df, feature_cols, cls_targets,
        halflife_years=args.sample_weight_halflife,
        calibrate=not args.no_calibrate,
    )

    train_end_date = cutoff.strftime("%Y-%m-%d") if cutoff is not None else None
    trained_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pack = {
        "features": feature_cols,
        "reg_models": reg_models,
        "cls_models": cls_models,
        "reg_targets": list(reg_models.keys()),
        "cls_targets": list(cls_models.keys()),
        "model_version": args.model_version,
        "train_end_date": train_end_date,
        "trained_at": trained_at,
    }

    export_feature_importance_reports(
        reg_models=reg_models,
        cls_models=cls_models,
        feature_cols=feature_cols,
        output_dir=MODEL_FEATURE_IMPORTANCE_DIR,
        model_version=args.model_version,
        trained_at=trained_at,
        train_end_date=train_end_date,
    )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(pack, f)

    logging.info(
        "Saved model package to %s (model_version=%s, train_end_date=%s, reg_targets=%s, cls_targets=%s)",
        args.output_pkl.resolve(),
        args.model_version,
        train_end_date,
        list(reg_models.keys()),
        list(cls_models.keys()),
    )


if __name__ == "__main__":
    main()
