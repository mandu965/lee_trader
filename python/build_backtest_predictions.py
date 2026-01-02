"""
build_backtest_predictions.py

Point-in-time 예측을 백필하여 research.prediction_history에 적재한다.

입력:
  - features CSV (일자별 특징)
  - model.pkl (model_predict.py에서 사용하는 포맷)
  - 대상 as_of_date 리스트 (start/end 또는 파일)
  - (선택) universe CSV (code 컬럼으로 필터)

출력:
  - 연구 테이블 research.prediction_history에 run_id를 붙여 append
  - (선택) CSV 저장 (--out-csv)
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from db import get_engine
except Exception:
    get_engine = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill prediction_history with run_id")
    p.add_argument("--features-csv", type=Path, default=Path("data/features.csv"))
    p.add_argument("--model-pkl", type=Path, default=Path("data/model.pkl"))
    p.add_argument("--universe-csv", type=Path, help="Optional universe file with 'code' column")
    p.add_argument("--dates-file", type=Path, help="Optional text file with as_of_date list (YYYY-MM-DD per line)")
    p.add_argument("--start-date", type=str, help="as_of_date range start (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, help="as_of_date range end (YYYY-MM-DD)")
    p.add_argument("--run-id", type=int, required=True, help="Backfill run_id (e.g., 4)")
    p.add_argument("--model-version", type=str, default=os.environ.get("MODEL_VERSION", "v1"))
    p.add_argument("--horizon-days", type=int, default=int(os.environ.get("HORIZON_DAYS", "60")))
    p.add_argument("--out-csv", type=Path, help="Optional path to save all predictions to CSV")
    p.add_argument("--log-interval", type=int, default=20, help="Log every N dates")
    return p.parse_args()


def load_model(model_path: Path) -> Dict[str, Any]:
    import pickle

    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found: {model_path}")
    with open(model_path, "rb") as f:
        pack = pickle.load(f)
    features = pack.get("features")
    reg_models = pack.get("reg_models", {})
    cls_models = pack.get("cls_models", {})
    reg_targets = pack.get("reg_targets", list(reg_models.keys()))
    cls_targets = pack.get("cls_targets", list(cls_models.keys()))
    if not features:
        raise ValueError("model pack missing 'features'")
    logging.info("Loaded model pack: %d reg targets, %d cls targets, %d features", len(reg_targets), len(cls_targets), len(features))
    return {
        "features": list(features),
        "reg_models": reg_models,
        "cls_models": cls_models,
        "reg_targets": list(reg_targets),
        "cls_targets": list(cls_targets),
    }


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        raise FileNotFoundError(f"features.csv not found: {features_csv}")
    df = pd.read_csv(features_csv, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must contain 'date' and 'code'")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    logging.info("Loaded features: %s (rows=%d)", features_csv, len(df))
    return df


def load_universe(path: Path | None) -> set[str] | None:
    if not path:
        return None
    if not path.exists():
        logging.warning("universe file not found: %s", path)
        return None
    df = pd.read_csv(path, dtype={"code": str})
    if "code" not in df.columns:
        logging.warning("universe file missing code column: %s", path)
        return None
    codes = set(df["code"].astype(str).str.zfill(6))
    logging.info("Loaded universe codes: %d", len(codes))
    return codes


def parse_dates(features_df: pd.DataFrame, args: argparse.Namespace) -> List[pd.Timestamp]:
    if args.dates_file and args.dates_file.exists():
        raw = [line.strip() for line in args.dates_file.read_text().splitlines() if line.strip()]
        dates = pd.to_datetime(raw)
        logging.info("Using dates from file: %d", len(dates))
    else:
        min_d = pd.to_datetime(args.start_date) if args.start_date else features_df["date"].min()
        max_d = pd.to_datetime(args.end_date) if args.end_date else features_df["date"].max()
        mask = (features_df["date"] >= min_d) & (features_df["date"] <= max_d)
        dates = features_df.loc[mask, "date"].dropna().unique()
        dates = pd.to_datetime(dates)
        logging.info("Using date range %s ~ %s (unique dates=%d)", min_d.date(), max_d.date(), len(dates))
    dates = sorted(pd.to_datetime(dates).unique())
    return dates


def predict_for_date(model_pack: Dict[str, Any], feats: pd.DataFrame) -> pd.DataFrame:
    """Predict regression/classification outputs for a single as_of_date subset."""
    feature_cols = model_pack["features"]
    reg_models: Dict[str, Any] = model_pack["reg_models"]
    cls_models: Dict[str, Any] = model_pack["cls_models"]
    reg_targets = model_pack["reg_targets"]
    cls_targets = model_pack["cls_targets"]

    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = feats[feature_cols].copy()
    codes = feats["code"].astype(str).values
    dates = feats["date"].dt.strftime("%Y-%m-%d").values
    out = pd.DataFrame({"date": dates, "code": codes})

    # Regression targets
    for target in reg_targets:
        model = reg_models.get(target)
        if model is None:
            continue
        pred = model.predict(X)
        if target == "target_log_30d":
            out["pred_return_30d"] = np.exp(np.clip(pred, -5.0, 5.0)) - 1.0
        elif target == "target_log_60d":
            out["pred_return_60d"] = np.exp(np.clip(pred, -5.0, 5.0)) - 1.0
        elif target == "target_log_90d":
            out["pred_return_90d"] = np.exp(np.clip(pred, -5.0, 5.0)) - 1.0
        elif target == "target_mdd_30d":
            out["pred_mdd_30d"] = pred
        elif target == "target_mdd_60d":
            out["pred_mdd_60d"] = pred
        elif target == "target_mdd_90d":
            out["pred_mdd_90d"] = pred
        else:
            out[target] = pred

    # Classification targets
    for target in cls_targets:
        model = cls_models.get(target)
        if model is None:
            continue
        col_name = None
        if target == "target_30d_top20":
            col_name = "prob_top20_30d"
        elif target == "target_60d_top20":
            col_name = "prob_top20_60d"
        elif target == "target_90d_top20":
            col_name = "prob_top20_90d"
        if not col_name:
            continue
        proba = model.predict_proba(X)[:, 1]
        out[col_name] = proba

    return out


def _percentile_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True) * 100.0
    return df.groupby("date", group_keys=False)[col].transform(_rank)


def compute_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """
    점수 계산:
      - ret_score: pred_return_60d/90d + pred_mdd 조합 기반
      - prob_score: prob_top20_60d * 100
      - qual_score: quality_score 날짜별 percentile (있을 때)
      - tech_score: vol_ma_20 날짜별 percentile (없으면 vol_20/vol_ratio_20)
      - risk_penalty: pred_mdd_60d 기반 단순 penalty
      - final_score: 가중합 * risk_penalty
    """
    df = preds.copy()
    needed_cols = [
        "pred_return_30d",
        "pred_return_60d",
        "pred_return_90d",
        "pred_mdd_30d",
        "pred_mdd_60d",
        "pred_mdd_90d",
        "prob_top20_30d",
        "prob_top20_60d",
        "prob_top20_90d",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ret_score: pred_return 조합을 pred_mdd로 조정 (사용 가능한 horizon 평균)
    ret_cols = ["pred_return_30d", "pred_return_60d", "pred_return_90d"]
    r_stack = np.vstack([df[c].values for c in ret_cols])
    r_comb = np.nanmean(r_stack, axis=0)
    mdd_cols = ["pred_mdd_30d", "pred_mdd_60d", "pred_mdd_90d"]
    mdd_stack = np.vstack([df[c].values for c in mdd_cols])
    pred_mdd_comb = np.nanmean(mdd_stack, axis=0)
    r_adj = r_comb / (1 + 3.0 * np.abs(pred_mdd_comb))
    r_mean = r_adj.mean()
    r_std = r_adj.std(ddof=0) if r_adj.std(ddof=0) and r_adj.std(ddof=0) > 0 else 1e-6
    df["ret_score"] = np.clip(50 + 10 * ((r_adj - r_mean) / r_std), 0, 100)

    # prob_score (30d 우선, 없으면 60d/90d)
    prob = df["prob_top20_30d"].fillna(df["prob_top20_60d"]).fillna(df["prob_top20_90d"]).fillna(0)
    df["prob_score"] = np.clip(prob * 100.0, 0, 100)

    # qual_score: quality_score percentile by date (optional)
    if "quality_score" in df.columns:
        df["qual_score"] = _percentile_by_date(df, "quality_score")
    else:
        df["qual_score"] = np.nan

    # tech_score: vol_ma_20 percentile fallback to vol_20 or vol_ratio_20
    if "vol_ma_20" in df.columns:
        df["tech_score"] = _percentile_by_date(df, "vol_ma_20")
    elif "vol_ratio_20" in df.columns:
        df["tech_score"] = _percentile_by_date(df, "vol_ratio_20")
    elif "vol_20" in df.columns:
        df["tech_score"] = _percentile_by_date(df, "vol_20")
    else:
        df["tech_score"] = np.nan

    # risk_penalty: pred_mdd 평균 기반 penalty (0.5~1.0)
    dd = df[["pred_mdd_30d", "pred_mdd_60d", "pred_mdd_90d"]].mean(axis=1, skipna=True).abs().fillna(0)
    risk_level = np.clip(dd / 0.4, 0, 1)
    df["risk_penalty"] = np.clip(1 - 0.5 * risk_level, 0.5, 1.0)

    # final_score: 가중합 후 penalty
    # 가중치 (ret, prob, qual, tech, bias) = 0.4, 0.25, 0.15, 0.10, 0.10
    bias_pred_score = 60.0
    base = (
        0.4 * df["ret_score"].fillna(0)
        + 0.25 * df["prob_score"].fillna(0)
        + 0.15 * df["qual_score"].fillna(0)
        + 0.10 * df["tech_score"].fillna(0)
        + 0.10 * bias_pred_score
    )
    df["final_score"] = np.clip(base * df["risk_penalty"].fillna(1.0), 0, 100)
    df["final_score_custom"] = np.nan
    return df


def save_prediction_history(run_id: int, model_version: str, horizon_days: int, df_all: pd.DataFrame) -> None:
    if not get_engine:
        logging.info("No DB engine available -> skip DB save")
        return
    try:
        eng = get_engine()
        out = df_all.copy()
        out["as_of_date"] = pd.to_datetime(out["date"]).dt.date
        out["run_id"] = int(run_id)
        out["model_version"] = model_version
        out["horizon_days"] = horizon_days

        cols = [
            "run_id",
            "as_of_date",
            "code",
            "model_version",
            "horizon_days",
            "pred_return_30d",
            "pred_return_60d",
            "pred_return_90d",
            "pred_mdd_30d",
            "pred_mdd_60d",
            "pred_mdd_90d",
            "prob_top20_30d",
            "prob_top20_60d",
            "prob_top20_90d",
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "final_score",
            "final_score_custom",
        ]
        missing = [c for c in cols if c not in out.columns]
        for c in missing:
            out[c] = np.nan

        out = out[cols]

        with eng.begin() as conn:
            conn.execute(
                text("DELETE FROM research.prediction_history WHERE run_id = :run_id"),
                {"run_id": int(run_id)},
            )
        out.to_sql("prediction_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Saved to research.prediction_history (run_id=%s, rows=%d)", run_id, len(out))
    except Exception:
        logging.exception("Failed to save prediction_history")


def main() -> None:
    setup_logging()
    args = parse_args()

    model_pack = load_model(args.model_pkl)
    feats = load_features(args.features_csv)
    universe_codes = load_universe(args.universe_csv)
    dates = parse_dates(feats, args)

    all_rows: List[pd.DataFrame] = []
    for i, d in enumerate(dates, 1):
        df_day = feats[feats["date"] == d].copy()
        if df_day.empty:
            logging.warning("No features for date %s -> skip", d.date())
            continue
        if universe_codes is not None:
            df_day = df_day[df_day["code"].isin(universe_codes)]
            if df_day.empty:
                logging.warning("Universe-filtered features empty for %s -> skip", d.date())
                continue
        preds = predict_for_date(model_pack, df_day)
        preds = compute_scores(preds)
        all_rows.append(preds)
        if i % args.log_interval == 0 or i == len(dates):
            logging.info("Predictions built for %s (%d/%d): rows=%d", d.date(), i, len(dates), len(preds))

    if not all_rows:
        logging.warning("No predictions generated -> exiting")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    if args.out_csv:
        df_all.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved predictions CSV: %s (rows=%d)", args.out_csv, len(df_all))

    save_prediction_history(args.run_id, args.model_version, args.horizon_days, df_all)


if __name__ == "__main__":
    main()
