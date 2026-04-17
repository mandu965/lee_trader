import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from db import create_research_model_run

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
ARTIFACTS_MODELS_DIR = Path("artifacts/models")

HORIZON_DAYS = 60
TOP_N = 20
MIN_TRAIN_MONTHS = 12


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run expanding-window quarterly walk-forward backtest")
    p.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--start-date", type=str, help="Optional overall start date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, help="Optional overall end date (YYYY-MM-DD)")
    p.add_argument(
        "--rebalance-freq",
        type=str,
        default="quarterly",
        help="Walk-forward rebalance frequency metadata. Defaults to quarterly.",
    )
    p.add_argument(
        "--universe-version",
        type=str,
        required=True,
        help="Universe version metadata stored in dim_model_run.config_json.",
    )
    p.add_argument(
        "--score-weights-json",
        type=str,
        required=True,
        help="JSON string for score weight metadata stored in dim_model_run.config_json.",
    )
    p.add_argument(
        "--score-formula-version",
        type=str,
        default="ranking_builder_v1",
        help="Score formula version metadata stored in dim_model_run.config_json.",
    )
    p.add_argument(
        "--min-train-months",
        type=int,
        default=MIN_TRAIN_MONTHS,
        help="Minimum expanding training span before the first quarterly retrain.",
    )
    return p.parse_args()


def load_feature_dates(path: Path, start_date: str | None, end_date: str | None) -> pd.Series:
    df = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(df["date"]).dropna().drop_duplicates().sort_values()
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
    if dates.empty:
        raise ValueError("No feature dates available for the requested range")
    return dates.reset_index(drop=True)


def build_quarterly_windows(dates: pd.Series, min_train_months: int) -> list[dict]:
    start_date = dates.iloc[0]
    min_train_end = start_date + pd.DateOffset(months=min_train_months)

    quarter_ends = (
        pd.DataFrame({"date": dates})
        .assign(quarter=lambda d: d["date"].dt.to_period("Q"))
        .groupby("quarter", as_index=False)["date"]
        .max()["date"]
    )
    quarter_ends = quarter_ends[quarter_ends >= min_train_end].reset_index(drop=True)

    windows: list[dict] = []
    for idx, train_end in enumerate(quarter_ends):
        future_dates = dates[dates > train_end]
        if future_dates.empty:
            continue
        pred_start = future_dates.iloc[0]
        next_train_end = quarter_ends.iloc[idx + 1] if idx + 1 < len(quarter_ends) else dates.iloc[-1]
        pred_end = min(next_train_end, dates.iloc[-1])
        pred_dates = dates[(dates >= pred_start) & (dates <= pred_end)]
        if pred_dates.empty:
            continue
        windows.append(
            {
                "train_start": start_date,
                "train_end": train_end,
                "pred_start": pred_dates.iloc[0],
                "pred_end": pred_dates.iloc[-1],
            }
        )
    return windows


def run_command(args: list[str]) -> None:
    logging.info("Running: %s", " ".join(args))
    subprocess.run(args, check=True)


def run_walkforward_window(
    window: dict,
    features_csv: Path,
    labels_csv: Path,
    *,
    rebalance_freq: str,
    universe_version: str,
    score_formula_version: str,
    score_weights: dict,
) -> int:
    train_start = window["train_start"]
    train_end = window["train_end"]
    pred_start = window["pred_start"]
    pred_end = window["pred_end"]

    model_version = f"wf_h{HORIZON_DAYS}_{train_end.strftime('%Y%m%d')}"
    config_json = {
        "strategy": "expanding_window_quarterly",
        "horizon_days": HORIZON_DAYS,
        "top_n": TOP_N,
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "predict_start": pred_start.strftime("%Y-%m-%d"),
        "predict_end": pred_end.strftime("%Y-%m-%d"),
        "rebalance_freq": rebalance_freq,
        "universe_version": universe_version,
        "score_formula_version": score_formula_version,
        "score_weights": score_weights,
    }
    run_id = create_research_model_run(
        run_type="walkforward_backtest",
        model_version=model_version,
        horizon_days=HORIZON_DAYS,
        top_n=TOP_N,
        train_start_date=train_start.date(),
        train_end_date=train_end.date(),
        config_json=config_json,
        comment="quarterly expanding-window walk-forward run",
    )

    ARTIFACTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_pkl = ARTIFACTS_MODELS_DIR / f"run_{run_id}_{model_version}.pkl"

    run_command(
        [
            sys.executable,
            "python/model_train.py",
            "--horizons",
            str(HORIZON_DAYS),
            "--features-csv",
            str(features_csv),
            "--labels-csv",
            str(labels_csv),
            "--train-end-date",
            train_end.strftime("%Y-%m-%d"),
            "--model-version",
            model_version,
            "--output-pkl",
            str(model_pkl),
        ]
    )
    run_command(
        [
            sys.executable,
            "python/build_backtest_predictions.py",
            "--features-csv",
            str(features_csv),
            "--model-pkl",
            str(model_pkl),
            "--start-date",
            pred_start.strftime("%Y-%m-%d"),
            "--end-date",
            pred_end.strftime("%Y-%m-%d"),
            "--run-id",
            str(run_id),
            "--model-version",
            model_version,
            "--horizon-days",
            str(HORIZON_DAYS),
        ]
    )
    run_command(
        [
            sys.executable,
            "python/build_backtest_ranking.py",
            "--run-id",
            str(run_id),
            "--start-date",
            pred_start.strftime("%Y-%m-%d"),
            "--end-date",
            pred_end.strftime("%Y-%m-%d"),
            "--top-n",
            str(TOP_N),
        ]
    )
    run_command(
        [
            sys.executable,
            "python/build_backtest_outcome.py",
            "--run-id",
            str(run_id),
            "--horizon-days",
            str(HORIZON_DAYS),
            "--start-date",
            pred_start.strftime("%Y-%m-%d"),
            "--end-date",
            pred_end.strftime("%Y-%m-%d"),
        ]
    )

    logging.info(
        "Completed walk-forward run_id=%s model_version=%s train=%s~%s predict=%s~%s",
        run_id,
        model_version,
        train_start.date(),
        train_end.date(),
        pred_start.date(),
        pred_end.date(),
    )
    return run_id


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        score_weights = json.loads(args.score_weights_json)
    except Exception as e:
        raise ValueError(f"Invalid --score-weights-json: {e}") from e
    if not isinstance(score_weights, dict):
        raise ValueError("--score-weights-json must decode to a JSON object")

    dates = load_feature_dates(args.features_csv, args.start_date, args.end_date)
    windows = build_quarterly_windows(dates, args.min_train_months)
    if not windows:
        raise RuntimeError("No valid walk-forward windows were generated")

    logging.info("Generated %d walk-forward windows", len(windows))
    logging.info("Window summary: %s", json.dumps(
        [
            {
                "train_end": w["train_end"].strftime("%Y-%m-%d"),
                "pred_start": w["pred_start"].strftime("%Y-%m-%d"),
                "pred_end": w["pred_end"].strftime("%Y-%m-%d"),
            }
            for w in windows
        ],
        ensure_ascii=True,
    ))

    run_ids: list[int] = []
    for window in windows:
        run_ids.append(
            run_walkforward_window(
                window,
                args.features_csv,
                args.labels_csv,
                rebalance_freq=args.rebalance_freq,
                universe_version=args.universe_version,
                score_formula_version=args.score_formula_version,
                score_weights=score_weights,
            )
        )

    logging.info("Walk-forward backtest completed. run_ids=%s", run_ids)


if __name__ == "__main__":
    main()
