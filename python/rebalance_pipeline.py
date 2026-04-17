"""
rebalance_pipeline.py

간단한 리밸런스용 일일 파이프라인.
- 30d(run_id 40), 60d(run_id 4), 90d(run_id 41) 예측 생성 및 prediction_history 적재
- 리밸런스 요일에 Top20 산출용 ranking_history, backtest_outcome 스냅샷 적재

주의: 스케줄러는 별도(github actions 등)에서 호출하며, 필요 시 RUNS/BASE 등만 맞춰 실행한다.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

RUNS = [
    {"run_id": 40, "horizon": 30, "model_pkl": "data/model_with_30d.pkl"},
    {"run_id": 4, "horizon": 60, "model_pkl": "data/model.pkl"},
    {"run_id": 41, "horizon": 90, "model_pkl": "data/model.pkl"},
]

FEATURES_CSV = "data/features.csv"
PRICES_CSV = "data/prices_daily_adjusted.csv"
BASE = os.environ.get("BASE", "outputs/research_plan_2025-12-16")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def run(cmd: List[str]) -> None:
    logging.info("Run: %s", " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    logging.info("Done (%.2fs)", time.perf_counter() - start)


def build_predictions(run_id: int, horizon: int, model_pkl: str) -> Path:
    out_csv = Path(BASE) / f"prediction_history_h{horizon}_run{run_id}_daily.csv"
    cmd = [
        sys.executable,
        "python/build_backtest_predictions.py",
        "--horizon-days",
        str(horizon),
        "--run-id",
        str(run_id),
        "--start-date",
        "2023-01-01",
        "--end-date",
        "2025-12-31",
        "--features-csv",
        FEATURES_CSV,
        "--model-pkl",
        model_pkl,
        "--out-csv",
        str(out_csv),
    ]
    run(cmd)
    return out_csv


def append_histories(run_id: int, horizon: int) -> None:
    env = os.environ.copy()
    env["HORIZON_DAYS"] = str(horizon)
    env["TOP_N"] = env.get("TOP_N", "20")
    # append_prediction_history / ranking_history / backtest_outcome
    cmd = [sys.executable, "python/run_pipeline_append_only.py", "--run-id", str(run_id)]
    logging.info("Appending histories for run_id=%s (horizon=%s)", run_id, horizon)
    subprocess.run(cmd, check=True, env=env)


def main():
    setup_logging()
    logging.info("Rebalance pipeline start (BASE=%s)", BASE)
    for r in RUNS:
        run_id = r["run_id"]
        horizon = r["horizon"]
        model_pkl = r["model_pkl"]
        logging.info("=== Horizon %sd / run_id %s ===", horizon, run_id)
        build_predictions(run_id, horizon, model_pkl)
        append_histories(run_id, horizon)
    logging.info("Rebalance pipeline completed for all runs.")


if __name__ == "__main__":
    main()
