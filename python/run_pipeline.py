# python/run_pipeline.py
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
try:
    from db import get_engine, log_pipeline_history
except Exception:
    get_engine = None
    log_pipeline_history = None

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lee_trader.db"

STEPS: List[Tuple[str, str]] = [
    ("fetch_market_data", "python/fetch_market_data.py"),   # 시장 데이터
    ("fetch_top_universe", "python/fetch_top_universe.py"),
    ("merge_universe", "python/merge_universe.py"),
    ("download_prices_kis", "python/download_prices_kis.py"),
    ("clean_prices", "python/clean_prices.py"),
    ("create_adjusted_prices", "python/create_adjusted_prices.py"),
    # ("fetch_fundamentals_dart", "python/fetch_fundamentals_dart.py"),
    ("quality_builder", "python/quality_builder.py"),
    ("feature_builder", "python/feature_builder.py"),
    ("scoring", "python/scoring.py"),
    ("label_builder", "python/label_builder.py"),
    ("model_train", "python/model_train.py"),
    ("model_predict", "python/model_predict.py"),
    ("ranking_builder", "python/ranking_builder.py"),
    # ("migrate_to_sqlite", "migrate_to_sqlite.py"),
]

# 핵심 종목 존재 여부를 파이프라인 끝에서 검증 (재발 방지)
CORE_CODES = ["005930", "000660", "035420"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def run_step(name: str, script: str) -> float:
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for step {name}: {script_path}")

    start_ts = time.perf_counter()
    logging.info("==> Running step: %s (%s)", name, script_path)
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    elapsed = time.perf_counter() - start_ts
    if result.returncode != 0:
        raise RuntimeError(f"Step {name} failed with return code {result.returncode}")
    logging.info("<== Step completed: %s (elapsed=%.2fs)", name, elapsed)
    return elapsed


def _load_codes_from_csv(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype={"code": str})
    if "code" not in df.columns:
        return set()
    return set(df["code"].astype(str).str.zfill(6).unique())


def _load_codes_from_db(table: str) -> set:
    # Prefer Postgres via SQLAlchemy engine if available
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                rows = conn.execute(f"SELECT DISTINCT code FROM {table}").fetchall()
            return set(str(r[0]).zfill(6) for r in rows if r and r[0] is not None)
        except Exception:
            pass

    if not DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            rows = cur.execute(f"SELECT DISTINCT code FROM {table}").fetchall()
            return set(str(r[0]).zfill(6) for r in rows if r and r[0] is not None)
    except Exception:
        return set()


def verify_core_codes(codes: list[str]) -> None:
    """Ensure critical codes exist in both CSV outputs and DB tables."""
    codes = [str(c).zfill(6) for c in codes]
    files = {
        "features.csv": DATA_DIR / "features.csv",
        "predictions.csv": DATA_DIR / "predictions.csv",
        "ranking_final.csv": DATA_DIR / "ranking_final.csv",
    }
    tables = ["features", "predictions", "daily_ranking"]

    missing = []

    for name, path in files.items():
        present = _load_codes_from_csv(path)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{name}: {','.join(miss)}")

    for tbl in tables:
        present = _load_codes_from_db(tbl)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{tbl} (DB): {','.join(miss)}")

    if missing:
        raise RuntimeError(f"Core code check failed -> { '; '.join(missing) }")

    logging.info("Core code check passed: %s", ", ".join(codes))


def main() -> None:
    setup_logging()
    ensure_data_dir()

    run_id = f"{datetime.utcnow().isoformat()}-{os.getpid()}"
    try:
        pipeline_start = time.perf_counter()
        for name, script in STEPS:
            if log_pipeline_history:
                log_pipeline_history(run_id, name, "start", None, None)
            elapsed = run_step(name, script)
            if log_pipeline_history:
                log_pipeline_history(run_id, name, "success", elapsed, None)

        verify_core_codes(CORE_CODES)
        total_elapsed = time.perf_counter() - pipeline_start
        if log_pipeline_history:
            log_pipeline_history(run_id, "pipeline", "success", total_elapsed, "all steps completed")
        logging.info("Pipeline finished successfully (total=%.2fs)", total_elapsed)
    except Exception as e:
        logging.exception("Pipeline error: %s", e)
        if log_pipeline_history:
            try:
                log_pipeline_history(run_id, "pipeline", "failed", None, str(e))
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
