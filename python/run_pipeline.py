# python/run_pipeline.py
import subprocess
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path("data")

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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def run_step(name: str, script: str) -> None:
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


def main() -> None:
    setup_logging()
    ensure_data_dir()

    try:
        pipeline_start = time.perf_counter()
        for name, script in STEPS:
            run_step(name, script)
        logging.info("Pipeline finished successfully (total=%.2fs)", time.perf_counter() - pipeline_start)
    except Exception as e:
        logging.exception("Pipeline error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
