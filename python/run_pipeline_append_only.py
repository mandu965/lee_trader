"""
run_pipeline_append_only.py

prediction_history / ranking_history / backtest_outcome만 append하는 경량 스크립트.
run_id와 환경변수(HORIZON_DAYS, MODEL_VERSION, TOP_N 등)를 받아 append_* 함수를 실행한다.
"""
import argparse
import logging
import sys

from run_pipeline import (
    append_prediction_history,
    append_ranking_history,
    append_backtest_outcome,
    log_table_stats,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="run_id to append into research.prediction_history/ranking_history")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_id = args.run_id

    logging.info("Append-only pipeline start (run_id=%s)", run_id)
    append_prediction_history(run_id)
    append_ranking_history(run_id)
    append_backtest_outcome(run_id)
    log_table_stats()
    logging.info("Append-only pipeline done.")


if __name__ == "__main__":
    sys.exit(main())
