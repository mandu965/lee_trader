"""
archive_predictions.py

매일 model_predict 실행 후 predictions.csv를 날짜별로 보존합니다.
evaluate_prediction_accuracy.py가 60d/90d 후 예측 정확도를 측정하는 데 사용합니다.

저장 위치: data/predictions_archive/YYYYMMDD.csv
파일명 기준: predictions.csv의 date 컬럼값 (최신 예측 기준일)

Usage:
    python archive_predictions.py
    python archive_predictions.py --force          # 이미 존재해도 덮어쓰기
    python archive_predictions.py --date 20260517  # 특정 날짜로 저장
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("archive_predictions")

DATA_DIR = Path("data")
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
ARCHIVE_DIR = DATA_DIR / "predictions_archive"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive predictions.csv by prediction date")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archive file if it exists",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override archive filename date (YYYYMMDD). Defaults to date from predictions.csv.",
    )
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=PREDICTIONS_CSV,
        help="Source predictions CSV path",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_DIR,
        help="Archive directory path",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not write a JSON sidecar manifest next to the archived CSV.",
    )
    return p.parse_args()


def resolve_prediction_date(pred_df: pd.DataFrame, override: str | None) -> str:
    if override:
        try:
            return pd.to_datetime(override).strftime("%Y%m%d")
        except Exception as e:
            raise ValueError(f"Invalid --date value '{override}': {e}") from e

    if "date" not in pred_df.columns:
        raise ValueError("predictions.csv has no 'date' column and --date was not specified")

    latest = pd.to_datetime(pred_df["date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        raise ValueError("Could not determine prediction date from predictions.csv")

    return latest.strftime("%Y%m%d")


def validate_predictions(pred_df: pd.DataFrame) -> None:
    required = {"date", "code"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")
    if pred_df.empty:
        raise ValueError("predictions.csv is empty")


def write_manifest(dest_csv: Path, pred_df: pd.DataFrame, archive_date: str) -> Path:
    manifest_path = dest_csv.with_suffix(".meta.json")
    payload = {
        "archive_date": archive_date,
        "csv_path": str(dest_csv),
        "row_count": int(len(pred_df)),
        "column_count": int(len(pred_df.columns)),
        "columns": list(pred_df.columns),
        "min_trade_date": str(pd.to_datetime(pred_df["date"], errors="coerce").min()),
        "max_trade_date": str(pd.to_datetime(pred_df["date"], errors="coerce").max()),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.predictions_csv.exists():
        LOGGER.error("predictions.csv not found: %s", args.predictions_csv.resolve())
        raise SystemExit(1)

    LOGGER.info("Loading predictions: %s", args.predictions_csv.resolve())
    pred_df = pd.read_csv(args.predictions_csv, dtype={"code": str})
    LOGGER.info("Loaded %d rows", len(pred_df))
    validate_predictions(pred_df)

    date_str = resolve_prediction_date(pred_df, args.date)

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    dest = args.archive_dir / f"{date_str}.csv"

    if dest.exists() and not args.force:
        LOGGER.info("Archive already exists (skipping): %s", dest.resolve())
        return

    shutil.copy2(args.predictions_csv, dest)
    LOGGER.info("Archived predictions -> %s (%d rows)", dest.resolve(), len(pred_df))
    if not args.skip_manifest:
        manifest_path = write_manifest(dest, pred_df, date_str)
        LOGGER.info("Wrote archive manifest -> %s", manifest_path.resolve())


if __name__ == "__main__":
    main()
