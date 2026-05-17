"""
evaluate_prediction_accuracy.py

archive_predictions.py로 저장된 과거 예측 파일과 실제 레이블을 비교하여
예측 정확도 리포트를 생성합니다.

평가 항목:
  - pred_return_60d  vs  target_60d    (RMSE, 방향 정확도)
  - pred_mdd_60d     vs  target_mdd_60d (RMSE, 방향 정확도)
  - prob_top20_60d   calibration        (구간별 실제 Top20 비율)

기본 동작: predictions_archive/에서 60일 이상 지난 아카이브를 자동 선택.

Usage:
    python evaluate_prediction_accuracy.py
    python evaluate_prediction_accuracy.py --archive-date 20260318
    python evaluate_prediction_accuracy.py --min-days 60
    python evaluate_prediction_accuracy.py --output-dir outputs/accuracy
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("evaluate_prediction_accuracy")

DATA_DIR = Path("data")
LABELS_CSV = DATA_DIR / "labels.csv"
ARCHIVE_DIR = DATA_DIR / "predictions_archive"
OUTPUTS_DIR = Path("outputs")

# 예측값-실제값 쌍 정의
REGRESSION_PAIRS = [
    ("pred_return_60d", "target_60d", "수익률 60d"),
    ("pred_mdd_60d", "target_mdd_60d", "MDD 60d"),
]
CALIBRATION_COL = ("prob_top20_60d", "target_60d_top20", "Top20 확률 Calibration")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate prediction accuracy against realized labels")
    p.add_argument(
        "--archive-date",
        type=str,
        default=None,
        help="평가할 아카이브 날짜 (YYYYMMDD). 미지정 시 자동 선택.",
    )
    p.add_argument(
        "--min-days",
        type=int,
        default=60,
        help="최소 경과 일수 (기본 60). 이 이상 지난 아카이브만 평가.",
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=LABELS_CSV,
        help="레이블 CSV 경로",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_DIR,
        help="아카이브 디렉토리 경로",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="리포트 출력 디렉토리",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="조건 맞는 모든 아카이브 평가 (기본: 최신 1개)",
    )
    return p.parse_args()


def find_archive_dates(archive_dir: Path, min_days: int) -> list[str]:
    """min_days 이상 지난 아카이브 날짜 목록을 내림차순으로 반환."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=min_days)

    dates = []
    for f in sorted(archive_dir.glob("????????.csv"), reverse=True):
        try:
            dt = pd.to_datetime(f.stem, format="%Y%m%d")
            if dt <= cutoff:
                dates.append(f.stem)
        except ValueError:
            continue
    return dates


def load_archive(archive_dir: Path, date_str: str) -> pd.DataFrame:
    path = archive_dir / f"{date_str}.csv"
    if not path.exists():
        raise FileNotFoundError(f"아카이브 파일 없음: {path.resolve()}")
    df = pd.read_csv(path, dtype={"code": str})
    LOGGER.info("아카이브 로드: %s (%d 행)", path.name, len(df))
    return df


def load_labels(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"레이블 파일 없음: {labels_csv.resolve()}")
    df = pd.read_csv(labels_csv, dtype={"code": str})
    LOGGER.info("레이블 로드: %s (%d 행)", labels_csv.name, len(df))
    return df


def merge_pred_label(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """예측 날짜와 동일한 date 기준으로 예측-레이블 병합."""
    pred_cols = ["date", "code"] + [
        col for col in pred_df.columns
        if col not in ("date", "code")
    ]
    pred = pred_df[[c for c in pred_cols if c in pred_df.columns]].copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.strftime("%Y-%m-%d")

    label_needed = ["date", "code", "target_60d", "target_mdd_60d", "target_60d_top20"]
    label = label_df[[c for c in label_needed if c in label_df.columns]].copy()
    label["date"] = pd.to_datetime(label["date"]).dt.strftime("%Y-%m-%d")

    merged = pred.merge(label, on=["date", "code"], how="inner")
    LOGGER.info("병합 결과: %d 행 (예측 %d, 레이블 %d)", len(merged), len(pred), len(label))
    return merged


def compute_regression_metrics(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    sub = df[[pred_col, actual_col]].dropna()
    n = len(sub)
    if n == 0:
        return {"n": 0, "rmse": None, "mae": None, "direction_acc": None, "corr": None}

    pred = sub[pred_col].values
    actual = sub[actual_col].values

    rmse = math.sqrt(np.mean((pred - actual) ** 2))
    mae = np.mean(np.abs(pred - actual))

    # 방향 정확도: 둘 다 양수/음수인 비율 (중립 0 제외)
    valid_dir = (pred != 0) & (actual != 0)
    if valid_dir.sum() > 0:
        direction_acc = float(np.mean(np.sign(pred[valid_dir]) == np.sign(actual[valid_dir])))
    else:
        direction_acc = None

    corr = float(np.corrcoef(pred, actual)[0, 1]) if n > 1 else None

    return {
        "n": n,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "direction_acc": round(direction_acc, 4) if direction_acc is not None else None,
        "corr": round(corr, 4) if corr is not None else None,
    }


def compute_calibration(df: pd.DataFrame, prob_col: str, actual_col: str, n_bins: int = 10) -> list[dict]:
    """prob_col을 n_bins 구간으로 나눠 실제 positive 비율과 비교."""
    sub = df[[prob_col, actual_col]].dropna()
    if len(sub) == 0:
        return []

    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (sub[prob_col] >= lo) & (sub[prob_col] < hi if i < n_bins - 1 else sub[prob_col] <= hi)
        bucket = sub[mask]
        n = len(bucket)
        actual_rate = float(bucket[actual_col].mean()) if n > 0 else None
        mid = round((lo + hi) / 2, 2)
        results.append({
            "prob_range": f"{lo:.1f}~{hi:.1f}",
            "mid": mid,
            "n": n,
            "actual_top20_rate": round(actual_rate, 4) if actual_rate is not None else None,
            "gap": round(actual_rate - mid, 4) if actual_rate is not None else None,
        })
    return results


def format_report(
    archive_date: str,
    df: pd.DataFrame,
    reg_results: list[tuple[str, dict]],
    cal_rows: list[dict],
) -> str:
    lines = []
    lines.append(f"# 예측 정확도 리포트 — {archive_date}")
    lines.append("")
    lines.append(f"- 예측 기준일: {archive_date}")
    lines.append(f"- 평가 행 수: {len(df):,}")
    lines.append(f"- 레이블 기준 날짜 분포: {df['date'].nunique()} 거래일")
    lines.append("")

    lines.append("## 회귀 예측 정확도")
    lines.append("")
    lines.append("| 항목 | N | RMSE | MAE | 방향 정확도 | 상관계수 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, m in reg_results:
        n = m["n"]
        rmse = f"{m['rmse']:.4f}" if m["rmse"] is not None else "—"
        mae = f"{m['mae']:.4f}" if m["mae"] is not None else "—"
        da = f"{m['direction_acc']:.1%}" if m["direction_acc"] is not None else "—"
        corr = f"{m['corr']:.4f}" if m["corr"] is not None else "—"
        lines.append(f"| {label} | {n:,} | {rmse} | {mae} | {da} | {corr} |")
    lines.append("")

    if cal_rows:
        lines.append("## prob_top20_60d Calibration")
        lines.append("")
        lines.append("| 예측 구간 | N | 실제 Top20 비율 | 차이 (실제-예측) |")
        lines.append("|---|---:|---:|---:|")
        for row in cal_rows:
            actual = f"{row['actual_top20_rate']:.1%}" if row["actual_top20_rate"] is not None else "—"
            gap = f"{row['gap']:+.1%}" if row["gap"] is not None else "—"
            lines.append(f"| {row['prob_range']} | {row['n']:,} | {actual} | {gap} |")
        lines.append("")
        lines.append("> 차이가 양수(+)이면 과소추정(모델이 낮게 예측), 음수(-)이면 과대추정.")
        lines.append("")

    lines.append("---")
    lines.append("*generated by evaluate_prediction_accuracy.py*")
    return "\n".join(lines)


def evaluate_one(
    archive_date: str,
    archive_dir: Path,
    label_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    pred_df = load_archive(archive_dir, archive_date)
    merged = merge_pred_label(pred_df, label_df)

    if len(merged) == 0:
        LOGGER.warning("[%s] 병합 결과 0행 — 레이블이 아직 생성되지 않았을 수 있습니다", archive_date)
        return None

    reg_results = []
    for pred_col, actual_col, label in REGRESSION_PAIRS:
        if pred_col in merged.columns and actual_col in merged.columns:
            m = compute_regression_metrics(merged, pred_col, actual_col)
            reg_results.append((label, m))
            LOGGER.info(
                "[%s] %s — RMSE=%.4f, 방향=%.1f%%, Corr=%.4f (N=%d)",
                archive_date, label,
                m["rmse"] or 0,
                (m["direction_acc"] or 0) * 100,
                m["corr"] or 0,
                m["n"],
            )
        else:
            LOGGER.warning("[%s] 컬럼 없음: %s 또는 %s", archive_date, pred_col, actual_col)

    prob_col, actual_col, _ = CALIBRATION_COL
    cal_rows = []
    if prob_col in merged.columns and actual_col in merged.columns:
        cal_rows = compute_calibration(merged, prob_col, actual_col)
    else:
        LOGGER.warning("[%s] Calibration 컬럼 없음: %s 또는 %s", archive_date, prob_col, actual_col)

    report_text = format_report(archive_date, merged, reg_results, cal_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"prediction_accuracy_report_{archive_date}.md"
    out_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("리포트 저장: %s", out_path.resolve())
    return out_path


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.archive_dir.exists():
        LOGGER.error("아카이브 디렉토리 없음: %s", args.archive_dir.resolve())
        raise SystemExit(1)

    label_df = load_labels(args.labels_csv)

    if args.archive_date:
        dates_to_eval = [pd.to_datetime(args.archive_date).strftime("%Y%m%d")]
    else:
        dates_to_eval = find_archive_dates(args.archive_dir, args.min_days)
        if not dates_to_eval:
            LOGGER.info("평가 가능한 아카이브 없음 (최소 %d일 경과 필요)", args.min_days)
            return
        if not args.all:
            dates_to_eval = dates_to_eval[:1]
        LOGGER.info("평가 대상 아카이브: %s", dates_to_eval)

    for archive_date in dates_to_eval:
        try:
            evaluate_one(archive_date, args.archive_dir, label_df, args.output_dir)
        except FileNotFoundError as e:
            LOGGER.error("%s", e)
        except Exception as e:
            LOGGER.error("[%s] 평가 실패: %s", archive_date, e, exc_info=True)


if __name__ == "__main__":
    main()
