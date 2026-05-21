"""
Evaluate realized prediction accuracy from archived predictions.

This script compares archived prediction snapshots from
`data/predictions_archive/YYYYMMDD.csv` against `data/labels.csv` and writes a
markdown report into `outputs/`.

Improvements over the initial version:
 - dynamically loads the label columns needed by the archived prediction file
 - evaluates 30d and 60d targets only when the archive is mature enough
 - emits a report even when no rows can be evaluated yet
 - allows a fixed reference date for reproducible source-only checks
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("evaluate_prediction_accuracy")

DATA_DIR = Path("data")
LABELS_CSV = DATA_DIR / "labels.csv"
ARCHIVE_DIR = DATA_DIR / "predictions_archive"
OUTPUTS_DIR = Path("outputs")


@dataclass(frozen=True)
class RegressionSpec:
    pred_col: str
    actual_col: str
    label: str
    horizon_days: int


@dataclass(frozen=True)
class CalibrationSpec:
    prob_col: str
    actual_col: str
    label: str
    horizon_days: int


REGRESSION_SPECS = [
    RegressionSpec("pred_return_30d", "target_30d", "Return 30d", 30),
    RegressionSpec("pred_return_60d", "target_60d", "Return 60d", 60),
    RegressionSpec("pred_mdd_30d", "target_mdd_30d", "MDD 30d", 30),
    RegressionSpec("pred_mdd_60d", "target_mdd_60d", "MDD 60d", 60),
]
CALIBRATION_SPEC = CalibrationSpec(
    "prob_top20_60d",
    "target_60d_top20",
    "Top20 Probability Calibration 60d",
    60,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy against realized labels")
    parser.add_argument(
        "--archive-date",
        type=str,
        default=None,
        help="Specific archive date to evaluate (YYYYMMDD).",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=60,
        help="Minimum archive age in days when auto-selecting archives.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=LABELS_CSV,
        help="Labels CSV path.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_DIR,
        help="Archived predictions directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory for markdown reports.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Reference date used to decide maturity (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every eligible archive instead of only the latest one.",
    )
    return parser.parse_args()


def resolve_reference_date(value: str | None) -> pd.Timestamp:
    if value:
        return pd.to_datetime(value).normalize()
    return pd.Timestamp.today().normalize()


def find_archive_dates(archive_dir: Path, min_days: int, reference_date: pd.Timestamp) -> list[str]:
    cutoff = reference_date - pd.Timedelta(days=min_days)
    dates: list[str] = []
    for file_path in sorted(archive_dir.glob("????????.csv"), reverse=True):
        try:
            archive_date = pd.to_datetime(file_path.stem, format="%Y%m%d")
        except ValueError:
            continue
        if archive_date <= cutoff:
            dates.append(file_path.stem)
    return dates


def load_archive(archive_dir: Path, archive_date: str) -> pd.DataFrame:
    path = archive_dir / f"{archive_date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Archive file not found: {path.resolve()}")
    df = pd.read_csv(path, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError(f"Archive file must contain date/code columns: {path.resolve()}")
    LOGGER.info("Loaded archive %s (%d rows)", path.name, len(df))
    return df


def load_labels(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv.resolve()}")
    df = pd.read_csv(labels_csv, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("labels.csv must contain date/code columns")
    LOGGER.info("Loaded labels %s (%d rows)", labels_csv.name, len(df))
    return df


def build_required_label_columns(pred_df: pd.DataFrame) -> set[str]:
    needed = {"date", "code"}
    pred_cols = set(pred_df.columns)
    for spec in REGRESSION_SPECS:
        if spec.pred_col in pred_cols:
            needed.add(spec.actual_col)
    if CALIBRATION_SPEC.prob_col in pred_cols:
        needed.add(CALIBRATION_SPEC.actual_col)
    return needed


def merge_pred_label(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    pred = pred_df.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    required_label_cols = [
        column for column in build_required_label_columns(pred_df)
        if column in label_df.columns
    ]
    label = label_df[required_label_cols].copy()
    label["date"] = pd.to_datetime(label["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = pred.merge(label, on=["date", "code"], how="inner")
    LOGGER.info("Merged rows=%d (pred=%d, labels=%d)", len(merged), len(pred), len(label))
    return merged


def compute_regression_metrics(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, float | int | None]:
    sub = df[[pred_col, actual_col]].dropna()
    n = len(sub)
    if n == 0:
        return {"n": 0, "rmse": None, "mae": None, "direction_acc": None, "corr": None}

    pred = sub[pred_col].to_numpy()
    actual = sub[actual_col].to_numpy()

    rmse = math.sqrt(np.mean((pred - actual) ** 2))
    mae = np.mean(np.abs(pred - actual))

    valid_dir = (pred != 0) & (actual != 0)
    direction_acc = None
    if int(valid_dir.sum()) > 0:
        direction_acc = float(np.mean(np.sign(pred[valid_dir]) == np.sign(actual[valid_dir])))

    corr = None
    if n > 1:
        corr_value = np.corrcoef(pred, actual)[0, 1]
        if np.isfinite(corr_value):
            corr = float(corr_value)

    return {
        "n": n,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "direction_acc": round(direction_acc, 4) if direction_acc is not None else None,
        "corr": round(corr, 4) if corr is not None else None,
    }


def compute_calibration(df: pd.DataFrame, prob_col: str, actual_col: str, n_bins: int = 10) -> list[dict[str, object]]:
    sub = df[[prob_col, actual_col]].dropna()
    if sub.empty:
        return []

    bins = np.linspace(0, 1, n_bins + 1)
    results: list[dict[str, object]] = []
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx < n_bins - 1:
            mask = (sub[prob_col] >= lo) & (sub[prob_col] < hi)
        else:
            mask = (sub[prob_col] >= lo) & (sub[prob_col] <= hi)
        bucket = sub[mask]
        actual_rate = float(bucket[actual_col].mean()) if not bucket.empty else None
        midpoint = round((lo + hi) / 2, 2)
        results.append(
            {
                "prob_range": f"{lo:.1f}~{hi:.1f}",
                "mid": midpoint,
                "n": len(bucket),
                "actual_top20_rate": round(actual_rate, 4) if actual_rate is not None else None,
                "gap": round(actual_rate - midpoint, 4) if actual_rate is not None else None,
            }
        )
    return results


def format_metric_value(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "-"
    return format(value, fmt)


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def format_report(
    archive_date: str,
    archive_age_days: int,
    merged: pd.DataFrame,
    matured_horizons: list[int],
    pending_horizons: list[int],
    reg_results: list[tuple[str, int, dict[str, float | int | None] | None, str]],
    cal_rows: list[dict[str, object]],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Prediction Accuracy Report - {archive_date}")
    lines.append("")
    lines.append(f"- archive_date: `{archive_date}`")
    lines.append(f"- archive_age_days: `{archive_age_days}`")
    lines.append(f"- merged_rows: `{len(merged):,}`")
    lines.append(f"- merged_trade_dates: `{merged['date'].nunique() if not merged.empty else 0}`")
    lines.append(f"- matured_horizons: `{', '.join(map(str, matured_horizons)) if matured_horizons else 'none'}`")
    lines.append(f"- pending_horizons: `{', '.join(map(str, pending_horizons)) if pending_horizons else 'none'}`")
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Regression Metrics")
    lines.append("")
    lines.append("| Metric | Horizon | Status | N | RMSE | MAE | Direction Acc | Corr |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for label, horizon_days, metrics, status in reg_results:
        if metrics is None:
            lines.append(f"| {label} | {horizon_days} | {status} | 0 | - | - | - | - |")
            continue
        lines.append(
            "| {label} | {horizon} | {status} | {n:,} | {rmse} | {mae} | {direction} | {corr} |".format(
                label=label,
                horizon=horizon_days,
                status=status,
                n=int(metrics["n"] or 0),
                rmse=format_metric_value(metrics["rmse"], ".4f"),
                mae=format_metric_value(metrics["mae"], ".4f"),
                direction=format_percent(metrics["direction_acc"]),
                corr=format_metric_value(metrics["corr"], ".4f"),
            )
        )
    lines.append("")

    lines.append("## Probability Calibration")
    lines.append("")
    if cal_rows:
        lines.append("| Probability Range | N | Actual Top20 Rate | Gap (Actual - Midpoint) |")
        lines.append("|---|---:|---:|---:|")
        for row in cal_rows:
            actual_rate = row["actual_top20_rate"]
            gap = row["gap"]
            lines.append(
                "| {prob_range} | {n:,} | {actual} | {gap_text} |".format(
                    prob_range=row["prob_range"],
                    n=int(row["n"]),
                    actual=format_percent(actual_rate if isinstance(actual_rate, float) else None),
                    gap_text=f"{gap:+.1%}" if isinstance(gap, float) else "-",
                )
            )
    else:
        lines.append("- No mature probability calibration rows available yet.")
    lines.append("")

    lines.append("---")
    lines.append("*generated by evaluate_prediction_accuracy.py*")
    return "\n".join(lines)


def evaluate_one(
    archive_date: str,
    archive_dir: Path,
    label_df: pd.DataFrame,
    output_dir: Path,
    reference_date: pd.Timestamp,
) -> Path:
    pred_df = load_archive(archive_dir, archive_date)
    merged = merge_pred_label(pred_df, label_df)

    archive_dt = pd.to_datetime(archive_date, format="%Y%m%d")
    archive_age_days = int((reference_date - archive_dt).days)

    available_horizons = sorted({spec.horizon_days for spec in REGRESSION_SPECS if spec.pred_col in pred_df.columns})
    matured_horizons = [h for h in available_horizons if archive_age_days >= h]
    pending_horizons = [h for h in available_horizons if archive_age_days < h]

    notes: list[str] = []
    if merged.empty:
        notes.append("No rows merged between archived predictions and labels for the same trade date.")
    if pending_horizons:
        notes.append(
            "Some targets are not mature yet for this archive: "
            + ", ".join(f"{h}d" for h in pending_horizons)
        )

    reg_results: list[tuple[str, int, dict[str, float | int | None] | None, str]] = []
    for spec in REGRESSION_SPECS:
        if spec.pred_col not in pred_df.columns:
            reg_results.append((spec.label, spec.horizon_days, None, "prediction_missing"))
            continue
        if spec.actual_col not in merged.columns:
            reg_results.append((spec.label, spec.horizon_days, None, "label_missing"))
            continue
        if archive_age_days < spec.horizon_days:
            reg_results.append((spec.label, spec.horizon_days, None, "not_matured"))
            continue

        metrics = compute_regression_metrics(merged, spec.pred_col, spec.actual_col)
        status = "ok" if int(metrics["n"] or 0) > 0 else "no_rows"
        reg_results.append((spec.label, spec.horizon_days, metrics, status))

    cal_rows: list[dict[str, object]] = []
    if (
        CALIBRATION_SPEC.prob_col in pred_df.columns
        and CALIBRATION_SPEC.actual_col in merged.columns
        and archive_age_days >= CALIBRATION_SPEC.horizon_days
    ):
        cal_rows = compute_calibration(merged, CALIBRATION_SPEC.prob_col, CALIBRATION_SPEC.actual_col)
    elif CALIBRATION_SPEC.prob_col in pred_df.columns:
        notes.append("Probability calibration is pending until the 60d horizon matures.")

    report_text = format_report(
        archive_date=archive_date,
        archive_age_days=archive_age_days,
        merged=merged,
        matured_horizons=matured_horizons,
        pending_horizons=pending_horizons,
        reg_results=reg_results,
        cal_rows=cal_rows,
        notes=notes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prediction_accuracy_report_{archive_date}.md"
    output_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("Wrote report %s", output_path.resolve())
    return output_path


def main() -> None:
    setup_logging()
    args = parse_args()
    reference_date = resolve_reference_date(args.as_of_date)

    if not args.archive_dir.exists():
        LOGGER.error("Archive directory not found: %s", args.archive_dir.resolve())
        raise SystemExit(1)

    label_df = load_labels(args.labels_csv)

    if args.archive_date:
        dates_to_eval = [pd.to_datetime(args.archive_date).strftime("%Y%m%d")]
    else:
        dates_to_eval = find_archive_dates(args.archive_dir, args.min_days, reference_date)
        if not dates_to_eval:
            LOGGER.info("No archive is old enough to evaluate yet (min_days=%d)", args.min_days)
            return
        if not args.all:
            dates_to_eval = dates_to_eval[:1]
        LOGGER.info("Archives selected for evaluation: %s", dates_to_eval)

    for archive_date in dates_to_eval:
        try:
            evaluate_one(
                archive_date=archive_date,
                archive_dir=args.archive_dir,
                label_df=label_df,
                output_dir=args.output_dir,
                reference_date=reference_date,
            )
        except FileNotFoundError as exc:
            LOGGER.error("%s", exc)
        except Exception as exc:
            LOGGER.error("[%s] evaluation failed: %s", archive_date, exc, exc_info=True)


if __name__ == "__main__":
    main()
