import argparse
import logging
from pathlib import Path

import pandas as pd


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def build_walkforward_splits(
    data_start_date: str,
    data_end_date: str,
    train_window_months: int,
    predict_window_months: int,
    step_months: int,
) -> pd.DataFrame:
    """
    Build non-overlapping walk-forward splits.

    Rules:
    - Each split has a fixed train window and fixed predict window.
    - `train_end` is always strictly earlier than `predict_start`.
    - Prediction windows do not overlap because the loop advances by `step_months`
      and this function validates that `step_months >= predict_window_months`.

    Output columns:
    - split_id
    - train_start
    - train_end
    - predict_start
    - predict_end
    """
    if train_window_months <= 0:
        raise ValueError("train_window_months must be > 0")
    if predict_window_months <= 0:
        raise ValueError("predict_window_months must be > 0")
    if step_months <= 0:
        raise ValueError("step_months must be > 0")
    if step_months < predict_window_months:
        raise ValueError(
            "step_months must be >= predict_window_months to keep prediction windows non-overlapping"
        )

    data_start = pd.to_datetime(data_start_date).normalize()
    data_end = pd.to_datetime(data_end_date).normalize()
    if data_start > data_end:
        raise ValueError("data_start_date must be <= data_end_date")

    rows: list[dict] = []
    split_id = 1

    # The first prediction starts immediately after the first train window ends.
    current_predict_start = data_start + pd.DateOffset(months=train_window_months)

    while current_predict_start <= data_end:
        # Train window is the fixed-length window immediately before predict_start.
        train_start = current_predict_start - pd.DateOffset(months=train_window_months)
        train_end = current_predict_start - pd.Timedelta(days=1)

        # Predict window is a fixed-length forward window clipped by the full data end date.
        predict_start = current_predict_start
        raw_predict_end = predict_start + pd.DateOffset(months=predict_window_months) - pd.Timedelta(days=1)
        predict_end = min(raw_predict_end, data_end)

        # Skip incomplete or invalid windows.
        if train_start < data_start:
            current_predict_start = current_predict_start + pd.DateOffset(months=step_months)
            continue
        if train_end >= predict_start:
            raise ValueError("Invalid split generated: train_end must be < predict_start")
        if predict_start > predict_end:
            break

        rows.append(
            {
                "split_id": split_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "predict_start": predict_start.strftime("%Y-%m-%d"),
                "predict_end": predict_end.strftime("%Y-%m-%d"),
            }
        )
        split_id += 1

        # Advance by step_months. Because step_months >= predict_window_months,
        # the next predict window cannot overlap the previous one.
        current_predict_start = current_predict_start + pd.DateOffset(months=step_months)

    out = pd.DataFrame(
        rows,
        columns=["split_id", "train_start", "train_end", "predict_start", "predict_end"],
    )

    if out.empty:
        logging.warning("No walk-forward splits were generated")
        return out

    # Validate non-overlap between neighboring prediction windows.
    pred_start = pd.to_datetime(out["predict_start"])
    pred_end = pd.to_datetime(out["predict_end"])
    overlap_mask = pred_start.iloc[1:].reset_index(drop=True) <= pred_end.iloc[:-1].reset_index(drop=True)
    if overlap_mask.any():
        raise ValueError("Generated prediction windows overlap, which violates the split contract")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build walk-forward split schedule")
    p.add_argument("--data-start-date", required=True, help="Full data start date (YYYY-MM-DD)")
    p.add_argument("--data-end-date", required=True, help="Full data end date (YYYY-MM-DD)")
    p.add_argument("--train-window-months", required=True, type=int, help="Training window length in months")
    p.add_argument("--predict-window-months", required=True, type=int, help="Prediction window length in months")
    p.add_argument("--step-months", required=True, type=int, help="Step size in months")
    p.add_argument("--out-csv", type=Path, help="Optional output CSV path")
    return p.parse_args()


def log_example_splits(df: pd.DataFrame) -> None:
    if df.empty:
        logging.info("No example splits to log")
        return

    # Show at least up to the first 3 splits so the caller can sanity-check the schedule.
    sample = df.head(3)
    for row in sample.to_dict(orient="records"):
        logging.info(
            "split_id=%s train=%s~%s predict=%s~%s",
            row["split_id"],
            row["train_start"],
            row["train_end"],
            row["predict_start"],
            row["predict_end"],
        )


def main() -> None:
    setup_logging()
    args = parse_args()

    df = build_walkforward_splits(
        data_start_date=args.data_start_date,
        data_end_date=args.data_end_date,
        train_window_months=args.train_window_months,
        predict_window_months=args.predict_window_months,
        step_months=args.step_months,
    )

    logging.info("Generated walk-forward splits: %d", len(df))
    log_example_splits(df)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False, encoding="utf-8")
        logging.info("Saved split CSV: %s", args.out_csv.resolve())

    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
