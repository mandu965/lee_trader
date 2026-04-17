import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import create_research_model_run, get_engine


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward backtest from a split schedule CSV")
    p.add_argument("--splits-csv", type=Path, required=True, help="CSV created by walkforward_splits.py")
    p.add_argument("--model-pkl", type=Path, required=True, help="Model package used by build_backtest_predictions.py")
    p.add_argument("--features-csv", type=Path, default=Path("data/features.csv"))
    p.add_argument("--model-version", type=str, default="wf_backfill")
    p.add_argument("--horizon-days", type=int, default=60)
    p.add_argument(
        "--horizon-days-list",
        type=str,
        help="Optional comma-separated horizon list, e.g. 60,90. When omitted, --horizon-days is used.",
    )
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--start-split-id", type=int, help="Resume from this split_id (inclusive)")
    p.add_argument(
        "--max-splits",
        type=int,
        help="Optional maximum number of splits to run in this invocation. Omit to run all remaining splits.",
    )
    p.add_argument("--rebalance-freq", type=str, default="quarterly")
    p.add_argument("--universe-version", type=str, default="unknown")
    p.add_argument(
        "--universe-mode",
        type=str,
        default="fixed_current_universe",
        help="Universe construction mode stored in config_json. Current project default uses a fixed current universe.",
    )
    p.add_argument(
        "--score-formula-version",
        type=str,
        default="ranking_builder_v1",
        help="Score formula version stored in dim_model_run.config_json for comparison-group tracking.",
    )
    p.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("outputs/walkforward_run_summary"),
        help="Prefix path for the post-run CSV/Markdown summary files.",
    )
    p.add_argument(
        "--summary-min-runs",
        type=int,
        default=8,
        help="Minimum run count passed to check_walkforward_runs.py for the final sufficiency judgment.",
    )
    p.add_argument(
        "--score-weights-json",
        type=str,
        default="{}",
        help="JSON object stored in dim_model_run.config_json",
    )
    return p.parse_args()


def resolve_horizon_days_list(args: argparse.Namespace) -> list[int]:
    """
    Resolve the effective horizon list for this invocation.

    Backward compatibility:
    - Existing callers that only pass `--horizon-days 60` keep working unchanged.
    - New callers can pass `--horizon-days-list 60,90` to create one run per
      split and per horizon.
    """
    if not args.horizon_days_list:
        return [int(args.horizon_days)]

    values = []
    for token in str(args.horizon_days_list).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as e:
            raise ValueError(f"Invalid horizon value in --horizon-days-list: {token}") from e

    if not values:
        raise ValueError("--horizon-days-list must contain at least one integer")

    deduped = []
    seen = set()
    for value in values:
        if value <= 0:
            raise ValueError(f"horizon_days must be > 0: {value}")
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"splits CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["split_id", "train_start", "train_end", "predict_start", "predict_end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"splits CSV missing required columns: {missing}")
    df = df[required].copy()
    for col in required[1:]:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    df["split_id"] = pd.to_numeric(df["split_id"], errors="raise").astype(int)
    return df.sort_values("split_id").reset_index(drop=True)


def create_run_for_split(
    split: pd.Series,
    *,
    run_type: str,
    model_version: str,
    horizon_days: int,
    top_n: int,
    rebalance_freq: str,
    universe_version: str,
    universe_mode: str,
    score_formula_version: str,
    score_weights: dict,
) -> int:
    config_json = {
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "predict_start": split["predict_start"],
        "predict_end": split["predict_end"],
        "rebalance_freq": rebalance_freq,
        "universe_version": universe_version,
        "universe_mode": universe_mode,
        "uses_fixed_current_universe": universe_mode == "fixed_current_universe",
        "historical_universe_built": universe_mode == "historical",
        "score_formula_version": score_formula_version,
        "score_weights": score_weights,
        "split_id": int(split["split_id"]),
    }
    return create_research_model_run(
        run_type=run_type,
        model_version=model_version,
        horizon_days=horizon_days,
        top_n=top_n,
        train_start_date=split["train_start"],
        train_end_date=split["train_end"],
        config_json=config_json,
        comment=f"walk-forward split {int(split['split_id'])}",
    )


def run_command(cmd: list[str], *, split_id: int, run_id: int, horizon_days: int) -> None:
    logging.info(
        "Running split_id=%s run_id=%s horizon_days=%s: %s",
        split_id,
        run_id,
        horizon_days,
        " ".join(cmd),
    )
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(
            "Command failed for split_id=%s run_id=%s horizon_days=%s exit_code=%s",
            split_id,
            run_id,
            horizon_days,
            e.returncode,
        )
        raise


def collect_run_row_counts(run_id: int) -> dict[str, int]:
    eng = get_engine()
    out = {}
    with eng.connect() as conn:
        for table in ["prediction_history", "ranking_history", "backtest_outcome"]:
            cnt = conn.execute(
                text(f"SELECT count(*) FROM research.{table} WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).scalar()
            out[table] = int(cnt or 0)
    return out


def log_empty_horizon_warning(*, split_id: int, run_id: int, horizon_days: int, counts: dict[str, int]) -> None:
    """
    Emit a focused warning when a 90d run writes no rows.

    The 60d path is the legacy default and should remain quiet unless the run
    outright fails. For 90d validation, an empty table is often the first sign
    of a horizon wiring bug or an upstream data gap, so we surface it
    explicitly in the runner logs.
    """
    if horizon_days != 90:
        return

    empty_tables = [table for table, count in counts.items() if int(count or 0) == 0]
    if not empty_tables:
        return

    logging.warning(
        "90d run produced empty result tables: split_id=%s run_id=%s horizon_days=%s empty_tables=%s counts=%s",
        split_id,
        run_id,
        horizon_days,
        ",".join(empty_tables),
        counts,
    )


def write_post_run_summary(summary_prefix: Path, min_runs: int) -> None:
    csv_path = summary_prefix.with_suffix(".csv")
    md_path = summary_prefix.with_suffix(".md")
    cmd = [
        sys.executable,
        "python/check_walkforward_runs.py",
        "--min-runs",
        str(min_runs),
        "--out-csv",
        str(csv_path),
        "--out-md",
        str(md_path),
    ]
    logging.info("Writing post-run walkforward summary: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    setup_logging()
    args = parse_args()
    horizon_days_list = resolve_horizon_days_list(args)
    try:
        score_weights = json.loads(args.score_weights_json)
    except Exception as e:
        raise ValueError(f"Invalid --score-weights-json: {e}") from e
    if not isinstance(score_weights, dict):
        raise ValueError("--score-weights-json must decode to a JSON object")

    splits = load_splits(args.splits_csv)
    if args.start_split_id is not None:
        splits = splits[splits["split_id"] >= args.start_split_id].reset_index(drop=True)
    if args.max_splits is not None:
        if args.max_splits <= 0:
            raise ValueError("--max-splits must be > 0")
        splits = splits.head(args.max_splits).reset_index(drop=True)
    if splits.empty:
        logging.warning("No splits to run after applying filters")
        return
    logging.info(
        "Running %d walk-forward splits after filters for horizons=%s",
        len(splits),
        horizon_days_list,
    )

    success_count = 0
    failure_count = 0
    skip_count = 0
    run_summaries: list[dict] = []

    for _, split in splits.iterrows():
        split_id = int(split["split_id"])
        predict_start = split["predict_start"]
        predict_end = split["predict_end"]

        if pd.to_datetime(predict_start) > pd.to_datetime(predict_end):
            logging.warning("Skipping split_id=%s because predict_start > predict_end", split_id)
            skip_count += 1
            continue

        for horizon_days in horizon_days_list:
            run_id = create_run_for_split(
                split,
                run_type="walkforward_backtest",
                model_version=args.model_version,
                horizon_days=horizon_days,
                top_n=args.top_n,
                rebalance_freq=args.rebalance_freq,
                universe_version=args.universe_version,
                universe_mode=args.universe_mode,
                score_formula_version=args.score_formula_version,
                score_weights=score_weights,
            )
            logging.info(
                "Created dim_model_run for split_id=%s run_id=%s horizon_days=%s train=%s~%s predict=%s~%s",
                split_id,
                run_id,
                horizon_days,
                split["train_start"],
                split["train_end"],
                predict_start,
                predict_end,
            )

            try:
                run_command(
                    [
                        sys.executable,
                        "python/build_backtest_predictions.py",
                        "--features-csv",
                        str(args.features_csv),
                        "--model-pkl",
                        str(args.model_pkl),
                        "--start-date",
                        predict_start,
                        "--end-date",
                        predict_end,
                        "--run-id",
                        str(run_id),
                        "--model-version",
                        args.model_version,
                        "--horizon-days",
                        str(horizon_days),
                    ],
                    split_id=split_id,
                    run_id=run_id,
                    horizon_days=horizon_days,
                )
                run_command(
                    [
                        sys.executable,
                        "python/build_backtest_ranking.py",
                        "--run-id",
                        str(run_id),
                        "--start-date",
                        predict_start,
                        "--end-date",
                        predict_end,
                        "--top-n",
                        str(args.top_n),
                    ],
                    split_id=split_id,
                    run_id=run_id,
                    horizon_days=horizon_days,
                )
                run_command(
                    [
                        sys.executable,
                        "python/build_backtest_outcome.py",
                        "--run-id",
                        str(run_id),
                        "--horizon-days",
                        str(horizon_days),
                        "--start-date",
                        predict_start,
                        "--end-date",
                        predict_end,
                    ],
                    split_id=split_id,
                    run_id=run_id,
                    horizon_days=horizon_days,
                )
                counts = collect_run_row_counts(run_id)
                log_empty_horizon_warning(
                    split_id=split_id,
                    run_id=run_id,
                    horizon_days=horizon_days,
                    counts=counts,
                )
                run_summaries.append(
                    {
                        "split_id": split_id,
                        "run_id": run_id,
                        "horizon_days": horizon_days,
                        **counts,
                    }
                )
                success_count += 1
            except Exception:
                logging.exception(
                    "Walk-forward split failed: split_id=%s run_id=%s horizon_days=%s",
                    split_id,
                    run_id,
                    horizon_days,
                )
                failure_count += 1

    logging.info(
        "Walk-forward run summary: success=%d failure=%d skipped=%d total=%d",
        success_count,
        failure_count,
        skip_count,
        len(splits),
    )

    if run_summaries:
        summary_df = pd.DataFrame(run_summaries)
        print(summary_df.to_string(index=False))

    write_post_run_summary(args.summary_prefix, args.summary_min_runs)


if __name__ == "__main__":
    main()
