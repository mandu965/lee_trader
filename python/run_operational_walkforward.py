from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RUN_HISTORY_DIR = DATA_DIR / "history" / "walkforward_runs"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operational wrapper around the existing walk-forward tooling."
    )
    parser.add_argument("--run-label", default="", help="Optional human-readable run label.")
    parser.add_argument("--model-pkl", type=Path, default=DATA_DIR / "model.pkl")
    parser.add_argument("--features-csv", type=Path, default=DATA_DIR / "features.csv")
    parser.add_argument("--splits-csv", type=Path, help="Reuse an existing split schedule CSV.")
    parser.add_argument("--split-start-date", help="Generate splits from this date (YYYY-MM-DD).")
    parser.add_argument("--split-end-date", help="Generate splits until this date (YYYY-MM-DD).")
    parser.add_argument("--train-days", type=int, default=252 * 2)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--horizon-days-list", default="60,90")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--model-version", default="operational_walkforward_v1")
    parser.add_argument("--score-formula-version", default="ranking_builder_v8_return_prob_tech_regime")
    parser.add_argument("--rebalance-freq", default="monthly")
    parser.add_argument("--universe-version", default="current_data_snapshot")
    parser.add_argument("--universe-mode", default="fixed_current_universe")
    parser.add_argument("--summary-min-runs", type=int, default=8)
    parser.add_argument(
        "--score-weights-json",
        default="{}",
        help="Optional JSON object stored with dim_model_run.config_json",
    )
    parser.add_argument(
        "--skip-score-validation",
        action="store_true",
        help="Skip build_walk_forward_score_validation_from_runs.py",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    logging.info("Running command: %s", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_run_dir(label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    run_dir = RUN_HISTORY_DIR / f"{stamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_generate_splits(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.splits_csv:
        splits_csv = args.splits_csv.resolve()
        if not splits_csv.exists():
            raise FileNotFoundError(f"splits csv not found: {splits_csv}")
        copied = run_dir / "inputs" / splits_csv.name
        ensure_parent(copied)
        shutil.copy2(splits_csv, copied)
        return copied

    if not args.split_start_date or not args.split_end_date:
        raise ValueError(
            "Either --splits-csv or both --split-start-date/--split-end-date are required."
        )

    generated = run_dir / "inputs" / "walkforward_splits.csv"
    ensure_parent(generated)
    run_command(
        [
            PYTHON,
            "python/walkforward_splits.py",
            "--start-date",
            args.split_start_date,
            "--end-date",
            args.split_end_date,
            "--train-days",
            str(args.train_days),
            "--test-days",
            str(args.test_days),
            "--step-days",
            str(args.step_days),
            "--out-csv",
            str(generated),
        ]
    )
    return generated


def build_manifest(args: argparse.Namespace, run_dir: Path, splits_csv: Path) -> dict:
    score_weights = json.loads(args.score_weights_json)
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_label": args.run_label or None,
        "run_dir": str(run_dir),
        "inputs": {
            "model_pkl": str(args.model_pkl),
            "features_csv": str(args.features_csv),
            "splits_csv": str(splits_csv),
        },
        "config": {
            "horizon_days_list": args.horizon_days_list,
            "top_n": args.top_n,
            "model_version": args.model_version,
            "score_formula_version": args.score_formula_version,
            "rebalance_freq": args.rebalance_freq,
            "universe_version": args.universe_version,
            "universe_mode": args.universe_mode,
            "summary_min_runs": args.summary_min_runs,
            "score_weights": score_weights,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "split_start_date": args.split_start_date,
            "split_end_date": args.split_end_date,
        },
        "artifacts": {},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_summary_note(run_dir: Path, manifest: dict, splits_csv: Path) -> None:
    try:
        split_df = pd.read_csv(splits_csv)
        split_count = int(len(split_df))
    except Exception:
        split_count = None

    lines = [
        "# 운영형 Walk-forward 실행 메모",
        "",
        f"- 생성 시각: {manifest['created_at']}",
        f"- run_label: {manifest['run_label'] or '(none)'}",
        f"- model_version: {manifest['config']['model_version']}",
        f"- score_formula_version: {manifest['config']['score_formula_version']}",
        f"- horizon_days_list: {manifest['config']['horizon_days_list']}",
        f"- top_n: {manifest['config']['top_n']}",
        f"- rebalance_freq: {manifest['config']['rebalance_freq']}",
        f"- universe_mode: {manifest['config']['universe_mode']}",
        f"- splits_csv: {splits_csv}",
    ]
    if split_count is not None:
        lines.append(f"- split_count: {split_count}")

    lines.extend(
        [
            "",
            "## 산출물",
            "",
            "- `inputs/`: split schedule 사본",
            "- `summaries/`: check_walkforward_runs, score validation 집계",
            "- `manifest.json`: 실행 파라미터와 산출물 인덱스",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()

    try:
        json.loads(args.score_weights_json)
    except Exception as exc:
        raise ValueError(f"Invalid --score-weights-json: {exc}") from exc

    run_dir = resolve_run_dir(args.run_label.strip())
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    splits_csv = maybe_generate_splits(args, run_dir)
    manifest = build_manifest(args, run_dir, splits_csv)
    write_summary_note(run_dir, manifest, splits_csv)

    summary_prefix = summaries_dir / "walkforward_run_summary"
    run_command(
        [
            PYTHON,
            "python/run_walkforward_backtest.py",
            "--splits-csv",
            str(splits_csv),
            "--model-pkl",
            str(args.model_pkl),
            "--features-csv",
            str(args.features_csv),
            "--model-version",
            args.model_version,
            "--horizon-days-list",
            args.horizon_days_list,
            "--top-n",
            str(args.top_n),
            "--rebalance-freq",
            args.rebalance_freq,
            "--universe-version",
            args.universe_version,
            "--universe-mode",
            args.universe_mode,
            "--score-formula-version",
            args.score_formula_version,
            "--summary-prefix",
            str(summary_prefix),
            "--summary-min-runs",
            str(args.summary_min_runs),
            "--score-weights-json",
            args.score_weights_json,
        ]
    )
    manifest["artifacts"]["walkforward_summary_csv"] = str(summary_prefix.with_suffix(".csv"))
    manifest["artifacts"]["walkforward_summary_md"] = str(summary_prefix.with_suffix(".md"))

    if not args.skip_score_validation:
        validation_md = summaries_dir / "walk_forward_score_validation.md"
        validation_csv = summaries_dir / "walk_forward_score_validation.csv"
        run_command(
            [
                PYTHON,
                "python/build_walk_forward_score_validation_from_runs.py",
            ]
        )
        default_md = OUTPUTS_DIR / "walk_forward_score_validation.md"
        default_csv = OUTPUTS_DIR / "walk_forward_score_validation.csv"
        if default_md.exists():
            shutil.copy2(default_md, validation_md)
            manifest["artifacts"]["score_validation_md"] = str(validation_md)
        if default_csv.exists():
            shutil.copy2(default_csv, validation_csv)
            manifest["artifacts"]["score_validation_csv"] = str(validation_csv)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Operational walk-forward run complete: %s", run_dir)


if __name__ == "__main__":
    main()
