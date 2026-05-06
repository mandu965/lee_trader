"""
run_backtest_experiments.py

Sequential experiment runner for RULE backtest parameter grids.
It creates isolated output directories per experiment, runs the backtest,
then runs the analysis step, and finally aggregates comparable metrics into
summary CSV/JSON/Markdown reports.

This script is research automation only. It must never call live trading APIs
and must not overwrite existing production rule backtest artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "backtest_experiments"


@dataclass
class ExperimentSpec:
    """One experiment parameter set."""

    experiment_id: str
    stop_loss: float
    trailing_stop: float
    max_holding_days: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run rule backtest experiment grid.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", default="rule", choices=["rule"], help="Current experiment backend supports rule only.")
    parser.add_argument("--initial-cash", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-loss-values", type=str, default="0.03,0.04,0.05,0.06,0.07")
    parser.add_argument("--trailing-stop-values", type=str, default="0.03,0.04,0.05")
    parser.add_argument("--max-holding-days-values", type=str, default="5,10,15,20")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve paths relative to project root."""
    return path if path.is_absolute() else ROOT / path


def _parse_float_list(text: str) -> list[float]:
    """Parse comma-separated float values."""
    values: list[float] = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _parse_int_list(text: str) -> list[int]:
    """Parse comma-separated integer values."""
    values: list[int] = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def build_grid(args: argparse.Namespace) -> list[ExperimentSpec]:
    """Build the experiment parameter grid."""
    stop_losses = _parse_float_list(args.stop_loss_values)
    trailing_stops = _parse_float_list(args.trailing_stop_values)
    max_holding_days_values = _parse_int_list(args.max_holding_days_values)

    specs: list[ExperimentSpec] = []
    combinations = list(itertools.product(stop_losses, trailing_stops, max_holding_days_values))
    if args.limit is not None:
        combinations = combinations[: max(args.limit, 0)]

    for idx, (stop_loss, trailing_stop, max_holding_days) in enumerate(combinations, start=1):
        specs.append(
            ExperimentSpec(
                experiment_id=f"exp_{idx:03d}",
                stop_loss=float(stop_loss),
                trailing_stop=float(trailing_stop),
                max_holding_days=int(max_holding_days),
            )
        )
    return specs


def _run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture stdout/stderr to files."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        return subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce to float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compute_ranking_score(row: dict[str, Any]) -> float:
    """
    Compute experiment ranking score.

    Formula:
    ranking_score =
      CAGR * 0.4
      + Sharpe * 0.3
      - abs(MDD) * 0.2
      + win_rate * 0.1

    All return-like values are treated as decimal fractions, not percent units.
    """
    cagr = _safe_float(row.get("CAGR"), 0.0)
    sharpe = _safe_float(row.get("Sharpe"), 0.0)
    mdd = abs(_safe_float(row.get("MDD"), 0.0))
    win_rate = _safe_float(row.get("win_rate"), 0.0)
    return cagr * 0.4 + sharpe * 0.3 - mdd * 0.2 + win_rate * 0.1


def _collect_summary_row(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    experiment_dir: Path,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    """Collect one aggregate summary row from generated outputs."""
    row = {
        "experiment_id": spec.experiment_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "strategy": args.strategy,
        "initial_cash": float(args.initial_cash),
        "stop_loss": spec.stop_loss,
        "trailing_stop": spec.trailing_stop,
        "max_holding_days": spec.max_holding_days,
        "final_total_value": None,
        "total_return": None,
        "CAGR": None,
        "MDD": None,
        "Sharpe": None,
        "win_rate": None,
        "total_trades": None,
        "avg_return": None,
        "avg_win": None,
        "avg_loss": None,
        "payoff_ratio": None,
        "profit_factor": None,
        "ranking_score": None,
        "output_dir": str(experiment_dir),
        "status": status,
        "error_message": error_message,
    }

    if status != "completed":
        return row

    analysis_path = experiment_dir / "analysis" / "analysis_summary.json"
    if not analysis_path.exists():
        row["status"] = "failed"
        row["error_message"] = "analysis_summary.json missing"
        return row

    analysis = _load_json(analysis_path)
    row.update({
        "final_total_value": _safe_float(analysis.get("final_total_value")),
        "total_return": _safe_float(analysis.get("total_return")),
        "CAGR": _safe_float(analysis.get("cagr")),
        "MDD": _safe_float(analysis.get("mdd")),
        "Sharpe": _safe_float(analysis.get("sharpe")),
        "win_rate": _safe_float(analysis.get("win_rate")),
        "total_trades": int(_safe_float(analysis.get("total_trades"), 0.0)),
        "avg_return": _safe_float(analysis.get("avg_return")),
        "avg_win": _safe_float(analysis.get("avg_win")),
        "avg_loss": _safe_float(analysis.get("avg_loss")),
        "payoff_ratio": _safe_float(analysis.get("payoff_ratio")),
        "profit_factor": _safe_float(analysis.get("profit_factor")),
    })
    row["ranking_score"] = _compute_ranking_score(row)
    return row


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """Render a simple markdown table."""
    if df.empty:
        return "_none_"
    work = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit is not None:
        work = work.head(limit).copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


def build_experiment_report(summary_df: pd.DataFrame, args: argparse.Namespace, specs: list[ExperimentSpec]) -> str:
    """Build markdown experiment report."""
    completed = summary_df.loc[summary_df["status"] == "completed"].copy() if not summary_df.empty else pd.DataFrame()
    top10 = completed.sort_values("ranking_score", ascending=False).head(10) if not completed.empty else completed
    worst10 = completed.sort_values(["total_return", "MDD"], ascending=[True, True]).head(10) if not completed.empty else completed

    best_return_text = "_none_"
    best_risk_text = "_none_"
    if not completed.empty:
        best_return = completed.sort_values("total_return", ascending=False).iloc[0]
        best_risk = completed.sort_values("ranking_score", ascending=False).iloc[0]
        best_return_text = (
            f"- best total_return: `{best_return['experiment_id']}` "
            f"(stop_loss={best_return['stop_loss']}, trailing_stop={best_return['trailing_stop']}, "
            f"max_holding_days={best_return['max_holding_days']}, total_return={best_return['total_return']:.4f})"
        )
        best_risk_text = (
            f"- best ranking_score: `{best_risk['experiment_id']}` "
            f"(stop_loss={best_risk['stop_loss']}, trailing_stop={best_risk['trailing_stop']}, "
            f"max_holding_days={best_risk['max_holding_days']}, ranking_score={best_risk['ranking_score']:.4f})"
        )

    lines = [
        "# Backtest Experiment Report",
        "",
        "## 1. Experiment Scope",
        f"- start_date: `{args.start_date}`",
        f"- end_date: `{args.end_date}`",
        f"- strategy: `{args.strategy}`",
        f"- initial_cash: `{args.initial_cash}`",
        f"- experiment_count: `{len(specs)}`",
        "",
        "## 2. Parameter Grid",
        f"- stop_loss values: `{args.stop_loss_values}`",
        f"- trailing_stop values: `{args.trailing_stop_values}`",
        f"- max_holding_days values: `{args.max_holding_days_values}`",
        "",
        "## 3. Top 10 Configurations",
        _markdown_table(
            top10,
            ["experiment_id", "stop_loss", "trailing_stop", "max_holding_days", "total_return", "CAGR", "MDD", "Sharpe", "win_rate", "ranking_score"],
        ),
        "",
        "## 4. Worst 10 Configurations",
        _markdown_table(
            worst10,
            ["experiment_id", "stop_loss", "trailing_stop", "max_holding_days", "total_return", "MDD", "Sharpe", "ranking_score"],
        ),
        "",
        "## 5. Best Return vs Best Risk-adjusted",
        best_return_text,
        best_risk_text,
        "",
        "## 6. Observations",
        f"- completed experiments: `{int((summary_df['status'] == 'completed').sum()) if not summary_df.empty else 0}`",
        f"- failed experiments: `{int((summary_df['status'] == 'failed').sum()) if not summary_df.empty else 0}`",
        "- ranking_score is a blended metric and should not replace direct review of the drawdown path.",
        "- if many rows cluster tightly, the current exit logic may still be less influential than candidate selection quality.",
        "",
        "## 7. Recommended Next Experiments",
        "- narrower stop_loss search around top-ranked settings",
        "- compare trailing_stop sensitivity only inside the best max_holding_days bucket",
        "- run a second grid on liquidity and candidate-count constraints",
        "- once rule exit logic is richer, add sector/cooldown variants to the grid",
        "- parallel execution support after confirming deterministic outputs",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    experiments_root = output_dir / "experiments"
    specs = build_grid(args)

    if args.dry_run:
        for spec in specs:
            print(asdict(spec))
        print(f"total_experiments={len(specs)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        experiment_dir = experiments_root / spec.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        backtest_cmd = [
            sys.executable,
            "python/walk_forward_backtest.py",
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--strategy", args.strategy,
            "--initial-cash", str(args.initial_cash),
            "--output-dir", str(experiment_dir),
            "--stop-loss", str(spec.stop_loss),
            "--trailing-stop", str(spec.trailing_stop),
            "--max-holding-days", str(spec.max_holding_days),
        ]

        analysis_cmd = [
            sys.executable,
            "python/analyze_backtest_results.py",
            "--input-dir", str(experiment_dir),
            "--output-dir", str(experiment_dir / "analysis"),
        ]

        status = "completed"
        error_message = ""

        backtest_result = _run_command(
            backtest_cmd,
            experiment_dir / "backtest_stdout.log",
            experiment_dir / "backtest_stderr.log",
        )
        if backtest_result.returncode != 0:
            status = "failed"
            error_message = f"backtest failed with exit code {backtest_result.returncode}"
            rows.append(_collect_summary_row(spec, args, experiment_dir, status, error_message))
            continue

        analysis_result = _run_command(
            analysis_cmd,
            experiment_dir / "analysis_stdout.log",
            experiment_dir / "analysis_stderr.log",
        )
        if analysis_result.returncode != 0:
            status = "failed"
            error_message = f"analysis failed with exit code {analysis_result.returncode}"

        rows.append(_collect_summary_row(spec, args, experiment_dir, status, error_message))

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and "ranking_score" in summary_df.columns:
        summary_df["ranking_score"] = pd.to_numeric(summary_df["ranking_score"], errors="coerce")
        summary_df = summary_df.sort_values(["ranking_score", "status"], ascending=[False, True], na_position="last").reset_index(drop=True)

    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    summary_payload = {
        "scope": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "strategy": args.strategy,
            "initial_cash": args.initial_cash,
            "experiment_count": len(specs),
        },
        "grid": {
            "stop_loss_values": args.stop_loss_values,
            "trailing_stop_values": args.trailing_stop_values,
            "max_holding_days_values": args.max_holding_days_values,
        },
        "rows": summary_df.to_dict(orient="records"),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_report.md").write_text(
        build_experiment_report(summary_df, args, specs),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
