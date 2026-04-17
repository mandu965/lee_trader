from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"

RUN_PIPELINE_SCRIPT = ROOT / "python" / "run_pipeline.py"
ARCHIVE_SCRIPT = ROOT / "python" / "archive_ranking_snapshot.py"
FORWARD_SCRIPT = ROOT / "python" / "build_operational_forward_returns.py"
BENCHMARK_SCRIPT = ROOT / "python" / "build_benchmark_comparison.py"
BUY_CANDIDATE_SCRIPT = ROOT / "python" / "buy_candidate_builder.py"
BUY_COMPARE_SCRIPT = ROOT / "python" / "build_buy_candidate_comparison.py"
CONFIDENCE_SCRIPT = ROOT / "python" / "calibrate_operational_confidence.py"
CONFIDENCE_V2_SCRIPT = ROOT / "python" / "build_confidence_score_v2.py"
BUYABILITY_SCRIPT = ROOT / "python" / "build_top20_buyability_report.py"
WF_ACCEPTANCE_SCRIPT = ROOT / "python" / "build_walkforward_acceptance.py"
THEME_ACCEPTANCE_SCRIPT = ROOT / "python" / "run_theme_overlay_acceptance.py"
REAL_EXECUTION_SCRIPT = ROOT / "python" / "build_real_execution_shadow_report.py"
MODEL_FAILURE_SCRIPT = ROOT / "python" / "build_model_failure_analysis.py"

STATUS_JSON = OUTPUT_DIR / "operational_daily_cycle_status.json"
REPORT_MD = OUTPUT_DIR / "operational_daily_cycle_report.md"
LOG_FILE = OUTPUT_DIR / "operational_daily_cycle.log"


@dataclass
class StepSpec:
    name: str
    command_builder: Callable[[argparse.Namespace], list[str]]
    required_inputs: list[Path]
    expected_outputs: list[Path]
    critical: bool
    wait_checker: Callable[[argparse.Namespace], tuple[bool, str]] | None = None
    skip_if_missing_inputs: bool = False
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily operational cycle after ranking production pipeline.")
    parser.add_argument("--skip-core-pipeline", action="store_true", help="Skip core run_pipeline.py and use existing artifacts.")
    parser.add_argument("--archive-as-of-date", default=None, help="Forwarded to run_pipeline.py / archive script when needed.")
    parser.add_argument("--continue-on-archive-error", action="store_true", help="Forwarded to run_pipeline.py.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for subprocess steps.")
    parser.add_argument("--log-file", type=Path, default=LOG_FILE)
    parser.add_argument("--status-json", type=Path, default=STATUS_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _check_forward_wait(_: argparse.Namespace) -> tuple[bool, str]:
    path = OUTPUT_DIR / "operational_forward_return_by_day.csv"
    if not path.exists():
        return False, ""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return True, "forward-return output is empty"
    if "maturity_state" in df.columns and df["maturity_state"].astype(str).eq("immature").all():
        return True, "all horizons are immature"
    if "matured_count" in df.columns and pd.to_numeric(df["matured_count"], errors="coerce").fillna(0).eq(0).all():
        return True, "no matured forward-return rows"
    return False, ""


def _check_benchmark_wait(_: argparse.Namespace) -> tuple[bool, str]:
    path = OUTPUT_DIR / "benchmark_comparison.csv"
    if not path.exists():
        return False, ""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return True, "benchmark comparison output is empty"
    if "dates_matured" in df.columns and pd.to_numeric(df["dates_matured"], errors="coerce").fillna(0).eq(0).all():
        return True, "benchmark horizons are still immature"
    return False, ""


def _check_confidence_wait(_: argparse.Namespace) -> tuple[bool, str]:
    path = OUTPUT_DIR / "confidence_calibration_operational_report.md"
    if not path.exists():
        return False, ""
    text = path.read_text(encoding="utf-8")
    if "insufficient_operational_history" in text or "too sparse" in text.lower():
        return True, "operational confidence history is still sparse"
    return False, ""


def _check_theme_acceptance_wait(_: argparse.Namespace) -> tuple[bool, str]:
    path = OUTPUT_DIR / "theme_overlay_acceptance_operational.md"
    if not path.exists():
        return False, ""
    text = path.read_text(encoding="utf-8")
    if "insufficient_matured_dates" in text:
        return True, "theme acceptance forward-return gate is waiting for matured dates"
    return False, ""


def build_steps(args: argparse.Namespace) -> list[StepSpec]:
    def pipeline_cmd(ns: argparse.Namespace) -> list[str]:
        cmd = [ns.python_exe, str(RUN_PIPELINE_SCRIPT)]
        if ns.archive_as_of_date:
            cmd.extend(["--archive-as-of-date", str(ns.archive_as_of_date)])
        if ns.continue_on_archive_error:
            cmd.append("--continue-on-archive-error")
        return cmd

    return [
        StepSpec(
            name="core_pipeline",
            command_builder=pipeline_cmd,
            required_inputs=[],
            expected_outputs=[DATA_DIR / "ranking_final.csv"],
            critical=True,
            description="Run the core daily research-to-ranking pipeline.",
        ),
        StepSpec(
            name="snapshot_archive",
            command_builder=lambda ns: [ns.python_exe, str(ARCHIVE_SCRIPT), "--skip-if-exists"],
            required_inputs=[DATA_DIR / "ranking_final.csv"],
            expected_outputs=[DATA_DIR / "ranking_snapshot_archive.csv"],
            critical=True,
            description="Archive the latest ranking snapshot and compact top20 archive.",
        ),
        StepSpec(
            name="forward_return_update",
            command_builder=lambda ns: [ns.python_exe, str(FORWARD_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_snapshot_archive.csv", DATA_DIR / "prices_daily_adjusted.csv"],
            expected_outputs=[OUTPUT_DIR / "operational_forward_return_report.md", OUTPUT_DIR / "operational_forward_return_by_day.csv"],
            critical=False,
            wait_checker=_check_forward_wait,
            skip_if_missing_inputs=True,
            description="Update operational forward-return tracking from archived snapshots.",
        ),
        StepSpec(
            name="benchmark_comparison",
            command_builder=lambda ns: [ns.python_exe, str(BENCHMARK_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_snapshot_archive.csv", DATA_DIR / "prices_daily_adjusted.csv"],
            expected_outputs=[OUTPUT_DIR / "benchmark_comparison_report.md", OUTPUT_DIR / "benchmark_comparison.csv"],
            critical=False,
            wait_checker=_check_benchmark_wait,
            skip_if_missing_inputs=True,
            description="Compare operational performance against benchmark baselines.",
        ),
        StepSpec(
            name="buy_candidate_generation",
            command_builder=lambda ns: [ns.python_exe, str(BUY_CANDIDATE_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv"],
            expected_outputs=[DATA_DIR / "buy_candidates_top5.csv", DATA_DIR / "buy_candidates_top8.csv", DATA_DIR / "buy_candidates_top10.csv"],
            critical=True,
            description="Generate operational buy candidate lists from latest ranking.",
        ),
        StepSpec(
            name="buy_candidate_comparison",
            command_builder=lambda ns: [ns.python_exe, str(BUY_COMPARE_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv", DATA_DIR / "buy_candidates_top5.csv", DATA_DIR / "buy_candidates_top8.csv", DATA_DIR / "buy_candidates_top10.csv"],
            expected_outputs=[OUTPUT_DIR / "buy_candidate_comparison_report.md", OUTPUT_DIR / "buy_candidate_comparison.csv"],
            critical=False,
            wait_checker=_check_forward_wait,
            skip_if_missing_inputs=True,
            description="Compare raw ranking top20 against operational buy candidate quality.",
        ),
        StepSpec(
            name="confidence_calibration",
            command_builder=lambda ns: [ns.python_exe, str(CONFIDENCE_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv", DATA_DIR / "prices_daily_adjusted.csv"],
            expected_outputs=[DATA_DIR / "confidence_calibration_operational.csv", OUTPUT_DIR / "confidence_calibration_operational_report.md"],
            critical=False,
            wait_checker=_check_confidence_wait,
            skip_if_missing_inputs=True,
            description="Refresh operational confidence calibration diagnostics.",
        ),
        StepSpec(
            name="confidence_score_v2",
            command_builder=lambda ns: [ns.python_exe, str(CONFIDENCE_V2_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv", DATA_DIR / "confidence_calibration_operational.csv"],
            expected_outputs=[DATA_DIR / "confidence_score_v2.json", OUTPUT_DIR / "confidence_score_v2.md"],
            critical=False,
            skip_if_missing_inputs=True,
            description="Build confidence v2 with alpha, execution, stability, and calibration axes.",
        ),
        StepSpec(
            name="walkforward_acceptance",
            command_builder=lambda ns: [ns.python_exe, str(WF_ACCEPTANCE_SCRIPT)],
            required_inputs=[OUTPUT_DIR / "walk_forward_score_validation.md", DATA_DIR / "confidence_calibration_operational.csv"],
            expected_outputs=[OUTPUT_DIR / "walkforward_acceptance.json", OUTPUT_DIR / "walkforward_acceptance.md"],
            critical=False,
            skip_if_missing_inputs=True,
            description="Evaluate whether current walk-forward evidence is accepted, conditional, or rejected.",
        ),
        StepSpec(
            name="top20_buyability",
            command_builder=lambda ns: [ns.python_exe, str(BUYABILITY_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv", DATA_DIR / "confidence_score_v2.json"],
            expected_outputs=[OUTPUT_DIR / "top20_buyability_report.json", OUTPUT_DIR / "top20_buyability_report.md"],
            critical=False,
            skip_if_missing_inputs=True,
            description="Classify Top20 names into buy-now, watchlist, paper-only, or block.",
        ),
        StepSpec(
            name="theme_overlay_acceptance",
            command_builder=lambda ns: [ns.python_exe, str(THEME_ACCEPTANCE_SCRIPT)],
            required_inputs=[DATA_DIR / "ranking_final.csv", DATA_DIR / "history" / "ranking"],
            expected_outputs=[OUTPUT_DIR / "theme_overlay_acceptance_operational.md"],
            critical=False,
            wait_checker=_check_theme_acceptance_wait,
            skip_if_missing_inputs=True,
            description="Generate promotion gate report for theme overlay acceptance.",
        ),
        StepSpec(
            name="real_execution_shadow",
            command_builder=lambda ns: [ns.python_exe, str(REAL_EXECUTION_SCRIPT)],
            required_inputs=[SERVING_DIR / "daily_recommendations.json", SERVING_DIR / "model_portfolio.json", SERVING_DIR / "buy_gate_status.json", DATA_DIR / "ranking_snapshot_archive.csv", DATA_DIR / "prices_daily_adjusted.csv"],
            expected_outputs=[DATA_DIR / "real_trade_log.csv", DATA_DIR / "real_vs_model_diff.csv", OUTPUT_DIR / "real_execution_report.md"],
            critical=False,
            skip_if_missing_inputs=True,
            description="Build real-money shadow execution log and paper-vs-real comparison report.",
        ),
        StepSpec(
            name="model_failure_analysis",
            command_builder=lambda ns: [ns.python_exe, str(MODEL_FAILURE_SCRIPT)],
            required_inputs=[DATA_DIR / "prices_daily_adjusted.csv", DATA_DIR / "market_status.csv", OUTPUT_DIR / "operational_buy_gate.json"],
            expected_outputs=[DATA_DIR / "model_failure_cases.csv", OUTPUT_DIR / "model_failure_analysis.md"],
            critical=False,
            skip_if_missing_inputs=True,
            description="Collect loss and underperformance failure cases from ranking history and cluster common failure patterns.",
        ),
    ]


def prerequisites_present(paths: list[Path]) -> tuple[bool, list[str]]:
    missing = []
    for path in paths:
        resolved = _resolve(path)
        if not resolved.exists():
            missing.append(str(resolved))
    return len(missing) == 0, missing


def outputs_present(paths: list[Path]) -> tuple[bool, list[str]]:
    missing = []
    for path in paths:
        resolved = _resolve(path)
        if not resolved.exists():
            missing.append(str(resolved))
    return len(missing) == 0, missing


def run_step(step: StepSpec, args: argparse.Namespace) -> dict[str, object]:
    required_ok, missing_inputs = prerequisites_present(step.required_inputs)
    started_at = datetime.now().isoformat(timespec="seconds")
    result: dict[str, object] = {
        "name": step.name,
        "description": step.description,
        "critical": step.critical,
        "required_inputs": [str(_resolve(p)) for p in step.required_inputs],
        "expected_outputs": [str(_resolve(p)) for p in step.expected_outputs],
        "missing_inputs": missing_inputs,
        "command": None,
        "status": None,
        "returncode": None,
        "wait_reason": None,
        "error": None,
        "started_at": started_at,
        "finished_at": None,
        "elapsed_sec": None,
    }

    if not required_ok:
        if step.skip_if_missing_inputs:
            result["status"] = "SKIPPED"
            result["error"] = "missing_inputs"
            result["finished_at"] = datetime.now().isoformat(timespec="seconds")
            result["elapsed_sec"] = 0.0
            logging.warning("step=%s skipped missing_inputs=%s", step.name, missing_inputs)
            return result
        result["status"] = "FAIL"
        result["error"] = f"missing_inputs: {missing_inputs}"
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result["elapsed_sec"] = 0.0
        logging.error("step=%s missing critical inputs=%s", step.name, missing_inputs)
        return result

    command = step.command_builder(args)
    result["command"] = command
    logging.info("step_start name=%s critical=%s command=%s", step.name, step.critical, " ".join(command))
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        result["returncode"] = completed.returncode
        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        outputs_ok, missing_outputs = outputs_present(step.expected_outputs)
        if completed.returncode != 0:
            result["status"] = "FAIL" if step.critical else "WARN"
            result["error"] = f"returncode={completed.returncode}"
            logging.error("step_end name=%s status=%s returncode=%s", step.name, result["status"], completed.returncode)
            return result
        if not outputs_ok:
            result["status"] = "FAIL" if step.critical else "WARN"
            result["error"] = f"missing_outputs: {missing_outputs}"
            logging.error("step_end name=%s status=%s missing_outputs=%s", step.name, result["status"], missing_outputs)
            return result
        wait = False
        wait_reason = ""
        if step.wait_checker is not None:
            wait, wait_reason = step.wait_checker(args)
        result["wait_reason"] = wait_reason or None
        result["status"] = "WAIT" if wait else "SUCCESS"
        logging.info("step_end name=%s status=%s elapsed_sec=%.3f", step.name, result["status"], result["elapsed_sec"])
        return result
    except Exception as exc:
        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result["status"] = "FAIL" if step.critical else "WARN"
        result["error"] = str(exc)
        logging.exception("step_exception name=%s", step.name)
        return result


def overall_status(step_results: list[dict[str, object]]) -> str:
    critical_fail = any(row["status"] == "FAIL" and row["critical"] for row in step_results)
    if critical_fail:
        return "FAIL"
    if any(row["status"] == "WARN" for row in step_results):
        return "WARN"
    if any(row["status"] == "WAIT" for row in step_results):
        return "WAIT"
    return "SUCCESS"


def build_dependency_rows(steps: list[StepSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": step.name,
                "critical": step.critical,
                "required_inputs": ", ".join(str(_resolve(p)) for p in step.required_inputs) if step.required_inputs else "(none)",
                "expected_outputs": ", ".join(str(_resolve(p)) for p in step.expected_outputs) if step.expected_outputs else "(none)",
                "mode": "warning_only" if not step.critical else "stop_on_fail",
            }
            for step in steps
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def build_report(args: argparse.Namespace, steps: list[StepSpec], results: list[dict[str, object]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dep_df = build_dependency_rows(steps)
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["elapsed_sec"] = result_df["elapsed_sec"].map(lambda x: _fmt_num(x, 3) if x is not None else "NA")
    summary = {
        "overall_status": overall_status(results),
        "success_count": int(sum(row["status"] == "SUCCESS" for row in results)),
        "wait_count": int(sum(row["status"] == "WAIT" for row in results)),
        "warn_count": int(sum(row["status"] == "WARN" for row in results)),
        "fail_count": int(sum(row["status"] == "FAIL" for row in results)),
        "skipped_count": int(sum(row["status"] == "SKIPPED" for row in results)),
    }

    lines = [
        "# Operational Daily Cycle Report",
        "",
        f"- generated_at: {generated_at}",
        f"- overall_status: {summary['overall_status']}",
        f"- skip_core_pipeline: {bool(args.skip_core_pipeline)}",
        f"- success_count: {summary['success_count']}",
        f"- wait_count: {summary['wait_count']}",
        f"- warn_count: {summary['warn_count']}",
        f"- fail_count: {summary['fail_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- log_file: {str(_resolve(args.log_file))}",
        f"- status_json: {str(_resolve(args.status_json))}",
        "",
        "## Execution Order",
        "1. core_pipeline",
        "2. snapshot_archive",
        "3. forward_return_update",
        "4. benchmark_comparison",
        "5. buy_candidate_generation",
        "6. buy_candidate_comparison",
        "7. confidence_calibration",
        "8. confidence_score_v2",
        "9. walkforward_acceptance",
        "10. top20_buyability",
        "11. theme_overlay_acceptance",
        "12. real_execution_shadow",
        "13. model_failure_analysis",
        "",
        "## Dependency Map",
        markdown_table(dep_df, ["step", "critical", "mode", "required_inputs", "expected_outputs"]),
        "",
        "## Step Results",
        markdown_table(result_df, ["name", "critical", "status", "elapsed_sec", "wait_reason", "error"]),
    ]

    waits = result_df.loc[result_df["status"].eq("WAIT")].copy() if not result_df.empty else pd.DataFrame()
    if not waits.empty:
        lines.extend(
            [
                "",
                "## WAIT States",
                markdown_table(waits, ["name", "wait_reason"]),
                "- WAIT means the step ran and produced outputs, but the result is not yet mature enough for full evaluation.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    setup_logging(_resolve(args.log_file))

    started_at = datetime.now().isoformat(timespec="seconds")
    logging.info("operational_daily_cycle_start skip_core_pipeline=%s", bool(args.skip_core_pipeline))

    all_steps = build_steps(args)
    results: list[dict[str, object]] = []
    for step in all_steps:
        if args.skip_core_pipeline and step.name == "core_pipeline":
            results.append(
                {
                    "name": step.name,
                    "description": step.description,
                    "critical": step.critical,
                    "required_inputs": [str(_resolve(p)) for p in step.required_inputs],
                    "expected_outputs": [str(_resolve(p)) for p in step.expected_outputs],
                    "missing_inputs": [],
                    "command": None,
                    "status": "SKIPPED",
                    "returncode": None,
                    "wait_reason": "skipped by --skip-core-pipeline",
                    "error": None,
                    "started_at": None,
                    "finished_at": None,
                    "elapsed_sec": 0.0,
                }
            )
            logging.info("step_skip name=core_pipeline reason=skip_core_pipeline")
            continue
        result = run_step(step, args)
        results.append(result)
        if result["status"] == "FAIL" and step.critical:
            logging.error("critical_step_failed name=%s", step.name)
            break

    finished_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "overall_status": overall_status(results),
        "log_file": str(_resolve(args.log_file)),
        "steps": results,
    }

    status_json = _resolve(args.status_json)
    report_md = _resolve(args.report_md)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_report(args, all_steps, results), encoding="utf-8")

    logging.info("operational_daily_cycle_end overall_status=%s", payload["overall_status"])
    print(f"overall_status: {payload['overall_status']}")
    print(f"status_json: {status_json}")
    print(f"report_md: {report_md}")
    return 1 if payload["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
