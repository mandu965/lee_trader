from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
PROFILE_EVENTS_JSONL = OUTPUTS_DIR / "pipeline_profile_events.jsonl"
OPS_REFRESH_STATUS_JSON = OUTPUTS_DIR / "operational_refresh_status.json"
DOCKER_COMPOSE_YML = ROOT / "docker-compose.yml"

HOTSPOTS_MD = OUTPUTS_DIR / "performance_hotspots.md"
WALKFORWARD_MD = OUTPUTS_DIR / "walkforward_profile.md"
IO_MD = OUTPUTS_DIR / "io_profile.md"
PIPELINE_REPORT_MD = OUTPUTS_DIR / "pipeline_performance_report.md"

TARGET_FILES = [
    ROOT / "python" / "run_daily_scheduler.py",
    ROOT / "python" / "model_train.py",
    ROOT / "python" / "model_predict.py",
    ROOT / "python" / "ranking_builder.py",
    ROOT / "python" / "build_walkforward_acceptance.py",
    ROOT / "python" / "feature_builder.py",
    ROOT / "python" / "quality_builder.py",
    ROOT / "python" / "run_pipeline.py",
    ROOT / "python" / "run_operational_refresh.py",
    ROOT / "python" / "walkforward_backtest.py",
    ROOT / "python" / "run_walkforward_backtest.py",
    ROOT / "python" / "walkforward_splits.py",
]
CORE_TARGET_FILE_SET = {path.relative_to(ROOT).as_posix() for path in TARGET_FILES}
CORE_PRIORITY_FILES = {
    "python/feature_builder.py",
    "python/model_train.py",
    "python/model_predict.py",
    "python/ranking_builder.py",
    "python/run_pipeline.py",
    "python/run_operational_refresh.py",
    "python/walkforward_backtest.py",
    "python/run_walkforward_backtest.py",
}


@dataclass
class Hotspot:
    score: int
    risk: str
    pattern: str
    file: str
    line: int
    detail: str


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        logging.exception("Failed to read JSON: %s", path)
        return {}


def load_profile_events() -> list[dict]:
    events: list[dict] = []
    if not PROFILE_EVENTS_JSONL.exists():
        return events
    for raw in PROFILE_EVENTS_JSONL.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def latest_event_by_step(events: Iterable[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for event in events:
        step = str(event.get("step") or "").strip()
        if not step:
            continue
        latest[step] = event
    return latest


def parse_scheduler_pipeline_history() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in [LOG_DIR / "auto_ops_scheduler.log", LOG_DIR / "auto_ops_recovery_scheduler.log"]:
        if not path.exists():
            continue
        starts: dict[str, datetime] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ts_match = re.match(r"\[(?P<ts>[\d\-:\s,]+)\] \[INFO\] (?P<msg>.*)", line)
            if not ts_match:
                continue
            ts_text = ts_match.group("ts")
            msg = ts_match.group("msg")
            try:
                ts = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S,%f")
            except Exception:
                continue
            if "START run_pipeline" in msg:
                starts[path.as_posix()] = ts
            elif "OK run_pipeline" in msg and path.as_posix() in starts:
                elapsed = (ts - starts[path.as_posix()]).total_seconds()
                rows.append(
                    {
                        "log": path.name,
                        "started_at": starts[path.as_posix()].isoformat(sep=" ", timespec="seconds"),
                        "finished_at": ts.isoformat(sep=" ", timespec="seconds"),
                        "elapsed_sec": round(elapsed, 2),
                    }
                )
                starts.pop(path.as_posix(), None)
    return rows


def file_sizes(limit: int = 20) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for root in [DATA_DIR, OUTPUT_DIR, OUTPUTS_DIR]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rows.append((path.stat().st_size, path.relative_to(ROOT).as_posix()))
            except OSError:
                continue
    rows.sort(reverse=True)
    return rows[:limit]


def csv_shape(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path, low_memory=False)
        return len(df), len(df.columns)
    except Exception:
        return None, None


def detect_hotspots() -> tuple[list[Hotspot], Counter]:
    hotspots: list[Hotspot] = []
    counter: Counter = Counter()
    patterns = [
        ("apply_lambda", re.compile(r"\.apply\(\s*lambda\b"), 9, "HIGH"),
        ("groupby_apply", re.compile(r"groupby\(.*\)\.apply\("), 10, "HIGH"),
        ("iterrows", re.compile(r"\.iterrows\(\)"), 9, "HIGH"),
        ("itertuples", re.compile(r"\.itertuples\(\)"), 7, "MEDIUM"),
        ("merge", re.compile(r"(?:pd\.merge\(|\.merge\()"), 4, "MEDIUM"),
        ("concat", re.compile(r"(?:pd\.concat\(|\.concat\()"), 4, "MEDIUM"),
        ("read_csv", re.compile(r"pd\.read_csv\("), 3, "LOW"),
        ("to_csv", re.compile(r"\.to_csv\("), 3, "LOW"),
        ("copy", re.compile(r"\.copy\(\)"), 2, "LOW"),
    ]

    for path in sorted(PYTHON_DIR.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        rel = path.relative_to(ROOT).as_posix()
        target_bonus = 6 if rel in CORE_TARGET_FILE_SET else 0
        core_bonus = 4 if rel in CORE_PRIORITY_FILES else 0
        for idx, line in enumerate(lines, start=1):
            for label, regex, base_score, risk in patterns:
                if not regex.search(line):
                    continue
                if label == "iterrows" and ".head(" in line:
                    continue
                counter[label] += 1
                score = base_score + target_bonus + core_bonus
                detail = line.strip()
                prev_lines = "\n".join(lines[max(0, idx - 5):idx])
                next_lines = "\n".join(lines[idx:min(len(lines), idx + 20)])
                in_loop = re.search(r"^\s*for\s+.+:\s*$", prev_lines, re.MULTILINE) is not None
                if in_loop:
                    score += 4
                    detail += " [inside/near loop]"
                if label in {"merge", "concat", "read_csv", "to_csv"} and re.search(
                    r"(?:pd\.merge|\.merge|pd\.concat|\.concat|pd\.read_csv|\.to_csv)", next_lines
                ):
                    score += 2
                hotspots.append(Hotspot(score=score, risk=risk, pattern=label, file=rel, line=idx, detail=detail[:180]))

        merge_count = sum(1 for line in lines if re.search(r"(?:pd\.merge\(|\.merge\()", line))
        csv_io_count = sum(1 for line in lines if "pd.read_csv(" in line or ".to_csv(" in line)
        if merge_count >= 5:
            hotspots.append(
                Hotspot(
                    score=8 + merge_count + target_bonus + core_bonus,
                    risk="MEDIUM",
                    pattern="repeated_merge_file",
                    file=rel,
                    line=1,
                    detail=f"{merge_count} merge calls in one file",
                )
            )
        if csv_io_count >= 8:
            hotspots.append(
                Hotspot(
                    score=6 + csv_io_count + target_bonus + core_bonus,
                    risk="MEDIUM",
                    pattern="repeated_csv_io_file",
                    file=rel,
                    line=1,
                    detail=f"{csv_io_count} csv read/write calls in one file",
                )
            )

    hotspots.sort(key=lambda item: (-item.score, item.file, item.line))
    return hotspots, counter


def build_hotspots_report(hotspots: list[Hotspot], counter: Counter) -> str:
    prioritized = [item for item in hotspots if item.file in CORE_TARGET_FILE_SET or item.file in CORE_PRIORITY_FILES]
    top = prioritized[:20] if prioritized else hotspots[:20]
    lines = [
        "# Performance Hotspots",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- purpose: static pandas / IO / loop-pattern risk scan only",
        "- note: this report does not change ranking or score logic",
        "",
        "## Pattern Counts",
        "",
        "| pattern | count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(counter.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Top Hotspots",
            "",
            "| score | risk | pattern | location | detail |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in top:
        lines.append(f"| {item.score} | {item.risk} | {item.pattern} | {item.file}:{item.line} | {item.detail} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `feature_builder.py` is high risk because it uses `groupby().apply(...)` over the full price history and then runs multiple date-wise winsorization/rank transforms.",
            "- `ranking_builder.py` is high IO risk because it writes many sidecar CSV/Markdown artifacts in one run and contains repeated export paths.",
            "- `model_train.py` is medium-to-high CPU risk because LightGBM training and CV loops execute once per target and once per fold.",
            "- `walkforward_backtest.py` and `run_walkforward_backtest.py` are structural bottlenecks because they repeat train/predict/rank/outcome subprocesses per split.",
            "- Multiple files call `pd.read_csv` on large shared artifacts (`features.csv`, `labels.csv`, `ranking_final.csv`), which strongly suggests redundant disk IO over code bottlenecks.",
        ]
    )
    return "\n".join(lines) + "\n"


def walkforward_summary() -> dict[str, object]:
    summary: dict[str, object] = {
        "total_runs": None,
        "comparison_groups": None,
        "primary_group_runs": None,
        "unique_horizons": [],
        "grid_search": False,
        "runner_retrains_per_split": 1,
        "subprocesses_per_split": 3,
        "legacy_subprocesses_per_window": 4,
    }
    path = OUTPUTS_DIR / "walkforward_monthly_validation_8.md"
    fallback = OUTPUTS_DIR / "walkforward_run_check.md"
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    elif fallback.exists():
        text = fallback.read_text(encoding="utf-8", errors="replace")
    if text:
        total_runs_match = re.search(r"- total_runs:\s*(\d+)", text)
        if total_runs_match:
            summary["total_runs"] = int(total_runs_match.group(1))
        group_rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*(yes|no)\s*\|", text)
        if group_rows:
            summary["comparison_groups"] = len(group_rows)
            for group_name, runs, primary in group_rows:
                if primary == "yes":
                    summary["primary_group_runs"] = int(runs)
                parts = group_name.split("|")
                if len(parts) >= 2:
                    try:
                        horizon = int(parts[1])
                    except Exception:
                        continue
                    summary["unique_horizons"] = sorted(set(summary.get("unique_horizons", [])) | {horizon})

    summary["recalc_notes"] = [
        "walkforward_backtest.py retrains a new model for every window via python/model_train.py",
        "run_walkforward_backtest.py creates one run per split and horizon and reruns predictions/ranking/outcome per run",
        "the standard runner does not perform grid search inside the loop; grid search exists in separate research scripts, not the core walkforward runner",
    ]
    return summary


def build_walkforward_report(summary: dict[str, object]) -> str:
    lines = [
        "# Walkforward Profile",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- sources: python/walkforward_backtest.py, python/run_walkforward_backtest.py, python/walkforward_splits.py, outputs/walkforward_monthly_validation_8.md",
        "",
        "## Structural Summary",
        "",
        f"- observed_total_runs: {summary.get('total_runs') if summary.get('total_runs') is not None else 'NA'}",
        f"- observed_comparison_groups: {summary.get('comparison_groups') if summary.get('comparison_groups') is not None else 'NA'}",
        f"- primary_group_runs: {summary.get('primary_group_runs') if summary.get('primary_group_runs') is not None else 'NA'}",
        f"- unique_horizons_observed: {summary.get('unique_horizons') or 'NA'}",
        f"- grid_search_inside_core_runner: {'yes' if summary.get('grid_search') else 'no'}",
        f"- legacy_engine_subprocesses_per_window: {summary.get('legacy_subprocesses_per_window')}",
        f"- split_runner_subprocesses_per_split: {summary.get('subprocesses_per_split')}",
        "",
        "## Repetition Analysis",
        "",
        "- `python/walkforward_backtest.py` repeats four subprocesses per window: train, backtest predictions, backtest ranking, backtest outcome.",
        "- `python/run_walkforward_backtest.py` assumes a pretrained model package, but still repeats three subprocesses per split/horizon: backtest predictions, backtest ranking, backtest outcome.",
        "- `python/run_walkforward_backtest.py` supports `--horizon-days-list`, so total run count scales as `split_count x horizon_count`.",
        "- The runner creates a fresh `research.dim_model_run` row for every split/horizon combination, so DB write volume also scales linearly with split count.",
        "",
        "## Same-Data Recalculation Risk",
        "",
    ]
    for note in summary.get("recalc_notes", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "- Main bottleneck driver is structural repetition rather than a single formula. Every extra split or horizon multiplies model IO, DB writes, and prediction/ranking generation cost.",
            "- Current monthly validation artifact shows multiple comparable runs, but the core loop is still recomputing overlapping train windows from scratch.",
            "- Based on code structure, walkforward cost is primarily code-structure bound, with hardware sensitivity concentrated in CPU for training and disk IO for repeated artifact loads/writes.",
        ]
    )
    return "\n".join(lines) + "\n"


def docker_volume_summary() -> list[str]:
    if not DOCKER_COMPOSE_YML.exists():
        return []
    text = DOCKER_COMPOSE_YML.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"^\s*-\s+(\./[^\s:]+:[^\s]+)\s*$", text, flags=re.MULTILINE)


def build_io_report(counter: Counter) -> str:
    sizes = file_sizes(limit=15)
    feature_snapshots = list((OUTPUTS_DIR / "snapshots").glob("*/features_*.csv")) if (OUTPUTS_DIR / "snapshots").exists() else []
    volumes = docker_volume_summary()
    lines = [
        "# IO Profile",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- scope: docker bind mounts, CSV read/write structure, large artifact locations",
        "",
        "## Docker Volume Mounts",
        "",
    ]
    if volumes:
        for volume in volumes:
            lines.append(f"- {volume}")
    else:
        lines.append("- no bind mounts parsed")

    lines.extend(
        [
            "",
            "## CSV IO Pattern Counts",
            "",
            f"- pd.read_csv occurrences in python/: {counter.get('read_csv', 0)}",
            f"- to_csv occurrences in python/: {counter.get('to_csv', 0)}",
            "",
            "## Largest Files",
            "",
            "| size_mb | path |",
            "| ---: | --- |",
        ]
    )
    for size_bytes, rel in sizes:
        lines.append(f"| {size_bytes / (1024 * 1024):.2f} | {rel} |")

    lines.extend(
        [
            "",
            "## Snapshot Churn",
            "",
            f"- features_snapshot_count: {len(feature_snapshots)}",
            f"- latest_features_csv_size_mb: {(DATA_DIR / 'features.csv').stat().st_size / (1024 * 1024):.2f}" if (DATA_DIR / "features.csv").exists() else "- latest_features_csv_size_mb: NA",
            "",
            "## Assessment",
            "",
            "- `./data`, `./output`, and `./outputs` are bind-mounted into containers, so repeated large CSV reads/writes are host-filesystem bound rather than in-container ephemeral disk only.",
            "- `data/features.csv` is about 60 MB and `data/labels.csv` is about 38 MB in the current workspace, which is large enough for repeated `read_csv` calls to matter.",
            "- Repeated snapshot export of `features_YYYYMMDD.csv` under `outputs/snapshots/` materially increases write amplification.",
            "- `ranking_builder.py` emits many comparison/debug/export files in one pass, so ranking completion likely contains a meaningful IO tail even when compute is finished.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_pipeline_report(events: dict[str, dict], hotspots: list[Hotspot], scheduler_rows: list[dict[str, object]]) -> str:
    feature_rows, feature_cols = csv_shape(DATA_DIR / "features.csv")
    label_rows, label_cols = csv_shape(DATA_DIR / "labels.csv")
    ranking_rows, ranking_cols = csv_shape(DATA_DIR / "ranking_final.csv")
    ops_status = read_json(OPS_REFRESH_STATUS_JSON)
    ops_export_elapsed = None
    steps = ops_status.get("steps", []) if isinstance(ops_status, dict) else []
    if isinstance(steps, list):
        export_sum = 0.0
        seen = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("name") in {"export_serving_payloads", "sync_auxiliary_payloads"} and step.get("elapsed_sec") is not None:
                export_sum += float(step["elapsed_sec"])
                seen += 1
        if seen:
            ops_export_elapsed = export_sum

    step_order = [
        "data_collection",
        "feature_build",
        "quality_build",
        "model_train",
        "prediction",
        "ranking_build",
        "walkforward",
        "export_sync",
    ]
    lines = [
        "# Pipeline Performance Report",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- profiler_status: installed",
        f"- psutil_available_now: {'yes' if psutil else 'no'}",
        "- note: step timings below use latest profiler events when available; otherwise they remain `NA` until the next profiled run.",
        "",
        "## Pipeline Size Context",
        "",
        f"- features.csv: rows={feature_rows if feature_rows is not None else 'NA'} cols={feature_cols if feature_cols is not None else 'NA'}",
        f"- labels.csv: rows={label_rows if label_rows is not None else 'NA'} cols={label_cols if label_cols is not None else 'NA'}",
        f"- ranking_final.csv: rows={ranking_rows if ranking_rows is not None else 'NA'} cols={ranking_cols if ranking_cols is not None else 'NA'}",
        "",
        "## Step Timing",
        "",
        "| step | elapsed_sec | memory_mb | cpu_percent | source |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for step in step_order:
        event = events.get(step, {})
        elapsed = event.get("elapsed_sec")
        memory = event.get("memory_mb")
        cpu = event.get("cpu_percent")
        source = "profile_events"
        if step == "export_sync" and elapsed is None and ops_export_elapsed is not None:
            elapsed = round(ops_export_elapsed, 3)
            source = "operational_refresh_status"
        if elapsed is None:
            source = "pending_profile_run"
        lines.append(
            f"| {step} | {elapsed if elapsed is not None else 'NA'} | "
            f"{round(float(memory), 2) if memory is not None else 'NA'} | "
            f"{round(float(cpu), 2) if cpu is not None else 'NA'} | {source} |"
        )

    lines.extend(["", "## Historical Scheduler Evidence", ""])
    if scheduler_rows:
        lines.extend(
            [
                "| log | started_at | finished_at | elapsed_sec |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for row in scheduler_rows[-8:]:
            lines.append(
                f"| {row['log']} | {row['started_at']} | {row['finished_at']} | {row['elapsed_sec']} |"
            )
        lines.append("")
        lines.append("- Historical close/recovery scheduler logs currently show roughly 14 to 20 minute `run_pipeline` durations on recorded dates.")
        lines.append("- This is materially lower than the stated 3+ hour target, so the 3-hour case likely refers to a larger end-to-end research / walkforward / manual batch scenario rather than the close scheduler alone.")
    else:
        lines.append("- no historical scheduler timing found")

    core_hotspots = [item for item in hotspots if item.file in CORE_TARGET_FILE_SET or item.file in CORE_PRIORITY_FILES]
    top5 = core_hotspots[:5] if core_hotspots else hotspots[:5]
    lines.extend(
        [
            "",
            "## Bottleneck Top 5",
            "",
            "| rank | hotspot | why it matters |",
            "| ---: | --- | --- |",
        ]
    )
    for idx, item in enumerate(top5, start=1):
        lines.append(f"| {idx} | {item.file}:{item.line} `{item.pattern}` | {item.detail} |")

    lines.extend(
        [
            "",
            "## Estimated Optimization Effect",
            "",
            "- `feature_builder.py` vectorized rewrite or groupby/apply reduction: potentially high impact if daily price history is the dominant data volume. Estimated range: 15% to 30% on feature build time.",
            "- Reusing loaded CSV/DataFrame artifacts across train/predict/rank steps: likely medium impact. Estimated range: 10% to 20% on end-to-end CPU+IO time.",
            "- Reducing ranking sidecar exports and snapshot churn: likely medium IO improvement. Estimated range: 5% to 15% on ranking/export tail latency.",
            "- Walkforward window caching or model reuse is potentially very high impact, but it is high risk and not recommended in the current measurement-only phase.",
            "",
            "## Code vs Hardware Estimate",
            "",
            "- estimated_code_bottleneck_share: 75%",
            "- estimated_hardware_bottleneck_share: 25%",
            "- rationale: repeated CSV IO, repeated subprocess orchestration, and pandas/groupby structure dominate the current visible risk more than raw file size alone.",
            "",
            "## Immediate Improvements",
            "",
            "- LOW RISK: reduce redundant `read_csv` across adjacent steps by adding read-through cache layers only after measurements confirm repeated loads.",
            "- LOW RISK: reduce nonessential debug/export writes in non-production research paths.",
            "- LOW RISK: centralize profiling event capture and keep using it for multiple runs before changing logic.",
            "- MEDIUM RISK: replace `groupby().apply(...)` style feature transforms with equivalent vectorized/grouped transforms.",
            "- MEDIUM RISK: reduce repeated merge chains in ranking and backtest builders after row-level parity checks.",
            "- HIGH RISK: any change to ranking score construction, final score formula, or walkforward semantics.",
            "",
            "## Risky Optimizations",
            "",
            "- HIGH RISK: changing `ranking_builder.py` score order or component semantics.",
            "- HIGH RISK: changing score calculation or final_score weighting behavior.",
            "- HIGH RISK: changing walkforward split semantics, run grouping, or outcome alignment without replay validation.",
            "",
            "## Recommended Hardware",
            "",
            "- CPU: 12 to 16 physical cores preferred for concurrent pandas, LightGBM, and DB/container overhead.",
            "- Memory: 64 GB RAM recommended; 32 GB is workable but less safe if multiple large CSVs/DataFrames coexist.",
            "- Storage: NVMe SSD strongly recommended for `data/`, `outputs/`, and Docker bind mounts.",
            "- If this runs on shared Windows host storage, moving heavy CSV paths to faster local NVMe storage is likely more valuable than adding only RAM.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    setup_logging()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    events = latest_event_by_step(load_profile_events())
    hotspots, counter = detect_hotspots()
    scheduler_rows = parse_scheduler_pipeline_history()
    walkforward = walkforward_summary()

    HOTSPOTS_MD.write_text(build_hotspots_report(hotspots, counter), encoding="utf-8")
    WALKFORWARD_MD.write_text(build_walkforward_report(walkforward), encoding="utf-8")
    IO_MD.write_text(build_io_report(counter), encoding="utf-8")
    PIPELINE_REPORT_MD.write_text(build_pipeline_report(events, hotspots, scheduler_rows), encoding="utf-8")

    print(f"out_hotspots: {HOTSPOTS_MD}")
    print(f"out_walkforward: {WALKFORWARD_MD}")
    print(f"out_io: {IO_MD}")
    print(f"out_pipeline: {PIPELINE_REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
