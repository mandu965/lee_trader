"""Scheduler health ledger — step-level execution recording.

Writes outputs/scheduler_health.json and outputs/scheduler_health_report.md
after each run_daily_scheduler.py cycle.  This is additive; it does NOT
replace auto_ops_scheduler_status.json.

Design constraint: every public function must swallow its own exceptions so
that health-ledger failures never propagate back to the scheduler.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HEALTH_JSON = ROOT / "outputs" / "scheduler_health.json"
_DEFAULT_HEALTH_MD = ROOT / "outputs" / "scheduler_health_report.md"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class JobResult:
    job_name: str
    command_set: str
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None
    status: str = "success"          # success | failed | skipped
    exit_code: int | None = None
    error_message: str | None = None
    output_rows: int | None = None
    output_file: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_health_config() -> dict:
    def _bool(key: str, default: bool) -> bool:
        raw = str(os.environ.get(key, "true" if default else "false")).strip().lower()
        if raw in {"1", "true", "yes", "on", "y"}:
            return True
        if raw in {"0", "false", "no", "off", "n"}:
            return False
        logging.warning("[scheduler_health] Unrecognised value for %s=%r; using default=%s", key, raw, default)
        return default

    def _path(key: str, default: Path) -> Path:
        raw = str(os.environ.get(key, "")).strip()
        if not raw:
            return default
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    return {
        "enabled": _bool("SCHEDULER_HEALTH_ENABLED", True),
        "json_path": _path("SCHEDULER_HEALTH_JSON", _DEFAULT_HEALTH_JSON),
        "report_path": _path("SCHEDULER_HEALTH_REPORT_MD", _DEFAULT_HEALTH_MD),
        "notify_on_failure": _bool("SCHEDULER_HEALTH_NOTIFY_ON_FAILURE", True),
    }


# ---------------------------------------------------------------------------
# Output metadata inference
# ---------------------------------------------------------------------------

_OUTPUT_MAP: dict[str, tuple[str, str]] = {
    "run_operational_refresh":       ("data/ranking_final.csv",                    "csv"),
    "submit_live_orders":            ("outputs/order_requests_execution.json",      "json"),
    "sync_live_account_holdings":    ("data/live_account_holdings.csv",             "csv"),
    "update_ai_position_state":      ("outputs/ai_position_state.json",            "json"),
    "sync_live_order_fills":         ("outputs/order_fill_sync.json",              "json"),
    "build_live_trade_consistency_report": ("outputs/live_trade_consistency_report.json", "json"),
    "build_live_trade_review":       ("outputs/live_trade_review.json",             "json"),
    "build_live_trade_review_summary": ("outputs/live_trade_review_summary.json",  "json"),
    "build_live_kpi_daily_report":   ("outputs/live_kpi_daily_report.json",        "json"),
    "build_live_closed_trade_report": ("outputs/live_closed_trade_report.json",    "json"),
    "build_quality_risk_guard_live_review": ("outputs/quality_risk_guard_live_review.json", "json"),
    "check_live_quality_guard_outputs": ("outputs/live_quality_guard_check.json",  "json"),
    "run_rule_after_close_cycle":    ("data/rule_signals.csv",                     "csv"),
    "run_rule_before_open_cycle":    ("outputs/rule_execution_results.json",       "json"),
    "run_rule_after_open_cycle":     ("outputs/rule_execution_fill_sync.json",     "json"),
    "run_manual_close_batch":        ("outputs/close_batch_result.json",           "json"),
    "sync_web_display_data":         ("outputs/web_display_sync.json",             "json"),
    "sync_rule_live_account_snapshot": ("outputs/rule_live_account_snapshot.json", "json"),
}


def _count_csv_rows(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        return max(0, len(rows) - 1)  # subtract header
    except Exception:
        return None


def _count_json_rows(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("items", "orders", "results", "jobs", "fills", "records"):
                if isinstance(data.get(key), list):
                    return len(data[key])
            return len(data)
    except Exception:
        pass
    return None


def infer_output_metadata(job_name: str) -> dict:
    entry = _OUTPUT_MAP.get(job_name)
    if entry is None:
        return {"output_file": None, "output_rows": None, "notes": "no_output_mapping"}

    rel_path, fmt = entry
    abs_path = ROOT / rel_path
    output_file = rel_path

    try:
        if not abs_path.exists():
            return {"output_file": output_file, "output_rows": None, "notes": "output_file_not_found"}

        if fmt == "csv":
            rows = _count_csv_rows(abs_path)
        else:
            rows = _count_json_rows(abs_path)

        if rows is None:
            return {"output_file": output_file, "output_rows": None, "notes": "row_count_unavailable"}

        notes = f"{Path(rel_path).name} rows={rows}"
        if job_name == "submit_live_orders":
            notes = _submit_orders_notes(abs_path, rows)
        return {"output_file": output_file, "output_rows": rows, "notes": notes}
    except Exception:
        return {"output_file": output_file, "output_rows": None, "notes": "row_count_unavailable"}


def _submit_orders_notes(path: Path, fallback_rows: int) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("items") or data.get("orders") or data.get("results") or []
        else:
            items = []
        submitted = sum(1 for x in items if isinstance(x, dict) and str(x.get("status", "")).lower() == "submitted")
        blocked = sum(1 for x in items if isinstance(x, dict) and str(x.get("status", "")).lower() == "blocked")
        failed = sum(1 for x in items if isinstance(x, dict) and str(x.get("status", "")).lower() == "failed")
        return f"submitted={submitted}, blocked={blocked}, failed={failed}"
    except Exception:
        return f"rows={fallback_rows}"


# ---------------------------------------------------------------------------
# record_job
# ---------------------------------------------------------------------------

def record_job(result: JobResult, path: Path | None = None) -> None:
    cfg = get_health_config()
    if not cfg["enabled"]:
        return
    try:
        _record_job_inner(result, path or cfg["json_path"])
    except Exception:
        logging.warning("[scheduler_health] record_job failed — health ledger write skipped", exc_info=True)


def _record_job_inner(result: JobResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    jobs: list[dict] = data.get("jobs") or []

    key = (result.command_set, result.job_name, result.started_at)
    existing_idx = next(
        (i for i, j in enumerate(jobs)
         if (j.get("command_set"), j.get("job_name"), j.get("started_at")) == key),
        None,
    )
    entry = asdict(result)
    if existing_idx is not None:
        jobs[existing_idx] = entry
    else:
        jobs.append(entry)

    data["jobs"] = jobs
    data["generated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# write_health_summary
# ---------------------------------------------------------------------------

def write_health_summary(
    date: str,
    command_set: str,
    path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    cfg = get_health_config()
    if not cfg["enabled"]:
        return
    try:
        _write_health_summary_inner(date, command_set, path or cfg["json_path"], report_path or cfg["report_path"])
    except Exception:
        logging.warning("[scheduler_health] write_health_summary failed", exc_info=True)


def _write_health_summary_inner(date: str, command_set: str, json_path: Path, md_path: Path) -> None:
    if not json_path.exists():
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    jobs: list[dict] = data.get("jobs") or []

    cs_jobs = [j for j in jobs if j.get("command_set") == command_set]

    successes = [j for j in cs_jobs if j.get("status") == "success"]
    failures  = [j for j in cs_jobs if j.get("status") == "failed"]
    skipped   = [j for j in cs_jobs if j.get("status") == "skipped"]

    durations = [j["duration_seconds"] for j in cs_jobs if j.get("duration_seconds") is not None]
    total_dur = round(sum(durations), 1) if durations else None

    started_ats  = sorted(j["started_at"] for j in cs_jobs if j.get("started_at"))
    finished_ats = sorted(j["finished_at"] for j in cs_jobs if j.get("finished_at"))

    summary = {
        "total_jobs": len(cs_jobs),
        "success": len(successes),
        "failed": len(failures),
        "skipped": len(skipped),
        "total_duration_seconds": total_dur,
        "first_started_at": started_ats[0] if started_ats else None,
        "last_finished_at": finished_ats[-1] if finished_ats else None,
    }

    data.update({"date": date, "command_set": command_set, "summary": summary})
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown_report(date, command_set, cs_jobs, summary, md_path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_STATUS_ICON = {"success": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]"}


def _write_markdown_report(
    date: str,
    command_set: str,
    jobs: list[dict],
    summary: dict,
    md_path: Path,
) -> None:
    total_dur = summary.get("total_duration_seconds")
    dur_label = f"{total_dur}초" if total_dur is not None else "-"

    lines: list[str] = [
        f"# Scheduler Health Report - {date}",
        "",
        f"command_set: {command_set} | 총 소요: {dur_label}",
        "",
        "| job | 상태 | 소요(초) | 산출물 | 비고 |",
        "|---|---:|---:|---|---|",
    ]

    for j in jobs:
        icon = _STATUS_ICON.get(str(j.get("status", "")), "?")
        status_label = f"{icon} {j.get('status', '-')}"
        dur = j.get("duration_seconds")
        dur_str = str(round(dur, 1)) if dur is not None else "-"
        output_file = j.get("output_file") or "-"
        rows = j.get("output_rows")
        if output_file != "-" and rows is not None:
            output_str = f"{output_file} / {rows} rows"
        else:
            output_str = output_file
        notes = j.get("notes") or ""
        if j.get("status") == "skipped" and not notes:
            notes = "previous step failed"
        elif j.get("status") == "failed" and not notes and j.get("error_message"):
            notes = str(j["error_message"])[:120]
        lines.append(f"| {j.get('job_name', '-')} | {status_label} | {dur_str} | {output_str} | {notes} |")

    failure_lines: list[str] = [
        "",
        "## 실패 상세",
        "",
    ]
    failed_jobs = [j for j in jobs if j.get("status") == "failed"]
    if failed_jobs:
        for j in failed_jobs:
            msg = j.get("error_message") or "unknown error"
            failure_lines.append(f"- {j.get('job_name', '-')}: {msg}")
    else:
        failure_lines.append("없음")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines + failure_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Failure notification
# ---------------------------------------------------------------------------

def notify_scheduler_failure_if_enabled(result: JobResult) -> None:
    cfg = get_health_config()
    if not cfg["enabled"]:
        return
    if not cfg["notify_on_failure"]:
        return
    if result.status != "failed":
        return
    try:
        from notifier import notify_warning  # type: ignore[import-not-found]
        notify_warning(
            f"Scheduler step failed: {result.job_name}",
            f"command_set={result.command_set} exit_code={result.exit_code}",
            {
                "job_name": result.job_name,
                "command_set": result.command_set,
                "exit_code": result.exit_code,
                "error_message": result.error_message,
            },
        )
    except Exception:
        logging.warning(
            "[scheduler_health] notify_scheduler_failure skipped: job=%s err=%s",
            result.job_name,
            result.error_message,
        )
