from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
HISTORY_DIR = BASE_DIR / "data" / "history" / "theme_shadow"
ARCHIVE_SCRIPT = BASE_DIR / "python" / "archive_ranking_snapshot.py"
SNAPSHOT_INVENTORY_SCRIPT = BASE_DIR / "python" / "report_ranking_snapshot_inventory.py"
CONFIDENCE_CALIBRATION_SCRIPT = BASE_DIR / "python" / "build_confidence_calibration_map.py"
CONFIDENCE_V2_SCRIPT = BASE_DIR / "python" / "build_confidence_score_v2.py"
TOP20_BUYABILITY_SCRIPT = BASE_DIR / "python" / "build_top20_buyability_report.py"
WALKFORWARD_ACCEPTANCE_SCRIPT = BASE_DIR / "python" / "build_walkforward_acceptance.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily theme shadow monitoring pipeline.")
    parser.add_argument("--skip-compute-theme-etf", action="store_true", help="Skip compute_theme_etf_daily.py")
    parser.add_argument("--skip-build-theme", action="store_true", help="Skip build_stock_theme_daily.py")
    parser.add_argument("--skip-ranking", action="store_true", help="Skip ranking_builder.py")
    parser.add_argument("--skip-acceptance-report", action="store_true", help="Skip build_theme_overlay_acceptance_report.py")
    parser.add_argument("--skip-monitor", action="store_true", help="Skip monitor_theme_shadow.py")
    parser.add_argument("--continue-on-archive-error", action="store_true", help="Continue even if archive_ranking_snapshot.py fails")
    parser.add_argument("--archive-as-of-date", default=None, help="Optional as_of_date forwarded to archive_ranking_snapshot.py")
    return parser.parse_args()


def run_step(step_name: str, cmd: list[str], log_buffer: list[str]) -> int:
    """Run one pipeline step and collect stdout/stderr into the shared log buffer."""
    start_line = f"[START] {step_name}"
    print(start_line)
    log_buffer.append(start_line)
    log_buffer.append(f"[CMD] {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        log_buffer.append("[STDOUT]")
        log_buffer.append(result.stdout.rstrip())
    if result.stderr:
        log_buffer.append("[STDERR]")
        log_buffer.append(result.stderr.rstrip())

    if result.returncode != 0:
        if result.stdout:
            print("[STDOUT tail]")
            print("\n".join(result.stdout.rstrip().splitlines()[-40:]))
        if result.stderr:
            print("[STDERR tail]")
            print("\n".join(result.stderr.rstrip().splitlines()[-80:]))
        fail_line = f"[FAIL] {step_name}"
        print(fail_line)
        log_buffer.append(fail_line)
        return result.returncode

    ok_line = f"[OK] {step_name}"
    print(ok_line)
    log_buffer.append(ok_line)
    return 0


def write_run_log(as_of_date: str, log_text: str) -> Path:
    """Write the full run log to the dated history folder."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    log_path = HISTORY_DIR / f"{as_of_date.replace('-', '')}_run.log"
    log_path.write_text(log_text, encoding="utf-8")
    return log_path


def _extract_archive_summary(log_buffer: list[str]) -> dict[str, str | bool | None]:
    snapshot_path = None
    snapshot_date = None
    archived = False
    archive_status = None
    for line in log_buffer:
        for raw_part in str(line).splitlines():
            text = raw_part.strip()
            if text.startswith("snapshot saved path:"):
                snapshot_path = text.split(":", 1)[1].strip()
            elif text.startswith("as_of_date:"):
                snapshot_date = text.split(":", 1)[1].strip()
            elif text.startswith("status:"):
                archive_status = text.split(":", 1)[1].strip()
    if snapshot_path and archive_status in (None, "saved", "skipped_existing"):
        archived = True
    return {
        "archived": archived,
        "snapshot_path": snapshot_path,
        "snapshot_date": snapshot_date,
        "archive_status": archive_status or "saved",
    }


def main() -> int:
    args = parse_args()
    as_of_date = datetime.now().strftime("%Y-%m-%d")
    log_buffer: list[str] = []

    steps: list[tuple[str, list[str], bool]] = [
        (
            "compute_theme_etf_daily",
            [PYTHON_EXE, str(BASE_DIR / "python" / "compute_theme_etf_daily.py")],
            args.skip_compute_theme_etf,
        ),
        (
            "build_stock_theme_daily",
            [PYTHON_EXE, str(BASE_DIR / "python" / "build_stock_theme_daily.py")],
            args.skip_build_theme,
        ),
        (
            "ranking_builder",
            [PYTHON_EXE, str(BASE_DIR / "python" / "ranking_builder.py")],
            args.skip_ranking,
        ),
        (
            "build_theme_overlay_acceptance_report",
            [PYTHON_EXE, str(BASE_DIR / "python" / "build_theme_overlay_acceptance_report.py")],
            args.skip_acceptance_report,
        ),
        (
            "monitor_theme_shadow",
            [PYTHON_EXE, str(BASE_DIR / "python" / "monitor_theme_shadow.py")],
            args.skip_monitor,
        ),
    ]

    failed_step = ""
    exit_code = 0
    archive_summary = {
        "archived": False,
        "snapshot_path": None,
        "snapshot_date": None,
    }

    for step_name, cmd, should_skip in steps:
        if should_skip:
            skip_line = f"[SKIP] {step_name}"
            print(skip_line)
            log_buffer.append(skip_line)
            continue

        exit_code = run_step(step_name, cmd, log_buffer)
        if exit_code != 0:
            failed_step = step_name
            break

    if exit_code == 0:
        archive_cmd = [PYTHON_EXE, str(ARCHIVE_SCRIPT), "--skip-if-exists"]
        if args.archive_as_of_date:
            archive_cmd.extend(["--as-of-date", str(args.archive_as_of_date)])
        archive_exit = run_step("archive_ranking_snapshot", archive_cmd, log_buffer)
        archive_summary = _extract_archive_summary(log_buffer)
        if archive_exit != 0:
            if args.continue_on_archive_error:
                warn_line = "ARCHIVE_STEP_FAILED_BUT_CONTINUED=true"
                print(warn_line)
                log_buffer.append(warn_line)
            else:
                exit_code = archive_exit
                failed_step = "archive_ranking_snapshot"
        else:
            if SNAPSHOT_INVENTORY_SCRIPT.exists():
                inventory_exit = run_step(
                    "ranking_snapshot_inventory",
                    [PYTHON_EXE, str(SNAPSHOT_INVENTORY_SCRIPT)],
                    log_buffer,
                )
                if inventory_exit != 0:
                    print("SNAPSHOT_INVENTORY_FAILED_BUT_CONTINUED=true")
                    log_buffer.append("SNAPSHOT_INVENTORY_FAILED_BUT_CONTINUED=true")
            if CONFIDENCE_CALIBRATION_SCRIPT.exists():
                calibration_exit = run_step(
                    "confidence_calibration_map",
                    [PYTHON_EXE, str(CONFIDENCE_CALIBRATION_SCRIPT)],
                    log_buffer,
                )
                if calibration_exit != 0:
                    print("CONFIDENCE_CALIBRATION_FAILED_BUT_CONTINUED=true")
                    log_buffer.append("CONFIDENCE_CALIBRATION_FAILED_BUT_CONTINUED=true")
            if CONFIDENCE_V2_SCRIPT.exists():
                confidence_v2_exit = run_step(
                    "confidence_score_v2",
                    [PYTHON_EXE, str(CONFIDENCE_V2_SCRIPT)],
                    log_buffer,
                )
                if confidence_v2_exit != 0:
                    print("CONFIDENCE_V2_FAILED_BUT_CONTINUED=true")
                    log_buffer.append("CONFIDENCE_V2_FAILED_BUT_CONTINUED=true")
            if WALKFORWARD_ACCEPTANCE_SCRIPT.exists():
                wf_acceptance_exit = run_step(
                    "walkforward_acceptance",
                    [PYTHON_EXE, str(WALKFORWARD_ACCEPTANCE_SCRIPT)],
                    log_buffer,
                )
                if wf_acceptance_exit != 0:
                    print("WALKFORWARD_ACCEPTANCE_FAILED_BUT_CONTINUED=true")
                    log_buffer.append("WALKFORWARD_ACCEPTANCE_FAILED_BUT_CONTINUED=true")
            if TOP20_BUYABILITY_SCRIPT.exists():
                buyability_exit = run_step(
                    "top20_buyability_report",
                    [PYTHON_EXE, str(TOP20_BUYABILITY_SCRIPT)],
                    log_buffer,
                )
                if buyability_exit != 0:
                    print("TOP20_BUYABILITY_FAILED_BUT_CONTINUED=true")
                    log_buffer.append("TOP20_BUYABILITY_FAILED_BUT_CONTINUED=true")

    if exit_code != 0:
        summary_line = f"FAILED_STEP={failed_step}"
        print(summary_line)
        log_buffer.append(summary_line)
        write_run_log(as_of_date, "\n".join(log_buffer) + "\n")
        return exit_code

    completion_line = "daily shadow monitoring completed"
    print(completion_line)
    log_buffer.append(completion_line)
    archive_line = (
        f"SNAPSHOT_ARCHIVED={archive_summary.get('archived')} "
        f"SNAPSHOT_PATH={archive_summary.get('snapshot_path')} "
        f"SNAPSHOT_DATE={archive_summary.get('snapshot_date')} "
        f"SNAPSHOT_STATUS={archive_summary.get('archive_status')}"
    )
    print(archive_line)
    log_buffer.append(archive_line)
    write_run_log(as_of_date, "\n".join(log_buffer) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
