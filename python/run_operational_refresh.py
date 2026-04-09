from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

THEME_SHADOW_SCRIPT = ROOT / "python" / "run_theme_shadow_daily.py"
BUY_CANDIDATE_SCRIPT = ROOT / "python" / "buy_candidate_builder.py"
BUY_COMPARE_SCRIPT = ROOT / "python" / "build_buy_candidate_comparison.py"
BUY_GATE_SCRIPT = ROOT / "python" / "build_operational_buy_gate.py"
PAPER_TRADING_SCRIPT = ROOT / "python" / "run_paper_trading_ledger.py"
PAPER_TRADING_DB_SYNC_SCRIPT = ROOT / "python" / "sync_paper_trading_db.py"
EXPORT_SCRIPT = ROOT / "python" / "export_serving_payloads.py"
SCORE_KPI_MONITOR_SCRIPT = ROOT / "python" / "score_kpi_monitor.py"
MARKET_STATUS_VALIDATION_SCRIPT = ROOT / "python" / "build_market_status_validation_report.py"
SHADOW_DAILY_REPORT_SCRIPT = ROOT / "python" / "build_shadow_quality_risk_guard_daily_report.py"
SHADOW_REPEATABILITY_REPORT_SCRIPT = ROOT / "python" / "build_shadow_quality_risk_guard_repeatability_report.py"
REPAIR_SHADOW_SNAPSHOTS_SCRIPT = ROOT / "python" / "repair_shadow_ranking_snapshots.py"
QUALITY_COVERAGE_REPORT_SCRIPT = ROOT / "python" / "build_quality_coverage_report.py"
QUALITY_RISK_GUARD_PROMOTION_REPORT_SCRIPT = ROOT / "python" / "build_quality_risk_guard_promotion_report.py"
SYNC_AUXILIARY_PAYLOADS_SCRIPT = ROOT / "python" / "sync_auxiliary_payloads.py"
LIVE_SYNC_SCRIPT = ROOT / "python" / "sync_live_account_holdings.py"
LIVE_PREVIEW_SCRIPT = ROOT / "python" / "build_live_order_preview.py"
OUTPUTS_DIR = ROOT / "outputs"
DATA_HISTORY_DIR = ROOT / "data" / "history"
BUY_GATE_JSON = OUTPUTS_DIR / "operational_buy_gate.json"
SCORE_KPI_JSON_CANDIDATES = [
    ROOT / "data" / "score_kpi_monitor.json",
    OUTPUTS_DIR / "score_kpi_monitor.json",
]
BUY_GATE_HISTORY_CSV = DATA_HISTORY_DIR / "operational_buy_gate_history.csv"
SCORE_KPI_HISTORY_CSV = DATA_HISTORY_DIR / "score_kpi_monitor_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh operational daily artifacts in one command.")
    parser.add_argument("--skip-theme-shadow", action="store_true", help="Skip run_theme_shadow_daily.py")
    parser.add_argument("--skip-paper-trading", action="store_true", help="Skip run_paper_trading_ledger.py")
    parser.add_argument("--skip-paper-trading-db", action="store_true", help="Skip sync_paper_trading_db.py")
    parser.add_argument(
        "--with-live-account",
        action="store_true",
        help="Also sync KIS live/demo holdings and build live order preview.",
    )
    parser.add_argument(
        "--skip-live-preview",
        action="store_true",
        help="With --with-live-account, skip build_live_order_preview.py after holdings sync.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print(f"[START] {name}")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[OK] {name}")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
        value = json.loads(normalized)
        return value if isinstance(value, dict) else None
    except Exception as exc:  # pragma: no cover - best-effort snapshot
        print(f"[WARN] failed to read {path.name}: {exc}")
        return None


def write_history_row(csv_path: Path, key_field: str, row: dict[str, object], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                existing = list(csv.DictReader(fh))
        except Exception as exc:  # pragma: no cover - best-effort snapshot
            print(f"[WARN] failed to load history {csv_path.name}: {exc}")
            existing = []

    key_value = str(row.get(key_field) or "").strip()
    if not key_value:
        return

    normalized_existing = [item for item in existing if str(item.get(key_field) or "").strip() != key_value]
    normalized_existing.append({name: "" if row.get(name) is None else str(row.get(name)) for name in fieldnames})
    normalized_existing.sort(key=lambda item: str(item.get(key_field) or ""))

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_existing)


def append_operational_history() -> None:
    gate = read_json(BUY_GATE_JSON) or {}
    score_kpi = None
    for candidate in SCORE_KPI_JSON_CANDIDATES:
        score_kpi = read_json(candidate)
        if score_kpi:
            break

    decision = gate.get("decisions", [{}])[0] if isinstance(gate.get("decisions"), list) and gate.get("decisions") else {}
    gate_row = {
        "as_of_date": gate.get("asof_date") or "",
        "generated_at": gate.get("generated_at") or "",
        "overall_status": gate.get("overall_status") or "",
        "primary_bucket": gate.get("primary_bucket") or "",
        "daily_cycle_status": gate.get("daily_cycle_status") or "",
        "matured_benchmark_dates": (((decision.get("benchmark") or {}).get("matured_dates_max")) if isinstance(decision, dict) else ""),
        "trusted_ratio_top20": (((decision.get("confidence_v2") or {}).get("trusted_ratio_top20")) if isinstance(decision, dict) else ""),
        "walkforward_acceptance": (((decision.get("walkforward_acceptance") or {}).get("status")) if isinstance(decision, dict) else ""),
        "buy_now_count": (((decision.get("buyability") or {}).get("buy_now_count")) if isinstance(decision, dict) else ""),
        "watchlist_count": (((decision.get("buyability") or {}).get("watchlist_count")) if isinstance(decision, dict) else ""),
        "blocked_count": (((decision.get("buyability") or {}).get("blocked_count")) if isinstance(decision, dict) else ""),
        "paper_only_count": (((decision.get("buyability") or {}).get("paper_only_count")) if isinstance(decision, dict) else ""),
    }
    write_history_row(
        BUY_GATE_HISTORY_CSV,
        "as_of_date",
        gate_row,
        [
            "as_of_date",
            "generated_at",
            "overall_status",
            "primary_bucket",
            "daily_cycle_status",
            "matured_benchmark_dates",
            "trusted_ratio_top20",
            "walkforward_acceptance",
            "buy_now_count",
            "watchlist_count",
            "blocked_count",
            "paper_only_count",
        ],
    )

    kpi_rows = score_kpi.get("kpis", []) if isinstance(score_kpi, dict) else []
    kpi_map = {
        str(item.get("metric") or ""): item
        for item in kpi_rows
        if isinstance(item, dict) and item.get("metric")
    }
    score_kpi_row = {
        "as_of_date": ((score_kpi.get("summary") or {}).get("latest_date")) if isinstance(score_kpi, dict) else "",
        "generated_at": ((score_kpi.get("metadata") or {}).get("generated_at")) if isinstance(score_kpi, dict) else "",
        "overall_status": ((score_kpi.get("summary") or {}).get("overall_status")) if isinstance(score_kpi, dict) else "",
        "score_formula_version": ((score_kpi.get("metadata") or {}).get("score_formula_version")) if isinstance(score_kpi, dict) else "",
        "alert_metric_count": sum(1 for item in kpi_rows if str(item.get("status") or "").upper() == "ALERT"),
        "watch_metric_count": sum(1 for item in kpi_rows if str(item.get("status") or "").upper() == "WATCH"),
        "top20_mean_confidence_score": (kpi_map.get("top20_mean_confidence_score") or {}).get("value", ""),
        "walkforward_top20_avg_return_60d": (kpi_map.get("walkforward_top20_avg_return_60d") or {}).get("value", ""),
        "confidence_high_bucket_hit_rate_60d": (kpi_map.get("confidence_high_bucket_hit_rate_60d") or {}).get("value", ""),
        "confidence_calibration_usable_bucket_count": (kpi_map.get("confidence_calibration_usable_bucket_count") or {}).get("value", ""),
    }
    write_history_row(
        SCORE_KPI_HISTORY_CSV,
        "as_of_date",
        score_kpi_row,
        [
            "as_of_date",
            "generated_at",
            "overall_status",
            "score_formula_version",
            "alert_metric_count",
            "watch_metric_count",
            "top20_mean_confidence_score",
            "walkforward_top20_avg_return_60d",
            "confidence_high_bucket_hit_rate_60d",
            "confidence_calibration_usable_bucket_count",
        ],
    )


def main() -> int:
    args = parse_args()

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_theme_shadow:
        steps.append(("theme_shadow_daily", [PYTHON, str(THEME_SHADOW_SCRIPT)]))
    steps.extend(
        [
            ("buy_candidate_builder", [PYTHON, str(BUY_CANDIDATE_SCRIPT)]),
            ("buy_candidate_comparison", [PYTHON, str(BUY_COMPARE_SCRIPT)]),
            ("score_kpi_monitor", [PYTHON, str(SCORE_KPI_MONITOR_SCRIPT)]),
            ("market_status_validation", [PYTHON, str(MARKET_STATUS_VALIDATION_SCRIPT)]),
            ("operational_buy_gate", [PYTHON, str(BUY_GATE_SCRIPT)]),
        ]
    )
    if not args.skip_paper_trading:
        steps.append(("paper_trading_ledger", [PYTHON, str(PAPER_TRADING_SCRIPT)]))
        if not args.skip_paper_trading_db:
            steps.append(("paper_trading_db_sync", [PYTHON, str(PAPER_TRADING_DB_SYNC_SCRIPT)]))
    if args.with_live_account:
        steps.append(("live_account_holdings_sync", [PYTHON, str(LIVE_SYNC_SCRIPT)]))
        if not args.skip_live_preview:
            steps.append(("live_order_preview", [PYTHON, str(LIVE_PREVIEW_SCRIPT)]))
    steps.append(("export_serving_payloads", [PYTHON, str(EXPORT_SCRIPT)]))
    steps.append(("shadow_quality_risk_guard_daily_report", [PYTHON, str(SHADOW_DAILY_REPORT_SCRIPT)]))
    steps.append(("repair_shadow_ranking_snapshots", [PYTHON, str(REPAIR_SHADOW_SNAPSHOTS_SCRIPT)]))
    steps.append(("shadow_quality_risk_guard_repeatability_report", [PYTHON, str(SHADOW_REPEATABILITY_REPORT_SCRIPT)]))
    steps.append(("quality_coverage_report", [PYTHON, str(QUALITY_COVERAGE_REPORT_SCRIPT)]))
    steps.append(("quality_risk_guard_promotion_report", [PYTHON, str(QUALITY_RISK_GUARD_PROMOTION_REPORT_SCRIPT)]))
    steps.append(("sync_auxiliary_payloads", [PYTHON, str(SYNC_AUXILIARY_PAYLOADS_SCRIPT)]))

    try:
        for name, command in steps:
            run_step(name, command)
        append_operational_history()
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] returncode={exc.returncode}")
        return exc.returncode

    print("[DONE] operational refresh completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
