from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from sync_auxiliary_payloads import sync_history_payload, sync_inventory_payload, sync_json_payload
from sync_csv_db_parity import sync_table, verify_table, SYNC_TABLES, VERIFY_TABLES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"
PYTHON = sys.executable

CORE_TABLES = {"stocks", "market_status", "predictions", "daily_ranking"}

JSON_PAYLOADS: list[tuple[str, Path, str | None]] = [
    ("daily_recommendations", SERVING_DIR / "daily_recommendations.json", "asof_date"),
    ("intraday_recommendations", SERVING_DIR / "intraday_recommendations.json", "asof_date"),
    ("operational_buy_gate", OUTPUT_DIR / "operational_buy_gate.json", "asof_date"),
    ("walkforward_acceptance", OUTPUT_DIR / "walkforward_acceptance.json", "asof_date"),
    ("score_kpi_monitor", DATA_DIR / "score_kpi_monitor.json", "asof_date"),
    ("top20_meaningfulness_report", OUTPUT_DIR / "top20_meaningfulness_report.json", "asof_date"),
    ("top20_buyability_report", OUTPUT_DIR / "top20_buyability_report.json", "asof_date"),
    ("buy_gate_status", SERVING_DIR / "buy_gate_status.json", "asof_date"),
    ("model_portfolio", SERVING_DIR / "model_portfolio.json", "asof_date"),
    ("performance_summary", SERVING_DIR / "performance_summary.json", "asof_date"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local display artifacts into the web DB.")
    parser.add_argument("--skip-core", action="store_true", help="Skip stocks/market_status/predictions/daily_ranking sync.")
    parser.add_argument("--skip-payloads", action="store_true", help="Skip app_payload_store JSON sync.")
    parser.add_argument("--skip-paper-trading", action="store_true", help="Skip research.paper_trading_* sync.")
    parser.add_argument("--skip-trades", action="store_true", help="Skip trades.csv -> trades sync.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sync_core_tables() -> None:
    for spec in SYNC_TABLES:
        if str(spec["name"]) not in CORE_TABLES:
            continue
        result = sync_table(spec)
        verify = verify_table(next(item for item in VERIFY_TABLES if item["name"] == spec["name"]))
        if verify["status"] != "ok":
            raise RuntimeError(f"core table parity failed: {spec['name']} -> {verify}")
        logging.info("Synced core table %s rows=%s latest=%s", spec["name"], result["csv_rows"], result["csv_latest_date"])


def sync_payloads() -> None:
    for payload_key, path, asof_field in JSON_PAYLOADS:
        sync_json_payload(payload_key, path, asof_field=asof_field)
    sync_json_payload("operational_daily_cycle_status", OUTPUT_DIR / "operational_daily_cycle_status.json")
    sync_json_payload(
        "shadow_quality_risk_guard_repeatability_report",
        OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json",
        asof_field="summary",
    )
    sync_history_payload(
        "operational_buy_gate_history",
        DATA_DIR / "history" / "operational_buy_gate_history.csv",
        asof_field="as_of_date",
    )
    sync_history_payload(
        "score_kpi_monitor_history",
        DATA_DIR / "history" / "score_kpi_monitor_history.csv",
        asof_field="as_of_date",
    )
    sync_inventory_payload()
    logging.info("Synced display payload store")


def run_script(script_name: str, *extra_args: str) -> None:
    script_path = ROOT / "python" / script_name
    command = [PYTHON, str(script_path), *extra_args]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    setup_logging()
    if not args.skip_core:
        sync_core_tables()
    if not args.skip_payloads:
        sync_payloads()
    if not args.skip_paper_trading:
        run_script("sync_paper_trading_db.py")
    if not args.skip_trades:
        trades_csv = DATA_DIR / "trades.csv"
        if trades_csv.exists():
            run_script("sync_trades_db.py")
        else:
            logging.info("Skip trades sync: %s not found", trades_csv)
    logging.info("Web display sync completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
