from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import db as db_module
from db import get_engine
from sync_auxiliary_payloads import sync_history_payload, sync_inventory_payload, sync_json_payload, sync_rows_payload
from sync_csv_db_parity import sync_table, verify_table, SYNC_TABLES, VERIFY_TABLES
from sync_live_trade_ledger import ensure_tables


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"
PYTHON = sys.executable

CORE_TABLES = {"stocks", "market_status", "features", "predictions", "daily_ranking"}

JSON_PAYLOADS: list[tuple[str, Path, str | None]] = [
    ("daily_recommendations", SERVING_DIR / "daily_recommendations.json", "asof_date"),
    ("intraday_recommendations", SERVING_DIR / "intraday_recommendations.json", "asof_date"),
    ("operational_buy_gate", OUTPUT_DIR / "operational_buy_gate.json", "asof_date"),
    ("walkforward_acceptance", OUTPUT_DIR / "walkforward_acceptance.json", "asof_date"),
    ("auto_ops_scheduler_status", OUTPUT_DIR / "auto_ops_scheduler_status.json", None),
    ("auto_ops_recovery_scheduler_status", OUTPUT_DIR / "auto_ops_recovery_scheduler_status.json", None),
    ("operational_daily_cycle_status", OUTPUT_DIR / "operational_daily_cycle_status.json", None),
    ("ops_operator_notes", OUTPUT_DIR / "ops_operator_notes.json", None),
    ("score_kpi_monitor", DATA_DIR / "score_kpi_monitor.json", "asof_date"),
    ("top20_meaningfulness_report", OUTPUT_DIR / "top20_meaningfulness_report.json", "asof_date"),
    ("top20_buyability_report", OUTPUT_DIR / "top20_buyability_report.json", "asof_date"),
    (
        "shadow_quality_risk_guard_repeatability_report",
        OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json",
        "summary",
    ),
    ("buy_gate_status", SERVING_DIR / "buy_gate_status.json", "asof_date"),
    ("model_portfolio", SERVING_DIR / "model_portfolio.json", "asof_date"),
    ("performance_summary", SERVING_DIR / "performance_summary.json", "asof_date"),
    ("live_account_balance_summary", OUTPUT_DIR / "live_account_balance_summary.json", None),
    ("trade_intents", OUTPUT_DIR / "trade_intents.json", "asof_date"),
    ("ai_entry_quality_score", OUTPUT_DIR / "ai_entry_quality_score.json", None),
    ("ai_filtered_top_candidates", OUTPUT_DIR / "ai_filtered_top_candidates.json", None),
    ("ai_selection_review_summary", OUTPUT_DIR / "ai_selection_review_summary.json", "asof_date"),
    ("watch_auto_buy_simulation", OUTPUT_DIR / "watch_auto_buy_simulation.json", "asof_date"),
    ("live_order_preview", OUTPUT_DIR / "live_order_preview.json", "asof_date"),
    ("order_requests_preview", OUTPUT_DIR / "order_requests_preview.json", "asof_date"),
    ("order_requests_execution", OUTPUT_DIR / "order_requests_execution.json", "executed_at"),
    ("live_order_fills", OUTPUT_DIR / "live_order_fills.json", "end_date"),
    ("live_trade_consistency_report", OUTPUT_DIR / "live_trade_consistency_report.json", "as_of_date"),
    ("live_trade_review_report", OUTPUT_DIR / "live_trade_review_report.json", "review_date"),
    ("live_trade_review_summary", OUTPUT_DIR / "live_trade_review_summary.json", "overview"),
    ("live_kpi_daily_report", OUTPUT_DIR / "live_kpi_daily_report.json", "as_of_date"),
    ("quality_risk_guard_live_review", OUTPUT_DIR / "quality_risk_guard_live_review.json", "as_of_date"),
    ("live_closed_trade_report", OUTPUT_DIR / "live_closed_trade_report.json", "latest_closed_date"),
    ("live_quality_guard_output_check", OUTPUT_DIR / "live_quality_guard_output_check.json", None),
    ("auto_ops_auto_buy_scheduler_status", OUTPUT_DIR / "auto_ops_auto_buy_scheduler_status.json", None),
    ("auto_ops_live_account_sync_scheduler_status", OUTPUT_DIR / "auto_ops_live_account_sync_scheduler_status.json", None),
    ("auto_trading_policy", OUTPUT_DIR / "auto_trading_policy.json", None),
    ("rule_dashboard_summary", OUTPUT_DIR / "rule_dashboard_summary.json", "as_of_date"),
    ("rule_signals_latest", OUTPUT_DIR / "rule_signals_latest.json", "as_of_date"),
    ("rule_portfolio_plan", OUTPUT_DIR / "rule_portfolio_plan.json", "as_of_date"),
    ("rule_order_preview", OUTPUT_DIR / "rule_order_preview.json", "as_of_date"),
    ("rule_account_paper_state", OUTPUT_DIR / "rule_account_paper_state.json", "as_of_date"),
    ("rule_account_live_state", OUTPUT_DIR / "rule_account_live_state.json", "as_of_date"),
    ("rule_strategy_backtest_report", OUTPUT_DIR / "rule_strategy_backtest_report.json", "latest_signal_date"),
    ("rule_execution_results", OUTPUT_DIR / "rule_execution_results.json", "as_of_date"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local display artifacts into the web DB.")
    parser.add_argument("--skip-core", action="store_true", help="Skip stocks/market_status/predictions/daily_ranking sync.")
    parser.add_argument("--skip-payloads", action="store_true", help="Skip app_payload_store JSON sync.")
    parser.add_argument("--skip-paper-trading", action="store_true", help="Skip research.paper_trading_* sync.")
    parser.add_argument("--skip-trades", action="store_true", help="Deprecated; trades are skipped unless --sync-trades-to-web is set.")
    parser.add_argument(
        "--sync-trades-to-web",
        action="store_true",
        help="Explicitly push data/trades.csv into web public.trades. Use only for recovery/import.",
    )
    parser.add_argument("--skip-meaningfulness-review", action="store_true", help="Skip research.meaningfulness_review_note sync.")
    parser.add_argument("--reset-first", action="store_true", help="Delete existing display tables first, then reload from local artifacts.")
    parser.add_argument(
        "--reset-trades",
        action="store_true",
        help="With --reset-first, also delete web public.trades. Use only for recovery/import.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def resolve_source_database_url() -> str:
    local_database_url = str(os.environ.get("LOCAL_DATABASE_URL", "")).strip()
    if local_database_url:
        logging.info("Using LOCAL_DATABASE_URL as source DB")
        return local_database_url
    return str(os.environ.get("DATABASE_URL", "")).strip()


def configure_target_database() -> str:
    web_database_url = str(os.environ.get("WEB_DATABASE_URL", "")).strip()
    if web_database_url:
        os.environ["DATABASE_URL"] = web_database_url
        db_module.get_database_url.cache_clear()
        db_module.get_engine.cache_clear()
        logging.info("Using WEB_DATABASE_URL as sync target")
        return "WEB_DATABASE_URL"
    logging.info("WEB_DATABASE_URL not set; falling back to DATABASE_URL")
    return "DATABASE_URL"


def sync_local_display_state_if_needed(args: argparse.Namespace, source_database_url: str) -> None:
    web_database_url = str(os.environ.get("WEB_DATABASE_URL", "")).strip()
    if not web_database_url or not source_database_url or web_database_url == source_database_url:
        return
    original_database_url = str(os.environ.get("DATABASE_URL", "")).strip()
    try:
        os.environ["DATABASE_URL"] = source_database_url
        db_module.get_database_url.cache_clear()
        db_module.get_engine.cache_clear()
        if not args.skip_payloads:
            sync_payloads()
            run_script("sync_live_trade_ledger.py")
        logging.info("Synced local display payload store before web sync")
    finally:
        if original_database_url:
            os.environ["DATABASE_URL"] = original_database_url
        db_module.get_database_url.cache_clear()
        db_module.get_engine.cache_clear()


def table_exists(conn, qualified_name: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": qualified_name}).scalar())


def reset_display_tables(*, include_trades: bool = False) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        delete_order = [
            ("research.meaningfulness_review_note", "DELETE FROM research.meaningfulness_review_note"),
            ("research.paper_trading_position", "DELETE FROM research.paper_trading_position"),
            ("research.paper_trading_nav", "DELETE FROM research.paper_trading_nav"),
            ("research.paper_trading_run", "DELETE FROM research.paper_trading_run"),
            ("research.app_payload_store", "DELETE FROM research.app_payload_store"),
            ("public.daily_ranking", "DELETE FROM daily_ranking"),
            ("public.predictions", "DELETE FROM predictions"),
            ("public.market_status", "DELETE FROM market_status"),
            ("public.etf_holdings_snapshot", "DELETE FROM etf_holdings_snapshot"),
            ("public.stocks", "DELETE FROM stocks"),
        ]
        if include_trades:
            delete_order.insert(5, ("public.trades", "DELETE FROM trades"))
        for qualified_name, statement in delete_order:
            if not table_exists(conn, qualified_name):
                continue
            conn.execute(text(statement))
            logging.info("Cleared %s", qualified_name)
        if not include_trades:
            logging.info("Preserved public.trades during reset; pass --reset-trades to clear it")


def verify_stocks_subset() -> dict[str, object]:
    csv_path = DATA_DIR / "universe.csv"
    if not csv_path.exists():
        return {"status": "missing_csv", "csv_rows": 0, "missing_codes": []}
    df = pd.read_csv(csv_path, dtype={"code": str}, low_memory=False)
    if df.empty:
        return {"status": "ok", "csv_rows": 0, "missing_codes": []}
    codes = sorted({str(code).zfill(6) for code in df["code"].dropna().tolist()})
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT code FROM stocks WHERE code = ANY(:codes)"), {"codes": codes}).fetchall()
    existing = {str(row[0]).zfill(6) for row in rows}
    missing = [code for code in codes if code not in existing]
    return {
        "status": "ok" if not missing else "missing_codes",
        "csv_rows": len(codes),
        "db_match_rows": len(existing),
        "missing_codes": missing[:20],
    }


def sync_core_tables() -> None:
    for spec in SYNC_TABLES:
        if str(spec["name"]) not in CORE_TABLES:
            continue
        result = sync_table(spec)
        if str(spec["name"]) == "stocks":
            verify = verify_stocks_subset()
        else:
            verify = verify_table(next(item for item in VERIFY_TABLES if item["name"] == spec["name"]))
        if verify["status"] != "ok":
            raise RuntimeError(f"core table parity failed: {spec['name']} -> {verify}")
        logging.info("Synced core table %s rows=%s latest=%s", spec["name"], result["csv_rows"], result["csv_latest_date"])


def sync_payloads() -> None:
    run_script("build_rule_web_payloads.py")
    for payload_key, path, asof_field in JSON_PAYLOADS:
        sync_json_payload(payload_key, path, asof_field=asof_field)
    live_summary_path = OUTPUT_DIR / "live_account_balance_summary.json"
    live_summary = {}
    if live_summary_path.exists():
        live_summary = json.loads(live_summary_path.read_text(encoding="utf-8-sig"))
    sync_rows_payload(
        "live_account_holdings",
        DATA_DIR / "live_account_holdings.csv",
        asof_date=live_summary.get("generated_at"),
        generated_at=live_summary.get("generated_at"),
        extra={
            "holding_count": live_summary.get("holding_count"),
            "env_dv": live_summary.get("env_dv"),
        },
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


def ensure_meaningfulness_review_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS research"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.meaningfulness_review_note (
                    analysis_date DATE NOT NULL,
                    code TEXT NOT NULL,
                    decision TEXT NULL,
                    note TEXT NULL,
                    updated_by TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (analysis_date, code)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_meaningfulness_review_note_updated
                ON research.meaningfulness_review_note(updated_at DESC)
                """
            )
        )


def sync_meaningfulness_review_notes(source_database_url: str) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip meaningfulness review sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip meaningfulness review sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip meaningfulness review sync: source and target DB are identical")
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as conn:
            source_exists = bool(
                conn.execute(text("SELECT to_regclass('research.meaningfulness_review_note')")).scalar()
            )
            if not source_exists:
                logging.info("Skip meaningfulness review sync: source table not found")
                return
            rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        """
                        SELECT analysis_date, code, decision, note, updated_by, created_at, updated_at
                        FROM research.meaningfulness_review_note
                        ORDER BY analysis_date DESC, updated_at DESC, code ASC
                        """
                    )
                ).mappings()
            ]

        ensure_meaningfulness_review_table(target_engine)
        with target_engine.begin() as conn:
            conn.execute(text("DELETE FROM research.meaningfulness_review_note"))
            if rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO research.meaningfulness_review_note (
                            analysis_date, code, decision, note, updated_by, created_at, updated_at
                        ) VALUES (
                            :analysis_date, :code, :decision, :note, :updated_by, :created_at, :updated_at
                        )
                        """
                    ),
                    rows,
                )
        logging.info("Synced meaningfulness review notes rows=%d", len(rows))
    finally:
        source_engine.dispose()
        target_engine.dispose()


def sync_live_order_fills_table(source_database_url: str) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip live order fill table sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip live order fill table sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip live order fill table sync: source and target DB are identical")
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as conn:
            source_exists = bool(conn.execute(text("SELECT to_regclass('research.live_order_fill')")).scalar())
            if not source_exists:
                logging.info("Skip live order fill table sync: source table not found")
                return
            rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        """
                        SELECT
                            request_id, broker_order_id, broker_org_order_id, as_of_date, filled_at,
                            code, name, side, filled_qty, filled_price, filled_amount,
                            fee, tax, fill_status, source, raw_response_json, created_at, updated_at
                        FROM research.live_order_fill
                        ORDER BY filled_at, broker_order_id, code, side
                        """
                    )
                ).mappings()
            ]

        ensure_tables()
        with target_engine.begin() as conn:
            for row in rows:
                row["raw_response_json"] = json.dumps(row.get("raw_response_json"), ensure_ascii=False, default=str)
                conn.execute(
                    text(
                        """
                        INSERT INTO research.live_order_fill (
                            request_id, broker_order_id, broker_org_order_id, as_of_date, filled_at,
                            code, name, side, filled_qty, filled_price, filled_amount,
                            fee, tax, fill_status, source, raw_response_json, created_at, updated_at
                        )
                        VALUES (
                            :request_id, :broker_order_id, :broker_org_order_id, :as_of_date, :filled_at,
                            :code, :name, :side, :filled_qty, :filled_price, :filled_amount,
                            :fee, :tax, :fill_status, :source, CAST(:raw_response_json AS jsonb), :created_at, :updated_at
                        )
                        ON CONFLICT (broker_order_id, code, side, filled_at, filled_qty, filled_price) DO UPDATE SET
                            request_id = COALESCE(EXCLUDED.request_id, research.live_order_fill.request_id),
                            broker_org_order_id = COALESCE(EXCLUDED.broker_org_order_id, research.live_order_fill.broker_org_order_id),
                            as_of_date = EXCLUDED.as_of_date,
                            name = EXCLUDED.name,
                            filled_amount = EXCLUDED.filled_amount,
                            fee = EXCLUDED.fee,
                            tax = EXCLUDED.tax,
                            fill_status = EXCLUDED.fill_status,
                            source = EXCLUDED.source,
                            raw_response_json = EXCLUDED.raw_response_json,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    row,
                )
        logging.info("Synced live order fill table rows=%d", len(rows))
    finally:
        source_engine.dispose()
        target_engine.dispose()


def sync_live_trade_review_table(source_database_url: str) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip live trade review table sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip live trade review table sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip live trade review table sync: source and target DB are identical")
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as conn:
            source_exists = bool(conn.execute(text("SELECT to_regclass('research.live_trade_review')")).scalar())
            if not source_exists:
                logging.info("Skip live trade review table sync: source table not found")
                return
            rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        """
                        SELECT
                            intent_id, request_id, code, review_date, pre_tags, post_tags,
                            outcome_label, review_note, next_action_note, reviewer, created_at, updated_at
                        FROM research.live_trade_review
                        WHERE reviewer = 'auto_review'
                        ORDER BY review_date, request_id, code
                        """
                    )
                ).mappings()
            ]

        ensure_tables()
        with target_engine.begin() as conn:
            conn.execute(text("DELETE FROM research.live_trade_review WHERE reviewer = 'auto_review'"))
            if rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO research.live_trade_review (
                            intent_id, request_id, code, review_date, pre_tags, post_tags,
                            outcome_label, review_note, next_action_note, reviewer, created_at, updated_at
                        )
                        VALUES (
                            :intent_id, :request_id, :code, :review_date, :pre_tags, :post_tags,
                            :outcome_label, :review_note, :next_action_note, :reviewer, :created_at, :updated_at
                        )
                        """
                    ),
                    rows,
                )
        logging.info("Synced live trade review table rows=%d", len(rows))
    finally:
        source_engine.dispose()
        target_engine.dispose()


def run_script(script_name: str, *extra_args: str) -> None:
    script_path = ROOT / "python" / script_name
    command = [PYTHON, str(script_path), *extra_args]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    setup_logging()
    load_environment()
    source_database_url = resolve_source_database_url()
    sync_local_display_state_if_needed(args, source_database_url)
    configure_target_database()
    if args.reset_first:
        reset_display_tables(include_trades=args.reset_trades)
    if not args.skip_core:
        sync_core_tables()
    if not args.skip_payloads:
        sync_payloads()
        run_script("sync_live_trade_ledger.py")
        sync_live_order_fills_table(source_database_url)
        sync_live_trade_review_table(source_database_url)
    if not args.skip_paper_trading:
        run_script("sync_paper_trading_db.py")
    if args.sync_trades_to_web:
        trades_csv = DATA_DIR / "trades.csv"
        if trades_csv.exists():
            run_script("sync_trades_db.py")
        else:
            logging.info("Skip trades sync: %s not found", trades_csv)
    else:
        logging.info("Skip trades.csv -> web trades sync; web public.trades is treated as source of truth")
    if not args.skip_meaningfulness_review:
        sync_meaningfulness_review_notes(source_database_url)
    logging.info("Web display sync completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
