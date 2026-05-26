from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import db as db_module
from db import get_engine
from sync_auxiliary_payloads import sync_history_payload, sync_inventory_payload, sync_json_payload, sync_rows_payload, read_csv_rows
from payload_store import upsert_json_payload
from sync_csv_db_parity import sync_table, verify_table, SYNC_TABLES, VERIFY_TABLES
from sync_live_trade_ledger import ensure_tables


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"
PYTHON = sys.executable
RANKING_ARCHIVE_DIR = DATA_DIR / "history" / "ranking"

CORE_TABLES = {"stocks", "market_status", "features", "predictions", "daily_ranking"}
US_RANKING_SOURCE_TABLE = "recommend.us_stock_rank_daily"
US_RANKING_SYNC_DATES = 60
KR_RANKING_SYNC_DATES = 60
US_RANKING_INSERT_BATCH_SIZE = 2000
FLOW_DAILY_SYNC_DATES = 60
US_TRADE_DDL_PATH = ROOT / "postgres" / "us_trade_tables.sql"
US_STOCK_DDL_PATH = ROOT / "postgres" / "us_stock_tables.sql"

FACT_PRICE_DAILY_SYNC_SPEC: dict[str, object] = {
    "table": "public.fact_price_daily", "date_column": "date", "lookback_dates": 180
}

US_WEB_TABLE_SYNC_SPECS: list[dict[str, object]] = [
    {"table": "market.us_stock_universe", "date_column": None, "lookback_dates": None},
    {"table": "trade.us_buy_candidate_log", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_buy_decision_log", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_risk_guard_log", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_paper_order", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_sell_decision_log", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_sell_signal_log", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_paper_sell_order", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "trade.us_paper_position_snapshot", "date_column": "snapshot_date", "lookback_dates": 60},
    {"table": "paper.us_stock_paper_account", "date_column": None, "lookback_dates": None},
    {"table": "paper.us_stock_paper_order", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "paper.us_stock_paper_fill", "date_column": "trade_date", "lookback_dates": 60},
    {"table": "paper.us_stock_paper_position", "date_column": None, "lookback_dates": None},
    {"table": "paper.us_stock_paper_account_snapshot", "date_column": "snapshot_date", "lookback_dates": 60},
]

JSON_PAYLOADS: list[tuple[str, Path, str | None]] = [
    ("daily_recommendations", SERVING_DIR / "daily_recommendations.json", "asof_date"),
    ("intraday_recommendations", SERVING_DIR / "intraday_recommendations.json", "asof_date"),
    ("operational_buy_gate", OUTPUT_DIR / "operational_buy_gate.json", "asof_date"),
    ("walkforward_acceptance", OUTPUT_DIR / "walkforward_acceptance.json", "asof_date"),
    ("auto_ops_scheduler_status", OUTPUT_DIR / "auto_ops_scheduler_status.json", None),
    ("auto_ops_recovery_scheduler_status", OUTPUT_DIR / "auto_ops_recovery_scheduler_status.json", None),
    ("operational_daily_cycle_status", OUTPUT_DIR / "operational_daily_cycle_status.json", None),
    ("ops_operator_notes", OUTPUT_DIR / "ops_operator_notes.json", None),
    ("model_feature_importance", DATA_DIR / "model_feature_importance.json", None),
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
    ("rule_before_open_scheduler_status", OUTPUT_DIR / "rule_before_open_scheduler_status.json", None),
    ("rule_after_open_scheduler_status", OUTPUT_DIR / "rule_after_open_scheduler_status.json", None),
    ("rule_after_close_scheduler_status", OUTPUT_DIR / "rule_after_close_scheduler_status.json", None),
    ("us_macro_scheduler_status", OUTPUT_DIR / "us_macro_scheduler_status.json", None),
    ("us_macro_shadow_scheduler_status", OUTPUT_DIR / "us_macro_shadow_scheduler_status.json", None),
    ("us_pipeline_scheduler_status", OUTPUT_DIR / "us_pipeline_scheduler_status.json", None),
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
    parser.add_argument(
        "--skip-core",
        action="store_true",
        help="Skip stocks/market_status/predictions/daily_ranking and recommend.us_stock_rank_daily sync.",
    )
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
    if load_dotenv:
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


def execute_sql_script(conn, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    sql = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not sql:
        return
    dbapi_conn = getattr(conn.connection, "driver_connection", None)
    if dbapi_conn is None:
        dbapi_conn = conn.connection.connection
    with dbapi_conn.cursor() as cursor:
        cursor.execute(sql)


def ensure_schema(conn, schema_name: str) -> None:
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))


def ensure_market_us_universe_table(conn) -> None:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS market"))
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS market.us_stock_universe (
                ticker          VARCHAR NOT NULL,
                name            VARCHAR,
                sector          VARCHAR,
                industry        VARCHAR,
                universe_tag    VARCHAR,
                is_active       VARCHAR,
                added_date      DATE,
                removed_date    DATE,
                data_source     VARCHAR,
                membership_rank INTEGER,
                source_note     TEXT,
                created_at      TIMESTAMPTZ,
                updated_at      TIMESTAMPTZ,
                PRIMARY KEY (ticker, universe_tag)
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_market_us_stock_universe_tag_rank
            ON market.us_stock_universe (universe_tag, membership_rank, ticker)
            """
        )
    )


def ensure_us_web_supporting_tables(conn) -> None:
    ensure_schema(conn, "trade")
    ensure_schema(conn, "raw")
    ensure_schema(conn, "feature")
    ensure_schema(conn, "label")
    ensure_schema(conn, "meta")
    ensure_schema(conn, "recommend")
    ensure_schema(conn, "paper")
    ensure_schema(conn, "risk")
    ensure_schema(conn, "research")
    if not table_exists(conn, "trade.us_buy_candidate_log"):
        execute_sql_script(conn, US_TRADE_DDL_PATH)
    if not table_exists(conn, "paper.us_stock_paper_account"):
        execute_sql_script(conn, US_STOCK_DDL_PATH)
    ensure_market_us_universe_table(conn)
    ensure_us_ranking_table(conn)


def get_table_columns(conn, qualified_name: str) -> list[dict[str, str]]:
    schema_name, table_name = qualified_name.split(".", 1)
    rows = conn.execute(
        text(
            """
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema = :schema_name AND table_name = :table_name
            ORDER BY ordinal_position
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    ).mappings()
    return [dict(row) for row in rows]


def sync_table_history(
    source_database_url: str,
    *,
    table_name: str,
    date_column: str | None,
    lookback_dates: int | None,
) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip %s sync: source DATABASE_URL not set", table_name)
        return
    if not target_url:
        logging.info("Skip %s sync: target DATABASE_URL not set", table_name)
        return
    if source_url == target_url:
        logging.info("Skip %s sync: source and target DB are identical", table_name)
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as source_conn:
            if not table_exists(source_conn, table_name):
                logging.info("Skip %s sync: source table not found", table_name)
                return
            source_columns = get_table_columns(source_conn, table_name)
            if not source_columns:
                logging.info("Skip %s sync: source table has no columns", table_name)
                return
            column_names = [str(col["column_name"]) for col in source_columns]
            json_columns = {str(col["column_name"]) for col in source_columns if str(col["udt_name"]) in {"json", "jsonb"}}
            select_columns_sql = ", ".join(column_names)

            scoped_dates: list[object] | None = None
            if date_column and lookback_dates:
                scoped_dates = [
                    row[0]
                    for row in source_conn.execute(
                        text(
                            f"""
                            SELECT {date_column}
                            FROM (
                                SELECT DISTINCT {date_column}
                                FROM {table_name}
                                ORDER BY {date_column} DESC
                                LIMIT :limit
                            ) recent
                            ORDER BY {date_column}
                            """
                        ),
                        {"limit": lookback_dates},
                    ).fetchall()
                ]
                if not scoped_dates:
                    logging.info("Skip %s sync: no source dates found", table_name)
                    return
                rows = [
                    dict(row)
                    for row in source_conn.execute(
                        text(
                            f"""
                            SELECT {select_columns_sql}
                            FROM {table_name}
                            WHERE {date_column} = ANY(:scoped_dates)
                            ORDER BY {date_column} DESC
                            """
                        ),
                        {"scoped_dates": scoped_dates},
                    ).mappings()
                ]
            else:
                rows = [
                    dict(row)
                    for row in source_conn.execute(
                        text(f"SELECT {select_columns_sql} FROM {table_name}")
                    ).mappings()
                ]

        with target_engine.begin() as target_conn:
            ensure_us_web_supporting_tables(target_conn)
            if not table_exists(target_conn, table_name):
                raise RuntimeError(f"target table missing after bootstrap: {table_name}")

            target_columns = get_table_columns(target_conn, table_name)
            target_column_names = [str(col["column_name"]) for col in target_columns]
            insert_columns = [name for name in column_names if name in target_column_names]
            if not insert_columns:
                logging.info("Skip %s sync: no shared columns with target", table_name)
                return

            if date_column and scoped_dates:
                target_conn.execute(
                    text(f"DELETE FROM {table_name} WHERE {date_column} = ANY(:scoped_dates)"),
                    {"scoped_dates": scoped_dates},
                )
            else:
                target_conn.execute(text(f"DELETE FROM {table_name}"))

            if rows:
                payload_rows: list[dict[str, object]] = []
                for row in rows:
                    payload_row: dict[str, object] = {}
                    for column in insert_columns:
                        value = row.get(column)
                        if column in json_columns and value is not None:
                            payload_row[column] = json.dumps(value, ensure_ascii=False, default=str)
                        else:
                            payload_row[column] = value
                    payload_rows.append(payload_row)
                column_sql = ", ".join(insert_columns)
                value_sql = ", ".join(
                    f"CAST(:{column} AS {('jsonb' if column in json_columns else 'TEXT')})"
                    if column in json_columns
                    else f":{column}"
                    for column in insert_columns
                )
                target_conn.execute(
                    text(f"INSERT INTO {table_name} ({column_sql}) VALUES ({value_sql})"),
                    payload_rows,
                )

        logging.info(
            "Synced table %s rows=%d%s",
            table_name,
            len(rows),
            f" dates={len(scoped_dates)}" if scoped_dates else "",
        )
    finally:
        source_engine.dispose()
        target_engine.dispose()


def sync_kr_ranking_history(source_database_url: str) -> None:
    archive_result = load_kr_ranking_history_from_archives(KR_RANKING_SYNC_DATES)
    if archive_result is not None:
        scoped_dates, archive_df = archive_result
        sync_kr_ranking_history_rows(scoped_dates, archive_df)
        return

    history_result = load_kr_ranking_history_from_db_history(source_database_url, KR_RANKING_SYNC_DATES)
    if history_result is not None:
        scoped_dates, history_df = history_result
        sync_kr_ranking_history_rows(scoped_dates, history_df)
        return

    logging.info("KR ranking archive/history unavailable; fallback to source DB daily_ranking sync")
    sync_table_history(
        source_database_url,
        table_name="public.daily_ranking",
        date_column="date",
        lookback_dates=KR_RANKING_SYNC_DATES,
    )


def load_kr_ranking_history_from_archives(lookback_dates: int) -> tuple[list[str], pd.DataFrame] | None:
    pattern = re.compile(r"(?P<yyyymmdd>\d{8})_ranking_final\.csv$", re.IGNORECASE)
    candidates: list[tuple[str, Path]] = []
    if RANKING_ARCHIVE_DIR.exists():
        for path in sorted(RANKING_ARCHIVE_DIR.glob("*_ranking_final.csv")):
            match = pattern.match(path.name)
            if not match:
                continue
            compact = match.group("yyyymmdd")
            date_value = f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
            candidates.append((date_value, path))
    if not candidates:
        return None

    selected = candidates[-lookback_dates:]
    frames: list[pd.DataFrame] = []
    scoped_dates: list[str] = []
    for date_value, path in selected:
        try:
            df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except Exception:
            logging.exception("Failed to read ranking archive %s", path)
            continue
        if df.empty:
            continue
        scoped_dates.append(date_value)
        frame = df.copy()
        frame["date"] = date_value
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame = frame.drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)
        frames.append(frame)

    if not frames or not scoped_dates:
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)
    logging.info(
        "Loaded KR ranking archives rows=%d dates=%d latest=%s",
        len(merged),
        len(scoped_dates),
        scoped_dates[-1],
    )
    return scoped_dates, merged


def load_kr_ranking_history_from_db_history(
    source_database_url: str,
    lookback_dates: int,
) -> tuple[list[str], pd.DataFrame] | None:
    source_url = str(source_database_url or "").strip()
    if not source_url:
        return None

    source_engine = create_engine(source_url, future=True)
    try:
        with source_engine.connect() as source_conn:
            if not table_exists(source_conn, "research.ranking_history"):
                return None

            recent_dates = [
                str(row[0])[:10]
                for row in source_conn.execute(
                    text(
                        """
                        SELECT as_of_date::text
                        FROM (
                            SELECT DISTINCT as_of_date
                            FROM research.ranking_history
                            ORDER BY as_of_date DESC
                            LIMIT :limit
                        ) recent
                        ORDER BY as_of_date
                        """
                    ),
                    {"limit": lookback_dates},
                ).fetchall()
            ]
            if not recent_dates:
                return None

            history_df = pd.read_sql_query(
                text(
                    """
                    WITH latest_runs AS (
                        SELECT as_of_date, MAX(run_id) AS run_id
                        FROM research.ranking_history
                        WHERE as_of_date = ANY(:scoped_dates)
                        GROUP BY as_of_date
                    )
                    SELECT
                        rh.as_of_date::text AS date,
                        LPAD(rh.code::text, 6, '0') AS code,
                        COALESCE(s.name, '') AS name,
                        rh.rank AS rank_final,
                        rh.rank AS live_rank,
                        rh.live_score,
                        rh.live_score_source,
                        rh.final_score,
                        rh.ret_score,
                        rh.prob_score,
                        rh.qual_score,
                        rh.tech_score,
                        rh.risk_penalty
                    FROM research.ranking_history rh
                    INNER JOIN latest_runs lr
                        ON lr.as_of_date = rh.as_of_date
                       AND lr.run_id = rh.run_id
                    LEFT JOIN public.stocks s
                        ON LPAD(s.code::text, 6, '0') = LPAD(rh.code::text, 6, '0')
                    WHERE rh.as_of_date = ANY(:scoped_dates)
                    ORDER BY rh.as_of_date ASC, rh.rank ASC, rh.code ASC
                    """
                ),
                source_conn,
                params={"scoped_dates": recent_dates},
            )
    finally:
        source_engine.dispose()

    if history_df.empty:
        return None

    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    history_df["code"] = history_df["code"].astype(str).str.zfill(6)
    history_df = history_df.dropna(subset=["date", "code"]).copy()
    history_df = history_df.drop_duplicates(subset=["date", "code"], keep="first").reset_index(drop=True)
    scoped_dates = sorted(history_df["date"].dropna().unique().tolist())
    if not scoped_dates:
        return None

    logging.info(
        "Loaded KR ranking history from research.ranking_history rows=%d dates=%d latest=%s",
        len(history_df),
        len(scoped_dates),
        scoped_dates[-1],
    )
    return scoped_dates, history_df


def sync_kr_ranking_history_rows(scoped_dates: list[str], archive_df: pd.DataFrame) -> None:
    if archive_df.empty or not scoped_dates:
        logging.info("Skip KR ranking archive sync: no archive rows")
        return

    target_engine = get_engine()
    try:
        with target_engine.begin() as target_conn:
            if not table_exists(target_conn, "public.daily_ranking"):
                raise RuntimeError("target public.daily_ranking missing after bootstrap")

            target_columns = get_table_columns(target_conn, "public.daily_ranking")
            target_column_names = [str(col["column_name"]) for col in target_columns]
            insert_columns = [name for name in archive_df.columns if name in target_column_names]
            if not insert_columns:
                logging.info("Skip KR ranking archive sync: no shared columns with target")
                return

            payload_df = archive_df.loc[:, insert_columns].copy()
            min_scoped_date = min(scoped_dates)
            target_conn.execute(
                text("DELETE FROM daily_ranking WHERE date < :min_scoped_date"),
                {"min_scoped_date": min_scoped_date},
            )
            target_conn.execute(
                text("DELETE FROM daily_ranking WHERE date::text = ANY(:scoped_dates)"),
                {"scoped_dates": scoped_dates},
            )
            payload_df.to_sql("daily_ranking", con=target_conn, if_exists="append", index=False)
            logging.info(
                "Synced KR ranking archive rows=%d dates=%d latest=%s",
                len(payload_df),
                len(scoped_dates),
                scoped_dates[-1],
            )
    finally:
        target_engine.dispose()


def reset_display_tables(*, include_trades: bool = False) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        delete_order = [
            (US_RANKING_SOURCE_TABLE, f"DELETE FROM {US_RANKING_SOURCE_TABLE}"),
            ("market.us_stock_universe", "DELETE FROM market.us_stock_universe"),
            ("paper.us_stock_paper_account_snapshot", "DELETE FROM paper.us_stock_paper_account_snapshot"),
            ("paper.us_stock_paper_position", "DELETE FROM paper.us_stock_paper_position"),
            ("paper.us_stock_paper_fill", "DELETE FROM paper.us_stock_paper_fill"),
            ("paper.us_stock_paper_order", "DELETE FROM paper.us_stock_paper_order"),
            ("paper.us_stock_paper_account", "DELETE FROM paper.us_stock_paper_account"),
            ("trade.us_paper_position_snapshot", "DELETE FROM trade.us_paper_position_snapshot"),
            ("trade.us_paper_sell_order", "DELETE FROM trade.us_paper_sell_order"),
            ("trade.us_sell_signal_log", "DELETE FROM trade.us_sell_signal_log"),
            ("trade.us_sell_decision_log", "DELETE FROM trade.us_sell_decision_log"),
            ("trade.us_paper_order", "DELETE FROM trade.us_paper_order"),
            ("trade.us_risk_guard_log", "DELETE FROM trade.us_risk_guard_log"),
            ("trade.us_buy_decision_log", "DELETE FROM trade.us_buy_decision_log"),
            ("trade.us_buy_candidate_log", "DELETE FROM trade.us_buy_candidate_log"),
            ("research.meaningfulness_review_note", "DELETE FROM research.meaningfulness_review_note"),
            ("research.paper_trading_position", "DELETE FROM research.paper_trading_position"),
            ("research.paper_trading_nav", "DELETE FROM research.paper_trading_nav"),
            ("research.paper_trading_run", "DELETE FROM research.paper_trading_run"),
            ("research.app_payload_store", "DELETE FROM research.app_payload_store"),
            ("public.fact_price_daily", "DELETE FROM public.fact_price_daily"),
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


def verify_daily_ranking_history_aware() -> dict[str, object]:
    csv_path = DATA_DIR / "ranking_final.csv"
    if not csv_path.exists():
        return {
            "status": "missing_csv",
            "csv_rows": 0,
            "db_rows": 0,
            "csv_latest_rows": 0,
            "db_latest_rows": 0,
            "csv_latest_date": None,
            "db_latest_date": None,
        }
    df = pd.read_csv(csv_path, dtype={"code": str}, low_memory=False)
    csv_rows = len(df.index)
    csv_latest_date = None
    csv_latest_rows = 0
    if csv_rows and "date" in df.columns:
        normalized_dates = pd.to_datetime(df["date"], errors="coerce")
        latest_ts = normalized_dates.max()
        if pd.notna(latest_ts):
            csv_latest_date = str(latest_ts.date())
            csv_latest_rows = int((normalized_dates.dt.strftime("%Y-%m-%d") == csv_latest_date).sum())
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) AS row_count,
                    MAX(date)::text AS latest_date,
                    COUNT(*) FILTER (WHERE date = (SELECT MAX(date) FROM daily_ranking)) AS latest_rows
                FROM daily_ranking
                """
            )
        ).mappings().one()
    db_rows = int(result["row_count"] or 0)
    db_latest_date = result["latest_date"]
    db_latest_rows = int(result["latest_rows"] or 0)
    status = "ok" if db_latest_date == csv_latest_date and db_latest_rows == csv_latest_rows else "mismatch"
    return {
        "status": status,
        "csv_rows": csv_rows,
        "db_rows": db_rows,
        "csv_latest_rows": csv_latest_rows,
        "db_latest_rows": db_latest_rows,
        "csv_latest_date": csv_latest_date,
        "db_latest_date": db_latest_date,
    }


def sync_core_tables() -> None:
    for spec in SYNC_TABLES:
        if str(spec["name"]) not in CORE_TABLES:
            continue
        result = sync_table(spec)
        if str(spec["name"]) == "stocks":
            verify = verify_stocks_subset()
        elif str(spec["name"]) == "daily_ranking":
            verify = verify_daily_ranking_history_aware()
        else:
            verify = verify_table(next(item for item in VERIFY_TABLES if item["name"] == spec["name"]))
        if verify["status"] != "ok":
            raise RuntimeError(f"core table parity failed: {spec['name']} -> {verify}")
        logging.info("Synced core table %s rows=%s latest=%s", spec["name"], result["csv_rows"], result["csv_latest_date"])


def ensure_us_ranking_table(conn) -> None:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS recommend"))
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS recommend.us_stock_rank_daily (
                trade_date              DATE        NOT NULL,
                symbol                  TEXT        NOT NULL,
                rank_no                 INTEGER,
                recommend_grade         TEXT,
                total_score             NUMERIC,
                momentum_score          NUMERIC,
                relative_strength_score NUMERIC,
                fundamental_score       NUMERIC,
                growth_score            NUMERIC,
                valuation_score         NUMERIC,
                risk_score              NUMERIC,
                feature_quality_score   NUMERIC,
                universe_group          TEXT,
                company_name            TEXT,
                sector                  TEXT,
                industry                TEXT,
                market_cap              NUMERIC,
                avg_volume              NUMERIC,
                is_etf                  BOOLEAN,
                is_active               BOOLEAN,
                data_status             TEXT,
                exclude_reason          TEXT,
                reason_summary          TEXT,
                score_detail_json       JSONB,
                source                  TEXT,
                created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, symbol, source)
            )
            """
        )
    )
    # ml_v1 도입 후 source 컬럼이 PK에 없으면 duplicate key 에러 발생.
    # 기존 (trade_date, symbol) PK를 (trade_date, symbol, source)로 마이그레이션.
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema = 'recommend'
                      AND table_name   = 'us_stock_rank_daily'
                      AND constraint_type = 'PRIMARY KEY'
                ) THEN
                    -- source 컬럼이 PK에 포함되지 않은 경우에만 재생성
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.key_column_usage
                        WHERE table_schema = 'recommend'
                          AND table_name   = 'us_stock_rank_daily'
                          AND constraint_name = (
                              SELECT constraint_name FROM information_schema.table_constraints
                              WHERE table_schema='recommend' AND table_name='us_stock_rank_daily'
                                AND constraint_type='PRIMARY KEY'
                          )
                          AND column_name = 'source'
                    ) THEN
                        ALTER TABLE recommend.us_stock_rank_daily
                            DROP CONSTRAINT us_stock_rank_daily_pkey;
                        ALTER TABLE recommend.us_stock_rank_daily
                            ADD PRIMARY KEY (trade_date, symbol, source);
                    END IF;
                END IF;
            END $$;
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_us_stock_rank_daily_trade_date
                ON recommend.us_stock_rank_daily (trade_date DESC)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_us_stock_rank_daily_grade
                ON recommend.us_stock_rank_daily (trade_date DESC, recommend_grade)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_us_stock_rank_daily_symbol_trade_date
                ON recommend.us_stock_rank_daily (symbol, trade_date DESC)
            """
        )
    )


def sync_us_ranking_table(source_database_url: str) -> None:
    if str(os.environ.get("SYNC_WEB_US_RANKING", "1")).strip() in ("0", "false", "no"):
        logging.info("Skip US ranking sync: SYNC_WEB_US_RANKING=0")
        return
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip US ranking sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip US ranking sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip US ranking sync: source and target DB are identical")
        return

    logging.info("Starting US ranking sync from %s", US_RANKING_SOURCE_TABLE)
    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as conn:
            source_exists = bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": US_RANKING_SOURCE_TABLE}).scalar())
            if not source_exists:
                logging.info("Skip US ranking sync: source table not found")
                return

            trade_dates = [
                row["trade_date"]
                for row in conn.execute(
                    text(
                        f"""
                        SELECT trade_date
                        FROM (
                            SELECT DISTINCT trade_date
                            FROM {US_RANKING_SOURCE_TABLE}
                            ORDER BY trade_date DESC
                            LIMIT :limit
                        ) recent
                        ORDER BY trade_date
                        """
                    ),
                    {"limit": US_RANKING_SYNC_DATES},
                ).mappings()
            ]
            if not trade_dates:
                logging.info("Skip US ranking sync: source table has no rows")
                return

            rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        f"""
                        SELECT
                            trade_date,
                            symbol,
                            rank_no,
                            recommend_grade,
                            total_score,
                            momentum_score,
                            relative_strength_score,
                            fundamental_score,
                            growth_score,
                            valuation_score,
                            risk_score,
                            feature_quality_score,
                            universe_group,
                            company_name,
                            sector,
                            industry,
                            market_cap,
                            avg_volume,
                            is_etf,
                            is_active,
                            data_status,
                            exclude_reason,
                            reason_summary,
                            score_detail_json,
                            source,
                            created_at,
                            updated_at
                        FROM {US_RANKING_SOURCE_TABLE}
                        WHERE trade_date = ANY(:trade_dates)
                        ORDER BY trade_date DESC, rank_no ASC NULLS LAST, symbol ASC
                        """
                    ),
                    {"trade_dates": trade_dates},
                ).mappings()
            ]
            logging.info(
                "Loaded US ranking source rows=%d dates=%d latest=%s",
                len(rows),
                len(trade_dates),
                trade_dates[-1].isoformat() if hasattr(trade_dates[-1], "isoformat") else str(trade_dates[-1]),
            )

        with target_engine.begin() as conn:
            ensure_us_web_supporting_tables(conn)
            conn.execute(
                text(f"DELETE FROM {US_RANKING_SOURCE_TABLE} WHERE trade_date = ANY(:trade_dates)"),
                {"trade_dates": trade_dates},
            )
            if rows:
                payload_rows = []
                for row in rows:
                    payload = dict(row)
                    payload["score_detail_json"] = json.dumps(payload.get("score_detail_json"), ensure_ascii=False, default=str)
                    payload_rows.append(payload)
                insert_stmt = text(
                    f"""
                    INSERT INTO {US_RANKING_SOURCE_TABLE} (
                        trade_date,
                        symbol,
                        rank_no,
                        recommend_grade,
                        total_score,
                        momentum_score,
                        relative_strength_score,
                        fundamental_score,
                        growth_score,
                        valuation_score,
                        risk_score,
                        feature_quality_score,
                        universe_group,
                        company_name,
                        sector,
                        industry,
                        market_cap,
                        avg_volume,
                        is_etf,
                        is_active,
                        data_status,
                        exclude_reason,
                        reason_summary,
                        score_detail_json,
                        source,
                        created_at,
                        updated_at
                    ) VALUES (
                        :trade_date,
                        :symbol,
                        :rank_no,
                        :recommend_grade,
                        :total_score,
                        :momentum_score,
                        :relative_strength_score,
                        :fundamental_score,
                        :growth_score,
                        :valuation_score,
                        :risk_score,
                        :feature_quality_score,
                        :universe_group,
                        :company_name,
                        :sector,
                        :industry,
                        :market_cap,
                        :avg_volume,
                        :is_etf,
                        :is_active,
                        :data_status,
                        :exclude_reason,
                        :reason_summary,
                        CAST(:score_detail_json AS jsonb),
                        :source,
                        :created_at,
                        :updated_at
                    )
                    """
                )
                total_batches = (len(payload_rows) + US_RANKING_INSERT_BATCH_SIZE - 1) // US_RANKING_INSERT_BATCH_SIZE
                for batch_index, start in enumerate(range(0, len(payload_rows), US_RANKING_INSERT_BATCH_SIZE), start=1):
                    batch = payload_rows[start : start + US_RANKING_INSERT_BATCH_SIZE]
                    conn.execute(insert_stmt, batch)
                    logging.info(
                        "Inserted US ranking batch %d/%d rows=%d",
                        batch_index,
                        total_batches,
                        len(batch),
                    )
        logging.info(
            "Synced US ranking table rows=%d dates=%d latest=%s",
            len(rows),
            len(trade_dates),
            trade_dates[-1].isoformat() if hasattr(trade_dates[-1], "isoformat") else str(trade_dates[-1]),
        )
    finally:
        source_engine.dispose()
        target_engine.dispose()


def sync_us_supporting_tables(source_database_url: str) -> None:
    logging.info("Starting supporting table sync: %s", FACT_PRICE_DAILY_SYNC_SPEC["table"])
    sync_table_history(
        source_database_url,
        table_name=str(FACT_PRICE_DAILY_SYNC_SPEC["table"]),
        date_column=FACT_PRICE_DAILY_SYNC_SPEC["date_column"],
        lookback_dates=FACT_PRICE_DAILY_SYNC_SPEC["lookback_dates"],
    )
    logging.info("Completed supporting table sync: %s", FACT_PRICE_DAILY_SYNC_SPEC["table"])

    if str(os.environ.get("SYNC_WEB_US_SUPPORTING", "1")).strip() in ("0", "false", "no"):
        logging.info("Skip US supporting table sync: SYNC_WEB_US_SUPPORTING=0")
        return

    for spec in US_WEB_TABLE_SYNC_SPECS:
        logging.info("Starting supporting table sync: %s", spec["table"])
        sync_table_history(
            source_database_url,
            table_name=str(spec["table"]),
            date_column=spec["date_column"],
            lookback_dates=spec["lookback_dates"],
        )
        logging.info("Completed supporting table sync: %s", spec["table"])


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
    # 수동매매 계좌 (KIS RULE 계좌) — 보유종목 0개여도 항상 동기화
    rule_summary_path = OUTPUT_DIR / "rule_account_holdings_summary.json"
    rule_summary = {}
    if rule_summary_path.exists():
        rule_summary = json.loads(rule_summary_path.read_text(encoding="utf-8-sig"))
    rule_items = read_csv_rows(DATA_DIR / "rule_account_holdings.csv") or []
    upsert_json_payload(
        "rule_account_holdings",
        {
            "entity": "rule_account_holdings",
            "generated_at": rule_summary.get("generated_at"),
            "holding_count": rule_summary.get("holding_count", 0),
            "cano_masked": rule_summary.get("cano_masked"),
            "cash_summary": rule_summary.get("cash_summary", {}),
            "items": rule_items,
        },
        generated_at=rule_summary.get("generated_at"),
        source_path=rule_summary_path,
    )
    logging.info("rule_account_holdings payload synced: %d items", len(rule_items))
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


def _ensure_us_macro_tables(conn) -> None:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS signal"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS signal.us_macro_feature_daily (
            id                          bigserial    PRIMARY KEY,
            us_trade_date               date         NOT NULL,
            kr_apply_date               date         NOT NULL,
            spy_ret_1d                  numeric,
            spy_ret_5d                  numeric,
            qqq_ret_1d                  numeric,
            qqq_ret_5d                  numeric,
            semiconductor_ret_1d        numeric,
            semiconductor_ret_5d        numeric,
            vix_ret_1d                  numeric,
            dxy_ret_1d                  numeric,
            tnx_ret_1d                  numeric,
            sector_breadth              numeric,
            top_sector                  varchar(100),
            bottom_sector               varchar(100),
            sector_strength_rank        jsonb,
            risk_on_flag                boolean,
            risk_off_flag               boolean,
            vix_spike_flag              boolean,
            semiconductor_strength_flag boolean,
            macro_status                varchar(50)  NOT NULL,
            macro_summary               text,
            data_source                 varchar(50)  NOT NULL DEFAULT 'yfinance',
            missing_tickers             text[],
            created_at                  timestamptz  NOT NULL DEFAULT now(),
            updated_at                  timestamptz  NOT NULL DEFAULT now(),
            UNIQUE (us_trade_date)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_us_macro_feature_kr_apply_date
            ON signal.us_macro_feature_daily (kr_apply_date DESC)
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS signal.kr_macro_overlay_log (
            id                  bigserial    PRIMARY KEY,
            run_date            date         NOT NULL,
            us_trade_date       date,
            kr_apply_date       date         NOT NULL,
            macro_status        varchar(50),
            engine_type         varchar(20)  NOT NULL,
            code                varchar(10)  NOT NULL,
            name                text,
            sector              text,
            original_score      numeric,
            macro_adjustment    numeric,
            adjusted_score      numeric,
            buy_blocked_flag    boolean,
            is_buy_candidate    boolean,
            overlay_reason      text,
            shadow_mode         boolean      NOT NULL DEFAULT true,
            created_at          timestamptz  NOT NULL DEFAULT now(),
            UNIQUE (run_date, engine_type, code)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_kr_macro_overlay_log_run_date
            ON signal.kr_macro_overlay_log (run_date DESC, engine_type)
    """))


def _ensure_flow_daily_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS public.flow_daily (
                date            DATE    NOT NULL,
                code            TEXT    NOT NULL,
                investor_type   TEXT    NOT NULL,
                net_buy_amount  DOUBLE PRECISION,
                net_buy_volume  DOUBLE PRECISION,
                fetch_status    TEXT,
                PRIMARY KEY (date, code, investor_type)
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_flow_daily_code_date
                ON public.flow_daily (code, date DESC)
            """
        )
    )
    # 웹DB 기존 테이블에 NOT NULL 컬럼이 있을 수 있으므로 nullable로 완화
    conn.execute(text("""
        DO $$
        DECLARE col TEXT;
        BEGIN
            FOR col IN
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'flow_daily'
                  AND is_nullable = 'NO'
                  AND column_name NOT IN ('date', 'code', 'investor_type')
            LOOP
                EXECUTE format('ALTER TABLE public.flow_daily ALTER COLUMN %I DROP NOT NULL', col);
            END LOOP;
        END $$;
    """))


def sync_flow_daily_table(source_database_url: str) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip flow_daily sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip flow_daily sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip flow_daily sync: source and target DB are identical")
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        with source_engine.connect() as conn:
            source_exists = bool(conn.execute(text("SELECT to_regclass('public.flow_daily')")).scalar())
            if not source_exists:
                logging.info("Skip flow_daily sync: source table not found")
                return

            scoped_dates = [
                row[0]
                for row in conn.execute(
                    text(
                        """
                        SELECT date FROM (
                            SELECT DISTINCT date FROM public.flow_daily
                            ORDER BY date DESC LIMIT :limit
                        ) recent ORDER BY date
                        """
                    ),
                    {"limit": FLOW_DAILY_SYNC_DATES},
                ).fetchall()
            ]
            if not scoped_dates:
                logging.info("Skip flow_daily sync: no source dates found")
                return

            rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        """
                        SELECT date, code, investor_type, net_buy_amount, net_buy_volume, fetch_status
                        FROM public.flow_daily
                        WHERE date = ANY(:scoped_dates)
                          AND fetch_status = 'success'
                          AND investor_type IN ('foreign', 'institution')
                        ORDER BY date DESC, code, investor_type
                        """
                    ),
                    {"scoped_dates": scoped_dates},
                ).mappings()
            ]

        with target_engine.begin() as conn:
            _ensure_flow_daily_table(conn)
            conn.execute(
                text("DELETE FROM public.flow_daily WHERE date = ANY(:scoped_dates)"),
                {"scoped_dates": scoped_dates},
            )
            if rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO public.flow_daily (
                            date, code, investor_type, net_buy_amount, net_buy_volume, fetch_status
                        ) VALUES (
                            :date, :code, :investor_type, :net_buy_amount, :net_buy_volume, :fetch_status
                        )
                        ON CONFLICT (date, code, investor_type) DO UPDATE SET
                            net_buy_amount = EXCLUDED.net_buy_amount,
                            net_buy_volume = EXCLUDED.net_buy_volume,
                            fetch_status   = EXCLUDED.fetch_status
                        """
                    ),
                    rows,
                )

        logging.info(
            "Synced flow_daily rows=%d dates=%d latest=%s",
            len(rows),
            len(scoped_dates),
            scoped_dates[-1].isoformat() if hasattr(scoped_dates[-1], "isoformat") else str(scoped_dates[-1]),
        )
    finally:
        source_engine.dispose()
        target_engine.dispose()


def sync_us_macro_signal_tables(source_database_url: str) -> None:
    source_url = str(source_database_url or "").strip()
    target_url = str(os.environ.get("DATABASE_URL", "")).strip()
    if not source_url:
        logging.info("Skip US macro signal sync: source DATABASE_URL not set")
        return
    if not target_url:
        logging.info("Skip US macro signal sync: target DATABASE_URL not set")
        return
    if source_url == target_url:
        logging.info("Skip US macro signal sync: source and target DB are identical")
        return

    source_engine = create_engine(source_url, future=True)
    target_engine = get_engine()
    try:
        # ── us_macro_feature_daily (last 90 days) ────────────────────────────
        with source_engine.connect() as conn:
            source_exists = bool(conn.execute(text("SELECT to_regclass('signal.us_macro_feature_daily')")).scalar())
            if not source_exists:
                logging.info("Skip US macro signal sync: signal.us_macro_feature_daily not found in source")
                return
            macro_rows = [
                dict(row)
                for row in conn.execute(text("""
                    SELECT us_trade_date, kr_apply_date,
                           spy_ret_1d, spy_ret_5d, qqq_ret_1d, qqq_ret_5d,
                           semiconductor_ret_1d, semiconductor_ret_5d,
                           vix_ret_1d, dxy_ret_1d, tnx_ret_1d,
                           sector_breadth, top_sector, bottom_sector,
                           sector_strength_rank,
                           risk_on_flag, risk_off_flag, vix_spike_flag, semiconductor_strength_flag,
                           macro_status, macro_summary, data_source, missing_tickers,
                           created_at, updated_at
                    FROM signal.us_macro_feature_daily
                    WHERE us_trade_date >= CURRENT_DATE - INTERVAL '90 days'
                    ORDER BY us_trade_date DESC
                """)).mappings()
            ]

        # ── kr_macro_overlay_log (last 30 run_dates) ─────────────────────────
        with source_engine.connect() as conn:
            overlay_source_exists = bool(conn.execute(text("SELECT to_regclass('signal.kr_macro_overlay_log')")).scalar())
            overlay_rows: list[dict] = []
            if overlay_source_exists:
                overlay_rows = [
                    dict(row)
                    for row in conn.execute(text("""
                        SELECT run_date, us_trade_date, kr_apply_date,
                               macro_status, engine_type,
                               code, name, sector,
                               original_score, macro_adjustment, adjusted_score,
                               buy_blocked_flag, is_buy_candidate,
                               overlay_reason, shadow_mode, created_at
                        FROM signal.kr_macro_overlay_log
                        WHERE run_date >= (
                            SELECT COALESCE(MIN(run_date), CURRENT_DATE - 30)
                            FROM (
                                SELECT DISTINCT run_date
                                FROM signal.kr_macro_overlay_log
                                ORDER BY run_date DESC
                                LIMIT 30
                            ) t
                        )
                        ORDER BY run_date DESC, engine_type, code
                    """)).mappings()
                ]

        # ── Write to target ───────────────────────────────────────────────────
        with target_engine.begin() as conn:
            _ensure_us_macro_tables(conn)
            for row in macro_rows:
                row["sector_strength_rank"] = (
                    json.dumps(row["sector_strength_rank"], ensure_ascii=False)
                    if row.get("sector_strength_rank") is not None else None
                )
                conn.execute(text("""
                    INSERT INTO signal.us_macro_feature_daily (
                        us_trade_date, kr_apply_date,
                        spy_ret_1d, spy_ret_5d, qqq_ret_1d, qqq_ret_5d,
                        semiconductor_ret_1d, semiconductor_ret_5d,
                        vix_ret_1d, dxy_ret_1d, tnx_ret_1d,
                        sector_breadth, top_sector, bottom_sector,
                        sector_strength_rank,
                        risk_on_flag, risk_off_flag, vix_spike_flag, semiconductor_strength_flag,
                        macro_status, macro_summary, data_source, missing_tickers,
                        created_at, updated_at
                    ) VALUES (
                        :us_trade_date, :kr_apply_date,
                        :spy_ret_1d, :spy_ret_5d, :qqq_ret_1d, :qqq_ret_5d,
                        :semiconductor_ret_1d, :semiconductor_ret_5d,
                        :vix_ret_1d, :dxy_ret_1d, :tnx_ret_1d,
                        :sector_breadth, :top_sector, :bottom_sector,
                        CAST(:sector_strength_rank AS jsonb),
                        :risk_on_flag, :risk_off_flag, :vix_spike_flag, :semiconductor_strength_flag,
                        :macro_status, :macro_summary, :data_source, :missing_tickers,
                        :created_at, :updated_at
                    )
                    ON CONFLICT (us_trade_date) DO UPDATE SET
                        kr_apply_date               = EXCLUDED.kr_apply_date,
                        spy_ret_1d                  = EXCLUDED.spy_ret_1d,
                        spy_ret_5d                  = EXCLUDED.spy_ret_5d,
                        qqq_ret_1d                  = EXCLUDED.qqq_ret_1d,
                        qqq_ret_5d                  = EXCLUDED.qqq_ret_5d,
                        semiconductor_ret_1d        = EXCLUDED.semiconductor_ret_1d,
                        semiconductor_ret_5d        = EXCLUDED.semiconductor_ret_5d,
                        vix_ret_1d                  = EXCLUDED.vix_ret_1d,
                        dxy_ret_1d                  = EXCLUDED.dxy_ret_1d,
                        tnx_ret_1d                  = EXCLUDED.tnx_ret_1d,
                        sector_breadth              = EXCLUDED.sector_breadth,
                        top_sector                  = EXCLUDED.top_sector,
                        bottom_sector               = EXCLUDED.bottom_sector,
                        sector_strength_rank        = EXCLUDED.sector_strength_rank,
                        risk_on_flag                = EXCLUDED.risk_on_flag,
                        risk_off_flag               = EXCLUDED.risk_off_flag,
                        vix_spike_flag              = EXCLUDED.vix_spike_flag,
                        semiconductor_strength_flag = EXCLUDED.semiconductor_strength_flag,
                        macro_status                = EXCLUDED.macro_status,
                        macro_summary               = EXCLUDED.macro_summary,
                        data_source                 = EXCLUDED.data_source,
                        missing_tickers             = EXCLUDED.missing_tickers,
                        updated_at                  = EXCLUDED.updated_at
                """), row)

            if overlay_rows:
                conn.execute(text("""
                    INSERT INTO signal.kr_macro_overlay_log (
                        run_date, us_trade_date, kr_apply_date,
                        macro_status, engine_type,
                        code, name, sector,
                        original_score, macro_adjustment, adjusted_score,
                        buy_blocked_flag, is_buy_candidate,
                        overlay_reason, shadow_mode, created_at
                    ) VALUES (
                        :run_date, :us_trade_date, :kr_apply_date,
                        :macro_status, :engine_type,
                        :code, :name, :sector,
                        :original_score, :macro_adjustment, :adjusted_score,
                        :buy_blocked_flag, :is_buy_candidate,
                        :overlay_reason, :shadow_mode, :created_at
                    )
                    ON CONFLICT (run_date, engine_type, code) DO UPDATE SET
                        us_trade_date    = EXCLUDED.us_trade_date,
                        kr_apply_date    = EXCLUDED.kr_apply_date,
                        macro_status     = EXCLUDED.macro_status,
                        original_score   = EXCLUDED.original_score,
                        macro_adjustment = EXCLUDED.macro_adjustment,
                        adjusted_score   = EXCLUDED.adjusted_score,
                        buy_blocked_flag = EXCLUDED.buy_blocked_flag,
                        is_buy_candidate = EXCLUDED.is_buy_candidate,
                        overlay_reason   = EXCLUDED.overlay_reason,
                        shadow_mode      = EXCLUDED.shadow_mode
                """), overlay_rows)

        logging.info(
            "Synced US macro signal tables: feature_daily=%d rows, overlay_log=%d rows",
            len(macro_rows), len(overlay_rows),
        )
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
        sync_kr_ranking_history(source_database_url)
        logging.info("Starting post-core US sync stages")
        sync_us_ranking_table(source_database_url)
        sync_us_supporting_tables(source_database_url)
        logging.info("Completed post-core US sync stages")
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
    sync_flow_daily_table(source_database_url)
    sync_us_macro_signal_tables(source_database_url)
    logging.info("Web display sync completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
