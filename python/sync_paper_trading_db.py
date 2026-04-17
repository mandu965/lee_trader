from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

POSITIONS_CSV = DATA_DIR / "paper_trading_positions.csv"
NAV_CSV = DATA_DIR / "paper_trading_nav.csv"
REPORT_MD = OUTPUT_DIR / "paper_trading_report.md"

POSITION_COLUMNS = [
    "paper_run_id",
    "strategy",
    "code",
    "name",
    "entry_date",
    "planned_exit_date",
    "exit_date",
    "entry_price_close",
    "entry_exec_price",
    "exit_price_close",
    "exit_exec_price",
    "shares",
    "entry_notional_gross",
    "exit_notional_net",
    "entry_cost_amount",
    "exit_cost_amount",
    "gross_return",
    "net_return",
    "source_rank",
    "selection_stage",
    "dominant_theme",
    "confidence_score",
    "final_score",
    "holding_age_trading_days",
    "remaining_holding_days",
    "holding_policy_code",
    "entry_action_code",
    "entry_action_reason",
    "current_action_code",
    "current_action_reason",
    "exit_action_code",
    "exit_action_reason",
    "status",
]

NAV_COLUMNS = [
    "paper_run_id",
    "strategy",
    "date",
    "cash",
    "market_value",
    "nav",
    "daily_return",
    "active_position_count",
    "opened_today",
    "duplicate_skip_count",
    "deployed_cash",
    "cumulative_return",
    "running_nav_max",
    "drawdown",
    "closed_trade_count",
    "closed_win_rate",
    "closed_win_count",
    "closed_trade_count_cum",
    "closed_win_count_cum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync paper-trading CSV artifacts into Postgres research tables.")
    parser.add_argument("--positions-csv", type=Path, default=POSITIONS_CSV)
    parser.add_argument("--nav-csv", type=Path, default=NAV_CSV)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--run-tag", default=None, help="Stable identifier for the synced paper-trading run.")
    parser.add_argument("--source-mode", default="historical_snapshot_ledger")
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--entry-fee-bps", type=float, default=0.0)
    parser.add_argument("--exit-fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=0.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=0.0)
    parser.add_argument("--comment", default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text_value = str(value).strip()
    return text_value or None


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    work = df.copy()
    work["strategy"] = work.get("strategy", pd.Series("", index=work.index)).fillna("").astype(str)
    work["code"] = work.get("code", pd.Series("", index=work.index)).astype(str).str.zfill(6)
    for col in [
        "name",
        "selection_stage",
        "dominant_theme",
        "holding_policy_code",
        "entry_action_code",
        "entry_action_reason",
        "current_action_code",
        "current_action_reason",
        "exit_action_code",
        "exit_action_reason",
        "status",
    ]:
        work[col] = work.get(col, pd.Series("", index=work.index)).map(_clean_text)
    for col in ["entry_date", "planned_exit_date", "exit_date"]:
        work[col] = pd.to_datetime(work.get(col, pd.Series(pd.NA, index=work.index)), errors="coerce").dt.date
    for col in [
        "entry_price_close",
        "entry_exec_price",
        "exit_price_close",
        "exit_exec_price",
        "shares",
        "entry_notional_gross",
        "exit_notional_net",
        "entry_cost_amount",
        "exit_cost_amount",
        "gross_return",
        "net_return",
        "confidence_score",
        "final_score",
    ]:
        work[col] = pd.to_numeric(work.get(col, pd.Series(pd.NA, index=work.index)), errors="coerce")
    for col in ["holding_age_trading_days", "remaining_holding_days"]:
        work[col] = pd.to_numeric(work.get(col, pd.Series(pd.NA, index=work.index)), errors="coerce").astype("Int64")
    work["source_rank"] = pd.to_numeric(work.get("source_rank", pd.Series(pd.NA, index=work.index)), errors="coerce").astype("Int64")
    work = work.drop_duplicates(subset=["strategy", "code", "entry_date"], keep="last").reset_index(drop=True)
    return work


def normalize_nav(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=NAV_COLUMNS)
    work = df.copy()
    work["strategy"] = work.get("strategy", pd.Series("", index=work.index)).fillna("").astype(str)
    work["date"] = pd.to_datetime(work.get("date", pd.Series(pd.NA, index=work.index)), errors="coerce").dt.date
    for col in [
        "cash",
        "market_value",
        "nav",
        "daily_return",
        "deployed_cash",
        "cumulative_return",
        "running_nav_max",
        "drawdown",
        "closed_win_rate",
    ]:
        work[col] = pd.to_numeric(work.get(col, pd.Series(pd.NA, index=work.index)), errors="coerce")
    for col in [
        "active_position_count",
        "opened_today",
        "duplicate_skip_count",
        "closed_trade_count",
        "closed_win_count",
        "closed_trade_count_cum",
        "closed_win_count_cum",
    ]:
        work[col] = pd.to_numeric(work.get(col, pd.Series(pd.NA, index=work.index)), errors="coerce").astype("Int64")
    work = work.drop_duplicates(subset=["strategy", "date"], keep="last").reset_index(drop=True)
    return work


def infer_asof_date(nav: pd.DataFrame, positions: pd.DataFrame) -> date | None:
    nav_dates = pd.to_datetime(nav.get("date"), errors="coerce")
    if not nav_dates.dropna().empty:
        return nav_dates.max().date()
    pos_dates = pd.to_datetime(positions.get("entry_date"), errors="coerce")
    if not pos_dates.dropna().empty:
        return pos_dates.max().date()
    return None


def ensure_tables() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS research"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.paper_trading_run (
                    paper_run_id BIGSERIAL PRIMARY KEY,
                    run_tag TEXT NOT NULL UNIQUE,
                    source_mode TEXT NOT NULL,
                    asof_date DATE,
                    hold_days INTEGER NOT NULL,
                    initial_nav NUMERIC,
                    entry_fee_bps NUMERIC,
                    exit_fee_bps NUMERIC,
                    entry_slippage_bps NUMERIC,
                    exit_slippage_bps NUMERIC,
                    positions_row_count INTEGER NOT NULL DEFAULT 0,
                    nav_row_count INTEGER NOT NULL DEFAULT 0,
                    source_positions_csv TEXT,
                    source_nav_csv TEXT,
                    source_report_md TEXT,
                    comment TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.paper_trading_position (
                    paper_run_id BIGINT NOT NULL REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL,
                    code VARCHAR(10) NOT NULL,
                    name TEXT,
                    entry_date DATE NOT NULL,
                    planned_exit_date DATE,
                    exit_date DATE,
                    entry_price_close NUMERIC,
                    entry_exec_price NUMERIC,
                    exit_price_close NUMERIC,
                    exit_exec_price NUMERIC,
                    shares NUMERIC,
                    entry_notional_gross NUMERIC,
                    exit_notional_net NUMERIC,
                    entry_cost_amount NUMERIC,
                    exit_cost_amount NUMERIC,
                    gross_return NUMERIC,
                    net_return NUMERIC,
                    source_rank INTEGER,
                    selection_stage TEXT,
                    dominant_theme TEXT,
                    confidence_score NUMERIC,
                    final_score NUMERIC,
                    holding_age_trading_days INTEGER,
                    remaining_holding_days INTEGER,
                    holding_policy_code TEXT,
                    entry_action_code TEXT,
                    entry_action_reason TEXT,
                    current_action_code TEXT,
                    current_action_reason TEXT,
                    exit_action_code TEXT,
                    exit_action_reason TEXT,
                    status TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (paper_run_id, strategy, code, entry_date)
                )
                """
            )
        )
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS holding_age_trading_days INTEGER"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS remaining_holding_days INTEGER"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS holding_policy_code TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS entry_action_code TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS entry_action_reason TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS current_action_code TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS current_action_reason TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS exit_action_code TEXT"))
        conn.execute(text("ALTER TABLE research.paper_trading_position ADD COLUMN IF NOT EXISTS exit_action_reason TEXT"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.paper_trading_nav (
                    paper_run_id BIGINT NOT NULL REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL,
                    date DATE NOT NULL,
                    cash NUMERIC,
                    market_value NUMERIC,
                    nav NUMERIC,
                    daily_return NUMERIC,
                    active_position_count INTEGER,
                    opened_today INTEGER,
                    duplicate_skip_count INTEGER,
                    deployed_cash NUMERIC,
                    cumulative_return NUMERIC,
                    running_nav_max NUMERIC,
                    drawdown NUMERIC,
                    closed_trade_count INTEGER,
                    closed_win_rate NUMERIC,
                    closed_win_count INTEGER,
                    closed_trade_count_cum INTEGER,
                    closed_win_count_cum INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (paper_run_id, strategy, date)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_trading_run_asof
                ON research.paper_trading_run(asof_date DESC, paper_run_id DESC)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_trading_position_lookup
                ON research.paper_trading_position(strategy, entry_date DESC, code)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_trading_nav_lookup
                ON research.paper_trading_nav(strategy, date DESC)
                """
            )
        )


def upsert_run(
    *,
    run_tag: str,
    source_mode: str,
    asof_date: datetime.date | None,
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    args: argparse.Namespace,
) -> int:
    engine = get_engine()
    params = {
        "run_tag": run_tag,
        "source_mode": source_mode,
        "asof_date": asof_date,
        "hold_days": args.hold_days,
        "initial_nav": args.initial_nav,
        "entry_fee_bps": args.entry_fee_bps,
        "exit_fee_bps": args.exit_fee_bps,
        "entry_slippage_bps": args.entry_slippage_bps,
        "exit_slippage_bps": args.exit_slippage_bps,
        "positions_row_count": int(len(positions)),
        "nav_row_count": int(len(nav)),
        "source_positions_csv": str(_resolve(args.positions_csv).relative_to(ROOT)),
        "source_nav_csv": str(_resolve(args.nav_csv).relative_to(ROOT)),
        "source_report_md": str(_resolve(args.report_md).relative_to(ROOT)),
        "comment": args.comment,
    }
    query = text(
        """
        INSERT INTO research.paper_trading_run (
            run_tag,
            source_mode,
            asof_date,
            hold_days,
            initial_nav,
            entry_fee_bps,
            exit_fee_bps,
            entry_slippage_bps,
            exit_slippage_bps,
            positions_row_count,
            nav_row_count,
            source_positions_csv,
            source_nav_csv,
            source_report_md,
            comment
        )
        VALUES (
            :run_tag,
            :source_mode,
            :asof_date,
            :hold_days,
            :initial_nav,
            :entry_fee_bps,
            :exit_fee_bps,
            :entry_slippage_bps,
            :exit_slippage_bps,
            :positions_row_count,
            :nav_row_count,
            :source_positions_csv,
            :source_nav_csv,
            :source_report_md,
            :comment
        )
        ON CONFLICT (run_tag) DO UPDATE SET
            source_mode = EXCLUDED.source_mode,
            asof_date = EXCLUDED.asof_date,
            hold_days = EXCLUDED.hold_days,
            initial_nav = EXCLUDED.initial_nav,
            entry_fee_bps = EXCLUDED.entry_fee_bps,
            exit_fee_bps = EXCLUDED.exit_fee_bps,
            entry_slippage_bps = EXCLUDED.entry_slippage_bps,
            exit_slippage_bps = EXCLUDED.exit_slippage_bps,
            positions_row_count = EXCLUDED.positions_row_count,
            nav_row_count = EXCLUDED.nav_row_count,
            source_positions_csv = EXCLUDED.source_positions_csv,
            source_nav_csv = EXCLUDED.source_nav_csv,
            source_report_md = EXCLUDED.source_report_md,
            comment = EXCLUDED.comment,
            updated_at = now()
        RETURNING paper_run_id
        """
    )
    with engine.begin() as conn:
        return int(conn.execute(query, params).scalar_one())


def replace_child_rows(paper_run_id: int, positions: pd.DataFrame, nav: pd.DataFrame) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM research.paper_trading_position WHERE paper_run_id = :paper_run_id"), {"paper_run_id": paper_run_id})
        conn.execute(text("DELETE FROM research.paper_trading_nav WHERE paper_run_id = :paper_run_id"), {"paper_run_id": paper_run_id})

    if not positions.empty:
        positions_out = positions.copy()
        positions_out["paper_run_id"] = paper_run_id
        positions_out = positions_out[POSITION_COLUMNS]
        positions_out.to_sql("paper_trading_position", engine, schema="research", if_exists="append", index=False, method="multi")

    if not nav.empty:
        nav_out = nav.copy()
        nav_out["paper_run_id"] = paper_run_id
        nav_out = nav_out[NAV_COLUMNS]
        nav_out.to_sql("paper_trading_nav", engine, schema="research", if_exists="append", index=False, method="multi")


def main() -> int:
    setup_logging()
    args = parse_args()

    positions = normalize_positions(_read_csv(args.positions_csv, dtype={"code": str}))
    nav = normalize_nav(_read_csv(args.nav_csv))
    if positions.empty and nav.empty:
        raise FileNotFoundError("paper trading CSV artifacts are missing or empty")
    try:
        ensure_tables()

        asof_date = infer_asof_date(nav, positions)
        run_tag = args.run_tag or f"paper_trading_ledger:{asof_date.isoformat() if asof_date else 'unknown'}"
        paper_run_id = upsert_run(
            run_tag=run_tag,
            source_mode=args.source_mode,
            asof_date=asof_date,
            positions=positions,
            nav=nav,
            args=args,
        )
        replace_child_rows(paper_run_id, positions, nav)
    except SQLAlchemyError as exc:
        logging.error("paper trading db sync failed: %s", exc)
        return 1

    logging.info(
        "synced paper trading to db run_tag=%s paper_run_id=%s asof_date=%s positions=%d nav_rows=%d",
        run_tag,
        paper_run_id,
        asof_date,
        len(positions),
        len(nav),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
