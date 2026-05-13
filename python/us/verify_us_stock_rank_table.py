from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text

from python.us.us_config import load_us_ranking_table_config
from python.us.us_db import delete_us_rank_rows, get_us_engine, relation_exists, upsert_us_rank_rows
from python.us.us_rank_design import build_sample_rank_rows


LOGGER = logging.getLogger("us_rank_verify")
DEFAULT_SAMPLE_SOURCE = "phase3_2_dryrun"


@dataclass(frozen=True)
class VerificationResult:
    ddl_ready: bool
    pk_ok: bool
    duplicate_guard_ok: bool
    index_ok: bool
    sample_insert_ok: bool
    top20_query_ok: bool
    grade_store_ok: bool
    json_store_ok: bool


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_ranking_table_config()
    parser = argparse.ArgumentParser(description="Verify recommend.us_stock_rank_daily table design.")
    parser.add_argument("--write-sample", action="store_true", help="Insert sample rows for DB validation.")
    parser.add_argument("--cleanup", action="store_true", help="Delete validation sample rows after checks.")
    parser.add_argument(
        "--trade-date",
        default=cfg.verify_sample_trade_date.isoformat(),
        help="Validation trade date. Default uses a reserved future date to avoid live collisions.",
    )
    parser.add_argument("--source", default=DEFAULT_SAMPLE_SOURCE, help="Sample source tag.")
    return parser.parse_args()


def _count_indexes() -> int:
    engine = get_us_engine()
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)::integer
                    FROM pg_indexes
                    WHERE schemaname = 'recommend'
                      AND tablename = 'us_stock_rank_daily'
                      AND indexname IN (
                          'idx_us_stock_rank_daily_trade_date_rank_no',
                          'idx_us_stock_rank_daily_trade_date_grade',
                          'idx_us_stock_rank_daily_symbol_trade_date'
                      )
                    """
                )
            ).scalar_one()
        )


def _pk_columns() -> list[str]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_class c
                  ON c.oid = i.indrelid
                JOIN pg_namespace n
                  ON n.oid = c.relnamespace
                JOIN pg_attribute a
                  ON a.attrelid = c.oid
                 AND a.attnum = ANY(i.indkey)
                WHERE n.nspname = 'recommend'
                  AND c.relname = 'us_stock_rank_daily'
                  AND i.indisprimary = true
                ORDER BY array_position(i.indkey, a.attnum)
                """
            )
        ).scalars().all()
    return [str(value) for value in rows]


def _top20_count(trade_date: date, source: str) -> int:
    engine = get_us_engine()
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*)::integer
                    FROM (
                        SELECT
                            trade_date,
                            rank_no,
                            symbol,
                            company_name,
                            recommend_grade,
                            total_score,
                            momentum_score,
                            relative_strength_score,
                            fundamental_score,
                            growth_score,
                            valuation_score,
                            risk_score,
                            reason_summary
                        FROM recommend.us_stock_rank_daily
                        WHERE trade_date = :trade_date
                          AND source = :source
                        ORDER BY rank_no
                        LIMIT 20
                    ) ranked
                    """
                ),
                {"trade_date": trade_date, "source": source},
            ).scalar_one()
        )


def verify_rank_table(*, trade_date: date, source: str, write_sample: bool, cleanup: bool) -> VerificationResult:
    ddl_ready = relation_exists("recommend.us_stock_rank_daily")
    pk_ok = False
    duplicate_guard_ok = False
    index_ok = False
    sample_insert_ok = False
    top20_query_ok = False
    grade_store_ok = False
    json_store_ok = False

    if ddl_ready:
        pk_ok = _pk_columns() == ["trade_date", "symbol"]
        index_ok = _count_indexes() == 3

    if ddl_ready and write_sample:
        sample_rows = build_sample_rank_rows(trade_date=trade_date, source=source)
        inserted = upsert_us_rank_rows(sample_rows)
        sample_insert_ok = inserted == len(sample_rows)
        top20_query_ok = _top20_count(trade_date, source) == len(sample_rows)

        engine = get_us_engine()
        with engine.connect() as conn:
            grade_store_ok = bool(
                conn.execute(
                    text(
                        """
                        SELECT COUNT(*)::integer
                        FROM recommend.us_stock_rank_daily
                        WHERE trade_date = :trade_date
                          AND source = :source
                          AND recommend_grade IS NOT NULL
                        """
                    ),
                    {"trade_date": trade_date, "source": source},
                ).scalar_one()
            )
            json_store_ok = bool(
                conn.execute(
                    text(
                        """
                        SELECT COUNT(*)::integer
                        FROM recommend.us_stock_rank_daily
                        WHERE trade_date = :trade_date
                          AND source = :source
                          AND score_detail_json IS NOT NULL
                          AND jsonb_typeof(score_detail_json) = 'object'
                        """
                    ),
                    {"trade_date": trade_date, "source": source},
                ).scalar_one()
            )

            duplicate_guard_ok = False
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO recommend.us_stock_rank_daily (
                            trade_date,
                            symbol,
                            rank_no,
                            recommend_grade,
                            source
                        ) VALUES (
                            :trade_date,
                            :symbol,
                            999,
                            'WATCH',
                            :source
                        )
                        """
                    ),
                    {"trade_date": trade_date, "symbol": "AAPL", "source": "pk_probe"},
                )
            except Exception:
                duplicate_guard_ok = True

        if cleanup:
            delete_us_rank_rows(trade_date=trade_date, source=source)
    elif ddl_ready:
        duplicate_guard_ok = pk_ok

    return VerificationResult(
        ddl_ready=ddl_ready,
        pk_ok=pk_ok,
        duplicate_guard_ok=duplicate_guard_ok,
        index_ok=index_ok,
        sample_insert_ok=sample_insert_ok,
        top20_query_ok=top20_query_ok,
        grade_store_ok=grade_store_ok,
        json_store_ok=json_store_ok,
    )


def main() -> None:
    cfg = load_us_ranking_table_config()
    setup_logging(cfg.log_level)
    args = parse_args()
    trade_date = date.fromisoformat(str(args.trade_date))
    result = verify_rank_table(
        trade_date=trade_date,
        source=str(args.source),
        write_sample=bool(args.write_sample),
        cleanup=bool(args.cleanup),
    )
    LOGGER.info("[verify]")
    LOGGER.info("- ddl_ready: %s", "ok" if result.ddl_ready else "fail")
    LOGGER.info("- pk_ok: %s", "ok" if result.pk_ok else "fail")
    LOGGER.info("- duplicate_guard_ok: %s", "ok" if result.duplicate_guard_ok else "fail")
    LOGGER.info("- index_ok: %s", "ok" if result.index_ok else "fail")
    LOGGER.info("- sample_insert_ok: %s", "ok" if result.sample_insert_ok else "not-run/fail")
    LOGGER.info("- grade_store_ok: %s", "ok" if result.grade_store_ok else "not-run/fail")
    LOGGER.info("- json_store_ok: %s", "ok" if result.json_store_ok else "not-run/fail")
    LOGGER.info("- top20_query_ok: %s", "ok" if result.top20_query_ok else "not-run/fail")


if __name__ == "__main__":
    main()
