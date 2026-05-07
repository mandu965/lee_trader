from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import create_engine, text

import db as db_module
from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "postgres" / "analytics_live_trade_views.sql"
REQUIRED_LIVE_TRADE_VIEWS = {
    "live_trade_fact",
    "live_trade_return_fact",
    "live_review_kpi",
    "live_score_bucket_kpi",
    "live_quality_guard_kpi",
    "live_closed_trade",
}


def _read_sql() -> str:
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"analytics view SQL not found: {SQL_PATH}")
    return SQL_PATH.read_text(encoding="utf-8")


def _existing_live_trade_views(engine) -> set[str]:
    sql = """
    SELECT viewname
    FROM pg_views
    WHERE schemaname = 'analytics'
      AND viewname = ANY(:view_names)
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"view_names": sorted(REQUIRED_LIVE_TRADE_VIEWS)}).mappings().all()
    return {str(row.get("viewname") or "").strip() for row in rows if row.get("viewname")}


def apply_analytics_views(*, database_url: str | None = None) -> None:
    if database_url:
        engine = create_engine(database_url, future=True)
    else:
        engine = get_engine()

    try:
        sql = _read_sql()
    except FileNotFoundError as exc:
        existing_views = _existing_live_trade_views(engine)
        if REQUIRED_LIVE_TRADE_VIEWS.issubset(existing_views):
            print(
                "analytics_views_apply_skipped: SQL file missing but required analytics views already exist "
                f"({SQL_PATH})"
            )
            return
        missing = ", ".join(sorted(REQUIRED_LIVE_TRADE_VIEWS - existing_views)) or "unknown"
        raise FileNotFoundError(
            f"{exc}. Required analytics views are also missing: {missing}. "
            "Restore postgres/analytics_live_trade_views.sql into the runtime image or run with an image built "
            "without excluding postgres/*.sql."
        ) from exc

    with engine.begin() as conn:
        conn.execute(text(sql))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or replace analytics live-trade views.")
    parser.add_argument(
        "--target",
        choices=["database", "web"],
        default="database",
        help="database uses DATABASE_URL; web uses WEB_DATABASE_URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target == "web":
        url = str(os.environ.get("WEB_DATABASE_URL", "")).strip()
        if not url:
            raise RuntimeError("WEB_DATABASE_URL is required for --target web")
        apply_analytics_views(database_url=url)
        print("analytics_views_applied: web")
    else:
        db_module.get_database_url.cache_clear()
        db_module.get_engine.cache_clear()
        apply_analytics_views()
        print("analytics_views_applied: database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
