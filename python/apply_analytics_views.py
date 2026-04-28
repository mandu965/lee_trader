from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import create_engine, text

import db as db_module
from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "postgres" / "analytics_live_trade_views.sql"


def _read_sql() -> str:
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"analytics view SQL not found: {SQL_PATH}")
    return SQL_PATH.read_text(encoding="utf-8")


def apply_analytics_views(*, database_url: str | None = None) -> None:
    sql = _read_sql()
    if database_url:
        engine = create_engine(database_url, future=True)
    else:
        engine = get_engine()
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
