"""Build sector-relative strength features into feature.us_stock_feature_daily.

Computes per-stock momentum relative to sector peers:
  - sector_rel_ret_20d: stock ret_20d minus sector-average ret_20d
  - sector_rel_ret_60d: stock ret_60d minus sector-average ret_60d
  - sector_rank_pct: percentile rank within sector by ret_20d (0=worst, 1=best)

Runs after build_us_features (needs ret_20d/ret_60d to be populated first).
Reads all tickers in a single batch query — efficient for group operations.
"""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_db import get_us_engine


LOGGER = logging.getLogger("us_sector_feature")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sector-relative strength features.")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--universe", default="NASDAQ100", help="Universe tag")
    return parser.parse_args()


def _load_sector_map(universe_tag: str) -> dict[str, str]:
    sql = text("""
        SELECT ticker, sector
        FROM market.us_stock_universe
        WHERE is_active = 'Y'
          AND universe_tag = :tag
          AND sector IS NOT NULL
          AND sector != 'ETF'
    """)
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, {"tag": universe_tag}).fetchall()
    return {r[0]: r[1] for r in rows}


def _load_feature_daily(
    tickers: list[str],
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    clauses = ["ticker = ANY(:tickers)", "ret_20d IS NOT NULL"]
    params: dict = {"tickers": tickers}
    if start_date:
        clauses.append("feature_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        clauses.append("feature_date <= :end_date")
        params["end_date"] = end_date
    where = " AND ".join(clauses)
    sql = text(f"SELECT ticker, feature_date, ret_20d, ret_60d FROM feature.us_stock_feature_daily WHERE {where}")
    engine = get_us_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def _compute_sector_features(df: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map)
    df = df.dropna(subset=["sector", "ret_20d"])

    # Sector average per date
    sector_avg = (
        df.groupby(["feature_date", "sector"])[["ret_20d", "ret_60d"]]
        .transform("mean")
    )
    df["sector_rel_ret_20d"] = df["ret_20d"] - sector_avg["ret_20d"]
    df["sector_rel_ret_60d"] = df["ret_60d"].sub(sector_avg["ret_60d"]).where(df["ret_60d"].notna())

    # Percentile rank within sector by ret_20d (0=worst, 1=best)
    df["sector_rank_pct"] = df.groupby(["feature_date", "sector"])["ret_20d"].rank(pct=True)

    return df[["ticker", "feature_date", "sector_rel_ret_20d", "sector_rel_ret_60d", "sector_rank_pct"]]


def _upsert_sector_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = text("""
        UPDATE feature.us_stock_feature_daily
        SET
            sector_rel_ret_20d = :sector_rel_ret_20d,
            sector_rel_ret_60d = :sector_rel_ret_60d,
            sector_rank_pct = :sector_rank_pct,
            updated_at = now()
        WHERE ticker = :ticker AND feature_date = :feature_date
    """)
    engine = get_us_engine()
    # Batch update in chunks to avoid locking too many rows
    chunk_size = 2000
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        with engine.begin() as conn:
            conn.execute(sql, chunk)
        total += len(chunk)
    return total


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        import math
        f = float(val)  # type: ignore[arg-type]
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def build_us_sector_features(
    *,
    universe_tag: str = "NASDAQ100",
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    sector_map = _load_sector_map(universe_tag)
    if not sector_map:
        LOGGER.warning("[US_SECTOR] No sector data found for universe=%s", universe_tag)
        return 0
    LOGGER.info("[US_SECTOR] Loaded sector map tickers=%s sectors=%s", len(sector_map), len(set(sector_map.values())))

    tickers = list(sector_map.keys())
    df = _load_feature_daily(tickers, start_date=start_date, end_date=end_date)
    if df.empty:
        LOGGER.warning("[US_SECTOR] No feature_daily rows found")
        return 0
    LOGGER.info("[US_SECTOR] Loaded feature_daily rows=%s dates=%s", len(df), df["feature_date"].nunique())

    result = _compute_sector_features(df, sector_map)
    LOGGER.info("[US_SECTOR] Computed sector features rows=%s", len(result))

    rows = []
    for rec in result.itertuples(index=False):
        rows.append({
            "ticker": rec.ticker,
            "feature_date": rec.feature_date,
            "sector_rel_ret_20d": _safe_float(rec.sector_rel_ret_20d),
            "sector_rel_ret_60d": _safe_float(rec.sector_rel_ret_60d),
            "sector_rank_pct": _safe_float(rec.sector_rank_pct),
        })

    updated = _upsert_sector_rows(rows)
    LOGGER.info("[US_SECTOR] Done updated=%s rows", updated)
    return updated


def main() -> None:
    setup_logging()
    args = parse_args()
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    try:
        from python.us.us_db import get_us_engine
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_SECTOR] DB connection failed: {exc}") from exc

    build_us_sector_features(
        universe_tag=args.universe,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
