from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_stock_config, resolve_universe_csv_path
from python.us.us_db import deactivate_missing_universe_rows, fetch_universe_rows, upsert_universe_rows


LOGGER = logging.getLogger("us_universe")


@dataclass(frozen=True)
class UniverseRow:
    ticker: str
    name: str
    sector: str
    industry: str
    universe_tag: str


@dataclass(frozen=True)
class UniverseLoadResult:
    universe_tag: str
    source_path: Path
    total_csv_rows: int
    inserted: int
    updated: int
    skipped: int
    deactivated: int
    deactivate_missing: bool
    dry_run: bool


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_stock_config()
    parser = argparse.ArgumentParser(description="Load a US universe CSV into market.us_stock_universe.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag. Phase 1 supports NASDAQ100 only.")
    parser.add_argument("--csv", default=str(cfg.default_universe_csv), help="Universe CSV path.")
    parser.add_argument(
        "--deactivate-missing",
        action="store_true",
        help="Mark missing existing members as inactive instead of leaving them untouched.",
    )
    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Load date for added_date / removed_date defaults. Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the CSV and print the plan without writing to DB.",
    )
    return parser.parse_args()


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def _normalize_ticker(value: str | None) -> str:
    return _normalize_text(value).upper()


def read_universe_csv(csv_path: Path, universe_tag: str) -> list[UniverseRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {csv_path}")

        required = {"ticker", "name", "sector", "industry", "universe_tag"}
        missing_cols = required.difference({col.strip() for col in reader.fieldnames})
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

        rows: list[UniverseRow] = []
        seen_tickers: set[str] = set()
        for line_no, raw in enumerate(reader, start=2):
            ticker = _normalize_ticker(raw.get("ticker"))
            name = _normalize_text(raw.get("name"))
            sector = _normalize_text(raw.get("sector"))
            industry = _normalize_text(raw.get("industry"))
            row_universe = _normalize_text(raw.get("universe_tag")).upper()

            if not ticker:
                raise ValueError(f"CSV validation failed at line {line_no}: ticker is required")
            if not row_universe:
                raise ValueError(f"CSV validation failed at line {line_no}: universe_tag is required")
            if row_universe != universe_tag:
                raise ValueError(
                    f"CSV validation failed at line {line_no}: universe_tag='{row_universe}' "
                    f"does not match target universe '{universe_tag}'"
                )
            if ticker in seen_tickers:
                raise ValueError(f"CSV validation failed: duplicate ticker '{ticker}' in universe '{universe_tag}'")
            seen_tickers.add(ticker)
            rows.append(
                UniverseRow(
                    ticker=ticker,
                    name=name,
                    sector=sector,
                    industry=industry,
                    universe_tag=row_universe,
                )
            )

    return rows


def build_upsert_payload(rows: list[UniverseRow], load_date: date, data_source: str) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in rows:
        payload.append(
            {
                "ticker": row.ticker,
                "name": row.name or None,
                "sector": row.sector or None,
                "industry": row.industry or None,
                "universe_tag": row.universe_tag,
                "is_active": "Y",
                "added_date": load_date,
                "removed_date": None,
                "data_source": data_source,
            }
        )
    return payload


def load_universe(
    *,
    universe_tag: str,
    csv_path: Path,
    load_date: date,
    deactivate_missing: bool = False,
    dry_run: bool = False,
    data_source: str = "static_csv",
) -> UniverseLoadResult:
    universe_tag = _normalize_text(universe_tag).upper()

    LOGGER.info("[US_UNIVERSE] Loading universe=%s", universe_tag)
    LOGGER.info("[US_UNIVERSE] source=%s", csv_path)

    rows = read_universe_csv(csv_path, universe_tag)
    LOGGER.info("[US_UNIVERSE] total_csv_rows=%d", len(rows))

    existing_rows = fetch_universe_rows(universe_tag) if not dry_run else []
    existing_tickers = {str(row["ticker"]).upper() for row in existing_rows}
    csv_tickers = [row.ticker for row in rows]

    inserted = sum(1 for ticker in csv_tickers if ticker not in existing_tickers)
    updated = sum(1 for ticker in csv_tickers if ticker in existing_tickers)
    skipped = 0

    payload = build_upsert_payload(rows, load_date=load_date, data_source=data_source)

    if dry_run:
        deactivated = max(0, len([t for t in existing_tickers if t not in set(csv_tickers)])) if deactivate_missing else 0
    else:
        upsert_universe_rows(payload)
        deactivated = (
            deactivate_missing_universe_rows(
                universe_tag=universe_tag,
                keep_tickers=csv_tickers,
                removed_date=load_date,
            )
            if deactivate_missing
            else 0
        )

    LOGGER.info("[US_UNIVERSE] inserted=%d updated=%d skipped=%d", inserted, updated, skipped)
    LOGGER.info("[US_UNIVERSE] deactivate_missing=%s", str(deactivate_missing).lower())
    if deactivate_missing:
        LOGGER.info("[US_UNIVERSE] deactivated=%d", deactivated)
    LOGGER.info("[US_UNIVERSE] Completed")

    return UniverseLoadResult(
        universe_tag=universe_tag,
        source_path=csv_path,
        total_csv_rows=len(rows),
        inserted=inserted,
        updated=updated,
        skipped=skipped,
        deactivated=deactivated,
        deactivate_missing=deactivate_missing,
        dry_run=dry_run,
    )


def main() -> None:
    setup_logging()
    args = parse_args()

    universe_tag = _normalize_text(args.universe).upper()
    csv_path = resolve_universe_csv_path(universe_tag, args.csv)
    load_date = date.fromisoformat(str(args.as_of_date))
    load_universe(
        universe_tag=universe_tag,
        csv_path=csv_path,
        load_date=load_date,
        deactivate_missing=bool(args.deactivate_missing),
        dry_run=bool(args.dry_run),
        data_source="static_csv",
    )


if __name__ == "__main__":
    main()
