from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_universe_filter_config
from python.us.us_db import fetch_meta_us_universe_rows, get_active_us_stock_universe, upsert_meta_us_universe_rows


LOGGER = logging.getLogger("us_meta_universe")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
MANUAL_ETFS = [
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP",
    "XLU", "XLB", "XLRE", "SMH", "SOXX",
]
LEVERAGED_OR_INVERSE = {
    "TQQQ": ("leveraged", "Leveraged ETF excluded from recommendation universe."),
    "SQQQ": ("inverse", "Inverse ETF excluded from recommendation universe."),
    "SOXL": ("leveraged", "Leveraged ETF excluded from recommendation universe."),
    "SOXS": ("inverse", "Inverse ETF excluded from recommendation universe."),
    "SPXL": ("leveraged", "Leveraged ETF excluded from recommendation universe."),
    "SPXS": ("inverse", "Inverse ETF excluded from recommendation universe."),
    "UPRO": ("leveraged", "Leveraged ETF excluded from recommendation universe."),
    "SH": ("inverse", "Inverse ETF excluded from recommendation universe."),
    "PSQ": ("inverse", "Inverse ETF excluded from recommendation universe."),
    "QID": ("inverse", "Inverse ETF excluded from recommendation universe."),
}
REQUEST_HEADERS = {
    "User-Agent": "lee_trader_project_c/phase3-1 (contact: local-dev-script)"
}


@dataclass(frozen=True)
class UniverseSeedRow:
    symbol: str
    company_name: str | None
    sector: str | None
    industry: str | None
    universe_group: str
    is_etf: bool
    exchange: str | None = None
    country: str | None = "US"
    currency: str | None = "USD"


@dataclass(frozen=True)
class InitUniverseResult:
    total_input_count: int
    deduped_count: int
    sp500_count: int
    nasdaq100_count: int
    etf_count: int
    active_count: int
    inactive_count: int
    excluded_count: int
    upserted_count: int
    error_count: int


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize meta.us_stock_universe for Phase 3-1.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate rows without writing to DB.")
    parser.add_argument("--include-etf", action="store_true", help="Explicitly allow standard ETFs to stay active.")
    parser.add_argument("--refresh", action="store_true", help="Refresh web-based universe sources.")
    return parser.parse_args()


def _normalize_symbol(value: str | None) -> str:
    return str(value or "").strip().upper()


def _fetch_html(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_wikitable(url: str, *, expected_headers: list[str]) -> list[dict[str, str]]:
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.select("table.wikitable"):
        header_cells = [cell.get_text(" ", strip=True) for cell in table.select("tr th")]
        if all(any(expected == header for header in header_cells) for expected in expected_headers):
            rows: list[dict[str, str]] = []
            trs = table.select("tr")
            if not trs:
                continue
            headers = [cell.get_text(" ", strip=True) for cell in trs[0].find_all(["th", "td"])]
            for tr in trs[1:]:
                cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
                if len(cells) < len(headers):
                    continue
                rows.append({headers[idx]: cells[idx] for idx in range(len(headers))})
            return rows
    raise ValueError(f"Could not find expected table at {url}")


def load_sp500_seed_rows() -> list[UniverseSeedRow]:
    rows: list[UniverseSeedRow] = []
    for item in _parse_wikitable(
        SP500_WIKI_URL,
        expected_headers=["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"],
    ):
        symbol = _normalize_symbol(item.get("Symbol"))
        if not symbol:
            continue
        rows.append(
            UniverseSeedRow(
                symbol=symbol,
                company_name=str(item.get("Security") or "").strip() or None,
                sector=str(item.get("GICS Sector") or "").strip() or None,
                industry=str(item.get("GICS Sub-Industry") or "").strip() or None,
                universe_group="SP500",
                is_etf=False,
                exchange="NYSE/NASDAQ",
            )
        )
    return rows


def load_nasdaq100_seed_rows() -> list[UniverseSeedRow]:
    local_csv = Path("data/us/nasdaq100_universe.csv")
    if local_csv.exists():
        frame = pd.read_csv(local_csv)
        rows: list[UniverseSeedRow] = []
        for item in frame.to_dict(orient="records"):
            symbol = _normalize_symbol(item.get("ticker"))
            if not symbol:
                continue
            rows.append(
                UniverseSeedRow(
                    symbol=symbol,
                    company_name=str(item.get("name") or "").strip() or None,
                    sector=str(item.get("sector") or "").strip() or None,
                    industry=str(item.get("industry") or "").strip() or None,
                    universe_group="NASDAQ100",
                    is_etf=False,
                    exchange="NASDAQ",
                )
            )
        return rows

    rows = []
    for item in _parse_wikitable(
        NASDAQ100_WIKI_URL,
        expected_headers=["Ticker", "Company"],
    ):
        symbol = _normalize_symbol(item.get("Ticker"))
        if not symbol:
            continue
        rows.append(
            UniverseSeedRow(
                symbol=symbol,
                company_name=str(item.get("Company") or "").strip() or None,
                sector=str(item.get("GICS Sector") or "").strip() or None,
                industry=str(item.get("GICS Sub-Industry") or "").strip() or None,
                universe_group="NASDAQ100",
                is_etf=False,
                exchange="NASDAQ",
            )
        )
    return rows


def load_manual_etf_seed_rows() -> list[UniverseSeedRow]:
    return [
        UniverseSeedRow(
            symbol=symbol,
            company_name=symbol,
            sector="ETF",
            industry="ETF",
            universe_group="ETF",
            is_etf=True,
            exchange="NYSE/NASDAQ",
        )
        for symbol in MANUAL_ETFS + list(LEVERAGED_OR_INVERSE.keys())
    ]


def _merge_seed_rows(rows: list[UniverseSeedRow], *, include_etf: bool, check_date: date) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in rows:
        current = merged.setdefault(
            row.symbol,
            {
                "symbol": row.symbol,
                "company_name": row.company_name,
                "market": "US",
                "sector": row.sector,
                "industry": row.industry,
                "universe_group": set(),
                "is_active": True,
                "is_etf": row.is_etf,
                "is_leveraged": False,
                "is_inverse": False,
                "source": set(),
                "market_cap": None,
                "avg_volume": None,
                "currency": row.currency,
                "country": row.country,
                "exchange": row.exchange,
                "first_included_date": check_date,
                "last_checked_date": check_date,
                "exclude_reason": None,
                "feature_quality_score": None,
            },
        )
        current["universe_group"].add(row.universe_group)
        current["source"].add(row.universe_group.lower())
        if not current.get("company_name") and row.company_name:
            current["company_name"] = row.company_name
        if not current.get("sector") and row.sector:
            current["sector"] = row.sector
        if not current.get("industry") and row.industry:
            current["industry"] = row.industry
        current["is_etf"] = bool(current["is_etf"] or row.is_etf)

    for symbol, row in merged.items():
        if symbol in LEVERAGED_OR_INVERSE:
            kind, reason = LEVERAGED_OR_INVERSE[symbol]
            row["is_active"] = False
            row["is_etf"] = True
            row["is_leveraged"] = kind == "leveraged"
            row["is_inverse"] = kind == "inverse"
            row["exclude_reason"] = reason
        elif row["is_etf"] and not include_etf:
            row["is_active"] = False
            row["exclude_reason"] = "ETF excluded by init option."

        row["universe_group"] = ",".join(sorted(row["universe_group"]))
        row["source"] = ",".join(sorted(row["source"]))

    return list(merged.values())


def init_us_stock_universe(*, dry_run: bool, include_etf: bool, refresh: bool) -> InitUniverseResult:
    del refresh  # reserved for future source refresh behavior
    sp500_rows = load_sp500_seed_rows()
    nasdaq100_rows = load_nasdaq100_seed_rows()
    etf_rows = load_manual_etf_seed_rows()
    combined = sp500_rows + nasdaq100_rows + etf_rows
    payload = _merge_seed_rows(combined, include_etf=include_etf, check_date=date.today())

    active_count = sum(1 for row in payload if row["is_active"])
    inactive_count = sum(1 for row in payload if not row["is_active"])
    excluded_count = sum(1 for row in payload if row["is_leveraged"] or row["is_inverse"])
    upserted_count = 0 if dry_run else upsert_meta_us_universe_rows(payload)

    LOGGER.info("[US_META_UNIVERSE] total_input=%s", len(combined))
    LOGGER.info("[US_META_UNIVERSE] deduped=%s", len(payload))
    LOGGER.info("[US_META_UNIVERSE] sp500=%s nasdaq100=%s etf=%s", len(sp500_rows), len(nasdaq100_rows), len(etf_rows))
    LOGGER.info("[US_META_UNIVERSE] active=%s inactive=%s excluded=%s", active_count, inactive_count, excluded_count)
    LOGGER.info("[US_META_UNIVERSE] upserted=%s dry_run=%s", upserted_count, str(dry_run).lower())

    return InitUniverseResult(
        total_input_count=len(combined),
        deduped_count=len(payload),
        sp500_count=len(sp500_rows),
        nasdaq100_count=len(nasdaq100_rows),
        etf_count=len(etf_rows),
        active_count=active_count,
        inactive_count=inactive_count,
        excluded_count=excluded_count,
        upserted_count=upserted_count,
        error_count=0,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    init_us_stock_universe(
        dry_run=bool(args.dry_run),
        include_etf=bool(args.include_etf or load_us_universe_filter_config().include_etf),
        refresh=bool(args.refresh),
    )
    cfg = load_us_universe_filter_config()
    if not args.dry_run:
        rows = get_active_us_stock_universe(
            min_market_cap=cfg.min_market_cap,
            min_avg_volume=cfg.min_avg_volume,
            min_feature_quality_score=cfg.min_feature_quality_score,
            include_etf=cfg.include_etf,
            exclude_leveraged=cfg.exclude_leveraged,
            exclude_inverse=cfg.exclude_inverse,
        )
        LOGGER.info("[US_META_UNIVERSE] filtered_active_candidates=%s", len(rows))


if __name__ == "__main__":
    main()
