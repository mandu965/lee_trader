import argparse
import logging
import sys
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine

try:
    from pykrx import stock as pykrx_stock
except Exception:
    pykrx_stock = None


LOGGER = logging.getLogger("download_etf_holdings")
SOURCE_NAME = "pykrx_pdf"
STOCK_NAME_CACHE: dict[str, str] = {}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ETF holdings snapshots using pykrx PDF data.")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Snapshot date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def parse_as_of_date(raw_value: str) -> tuple[str, str]:
    as_of_date = datetime.strptime(raw_value, "%Y-%m-%d").date()
    return as_of_date.isoformat(), as_of_date.strftime("%Y%m%d")


def get_table(name: str) -> Table:
    metadata = MetaData()
    return Table(name, metadata, autoload_with=get_engine())


def load_active_etfs() -> list[dict[str, Any]]:
    query = text(
        """
        SELECT etf_code, etf_name
        FROM etf_master
        WHERE is_active = true
        ORDER BY etf_code
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    text_value = str(value).replace(",", "").strip()
    if not text_value or text_value == "-":
        return None
    try:
        return float(text_value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def get_stock_name(stock_code: str) -> str:
    cached = STOCK_NAME_CACHE.get(stock_code)
    if cached is not None:
        return cached
    if pykrx_stock is None:
        return ""
    try:
        stock_name = str(pykrx_stock.get_market_ticker_name(stock_code) or "").strip()
    except Exception:
        stock_name = ""
    STOCK_NAME_CACHE[stock_code] = stock_name
    return stock_name


def fetch_pdf_snapshot(krx_date: str, etf_code: str) -> list[dict[str, Any]]:
    if pykrx_stock is None:
        raise RuntimeError("pykrx is not installed or failed to import")

    df = pykrx_stock.get_etf_portfolio_deposit_file(etf_code, krx_date)
    if df is None or getattr(df, "empty", True):
        return []

    df = df.reset_index()
    required_cols = {"티커", "비중"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise RuntimeError(f"PDF response missing columns: {sorted(missing_cols)}")

    rows: list[dict[str, Any]] = []
    for rank, record in enumerate(df.to_dict(orient="records"), start=1):
        stock_code = str(record.get("티커") or "").strip().zfill(6)
        stock_name = get_stock_name(stock_code)
        if not stock_code:
            continue
        rows.append(
            {
                "etf_code": etf_code,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "holding_weight": _to_float(record.get("비중")),
                "holding_quantity": _to_float(record.get("계약수")),
                "market_value": _to_int(record.get("금액")),
                "rank_in_etf": rank,
                "raw_payload_json": None,
            }
        )
    return rows


def upsert_stock_stubs(stock_rows: list[dict[str, str]]) -> int:
    if not stock_rows:
        return 0

    deduped: dict[str, dict[str, str]] = {}
    for row in stock_rows:
        deduped[row["code"]] = row

    stocks = get_table("stocks")
    stmt = insert(stocks).values(list(deduped.values()))
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["code"],
        set_={
            "name": text("COALESCE(NULLIF(EXCLUDED.name, ''), stocks.name)"),
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def upsert_holdings_rows(holdings_rows: list[dict[str, Any]], as_of_date: str) -> int:
    if not holdings_rows:
        return 0

    table = get_table("etf_holdings_snapshot")
    payload = [
        {
            "as_of_date": as_of_date,
            "etf_code": row["etf_code"],
            "stock_code": row["stock_code"],
            "stock_name": row["stock_name"],
            "holding_weight": row["holding_weight"],
            "holding_quantity": row["holding_quantity"],
            "market_value": row["market_value"],
            "rank_in_etf": row["rank_in_etf"],
            "source_name": SOURCE_NAME,
            "raw_payload_json": row["raw_payload_json"],
        }
        for row in holdings_rows
    ]

    stmt = insert(table).values(payload)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["as_of_date", "etf_code", "stock_code"],
        set_={
            "stock_name": stmt.excluded.stock_name,
            "holding_weight": stmt.excluded.holding_weight,
            "holding_quantity": stmt.excluded.holding_quantity,
            "market_value": stmt.excluded.market_value,
            "rank_in_etf": stmt.excluded.rank_in_etf,
            "source_name": stmt.excluded.source_name,
            "raw_payload_json": stmt.excluded.raw_payload_json,
            "collected_at": text("now()"),
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def print_summary(
    *,
    as_of_date: str,
    etf_total: int,
    etf_success: int,
    etf_failed: int,
    holdings_rows: int,
    stock_rows: int,
) -> None:
    print(
        "ETF holdings load completed "
        f"as_of_date={as_of_date} etf_total={etf_total} etf_success={etf_success} "
        f"etf_failed={etf_failed} stock_rows={stock_rows} holdings_rows={holdings_rows}"
    )


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        as_of_date, krx_date = parse_as_of_date(args.as_of_date)
        LOGGER.info("Starting ETF holdings download as_of_date=%s krx_date=%s", as_of_date, krx_date)

        active_etfs = load_active_etfs()
        LOGGER.info("Loaded %s active ETFs from etf_master", len(active_etfs))
        if not active_etfs:
            print_summary(
                as_of_date=as_of_date,
                etf_total=0,
                etf_success=0,
                etf_failed=0,
                holdings_rows=0,
                stock_rows=0,
            )
            return 0

        all_holdings_rows: list[dict[str, Any]] = []
        success_count = 0
        failure_count = 0

        for etf in active_etfs:
            etf_code = str(etf["etf_code"])
            etf_name = str(etf.get("etf_name") or "")
            try:
                rows = fetch_pdf_snapshot(krx_date, etf_code)
                if not rows:
                    failure_count += 1
                    LOGGER.warning("Empty PDF holdings etf_code=%s etf_name=%s", etf_code, etf_name)
                    continue
                all_holdings_rows.extend(rows)
                success_count += 1
                LOGGER.info(
                    "Fetched ETF holdings etf_code=%s etf_name=%s holdings=%s",
                    etf_code,
                    etf_name,
                    len(rows),
                )
            except Exception as exc:
                failure_count += 1
                LOGGER.warning("ETF holdings fetch failed etf_code=%s etf_name=%s error=%s", etf_code, etf_name, exc)
                continue

        stock_rows = [
            {"code": row["stock_code"], "name": row["stock_name"]}
            for row in all_holdings_rows
        ]
        stock_upserted = upsert_stock_stubs(stock_rows)
        holdings_upserted = upsert_holdings_rows(all_holdings_rows, as_of_date)

        LOGGER.info(
            "ETF holdings download finished etf_total=%s success=%s failed=%s stock_rows=%s holdings_rows=%s",
            len(active_etfs),
            success_count,
            failure_count,
            stock_upserted,
            holdings_upserted,
        )
        print_summary(
            as_of_date=as_of_date,
            etf_total=len(active_etfs),
            etf_success=success_count,
            etf_failed=failure_count,
            holdings_rows=holdings_upserted,
            stock_rows=stock_upserted,
        )
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while loading ETF holdings: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("ETF holdings load failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
