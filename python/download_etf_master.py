import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine

try:
    from pykrx import stock as pykrx_stock
except Exception:
    pykrx_stock = None


LOGGER = logging.getLogger("download_etf_master")
DEFAULT_LOOKBACK_DAYS = 14
DATA_DIR = Path("data")
PRICES_RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
ISSUER_PREFIX_MAP = {
    "KODEX": "Samsung Asset Management",
    "TIGER": "Mirae Asset Global Investments",
    "KOSEF": "Kiwoom Asset Management",
    "KBSTAR": "KB Asset Management",
    "KINDEX": "Korea Investment Management",
    "HANARO": "NH-Amundi Asset Management",
    "ARIRANG": "Hanwha Asset Management",
    "ACE": "Korea Investment Management",
    "SOL": "Shinhan Asset Management",
    "TIMEFOLIO": "TIMEFOLIO Asset Management",
    "TREX": "Yuanta Asset Management",
    "FOCUS": "Eugene Asset Management",
    "WOORI": "Woori Asset Management",
    "1Q": "Hana Asset Management",
    "PLUS": "Hanwha Asset Management",
    "RISE": "KB Asset Management",
    "TRUSTON": "Truston Asset Management",
    "BNK": "BNK Asset Management",
}
ETF_TYPE_KEYWORDS = {
    "active": ["ACTIVE", "\uc561\ud2f0\ube0c"],
    "leveraged": ["LEVERAGE", "\ub808\ubc84\ub9ac\uc9c0"],
    "inverse": ["INVERSE", "\uc778\ubc84\uc2a4"],
    "bond": ["BOND", "TREASURY", "\ucc44\uad8c", "\uad6d\ucc44", "\ud68c\uc0ac\ucc44", "\ud1b5\uc548"],
    "commodity": ["OIL", "GOLD", "SILVER", "COPPER", "\uc6d0\uc720", "\uae08", "\uc740", "\uad6c\ub9ac", "\uc6d0\uc790\uc7ac"],
    "currency": ["USD", "JPY", "EUR", "CURRENCY", "FX", "\ub2ec\ub7ec", "\ud658\uc728", "\ud1b5\ud654", "\uc5d4\uc120\ubb3c", "\uc720\ub85c"],
    "equity_global": [
        "NASDAQ", "NYSE", "S&P", "MSCI", "CHINA", "JAPAN", "EUROPE", "GLOBAL", "USA",
        "\ubbf8\uad6d", "\uc77c\ubcf8", "\uc911\uad6d", "\uc720\ub7fd", "\uae00\ub85c\ubc8c", "\uc120\uc9c4\uad6d", "\uc2e0\ud765\uad6d", "\ud574\uc678",
    ],
    "equity_thematic": [
        "AI", "ROBOT", "GAME", "BIO", "SEMICONDUCTOR", "BATTERY", "TECH", "INTERNET", "POWER",
        "\ubc18\ub3c4\uccb4", "2\ucc28\uc804\uc9c0", "\ub85c\ubd07", "\ubc14\uc774\uc624", "\uc804\ub825", "\ud14c\ud06c", "\uc778\ud130\ub137", "\uac8c\uc784",
    ],
    "equity_income": ["DIVIDEND", "COVEREDCALL", "PREFERRED", "\ubc30\ub2f9", "\uace0\ubc30\ub2f9", "\ucee4\ubc84\ub4dc\ucf5c", "\uc6b0\uc120\uc8fc"],
    "equity_index": ["KOSPI", "KOSDAQ", "200", "150", "300", "100", "\ucf54\uc2a4\ud53c", "\ucf54\uc2a4\ub2e5"],
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.raiseExceptions = False
    for logger_name in [
        "pykrx",
        "pykrx.website",
        "pykrx.website.comm",
        "urllib3",
        "requests",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def _recent_business_dates(lookback_days: int) -> list[str]:
    today = date.today()
    candidates: list[str] = []
    for offset in range(lookback_days + 1):
        current = today - timedelta(days=offset)
        if current.weekday() < 5:
            candidates.append(current.strftime("%Y%m%d"))
    return candidates


def _extract_tickers_from_price_change(df: Any) -> list[str]:
    if df is None or getattr(df, "empty", True):
        return []
    tickers = [str(idx).strip().zfill(6) for idx in list(df.index)]
    return [ticker for ticker in tickers if ticker.isdigit()]


def _call_pykrx_quietly(func, *args, **kwargs):
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return func(*args, **kwargs)
    finally:
        logging.disable(previous_disable)


def get_etf_tickers() -> list[str]:
    if pykrx_stock is None:
        raise RuntimeError("pykrx is not installed or failed to import")

    errors: list[str] = []
    for business_date in _recent_business_dates(DEFAULT_LOOKBACK_DAYS):
        try:
            tickers = _call_pykrx_quietly(pykrx_stock.get_etf_ticker_list, business_date)
            tickers = [str(ticker).strip().zfill(6) for ticker in tickers if str(ticker).strip()]
            if tickers:
                LOGGER.info("Loaded %s ETF tickers from pykrx date=%s", len(tickers), business_date)
                return tickers
        except Exception as exc:
            errors.append(f"{business_date}: ticker_list failed: {exc}")

        try:
            df = _call_pykrx_quietly(pykrx_stock.get_etf_price_change_by_ticker, business_date, business_date)
            tickers = _extract_tickers_from_price_change(df)
            if tickers:
                LOGGER.info("Loaded %s ETF tickers from pykrx price_change date=%s", len(tickers), business_date)
                return tickers
        except Exception as exc:
            errors.append(f"{business_date}: price_change failed: {exc}")

    joined = " | ".join(errors[-6:]) if errors else "no attempts"
    raise RuntimeError(f"Unable to load ETF ticker list from pykrx. recent_errors={joined}")


def infer_issuer_name(etf_name: str) -> str | None:
    upper_name = (etf_name or "").upper()
    for prefix, issuer_name in ISSUER_PREFIX_MAP.items():
        if upper_name.startswith(prefix):
            return issuer_name
    return None


def infer_etf_type(etf_name: str) -> str:
    raw_name = etf_name or ""
    upper_name = raw_name.upper()
    for etf_type, keywords in ETF_TYPE_KEYWORDS.items():
        if any(keyword in raw_name or keyword in upper_name for keyword in keywords):
            return etf_type
    return "equity"


def normalize_etf_name(raw_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (raw_name or "").strip())
    if not cleaned:
        raise ValueError("empty ETF name")
    return cleaned


def _read_theme_etf_master_name_map() -> dict[str, str]:
    if not THEME_ETF_MASTER_CSV.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(THEME_ETF_MASTER_CSV)
    except Exception as exc:
        LOGGER.warning("Failed to read theme_etf_master for fallback names: %s", exc)
        return {}
    if df.empty or "etf_code" not in df.columns or "etf_name" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for row in df.itertuples(index=False):
        code = str(getattr(row, "etf_code", "")).strip().zfill(6)
        name = str(getattr(row, "etf_name", "")).strip()
        if code.isdigit() and name:
            out[code] = name
    return out


def _read_existing_etf_master_name_map() -> dict[str, str]:
    try:
        with get_engine().begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT etf_code, etf_name
                    FROM etf_master
                    WHERE etf_code IS NOT NULL
                    """
                )
            ).mappings().all()
    except Exception as exc:
        LOGGER.warning("Failed to read existing etf_master for fallback names: %s", exc)
        return {}

    out: dict[str, str] = {}
    for row in rows:
        code = str(row.get("etf_code") or "").strip().zfill(6)
        name = str(row.get("etf_name") or "").strip()
        if code.isdigit() and name:
            out[code] = name
    return out


def load_local_etf_rows_from_prices() -> list[dict[str, Any]]:
    if not PRICES_RAW_CSV.exists():
        LOGGER.warning("Local ETF fallback skipped: prices_raw not found path=%s", PRICES_RAW_CSV)
        return []
    try:
        import pandas as pd

        df = pd.read_csv(PRICES_RAW_CSV, usecols=["code", "asset_type"], low_memory=False)
    except Exception as exc:
        LOGGER.warning("Failed to read prices_raw for ETF fallback: %s", exc)
        return []

    if df.empty or "asset_type" not in df.columns or "code" not in df.columns:
        LOGGER.warning("prices_raw missing required columns for ETF fallback")
        return []

    etf_codes = (
        df.loc[df["asset_type"].astype(str).str.lower() == "etf", "code"]
        .astype(str)
        .str.strip()
        .str.zfill(6)
    )
    etf_codes = sorted({code for code in etf_codes if code.isdigit()})
    if not etf_codes:
        LOGGER.warning("Local ETF fallback found zero ETF codes in prices_raw")
        return []

    name_map = _read_existing_etf_master_name_map()
    name_map.update(_read_theme_etf_master_name_map())
    rows: list[dict[str, Any]] = []
    missing_name_codes: list[str] = []

    for code in etf_codes:
        etf_name = name_map.get(code)
        if not etf_name:
            missing_name_codes.append(code)
            etf_name = f"ETF_{code}"
        rows.append(
            {
                "etf_code": code,
                "etf_name": normalize_etf_name(etf_name),
                "issuer_name": infer_issuer_name(etf_name),
                "asset_class": infer_etf_type(etf_name),
                "market": "ETF",
                "is_active": True,
            }
        )

    LOGGER.info(
        "Local ETF fallback rows=%s missing_names=%s sample_missing=%s",
        len(rows),
        len(missing_name_codes),
        missing_name_codes[:5],
    )
    return rows


def ensure_etf_master_table() -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS etf_master (
                    etf_code TEXT PRIMARY KEY,
                    etf_name TEXT NOT NULL,
                    issuer_name TEXT NULL,
                    asset_class TEXT NULL,
                    market TEXT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_etf_master_is_active
                ON etf_master (is_active)
                """
            )
        )


def collect_etf_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    try:
        tickers = get_etf_tickers()
        source = "pykrx"
    except Exception as exc:
        LOGGER.warning("Primary ETF ticker load failed -> fallback to local ETF rows: %s", exc)
        rows = load_local_etf_rows_from_prices()
        return rows, [], "local_prices_fallback"

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for ticker in tickers:
        try:
            etf_name = normalize_etf_name(pykrx_stock.get_etf_ticker_name(ticker))
            rows.append(
                {
                    "etf_code": ticker,
                    "etf_name": etf_name,
                    "issuer_name": infer_issuer_name(etf_name),
                    "asset_class": infer_etf_type(etf_name),
                    "market": "ETF",
                    "is_active": True,
                }
            )
        except Exception as exc:
            failures.append({"etf_code": ticker, "error": str(exc)})
            LOGGER.warning("ETF metadata fetch failed etf_code=%s error=%s", ticker, exc)

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["etf_code"]] = row
    return list(deduped.values()), failures, source


def get_etf_master_table() -> Table:
    ensure_etf_master_table()
    metadata = MetaData()
    return Table("etf_master", metadata, autoload_with=get_engine())


def upsert_etf_master(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    etf_master = get_etf_master_table()
    stmt = insert(etf_master).values(rows)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["etf_code"],
        set_={
            "etf_name": stmt.excluded.etf_name,
            "issuer_name": stmt.excluded.issuer_name,
            "asset_class": stmt.excluded.asset_class,
            "market": stmt.excluded.market,
            "is_active": stmt.excluded.is_active,
            "updated_at": text("now()"),
        },
    )

    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def deactivate_missing_etfs(active_codes: list[str]) -> int:
    with get_engine().begin() as conn:
        if active_codes:
            result = conn.execute(
                text(
                    """
                    UPDATE etf_master
                    SET is_active = false,
                        updated_at = now()
                    WHERE is_active = true
                      AND etf_code NOT IN :active_codes
                    """
                ).bindparams(bindparam("active_codes", expanding=True)),
                {"active_codes": active_codes},
            )
        else:
            result = conn.execute(
                text(
                    """
                    UPDATE etf_master
                    SET is_active = false,
                        updated_at = now()
                    WHERE is_active = true
                    """
                )
            )
    return int(result.rowcount or 0)


def print_summary(*, fetched_count: int, upserted_count: int, deactivated_count: int, failure_count: int) -> None:
    print(
        "ETF master load completed "
        f"fetched={fetched_count} upserted={upserted_count} "
        f"deactivated={deactivated_count} failures={failure_count}"
    )


def main() -> int:
    setup_logging()
    LOGGER.info("Starting ETF master download")

    try:
        rows, failures, source = collect_etf_rows()
        if not rows:
            LOGGER.error("No ETF rows collected")
            print_summary(fetched_count=0, upserted_count=0, deactivated_count=0, failure_count=len(failures))
            return 1

        upserted_count = upsert_etf_master(rows)
        if source == "pykrx":
            deactivated_count = deactivate_missing_etfs([row["etf_code"] for row in rows])
        else:
            deactivated_count = 0
            LOGGER.info("Skip deactivate_missing_etfs because source=%s is partial fallback", source)

        LOGGER.info(
            "ETF master download finished source=%s fetched=%s upserted=%s deactivated=%s failures=%s",
            source,
            len(rows),
            upserted_count,
            deactivated_count,
            len(failures),
        )
        print_summary(
            fetched_count=len(rows),
            upserted_count=upserted_count,
            deactivated_count=deactivated_count,
            failure_count=len(failures),
        )
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while loading ETF master: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("ETF master load failed: %s", exc)
        return 1


if __name__ == "__main__":
<<<<<<< HEAD
    sys.exit(main())
=======
    sys.exit(main())
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
