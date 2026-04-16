from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine, replace_table_rows_pg


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_JSON = OUTPUT_DIR / "csv_db_parity_report.json"
REPORT_MD = OUTPUT_DIR / "csv_db_parity_report.md"
PRICES_CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"
PRICES_ADJUSTED_CSV = DATA_DIR / "prices_daily_adjusted.csv"

SYNC_TABLES = [
    {
        "name": "stocks",
        "csv_path": DATA_DIR / "universe.csv",
        "table": "stocks",
        "date_col": None,
        "key_cols": ["code"],
        "columns": ["code", "name", "market", "sector", "listed_at", "delisted_at"],
    },
    {
        "name": "market_status",
        "csv_path": DATA_DIR / "market_status.csv",
        "table": "market_status",
        "date_col": "date",
        "key_cols": ["date"],
        "columns": ["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d", "market_up"],
    },
    {
        "name": "fundamentals",
        "csv_path": DATA_DIR / "fundamentals.csv",
        "table": "fundamentals",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": ["date", "code", "roe", "op_margin", "debt_ratio", "ocf_to_assets", "net_margin"],
    },
    {
        "name": "quality",
        "csv_path": DATA_DIR / "quality.csv",
        "table": "quality",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": None,
    },
    {
        "name": "prices_adjusted",
        "csv_path": PRICES_ADJUSTED_CSV,
        "table": "prices_adjusted",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": ["date", "code", "adj_open", "adj_high", "adj_low", "adj_close", "volume"],
    },
    {
        "name": "fact_price_daily",
        "csv_path": PRICES_CLEAN_CSV,
        "table": "fact_price_daily",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": ["date", "code", "open", "high", "low", "close", "adj_close", "volume", "value", "market_cap", "listed_shares"],
    },
    {
        "name": "features",
        "csv_path": DATA_DIR / "features.csv",
        "table": "features",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": None,
    },
    {
        "name": "labels",
        "csv_path": DATA_DIR / "labels.csv",
        "table": "labels",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": None,
    },
    {
        "name": "predictions",
        "csv_path": DATA_DIR / "predictions.csv",
        "table": "predictions",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": None,
    },
    {
        "name": "daily_ranking",
        "csv_path": DATA_DIR / "ranking_final.csv",
        "table": "daily_ranking",
        "date_col": "date",
        "key_cols": ["date", "code"],
        "columns": None,
    },
]

VERIFY_TABLES = [
    {"name": "stocks", "csv_path": DATA_DIR / "universe.csv", "table": "stocks", "date_col": None},
    {"name": "market_status", "csv_path": DATA_DIR / "market_status.csv", "table": "market_status", "date_col": "date"},
    {"name": "fundamentals", "csv_path": DATA_DIR / "fundamentals.csv", "table": "fundamentals", "date_col": "date"},
    {"name": "quality", "csv_path": DATA_DIR / "quality.csv", "table": "quality", "date_col": "date"},
    {"name": "prices_adjusted", "csv_path": PRICES_ADJUSTED_CSV, "table": "prices_adjusted", "date_col": "date"},
    {"name": "fact_price_daily", "csv_path": PRICES_CLEAN_CSV, "table": "fact_price_daily", "date_col": "date"},
    {"name": "features", "csv_path": DATA_DIR / "features.csv", "table": "features", "date_col": "date"},
    {"name": "labels", "csv_path": DATA_DIR / "labels.csv", "table": "labels", "date_col": "date"},
    {"name": "predictions", "csv_path": DATA_DIR / "predictions.csv", "table": "predictions", "date_col": "date"},
    {"name": "daily_ranking", "csv_path": DATA_DIR / "ranking_final.csv", "table": "daily_ranking", "date_col": "date"},
]

VALID_TABLE_NAMES = {str(spec["name"]) for spec in SYNC_TABLES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync CSV outputs into Postgres and verify parity.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional table names to limit sync/verify scope. Example: market_status predictions daily_ranking",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def build_prices_adjusted_df() -> pd.DataFrame:
    return read_csv(PRICES_ADJUSTED_CSV)


def build_fact_price_daily_df() -> pd.DataFrame:
    clean = read_csv(PRICES_CLEAN_CSV)
    adjusted = read_csv(PRICES_ADJUSTED_CSV)
    if clean.empty or adjusted.empty:
        return pd.DataFrame()

    clean = clean.copy()
    adjusted = adjusted.copy()
    if "code" in clean.columns:
        clean["code"] = clean["code"].astype(str).str.zfill(6)
    if "code" in adjusted.columns:
        adjusted["code"] = adjusted["code"].astype(str).str.zfill(6)
    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "date" in adjusted.columns:
        adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = clean.merge(
        adjusted[["date", "code", "adj_close"]],
        on=["date", "code"],
        how="left",
    )
    merged["value"] = pd.NA
    merged["market_cap"] = pd.NA
    merged["listed_shares"] = pd.NA
    return merged


def normalize_dates(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    if not date_col or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def latest_date(df: pd.DataFrame, date_col: str | None) -> str | None:
    if not date_col or date_col not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def db_snapshot(table: str, date_col: str | None) -> dict[str, object]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
        latest = None
        if date_col:
            latest = conn.execute(text(f"SELECT to_char(max({date_col}), 'YYYY-MM-DD') FROM {table}")).scalar()
    return {"rows": int(rows or 0), "latest_date": latest}


@lru_cache(maxsize=None)
def get_table_column_types(table: str) -> dict[str, str]:
    engine = get_engine()
    query = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table_name": table}).mappings().all()
    return {str(row["column_name"]): str(row["data_type"]).lower() for row in rows}


def _coerce_bool(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    if normalized.dtype == bool:
        return normalized
    text_values = normalized.astype(str).str.strip().str.lower()
    mapped = text_values.map(
        {
            "true": True,
            "t": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "f": False,
            "0": False,
            "no": False,
            "n": False,
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "<na>": pd.NA,
        }
    )
    return mapped.astype("boolean")


def normalize_for_pg_table(df: pd.DataFrame, table: str, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    column_types = get_table_column_types(table)
    integer_types = {"smallint", "integer", "bigint"}
    numeric_types = {"real", "double precision", "numeric", "decimal"}
    date_types = {"date"}
    timestamp_types = {"timestamp without time zone", "timestamp with time zone"}

    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
            continue
        data_type = column_types.get(col)
        if data_type in integer_types:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
        elif data_type in numeric_types:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif data_type == "boolean":
            out[col] = _coerce_bool(out[col])
        elif data_type in date_types:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif data_type in timestamp_types:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    return out


def upsert_stocks(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in ["listed_at", "delisted_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    replace_table_rows_pg(
        "stocks",
        out,
        columns=["code", "name", "market", "sector", "listed_at", "delisted_at"],
    )


def sync_table(spec: dict[str, object]) -> dict[str, object]:
    if spec["name"] == "prices_adjusted":
        raw_df = build_prices_adjusted_df()
    elif spec["name"] == "fact_price_daily":
        raw_df = build_fact_price_daily_df()
    else:
        raw_df = read_csv(spec["csv_path"])

    df = normalize_dates(raw_df, spec.get("date_col"))
    if df.empty:
        return {
            "name": spec["name"],
            "table": spec["table"],
            "status": "missing_csv",
            "csv_rows": 0,
            "csv_latest_date": None,
            "db_rows": None,
            "db_latest_date": None,
        }

    configured_columns = spec.get("columns")
    table_column_types = get_table_column_types(str(spec["table"]))
    if configured_columns is None:
        columns = [col for col in df.columns if col in table_column_types]
    else:
        columns = [col for col in configured_columns if col in table_column_types]
    out = df.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[columns]
    out = normalize_for_pg_table(out, str(spec["table"]), columns)
    out = out.drop_duplicates(subset=spec["key_cols"], keep="last").reset_index(drop=True)
    if spec["table"] == "stocks":
        upsert_stocks(out)
    else:
        replace_table_rows_pg(spec["table"], out, columns=columns)
    db = db_snapshot(spec["table"], spec.get("date_col"))
    csv_latest = latest_date(out, spec.get("date_col"))
    status = "ok" if db["rows"] == len(out) and db["latest_date"] == csv_latest else "mismatch"
    return {
        "name": spec["name"],
        "table": spec["table"],
        "status": status,
        "csv_rows": int(len(out)),
        "csv_latest_date": csv_latest,
        "db_rows": db["rows"],
        "db_latest_date": db["latest_date"],
    }


def verify_table(spec: dict[str, object]) -> dict[str, object]:
    if spec["name"] == "prices_adjusted":
        raw_df = build_prices_adjusted_df()
    elif spec["name"] == "fact_price_daily":
        raw_df = build_fact_price_daily_df()
    else:
        raw_df = read_csv(spec["csv_path"])
    df = normalize_dates(raw_df, spec.get("date_col"))
    db = db_snapshot(spec["table"], spec.get("date_col"))
    csv_latest = latest_date(df, spec.get("date_col"))
    status = "ok" if db["rows"] == len(df) and db["latest_date"] == csv_latest else "mismatch"
    return {
        "name": spec["name"],
        "table": spec["table"],
        "status": status,
        "csv_rows": int(len(df)),
        "csv_latest_date": csv_latest,
        "db_rows": db["rows"],
        "db_latest_date": db["latest_date"],
    }


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# CSV DB Parity Report",
        "",
        f"- generated_at: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "| name | table | status | csv_rows | csv_latest_date | db_rows | db_latest_date |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['table']} | {row['status']} | {row['csv_rows']} | {row['csv_latest_date'] or ''} | {row['db_rows'] if row['db_rows'] is not None else ''} | {row['db_latest_date'] or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def select_specs(names: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not names:
        return SYNC_TABLES, VERIFY_TABLES
    normalized = [str(name).strip() for name in names if str(name).strip()]
    unknown = sorted(set(normalized) - VALID_TABLE_NAMES)
    if unknown:
        raise ValueError(f"unknown table names: {', '.join(unknown)}")
    selected = set(normalized)
    sync_specs = [spec for spec in SYNC_TABLES if str(spec["name"]) in selected]
    verify_specs = [spec for spec in VERIFY_TABLES if str(spec["name"]) in selected]
    return sync_specs, verify_specs


def main() -> int:
    args = parse_args()
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sync_specs, verify_specs = select_specs(args.only)
    rows: list[dict[str, object]] = []
    for spec in sync_specs:
        try:
            rows.append(sync_table(spec))
        except Exception as exc:
            logging.exception("sync failed for %s", spec["name"])
            rows.append(
                {
                    "name": spec["name"],
                    "table": spec["table"],
                    "status": "error",
                    "csv_rows": 0,
                    "csv_latest_date": None,
                    "db_rows": None,
                    "db_latest_date": None,
                    "error": str(exc),
                }
            )
    for spec in verify_specs:
        rows.append(verify_table(spec))
    REPORT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(build_markdown(rows), encoding="utf-8")
    mismatch_count = sum(1 for row in rows if row["status"] != "ok")
    logging.info("csv_db_parity_report saved mismatch_count=%d", mismatch_count)
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
