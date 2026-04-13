from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine, replace_table_rows_pg


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_JSON = OUTPUT_DIR / "csv_db_parity_report.json"
REPORT_MD = OUTPUT_DIR / "csv_db_parity_report.md"

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
        "name": "features",
        "csv_path": DATA_DIR / "features.csv",
        "table": "features",
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
    {"name": "features", "csv_path": DATA_DIR / "features.csv", "table": "features", "date_col": "date"},
    {"name": "predictions", "csv_path": DATA_DIR / "predictions.csv", "table": "predictions", "date_col": "date"},
    {"name": "daily_ranking", "csv_path": DATA_DIR / "ranking_final.csv", "table": "daily_ranking", "date_col": "date"},
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


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


def upsert_stocks(df: pd.DataFrame) -> None:
    engine = get_engine()
    out = df.copy()
    for col in ["listed_at", "delisted_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype(object)
            out.loc[out[col].isna(), col] = None
    records = out.astype(object).where(pd.notna(out), None).to_dict(orient="records")
    stmt = text(
        """
        INSERT INTO stocks (code, name, market, sector, listed_at, delisted_at)
        VALUES (:code, :name, :market, :sector, :listed_at, :delisted_at)
        ON CONFLICT (code) DO UPDATE SET
            name = EXCLUDED.name,
            market = EXCLUDED.market,
            sector = EXCLUDED.sector,
            listed_at = EXCLUDED.listed_at,
            delisted_at = EXCLUDED.delisted_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, records)


def sync_table(spec: dict[str, object]) -> dict[str, object]:
    df = normalize_dates(read_csv(spec["csv_path"]), spec.get("date_col"))
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
    columns = list(df.columns) if configured_columns is None else [col for col in configured_columns if col in df.columns]
    out = df.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.zfill(6)
    if configured_columns is not None:
        for col in columns:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[columns]
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
    df = normalize_dates(read_csv(spec["csv_path"]), spec.get("date_col"))
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


def main() -> int:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in SYNC_TABLES:
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
    for spec in VERIFY_TABLES:
        rows.append(verify_table(spec))
    REPORT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(build_markdown(rows), encoding="utf-8")
    mismatch_count = sum(1 for row in rows if row["status"] != "ok")
    logging.info("csv_db_parity_report saved mismatch_count=%d", mismatch_count)
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
