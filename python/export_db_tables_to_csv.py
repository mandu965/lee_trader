"""
Export Postgres tables to per-table CSV files for manual Supabase import.

Default behavior:
- reads DATABASE_URL from environment or repo .env via python/db.py
- exports every BASE TABLE in public and research schemas
- writes UTF-8 BOM CSV files so Excel/Supabase column headers display cleanly
- stores outputs under exports/db_csv/<schema>/<table>.csv
- writes a manifest.json with row counts and file paths
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine


DEFAULT_SCHEMAS = ("public", "research")
DEFAULT_OUTPUT_DIR = Path("exports/db_csv")


@dataclass(frozen=True)
class TableRef:
    schema: str
    name: str

    @property
    def fq_name(self) -> str:
        return f"{self.schema}.{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Postgres tables to CSV for manual Supabase import."
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=list(DEFAULT_SCHEMAS),
        help="Schemas to export. Default: public research",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=[],
        help="Optional explicit tables in schema.table format.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also emit CSV files for empty tables. Default keeps them too; this flag is for clarity in logs.",
    )
    parser.add_argument(
        "--exclude-tables",
        nargs="*",
        default=[],
        help="Optional schema.table entries to skip.",
    )
    return parser.parse_args()


def parse_table_ref(raw: str) -> TableRef:
    parts = raw.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"expected schema.table, got: {raw}")
    return TableRef(schema=parts[0], name=parts[1])


def list_base_tables(schemas: list[str]) -> list[TableRef]:
    eng = get_engine()
    sql = text(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema = ANY(:schemas)
        ORDER BY table_schema, table_name
        """
    )
    with eng.connect() as conn:
        rows = conn.execute(sql, {"schemas": schemas}).fetchall()
    return [TableRef(schema=row.table_schema, name=row.table_name) for row in rows]


def select_export_tables(
    schemas: list[str], explicit_tables: list[str], excluded_tables: list[str]
) -> list[TableRef]:
    exclude_set = {parse_table_ref(item).fq_name for item in excluded_tables}
    if explicit_tables:
        refs = [parse_table_ref(item) for item in explicit_tables]
        return [ref for ref in refs if ref.fq_name not in exclude_set]
    refs = list_base_tables(schemas)
    return [ref for ref in refs if ref.fq_name not in exclude_set]


def export_table(table_ref: TableRef, output_dir: Path) -> dict[str, object]:
    eng = get_engine()
    schema_dir = output_dir / table_ref.schema
    schema_dir.mkdir(parents=True, exist_ok=True)
    out_path = schema_dir / f"{table_ref.name}.csv"
    sql = text(f'SELECT * FROM "{table_ref.schema}"."{table_ref.name}"')
    df = pd.read_sql_query(sql, eng)
    df = normalize_dataframe_for_csv(eng, table_ref, df)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "schema": table_ref.schema,
        "table": table_ref.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "output": out_path.as_posix(),
    }


def get_column_types(table_ref: TableRef) -> dict[str, str]:
    eng = get_engine()
    sql = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with eng.connect() as conn:
        rows = conn.execute(
            sql,
            {"schema": table_ref.schema, "table": table_ref.name},
        ).fetchall()
    return {row.column_name: row.data_type for row in rows}


def normalize_dataframe_for_csv(eng, table_ref: TableRef, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    column_types = get_column_types(table_ref)
    integer_types = {"smallint", "integer", "bigint"}
    json_types = {"json", "jsonb"}

    for column, data_type in column_types.items():
        if column not in out.columns:
            continue
        if data_type in integer_types:
            series = pd.to_numeric(out[column], errors="coerce")
            if series.notna().any():
                out[column] = series.round().astype("Int64")
            else:
                out[column] = pd.Series([pd.NA] * len(out), dtype="Int64")
        elif data_type in json_types:
            out[column] = out[column].map(_json_cell_to_text)

    return out


def _json_cell_to_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return None
        try:
            return json.dumps(json.loads(text_value), ensure_ascii=False)
        except Exception:
            return text_value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return json.dumps(value, ensure_ascii=False)


def write_manifest(output_dir: Path, items: list[dict[str, object]]) -> Path:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "table_count": len(items),
        "tables": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = select_export_tables(args.schemas, args.tables, args.exclude_tables)
    if not tables:
        raise SystemExit("no tables selected for export")

    manifest_items: list[dict[str, object]] = []
    print(f"[START] export tables -> {output_dir}")
    for table_ref in tables:
        item = export_table(table_ref, output_dir)
        manifest_items.append(item)
        print(
            f"[OK] {table_ref.fq_name} rows={item['rows']} file={item['output']}"
        )

    manifest_path = write_manifest(output_dir, manifest_items)
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
