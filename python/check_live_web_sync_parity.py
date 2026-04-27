from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_web_sync_parity_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_web_sync_parity_report.md"

PAYLOAD_KEYS = [
    "trade_intents",
    "order_requests_preview",
    "order_requests_execution",
    "live_order_fills",
    "live_trade_consistency_report",
    "live_trade_review_report",
    "live_trade_review_summary",
]

LIVE_TABLES = [
    "research.live_trade_decision",
    "research.live_order_request",
    "research.live_order_execution",
    "research.live_order_fill",
    "research.live_position_snapshot",
    "research.live_trade_review",
]

TABLE_COUNT_SQL = {
    "research.live_trade_decision": """
        SELECT COUNT(*)
        FROM research.live_trade_decision
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM research.live_trade_decision)
    """,
    "research.live_order_request": """
        SELECT COUNT(*)
        FROM research.live_order_request
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM research.live_order_request)
    """,
    "research.live_order_execution": """
        SELECT COUNT(*)
        FROM research.live_order_execution
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM research.live_order_execution)
    """,
    "research.live_order_fill": "SELECT COUNT(*) FROM research.live_order_fill",
    "research.live_position_snapshot": """
        SELECT COUNT(*)
        FROM research.live_position_snapshot
        WHERE snapshot_at = (SELECT MAX(snapshot_at) FROM research.live_position_snapshot)
    """,
    "research.live_trade_review": """
        SELECT COUNT(*)
        FROM research.live_trade_review
        WHERE reviewer = 'auto_review'
    """,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local live trading display data with the web DB.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--local-database-url", default="")
    parser.add_argument("--web-database-url", default="")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _payload_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _connect(url: str):
    return create_engine(url, future=True, connect_args={"connect_timeout": int(os.environ.get("PG_CONNECT_TIMEOUT", "5"))})


def _table_exists(conn, table_name: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": table_name}).scalar())


def _payload_rows(engine) -> dict[str, dict[str, Any]]:
    with engine.connect() as conn:
        if not _table_exists(conn, "research.app_payload_store"):
            return {}
        rows = conn.execute(
            text(
                """
                SELECT payload_key, payload_json, asof_date, generated_at, updated_at
                FROM research.app_payload_store
                WHERE payload_key = ANY(:keys)
                ORDER BY payload_key
                """
            ),
            {"keys": PAYLOAD_KEYS},
        ).mappings().all()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row.get("payload_json")
        out[str(row["payload_key"])] = {
            "present": True,
            "asof_date": row.get("asof_date"),
            "generated_at": row.get("generated_at"),
            "updated_at": row.get("updated_at"),
            "payload_hash": _payload_hash(payload),
        }
    return out


def _table_counts(engine) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with engine.connect() as conn:
        for table in LIVE_TABLES:
            if not _table_exists(conn, table):
                out[table] = {"present": False, "count": 0}
                continue
            count_sql = TABLE_COUNT_SQL.get(table, f"SELECT COUNT(*) FROM {table}")
            count = int(conn.execute(text(count_sql)).scalar() or 0)
            out[table] = {"present": True, "count": count}
    return out


def _compare_payloads(local: dict[str, dict[str, Any]], web: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in PAYLOAD_KEYS:
        local_row = local.get(key, {"present": False})
        web_row = web.get(key, {"present": False})
        rows.append(
            {
                "payload_key": key,
                "local_present": bool(local_row.get("present")),
                "web_present": bool(web_row.get("present")),
                "local_asof_date": local_row.get("asof_date"),
                "web_asof_date": web_row.get("asof_date"),
                "local_generated_at": local_row.get("generated_at"),
                "web_generated_at": web_row.get("generated_at"),
                "hash_match": bool(local_row.get("payload_hash") and local_row.get("payload_hash") == web_row.get("payload_hash")),
            }
        )
    return rows


def _compare_tables(local: dict[str, dict[str, Any]], web: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in LIVE_TABLES:
        local_row = local.get(table, {"present": False, "count": 0})
        web_row = web.get(table, {"present": False, "count": 0})
        rows.append(
            {
                "table": table,
                "local_present": bool(local_row.get("present")),
                "web_present": bool(web_row.get("present")),
                "local_count": int(local_row.get("count") or 0),
                "web_count": int(web_row.get("count") or 0),
                "count_match": int(local_row.get("count") or 0) == int(web_row.get("count") or 0),
            }
        )
    return rows


def build_report(local_url: str, web_url: str) -> dict[str, Any]:
    local_engine = _connect(local_url)
    web_engine = _connect(web_url)
    local_payloads = _payload_rows(local_engine)
    web_payloads = _payload_rows(web_engine)
    local_tables = _table_counts(local_engine)
    web_tables = _table_counts(web_engine)
    payload_comparison = _compare_payloads(local_payloads, web_payloads)
    table_comparison = _compare_tables(local_tables, web_tables)

    warnings: list[str] = []
    for row in payload_comparison:
        if not row["local_present"] or not row["web_present"]:
            warnings.append(f"payload missing: {row['payload_key']}")
        elif not row["hash_match"]:
            warnings.append(f"payload differs: {row['payload_key']}")
    for row in table_comparison:
        if not row["local_present"] or not row["web_present"]:
            warnings.append(f"table missing: {row['table']}")
        elif not row["count_match"]:
            warnings.append(f"table count differs: {row['table']} local={row['local_count']} web={row['web_count']}")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "payload_comparison": payload_comparison,
        "table_comparison": table_comparison,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Web Sync Parity Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- warning_count: `{report['warning_count']}`",
        "",
        "## Warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Payloads", "", "| payload | local | web | asof_match | hash_match |", "| --- | --- | --- | --- | --- |"])
    for row in report["payload_comparison"]:
        asof_match = str(row.get("local_asof_date")) == str(row.get("web_asof_date"))
        lines.append(
            f"| {row['payload_key']} | {row['local_asof_date'] or '-'} | {row['web_asof_date'] or '-'} | {asof_match} | {row['hash_match']} |"
        )

    lines.extend(["", "## Tables", "", "| table | local_count | web_count | match |", "| --- | ---: | ---: | --- |"])
    for row in report["table_comparison"]:
        lines.append(f"| {row['table']} | {row['local_count']} | {row['web_count']} | {row['count_match']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    local_url = args.local_database_url or os.environ.get("LOCAL_DATABASE_URL") or os.environ.get("DATABASE_URL")
    web_url = args.web_database_url or os.environ.get("WEB_DATABASE_URL")
    if not local_url:
        raise SystemExit("local database URL is required")
    if not web_url:
        raise SystemExit("WEB_DATABASE_URL is required")

    report = build_report(local_url, web_url)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"live_web_sync_parity_report_json: {out_json}")
    print(f"live_web_sync_parity_report_md: {out_md}")
    print(f"warning_count: {report['warning_count']}")
    return 2 if report["warning_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
