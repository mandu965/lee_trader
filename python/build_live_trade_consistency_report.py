from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_trade_consistency_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_trade_consistency_report.md"


LIVE_TABLES = {
    "decision": "research.live_trade_decision",
    "request": "research.live_order_request",
    "execution": "research.live_order_execution",
    "fill": "research.live_order_fill",
    "position": "research.live_position_snapshot",
    "review": "research.live_trade_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only live trade consistency report.")
    parser.add_argument("--as-of-date", default="", help="YYYY-MM-DD. Defaults to latest live decision/request/execution date.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 when warnings are present.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _scalar(conn, sql: str, params: dict[str, Any] | None = None) -> Any:
    return conn.execute(text(sql), params or {}).scalar()


def _rows(conn, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]


def _table_exists(conn, qualified_name: str) -> bool:
    return bool(_scalar(conn, "SELECT to_regclass(:name)", {"name": qualified_name}))


def _count(conn, table: str, where_sql: str = "", params: dict[str, Any] | None = None) -> int:
    if not _table_exists(conn, table):
        return 0
    suffix = f" WHERE {where_sql}" if where_sql else ""
    return int(_scalar(conn, f"SELECT COUNT(*) FROM {table}{suffix}", params) or 0)


def resolve_as_of_date(conn, requested: str) -> str | None:
    requested = str(requested or "").strip()
    if requested:
        return requested

    candidates: list[str] = []
    for table in (
        "research.live_order_execution",
        "research.live_order_request",
        "research.live_trade_decision",
    ):
        if not _table_exists(conn, table):
            continue
        value = _scalar(conn, f"SELECT MAX(as_of_date) FROM {table}")
        if value:
            candidates.append(str(value))
    return max(candidates) if candidates else None


def build_report(as_of_date: str | None) -> dict[str, Any]:
    engine = get_engine()
    with engine.begin() as conn:
        table_status = {key: _table_exists(conn, table) for key, table in LIVE_TABLES.items()}
        if not as_of_date:
            as_of_date = resolve_as_of_date(conn, "")

        params = {"as_of_date": as_of_date} if as_of_date else {}
        date_filter = "as_of_date = CAST(:as_of_date AS date)" if as_of_date else ""

        counts = {
            "intent_count": _count(conn, LIVE_TABLES["decision"], date_filter, params),
            "buy_intent_count": _count(conn, LIVE_TABLES["decision"], f"{date_filter} AND intent_type = 'BUY'", params) if as_of_date else 0,
            "request_count": _count(conn, LIVE_TABLES["request"], date_filter, params),
            "executable_request_count": _count(conn, LIVE_TABLES["request"], f"{date_filter} AND executable_now", params) if as_of_date else 0,
            "execution_count": _count(conn, LIVE_TABLES["execution"], date_filter, params),
            "submitted_count": _count(conn, LIVE_TABLES["execution"], f"{date_filter} AND submission_status = 'submitted'", params) if as_of_date else 0,
            "failed_count": _count(conn, LIVE_TABLES["execution"], f"{date_filter} AND submission_status = 'failed'", params) if as_of_date else 0,
            "skipped_count": _count(conn, LIVE_TABLES["execution"], f"{date_filter} AND submission_status = 'skipped'", params) if as_of_date else 0,
            "filled_count": _count(conn, LIVE_TABLES["fill"], date_filter, params),
        }

        request_without_execution = []
        submitted_without_fill = []
        execution_status_counts = []
        latest_snapshot = None

        if as_of_date and table_status["request"] and table_status["execution"]:
            request_without_execution = _rows(
                conn,
                """
                SELECT r.request_id, r.intent_id, r.code, r.name, r.side, r.executable_now, r.blocked_reason
                FROM research.live_order_request r
                LEFT JOIN research.live_order_execution e ON e.request_id = r.request_id
                WHERE r.as_of_date = CAST(:as_of_date AS date)
                  AND e.request_id IS NULL
                ORDER BY r.executable_now DESC, r.request_id
                LIMIT 50
                """,
                params,
            )

        if as_of_date and table_status["execution"] and table_status["fill"]:
            submitted_without_fill = _rows(
                conn,
                """
                SELECT e.request_id, e.intent_id, e.broker_order_id, e.code, e.name, e.side, e.submitted_at
                FROM research.live_order_execution e
                LEFT JOIN research.live_order_fill f
                  ON f.broker_order_id = e.broker_order_id
                 AND f.code = e.code
                 AND f.side = e.side
                WHERE e.as_of_date = CAST(:as_of_date AS date)
                  AND e.submission_status = 'submitted'
                  AND e.broker_order_id IS NOT NULL
                  AND f.fill_id IS NULL
                ORDER BY e.submitted_at DESC, e.request_id
                LIMIT 50
                """,
                params,
            )

        if as_of_date and table_status["execution"]:
            execution_status_counts = _rows(
                conn,
                """
                SELECT submission_status, COUNT(*) AS count
                FROM research.live_order_execution
                WHERE as_of_date = CAST(:as_of_date AS date)
                GROUP BY submission_status
                ORDER BY submission_status
                """,
                params,
            )

        if table_status["position"]:
            latest_snapshot = _rows(
                conn,
                """
                SELECT snapshot_date, COUNT(*) AS holding_count, MAX(snapshot_at) AS latest_snapshot_at
                FROM research.live_position_snapshot
                GROUP BY snapshot_date
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
            )
            latest_snapshot = latest_snapshot[0] if latest_snapshot else None

    warnings: list[str] = []
    if not as_of_date:
        warnings.append("No live trade as_of_date was found.")
    if counts["request_count"] and not counts["execution_count"]:
        warnings.append("Order requests exist, but no execution rows exist for the basis date.")
    if counts["submitted_count"] and counts["filled_count"] == 0:
        warnings.append("Submitted broker orders exist, but no fill rows exist for the basis date.")
    if submitted_without_fill:
        warnings.append(f"{len(submitted_without_fill)} submitted broker orders have no matching fill row.")
    if request_without_execution:
        warnings.append(f"{len(request_without_execution)} order requests have no matching execution row.")
    if latest_snapshot and as_of_date and str(latest_snapshot.get("snapshot_date")) < as_of_date:
        warnings.append("Latest position snapshot is older than the report basis date.")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "table_status": table_status,
        "counts": counts,
        "execution_status_counts": execution_status_counts,
        "request_without_execution": request_without_execution,
        "submitted_without_fill": submitted_without_fill,
        "latest_position_snapshot": latest_snapshot,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Live Trade Consistency Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- as_of_date: `{report.get('as_of_date') or '-'}`",
        f"- warning_count: `{report['warning_count']}`",
        "",
        "## Summary",
        "",
        "| metric | count |",
        "| --- | ---: |",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Execution Status", "", "| status | count |", "| --- | ---: |"])
    for row in report["execution_status_counts"]:
        lines.append(f"| {row.get('submission_status') or '-'} | {row.get('count') or 0} |")
    if not report["execution_status_counts"]:
        lines.append("| - | 0 |")

    snapshot = report.get("latest_position_snapshot") or {}
    lines.extend(
        [
            "",
            "## Latest Position Snapshot",
            "",
            f"- snapshot_date: `{snapshot.get('snapshot_date') or '-'}`",
            f"- holding_count: `{snapshot.get('holding_count') or 0}`",
            f"- latest_snapshot_at: `{snapshot.get('latest_snapshot_at') or '-'}`",
            "",
            "## Submitted Without Fill",
            "",
            "| request_id | broker_order_id | code | side | submitted_at |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["submitted_without_fill"]:
        lines.append(
            "| {request_id} | {broker_order_id} | {code} | {side} | {submitted_at} |".format(
                request_id=row.get("request_id") or "",
                broker_order_id=row.get("broker_order_id") or "",
                code=row.get("code") or "",
                side=row.get("side") or "",
                submitted_at=row.get("submitted_at") or "",
            )
        )
    if not report["submitted_without_fill"]:
        lines.append("| - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Request Without Execution",
            "",
            "| request_id | code | side | executable_now | blocked_reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["request_without_execution"]:
        lines.append(
            "| {request_id} | {code} | {side} | {executable_now} | {blocked_reason} |".format(
                request_id=row.get("request_id") or "",
                code=row.get("code") or "",
                side=row.get("side") or "",
                executable_now=row.get("executable_now"),
                blocked_reason=str(row.get("blocked_reason") or "").replace("|", "/"),
            )
        )
    if not report["request_without_execution"]:
        lines.append("| - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)

    engine = get_engine()
    with engine.begin() as conn:
        as_of_date = resolve_as_of_date(conn, args.as_of_date)

    report = build_report(as_of_date)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"live_trade_consistency_report_json: {out_json}")
    print(f"live_trade_consistency_report_md: {out_md}")
    print(f"warning_count: {report['warning_count']}")
    get_engine().dispose()
    return 2 if args.strict and report["warning_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
