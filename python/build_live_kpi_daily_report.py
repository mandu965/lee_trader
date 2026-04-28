from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from apply_analytics_views import apply_analytics_views
from db import get_engine
from payload_store import upsert_json_payload
from report_json import sanitize_for_json, write_json_strict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_kpi_daily_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_kpi_daily_report.md"
LIVE_BALANCE_JSON = OUTPUT_DIR / "live_account_balance_summary.json"
CONSISTENCY_JSON = OUTPUT_DIR / "live_trade_consistency_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live-trading KPI report from analytics views.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--skip-ensure-views", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    engine = get_engine()
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]


def _one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


def _sample_status(observed_count: int) -> str:
    if observed_count < 30:
        return "INSUFFICIENT_SAMPLE"
    if observed_count < 100:
        return "MONITOR_ONLY"
    return "ACTIONABLE"


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.0f}"


def build_report() -> dict[str, Any]:
    overview = _one(
        """
        SELECT
            MAX(as_of_date)::text AS as_of_date,
            COUNT(*)::integer AS fill_count,
            COUNT(DISTINCT request_id)::integer AS filled_request_count,
            COALESCE(SUM(filled_amount), 0)::numeric AS filled_amount,
            COUNT(*) FILTER (WHERE submission_status IS NULL)::integer AS fill_without_execution_count,
            COUNT(*) FILTER (WHERE production_rank IS NULL)::integer AS missing_ranking_context_count,
            COUNT(*) FILTER (WHERE shadow_quality_risk_guard_applied)::integer AS guard_applied_fill_count
        FROM analytics.live_trade_fact
        """
    )
    as_of_date = overview.get("as_of_date")

    today_counts = _one(
        """
        SELECT
            (SELECT COUNT(*) FROM research.live_trade_decision WHERE as_of_date = CAST(:as_of_date AS date))::integer AS decision_count,
            (SELECT COUNT(*) FROM research.live_order_request WHERE as_of_date = CAST(:as_of_date AS date))::integer AS request_count,
            (SELECT COUNT(*) FROM research.live_order_execution WHERE as_of_date = CAST(:as_of_date AS date))::integer AS execution_count,
            (SELECT COUNT(*) FROM research.live_order_fill WHERE as_of_date = CAST(:as_of_date AS date))::integer AS fill_count
        """,
        {"as_of_date": as_of_date} if as_of_date else {"as_of_date": "1900-01-01"},
    )
    horizon_summary = _rows(
        """
        SELECT
            horizon,
            COUNT(*)::integer AS count,
            COUNT(signed_return)::integer AS observed_count,
            AVG(signed_return) AS avg_return,
            AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
            MIN(signed_return) AS max_loss
        FROM analytics.live_trade_return_fact
        GROUP BY horizon
        ORDER BY horizon
        """
    )
    for row in horizon_summary:
        row["sample_status"] = _sample_status(int(row.get("observed_count") or 0))

    by_intent = _rows(
        """
        SELECT horizon, intent_type, SUM(count)::integer AS count, SUM(observed_count)::integer AS observed_count,
               AVG(avg_return) AS avg_return, AVG(win_rate) AS win_rate
        FROM analytics.live_review_kpi
        GROUP BY horizon, intent_type
        ORDER BY horizon, observed_count DESC, count DESC, intent_type
        """
    )
    by_rank_bucket = _rows(
        """
        SELECT horizon, rank_bucket, SUM(count)::integer AS count, SUM(observed_count)::integer AS observed_count,
               AVG(avg_return) AS avg_return, AVG(win_rate) AS win_rate
        FROM analytics.live_score_bucket_kpi
        GROUP BY horizon, rank_bucket
        ORDER BY horizon, rank_bucket
        """
    )
    by_score_bucket = _rows(
        """
        SELECT horizon, final_score_bucket, SUM(count)::integer AS count, SUM(observed_count)::integer AS observed_count,
               AVG(avg_return) AS avg_return, AVG(win_rate) AS win_rate
        FROM analytics.live_score_bucket_kpi
        GROUP BY horizon, final_score_bucket
        ORDER BY horizon, final_score_bucket
        """
    )
    by_confidence_bucket = _rows(
        """
        SELECT horizon, confidence_bucket, SUM(count)::integer AS count, SUM(observed_count)::integer AS observed_count,
               AVG(avg_return) AS avg_return, AVG(win_rate) AS win_rate
        FROM analytics.live_score_bucket_kpi
        GROUP BY horizon, confidence_bucket
        ORDER BY horizon, confidence_bucket
        """
    )
    by_risk_bucket = _rows(
        """
        SELECT horizon, risk_penalty_bucket, SUM(count)::integer AS count, SUM(observed_count)::integer AS observed_count,
               AVG(avg_return) AS avg_return, AVG(win_rate) AS win_rate
        FROM analytics.live_score_bucket_kpi
        GROUP BY horizon, risk_penalty_bucket
        ORDER BY horizon, risk_penalty_bucket
        """
    )

    balance = _read_json(LIVE_BALANCE_JSON)
    consistency = _read_json(CONSISTENCY_JSON)
    derived = balance.get("derived_metrics") if isinstance(balance.get("derived_metrics"), dict) else {}
    cash_summary = balance.get("cash_summary") if isinstance(balance.get("cash_summary"), dict) else {}
    warnings: list[str] = []
    observed_d5 = next((int(row.get("observed_count") or 0) for row in horizon_summary if int(row.get("horizon") or -1) == 5), 0)
    if observed_d5 < 30:
        warnings.append("D+5 observed_count is below 30; do not change trading parameters from this report.")
    if int(overview.get("missing_ranking_context_count") or 0) > 0:
        warnings.append("Some fills have no ranking context; review ledger/ranking sync before promotion analysis.")
    if int((consistency.get("warning_count") if isinstance(consistency, dict) else 0) or 0) > 0:
        warnings.append("Live trade consistency report has warnings.")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "overview": overview,
        "account": {
            "generated_at": balance.get("generated_at"),
            "total_assets": derived.get("total_assets") or cash_summary.get("tot_evlu_amt"),
            "cash_amount": derived.get("cash_amount") or cash_summary.get("dnca_tot_amt"),
            "cash_ratio": derived.get("cash_ratio"),
            "holding_count": balance.get("holding_count"),
        },
        "today_counts": today_counts,
        "consistency": {
            "warning_count": consistency.get("warning_count") if isinstance(consistency, dict) else None,
            "counts": consistency.get("counts") if isinstance(consistency, dict) else {},
        },
        "horizon_summary": horizon_summary,
        "by_intent": by_intent,
        "by_rank_bucket": by_rank_bucket,
        "by_score_bucket": by_score_bucket,
        "by_confidence_bucket": by_confidence_bucket,
        "by_risk_penalty_bucket": by_risk_bucket,
        "sample_status": _sample_status(max((int(row.get("observed_count") or 0) for row in horizon_summary), default=0)),
        "warnings": warnings,
    }
    return sanitize_for_json(report)


def _render_group(title: str, rows: list[dict[str, Any]], key: str, *, horizon: int = 0, limit: int = 8) -> list[str]:
    filtered = [row for row in rows if int(row.get("horizon") or -1) == horizon][:limit]
    lines = ["", f"## {title}", "", "| group | count | observed | avg_return | win_rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in filtered:
        lines.append(
            f"| {row.get(key) or '-'} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_pct(row.get('avg_return'))} | {_pct(row.get('win_rate'))} |"
        )
    if not filtered:
        lines.append("| - | 0 | 0 | - | - |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    account = report["account"]
    today = report["today_counts"]
    lines = [
        "# Live KPI Daily Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- as_of_date: `{report.get('as_of_date') or '-'}`",
        f"- sample_status: `{report.get('sample_status')}`",
        f"- fills: `{overview.get('fill_count') or 0}` / amount: `{_num(overview.get('filled_amount'))}`",
        f"- total_assets: `{_num(account.get('total_assets'))}` / cash: `{_num(account.get('cash_amount'))}` / cash_ratio: `{_pct(account.get('cash_ratio'))}`",
        f"- today decision/request/execution/fill: `{today.get('decision_count') or 0}` / `{today.get('request_count') or 0}` / `{today.get('execution_count') or 0}` / `{today.get('fill_count') or 0}`",
        "",
        "## Horizon Summary",
        "",
        "| horizon | count | observed | avg_return | win_rate | max_loss | sample_status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["horizon_summary"]:
        lines.append(
            f"| D+{row.get('horizon')} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_pct(row.get('avg_return'))} | {_pct(row.get('win_rate'))} | {_pct(row.get('max_loss'))} | {row.get('sample_status')} |"
        )
    warnings = report.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(_render_group("D0 By Intent", report["by_intent"], "intent_type", horizon=0))
    lines.extend(_render_group("D0 By Rank Bucket", report["by_rank_bucket"], "rank_bucket", horizon=0))
    lines.extend(_render_group("D0 By Score Bucket", report["by_score_bucket"], "final_score_bucket", horizon=0))
    lines.extend(_render_group("D0 By Confidence Bucket", report["by_confidence_bucket"], "confidence_bucket", horizon=0))
    lines.extend(_render_group("D0 By Risk Penalty Bucket", report["by_risk_penalty_bucket"], "risk_penalty_bucket", horizon=0))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.skip_ensure_views:
        apply_analytics_views()
    report = build_report()
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    write_json_strict(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    upsert_json_payload(
        "live_kpi_daily_report",
        report,
        asof_date=report.get("as_of_date"),
        generated_at=report.get("generated_at"),
        source_path=out_json,
    )
    print(f"live_kpi_daily_report_json: {out_json}")
    print(f"live_kpi_daily_report_md: {out_md}")
    print(f"sample_status: {report.get('sample_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
