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
DEFAULT_OUT_JSON = OUTPUT_DIR / "quality_risk_guard_live_review.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "quality_risk_guard_live_review.md"
DEFAULT_CLOSED_TRADE_JSON = OUTPUT_DIR / "live_closed_trade_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review quality_risk_guard shadow performance using live fills.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--skip-ensure-views", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    engine = get_engine()
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]


def _one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


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


def _closed_trade_summary(closed_report: dict[str, Any]) -> dict[str, Any]:
    overview = closed_report.get("overview") if isinstance(closed_report.get("overview"), dict) else {}
    by_quality_guard = closed_report.get("by_quality_guard")
    if not isinstance(by_quality_guard, list):
        by_quality_guard = []
    return {
        "available": bool(overview),
        "latest_closed_date": closed_report.get("latest_closed_date") or overview.get("latest_closed_date"),
        "sample_status": closed_report.get("sample_status"),
        "closed_trade_count": overview.get("closed_trade_count"),
        "observed_count": overview.get("observed_count"),
        "realized_net_pnl": overview.get("realized_net_pnl"),
        "avg_realized_return": overview.get("avg_realized_return"),
        "win_rate": overview.get("win_rate"),
        "max_loss": overview.get("max_loss"),
        "snapshot_fallback_count": overview.get("snapshot_fallback_count"),
        "unmatched_count": overview.get("unmatched_count"),
        "match_method": overview.get("match_method"),
        "by_quality_guard": by_quality_guard,
        "warnings": closed_report.get("warnings") if isinstance(closed_report.get("warnings"), list) else [],
    }


def _promotion_status(
    summary: dict[str, Any],
    d5: dict[str, Any],
    guard_groups: list[dict[str, Any]],
    closed_summary: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    d5_observed = int(d5.get("observed_count") or 0)
    applied = next((row for row in guard_groups if bool(row.get("shadow_quality_risk_guard_applied"))), {})
    not_applied = next((row for row in guard_groups if not bool(row.get("shadow_quality_risk_guard_applied"))), {})
    applied_observed = int(applied.get("observed_count") or 0)
    not_applied_observed = int(not_applied.get("observed_count") or 0)

    if d5_observed < 30:
        blockers.append("D+5 observed_count is below 30.")
    if applied_observed < 30:
        blockers.append("Guard-applied observed_count is below 30.")
    if not_applied_observed < 30:
        blockers.append("Guard-not-applied observed_count is below 30.")
    if not summary.get("has_production_shadow_top20_comparison"):
        blockers.append("Production top20 vs shadow top20 comparison is not available.")
    if not closed_summary.get("available"):
        blockers.append("Closed-trade report is not available.")
    else:
        closed_observed = int(closed_summary.get("observed_count") or 0)
        snapshot_fallback_count = int(closed_summary.get("snapshot_fallback_count") or 0)
        unmatched_count = int(closed_summary.get("unmatched_count") or 0)
        if closed_observed < 30:
            blockers.append("Closed-trade observed_count is below 30.")
        if snapshot_fallback_count > 0:
            blockers.append("Closed-trade PnL uses position snapshot avg_price fallback; lot-level evidence is approximate.")
        if unmatched_count > 0:
            blockers.append("Closed-trade report still has unmatched SELL basis rows.")

    if blockers:
        return "KEEP_SHADOW", blockers
    if d5_observed < 100:
        return "REVIEW_READY", ["Sample is monitorable but still below ACTIONABLE threshold."]
    return "PROMOTE_CANDIDATE", []


def build_report() -> dict[str, Any]:
    closed_summary = _closed_trade_summary(_read_json(DEFAULT_CLOSED_TRADE_JSON))
    overview = _one(
        """
        SELECT
            MAX(as_of_date)::text AS as_of_date,
            COUNT(*)::integer AS fill_count,
            COUNT(*) FILTER (WHERE shadow_quality_risk_guard_applied)::integer AS guard_applied_count,
            COUNT(*) FILTER (WHERE shadow_quality_risk_guard_applied = false)::integer AS guard_not_applied_count,
            COUNT(*) FILTER (WHERE production_rank <= 20)::integer AS production_top20_count,
            COUNT(*) FILTER (WHERE shadow_rank_quality_risk_guard <= 20)::integer AS shadow_top20_count,
            BOOL_OR(production_rank <= 20) AND BOOL_OR(shadow_rank_quality_risk_guard <= 20) AS has_production_shadow_top20_comparison
        FROM analytics.live_trade_fact
        """
    )
    horizon_summary = _rows(
        """
        SELECT
            horizon,
            COUNT(*)::integer AS count,
            COUNT(signed_return)::integer AS observed_count,
            AVG(signed_return) AS avg_return,
            AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
            AVG(CASE WHEN signed_return < 0 THEN signed_return END) AS avg_downside_return,
            MIN(signed_return) AS max_loss
        FROM analytics.live_trade_return_fact
        GROUP BY horizon
        ORDER BY horizon
        """
    )
    for row in horizon_summary:
        row["sample_status"] = _sample_status(int(row.get("observed_count") or 0))

    by_guard_applied = _rows(
        """
        SELECT
            horizon,
            shadow_quality_risk_guard_applied,
            SUM(count)::integer AS count,
            SUM(observed_count)::integer AS observed_count,
            AVG(avg_return) AS avg_return,
            AVG(weighted_avg_return) AS weighted_avg_return,
            AVG(win_rate) AS win_rate,
            AVG(expectancy) AS expectancy,
            AVG(avg_downside_return) AS avg_downside_return,
            MIN(max_loss) AS max_loss
        FROM analytics.live_quality_guard_kpi
        GROUP BY horizon, shadow_quality_risk_guard_applied
        ORDER BY horizon, shadow_quality_risk_guard_applied
        """
    )
    by_penalty_bucket = _rows(
        """
        SELECT
            horizon,
            guard_penalty_bucket,
            SUM(count)::integer AS count,
            SUM(observed_count)::integer AS observed_count,
            AVG(avg_return) AS avg_return,
            AVG(win_rate) AS win_rate,
            AVG(expectancy) AS expectancy,
            MIN(max_loss) AS max_loss
        FROM analytics.live_quality_guard_kpi
        GROUP BY horizon, guard_penalty_bucket
        ORDER BY horizon, guard_penalty_bucket
        """
    )
    by_shadow_delta = _rows(
        """
        SELECT
            horizon,
            shadow_rank_delta_bucket,
            SUM(count)::integer AS count,
            SUM(observed_count)::integer AS observed_count,
            AVG(avg_return) AS avg_return,
            AVG(win_rate) AS win_rate,
            AVG(expectancy) AS expectancy,
            MIN(max_loss) AS max_loss
        FROM analytics.live_quality_guard_kpi
        GROUP BY horizon, shadow_rank_delta_bucket
        ORDER BY horizon, shadow_rank_delta_bucket
        """
    )
    top20_comparison = _rows(
        """
        SELECT
            horizon,
            production_rank_bucket,
            shadow_rank_bucket,
            COUNT(*)::integer AS count,
            COUNT(signed_return)::integer AS observed_count,
            AVG(signed_return) AS avg_return,
            AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
            AVG(CASE WHEN signed_return < 0 THEN signed_return END) AS avg_downside_return,
            MIN(signed_return) AS max_loss
        FROM analytics.live_trade_return_fact
        GROUP BY horizon, production_rank_bucket, shadow_rank_bucket
        ORDER BY horizon, production_rank_bucket, shadow_rank_bucket
        """
    )

    d5 = next((row for row in horizon_summary if int(row.get("horizon") or -1) == 5), {})
    d5_groups = [row for row in by_guard_applied if int(row.get("horizon") or -1) == 5]
    promotion_status, blockers = _promotion_status(overview, d5, d5_groups, closed_summary)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": overview.get("as_of_date"),
        "overview": overview,
        "horizon_summary": horizon_summary,
        "by_guard_applied": by_guard_applied,
        "by_penalty_bucket": by_penalty_bucket,
        "by_shadow_rank_delta": by_shadow_delta,
        "top20_comparison": top20_comparison,
        "closed_trade_summary": closed_summary,
        "sample_status": _sample_status(max((int(row.get("observed_count") or 0) for row in horizon_summary), default=0)),
        "promotion_status": promotion_status,
        "promotion_blockers": blockers,
    }
    return sanitize_for_json(report)


def _render_rows(title: str, rows: list[dict[str, Any]], key: str, *, horizon: int = 0) -> list[str]:
    filtered = []
    for row in rows:
        if row.get("horizon") is None:
            continue
        if int(row.get("horizon")) == horizon:
            filtered.append(row)
    lines = ["", f"## {title}", "", "| group | count | observed | avg_return | win_rate | max_loss |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in filtered:
        lines.append(
            f"| {row.get(key)} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_pct(row.get('avg_return'))} | {_pct(row.get('win_rate'))} | {_pct(row.get('max_loss'))} |"
        )
    if not filtered:
        lines.append("| - | 0 | 0 | - | - | - |")
    return lines


def _render_closed_quality_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Closed Trade By Quality Guard",
        "",
        "| guard_penalty | shadow_delta | count | observed | pnl | avg_return | win_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('guard_penalty_bucket') or '-'} | {row.get('shadow_rank_delta_bucket') or '-'} | "
            f"{row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_num(row.get('realized_net_pnl'))} | {_pct(row.get('avg_realized_return'))} | {_pct(row.get('win_rate'))} |"
        )
    if not rows:
        lines.append("| - | - | 0 | 0 | - | - | - |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    closed = report.get("closed_trade_summary") or {}
    lines = [
        "# Quality Risk Guard Live Review",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- as_of_date: `{report.get('as_of_date') or '-'}`",
        f"- promotion_status: `{report.get('promotion_status')}`",
        f"- sample_status: `{report.get('sample_status')}`",
        f"- fills: `{overview.get('fill_count') or 0}`",
        f"- guard_applied/not_applied: `{overview.get('guard_applied_count') or 0}` / `{overview.get('guard_not_applied_count') or 0}`",
        f"- production_top20/shadow_top20: `{overview.get('production_top20_count') or 0}` / `{overview.get('shadow_top20_count') or 0}`",
        f"- closed_trade_observed: `{closed.get('observed_count') or 0}`",
        f"- closed_trade_pnl: `{_num(closed.get('realized_net_pnl'))}`",
        f"- closed_trade_avg_return: `{_pct(closed.get('avg_realized_return'))}`",
        f"- closed_trade_snapshot_fallback_count: `{closed.get('snapshot_fallback_count') or 0}`",
        "",
        "## Horizon Summary",
        "",
        "| horizon | count | observed | avg_return | win_rate | downside | max_loss | sample_status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["horizon_summary"]:
        lines.append(
            f"| D+{row.get('horizon')} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_pct(row.get('avg_return'))} | {_pct(row.get('win_rate'))} | {_pct(row.get('avg_downside_return'))} | "
            f"{_pct(row.get('max_loss'))} | {row.get('sample_status')} |"
        )
    lines.extend(["", "## Promotion Blockers", ""])
    blockers = report.get("promotion_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Closed Trade Summary", ""])
    if closed.get("available"):
        lines.extend(
            [
                f"- latest_closed_date: `{closed.get('latest_closed_date') or '-'}`",
                f"- sample_status: `{closed.get('sample_status') or '-'}`",
                f"- match_method: `{closed.get('match_method') or '-'}`",
                f"- closed_trade_count: `{closed.get('closed_trade_count') or 0}`",
                f"- observed_count: `{closed.get('observed_count') or 0}`",
                f"- realized_net_pnl: `{_num(closed.get('realized_net_pnl'))}`",
                f"- avg_realized_return: `{_pct(closed.get('avg_realized_return'))}`",
                f"- win_rate: `{_pct(closed.get('win_rate'))}`",
                f"- snapshot_fallback_count: `{closed.get('snapshot_fallback_count') or 0}`",
                f"- unmatched_count: `{closed.get('unmatched_count') or 0}`",
            ]
        )
        warnings = closed.get("warnings") if isinstance(closed.get("warnings"), list) else []
        lines.extend(f"- warning: {warning}" for warning in warnings)
    else:
        lines.append("- Closed trade report is not available.")
    lines.extend(_render_closed_quality_rows(closed.get("by_quality_guard") or []))
    lines.extend(_render_rows("D0 Guard Applied", report["by_guard_applied"], "shadow_quality_risk_guard_applied", horizon=0))
    lines.extend(_render_rows("D0 Penalty Bucket", report["by_penalty_bucket"], "guard_penalty_bucket", horizon=0))
    lines.extend(_render_rows("D0 Shadow Rank Delta", report["by_shadow_rank_delta"], "shadow_rank_delta_bucket", horizon=0))
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
        "quality_risk_guard_live_review",
        report,
        asof_date=report.get("as_of_date"),
        generated_at=report.get("generated_at"),
        source_path=out_json,
    )
    print(f"quality_risk_guard_live_review_json: {out_json}")
    print(f"quality_risk_guard_live_review_md: {out_md}")
    print(f"promotion_status: {report.get('promotion_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
