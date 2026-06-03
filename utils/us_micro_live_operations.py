from __future__ import annotations

import csv
from collections import Counter
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path

from python.us.us_db import (
    fetch_rank_component_rows_between,
    fetch_us_live_daily_risk_usage_rows,
    fetch_us_live_kill_switch_rows,
    fetch_us_live_order_approval_rows,
    fetch_us_live_order_block_log_rows,
    fetch_us_micro_order_fill_rows,
    fetch_us_micro_reconciliation_result_rows,
)
from utils.us_live_kill_switch import activate_kill_switch
from utils.us_micro_order_request import get_micro_order_events, list_micro_orders

try:
    from python.notifier import notify_critical, notify_warning
except ImportError:
    notify_critical = None
    notify_warning = None


HEALTH_STATES = {"HEALTHY", "ATTENTION", "DEGRADED", "CRITICAL"}
ACTION_SEVERITIES = {"INFO", "WARNING", "ERROR", "CRITICAL"}


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _safe_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _safe_float(value: object) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_trade_date(trade_date: str) -> date:
    return date.fromisoformat(str(trade_date)[:10])


def _output_dir(path: str | None = None) -> Path:
    raw = path or _text("US_MICRO_OPS_REPORT_OUTPUT_DIR", "output/us_stock_micro_live")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _collect_order_events(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for row in orders:
        micro_order_id = _safe_str(row.get("micro_order_id"))
        if not micro_order_id:
            continue
        events.extend(get_micro_order_events(micro_order_id))
    return events


def _collect_fills(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in orders:
        rows.extend(fetch_us_micro_order_fill_rows(micro_order_id=_safe_str(order.get("micro_order_id"))))
    return rows


def _summarize_ranking(rank_rows: list[dict[str, object]]) -> dict[str, object]:
    grades = Counter(_safe_str(row.get("recommend_grade")).upper() for row in rank_rows)
    eligible_buy = sum(1 for row in rank_rows if _safe_str(row.get("recommend_grade")).upper() in {"BUY", "STRONG_BUY"})
    return {
        "top20_count": len(rank_rows),
        "buy_or_better_count": eligible_buy,
        "exclude_count": grades.get("EXCLUDE", 0),
        "grade_counts": dict(grades),
        "rows": rank_rows,
    }


def _summarize_precheck(
    rank_rows: list[dict[str, object]],
    approvals: list[dict[str, object]],
    block_logs: list[dict[str, object]],
) -> dict[str, object]:
    candidate_symbols = {_safe_str(row.get("symbol")).upper() for row in rank_rows if _safe_str(row.get("recommend_grade")).upper() in {"BUY", "STRONG_BUY"}}
    approval_symbols = {_safe_str(row.get("symbol")).upper() for row in approvals}
    error_symbols = {
        _safe_str(row.get("symbol")).upper()
        for row in block_logs
        if _safe_str(row.get("severity")).upper() in {"ERROR", "CRITICAL"} and "error" in _safe_str(row.get("block_reason_code")).lower()
    }
    block_symbols = {_safe_str(row.get("symbol")).upper() for row in block_logs}
    allow_symbols = candidate_symbols - approval_symbols - block_symbols
    block_reason_counts = Counter(_safe_str(row.get("block_reason_code")).lower() or "unknown" for row in block_logs)
    return {
        "ALLOW": len(allow_symbols),
        "REQUIRE_APPROVAL": len(approval_symbols & candidate_symbols) if candidate_symbols else len(approval_symbols),
        "BLOCK": len(block_symbols - error_symbols),
        "ERROR": len(error_symbols),
        "top_block_reasons": block_reason_counts.most_common(10),
    }


def _summarize_approvals(approvals: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(_safe_str(row.get("approval_status")).upper() for row in approvals)
    return {
        "counts": dict(counts),
        "pending": counts.get("PENDING", 0),
        "approved": counts.get("APPROVED", 0),
        "rejected": counts.get("REJECTED", 0),
        "expired": counts.get("EXPIRED", 0),
        "rows": approvals,
    }


def _summarize_orders(orders: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(_safe_str(row.get("request_status")).upper() for row in orders)
    execution_modes = Counter(_safe_str(row.get("execution_mode")).upper() for row in orders)
    return {"counts": dict(counts), "execution_modes": dict(execution_modes), "rows": orders}


def _summarize_fills(fills: list[dict[str, object]]) -> dict[str, object]:
    total_amount = sum(_safe_float(row.get("filled_amount_usd")) or 0.0 for row in fills)
    return {"fill_count": len(fills), "total_filled_amount_usd": round(total_amount, 6), "rows": fills}


def _summarize_reconciliation(results: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(_safe_str(row.get("recon_status")).upper() for row in results)
    severity_counts = Counter(_safe_str(row.get("severity")).upper() for row in results)
    critical_rows = [row for row in results if _safe_str(row.get("severity")).upper() == "CRITICAL"]
    return {
        "counts": dict(counts),
        "severity_counts": dict(severity_counts),
        "match": counts.get("MATCH", 0),
        "mismatch": counts.get("MISMATCH", 0),
        "critical": severity_counts.get("CRITICAL", 0),
        "error": severity_counts.get("ERROR", 0) + counts.get("ERROR", 0),
        "rows": results,
        "critical_rows": critical_rows,
    }


def _summarize_kill_switches(rows: list[dict[str, object]]) -> dict[str, object]:
    active = [row for row in rows if bool(row.get("is_active"))]
    return {"active_count": len(active), "active_rows": active, "rows": rows}


def _summarize_daily_risk_usage(rows: list[dict[str, object]]) -> dict[str, object]:
    row = rows[0] if rows else {}
    return {
        "row": row,
        "total_order_count": _safe_int(row.get("total_order_count")) or 0,
        "failed_order_count": _safe_int(row.get("failed_order_count")) or 0,
        "blocked_order_count": _safe_int(row.get("blocked_order_count")) or 0,
        "new_buy_count": _safe_int(row.get("new_buy_count")) or 0,
        "buy_amount_usd": _safe_float(row.get("buy_amount_usd")) or 0.0,
        "sell_amount_usd": _safe_float(row.get("sell_amount_usd")) or 0.0,
    }


def _order_age_minutes(row: dict[str, object]) -> float | None:
    for key in ("last_sync_at", "created_at"):
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, datetime):
            dt = value
        else:
            text = _safe_str(value).replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    return None


def generate_action_required(report: dict[str, object]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    kill_summary = report["kill_switch"]
    approval_summary = report["approvals"]
    order_summary = report["orders"]
    recon_summary = report["reconciliation"]
    risk_summary = report["daily_risk_usage"]
    block_logs = report["block_logs"]["rows"]
    orders = order_summary["rows"]

    if kill_summary["active_count"] > 0:
        actions.append({"severity": "CRITICAL", "message": "Kill Switch is active. Confirm whether this is intentional.", "reason_code": "kill_switch_active"})
    if report["precheck"]["ERROR"] > 0:
        actions.append({"severity": "ERROR", "message": "Pre-Trade Check ERROR exists. Review block logs before any Micro Live progression.", "reason_code": "precheck_error_exists"})
    if approval_summary["pending"] > 0:
        actions.append({"severity": "WARNING", "message": f"{approval_summary['pending']} approval requests are pending.", "reason_code": "approval_pending"})
    if approval_summary["expired"] > 0:
        actions.append({"severity": "WARNING", "message": f"{approval_summary['expired']} approval requests expired. Re-run Pre-Trade Check before reuse.", "reason_code": "approval_expired"})
    if order_summary["counts"].get("SYNC_ERROR", 0) > 0:
        actions.append({"severity": "ERROR", "message": "SYNC_ERROR exists in Micro Orders. Check broker-status sync path.", "reason_code": "micro_order_sync_error"})
    if order_summary["counts"].get("ORDER_UNKNOWN", 0) > 0:
        actions.append({"severity": "ERROR", "message": "ORDER_UNKNOWN exists. Check broker status mapping.", "reason_code": "order_unknown_exists"})
    if order_summary["counts"].get("ORDER_PARTIALLY_FILLED", 0) > 0:
        actions.append({"severity": "WARNING", "message": "Partially filled orders exist. Review remaining quantity manually.", "reason_code": "partial_fill_exists"})
    stale_threshold = _safe_float(_text("US_MICRO_OPS_STALE_OPEN_ORDER_MINUTES", "60")) or 60.0
    stale_open = [
        row for row in orders
        if _safe_str(row.get("request_status")).upper() == "ORDER_OPEN" and ((_order_age_minutes(row) or 0.0) >= stale_threshold)
    ]
    if stale_open:
        actions.append({"severity": "WARNING", "message": f"{len(stale_open)} open orders are stale beyond {int(stale_threshold)} minutes.", "reason_code": "stale_open_order"})
    if order_summary["counts"].get("ORDER_REJECTED", 0) > 0:
        actions.append({"severity": "WARNING", "message": "Rejected orders exist. Review reject reason before retrying anything manually.", "reason_code": "order_rejected_exists"})
    if recon_summary["mismatch"] > 0:
        actions.append({"severity": "ERROR", "message": "Reconciliation mismatch exists. Do not trust internal state blindly.", "reason_code": "reconciliation_mismatch"})
    if recon_summary["critical"] > 0:
        actions.append({"severity": "CRITICAL", "message": "Reconciliation CRITICAL mismatch detected. Do not proceed with Micro Live.", "reason_code": "reconciliation_critical"})
    total_order_limit = _safe_int(risk_summary["row"].get("max_daily_order_count")) or 3
    if total_order_limit and risk_summary["total_order_count"] >= max(1, total_order_limit - 1):
        actions.append({"severity": "WARNING", "message": "Daily risk usage is near the total-order-count limit.", "reason_code": "daily_risk_near_limit"})
    repeat_blocks = Counter(_safe_str(row.get("block_reason_code")).lower() or "unknown" for row in block_logs)
    repeated = [(code, count) for code, count in repeat_blocks.items() if count >= 3]
    for code, count in repeated[:3]:
        actions.append({"severity": "INFO", "message": f"Block reason `{code}` repeated {count} times.", "reason_code": "repeated_block_reason"})
    duplicate_keys = Counter(
        (
            _safe_str(row.get("symbol")).upper(),
            _safe_str(row.get("side")).upper(),
            _safe_str(row.get("request_status")).upper(),
        )
        for row in orders
    )
    if any(count >= 2 and status in {"CREATED", "READY_TO_SEND", "SENT", "ACCEPTED", "ORDER_OPEN"} for (_, _, status), count in duplicate_keys.items()):
        actions.append({"severity": "CRITICAL", "message": "Potential duplicate Micro Order detected for the same symbol/side/status cluster.", "reason_code": "duplicate_order_detected"})
    actions.sort(key=lambda item: {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}.get(item["severity"], 9))
    return actions


def derive_health_status(report: dict[str, object]) -> dict[str, str]:
    actions = report["actions"]
    if any(item["severity"] == "CRITICAL" for item in actions):
        reasons = [item["message"] for item in actions if item["severity"] == "CRITICAL"][:2]
        return {"status": "CRITICAL", "reason": "; ".join(reasons)}
    if any(item["severity"] == "ERROR" for item in actions):
        reasons = [item["message"] for item in actions if item["severity"] == "ERROR"][:2]
        return {"status": "DEGRADED", "reason": "; ".join(reasons)}
    if any(item["severity"] == "WARNING" for item in actions):
        reasons = [item["message"] for item in actions if item["severity"] == "WARNING"][:2]
        return {"status": "ATTENTION", "reason": "; ".join(reasons)}
    return {"status": "HEALTHY", "reason": "No critical or error operational issue detected."}


def build_micro_live_operations_report(
    *,
    trade_date: str,
    account_id: str,
    include_ranking: bool = True,
    include_precheck: bool = True,
    include_approvals: bool = True,
    include_orders: bool = True,
    include_fills: bool = True,
    include_reconciliation: bool = True,
    include_kill_switch: bool = True,
    include_actions: bool = True,
    activate_kill_on_critical: bool = False,
    performed_by: str = "SYSTEM",
) -> dict[str, object]:
    trade_date_value = _parse_trade_date(trade_date)
    rank_rows = fetch_rank_component_rows_between(start_date=trade_date_value, end_date=trade_date_value, source="rule_v1") if include_ranking else []
    rank_rows = [row for row in rank_rows if _safe_int(row.get("rank_no")) is not None and int(row.get("rank_no")) <= 20]
    approvals = fetch_us_live_order_approval_rows(trade_date=trade_date_value, account_id=account_id) if include_approvals else []
    block_logs = fetch_us_live_order_block_log_rows(trade_date=trade_date_value, account_id=account_id) if include_precheck else []
    orders = list_micro_orders(trade_date=trade_date, account_id=account_id) if include_orders else []
    order_events = _collect_order_events(orders) if include_orders else []
    fills = _collect_fills(orders) if include_fills else []
    recon_results = fetch_us_micro_reconciliation_result_rows(recon_date=trade_date_value, account_id=account_id) if include_reconciliation else []
    kill_switch_rows = fetch_us_live_kill_switch_rows() if include_kill_switch else []
    daily_risk_rows = fetch_us_live_daily_risk_usage_rows(trade_date=trade_date_value, account_id=account_id)

    report = {
        "trade_date": trade_date_value.isoformat(),
        "account_id": account_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "US_LIVE_ORDER_ENABLED": _flag("US_LIVE_ORDER_ENABLED", "false"),
            "US_MICRO_ALLOW_LIVE": _flag("US_MICRO_ALLOW_LIVE", "false"),
            "US_MICRO_REAL_ORDER_BLOCKED": _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"),
            "US_MICRO_ALLOW_LIVE_STATUS_QUERY": _flag("US_MICRO_ALLOW_LIVE_STATUS_QUERY", "false"),
            "US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY": _flag("US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY", "false"),
        },
        "ranking": _summarize_ranking(rank_rows),
        "precheck": _summarize_precheck(rank_rows, approvals, block_logs),
        "block_logs": {"rows": block_logs, "count": len(block_logs)},
        "approvals": _summarize_approvals(approvals),
        "orders": _summarize_orders(orders),
        "order_events": {"rows": order_events, "count": len(order_events)},
        "fills": _summarize_fills(fills),
        "reconciliation": _summarize_reconciliation(recon_results),
        "kill_switch": _summarize_kill_switches(kill_switch_rows),
        "daily_risk_usage": _summarize_daily_risk_usage(daily_risk_rows),
    }
    report["actions"] = generate_action_required(report) if include_actions else []
    report["health"] = derive_health_status(report)
    report["kill_switch_recommended"] = any(item["severity"] == "CRITICAL" for item in report["actions"])
    report["kill_switch_triggered"] = False
    if activate_kill_on_critical and report["kill_switch_recommended"]:
        if not report["kill_switch"]["active_count"]:
            activate_kill_switch(
                scope="ACCOUNT",
                target_value=account_id,
                reason_code="micro_live_operations_critical",
                reason_detail=report["health"]["reason"],
                performed_by=performed_by,
                trigger_source="MICRO_LIVE_OPERATIONS_REPORT",
                trigger_ref_id=f"{trade_date}:{account_id}",
            )
        report["kill_switch_triggered"] = True
    return report


def render_operations_console(report: dict[str, object]) -> str:
    ranking = report["ranking"]
    precheck = report["precheck"]
    approvals = report["approvals"]
    orders = report["orders"]
    fills = report["fills"]
    recon = report["reconciliation"]
    kill_switch = report["kill_switch"]
    risk = report["daily_risk_usage"]
    lines = [
        "[US Micro Live Operations Report]",
        f"Trade Date: {report['trade_date']}",
        f"Account: {report['account_id']}",
        f"System Health: {report['health']['status']}",
        f"Reason: {report['health']['reason']}",
        "",
        "[Safety Status]",
        f"US_LIVE_ORDER_ENABLED: {str(report['safety']['US_LIVE_ORDER_ENABLED']).lower()}",
        f"US_MICRO_ALLOW_LIVE: {str(report['safety']['US_MICRO_ALLOW_LIVE']).lower()}",
        f"US_MICRO_REAL_ORDER_BLOCKED: {str(report['safety']['US_MICRO_REAL_ORDER_BLOCKED']).lower()}",
        f"Active Kill Switches: {kill_switch['active_count']}",
    ]
    for row in kill_switch["active_rows"][:5]:
        lines.append(f"- {row.get('kill_switch_id')} / {row.get('reason_code') or '-'} / activated_by={row.get('activated_by') or '-'}")
    lines.extend(
        [
            "",
            "[Ranking Summary]",
            f"Top20 Candidates: {ranking['top20_count']}",
            f"BUY or Better: {ranking['buy_or_better_count']}",
            f"EXCLUDE: {ranking['exclude_count']}",
            "",
            "[Pre-Trade Check Summary]",
            f"ALLOW: {precheck['ALLOW']}",
            f"REQUIRE_APPROVAL: {precheck['REQUIRE_APPROVAL']}",
            f"BLOCK: {precheck['BLOCK']}",
            f"ERROR: {precheck['ERROR']}",
            "",
            "Top Block Reasons:",
        ]
    )
    for code, count in precheck["top_block_reasons"][:5]:
        lines.append(f"- {code}: {count}")
    if not precheck["top_block_reasons"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "[Approval Summary]",
            f"PENDING: {approvals['pending']}",
            f"APPROVED: {approvals['approved']}",
            f"REJECTED: {approvals['rejected']}",
            f"EXPIRED: {approvals['expired']}",
            "",
            "[Micro Order Summary]",
        ]
    )
    for key in ["CREATED", "READY_TO_SEND", "ACCEPTED", "ORDER_OPEN", "ORDER_PARTIALLY_FILLED", "ORDER_FILLED", "ORDER_REJECTED", "ORDER_UNKNOWN", "SYNC_ERROR"]:
        lines.append(f"{key}: {orders['counts'].get(key, 0)}")
    lines.extend(
        [
            "",
            "[Fill Summary]",
            f"Fill Count: {fills['fill_count']}",
            f"Total Filled Amount: {fills['total_filled_amount_usd']:.2f} USD",
            "",
            "[Reconciliation Summary]",
            f"MATCH: {recon['match']}",
            f"MISMATCH: {recon['mismatch']}",
            f"CRITICAL: {recon['critical']}",
            f"ERROR: {recon['error']}",
            "",
            "[Daily Risk Usage]",
            f"Total Orders: {risk['total_order_count']}",
            f"Failed Orders: {risk['failed_order_count']}",
            f"Blocked Orders: {risk['blocked_order_count']}",
            f"New BUY Count: {risk['new_buy_count']}",
            "",
            "[Action Required]",
        ]
    )
    if report["actions"]:
        for idx, item in enumerate(report["actions"], start=1):
            lines.append(f"{idx}. [{item['severity']}] {item['message']}")
    else:
        lines.append("None")
    lines.extend(
        [
            "",
            "[Safety]",
            "No orders were created.",
            "No orders were sent.",
            "No live account write occurred.",
        ]
    )
    return "\n".join(lines)


def render_operations_markdown(report: dict[str, object]) -> str:
    lines = [
        "# 미국주식 Micro Live 운영 리포트",
        "",
        "## 1. 개요",
        "",
        f"- Trade Date: `{report['trade_date']}`",
        f"- Account ID: `{report['account_id']}`",
        f"- Report Generated At: `{report['generated_at']}`",
        f"- Execution Mode: `{', '.join(sorted(k for k, v in report['orders']['execution_modes'].items() if v)) or 'N/A'}`",
        f"- Safety Mode: `US_MICRO_REAL_ORDER_BLOCKED={str(report['safety']['US_MICRO_REAL_ORDER_BLOCKED']).lower()}`",
        f"- System Health: `{report['health']['status']}`",
        f"- Reason: `{report['health']['reason']}`",
        "",
        "## 2. Safety Status",
        "",
        f"- Live Order Enabled: `{str(report['safety']['US_LIVE_ORDER_ENABLED']).lower()}`",
        f"- Micro Live Allowed: `{str(report['safety']['US_MICRO_ALLOW_LIVE']).lower()}`",
        f"- Real Order Blocked: `{str(report['safety']['US_MICRO_REAL_ORDER_BLOCKED']).lower()}`",
        f"- Active Kill Switches: `{report['kill_switch']['active_count']}`",
        "",
        "## 3. Ranking Summary",
        "",
        f"- Top20 Candidates: `{report['ranking']['top20_count']}`",
        f"- BUY or Better: `{report['ranking']['buy_or_better_count']}`",
        f"- EXCLUDE: `{report['ranking']['exclude_count']}`",
        "",
        "## 4. Pre-Trade Check Summary",
        "",
        f"- ALLOW: `{report['precheck']['ALLOW']}`",
        f"- REQUIRE_APPROVAL: `{report['precheck']['REQUIRE_APPROVAL']}`",
        f"- BLOCK: `{report['precheck']['BLOCK']}`",
        f"- ERROR: `{report['precheck']['ERROR']}`",
        "",
        "## 5. Block Log Summary",
        "",
        f"- Block Rows: `{report['block_logs']['count']}`",
        "",
        "## 6. Approval Summary",
        "",
        f"- PENDING: `{report['approvals']['pending']}`",
        f"- APPROVED: `{report['approvals']['approved']}`",
        f"- REJECTED: `{report['approvals']['rejected']}`",
        f"- EXPIRED: `{report['approvals']['expired']}`",
        "",
        "## 7. Micro Order Summary",
        "",
    ]
    for key, value in sorted(report["orders"]["counts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 8. Fill Summary",
            "",
            f"- Fill Count: `{report['fills']['fill_count']}`",
            f"- Total Filled Amount USD: `{report['fills']['total_filled_amount_usd']:.2f}`",
            "",
            "## 9. Reconciliation Summary",
            "",
            f"- MATCH: `{report['reconciliation']['match']}`",
            f"- MISMATCH: `{report['reconciliation']['mismatch']}`",
            f"- CRITICAL: `{report['reconciliation']['critical']}`",
            f"- ERROR: `{report['reconciliation']['error']}`",
            "",
            "## 10. Daily Risk Usage",
            "",
            f"- Total Orders: `{report['daily_risk_usage']['total_order_count']}`",
            f"- Failed Orders: `{report['daily_risk_usage']['failed_order_count']}`",
            f"- Blocked Orders: `{report['daily_risk_usage']['blocked_order_count']}`",
            "",
            "## 11. Action Required",
            "",
        ]
    )
    if report["actions"]:
        for item in report["actions"]:
            lines.append(f"- `{item['severity']}` {item['message']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 12. Error / Warning Details",
            "",
        ]
    )
    issue_rows = [item for item in report["actions"] if item["severity"] in {"WARNING", "ERROR", "CRITICAL"}]
    if issue_rows:
        for item in issue_rows:
            lines.append(f"- `{item['reason_code']}`: {item['message']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 13. Operator Checklist",
            "",
            "- Confirm active Kill Switch rows are intentional.",
            "- Review pending or expired approvals.",
            "- Review stale open or partial-filled orders.",
            "- Review reconciliation mismatch or critical rows before any next-phase expansion.",
            "",
            "## 14. 주의사항",
            "",
            "- 이 리포트는 운영 상태 확인용입니다.",
            "- 이 리포트는 주문을 생성하지 않습니다.",
            "- 이 리포트는 주문을 전송하지 않습니다.",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def write_operations_csv(report: dict[str, object], output_dir: str | None = None) -> list[Path]:
    out_dir = _output_dir(output_dir)
    trade_date = report["trade_date"]
    account_id = report["account_id"]
    files: list[Path] = []
    orders_path = out_dir / f"micro_live_orders_{trade_date}_{account_id}.csv"
    _write_csv(
        orders_path,
        report["orders"]["rows"],
        [
            "trade_date",
            "account_id",
            "micro_order_id",
            "symbol",
            "side",
            "execution_mode",
            "order_amount_usd",
            "request_status",
            "broker_order_id",
            "last_broker_status",
            "last_sync_at",
            "reject_reason_code",
            "reject_reason_detail",
        ],
    )
    files.append(orders_path)
    approvals_path = out_dir / f"micro_live_approvals_{trade_date}_{account_id}.csv"
    _write_csv(
        approvals_path,
        report["approvals"]["rows"],
        [
            "approval_id",
            "trade_date",
            "account_id",
            "symbol",
            "side",
            "requested_order_amount_usd",
            "precheck_decision",
            "approval_status",
            "requested_at",
            "approved_at",
            "rejected_at",
            "expires_at",
            "approval_reason",
            "reject_reason",
        ],
    )
    files.append(approvals_path)
    blocks_path = out_dir / f"micro_live_blocks_{trade_date}_{account_id}.csv"
    _write_csv(
        blocks_path,
        report["block_logs"]["rows"],
        [
            "trade_date",
            "account_id",
            "symbol",
            "side",
            "block_reason_code",
            "block_reason_detail",
            "check_stage",
            "severity",
            "created_at",
        ],
    )
    files.append(blocks_path)
    recon_path = out_dir / f"micro_live_reconciliation_{trade_date}_{account_id}.csv"
    _write_csv(
        recon_path,
        report["reconciliation"]["rows"],
        [
            "recon_date",
            "account_id",
            "execution_mode",
            "recon_type",
            "symbol",
            "micro_order_id",
            "broker_order_id",
            "recon_status",
            "severity",
            "reason_code",
            "reason_detail",
            "qty_diff",
            "amount_diff_usd",
            "cash_diff_usd",
            "created_at",
        ],
    )
    files.append(recon_path)
    return files


def write_operations_markdown(report: dict[str, object], output_dir: str | None = None) -> Path:
    out_dir = _output_dir(output_dir)
    path = out_dir / f"micro_live_ops_{report['trade_date']}_{report['account_id']}.md"
    path.write_text(render_operations_markdown(report), encoding="utf-8")
    return path


def maybe_notify_operations_report(report: dict[str, object]) -> bool:
    if not _flag("US_MICRO_OPS_REPORT_NOTIFY_ENABLED", "false"):
        return False
    has_critical = any(item["severity"] == "CRITICAL" for item in report["actions"])
    has_error = any(item["severity"] == "ERROR" for item in report["actions"])
    title = f"US Micro Live Ops {report['health']['status']}"
    message = report["health"]["reason"]
    details = {
        "trade_date": report["trade_date"],
        "account_id": report["account_id"],
        "critical_actions": sum(1 for item in report["actions"] if item["severity"] == "CRITICAL"),
        "error_actions": sum(1 for item in report["actions"] if item["severity"] == "ERROR"),
        "active_kill_switches": report["kill_switch"]["active_count"],
    }
    try:
        if has_critical and _flag("US_MICRO_OPS_NOTIFY_ON_CRITICAL", "true") and notify_critical is not None:
            return bool(notify_critical(title, message, details))
        if has_error and _flag("US_MICRO_OPS_NOTIFY_ON_ERROR", "true") and notify_warning is not None:
            return bool(notify_warning(title, message, details))
    except Exception:
        return False
    return False
