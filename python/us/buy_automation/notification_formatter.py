from __future__ import annotations


def format_notification_summary(report: dict[str, object]) -> str:
    validation = report.get("validation_summary", {})
    block_counts = validation.get("block_counts", {}) if isinstance(validation, dict) else {}
    top_block_reason = next(iter(block_counts.keys()), "NONE")
    fail_safe = "YES" if int(validation.get("fail_safe_block_count", 0) or 0) > 0 else "NO"
    return "\n".join(
        [
            "[US BUY Automation Summary]",
            f"date: {report.get('trade_date') or '-'}",
            f"mode: {report.get('mode') or '-'}",
            f"candidates: {report.get('loaded_candidates', 0)}",
            f"allowed: {report.get('allowed_candidates', 0)}",
            f"blocked: {report.get('blocked_candidates', 0)}",
            f"paper_orders: {len(report.get('paper_orders', []))}",
            f"top_block_reason: {top_block_reason}",
            f"fail_safe: {fail_safe}",
        ]
    )


def format_notification_detail(report: dict[str, object]) -> str:
    allowed_rows = [row for row in report.get("candidates", []) if row.get("allowed")]
    blocked_rows = [row for row in report.get("candidates", []) if not row.get("allowed")]
    paper_rows = report.get("paper_performance", {}).get("rows", []) if isinstance(report.get("paper_performance"), dict) else []
    validation = report.get("validation_summary", {})
    lines = [
        "[US BUY Automation Detail]",
        f"date: {report.get('trade_date') or '-'}",
        f"mode: {report.get('mode') or '-'}",
        f"live_readiness: NOT_EVALUATED",
        "",
        "[Allowed Candidates]",
    ]
    if not allowed_rows:
        lines.append("- none")
    else:
        for row in allowed_rows:
            lines.append(f"- {row.get('symbol')} rank={row.get('rank')} score={row.get('score')} amount={row.get('allocated_amount_usd')}")
    lines.extend(["", "[Blocked Candidates]"])
    if not blocked_rows:
        lines.append("- none")
    else:
        for row in blocked_rows:
            lines.append(f"- {row.get('symbol')}: {', '.join(row.get('block_reasons') or ['UNKNOWN'])}")
    lines.extend(["", "[Paper Orders]"])
    if not paper_rows:
        lines.append("- none")
    else:
        for row in paper_rows:
            lines.append(
                f"- {row.get('symbol')} pnl_pct={row.get('unrealized_pnl_pct')} benchmark={row.get('benchmark_return_pct')} status={row.get('status')}"
            )
    lines.extend(
        [
            "",
            f"data_missing_symbols: {', '.join(validation.get('data_missing_symbols', [])) or 'none'}",
            f"fail_safe_block_count: {validation.get('fail_safe_block_count', 0)}",
            "live_transition: NO",
        ]
    )
    return "\n".join(lines)
