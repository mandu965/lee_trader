from __future__ import annotations


def render_console_notification(payload: dict[str, object], *, severity: str) -> str:
    buy = payload.get("buy") if isinstance(payload.get("buy"), dict) else {}
    sell = payload.get("sell") if isinstance(payload.get("sell"), dict) else {}
    risk = payload.get("risk") if isinstance(payload.get("risk"), dict) else {}
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    lines = [
        f"[US Paper Trading Dashboard] {severity}",
        "",
        f"date: {payload.get('trade_date')}",
        f"mode: {payload.get('mode')}",
        f"status: {payload.get('status')}",
        "",
        "BUY:",
        f"- candidates: {buy.get('candidates')}",
        f"- final allowed: {buy.get('final_allowed')}",
        f"- conflict blocked: {buy.get('conflict_blocked')}",
        "",
        "SELL:",
        f"- positions: {sell.get('positions')}",
        f"- sell signals: {sell.get('sell_signals')}",
        f"- review required: {sell.get('review_required')}",
        "",
        "Risk/Data:",
        f"- data missing rate: {risk.get('data_missing_rate')}%",
        f"- fail-safe: {'YES' if risk.get('fail_safe_triggered') else 'NO'}",
        f"- top warning: {risk.get('top_warning_reason')}",
        "",
        "Health:",
        f"- scheduler: {health.get('scheduler_status')}",
        f"- dashboard: {health.get('dashboard_status')}",
        "",
    ]
    if readiness:
        lines.extend(
            [
                "Readiness:",
                f"- live_ready: {str(bool(readiness.get('live_ready'))).lower()}",
                f"- readiness_score: {readiness.get('readiness_score')}",
                f"- manual approval required: {str(bool(readiness.get('manual_approval_required'))).lower()}",
                "",
            ]
        )
    lines.extend(["Notice:", str(payload.get("notice") or "Paper Trading only. No live orders were executed.")])
    return "\n".join(lines)


def run_console_adapter(payload: dict[str, object], *, severity: str, emit_console: bool = True) -> dict[str, object]:
    text = render_console_notification(payload, severity=severity)
    if emit_console:
        print(text)
    return {"channel": "CONSOLE", "status": "SUCCESS", "text": text}
