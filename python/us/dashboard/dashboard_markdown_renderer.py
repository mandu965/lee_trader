from __future__ import annotations


PAPER_NOTICE = (
    "This report is based on Paper Trading data only. It does not represent real account holdings, "
    "real account PnL, or live trading execution."
)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "not_available"
    return str(value)


def _add_key_values(lines: list[str], mapping: dict[str, object], keys: list[str]) -> None:
    for key in keys:
        lines.append(f"- `{key}`: `{_fmt(mapping.get(key))}`")


def _add_table(lines: list[str], headers: list[str], rows: list[dict[str, object]], keys: list[str], limit: int = 10) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows[:limit]:
        values = [_fmt(row.get(key)) for key in keys]
        lines.append("| " + " | ".join(values) + " |")


def render_dashboard_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# US Paper Trading Dashboard",
        "",
        PAPER_NOTICE,
        "",
        "## 1. Summary",
    ]
    meta = payload.get("meta") or {}
    _add_key_values(lines, meta, ["trade_date", "generated_at", "mode", "paper_trading_only", "live_trading_enabled"])

    lines.extend(["", "## 2. Daily Overview"])
    _add_key_values(
        lines,
        payload.get("daily_overview") or {},
        [
            "status",
            "final_status",
            "orchestration_executed",
            "buy_candidates",
            "final_buy_allowed",
            "sell_signals",
            "hold_count",
            "review_required_count",
            "conflict_blocked_count",
            "paper_buy_orders",
            "paper_sell_orders",
            "fail_safe_triggered",
            "top_warning_reason",
        ],
    )

    lines.extend(["", "## 3. Paper Portfolio"])
    _add_key_values(
        lines,
        payload.get("paper_portfolio_summary") or {},
        [
            "status",
            "open_position_count",
            "closed_position_count",
            "total_invested_amount",
            "current_paper_value",
            "unrealized_paper_pnl",
            "unrealized_paper_pnl_pct",
            "realized_paper_pnl",
            "realized_paper_pnl_pct",
            "total_paper_pnl",
            "total_paper_pnl_pct",
            "average_holding_days",
            "largest_position_symbol",
            "largest_position_weight",
            "cash_simulation_status",
        ],
    )

    lines.extend(["", "## 4. BUY Decision Monitor"])
    buy_monitor = payload.get("buy_decision_monitor") or {}
    _add_key_values(lines, buy_monitor, ["status", "total_candidates", "risk_blocked_count", "conflict_blocked_count", "final_allowed_count"])
    if buy_monitor.get("items"):
        _add_table(
            lines,
            ["Symbol", "Rank", "Score", "Decision", "Block Reasons"],
            buy_monitor["items"],
            ["symbol", "rank", "score", "final_buy_decision", "block_reasons"],
        )

    lines.extend(["", "## 5. SELL Decision Monitor"])
    sell_monitor = payload.get("sell_decision_monitor") or {}
    _add_key_values(lines, sell_monitor, ["status", "loaded_positions", "hold_count", "sell_count", "partial_sell_count", "review_required_count"])
    if sell_monitor.get("items"):
        _add_table(
            lines,
            ["Symbol", "Decision", "PnL Pct", "Holding Days", "Exit Reason"],
            sell_monitor["items"],
            ["symbol", "sell_decision", "unrealized_pnl_pct", "holding_days", "exit_reason"],
        )

    lines.extend(["", "## 6. Conflict Guard Monitor"])
    conflict_monitor = payload.get("conflict_guard_monitor") or {}
    _add_key_values(
        lines,
        conflict_monitor,
        [
            "status",
            "conflict_count",
            "open_position_exists_count",
            "sell_signal_exists_count",
            "review_required_symbol_count",
            "cooldown_active_count",
            "duplicate_buy_count",
        ],
    )
    if conflict_monitor.get("items"):
        _add_table(
            lines,
            ["Symbol", "Final Action", "Conflict Reasons"],
            conflict_monitor["items"],
            ["symbol", "final_action", "conflict_reasons"],
        )

    lines.extend(["", "## 7. Paper Performance"])
    _add_key_values(
        lines,
        payload.get("paper_performance_monitor") or {},
        [
            "status",
            "lookback_days",
            "cumulative_paper_return_pct",
            "daily_paper_return_pct",
            "weekly_paper_return_pct",
            "monthly_paper_return_pct",
            "win_rate",
            "loss_rate",
            "average_trade_return_pct",
            "median_trade_return_pct",
            "best_trade",
            "worst_trade",
            "max_drawdown_pct",
            "trade_count",
            "active_position_count",
            "sample_status",
        ],
    )

    lines.extend(["", "## 8. Benchmark Comparison"])
    _add_key_values(
        lines,
        payload.get("benchmark_comparison") or {},
        [
            "status",
            "paper_return_pct",
            "spy_return_pct",
            "qqq_return_pct",
            "excess_return_vs_spy",
            "excess_return_vs_qqq",
            "benchmark_win_vs_spy",
            "benchmark_win_vs_qqq",
            "benchmark_data_missing",
        ],
    )

    lines.extend(["", "## 9. Risk / Data Quality"])
    _add_key_values(
        lines,
        payload.get("risk_data_quality_monitor") or {},
        [
            "status",
            "data_missing_count",
            "data_missing_rate",
            "price_data_missing_count",
            "benchmark_data_missing_count",
            "financial_data_missing_count",
            "invalid_decision_log_count",
            "block_reason_missing_count",
            "portfolio_state_inconsistent_count",
            "fail_safe_triggered_count",
            "review_required_count",
            "risk_status",
        ],
    )

    lines.extend(["", "## 10. Scheduler / Health Check"])
    _add_key_values(
        lines,
        payload.get("scheduler_health_monitor") or {},
        [
            "status",
            "scheduler_run_status",
            "last_run_at",
            "last_success_at",
            "duplicate_run_detected",
            "stale_lock_removed",
            "scheduler_success_rate",
            "health_check_status",
            "report_generated",
            "json_report_exists",
            "markdown_report_exists",
            "pipeline_should_fail",
            "warning_count",
            "error_count",
        ],
    )

    lines.extend(["", "## 11. LIVE Readiness"])
    readiness = payload.get("live_readiness_monitor") or {}
    _add_key_values(
        lines,
        readiness,
        [
            "status",
            "live_ready",
            "readiness_score",
            "manual_approval_required",
            "min_shadow_days_met",
            "min_paper_days_met",
            "min_paper_orders_met",
            "win_rate_met",
            "max_drawdown_met",
            "excess_return_met",
            "data_missing_rate_met",
            "scheduler_success_rate_met",
            "not_ready_reasons",
        ],
    )
    lines.append(f"- `live_transition_note`: `{_fmt(readiness.get('live_transition_note'))}`")

    lines.extend(["", "## 12. Warnings / Errors"])
    warnings = payload.get("warnings") or []
    errors = payload.get("errors") or []
    lines.append("### Warnings")
    if warnings:
        for item in warnings[:20]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### Errors")
    if errors:
        for item in errors[:20]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")

    return "\n".join(lines)
