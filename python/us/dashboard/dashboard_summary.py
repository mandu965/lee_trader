from __future__ import annotations

from collections import Counter
from statistics import median

from python.us.dashboard.config import DashboardConfig
from python.us.buy_automation.paper_backtest_summary import build_paper_backtest_summary


LIVE_READINESS_NOTE = "live_ready=true는 자동 실매매 전환을 의미하지 않으며, LIVE 전환은 별도 Phase와 수동 승인 이후에만 가능하다."


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _status_with_level(*, has_error: bool = False, has_warning: bool = False, data_missing: bool = False, not_available: bool = False) -> str:
    if has_error:
        return "ERROR"
    if data_missing:
        return "DATA_MISSING"
    if not_available:
        return "NOT_AVAILABLE"
    if has_warning:
        return "WARNING"
    return "OK"


def _top_counts(counter: Counter[str], limit: int = 5) -> list[dict[str, object]]:
    return [{"reason": key, "count": count} for key, count in counter.most_common(limit)]


def _benchmark_from_readiness(readiness: dict[str, object] | None, key: str) -> float | None:
    if not isinstance(readiness, dict):
        return None
    perf = readiness.get("paper_performance_summary") or {}
    if isinstance(perf, dict):
        return _safe_float(perf.get(key))
    return None


def build_daily_overview(raw_data: dict[str, object]) -> dict[str, object]:
    integrated_report = raw_data.get("integrated_report") if isinstance(raw_data.get("integrated_report"), dict) else {}
    buy_decisions = list(raw_data.get("buy_decisions") or [])
    sell_decisions = list(raw_data.get("sell_decisions") or [])
    conflicts = list(raw_data.get("conflicts") or [])
    paper_buy_orders = list(raw_data.get("paper_buy_orders") or [])
    paper_sell_orders = list(raw_data.get("paper_sell_orders") or [])
    orchestration_logs = list(raw_data.get("orchestration_logs") or [])
    missing_sources = set(raw_data.get("missing_sources") or [])
    warnings = list(raw_data.get("load_warnings") or [])
    errors: list[str] = []

    buy_candidates = _safe_int(integrated_report.get("buy_summary", {}).get("loaded_candidates")) if integrated_report else len(buy_decisions)
    final_buy_allowed = _safe_int(integrated_report.get("buy_summary", {}).get("final_buy_candidates")) if integrated_report else sum(
        1 for row in buy_decisions if str(row.get("decision") or row.get("final_buy_decision") or "").upper() in {"ALLOW", "ALLOWED"}
        or bool(row.get("allowed"))
    )
    sell_signals = _safe_int(integrated_report.get("sell_summary", {}).get("sell")) if integrated_report else sum(
        1 for row in sell_decisions if str(row.get("decision") or row.get("sell_decision") or "").upper() == "SELL"
    )
    hold_count = _safe_int(integrated_report.get("sell_summary", {}).get("hold")) if integrated_report else sum(
        1 for row in sell_decisions if str(row.get("decision") or row.get("sell_decision") or "").upper() == "HOLD"
    )
    review_required_count = _safe_int(integrated_report.get("sell_summary", {}).get("review_required")) if integrated_report else sum(
        1 for row in sell_decisions if str(row.get("decision") or row.get("sell_decision") or "").upper() == "REVIEW_REQUIRED" or bool(row.get("review_required"))
    )
    conflict_blocked_count = _safe_int(integrated_report.get("conflict_summary", {}).get("TOTAL_CONFLICT_BLOCKED")) if integrated_report else sum(
        1 for row in conflicts if _safe_list(row.get("conflict_reasons"))
    )
    fail_safe_triggered = bool(integrated_report.get("fail_safe_triggered")) if integrated_report else any(
        "FAILSAFE" in str(item).upper() for item in warnings
    )
    mode = integrated_report.get("mode") if integrated_report else (orchestration_logs[0].get("mode") if orchestration_logs else "UNKNOWN")
    orchestration_executed = bool(integrated_report or orchestration_logs)
    final_status = "SUCCESS" if integrated_report.get("success", False) or (orchestration_logs and orchestration_logs[0].get("success")) else "ERROR"
    top_warning_reason = warnings[0] if warnings else None
    if review_required_count > 0 and not top_warning_reason:
        top_warning_reason = "REVIEW_REQUIRED_PRESENT"
    if "integrated_daily_report" in missing_sources:
        errors.append("INTEGRATED_DAILY_REPORT_MISSING")

    status = _status_with_level(
        has_error=bool(errors) or (orchestration_executed and final_status == "ERROR"),
        has_warning=review_required_count > 0 or conflict_blocked_count > 0 or bool(warnings),
        data_missing=not orchestration_executed or "integrated_daily_report" in missing_sources,
    )
    if status == "OK" and final_status == "ERROR":
        status = "ERROR"

    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "trade_date": raw_data.get("trade_date"),
        "generated_at": raw_data.get("loaded_at"),
        "mode": mode,
        "final_status": final_status,
        "orchestration_executed": orchestration_executed,
        "buy_candidates": buy_candidates,
        "final_buy_allowed": final_buy_allowed,
        "sell_signals": sell_signals,
        "hold_count": hold_count,
        "review_required_count": review_required_count,
        "conflict_blocked_count": conflict_blocked_count,
        "paper_buy_orders": len(paper_buy_orders),
        "paper_sell_orders": len(paper_sell_orders),
        "fail_safe_triggered": fail_safe_triggered,
        "top_warning_reason": top_warning_reason,
    }


def build_paper_portfolio_summary(raw_data: dict[str, object]) -> dict[str, object]:
    positions = list(raw_data.get("paper_positions") or raw_data.get("paper_position_snapshots") or [])
    sell_orders = list(raw_data.get("paper_sell_orders") or [])
    warnings: list[str] = []
    errors: list[str] = []

    open_positions = [row for row in positions if _safe_float(row.get("remaining_quantity")) not in {None, 0.0}]
    closed_position_count = sum(1 for row in sell_orders if str(row.get("sell_action") or "").upper() == "FULL_SELL")

    total_invested = 0.0
    current_value = 0.0
    unrealized_pnl = 0.0
    largest_position_symbol = None
    largest_position_value = 0.0
    holding_days_values: list[float] = []
    sector_counter: Counter[str] = Counter()

    for row in open_positions:
        qty = _safe_float(row.get("remaining_quantity") or row.get("quantity")) or 0.0
        entry = _safe_float(row.get("avg_entry_price") or row.get("entry_price")) or 0.0
        latest = _safe_float(row.get("latest_price")) or 0.0
        invested = qty * entry
        value = qty * latest
        total_invested += invested
        current_value += value
        unrealized_pnl += _safe_float(row.get("unrealized_pnl")) or (value - invested)
        if value > largest_position_value:
            largest_position_value = value
            largest_position_symbol = row.get("symbol")
        if row.get("holding_days") is not None:
            holding_days_values.append(float(row.get("holding_days")))
        sector = str(row.get("sector") or "").strip()
        if sector:
            sector_counter[sector] += 1

    realized_pnl = sum(_safe_float(row.get("realized_paper_pnl")) or 0.0 for row in sell_orders)
    unrealized_pnl_pct = ((current_value / total_invested) - 1.0) if total_invested > 0 else None
    realized_basis = sum(_safe_float(row.get("sell_amount")) or 0.0 for row in sell_orders)
    realized_pnl_pct = (realized_pnl / realized_basis) if realized_basis > 0 else None
    total_paper_pnl = realized_pnl + unrealized_pnl
    total_paper_pnl_pct = ((current_value + realized_basis) / (total_invested + max(realized_basis - realized_pnl, 0.0)) - 1.0) if (total_invested > 0 or realized_basis > 0) else None
    largest_position_weight = (largest_position_value / current_value) if current_value > 0 else None

    sector_concentration: dict[str, float] | str
    if sector_counter:
        total_sector = sum(sector_counter.values())
        sector_concentration = {key: round(count / total_sector, 6) for key, count in sector_counter.items()}
    else:
        sector_concentration = "DATA_MISSING"
        warnings.append("SECTOR_CONCENTRATION_DATA_MISSING")

    cash_simulation_status = "NOT_AVAILABLE"
    warnings.append("CASH_SIMULATION_NOT_AVAILABLE")

    status = _status_with_level(
        has_warning=bool(warnings),
        data_missing=not bool(open_positions) and not bool(sell_orders),
        not_available=False,
    )

    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "open_position_count": len(open_positions),
        "closed_position_count": closed_position_count,
        "total_invested_amount": round(total_invested, 6),
        "current_paper_value": round(current_value, 6),
        "unrealized_paper_pnl": round(unrealized_pnl, 6),
        "unrealized_paper_pnl_pct": unrealized_pnl_pct,
        "realized_paper_pnl": round(realized_pnl, 6),
        "realized_paper_pnl_pct": realized_pnl_pct,
        "total_paper_pnl": round(total_paper_pnl, 6),
        "total_paper_pnl_pct": total_paper_pnl_pct,
        "average_holding_days": (sum(holding_days_values) / len(holding_days_values)) if holding_days_values else None,
        "largest_position_symbol": largest_position_symbol,
        "largest_position_weight": largest_position_weight,
        "sector_concentration": sector_concentration,
        "cash_simulation_status": cash_simulation_status,
    }


def build_buy_decision_monitor(raw_data: dict[str, object]) -> dict[str, object]:
    rows = list(raw_data.get("buy_decisions") or [])
    block_counter: Counter[str] = Counter()
    conflict_counter: Counter[str] = Counter()
    items: list[dict[str, object]] = []
    risk_blocked_count = 0
    conflict_blocked_count = 0
    final_allowed_count = 0

    for row in rows:
        block_reasons = [str(item) for item in _safe_list(row.get("block_reasons")) if str(item)]
        conflict_reasons = [str(item) for item in _safe_list(row.get("conflict_reasons")) if str(item)]
        allowed = bool(row.get("allowed")) or str(row.get("decision") or row.get("final_buy_decision") or "").upper() in {"ALLOW", "ALLOWED"}
        if allowed:
            final_allowed_count += 1
        if block_reasons and not allowed:
            risk_blocked_count += 1
        if conflict_reasons:
            conflict_blocked_count += 1
        for reason in block_reasons:
            block_counter[reason] += 1
        for reason in conflict_reasons:
            conflict_counter[reason] += 1
        items.append(
            {
                "symbol": row.get("symbol"),
                "rank": row.get("rank") if row.get("rank") is not None else row.get("rank_no"),
                "score": row.get("score") if row.get("score") is not None else row.get("total_score"),
                "probability": row.get("probability"),
                "risk_guard_result": "PASS" if bool(row.get("risk_allowed", allowed and not block_reasons)) else "BLOCK",
                "conflict_guard_result": "PASS" if not conflict_reasons else "BLOCK",
                "final_buy_decision": "ALLOWED" if allowed else ("REVIEW_REQUIRED" if any("REVIEW" in reason.upper() for reason in block_reasons + conflict_reasons) else "BLOCKED"),
                "block_reasons": block_reasons,
                "conflict_reasons": conflict_reasons,
                "allocated_paper_amount": row.get("allocated_amount_usd") if row.get("allocated_amount_usd") is not None else row.get("planned_order_amount_usd"),
                "paper_buy_order_created": allowed,
            }
        )

    status = _status_with_level(data_missing=not rows, has_warning=bool(conflict_blocked_count))
    return {
        "status": status,
        "items": items,
        "warnings": list(raw_data.get("load_warnings") or []),
        "errors": [],
        "total_candidates": len(rows),
        "risk_blocked_count": risk_blocked_count,
        "conflict_blocked_count": conflict_blocked_count,
        "final_allowed_count": final_allowed_count,
        "top_block_reasons": _top_counts(block_counter),
        "top_conflict_reasons": _top_counts(conflict_counter),
    }


def build_sell_decision_monitor(raw_data: dict[str, object]) -> dict[str, object]:
    decisions = list(raw_data.get("sell_decisions") or [])
    sell_orders = {str(item.get("source_sell_decision_id") or ""): item for item in list(raw_data.get("paper_sell_orders") or [])}
    items: list[dict[str, object]] = []
    exit_counter: Counter[str] = Counter()
    hold_count = 0
    sell_count = 0
    partial_sell_count = 0
    review_required_count = 0

    for row in decisions:
        decision = str(row.get("decision") or row.get("sell_decision") or "").upper()
        if decision == "HOLD":
            hold_count += 1
        elif decision == "SELL":
            sell_count += 1
        elif decision == "PARTIAL_SELL":
            partial_sell_count += 1
        elif decision == "REVIEW_REQUIRED":
            review_required_count += 1
        exit_reason = row.get("exit_reason")
        if exit_reason:
            exit_counter[str(exit_reason)] += 1
        latest_price = _safe_float(row.get("latest_price"))
        highest_price = _safe_float(row.get("highest_price_since_entry"))
        drawdown_from_high_pct = None
        if latest_price is not None and highest_price not in {None, 0.0}:
            drawdown_from_high_pct = (latest_price / highest_price) - 1.0
        items.append(
            {
                "symbol": row.get("symbol"),
                "paper_position_id": row.get("paper_position_id"),
                "entry_trade_date": row.get("entry_trade_date"),
                "avg_entry_price": row.get("avg_entry_price"),
                "latest_price": row.get("latest_price"),
                "unrealized_pnl_pct": row.get("unrealized_pnl_pct"),
                "highest_price_since_entry": row.get("highest_price_since_entry"),
                "drawdown_from_high_pct": drawdown_from_high_pct,
                "holding_days": row.get("holding_days"),
                "sell_decision": decision or None,
                "sell_action": row.get("sell_action"),
                "sell_ratio": row.get("sell_ratio"),
                "exit_reason": exit_reason,
                "review_required": bool(row.get("review_required")) or decision == "REVIEW_REQUIRED",
                "applied_rules": row.get("applied_rules") or [],
                "paper_sell_order_created": str(row.get("sell_decision_id") or "") in sell_orders,
            }
        )

    status = _status_with_level(data_missing=not decisions, has_warning=review_required_count > 0)
    return {
        "status": status,
        "items": items,
        "warnings": [],
        "errors": [],
        "loaded_positions": len(decisions),
        "hold_count": hold_count,
        "sell_count": sell_count,
        "partial_sell_count": partial_sell_count,
        "review_required_count": review_required_count,
        "top_exit_reasons": _top_counts(exit_counter),
    }


def build_conflict_guard_monitor(raw_data: dict[str, object]) -> dict[str, object]:
    rows = list(raw_data.get("conflicts") or [])
    counts = Counter()
    items: list[dict[str, object]] = []
    for row in rows:
        reasons = [str(item) for item in _safe_list(row.get("conflict_reasons")) if str(item)]
        for reason in reasons:
            counts[reason] += 1
        symbol = str(row.get("symbol") or "").upper()
        items.append(
            {
                "symbol": symbol,
                "buy_candidate": True,
                "open_position_exists": "OPEN_POSITION_EXISTS" in reasons,
                "sell_signal_exists": "SELL_SIGNAL_EXISTS" in reasons,
                "review_required": "REVIEW_REQUIRED_SYMBOL" in reasons,
                "cooldown_active": "COOLDOWN_ACTIVE" in reasons,
                "duplicate_buy": "DUPLICATE_BUY" in reasons,
                "conflict_reasons": reasons,
                "final_action": "BUY_ALLOWED" if not reasons else "BUY_BLOCKED",
            }
        )
    status = _status_with_level(data_missing=not rows, has_warning=bool(rows))
    return {
        "status": status,
        "items": items,
        "warnings": [],
        "errors": [],
        "conflict_count": sum(1 for row in rows if _safe_list(row.get("conflict_reasons"))),
        "open_position_exists_count": counts.get("OPEN_POSITION_EXISTS", 0),
        "sell_signal_exists_count": counts.get("SELL_SIGNAL_EXISTS", 0),
        "review_required_symbol_count": counts.get("REVIEW_REQUIRED_SYMBOL", 0),
        "cooldown_active_count": counts.get("COOLDOWN_ACTIVE", 0),
        "duplicate_buy_count": counts.get("DUPLICATE_BUY", 0),
        "portfolio_state_inconsistent_count": counts.get("PORTFOLIO_STATE_INCONSISTENT", 0),
    }


def build_paper_performance_monitor(raw_data: dict[str, object], cfg: DashboardConfig) -> dict[str, object]:
    lookback_days = cfg.default_lookback_days
    daily = build_paper_backtest_summary(days=1, benchmark_symbol="SPY")
    weekly = build_paper_backtest_summary(days=5, benchmark_symbol="SPY")
    monthly = build_paper_backtest_summary(days=20, benchmark_symbol="SPY")
    main = build_paper_backtest_summary(days=lookback_days, benchmark_symbol="SPY")
    all_period = build_paper_backtest_summary(days=None, benchmark_symbol="SPY")

    returns = []
    for row in main.get("rows", []):
        value = _safe_float(row.get("unrealized_pnl_pct"))
        if value is not None:
            returns.append(value)
    sample_status = "OK"
    warnings: list[str] = []
    if _safe_int(main.get("paper_order_count")) <= 0:
        sample_status = "DATA_MISSING"
        warnings.append("NO_PAPER_ORDERS")
    elif _safe_int(main.get("paper_order_count")) < 3:
        sample_status = "NOT_ENOUGH_SAMPLE"
        warnings.append("NOT_ENOUGH_SAMPLE")
    elif not raw_data.get("paper_sell_orders"):
        sample_status = "NOT_AVAILABLE"
        warnings.append("REALIZED_PERFORMANCE_NOT_AVAILABLE")

    status = _status_with_level(
        data_missing=sample_status == "DATA_MISSING",
        not_available=sample_status == "NOT_AVAILABLE",
        has_warning=sample_status == "NOT_ENOUGH_SAMPLE",
    )
    return {
        "status": status,
        "warnings": warnings,
        "errors": [],
        "lookback_days": lookback_days,
        "cumulative_paper_return_pct": all_period.get("total_return_pct"),
        "daily_paper_return_pct": daily.get("total_return_pct"),
        "weekly_paper_return_pct": weekly.get("total_return_pct"),
        "monthly_paper_return_pct": monthly.get("total_return_pct"),
        "win_rate": main.get("win_rate"),
        "loss_rate": main.get("loss_rate"),
        "average_trade_return_pct": main.get("avg_return_pct"),
        "median_trade_return_pct": median(returns) if returns else None,
        "best_trade": main.get("best_trade_return_pct"),
        "worst_trade": main.get("worst_trade_return_pct"),
        "max_drawdown_pct": main.get("max_drawdown_pct"),
        "average_holding_days": main.get("avg_holding_days"),
        "trade_count": main.get("paper_order_count"),
        "active_position_count": len([row for row in list(raw_data.get("paper_positions") or []) if _safe_float(row.get("remaining_quantity")) not in {None, 0.0}]),
        "sample_status": sample_status,
    }


def build_benchmark_comparison(raw_data: dict[str, object], cfg: DashboardConfig) -> dict[str, object]:
    readiness = raw_data.get("readiness") if isinstance(raw_data.get("readiness"), dict) else {}
    main = build_paper_backtest_summary(days=cfg.default_lookback_days, benchmark_symbol="SPY", compare_qqq=True)
    paper_return = _benchmark_from_readiness(readiness, "total_return_pct")
    if paper_return is None:
        paper_return = main.get("total_return_pct")
    spy_return = _benchmark_from_readiness(readiness, "benchmark_return_pct")
    if spy_return is None:
        spy_return = main.get("benchmark_return_pct")
    qqq_from_readiness = None
    if isinstance(readiness, dict):
        qqq_from_readiness = ((readiness.get("paper_performance_summary") or {}).get("qqq_comparison") or {}).get("benchmark_return_pct")
    qqq_return = qqq_from_readiness if qqq_from_readiness is not None else ((main.get("qqq_comparison") or {}).get("benchmark_return_pct"))
    excess_spy = _benchmark_from_readiness(readiness, "excess_return_pct")
    if excess_spy is None and paper_return is not None and spy_return is not None:
        excess_spy = paper_return - spy_return
    excess_qqq = ((readiness.get("paper_performance_summary") or {}).get("qqq_comparison") or {}).get("excess_return_pct") if isinstance(readiness, dict) else None
    if excess_qqq is None and paper_return is not None and qqq_return is not None:
        excess_qqq = paper_return - qqq_return
    benchmark_missing = bool(main.get("benchmark_data_missing")) or spy_return is None
    status = _status_with_level(data_missing=benchmark_missing)
    return {
        "status": status,
        "warnings": ["BENCHMARK_DATA_MISSING"] if benchmark_missing else [],
        "errors": [],
        "paper_return_pct": paper_return,
        "spy_return_pct": spy_return,
        "qqq_return_pct": qqq_return,
        "excess_return_vs_spy": excess_spy,
        "excess_return_vs_qqq": excess_qqq,
        "benchmark_win_vs_spy": excess_spy is not None and excess_spy > 0,
        "benchmark_win_vs_qqq": excess_qqq is not None and excess_qqq > 0,
        "rolling_excess_return": excess_spy,
        "benchmark_data_missing": benchmark_missing,
    }


def build_risk_data_quality_monitor(raw_data: dict[str, object], cfg: DashboardConfig) -> dict[str, object]:
    missing_sources = list(raw_data.get("missing_sources") or [])
    buy_rows = list(raw_data.get("buy_decisions") or [])
    sell_rows = list(raw_data.get("sell_decisions") or [])
    orchestration_logs = list(raw_data.get("orchestration_logs") or [])

    price_missing = 0
    benchmark_missing = 0
    financial_missing = 0
    invalid_decision_log = 0
    block_reason_missing = 0
    review_required = 0

    for row in buy_rows:
        reasons = [str(item).upper() for item in _safe_list(row.get("block_reasons"))]
        if not bool(row.get("allowed")) and not reasons and str(row.get("decision") or "").upper() not in {"ALLOW", "ALLOWED"}:
            invalid_decision_log += 1
            block_reason_missing += 1
        price_missing += sum(1 for reason in reasons if "PRICE_DATA_MISSING" in reason)
        benchmark_missing += sum(1 for reason in reasons if "BENCHMARK" in reason and "MISSING" in reason)
        financial_missing += sum(1 for reason in reasons if "FINANCIAL_DATA_MISSING" in reason)

    for row in sell_rows:
        reasons = [str(item.get("reason") or item) for item in _safe_list(row.get("applied_rules"))]
        review_required += 1 if bool(row.get("review_required")) or str(row.get("decision") or "").upper() == "REVIEW_REQUIRED" else 0
        benchmark_missing += sum(1 for reason in reasons if "BENCHMARK_DATA_MISSING" in str(reason).upper())
        price_missing += sum(1 for reason in reasons if "PRICE_DATA_MISSING" in str(reason).upper())

    fail_safe_triggered_count = sum(1 for row in orchestration_logs if row.get("fail_safe_triggered"))
    total_checks = max(len(buy_rows) + len(sell_rows) + len(missing_sources), 1)
    data_missing_count = len(missing_sources) + price_missing + benchmark_missing + financial_missing
    data_missing_rate = min((data_missing_count / total_checks) * 100.0, 100.0) if total_checks > 0 else 100.0

    if data_missing_rate <= cfg.data_missing_warning_pct:
        risk_status = "NORMAL"
    elif data_missing_rate <= cfg.data_missing_critical_pct:
        risk_status = "WARNING"
    else:
        risk_status = "CRITICAL"

    status = "OK" if risk_status == "NORMAL" else ("WARNING" if risk_status == "WARNING" else "ERROR")
    portfolio_state_inconsistent_count = sum(1 for item in missing_sources if "paper_position" in item.lower())
    return {
        "status": status,
        "warnings": list(raw_data.get("load_warnings") or []),
        "errors": [],
        "data_missing_count": data_missing_count,
        "data_missing_rate": round(data_missing_rate, 2),
        "price_data_missing_count": price_missing,
        "benchmark_data_missing_count": benchmark_missing,
        "financial_data_missing_count": financial_missing,
        "invalid_decision_log_count": invalid_decision_log,
        "block_reason_missing_count": block_reason_missing,
        "portfolio_state_inconsistent_count": portfolio_state_inconsistent_count,
        "fail_safe_triggered_count": fail_safe_triggered_count,
        "review_required_count": review_required,
        "risk_status": risk_status,
        "missing_sources": missing_sources,
    }


def build_scheduler_health_monitor(raw_data: dict[str, object]) -> dict[str, object]:
    logs = list(raw_data.get("scheduler_run_logs") or [])
    health_rows = list(raw_data.get("scheduler_health_rows") or [])
    latest = logs[0] if logs else {}
    success_logs = [row for row in logs if row.get("success")]
    health_status = health_rows[0].get("health_status") if health_rows else ("PASS" if latest.get("health_check_passed") else "DATA_MISSING")
    duplicate_run_detected = any("DUPLICATE_RUN_DETECTED" in ",".join(row.get("errors") or []) for row in logs)
    stale_lock_removed = any(bool(row.get("stale_lock_removed")) for row in logs)
    success_rate = (len(success_logs) / len(logs)) if logs else None
    warnings = []
    if not logs:
        warnings.append("SCHEDULER_RUN_LOG_MISSING")
    status = _status_with_level(
        has_error=bool(logs) and not bool(latest.get("success")),
        has_warning=duplicate_run_detected or health_status in {"WARNING", "UNKNOWN"},
        data_missing=not logs,
    )
    return {
        "status": status,
        "warnings": warnings,
        "errors": [],
        "scheduler_run_status": "SUCCESS" if latest.get("success") else ("DATA_MISSING" if not logs else "FAILED"),
        "last_run_at": latest.get("generated_at") or latest.get("created_at"),
        "last_success_at": success_logs[0].get("generated_at") if success_logs else None,
        "duplicate_run_detected": duplicate_run_detected,
        "stale_lock_removed": stale_lock_removed,
        "scheduler_success_rate": success_rate,
        "health_check_status": health_status,
        "report_generated": latest.get("report_generated"),
        "json_report_exists": bool(raw_data.get("integrated_report")),
        "markdown_report_exists": True if raw_data.get("integrated_report") else False,
        "pipeline_should_fail": latest.get("pipeline_should_fail"),
        "warning_count": len(_safe_list(health_rows[0].get("warnings") if health_rows else [])),
        "error_count": len(_safe_list(health_rows[0].get("errors") if health_rows else [])),
    }


def build_live_readiness_monitor(raw_data: dict[str, object]) -> dict[str, object]:
    readiness = raw_data.get("readiness") if isinstance(raw_data.get("readiness"), dict) else None
    if not readiness:
        return {
            "status": "DATA_MISSING",
            "warnings": ["READINESS_REPORT_MISSING"],
            "errors": [],
            "live_ready": False,
            "readiness_score": None,
            "manual_approval_required": True,
            "min_shadow_days_met": None,
            "min_paper_days_met": None,
            "min_paper_orders_met": None,
            "win_rate_met": None,
            "max_drawdown_met": None,
            "excess_return_met": None,
            "data_missing_rate_met": None,
            "scheduler_success_rate_met": None,
            "not_ready_reasons": ["READINESS_REPORT_MISSING"],
            "live_transition_note": LIVE_READINESS_NOTE,
        }

    reasons = [str(item) for item in _safe_list(readiness.get("reasons")) if str(item)]
    policy = readiness.get("promotion_policy") or {}
    ops = readiness.get("operational_stability") or {}
    perf = readiness.get("paper_performance_summary") or {}

    shadow_days = _safe_int(ops.get("shadow_days"))
    paper_days = _safe_int(ops.get("paper_days"))
    paper_orders = _safe_int(perf.get("paper_order_count"))
    win_rate = _safe_float(perf.get("win_rate"))
    max_drawdown = _safe_float(perf.get("max_drawdown_pct"))
    excess_return = _safe_float(perf.get("excess_return_pct"))
    data_missing_rate = _safe_float(ops.get("data_missing_rate"))
    scheduler_success_rate = _safe_float(ops.get("scheduler_success_rate"))

    min_shadow_days_met = shadow_days >= _safe_int(policy.get("min_shadow_days"))
    min_paper_days_met = paper_days >= _safe_int(policy.get("min_paper_days"))
    min_paper_orders_met = paper_orders >= _safe_int(policy.get("min_paper_orders"))
    win_rate_met = win_rate is not None and win_rate >= (_safe_float(policy.get("min_win_rate_pct")) or 0.0)
    max_drawdown_met = max_drawdown is not None and max_drawdown <= (_safe_float(policy.get("max_drawdown_pct")) or 0.0)
    excess_return_met = excess_return is not None and excess_return >= (_safe_float(policy.get("min_excess_return_pct")) or 0.0)
    data_missing_rate_met = data_missing_rate is not None and data_missing_rate <= (_safe_float(policy.get("max_data_missing_rate_pct")) or 0.0)
    scheduler_success_rate_met = scheduler_success_rate is not None and scheduler_success_rate >= (_safe_float(policy.get("min_scheduler_success_rate_pct")) or 0.0)

    status = _status_with_level(has_warning=not readiness.get("live_ready"), data_missing=False)
    return {
        "status": status,
        "warnings": [],
        "errors": [],
        "live_ready": bool(readiness.get("live_ready")),
        "readiness_score": readiness.get("readiness_score"),
        "manual_approval_required": bool(readiness.get("manual_approval_required", True)),
        "min_shadow_days_met": min_shadow_days_met,
        "min_paper_days_met": min_paper_days_met,
        "min_paper_orders_met": min_paper_orders_met,
        "win_rate_met": win_rate_met,
        "max_drawdown_met": max_drawdown_met,
        "excess_return_met": excess_return_met,
        "data_missing_rate_met": data_missing_rate_met,
        "scheduler_success_rate_met": scheduler_success_rate_met,
        "not_ready_reasons": reasons,
        "live_transition_note": LIVE_READINESS_NOTE,
    }
