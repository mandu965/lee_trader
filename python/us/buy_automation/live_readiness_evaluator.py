from __future__ import annotations

from datetime import date, datetime, timezone

from python.us.buy_automation.paper_backtest_summary import (
    build_paper_backtest_summary,
    load_buy_automation_run_logs,
    load_scheduler_job_logs,
)
from python.us.buy_automation.promotion_policy import LivePromotionPolicy, load_live_promotion_policy
from python.us.buy_automation.validation_summary import summarize_validation


def _unique_trade_dates(logs: list[dict[str, object]], *, mode: str | None = None) -> int:
    selected = []
    for row in logs:
        row_mode = str(row.get("mode") or "").upper()
        if mode and row_mode != mode.upper():
            continue
        trade_date = row.get("trade_date")
        if trade_date:
            selected.append(str(trade_date))
    return len(set(selected))


def _aggregate_validation(logs: list[dict[str, object]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for row in logs:
        for candidate in row.get("candidates") or []:
            if isinstance(candidate, dict):
                candidates.append(candidate)
    return summarize_validation(candidates)


def _candidate_data_missing_rate(logs: list[dict[str, object]]) -> float | None:
    candidates = []
    data_missing = 0
    for row in logs:
        for candidate in row.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidates.append(candidate)
            reasons = [str(reason).upper() for reason in (candidate.get("block_reasons") or [])]
            if any("MISSING" in reason for reason in reasons):
                data_missing += 1
    if not candidates:
        return None
    return data_missing / len(candidates)


def _scheduler_success_rate(logs: list[dict[str, object]]) -> float | None:
    enabled_rows = [row for row in logs if row.get("enabled")]
    if not enabled_rows:
        return None
    success_count = sum(1 for row in enabled_rows if row.get("success"))
    return success_count / len(enabled_rows)


def _report_success_rate(logs: list[dict[str, object]]) -> float | None:
    enabled_rows = [row for row in logs if row.get("enabled")]
    if not enabled_rows:
        return None
    success_count = sum(1 for row in enabled_rows if row.get("report_executed"))
    return success_count / len(enabled_rows)


def _live_block_evidence_count(logs: list[dict[str, object]]) -> int:
    return sum(1 for row in logs if str(row.get("error") or "").upper() == "LIVE_DISABLED_IN_SCHEDULER")


def _score_from_reasons(reasons: list[str]) -> int:
    penalties = {
        "BENCHMARK_DATA_MISSING": 20,
        "PAPER_DAYS_BELOW_MINIMUM": 15,
        "PAPER_ORDERS_BELOW_MINIMUM": 15,
        "EXCESS_RETURN_BELOW_THRESHOLD": 15,
        "MAX_DRAWDOWN_TOO_HIGH": 10,
        "WIN_RATE_BELOW_THRESHOLD": 10,
        "DATA_MISSING_RATE_TOO_HIGH": 10,
        "SCHEDULER_SUCCESS_RATE_BELOW_THRESHOLD": 10,
        "SHADOW_DAYS_BELOW_MINIMUM": 10,
        "INVALID_DECISION_LOG_PRESENT": 10,
        "REPORT_SUCCESS_RATE_BELOW_THRESHOLD": 5,
        "FAILSAFE_EVIDENCE_MISSING": 5,
        "LIVE_BLOCK_EVIDENCE_MISSING": 5,
    }
    score = 100
    for reason in reasons:
        score -= penalties.get(reason, 5)
    return max(0, score)


def evaluate_live_readiness(
    *,
    days: int = 60,
    benchmark_symbol: str | None = None,
    input_dir: str | None = None,
    compare_qqq: bool | None = None,
    policy: LivePromotionPolicy | None = None,
    as_of_date: date | None = None,
) -> dict[str, object]:
    readiness_policy = policy or load_live_promotion_policy()
    benchmark = str(benchmark_symbol or readiness_policy.benchmark_symbol).upper()
    compare_with_qqq = readiness_policy.compare_qqq if compare_qqq is None else bool(compare_qqq)

    automation_logs = load_buy_automation_run_logs(input_dir=input_dir)
    scheduler_logs = load_scheduler_job_logs(input_dir=input_dir)
    performance = build_paper_backtest_summary(
        days=days,
        benchmark_symbol=benchmark,
        compare_qqq=compare_with_qqq,
        input_dir=input_dir,
        as_of_date=as_of_date,
    )
    validation = _aggregate_validation(automation_logs)
    shadow_days = _unique_trade_dates(automation_logs, mode="SHADOW")
    paper_days = _unique_trade_dates(automation_logs, mode="PAPER")
    scheduler_success_rate = _scheduler_success_rate(scheduler_logs)
    report_success_rate = _report_success_rate(scheduler_logs)
    data_missing_rate = _candidate_data_missing_rate(automation_logs)
    live_block_evidence_count = _live_block_evidence_count(scheduler_logs)
    fail_safe_observed = int(validation.get("fail_safe_block_count", 0) or 0) > 0

    reasons: list[str] = []
    if shadow_days < readiness_policy.min_shadow_days:
        reasons.append("SHADOW_DAYS_BELOW_MINIMUM")
    if paper_days < readiness_policy.min_paper_days:
        reasons.append("PAPER_DAYS_BELOW_MINIMUM")
    if int(performance.get("paper_order_count", 0) or 0) < readiness_policy.min_paper_orders:
        reasons.append("PAPER_ORDERS_BELOW_MINIMUM")
    if performance.get("benchmark_data_missing"):
        reasons.append("BENCHMARK_DATA_MISSING")
    if readiness_policy.require_positive_excess_return and not performance.get("positive_excess_return"):
        reasons.append("EXCESS_RETURN_BELOW_THRESHOLD")
    elif performance.get("excess_return_pct") is not None and performance.get("excess_return_pct") < readiness_policy.min_excess_return_pct:
        reasons.append("EXCESS_RETURN_BELOW_THRESHOLD")
    if performance.get("max_drawdown_pct") is None or performance.get("max_drawdown_pct") > readiness_policy.max_drawdown_pct:
        reasons.append("MAX_DRAWDOWN_TOO_HIGH")
    if performance.get("win_rate") is None or performance.get("win_rate") < readiness_policy.min_win_rate_pct:
        reasons.append("WIN_RATE_BELOW_THRESHOLD")
    if data_missing_rate is None or data_missing_rate > readiness_policy.max_data_missing_rate_pct:
        reasons.append("DATA_MISSING_RATE_TOO_HIGH")
    if not fail_safe_observed:
        reasons.append("FAILSAFE_EVIDENCE_MISSING")
    if live_block_evidence_count <= 0:
        reasons.append("LIVE_BLOCK_EVIDENCE_MISSING")
    if scheduler_success_rate is None or scheduler_success_rate < readiness_policy.min_scheduler_success_rate_pct:
        reasons.append("SCHEDULER_SUCCESS_RATE_BELOW_THRESHOLD")
    if report_success_rate is None or report_success_rate < readiness_policy.min_scheduler_success_rate_pct:
        reasons.append("REPORT_SUCCESS_RATE_BELOW_THRESHOLD")
    if validation.get("invalid_decision_logs"):
        reasons.append("INVALID_DECISION_LOG_PRESENT")

    unique_reasons = list(dict.fromkeys(reasons))
    readiness_score = _score_from_reasons(unique_reasons)
    live_ready = not unique_reasons and readiness_policy.readiness_enabled

    return {
        "evaluation_generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_period_days": days,
        "benchmark_symbol": benchmark,
        "live_ready": live_ready,
        "readiness_score": readiness_score,
        "decision": "REVIEW_READY" if live_ready else "NOT_READY",
        "reasons": unique_reasons,
        "manual_approval_required": True,
        "promotion_policy": {
            "min_shadow_days": readiness_policy.min_shadow_days,
            "min_paper_days": readiness_policy.min_paper_days,
            "min_paper_orders": readiness_policy.min_paper_orders,
            "min_win_rate_pct": readiness_policy.min_win_rate_pct,
            "max_drawdown_pct": readiness_policy.max_drawdown_pct,
            "max_data_missing_rate_pct": readiness_policy.max_data_missing_rate_pct,
            "min_scheduler_success_rate_pct": readiness_policy.min_scheduler_success_rate_pct,
            "require_positive_excess_return": readiness_policy.require_positive_excess_return,
            "require_manual_approval": readiness_policy.require_manual_approval,
        },
        "operational_stability": {
            "shadow_days": shadow_days,
            "paper_days": paper_days,
            "scheduler_success_rate": scheduler_success_rate,
            "report_success_rate": report_success_rate,
            "invalid_decision_log_count": len(validation.get("invalid_decision_logs") or []),
            "block_reason_missing_count": len(validation.get("invalid_decision_logs") or []),
            "data_missing_rate": data_missing_rate,
            "fail_safe_observed": fail_safe_observed,
            "live_disabled_in_scheduler_count": live_block_evidence_count,
        },
        "paper_performance_summary": performance,
        "validation_summary": validation,
        "live_transition_note": "LIVE automatic promotion is prohibited. Manual approval remains mandatory.",
    }
