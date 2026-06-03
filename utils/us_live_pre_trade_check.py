from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import json

from python.us.us_db import (
    fetch_latest_daily_feature_snapshots,
    fetch_market_regime_rows_between,
    fetch_meta_us_universe_rows,
    fetch_mixed_price_rows_for_tickers_between,
    fetch_rank_component_rows_between,
    fetch_us_live_daily_risk_usage_rows,
    insert_us_live_order_block_log_rows,
)
from utils.us_live_kill_switch import check_kill_switch_for_order_candidate
from utils.us_live_risk_policy import load_us_live_risk_policy


@dataclass(frozen=True)
class UsLiveOrderCandidate:
    trade_date: str
    account_id: str
    policy_id: str
    symbol: str
    side: str
    requested_order_amount_usd: float | None
    requested_qty: float | None
    requested_order_type: str
    requested_limit_price: float | None
    candidate_source: str
    strategy_name: str | None
    rank_no: int | None
    recommend_grade: str | None
    total_score: float | None
    reason: str | None


@dataclass(frozen=True)
class CheckStageResult:
    stage: str
    status: str
    reason_codes: list[str] = field(default_factory=list)
    reason_details: list[str] = field(default_factory=list)
    severity: str = "INFO"


@dataclass(frozen=True)
class UsLivePreTradeCheckResult:
    decision: str
    symbol: str
    side: str
    reason_codes: list[str]
    reason_details: list[str]
    severity: str
    check_results: dict[str, str]
    requires_manual_approval: bool
    blocked: bool
    created_at: str


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_trade_date(value: str) -> date:
    return date.fromisoformat(str(value).strip())


def _stage(stage: str, status: str, code: str | None = None, detail: str | None = None, *, severity: str = "INFO") -> CheckStageResult:
    return CheckStageResult(
        stage=stage,
        status=status,
        reason_codes=[code] if code else [],
        reason_details=[detail] if detail else [],
        severity=severity,
    )


def _merge_stage_results(stage_results: list[CheckStageResult]) -> UsLivePreTradeCheckResult:
    precedence = {"ERROR": 4, "BLOCK": 3, "REQUIRE_APPROVAL": 2, "WARNING": 1, "PASS": 0}
    decision = "ALLOW"
    severity = "INFO"
    reason_codes: list[str] = []
    reason_details: list[str] = []
    check_results: dict[str, str] = {}
    for item in stage_results:
        check_results[item.stage] = item.status
        reason_codes.extend(item.reason_codes)
        reason_details.extend(item.reason_details)
        if precedence.get(item.status, 0) > precedence.get(decision, 0):
            decision = item.status if item.status in {"ERROR", "BLOCK", "REQUIRE_APPROVAL"} else decision
        if item.severity in {"CRITICAL", "ERROR"}:
            severity = item.severity
        elif item.severity == "WARNING" and severity == "INFO":
            severity = "WARNING"
    if decision == "ALLOW" and any(item.status == "REQUIRE_APPROVAL" for item in stage_results):
        decision = "REQUIRE_APPROVAL"
    if decision == "ALLOW" and any(item.status == "BLOCK" for item in stage_results):
        decision = "BLOCK"
    if any(item.status == "ERROR" for item in stage_results):
        decision = "ERROR"
        if severity == "INFO":
            severity = "ERROR"
    blocked = decision in {"BLOCK", "ERROR"}
    requires_manual_approval = decision == "REQUIRE_APPROVAL" or "manual_approval_required" in reason_codes
    return UsLivePreTradeCheckResult(
        decision=decision,
        symbol="",
        side="",
        reason_codes=reason_codes,
        reason_details=reason_details,
        severity=severity,
        check_results=check_results,
        requires_manual_approval=requires_manual_approval,
        blocked=blocked,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _is_truthy(value: object) -> bool:
    return bool(value)


def _is_tech_heavy(sector_text: str | None) -> bool:
    sector = str(sector_text or "").strip().lower()
    return "tech" in sector or "semiconductor" in sector or "communication" in sector


def _candidate_dict(candidate: UsLiveOrderCandidate) -> dict[str, object]:
    return {
        "trade_date": candidate.trade_date,
        "account_id": candidate.account_id,
        "policy_id": candidate.policy_id,
        "symbol": candidate.symbol.upper(),
        "side": candidate.side.upper(),
        "requested_order_amount_usd": candidate.requested_order_amount_usd,
        "requested_qty": candidate.requested_qty,
        "requested_order_type": candidate.requested_order_type.upper(),
        "requested_limit_price": candidate.requested_limit_price,
        "candidate_source": candidate.candidate_source,
        "strategy_name": candidate.strategy_name,
        "rank_no": candidate.rank_no,
        "recommend_grade": candidate.recommend_grade,
        "total_score": candidate.total_score,
        "reason": candidate.reason,
    }


def _load_context(candidate: UsLiveOrderCandidate, policy: dict[str, object]) -> dict[str, object]:
    symbol = candidate.symbol.upper()
    trade_date = _parse_trade_date(candidate.trade_date)
    rank_rows = fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source="rule_v1")
    rank_map = {str(row.get("symbol") or "").upper(): row for row in rank_rows}
    universe_rows = fetch_meta_us_universe_rows()
    universe_map = {str(row.get("symbol") or "").upper(): row for row in universe_rows}
    daily_feature = fetch_latest_daily_feature_snapshots([symbol], trade_date=trade_date).get(symbol)
    regime_rows = fetch_market_regime_rows_between(start_date=trade_date, end_date=trade_date)
    regime_row = regime_rows[0] if regime_rows else None
    usage_rows = fetch_us_live_daily_risk_usage_rows(
        trade_date=trade_date,
        policy_id=str(policy.get("policy_id") or candidate.policy_id),
        account_id=candidate.account_id,
    )
    usage_row = usage_rows[0] if usage_rows else None
    price_rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=[symbol, "SPY", "QQQ"],
        start_date=trade_date - timedelta(days=10),
        end_date=trade_date,
    )
    grouped_prices: dict[str, list[dict[str, object]]] = {}
    for row in price_rows:
        ticker = str(row.get("ticker") or "").upper()
        grouped_prices.setdefault(ticker, []).append(row)
    for ticker in grouped_prices:
        grouped_prices[ticker].sort(key=lambda item: item.get("trade_date") or date.min)
    return {
        "trade_date": trade_date,
        "rank_row": rank_map.get(symbol),
        "universe_row": universe_map.get(symbol),
        "daily_feature": daily_feature,
        "regime_row": regime_row,
        "usage_row": usage_row,
        "grouped_prices": grouped_prices,
    }


def check_system_flags(candidate: UsLiveOrderCandidate, policy: dict[str, object]) -> CheckStageResult:
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    reason_codes: list[str] = []
    reason_details: list[str] = []
    if not _is_truthy(safety.get("real_order_blocked", True)):
        return CheckStageResult("SYSTEM_FLAG", "ERROR", ["real_order_blocked_check_failed"], ["US_LIVE_REAL_ORDER_BLOCKED must remain true."], "CRITICAL")
    if not _is_truthy(safety.get("live_trading_enabled", False)):
        reason_codes.append("live_disabled")
        reason_details.append("US_LIVE_TRADING_ENABLED=false")
    if not _is_truthy(safety.get("live_order_enabled", False)):
        reason_codes.append("order_disabled")
        reason_details.append("US_LIVE_ORDER_ENABLED=false")
    side = candidate.side.upper()
    if side == "BUY" and not _is_truthy(safety.get("buy_enabled", False)):
        reason_codes.append("buy_disabled")
        reason_details.append("US_LIVE_BUY_ENABLED=false")
    if side == "SELL" and not _is_truthy(safety.get("sell_enabled", False)):
        reason_codes.append("sell_disabled")
        reason_details.append("US_LIVE_SELL_ENABLED=false")
    if reason_codes:
        return CheckStageResult("SYSTEM_FLAG", "BLOCK", reason_codes, reason_details, "ERROR")
    return _stage("SYSTEM_FLAG", "PASS")


def check_kill_switch(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    matched = check_kill_switch_for_order_candidate(candidate)
    if matched["active"]:
        return CheckStageResult(
            "KILL_SWITCH",
            "BLOCK",
            list(matched["reason_codes"]),
            list(matched["reason_details"]) or ["kill switch active"],
            "CRITICAL",
        )
    return _stage("KILL_SWITCH", "PASS")


def check_ranking(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    rank_row = context.get("rank_row")
    strategy = policy.get("strategy", {}) if isinstance(policy.get("strategy"), dict) else {}
    buy_grades = {str(item).upper() for item in strategy.get("buy_grades", ["STRONG_BUY", "BUY"])}
    sell_grades = {str(item).upper() for item in strategy.get("sell_grades", ["HOLD", "EXCLUDE"])}
    max_rank_no = int(strategy.get("max_rank_no", 20) or 20)
    side = candidate.side.upper()
    if not isinstance(rank_row, dict):
        if side == "BUY":
            return _stage("RANKING", "BLOCK", "rank_data_missing", "Ranking row is missing for BUY candidate.", severity="ERROR")
        return _stage("RANKING", "REQUIRE_APPROVAL", "rank_data_missing", "Ranking row is missing for SELL candidate.", severity="WARNING")
    grade = str(rank_row.get("recommend_grade") or "").upper()
    rank_no = _safe_int(rank_row.get("rank_no"))
    total_score = _safe_float(rank_row.get("total_score"))
    data_status = str(rank_row.get("data_status") or "").upper()
    exclude_reason = str(rank_row.get("exclude_reason") or "").strip()
    if side == "BUY":
        if grade not in buy_grades:
            return _stage("RANKING", "BLOCK", "grade_not_allowed", f"BUY grade is not allowed: {grade}", severity="ERROR")
        if rank_no is None or rank_no > max_rank_no:
            return _stage("RANKING", "BLOCK", "rank_out_of_range", f"Rank {rank_no} is outside Top{max_rank_no}.", severity="ERROR")
        if total_score is None:
            return _stage("RANKING", "BLOCK", "total_score_missing", "total_score is missing.", severity="ERROR")
        if data_status not in {"OK", "PARTIAL_DATA"}:
            return _stage("RANKING", "BLOCK", "data_status_invalid", f"Ranking data_status is {data_status}.", severity="ERROR")
        if grade == "EXCLUDE":
            return _stage("RANKING", "BLOCK", "grade_not_allowed", "recommend_grade=EXCLUDE", severity="ERROR")
        if exclude_reason:
            return _stage("RANKING", "BLOCK", "exclude_reason_exists", exclude_reason, severity="ERROR")
        return _stage("RANKING", "PASS")
    if data_status == "ERROR":
        return _stage("RANKING", "REQUIRE_APPROVAL", "data_status_invalid", "SELL candidate has ranking data_status=ERROR.", severity="WARNING")
    if exclude_reason:
        return _stage("RANKING", "PASS", "exclude_reason_exists", f"SELL candidate reason: {exclude_reason}", severity="INFO")
    if grade in sell_grades:
        return _stage("RANKING", "PASS")
    if rank_no is not None and rank_no > max_rank_no:
        return _stage("RANKING", "PASS")
    return _stage("RANKING", "REQUIRE_APPROVAL", "approval_required_for_sell", "SELL candidate is not a clear rank-exit or downgrade case.", severity="WARNING")


def check_instrument(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    universe_row = context.get("universe_row")
    instrument_policy = policy.get("instrument", {}) if isinstance(policy.get("instrument"), dict) else {}
    if not isinstance(universe_row, dict):
        return _stage("INSTRUMENT", "BLOCK", "symbol_not_in_universe", "Symbol is not present in meta.us_stock_universe.", severity="ERROR")
    if not bool(universe_row.get("is_active")):
        return _stage("INSTRUMENT", "BLOCK", "inactive_universe", "Universe row is inactive.", severity="ERROR")
    if bool(universe_row.get("is_leveraged")) and bool(instrument_policy.get("block_leveraged_etf", True)):
        return _stage("INSTRUMENT", "BLOCK", "leveraged_etf_blocked", "Leveraged ETF is blocked.", severity="ERROR")
    if bool(universe_row.get("is_inverse")) and bool(instrument_policy.get("block_inverse_etf", True)):
        return _stage("INSTRUMENT", "BLOCK", "inverse_etf_blocked", "Inverse ETF is blocked.", severity="ERROR")
    if bool(universe_row.get("is_etf")) and not bool(instrument_policy.get("allow_etf", True)):
        return _stage("INSTRUMENT", "BLOCK", "etf_not_allowed", "ETF candidates are disabled by policy.", severity="ERROR")
    currency = str(universe_row.get("currency") or "USD").upper()
    if currency and currency != "USD":
        return _stage("INSTRUMENT", "BLOCK", "unsupported_currency", f"Unsupported currency: {currency}", severity="ERROR")
    return _stage("INSTRUMENT", "PASS")


def check_price_and_volatility(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    trade_date = context["trade_date"]
    grouped_prices = context.get("grouped_prices") if isinstance(context.get("grouped_prices"), dict) else {}
    symbol = candidate.symbol.upper()
    rows = list(grouped_prices.get(symbol) or [])
    if not rows:
        return _stage("PRICE", "BLOCK", "price_missing", "Price history is missing.", severity="ERROR")
    latest = rows[-1]
    latest_price = _safe_float(latest.get("close_price"))
    if latest_price is None:
        latest_price = _safe_float(latest.get("adj_close_price"))
    if latest_price is None:
        latest_price = _safe_float(latest.get("close"))
    if latest_price is None or latest_price <= 0:
        return _stage("PRICE", "BLOCK", "invalid_price", "Latest price is missing or invalid.", severity="ERROR")
    prev_close = None
    if len(rows) >= 2:
        prev = rows[-2]
        prev_close = _safe_float(prev.get("close_price")) or _safe_float(prev.get("adj_close_price")) or _safe_float(prev.get("close"))
    gap_pct = None if prev_close in {None, 0} else (latest_price - prev_close) / prev_close
    daily_feature = context.get("daily_feature") if isinstance(context.get("daily_feature"), dict) else {}
    vol20 = _safe_float((daily_feature or {}).get("volatility_20d"))
    market_policy = policy.get("market", {}) if isinstance(policy.get("market"), dict) else {}
    side = candidate.side.upper()
    if side == "BUY" and gap_pct is not None:
        if gap_pct > float(market_policy.get("block_buy_on_symbol_gap_up_pct", 0.05) or 0.05):
            return _stage("PRICE", "BLOCK", "gap_up_blocked", f"Gap up {gap_pct:.2%} exceeds policy.", severity="ERROR")
        if gap_pct < float(market_policy.get("block_buy_on_symbol_gap_down_pct", -0.05) or -0.05):
            return _stage("PRICE", "BLOCK", "gap_down_blocked", f"Gap down {gap_pct:.2%} exceeds policy.", severity="ERROR")
    if vol20 is not None and vol20 > float(market_policy.get("max_symbol_volatility_20d", 0.05) or 0.05):
        status = "BLOCK" if side == "BUY" else "REQUIRE_APPROVAL"
        return _stage("PRICE", status, "volatility_too_high", f"20d volatility {vol20:.4f} exceeds policy.", severity="WARNING" if status == "REQUIRE_APPROVAL" else "ERROR")
    return _stage("PRICE", "PASS")


def check_market_regime(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    regime_row = context.get("regime_row")
    side = candidate.side.upper()
    if side != "BUY":
        return _stage("MARKET", "PASS")
    if not isinstance(regime_row, dict):
        return _stage("MARKET", "REQUIRE_APPROVAL", "market_regime_unknown", "Market regime row is missing.", severity="WARNING")
    market_policy = policy.get("market", {}) if isinstance(policy.get("market"), dict) else {}
    market_regime = str(regime_row.get("market_regime") or "UNKNOWN").upper()
    spy_ret = _safe_float(regime_row.get("spy_daily_ret_1d"))
    qqq_ret = _safe_float(regime_row.get("qqq_daily_ret_1d"))
    if bool(market_policy.get("block_bear_high_vol_regime", True)) and market_regime == "BEAR_HIGH_VOL":
        return _stage("MARKET", "BLOCK", "bear_high_vol_regime_blocked", "BEAR_HIGH_VOL regime blocks new BUY.", severity="ERROR")
    if spy_ret is not None and spy_ret <= float(market_policy.get("block_buy_on_spy_drop_pct", -0.02) or -0.02):
        return _stage("MARKET", "BLOCK", "spy_drop_blocked", f"SPY daily return {spy_ret:.2%} exceeded downside block.", severity="ERROR")
    sector_text = ""
    universe_row = context.get("universe_row")
    if isinstance(universe_row, dict):
        sector_text = str(universe_row.get("sector") or "")
    if _is_tech_heavy(sector_text) and qqq_ret is not None and qqq_ret <= float(market_policy.get("block_buy_on_qqq_drop_pct", -0.025) or -0.025):
        return _stage("MARKET", "BLOCK", "qqq_drop_blocked", f"QQQ daily return {qqq_ret:.2%} exceeded downside block.", severity="ERROR")
    return _stage("MARKET", "PASS")


def check_daily_limits(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    usage_row = context.get("usage_row")
    if not isinstance(usage_row, dict):
        return _stage("DAILY_LIMIT", "REQUIRE_APPROVAL", "daily_risk_usage_missing", "Daily risk-usage row is missing.", severity="WARNING")
    order_policy = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    requested_amount = _safe_float(candidate.requested_order_amount_usd) or 0.0
    side = candidate.side.upper()
    total_order_count = _safe_int(usage_row.get("total_order_count")) or 0
    if total_order_count >= int(order_policy.get("max_daily_order_count", 3) or 3):
        return _stage("DAILY_LIMIT", "BLOCK", "daily_order_count_exceeded", "Daily total order count would exceed policy.", severity="ERROR")
    failed_order_count = _safe_int(usage_row.get("failed_order_count")) or 0
    if failed_order_count >= int(order_policy.get("max_daily_order_failures", 3) or 3):
        return _stage("DAILY_LIMIT", "BLOCK", "daily_order_failure_limit_exceeded", "Daily order failure threshold exceeded.", severity="ERROR")
    if side == "BUY":
        current_buy_amount = _safe_float(usage_row.get("buy_amount_usd")) or 0.0
        if current_buy_amount + requested_amount > float(order_policy.get("max_daily_buy_amount_usd", 100) or 100):
            return _stage("DAILY_LIMIT", "BLOCK", "daily_buy_amount_exceeded", "Daily BUY amount would exceed policy.", severity="ERROR")
        new_buy_count = _safe_int(usage_row.get("new_buy_count")) or 0
        if new_buy_count >= int(order_policy.get("max_daily_new_buys", 1) or 1):
            return _stage("DAILY_LIMIT", "BLOCK", "daily_new_buy_count_exceeded", "Daily new BUY count would exceed policy.", severity="ERROR")
    if side == "SELL":
        current_sell_amount = _safe_float(usage_row.get("sell_amount_usd")) or 0.0
        if current_sell_amount + requested_amount > float(order_policy.get("max_daily_sell_amount_usd", 500) or 500):
            return _stage("DAILY_LIMIT", "BLOCK", "daily_sell_amount_exceeded", "Daily SELL amount would exceed policy.", severity="ERROR")
    return _stage("DAILY_LIMIT", "PASS")


def check_position_limits(candidate: UsLiveOrderCandidate, policy: dict[str, object]) -> CheckStageResult:
    order_policy = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    requested_amount = _safe_float(candidate.requested_order_amount_usd)
    if requested_amount is None:
        return _stage("POSITION", "REQUIRE_APPROVAL", "account_snapshot_missing", "Requested order amount is missing and no live account snapshot exists.", severity="WARNING")
    if requested_amount > float(order_policy.get("max_order_amount_usd", 50) or 50):
        return _stage("POSITION", "BLOCK", "max_order_amount_exceeded", "Requested order amount exceeds policy.", severity="ERROR")
    if requested_amount < float(order_policy.get("min_order_amount_usd", 10) or 10):
        return _stage("POSITION", "BLOCK", "min_order_amount_not_met", "Requested order amount is below policy minimum.", severity="ERROR")
    return _stage("POSITION", "REQUIRE_APPROVAL", "account_snapshot_missing", "Live account snapshot integration is not implemented yet.", severity="WARNING")


def check_sector_limits(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    universe_row = context.get("universe_row")
    sector = str((universe_row or {}).get("sector") or "").strip() if isinstance(universe_row, dict) else ""
    if not sector:
        return _stage("SECTOR", "REQUIRE_APPROVAL", "sector_data_missing", "Sector exposure cannot be validated without live holdings context.", severity="WARNING")
    return _stage("SECTOR", "REQUIRE_APPROVAL", "position_data_missing", "Live position and sector exposure validation is not implemented yet.", severity="WARNING")


def check_time_window(candidate: UsLiveOrderCandidate, policy: dict[str, object], context: dict[str, object]) -> CheckStageResult:
    time_policy = policy.get("time", {}) if isinstance(policy.get("time"), dict) else {}
    if not bool(time_policy.get("regular_session_only", True)):
        return _stage("TIME_WINDOW", "PASS")
    trade_date = context["trade_date"]
    ny_now = datetime.now(ZoneInfo("America/New_York"))
    if ny_now.date() != trade_date:
        return _stage("TIME_WINDOW", "REQUIRE_APPROVAL", "time_window_unknown", "Trade date is not the current US session date.", severity="WARNING")
    if ny_now.weekday() >= 5:
        return _stage("TIME_WINDOW", "BLOCK", "market_closed", "US market is closed on weekend.", severity="ERROR")
    open_time = time(9, 30)
    close_time = time(16, 0)
    first_block = int(time_policy.get("block_first_minutes_after_open", 15) or 15)
    last_block = int(time_policy.get("block_last_minutes_before_close", 15) or 15)
    current_time = ny_now.time()
    if current_time < open_time:
        return _stage("TIME_WINDOW", "BLOCK", "premarket_blocked", "Premarket orders are blocked.", severity="ERROR")
    if current_time >= close_time:
        return _stage("TIME_WINDOW", "BLOCK", "afterhours_blocked", "After-hours orders are blocked.", severity="ERROR")
    if current_time < (datetime.combine(trade_date, open_time) + timedelta(minutes=first_block)).time():
        return _stage("TIME_WINDOW", "BLOCK", "near_market_open_blocked", "Orders near market open are blocked.", severity="ERROR")
    if current_time >= (datetime.combine(trade_date, close_time) - timedelta(minutes=last_block)).time():
        return _stage("TIME_WINDOW", "BLOCK", "near_market_close_blocked", "Orders near market close are blocked.", severity="ERROR")
    return _stage("TIME_WINDOW", "PASS")


def check_manual_approval(candidate: UsLiveOrderCandidate, policy: dict[str, object]) -> CheckStageResult:
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    if bool(safety.get("require_manual_approval", True)):
        code = "approval_required_for_sell" if candidate.side.upper() == "SELL" else "manual_approval_required"
        detail = "Manual approval is required by policy." if candidate.side.upper() != "SELL" else "SELL candidates require manual approval in current policy."
        return _stage("APPROVAL", "REQUIRE_APPROVAL", code, detail, severity="WARNING")
    return _stage("APPROVAL", "PASS")


def build_block_log_rows(candidate: UsLiveOrderCandidate, result: UsLivePreTradeCheckResult) -> list[dict[str, object]]:
    trade_date = _parse_trade_date(candidate.trade_date)
    rows: list[dict[str, object]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    for idx, reason_code in enumerate(result.reason_codes, start=1):
        rows.append(
            {
                "block_id": f"USBLOCK_{trade_date.strftime('%Y%m%d')}_{candidate.account_id}_{candidate.symbol.upper()}_{candidate.side.upper()}_{reason_code}_{timestamp}_{idx}",
                "trade_date": trade_date,
                "policy_id": candidate.policy_id,
                "account_id": candidate.account_id,
                "symbol": candidate.symbol.upper(),
                "side": candidate.side.upper(),
                "candidate_source": candidate.candidate_source,
                "rank_no": candidate.rank_no,
                "recommend_grade": candidate.recommend_grade,
                "total_score": candidate.total_score,
                "requested_order_amount_usd": candidate.requested_order_amount_usd,
                "requested_qty": candidate.requested_qty,
                "requested_order_type": candidate.requested_order_type.upper(),
                "block_reason_code": reason_code,
                "block_reason_detail": result.reason_details[min(idx - 1, len(result.reason_details) - 1)] if result.reason_details else "",
                "check_stage": next((stage for stage, status in result.check_results.items() if status in {"BLOCK", "ERROR", "REQUIRE_APPROVAL"}), "UNKNOWN"),
                "severity": result.severity,
            }
        )
    return rows


def run_us_live_pre_trade_check(candidate: UsLiveOrderCandidate, *, write_block_log: bool = False, log_requires_approval: bool = False) -> UsLivePreTradeCheckResult:
    policy = load_us_live_risk_policy(candidate.policy_id)
    context = _load_context(candidate, policy)
    stage_results = [
        check_system_flags(candidate, policy),
        check_kill_switch(candidate, policy, context),
        check_ranking(candidate, policy, context),
        check_instrument(candidate, policy, context),
        check_price_and_volatility(candidate, policy, context),
        check_market_regime(candidate, policy, context),
        check_daily_limits(candidate, policy, context),
        check_position_limits(candidate, policy),
        check_sector_limits(candidate, policy, context),
        check_time_window(candidate, policy, context),
        check_manual_approval(candidate, policy),
    ]
    merged = _merge_stage_results(stage_results)
    result = UsLivePreTradeCheckResult(
        decision=merged.decision,
        symbol=candidate.symbol.upper(),
        side=candidate.side.upper(),
        reason_codes=merged.reason_codes,
        reason_details=merged.reason_details,
        severity=merged.severity,
        check_results=merged.check_results,
        requires_manual_approval=merged.requires_manual_approval,
        blocked=merged.blocked,
        created_at=merged.created_at,
    )
    if write_block_log and (result.decision in {"BLOCK", "ERROR"} or (log_requires_approval and result.decision == "REQUIRE_APPROVAL")):
        insert_us_live_order_block_log_rows(build_block_log_rows(candidate, result))
    return result


def run_batch_us_live_pre_trade_check(
    candidates: list[UsLiveOrderCandidate],
    *,
    write_block_log: bool = False,
    log_requires_approval: bool = False,
) -> list[UsLivePreTradeCheckResult]:
    return [
        run_us_live_pre_trade_check(item, write_block_log=write_block_log, log_requires_approval=log_requires_approval)
        for item in candidates
    ]


def result_to_markdown(candidate: UsLiveOrderCandidate, result: UsLivePreTradeCheckResult) -> str:
    lines = [
        "# US Live Pre-Trade Check",
        "",
        f"- Trade Date: {candidate.trade_date}",
        f"- Account: {candidate.account_id}",
        f"- Policy: {candidate.policy_id}",
        f"- Symbol: {candidate.symbol.upper()}",
        f"- Side: {candidate.side.upper()}",
        f"- Amount USD: {candidate.requested_order_amount_usd}",
        f"- Order Type: {candidate.requested_order_type.upper()}",
        f"- Decision: {result.decision}",
        f"- Severity: {result.severity}",
        "",
        "## Check Results",
        "",
    ]
    for stage, status in result.check_results.items():
        lines.append(f"- `{stage}`: `{status}`")
    lines.extend(["", "## Reason Codes", ""])
    if result.reason_codes:
        for code, detail in zip(result.reason_codes, result.reason_details or [""] * len(result.reason_codes)):
            lines.append(f"- `{code}`: {detail}")
    else:
        lines.append("- none")
    lines.extend(["", "## Safety", "", "- No real order API was called.", "- No live order was created."])
    return "\n".join(lines)


def results_to_json(results: list[UsLivePreTradeCheckResult]) -> str:
    payload = [
        {
            "decision": item.decision,
            "symbol": item.symbol,
            "side": item.side,
            "reason_codes": item.reason_codes,
            "reason_details": item.reason_details,
            "severity": item.severity,
            "check_results": item.check_results,
            "requires_manual_approval": item.requires_manual_approval,
            "blocked": item.blocked,
            "created_at": item.created_at,
        }
        for item in results
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
