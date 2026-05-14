from __future__ import annotations

from python.us.sell_automation.config import SellAutomationConfig


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_ratio(score: object) -> float | None:
    value = _safe_float(score)
    if value is None:
        return None
    if abs(value) > 1.0:
        return value / 100.0
    return value


def _rule(
    name: str,
    *,
    result: str,
    value: object,
    threshold: object,
    action: str,
    reason: str | None,
) -> dict[str, object]:
    return {
        "rule": name,
        "result": result,
        "value": value,
        "threshold": threshold,
        "action": action,
        "reason": reason,
    }


def evaluate_sell_rules(
    position: dict[str, object],
    cfg: SellAutomationConfig,
    *,
    market_context: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    market_context = market_context or {}
    flags = list(position.get("data_quality_flags") or [])
    latest_price = _safe_float(position.get("latest_price"))
    avg_entry_price = _safe_float(position.get("avg_entry_price"))
    remaining_quantity = _safe_float(position.get("remaining_quantity"))
    unrealized_pnl_pct = _safe_float(position.get("unrealized_pnl_pct"))
    highest_price_since_entry = _safe_float(position.get("highest_price_since_entry"))
    holding_days = _safe_float(position.get("holding_days"))
    latest_rank = _safe_float(position.get("latest_rank"))
    latest_score_ratio = _score_ratio(position.get("latest_score"))
    latest_probability = _safe_float(position.get("latest_probability"))
    benchmark_return_pct = _safe_float(position.get("benchmark_return_pct"))
    symbol_return_pct = _safe_float(position.get("symbol_return_pct"))
    benchmark_drawdown_pct = _safe_float(market_context.get("benchmark_drawdown_pct"))

    rules: list[dict[str, object]] = []

    if flags or latest_price is None or avg_entry_price is None or remaining_quantity is None or remaining_quantity <= 0:
        reason = flags[0] if flags else "POSITION_DATA_INVALID"
        rules.append(
            _rule(
                "DATA_QUALITY_CHECK",
                result="FAIL",
                value=None,
                threshold="NO_DATA_ERROR",
                action="REVIEW_REQUIRED",
                reason=reason,
            )
        )
    else:
        rules.append(
            _rule(
                "DATA_QUALITY_CHECK",
                result="PASS",
                value="OK",
                threshold="NO_DATA_ERROR",
                action="HOLD",
                reason=None,
            )
        )

    if not cfg.risk_off_exit_enabled:
        rules.append(_rule("RISK_OFF_MARKET_EXIT", result="PASS", value=None, threshold=cfg.market_drawdown_exit_pct, action="HOLD", reason=None))
    elif benchmark_drawdown_pct is None:
        rules.append(
            _rule(
                "RISK_OFF_MARKET_EXIT",
                result="UNKNOWN",
                value=None,
                threshold=cfg.market_drawdown_exit_pct,
                action="REVIEW_REQUIRED",
                reason="MARKET_DATA_MISSING",
            )
        )
    elif benchmark_drawdown_pct <= cfg.market_drawdown_exit_pct:
        rules.append(
            _rule(
                "RISK_OFF_MARKET_EXIT",
                result="FAIL",
                value=benchmark_drawdown_pct,
                threshold=cfg.market_drawdown_exit_pct,
                action="FULL_SELL",
                reason="RISK_OFF_MARKET_EXIT",
            )
        )
    else:
        rules.append(
            _rule(
                "RISK_OFF_MARKET_EXIT",
                result="PASS",
                value=benchmark_drawdown_pct,
                threshold=cfg.market_drawdown_exit_pct,
                action="HOLD",
                reason=None,
            )
        )

    if unrealized_pnl_pct is None:
        rules.append(_rule("STOP_LOSS", result="UNKNOWN", value=None, threshold=cfg.stop_loss_pct, action="REVIEW_REQUIRED", reason="PRICE_DATA_MISSING"))
    elif unrealized_pnl_pct <= cfg.stop_loss_pct:
        rules.append(_rule("STOP_LOSS", result="FAIL", value=unrealized_pnl_pct, threshold=cfg.stop_loss_pct, action="FULL_SELL", reason="STOP_LOSS"))
    else:
        rules.append(_rule("STOP_LOSS", result="PASS", value=unrealized_pnl_pct, threshold=cfg.stop_loss_pct, action="HOLD", reason=None))

    trailing_drawdown_pct = None
    trailing_threshold = -cfg.trailing_stop_pct
    if latest_price is not None and highest_price_since_entry and highest_price_since_entry > 0:
        trailing_drawdown_pct = round((latest_price / highest_price_since_entry) - 1.0, 6)
    if trailing_drawdown_pct is None:
        rules.append(
            _rule(
                "TRAILING_STOP",
                result="UNKNOWN",
                value=None,
                threshold=trailing_threshold,
                action="REVIEW_REQUIRED",
                reason="HIGH_WATER_MARK_MISSING",
            )
        )
    elif trailing_drawdown_pct <= trailing_threshold:
        rules.append(
            _rule(
                "TRAILING_STOP",
                result="FAIL",
                value=trailing_drawdown_pct,
                threshold=trailing_threshold,
                action="FULL_SELL",
                reason="TRAILING_STOP",
            )
        )
    else:
        rules.append(_rule("TRAILING_STOP", result="PASS", value=trailing_drawdown_pct, threshold=trailing_threshold, action="HOLD", reason=None))

    if holding_days is None:
        rules.append(
            _rule(
                "MAX_HOLDING_DAYS",
                result="UNKNOWN",
                value=None,
                threshold=cfg.max_holding_days,
                action="REVIEW_REQUIRED",
                reason="HOLDING_DAYS_MISSING",
            )
        )
    elif holding_days >= cfg.max_holding_days:
        rules.append(_rule("MAX_HOLDING_DAYS", result="FAIL", value=holding_days, threshold=cfg.max_holding_days, action="FULL_SELL", reason="MAX_HOLDING_DAYS"))
    else:
        rules.append(_rule("MAX_HOLDING_DAYS", result="PASS", value=holding_days, threshold=cfg.max_holding_days, action="HOLD", reason=None))

    deterioration_reason = None
    deterioration_value: object = None
    if latest_rank is None and latest_score_ratio is None and (cfg.min_prob_hold <= 0 or latest_probability is None):
        rules.append(
            _rule(
                "RANK_SCORE_DETERIORATION",
                result="UNKNOWN",
                value=None,
                threshold={
                    "rank_exit_threshold": cfg.rank_exit_threshold,
                    "min_score_hold": cfg.min_score_hold,
                    "min_prob_hold": cfg.min_prob_hold,
                },
                action="REVIEW_REQUIRED",
                reason="RANK_DATA_MISSING",
            )
        )
    else:
        if latest_rank is not None and latest_rank > cfg.rank_exit_threshold:
            deterioration_reason = "RANK_EXIT_THRESHOLD"
            deterioration_value = latest_rank
        elif latest_score_ratio is not None and latest_score_ratio < cfg.min_score_hold:
            deterioration_reason = "SCORE_BELOW_HOLD"
            deterioration_value = latest_score_ratio
        elif cfg.min_prob_hold > 0 and latest_probability is None:
            deterioration_reason = "PROBABILITY_MISSING"
        elif latest_probability is not None and latest_probability < cfg.min_prob_hold:
            deterioration_reason = "PROBABILITY_BELOW_HOLD"
            deterioration_value = latest_probability

        if deterioration_reason == "PROBABILITY_MISSING":
            rules.append(
                _rule(
                    "RANK_SCORE_DETERIORATION",
                    result="UNKNOWN",
                    value=None,
                    threshold={
                        "rank_exit_threshold": cfg.rank_exit_threshold,
                        "min_score_hold": cfg.min_score_hold,
                        "min_prob_hold": cfg.min_prob_hold,
                    },
                    action="REVIEW_REQUIRED",
                    reason=deterioration_reason,
                )
            )
        elif deterioration_reason:
            rules.append(
                _rule(
                    "RANK_SCORE_DETERIORATION",
                    result="FAIL",
                    value=deterioration_value,
                    threshold={
                        "rank_exit_threshold": cfg.rank_exit_threshold,
                        "min_score_hold": cfg.min_score_hold,
                        "min_prob_hold": cfg.min_prob_hold,
                    },
                    action="FULL_SELL",
                    reason="RANK_SCORE_DETERIORATION",
                )
            )
        else:
            rules.append(
                _rule(
                    "RANK_SCORE_DETERIORATION",
                    result="PASS",
                    value={
                        "rank": latest_rank,
                        "score_ratio": latest_score_ratio,
                        "probability": latest_probability,
                    },
                    threshold={
                        "rank_exit_threshold": cfg.rank_exit_threshold,
                        "min_score_hold": cfg.min_score_hold,
                        "min_prob_hold": cfg.min_prob_hold,
                    },
                    action="HOLD",
                    reason=None,
                )
            )

    if not cfg.require_benchmark_strength:
        rules.append(_rule("BENCHMARK_UNDERPERFORMANCE", result="PASS", value=None, threshold=cfg.benchmark_underperform_pct, action="HOLD", reason=None))
    elif symbol_return_pct is None or benchmark_return_pct is None:
        rules.append(
            _rule(
                "BENCHMARK_UNDERPERFORMANCE",
                result="UNKNOWN",
                value=None,
                threshold=cfg.benchmark_underperform_pct,
                action="REVIEW_REQUIRED",
                reason="BENCHMARK_DATA_MISSING",
            )
        )
    else:
        excess_return_pct = round(symbol_return_pct - benchmark_return_pct, 6)
        if excess_return_pct <= cfg.benchmark_underperform_pct:
            rules.append(
                _rule(
                    "BENCHMARK_UNDERPERFORMANCE",
                    result="FAIL",
                    value=excess_return_pct,
                    threshold=cfg.benchmark_underperform_pct,
                    action="FULL_SELL",
                    reason="BENCHMARK_UNDERPERFORMANCE",
                )
            )
        else:
            rules.append(
                _rule(
                    "BENCHMARK_UNDERPERFORMANCE",
                    result="PASS",
                    value=excess_return_pct,
                    threshold=cfg.benchmark_underperform_pct,
                    action="HOLD",
                    reason=None,
                )
            )

    if unrealized_pnl_pct is None:
        rules.append(_rule("TAKE_PROFIT", result="UNKNOWN", value=None, threshold=cfg.take_profit_pct, action="REVIEW_REQUIRED", reason="PRICE_DATA_MISSING"))
    elif unrealized_pnl_pct >= cfg.take_profit_pct:
        rules.append(
            _rule(
                "TAKE_PROFIT",
                result="FAIL",
                value=unrealized_pnl_pct,
                threshold=cfg.take_profit_pct,
                action="PARTIAL_SELL" if cfg.partial_take_profit_enabled else "FULL_SELL",
                reason="TAKE_PROFIT",
            )
        )
    else:
        rules.append(_rule("TAKE_PROFIT", result="PASS", value=unrealized_pnl_pct, threshold=cfg.take_profit_pct, action="HOLD", reason=None))

    rules.append(_rule("HOLD", result="PASS", value="NO_EXIT_TRIGGER", threshold=None, action="HOLD", reason=None))
    return rules
