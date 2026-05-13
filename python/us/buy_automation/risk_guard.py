from __future__ import annotations

from datetime import date

from python.us.buy_automation.config import BuyAutomationConfig


GRADE_ORDER = {
    "EXCLUDE": 0,
    "HOLD": 1,
    "WATCH": 2,
    "BUY": 3,
    "STRONG_BUY": 4,
}


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _grade_passes(candidate_grade: str, minimum_grade: str) -> bool:
    return GRADE_ORDER.get(candidate_grade.upper(), -1) >= GRADE_ORDER.get(minimum_grade.upper(), 999)


def _add_rule(
    applied_rules: list[dict[str, object]],
    *,
    rule: str,
    passed: bool,
    reason_code: str | None = None,
    value: object | None = None,
    threshold: object | None = None,
) -> None:
    row: dict[str, object] = {
        "rule": rule,
        "result": "PASS" if passed else "FAIL",
    }
    if reason_code:
        row["reason_code"] = reason_code
    if value is not None:
        row["value"] = value
    if threshold is not None:
        row["threshold"] = threshold
    applied_rules.append(row)


def evaluate_candidate(
    candidate: dict[str, object],
    cfg: BuyAutomationConfig,
    *,
    selected_count: int,
    selected_amount_usd: float,
    recent_buy_symbols: set[str] | None = None,
    kill_switch_active: bool = False,
) -> dict[str, object]:
    recent_buy_symbols = recent_buy_symbols or set()
    symbol = str(candidate.get("symbol") or "").upper()
    applied_rules: list[dict[str, object]] = []
    block_reasons: list[str] = []

    def fail(rule: str, reason_code: str, value: object | None = None, threshold: object | None = None) -> None:
        block_reasons.append(reason_code)
        _add_rule(applied_rules, rule=rule, passed=False, reason_code=reason_code, value=value, threshold=threshold)

    def passed(rule: str, value: object | None = None, threshold: object | None = None) -> None:
        _add_rule(applied_rules, rule=rule, passed=True, value=value, threshold=threshold)

    if kill_switch_active and cfg.block_on_kill_switch:
        fail("KILL_SWITCH", "KILL_SWITCH_ACTIVE")
    else:
        passed("KILL_SWITCH")

    price = _safe_float(candidate.get("reference_price"))
    if price is None:
        fail("PRICE_RANGE", "PRICE_DATA_MISSING")
    elif price < cfg.min_price or price > cfg.max_price:
        fail("PRICE_RANGE", "PRICE_OUT_OF_RANGE", value=price, threshold=f"{cfg.min_price}..{cfg.max_price}")
    else:
        passed("PRICE_RANGE", value=price, threshold=f"{cfg.min_price}..{cfg.max_price}")

    score = _safe_float(candidate.get("score"))
    if score is None:
        fail("SCORE_THRESHOLD", "SCORE_MISSING")
    elif score < cfg.min_score:
        fail("SCORE_THRESHOLD", "SCORE_BELOW_THRESHOLD", value=score, threshold=cfg.min_score)
    else:
        passed("SCORE_THRESHOLD", value=score, threshold=cfg.min_score)

    probability = _safe_float(candidate.get("probability"))
    if cfg.min_prob > 0:
        if probability is None:
            fail("PROBABILITY_THRESHOLD", "PROBABILITY_MISSING")
        elif probability < cfg.min_prob:
            fail("PROBABILITY_THRESHOLD", "PROBABILITY_BELOW_THRESHOLD", value=probability, threshold=cfg.min_prob)
        else:
            passed("PROBABILITY_THRESHOLD", value=probability, threshold=cfg.min_prob)
    else:
        passed("PROBABILITY_THRESHOLD", value=probability, threshold=cfg.min_prob)

    rank = int(candidate.get("rank") or 999999)
    if rank > cfg.top_n:
        fail("RANK_LIMIT", "RANK_OUTSIDE_TOP_N", value=rank, threshold=cfg.top_n)
    else:
        passed("RANK_LIMIT", value=rank, threshold=cfg.top_n)

    grade = str(candidate.get("recommend_grade") or "").upper()
    if not _grade_passes(grade, cfg.min_grade):
        fail("GRADE_THRESHOLD", "GRADE_BELOW_THRESHOLD", value=grade, threshold=cfg.min_grade)
    else:
        passed("GRADE_THRESHOLD", value=grade, threshold=cfg.min_grade)

    if cfg.require_financial_data:
        financial_quality_score = _safe_float(candidate.get("financial_quality_score"))
        if not candidate.get("financial_feature"):
            fail("FINANCIAL_DATA", "FINANCIAL_DATA_MISSING")
        elif financial_quality_score is None:
            fail("FINANCIAL_DATA", "FINANCIAL_QUALITY_MISSING")
        else:
            passed("FINANCIAL_DATA", value=financial_quality_score)
    else:
        passed("FINANCIAL_DATA")

    if cfg.require_benchmark_strength:
        relative_strength = candidate.get("relative_strength") or {}
        rs_values = [
            _safe_float(relative_strength.get("rs_spy_20d")),
            _safe_float(relative_strength.get("rs_qqq_20d")),
            _safe_float(relative_strength.get("rs_spy_60d")),
            _safe_float(relative_strength.get("rs_qqq_60d")),
        ]
        usable = [value for value in rs_values if value is not None]
        if not relative_strength or not usable:
            fail("BENCHMARK_STRENGTH", "BENCHMARK_STRENGTH_MISSING")
        elif max(usable) <= 0:
            fail("BENCHMARK_STRENGTH", "BENCHMARK_STRENGTH_WEAK", value=max(usable), threshold="> 0")
        else:
            passed("BENCHMARK_STRENGTH", value=max(usable), threshold="> 0")
    else:
        passed("BENCHMARK_STRENGTH")

    gap_up_pct = _safe_float(candidate.get("gap_up_pct"))
    if gap_up_pct is None:
        fail("GAP_UP", "GAP_UP_DATA_MISSING")
    elif gap_up_pct > cfg.max_gap_up_pct:
        fail("GAP_UP", "GAP_UP_TOO_HIGH", value=gap_up_pct, threshold=cfg.max_gap_up_pct)
    else:
        passed("GAP_UP", value=gap_up_pct, threshold=cfg.max_gap_up_pct)

    intraday_change_pct = _safe_float(candidate.get("intraday_change_pct"))
    if intraday_change_pct is None:
        fail("INTRADAY_CHANGE", "INTRADAY_CHANGE_DATA_MISSING")
    elif intraday_change_pct > cfg.max_intraday_change_pct:
        fail("INTRADAY_CHANGE", "INTRADAY_CHANGE_TOO_HIGH", value=intraday_change_pct, threshold=cfg.max_intraday_change_pct)
    else:
        passed("INTRADAY_CHANGE", value=intraday_change_pct, threshold=cfg.max_intraday_change_pct)

    volatility_20d = _safe_float(candidate.get("volatility_20d"))
    if volatility_20d is None:
        fail("VOLATILITY", "VOLATILITY_DATA_MISSING")
    elif volatility_20d > cfg.max_volatility_pct:
        fail("VOLATILITY", "VOLATILITY_TOO_HIGH", value=volatility_20d, threshold=cfg.max_volatility_pct)
    else:
        passed("VOLATILITY", value=volatility_20d, threshold=cfg.max_volatility_pct)

    if symbol in recent_buy_symbols:
        fail("COOLDOWN", "COOLDOWN_ACTIVE", value=symbol, threshold=cfg.cooldown_days)
    else:
        passed("COOLDOWN", value=symbol, threshold=cfg.cooldown_days)

    if selected_count >= cfg.max_daily_symbols:
        fail("DAILY_SYMBOL_LIMIT", "DAILY_SYMBOL_LIMIT_REACHED", value=selected_count, threshold=cfg.max_daily_symbols)
    else:
        passed("DAILY_SYMBOL_LIMIT", value=selected_count, threshold=cfg.max_daily_symbols)

    remaining_amount = max(0.0, cfg.max_daily_amount_usd - selected_amount_usd)
    proposed_amount = min(cfg.max_per_symbol_amount_usd, remaining_amount)
    if proposed_amount <= 0:
        fail("DAILY_AMOUNT_LIMIT", "DAILY_AMOUNT_LIMIT_REACHED", value=selected_amount_usd, threshold=cfg.max_daily_amount_usd)
    else:
        passed("DAILY_AMOUNT_LIMIT", value=selected_amount_usd, threshold=cfg.max_daily_amount_usd)

    if cfg.failsafe_on_data_error and str(candidate.get("data_status") or "").upper() in {"ERROR", "MISSING_PRICE_FEATURE", "MISSING"}:
        fail("DATA_STATUS", "DATA_ERROR_FAILSAFE", value=candidate.get("data_status"))
    else:
        passed("DATA_STATUS", value=candidate.get("data_status"))

    return {
        "symbol": symbol,
        "allowed": len(block_reasons) == 0,
        "block_reasons": block_reasons,
        "applied_rules": applied_rules,
        "proposed_amount_usd": round(proposed_amount, 6),
    }
