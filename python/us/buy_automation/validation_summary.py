from __future__ import annotations

from collections import Counter, defaultdict


def summarize_validation(candidates: list[dict[str, object]]) -> dict[str, object]:
    block_counts: Counter[str] = Counter()
    symbol_blocks: dict[str, list[str]] = {}
    rule_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "UNKNOWN": 0})
    data_missing_symbols: list[str] = []
    rule_not_ready_symbols: list[str] = []
    invalid_decision_logs: list[str] = []
    parse_errors: list[str] = []
    fail_safe_block_count = 0
    automation_disabled_count = 0
    live_not_implemented = False

    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "UNKNOWN").upper()
        block_reasons = candidate.get("block_reasons")
        if not isinstance(block_reasons, list):
            parse_errors.append(symbol)
            block_reasons = ["REPORT_PARSE_ERROR"]
        if not candidate.get("allowed") and not block_reasons:
            invalid_decision_logs.append(symbol)
            block_reasons = ["INVALID_DECISION_LOG"]

        symbol_blocks[symbol] = [str(reason) for reason in block_reasons]
        for reason in symbol_blocks[symbol]:
            block_counts[reason] += 1
            upper_reason = reason.upper()
            if "DATA_MISSING" in upper_reason:
                data_missing_symbols.append(symbol)
            if "RULE_NOT_READY" in upper_reason:
                rule_not_ready_symbols.append(symbol)
            if "FAILSAFE" in upper_reason or upper_reason in {
                "PRICE_DATA_MISSING",
                "SCORE_MISSING",
                "PROBABILITY_MISSING",
                "FINANCIAL_DATA_MISSING",
                "BENCHMARK_STRENGTH_MISSING",
            }:
                fail_safe_block_count += 1
            if upper_reason == "AUTOMATION_DISABLED":
                automation_disabled_count += 1
            if upper_reason == "LIVE_NOT_IMPLEMENTED":
                live_not_implemented = True

        applied_rules = candidate.get("applied_rules")
        if not isinstance(applied_rules, list):
            parse_errors.append(symbol)
            continue
        for rule_row in applied_rules:
            if not isinstance(rule_row, dict):
                parse_errors.append(symbol)
                continue
            rule_name = str(rule_row.get("rule") or "UNKNOWN").upper()
            result = str(rule_row.get("result") or "UNKNOWN").upper()
            if result not in {"PASS", "FAIL", "UNKNOWN"}:
                result = "UNKNOWN"
            rule_summary[rule_name][result] += 1

    if invalid_decision_logs:
        block_counts["INVALID_DECISION_LOG"] += len(invalid_decision_logs)
    if parse_errors:
        block_counts["REPORT_PARSE_ERROR"] += len(parse_errors)

    return {
        "block_counts": dict(sorted(block_counts.items())),
        "symbol_block_reasons": symbol_blocks,
        "rule_summary": dict(sorted(rule_summary.items())),
        "data_missing_symbols": sorted(set(data_missing_symbols)),
        "rule_not_ready_symbols": sorted(set(rule_not_ready_symbols)),
        "automation_disabled_count": automation_disabled_count,
        "live_not_implemented": live_not_implemented,
        "fail_safe_block_count": fail_safe_block_count,
        "invalid_decision_logs": sorted(set(invalid_decision_logs)),
        "parse_errors": sorted(set(parse_errors)),
    }
