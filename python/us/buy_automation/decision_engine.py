from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from datetime import datetime, timezone
import json
import uuid

from sqlalchemy import text

from python.us.buy_automation.candidate_loader import load_buy_candidates
from python.us.buy_automation.config import BuyAutomationConfig, load_buy_automation_config
from python.us.buy_automation.logger import persist_buy_automation_logs
from python.us.buy_automation.paper_order import build_paper_order
from python.us.buy_automation.risk_guard import evaluate_candidate
from python.us.trade_orchestration.conflict_guard import build_conflict_rule_rows, check_buy_conflict
from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.us_config import parse_iso_date
from python.us.us_db import get_us_engine, relation_exists
from utils.us_live_kill_switch import list_active_kill_switches


RECENT_BUY_FROM_PAPER_SQL = text(
    """
    SELECT DISTINCT symbol
    FROM paper.us_stock_paper_order
    WHERE account_id = :account_id
      AND side = 'BUY'
      AND trade_date BETWEEN :start_date AND :end_date
      AND status IN ('CREATED', 'FILLED', 'PENDING', 'PARTIALLY_FILLED')
    """
)

RECENT_BUY_FROM_MICRO_SQL = text(
    """
    SELECT DISTINCT symbol
    FROM live.us_stock_micro_order_request
    WHERE account_id = :account_id
      AND side = 'BUY'
      AND trade_date BETWEEN :start_date AND :end_date
      AND request_status NOT IN ('CANCELED', 'REJECTED', 'FAILED', 'ERROR')
    """
)

RECENT_BUY_FROM_APPROVAL_SQL = text(
    """
    SELECT DISTINCT symbol
    FROM risk.us_stock_live_order_approval
    WHERE account_id = :account_id
      AND side = 'BUY'
      AND trade_date BETWEEN :start_date AND :end_date
      AND approval_status IN ('PENDING', 'APPROVED')
    """
)


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _recent_buy_symbols(account_id: str, trade_date: date, cooldown_days: int) -> set[str]:
    if cooldown_days <= 0:
        return set()
    start_date = trade_date - timedelta(days=cooldown_days)
    symbols: set[str] = set()
    try:
        engine = get_us_engine()
        with engine.connect() as conn:
            if relation_exists("paper.us_stock_paper_order"):
                rows = conn.execute(
                    RECENT_BUY_FROM_PAPER_SQL,
                    {"account_id": account_id, "start_date": start_date, "end_date": trade_date},
                ).mappings().all()
                symbols.update(str(row.get("symbol") or "").upper() for row in rows)
            if relation_exists("live.us_stock_micro_order_request"):
                rows = conn.execute(
                    RECENT_BUY_FROM_MICRO_SQL,
                    {"account_id": account_id, "start_date": start_date, "end_date": trade_date},
                ).mappings().all()
                symbols.update(str(row.get("symbol") or "").upper() for row in rows)
            if relation_exists("risk.us_stock_live_order_approval"):
                rows = conn.execute(
                    RECENT_BUY_FROM_APPROVAL_SQL,
                    {"account_id": account_id, "start_date": start_date, "end_date": trade_date},
                ).mappings().all()
                symbols.update(str(row.get("symbol") or "").upper() for row in rows)
    except Exception:
        return set()
    return {symbol for symbol in symbols if symbol}


def _is_kill_switch_active(account_id: str, symbol: str, sector: str | None) -> bool:
    try:
        active_rows = list_active_kill_switches()
    except Exception:
        return False
    symbol = str(symbol or "").upper()
    sector = str(sector or "").upper()
    account_id = str(account_id or "").upper()
    for row in active_rows:
        scope = str(row.get("scope") or "").upper()
        target_value = str(row.get("target_value") or "").upper()
        if scope in {"GLOBAL", "BUY"}:
            return True
        if scope == "SYMBOL" and target_value == symbol:
            return True
        if scope == "SECTOR" and sector and target_value == sector:
            return True
        if scope == "ACCOUNT" and target_value == account_id:
            return True
    return False


def _severity_for_candidate(allowed: bool, block_reasons: list[str]) -> str:
    if allowed:
        return "INFO"
    critical_codes = {
        "KILL_SWITCH_ACTIVE",
        "DATA_ERROR_FAILSAFE",
        "PRICE_DATA_MISSING",
        "SCORE_MISSING",
        "PROBABILITY_MISSING",
        "FINANCIAL_DATA_MISSING",
        "BENCHMARK_STRENGTH_MISSING",
    }
    if any(code in critical_codes for code in block_reasons):
        return "ERROR"
    return "WARNING"


def _attach_ids(candidate: dict[str, object]) -> dict[str, object]:
    symbol = str(candidate.get("symbol") or "UNKNOWN").upper()
    token = uuid.uuid4().hex[:10].upper()
    candidate["candidate_id"] = f"USBUYCAND_{symbol}_{token}"
    candidate["decision_id"] = f"USBUYDEC_{symbol}_{token}"
    candidate["guard_log_id"] = f"USBUYGUARD_{symbol}_{token}"
    return candidate


def _apply_mode_overrides(candidate: dict[str, object], cfg: BuyAutomationConfig) -> None:
    block_reasons = list(candidate.get("block_reasons") or [])
    if not cfg.enabled:
        block_reasons.append("AUTOMATION_DISABLED")
        candidate["allowed"] = False
    if cfg.mode == "LIVE":
        block_reasons.append("LIVE_NOT_IMPLEMENTED")
        candidate["allowed"] = False
    candidate["block_reasons"] = block_reasons


def run_buy_automation(
    *,
    trade_date: str | None = None,
    account_id: str = "US_BUY_SHADOW",
    persist_logs: bool = True,
    portfolio_state: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg = load_buy_automation_config()
    trade_cfg = load_trade_orchestration_config()
    requested_trade_date = parse_iso_date(trade_date, field_name="trade_date") if trade_date else None
    load_result = load_buy_candidates(cfg, requested_trade_date=requested_trade_date)
    effective_trade_date = load_result.get("trade_date")
    candidates = list(load_result.get("candidates") or [])
    events = list(load_result.get("events") or [])

    recent_buy_symbols: set[str] = set()
    if isinstance(effective_trade_date, date):
        recent_buy_symbols = _recent_buy_symbols(account_id, effective_trade_date, cfg.cooldown_days)

    selected_count = 0
    selected_amount_usd = 0.0
    paper_orders: list[dict[str, object]] = []
    final_candidates: list[dict[str, object]] = []
    allowed_before_conflict = 0
    allowed_after_conflict = 0
    conflict_blocked_candidates = 0

    for raw_candidate in sorted(candidates, key=lambda row: (int(row.get("rank") or 999999), str(row.get("symbol") or ""))):
        candidate = _attach_ids(dict(raw_candidate))
        kill_switch_active = _is_kill_switch_active(account_id, str(candidate.get("symbol") or ""), candidate.get("sector"))
        guard = evaluate_candidate(
            candidate,
            cfg,
            selected_count=selected_count,
            selected_amount_usd=selected_amount_usd,
            recent_buy_symbols=recent_buy_symbols,
            kill_switch_active=kill_switch_active,
        )
        candidate.update(guard)
        candidate["risk_allowed"] = bool(candidate.get("allowed"))
        if candidate["risk_allowed"]:
            allowed_before_conflict += 1
        candidate["conflict_checked"] = False
        candidate["conflict_blocked"] = False
        candidate["conflict_reasons"] = []
        candidate["related_position_id"] = None
        candidate["related_sell_signal"] = None
        candidate["cooldown_until"] = None
        candidate["buy_allowed_after_conflict_check"] = bool(candidate.get("allowed"))
        if portfolio_state is not None:
            conflict_result = check_buy_conflict(candidate, portfolio_state, trade_cfg)
            candidate["conflict_checked"] = True
            candidate["buy_allowed_after_conflict_check"] = bool(conflict_result.get("buy_allowed_after_conflict_check"))
            candidate["conflict_reasons"] = list(conflict_result.get("conflict_reasons") or [])
            candidate["related_position_id"] = conflict_result.get("related_position_id")
            candidate["related_sell_signal"] = conflict_result.get("sell_signal")
            candidate["cooldown_until"] = conflict_result.get("cooldown_until")
            if candidate["risk_allowed"] and not candidate["buy_allowed_after_conflict_check"]:
                candidate["allowed"] = False
                candidate["conflict_blocked"] = True
                conflict_blocked_candidates += 1
                candidate["block_reasons"] = list(candidate.get("block_reasons") or []) + list(candidate["conflict_reasons"])
                candidate["applied_rules"] = list(candidate.get("applied_rules") or []) + build_conflict_rule_rows(conflict_result)
            elif candidate["risk_allowed"] and candidate["buy_allowed_after_conflict_check"]:
                allowed_after_conflict += 1
                candidate["applied_rules"] = list(candidate.get("applied_rules") or []) + build_conflict_rule_rows(conflict_result)
        else:
            if candidate["risk_allowed"]:
                allowed_after_conflict += 1
        _apply_mode_overrides(candidate, cfg)
        candidate["severity"] = _severity_for_candidate(bool(candidate.get("allowed")), list(candidate.get("block_reasons") or []))
        candidate["allocated_amount_usd"] = round(float(candidate.get("proposed_amount_usd") or 0.0), 6) if candidate.get("allowed") else 0.0

        if candidate.get("allowed"):
            selected_count += 1
            selected_amount_usd += float(candidate.get("allocated_amount_usd") or 0.0)
            if cfg.mode == "PAPER":
                paper_order = build_paper_order(
                    trade_date=str(effective_trade_date),
                    symbol=str(candidate.get("symbol")),
                    allocated_amount_usd=float(candidate.get("allocated_amount_usd") or 0.0),
                    reference_price=_safe_float(candidate.get("reference_price")),
                    mode=cfg.mode,
                )
                paper_order["source_decision_id"] = candidate["decision_id"]
                paper_orders.append(paper_order)
        final_candidates.append(candidate)

    block_summary = Counter()
    for candidate in final_candidates:
        reasons = candidate.get("block_reasons") or []
        if not reasons and candidate.get("allowed"):
            block_summary["ALLOW"] += 1
        for reason in reasons:
            block_summary[str(reason)] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_date": str(effective_trade_date) if effective_trade_date else None,
        "account_id": account_id,
        "mode": cfg.mode,
        "enabled": cfg.enabled,
        "ranking_source": cfg.ranking_source,
        "config_snapshot": {
            "top_n": cfg.top_n,
            "min_score": cfg.min_score,
            "min_prob": cfg.min_prob,
            "max_daily_symbols": cfg.max_daily_symbols,
            "max_daily_amount_usd": cfg.max_daily_amount_usd,
            "max_per_symbol_amount_usd": cfg.max_per_symbol_amount_usd,
        },
        "config_warnings": list(cfg.warnings),
        "events": events,
        "loaded_candidates": len(candidates),
        "allowed_before_conflict": allowed_before_conflict,
        "allowed_after_conflict": allowed_after_conflict,
        "conflict_blocked_candidates": conflict_blocked_candidates,
        "allowed_candidates": sum(1 for item in final_candidates if item.get("allowed")),
        "blocked_candidates": sum(1 for item in final_candidates if not item.get("allowed")),
        "paper_orders": paper_orders,
        "candidates": final_candidates,
        "block_summary": dict(sorted(block_summary.items())),
        "recent_buy_symbols": sorted(recent_buy_symbols),
        "portfolio_state_status": (portfolio_state or {}).get("status"),
    }
    if persist_logs:
        report["log_persistence"] = persist_buy_automation_logs(cfg=cfg, report=report, account_id=account_id)
    return report
