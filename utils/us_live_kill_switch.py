from __future__ import annotations

from datetime import date, datetime, timezone
import re
from typing import Any

from python.us.us_db import (
    ensure_us_live_risk_tables,
    fetch_market_regime_rows_between,
    fetch_meta_us_universe_rows,
    fetch_us_live_daily_risk_usage_rows,
    fetch_us_live_kill_switch_rows,
    fetch_us_live_order_block_log_rows,
    insert_us_live_kill_switch_event_log_rows,
    upsert_us_live_kill_switch_rows,
)
from utils.us_live_risk_policy import load_us_live_risk_policy


def _safe_str(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


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


def _normalize_token(value: str | None, default: str = "ALL") -> str:
    text = _safe_str(value, default).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or default


def build_kill_switch_id(scope: str, target_value: str | None = None) -> str:
    scope_token = _normalize_token(scope)
    if scope_token == "GLOBAL":
        return "US_LIVE_GLOBAL_KILL"
    if scope_token in {"BUY", "SELL"}:
        return f"US_LIVE_{scope_token}_KILL"
    target_token = _normalize_token(target_value)
    return f"US_LIVE_{scope_token}_{target_token}_KILL"


def _normalize_target_value(scope: str, target_value: str | None = None) -> str | None:
    scope_token = _normalize_token(scope)
    if scope_token == "GLOBAL":
        return "ALL"
    if scope_token in {"BUY", "SELL"}:
        return scope_token
    normalized = _safe_str(target_value)
    return normalized.upper() if normalized else None


def _build_default_status(scope: str, target_value: str | None = None) -> dict[str, object]:
    scope_token = _normalize_token(scope)
    normalized_target = _normalize_target_value(scope_token, target_value)
    return {
        "kill_switch_id": build_kill_switch_id(scope_token, normalized_target),
        "scope": scope_token,
        "target_value": normalized_target,
        "is_active": False,
        "reason_code": None,
        "reason_detail": None,
        "activated_at": None,
        "activated_by": None,
        "cleared_at": None,
        "cleared_by": None,
        "clear_reason": None,
    }


def _build_event_row(
    *,
    kill_switch_id: str,
    scope: str,
    target_value: str | None,
    event_type: str,
    reason_code: str | None,
    reason_detail: str | None,
    trigger_source: str,
    trigger_ref_id: str | None,
    performed_by: str | None,
    before_is_active: bool | None,
    after_is_active: bool | None,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return {
        "event_id": f"USKSEVT_{kill_switch_id}_{event_type}_{timestamp}",
        "kill_switch_id": kill_switch_id,
        "scope": scope,
        "target_value": target_value,
        "event_type": event_type,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "trigger_source": trigger_source,
        "trigger_ref_id": trigger_ref_id,
        "performed_by": performed_by,
        "before_is_active": before_is_active,
        "after_is_active": after_is_active,
    }


def get_kill_switch_status(scope: str, target_value: str | None = None) -> dict[str, object]:
    ensure_us_live_risk_tables()
    kill_switch_id = build_kill_switch_id(scope, target_value)
    rows = fetch_us_live_kill_switch_rows(kill_switch_id=kill_switch_id)
    if rows:
        return rows[0]
    return _build_default_status(scope, target_value)


def is_kill_switch_active(scope: str, target_value: str | None = None) -> bool:
    return bool(get_kill_switch_status(scope, target_value).get("is_active"))


def activate_kill_switch(
    scope: str,
    target_value: str | None,
    reason_code: str,
    reason_detail: str,
    performed_by: str = "SYSTEM",
    trigger_source: str = "MANUAL",
    trigger_ref_id: str | None = None,
) -> dict[str, object]:
    scope_token = _normalize_token(scope)
    performed_by = _safe_str(performed_by)
    if not reason_code.strip():
        raise ValueError("reason_code is required for kill-switch activation.")
    if not reason_detail.strip():
        raise ValueError("reason_detail is required for kill-switch activation.")
    if not performed_by:
        raise ValueError("performed_by is required for kill-switch activation.")
    if scope_token in {"SYMBOL", "SECTOR", "ACCOUNT"} and not _safe_str(target_value):
        raise ValueError(f"target_value is required for scope={scope_token}.")

    ensure_us_live_risk_tables()
    current = get_kill_switch_status(scope_token, target_value)
    now = datetime.now(timezone.utc)
    activated_at = current.get("activated_at") or now
    row = {
        "kill_switch_id": build_kill_switch_id(scope_token, target_value),
        "scope": scope_token,
        "target_value": _normalize_target_value(scope_token, target_value),
        "is_active": True,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "activated_at": activated_at,
        "activated_by": current.get("activated_by") or performed_by,
        "cleared_at": None,
        "cleared_by": None,
        "clear_reason": None,
    }
    upsert_us_live_kill_switch_rows([row])
    insert_us_live_kill_switch_event_log_rows([
        _build_event_row(
            kill_switch_id=row["kill_switch_id"],
            scope=scope_token,
            target_value=row["target_value"],
            event_type="ACTIVATE",
            reason_code=reason_code,
            reason_detail=reason_detail,
            trigger_source=trigger_source,
            trigger_ref_id=trigger_ref_id,
            performed_by=performed_by,
            before_is_active=bool(current.get("is_active")),
            after_is_active=True,
        )
    ])
    return get_kill_switch_status(scope_token, target_value)


def clear_kill_switch(
    scope: str,
    target_value: str | None,
    clear_reason: str,
    performed_by: str,
) -> dict[str, object]:
    scope_token = _normalize_token(scope)
    performed_by = _safe_str(performed_by)
    clear_reason = _safe_str(clear_reason)
    if not clear_reason:
        raise ValueError("clear_reason is required for kill-switch clear.")
    if not performed_by:
        raise ValueError("performed_by is required for kill-switch clear.")
    if scope_token in {"SYMBOL", "SECTOR", "ACCOUNT"} and not _safe_str(target_value):
        raise ValueError(f"target_value is required for scope={scope_token}.")

    ensure_us_live_risk_tables()
    current = get_kill_switch_status(scope_token, target_value)
    row = {
        "kill_switch_id": build_kill_switch_id(scope_token, target_value),
        "scope": scope_token,
        "target_value": _normalize_target_value(scope_token, target_value),
        "is_active": False,
        "reason_code": current.get("reason_code"),
        "reason_detail": current.get("reason_detail"),
        "activated_at": current.get("activated_at"),
        "activated_by": current.get("activated_by"),
        "cleared_at": datetime.now(timezone.utc),
        "cleared_by": performed_by,
        "clear_reason": clear_reason,
    }
    upsert_us_live_kill_switch_rows([row])
    insert_us_live_kill_switch_event_log_rows([
        _build_event_row(
            kill_switch_id=row["kill_switch_id"],
            scope=scope_token,
            target_value=row["target_value"],
            event_type="CLEAR",
            reason_code=current.get("reason_code"),
            reason_detail=clear_reason,
            trigger_source="MANUAL",
            trigger_ref_id=None,
            performed_by=performed_by,
            before_is_active=bool(current.get("is_active")),
            after_is_active=False,
        )
    ])
    return get_kill_switch_status(scope_token, target_value)


def list_kill_switches(*, active_only: bool = False) -> list[dict[str, object]]:
    ensure_us_live_risk_tables()
    rows = fetch_us_live_kill_switch_rows()
    if active_only:
        rows = [row for row in rows if bool(row.get("is_active"))]
    return rows


def list_active_kill_switches() -> list[dict[str, object]]:
    return list_kill_switches(active_only=True)


def _lookup_symbol_sector(symbol: str) -> str | None:
    universe_rows = fetch_meta_us_universe_rows()
    for row in universe_rows:
        if _safe_str(row.get("symbol")).upper() == symbol.upper():
            sector = _safe_str(row.get("sector"))
            return sector.upper() if sector else None
    return None


def check_kill_switch_for_order_candidate(candidate) -> dict[str, object]:
    symbol = _safe_str(getattr(candidate, "symbol", None)).upper()
    side = _safe_str(getattr(candidate, "side", None)).upper()
    account_id = _safe_str(getattr(candidate, "account_id", None))
    sector = _lookup_symbol_sector(symbol) if symbol else None
    active_rows = list_active_kill_switches()
    matches: list[dict[str, object]] = []

    def add_match(row: dict[str, object], code: str) -> None:
        matches.append(
            {
                "kill_switch_id": row.get("kill_switch_id"),
                "scope": row.get("scope"),
                "target_value": row.get("target_value"),
                "reason_code": code,
                "reason_detail": row.get("reason_detail") or f"{row.get('kill_switch_id')} is active",
            }
        )

    for row in active_rows:
        scope = _normalize_token(row.get("scope"))
        target = _safe_str(row.get("target_value")).upper()
        if scope == "GLOBAL":
            add_match(row, "global_kill_switch_active")
        elif scope == "BUY" and side == "BUY":
            add_match(row, "buy_kill_switch_active")
        elif scope == "SELL" and side == "SELL":
            add_match(row, "sell_kill_switch_active")
        elif scope == "SYMBOL" and symbol and target == symbol:
            add_match(row, "symbol_kill_switch_active")
        elif scope == "SECTOR" and sector and target == sector:
            add_match(row, "sector_kill_switch_active")
        elif scope == "ACCOUNT" and account_id and target == account_id.upper():
            add_match(row, "account_kill_switch_active")

    return {
        "active": bool(matches),
        "matches": matches,
        "reason_codes": [item["reason_code"] for item in matches],
        "reason_details": [str(item["reason_detail"]) for item in matches],
    }


def evaluate_kill_switch_triggers(trade_date: str, account_id: str, policy_id: str) -> list[dict[str, object]]:
    policy = load_us_live_risk_policy(policy_id)
    order_policy = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    market_policy = policy.get("market", {}) if isinstance(policy.get("market"), dict) else {}
    trade_date_value = date.fromisoformat(str(trade_date))
    usage_rows = fetch_us_live_daily_risk_usage_rows(
        trade_date=trade_date_value,
        policy_id=str(policy.get("policy_id") or policy_id),
        account_id=account_id,
    )
    usage_row = usage_rows[0] if usage_rows else None
    block_rows = fetch_us_live_order_block_log_rows(
        trade_date=trade_date_value,
        policy_id=str(policy.get("policy_id") or policy_id),
        account_id=account_id,
    )
    regime_rows = fetch_market_regime_rows_between(start_date=trade_date_value, end_date=trade_date_value)
    regime_row = regime_rows[0] if regime_rows else None
    triggers: list[dict[str, object]] = []

    max_failures = int(order_policy.get("max_daily_order_failures", 3) or 3)
    if isinstance(usage_row, dict):
        failed_order_count = _safe_int(usage_row.get("failed_order_count")) or 0
        blocked_order_count = _safe_int(usage_row.get("blocked_order_count")) or 0
        if failed_order_count >= max_failures:
            triggers.append(
                {
                    "scope": "GLOBAL",
                    "target_value": None,
                    "reason_code": "order_failure_limit",
                    "reason_detail": f"failed_order_count={failed_order_count} exceeded threshold={max_failures}",
                    "trigger_source": "DAILY_RISK_USAGE",
                    "trigger_ref_id": f"{trade_date}:{account_id}",
                }
            )
        if blocked_order_count >= int(order_policy.get("max_daily_order_count", 3) or 3):
            triggers.append(
                {
                    "scope": "BUY",
                    "target_value": None,
                    "reason_code": "data_error",
                    "reason_detail": f"blocked_order_count={blocked_order_count} indicates unstable pre-trade state.",
                    "trigger_source": "DAILY_RISK_USAGE",
                    "trigger_ref_id": f"{trade_date}:{account_id}",
                }
            )

    error_block_count = sum(1 for row in block_rows if _safe_str(row.get("severity")).upper() in {"ERROR", "CRITICAL"})
    if error_block_count >= max_failures:
        triggers.append(
            {
                "scope": "BUY",
                "target_value": None,
                "reason_code": "data_error",
                "reason_detail": f"pre-trade ERROR/CRITICAL block count={error_block_count} exceeded threshold={max_failures}",
                "trigger_source": "PRE_TRADE_CHECK",
                "trigger_ref_id": f"{trade_date}:{account_id}",
            }
        )

    market_regime = _safe_str((regime_row or {}).get("market_regime")).upper() if isinstance(regime_row, dict) else ""
    if bool(market_policy.get("block_bear_high_vol_regime", True)) and market_regime == "BEAR_HIGH_VOL":
        triggers.append(
            {
                "scope": "BUY",
                "target_value": None,
                "reason_code": "market_crash",
                "reason_detail": "BEAR_HIGH_VOL regime detected while policy blocks new BUY orders.",
                "trigger_source": "MARKET_REGIME_MONITOR",
                "trigger_ref_id": str(trade_date),
            }
        )

    return triggers
