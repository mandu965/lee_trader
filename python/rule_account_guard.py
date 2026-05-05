from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common_live_risk_guard import evaluate_common_buy_guard
from rule_signal_builder import ENGINE_TYPE, ROOT, STRATEGY_ID


RULE_ACCOUNT_ID = "RULE_ACCOUNT_01"
VALID_RUN_MODES = {"paper", "pilot", "live"}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env", override=False)


_load_env()


@dataclass(frozen=True)
class AccountProfile:
    account_id: str
    strategy_id: str
    engine_type: str
    run_mode: str
    cano: str | None
    acnt_prdt_cd: str | None
    paper_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "engine_type": self.engine_type,
            "run_mode": self.run_mode,
            "cano_configured": bool(self.cano),
            "acnt_prdt_cd_configured": bool(self.acnt_prdt_cd),
            "paper_only": self.paper_only,
        }


def _flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _optional_positive_float_env(name: str) -> float | None:
    try:
        value = float(str(os.getenv(name, "")).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _optional_positive_int_env(name: str) -> int | None:
    try:
        value = int(float(str(os.getenv(name, "")).strip()))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _normalize_run_mode(run_mode: str | None) -> str:
    mode = str(run_mode or os.getenv("RULE_TRADING_RUN_MODE", "paper")).strip().lower()
    return mode if mode in VALID_RUN_MODES else mode


def resolve_trading_account(engine_type: str = ENGINE_TYPE, run_mode: str | None = None) -> AccountProfile:
    mode = _normalize_run_mode(run_mode)
    if engine_type != ENGINE_TYPE:
        return AccountProfile(
            account_id="UNKNOWN",
            strategy_id=STRATEGY_ID,
            engine_type=engine_type,
            run_mode=mode,
            cano=None,
            acnt_prdt_cd=None,
            paper_only=True,
        )

    cano = os.getenv("KIS_RULE_CANO")
    acnt_prdt_cd = os.getenv("KIS_RULE_ACNT_PRDT_CD")
    return AccountProfile(
        account_id=RULE_ACCOUNT_ID,
        strategy_id=STRATEGY_ID,
        engine_type=ENGINE_TYPE,
        run_mode=mode,
        cano=cano,
        acnt_prdt_cd=acnt_prdt_cd,
        paper_only=mode == "paper",
    )


def validate_account_profile(account_profile: AccountProfile | dict[str, Any]) -> tuple[bool, list[str]]:
    profile = account_profile if isinstance(account_profile, AccountProfile) else AccountProfile(
        account_id=str(account_profile.get("account_id") or ""),
        strategy_id=str(account_profile.get("strategy_id") or STRATEGY_ID),
        engine_type=str(account_profile.get("engine_type") or ""),
        run_mode=str(account_profile.get("run_mode") or ""),
        cano=account_profile.get("cano"),
        acnt_prdt_cd=account_profile.get("acnt_prdt_cd"),
        paper_only=bool(account_profile.get("paper_only")),
    )
    reasons: list[str] = []
    if profile.account_id != RULE_ACCOUNT_ID:
        reasons.append("account_id_mismatch")
    if profile.strategy_id != STRATEGY_ID:
        reasons.append("strategy_id_mismatch")
    if profile.engine_type != ENGINE_TYPE:
        reasons.append("engine_type_mismatch")
    if profile.run_mode not in VALID_RUN_MODES:
        reasons.append("invalid_run_mode")
    if profile.run_mode in {"pilot", "live"} and not profile.cano:
        reasons.append("kis_rule_cano_missing")
    if profile.run_mode in {"pilot", "live"} and not profile.acnt_prdt_cd:
        reasons.append("kis_rule_acnt_prdt_cd_missing")
    return not reasons, reasons


def _build_common_risk_context(order_context: dict[str, Any], run_mode: str) -> dict[str, Any]:
    context = {
        "now": order_context.get("now"),
        "as_of_date": order_context.get("as_of_date"),
        "execution_date": order_context.get("execution_date"),
        "trade_date": order_context.get("trade_date"),
        "side": order_context.get("side"),
        "code": order_context.get("code") or order_context.get("symbol"),
        "order_amount": order_context.get("order_amount"),
        "order_qty": order_context.get("order_qty"),
        "reference_price": order_context.get("reference_price"),
        "market_defensive_mode": order_context.get("market_defensive_mode"),
        "run_mode": run_mode,
    }
    for key in [
        "holdings_csv",
        "balance_json",
        "fills_json",
        "market_status_csv",
        "holdings_generated_at",
        "fills_generated_at",
        "daily_buy_amount_used",
        "weekly_buy_amount_used",
        "daily_loss_pct",
        "weekly_loss_pct",
    ]:
        if key in order_context:
            context[key] = order_context.get(key)
    return context


def evaluate_rule_order_guard(order_context: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    run_mode = _normalize_run_mode(order_context.get("run_mode"))
    account = resolve_trading_account(str(order_context.get("engine_type") or ENGINE_TYPE), run_mode)
    account_ok, account_reasons = validate_account_profile(account)
    reasons = list(account_reasons)

    side = str(order_context.get("side") or "NONE").upper()
    if side == "NONE":
        reasons.append("no_order_action")
    if not account_ok:
        pass
    if str(order_context.get("account_id") or "") != RULE_ACCOUNT_ID:
        reasons.append("account_id_mismatch")
    if str(order_context.get("strategy_id") or STRATEGY_ID) != STRATEGY_ID:
        reasons.append("strategy_id_mismatch")
    if str(order_context.get("engine_type") or ENGINE_TYPE) != ENGINE_TYPE:
        reasons.append("engine_type_mismatch")
    if run_mode not in VALID_RUN_MODES:
        reasons.append("invalid_run_mode")

    if run_mode == "paper":
        if side in {"BUY", "SELL"}:
            reasons.append("paper_mode_no_order_submission")
    else:
        if not _flag("RULE_LIVE_ENABLED", "0"):
            reasons.append("rule_live_disabled")
        if not _flag("RULE_ORDER_SUBMIT_ENABLED", "0"):
            reasons.append("rule_order_submit_disabled")

    if _flag("RULE_KILL_SWITCH", "0"):
        reasons.append("kill_switch_on")

    order_amount = float(order_context.get("order_amount") or 0.0)
    order_qty = int(float(order_context.get("order_qty") or 0))
    daily_order_amount_used = float(order_context.get("daily_order_amount_used") or 0.0)
    min_order_amount = _float_env("RULE_MIN_ORDER_AMOUNT", 100_000.0)
    max_order_amount = _float_env("RULE_MAX_ORDER_AMOUNT", 1_000_000.0)
    max_daily_order_amount = _optional_positive_float_env("RULE_MAX_DAILY_ORDER_AMOUNT")
    pilot_max_order_amount = _optional_positive_float_env("RULE_PILOT_MAX_ORDER_AMOUNT")
    pilot_max_order_qty = _optional_positive_int_env("RULE_PILOT_MAX_ORDER_QTY")
    if side in {"BUY", "SELL"} and order_amount < min_order_amount:
        reasons.append("final_order_amount_below_min_order_amount")
    if side in {"BUY", "SELL"} and order_qty <= 0:
        reasons.append("order_qty_zero")
    if side in {"BUY", "SELL"} and order_amount > max_order_amount:
        reasons.append("order_amount_exceeds_limit")
    if side in {"BUY", "SELL"} and max_daily_order_amount is not None and daily_order_amount_used + order_amount > max_daily_order_amount:
        reasons.append("daily_order_amount_exceeds_limit")
    if run_mode == "pilot" and side in {"BUY", "SELL"}:
        if pilot_max_order_amount is not None and order_amount > pilot_max_order_amount:
            reasons.append("pilot_order_amount_exceeds_limit")
        if pilot_max_order_qty is not None and order_qty > pilot_max_order_qty:
            reasons.append("pilot_order_qty_exceeds_limit")

    if side == "BUY":
        if order_context.get("signal_strength") != "strong_entry":
            reasons.append("buy_requires_strong_entry")
        if bool(order_context.get("market_defensive_mode")):
            reasons.append("market_defensive_mode")
        if bool(order_context.get("gap_risk_blocked")):
            reasons.append(str(order_context.get("gap_risk_reason") or "gap_risk_blocked"))
        if not bool(order_context.get("trading_value_pass", True)):
            reasons.append(str(order_context.get("trading_value_block_reason") or "trading_value_failed"))
        if not bool(order_context.get("sector_limit_pass", True)):
            reasons.append("sector_limit_failed")
        if not bool(order_context.get("cooldown_pass", True)):
            reasons.append("cooldown_failed")
        if not bool(order_context.get("cash_limit_pass", True)):
            reasons.append("cash_limit_failed")

    base_reasons = list(dict.fromkeys(reason for reason in reasons if reason and reason != "none"))

    common_allowed = True
    common_reasons: list[str] = []
    common_snapshot: dict[str, Any] = {
        "side": side,
        "buy_allowed": True,
        "block_reasons": [],
        "bypass_reason": "side_not_buy",
    }
    if side == "BUY":
        common_allowed, common_reasons, common_snapshot = evaluate_common_buy_guard(
            _build_common_risk_context(order_context, run_mode)
        )

    merged_reasons = list(dict.fromkeys([*base_reasons, *common_reasons]))
    details = {
        "base_allowed": not base_reasons,
        "base_block_reasons": base_reasons,
        "common_risk_allowed": common_allowed,
        "common_risk_block_reasons": common_reasons,
        "common_risk_snapshot": common_snapshot,
        "guard_limits": {
            "min_order_amount": min_order_amount,
            "max_order_amount": max_order_amount,
            "max_daily_order_amount": max_daily_order_amount,
            "daily_order_amount_used": daily_order_amount_used,
            "pilot_max_order_amount": pilot_max_order_amount,
            "pilot_max_order_qty": pilot_max_order_qty,
        },
    }
    return not merged_reasons, merged_reasons, details


def assert_order_allowed(order_context: dict[str, Any]) -> tuple[bool, list[str]]:
    allowed, reasons, _ = evaluate_rule_order_guard(order_context)
    return allowed, reasons
