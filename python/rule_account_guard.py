from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def assert_order_allowed(order_context: dict[str, Any]) -> tuple[bool, list[str]]:
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
    min_order_amount = _float_env("RULE_MIN_ORDER_AMOUNT", 100_000.0)
    max_order_amount = _float_env("RULE_MAX_ORDER_AMOUNT", 1_000_000.0)
    if side in {"BUY", "SELL"} and order_amount < min_order_amount:
        reasons.append("final_order_amount_below_min_order_amount")
    if side in {"BUY", "SELL"} and order_qty <= 0:
        reasons.append("order_qty_zero")
    if side in {"BUY", "SELL"} and order_amount > max_order_amount:
        reasons.append("order_amount_exceeds_limit")

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

    unique_reasons = list(dict.fromkeys(reason for reason in reasons if reason and reason != "none"))
    return not unique_reasons, unique_reasons
