from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rule_account_guard import evaluate_rule_order_guard, resolve_trading_account, validate_account_profile
from rule_signal_builder import ENGINE_TYPE, STRATEGY_ID, ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"

DEFAULT_PLAN = OUTPUT_DIR / "rule_portfolio_plan.json"
DEFAULT_OUT = OUTPUT_DIR / "rule_order_preview.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-only RULE order preview.")
    parser.add_argument("--portfolio-plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-mode", default=os.getenv("RULE_TRADING_RUN_MODE", "paper"))
    return parser.parse_args()


def cfg_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def optional_cfg_float(name: str) -> float | None:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def optional_cfg_int(name: str) -> int | None:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError:
        return None
    return value if value > 0 else None


def load_plan(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule portfolio plan not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def calculate_order_quantity(order_amount: float, execution_price: float | None) -> int:
    if not execution_price or execution_price <= 0:
        return 0
    return max(int(math.floor(order_amount / execution_price)), 0)


def validate_order_size(order_qty: int, order_amount: float, min_order_amount: float) -> tuple[bool, str]:
    if order_amount < min_order_amount:
        return False, "final_order_amount_below_min_order_amount"
    if order_qty <= 0:
        return False, "order_qty_zero"
    return True, "none"


def build_rule_order_preview(plan: dict[str, Any], run_mode: str = "paper") -> dict[str, Any]:
    min_order_amount = cfg_float("RULE_MIN_ORDER_AMOUNT", 100_000.0)
    market_order_enabled = str(os.getenv("MARKET_ORDER_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    account_profile = resolve_trading_account(ENGINE_TYPE, run_mode)
    account_profile_ok, account_profile_reasons = validate_account_profile(account_profile)
    account_state = plan.get("account_state") or {}
    total_equity = float(account_state.get("total_equity") or 0.0)
    cash = float(account_state.get("cash") or 0.0)
    min_cash_weight = float((plan.get("config") or {}).get("min_cash_weight", 0.20))
    available_buying_power = max(0.0, cash - total_equity * min_cash_weight)
    pilot_max_order_amount = optional_cfg_float("RULE_PILOT_MAX_ORDER_AMOUNT") if run_mode == "pilot" else None
    pilot_max_order_qty = optional_cfg_int("RULE_PILOT_MAX_ORDER_QTY") if run_mode == "pilot" else None

    items: list[dict[str, Any]] = []
    for idx, item in enumerate(plan.get("items") or [], 1):
        action = item.get("portfolio_action")
        side = "BUY" if action == "buy" else "SELL" if action in {"reduce", "exit"} else "NONE"
        expected_price = item.get("expected_entry_price")
        try:
            expected_price = float(expected_price) if expected_price is not None else None
        except Exception:
            expected_price = None

        target_amount = float(item.get("target_amount") or 0.0)
        current_amount = float(item.get("current_amount") or 0.0)
        order_amount = 0.0
        if side == "BUY":
            order_amount = min(target_amount, available_buying_power)
        elif side == "SELL":
            if action == "exit":
                order_amount = current_amount
            elif action == "reduce":
                order_amount = max(current_amount - target_amount, 0.0)
            else:
                order_amount = current_amount
        raw_order_amount = order_amount
        if pilot_max_order_amount is not None and side in {"BUY", "SELL"}:
            order_amount = min(order_amount, pilot_max_order_amount)
        raw_order_qty = calculate_order_quantity(order_amount, expected_price)
        order_qty = raw_order_qty
        if pilot_max_order_qty is not None and side in {"BUY", "SELL"}:
            order_qty = min(order_qty, pilot_max_order_qty)
            if expected_price and expected_price > 0:
                order_amount = min(order_amount, float(order_qty) * expected_price)
        size_ok, size_reason = validate_order_size(order_qty, order_amount, min_order_amount)

        order_type = "limit"
        limit_price = None
        if expected_price:
            if side == "BUY":
                limit_price = int(expected_price * 1.01)
            elif side == "SELL":
                limit_price = int(expected_price * 0.99)

        gap_risk_reason = item.get("gap_risk_reason")
        trading_value_block_reason = item.get("trading_value_block_reason")
        order_context = {
            "account_id": account_profile.account_id,
            "strategy_id": STRATEGY_ID,
            "engine_type": ENGINE_TYPE,
            "run_mode": run_mode,
            "as_of_date": plan.get("as_of_date"),
            "execution_date": datetime.now().strftime("%Y-%m-%d"),
            "side": side,
            "code": item.get("code"),
            "order_qty": order_qty,
            "order_amount": order_amount,
            "reference_price": expected_price,
            "signal_strength": item.get("signal_strength"),
            "market_defensive_mode": bool(item.get("market_defensive_mode")),
            "gap_risk_blocked": gap_risk_reason not in {None, "", "none"},
            "gap_risk_reason": gap_risk_reason,
            "trading_value_pass": trading_value_block_reason in {None, "", "none"},
            "trading_value_block_reason": trading_value_block_reason,
            "sector_limit_pass": bool(item.get("sector_limit_pass", True)),
            "cooldown_pass": bool(item.get("cooldown_pass", True)),
            "cash_limit_pass": bool(item.get("cash_limit_pass", True)),
            "daily_order_amount_used": 0.0,
        }
        if not size_ok and side in {"BUY", "SELL"}:
            # Keep the original sizing reason when it is more specific than the guard default.
            order_context["order_qty"] = order_qty
            order_context["order_amount"] = order_amount
        order_allowed, block_reasons, guard_details = evaluate_rule_order_guard(order_context)
        if not size_ok and side in {"BUY", "SELL"} and size_reason not in block_reasons:
            block_reasons.append(size_reason)

        items.append(
            {
                "order_id": f"RULE-PREVIEW-{plan.get('as_of_date')}-{idx:03d}",
                "parent_order_id": None,
                "account_id": account_profile.account_id,
                "strategy_id": STRATEGY_ID,
                "engine_type": ENGINE_TYPE,
                "run_mode": run_mode,
                "symbol": item.get("code"),
                "code": item.get("code"),
                "name": item.get("name"),
                "side": side,
                "order_type": order_type,
                "market_order_enabled": bool(market_order_enabled),
                "expected_execution_price": expected_price,
                "limit_price": limit_price,
                "order_qty": order_qty,
                "order_amount": order_amount,
                "portfolio_action": action,
                "portfolio_action_reason": item.get("portfolio_action_reason"),
                "portfolio_action_detail_reasons": list(item.get("portfolio_action_detail_reasons") or []),
                "holding_days": item.get("holding_days"),
                "current_return_pct": item.get("current_return_pct"),
                "highest_return_since_entry": item.get("highest_return_since_entry"),
                "drawdown_from_high_pct": item.get("drawdown_from_high_pct"),
                "order_allowed": order_allowed,
                "order_block_reason": ";".join(dict.fromkeys(block_reasons)) if block_reasons else "none",
                "common_risk_allowed": bool(guard_details.get("common_risk_allowed", True)),
                "common_risk_block_reasons": list(guard_details.get("common_risk_block_reasons") or []),
                "common_risk_snapshot": dict(guard_details.get("common_risk_snapshot") or {}),
                "signal_strength": item.get("signal_strength"),
                "gap_risk_reason": gap_risk_reason,
                "trading_value_block_reason": trading_value_block_reason,
                "sector_limit_pass": bool(item.get("sector_limit_pass", True)),
                "cooldown_pass": bool(item.get("cooldown_pass", True)),
                "cash_limit_pass": bool(item.get("cash_limit_pass", True)),
                "order_status": "planned",
                "retry_count": 0,
                "fallback_enabled": False,
                "fallback_type": "none",
                "fallback_reason": None,
                "original_limit_price": limit_price,
                "fallback_limit_price": None,
                "fallback_order_allowed": False,
                "fallback_block_reason": None,
                "pilot_order_amount_cap_applied": bool(
                    pilot_max_order_amount is not None and side in {"BUY", "SELL"} and raw_order_amount > order_amount
                ),
                "pilot_order_qty_cap_applied": bool(
                    pilot_max_order_qty is not None and side in {"BUY", "SELL"} and raw_order_qty > order_qty
                ),
                "filled_qty": 0,
                "unfilled_qty": order_qty,
                "filled_amount": 0.0,
                "avg_fill_price": None,
                "execution_checked_at": None,
                "reconciliation_status": "not_started",
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": plan.get("as_of_date"),
        "account_id": account_profile.account_id,
        "strategy_id": STRATEGY_ID,
        "engine_type": ENGINE_TYPE,
        "run_mode": run_mode,
        "paper_only": run_mode == "paper",
        "account_profile": account_profile.to_dict(),
        "account_profile_valid": account_profile_ok,
        "account_profile_block_reasons": account_profile_reasons,
        "trading_day_valid": None,
        "trading_day_reason": "not_checked_in_after_close_preview",
        "market_data_available": None,
        "api_health_status": "not_called_paper_preview",
        "api_failure_reason": None,
        "previous_reconciliation_found": None,
        "previous_reconciliation_status": "not_checked",
        "new_orders_blocked_by_reconciliation": False,
        "reconciliation_block_reason": None,
        "items": items,
        "summary": {
            "request_count": len(items),
            "buy_preview_count": sum(1 for row in items if row["side"] == "BUY"),
            "sell_preview_count": sum(1 for row in items if row["side"] == "SELL"),
            "order_allowed_count": sum(1 for row in items if row["order_allowed"]),
        },
    }


def main() -> None:
    args = parse_args()
    plan = load_plan(args.portfolio_plan_json)
    payload = build_rule_order_preview(plan, args.run_mode)
    out = resolve(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
