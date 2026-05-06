from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rule_signal_builder import ENGINE_TYPE, STRATEGY_ID, ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_STATE = OUTPUT_DIR / "rule_account_paper_state.json"
DEFAULT_PLAN = OUTPUT_DIR / "rule_portfolio_plan.json"
DEFAULT_INTENTS = OUTPUT_DIR / "rule_trade_intents.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-only RULE portfolio plan and trade intents.")
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out-plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-intents-json", type=Path, default=DEFAULT_INTENTS)
    parser.add_argument("--run-mode", default=os.getenv("RULE_TRADING_RUN_MODE", "paper"))
    return parser.parse_args()


def cfg_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def cfg_int(name: str, default: int) -> int:
    return int(float(os.getenv(name, str(default))))


def _flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def load_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule signals not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["entry_signal", "strong_entry_signal", "market_defensive_mode", "sector_limit_pass", "cooldown_pass", "cash_limit_pass"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["date", "code"])


def load_state(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        return {
            "generated_at": None,
            "total_equity": 10_000_000.0,
            "cash": 10_000_000.0,
            "positions": [],
            "recent_trades": [],
        }
    return json.loads(path.read_text(encoding="utf-8-sig"))


def position_frame(state: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(state.get("positions") or [])
    if df.empty:
        return pd.DataFrame(columns=["code", "name", "sector", "qty", "amount", "weight", "entry_price", "last_price", "entry_date", "highest_price", "highest_return_since_entry"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["qty", "amount", "weight", "entry_price", "last_price", "highest_price", "highest_return_since_entry"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    return df


def evaluate_position_risk(
    row: pd.Series,
    current_weight: float,
    latest_date: pd.Timestamp,
) -> tuple[str | None, str | None, float | None, int | None, float | None, float | None, list[str]]:
    max_holding_days = cfg_int("RULE_MAX_HOLDING_DAYS", 10)
    max_holding_days_defensive = cfg_int("RULE_MAX_HOLDING_DAYS_DEFENSIVE", 7)
    max_holding_profit_buffer = cfg_float("RULE_MAX_HOLDING_DAYS_PROFIT_BUFFER", 0.02)
    profit_target_pct = cfg_float("RULE_PROFIT_TARGET_PCT", 0.0)
    stop_loss_pct = cfg_float("RULE_STOP_LOSS_PCT", 0.05)
    trailing_stop_pct = cfg_float("RULE_TRAILING_STOP_PCT", 0.04)
    trailing_stop_min_profit_pct = cfg_float("RULE_TRAILING_STOP_MIN_PROFIT_PCT", 0.03)

    reasons: list[str] = []
    entry_price = _float(row.get("entry_price"))
    last_price = _float(row.get("last_price")) or _float(row.get("close"))
    highest_price = _float(row.get("highest_price"))
    highest_return_since_entry = _float(row.get("highest_return_since_entry"))
    entry_date = pd.to_datetime(row.get("entry_date"), errors="coerce")

    current_return_pct = None
    holding_days = None
    drawdown_from_high_pct = None

    if entry_price is None or entry_price <= 0:
        reasons.append("entry_price_missing")
    if last_price is None or last_price <= 0:
        reasons.append("last_price_missing")
    if pd.isna(entry_date):
        reasons.append("entry_date_missing")
    if highest_price is None or highest_price <= 0:
        reasons.append("highest_price_missing")

    if entry_price and entry_price > 0 and last_price and last_price > 0:
        current_return_pct = (last_price / entry_price) - 1.0
    if pd.notna(entry_date):
        holding_days = max((latest_date.normalize() - pd.Timestamp(entry_date).normalize()).days, 0)
    if highest_return_since_entry is None and entry_price and entry_price > 0 and highest_price and highest_price > 0:
        highest_return_since_entry = (highest_price / entry_price) - 1.0
    if current_return_pct is not None and highest_return_since_entry is not None:
        drawdown_from_high_pct = current_return_pct - highest_return_since_entry
    elif highest_price and highest_price > 0 and last_price and last_price > 0:
        drawdown_from_high_pct = (last_price / highest_price) - 1.0

    if current_return_pct is not None and current_return_pct <= -abs(stop_loss_pct):
        return "exit", "stop_loss_exit", current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons

    if (
        highest_return_since_entry is not None
        and drawdown_from_high_pct is not None
        and highest_return_since_entry >= trailing_stop_min_profit_pct
        and drawdown_from_high_pct <= -abs(trailing_stop_pct)
    ):
        if current_return_pct is not None and current_return_pct <= 0:
            return "exit", "trailing_stop_exit", current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons
        return "reduce", "trailing_stop_reduce", current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons

    if profit_target_pct > 0 and current_return_pct is not None and current_return_pct >= profit_target_pct:
        return "reduce", "profit_target_reduce", current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons

    is_defensive = str(row.get("market_defensive_mode", "")).strip().lower() in {"true", "1", "yes"}
    effective_max_holding_days = max_holding_days_defensive if is_defensive else max_holding_days
    if holding_days is not None and holding_days > effective_max_holding_days:
        if current_return_pct is None:
            reasons.append("holding_days_exit_data_incomplete")
        elif current_return_pct < max_holding_profit_buffer:
            action = "reduce" if current_return_pct > 0 else "exit"
            holding_reason = "max_holding_days_defensive_exit" if is_defensive else "max_holding_days_exit"
            return action, holding_reason, current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons

    return None, None, current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, reasons


def recent_trade_codes(state: dict[str, Any], latest_date: pd.Timestamp, cooldown_days: int) -> set[str]:
    rows = state.get("recent_trades") or []
    blocked: set[str] = set()
    for row in rows:
        code = str(row.get("code") or "").zfill(6)
        exit_date = pd.to_datetime(row.get("exit_date") or row.get("date"), errors="coerce")
        if code and pd.notna(exit_date) and (latest_date - exit_date).days <= cooldown_days:
            blocked.add(code)
    return blocked


def build_rule_portfolio_plan(signals: pd.DataFrame, account_state: dict[str, Any], run_mode: str = "paper") -> dict[str, Any]:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    total_equity = float(account_state.get("total_equity") or 10_000_000.0)
    cash = float(account_state.get("cash") or total_equity)
    max_positions = cfg_int("RULE_MAX_POSITIONS", 5)
    max_position_weight = cfg_float("RULE_MAX_POSITION_WEIGHT", 0.15)
    new_entry_weight = cfg_float("RULE_NEW_ENTRY_WEIGHT", 0.05)
    min_cash_weight = cfg_float("RULE_MIN_CASH_WEIGHT", 0.20)
    max_sector_weight = cfg_float("RULE_MAX_SECTOR_WEIGHT", 0.35)
    cooldown_days = cfg_int("RULE_COOLDOWN_DAYS", 5)
    allow_entry_signal = _flag("RULE_ALLOW_ENTRY_SIGNAL", "0")
    entry_allow_ratio = cfg_float("RULE_ENTRY_ALLOW_RATIO", 0.6)
    positions = position_frame(account_state)
    held_codes = set(positions["code"].astype(str)) if not positions.empty else set()
    cooldown_codes = recent_trade_codes(account_state, latest_date, cooldown_days)
    current_cash_weight = cash / total_equity if total_equity else 0.0
    market_defensive_mode = bool(latest.get("market_defensive_mode", pd.Series([False])).fillna(False).any())

    sector_exposure: dict[str, float] = {}
    if not positions.empty:
        for sector, group in positions.groupby(positions.get("sector", pd.Series("(none)", index=positions.index)).fillna("(none)").astype(str)):
            sector_exposure[sector] = float(pd.to_numeric(group.get("weight"), errors="coerce").fillna(0).sum())

    rows: list[dict[str, Any]] = []
    for _, row in latest.sort_values(["rule_score_v2", "rule_score", "liquidity_score", "vol_20"], ascending=[False, False, False, True]).iterrows():
        code = str(row.get("code") or "").zfill(6)
        sector = str(row.get("sector") or "(none)")
        held = code in held_codes
        current_weight = 0.0
        current_qty = 0
        current_amount = 0.0
        position_meta: dict[str, Any] = {}
        if held and not positions.empty:
            pos = positions.loc[positions["code"] == code]
            if not pos.empty:
                current_weight = float(pd.to_numeric(pos["weight"], errors="coerce").fillna(0).max())
                current_qty = int(pd.to_numeric(pos["qty"], errors="coerce").fillna(0).max())
                current_amount = float(pd.to_numeric(pos["amount"], errors="coerce").fillna(0).max())
                position_meta = pos.iloc[0].to_dict()

        sector_after = sector_exposure.get(sector, 0.0) + (new_entry_weight if not held else 0.0)
        sector_limit_pass = sector_after <= max_sector_weight or held
        cooldown_pass = code not in cooldown_codes or held
        cash_limit_pass = (current_cash_weight - new_entry_weight) >= min_cash_weight or held
        max_position_allowed = current_weight < max_position_weight

        action = "skip"
        reason = "no_action"
        detail_reasons: list[str] = []
        target_weight = 0.0
        current_return_pct = None
        holding_days = None
        highest_return_since_entry = None
        drawdown_from_high_pct = None
        if held:
            risk_row = row.copy()
            for key, value in position_meta.items():
                risk_row[key] = value
            risk_action, risk_reason, current_return_pct, holding_days, highest_return_since_entry, drawdown_from_high_pct, missing_reasons = evaluate_position_risk(
                risk_row,
                current_weight,
                latest_date,
            )
            detail_reasons.extend(missing_reasons)
            if risk_action and risk_reason:
                action = risk_action
                reason = risk_reason
                if action == "exit":
                    target_weight = 0.0
                elif reason == "profit_target_reduce":
                    target_weight = max(current_weight * 0.7, 0.0)
                elif current_return_pct is not None and current_return_pct > 0:
                    target_weight = max(current_weight * 0.4, 0.0)
                else:
                    target_weight = max(current_weight * 0.5, 0.0)
            elif bool(row.get("market_defensive_mode")) and (float(row.get("rule_score_v2") or 0.0) < 45.0):
                action = "reduce"
                reason = "defensive_rule_score_v2_drop"
                if current_return_pct is not None and current_return_pct > 0:
                    target_weight = max(current_weight * 0.4, 0.0)
                else:
                    target_weight = max(current_weight * 0.5, 0.0)
            elif float(row.get("rule_score_v2") or 0.0) < 35.0 or bool(row.get("gap_risk_blocked")):
                action = "exit"
                reason = "rule_exit_condition"
                target_weight = 0.0
            else:
                action = "hold"
                reason = "held_and_hold_conditions_pass"
                target_weight = current_weight
        elif market_defensive_mode:
            reason = "market_defensive_mode_buy_blocked"
        elif not bool(row.get("strong_entry_signal")) and not (allow_entry_signal and bool(row.get("entry_signal"))):
            reason = "not_strong_entry_signal"
        elif len(held_codes) + sum(1 for item in rows if item.get("portfolio_action") == "buy") >= max_positions:
            reason = "max_positions_reached"
        elif not bool(row.get("strong_entry_signal")) and len(held_codes) + sum(1 for item in rows if item.get("portfolio_action") == "buy") >= int(max_positions * entry_allow_ratio):
            reason = "entry_allow_ratio_positions_reached"
        elif not sector_limit_pass:
            reason = "sector_limit_failed"
        elif not cooldown_pass:
            reason = "cooldown_failed"
        elif not cash_limit_pass:
            reason = "cash_limit_failed"
        elif not max_position_allowed:
            reason = "position_limit_failed"
        else:
            action = "buy"
            if bool(row.get("strong_entry_signal")):
                reason = "strong_entry_selected"
                target_weight = new_entry_weight
            else:
                reason = "entry_signal_relaxed_selected"
                target_weight = new_entry_weight * 0.7
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + (target_weight if not held else 0.0)

        target_amount = total_equity * target_weight
        rows.append(
            {
                "date": latest_date.date().isoformat(),
                "code": code,
                "name": row.get("name"),
                "sector": sector,
                "portfolio_action": action,
                "portfolio_action_reason": reason,
                "portfolio_action_detail_reasons": detail_reasons,
                "target_weight": target_weight,
                "current_weight": current_weight,
                "current_qty": current_qty,
                "current_amount": current_amount,
                "target_amount": target_amount,
                "entry_price": _float(position_meta.get("entry_price")) if position_meta else None,
                "last_price": _float(position_meta.get("last_price")) if position_meta else _float(row.get("close")),
                "entry_date": pd.to_datetime(position_meta.get("entry_date"), errors="coerce").date().isoformat() if position_meta and pd.notna(pd.to_datetime(position_meta.get("entry_date"), errors="coerce")) else None,
                "holding_days": holding_days,
                "current_return_pct": current_return_pct,
                "highest_price": _float(position_meta.get("highest_price")) if position_meta else None,
                "highest_return_since_entry": highest_return_since_entry,
                "drawdown_from_high_pct": drawdown_from_high_pct,
                "max_position_allowed": bool(max_position_allowed),
                "sector_limit_pass": bool(sector_limit_pass),
                "cooldown_pass": bool(cooldown_pass),
                "cash_limit_pass": bool(cash_limit_pass),
                "rule_score": _float(row.get("rule_score")),
                "rule_score_v2": _float(row.get("rule_score_v2")),
                "liquidity_score": _float(row.get("liquidity_score")),
                "vol_20": _float(row.get("vol_20")),
                "signal_strength": row.get("signal_strength"),
                "entry_signal": bool(row.get("entry_signal")),
                "strong_entry_signal": bool(row.get("strong_entry_signal")),
                "expected_entry_price": _float(row.get("expected_entry_price")),
                "gap_risk_reason": row.get("gap_risk_reason"),
                "trading_value_block_reason": row.get("trading_value_block_reason"),
                "market_defensive_mode": bool(row.get("market_defensive_mode")),
            }
        )

    plan = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": latest_date.date().isoformat(),
        "account_id": "RULE_ACCOUNT_01",
        "strategy_id": STRATEGY_ID,
        "engine_type": ENGINE_TYPE,
        "run_mode": run_mode,
        "config": {
            "max_positions": max_positions,
            "max_position_weight": max_position_weight,
            "new_entry_weight": new_entry_weight,
            "min_cash_weight": min_cash_weight,
            "max_sector_weight": max_sector_weight,
            "cooldown_days": cooldown_days,
            "allow_entry_signal": allow_entry_signal,
            "entry_allow_ratio": entry_allow_ratio,
        },
        "account_state": {
            "total_equity": total_equity,
            "cash": cash,
            "cash_weight": current_cash_weight,
            "position_count": int(len(positions)),
        },
        "items": rows,
        "summary": {
            "hold_count": sum(1 for item in rows if item["portfolio_action"] == "hold"),
            "buy_count": sum(1 for item in rows if item["portfolio_action"] == "buy"),
            "reduce_count": sum(1 for item in rows if item["portfolio_action"] == "reduce"),
            "exit_count": sum(1 for item in rows if item["portfolio_action"] == "exit"),
            "skip_count": sum(1 for item in rows if item["portfolio_action"] == "skip"),
        },
    }
    return plan


def _float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def build_trade_intents(plan: dict[str, Any]) -> dict[str, Any]:
    intents = []
    for item in plan.get("items") or []:
        action = item.get("portfolio_action")
        if action not in {"buy", "reduce", "exit"}:
            continue
        intent_type = "BUY" if action == "buy" else "TRIM" if action == "reduce" else "EXIT"
        intents.append(
            {
                "intent_id": f"{plan['as_of_date']}:{intent_type}:{item['code']}",
                "account_id": plan["account_id"],
                "strategy_id": plan["strategy_id"],
                "engine_type": plan["engine_type"],
                "run_mode": plan["run_mode"],
                "as_of_date": plan["as_of_date"],
                "code": item["code"],
                "name": item.get("name"),
                "intent_type": intent_type,
                "target_weight": item.get("target_weight"),
                "target_amount": item.get("target_amount"),
                "reason": item.get("portfolio_action_reason"),
                "detail_reasons": list(item.get("portfolio_action_detail_reasons") or []),
                "signal_strength": item.get("signal_strength"),
                "executable": False,
                "paper_only": True,
            }
        )
    return {
        "generated_at": plan["generated_at"],
        "as_of_date": plan["as_of_date"],
        "account_id": plan["account_id"],
        "strategy_id": plan["strategy_id"],
        "engine_type": plan["engine_type"],
        "run_mode": plan["run_mode"],
        "intents": intents,
        "summary": {"intent_count": len(intents)},
    }


def main() -> None:
    args = parse_args()
    signals = load_signals(args.signals_csv)
    state = load_state(args.state_json)
    plan = build_rule_portfolio_plan(signals, state, args.run_mode)
    intents = build_trade_intents(plan)
    out_plan = resolve(args.out_plan_json)
    out_intents = resolve(args.out_intents_json)
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_intents.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    out_intents.write_text(json.dumps(intents, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_plan}")
    print(f"saved {out_intents}")


if __name__ == "__main__":
    main()
