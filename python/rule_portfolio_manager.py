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
        return pd.DataFrame(columns=["code", "name", "sector", "qty", "amount", "weight", "entry_price"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["qty", "amount", "weight", "entry_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
        if held and not positions.empty:
            pos = positions.loc[positions["code"] == code]
            if not pos.empty:
                current_weight = float(pd.to_numeric(pos["weight"], errors="coerce").fillna(0).max())
                current_qty = int(pd.to_numeric(pos["qty"], errors="coerce").fillna(0).max())
                current_amount = float(pd.to_numeric(pos["amount"], errors="coerce").fillna(0).max())

        sector_after = sector_exposure.get(sector, 0.0) + (new_entry_weight if not held else 0.0)
        sector_limit_pass = sector_after <= max_sector_weight or held
        cooldown_pass = code not in cooldown_codes or held
        cash_limit_pass = (current_cash_weight - new_entry_weight) >= min_cash_weight or held
        max_position_allowed = current_weight < max_position_weight

        action = "skip"
        reason = "no_action"
        target_weight = 0.0
        if held:
            if bool(row.get("market_defensive_mode")) and (float(row.get("rule_score_v2") or 0.0) < 45.0):
                action = "reduce"
                reason = "defensive_rule_score_v2_drop"
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
        elif not bool(row.get("strong_entry_signal")):
            reason = "not_strong_entry_signal"
        elif len(held_codes) + sum(1 for item in rows if item.get("portfolio_action") == "buy") >= max_positions:
            reason = "max_positions_reached"
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
            reason = "strong_entry_selected"
            target_weight = new_entry_weight
            sector_exposure[sector] = sector_after

        target_amount = total_equity * target_weight
        rows.append(
            {
                "date": latest_date.date().isoformat(),
                "code": code,
                "name": row.get("name"),
                "sector": sector,
                "portfolio_action": action,
                "portfolio_action_reason": reason,
                "target_weight": target_weight,
                "current_weight": current_weight,
                "current_qty": current_qty,
                "current_amount": current_amount,
                "target_amount": target_amount,
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
