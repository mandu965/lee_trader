from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import rule_order_preview_builder
import rule_paper_state_manager
import rule_portfolio_manager
from rule_signal_builder import ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "rule_paper_state_validation_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "rule_paper_state_validation_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RULE paper hold/exit/cooldown scenarios.")
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--run-mode", default="paper")
    return parser.parse_args()


def load_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["entry_signal", "strong_entry_signal", "gap_risk_blocked", "market_defensive_mode"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["date", "code"])


def latest_rows(signals: pd.DataFrame) -> pd.DataFrame:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    latest["rule_score_v2"] = pd.to_numeric(latest.get("rule_score_v2"), errors="coerce")
    latest["expected_entry_price"] = pd.to_numeric(latest.get("expected_entry_price"), errors="coerce")
    latest["close"] = pd.to_numeric(latest.get("close"), errors="coerce")
    return latest


def pick_strong_codes(latest: pd.DataFrame) -> list[str]:
    rows = latest.loc[latest["strong_entry_signal"]].sort_values(["rule_score_v2", "rule_score"], ascending=[False, False])
    return rows["code"].astype(str).tolist()


def pick_exit_code(latest: pd.DataFrame) -> str:
    rows = latest.loc[(latest["gap_risk_blocked"]) | (latest["rule_score_v2"] < 35)].sort_values(
        ["gap_risk_blocked", "rule_score_v2"], ascending=[False, True]
    )
    if rows.empty:
        raise RuntimeError("no exit candidate found in latest signals")
    return str(rows.iloc[0]["code"]).zfill(6)


def row_map(latest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["code"]).zfill(6): row.to_dict() for _, row in latest.iterrows()}


def build_position(code: str, row: dict[str, Any], qty: int = 10) -> dict[str, Any]:
    price = float(row.get("close") or row.get("expected_entry_price") or 0.0)
    amount = qty * price
    return {
        "code": code,
        "name": row.get("name"),
        "sector": row.get("sector"),
        "qty": qty,
        "amount": amount,
        "weight": amount / 10_000_000.0,
        "entry_price": price,
        "last_price": price,
    }


def summarize_plan(plan: dict[str, Any], codes: list[str]) -> dict[str, Any]:
    items = {str(item.get("code") or "").zfill(6): item for item in plan.get("items") or []}
    return {code: {"action": items.get(code, {}).get("portfolio_action"), "reason": items.get(code, {}).get("portfolio_action_reason")} for code in codes}


def scenario_hold_repeat(latest: pd.DataFrame, run_mode: str) -> dict[str, Any]:
    rows = row_map(latest)
    strong_codes = pick_strong_codes(latest)[:2]
    state = rule_paper_state_manager.default_state()
    positions = [build_position(code, rows[code], qty=3) for code in strong_codes]
    total_market = sum(float(p["amount"]) for p in positions)
    state["cash"] = 10_000_000.0 - total_market
    state["total_equity"] = 10_000_000.0
    state["positions"] = positions
    state["last_applied_order_ids"] = [f"RULE-PREVIEW-{latest['date'].max().date()}-{idx+1:03d}" for idx in range(len(positions))]
    plan = rule_portfolio_manager.build_rule_portfolio_plan(latest, state, run_mode)
    preview = rule_order_preview_builder.build_rule_order_preview(plan, run_mode)
    return {
        "scenario": "hold_repeat_run",
        "plan_summary": plan.get("summary"),
        "buy_preview_count": (preview.get("summary") or {}).get("buy_preview_count", 0),
        "tracked_codes": summarize_plan(plan, strong_codes),
        "passed": plan.get("summary", {}).get("hold_count", 0) >= len(strong_codes) and (preview.get("summary") or {}).get("buy_preview_count", 0) == 0,
    }


def scenario_exit_flow(latest: pd.DataFrame, run_mode: str) -> dict[str, Any]:
    rows = row_map(latest)
    exit_code = pick_exit_code(latest)
    state = rule_paper_state_manager.default_state()
    position = build_position(exit_code, rows[exit_code], qty=10)
    state["positions"] = [position]
    state["cash"] = 10_000_000.0 - float(position["amount"])
    state["total_equity"] = 10_000_000.0
    plan = rule_portfolio_manager.build_rule_portfolio_plan(latest, state, run_mode)
    preview = rule_order_preview_builder.build_rule_order_preview(plan, run_mode)
    latest_date, signal_map = rule_paper_state_manager.latest_signal_map(latest)
    updated = rule_paper_state_manager.apply_paper_preview(state, preview, signal_map, latest_date)
    plan_item = summarize_plan(plan, [exit_code]).get(exit_code, {})
    sell_preview_count = sum(1 for row in preview.get("items") or [] if row.get("side") == "SELL")
    remaining_codes = [str(row.get("code") or "").zfill(6) for row in updated.get("positions") or []]
    cooldown_codes = [str(code).zfill(6) for code in (updated.get("cooldown_codes") or [])]
    return {
        "scenario": "exit_flow",
        "code": exit_code,
        "plan_item": plan_item,
        "sell_preview_count": sell_preview_count,
        "remaining_position_codes": remaining_codes,
        "cooldown_codes": cooldown_codes,
        "passed": plan_item.get("action") == "exit" and sell_preview_count >= 1 and exit_code not in remaining_codes and exit_code in cooldown_codes,
    }


def scenario_reduce_flow(latest: pd.DataFrame, run_mode: str) -> dict[str, Any]:
    rows = row_map(latest)
    candidates = (
        latest.loc[(latest["rule_score_v2"] < 45) & (latest["rule_score_v2"] >= 35)]
        .sort_values("rule_score_v2")
    )
    if candidates.empty:
        raise RuntimeError("no reduce candidate found in latest signals")
    code = str(candidates.iloc[0]["code"]).zfill(6)
    state = rule_paper_state_manager.default_state()
    position = build_position(code, rows[code], qty=10)
    state["positions"] = [position]
    state["cash"] = 10_000_000.0 - float(position["amount"])
    state["total_equity"] = 10_000_000.0
    scenario_latest = latest.copy()
    scenario_latest["market_defensive_mode"] = True
    scenario_latest["market_entry_allowed"] = False
    plan = rule_portfolio_manager.build_rule_portfolio_plan(scenario_latest, state, run_mode)
    preview = rule_order_preview_builder.build_rule_order_preview(plan, run_mode)
    latest_date, signal_map = rule_paper_state_manager.latest_signal_map(scenario_latest)
    updated = rule_paper_state_manager.apply_paper_preview(state, preview, signal_map, latest_date)
    plan_item = summarize_plan(plan, [code]).get(code, {})
    sell_preview_count = sum(1 for row in preview.get("items") or [] if row.get("side") == "SELL")
    updated_map = {str(row.get("code") or "").zfill(6): row for row in updated.get("positions") or []}
    remaining_qty = int(float(updated_map.get(code, {}).get("qty") or 0))
    return {
        "scenario": "reduce_flow",
        "code": code,
        "plan_item": plan_item,
        "sell_preview_count": sell_preview_count,
        "remaining_qty": remaining_qty,
        "cooldown_codes": [str(item).zfill(6) for item in (updated.get("cooldown_codes") or [])],
        "passed": plan_item.get("action") == "reduce" and sell_preview_count >= 1 and 0 < remaining_qty < int(position["qty"]),
    }


def scenario_cooldown_block(latest: pd.DataFrame, run_mode: str) -> dict[str, Any]:
    rows = row_map(latest)
    strong_codes = pick_strong_codes(latest)
    code = strong_codes[0]
    latest_date = latest["date"].max()
    state = rule_paper_state_manager.default_state()
    state["recent_trades"] = [
        {
            "date": latest_date.date().isoformat(),
            "exit_date": latest_date.date().isoformat(),
            "code": code,
            "name": rows[code].get("name"),
            "side": "SELL",
            "qty": 5,
            "price": float(rows[code].get("close") or 0.0),
            "amount": float(rows[code].get("close") or 0.0) * 5,
        }
    ]
    plan = rule_portfolio_manager.build_rule_portfolio_plan(latest, state, run_mode)
    plan_item = summarize_plan(plan, [code]).get(code, {})
    return {
        "scenario": "cooldown_block",
        "code": code,
        "plan_item": plan_item,
        "passed": plan_item.get("action") == "skip" and plan_item.get("reason") == "cooldown_failed",
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# RULE Paper State Validation",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- signal_date: `{report['signal_date']}`",
        "",
        "| scenario | passed | key_result |",
        "| --- | --- | --- |",
    ]
    for row in report["scenarios"]:
        key_result = row.get("key_result") or ""
        lines.append(f"| {row['scenario']} | {'yes' if row['passed'] else 'no'} | {key_result} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    signals = load_signals(args.signals_csv)
    latest = latest_rows(signals)
    latest_date = latest["date"].max()

    hold = scenario_hold_repeat(latest, args.run_mode)
    hold["key_result"] = f"hold_count={hold['plan_summary']['hold_count']}, buy_preview={hold['buy_preview_count']}"

    exit_flow = scenario_exit_flow(latest, args.run_mode)
    exit_flow["key_result"] = f"action={exit_flow['plan_item']['action']}, sell_preview={exit_flow['sell_preview_count']}"

    reduce_flow = scenario_reduce_flow(latest, args.run_mode)
    reduce_flow["key_result"] = f"action={reduce_flow['plan_item']['action']}, remaining_qty={reduce_flow['remaining_qty']}"

    cooldown = scenario_cooldown_block(latest, args.run_mode)
    cooldown["key_result"] = f"action={cooldown['plan_item']['action']}, reason={cooldown['plan_item']['reason']}"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": latest_date.date().isoformat(),
        "scenarios": [hold, exit_flow, reduce_flow, cooldown],
    }

    out_json = resolve(args.out_json)
    out_md = resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_report(report), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
