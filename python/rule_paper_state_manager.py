from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

DEFAULT_STATE = OUTPUT_DIR / "rule_account_paper_state.json"
DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_REPORT = OUTPUT_DIR / "rule_paper_state_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update RULE paper account state from order preview.")
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out-report-md", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule signals not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.dropna(subset=["date", "code"])


def default_state() -> dict[str, Any]:
    return {
        "generated_at": None,
        "as_of_date": None,
        "total_equity": 10_000_000.0,
        "cash": 10_000_000.0,
        "positions": [],
        "recent_trades": [],
        "cooldown_codes": [],
        "last_applied_order_ids": [],
    }


def latest_signal_map(signals: pd.DataFrame) -> tuple[pd.Timestamp, dict[str, dict[str, Any]]]:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    latest["close"] = pd.to_numeric(latest.get("close"), errors="coerce")
    latest["expected_entry_price"] = pd.to_numeric(latest.get("expected_entry_price"), errors="coerce")
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in latest.iterrows():
        mapping[str(row["code"]).zfill(6)] = row.to_dict()
    return latest_date, mapping


def normalize_positions(positions: list[dict[str, Any]], signal_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in positions:
        code = str(row.get("code") or "").zfill(6)
        if not code:
            continue
        qty = int(float(row.get("qty") or 0))
        if qty <= 0:
            continue
        signal = signal_map.get(code, {})
        last_price = _float(signal.get("close")) or _float(row.get("last_price")) or _float(row.get("entry_price")) or 0.0
        market_value = qty * last_price
        normalized.append(
            {
                "code": code,
                "name": signal.get("name") or row.get("name"),
                "sector": signal.get("sector") or row.get("sector"),
                "qty": qty,
                "entry_price": _float(row.get("entry_price")) or last_price,
                "last_price": last_price,
                "market_value": market_value,
                "amount": market_value,
                "weight": 0.0,
            }
        )
    return normalized


def apply_paper_preview(
    state: dict[str, Any],
    preview: dict[str, Any],
    signal_map: dict[str, dict[str, Any]],
    latest_date: pd.Timestamp,
) -> dict[str, Any]:
    positions = normalize_positions(state.get("positions") or [], signal_map)
    pos_by_code = {row["code"]: row for row in positions}
    cash = float(state.get("cash") or 0.0)
    recent_trades = list(state.get("recent_trades") or [])
    applied_ids = set(state.get("last_applied_order_ids") or [])
    simulated_orders: list[dict[str, Any]] = []

    for item in preview.get("items") or []:
        order_id = str(item.get("order_id") or "")
        if not order_id or order_id in applied_ids:
            continue
        side = str(item.get("side") or "NONE").upper()
        code = str(item.get("code") or "").zfill(6)
        qty = int(float(item.get("order_qty") or 0))
        price = _float(item.get("expected_execution_price")) or 0.0
        if side == "BUY" and qty > 0 and price > 0:
            filled_amount = qty * price
            if cash >= filled_amount:
                existing = pos_by_code.get(code)
                if existing:
                    total_qty = existing["qty"] + qty
                    avg_price = ((existing["entry_price"] * existing["qty"]) + filled_amount) / total_qty
                    existing["qty"] = total_qty
                    existing["entry_price"] = avg_price
                    existing["last_price"] = _float(signal_map.get(code, {}).get("close")) or price
                    existing["market_value"] = existing["qty"] * existing["last_price"]
                    existing["amount"] = existing["market_value"]
                else:
                    signal = signal_map.get(code, {})
                    last_price = _float(signal.get("close")) or price
                    pos_by_code[code] = {
                        "code": code,
                        "name": signal.get("name") or item.get("name"),
                        "sector": signal.get("sector"),
                        "qty": qty,
                        "entry_price": price,
                        "last_price": last_price,
                        "market_value": qty * last_price,
                        "amount": qty * last_price,
                        "weight": 0.0,
                    }
                cash -= filled_amount
                recent_trades.append(
                    {
                        "date": latest_date.date().isoformat(),
                        "code": code,
                        "name": item.get("name"),
                        "side": "BUY",
                        "qty": qty,
                        "price": price,
                        "amount": filled_amount,
                    }
                )
                simulated_orders.append({"order_id": order_id, "code": code, "side": "BUY", "qty": qty, "price": price})
                applied_ids.add(order_id)
        elif side == "SELL" and code in pos_by_code:
            existing = pos_by_code[code]
            sell_qty = existing["qty"] if str(item.get("portfolio_action")) == "exit" else max(existing["qty"] // 2, 1)
            sell_qty = min(sell_qty, existing["qty"])
            if sell_qty <= 0 or price <= 0:
                continue
            proceeds = sell_qty * price
            existing["qty"] -= sell_qty
            cash += proceeds
            recent_trades.append(
                {
                    "date": latest_date.date().isoformat(),
                    "exit_date": latest_date.date().isoformat(),
                    "code": code,
                    "name": item.get("name"),
                    "side": "SELL",
                    "qty": sell_qty,
                    "price": price,
                    "amount": proceeds,
                }
            )
            if existing["qty"] <= 0:
                pos_by_code.pop(code, None)
            else:
                last_price = _float(signal_map.get(code, {}).get("close")) or price
                existing["last_price"] = last_price
                existing["market_value"] = existing["qty"] * last_price
                existing["amount"] = existing["market_value"]
            simulated_orders.append({"order_id": order_id, "code": code, "side": "SELL", "qty": sell_qty, "price": price})
            applied_ids.add(order_id)

    positions = list(pos_by_code.values())
    for row in positions:
        signal = signal_map.get(row["code"], {})
        last_price = _float(signal.get("close")) or row.get("last_price") or row.get("entry_price") or 0.0
        row["last_price"] = last_price
        row["market_value"] = row["qty"] * last_price
        row["amount"] = row["market_value"]

    market_value = sum(float(row.get("market_value") or 0.0) for row in positions)
    total_equity = cash + market_value
    for row in positions:
        row["weight"] = (float(row.get("market_value") or 0.0) / total_equity) if total_equity > 0 else 0.0

    cooldown_codes = sorted(
        {
            str(row.get("code") or "").zfill(6)
            for row in recent_trades
            if str(row.get("side") or "").upper() == "SELL"
            and pd.notna(pd.to_datetime(row.get("exit_date") or row.get("date"), errors="coerce"))
            and (latest_date - pd.to_datetime(row.get("exit_date") or row.get("date"), errors="coerce")).days <= 5
        }
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": latest_date.date().isoformat(),
        "total_equity": total_equity,
        "cash": cash,
        "positions": sorted(positions, key=lambda row: float(row.get("weight") or 0.0), reverse=True),
        "recent_trades": recent_trades[-200:],
        "cooldown_codes": cooldown_codes,
        "last_applied_order_ids": sorted(applied_ids),
        "simulated_orders": simulated_orders,
    }


def render_report(state: dict[str, Any]) -> str:
    positions = state.get("positions") or []
    simulated = state.get("simulated_orders") or []
    lines = [
        "# RULE Paper State Report",
        "",
        f"- generated_at: `{state.get('generated_at')}`",
        f"- as_of_date: `{state.get('as_of_date')}`",
        f"- total_equity: `{float(state.get('total_equity') or 0.0):,.0f}`",
        f"- cash: `{float(state.get('cash') or 0.0):,.0f}`",
        f"- position_count: `{len(positions)}`",
        f"- simulated_order_count: `{len(simulated)}`",
        f"- cooldown_code_count: `{len(state.get('cooldown_codes') or [])}`",
        "",
        "## Positions",
        "",
    ]
    if not positions:
        lines.append("_No open positions._")
    else:
        lines.extend(["| code | name | qty | entry_price | last_price | market_value | weight |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in positions:
            lines.append(
                "| {code} | {name} | {qty} | {entry:.0f} | {last:.0f} | {mv:.0f} | {wt:.2%} |".format(
                    code=row.get("code") or "",
                    name=row.get("name") or "",
                    qty=int(row.get("qty") or 0),
                    entry=float(row.get("entry_price") or 0.0),
                    last=float(row.get("last_price") or 0.0),
                    mv=float(row.get("market_value") or 0.0),
                    wt=float(row.get("weight") or 0.0),
                )
            )
    lines.extend(["", "## Simulated Orders", ""])
    if not simulated:
        lines.append("_No simulated orders applied._")
    else:
        lines.extend(["| order_id | code | side | qty | price |", "| --- | --- | --- | ---: | ---: |"])
        for row in simulated:
            lines.append(
                "| {order_id} | {code} | {side} | {qty} | {price:.0f} |".format(
                    order_id=row.get("order_id") or "",
                    code=row.get("code") or "",
                    side=row.get("side") or "",
                    qty=int(row.get("qty") or 0),
                    price=float(row.get("price") or 0.0),
                )
            )
    lines.extend(["", "## Cooldown Codes", ""])
    cooldown_codes = state.get("cooldown_codes") or []
    if not cooldown_codes:
        lines.append("_No cooldown codes._")
    else:
        lines.extend(f"- `{code}`" for code in cooldown_codes)
    lines.append("")
    return "\n".join(lines)


def _float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    state = load_json(args.state_json, default_state())
    preview = load_json(resolve(args.order_preview_json), {})
    signals = load_signals(args.signals_csv)
    latest_date, signal_map = latest_signal_map(signals)
    updated = apply_paper_preview(state, preview, signal_map, latest_date)
    out_json = resolve(args.out_json)
    out_md = resolve(args.out_report_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_report(updated), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
