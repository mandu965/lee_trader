from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kis_client import KISClient
from kis_live_account import inquire_balance, summarize_cash
from rule_market_open_snapshot import resolve_rule_account_env
from rule_signal_builder import ROOT, resolve
from sync_live_account_holdings import normalize_holdings


OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

DEFAULT_STATE_JSON = OUTPUT_DIR / "rule_account_live_state.json"
DEFAULT_HOLDINGS_CSV = DATA_DIR / "rule_account_live_holdings.csv"
DEFAULT_RESULTS_JSON = OUTPUT_DIR / "rule_execution_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync RULE live account snapshot into state JSON.")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--out-state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--out-holdings-csv", type=Path, default=DEFAULT_HOLDINGS_CSV)
    parser.add_argument("--execution-results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _load_recent_trades(path: Path) -> list[dict[str, Any]]:
    resolved = resolve(path)
    if not resolved.exists():
        return []
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    trades: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        status = str(item.get("order_status") or "")
        filled_qty = int(float(item.get("filled_qty") or 0))
        if status not in {"filled", "partial_filled", "simulated_filled"} or filled_qty <= 0:
            continue
        trades.append(
            {
                "date": str(item.get("filled_at") or item.get("submitted_at") or payload.get("as_of_date") or "")[:10],
                "side": str(item.get("side") or "").upper(),
                "code": str(item.get("code") or "").zfill(6),
                "qty": filled_qty,
                "price": _float(item.get("avg_fill_price")),
                "order_id": item.get("order_id"),
                "broker_order_id": item.get("broker_order_id"),
            }
        )
    return trades[-200:]


def build_rule_live_account_state(*, as_of_date: str, execution_results_json: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    client = KISClient.from_rule_env()
    client.issue_access_token()
    account = resolve_rule_account_env()
    holdings_df, summary_df = inquire_balance(client, account)
    holdings = normalize_holdings(holdings_df)
    cash_summary = summarize_cash(summary_df)

    positions = [
        {
            "code": str(row.get("code") or "").zfill(6),
            "name": row.get("name"),
            "sector": None,
            "qty": int(float(row.get("qty") or 0)),
            "entry_price": float(row.get("avg_price") or 0.0),
            "last_price": float(row.get("current_price") or 0.0),
            "market_value": float(row.get("eval_amount") or 0.0),
            "amount": float(row.get("eval_amount") or 0.0),
            "weight": float(row.get("weight") or 0.0),
            "pnl_amount": float(row.get("pnl_amount") or 0.0) if row.get("pnl_amount") is not None else None,
            "pnl_pct": float(row.get("pnl_pct") or 0.0) if row.get("pnl_pct") is not None else None,
        }
        for row in holdings.to_dict(orient="records")
    ]
    total_equity = _float(cash_summary.get("tot_evlu_amt")) or (
        (_float(cash_summary.get("dnca_tot_amt")) or 0.0) + sum(float(row.get("market_value") or 0.0) for row in positions)
    )
    state = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date or datetime.now().strftime("%Y-%m-%d"),
        "account_mode": "live",
        "total_equity": total_equity,
        "cash": _float(cash_summary.get("dnca_tot_amt")) or 0.0,
        "positions": positions,
        "recent_trades": _load_recent_trades(execution_results_json),
        "cooldown_codes": [],
        "last_applied_order_ids": [],
        "cash_summary": cash_summary,
    }
    return state, holdings


def main() -> None:
    args = parse_args()
    state, holdings = build_rule_live_account_state(
        as_of_date=str(args.as_of_date or datetime.now().strftime("%Y-%m-%d")),
        execution_results_json=args.execution_results_json,
    )
    out_state = resolve(args.out_state_json)
    out_holdings = resolve(args.out_holdings_csv)
    out_state.parent.mkdir(parents=True, exist_ok=True)
    out_holdings.parent.mkdir(parents=True, exist_ok=True)
    out_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    holdings.to_csv(out_holdings, index=False, encoding="utf-8-sig")
    print(f"saved {out_state}")
    print(f"saved {out_holdings}")


if __name__ == "__main__":
    main()
