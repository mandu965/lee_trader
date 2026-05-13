from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import parse_iso_date
from python.us.us_db import (
    ensure_us_live_risk_tables,
    fetch_us_live_daily_risk_usage_rows,
    fetch_us_live_kill_switch_rows,
    upsert_us_live_daily_risk_usage_rows,
    upsert_us_live_kill_switch_rows,
)
from utils.us_live_risk_policy import load_us_live_risk_policy


def _build_default_rows(policy_id: str, account_id: str, trade_date: date) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kill_rows = [
        {
            "kill_switch_id": "US_LIVE_GLOBAL_KILL",
            "scope": "GLOBAL",
            "target_value": "ALL",
            "is_active": False,
            "reason_code": None,
            "reason_detail": None,
            "activated_at": None,
            "activated_by": None,
            "cleared_at": None,
            "cleared_by": None,
            "clear_reason": None,
        },
        {
            "kill_switch_id": "US_LIVE_BUY_KILL",
            "scope": "BUY",
            "target_value": "BUY",
            "is_active": False,
            "reason_code": None,
            "reason_detail": None,
            "activated_at": None,
            "activated_by": None,
            "cleared_at": None,
            "cleared_by": None,
            "clear_reason": None,
        },
        {
            "kill_switch_id": "US_LIVE_SELL_KILL",
            "scope": "SELL",
            "target_value": "SELL",
            "is_active": False,
            "reason_code": None,
            "reason_detail": None,
            "activated_at": None,
            "activated_by": None,
            "cleared_at": None,
            "cleared_by": None,
            "clear_reason": None,
        },
    ]
    usage_rows = [
        {
            "trade_date": trade_date,
            "policy_id": policy_id,
            "account_id": account_id,
            "buy_order_count": 0,
            "sell_order_count": 0,
            "total_order_count": 0,
            "new_buy_count": 0,
            "buy_amount_usd": 0,
            "sell_amount_usd": 0,
            "total_order_amount_usd": 0,
            "failed_order_count": 0,
            "rejected_order_count": 0,
            "blocked_order_count": 0,
            "realized_pnl_usd": None,
            "unrealized_pnl_usd": None,
            "daily_pnl_usd": None,
            "daily_pnl_pct": None,
            "max_position_weight": None,
            "max_sector_weight": None,
            "cash_weight": None,
            "data_status": "INIT",
        }
    ]
    return kill_rows, usage_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize US live risk-state tables with safe default rows.")
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_us_live_risk_policy(args.policy_id)
    policy_id = str(policy.get("policy_id") or "US_LIVE_RULE_V1")
    account_cfg = policy.get("account", {}) if isinstance(policy.get("account"), dict) else {}
    account_id = str(account_cfg.get("account_id") or policy_id).strip() or policy_id
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date") if args.trade_date else date.today()
    kill_rows, usage_rows = _build_default_rows(policy_id, account_id, trade_date)
    if args.dry_run:
        print("[US Live Risk State Dry Run]")
        print(f"Policy ID: {policy_id}")
        print(f"Account ID: {account_id}")
        print(f"Trade Date: {trade_date.isoformat()}")
        print(f"Kill Switch Rows: {len(kill_rows)}")
        print(f"Daily Usage Rows: {len(usage_rows)}")
        return 0
    ensure_us_live_risk_tables()
    upsert_us_live_kill_switch_rows(kill_rows)
    upsert_us_live_daily_risk_usage_rows(usage_rows)
    stored_kill = fetch_us_live_kill_switch_rows()
    stored_usage = fetch_us_live_daily_risk_usage_rows(trade_date=trade_date, policy_id=policy_id, account_id=account_id)
    print("[US Live Risk State]")
    print(f"Policy ID: {policy_id}")
    print(f"Account ID: {account_id}")
    print(f"Trade Date: {trade_date.isoformat()}")
    print(f"Kill Switch Rows: {len(stored_kill)}")
    print(f"Daily Usage Rows: {len(stored_usage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
