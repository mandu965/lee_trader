from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_kill_switch import activate_kill_switch, evaluate_kill_switch_triggers
from utils.us_live_risk_policy import load_us_live_risk_policy
from utils.us_live_trading_safety import assert_us_live_pre_trade_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate US live kill-switch auto-trigger conditions without placing any real order.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--performed-by", default="SYSTEM")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_us_live_risk_policy(args.policy_id)
    policy_id = str(policy.get("policy_id") or args.policy_id or "US_LIVE_RULE_V1")
    assert_us_live_pre_trade_only(
        policy_id=policy_id,
        message="[SAFETY] Kill switch management only. Real order APIs are blocked.",
    )
    triggers = evaluate_kill_switch_triggers(args.trade_date, args.account_id, policy_id)
    print("[US Live Kill Switch Evaluation]")
    print(f"Trade Date: {args.trade_date}")
    print(f"Account ID: {args.account_id}")
    print(f"Policy ID: {policy_id}")
    print(f"Trigger Count: {len(triggers)}")
    if not triggers:
        print("No kill-switch trigger conditions detected.")
        return 0
    for item in triggers:
        print(
            f"- Scope={item.get('scope')} Target={item.get('target_value') or 'ALL'} "
            f"Reason={item.get('reason_code')} Detail={item.get('reason_detail')}"
        )
    if args.dry_run or not args.activate:
        return 0
    for item in triggers:
        activate_kill_switch(
            scope=str(item.get("scope") or "GLOBAL"),
            target_value=item.get("target_value"),
            reason_code=str(item.get("reason_code") or "unknown"),
            reason_detail=str(item.get("reason_detail") or "auto trigger"),
            performed_by=args.performed_by,
            trigger_source=str(item.get("trigger_source") or "SYSTEM"),
            trigger_ref_id=str(item.get("trigger_ref_id") or "") or None,
        )
    print("Kill switch activation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
