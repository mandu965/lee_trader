from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_kill_switch import (
    activate_kill_switch,
    build_kill_switch_id,
    clear_kill_switch,
    list_kill_switches,
)
from utils.us_live_trading_safety import assert_us_live_pre_trade_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage US live kill-switch state without placing any real order.")
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--scope", choices=["GLOBAL", "BUY", "SELL", "SYMBOL", "SECTOR", "ACCOUNT"], default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--reason-code", default=None)
    parser.add_argument("--reason-detail", default=None)
    parser.add_argument("--clear-reason", default=None)
    parser.add_argument("--performed-by", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_scope_target(scope: str | None, target: str | None) -> None:
    if not scope:
        raise ValueError("--scope is required.")
    if scope in {"SYMBOL", "SECTOR", "ACCOUNT"} and not str(target or "").strip():
        raise ValueError(f"--target is required for scope={scope}.")


def _render_list(rows: list[dict[str, object]], *, active_only: bool) -> None:
    active_rows = [row for row in rows if bool(row.get("is_active"))]
    inactive_rows = [row for row in rows if not bool(row.get("is_active"))]
    print("[US Live Kill Switch Status]")
    print("")
    print(f"Active Kill Switches: {len(active_rows)}")
    print("")
    if active_rows:
        print("Kill ID | Scope | Target | Reason | Activated At | By")
        for row in active_rows:
            print(
                f"{row.get('kill_switch_id')} | {row.get('scope')} | {row.get('target_value') or 'ALL'} | "
                f"{row.get('reason_code') or ''} | {row.get('activated_at') or ''} | {row.get('activated_by') or ''}"
            )
    else:
        print("- none")
    if not active_only:
        print("")
        print("Inactive:")
        if inactive_rows:
            for row in inactive_rows:
                print(f"- {row.get('kill_switch_id')}")
        else:
            print("- none")


def main() -> int:
    args = parse_args()
    assert_us_live_pre_trade_only(
        policy_id=args.policy_id,
        message="[SAFETY] Kill switch management only. Real order APIs are blocked.",
    )
    if args.list or (not args.activate and not args.clear):
        rows = list_kill_switches(active_only=args.active_only)
        _render_list(rows, active_only=args.active_only)
        return 0

    _require_scope_target(args.scope, args.target)
    if args.activate:
        if not str(args.reason_code or "").strip():
            raise ValueError("--reason-code is required for --activate.")
        if not str(args.reason_detail or "").strip():
            raise ValueError("--reason-detail is required for --activate.")
        if not str(args.performed_by or "").strip():
            raise ValueError("--performed-by is required for --activate.")
        kill_switch_id = build_kill_switch_id(args.scope, args.target)
        if args.dry_run:
            print("[US Live Kill Switch Dry Run]")
            print(f"Action: ACTIVATE")
            print(f"Kill Switch ID: {kill_switch_id}")
            print(f"Scope: {args.scope}")
            print(f"Target: {args.target or 'ALL'}")
            print(f"Reason Code: {args.reason_code}")
            print(f"Reason Detail: {args.reason_detail}")
            print(f"Performed By: {args.performed_by}")
            return 0
        row = activate_kill_switch(
            args.scope,
            args.target,
            reason_code=args.reason_code,
            reason_detail=args.reason_detail,
            performed_by=args.performed_by,
            trigger_source="MANUAL",
        )
        print("[US Live Kill Switch]")
        print(f"Action: ACTIVATE")
        print(f"Kill Switch ID: {row.get('kill_switch_id')}")
        print(f"Active: {row.get('is_active')}")
        return 0

    if args.clear:
        if not str(args.clear_reason or "").strip():
            raise ValueError("--clear-reason is required for --clear.")
        if not str(args.performed_by or "").strip():
            raise ValueError("--performed-by is required for --clear.")
        kill_switch_id = build_kill_switch_id(args.scope, args.target)
        if args.dry_run:
            print("[US Live Kill Switch Dry Run]")
            print("Action: CLEAR")
            print(f"Kill Switch ID: {kill_switch_id}")
            print(f"Clear Reason: {args.clear_reason}")
            print(f"Performed By: {args.performed_by}")
            return 0
        row = clear_kill_switch(
            args.scope,
            args.target,
            clear_reason=args.clear_reason,
            performed_by=args.performed_by,
        )
        print("[US Live Kill Switch]")
        print("Action: CLEAR")
        print(f"Kill Switch ID: {row.get('kill_switch_id')}")
        print(f"Active: {row.get('is_active')}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
