from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_live_operations import build_micro_live_operations_report, render_operations_console
from utils.us_micro_reconciliation import render_reconciliation_console, run_micro_reconciliation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US Micro Live daily operational checks without creating or sending any order.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--execution-mode", default="MOCK", choices=["MOCK", "SANDBOX", "LIVE"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-cash", action="store_true")
    parser.add_argument("--activate-kill-on-critical", action="store_true")
    parser.add_argument("--performed-by", default="SYSTEM")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_micro_live_operations_report(
            trade_date=args.trade_date,
            account_id=args.account_id,
            activate_kill_on_critical=(args.activate_kill_on_critical and not args.dry_run),
            performed_by=args.performed_by,
        )
        print(render_operations_console(report))
        print("")
        recon = run_micro_reconciliation(
            account_id=args.account_id,
            recon_date=args.trade_date,
            execution_mode=args.execution_mode,
            include_orders=True,
            include_fills=True,
            include_positions=True,
            include_cash=args.include_cash,
            dry_run=args.dry_run,
            trigger_kill_on_critical=(args.activate_kill_on_critical and not args.dry_run),
        )
        print(render_reconciliation_console(recon))
        return 0
    except Exception as exc:
        print("[US Micro Live Daily Check]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
