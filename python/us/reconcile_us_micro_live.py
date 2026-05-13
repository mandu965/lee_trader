from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_reconciliation import (
    render_reconciliation_console,
    render_reconciliation_markdown,
    run_micro_reconciliation,
    write_reconciliation_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US Micro Live reconciliation without creating or sending any new order.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--recon-date", required=True)
    parser.add_argument("--execution-mode", default="MOCK", choices=["MOCK", "SANDBOX", "LIVE"])
    parser.add_argument("--include-orders", action="store_true")
    parser.add_argument("--include-fills", action="store_true")
    parser.add_argument("--include-positions", action="store_true")
    parser.add_argument("--include-cash", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trigger-kill-on-critical", action="store_true")
    parser.add_argument("--format", default="console", choices=["console", "markdown"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    include_orders = args.include_orders or not any([args.include_orders, args.include_fills, args.include_positions, args.include_cash])
    include_fills = args.include_fills or not any([args.include_orders, args.include_fills, args.include_positions, args.include_cash])
    include_positions = args.include_positions or not any([args.include_orders, args.include_fills, args.include_positions, args.include_cash])
    try:
        report = run_micro_reconciliation(
            account_id=args.account_id,
            recon_date=args.recon_date,
            execution_mode=args.execution_mode,
            include_orders=include_orders,
            include_fills=include_fills,
            include_positions=include_positions,
            include_cash=args.include_cash,
            dry_run=args.dry_run,
            trigger_kill_on_critical=args.trigger_kill_on_critical,
        )
        if args.format == "markdown":
            markdown = render_reconciliation_markdown(report)
            print(markdown)
            output_path = write_reconciliation_markdown(report)
            print("")
            print(f"saved {output_path}")
        else:
            print(render_reconciliation_console(report))
        return 0
    except Exception as exc:
        print("[US Micro Live Reconciliation]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
