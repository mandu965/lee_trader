from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_live_operations import (
    build_micro_live_operations_report,
    maybe_notify_operations_report,
    render_operations_console,
    render_operations_markdown,
    write_operations_csv,
    write_operations_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US Micro Live operations report without creating or sending any order.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--format", default="console", choices=["console", "markdown", "csv"])
    parser.add_argument("--include-ranking", action="store_true")
    parser.add_argument("--include-precheck", action="store_true")
    parser.add_argument("--include-approvals", action="store_true")
    parser.add_argument("--include-orders", action="store_true")
    parser.add_argument("--include-fills", action="store_true")
    parser.add_argument("--include-reconciliation", action="store_true")
    parser.add_argument("--include-kill-switch", action="store_true")
    parser.add_argument("--include-actions", action="store_true")
    parser.add_argument("--activate-kill-on-critical", action="store_true")
    parser.add_argument("--performed-by", default="SYSTEM")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _default_include(args: argparse.Namespace, field_names: list[str]) -> dict[str, bool]:
    explicit = any(getattr(args, name) for name in field_names)
    return {name: (getattr(args, name) or not explicit) for name in field_names}


def main() -> int:
    args = parse_args()
    include = _default_include(
        args,
        [
            "include_ranking",
            "include_precheck",
            "include_approvals",
            "include_orders",
            "include_fills",
            "include_reconciliation",
            "include_kill_switch",
            "include_actions",
        ],
    )
    try:
        report = build_micro_live_operations_report(
            trade_date=args.trade_date,
            account_id=args.account_id,
            include_ranking=include["include_ranking"],
            include_precheck=include["include_precheck"],
            include_approvals=include["include_approvals"],
            include_orders=include["include_orders"],
            include_fills=include["include_fills"],
            include_reconciliation=include["include_reconciliation"],
            include_kill_switch=include["include_kill_switch"],
            include_actions=include["include_actions"],
            activate_kill_on_critical=args.activate_kill_on_critical,
            performed_by=args.performed_by,
        )
        if args.format == "console":
            print(render_operations_console(report))
        elif args.format == "markdown":
            print(render_operations_markdown(report))
            path = write_operations_markdown(report, output_dir=args.output_dir)
            print("")
            print(f"saved {path}")
        else:
            files = write_operations_csv(report, output_dir=args.output_dir)
            for path in files:
                print(f"saved {path}")
        maybe_notify_operations_report(report)
        return 0
    except Exception as exc:
        print("[US Micro Live Operations Report]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
