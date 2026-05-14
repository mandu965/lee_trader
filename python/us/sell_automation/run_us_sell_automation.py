from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.sell_automation.sell_decision_engine import run_sell_automation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US SELL automation skeleton without any broker API or real SELL execution.")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD. Defaults to latest available ranking or paper order trade date.")
    parser.add_argument("--account-id", default="US_SELL_SHADOW")
    parser.add_argument("--no-persist", action="store_true", help="Do not write JSON/DB logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_sell_automation(
            trade_date=args.trade_date,
            account_id=args.account_id,
            persist_logs=not args.no_persist,
        )
        print("[US SELL AUTOMATION]")
        print(f"mode={report['mode']}")
        print(f"enabled={1 if report['automation_enabled'] else 0}")
        print(f"trade_date={report.get('trade_date') or '-'}")
        print("")
        print(f"loaded_positions={report['loaded_positions']}")
        print(f"hold_positions={report['hold_positions']}")
        print(f"sell_signals={report['sell_signals']}")
        print(f"partial_sell_signals={report['partial_sell_signals']}")
        print(f"review_required={report['review_required']}")
        print(f"paper_sell_orders={len(report['paper_sell_orders'])}")
        print("")
        print("reason_summary:")
        if report["reason_summary"]:
            for reason_code, count in report["reason_summary"].items():
                print(f"- {reason_code}: {count}")
        else:
            print("- NONE: 0")
        if report.get("config_warnings"):
            print("")
            print("config_warnings:")
            for warning in report["config_warnings"]:
                print(f"- {warning}")
        if report.get("events"):
            print("")
            print("events:")
            for event in report["events"]:
                print(f"- {event.get('reason_code')}: {event.get('detail')}")
        if report.get("log_persistence"):
            print("")
            print(f"log_json={report['log_persistence'].get('json_path')}")
        return 0
    except Exception as exc:
        print("[US SELL AUTOMATION]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
