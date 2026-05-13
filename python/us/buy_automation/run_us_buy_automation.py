from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.buy_automation.decision_engine import run_buy_automation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US BUY automation skeleton without any real order or broker API call.")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD. Defaults to latest available ranking trade date.")
    parser.add_argument("--account-id", default="US_BUY_SHADOW")
    parser.add_argument("--no-persist", action="store_true", help="Do not write JSON/DB logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_buy_automation(
            trade_date=args.trade_date,
            account_id=args.account_id,
            persist_logs=not args.no_persist,
        )
        print("[US BUY AUTOMATION]")
        print(f"mode={report['mode']}")
        print(f"enabled={1 if report['enabled'] else 0}")
        print(f"trade_date={report.get('trade_date') or '-'}")
        print("")
        print(f"loaded_candidates={report['loaded_candidates']}")
        print(f"allowed_candidates={report['allowed_candidates']}")
        print(f"blocked_candidates={report['blocked_candidates']}")
        print(f"paper_orders={len(report['paper_orders'])}")
        print("")
        print("block_summary:")
        for reason_code, count in report["block_summary"].items():
            print(f"- {reason_code}: {count}")
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
        print("[US BUY AUTOMATION]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
