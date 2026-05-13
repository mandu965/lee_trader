from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.buy_automation.report_generator import (
    finalize_buy_report,
    load_buy_automation_run_log,
    render_buy_report_console,
    render_buy_report_markdown,
    write_buy_report_json,
    write_buy_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a daily report for the US BUY automation skeleton without any external API call.")
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--format", default="console", choices=["console", "json", "markdown"])
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw_log = load_buy_automation_run_log(trade_date=args.trade_date, input_dir=args.input_dir)
        report = finalize_buy_report(raw_log)
        if args.format == "console":
            print(render_buy_report_console(report))
        elif args.format == "json":
            path = write_buy_report_json(report, output_dir=args.output_dir)
            print(f"saved {path}")
        else:
            print(render_buy_report_markdown(report))
            path = write_buy_report_markdown(report, output_dir=args.output_dir)
            print("")
            print(f"saved {path}")
        return 0
    except Exception as exc:
        print("[US BUY Automation Report]")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
