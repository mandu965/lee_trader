from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.trade_orchestration.daily_trade_orchestrator import run_trade_orchestration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US trade orchestration without any broker API or real BUY/SELL execution.")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD.")
    parser.add_argument("--mode", default=None, help="Optional mode override via environment.")
    parser.add_argument("--account-id", default="US_TRADE_SHADOW")
    parser.add_argument("--no-persist", action="store_true", help="Do not write JSON/DB logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode:
        __import__("os").environ["US_TRADE_ORCHESTRATION_MODE"] = str(args.mode).upper()
    result = run_trade_orchestration(
        trade_date=args.trade_date,
        account_id=args.account_id,
        persist_logs=not args.no_persist,
    )
    report = result.get("integrated_report") or {
        "mode": result.get("mode"),
        "orchestration_enabled": result.get("enabled"),
        "trade_date": result.get("trade_date"),
        "success": result.get("success"),
        "sell_summary": result.get("sell_summary", {}),
        "buy_summary": result.get("buy_summary", {}),
        "conflict_summary": result.get("conflict_summary", {}),
    }
    print("[US TRADE ORCHESTRATION]")
    print(f"mode={report.get('mode')}")
    print(f"enabled={1 if report.get('orchestration_enabled') else 0}")
    print(f"trade_date={report.get('trade_date')}")
    print("")
    print("SELL:")
    print(f"loaded_positions={report.get('sell_summary', {}).get('loaded_positions', 0)}")
    print(f"hold={report.get('sell_summary', {}).get('hold', 0)}")
    print(f"sell={report.get('sell_summary', {}).get('sell', 0)}")
    print(f"partial_sell={report.get('sell_summary', {}).get('partial_sell', 0)}")
    print(f"review_required={report.get('sell_summary', {}).get('review_required', 0)}")
    print("")
    print("BUY:")
    print(f"loaded_candidates={report.get('buy_summary', {}).get('loaded_candidates', 0)}")
    print(f"allowed_before_conflict={report.get('buy_summary', {}).get('risk_guard_passed', report.get('buy_summary', {}).get('allowed_before_conflict', 0))}")
    print(f"conflict_blocked={report.get('conflict_summary', {}).get('TOTAL_CONFLICT_BLOCKED', result.get('buy_summary', {}).get('conflict_blocked', 0))}")
    print(f"allowed_after_conflict={report.get('buy_summary', {}).get('conflict_guard_passed', report.get('buy_summary', {}).get('allowed_after_conflict', 0))}")
    print("")
    print("CONFLICT:")
    for key in ("OPEN_POSITION_EXISTS", "SELL_SIGNAL_EXISTS", "REVIEW_REQUIRED_SYMBOL", "COOLDOWN_ACTIVE"):
        print(f"{key}={report.get('conflict_summary', {}).get(key, 0)}")
    print("")
    print("PAPER:")
    print(f"paper_buy_orders={result.get('paper_orders', {}).get('buy_orders', 0)}")
    print(f"paper_sell_orders={result.get('paper_orders', {}).get('sell_orders', 0)}")
    print("")
    print(f"final_status={'SUCCESS' if result.get('success') else 'FAILED'}")
    return 0 if result.get("success") or not result.get("pipeline_should_fail") else 1


if __name__ == "__main__":
    raise SystemExit(main())
