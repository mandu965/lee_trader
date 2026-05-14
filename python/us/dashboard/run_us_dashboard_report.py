from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.scheduler_integration import run_dashboard_scheduler_integration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a file-based US Paper Trading dashboard report.")
    parser.add_argument("--trade-date", default=None, help="Optional YYYY-MM-DD.")
    parser.add_argument("--format", default=None, choices=["json", "markdown", "all"], help="Optional output format override.")
    parser.add_argument("--force", action="store_true", help="Allow manual run even if US_DASHBOARD_ENABLED=0.")
    return parser.parse_args()


def _selected_formats(arg_value: str | None, default_formats: tuple[str, ...]) -> tuple[str, ...]:
    if arg_value is None or arg_value == "all":
        return default_formats
    return (arg_value,)


def _overall_status(payload: dict[str, object]) -> str:
    statuses = [
        str((payload.get(section) or {}).get("status") or "OK")
        for section in (
            "daily_overview",
            "paper_portfolio_summary",
            "buy_decision_monitor",
            "sell_decision_monitor",
            "conflict_guard_monitor",
            "paper_performance_monitor",
            "benchmark_comparison",
            "risk_data_quality_monitor",
            "scheduler_health_monitor",
            "live_readiness_monitor",
        )
    ]
    if "ERROR" in statuses:
        return "ERROR"
    if "DATA_MISSING" in statuses:
        return "DATA_MISSING"
    if "WARNING" in statuses:
        return "WARNING"
    return "OK"


def main() -> int:
    args = parse_args()
    cfg = load_dashboard_config()
    if not cfg.enabled and not args.force:
        print("[US PAPER TRADING DASHBOARD]")
        print("status=DISABLED")
        print("reason=US_DASHBOARD_ENABLED=0")
        print("hint=use --force for manual generation")
        return 0

    result = run_dashboard_scheduler_integration(
        trade_date=args.trade_date,
        force=args.force,
        formats=_selected_formats(args.format, cfg.formats),
    )
    payload = result.get("payload") or {}
    paths = {
        key: value
        for key, value in {
            "json": result.get("json_report_path"),
            "markdown": result.get("markdown_report_path"),
            "latest_json": result.get("latest_json_path"),
            "latest_markdown": result.get("latest_markdown_path"),
        }.items()
        if value
    }

    daily = payload.get("daily_overview") or {}
    risk = payload.get("risk_data_quality_monitor") or {}
    health = payload.get("scheduler_health_monitor") or {}
    readiness = payload.get("live_readiness_monitor") or {}

    print("[US PAPER TRADING DASHBOARD]")
    print(f"trade_date={(payload.get('meta') or {}).get('trade_date') or result.get('trade_date')}")
    print(f"mode={(payload.get('meta') or {}).get('mode')}")
    print(f"status={_overall_status(payload) if payload else ('ERROR' if not result.get('success') else 'OK')}")
    print("")
    print(f"daily_status={daily.get('status')}")
    print(f"open_positions={(payload.get('paper_portfolio_summary') or {}).get('open_position_count')}")
    print(f"final_buy_allowed={daily.get('final_buy_allowed')}")
    print(f"sell_signals={daily.get('sell_signals')}")
    print(f"review_required={daily.get('review_required_count')}")
    print(f"conflict_blocked={daily.get('conflict_blocked_count')}")
    print(f"data_missing_rate={risk.get('data_missing_rate')}%")
    print(f"health_check={health.get('health_check_status')}")
    print(f"live_ready={str(bool(readiness.get('live_ready'))).lower()}")
    if result.get("errors"):
        print("")
        print(f"errors={','.join(result.get('errors') or [])}")
    print("")
    print("output:")
    for key in ("json", "markdown", "latest_json", "latest_markdown"):
        if key in paths:
            print(f"- {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
