from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMON_REQUIRED_FILES = {
    "postgres/live_trade_ledger_tables.sql": "live trade ledger schema SQL is missing",
}

COMMAND_SET_REQUIRED_FILES = {
    "auto_buy": {
        "postgres/analytics_live_trade_views.sql": "analytics live-trade view SQL is missing",
    },
    "live_sync": {
        "postgres/analytics_live_trade_views.sql": "analytics live-trade view SQL is missing",
    },
}


def _required_files_for(command_set: str) -> dict[str, str]:
    required = dict(COMMON_REQUIRED_FILES)
    required.update(COMMAND_SET_REQUIRED_FILES.get(str(command_set or "").strip().lower(), {}))
    return required


def validate_runtime_assets(command_set: str) -> list[str]:
    issues: list[str] = []
    for relative_path, message in _required_files_for(command_set).items():
        candidate = ROOT / relative_path
        if not candidate.exists():
            issues.append(f"{message}: {candidate}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate required runtime files before scheduler/trading execution.")
    parser.add_argument("--command-set", default="close", help="Scheduler command set such as close, auto_buy, or live_sync.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required runtime files are missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = validate_runtime_assets(args.command_set)
    if issues:
        for issue in issues:
            print(f"[RUNTIME_ASSET_ERROR] {issue}")
        return 1 if args.strict else 0
    print(f"runtime_assets_ok: command_set={args.command_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
