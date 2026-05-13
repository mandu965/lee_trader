from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_live_risk_policy import collect_us_live_risk_policy_issues, load_us_live_risk_policy


@dataclass(frozen=True)
class ValidationReport:
    policy: dict[str, object]
    ok: int
    warnings: int
    errors: int
    issues: list[object]
    result: str


def build_validation_report(policy_id: str | None) -> ValidationReport:
    policy = load_us_live_risk_policy(policy_id)
    issues = collect_us_live_risk_policy_issues(policy)
    warnings = sum(1 for item in issues if getattr(item, "level", "") == "WARNING")
    errors = sum(1 for item in issues if getattr(item, "level", "") == "ERROR")
    checks = 11
    ok = max(0, checks - warnings - errors)
    result = "SAFE_DEFAULT" if errors == 0 and warnings == 0 else ("SAFE_WITH_WARNINGS" if errors == 0 else "UNSAFE")
    return ValidationReport(policy=policy, ok=ok, warnings=warnings, errors=errors, issues=issues, result=result)


def _render_console(report: ValidationReport) -> None:
    policy = report.policy
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    account = policy.get("account", {}) if isinstance(policy.get("account"), dict) else {}
    order = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    position = policy.get("position", {}) if isinstance(policy.get("position"), dict) else {}
    sector = policy.get("sector", {}) if isinstance(policy.get("sector"), dict) else {}
    print("[US Live Risk Policy Validation]")
    print(f"Policy ID: {policy.get('policy_id')}")
    print("")
    print("Safety Flags:")
    print(f"- live_trading_enabled: {str(bool(safety.get('live_trading_enabled'))).lower()}")
    print(f"- live_order_enabled: {str(bool(safety.get('live_order_enabled'))).lower()}")
    print(f"- buy_enabled: {str(bool(safety.get('buy_enabled'))).lower()}")
    print(f"- sell_enabled: {str(bool(safety.get('sell_enabled'))).lower()}")
    print(f"- require_manual_approval: {str(bool(safety.get('require_manual_approval'))).lower()}")
    print(f"- real_order_blocked: {str(bool(safety.get('real_order_blocked'))).lower()}")
    print("")
    print("Risk Limits:")
    print(f"- max_order_amount_usd: {order.get('max_order_amount_usd')}")
    print(f"- max_daily_buy_amount_usd: {order.get('max_daily_buy_amount_usd')}")
    print(f"- max_daily_order_count: {order.get('max_daily_order_count')}")
    print(f"- max_position_weight: {(float(position.get('max_position_weight') or 0) * 100):.1f}%")
    print(f"- max_sector_weight: {(float(sector.get('max_sector_weight') or 0) * 100):.1f}%")
    print(f"- min_cash_weight: {(float(account.get('min_cash_weight') or 0) * 100):.1f}%")
    print("")
    print("Validation:")
    print(f"OK: {report.ok}")
    print(f"Warnings: {report.warnings}")
    print(f"Errors: {report.errors}")
    if report.issues:
        print("")
        for item in report.issues:
            print(f"{item.level}:")
            print(f"- {item.message}")
    print("")
    print(f"Result: {report.result}")


def _write_markdown(report: ValidationReport) -> Path:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "outputs" / "us_stock_live_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"live_risk_policy_validation_{report.policy.get('policy_id')}.md"
    lines = [
        "# US Live Risk Policy Validation",
        "",
        f"- Policy ID: {report.policy.get('policy_id')}",
        f"- Result: {report.result}",
        f"- OK: {report.ok}",
        f"- Warnings: {report.warnings}",
        f"- Errors: {report.errors}",
        "",
        "## Issues",
        "",
    ]
    if report.issues:
        for item in report.issues:
            lines.append(f"- [{item.level}] {item.message}")
    else:
        lines.append("- OK")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate US live risk policy defaults and safety thresholds.")
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_validation_report(args.policy_id)
    if args.format == "markdown":
        print(_write_markdown(report))
    else:
        _render_console(report)
    if args.fail_on_error and report.errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
