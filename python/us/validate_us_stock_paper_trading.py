from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.paper_rebalance import load_policy, safe_float
from python.us.simulate_us_stock_paper_fills import validate_paper_account_integrity
from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import (
    fetch_rank_component_rows_between,
    fetch_us_paper_account_rows,
    fetch_us_paper_account_snapshot_rows,
    fetch_us_paper_fill_rows,
    fetch_us_paper_order_rows,
    fetch_us_paper_position_rows,
)
from utils.paper_trading_safety import assert_paper_trading_only


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str


def _fmt_pct(value: object) -> str:
    number = safe_float(value)
    return "N/A" if number is None else f"{number * 100:.2f}%"


def _load_sector_map(snapshot_date: date | None) -> dict[str, str]:
    if snapshot_date is None:
        return {}
    rows = fetch_rank_component_rows_between(
        start_date=snapshot_date - timedelta(days=7),
        end_date=snapshot_date,
        source="rule_v1",
    )
    latest_by_symbol: dict[str, tuple[date, str]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        trade_date = row.get("trade_date")
        sector = str(row.get("sector") or "").strip()
        if not symbol or not isinstance(trade_date, date):
            continue
        current = latest_by_symbol.get(symbol)
        if current is None or trade_date > current[0]:
            latest_by_symbol[symbol] = (trade_date, sector or "UNKNOWN")
    return {symbol: sector for symbol, (_, sector) in latest_by_symbol.items()}


def collect_paper_validation(account_id: str, snapshot_date: date | None = None) -> dict[str, object]:
    cfg = load_us_paper_trading_config(account_id=account_id)
    policy = load_policy(account_id)
    account_rows = fetch_us_paper_account_rows(account_id=account_id)
    issues: list[ValidationIssue] = []
    if not account_rows:
        issues.append(ValidationIssue("ERROR", "account_not_found", f"Paper account not found: {account_id}"))
        return {"checks": 1, "ok": 0, "warnings": 0, "errors": 1, "issues": issues}

    account_row = account_rows[0]
    if str(account_row.get("status") or "").upper() != "ACTIVE":
        issues.append(ValidationIssue("WARNING", "account_not_active", f"Paper account status is {account_row.get('status')}"))

    position_rows = fetch_us_paper_position_rows(account_id=account_id)
    order_rows = fetch_us_paper_order_rows(account_id=account_id)
    fill_rows = fetch_us_paper_fill_rows(account_id=account_id)
    snapshot_rows = fetch_us_paper_account_snapshot_rows(account_id=account_id, snapshot_date=snapshot_date)
    if snapshot_date is None:
        snapshot_rows = fetch_us_paper_account_snapshot_rows(account_id=account_id)
    snapshot_row = snapshot_rows[0] if snapshot_rows else None

    for code in validate_paper_account_integrity(account_id):
        issues.append(ValidationIssue("ERROR", code, code))

    sector_map = _load_sector_map(snapshot_row.get("snapshot_date") if isinstance(snapshot_row, dict) else snapshot_date)
    equity_value = safe_float(account_row.get("equity_value")) or 0.0
    sector_exposure: dict[str, float] = {}
    unknown_sector_count = 0

    filled_ids = {str(row.get("paper_order_id") or "") for row in fill_rows}
    for row in position_rows:
        symbol = str(row.get("symbol") or "").upper()
        qty = safe_float(row.get("qty")) or 0.0
        last_price = safe_float(row.get("last_price")) or 0.0
        market_value = safe_float(row.get("market_value")) or 0.0
        cost_amount = safe_float(row.get("cost_amount")) or 0.0
        unrealized_pnl = safe_float(row.get("unrealized_pnl")) or 0.0
        unrealized_pnl_pct = safe_float(row.get("unrealized_pnl_pct"))
        status = str(row.get("status") or "").upper()

        if status == "OPEN" and qty <= 0:
            issues.append(ValidationIssue("ERROR", "open_position_nonpositive_qty", f"{symbol} OPEN position qty must be > 0"))
        if status == "CLOSED" and abs(qty) > 1e-9:
            issues.append(ValidationIssue("ERROR", "closed_position_nonzero_qty", f"{symbol} CLOSED position qty must be 0"))
        if abs(market_value - (qty * last_price)) > 1e-4:
            issues.append(ValidationIssue("ERROR", "position_market_value_mismatch", f"{symbol} market_value is inconsistent with qty * last_price"))
        if abs(unrealized_pnl - (market_value - cost_amount)) > 1e-4:
            issues.append(ValidationIssue("ERROR", "position_unrealized_pnl_mismatch", f"{symbol} unrealized_pnl is inconsistent with market_value - cost_amount"))
        if cost_amount > 0 and unrealized_pnl_pct is not None:
            expected_pct = (market_value - cost_amount) / cost_amount
            if abs(unrealized_pnl_pct - expected_pct) > 1e-4:
                issues.append(ValidationIssue("WARNING", "position_unrealized_pnl_pct_mismatch", f"{symbol} unrealized_pnl_pct differs from current valuation"))
        if status == "OPEN" and equity_value > 0:
            weight = market_value / equity_value
            if weight > policy.max_position_weight + 1e-9:
                issues.append(ValidationIssue("WARNING", "position_weight_limit", f"{symbol} weight {_fmt_pct(weight)} exceeds max {_fmt_pct(policy.max_position_weight)}"))
            sector = sector_map.get(symbol, "UNKNOWN")
            if sector == "UNKNOWN":
                unknown_sector_count += 1
            else:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + market_value

    if unknown_sector_count:
        issues.append(ValidationIssue("WARNING", "sector_exposure_unknown", f"Sector exposure unknown for {unknown_sector_count} open positions"))
    if equity_value > 0:
        for sector, exposure in sector_exposure.items():
            weight = exposure / equity_value
            if weight > policy.max_sector_weight + 1e-9:
                issues.append(ValidationIssue("WARNING", "sector_weight_limit", f"{sector} weight {_fmt_pct(weight)} exceeds max {_fmt_pct(policy.max_sector_weight)}"))

    for row in order_rows:
        order_id = str(row.get("paper_order_id") or "")
        status = str(row.get("status") or "").upper()
        trade_date = row.get("trade_date")
        reject_reason = str(row.get("reject_reason") or "").strip()
        if status == "FILLED" and order_id not in filled_ids:
            issues.append(ValidationIssue("ERROR", "filled_order_missing_fill", f"FILLED order missing fill: {order_id}"))
        if status == "REJECTED" and not reject_reason:
            issues.append(ValidationIssue("WARNING", "rejected_order_missing_reason", f"REJECTED order missing reject_reason: {order_id}"))
        if status == "ERROR":
            issues.append(ValidationIssue("WARNING", "error_order_present", f"ERROR order exists: {order_id}"))
        if status == "CREATED" and isinstance(trade_date, date):
            age_days = ((snapshot_date or date.today()) - trade_date).days
            if age_days > 3:
                issues.append(ValidationIssue("WARNING", "stale_created_order", f"CREATED order older than 3 days: {order_id}"))

    for row in fill_rows:
        fill_id = str(row.get("paper_fill_id") or "")
        qty = safe_float(row.get("filled_qty")) or 0.0
        price = safe_float(row.get("filled_price")) or 0.0
        amount = safe_float(row.get("filled_amount")) or 0.0
        if abs(amount - (qty * price)) > 1e-4:
            issues.append(ValidationIssue("ERROR", "fill_amount_mismatch", f"{fill_id} filled_amount is inconsistent with qty * price"))

    if snapshot_row is None:
        issues.append(ValidationIssue("WARNING", "snapshot_missing", "Latest snapshot is missing"))
    else:
        snapshot_equity = safe_float(snapshot_row.get("equity_value")) or 0.0
        snapshot_cash = safe_float(snapshot_row.get("cash_balance")) or 0.0
        snapshot_market = safe_float(snapshot_row.get("market_value")) or 0.0
        if abs(snapshot_equity - (snapshot_cash + snapshot_market)) > 1e-4:
            issues.append(ValidationIssue("ERROR", "snapshot_equity_mismatch", "Snapshot equity_value is inconsistent with cash_balance + market_value"))
        if snapshot_row.get("daily_return_pct") is None:
            issues.append(ValidationIssue("WARNING", "snapshot_daily_return_missing", "Latest snapshot daily_return_pct is missing"))

    warning_count = sum(1 for item in issues if item.level == "WARNING")
    error_count = sum(1 for item in issues if item.level == "ERROR")
    checks = 5 + len(position_rows) * 4 + len(order_rows) + len(fill_rows)
    ok = max(0, checks - warning_count - error_count)
    return {
        "checks": checks,
        "ok": ok,
        "warnings": warning_count,
        "errors": error_count,
        "issues": issues,
        "account": account_row,
        "snapshot": snapshot_row,
        "positions": position_rows,
        "orders": order_rows,
        "fills": fill_rows,
    }


def _render_console(account_id: str, report: dict[str, object]) -> None:
    print("[Paper Trading Validation]")
    print(f"Account: {account_id}")
    print("")
    print(f"Checks: {report['checks']}")
    print(f"OK: {report['ok']}")
    print(f"Warnings: {report['warnings']}")
    print(f"Errors: {report['errors']}")
    issues = report["issues"]
    if issues:
        print("")
        warning_lines = [item.message for item in issues if item.level == "WARNING"]
        error_lines = [item.message for item in issues if item.level == "ERROR"]
        if warning_lines:
            print("[Warnings]")
            for line in warning_lines:
                print(f"- {line}")
        if error_lines:
            print("[Errors]")
            for line in error_lines:
                print(f"- {line}")


def _write_markdown(account_id: str, report: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"paper_validation_{account_id}.md"
    lines = [
        "# US Paper Trading Validation",
        "",
        f"- Account: {account_id}",
        f"- Checks: {report['checks']}",
        f"- OK: {report['ok']}",
        f"- Warnings: {report['warnings']}",
        f"- Errors: {report['errors']}",
        "",
        "## Issues",
        "",
    ]
    issues = report["issues"]
    if not issues:
        lines.append("- OK")
    else:
        for item in issues:
            lines.append(f"- [{item.level}] {item.message}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate US paper trading account integrity and operating state.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    assert_paper_trading_only(account_id=args.account_id, message="[SAFETY] Paper trading validation only. Real order APIs are blocked.")
    snapshot_date = parse_iso_date(args.snapshot_date, field_name="snapshot_date") if args.snapshot_date else None
    report = collect_paper_validation(args.account_id, snapshot_date=snapshot_date)
    if args.format == "markdown":
        print(_write_markdown(args.account_id, report, cfg.report_output_dir))
    else:
        _render_console(args.account_id, report)
    if args.fail_on_error and report["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
