from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.paper_rebalance import safe_float
from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import (
    fetch_rank_component_rows_between,
    fetch_us_paper_account_rows,
    fetch_us_paper_account_snapshot_rows,
    fetch_us_paper_fill_rows,
    fetch_us_paper_order_rows,
    fetch_us_paper_position_rows,
)
from python.us.validate_us_stock_paper_trading import collect_paper_validation
from utils.paper_trading_safety import assert_paper_trading_only


SUPPORTED_FORMATS = {"console", "markdown", "csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report US paper trading account state and performance.")
    parser.add_argument("--account-id", required=True, help="Paper account ID.")
    parser.add_argument("--format", choices=sorted(SUPPORTED_FORMATS), default="console")
    parser.add_argument("--snapshot-date", default=None, help="Optional snapshot date. Format: YYYY-MM-DD.")
    parser.add_argument("--show-positions", action="store_true")
    parser.add_argument("--show-orders", action="store_true")
    parser.add_argument("--show-fills", action="store_true")
    return parser.parse_args()


def _fmt_money(value: object) -> str:
    num = safe_float(value)
    return "N/A" if num is None else f"{num:,.2f}"


def _fmt_pct(value: object) -> str:
    num = safe_float(value)
    return "N/A" if num is None else f"{num * 100:.2f}%"


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _latest_snapshot_or_none(rows: list[dict[str, object]], snapshot_date: date | None) -> dict[str, object] | None:
    if snapshot_date is not None:
        for row in rows:
            if row.get("snapshot_date") == snapshot_date:
                return row
        return None
    return rows[0] if rows else None


def _build_summary(account_row: dict[str, object], snapshot_row: dict[str, object] | None) -> dict[str, object]:
    summary = dict(account_row)
    if snapshot_row:
        summary.update(snapshot_row)
    return summary


def _latest_rank_sector_map(snapshot_date: date | None) -> dict[str, str]:
    if snapshot_date is None:
        return {}
    rows = fetch_rank_component_rows_between(
        start_date=snapshot_date,
        end_date=snapshot_date,
        source="rule_v1",
    )
    return {
        str(row.get("symbol") or "").upper(): str(row.get("sector") or "").strip() or "UNKNOWN"
        for row in rows
        if str(row.get("symbol") or "").strip()
    }


def _operation_status(
    *,
    summary: dict[str, object],
    positions: list[dict[str, object]],
    orders: list[dict[str, object]],
    fills: list[dict[str, object]],
    snapshots: list[dict[str, object]],
    validation: dict[str, object],
) -> dict[str, object]:
    latest_order_date = max((row.get("trade_date") for row in orders if isinstance(row.get("trade_date"), date)), default=None)
    latest_fill_date = max((row.get("trade_date") for row in fills if isinstance(row.get("trade_date"), date)), default=None)
    latest_snapshot_date = max((row.get("snapshot_date") for row in snapshots if isinstance(row.get("snapshot_date"), date)), default=None)
    latest_rebalance_date = latest_order_date
    equity_value = safe_float(summary.get("equity_value")) or 0.0
    cash_balance = safe_float(summary.get("cash_balance")) or 0.0
    cash_weight = (cash_balance / equity_value) if equity_value > 0 else None
    sector_map = _latest_rank_sector_map(latest_snapshot_date)
    sector_exposure: dict[str, float] = {}
    max_position_weight = 0.0
    for row in positions:
        if str(row.get("status") or "").upper() != "OPEN":
            continue
        market_value = safe_float(row.get("market_value")) or 0.0
        if equity_value > 0:
            max_position_weight = max(max_position_weight, market_value / equity_value)
        symbol = str(row.get("symbol") or "").upper()
        sector = sector_map.get(symbol, "UNKNOWN")
        if sector != "UNKNOWN":
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + market_value
    max_sector_weight = 0.0
    if equity_value > 0:
        max_sector_weight = max((value / equity_value for value in sector_exposure.values()), default=0.0)
    return {
        "last_rebalance_date": latest_rebalance_date,
        "last_order_date": latest_order_date,
        "last_fill_date": latest_fill_date,
        "last_snapshot_date": latest_snapshot_date,
        "created_orders": sum(1 for row in orders if str(row.get("status") or "").upper() == "CREATED"),
        "rejected_orders": sum(1 for row in orders if str(row.get("status") or "").upper() == "REJECTED"),
        "error_orders": sum(1 for row in orders if str(row.get("status") or "").upper() == "ERROR"),
        "open_positions": sum(1 for row in positions if str(row.get("status") or "").upper() == "OPEN"),
        "cash_weight": cash_weight,
        "max_position_weight": max_position_weight if equity_value > 0 else None,
        "max_sector_weight": max_sector_weight if equity_value > 0 else None,
        "validation_warnings": validation["warnings"],
        "validation_errors": validation["errors"],
    }


def _load_report_payload(account_id: str, snapshot_date: date | None) -> dict[str, object]:
    account_rows = fetch_us_paper_account_rows(account_id=account_id)
    if not account_rows:
        raise ValueError(f"paper account not found: {account_id}")
    account_row = account_rows[0]
    snapshot_rows = fetch_us_paper_account_snapshot_rows(account_id=account_id)
    snapshot_row = _latest_snapshot_or_none(snapshot_rows, snapshot_date)
    position_rows = fetch_us_paper_position_rows(account_id=account_id)
    order_rows = fetch_us_paper_order_rows(account_id=account_id)
    fill_rows = fetch_us_paper_fill_rows(account_id=account_id)
    validation = collect_paper_validation(account_id, snapshot_date=snapshot_date)
    summary = _build_summary(account_row, snapshot_row)
    summary["cash_weight"] = ((safe_float(summary.get("cash_balance")) or 0.0) / (safe_float(summary.get("equity_value")) or 1.0)) if (safe_float(summary.get("equity_value")) or 0.0) > 0 else None
    summary["open_position_count"] = sum(1 for row in position_rows if str(row.get("status") or "").upper() == "OPEN")
    operation_status = _operation_status(
        summary=summary,
        positions=position_rows,
        orders=order_rows,
        fills=fill_rows,
        snapshots=snapshot_rows,
        validation=validation,
    )
    return {
        "account": account_row,
        "summary": summary,
        "snapshot": snapshot_row,
        "snapshots": snapshot_rows,
        "positions": position_rows,
        "orders": order_rows,
        "fills": fill_rows,
        "validation": validation,
        "operation_status": operation_status,
    }


def _print_console(payload: dict[str, object], *, show_positions: bool, show_orders: bool, show_fills: bool) -> None:
    summary = payload["summary"]
    positions = payload["positions"]
    orders = payload["orders"]
    fills = payload["fills"]
    op = payload["operation_status"]
    validation = payload["validation"]
    print("[US Stock Paper Trading Report]")
    print(f"Account: {summary.get('account_id')}")
    print(f"Snapshot Date: {_format_date(summary.get('snapshot_date'))}")
    print("")
    print("[Account Summary]")
    print(f"Initial Cash:      {_fmt_money(summary.get('initial_cash'))}")
    print(f"Cash Balance:      {_fmt_money(summary.get('cash_balance'))}")
    print(f"Market Value:      {_fmt_money(summary.get('market_value'))}")
    print(f"Equity Value:      {_fmt_money(summary.get('equity_value'))}")
    print(f"Realized PnL:      {_fmt_money(summary.get('realized_pnl'))}")
    print(f"Unrealized PnL:    {_fmt_money(summary.get('unrealized_pnl'))}")
    print(f"Total PnL:         {_fmt_money(summary.get('total_pnl'))}")
    print(f"Total PnL %:       {_fmt_pct(summary.get('total_pnl_pct'))}")
    print(f"Daily Return %:    {_fmt_pct(summary.get('daily_return_pct'))}")
    print(f"Excess vs SPY:     {_fmt_pct(summary.get('excess_return_vs_spy'))}")
    print(f"Excess vs QQQ:     {_fmt_pct(summary.get('excess_return_vs_qqq'))}")
    print(f"Cash Weight:       {_fmt_pct(summary.get('cash_weight'))}")
    print(f"Position Count:    {summary.get('open_position_count')}")
    print("")
    print("[Operation Status]")
    print(f"Last Rebalance Date: {_format_date(op.get('last_rebalance_date'))}")
    print(f"Last Order Date:     {_format_date(op.get('last_order_date'))}")
    print(f"Last Fill Date:      {_format_date(op.get('last_fill_date'))}")
    print(f"Last Snapshot Date:  {_format_date(op.get('last_snapshot_date'))}")
    print(f"Created Orders:      {op.get('created_orders')}")
    print(f"Rejected Orders:     {op.get('rejected_orders')}")
    print(f"Error Orders:        {op.get('error_orders')}")
    print(f"Open Positions:      {op.get('open_positions')}")
    print(f"Cash Weight:         {_fmt_pct(op.get('cash_weight'))}")
    print(f"Max Position Weight: {_fmt_pct(op.get('max_position_weight'))}")
    print(f"Max Sector Weight:   {_fmt_pct(op.get('max_sector_weight'))}")
    print(f"Validation Warnings: {op.get('validation_warnings')}")
    print(f"Validation Errors:   {op.get('validation_errors')}")

    if show_positions or True:
        print("")
        print("[Open Positions]")
        print("Symbol | Qty | Avg Price | Last Price | Market Value | Unrealized PnL | PnL %")
        for row in positions:
            if str(row.get("status") or "").upper() != "OPEN":
                continue
            print(
                f"{row.get('symbol')} | {row.get('qty')} | {_fmt_money(row.get('avg_price'))} | {_fmt_money(row.get('last_price'))} | "
                f"{_fmt_money(row.get('market_value'))} | {_fmt_money(row.get('unrealized_pnl'))} | {_fmt_pct(row.get('unrealized_pnl_pct'))}"
            )

    if show_orders:
        print("")
        print("[Recent Orders]")
        print("Trade Date | Symbol | Side | Qty | Status | Reason")
        for row in orders[:10]:
            print(
                f"{_format_date(row.get('trade_date'))} | {row.get('symbol')} | {row.get('side')} | {row.get('order_qty')} | "
                f"{row.get('status')} | {str(row.get('reason') or '')[:60]}"
            )

    if show_fills:
        print("")
        print("[Recent Fills]")
        print("Fill Date | Symbol | Side | Qty | Price | Amount | Commission")
        for row in fills[:10]:
            print(
                f"{_format_date(row.get('trade_date'))} | {row.get('symbol')} | {row.get('side')} | {row.get('filled_qty')} | "
                f"{_fmt_money(row.get('filled_price'))} | {_fmt_money(row.get('filled_amount'))} | {_fmt_money(row.get('commission'))}"
            )

    print("")
    print("[Integrity Check]")
    print(f"Checks: {validation['checks']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Errors: {validation['errors']}")
    for item in validation["issues"][:20]:
        print(f"- [{item.level}] {item.message}")


def _write_markdown(payload: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    positions = payload["positions"]
    orders = payload["orders"]
    fills = payload["fills"]
    op = payload["operation_status"]
    validation = payload["validation"]
    lines = [
        "# US Stock Paper Trading Report",
        "",
        "## 1. Account Summary",
        "",
        f"- Account ID: {summary.get('account_id')}",
        f"- Snapshot Date: {_format_date(summary.get('snapshot_date'))}",
        f"- Initial Cash: {_fmt_money(summary.get('initial_cash'))}",
        f"- Cash Balance: {_fmt_money(summary.get('cash_balance'))}",
        f"- Market Value: {_fmt_money(summary.get('market_value'))}",
        f"- Equity Value: {_fmt_money(summary.get('equity_value'))}",
        f"- Total PnL: {_fmt_money(summary.get('total_pnl'))}",
        f"- Total PnL %: {_fmt_pct(summary.get('total_pnl_pct'))}",
        f"- Daily Return %: {_fmt_pct(summary.get('daily_return_pct'))}",
        f"- Excess vs SPY: {_fmt_pct(summary.get('excess_return_vs_spy'))}",
        f"- Excess vs QQQ: {_fmt_pct(summary.get('excess_return_vs_qqq'))}",
        "",
        "## 2. Operation Status",
        "",
        f"- Last Rebalance Date: {_format_date(op.get('last_rebalance_date'))}",
        f"- Last Order Date: {_format_date(op.get('last_order_date'))}",
        f"- Last Fill Date: {_format_date(op.get('last_fill_date'))}",
        f"- Last Snapshot Date: {_format_date(op.get('last_snapshot_date'))}",
        f"- Created Orders: {op.get('created_orders')}",
        f"- Rejected Orders: {op.get('rejected_orders')}",
        f"- Error Orders: {op.get('error_orders')}",
        f"- Open Positions: {op.get('open_positions')}",
        f"- Cash Weight: {_fmt_pct(op.get('cash_weight'))}",
        f"- Max Position Weight: {_fmt_pct(op.get('max_position_weight'))}",
        f"- Max Sector Weight: {_fmt_pct(op.get('max_sector_weight'))}",
        f"- Validation Warnings: {op.get('validation_warnings')}",
        f"- Validation Errors: {op.get('validation_errors')}",
        "",
        "## 3. Positions",
        "",
        "| Symbol | Qty | Avg Price | Last Price | Market Value | Unrealized PnL | PnL % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in positions:
        if str(row.get("status") or "").upper() != "OPEN":
            continue
        lines.append(
            f"| {row.get('symbol')} | {row.get('qty')} | {_fmt_money(row.get('avg_price'))} | {_fmt_money(row.get('last_price'))} | "
            f"{_fmt_money(row.get('market_value'))} | {_fmt_money(row.get('unrealized_pnl'))} | {_fmt_pct(row.get('unrealized_pnl_pct'))} |"
        )
    lines.extend(["", "## 4. Recent Orders", "", "| Trade Date | Symbol | Side | Qty | Status | Reject |", "|---|---|---|---:|---|---|"])
    for row in orders[:10]:
        lines.append(
            f"| {_format_date(row.get('trade_date'))} | {row.get('symbol')} | {row.get('side')} | {row.get('order_qty')} | {row.get('status')} | {row.get('reject_reason') or ''} |"
        )
    lines.extend(["", "## 5. Recent Fills", "", "| Fill Date | Symbol | Side | Qty | Price | Amount | Commission |", "|---|---|---|---:|---:|---:|---:|"])
    for row in fills[:10]:
        lines.append(
            f"| {_format_date(row.get('trade_date'))} | {row.get('symbol')} | {row.get('side')} | {row.get('filled_qty')} | {_fmt_money(row.get('filled_price'))} | {_fmt_money(row.get('filled_amount'))} | {_fmt_money(row.get('commission'))} |"
        )
    lines.extend(["", "## 6. Integrity Check", ""])
    if validation["issues"]:
        for item in validation["issues"]:
            lines.append(f"- [{item.level}] {item.message}")
    else:
        lines.append("- OK")
    path = output_dir / f"paper_report_{summary.get('account_id')}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_csv(payload: dict[str, object], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    paths: list[Path] = []
    snapshot_path = output_dir / f"paper_account_snapshot_{summary.get('account_id')}.csv"
    with snapshot_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "account_id", "snapshot_date", "cash_balance", "reserved_cash", "market_value", "equity_value",
                "realized_pnl", "unrealized_pnl", "total_pnl", "total_pnl_pct", "daily_return_pct",
                "spy_return_pct", "qqq_return_pct", "excess_return_vs_spy", "excess_return_vs_qqq", "position_count",
            ],
        )
        writer.writeheader()
        for row in payload["snapshots"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    paths.append(snapshot_path)

    positions_path = output_dir / f"paper_positions_{summary.get('account_id')}.csv"
    with positions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "account_id", "symbol", "qty", "avg_price", "cost_amount", "last_price", "last_price_date",
                "market_value", "unrealized_pnl", "unrealized_pnl_pct", "realized_pnl", "status",
            ],
        )
        writer.writeheader()
        for row in payload["positions"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    paths.append(positions_path)

    orders_path = output_dir / f"paper_orders_{summary.get('account_id')}.csv"
    with orders_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "paper_order_id", "trade_date", "symbol", "side", "order_type", "order_qty",
                "order_price", "order_amount", "status", "reason", "reject_reason",
            ],
        )
        writer.writeheader()
        for row in payload["orders"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    paths.append(orders_path)

    fills_path = output_dir / f"paper_fills_{summary.get('account_id')}.csv"
    with fills_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "paper_fill_id", "paper_order_id", "trade_date", "symbol", "side", "filled_qty",
                "filled_price", "filled_amount", "commission", "slippage_amount", "fill_status", "created_at",
            ],
        )
        writer.writeheader()
        for row in payload["fills"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    paths.append(fills_path)
    return paths


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    assert_paper_trading_only(account_id=args.account_id, message="[SAFETY] Paper trading evaluation only. Real order APIs are blocked.")
    snapshot_date = parse_iso_date(args.snapshot_date, field_name="snapshot_date") if args.snapshot_date else None
    payload = _load_report_payload(args.account_id, snapshot_date)
    if args.format == "console":
        _print_console(payload, show_positions=args.show_positions or True, show_orders=args.show_orders, show_fills=args.show_fills)
        return 0
    if args.format == "markdown":
        print(_write_markdown(payload, cfg.report_output_dir))
        return 0
    if args.format == "csv":
        for path in _write_csv(payload, cfg.report_output_dir):
            print(path)
        return 0
    raise ValueError(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    raise SystemExit(main())
