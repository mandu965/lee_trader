from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.generate_us_stock_paper_orders import build_paper_orders
from python.us.paper_rebalance import load_policy
from python.us.us_config import load_us_paper_trading_config, parse_iso_date
from python.us.us_db import fetch_rank_component_rows_between, fetch_us_paper_account_rows, fetch_us_paper_order_rows, fetch_us_paper_position_rows
from utils.paper_trading_safety import assert_paper_trading_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan US paper trading rebalance without creating orders.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _print_console(*, trade_date: date, account_id: str, orders: list[dict[str, object]], counts: dict[str, object]) -> None:
    created = [row for row in orders if str(row.get("status") or "") == "CREATED"]
    sells = [row for row in created if str(row.get("side") or "") == "SELL"]
    buys = [row for row in created if str(row.get("side") or "") == "BUY"]
    print("[Paper Rebalance Plan]")
    print(f"Trade Date: {trade_date.isoformat()}")
    print(f"Account: {account_id}")
    print("")
    print(f"Current Positions: {counts['positions_read']}")
    print(f"Target Candidates: {counts['target_candidates']}")
    print(f"Sell Candidates: {counts['sell_candidates']}")
    print(f"Buy Candidates: {counts['buy_candidates']}")
    print(f"Hold Candidates: {counts['hold_candidates']}")
    if sells:
        print("")
        print("[SELL Plan]")
        print("Symbol | Qty | Reason")
        for row in sells[:20]:
            print(f"{row.get('symbol')} | {row.get('order_qty')} | {row.get('reason')}")
    if buys:
        print("")
        print("[BUY Plan]")
        print("Symbol | Rank | Grade | Amount | Reason")
        for row in buys[:20]:
            print(f"{row.get('symbol')} | {row.get('rank_no')} | {row.get('recommend_grade')} | {row.get('order_amount')} | {row.get('reason')}")


def _write_markdown(*, trade_date: date, account_id: str, orders: list[dict[str, object]], counts: dict[str, object], output_dir: Path) -> Path:
    created = [row for row in orders if str(row.get("status") or "") == "CREATED"]
    sells = [row for row in created if str(row.get("side") or "") == "SELL"]
    buys = [row for row in created if str(row.get("side") or "") == "BUY"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"paper_rebalance_plan_{account_id}_{trade_date:%Y%m%d}.md"
    lines = [
        "# US Paper Rebalance Plan",
        "",
        f"- Trade Date: {trade_date.isoformat()}",
        f"- Account: {account_id}",
        f"- Current Positions: {counts['positions_read']}",
        f"- Target Candidates: {counts['target_candidates']}",
        f"- Sell Candidates: {counts['sell_candidates']}",
        f"- Buy Candidates: {counts['buy_candidates']}",
        f"- Hold Candidates: {counts['hold_candidates']}",
        "",
        "## SELL Plan",
        "",
        "| Symbol | Qty | Reason |",
        "|---|---:|---|",
    ]
    for row in sells:
        lines.append(f"| {row.get('symbol')} | {row.get('order_qty')} | {row.get('reason')} |")
    lines.extend(["", "## BUY Plan", "", "| Symbol | Rank | Grade | Amount | Reason |", "|---|---:|---|---:|---|"])
    for row in buys:
        lines.append(f"| {row.get('symbol')} | {row.get('rank_no')} | {row.get('recommend_grade')} | {row.get('order_amount')} | {row.get('reason')} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    cfg = load_us_paper_trading_config(account_id=args.account_id)
    assert_paper_trading_only(account_id=args.account_id, message="[SAFETY] Paper trading only. Real order APIs are blocked.")
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    account_rows = fetch_us_paper_account_rows(account_id=args.account_id)
    if not account_rows:
        raise RuntimeError(f"Paper account not found: {args.account_id}")
    account_row = account_rows[0]
    policy = load_policy(args.account_id)
    rank_rows = fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source="rule_v1")
    position_rows = fetch_us_paper_position_rows(account_id=args.account_id, status="OPEN")
    existing_order_rows = fetch_us_paper_order_rows(account_id=args.account_id, trade_date=trade_date, strategy_name=f"US_RANK_{policy.selection_rule}")
    orders, counts = build_paper_orders(
        trade_date=trade_date,
        account_row=account_row,
        rank_rows=rank_rows,
        position_rows=position_rows,
        existing_order_rows=existing_order_rows,
        policy=policy,
        side_option="ALL",
        replace_existing=False,
    )
    if args.format == "console":
        _print_console(trade_date=trade_date, account_id=args.account_id, orders=orders, counts=counts)
        return 0
    path = _write_markdown(trade_date=trade_date, account_id=args.account_id, orders=orders, counts=counts, output_dir=cfg.report_output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
