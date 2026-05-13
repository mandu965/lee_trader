from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import parse_iso_date
from python.us.us_db import fetch_rank_component_rows_between
from utils.us_live_pre_trade_check import (
    UsLiveOrderCandidate,
    result_to_markdown,
    run_batch_us_live_pre_trade_check,
    run_us_live_pre_trade_check,
)
from utils.us_live_order_approval import create_order_approval_request
from utils.us_live_risk_policy import load_us_live_risk_policy
from utils.us_live_trading_safety import assert_us_live_pre_trade_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US live pre-trade checks without placing any real order.")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--policy-id", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    parser.add_argument("--amount-usd", type=float, default=None)
    parser.add_argument("--qty", type=float, default=None)
    parser.add_argument("--order-type", choices=["LIMIT", "MARKET"], default="LIMIT")
    parser.add_argument("--limit-price", type=float, default=None)
    parser.add_argument("--candidate-source", default="MANUAL")
    parser.add_argument("--strategy-name", default=None)
    parser.add_argument("--from-ranking", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--format", choices=["console", "markdown"], default="console")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-requires-approval", action="store_true")
    parser.add_argument("--create-approval-request", action="store_true")
    parser.add_argument("--requested-by", default="SYSTEM")
    parser.add_argument("--approval-expires-minutes", type=int, default=None)
    parser.add_argument("--replace-existing-approval", action="store_true")
    return parser.parse_args()


def _default_amount(policy: dict[str, object], requested: float | None) -> float:
    order = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    cap = float(order.get("max_order_amount_usd", 50) or 50)
    return min(cap, requested if requested is not None else cap)


def _build_candidate(
    *,
    args: argparse.Namespace,
    policy: dict[str, object],
    symbol: str,
    candidate_source: str,
    strategy_name: str | None,
    rank_row: dict[str, object] | None = None,
) -> UsLiveOrderCandidate:
    return UsLiveOrderCandidate(
        trade_date=args.trade_date,
        account_id=args.account_id,
        policy_id=str(policy.get("policy_id") or args.policy_id or "US_LIVE_RULE_V1"),
        symbol=symbol.upper(),
        side=args.side.upper(),
        requested_order_amount_usd=_default_amount(policy, args.amount_usd),
        requested_qty=args.qty,
        requested_order_type=args.order_type.upper(),
        requested_limit_price=args.limit_price,
        candidate_source=candidate_source,
        strategy_name=strategy_name,
        rank_no=int(rank_row.get("rank_no") or 0) if isinstance(rank_row, dict) and rank_row.get("rank_no") is not None else None,
        recommend_grade=str(rank_row.get("recommend_grade") or "") if isinstance(rank_row, dict) else None,
        total_score=float(rank_row.get("total_score") or 0) if isinstance(rank_row, dict) and rank_row.get("total_score") is not None else None,
        reason=str(rank_row.get("reason_summary") or "") if isinstance(rank_row, dict) else None,
    )


def _manual_candidates(args: argparse.Namespace, policy: dict[str, object]) -> list[UsLiveOrderCandidate]:
    if not args.symbol:
        raise ValueError("--symbol is required unless --from-ranking is used.")
    return [
        _build_candidate(
            args=args,
            policy=policy,
            symbol=args.symbol,
            candidate_source=args.candidate_source,
            strategy_name=args.strategy_name,
        )
    ]


def _ranking_candidates(args: argparse.Namespace, policy: dict[str, object]) -> list[UsLiveOrderCandidate]:
    if args.side.upper() != "BUY":
        raise ValueError("--from-ranking currently supports BUY candidates only.")
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    rows = fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source="rule_v1")
    strategy = policy.get("strategy", {}) if isinstance(policy.get("strategy"), dict) else {}
    buy_grades = {str(item).upper() for item in strategy.get("buy_grades", ["STRONG_BUY", "BUY"])}
    selected = [
        row for row in rows
        if int(row.get("rank_no") or 999999) <= int(args.top_n)
        and str(row.get("recommend_grade") or "").upper() in buy_grades
        and str(row.get("recommend_grade") or "").upper() != "EXCLUDE"
    ]
    selected.sort(key=lambda row: (int(row.get("rank_no") or 999999), -float(row.get("total_score") or 0), str(row.get("symbol") or "")))
    return [
        _build_candidate(
            args=args,
            policy=policy,
            symbol=str(row.get("symbol") or ""),
            candidate_source="RANK",
            strategy_name=str(row.get("source") or "rule_v1"),
            rank_row=row,
        )
        for row in selected
    ]


def _render_console(candidate: UsLiveOrderCandidate, result, approval_row: dict[str, object] | None = None) -> None:
    print("[US Live Pre-Trade Check]")
    print(f"Trade Date: {candidate.trade_date}")
    print(f"Account: {candidate.account_id}")
    print(f"Policy: {candidate.policy_id}")
    print("")
    print(f"Symbol: {candidate.symbol}")
    print(f"Side: {candidate.side}")
    print(f"Amount: {candidate.requested_order_amount_usd} USD")
    print(f"Order Type: {candidate.requested_order_type}")
    print("")
    print(f"Decision: {result.decision}")
    print(f"Severity: {result.severity}")
    print("")
    print("[Check Results]")
    for stage, status in result.check_results.items():
        print(f"{stage}: {status}")
    print("")
    print("[Reason Codes]")
    if result.reason_codes:
        for code in result.reason_codes:
            print(f"- {code}")
    else:
        print("- none")
    if approval_row:
        print("")
        print("[Approval Request]")
        print(f"Approval ID: {approval_row.get('approval_id')}")
        print(f"Status: {approval_row.get('approval_status')}")
        print(f"Expires At: {approval_row.get('expires_at')}")
    print("")
    print("[Safety]")
    print("No real order API was called.")
    print("No live order was created.")


def main() -> int:
    args = parse_args()
    policy = load_us_live_risk_policy(args.policy_id)
    assert_us_live_pre_trade_only(policy_id=args.policy_id, message="[SAFETY] Pre-trade check only. Real order APIs are blocked.")
    candidates = _ranking_candidates(args, policy) if args.from_ranking else _manual_candidates(args, policy)
    if not candidates:
        print("[US Live Pre-Trade Check]")
        print("No candidates found.")
        return 0
    results = run_batch_us_live_pre_trade_check(
        candidates,
        write_block_log=not args.dry_run,
        log_requires_approval=args.log_requires_approval,
    )
    approval_rows: list[dict[str, object] | None] = []
    for candidate, result in zip(candidates, results):
        approval_row = None
        if args.create_approval_request and not args.dry_run and result.decision in {"ALLOW", "REQUIRE_APPROVAL"}:
            approval_row = create_order_approval_request(
                candidate,
                result,
                requested_by=args.requested_by,
                expires_minutes=args.approval_expires_minutes,
                replace_existing=args.replace_existing_approval,
            )
        approval_rows.append(approval_row)
    if args.format == "markdown":
        output_dir = Path(__file__).resolve().parents[2] / "outputs" / "us_stock_live_risk"
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(candidates) == 1:
            path = output_dir / f"pre_trade_check_{candidates[0].symbol}_{candidates[0].side}_{candidates[0].trade_date.replace('-', '')}.md"
            text = result_to_markdown(candidates[0], results[0])
            if approval_rows[0]:
                text += (
                    "\n\n## Approval Request\n\n"
                    f"- Approval ID: {approval_rows[0].get('approval_id')}\n"
                    f"- Status: {approval_rows[0].get('approval_status')}\n"
                    f"- Expires At: {approval_rows[0].get('expires_at')}\n"
                )
            path.write_text(text, encoding="utf-8")
            print(path)
        else:
            path = output_dir / f"pre_trade_check_batch_{args.trade_date.replace('-', '')}_{args.account_id}.md"
            lines = ["# US Live Pre-Trade Check Batch", ""]
            for candidate, result, approval_row in zip(candidates, results, approval_rows):
                lines.append(result_to_markdown(candidate, result))
                if approval_row:
                    lines.append("")
                    lines.append("## Approval Request")
                    lines.append("")
                    lines.append(f"- Approval ID: {approval_row.get('approval_id')}")
                    lines.append(f"- Status: {approval_row.get('approval_status')}")
                    lines.append(f"- Expires At: {approval_row.get('expires_at')}")
                lines.append("")
            path.write_text("\n".join(lines), encoding="utf-8")
            print(path)
    else:
        for candidate, result, approval_row in zip(candidates, results, approval_rows):
            _render_console(candidate, result, approval_row)
            if len(candidates) > 1:
                print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
