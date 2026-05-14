from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from python.us.trade_orchestration.config import TradeOrchestrationConfig


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_integrated_report(
    orchestration_result: dict[str, object],
    cfg: TradeOrchestrationConfig,
) -> dict[str, object]:
    sell_report = dict(orchestration_result.get("sell_report") or {})
    buy_report = dict(orchestration_result.get("buy_report") or {})
    portfolio_state = dict(orchestration_result.get("portfolio_state") or {})
    conflict_summary = dict(orchestration_result.get("conflict_summary") or {})

    total_invested = 0.0
    current_value = 0.0
    unrealized_pnl = 0.0
    for position in portfolio_state.get("open_positions") or []:
        qty = float(position.get("remaining_quantity") or 0.0)
        avg_entry = float(position.get("avg_entry_price") or 0.0)
        latest_price = float(position.get("latest_price") or 0.0)
        total_invested += qty * avg_entry
        current_value += qty * latest_price
        unrealized_pnl += float(position.get("unrealized_pnl") or 0.0)

    symbol_rows: list[dict[str, object]] = []
    symbol_set = sorted(
        {
            str(item.get("symbol") or "").upper()
            for item in (portfolio_state.get("open_positions") or [])
        }
        | {
            str(item.get("symbol") or "").upper()
            for item in (buy_report.get("candidates") or [])
        }
        | {
            str(item.get("symbol") or "").upper()
            for item in (sell_report.get("decisions") or [])
        }
    )
    buy_map = {str(item.get("symbol") or "").upper(): item for item in buy_report.get("candidates") or []}
    sell_map = {str(item.get("symbol") or "").upper(): item for item in sell_report.get("decisions") or []}
    conflict_map = {str(item.get("symbol") or "").upper(): item for item in orchestration_result.get("conflict_results") or []}
    open_map = portfolio_state.get("open_position_map") or {}

    for symbol in symbol_set:
        buy_item = buy_map.get(symbol, {})
        sell_item = sell_map.get(symbol, {})
        conflict_item = conflict_map.get(symbol, {})
        position_item = open_map.get(symbol, {})
        final_action = "NONE"
        reasons: list[str] = []
        if sell_item:
            final_action = str(sell_item.get("decision") or "NONE")
            if sell_item.get("exit_reason"):
                reasons.append(str(sell_item.get("exit_reason")))
        elif buy_item:
            final_action = "BUY" if buy_item.get("allowed") else "BLOCK"
            reasons.extend([str(reason) for reason in buy_item.get("block_reasons") or []])
        if conflict_item and conflict_item.get("conflict_reasons"):
            reasons.extend([str(reason) for reason in conflict_item.get("conflict_reasons") or []])
        symbol_rows.append(
            {
                "symbol": symbol,
                "current_position_status": position_item.get("status"),
                "buy_decision": "ALLOW" if buy_item.get("allowed") else ("BLOCK" if buy_item else None),
                "sell_decision": sell_item.get("decision"),
                "conflict_result": conflict_item.get("conflict_reasons"),
                "final_action": final_action,
                "reasons": reasons,
            }
        )

    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_date": orchestration_result.get("trade_date"),
        "mode": orchestration_result.get("mode"),
        "orchestration_enabled": orchestration_result.get("enabled"),
        "success": orchestration_result.get("success"),
        "fail_safe_triggered": orchestration_result.get("fail_safe_triggered", False),
        "sell_summary": {
            "loaded_positions": sell_report.get("loaded_positions", 0),
            "hold": sell_report.get("hold_positions", 0),
            "sell": sell_report.get("sell_signals", 0),
            "partial_sell": sell_report.get("partial_sell_signals", 0),
            "review_required": sell_report.get("review_required", 0),
            "exit_reason_summary": sell_report.get("reason_summary", {}),
        },
        "buy_summary": {
            "loaded_candidates": buy_report.get("loaded_candidates", 0),
            "risk_guard_passed": buy_report.get("allowed_before_conflict", 0),
            "conflict_guard_passed": buy_report.get("allowed_after_conflict", 0),
            "final_buy_candidates": buy_report.get("allowed_candidates", 0),
            "block_summary": buy_report.get("block_summary", {}),
        },
        "conflict_summary": conflict_summary,
        "paper_portfolio_summary": {
            "open_position_count": len(portfolio_state.get("open_positions") or []),
            "total_invested_amount": round(total_invested, 6),
            "current_value": round(current_value, 6),
            "unrealized_paper_pnl": round(unrealized_pnl, 6),
            "realized_paper_pnl": None,
        },
        "symbol_details": symbol_rows,
        "warnings": list(orchestration_result.get("warnings") or []),
        "portfolio_state_status": portfolio_state.get("status"),
    }


def render_integrated_report_console(report: dict[str, object]) -> str:
    conflict_summary = report.get("conflict_summary") or {}
    lines = [
        "[US TRADE ORCHESTRATION]",
        f"mode={report.get('mode')}",
        f"enabled={1 if report.get('orchestration_enabled') else 0}",
        f"trade_date={report.get('trade_date')}",
        "",
        "SELL:",
        f"loaded_positions={report.get('sell_summary', {}).get('loaded_positions', 0)}",
        f"hold={report.get('sell_summary', {}).get('hold', 0)}",
        f"sell={report.get('sell_summary', {}).get('sell', 0)}",
        f"partial_sell={report.get('sell_summary', {}).get('partial_sell', 0)}",
        f"review_required={report.get('sell_summary', {}).get('review_required', 0)}",
        "",
        "BUY:",
        f"loaded_candidates={report.get('buy_summary', {}).get('loaded_candidates', 0)}",
        f"allowed_before_conflict={report.get('buy_summary', {}).get('risk_guard_passed', 0)}",
        f"conflict_blocked={conflict_summary.get('TOTAL_CONFLICT_BLOCKED', 0)}",
        f"allowed_after_conflict={report.get('buy_summary', {}).get('conflict_guard_passed', 0)}",
        "",
        "CONFLICT:",
    ]
    for key in (
        "OPEN_POSITION_EXISTS",
        "SELL_SIGNAL_EXISTS",
        "REVIEW_REQUIRED_SYMBOL",
        "COOLDOWN_ACTIVE",
        "DUPLICATE_BUY",
        "PORTFOLIO_STATE_INCONSISTENT",
    ):
        lines.append(f"{key}={conflict_summary.get(key, 0)}")
    lines.extend(
        [
            "",
            "PAPER:",
            f"paper_buy_orders={report.get('buy_summary', {}).get('final_buy_candidates', 0)}",
            f"paper_sell_orders={report.get('sell_summary', {}).get('sell', 0)}",
            "",
            f"final_status={'SUCCESS' if report.get('success') else 'FAILED'}",
        ]
    )
    return "\n".join(lines)


def render_integrated_report_markdown(report: dict[str, object]) -> str:
    lines = [
        "# US Integrated Trade Report",
        "",
        "## Execution Summary",
        f"- Trade Date: `{report.get('trade_date')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Orchestration Enabled: `{report.get('orchestration_enabled')}`",
        f"- Success: `{report.get('success')}`",
        f"- Fail Safe Triggered: `{report.get('fail_safe_triggered')}`",
        "",
        "## SELL Summary",
        f"- Loaded Positions: `{report.get('sell_summary', {}).get('loaded_positions', 0)}`",
        f"- HOLD: `{report.get('sell_summary', {}).get('hold', 0)}`",
        f"- SELL: `{report.get('sell_summary', {}).get('sell', 0)}`",
        f"- PARTIAL_SELL: `{report.get('sell_summary', {}).get('partial_sell', 0)}`",
        f"- REVIEW_REQUIRED: `{report.get('sell_summary', {}).get('review_required', 0)}`",
        "",
        "## BUY Summary",
        f"- Loaded Candidates: `{report.get('buy_summary', {}).get('loaded_candidates', 0)}`",
        f"- Risk Guard Passed: `{report.get('buy_summary', {}).get('risk_guard_passed', 0)}`",
        f"- Conflict Guard Passed: `{report.get('buy_summary', {}).get('conflict_guard_passed', 0)}`",
        f"- Final BUY Candidates: `{report.get('buy_summary', {}).get('final_buy_candidates', 0)}`",
        "",
        "## Conflict Summary",
    ]
    for key, value in (report.get("conflict_summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Paper Portfolio Summary",
            f"- Open Positions: `{report.get('paper_portfolio_summary', {}).get('open_position_count', 0)}`",
            f"- Total Invested Amount: `{report.get('paper_portfolio_summary', {}).get('total_invested_amount')}`",
            f"- Current Value: `{report.get('paper_portfolio_summary', {}).get('current_value')}`",
            f"- Unrealized Paper PnL: `{report.get('paper_portfolio_summary', {}).get('unrealized_paper_pnl')}`",
            f"- Realized Paper PnL: `{report.get('paper_portfolio_summary', {}).get('realized_paper_pnl')}`",
            "",
            "## Symbol Details",
        ]
    )
    if not report.get("symbol_details"):
        lines.append("- none")
    else:
        for row in report["symbol_details"]:
            lines.append(
                f"- `{row.get('symbol')}` position=`{row.get('current_position_status')}` buy=`{row.get('buy_decision')}` sell=`{row.get('sell_decision')}` final=`{row.get('final_action')}` reasons=`{', '.join(row.get('reasons') or []) or 'none'}`"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "- All values are Paper-only review artifacts.",
            "- No real account balance or real account position is used.",
            "- No broker API or real BUY/SELL order path is enabled.",
        ]
    )
    return "\n".join(lines)


def write_integrated_report_json(report: dict[str, object], cfg: TradeOrchestrationConfig) -> Path:
    cfg.report_output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.report_output_dir / f"{report.get('trade_date')}_integrated_trade_report.json"
    path.write_text(_json_text(report), encoding="utf-8")
    return path


def write_integrated_report_markdown(report: dict[str, object], cfg: TradeOrchestrationConfig) -> Path:
    cfg.report_output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.report_output_dir / f"{report.get('trade_date')}_integrated_trade_report.md"
    path.write_text(render_integrated_report_markdown(report), encoding="utf-8")
    return path
