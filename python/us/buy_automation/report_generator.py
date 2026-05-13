from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from python.us.buy_automation.paper_performance import build_paper_performance
from python.us.buy_automation.validation_summary import summarize_validation
from python.us.buy_automation.notification_formatter import (
    format_notification_detail,
    format_notification_summary,
)


def report_output_dir() -> Path:
    root_dir = Path(__file__).resolve().parents[3]
    raw = str(os.environ.get("US_BUY_REPORT_OUTPUT_DIR", "reports/lee_trader_us/buy_automation")).strip() or "reports/lee_trader_us/buy_automation"
    path = Path(raw)
    return path if path.is_absolute() else root_dir / path


def log_input_dir() -> Path:
    root_dir = Path(__file__).resolve().parents[3]
    raw = str(os.environ.get("US_BUY_LOG_INPUT_DIR", "output/us_stock_buy_automation")).strip() or "output/us_stock_buy_automation"
    path = Path(raw)
    return path if path.is_absolute() else root_dir / path


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def load_buy_automation_run_log(*, trade_date: str | None = None, input_dir: str | None = None) -> dict[str, object]:
    directory = Path(input_dir) if input_dir else log_input_dir()
    if not directory.exists():
        raise FileNotFoundError(f"BUY automation log directory not found: {directory}")
    files = sorted(directory.glob("buy_automation_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No BUY automation JSON logs found under: {directory}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if trade_date is None or str(payload.get("trade_date")) == str(trade_date):
            payload["_source_json_path"] = str(path)
            return payload
    raise FileNotFoundError(f"No BUY automation JSON log found for trade_date={trade_date}")


def build_buy_report(raw_log: dict[str, object]) -> dict[str, object]:
    candidates = list(raw_log.get("candidates") or [])
    allowed_rows = [row for row in candidates if row.get("allowed")]
    blocked_rows = [row for row in candidates if not row.get("allowed")]
    validation = summarize_validation(candidates)
    paper_performance = build_paper_performance(
        list(raw_log.get("paper_orders") or []),
        benchmark_symbol=os.environ.get("US_BUY_BENCHMARK_SYMBOL", "SPY"),
    )
    config_snapshot = dict(raw_log.get("config_snapshot") or {})
    max_daily_amount = float(config_snapshot.get("max_daily_amount_usd") or 0.0)
    max_daily_symbols = int(config_snapshot.get("max_daily_symbols") or 0)
    used_amount = sum(float(row.get("allocated_amount_usd") or 0.0) for row in allowed_rows)
    used_symbols = len(allowed_rows)
    data_missing = bool(validation.get("data_missing_symbols")) or bool(raw_log.get("events"))
    fail_safe_triggered = int(validation.get("fail_safe_block_count", 0) or 0) > 0
    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_date": raw_log.get("trade_date"),
        "mode": raw_log.get("mode"),
        "automation_enabled": raw_log.get("enabled"),
        "source_json_path": raw_log.get("_source_json_path"),
        "loaded_candidates": raw_log.get("loaded_candidates", 0),
        "allowed_candidates": len(allowed_rows),
        "blocked_candidates": len(blocked_rows),
        "paper_order_count": len(raw_log.get("paper_orders", [])),
        "final_candidates": allowed_rows,
        "blocked_candidates_detail": blocked_rows,
        "candidates": candidates,
        "events": list(raw_log.get("events") or []),
        "config_snapshot": config_snapshot,
        "amount_limit_usage_pct": round((used_amount / max_daily_amount), 6) if max_daily_amount > 0 else None,
        "symbol_limit_usage_pct": round((used_symbols / max_daily_symbols), 6) if max_daily_symbols > 0 else None,
        "used_amount_usd": round(used_amount, 6),
        "used_symbol_count": used_symbols,
        "data_missing": data_missing,
        "fail_safe_triggered": fail_safe_triggered,
        "validation_summary": validation,
        "paper_orders": list(raw_log.get("paper_orders") or []),
        "paper_performance": paper_performance,
        "live_transition_readiness": "NOT_EVALUATED",
        "notification_summary": format_notification_summary({"validation_summary": validation, **raw_log}),
        "notification_detail": "",
    }


def render_buy_report_console(report: dict[str, object]) -> str:
    validation = report.get("validation_summary", {})
    lines = [
        "[US BUY Automation Report]",
        f"generated_at={report.get('report_generated_at')}",
        f"trade_date={report.get('trade_date')}",
        f"mode={report.get('mode')}",
        f"automation_enabled={1 if report.get('automation_enabled') else 0}",
        "",
        f"loaded_candidates={report.get('loaded_candidates', 0)}",
        f"allowed_candidates={report.get('allowed_candidates', 0)}",
        f"blocked_candidates={report.get('blocked_candidates', 0)}",
        f"paper_orders={report.get('paper_order_count', 0)}",
        "",
        f"amount_limit_usage_pct={report.get('amount_limit_usage_pct')}",
        f"symbol_limit_usage_pct={report.get('symbol_limit_usage_pct')}",
        f"data_missing={report.get('data_missing')}",
        f"fail_safe_triggered={report.get('fail_safe_triggered')}",
        "",
        "[Block Summary]",
    ]
    for reason, count in (validation.get("block_counts") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("[Rule Summary]")
    for rule_name, stats in (validation.get("rule_summary") or {}).items():
        lines.append(
            f"- {rule_name}: PASS {stats.get('PASS', 0)} / FAIL {stats.get('FAIL', 0)} / UNKNOWN {stats.get('UNKNOWN', 0)}"
        )
    return "\n".join(lines)


def render_buy_report_markdown(report: dict[str, object]) -> str:
    validation = report.get("validation_summary", {})
    paper_summary = report.get("paper_performance", {}).get("summary", {}) if isinstance(report.get("paper_performance"), dict) else {}
    lines = [
        "# US BUY Automation Report",
        "",
        "## Overview",
        f"- Report Generated At: `{report.get('report_generated_at')}`",
        f"- Trade Date: `{report.get('trade_date')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Automation Enabled: `{report.get('automation_enabled')}`",
        "",
        "## Daily Summary",
        f"- Loaded Candidates: `{report.get('loaded_candidates', 0)}`",
        f"- Allowed Candidates: `{report.get('allowed_candidates', 0)}`",
        f"- Blocked Candidates: `{report.get('blocked_candidates', 0)}`",
        f"- PAPER Orders: `{report.get('paper_order_count', 0)}`",
        f"- Amount Limit Usage: `{report.get('amount_limit_usage_pct')}`",
        f"- Symbol Limit Usage: `{report.get('symbol_limit_usage_pct')}`",
        f"- Data Missing: `{report.get('data_missing')}`",
        f"- Fail-Safe Triggered: `{report.get('fail_safe_triggered')}`",
        "",
        "## Block Summary",
    ]
    for reason, count in (validation.get("block_counts") or {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Allowed Candidates"])
    if not report.get("final_candidates"):
        lines.append("- none")
    else:
        for row in report["final_candidates"]:
            lines.append(
                f"- `{row.get('symbol')}` rank=`{row.get('rank')}` score=`{row.get('score')}` prob=`{row.get('probability')}` amount=`{row.get('allocated_amount_usd')}`"
            )
    lines.extend(["", "## Blocked Candidates"])
    if not report.get("blocked_candidates_detail"):
        lines.append("- none")
    else:
        for row in report["blocked_candidates_detail"]:
            lines.append(
                f"- `{row.get('symbol')}` rank=`{row.get('rank')}` score=`{row.get('score')}` reasons=`{', '.join(row.get('block_reasons') or [])}`"
            )
    lines.extend(
        [
            "",
            "## Validation Notes",
            f"- Invalid Decision Logs: `{', '.join(validation.get('invalid_decision_logs', [])) or 'none'}`",
            f"- Parse Errors: `{', '.join(validation.get('parse_errors', [])) or 'none'}`",
            f"- Data Missing Symbols: `{', '.join(validation.get('data_missing_symbols', [])) or 'none'}`",
            f"- Rule Not Ready Symbols: `{', '.join(validation.get('rule_not_ready_symbols', [])) or 'none'}`",
            "",
            "## PAPER Performance",
            f"- Benchmark Symbol: `{report.get('paper_performance', {}).get('benchmark_symbol')}`",
            f"- Tracked Orders: `{paper_summary.get('count', 0)}`",
            f"- Price Data Missing Count: `{paper_summary.get('price_data_missing_count', 0)}`",
            "",
            "## Limitations",
            "- This report does not send any order.",
            "- This report does not call broker APIs.",
            "- LIVE transition readiness is not evaluated here and remains `NOT_EVALUATED`.",
        ]
    )
    return "\n".join(lines)


def write_buy_report_json(report: dict[str, object], *, output_dir: str | None = None) -> Path:
    directory = Path(output_dir) if output_dir else report_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.get('trade_date')}_buy_report.json"
    path.write_text(_json_text(report), encoding="utf-8")
    return path


def write_buy_report_markdown(report: dict[str, object], *, output_dir: str | None = None) -> Path:
    directory = Path(output_dir) if output_dir else report_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.get('trade_date')}_buy_report.md"
    path.write_text(render_buy_report_markdown(report), encoding="utf-8")
    return path


def finalize_buy_report(raw_log: dict[str, object]) -> dict[str, object]:
    report = build_buy_report(raw_log)
    report["notification_detail"] = format_notification_detail(report)
    return report
