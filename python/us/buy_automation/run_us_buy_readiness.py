from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.buy_automation.live_readiness_evaluator import evaluate_live_readiness
from python.us.buy_automation.paper_backtest_summary import build_paper_backtest_summary
from python.us.buy_automation.promotion_policy import load_live_promotion_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PAPER/SHADOW BUY automation readiness for future LIVE review.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--format", default="console", choices=["console", "json", "markdown"])
    return parser.parse_args()


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _render_console(report: dict[str, object]) -> str:
    perf = report.get("paper_performance_summary") or {}
    ops = report.get("operational_stability") or {}
    lines = [
        "[US BUY LIVE READINESS]",
        f"evaluation_date={str(date.today())}",
        f"period_days={report.get('evaluation_period_days')}",
        f"benchmark={report.get('benchmark_symbol')}",
        f"live_ready={str(report.get('live_ready')).lower()}",
        f"decision={report.get('decision')}",
        f"readiness_score={report.get('readiness_score')}",
        "",
        f"paper_order_count={perf.get('paper_order_count', 0)}",
        f"total_return_pct={perf.get('total_return_pct')}",
        f"benchmark_return_pct={perf.get('benchmark_return_pct')}",
        f"excess_return_pct={perf.get('excess_return_pct')}",
        f"win_rate={perf.get('win_rate')}",
        f"max_drawdown_pct={perf.get('max_drawdown_pct')}",
        f"data_missing_rate={perf.get('data_missing_rate')}",
        f"scheduler_success_rate={ops.get('scheduler_success_rate')}",
        f"manual_approval_required={str(report.get('manual_approval_required')).lower()}",
        "",
        "reasons:",
    ]
    reasons = report.get("reasons") or []
    if not reasons:
        lines.append("- none")
    else:
        for reason in reasons:
            lines.append(f"- {reason}")
    lines.append("")
    lines.append("LIVE automatic promotion is prohibited.")
    return "\n".join(lines)


def _render_markdown(report: dict[str, object], windows: list[dict[str, object]]) -> str:
    perf = report.get("paper_performance_summary") or {}
    ops = report.get("operational_stability") or {}
    lines = [
        "# US BUY LIVE Readiness Report",
        "",
        "## Overview",
        f"- Evaluation Generated At: `{report.get('evaluation_generated_at')}`",
        f"- Evaluation Period Days: `{report.get('evaluation_period_days')}`",
        f"- Benchmark: `{report.get('benchmark_symbol')}`",
        f"- Live Ready: `{report.get('live_ready')}`",
        f"- Decision: `{report.get('decision')}`",
        f"- Readiness Score: `{report.get('readiness_score')}`",
        f"- Manual Approval Required: `{report.get('manual_approval_required')}`",
        "",
        "## Selected Period Performance",
        f"- Paper Order Count: `{perf.get('paper_order_count', 0)}`",
        f"- Total Return Pct: `{perf.get('total_return_pct')}`",
        f"- Benchmark Return Pct: `{perf.get('benchmark_return_pct')}`",
        f"- Excess Return Pct: `{perf.get('excess_return_pct')}`",
        f"- Win Rate: `{perf.get('win_rate')}`",
        f"- Max Drawdown Pct: `{perf.get('max_drawdown_pct')}`",
        f"- Data Missing Rate: `{perf.get('data_missing_rate')}`",
        "",
        "## Operational Stability",
        f"- Shadow Days: `{ops.get('shadow_days')}`",
        f"- Paper Days: `{ops.get('paper_days')}`",
        f"- Scheduler Success Rate: `{ops.get('scheduler_success_rate')}`",
        f"- Report Success Rate: `{ops.get('report_success_rate')}`",
        f"- Invalid Decision Log Count: `{ops.get('invalid_decision_log_count')}`",
        f"- LIVE Disabled In Scheduler Count: `{ops.get('live_disabled_in_scheduler_count')}`",
        "",
        "## Period Windows",
    ]
    for window in windows:
        lines.append(
            f"- `{window.get('period_label')}`: paper_orders=`{window.get('paper_order_count')}` total_return_pct=`{window.get('total_return_pct')}` excess_return_pct=`{window.get('excess_return_pct')}` status=`{window.get('status')}`"
        )
    lines.extend(["", "## Not Ready Reasons"])
    reasons = report.get("reasons") or []
    if not reasons:
        lines.append("- none")
    else:
        for reason in reasons:
            lines.append(f"- `{reason}`")
    lines.extend(
        [
            "",
            "## Safety Notice",
            "- `live_ready=true` means review-eligible only.",
            "- Manual approval remains mandatory.",
            "- LIVE automatic promotion is prohibited.",
        ]
    )
    return "\n".join(lines)


def _write_json(report: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{date.today().isoformat()}_live_readiness.json"
    path.write_text(_json_text(report), encoding="utf-8")
    return path


def _write_markdown(report: dict[str, object], windows: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{date.today().isoformat()}_live_readiness.md"
    path.write_text(_render_markdown(report, windows), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    policy = load_live_promotion_policy()
    report = evaluate_live_readiness(days=args.days, benchmark_symbol=args.benchmark, policy=policy)
    windows = [
        build_paper_backtest_summary(days=20, benchmark_symbol=report["benchmark_symbol"], compare_qqq=policy.compare_qqq),
        build_paper_backtest_summary(days=60, benchmark_symbol=report["benchmark_symbol"], compare_qqq=policy.compare_qqq),
        build_paper_backtest_summary(days=120, benchmark_symbol=report["benchmark_symbol"], compare_qqq=policy.compare_qqq),
        build_paper_backtest_summary(days=None, benchmark_symbol=report["benchmark_symbol"], compare_qqq=policy.compare_qqq),
    ]
    final_payload = dict(report)
    final_payload["period_windows"] = windows
    final_payload["evaluation_date"] = date.today().isoformat()
    if args.format == "console":
        print(_render_console(final_payload))
    elif args.format == "json":
        path = _write_json(final_payload, policy.output_dir)
        print(path)
    else:
        path = _write_markdown(final_payload, windows, policy.output_dir)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
