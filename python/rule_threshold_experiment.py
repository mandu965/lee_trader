from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

import rule_backtest
import rule_daily_report
import rule_order_preview_builder
import rule_paper_state_manager
import rule_portfolio_manager
import rule_signal_builder
from rule_signal_builder import ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "rule_threshold_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-only RULE threshold experiments.")
    parser.add_argument("--v2-thresholds", default="45,50,55")
    parser.add_argument("--rule-score-min", type=float, default=70.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--run-mode", default=os.getenv("RULE_TRADING_RUN_MODE", "paper"))
    return parser.parse_args()


@contextmanager
def patched_env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("at least one v2 threshold is required")
    return out


def tag_for(rule_score_min: float, v2_min: float) -> str:
    score = f"{rule_score_min:g}".replace(".", "p")
    v2 = f"{v2_min:g}".replace(".", "p")
    return f"score{score}_v2{v2}"


def _fmt_pct(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "NA"


def latest_summary(signals: pd.DataFrame, plan: dict, preview: dict, backtest_report: dict) -> dict:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    plan_items = {str(item.get("code") or "").zfill(6): item for item in plan.get("items") or []}
    buy_items = [item for item in preview.get("items") or [] if item.get("side") == "BUY"]
    top_buy = []
    for item in buy_items[:10]:
        code = str(item.get("code") or "").zfill(6)
        plan_item = plan_items.get(code, {})
        top_buy.append(
            {
            "code": item.get("code"),
            "name": item.get("name"),
            "sector": plan_item.get("sector"),
            "rule_score": plan_item.get("rule_score"),
            "rule_score_v2": plan_item.get("rule_score_v2"),
            "order_qty": item.get("order_qty"),
            "order_amount": item.get("order_amount"),
            "expected_execution_price": item.get("expected_execution_price"),
            "order_block_reason": item.get("order_block_reason"),
            }
        )
    return {
        "signal_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "candidate_count": int(len(latest)),
        "entry_signal_count": int(latest.get("entry_signal", pd.Series(False, index=latest.index)).sum()),
        "strong_entry_signal_count": int(latest.get("strong_entry_signal", pd.Series(False, index=latest.index)).sum()),
        "gap_risk_blocked_count": int(latest.get("gap_risk_blocked", pd.Series(False, index=latest.index)).sum()),
        "trading_value_failed_count": int((~latest.get("trading_value_pass", pd.Series(True, index=latest.index))).sum()),
        "portfolio_summary": plan.get("summary") or {},
        "order_preview_summary": preview.get("summary") or {},
        "top_buy_preview": top_buy,
        "backtest_entry_signal": (backtest_report.get("summary") or {}).get("entry_signal") or {},
        "backtest_strong_entry_signal": (backtest_report.get("summary") or {}).get("strong_entry_signal") or {},
        "portfolio_equity_strong_entry": (backtest_report.get("portfolio_equity_curve") or {}).get("strong_entry_signal") or {},
        "sector_strong_entry_top": ((backtest_report.get("sector") or {}).get("strong_entry_signal") or [])[:5],
    }


def _fmt_float(value: object, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def write_experiment_md(summary: dict) -> str:
    lines = [
        "# RULE Threshold Experiment Report",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- run_mode: `{summary['run_mode']}`",
        "",
        "| tag | rule_score_min | rule_score_v2_min | latest strong | buy preview | D+5 avg | D+20 avg | D+60 avg | D+20 win | portfolio MDD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["experiments"]:
        strong = item["latest"]["strong_entry_signal_count"]
        buy = (item["latest"].get("order_preview_summary") or {}).get("buy_preview_count", 0)
        metrics = item["latest"].get("backtest_strong_entry_signal") or {}
        d5 = metrics.get("avg_return_d5")
        d20 = metrics.get("avg_return_d20")
        d60 = metrics.get("avg_return_d60")
        win = metrics.get("win_rate_d20")
        portfolio_metrics = item["latest"].get("portfolio_equity_strong_entry") or {}
        portfolio_mdd = portfolio_metrics.get("mdd_d20_portfolio_equity")
        lines.append(
            "| {tag} | {score:g} | {v2:g} | {strong} | {buy} | {d5} | {d20} | {d60} | {win} | {mdd} |".format(
                tag=item["tag"],
                score=item["rule_score_min"],
                v2=item["rule_score_v2_min"],
                strong=strong,
                buy=buy,
                d5="NA" if d5 is None else f"{float(d5) * 100:.2f}%",
                d20="NA" if d20 is None else f"{float(d20) * 100:.2f}%",
                d60="NA" if d60 is None else f"{float(d60) * 100:.2f}%",
                win="NA" if win is None else f"{float(win) * 100:.2f}%",
                mdd="NA" if portfolio_mdd is None else f"{float(portfolio_mdd) * 100:.2f}%",
            )
        )
    lines.append("")
    for item in summary["experiments"]:
        lines.extend(
            [
                f"## {item['tag']}",
                "",
                f"- latest strong_entry_signal: `{item['latest']['strong_entry_signal_count']}`",
                f"- portfolio buy_count: `{(item['latest'].get('portfolio_summary') or {}).get('buy_count', 0)}`",
                f"- order buy_preview_count: `{(item['latest'].get('order_preview_summary') or {}).get('buy_preview_count', 0)}`",
                f"- strong D+5 / D+20 / D+60: `{_fmt_pct((item['latest'].get('backtest_strong_entry_signal') or {}).get('avg_return_d5'))}` / `{_fmt_pct((item['latest'].get('backtest_strong_entry_signal') or {}).get('avg_return_d20'))}` / `{_fmt_pct((item['latest'].get('backtest_strong_entry_signal') or {}).get('avg_return_d60'))}`",
                f"- portfolio equity MDD D+20: `{_fmt_pct((item['latest'].get('portfolio_equity_strong_entry') or {}).get('mdd_d20_portfolio_equity'))}`",
                "",
            ]
        )
        top = item["latest"].get("top_buy_preview") or []
        if not top:
            lines.append("_No buy preview items._")
            lines.append("")
            continue
        lines.extend(["| code | name | sector | rule_score_v2 | rule_score | qty | amount | expected_price | block_reason |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
        for row in top:
            lines.append(
                "| {code} | {name} | {sector} | {v2:.2f} | {score:.2f} | {qty} | {amount:.0f} | {price:.0f} | {reason} |".format(
                    code=row.get("code") or "",
                    name=row.get("name") or "",
                    sector=row.get("sector") or "",
                    v2=float(row.get("rule_score_v2") or 0.0),
                    score=float(row.get("rule_score") or 0.0),
                    qty=int(row.get("order_qty") or 0),
                    amount=float(row.get("order_amount") or 0.0),
                    price=float(row.get("expected_execution_price") or 0.0),
                    reason=row.get("order_block_reason") or "",
                )
            )
        lines.append("")
        sectors = item["latest"].get("sector_strong_entry_top") or []
        if sectors:
            lines.extend(["| sector | trades | D+20 avg | D+20 win | payoff |", "| --- | ---: | ---: | ---: | ---: |"])
            for row in sectors:
                lines.append(
                    "| {group} | {trades} | {d20} | {win} | {payoff} |".format(
                        group=row.get("group") or "",
                        trades=int(row.get("trade_count") or 0),
                        d20=_fmt_pct(row.get("avg_return_d20")),
                        win=_fmt_pct(row.get("win_rate_d20")),
                        payoff=_fmt_float(row.get("payoff_ratio_d20")),
                    )
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.v2_thresholds)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_mode": args.run_mode,
        "experiments": [],
    }

    for v2_min in thresholds:
        tag = tag_for(args.rule_score_min, v2_min)
        env = {
            "RULE_STRONG_RULE_SCORE_MIN": f"{args.rule_score_min:g}",
            "RULE_STRONG_RULE_SCORE_V2_MIN": f"{v2_min:g}",
        }
        with patched_env(env):
            signals = rule_signal_builder.build_signals(run_mode=args.run_mode)
            signals_csv = out_dir / f"rule_signals_{tag}.csv"
            out_signals = signals.copy()
            out_signals["date"] = pd.to_datetime(out_signals["date"]).dt.strftime("%Y-%m-%d")
            out_signals.to_csv(signals_csv, index=False, encoding="utf-8-sig")

            backtest_report = rule_backtest.build_report(signals, rule_backtest.read_prices(rule_signal_builder.DEFAULT_PRICES))
            backtest_json = out_dir / f"rule_strategy_backtest_report_{tag}.json"
            backtest_md = out_dir / f"rule_strategy_backtest_report_{tag}.md"
            backtest_json.write_text(json.dumps(backtest_report, ensure_ascii=False, indent=2), encoding="utf-8")
            backtest_md.write_text(rule_backtest.render_md(backtest_report), encoding="utf-8")

            account_state = rule_paper_state_manager.default_state()
            plan = rule_portfolio_manager.build_rule_portfolio_plan(signals, account_state, args.run_mode)
            plan_json = out_dir / f"rule_portfolio_plan_{tag}.json"
            intents_json = out_dir / f"rule_trade_intents_{tag}.json"
            intents = rule_portfolio_manager.build_trade_intents(plan)
            plan_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            intents_json.write_text(json.dumps(intents, ensure_ascii=False, indent=2), encoding="utf-8")

            preview = rule_order_preview_builder.build_rule_order_preview(plan, args.run_mode)
            preview_json = out_dir / f"rule_order_preview_{tag}.json"
            preview_json.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

            daily_md = out_dir / f"rule_strategy_daily_report_{tag}.md"
            daily_md.write_text(rule_daily_report.render_report(signals, plan, preview), encoding="utf-8")

        summary["experiments"].append(
            {
                "tag": tag,
                "rule_score_min": args.rule_score_min,
                "rule_score_v2_min": v2_min,
                "files": {
                    "signals_csv": str(signals_csv),
                    "backtest_json": str(backtest_json),
                    "backtest_md": str(backtest_md),
                    "portfolio_plan_json": str(plan_json),
                    "trade_intents_json": str(intents_json),
                    "order_preview_json": str(preview_json),
                    "daily_report_md": str(daily_md),
                },
                "latest": latest_summary(signals, plan, preview, backtest_report),
            }
        )

    summary_json = out_dir / "rule_threshold_experiment_summary.json"
    summary_md = out_dir / "rule_threshold_experiment_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(write_experiment_md(summary), encoding="utf-8")
    print(f"saved {summary_json}")
    print(f"saved {summary_md}")


if __name__ == "__main__":
    main()
