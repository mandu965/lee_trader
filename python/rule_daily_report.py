from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from rule_signal_builder import ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_PLAN = OUTPUT_DIR / "rule_portfolio_plan.json"
DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_OUT = OUTPUT_DIR / "rule_strategy_daily_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RULE daily paper report.")
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--portfolio-plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule signals not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [
        "entry_signal",
        "strong_entry_signal",
        "gap_risk_blocked",
        "trading_value_pass",
        "market_entry_allowed",
        "market_defensive_mode",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["date"])


def load_json(path: Path) -> dict:
    path = resolve(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _num(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def render_report(signals: pd.DataFrame, plan: dict, preview: dict) -> str:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    preview_items = preview.get("items") or []
    block_reasons = Counter()
    for item in preview_items:
        reason = str(item.get("order_block_reason") or "none")
        for part in reason.split(";"):
            if part:
                block_reasons[part] += 1

    strong = latest.loc[latest.get("strong_entry_signal", False).fillna(False)].copy()
    strong = strong.sort_values(["rule_score_v2", "rule_score"], ascending=[False, False]).head(20)
    score_threshold_rows = build_score_threshold_rows(latest)
    v2_threshold_rows = build_v2_threshold_rows(latest)
    tv_threshold_rows = build_trading_value_rows(latest)

    lines = [
        "# RULE Strategy Daily Report",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- signal_date: `{latest_date.date().isoformat() if pd.notna(latest_date) else 'NA'}`",
        f"- run_mode: `{preview.get('run_mode') or plan.get('run_mode') or 'paper'}`",
        f"- paper_only: `{preview.get('paper_only', True)}`",
        "",
        "## Candidate Summary",
        "",
        f"- total_candidates: `{len(latest)}`",
        f"- entry_signal: `{int(latest.get('entry_signal', pd.Series(False, index=latest.index)).sum())}`",
        f"- strong_entry_signal: `{int(latest.get('strong_entry_signal', pd.Series(False, index=latest.index)).sum())}`",
        f"- gap_risk_blocked: `{int(latest.get('gap_risk_blocked', pd.Series(False, index=latest.index)).sum())}`",
        f"- trading_value_failed: `{int((~latest.get('trading_value_pass', pd.Series(True, index=latest.index))).sum())}`",
        "",
        "## Score Threshold Diagnostics",
        "",
        "| rule_score_min | candidates | entry_signal | strong_entry_signal |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in score_threshold_rows:
        lines.append(
            f"| {row['threshold']} | {row['candidate_count']} | {row['entry_count']} | {row['strong_count']} |"
        )
    lines.extend(
        [
            "",
            "## rule_score_v2 Threshold Diagnostics",
            "",
            "| rule_score_min | rule_score_v2_min | candidates | entry_signal | simulated_strong_entry |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in v2_threshold_rows:
        lines.append(
            f"| {row['rule_score_min']} | {row['rule_score_v2_min']} | {row['candidate_count']} | {row['entry_count']} | {row['simulated_strong_entry_count']} |"
        )
    lines.extend(
        [
            "",
            "## Trading Value Filter Diagnostics",
            "",
            "| trading_value_ma20_min | pass_count | pass_rate | strong_entry_pass_count |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tv_threshold_rows:
        lines.append(
            f"| {row['threshold']:,.0f} | {row['pass_count']} | {row['pass_rate']:.2%} | {row['strong_entry_pass_count']} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio Plan",
            "",
            f"- hold_count: `{(plan.get('summary') or {}).get('hold_count', 0)}`",
            f"- buy_count: `{(plan.get('summary') or {}).get('buy_count', 0)}`",
            f"- reduce_count: `{(plan.get('summary') or {}).get('reduce_count', 0)}`",
            f"- exit_count: `{(plan.get('summary') or {}).get('exit_count', 0)}`",
            f"- skip_count: `{(plan.get('summary') or {}).get('skip_count', 0)}`",
            "",
            "## Order Preview",
            "",
            f"- request_count: `{(preview.get('summary') or {}).get('request_count', 0)}`",
            f"- buy_preview_count: `{(preview.get('summary') or {}).get('buy_preview_count', 0)}`",
            f"- sell_preview_count: `{(preview.get('summary') or {}).get('sell_preview_count', 0)}`",
            f"- order_allowed_count: `{(preview.get('summary') or {}).get('order_allowed_count', 0)}`",
            "",
            "## Block Reasons",
            "",
        ]
    )
    if block_reasons:
        lines.extend(f"- {reason}: `{count}`" for reason, count in block_reasons.most_common())
    else:
        lines.append("- none")

    lines.extend(["", "## Strong Entry Candidates", ""])
    if strong.empty:
        lines.append("_No strong entry candidates._")
    else:
        lines.append("| code | name | sector | rule_score_v2 | rule_score | expected_entry_price |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for _, row in strong.iterrows():
            lines.append(
                "| {code} | {name} | {sector} | {v2:.2f} | {v1:.2f} | {price:.0f} |".format(
                    code=row.get("code", ""),
                    name=str(row.get("name") or ""),
                    sector=str(row.get("sector") or ""),
                    v2=_num(row.get("rule_score_v2")),
                    v1=_num(row.get("rule_score")),
                    price=_num(row.get("expected_entry_price")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def build_score_threshold_rows(latest: pd.DataFrame) -> list[dict[str, int]]:
    rows = []
    score = pd.to_numeric(latest.get("rule_score"), errors="coerce")
    entry = latest.get("entry_signal", pd.Series(False, index=latest.index)).fillna(False)
    strong = latest.get("strong_entry_signal", pd.Series(False, index=latest.index)).fillna(False)
    for threshold in [60, 65, 70, 75, 80]:
        mask = score >= threshold
        rows.append(
            {
                "threshold": threshold,
                "candidate_count": int(mask.sum()),
                "entry_count": int((mask & entry).sum()),
                "strong_count": int((mask & strong).sum()),
            }
        )
    return rows


def build_v2_threshold_rows(latest: pd.DataFrame) -> list[dict[str, int]]:
    rows = []
    rule_score = pd.to_numeric(latest.get("rule_score"), errors="coerce")
    rule_score_v2 = pd.to_numeric(latest.get("rule_score_v2"), errors="coerce")
    entry = latest.get("entry_signal", pd.Series(False, index=latest.index)).fillna(False)
    trading_value_pass = latest.get("trading_value_pass", pd.Series(True, index=latest.index)).fillna(True)
    gap_ok = ~latest.get("gap_risk_blocked", pd.Series(False, index=latest.index)).fillna(False)
    market_ok = latest.get("market_entry_allowed", pd.Series(True, index=latest.index)).fillna(True)
    for threshold in [35, 40, 45, 50, 55, 60]:
        mask = (rule_score >= 70) & (rule_score_v2 >= threshold) & trading_value_pass & gap_ok & market_ok
        rows.append(
            {
                "rule_score_min": 70,
                "rule_score_v2_min": threshold,
                "candidate_count": int(mask.sum()),
                "entry_count": int((mask & entry).sum()),
                "simulated_strong_entry_count": int(mask.sum()),
            }
        )
    return rows


def build_trading_value_rows(latest: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    tv = pd.to_numeric(latest.get("trading_value_ma_20"), errors="coerce")
    strong = latest.get("strong_entry_signal", pd.Series(False, index=latest.index)).fillna(False)
    valid = tv.notna()
    denom = max(int(valid.sum()), 1)
    for threshold in [500_000_000, 1_000_000_000, 2_000_000_000]:
        mask = tv >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "pass_count": int(mask.sum()),
                "pass_rate": float(mask.sum() / denom),
                "strong_entry_pass_count": int((mask & strong).sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    signals = load_signals(args.signals_csv)
    plan = load_json(args.portfolio_plan_json)
    preview = load_json(args.order_preview_json)
    out = resolve(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(signals, plan, preview), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
