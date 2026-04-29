from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import rule_backtest
import rule_order_preview_builder
import rule_paper_state_manager
import rule_portfolio_manager
import rule_signal_builder
from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "rule_formula_review.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "rule_formula_review.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review RULE formula alternatives against current baseline.")
    parser.add_argument("--run-mode", default="paper")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    val = _float(value)
    return "NA" if val is None else f"{val * 100:.2f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    val = _float(value)
    return "NA" if val is None else f"{val:.{digits}f}"


def additive_score(frame: pd.DataFrame) -> pd.Series:
    trend = pd.to_numeric(frame.get("trend_component"), errors="coerce").fillna(0.0)
    liquidity = pd.to_numeric(frame.get("liquidity_component"), errors="coerce").fillna(0.0)
    stability = pd.to_numeric(frame.get("stability_component"), errors="coerce").fillna(0.0)
    regime = pd.to_numeric(frame.get("regime_component"), errors="coerce").fillna(0.0)
    score = (0.35 * trend) + (0.30 * liquidity) + (0.20 * stability) + (0.15 * regime)
    return (score * 100.0).clip(0, 100)


def percentile_rank_by_date(frame: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(frame.get(col), errors="coerce")
    return values.groupby(frame["date"]).rank(pct=True, ascending=True, method="average")


def base_conditions(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame.get("entry_signal"), index=frame.index).fillna(False) | pd.Series(frame.get("strong_entry_signal"), index=frame.index).fillna(False)


def prepare_variant(signals: pd.DataFrame, variant_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = signals.copy()
    meta: dict[str, Any] = {"name": variant_name}
    current_v2 = pd.to_numeric(frame.get("rule_score_v2"), errors="coerce")
    current_score = pd.to_numeric(frame.get("rule_score"), errors="coerce")
    add_v2 = additive_score(frame)
    base = base_conditions(frame)

    if variant_name == "baseline_current":
        meta["description"] = "Current multiplicative v2 and current strong threshold."
        meta["score_column"] = "rule_score_v2"
        meta["strong_rule"] = "rule_score >= 70 AND rule_score_v2 >= 50"
    elif variant_name == "current_relaxed_v2_45":
        frame["strong_entry_signal"] = base & (current_score >= 70.0) & (current_v2 >= 45.0)
        frame["signal_strength"] = np.where(frame["strong_entry_signal"], "strong_entry", np.where(frame["entry_signal"], "entry", "none"))
        meta["description"] = "Current multiplicative v2 with relaxed strong threshold."
        meta["score_column"] = "rule_score_v2"
        meta["strong_rule"] = "rule_score >= 70 AND rule_score_v2 >= 45"
    elif variant_name == "additive_v2_60":
        frame["rule_score_v2"] = add_v2
        frame["strong_entry_signal"] = base & (current_score >= 70.0) & (frame["rule_score_v2"] >= 60.0)
        frame["signal_strength"] = np.where(frame["strong_entry_signal"], "strong_entry", np.where(frame["entry_signal"], "entry", "none"))
        meta["description"] = "Additive v2 with fixed threshold."
        meta["score_column"] = "rule_score_v2_additive"
        meta["strong_rule"] = "rule_score >= 70 AND additive_v2 >= 60"
    elif variant_name == "additive_top15":
        frame["rule_score_v2"] = add_v2
        top_rank = percentile_rank_by_date(frame, "rule_score_v2")
        frame["strong_entry_signal"] = base & (current_score >= 70.0) & (frame["rule_score_v2"] >= 58.0) & (top_rank >= 0.85)
        frame["signal_strength"] = np.where(frame["strong_entry_signal"], "strong_entry", np.where(frame["entry_signal"], "entry", "none"))
        meta["description"] = "Additive v2 with top-15% daily rank gate."
        meta["score_column"] = "rule_score_v2_additive"
        meta["strong_rule"] = "rule_score >= 70 AND additive_v2 >= 58 AND daily_rank >= 85pct"
    else:
        raise ValueError(f"unknown variant: {variant_name}")

    return frame, meta


def latest_summary(signals: pd.DataFrame, plan: dict[str, Any], preview: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    latest_date = pd.to_datetime(signals["date"]).max()
    latest = signals.loc[pd.to_datetime(signals["date"]) == latest_date].copy()
    return {
        "signal_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "strong_entry_signal_count": int(pd.Series(latest.get("strong_entry_signal"), index=latest.index).fillna(False).sum()),
        "entry_signal_count": int(pd.Series(latest.get("entry_signal"), index=latest.index).fillna(False).sum()),
        "buy_count": int((plan.get("summary") or {}).get("buy_count") or 0),
        "preview_buy_count": int((preview.get("summary") or {}).get("buy_preview_count") or 0),
        "top_strong": (
            latest.loc[pd.Series(latest.get("strong_entry_signal"), index=latest.index).fillna(False)]
            .sort_values(["rule_score_v2", "rule_score"], ascending=[False, False])
            .head(5)[["code", "name", "sector", "rule_score", "rule_score_v2", "expected_entry_price"]]
            .where(pd.notna, None)
            .to_dict(orient="records")
        ),
        "strong_metrics": (report.get("summary") or {}).get("strong_entry_signal") or {},
        "portfolio_metrics": (report.get("portfolio_equity_curve") or {}).get("strong_entry_signal") or {},
        "test_2024_strong_metrics": (report.get("test_summary_2024_plus") or {}).get("strong_entry_signal") or {},
    }


def build_review(signals: pd.DataFrame, run_mode: str) -> dict[str, Any]:
    variants = [
        "baseline_current",
        "current_relaxed_v2_45",
        "additive_v2_60",
        "additive_top15",
    ]
    review_items: list[dict[str, Any]] = []
    default_state = rule_paper_state_manager.default_state()
    for variant in variants:
        variant_signals, meta = prepare_variant(signals, variant)
        report = rule_backtest.build_report(variant_signals, rule_backtest.read_prices(rule_signal_builder.DEFAULT_PRICES))
        plan = rule_portfolio_manager.build_rule_portfolio_plan(variant_signals, default_state, run_mode)
        preview = rule_order_preview_builder.build_rule_order_preview(plan, run_mode)
        review_items.append(
            {
                "variant": variant,
                "meta": meta,
                "latest": latest_summary(variant_signals, plan, preview, report),
                "backtest": {
                    "strong": (report.get("summary") or {}).get("strong_entry_signal") or {},
                    "entry": (report.get("summary") or {}).get("entry_signal") or {},
                    "strong_2024_plus": (report.get("test_summary_2024_plus") or {}).get("strong_entry_signal") or {},
                    "portfolio_strong": (report.get("portfolio_equity_curve") or {}).get("strong_entry_signal") or {},
                },
            }
        )

    recommendation = "baseline_current"
    def rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
        strong = item["backtest"]["strong_2024_plus"]
        portfolio = item["backtest"]["portfolio_strong"]
        avg20 = _float(strong.get("avg_return_d20")) or -999.0
        mdd = abs(_float(portfolio.get("mdd_d20_portfolio_equity")) or 999.0)
        latest_count = float(item["latest"]["strong_entry_signal_count"])
        return (avg20, -mdd, latest_count)
    recommendation = sorted(review_items, key=rank_key, reverse=True)[0]["variant"]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_mode": run_mode,
        "review_items": review_items,
        "recommendation": recommendation,
        "review_note": "Prefer alternatives that improve 2024+ strong D+20 while keeping candidate supply non-zero and portfolio drawdown controlled.",
    }


def render_md(review: dict[str, Any]) -> str:
    lines = [
        "# RULE Formula Review",
        "",
        f"- generated_at: `{review['generated_at']}`",
        f"- run_mode: `{review['run_mode']}`",
        f"- recommendation: `{review['recommendation']}`",
        f"- note: {review['review_note']}",
        "",
        "| variant | rule | latest strong | latest buy | 2024+ D+20 | 2024+ win | portfolio D+20 | portfolio MDD | final return |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in review["review_items"]:
        strong = item["backtest"]["strong_2024_plus"]
        portfolio = item["backtest"]["portfolio_strong"]
        lines.append(
            "| {variant} | {rule} | {latest_strong} | {latest_buy} | {d20} | {win} | {pd20} | {mdd} | {final_return} |".format(
                variant=item["variant"],
                rule=item["meta"]["strong_rule"],
                latest_strong=item["latest"]["strong_entry_signal_count"],
                latest_buy=item["latest"]["buy_count"],
                d20=_fmt_pct(strong.get("avg_return_d20")),
                win=_fmt_pct(strong.get("win_rate_d20")),
                pd20=_fmt_pct(portfolio.get("avg_return_d20")),
                mdd=_fmt_pct(portfolio.get("mdd_d20_portfolio_equity")),
                final_return=_fmt_pct(portfolio.get("final_return_d20_portfolio_equity")),
            )
        )
    lines.append("")
    for item in review["review_items"]:
        lines.extend(
            [
                f"## {item['variant']}",
                "",
                f"- description: {item['meta']['description']}",
                f"- strong_rule: `{item['meta']['strong_rule']}`",
                f"- latest strong candidates: `{item['latest']['strong_entry_signal_count']}`",
                f"- latest portfolio buy_count: `{item['latest']['buy_count']}`",
                f"- latest preview buy_count: `{item['latest']['preview_buy_count']}`",
                f"- strong 2024+ D+5 / D+20 / D+60: `{_fmt_pct(item['backtest']['strong_2024_plus'].get('avg_return_d5'))}` / `{_fmt_pct(item['backtest']['strong_2024_plus'].get('avg_return_d20'))}` / `{_fmt_pct(item['backtest']['strong_2024_plus'].get('avg_return_d60'))}`",
                f"- portfolio strong D+20 / MDD / final: `{_fmt_pct(item['backtest']['portfolio_strong'].get('avg_return_d20'))}` / `{_fmt_pct(item['backtest']['portfolio_strong'].get('mdd_d20_portfolio_equity'))}` / `{_fmt_pct(item['backtest']['portfolio_strong'].get('final_return_d20_portfolio_equity'))}`",
                "",
            ]
        )
        top = item["latest"].get("top_strong") or []
        if top:
            lines.extend(["| code | name | sector | rule_score | variant_score | expected_entry_price |", "| --- | --- | --- | ---: | ---: | ---: |"])
            for row in top:
                lines.append(
                    "| {code} | {name} | {sector} | {score} | {v2} | {price} |".format(
                        code=row.get("code") or "",
                        name=row.get("name") or "",
                        sector=row.get("sector") or "",
                        score=_fmt_num(row.get("rule_score")),
                        v2=_fmt_num(row.get("rule_score_v2")),
                        price=_fmt_num(row.get("expected_entry_price"), 0),
                    )
                )
            lines.append("")
        else:
            lines.extend(["_No strong candidates on latest date._", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    signals = rule_signal_builder.build_signals(run_mode=args.run_mode)
    review = build_review(signals, args.run_mode)
    out_json = resolve(args.out_json)
    out_md = resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(review), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
