from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scoring.final_score import BULL_WEIGHT_PROFILE, DEFENSIVE_WEIGHT_PROFILE, NEUTRAL_WEIGHT_PROFILE


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOC_DIR = ROOT / "docs" / "web_program_docs"
OUTPUT_DIR = ROOT / "outputs"


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2, *, already_percent: bool = False) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    multiplier = 1.0 if already_percent else 100.0
    return f"{float(x) * multiplier:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def _first_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    for splitter in [". ", ".\n"]:
        if splitter in text:
            return text.split(splitter)[0].strip() + "."
    return text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_scoring_method_overview(generated_at: str, ranking_df: pd.DataFrame) -> str:
    latest_regime = str(ranking_df.get("regime", pd.Series(dtype=str)).dropna().mode().iloc[0]) if "regime" in ranking_df.columns and not ranking_df["regime"].dropna().empty else "defensive"
    regime_weights = pd.DataFrame(
        [
            {
                "regime": "bull",
                "ret": _fmt_pct(BULL_WEIGHT_PROFILE.ret),
                "prob": _fmt_pct(BULL_WEIGHT_PROFILE.prob),
                "tech": _fmt_pct(BULL_WEIGHT_PROFILE.tech),
                "quality": _fmt_pct(BULL_WEIGHT_PROFILE.qual),
                "valuation": _fmt_pct(BULL_WEIGHT_PROFILE.valuation),
                "risk_penalty_strength": _fmt_num(BULL_WEIGHT_PROFILE.risk_penalty, 2),
            },
            {
                "regime": "neutral",
                "ret": _fmt_pct(NEUTRAL_WEIGHT_PROFILE.ret),
                "prob": _fmt_pct(NEUTRAL_WEIGHT_PROFILE.prob),
                "tech": _fmt_pct(NEUTRAL_WEIGHT_PROFILE.tech),
                "quality": _fmt_pct(NEUTRAL_WEIGHT_PROFILE.qual),
                "valuation": _fmt_pct(NEUTRAL_WEIGHT_PROFILE.valuation),
                "risk_penalty_strength": _fmt_num(NEUTRAL_WEIGHT_PROFILE.risk_penalty, 2),
            },
            {
                "regime": "defensive",
                "ret": _fmt_pct(DEFENSIVE_WEIGHT_PROFILE.ret),
                "prob": _fmt_pct(DEFENSIVE_WEIGHT_PROFILE.prob),
                "tech": _fmt_pct(DEFENSIVE_WEIGHT_PROFILE.tech),
                "quality": _fmt_pct(DEFENSIVE_WEIGHT_PROFILE.qual),
                "valuation": _fmt_pct(DEFENSIVE_WEIGHT_PROFILE.valuation),
                "risk_penalty_strength": _fmt_num(DEFENSIVE_WEIGHT_PROFILE.risk_penalty, 2),
            },
        ]
    )
    return (
        "# Scoring Method Overview\n\n"
        f"- generated_at: {generated_at}\n"
        f"- latest_snapshot_date: {ranking_df['date'].max().strftime('%Y-%m-%d') if 'date' in ranking_df.columns else 'NA'}\n"
        f"- latest_detected_regime: {latest_regime}\n\n"
        "## What The Score Means\n\n"
        "The system builds one final recommendation score for each stock. This score is not a single forecast. "
        "It combines expected return, the chance of ranking near the top, technical trend quality, business quality, valuation support, "
        "and a risk penalty for names expected to suffer larger drawdowns.\n\n"
        "In plain language, a higher score means the stock looks better on several dimensions at the same time, not just on one strong signal.\n\n"
        "## Main Inputs\n\n"
        "- `ret_score`: expected return signal from the production model.\n"
        "- `prob_score`: how often the stock looks like a likely top-ranked candidate versus peers on the same date.\n"
        "- `tech_score`: trend, momentum, stability, and volume-based technical quality.\n"
        "- `quality_score`: financial quality inputs such as profitability and balance-sheet strength.\n"
        "- `valuation_score`: valuation support when cheaper fundamentals improve the setup.\n"
        "- `risk_penalty`: a deduction when predicted drawdown risk is high.\n\n"
        "## Regime-Based Weights\n\n"
        "The system changes weights depending on market regime. In stronger markets it leans more on return and technical momentum. "
        "In defensive markets it gives more room to quality, valuation, and drawdown control.\n\n"
        f"{_markdown_table(regime_weights, list(regime_weights.columns))}\n\n"
        "## How To Read It\n\n"
        "- A high score with high confidence is stronger than a high score with weak confidence.\n"
        "- A stock can still rank well in a defensive regime even if its momentum is not the strongest, as long as quality and risk are better.\n"
        "- The score is relative within each date. It is most useful for comparing candidates against each other, not as a standalone promise of future return.\n"
    )


def build_confidence_guide(generated_at: str, calibration_df: pd.DataFrame) -> str:
    cal_5d = calibration_df.loc[
        (calibration_df["source_mode"] == "operational") & (calibration_df["horizon_days"] == 5)
    ].copy()
    cal_5d["bucket_hit_rate"] = cal_5d["hit_rate"].map(lambda x: _fmt_pct(x))
    cal_5d["avg_raw_confidence_score"] = cal_5d["avg_raw_confidence_score"].map(_fmt_num)
    cal_5d["calibrated_confidence_score"] = cal_5d["calibrated_confidence_score"].map(_fmt_num)
    cal_5d = cal_5d.rename(
        columns={
            "bucket_label": "bucket",
            "rows": "sample_rows",
            "avg_raw_confidence_score": "avg_raw_confidence",
            "bucket_hit_rate": "realized_hit_rate",
            "calibrated_confidence_score": "operational_calibrated",
            "status": "bucket_status",
        }
    )
    monotonic_ok = (
        len(cal_5d.loc[cal_5d["bucket_status"].eq("stable")]) >= 2
        and cal_5d.loc[cal_5d["bucket_status"].eq("stable"), "hit_rate"].is_monotonic_increasing
    )
    return (
        "# Confidence Interpretation Guide\n\n"
        f"- generated_at: {generated_at}\n"
        "- source: operational confidence calibration table\n"
        f"- 5d_monotonicity_check: {'pass' if monotonic_ok else 'fail'}\n\n"
        "## What Confidence Means\n\n"
        "Confidence is not expected return. It is a reliability signal. It describes how much trust the system has in the current recommendation, "
        "based on data quality, model reliability, signal agreement, and fit with the current market regime.\n\n"
        "## Practical Interpretation\n\n"
        "- `80-100`: strong internal conviction, but still not a guarantee.\n"
        "- `60-80`: usable signal, but current evidence should be checked against liquidity and market conditions.\n"
        "- Below `60`: usually not suitable for operational buy lists unless there is a special reason.\n\n"
        "## Current Operational Calibration Reality\n\n"
        "The live calibration dataset is still small. On the 5-day horizon, the current operational history does not show a clean monotonic relationship "
        "between higher confidence and better realized hit rate.\n\n"
        f"{_markdown_table(cal_5d[['bucket', 'sample_rows', 'avg_raw_confidence', 'realized_hit_rate', 'operational_calibrated', 'bucket_status']], ['bucket', 'sample_rows', 'avg_raw_confidence', 'realized_hit_rate', 'operational_calibrated', 'bucket_status'])}\n\n"
        "## Current User Guidance\n\n"
        "- Treat confidence as a supporting indicator, not as standalone buy permission.\n"
        "- High confidence still requires buy-gate approval and acceptable liquidity.\n"
        "- Because operational monotonicity is not stable yet, confidence is currently best read as provisional rather than fully calibrated.\n"
    )


def build_buy_gate_summary(generated_at: str, gate: dict) -> str:
    rows = []
    for decision in gate.get("decisions", []):
        rows.append(
            {
                "bucket": f"top{decision['bucket']}",
                "status": decision["status"],
                "avg_final_score": _fmt_num(decision["static"].get("avg_final_score")),
                "avg_confidence_score": _fmt_num(decision["static"].get("avg_confidence_score")),
                "liquidity_risk_ratio": _fmt_pct(decision["static"].get("liquidity_risk_ratio")),
                "matured_benchmark_dates": str(decision["benchmark"].get("matured_dates_max")),
                "confidence_reliable": str(decision["confidence"].get("reliable")),
                "reason": decision["reason_summary"],
            }
        )
    frame = pd.DataFrame(rows)
    return (
        "# Operational Buy Gate Summary\n\n"
        f"- generated_at: {generated_at}\n"
        f"- asof_date: {gate.get('asof_date', 'NA')}\n"
        f"- overall_status: {gate.get('overall_status', 'NA')}\n"
        f"- primary_bucket: top{gate.get('primary_bucket', 'NA')}\n"
        f"- daily_cycle_status: {gate.get('daily_cycle_status', 'NA')}\n\n"
        "## What This Gate Does\n\n"
        "The buy gate is the final operational safety layer. It prevents the system from automatically approving a live buy list "
        "when benchmark outperformance has not been confirmed, confidence calibration is still weak, or liquidity risk is too high.\n\n"
        "## Latest Decision Snapshot\n\n"
        f"{_markdown_table(frame, list(frame.columns))}\n\n"
        "## How To Read The Status\n\n"
        "- `BUY_ALLOWED`: live evidence is strong enough to support buying.\n"
        "- `WATCH`: some positive evidence exists, but it is not strong enough for automatic approval.\n"
        "- `HOLD`: the current list may still be interesting, but the system does not yet have enough mature evidence.\n"
        "- `BLOCK`: risk conditions are strong enough that the list should not be used for buying.\n\n"
        "## Current Message\n\n"
        f"The current system status is `{gate.get('overall_status', 'NA')}`. For the primary `top{gate.get('primary_bucket', 'NA')}` bucket, "
        "the main blocker is lack of mature benchmark evidence and insufficiently reliable operational confidence calibration.\n"
    )


def build_current_portfolio_rationale(generated_at: str, portfolio_df: pd.DataFrame, gate: dict) -> str:
    shown = portfolio_df.copy()
    shown["target_weight"] = shown["target_weight"].map(_fmt_pct)
    shown["final_score"] = shown["final_score"].map(_fmt_num)
    shown["confidence_score"] = shown["confidence_score"].map(_fmt_num)
    shown["liquidity_score"] = shown["liquidity_score"].map(_fmt_num)
    shown["comment"] = shown["explain_text"].map(_first_sentence)
    shown["keep_from_previous"] = shown["keep_from_previous"].map(lambda x: "kept" if bool(x) else "new")
    cash_buffer = _fmt_pct(portfolio_df["cash_buffer"].iloc[0]) if not portfolio_df.empty else "NA"
    return (
        "# Current Portfolio Rationale\n\n"
        f"- generated_at: {generated_at}\n"
        f"- asof_date: {portfolio_df['asof_date'].iloc[0] if not portfolio_df.empty else 'NA'}\n"
        "- portfolio_reference: model_portfolio_top5\n"
        f"- operational_gate_status: {gate.get('overall_status', 'NA')}\n"
        f"- cash_buffer: {cash_buffer}\n\n"
        "## Portfolio Construction Summary\n\n"
        "This is the current model portfolio proposal built from the Top5 buy candidates. "
        "Weights are not equal. Higher-confidence, higher-score names receive more weight, while weaker-liquidity names are reduced. "
        "A small cash buffer is reserved by default.\n\n"
        "## Current Holdings\n\n"
        f"{_markdown_table(shown[['portfolio_rank', 'code', 'name', 'target_weight', 'final_score', 'confidence_score', 'liquidity_score', 'keep_from_previous', 'comment']], ['portfolio_rank', 'code', 'name', 'target_weight', 'final_score', 'confidence_score', 'liquidity_score', 'keep_from_previous', 'comment'])}\n\n"
        "## Plain-Language Takeaway\n\n"
        "The current proposal is concentrated in high-scoring names with relatively strong confidence, but it should still be treated as a model portfolio. "
        "The buy gate currently remains on HOLD, so this portfolio is suitable for monitoring and paper-trading review rather than automatic live deployment.\n"
    )


def build_recent_performance_summary(
    generated_at: str,
    nav_df: pd.DataFrame,
    forward_report: str,
    benchmark_report: str,
) -> str:
    latest_nav = nav_df.sort_values(["strategy", "date"]).groupby("strategy").tail(1).copy()
    latest_nav["cumulative_return_pct"] = latest_nav["cumulative_return"].map(_fmt_pct)
    latest_nav["running_mdd_pct"] = latest_nav["drawdown"].map(_fmt_pct)
    latest_nav["nav"] = latest_nav["nav"].map(_fmt_num)
    latest_nav["open_positions"] = latest_nav["active_position_count"].astype(str)
    latest_nav["closed_positions"] = latest_nav["closed_trade_count"].astype(str)
    maturity_line = "All tracked forward-return horizons are currently immature."
    benchmark_line = "Benchmark comparison is not yet mature enough to show excess return."
    if "latest_snapshot_state" in forward_report and "immature" not in forward_report:
        maturity_line = "At least one forward-return horizon is mature."
    if "| 0             |" not in benchmark_report:
        benchmark_line = "At least one benchmark comparison horizon is mature."
    return (
        "# Recent Performance Summary\n\n"
        f"- generated_at: {generated_at}\n"
        "- source_set: paper_trading_nav, operational_forward_return_report, benchmark_comparison_report\n\n"
        "## Paper Trading Snapshot\n\n"
        f"{_markdown_table(latest_nav[['strategy', 'date', 'nav', 'cumulative_return_pct', 'running_mdd_pct', 'open_positions', 'closed_positions']], ['strategy', 'date', 'nav', 'cumulative_return_pct', 'running_mdd_pct', 'open_positions', 'closed_positions'])}\n\n"
        "## Forward Return Tracking\n\n"
        f"- {maturity_line}\n"
        "- The latest operational snapshot date is 2026-03-27, so 5d/20d/60d/90d forward returns have not matured yet.\n\n"
        "## Benchmark Context\n\n"
        f"- {benchmark_line}\n"
        "- KOSPI and equal-weight universe benchmarks are available.\n"
        "- KOSDAQ benchmark series is currently unavailable in the local dataset.\n\n"
        "## What Users Should Take From This\n\n"
        "The paper portfolio can already be monitored day by day, but hard claims about true excess return are still premature. "
        "At this stage the system is better viewed as a monitored research-and-operations stack than a fully validated live allocation engine.\n"
    )


def build_known_limitations(
    generated_at: str,
    gate: dict,
    calibration_df: pd.DataFrame,
    benchmark_report: str,
    portfolio_df: pd.DataFrame,
) -> str:
    stable_5d = calibration_df.loc[
        (calibration_df["source_mode"] == "operational") & (calibration_df["horizon_days"] == 5) & (calibration_df["status"] == "stable")
    ].copy()
    no_theme_weight = portfolio_df.groupby("dominant_theme")["target_weight"].sum().get("(none)", 0.0) if not portfolio_df.empty else 0.0
    limitations = [
        "Operational forward-return history is still too short. The current archive has only one latest snapshot date for the newest cycle, so all main horizons are still immature.",
        "Benchmark outperformance is not yet proven. The buy gate remains on HOLD because mature benchmark dates are still below the required threshold.",
        "Confidence calibration is not yet trustworthy enough for live use. On the 5-day horizon, higher confidence has not shown a clean monotonic improvement in realized hit rate.",
        f"Theme diversification is currently weak. The latest model portfolio still allocates about {_fmt_pct(no_theme_weight)} to `(none)` theme names because alternative theme coverage is limited.",
        "Top8 and Top10 lists carry more liquidity risk than Top5. The current operational gate blocks them because the very-low-liquidity ratio is too high.",
        "KOSDAQ benchmark comparison is not available in the local dataset, so cross-market benchmark review is incomplete.",
        "Paper trading is still in the early stage. Current portfolios have no closed 20-day trades yet, so realized win rate is not meaningful.",
    ]
    stable_rows = stable_5d[["bucket_label", "rows", "hit_rate"]].copy()
    if not stable_rows.empty:
        stable_rows["hit_rate"] = stable_rows["hit_rate"].map(_fmt_pct)
    body = [
        "# Known Limitations",
        "",
        f"- generated_at: {generated_at}",
        f"- current_buy_gate_status: {gate.get('overall_status', 'NA')}",
        "",
        "## Current Limitations",
        "",
        *[f"- {item}" for item in limitations],
        "",
        "## Supporting Detail",
        "",
    ]
    if not stable_rows.empty:
        body.extend(
            [
                "Stable 5d confidence buckets:",
                "",
                _markdown_table(stable_rows, list(stable_rows.columns)),
                "",
            ]
        )
    body.append(
        "These limitations do not mean the system is unusable. They mean the current state is appropriate for monitored operation, paper trading, and continued evidence collection rather than automatic live deployment."
    )
    return "\n".join(body) + "\n"


def main() -> int:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    ranking_path = DATA_DIR / "ranking_final.csv"
    ranking_df = pd.read_csv(ranking_path)
    if "date" in ranking_df.columns:
        ranking_df["date"] = pd.to_datetime(ranking_df["date"], errors="coerce")

    gate = load_json(OUTPUT_DIR / "operational_buy_gate.json")
    calibration_df = pd.read_csv(DATA_DIR / "confidence_calibration_operational.csv")
    portfolio_top5 = pd.read_csv(DATA_DIR / "model_portfolio_top5.csv")
    nav_df = pd.read_csv(DATA_DIR / "paper_trading_nav.csv")
    forward_report = (OUTPUT_DIR / "operational_forward_return_report.md").read_text(encoding="utf-8")
    benchmark_report = (OUTPUT_DIR / "benchmark_comparison_report.md").read_text(encoding="utf-8")

    documents = {
        "scoring_method_overview.md": build_scoring_method_overview(generated_at, ranking_df),
        "confidence_interpretation_guide.md": build_confidence_guide(generated_at, calibration_df),
        "operational_buy_gate_summary.md": build_buy_gate_summary(generated_at, gate),
        "current_portfolio_rationale.md": build_current_portfolio_rationale(generated_at, portfolio_top5, gate),
        "recent_performance_summary.md": build_recent_performance_summary(generated_at, nav_df, forward_report, benchmark_report),
        "known_limitations.md": build_known_limitations(generated_at, gate, calibration_df, benchmark_report, portfolio_top5),
    }

    for filename, content in documents.items():
        (DOC_DIR / filename).write_text(content, encoding="utf-8")
        print(f"wrote: {DOC_DIR / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
