from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from ranking_builder import (  # noqa: E402
    BULL_WEIGHT_PROFILE,
    DEFENSIVE_WEIGHT_PROFILE,
    NEUTRAL_WEIGHT_PROFILE_BASELINE,
    NEUTRAL_WEIGHT_PROFILE_EXPERIMENTAL,
)


INPUT_CSV = ROOT / "data" / "ranking_final.csv"
OUTPUT_MD = ROOT / "outputs" / "neutral_tech_ab_test.md"
TOP_N = 20


def _fmt(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(item) for item in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def load_latest() -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = str(df["date"].dropna().max())
    latest = df.loc[df["date"] == latest_date].copy()
    latest["final_score"] = pd.to_numeric(latest["final_score"], errors="coerce")
    latest = latest.sort_values("final_score", ascending=False).reset_index(drop=True)
    score_formula_version = str(latest["score_formula_version"].dropna().iloc[0]) if "score_formula_version" in latest.columns and latest["score_formula_version"].notna().any() else "NA"
    return latest, latest_date, score_formula_version


def safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    sample = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sample) < 2:
        return float("nan")
    return float(sample[left].corr(sample[right]))


def overlap_ratio(df: pd.DataFrame, final_col: str, score_col: str, top_n: int = TOP_N) -> float:
    final_set = set(df.sort_values(final_col, ascending=False).head(top_n)["code"].astype(str))
    comp_set = set(df.sort_values(score_col, ascending=False).head(top_n)["code"].astype(str))
    return len(final_set & comp_set) / float(top_n) if top_n else float("nan")


def top20_driver_frequency(df: pd.DataFrame, final_col: str, top_n: int = TOP_N) -> dict[str, int]:
    top = df.sort_values(final_col, ascending=False).head(top_n)
    values: list[str] = []
    for col in ["score_driver_1", "score_driver_2", "score_driver_3"]:
        if col in top.columns:
            values.extend(top[col].dropna().astype(str).tolist())
    return dict(Counter(values).most_common(10))


def mean_confidence(df: pd.DataFrame, final_col: str, top_n: int = TOP_N) -> float:
    top = df.sort_values(final_col, ascending=False).head(top_n)
    return float(pd.to_numeric(top["confidence_score"], errors="coerce").mean())


def choose_profile(regime: str, neutral_profile: dict[str, float | str]) -> dict[str, float | str]:
    regime_clean = str(regime or "").lower()
    if regime_clean == "bull":
        return BULL_WEIGHT_PROFILE
    if regime_clean == "neutral":
        return neutral_profile
    return DEFENSIVE_WEIGHT_PROFILE


def apply_profile_scores(df: pd.DataFrame, *, score_col: str, profile_col: str, neutral_profile: dict[str, float | str]) -> pd.DataFrame:
    work = df.copy()
    weights = work["regime"].apply(lambda regime: choose_profile(regime, neutral_profile))
    work[profile_col] = weights.apply(lambda item: item["profile"])
    work[f"{score_col}_ret"] = weights.apply(lambda item: float(item["ret"]))
    work[f"{score_col}_prob"] = weights.apply(lambda item: float(item["prob"]))
    work[f"{score_col}_tech"] = weights.apply(lambda item: float(item["tech"]))
    work[f"{score_col}_qual"] = weights.apply(lambda item: float(item["qual"]))
    work[f"{score_col}_valuation"] = weights.apply(lambda item: float(item["valuation"]))
    work[f"{score_col}_risk"] = weights.apply(lambda item: float(item["risk_penalty"]))

    for col in ["ret_score", "prob_score", "tech_score", "qual_score", "valuation_score", "risk_penalty"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    work[score_col] = (
        work[f"{score_col}_ret"] * work["ret_score"]
        + work[f"{score_col}_prob"] * work["prob_score"]
        + work[f"{score_col}_tech"] * work["tech_score"]
        + work[f"{score_col}_qual"] * work["qual_score"]
        + work[f"{score_col}_valuation"] * work["valuation_score"]
        - work[f"{score_col}_risk"] * work["risk_penalty"]
    ).clip(lower=0.0, upper=100.0)
    return work


def collect_metrics(df: pd.DataFrame, final_col: str) -> dict[str, object]:
    return {
        "overlap_final_ret": overlap_ratio(df, final_col, "ret_score"),
        "overlap_final_prob": overlap_ratio(df, final_col, "prob_score"),
        "overlap_final_tech": overlap_ratio(df, final_col, "tech_score"),
        "corr_final_risk_penalty": safe_corr(df, final_col, "risk_penalty"),
        "corr_final_tech": safe_corr(df, final_col, "tech_score"),
        "top20_mean_confidence_score": mean_confidence(df, final_col),
        "top20_driver_frequency": top20_driver_frequency(df, final_col),
    }


def summarize_decision(baseline: dict[str, object], experimental: dict[str, object]) -> tuple[str, list[str]]:
    tech_gain = float(experimental["overlap_final_tech"]) - float(baseline["overlap_final_tech"])
    ret_drop = float(experimental["overlap_final_ret"]) - float(baseline["overlap_final_ret"])
    prob_drop = float(experimental["overlap_final_prob"]) - float(baseline["overlap_final_prob"])
    risk_corr_delta = abs(float(experimental["corr_final_risk_penalty"])) - abs(float(baseline["corr_final_risk_penalty"]))
    confidence_delta = float(experimental["top20_mean_confidence_score"]) - float(baseline["top20_mean_confidence_score"])

    notes = [
        f"- tech overlap delta: {_fmt(tech_gain)}",
        f"- ret overlap delta: {_fmt(ret_drop)}",
        f"- prob overlap delta: {_fmt(prob_drop)}",
        f"- abs corr(final, risk_penalty) delta: {_fmt(risk_corr_delta)}",
        f"- top20 mean confidence delta: {_fmt(confidence_delta)}",
    ]

    passes_tech_goal = float(experimental["overlap_final_tech"]) >= 0.25
    holds_ret_prob = ret_drop >= -0.05 and prob_drop >= -0.05
    risk_not_worse = risk_corr_delta <= 0.03

    if passes_tech_goal and tech_gain > 0 and holds_ret_prob and risk_not_worse:
        return "experimental neutral profile is suitable for follow-up validation", notes
    return "baseline neutral profile is more appropriate", notes


def build_report(df: pd.DataFrame, latest_date: str, score_formula_version: str) -> str:
    baseline_df = apply_profile_scores(
        df,
        score_col="ab_final_baseline",
        profile_col="ab_profile_baseline",
        neutral_profile=NEUTRAL_WEIGHT_PROFILE_BASELINE,
    )
    experimental_df = apply_profile_scores(
        df,
        score_col="ab_final_experimental",
        profile_col="ab_profile_experimental",
        neutral_profile=NEUTRAL_WEIGHT_PROFILE_EXPERIMENTAL,
    )

    baseline_metrics = collect_metrics(baseline_df, "ab_final_baseline")
    experimental_metrics = collect_metrics(experimental_df, "ab_final_experimental")
    decision, decision_notes = summarize_decision(baseline_metrics, experimental_metrics)

    stat = INPUT_CSV.stat()
    rows = [
        ["overlap(final, ret)", _fmt(baseline_metrics["overlap_final_ret"]), _fmt(experimental_metrics["overlap_final_ret"])],
        ["overlap(final, prob)", _fmt(baseline_metrics["overlap_final_prob"]), _fmt(experimental_metrics["overlap_final_prob"])],
        ["overlap(final, tech)", _fmt(baseline_metrics["overlap_final_tech"]), _fmt(experimental_metrics["overlap_final_tech"])],
        ["corr(final, risk_penalty)", _fmt(baseline_metrics["corr_final_risk_penalty"]), _fmt(experimental_metrics["corr_final_risk_penalty"])],
        ["corr(final, tech)", _fmt(baseline_metrics["corr_final_tech"]), _fmt(experimental_metrics["corr_final_tech"])],
        ["top20 mean confidence_score", _fmt(baseline_metrics["top20_mean_confidence_score"]), _fmt(experimental_metrics["top20_mean_confidence_score"])],
        ["top20 dominant driver frequency", str(baseline_metrics["top20_driver_frequency"]), str(experimental_metrics["top20_driver_frequency"])],
    ]

    neutral_rows = int(df["regime"].astype(str).str.lower().eq("neutral").sum())
    lines = [
        "# Neutral Tech A/B Test",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- latest_date: {latest_date}",
        f"- score_formula_version: {score_formula_version}",
        f"- source_ranking_file: {INPUT_CSV.name}; rows={len(df)}; modified_at={datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        "- recomputed_from_current_code: true",
        "",
        "## neutral profile setup",
        f"- baseline: {NEUTRAL_WEIGHT_PROFILE_BASELINE}",
        f"- experimental: {NEUTRAL_WEIGHT_PROFILE_EXPERIMENTAL}",
        f"- latest_date neutral rows: {neutral_rows}",
        "- risk_penalty multiplier is unchanged in this experiment.",
        "",
        "## metric comparison",
        _markdown_table(rows, ["metric", "baseline", "experimental"]),
        "",
        "## decision",
        f"- conclusion: {decision}",
        *decision_notes,
        "",
        "## interpretation",
        "- This experiment only changes neutral weights.",
        "- ret/prob guardrail is evaluated through top20 overlap retention rather than raw score magnitude.",
        "- confidence_score is unchanged as a formula, but top20 membership can change its top20 mean.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    latest, latest_date, score_formula_version = load_latest()
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_report(latest, latest_date, score_formula_version), encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
