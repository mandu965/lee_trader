import logging
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path("data/ranking_final.csv")
OUTPUT_MD = Path("outputs/confidence_diagnostics.md")
TOP_N = 10


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _fmt(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    sample = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sample) < 2:
        return float("nan")
    return float(sample[left].corr(sample[right]))


def load_ranking() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"ranking CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def ensure_columns(df: pd.DataFrame) -> None:
    required = [
        "final_score",
        "confidence_score",
        "confidence_grade",
        "confidence_penalty",
        "component_coverage_ratio",
        "fallback_count",
        "confidence_reason",
        "ret_score_missing",
        "prob_score_missing",
        "qual_score_missing",
        "tech_score_missing",
        "safety_score_missing",
        "liquidity_score_missing",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ret_score_missing",
        "prob_score_missing",
        "qual_score_missing",
        "tech_score_missing",
        "safety_score_missing",
        "liquidity_score_missing",
    ]
    rows = []
    for col in cols:
        series = df[col].fillna(False).astype(bool)
        rows.append({"column": col, "missing_ratio": float(series.mean()), "missing_count": int(series.sum())})
    return pd.DataFrame(rows)


def build_mismatch_cases(df: pd.DataFrame, *, high_final_low_conf: bool) -> pd.DataFrame:
    work = df.copy()
    work["final_score"] = pd.to_numeric(work["final_score"], errors="coerce")
    work["confidence_score"] = pd.to_numeric(work["confidence_score"], errors="coerce")
    if high_final_low_conf:
        mask = work["final_score"].ge(60.0) & work["confidence_score"].lt(55.0)
        sort_cols = ["final_score", "confidence_score"]
        ascending = [False, True]
    else:
        mask = work["final_score"].lt(40.0) & work["confidence_score"].ge(70.0)
        sort_cols = ["confidence_score", "final_score"]
        ascending = [False, True]
    keep = [
        "date",
        "code",
        "name",
        "final_score",
        "confidence_score",
        "confidence_grade",
        "component_coverage_ratio",
        "fallback_count",
        "confidence_reason",
    ]
    keep = [col for col in keep if col in work.columns]
    return work.loc[mask, keep].sort_values(sort_cols, ascending=ascending).head(TOP_N)


def build_markdown(df: pd.DataFrame) -> str:
    numeric_cols = ["final_score", "confidence_score", "confidence_penalty", "component_coverage_ratio", "fallback_count"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_summary = build_missing_summary(df)
    fallback = pd.to_numeric(df["fallback_count"], errors="coerce")
    high_final_low_conf = build_mismatch_cases(df, high_final_low_conf=True)
    low_final_high_conf = build_mismatch_cases(df, high_final_low_conf=False)
    grade_dist = df["confidence_grade"].fillna("NA").value_counts(dropna=False).sort_index()

    lines: list[str] = []
    lines.append("# Confidence Diagnostics")
    lines.append("")
    lines.append("## summary")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- date_range: {df['date'].min()} ~ {df['date'].max()}")
    lines.append(f"- confidence_score_mean: {_fmt(df['confidence_score'].mean())}")
    lines.append(f"- confidence_score_p25: {_fmt(df['confidence_score'].quantile(0.25))}")
    lines.append(f"- confidence_score_p50: {_fmt(df['confidence_score'].median())}")
    lines.append(f"- confidence_score_p75: {_fmt(df['confidence_score'].quantile(0.75))}")
    lines.append("")
    lines.append("## confidence definition")
    lines.append("- `confidence_score` is an evidence-quality meta metric, not a return score.")
    lines.append("- It reflects component coverage, fallback usage, quality factor support, and structural support from technical/risk-control inputs.")
    lines.append("")
    lines.append("## score vs confidence separation")
    lines.append("- `final_score` remains the recommendation score.")
    lines.append("- `confidence_score` is tracked independently and is not multiplied into or subtracted from `final_score`.")
    lines.append(f"- corr(final_score, confidence_score): {_fmt(_safe_corr(df, 'final_score', 'confidence_score'))}")
    lines.append("")
    lines.append("## missing flag summary")
    for _, row in missing_summary.iterrows():
        lines.append(f"- {row['column']}: ratio={_fmt(row['missing_ratio'])}, count={int(row['missing_count'])}")
    lines.append("")
    lines.append("## fallback summary")
    lines.append(f"- fallback_count_mean: {_fmt(fallback.mean())}")
    lines.append(f"- fallback_count_max: {_fmt(fallback.max(), 0)}")
    lines.append(f"- fallback_count_zero_ratio: {_fmt((fallback.fillna(0) == 0).mean())}")
    lines.append(f"- confidence_penalty_mean: {_fmt(df['confidence_penalty'].mean())}")
    lines.append("")
    lines.append("## confidence distribution")
    lines.append(f"- min: {_fmt(df['confidence_score'].min())}")
    lines.append(f"- max: {_fmt(df['confidence_score'].max())}")
    lines.append(f"- std: {_fmt(df['confidence_score'].std(ddof=0))}")
    lines.append(f"- component_coverage_ratio_mean: {_fmt(df['component_coverage_ratio'].mean())}")
    lines.append("")
    lines.append("## confidence grade distribution")
    for grade, count in grade_dist.items():
        lines.append(f"- {grade}: {int(count)}")
    lines.append("")
    lines.append("## notable mismatch cases")
    lines.append("- High final_score but low confidence")
    if len(high_final_low_conf):
        for _, row in high_final_low_conf.iterrows():
            lines.append(
                f"  - {row.get('date', 'NA')} {row.get('code', 'NA')} final={_fmt(row.get('final_score'))} "
                f"confidence={_fmt(row.get('confidence_score'))} grade={row.get('confidence_grade', 'NA')} "
                f"coverage={_fmt(row.get('component_coverage_ratio'))} fallback={_fmt(row.get('fallback_count'), 0)} "
                f"reason={row.get('confidence_reason', '')}"
            )
    else:
        lines.append("  - none")
    lines.append("- Low final_score but high confidence")
    if len(low_final_high_conf):
        for _, row in low_final_high_conf.iterrows():
            lines.append(
                f"  - {row.get('date', 'NA')} {row.get('code', 'NA')} final={_fmt(row.get('final_score'))} "
                f"confidence={_fmt(row.get('confidence_score'))} grade={row.get('confidence_grade', 'NA')} "
                f"coverage={_fmt(row.get('component_coverage_ratio'))} fallback={_fmt(row.get('fallback_count'), 0)} "
                f"reason={row.get('confidence_reason', '')}"
            )
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("## interpretation")
    lines.append("- Low confidence should mainly reflect missing inputs or fallback-heavy construction, not simply a weak alpha score.")
    lines.append("- High confidence with low final_score means the system has enough evidence, but that evidence currently argues against the name.")
    lines.append("")
    lines.append("## remaining limitations")
    lines.append("- Some fallback flags are still proxy-based and map to neutral fill usage rather than a deep source lineage graph.")
    lines.append("- Confidence is only as good as the availability tracking of upstream builders.")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    df = load_ranking()
    ensure_columns(df)
    report = build_markdown(df)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    logging.info("Saved confidence diagnostics: %s", OUTPUT_MD.resolve())


if __name__ == "__main__":
    main()
