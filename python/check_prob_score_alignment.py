import logging
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path("data/ranking_final.csv")
OUTPUT_MD = Path("outputs/prob_score_diagnostics.md")
TOP_N = 20


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    cols = [left, right]
    if any(col not in df.columns for col in cols):
        return float("nan")
    sample = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sample) < 2:
        return float("nan")
    return float(sample[left].corr(sample[right]))


def _fmt(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def load_ranking() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"ranking CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def ensure_columns(df: pd.DataFrame) -> None:
    required = ["date", "code", "prob_top20_60d", "prob_score_raw", "prob_score", "prob_rank_pct", "final_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")


def build_date_distribution(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("date", dropna=False)
    return grouped.agg(
        rows=("code", "size"),
        prob_score_mean=("prob_score", "mean"),
        prob_score_std=("prob_score", "std"),
        prob_score_min=("prob_score", "min"),
        prob_score_p25=("prob_score", lambda s: s.quantile(0.25)),
        prob_score_p50=("prob_score", "median"),
        prob_score_p75=("prob_score", lambda s: s.quantile(0.75)),
        prob_score_max=("prob_score", "max"),
    ).reset_index()


def build_top20_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"] == latest_date].copy()
    latest = latest.sort_values(["final_score", "prob_score"], ascending=[False, False]).head(TOP_N)
    keep_cols = [
        "date",
        "code",
        "name",
        "final_score",
        "prob_top20_60d",
        "prob_score_raw",
        "prob_rank_pct",
        "prob_score",
        "prob_score_missing",
        "rank_final",
    ]
    keep_cols = [col for col in keep_cols if col in latest.columns]
    return latest[keep_cols]


def build_markdown(df: pd.DataFrame) -> str:
    numeric_cols = ["prob_top20_60d", "prob_score_raw", "prob_score", "prob_rank_pct", "final_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "prob_score_missing" in df.columns:
        df["prob_score_missing"] = df["prob_score_missing"].astype("boolean")

    date_dist = build_date_distribution(df)
    top20 = build_top20_snapshot(df)

    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"] == latest_date].copy()
    latest_missing_ratio = float(latest["prob_score_missing"].fillna(False).mean()) if "prob_score_missing" in latest.columns and len(latest) else float("nan")
    all_missing_ratio = float(df["prob_score_missing"].fillna(False).mean()) if "prob_score_missing" in df.columns and len(df) else float("nan")

    lines: list[str] = []
    lines.append("# Prob Score Diagnostics")
    lines.append("")
    lines.append("## summary")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- date_range: {df['date'].min()} ~ {df['date'].max()}")
    lines.append(f"- latest_date: {latest_date}")
    lines.append(f"- prob_score_missing_ratio_all: {_fmt(all_missing_ratio)}")
    lines.append(f"- prob_score_missing_ratio_latest: {_fmt(latest_missing_ratio)}")
    lines.append("")
    lines.append("## raw probability vs operational probability score")
    lines.append("- `prob_top20_60d`: raw model probability for top-bucket entry.")
    lines.append("- `prob_score_raw`: absolute conversion `clip(prob_top20_60d * 100, 0, 100)`.")
    lines.append("- `prob_score`: same-date percentile operating score used in `final_score`.")
    lines.append(f"- corr(prob_top20_60d, prob_score_raw): {_fmt(_safe_corr(df, 'prob_top20_60d', 'prob_score_raw'))}")
    lines.append(f"- corr(prob_top20_60d, prob_score): {_fmt(_safe_corr(df, 'prob_top20_60d', 'prob_score'))}")
    lines.append(f"- corr(prob_score_raw, prob_score): {_fmt(_safe_corr(df, 'prob_score_raw', 'prob_score'))}")
    lines.append("")
    lines.append("## date-wise distribution summary")
    if len(date_dist):
        lines.append(f"- dates_covered: {len(date_dist)}")
        lines.append(f"- mean_of_date_means: {_fmt(date_dist['prob_score_mean'].mean())}")
        lines.append(f"- mean_of_date_stds: {_fmt(date_dist['prob_score_std'].fillna(0.0).mean())}")
        lines.append(f"- min_prob_score_across_dates: {_fmt(date_dist['prob_score_min'].min())}")
        lines.append(f"- max_prob_score_across_dates: {_fmt(date_dist['prob_score_max'].max())}")
    else:
        lines.append("- no date-wise distribution available")
    lines.append("")
    lines.append("## missingness summary")
    lines.append(f"- missing prob_top20_60d ratio: {_fmt(df['prob_top20_60d'].isna().mean())}")
    lines.append(f"- missing prob_score_raw ratio: {_fmt(df['prob_score_raw'].isna().mean())}")
    lines.append(f"- fallback-applied ratio via prob_score_missing: {_fmt(all_missing_ratio)}")
    lines.append("")
    lines.append("## correlation with final_score")
    lines.append(f"- corr(final_score, prob_score): {_fmt(_safe_corr(df, 'final_score', 'prob_score'))}")
    lines.append(f"- corr(final_score, prob_score_raw): {_fmt(_safe_corr(df, 'final_score', 'prob_score_raw'))}")
    lines.append(f"- corr(final_score, prob_top20_60d): {_fmt(_safe_corr(df, 'final_score', 'prob_top20_60d'))}")
    lines.append("")
    lines.append("## top20 snapshot")
    if len(top20):
        lines.append(f"- latest_date_top20: {latest_date}")
        lines.append("")
        lines.append("| code | final_score | prob_top20_60d | prob_score_raw | prob_rank_pct | prob_score | prob_score_missing |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for _, row in top20.iterrows():
            lines.append(
                f"| {row.get('code', 'NA')} | {_fmt(row.get('final_score'))} | {_fmt(row.get('prob_top20_60d'))} | "
                f"{_fmt(row.get('prob_score_raw'))} | {_fmt(row.get('prob_rank_pct'))} | {_fmt(row.get('prob_score'))} | "
                f"{row.get('prob_score_missing', 'NA')} |"
            )
    else:
        lines.append("- top20 snapshot unavailable")
    lines.append("")
    lines.append("## interpretation")
    corr_relative = _safe_corr(df, "final_score", "prob_score")
    corr_raw = _safe_corr(df, "final_score", "prob_score_raw")
    if np.isfinite(corr_relative) and np.isfinite(corr_raw):
        dominant = "prob_score" if abs(corr_relative) >= abs(corr_raw) else "prob_score_raw"
        lines.append(f"- stronger direct alignment with `final_score` currently appears on `{dominant}`.")
    lines.append("- `final_score` should be read as using the relative operating `prob_score`, not the absolute raw probability conversion.")
    lines.append("- Large gaps between `prob_score_raw` and `prob_score` indicate that the same raw probability can map to different operating strength depending on the date slice.")
    lines.append("")
    lines.append("## remaining limitations")
    lines.append("- Relative probability scoring depends on the daily universe composition, so cross-date direct comparison of `prob_score` should be avoided.")
    lines.append("- If a date has widespread missing probability inputs, fallback values can compress cross-sectional separation for that day.")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    df = load_ranking()
    ensure_columns(df)
    report = build_markdown(df)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    logging.info("Saved prob score diagnostics: %s", OUTPUT_MD.resolve())


if __name__ == "__main__":
    main()
