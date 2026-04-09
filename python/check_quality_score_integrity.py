import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quality_builder import QUALITY_FACTOR_SPECS


DEFAULT_INPUT_CSV = Path("data/quality.csv")
DEFAULT_OUT_MD = Path("outputs/quality_score_diagnostics.md")
EXPECTED_DIRECTIONS = {factor: int(spec["direction"]) for factor, spec in QUALITY_FACTOR_SPECS.items()}
FACTOR_WEIGHTS = {factor: float(spec["weight"]) for factor, spec in QUALITY_FACTOR_SPECS.items()}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check production quality score temporal integrity and factor behavior")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_quality(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"quality csv not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["quality_score", "quality_raw_score", "quality_factor_count", "quality_missing_ratio", "quality_score_confidence", *EXPECTED_DIRECTIONS.keys()]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_empty_"
    render = df.copy()
    for col in render.columns:
        if pd.api.types.is_numeric_dtype(render[col]):
            render[col] = render[col].map(lambda x: format_float(x, digits))
        else:
            render[col] = render[col].fillna("NA").astype(str)
    headers = [str(col) for col in render.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in render.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_date_distribution(df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for date, group in df.groupby("date", dropna=False):
        score = pd.to_numeric(group.get("quality_score"), errors="coerce")
        metrics.append(
            {
                "date": date,
                "rows": int(len(group)),
                "mean": float(score.mean()) if score.notna().any() else np.nan,
                "std": float(score.std(ddof=0)) if score.notna().any() else np.nan,
                "min": float(score.min()) if score.notna().any() else np.nan,
                "p25": float(score.quantile(0.25)) if score.notna().any() else np.nan,
                "p50": float(score.quantile(0.50)) if score.notna().any() else np.nan,
                "p75": float(score.quantile(0.75)) if score.notna().any() else np.nan,
                "max": float(score.max()) if score.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(metrics).sort_values("date").reset_index(drop=True)


def build_factor_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in EXPECTED_DIRECTIONS:
        available_in_csv = factor in df.columns
        nonnull_ratio = float(df[factor].notna().mean()) if available_in_csv else np.nan
        rows.append(
            {
                "factor": factor,
                "configured_weight": FACTOR_WEIGHTS[factor],
                "available_in_csv": available_in_csv,
                "source_usable": bool(available_in_csv and nonnull_ratio > 0.0),
                "nonnull_ratio": nonnull_ratio,
            }
        )
    return pd.DataFrame(rows)


def build_missingness_summary(df: pd.DataFrame) -> dict[str, float | int]:
    factor_count = pd.to_numeric(df.get("quality_factor_count"), errors="coerce")
    missing_ratio = pd.to_numeric(df.get("quality_missing_ratio"), errors="coerce")
    confidence = pd.to_numeric(df.get("quality_score_confidence"), errors="coerce")
    return {
        "rows": int(len(df)),
        "factor_count_mean": float(factor_count.mean()) if factor_count.notna().any() else np.nan,
        "factor_count_min": float(factor_count.min()) if factor_count.notna().any() else np.nan,
        "factor_count_max": float(factor_count.max()) if factor_count.notna().any() else np.nan,
        "missing_ratio_mean": float(missing_ratio.mean()) if missing_ratio.notna().any() else np.nan,
        "missing_ratio_p75": float(missing_ratio.quantile(0.75)) if missing_ratio.notna().any() else np.nan,
        "confidence_mean": float(confidence.mean()) if confidence.notna().any() else np.nan,
        "confidence_p25": float(confidence.quantile(0.25)) if confidence.notna().any() else np.nan,
    }


def build_directional_sanity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score = pd.to_numeric(df.get("quality_score"), errors="coerce")
    for factor, expected_sign in EXPECTED_DIRECTIONS.items():
        if factor not in df.columns:
            rows.append(
                {
                    "factor": factor,
                    "expected_sign": expected_sign,
                    "overall_corr": np.nan,
                    "matching_date_ratio": np.nan,
                    "status": "missing_column",
                }
            )
            continue
        factor_series = pd.to_numeric(df[factor], errors="coerce")
        if factor_series.notna().sum() == 0:
            rows.append(
                {
                    "factor": factor,
                    "expected_sign": expected_sign,
                    "overall_corr": np.nan,
                    "matching_date_ratio": np.nan,
                    "status": "no_source_data",
                }
            )
            continue
        work = pd.concat([factor_series, score], axis=1).dropna()
        overall_corr = float(work.iloc[:, 0].corr(work.iloc[:, 1])) if len(work) >= 2 else np.nan
        date_signs = []
        for _date, group in df.groupby("date", dropna=False):
            sample = group[[factor, "quality_score"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sample) >= 2:
                corr = float(sample[factor].corr(sample["quality_score"]))
                if not pd.isna(corr):
                    date_signs.append(np.sign(corr) == expected_sign or np.isclose(corr, 0.0))
        match_ratio = float(np.mean(date_signs)) if date_signs else np.nan
        status = "ok"
        if pd.notna(overall_corr) and np.sign(overall_corr) not in {0.0, expected_sign}:
            status = "sign_mismatch"
        rows.append(
            {
                "factor": factor,
                "expected_sign": expected_sign,
                "overall_corr": overall_corr,
                "matching_date_ratio": match_ratio,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_interpretation(availability: pd.DataFrame, missingness: dict[str, float | int], sanity: pd.DataFrame) -> list[str]:
    comments = []
    missing_factors = availability.loc[~availability["source_usable"].fillna(False), "factor"].tolist()
    if missing_factors:
        comments.append(f"- Missing or unusable configured factors in current source: {', '.join(missing_factors)}.")
    else:
        comments.append("- All configured quality factors are present in the current quality source.")

    comments.append(
        f"- Mean factor coverage is {format_float(missingness.get('factor_count_mean'))} and mean confidence is {format_float(missingness.get('confidence_mean'))}."
    )
    mismatch = sanity.loc[sanity["status"] == "sign_mismatch", "factor"].tolist()
    if mismatch:
        comments.append(f"- Directional sign mismatch detected for: {', '.join(mismatch)}.")
    else:
        comments.append("- Factor direction checks are broadly aligned with expected signs.")
    return comments


def render_markdown(
    *,
    df: pd.DataFrame,
    date_dist: pd.DataFrame,
    availability: pd.DataFrame,
    missingness: dict[str, float | int],
    sanity: pd.DataFrame,
) -> str:
    lines = [
        "# Quality Score Diagnostics",
        "",
        "## Summary",
        f"- rows: {len(df)}",
        f"- dates: {df['date'].nunique() if 'date' in df.columns else 0}",
        f"- quality_score_mean: {format_float(pd.to_numeric(df.get('quality_score'), errors='coerce').mean() if 'quality_score' in df.columns else np.nan)}",
        f"- quality_score_confidence_mean: {format_float(missingness.get('confidence_mean'))}",
        "",
        "## quality factor configuration",
    ]
    for factor, weight in FACTOR_WEIGHTS.items():
        direction = "high is good" if EXPECTED_DIRECTIONS[factor] > 0 else "low is good"
        lines.append(f"- `{factor}`: weight={weight:.2f}, direction={direction}")
    lines.extend(
        [
            "",
            "## date-wise score distribution",
            dataframe_to_markdown(date_dist),
            "",
            "## factor availability summary",
            dataframe_to_markdown(availability),
            "",
            "## missingness summary",
            f"- factor_count_mean: {format_float(missingness.get('factor_count_mean'))}",
            f"- factor_count_min: {format_float(missingness.get('factor_count_min'))}",
            f"- factor_count_max: {format_float(missingness.get('factor_count_max'))}",
            f"- quality_missing_ratio_mean: {format_float(missingness.get('missing_ratio_mean'))}",
            f"- quality_missing_ratio_p75: {format_float(missingness.get('missing_ratio_p75'))}",
            f"- quality_score_confidence_mean: {format_float(missingness.get('confidence_mean'))}",
            f"- quality_score_confidence_p25: {format_float(missingness.get('confidence_p25'))}",
            "",
            "## directional sanity check",
            dataframe_to_markdown(sanity),
            "",
            "## interpretation",
        ]
    )
    lines.extend(build_interpretation(availability, missingness, sanity))
    lines.extend(
        [
            "",
            "## remaining limitations",
            "- Quality source coverage still depends on what is available in `fundamentals.csv`.",
            "- Annual fundamentals are forward-filled into daily features later, so freshness is limited by source reporting cadence.",
            "- Confidence reflects factor completeness, not accounting-data correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    logging.info("Saved quality diagnostics: %s", path.resolve())


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_quality(args.input_csv)
    if df.empty:
        raise ValueError("quality input is empty")

    date_dist = build_date_distribution(df)
    availability = build_factor_availability(df)
    missingness = build_missingness_summary(df)
    sanity = build_directional_sanity(df)
    markdown = render_markdown(
        df=df,
        date_dist=date_dist,
        availability=availability,
        missingness=missingness,
        sanity=sanity,
    )
    save_markdown(args.out_md, markdown)


if __name__ == "__main__":
    main()
