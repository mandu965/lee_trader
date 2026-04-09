import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
DEFAULT_OUT_CSV = Path("outputs/tech_score_top20_summary.csv")
DEFAULT_OUT_MD = Path("outputs/tech_score_diagnostics.md")
TOP_N = 20
TECH_INTERMEDIATE_COLUMNS = [
    "tech_trend_score",
    "tech_momentum_score",
    "tech_stability_score",
    "tech_volume_score",
    "tech_liquidity_guard",
    "tech_source",
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check tech_score quality and contribution diagnostics")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_ranking(input_csv: Path, date_filter: str | None) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"ranking CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if date_filter:
        df = df.loc[df.get("date", pd.Series(index=df.index, dtype=object)) == date_filter].copy()
    return df


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    render = df.copy()
    for col in render.columns:
        if pd.api.types.is_numeric_dtype(render[col]):
            render[col] = render[col].map(lambda x: format_float(x))
        else:
            render[col] = render[col].fillna("NA").astype(str)
    headers = [str(col) for col in render.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in render.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def compute_distribution(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "rows": int(len(numeric)),
        "nonnull": int(numeric.notna().sum()),
        "null_ratio": float(numeric.isna().mean()) if len(numeric) else np.nan,
        "unique": int(numeric.nunique(dropna=True)),
        "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
        "std": float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan,
        "min": float(numeric.min()) if numeric.notna().any() else np.nan,
        "p25": float(numeric.quantile(0.25)) if numeric.notna().any() else np.nan,
        "p50": float(numeric.quantile(0.50)) if numeric.notna().any() else np.nan,
        "p75": float(numeric.quantile(0.75)) if numeric.notna().any() else np.nan,
        "max": float(numeric.max()) if numeric.notna().any() else np.nan,
    }


def detect_constant_values(series: pd.Series) -> tuple[bool, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    unique_count = int(numeric.nunique(dropna=True))
    return unique_count <= 1, unique_count


def latest_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    latest_date = str(df["date"].dropna().max()) if "date" in df.columns and df["date"].notna().any() else "all_dates"
    latest = df.loc[df["date"] == latest_date].copy() if latest_date != "all_dates" else df.copy()
    return latest.sort_values(["final_score"], ascending=[False]), latest_date


def overlap_ratio(df: pd.DataFrame, score_col: str, top_n: int = TOP_N) -> float:
    final_set = set(df.sort_values(["final_score"], ascending=[False]).head(top_n)["code"].astype(str))
    comp_set = set(df.sort_values([score_col], ascending=[False]).head(top_n)["code"].astype(str))
    return len(final_set & comp_set) / float(top_n) if top_n else np.nan


def compute_correlation(df: pd.DataFrame, left: str, right: str) -> float:
    cols = [left, right]
    if any(col not in df.columns for col in cols):
        return np.nan
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(corr_df) < 2:
        return np.nan
    return float(corr_df[left].corr(corr_df[right]))


def build_topn_summary(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(
        work,
        [
            "final_score",
            "tech_score",
            "w_tech",
            "contrib_tech",
            "tech_trend_score",
            "tech_momentum_score",
            "tech_stability_score",
            "tech_volume_score",
            "tech_liquidity_guard",
        ],
    )
    work = work.sort_values(["final_score", "tech_score"], ascending=[False, False]).head(top_n).copy()
    base_cols = [col for col in ["date", "code", "name", "sector", "market", "regime"] if col in work.columns]
    metric_cols = [col for col in ["final_score", "tech_score", "w_tech", "contrib_tech"] if col in work.columns]
    extra_cols = [col for col in TECH_INTERMEDIATE_COLUMNS if col in work.columns]
    return work[base_cols + metric_cols + extra_cols]


def compute_by_date_std(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        return pd.Series(dtype=float)
    temp = df.copy()
    temp["tech_score"] = pd.to_numeric(temp["tech_score"], errors="coerce")
    return temp.groupby("date")["tech_score"].std(ddof=0).dropna()


def compute_guard_diagnostics(df: pd.DataFrame) -> dict[str, float | int]:
    if "tech_liquidity_guard" not in df.columns:
        return {}
    guard = pd.to_numeric(df["tech_liquidity_guard"], errors="coerce")
    tech = pd.to_numeric(df.get("tech_score"), errors="coerce")
    contrib = pd.to_numeric(df.get("contrib_tech"), errors="coerce")
    return {
        "guard_mean": float(guard.mean()) if guard.notna().any() else np.nan,
        "guard_std": float(guard.std(ddof=0)) if guard.notna().any() else np.nan,
        "guard_lt_1_ratio": float((guard < 1.0).mean()) if len(guard) else np.nan,
        "guard_lt_0_9_ratio": float((guard < 0.9).mean()) if len(guard) else np.nan,
        "tech_mean_guard_lt_1": float(tech[guard < 1.0].mean()) if (guard < 1.0).any() else np.nan,
        "tech_mean_guard_eq_1": float(tech[guard >= 1.0].mean()) if (guard >= 1.0).any() else np.nan,
        "contrib_mean_guard_lt_1": float(contrib[guard < 1.0].mean()) if (guard < 1.0).any() else np.nan,
        "contrib_mean_guard_eq_1": float(contrib[guard >= 1.0].mean()) if (guard >= 1.0).any() else np.nan,
        "corr_guard_vs_tech": compute_correlation(pd.DataFrame({"guard": guard, "tech": tech}), "guard", "tech"),
        "corr_guard_vs_contrib": compute_correlation(pd.DataFrame({"guard": guard, "contrib": contrib}), "guard", "contrib"),
    }


def dominant_driver_counts(df: pd.DataFrame, top_n: int) -> dict[str, int]:
    latest, _ = latest_slice(df)
    top = latest.head(top_n)
    values = pd.concat(
        [top[col].dropna().astype(str) for col in ["score_driver_1", "score_driver_2", "score_driver_3"] if col in top.columns],
        ignore_index=True,
    )
    return values.value_counts().to_dict() if len(values) else {}


def build_markdown(
    df: pd.DataFrame,
    latest: pd.DataFrame,
    latest_date: str,
    distribution: dict[str, float | int],
    by_date_std: pd.Series,
    corr: float,
    overlap: float,
    guard_diag: dict[str, float | int],
    top_summary: pd.DataFrame,
    same_value: bool,
    unique_count: int,
) -> str:
    contrib_all = pd.to_numeric(df.get("contrib_tech"), errors="coerce")
    contrib_top = pd.to_numeric(latest.head(TOP_N).get("contrib_tech"), errors="coerce")
    lines = []
    lines.append("# Tech Score Diagnostics")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- latest_date: {latest_date}")
    lines.append(f"- tech_score_unique_count: {distribution['unique']}")
    lines.append(f"- tech_score_std: {format_float(distribution['std'])}")
    lines.append(f"- tech_score_by_date_std_mean: {format_float(by_date_std.mean() if len(by_date_std) else np.nan)}")
    lines.append(f"- tech_score_by_date_std_latest: {format_float(by_date_std.loc[latest_date] if latest_date in by_date_std.index else np.nan)}")
    lines.append(f"- contrib_tech_mean: {format_float(contrib_all.mean())}")
    lines.append(f"- contrib_tech_top20_mean: {format_float(contrib_top.mean())}")
    lines.append(f"- overlap(final_score top20, tech_score top20): {format_float(overlap)}")
    lines.append(f"- corr(final_score, tech_score): {format_float(corr)}")
    lines.append("")
    lines.append("## Distribution")
    lines.append(f"- null_ratio: {format_float(distribution['null_ratio'])}")
    lines.append(f"- mean: {format_float(distribution['mean'])}")
    lines.append(f"- std: {format_float(distribution['std'])}")
    lines.append(f"- range: {format_float(distribution['min'])} / {format_float(distribution['p25'])} / {format_float(distribution['p50'])} / {format_float(distribution['p75'])} / {format_float(distribution['max'])}")
    lines.append(f"- all_values_identical: {'YES' if same_value else 'NO'}")
    lines.append(f"- unique_value_count: {unique_count}")
    lines.append("")
    lines.append("## Liquidity Guard")
    if not guard_diag:
        lines.append("- tech_liquidity_guard column unavailable")
    else:
        for key, value in guard_diag.items():
            lines.append(f"- {key}: {format_float(value)}")
    lines.append("")
    lines.append("## Top20 Driver Mix")
    driver_counts = dominant_driver_counts(df, TOP_N)
    lines.append(f"- driver_counts: {driver_counts}")
    lines.append("")
    lines.append("## Top20 Snapshot")
    if top_summary.empty:
        lines.append("- none")
    else:
        lines.append(dataframe_to_markdown(top_summary))
    lines.append("")
    lines.append("## Interpretation")
    if overlap < 0.20:
        lines.append("- tech_score 분산은 있어도 실제 top rank 반영은 약한 편입니다.")
    elif overlap < 0.35:
        lines.append("- tech_score는 보조 축으로 작동하지만 top rank 지배력은 아직 제한적입니다.")
    else:
        lines.append("- tech_score가 top rank에 의미 있게 반영되고 있습니다.")
    if guard_diag:
        guard_ratio = float(guard_diag.get("guard_lt_1_ratio", np.nan))
        if np.isfinite(guard_ratio) and guard_ratio >= 0.25:
            lines.append("- liquidity guard 적용 비율이 높아 억제 강도를 재점검할 필요가 있습니다.")
        else:
            lines.append("- liquidity guard는 존재하지만 전면적 억제 수준은 아닙니다.")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")

    required_cols = {"tech_score", "final_score"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")

    df = ensure_numeric(df, ["tech_score", "final_score", "contrib_tech", "tech_liquidity_guard", "w_tech"])
    latest, latest_date = latest_slice(df)
    distribution = compute_distribution(df["tech_score"])
    same_value, unique_count = detect_constant_values(df["tech_score"])
    corr = compute_correlation(df, "final_score", "tech_score")
    top_summary = build_topn_summary(df, args.top_n)
    by_date_std = compute_by_date_std(df)
    overlap = overlap_ratio(latest, "tech_score", args.top_n)
    guard_diag = compute_guard_diagnostics(df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    top_summary.to_csv(args.out_csv, index=False, encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_markdown(df, latest, latest_date, distribution, by_date_std, corr, overlap, guard_diag, top_summary, same_value, unique_count),
        encoding="utf-8",
    )
    logging.info("Saved tech_score diagnostics markdown: %s", args.out_md.resolve())
    logging.info("Saved tech_score diagnostics CSV: %s", args.out_csv.resolve())


if __name__ == "__main__":
    main()
