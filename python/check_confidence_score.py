import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
TOP_N = 20
CONF_COMPONENT_WEIGHTS = {
    "data_quality_conf": 0.30,
    "signal_consistency_conf": 0.30,
    "risk_stability_conf": 0.25,
    "backtest_reliability_conf": 0.15,
}
REQUIRED_COLUMNS = [
    "final_score",
    "confidence_score",
    "confidence_grade",
    "data_quality_conf",
    "signal_consistency_conf",
    "risk_stability_conf",
    "backtest_reliability_conf",
]
NUMERIC_REQUIRED_COLUMNS = [
    "final_score",
    "confidence_score",
    "data_quality_conf",
    "signal_consistency_conf",
    "risk_stability_conf",
    "backtest_reliability_conf",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check confidence_score distribution and dominance diagnostics")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"ranking CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"top rows to sample (default: {TOP_N})")
    parser.add_argument("--out-csv", type=Path, help="optional path to save low-confidence exception rows")
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


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def compute_relationship(df: pd.DataFrame) -> dict[str, float]:
    work = df[["final_score", "confidence_score"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 2:
        return {"corr": np.nan, "mean_gap": np.nan}
    return {
        "corr": float(work["final_score"].corr(work["confidence_score"])),
        "mean_gap": float((work["final_score"] - work["confidence_score"]).mean()),
    }


def build_high_confidence_samples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(work, ["final_score", "confidence_score"])
    cols = [col for col in ["date", "code", "name", "final_score", "confidence_score", "confidence_grade", "confidence_explain_text"] if col in work.columns]
    return work.sort_values(["confidence_score", "final_score"], ascending=[False, False]).head(top_n)[cols]


def build_high_final_low_confidence_samples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(work, ["final_score", "confidence_score"])
    high_final_threshold = work["final_score"].quantile(0.75)
    low_conf_threshold = work["confidence_score"].quantile(0.25)
    mask = (work["final_score"] >= high_final_threshold) & (work["confidence_score"] <= low_conf_threshold)
    cols = [col for col in [
        "date",
        "code",
        "name",
        "final_score",
        "confidence_score",
        "confidence_grade",
        "data_quality_conf",
        "signal_consistency_conf",
        "risk_stability_conf",
        "backtest_reliability_conf",
        "confidence_penalty",
        "confidence_explain_text",
    ] if col in work.columns]
    return work.loc[mask].sort_values(["final_score", "confidence_score"], ascending=[False, True]).head(top_n)[cols]


def build_dominance_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    work = df.copy()
    component_cols = list(CONF_COMPONENT_WEIGHTS.keys())
    work = ensure_numeric(work, component_cols + ["confidence_score"])

    weighted_cols = []
    for col, weight in CONF_COMPONENT_WEIGHTS.items():
        weighted_col = f"{col}_weighted"
        work[weighted_col] = work[col].fillna(0.0) * weight
        weighted_cols.append(weighted_col)

    weighted_frame = work[weighted_cols].copy()
    dominant_weighted_col = weighted_frame.idxmax(axis=1)
    dominant_weighted_val = weighted_frame.max(axis=1)
    weighted_sum = weighted_frame.sum(axis=1).replace(0.0, np.nan)

    work["dominant_conf_axis"] = dominant_weighted_col.str.replace("_weighted", "", regex=False)
    work["dominant_conf_value"] = dominant_weighted_val
    work["dominant_conf_share"] = dominant_weighted_val / weighted_sum
    work["is_over_dominant"] = work["dominant_conf_share"] > 0.50

    dominant_counts = work["dominant_conf_axis"].value_counts(dropna=False)
    summary = {
        "over_dominant_ratio": float(work["is_over_dominant"].mean()) if len(work) else np.nan,
        "top_dominant_axis": str(dominant_counts.index[0]) if len(dominant_counts) else "NA",
        "top_dominant_count": int(dominant_counts.iloc[0]) if len(dominant_counts) else 0,
    }
    return work, summary


def maybe_save_csv(df: pd.DataFrame, out_csv: Path | None) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved confidence check CSV: %s (rows=%d)", out_csv.resolve(), len(df))


def print_report(
    df: pd.DataFrame,
    distribution: dict[str, float | int],
    relationship: dict[str, float],
    high_conf_df: pd.DataFrame,
    exception_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
    dominance_summary: dict[str, float | str],
) -> None:
    print("=== confidence_score check ===")
    print(f"rows={len(df)}")
    if "date" in df.columns and df["date"].notna().any():
        print(f"date_range={df['date'].min()} ~ {df['date'].max()}")
    print("")

    print("[distribution]")
    print(f"confidence_null_ratio={distribution['null_ratio']:.4f}")
    print(f"confidence_unique={distribution['unique']}")
    print(f"confidence_mean={distribution['mean']:.4f}")
    print(f"confidence_std={distribution['std']:.4f}")
    print(
        "confidence_range="
        f"{distribution['min']:.4f} / {distribution['p25']:.4f} / "
        f"{distribution['p50']:.4f} / {distribution['p75']:.4f} / {distribution['max']:.4f}"
    )
    print("")

    print("[relationship to final_score]")
    corr = relationship.get("corr", np.nan)
    mean_gap = relationship.get("mean_gap", np.nan)
    print(f"corr_final_vs_confidence={corr:.4f}" if pd.notna(corr) else "corr_final_vs_confidence=NA")
    print(f"mean_final_minus_confidence={mean_gap:.4f}" if pd.notna(mean_gap) else "mean_final_minus_confidence=NA")
    print("")

    print("[high confidence top samples]")
    if high_conf_df.empty:
        print("No rows available.")
    else:
        print(high_conf_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[high final / low confidence samples]")
    if exception_df.empty:
        print("No rows matched.")
    else:
        print(exception_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[dominance check]")
    print(f"over_dominant_ratio={dominance_summary['over_dominant_ratio']:.4f}")
    print(f"top_dominant_axis={dominance_summary['top_dominant_axis']}")
    print(f"top_dominant_count={dominance_summary['top_dominant_count']}")
    over_dom = dominance_df.loc[dominance_df["is_over_dominant"]].copy()
    if over_dom.empty:
        print("No over-dominant rows detected.")
    else:
        cols = [col for col in ["date", "code", "name", "confidence_score", "dominant_conf_axis", "dominant_conf_share"] if col in over_dom.columns]
        print(over_dom[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")
    ensure_columns(df, REQUIRED_COLUMNS)
    df = ensure_numeric(df, NUMERIC_REQUIRED_COLUMNS)

    logging.info("Loaded ranking CSV: %s (rows=%d)", args.input_csv.resolve(), len(df))

    distribution = compute_distribution(df["confidence_score"])
    relationship = compute_relationship(df)
    high_conf_df = build_high_confidence_samples(df, args.top_n)
    exception_df = build_high_final_low_confidence_samples(df, args.top_n)
    dominance_df, dominance_summary = build_dominance_report(df)

    if pd.notna(dominance_summary["over_dominant_ratio"]) and float(dominance_summary["over_dominant_ratio"]) > 0.30:
        logging.warning("Confidence axis dominance is high: over_dominant_ratio=%.4f", float(dominance_summary["over_dominant_ratio"]))

    print_report(df, distribution, relationship, high_conf_df, exception_df, dominance_df, dominance_summary)
    maybe_save_csv(exception_df, args.out_csv)


if __name__ == "__main__":
    main()
