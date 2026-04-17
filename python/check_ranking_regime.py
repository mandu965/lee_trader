import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
TOP_N = 20

BULL_WEIGHTS = {
    "w_tech": 0.15,
    "w_ret": 0.35,
    "w_prob": 0.25,
    "w_qual": 0.10,
    "w_safety": 0.10,
    "w_liquidity": 0.05,
}
DEFENSIVE_WEIGHTS = {
    "w_tech": 0.12,
    "w_ret": 0.22,
    "w_prob": 0.20,
    "w_qual": 0.18,
    "w_safety": 0.21,
    "w_liquidity": 0.07,
}
REQUIRED_COLUMNS = [
    "regime",
    "final_score",
    "pred_score",
    "tech_score",
    "qual_score",
]
WEIGHT_COLUMNS = list(BULL_WEIGHTS.keys())


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ranking_builder regime branch results")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"ranking CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"top rows to print (default: {TOP_N})")
    parser.add_argument("--out-csv", type=Path, help="optional path to save top sample CSV")
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


def compute_regime_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["regime"].fillna("NA").astype(str).value_counts(dropna=False).rename_axis("regime").reset_index(name="count")
    dist["ratio"] = dist["count"] / max(len(df), 1)
    return dist


def validate_branch_masks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df.copy()
    work["bull_mask"] = work["regime"].astype(str).eq("bull")
    work["defensive_mask"] = work["regime"].astype(str).eq("defensive")
    work["mask_overlap"] = work["bull_mask"] & work["defensive_mask"]
    work["mask_missing"] = ~(work["bull_mask"] | work["defensive_mask"])

    if all(col in work.columns for col in WEIGHT_COLUMNS):
        work = ensure_numeric(work, WEIGHT_COLUMNS)
        work["bull_weight_match"] = work["bull_mask"]
        work["defensive_weight_match"] = work["defensive_mask"]
        for col, expected in BULL_WEIGHTS.items():
            work["bull_weight_match"] = work["bull_weight_match"] & np.isclose(work[col], expected, equal_nan=False)
        for col, expected in DEFENSIVE_WEIGHTS.items():
            work["defensive_weight_match"] = work["defensive_weight_match"] & np.isclose(work[col], expected, equal_nan=False)
        work["branch_weight_ok"] = np.where(work["bull_mask"], work["bull_weight_match"], np.where(work["defensive_mask"], work["defensive_weight_match"], False))
    else:
        work["branch_weight_ok"] = np.nan

    summary = {
        "bull_count": int(work["bull_mask"].sum()),
        "defensive_count": int(work["defensive_mask"].sum()),
        "overlap_count": int(work["mask_overlap"].sum()),
        "missing_count": int(work["mask_missing"].sum()),
        "branch_weight_ok_count": int(work["branch_weight_ok"].fillna(False).sum()) if "branch_weight_ok" in work.columns else 0,
    }
    return work, summary


def build_top_samples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(work, ["final_score", "pred_score", "tech_score", "qual_score"])
    cols = [
        col for col in [
            "date",
            "code",
            "name",
            "regime",
            "final_score",
            "pred_score",
            "ret_score",
            "tech_score",
            "qual_score",
            "w_tech",
            "w_ret",
            "w_prob",
            "w_qual",
            "w_safety",
            "w_liquidity",
        ]
        if col in work.columns
    ]
    return work.sort_values(["final_score"], ascending=[False]).head(top_n)[cols]


def build_regime_comparison(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    metric_cols = [col for col in ["final_score", "pred_score", "tech_score", "qual_score"] if col in work.columns]
    work = ensure_numeric(work, metric_cols)
    if not metric_cols:
        return pd.DataFrame()
    out = work.groupby("regime", dropna=False)[metric_cols].mean().reset_index()
    out["count"] = work.groupby("regime", dropna=False).size().values
    return out


def compute_final_score_nan_summary(df: pd.DataFrame) -> dict[str, float | int]:
    score = pd.to_numeric(df["final_score"], errors="coerce")
    return {
        "nan_count": int(score.isna().sum()),
        "nan_ratio": float(score.isna().mean()) if len(score) else np.nan,
    }


def detect_single_branch(df: pd.DataFrame) -> dict[str, object]:
    regime_counts = df["regime"].fillna("NA").astype(str).value_counts(dropna=False)
    unique_regimes = int(regime_counts.shape[0])
    single_branch = unique_regimes <= 1
    only_regime = str(regime_counts.index[0]) if len(regime_counts) else "NA"
    return {
        "single_branch": single_branch,
        "unique_regime_count": unique_regimes,
        "only_regime": only_regime,
    }


def maybe_save_csv(df: pd.DataFrame, out_csv: Path | None) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved ranking regime check CSV: %s (rows=%d)", out_csv.resolve(), len(df))


def print_report(
    df: pd.DataFrame,
    regime_dist: pd.DataFrame,
    branch_summary: dict[str, int],
    top_samples: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    nan_summary: dict[str, float | int],
    single_branch: dict[str, object],
) -> None:
    print("=== ranking regime check ===")
    print(f"rows={len(df)}")
    if "date" in df.columns and df["date"].notna().any():
        print(f"date_range={df['date'].min()} ~ {df['date'].max()}")
    print("")

    print("[regime distribution]")
    print(regime_dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[branch mask validation]")
    print(f"bull_count={branch_summary['bull_count']}")
    print(f"defensive_count={branch_summary['defensive_count']}")
    print(f"mask_overlap_count={branch_summary['overlap_count']}")
    print(f"mask_missing_count={branch_summary['missing_count']}")
    print(f"branch_weight_ok_count={branch_summary['branch_weight_ok_count']}")
    print("")

    print("[top final_score samples]")
    if top_samples.empty:
        print("No rows available.")
    else:
        print(top_samples.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[regime mean comparison]")
    if regime_comparison.empty:
        print("No comparison columns available.")
    else:
        print(regime_comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[final_score null check]")
    print(f"final_score_nan_count={nan_summary['nan_count']}")
    print(f"final_score_nan_ratio={nan_summary['nan_ratio']:.4f}")
    print("")

    print("[single branch detection]")
    print(f"single_branch_only={'YES' if bool(single_branch['single_branch']) else 'NO'}")
    print(f"unique_regime_count={single_branch['unique_regime_count']}")
    print(f"only_regime={single_branch['only_regime']}")


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")
    ensure_columns(df, REQUIRED_COLUMNS)

    logging.info("Loaded ranking CSV: %s (rows=%d)", args.input_csv.resolve(), len(df))

    regime_dist = compute_regime_distribution(df)
    checked_df, branch_summary = validate_branch_masks(df)
    top_samples = build_top_samples(checked_df, args.top_n)
    regime_comparison = build_regime_comparison(checked_df)
    nan_summary = compute_final_score_nan_summary(checked_df)
    single_branch = detect_single_branch(checked_df)

    if branch_summary["overlap_count"] > 0 or branch_summary["missing_count"] > 0:
        logging.warning(
            "Branch mask anomaly detected: overlap=%d missing=%d",
            branch_summary["overlap_count"],
            branch_summary["missing_count"],
        )
    if branch_summary["branch_weight_ok_count"] < len(checked_df):
        logging.warning(
            "Branch weight mismatch detected: ok=%d total=%d",
            branch_summary["branch_weight_ok_count"],
            len(checked_df),
        )
    if bool(single_branch["single_branch"]):
        logging.warning("All rows are using a single regime branch: %s", single_branch["only_regime"])

    print_report(checked_df, regime_dist, branch_summary, top_samples, regime_comparison, nan_summary, single_branch)
    maybe_save_csv(top_samples, args.out_csv)


if __name__ == "__main__":
    main()
