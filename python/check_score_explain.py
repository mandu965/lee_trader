<<<<<<< HEAD
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
TOP_N = 20
REQUIRED_COLUMNS = [
    "final_score",
    "final_score_raw",
    "contrib_tech",
    "contrib_ret",
    "contrib_prob",
    "contrib_qual",
    "contrib_safety",
    "contrib_liquidity",
    "contrib_penalty",
    "explain_text",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check score explain consistency and text coverage")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"ranking CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"top rows to sample (default: {TOP_N})")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="score equality tolerance")
    parser.add_argument("--out-csv", type=Path, help="optional path to save contradiction rows")
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


def compute_score_consistency(df: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    numeric_cols = [
        "final_score",
        "final_score_raw",
        "contrib_tech",
        "contrib_ret",
        "contrib_prob",
        "contrib_qual",
        "contrib_safety",
        "contrib_liquidity",
        "contrib_penalty",
    ]
    work = ensure_numeric(work, numeric_cols)

    work["components_sum_raw"] = (
        work["contrib_tech"].fillna(0.0)
        + work["contrib_ret"].fillna(0.0)
        + work["contrib_prob"].fillna(0.0)
        + work["contrib_qual"].fillna(0.0)
        + work["contrib_safety"].fillna(0.0)
        + work["contrib_liquidity"].fillna(0.0)
    )
    work["expected_final_score"] = (work["final_score_raw"].fillna(0.0) + work["contrib_penalty"].fillna(0.0)).clip(lower=0.0, upper=100.0)
    work["raw_sum_diff"] = (work["final_score_raw"] - work["components_sum_raw"]).abs()
    work["final_sum_diff"] = (work["final_score"] - work["expected_final_score"]).abs()
    work["is_raw_sum_match"] = work["raw_sum_diff"] <= tolerance
    work["is_final_sum_match"] = work["final_sum_diff"] <= tolerance

    summary = {
        "raw_match_ratio": float(work["is_raw_sum_match"].mean()) if len(work) else np.nan,
        "final_match_ratio": float(work["is_final_sum_match"].mean()) if len(work) else np.nan,
        "raw_mismatch_count": int((~work["is_raw_sum_match"]).sum()),
        "final_mismatch_count": int((~work["is_final_sum_match"]).sum()),
        "max_raw_diff": float(work["raw_sum_diff"].max()) if len(work) else np.nan,
        "max_final_diff": float(work["final_sum_diff"].max()) if len(work) else np.nan,
    }
    return work, summary


def compute_empty_explain_ratio(df: pd.DataFrame) -> float:
    explain = df["explain_text"].fillna("").astype(str).str.strip()
    if len(explain) == 0:
        return np.nan
    return float(explain.eq("").mean())


def _format_1dp(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.1f}"


def _clean_text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def detect_text_contradictions(row: pd.Series) -> list[str]:
    issues: list[str] = []
    text = _clean_text_value(row.get("explain_text"))
    if not text:
        return ["empty_explain_text"]

    regime = _clean_text_value(row.get("regime"))
    if regime and regime not in text:
        issues.append("regime_missing_in_text")

    final_score_text = _format_1dp(row.get("final_score"))
    if final_score_text and final_score_text not in text:
        issues.append("final_score_missing_in_text")

    field_specs = [
        ("ret_score", "ret_score"),
        ("prob_score", "prob_score"),
        ("qual_score", "qual_score"),
        ("tech_score", "tech_score"),
    ]
    for col, label in field_specs:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            if label not in text:
                issues.append(f"{col}_label_missing")
            formatted = _format_1dp(value)
            if formatted and formatted not in text:
                issues.append(f"{col}_value_missing")

    top_positive_factor = _clean_text_value(row.get("top_positive_factor"))
    if top_positive_factor:
        if top_positive_factor not in text:
            issues.append("top_positive_factor_missing")
        pos_value = _format_1dp(row.get("top_positive_value"))
        if pos_value and pos_value not in text:
            issues.append("top_positive_value_missing")

    top_negative_factor = _clean_text_value(row.get("top_negative_factor"))
    if top_negative_factor:
        if top_negative_factor not in text:
            issues.append("top_negative_factor_missing")
        neg_value = _format_1dp(row.get("top_negative_value"))
        if neg_value and neg_value not in text:
            issues.append("top_negative_value_missing")

    return issues


def build_contradiction_report(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["contradiction_issues"] = work.apply(detect_text_contradictions, axis=1)
    work["contradiction_count"] = work["contradiction_issues"].apply(len)
    work["has_contradiction"] = work["contradiction_count"] > 0
    work["contradiction_issues"] = work["contradiction_issues"].apply(lambda items: "|".join(items))
    return work


def build_topn_samples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(work, ["final_score", "top_positive_value", "top_negative_value"])
    cols = [col for col in [
        "date",
        "code",
        "name",
        "regime",
        "final_score",
        "top_positive_factor",
        "top_positive_value",
        "top_negative_factor",
        "top_negative_value",
        "explain_text",
    ] if col in work.columns]
    return work.sort_values(["final_score"], ascending=[False]).head(top_n)[cols]


def maybe_save_csv(df: pd.DataFrame, out_csv: Path | None) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved score explain check CSV: %s (rows=%d)", out_csv.resolve(), len(df))


def print_report(
    df: pd.DataFrame,
    summary: dict[str, float],
    empty_ratio: float,
    contradiction_df: pd.DataFrame,
    top_samples: pd.DataFrame,
) -> None:
    print("=== score explain check ===")
    print(f"rows={len(df)}")
    if "date" in df.columns and df["date"].notna().any():
        print(f"date_range={df['date'].min()} ~ {df['date'].max()}")
    print("")

    print("[score consistency]")
    print(f"raw_match_ratio={summary['raw_match_ratio']:.4f}")
    print(f"final_match_ratio={summary['final_match_ratio']:.4f}")
    print(f"raw_mismatch_count={summary['raw_mismatch_count']}")
    print(f"final_mismatch_count={summary['final_mismatch_count']}")
    print(f"max_raw_diff={summary['max_raw_diff']:.8f}")
    print(f"max_final_diff={summary['max_final_diff']:.8f}")
    print("")

    print("[text coverage]")
    print(f"empty_explain_ratio={empty_ratio:.4f}")
    print(f"contradiction_row_count={int(contradiction_df['has_contradiction'].sum())}")
    print("")

    print("[top 20 explain samples]")
    if top_samples.empty:
        print("No rows available.")
    else:
        print(top_samples.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[contradiction summary]")
    contradiction_rows = contradiction_df.loc[contradiction_df["has_contradiction"]].copy()
    if contradiction_rows.empty:
        print("No contradictions detected.")
    else:
        cols = [col for col in ["date", "code", "name", "final_score", "contradiction_issues"] if col in contradiction_rows.columns]
        print(contradiction_rows[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")
    ensure_columns(df, REQUIRED_COLUMNS)

    logging.info("Loaded ranking CSV: %s (rows=%d)", args.input_csv.resolve(), len(df))

    checked_df, summary = compute_score_consistency(df, args.tolerance)
    empty_ratio = compute_empty_explain_ratio(checked_df)
    contradiction_df = build_contradiction_report(checked_df)
    top_samples = build_topn_samples(checked_df, args.top_n)

    if summary["raw_mismatch_count"] > 0 or summary["final_mismatch_count"] > 0:
        logging.warning(
            "Score consistency mismatch detected: raw=%d final=%d",
            summary["raw_mismatch_count"],
            summary["final_mismatch_count"],
        )
    if contradiction_df["has_contradiction"].any():
        logging.warning("Explain contradictions detected: %d rows", int(contradiction_df["has_contradiction"].sum()))

    print_report(checked_df, summary, empty_ratio, contradiction_df, top_samples)
    maybe_save_csv(contradiction_df.loc[contradiction_df["has_contradiction"]], args.out_csv)


if __name__ == "__main__":
    main()
=======
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
TOP_N = 20
REQUIRED_COLUMNS = [
    "final_score",
    "final_score_raw",
    "contrib_tech",
    "contrib_ret",
    "contrib_prob",
    "contrib_qual",
    "contrib_safety",
    "contrib_liquidity",
    "contrib_penalty",
    "explain_text",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check score explain consistency and text coverage")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"ranking CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"top rows to sample (default: {TOP_N})")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="score equality tolerance")
    parser.add_argument("--out-csv", type=Path, help="optional path to save contradiction rows")
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


def compute_score_consistency(df: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    numeric_cols = [
        "final_score",
        "final_score_raw",
        "contrib_tech",
        "contrib_ret",
        "contrib_prob",
        "contrib_qual",
        "contrib_safety",
        "contrib_liquidity",
        "contrib_penalty",
    ]
    work = ensure_numeric(work, numeric_cols)

    work["components_sum_raw"] = (
        work["contrib_tech"].fillna(0.0)
        + work["contrib_ret"].fillna(0.0)
        + work["contrib_prob"].fillna(0.0)
        + work["contrib_qual"].fillna(0.0)
        + work["contrib_safety"].fillna(0.0)
        + work["contrib_liquidity"].fillna(0.0)
    )
    work["expected_final_score"] = (work["final_score_raw"].fillna(0.0) + work["contrib_penalty"].fillna(0.0)).clip(lower=0.0, upper=100.0)
    work["raw_sum_diff"] = (work["final_score_raw"] - work["components_sum_raw"]).abs()
    work["final_sum_diff"] = (work["final_score"] - work["expected_final_score"]).abs()
    work["is_raw_sum_match"] = work["raw_sum_diff"] <= tolerance
    work["is_final_sum_match"] = work["final_sum_diff"] <= tolerance

    summary = {
        "raw_match_ratio": float(work["is_raw_sum_match"].mean()) if len(work) else np.nan,
        "final_match_ratio": float(work["is_final_sum_match"].mean()) if len(work) else np.nan,
        "raw_mismatch_count": int((~work["is_raw_sum_match"]).sum()),
        "final_mismatch_count": int((~work["is_final_sum_match"]).sum()),
        "max_raw_diff": float(work["raw_sum_diff"].max()) if len(work) else np.nan,
        "max_final_diff": float(work["final_sum_diff"].max()) if len(work) else np.nan,
    }
    return work, summary


def compute_empty_explain_ratio(df: pd.DataFrame) -> float:
    explain = df["explain_text"].fillna("").astype(str).str.strip()
    if len(explain) == 0:
        return np.nan
    return float(explain.eq("").mean())


def _format_1dp(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.1f}"


def _clean_text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def detect_text_contradictions(row: pd.Series) -> list[str]:
    issues: list[str] = []
    text = _clean_text_value(row.get("explain_text"))
    if not text:
        return ["empty_explain_text"]

    regime = _clean_text_value(row.get("regime"))
    if regime and regime not in text:
        issues.append("regime_missing_in_text")

    final_score_text = _format_1dp(row.get("final_score"))
    if final_score_text and final_score_text not in text:
        issues.append("final_score_missing_in_text")

    field_specs = [
        ("ret_score", "ret_score"),
        ("prob_score", "prob_score"),
        ("qual_score", "qual_score"),
        ("tech_score", "tech_score"),
    ]
    for col, label in field_specs:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            if label not in text:
                issues.append(f"{col}_label_missing")
            formatted = _format_1dp(value)
            if formatted and formatted not in text:
                issues.append(f"{col}_value_missing")

    top_positive_factor = _clean_text_value(row.get("top_positive_factor"))
    if top_positive_factor:
        if top_positive_factor not in text:
            issues.append("top_positive_factor_missing")
        pos_value = _format_1dp(row.get("top_positive_value"))
        if pos_value and pos_value not in text:
            issues.append("top_positive_value_missing")

    top_negative_factor = _clean_text_value(row.get("top_negative_factor"))
    if top_negative_factor:
        if top_negative_factor not in text:
            issues.append("top_negative_factor_missing")
        neg_value = _format_1dp(row.get("top_negative_value"))
        if neg_value and neg_value not in text:
            issues.append("top_negative_value_missing")

    return issues


def build_contradiction_report(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["contradiction_issues"] = work.apply(detect_text_contradictions, axis=1)
    work["contradiction_count"] = work["contradiction_issues"].apply(len)
    work["has_contradiction"] = work["contradiction_count"] > 0
    work["contradiction_issues"] = work["contradiction_issues"].apply(lambda items: "|".join(items))
    return work


def build_topn_samples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work = ensure_numeric(work, ["final_score", "top_positive_value", "top_negative_value"])
    cols = [col for col in [
        "date",
        "code",
        "name",
        "regime",
        "final_score",
        "top_positive_factor",
        "top_positive_value",
        "top_negative_factor",
        "top_negative_value",
        "explain_text",
    ] if col in work.columns]
    return work.sort_values(["final_score"], ascending=[False]).head(top_n)[cols]


def maybe_save_csv(df: pd.DataFrame, out_csv: Path | None) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved score explain check CSV: %s (rows=%d)", out_csv.resolve(), len(df))


def print_report(
    df: pd.DataFrame,
    summary: dict[str, float],
    empty_ratio: float,
    contradiction_df: pd.DataFrame,
    top_samples: pd.DataFrame,
) -> None:
    print("=== score explain check ===")
    print(f"rows={len(df)}")
    if "date" in df.columns and df["date"].notna().any():
        print(f"date_range={df['date'].min()} ~ {df['date'].max()}")
    print("")

    print("[score consistency]")
    print(f"raw_match_ratio={summary['raw_match_ratio']:.4f}")
    print(f"final_match_ratio={summary['final_match_ratio']:.4f}")
    print(f"raw_mismatch_count={summary['raw_mismatch_count']}")
    print(f"final_mismatch_count={summary['final_mismatch_count']}")
    print(f"max_raw_diff={summary['max_raw_diff']:.8f}")
    print(f"max_final_diff={summary['max_final_diff']:.8f}")
    print("")

    print("[text coverage]")
    print(f"empty_explain_ratio={empty_ratio:.4f}")
    print(f"contradiction_row_count={int(contradiction_df['has_contradiction'].sum())}")
    print("")

    print("[top 20 explain samples]")
    if top_samples.empty:
        print("No rows available.")
    else:
        print(top_samples.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("")

    print("[contradiction summary]")
    contradiction_rows = contradiction_df.loc[contradiction_df["has_contradiction"]].copy()
    if contradiction_rows.empty:
        print("No contradictions detected.")
    else:
        cols = [col for col in ["date", "code", "name", "final_score", "contradiction_issues"] if col in contradiction_rows.columns]
        print(contradiction_rows[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")
    ensure_columns(df, REQUIRED_COLUMNS)

    logging.info("Loaded ranking CSV: %s (rows=%d)", args.input_csv.resolve(), len(df))

    checked_df, summary = compute_score_consistency(df, args.tolerance)
    empty_ratio = compute_empty_explain_ratio(checked_df)
    contradiction_df = build_contradiction_report(checked_df)
    top_samples = build_topn_samples(checked_df, args.top_n)

    if summary["raw_mismatch_count"] > 0 or summary["final_mismatch_count"] > 0:
        logging.warning(
            "Score consistency mismatch detected: raw=%d final=%d",
            summary["raw_mismatch_count"],
            summary["final_mismatch_count"],
        )
    if contradiction_df["has_contradiction"].any():
        logging.warning("Explain contradictions detected: %d rows", int(contradiction_df["has_contradiction"].sum()))

    print_report(checked_df, summary, empty_ratio, contradiction_df, top_samples)
    maybe_save_csv(contradiction_df.loc[contradiction_df["has_contradiction"]], args.out_csv)


if __name__ == "__main__":
    main()
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
