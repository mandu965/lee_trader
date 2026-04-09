import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/ranking_final.csv")
DEFAULT_OUT_MD = Path("outputs/final_score_dominance_report.md")
DEFAULT_OUT_CSV = Path("outputs/final_score_top20_components.csv")
TOP_N = 20
COMPONENT_COLUMNS = [
    "tech_score",
    "ret_score",
    "prob_score",
    "qual_score",
    "safety_score",
    "liquidity_score",
    "risk_penalty",
]
CONTRIBUTION_COLUMNS = [
    "contrib_tech",
    "contrib_ret",
    "contrib_prob",
    "contrib_qual",
    "contrib_safety",
    "contrib_liquidity",
    "contrib_penalty",
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose final_score dominance and risk penalty impact")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--date", type=str, help="optional filter date YYYY-MM-DD")
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


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_empty_"
    render = df.copy()
    for col in render.columns:
        if pd.api.types.is_numeric_dtype(render[col]):
            render[col] = render[col].map(lambda x: format_float(x, digits=digits))
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


def choose_top_snapshot_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "date" in df.columns and df["date"].notna().any():
        latest_date = str(df["date"].dropna().max())
        latest_df = df.loc[df["date"] == latest_date].copy()
        if not latest_df.empty:
            return latest_df, latest_date
    return df.copy(), "all_dates"


def compute_component_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for col in COMPONENT_COLUMNS:
        if col not in df.columns:
            continue
        work = df[["final_score", col]].apply(pd.to_numeric, errors="coerce").dropna()
        corr = float(work["final_score"].corr(work[col])) if len(work) >= 2 else np.nan
        rows.append(
            {
                "component": col,
                "corr_with_final_score": corr,
                "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
                "nonnull_rows": int(len(work)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_corr", "component"], ascending=[False, True]).reset_index(drop=True)


def compute_regime_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "regime" not in df.columns:
        return pd.DataFrame(columns=["regime", "count", "ratio"])
    dist = df["regime"].fillna("NA").astype(str).value_counts(dropna=False).rename_axis("regime").reset_index(name="count")
    dist["ratio"] = dist["count"] / max(len(df), 1)
    return dist


def compute_risk_penalty_summary(df: pd.DataFrame) -> dict[str, float | int]:
    penalty = pd.to_numeric(df.get("risk_penalty"), errors="coerce")
    before = pd.to_numeric(df.get("final_score_before_penalty"), errors="coerce")
    return {
        "rows": int(len(df)),
        "nonnull": int(penalty.notna().sum()),
        "mean": float(penalty.mean()) if penalty.notna().any() else np.nan,
        "median": float(penalty.median()) if penalty.notna().any() else np.nan,
        "p75": float(penalty.quantile(0.75)) if penalty.notna().any() else np.nan,
        "p90": float(penalty.quantile(0.90)) if penalty.notna().any() else np.nan,
        "max": float(penalty.max()) if penalty.notna().any() else np.nan,
        "zero_ratio": float((penalty.fillna(0.0) == 0.0).mean()) if len(penalty) else np.nan,
        "ge_6_ratio": float((penalty.fillna(0.0) >= 6.0).mean()) if len(penalty) else np.nan,
        "mean_penalty_share": float((penalty / before.replace(0.0, np.nan)).mean()) if before.notna().any() else np.nan,
    }


def build_top_snapshot(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_df, scope = choose_top_snapshot_frame(df)
    sort_specs = [("rank_final", True), ("final_score", False), ("rank_v2", True), ("final_score_v2", False)]
    order_cols = [col for col, _ in sort_specs if col in top_df.columns]
    ascending = [asc for col, asc in sort_specs if col in top_df.columns]
    if order_cols:
        top_df = top_df.sort_values(order_cols, ascending=ascending)
    else:
        top_df = top_df.sort_values(["final_score"], ascending=[False])

    base_cols = [col for col in ["date", "code", "name", "market", "sector", "regime", "rank_final", "rank_v2"] if col in top_df.columns]
    score_cols = [col for col in ["final_score", "final_score_before_penalty", "final_score_v2", "risk_penalty", "w_risk_penalty"] if col in top_df.columns]
    component_cols = [col for col in COMPONENT_COLUMNS if col in top_df.columns and col not in score_cols]
    contribution_cols = [col for col in CONTRIBUTION_COLUMNS if col in top_df.columns]
    out = top_df.head(top_n)[base_cols + score_cols + component_cols + contribution_cols].copy()
    out.attrs["scope"] = scope
    return out


def build_penalty_pressure_examples(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = df.copy()
    work["risk_penalty"] = pd.to_numeric(work.get("risk_penalty"), errors="coerce").fillna(0.0)
    work["w_risk_penalty"] = pd.to_numeric(work.get("w_risk_penalty"), errors="coerce").fillna(1.0)
    work["final_score"] = pd.to_numeric(work.get("final_score"), errors="coerce")
    work["penalty_impact"] = work["risk_penalty"] * work["w_risk_penalty"]
    work["final_score_before_penalty"] = work["final_score"].fillna(0.0) + work["penalty_impact"]
    cols = [
        col
        for col in [
            "date",
            "code",
            "name",
            "regime",
            "final_score",
            "final_score_before_penalty",
            "risk_penalty",
            "w_risk_penalty",
            "penalty_impact",
            "score_drag_1",
            "score_drag_2",
        ]
        if col in work.columns
    ]
    return work.sort_values(["penalty_impact", "final_score_before_penalty"], ascending=[False, False]).head(top_n)[cols]


def build_penalty_drag_summary(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_df, _ = choose_top_snapshot_frame(df)
    top = top_df.sort_values(["final_score"], ascending=[False]).head(top_n).copy()
    values = []
    for col in ["score_drag_1", "score_drag_2"]:
        if col in top.columns:
            values.extend(top[col].dropna().astype(str).tolist())
    if not values:
        return pd.DataFrame(columns=["drag_code", "count"])
    return pd.Series(values).value_counts().rename_axis("drag_code").reset_index(name="count")


def interpret_comments(corr_df: pd.DataFrame, risk_summary: dict[str, float | int], drag_summary: pd.DataFrame) -> list[str]:
    comments: list[str] = []
    if not corr_df.empty and pd.notna(corr_df.iloc[0]["corr_with_final_score"]):
        leader = corr_df.iloc[0]
        comments.append(f"- final_score dominance leader is `{leader['component']}` (corr={format_float(leader['corr_with_final_score'])}).")
    else:
        comments.append("- component correlation could not be computed reliably.")

    mean_penalty_share = risk_summary.get("mean_penalty_share")
    ge_6_ratio = risk_summary.get("ge_6_ratio")
    if pd.notna(mean_penalty_share) and pd.notna(ge_6_ratio):
        if float(mean_penalty_share) >= 0.12 or float(ge_6_ratio) >= 0.25:
            comments.append(
                f"- risk_penalty is still influential (mean_share={format_float(mean_penalty_share)}, ge_6_ratio={format_float(ge_6_ratio)})."
            )
        else:
            comments.append(
                f"- risk_penalty influence looks controlled (mean_share={format_float(mean_penalty_share)}, ge_6_ratio={format_float(ge_6_ratio)})."
            )
    else:
        comments.append("- risk_penalty intensity could not be estimated.")

    if not drag_summary.empty:
        elevated_count = int(
            drag_summary.loc[
                drag_summary["drag_code"].isin(["elevated_risk_penalty", "very_high_risk_penalty"]),
                "count",
            ].sum()
        )
        comments.append(f"- elevated_risk_penalty family repeats {elevated_count} times in top20 drag labels.")

    target_components = corr_df.set_index("component")["abs_corr"].to_dict() if not corr_df.empty else {}
    tqp = [target_components.get(col, np.nan) for col in ["tech_score", "qual_score", "prob_score"]]
    valid_tqp = [float(x) for x in tqp if pd.notna(x)]
    if valid_tqp:
        avg_tqp = float(np.mean(valid_tqp))
        comments.append(f"- average abs corr of tech/qual/prob is {format_float(avg_tqp)}.")
    return comments


def render_markdown(
    *,
    df: pd.DataFrame,
    snapshot_scope: str,
    corr_df: pd.DataFrame,
    regime_dist: pd.DataFrame,
    risk_summary: dict[str, float | int],
    top_df: pd.DataFrame,
    penalty_examples: pd.DataFrame,
    penalty_drag_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Final Score Dominance Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- rows: {len(df)}")
    if "date" in df.columns and df["date"].notna().any():
        lines.append(f"- date_range: {df['date'].min()} ~ {df['date'].max()}")
    lines.append(f"- top20_snapshot_scope: {snapshot_scope}")
    if not corr_df.empty and pd.notna(corr_df.iloc[0]["corr_with_final_score"]):
        lines.append(f"- strongest_component: {corr_df.iloc[0]['component']} (corr={format_float(corr_df.iloc[0]['corr_with_final_score'])})")
    lines.append(f"- mean_risk_penalty: {format_float(risk_summary.get('mean'))}")
    lines.append("")

    lines.append("## Regime Distribution")
    lines.append(dataframe_to_markdown(regime_dist) if not regime_dist.empty else "regime column unavailable.")
    lines.append("")

    lines.append("## Component Correlation With final_score")
    lines.append(dataframe_to_markdown(corr_df) if not corr_df.empty else "component correlation unavailable.")
    lines.append("")

    lines.append("## Risk Penalty Summary")
    for key in ["nonnull", "mean", "median", "p75", "p90", "max", "zero_ratio", "ge_6_ratio", "mean_penalty_share"]:
        digits = 0 if key == "nonnull" else 4
        lines.append(f"- {key}: {format_float(risk_summary.get(key), digits=digits)}")
    lines.append("")

    lines.append("## Top20 Component Snapshot")
    lines.append(dataframe_to_markdown(top_df) if not top_df.empty else "top20 snapshot unavailable.")
    lines.append("")

    lines.append("## Penalty Pressure Examples")
    lines.append(dataframe_to_markdown(penalty_examples) if not penalty_examples.empty else "penalty pressure examples unavailable.")
    lines.append("")

    lines.append("## Top20 Drag Repetition")
    if penalty_drag_summary.empty:
        lines.append("drag summary unavailable.")
    else:
        lines.append(dataframe_to_markdown(penalty_drag_summary))
        elevated_count = int(
            penalty_drag_summary.loc[
                penalty_drag_summary["drag_code"].isin(["elevated_risk_penalty", "very_high_risk_penalty"]),
                "count",
            ].sum()
        )
        lines.append(f"- elevated_risk_penalty_repeat_top20: {elevated_count}")
    lines.append("")

    lines.append("## Interpretation")
    lines.extend(interpret_comments(corr_df, risk_summary, penalty_drag_summary))
    lines.append("")
    return "\n".join(lines)


def save_outputs(out_md: Path, out_csv: Path, markdown: str, top_df: pd.DataFrame) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    top_df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved markdown report: %s", out_md.resolve())
    logging.info("Saved top20 component CSV: %s", out_csv.resolve())


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_ranking(args.input_csv, args.date)
    if df.empty:
        raise ValueError("input ranking data is empty after filtering")

    required = {"final_score", "risk_penalty"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")

    df = ensure_numeric(df, ["final_score", "final_score_v2", "rank_final", "rank_v2", "w_risk_penalty", *COMPONENT_COLUMNS, *CONTRIBUTION_COLUMNS])
    df["final_score_before_penalty"] = (
        pd.to_numeric(df["final_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["risk_penalty"], errors="coerce").fillna(0.0) * pd.to_numeric(df.get("w_risk_penalty"), errors="coerce").fillna(1.0)
    )

    corr_df = compute_component_correlation(df)
    regime_dist = compute_regime_distribution(df)
    risk_summary = compute_risk_penalty_summary(df)
    top_df = build_top_snapshot(df, args.top_n)
    penalty_examples = build_penalty_pressure_examples(df, args.top_n)
    penalty_drag_summary = build_penalty_drag_summary(df, args.top_n)
    snapshot_scope = str(top_df.attrs.get("scope", "all_dates"))
    markdown = render_markdown(
        df=df,
        snapshot_scope=snapshot_scope,
        corr_df=corr_df,
        regime_dist=regime_dist,
        risk_summary=risk_summary,
        top_df=top_df,
        penalty_examples=penalty_examples,
        penalty_drag_summary=penalty_drag_summary,
    )
    save_outputs(args.out_md, args.out_csv, markdown, top_df)


if __name__ == "__main__":
    main()
