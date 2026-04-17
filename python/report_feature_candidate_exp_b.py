from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "feature_candidate_exp_b.csv"
OUT_MD = DATA_DIR / "feature_candidate_exp_b_displaced_no_theme.md"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def load_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"missing input csv: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("feature_candidate_exp_b.csv is empty")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["regime"] = df.get("regime", "").fillna("").astype(str)
    df["dominant_theme"] = df.get("dominant_theme", "(none)").fillna("(none)").replace("", "(none)").astype(str)
    for col in [
        "theme_score",
        "theme_confidence",
        "baseline_final_score",
        "candidate_final_score",
        "score_delta",
        "baseline_rank",
        "candidate_rank",
        "rank_delta",
        "baseline_risk_penalty",
        "candidate_risk_penalty",
        "penalty_delta",
        "has_theme_flag",
    ]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
    df["candidate_applied_flag"] = df.get("candidate_applied_flag", False).fillna(False).astype(bool)
    df["candidate_reason"] = df.get("candidate_reason", "candidate_disabled").fillna("candidate_disabled").astype(str)
    return df


def build_report(df: pd.DataFrame) -> str:
    baseline_near = set(df.loc[df["baseline_rank"].between(15, 30), "symbol"].astype(str))
    candidate_near = set(df.loc[df["candidate_rank"].between(15, 30), "symbol"].astype(str))
    entered_near = candidate_near - baseline_near

    entered_near_df = df.loc[df["symbol"].astype(str).isin(entered_near)].copy()
    entered_near_no_theme = entered_near_df.loc[entered_near_df["has_theme_flag"].eq(0)].copy()

    displaced_from_top20 = df.loc[
        df["symbol"].astype(str).isin(candidate_near)
        & df["baseline_rank"].le(20)
        & df["has_theme_flag"].eq(0)
    ].copy()

    top_lifted_themed = df.loc[df["has_theme_flag"].eq(1)].sort_values(
        ["rank_delta", "candidate_rank"],
        ascending=[False, True],
    ).head(10)

    lines = [
        "# Feature Candidate Exp-B Displaced No-Theme Memo",
        "",
        "## Overview",
        "- This memo isolates whether near-top20 no-theme ratio increased because no-theme names were lifted, or because themed names displaced them downward.",
        "",
        "## Key Counts",
        f"- entered_near_count={int(len(entered_near_df))}",
        f"- entered_near_no_theme_count={int(len(entered_near_no_theme))}",
        f"- displaced_top20_no_theme_count={int(len(displaced_from_top20))}",
        f"- no_theme_positive_rank_delta_count={int((df.loc[df['has_theme_flag'].eq(0), 'rank_delta'] > 0).sum())}",
        "",
        "## Interpretation",
    ]

    if int((df.loc[df["has_theme_flag"].eq(0), "rank_delta"] > 0).sum()) == 0:
        lines.append("- No-theme names were not positively lifted by the candidate. The near-top20 ratio increase is a displacement effect, not a no-theme uplift effect.")
    else:
        lines.append("- Some no-theme names did receive positive rank delta. Review these names before candidate promotion.")

    lines.extend(["", "## Entered Near-Top20 No-Theme Names"])
    if entered_near_no_theme.empty:
        lines.append("- none")
    else:
        for row in entered_near_no_theme.sort_values(["candidate_rank", "rank_delta"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"- {row.symbol} {row.name}: baseline_rank={int(row.baseline_rank)}, candidate_rank={int(row.candidate_rank)}, "
                f"rank_delta={int(row.rank_delta)}, reason={row.candidate_reason}"
            )

    lines.extend(["", "## Displaced Top20 No-Theme Names"])
    if displaced_from_top20.empty:
        lines.append("- none")
    else:
        for row in displaced_from_top20.sort_values(["candidate_rank", "baseline_rank"], ascending=[True, True]).itertuples(index=False):
            lines.append(
                f"- {row.symbol} {row.name}: baseline_rank={int(row.baseline_rank)}, candidate_rank={int(row.candidate_rank)}, "
                f"rank_delta={int(row.rank_delta)}, reason={row.candidate_reason}"
            )

    lines.extend(["", "## Top Lifted Themed Names"])
    if top_lifted_themed.empty:
        lines.append("- none")
    else:
        for row in top_lifted_themed.itertuples(index=False):
            lines.append(
                f"- {row.symbol} {row.name}: theme={row.dominant_theme}, baseline_rank={int(row.baseline_rank)}, "
                f"candidate_rank={int(row.candidate_rank)}, rank_delta={int(row.rank_delta)}, reason={row.candidate_reason}"
            )

    lines.extend([
        "",
        "## Recommendation",
        "- Use this memo together with feature_candidate_exp_b_summary.md. If no_theme_positive_rank_delta_count stays at zero, the remaining review question is whether displacement of top20 no-theme names is acceptable.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    df = load_input()
    OUT_MD.write_text(build_report(df), encoding="utf-8")
    logging.info("Saved displaced no-theme memo: %s", OUT_MD.resolve())
    print(f"generated_files={[str(OUT_MD)]}")
    print("example=python python\\report_feature_candidate_exp_b.py")


if __name__ == "__main__":
    main()
