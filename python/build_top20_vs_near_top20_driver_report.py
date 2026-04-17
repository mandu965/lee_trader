from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from build_theme_overlay_final_analysis import TOP_N, load_latest_ranking


DATA_DIR = Path("data")
OUTPUT_MD = DATA_DIR / "top20_vs_near_top20_driver_report.md"
OUTPUT_CSV = DATA_DIR / "top20_vs_near_top20_driver_report.csv"

LOGGER = logging.getLogger("build_top20_vs_near_top20_driver_report")

CORE_COLS = [
    "final_score",
    "final_score_v3",
    "theme_score",
    "theme_confidence",
    "ret_score",
    "prob_score",
    "tech_score",
    "qual_score",
    "safety_score",
    "liquidity_score",
    "risk_penalty",
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _avg_series(df: pd.DataFrame) -> dict[str, float]:
    return {col: float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean()) if not df.empty else 0.0 for col in CORE_COLS}


def build_driver_gap_table(top20_no_theme: pd.DataFrame, near_themed: pd.DataFrame) -> pd.DataFrame:
    top_avg = _avg_series(top20_no_theme)
    near_avg = _avg_series(near_themed)
    rows = []
    for col in CORE_COLS:
        gap = top_avg[col] - near_avg[col]
        direction = "top20_non_theme_advantage" if gap > 0 else ("near_top20_themed_advantage" if gap < 0 else "flat")
        rows.append(
            {
                "metric": col,
                "top20_no_theme_avg": top_avg[col],
                "near_top20_themed_avg": near_avg[col],
                "gap_top20_minus_near": gap,
                "advantage_side": direction,
                "explain_ko": f"{col} 기준으로 top20 비테마 평균과 near-top20 테마 평균의 차이는 {gap:+.2f}다.",
                "explain_en": f"The gap in {col} between non-theme top20 names and themed near-top20 names is {gap:+.2f}.",
            }
        )
    return pd.DataFrame(rows).sort_values("gap_top20_minus_near", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def build_markdown(latest: pd.DataFrame, gap_df: pd.DataFrame) -> str:
    top20_no_theme = latest[(latest["base_rank"].le(TOP_N)) & (latest["is_no_theme"])].copy()
    near_themed = latest[(latest["base_rank"].between(TOP_N + 1, 40)) & (latest["is_themed"])].copy()
    lines = [
        "# Top20 vs Near-Top20 Driver Report",
        "",
        "## Scope",
        f"- non_theme_top20_count: {len(top20_no_theme)}",
        f"- themed_rank21_40_count: {len(near_themed)}",
        "",
        "## Largest Gaps",
    ]
    for row in gap_df.head(6).itertuples(index=False):
        lines.append(
            f"- {row.metric}: top20_non_theme_avg={float(row.top20_no_theme_avg):.2f}, "
            f"near_top20_themed_avg={float(row.near_top20_themed_avg):.2f}, gap={float(row.gap_top20_minus_near):+.2f}"
        )
    lines.extend([
        "",
        "## Interpretation",
        "- 한국어: top20 비테마 종목이 유지되는 핵심 이유가 `ret_score`, `prob_score`, `qual_score` 쪽인지 확인하기 위한 보고서다.",
        "- English: This report checks whether non-theme top20 names are defended mainly by return, probability, and quality factors.",
        "",
        "## Sample: Non-Theme Top20",
    ])
    for row in top20_no_theme.head(8).itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, final_score={float(row.final_score):.2f}, "
            f"ret={float(row.ret_score):.2f}, prob={float(row.prob_score):.2f}, qual={float(row.qual_score):.2f}"
        )
    lines.extend(["", "## Sample: Themed Near-Top20"])
    for row in near_themed.head(8).itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, theme={row.dominant_theme}, "
            f"theme_score={float(row.theme_score):.2f}, ret={float(row.ret_score):.2f}, prob={float(row.prob_score):.2f}"
        )
    lines.extend([
        "",
        "## Conclusion",
        "- 한국어: top20 진입 실패 원인이 단순히 theme_weight 부족인지, 아니면 base 축 열세인지 운영자가 빠르게 판단하도록 돕는다.",
        "- English: The goal is to separate a pure theme-weight issue from a broader base-score deficit.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    top20_no_theme = latest[(latest["base_rank"].le(TOP_N)) & (latest["is_no_theme"])].copy()
    near_themed = latest[(latest["base_rank"].between(TOP_N + 1, 40)) & (latest["is_themed"])].copy()
    gap_df = build_driver_gap_table(top20_no_theme, near_themed)
    gap_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_MD.write_text(build_markdown(latest, gap_df), encoding="utf-8")
    LOGGER.info("Saved %s", OUTPUT_CSV.resolve())
    LOGGER.info("Saved %s", OUTPUT_MD.resolve())
    print(f"generated_files={[str(OUTPUT_CSV), str(OUTPUT_MD)]}")


if __name__ == "__main__":
    main()
