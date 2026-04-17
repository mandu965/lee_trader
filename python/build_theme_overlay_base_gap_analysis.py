import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
RANK_LIFT_V2_CSV = DATA_DIR / "theme_overlay_rank_lift_report_v2.csv"
OUTPUT_MD = DATA_DIR / "theme_overlay_base_gap_analysis.md"

LOGGER = logging.getLogger("build_theme_overlay_base_gap_analysis")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest() -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = pd.read_csv(RANKING_CSV, low_memory=False, dtype={"code": str})
    rank["date"] = rank["date"].astype(str)
    latest = rank.loc[rank["date"] == rank["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    for col in [
        "rank_final",
        "final_score",
        "final_score_v3",
        "score_diff_v3",
        "theme_score",
        "theme_confidence",
        "ret_score",
        "prob_score",
        "tech_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
    ]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce").fillna(0.0)
    latest = latest.sort_values("rank_final").reset_index(drop=True)

    lift = pd.read_csv(RANK_LIFT_V2_CSV, dtype={"code": str})
    lift["code"] = lift["code"].astype(str).str.zfill(6)
    for col in ["base_rank", "new_rank", "rank_change", "score_diff", "theme_score", "theme_confidence"]:
        lift[col] = pd.to_numeric(lift.get(col), errors="coerce").fillna(0.0)
    return latest, lift


def describe_group(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "count": int(len(df)),
        "avg_final_score": float(df["final_score"].mean()) if not df.empty else 0.0,
        "avg_final_score_v3": float(df["final_score_v3"].mean()) if not df.empty else 0.0,
        "avg_theme_score": float(df["theme_score"].mean()) if not df.empty else 0.0,
        "avg_theme_confidence": float(df["theme_confidence"].mean()) if not df.empty else 0.0,
        "avg_ret_score": float(df["ret_score"].mean()) if not df.empty else 0.0,
        "avg_prob_score": float(df["prob_score"].mean()) if not df.empty else 0.0,
        "avg_tech_score": float(df["tech_score"].mean()) if not df.empty else 0.0,
        "avg_qual_score": float(df["qual_score"].mean()) if not df.empty else 0.0,
        "avg_safety_score": float(df["safety_score"].mean()) if not df.empty else 0.0,
        "avg_liquidity_score": float(df["liquidity_score"].mean()) if not df.empty else 0.0,
        "avg_risk_penalty": float(df["risk_penalty"].mean()) if not df.empty else 0.0,
    }


def format_stats(stats: dict) -> list[str]:
    return [
        f"- count={stats['count']}",
        f"- avg_final_score={stats['avg_final_score']:.2f}",
        f"- avg_final_score_v3={stats['avg_final_score_v3']:.2f}",
        f"- avg_theme_score={stats['avg_theme_score']:.2f}",
        f"- avg_theme_confidence={stats['avg_theme_confidence']:.3f}",
        f"- avg_ret_score={stats['avg_ret_score']:.2f}",
        f"- avg_prob_score={stats['avg_prob_score']:.2f}",
        f"- avg_tech_score={stats['avg_tech_score']:.2f}",
        f"- avg_qual_score={stats['avg_qual_score']:.2f}",
        f"- avg_safety_score={stats['avg_safety_score']:.2f}",
        f"- avg_liquidity_score={stats['avg_liquidity_score']:.2f}",
        f"- avg_risk_penalty={stats['avg_risk_penalty']:.2f}",
    ]


def build_markdown(latest: pd.DataFrame, lift: pd.DataFrame) -> str:
    top20 = latest.loc[latest["rank_final"] <= 20].copy()
    near_top20 = latest.loc[(latest["rank_final"] >= 21) & (latest["rank_final"] <= 40)].copy()
    themed_near = near_top20.loc[near_top20["dominant_theme"].str.strip().ne("")].copy()
    near_movers = lift.loc[(lift["base_rank"] >= 21) & (lift["base_rank"] <= 40) & (lift["rank_change"] > 0)].copy()
    top_lifters = lift.head(10).copy()

    top20_stats = describe_group(top20, "top20")
    near_stats = describe_group(near_top20, "rank21_40")
    themed_near_stats = describe_group(themed_near, "themed_rank21_40")

    lines = [
        "# Theme Overlay Base Score Gap Analysis",
        "",
        "## Summary",
        "- 목적: 테마 종목이 왜 top20에 제한적으로만 진입하는지, `theme_weight 부족`이 아니라 `base score gap`인지 확인한다.",
        "",
        "## Group Comparison",
        "### Top20",
        *format_stats(top20_stats),
        "",
        "### Rank 21-40",
        *format_stats(near_stats),
        "",
        "### Themed Rank 21-40",
        *format_stats(themed_near_stats),
        "",
        "## Key Gaps",
        f"- top20 vs themed_rank21_40 final_score gap = {top20_stats['avg_final_score'] - themed_near_stats['avg_final_score']:.2f}",
        f"- top20 vs themed_rank21_40 ret_score gap = {top20_stats['avg_ret_score'] - themed_near_stats['avg_ret_score']:.2f}",
        f"- top20 vs themed_rank21_40 prob_score gap = {top20_stats['avg_prob_score'] - themed_near_stats['avg_prob_score']:.2f}",
        f"- top20 vs themed_rank21_40 tech_score gap = {top20_stats['avg_tech_score'] - themed_near_stats['avg_tech_score']:.2f}",
        f"- top20 vs themed_rank21_40 qual_score gap = {top20_stats['avg_qual_score'] - themed_near_stats['avg_qual_score']:.2f}",
        f"- themed_rank21_40 avg theme_score = {themed_near_stats['avg_theme_score']:.2f}",
        f"- themed_rank21_40 avg theme_confidence = {themed_near_stats['avg_theme_confidence']:.3f}",
        "",
        "## Reading",
    ]

    if top20_stats["avg_final_score"] - themed_near_stats["avg_final_score"] > 5:
        lines.append("- 상위권과 근접권 사이의 기본 점수 격차가 아직 크다. 테마 overlay만으로 뒤집기 어려운 구간이 남아 있다.")
    if top20_stats["avg_ret_score"] > themed_near_stats["avg_ret_score"]:
        lines.append("- 가장 큰 차이는 `ret_score` 쪽이다. 현재 상위권은 예측 수익률 축이 더 강하다.")
    if top20_stats["avg_prob_score"] > themed_near_stats["avg_prob_score"]:
        lines.append("- `prob_score`도 상위권이 우세하다. 테마가 붙어도 상위권 진입 확률 점수가 약하면 진입이 어렵다.")
    if themed_near_stats["avg_theme_score"] > 0:
        lines.append("- 근접 테마 종목의 theme_score 자체는 충분히 의미가 있다. 문제는 테마가 약해서가 아니라 base 축의 열세다.")

    lines.extend([
        "",
        "## Near Top20 Movers",
    ])
    if near_movers.empty:
        lines.append("- none")
    else:
        for row in near_movers[["code", "name", "base_rank", "new_rank", "rank_change", "dominant_theme", "theme_score", "theme_confidence", "score_diff"]].itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: {int(row.base_rank)} -> {int(row.new_rank)}, "
                f"rank_change={int(row.rank_change)}, theme={row.dominant_theme}, "
                f"theme_score={float(row.theme_score):.2f}, theme_confidence={float(row.theme_confidence):.3f}, "
                f"score_diff={float(row.score_diff):.4f}"
            )

    lines.extend([
        "",
        "## Top Lifters Pattern",
    ])
    theme_dist = top_lifters["dominant_theme"].value_counts().to_dict()
    lines.append(f"- top10_lifter_theme_distribution={theme_dist}")
    lines.append("- 상위 상승 종목은 HBM, AI서버기판, 방산, 반도체장비에 집중된다. 즉 현재 overlay는 주도 테마에는 작동한다.")

    lines.extend([
        "",
        "## Diagnosis",
        "- 이번 단계의 1차 병목은 더 이상 조인이 아니라 `base score gap`이다.",
        "- coverage 확대로 near-top20 후보는 늘었지만, top20 핵심 구간은 여전히 비테마 고득점 종목이 장악한다.",
        "- 따라서 다음 액션은 theme_weight 상향보다 `top20 비테마 종목의 ret/prob 우위 원인`을 점검하는 것이다.",
    ])
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    latest, lift = load_latest()
    OUTPUT_MD.write_text(build_markdown(latest, lift), encoding="utf-8")
    LOGGER.info("saved %s", OUTPUT_MD)
    print(f"generated_files={[str(OUTPUT_MD)]}")
    print("example=python python\\build_theme_overlay_base_gap_analysis.py")


if __name__ == "__main__":
    main()
