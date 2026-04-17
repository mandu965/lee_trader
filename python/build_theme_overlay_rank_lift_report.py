import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUTPUT_CSV = DATA_DIR / "theme_overlay_rank_lift_report.csv"
OUTPUT_MD = DATA_DIR / "theme_overlay_rank_lift_report.md"

LOGGER = logging.getLogger("build_theme_overlay_rank_lift_report")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest_ranking(path: Path = RANKING_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ranking file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"ranking file is empty: {path}")

    df["date"] = df["date"].astype(str)
    latest_date = df["date"].max()
    latest = df.loc[df["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str)
    latest["name"] = latest["name"].fillna("").astype(str)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    latest["explain"] = latest.get("explain_text", "").fillna("").astype(str)

    numeric_cols = [
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "theme_score",
        "theme_confidence",
    ]
    for col in numeric_cols:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce").fillna(0.0)

    latest["base_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)
    latest["new_rank"] = latest["final_score_v3"].rank(method="first", ascending=False).astype(int)
    latest["rank_change"] = latest["base_rank"] - latest["new_rank"]
    latest["score_diff"] = latest["final_score_v3"] - latest["final_score"]
    return latest


def build_report_df(latest: pd.DataFrame) -> pd.DataFrame:
    themed = latest.loc[
        (latest["theme_score"] > 0.0)
        | (latest["theme_confidence"] > 0.0)
        | (latest["dominant_theme"].str.strip() != "")
    ].copy()

    themed = themed.loc[:, [
        "date",
        "code",
        "name",
        "base_rank",
        "new_rank",
        "rank_change",
        "final_score",
        "final_score_v3",
        "score_diff",
        "theme_score",
        "theme_confidence",
        "dominant_theme",
        "explain",
    ]]
    themed = themed.sort_values(
        ["rank_change", "score_diff", "theme_score", "code"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return themed


def build_markdown(report_df: pd.DataFrame, latest: pd.DataFrame) -> str:
    theme_count = int(len(report_df))
    avg_rank_change = float(report_df["rank_change"].mean()) if theme_count else 0.0
    median_rank_change = float(report_df["rank_change"].median()) if theme_count else 0.0

    top10 = report_df.head(10).copy()
    theme_summary = (
        report_df.assign(dominant_theme=report_df["dominant_theme"].replace("", "(none)"))
        .groupby("dominant_theme", as_index=False)
        .agg(
            stock_count=("code", "count"),
            avg_rank_change=("rank_change", "mean"),
        )
        .sort_values(["avg_rank_change", "stock_count", "dominant_theme"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    near_top20 = report_df.loc[
        (report_df["base_rank"] >= 21) & (report_df["base_rank"] <= 40) & (report_df["new_rank"] <= 30)
    ].copy()

    top20_v3 = latest.loc[latest["new_rank"] <= 20].copy()
    top20_theme_dist = top20_v3["dominant_theme"].replace("", "(none)").value_counts().to_dict()

    lines: list[str] = [
        "# Theme Overlay Rank Lift Report",
        "",
        f"- theme_applied_stock_count={theme_count}",
        f"- average_rank_change={avg_rank_change:.2f}",
        f"- median_rank_change={median_rank_change:.2f}",
        "",
        "## Top 10 Rank Lifters",
    ]
    for row in top10.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, new_rank={int(row.new_rank)}, "
            f"rank_change={int(row.rank_change)}, score_diff={float(row.score_diff):.4f}, "
            f"theme={row.dominant_theme}, theme_score={float(row.theme_score):.2f}, "
            f"theme_confidence={float(row.theme_confidence):.3f}"
        )

    lines.extend([
        "",
        "## Theme Average Rank Change",
    ])
    for row in theme_summary.itertuples(index=False):
        lines.append(
            f"- {row.dominant_theme}: stock_count={int(row.stock_count)}, avg_rank_change={float(row.avg_rank_change):.2f}"
        )

    lines.extend([
        "",
        "## Near Top20 Movers",
    ])
    if near_top20.empty:
        lines.append("- No theme-applied stock moved from base rank 21-40 into the top-30 zone.")
    else:
        for row in near_top20.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, new_rank={int(row.new_rank)}, "
                f"rank_change={int(row.rank_change)}, theme={row.dominant_theme}"
            )

    lines.extend([
        "",
        "## Top20 Non-Entry Interpretation",
        f"- top20_theme_distribution_v3={top20_theme_dist}",
    ])
    if top20_v3["dominant_theme"].replace("", "(none)").eq("(none)").all():
        lines.append("- Top20 is still dominated by non-theme names. Current overlay strength is enough to lift mid ranks, but not enough to overcome the base score gap to the top tier.")
    else:
        lines.append("- Some themed names entered top20. Review whether the lift is concentrated in one theme before increasing weights further.")

    lines.extend([
        "",
        "## Next Action",
        "- Expand stock_theme_map coverage before increasing theme overlay weights. Coverage is the limiting factor, not just score strength.",
    ])
    return "\n".join(lines)


def export_report(report_df: pd.DataFrame, latest: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_MD.write_text(build_markdown(report_df, latest), encoding="utf-8")
    LOGGER.info("Saved theme overlay rank lift CSV: %s rows=%d", OUTPUT_CSV.resolve(), len(report_df))
    LOGGER.info("Saved theme overlay rank lift markdown: %s", OUTPUT_MD.resolve())


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    report_df = build_report_df(latest)
    export_report(report_df, latest)
    print(f"generated_files={[str(OUTPUT_CSV), str(OUTPUT_MD)]}")
    print("example=python python\\build_theme_overlay_rank_lift_report.py")


if __name__ == "__main__":
    main()
