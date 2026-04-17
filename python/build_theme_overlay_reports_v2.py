import logging
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

RANKING_CSV = DATA_DIR / "ranking_final.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
STOCK_THEME_DAILY_OUTPUT = OUTPUT_DIR / "stock_theme_daily.csv"
STOCK_THEME_DAILY_SUMMARY_OUTPUT = OUTPUT_DIR / "stock_theme_daily_summary.csv"

JOIN_DEBUG_MD_V2 = DATA_DIR / "theme_overlay_join_debug_report_v2.md"
RANK_LIFT_CSV_V2 = DATA_DIR / "theme_overlay_rank_lift_report_v2.csv"
RANK_LIFT_MD_V2 = DATA_DIR / "theme_overlay_rank_lift_report_v2.md"

LOGGER = logging.getLogger("build_theme_overlay_reports_v2")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sync_data_outputs() -> None:
    copies = [
        (STOCK_THEME_DAILY_OUTPUT, DATA_DIR / "stock_theme_daily.csv"),
        (STOCK_THEME_DAILY_SUMMARY_OUTPUT, DATA_DIR / "stock_theme_daily_summary.csv"),
        (OUTPUT_DIR / "theme_etf_latest_rank.csv", DATA_DIR / "theme_etf_latest_rank.csv"),
        (OUTPUT_DIR / "theme_etf_validation.md", DATA_DIR / "theme_etf_validation.md"),
        (OUTPUT_DIR / "theme_factor_contribution_check.csv", DATA_DIR / "theme_factor_contribution_check.csv"),
        (OUTPUT_DIR / "theme_factor_contribution_check.md", DATA_DIR / "theme_factor_contribution_check.md"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            LOGGER.info("synced %s -> %s", src, dst)


def _to_bool_theme(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def build_join_debug_v2() -> dict:
    rank = pd.read_csv(RANKING_CSV, dtype={"code": str}, low_memory=False)
    rank["date"] = rank["date"].astype(str)
    rank["code"] = rank["code"].astype(str).str.zfill(6)
    latest_date = rank["date"].max()
    latest_rank = rank.loc[rank["date"] == latest_date].copy()
    latest_rank["dominant_theme"] = latest_rank.get("dominant_theme", "").fillna("").astype(str)

    std = pd.read_csv(DATA_DIR / "stock_theme_daily.csv", dtype={"code": str})
    std["date"] = std["date"].astype(str)
    std["code"] = std["code"].astype(str).str.zfill(6)
    latest_std = std.loc[std["date"] == latest_date].copy()
    latest_std["dominant_theme"] = latest_std["dominant_theme"].fillna("").astype(str)

    stm = pd.read_csv(STOCK_THEME_MAP_CSV, dtype={"code": str})
    stm["code"] = stm["code"].astype(str).str.zfill(6)
    mapped_codes = set(stm["code"])
    latest_std_codes = set(latest_std["code"])
    latest_rank_codes = set(latest_rank["code"])
    overlap_codes = latest_rank_codes & latest_std_codes

    merged = latest_rank.merge(
        latest_std.loc[:, ["date", "code", "dominant_theme", "theme_score", "theme_confidence"]],
        on=["date", "code"],
        how="left",
        suffixes=("_ranking", "_stock_theme_daily"),
    )
    merged["in_stock_theme_daily"] = merged["code"].isin(latest_std_codes)
    merged["ranking_has_theme"] = _to_bool_theme(merged["dominant_theme_ranking"])
    merged["stock_theme_daily_has_theme"] = _to_bool_theme(merged["dominant_theme_stock_theme_daily"])
    merged["join_gap_flag"] = (
        merged["in_stock_theme_daily"] & merged["stock_theme_daily_has_theme"] & ~merged["ranking_has_theme"]
    )
    gap_count = int(merged["join_gap_flag"].sum())
    ranking_theme_filled_count = int(_to_bool_theme(latest_rank["dominant_theme"]).sum())

    lines = [
        "# Theme Overlay Join Debug Report V2",
        "",
        f"- latest_date={latest_date}",
        f"- latest_ranking_rows={len(latest_rank)}",
        f"- latest_stock_theme_daily_rows={len(latest_std)}",
        f"- mapped_code_count={len(mapped_codes)}",
        f"- ranking_and_stock_theme_overlap={len(overlap_codes)}",
        f"- ranking_theme_filled_count={ranking_theme_filled_count}",
        f"- join_gap_count={gap_count}",
        "",
        "## Interpretation",
        "- `join_gap_count=0` means the current bottleneck is coverage or score strength, not date/code join failure.",
        "- `ranking_theme_filled_count` is the actual latest ranking impact count after the refreshed pipeline run.",
    ]
    if gap_count:
        lines.extend(["", "## Join Gaps"])
        for row in merged.loc[merged["join_gap_flag"], ["rank_final", "code", "name", "dominant_theme_stock_theme_daily"]].sort_values("rank_final").head(30).itertuples(index=False):
            lines.append(f"- rank={int(row.rank_final)} {row.code} {row.name} / stock_theme_daily_theme={row.dominant_theme_stock_theme_daily}")
    else:
        lines.extend(["", "## Join Gaps", "- none"])

    JOIN_DEBUG_MD_V2.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("saved %s", JOIN_DEBUG_MD_V2)
    return {
        "latest_date": latest_date,
        "latest_ranking_rows": int(len(latest_rank)),
        "latest_stock_theme_daily_rows": int(len(latest_std)),
        "mapped_code_count": int(len(mapped_codes)),
        "ranking_and_stock_theme_overlap": int(len(overlap_codes)),
        "ranking_theme_filled_count": ranking_theme_filled_count,
        "join_gap_count": gap_count,
    }


def build_rank_lift_v2(before_theme_filled_count: int = 65) -> dict:
    latest = pd.read_csv(RANKING_CSV, low_memory=False, dtype={"code": str})
    latest["date"] = latest["date"].astype(str)
    latest = latest.loc[latest["date"] == latest["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["name"] = latest["name"].fillna("").astype(str)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    latest["explain"] = latest.get("explain_text", "").fillna("").astype(str)

    for col in ["final_score", "final_score_v3", "theme_score", "theme_confidence"]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce").fillna(0.0)

    latest["base_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)
    latest["new_rank"] = latest["final_score_v3"].rank(method="first", ascending=False).astype(int)
    latest["rank_change"] = latest["base_rank"] - latest["new_rank"]
    latest["score_diff"] = latest["final_score_v3"] - latest["final_score"]

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
    ]].sort_values(["rank_change", "score_diff", "theme_score", "code"], ascending=[False, False, False, True]).reset_index(drop=True)
    themed.to_csv(RANK_LIFT_CSV_V2, index=False, encoding="utf-8-sig")

    theme_summary = (
        themed.assign(dominant_theme=themed["dominant_theme"].replace("", "(none)"))
        .groupby("dominant_theme", as_index=False)
        .agg(stock_count=("code", "count"), avg_rank_change=("rank_change", "mean"))
        .sort_values(["avg_rank_change", "stock_count", "dominant_theme"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    top10 = themed.head(10)
    near_top20 = themed.loc[(themed["base_rank"] >= 21) & (themed["base_rank"] <= 40) & (themed["rank_change"] > 0)].copy()
    top20_v3 = latest.loc[latest["new_rank"] <= 20].copy()
    top20_theme_dist = top20_v3["dominant_theme"].replace("", "(none)").value_counts().to_dict()

    applied_count = int(len(themed))
    avg_rank_change = float(themed["rank_change"].mean()) if applied_count else 0.0
    median_rank_change = float(themed["rank_change"].median()) if applied_count else 0.0
    current_top20_theme_count = int(top20_v3["dominant_theme"].fillna("").astype(str).str.strip().ne("").sum())
    previous_top20_theme_count = 3
    previous_top100_coverage = 32
    previous_coverage_ratio = before_theme_filled_count / 199.0
    after_coverage_ratio = applied_count / 199.0

    winner_themes = theme_summary.loc[theme_summary["avg_rank_change"] > 10, "dominant_theme"].tolist()
    loser_themes = theme_summary.loc[theme_summary["avg_rank_change"] < -5, "dominant_theme"].tolist()

    lines = [
        "# Theme Overlay Rank Lift Report V2",
        "",
        f"- theme_applied_stock_count={applied_count}",
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

    lines.extend(["", "## Theme Average Rank Change"])
    for row in theme_summary.itertuples(index=False):
        lines.append(f"- {row.dominant_theme}: stock_count={int(row.stock_count)}, avg_rank_change={float(row.avg_rank_change):.2f}")

    lines.extend(["", "## Near Top20 Movers"])
    if near_top20.empty:
        lines.append("- none")
    else:
        for row in near_top20.sort_values(["new_rank", "rank_change"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, new_rank={int(row.new_rank)}, "
                f"rank_change={int(row.rank_change)}, theme={row.dominant_theme}"
            )

    lines.extend(["", "## Top20 Theme Distribution", f"- top20_theme_distribution_v3={top20_theme_dist}"])

    lines.extend([
        "",
        "## Before / After Comparison",
        f"- ranking_theme_filled_count: {before_theme_filled_count} -> {applied_count}",
        f"- latest_ranking_theme_coverage_ratio: {previous_coverage_ratio:.3f} -> {after_coverage_ratio:.3f}",
        f"- top100_coverage: {previous_top100_coverage}/100 -> {int((latest['dominant_theme'].fillna('').astype(str).str.strip().ne('')).head(100).sum())}/100",
        f"- top20_theme_applied_count: {previous_top20_theme_count} -> {current_top20_theme_count}",
        f"- near_top20_mover_count={len(near_top20)}",
        f"- winner_themes={winner_themes}",
        f"- loser_themes={loser_themes}",
        f"- new_top20_entries_with_theme={max(current_top20_theme_count - previous_top20_theme_count, 0)}",
    ])

    lines.extend([
        "",
        "## Next Action",
        "- theme_weight 추가 상향보다 먼저 상위권 비테마 종목의 base score gap을 줄일 수 있는지 점검하십시오.",
    ])

    RANK_LIFT_MD_V2.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("saved %s and %s", RANK_LIFT_CSV_V2, RANK_LIFT_MD_V2)
    return {
        "theme_applied_stock_count": applied_count,
        "average_rank_change": avg_rank_change,
        "median_rank_change": median_rank_change,
        "top20_theme_applied_count": current_top20_theme_count,
        "near_top20_mover_count": int(len(near_top20)),
    }


def main() -> None:
    setup_logging()
    sync_data_outputs()
    join_stats = build_join_debug_v2()
    rank_stats = build_rank_lift_v2()
    print(f"generated_files={[str(DATA_DIR / 'stock_theme_daily.csv'), str(DATA_DIR / 'stock_theme_daily_summary.csv'), str(JOIN_DEBUG_MD_V2), str(RANK_LIFT_CSV_V2), str(RANK_LIFT_MD_V2)]}")
    print(f"join_stats={join_stats}")
    print(f"rank_stats={rank_stats}")
    print("example=python python\\build_theme_overlay_reports_v2.py")


if __name__ == "__main__":
    main()
