import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

RANKING_CSV = DATA_DIR / "ranking_final.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
STOCK_THEME_DAILY_CSV = OUTPUT_DIR / "stock_theme_daily.csv"

REPORT_CSV = DATA_DIR / "theme_overlay_join_debug_report.csv"
REPORT_MD = DATA_DIR / "theme_overlay_join_debug_report.md"

LOGGER = logging.getLogger("build_theme_overlay_join_debug_report")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _to_bool_theme(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def main() -> None:
    setup_logging()

    rank = pd.read_csv(RANKING_CSV, low_memory=False)
    rank["date"] = rank["date"].astype(str)
    rank["code"] = rank["code"].astype(str).str.zfill(6)
    latest_date = rank["date"].max()
    latest_rank = rank.loc[rank["date"] == latest_date].copy()
    latest_rank["dominant_theme"] = latest_rank.get("dominant_theme", "").fillna("").astype(str)
    latest_rank["theme_score"] = pd.to_numeric(latest_rank.get("theme_score"), errors="coerce").fillna(0.0)
    latest_rank["theme_confidence"] = pd.to_numeric(latest_rank.get("theme_confidence"), errors="coerce").fillna(0.0)

    std = pd.read_csv(STOCK_THEME_DAILY_CSV, dtype={"code": str})
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

    merged["has_mapping"] = merged["code"].isin(mapped_codes)
    merged["in_stock_theme_daily"] = merged["code"].isin(latest_std_codes)
    merged["ranking_has_theme"] = _to_bool_theme(merged["dominant_theme_ranking"])
    merged["stock_theme_daily_has_theme"] = _to_bool_theme(merged["dominant_theme_stock_theme_daily"])
    merged["join_gap_flag"] = (
        merged["in_stock_theme_daily"]
        & merged["stock_theme_daily_has_theme"]
        & ~merged["ranking_has_theme"]
    )

    gap_df = merged.loc[merged["join_gap_flag"], [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "rank_final",
        "dominant_theme_ranking",
        "theme_score_ranking",
        "theme_confidence_ranking",
        "dominant_theme_stock_theme_daily",
        "theme_score_stock_theme_daily",
        "theme_confidence_stock_theme_daily",
    ]].sort_values("rank_final")
    gap_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# Theme Overlay Join Debug Report",
        "",
        f"- latest_date={latest_date}",
        f"- latest_ranking_rows={len(latest_rank)}",
        f"- latest_stock_theme_daily_rows={len(latest_std)}",
        f"- mapped_code_count={len(mapped_codes)}",
        f"- ranking_and_stock_theme_overlap={len(overlap_codes)}",
        f"- ranking_theme_filled_count={int(_to_bool_theme(latest_rank['dominant_theme']).sum())}",
        f"- join_gap_count={len(gap_df)}",
        "",
        "## Interpretation",
        "- `join_gap_count > 0` means the latest ranking output still missed theme data even though the latest stock_theme_daily had a matching date/code row.",
        "- If this count falls after re-running ranking_builder sequentially, the root cause was stale read timing rather than mapping coverage.",
        "",
        "## Join Gap Sample",
    ]
    if gap_df.empty:
        lines.append("- No join gap detected on the latest date.")
    else:
        for row in gap_df.head(30).itertuples(index=False):
            lines.append(
                f"- rank={int(row.rank_final)} {row.code} {row.name}: "
                f"ranking_theme={row.dominant_theme_ranking or '(blank)'} / "
                f"stock_theme_daily_theme={row.dominant_theme_stock_theme_daily or '(blank)'} / "
                f"stock_theme_score={float(row.theme_score_stock_theme_daily):.2f}"
            )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved theme overlay join debug CSV: %s rows=%d", REPORT_CSV.resolve(), len(gap_df))
    LOGGER.info("Saved theme overlay join debug markdown: %s", REPORT_MD.resolve())
    print(f"generated_files={[str(REPORT_CSV), str(REPORT_MD)]}")
    print("example=python python\\build_theme_overlay_join_debug_report.py")


if __name__ == "__main__":
    main()
