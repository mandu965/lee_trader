from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_stock_theme_daily import build_stock_theme_daily, load_theme_etf_daily
from theme_mapping_utils import STOCK_THEME_MAP_COLUMNS, standardize_stock_theme_map


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
RANKING_CSV = DATA_DIR / "ranking_final.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
THEME_ETF_DAILY_CSV = OUTPUT_DIR / "theme_etf_daily.csv"
CURRENT_STOCK_THEME_DAILY_CSV = OUTPUT_DIR / "stock_theme_daily.csv"

EXPANDED_MAP_CSV = DATA_DIR / "stock_theme_map_expanded.csv"
TOP20_REVIEW_CSV = DATA_DIR / "theme_mapping_top20_review.csv"
GAP_REPORT_MD = DATA_DIR / "theme_mapping_gap_report.md"

ADDITIONAL_ROWS = [
    {
        "code": "358570",
        "name": "지아이이노베이션",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "mapping_source": "manual_expand_v3_gap_report",
        "source_note": "Immunology-focused biotech pipeline company; direct fit to existing bio theme.",
        "is_primary": True,
    },
    {
        "code": "097230",
        "name": "HJ중공업",
        "theme_id": "SHIP",
        "theme_name": "조선",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.85,
        "theme_weight": 0.85,
        "mapping_source": "manual_expand_v3_gap_report",
        "source_note": "Shipbuilding and naval-vessel exposure is direct enough for the current shipbuilding theme bucket.",
        "is_primary": True,
    },
    {
        "code": "298380",
        "name": "에이비엘바이오",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.95,
        "theme_weight": 0.95,
        "mapping_source": "manual_expand_v3_gap_report",
        "source_note": "Clinical-stage biotech with antibody pipeline; direct fit to bio theme.",
        "is_primary": True,
    },
]

REMOVAL_RULES = [
    {
        "code": "011200",
        "theme_id": "SHIP",
        "reason": "Shipping carrier is adjacent to shipping cycle but not a shipbuilding pure-play.",
    },
    {
        "code": "017890",
        "theme_id": "POWER",
        "reason": "Business description does not support a power-equipment mapping.",
    },
    {
        "code": "336200",
        "theme_id": "AIPCB",
        "reason": "Fuel-cell business does not fit the AI server substrate theme.",
    },
]

TOP40_ACTIONS = {
    "006730": {"action": "no_fit", "reason": "Current ETF taxonomy has no hotel/leisure theme proxy."},
    "030200": {"action": "no_fit", "reason": "Current ETF taxonomy has no telecom theme proxy."},
    "323410": {"action": "no_fit", "reason": "Platform banking is outside the current eight ETF themes."},
    "016360": {"action": "no_fit", "reason": "Brokerage/financials are outside the current eight ETF themes."},
    "293490": {"action": "no_fit", "reason": "Game content is outside the current eight ETF themes."},
    "358570": {"action": "add", "theme_id": "BIO", "reason": "Direct biotech pipeline exposure fits the bio theme."},
    "017670": {"action": "no_fit", "reason": "Current ETF taxonomy has no telecom theme proxy."},
    "011200": {"action": "remove", "theme_id": "SHIP", "reason": "Shipping line is not a direct shipbuilding theme constituent."},
    "032640": {"action": "no_fit", "reason": "Current ETF taxonomy has no telecom theme proxy."},
    "003490": {"action": "no_fit", "reason": "Current ETF taxonomy has no airline theme proxy."},
    "278470": {"action": "no_fit", "reason": "Beauty/cosmetics is outside the current eight ETF themes."},
    "138930": {"action": "no_fit", "reason": "Financial holding company is outside the current eight ETF themes."},
    "097230": {"action": "add", "theme_id": "SHIP", "reason": "Shipbuilding and naval-vessel exposure fits the ship theme."},
    "005385": {"action": "no_fit", "reason": "Preferred auto share should not be forced into battery theme without a direct business basis."},
    "005387": {"action": "no_fit", "reason": "Preferred auto share should not be forced into battery theme without a direct business basis."},
    "005940": {"action": "no_fit", "reason": "Brokerage/financials are outside the current eight ETF themes."},
    "039490": {"action": "no_fit", "reason": "Brokerage/financials are outside the current eight ETF themes."},
    "066570": {"action": "review_new_theme", "reason": "Electronics/AI hardware adjacency exists, but current AIPCB theme is too narrow to map broad consumer electronics."},
    "259960": {"action": "no_fit", "reason": "Game content is outside the current eight ETF themes."},
    "377300": {"action": "no_fit", "reason": "Fintech/platform payments are outside the current eight ETF themes."},
    "298380": {"action": "add", "theme_id": "BIO", "reason": "Clinical biotech pipeline exposure fits the bio theme directly."},
    "003550": {"action": "review_new_theme", "reason": "Holdco exposure spans multiple sectors; current eight ETF themes are too narrow for a clean mapping."},
    "486990": {"action": "review_new_theme", "reason": "AI software exposure is real, but the current ETF theme master has no standalone AI software proxy."},
    "035720": {"action": "no_fit", "reason": "Platform/internet exposure is outside the current eight ETF themes."},
    "180640": {"action": "review_new_theme", "reason": "Holding-company exposure is tied to airline/logistics rather than any current ETF theme."},
    "029780": {"action": "no_fit", "reason": "Card/consumer finance is outside the current eight ETF themes."},
}

MULTI_THEME_REVIEW = {
    "000150": "Review: holdco mapped to both power and defense; acceptable as a low-weight proxy but not a pure-play.",
    "056190": "Review: battery and semiconductor capex overlap exists, but both links are indirect.",
    "067310": "Review: HBM is the stronger primary link; AI PCB adjacency is secondary only.",
    "039030": "Keep: semiconductor equipment remains primary over HBM adjacency.",
    "042700": "Keep: HBM primary / semiconductor equipment secondary is explainable.",
    "089030": "Keep: HBM and semiconductor test-equipment overlap is explainable.",
    "240810": "Keep: equipment primary / HBM secondary remains explainable.",
    "319660": "Keep: equipment primary / HBM secondary remains explainable.",
    "403870": "Keep: equipment primary / HBM secondary remains explainable.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and expand stock-theme mappings around top20 / near-top20 names.")
    parser.add_argument("--ranking-csv", default=str(RANKING_CSV))
    parser.add_argument("--stock-theme-map", default=str(STOCK_THEME_MAP_CSV))
    parser.add_argument("--theme-etf-master", default=str(THEME_ETF_MASTER_CSV))
    parser.add_argument("--theme-etf-daily", default=str(THEME_ETF_DAILY_CSV))
    parser.add_argument("--current-stock-theme-daily", default=str(CURRENT_STOCK_THEME_DAILY_CSV))
    parser.add_argument("--expanded-map-csv", default=str(EXPANDED_MAP_CSV))
    parser.add_argument("--top20-review-csv", default=str(TOP20_REVIEW_CSV))
    parser.add_argument("--gap-report-md", default=str(GAP_REPORT_MD))
    return parser.parse_args()


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def load_latest_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["rank_final"] = pd.to_numeric(df["rank_final"], errors="coerce")
    latest_date = df["date"].dropna().max()
    latest = df[df["date"] == latest_date].copy()
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    latest["theme_score"] = pd.to_numeric(latest.get("theme_score"), errors="coerce").fillna(0.0)
    latest["theme_confidence"] = pd.to_numeric(latest.get("theme_confidence"), errors="coerce").fillna(0.0)
    return latest.sort_values(["rank_final", "code"]).reset_index(drop=True)


def summarize_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["code", "theme_ids", "theme_names", "theme_count"])
    summary = (
        df.groupby("code", dropna=False)
        .agg(
            name=("name", "first"),
            theme_ids=("theme_id", lambda s: "|".join(sorted(set(map(str, s))))),
            theme_names=("theme_name", lambda s: "|".join(sorted(set(map(str, s))))),
            theme_count=("theme_id", "nunique"),
        )
        .reset_index()
    )
    return summary


def apply_expansion(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = base_df.copy()
    removed_notes: list[str] = []
    added_notes: list[str] = []
    for rule in REMOVAL_RULES:
        mask = work["code"].eq(rule["code"]) & work["theme_id"].eq(rule["theme_id"])
        if mask.any():
            work = work.loc[~mask].copy()
            removed_notes.append(f"{rule['code']} {rule['theme_id']}: {rule['reason']}")
    additions = pd.DataFrame(ADDITIONAL_ROWS)
    additions["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not additions.empty:
        work = pd.concat([work, additions], ignore_index=True)
        for row in ADDITIONAL_ROWS:
            added_notes.append(f"{row['code']} {row['name']} -> {row['theme_id']}: {row['source_note']}")
    expanded = standardize_stock_theme_map(work)
    expanded = expanded.sort_values(["code", "theme_id"]).reset_index(drop=True)
    return expanded, removed_notes, added_notes


def load_current_theme_daily(path: Path) -> pd.DataFrame:
    df = _read_csv(path, dtype={"code": str})
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["theme_score"] = pd.to_numeric(df.get("theme_score"), errors="coerce").fillna(0.0)
    df["theme_confidence"] = pd.to_numeric(df.get("theme_confidence"), errors="coerce").fillna(0.0)
    latest_date = df["date"].dropna().max()
    return df[df["date"] == latest_date].copy()


def build_projected_theme_daily(expanded_df: pd.DataFrame, theme_etf_daily_path: Path, theme_etf_master_path: Path) -> pd.DataFrame:
    theme_etf_df = load_theme_etf_daily(theme_etf_daily_path)
    if theme_etf_df.empty:
        return pd.DataFrame()
    latest_date = theme_etf_df["date"].dropna().max()
    latest_theme_df = theme_etf_df[theme_etf_df["date"] == latest_date].copy()
    projected_daily_df, _ = build_stock_theme_daily(
        expanded_df,
        latest_theme_df,
        theme_etf_master_path=theme_etf_master_path,
    )
    return projected_daily_df


def _theme_presence(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def build_top40_review(latest_rank: pd.DataFrame, base_map: pd.DataFrame, expanded_map: pd.DataFrame) -> pd.DataFrame:
    top40 = latest_rank[latest_rank["rank_final"].between(1, 40)].copy()
    base_summary = summarize_map(base_map).rename(
        columns={
            "theme_ids": "current_theme_ids",
            "theme_names": "current_theme_names",
            "theme_count": "current_theme_count",
        }
    )
    expanded_summary = summarize_map(expanded_map).rename(
        columns={
            "theme_ids": "expanded_theme_ids",
            "theme_names": "expanded_theme_names",
            "theme_count": "expanded_theme_count",
        }
    )
    review = top40.merge(base_summary, on=["code"], how="left").merge(expanded_summary, on=["code"], how="left")
    review["bucket"] = review["rank_final"].apply(lambda value: "top20" if value <= 20 else "near_top20")
    review["current_theme_ids"] = review["current_theme_ids"].fillna("")
    review["current_theme_names"] = review["current_theme_names"].fillna("")
    review["expanded_theme_ids"] = review["expanded_theme_ids"].fillna("")
    review["expanded_theme_names"] = review["expanded_theme_names"].fillna("")
    review["current_theme_count"] = pd.to_numeric(review["current_theme_count"], errors="coerce").fillna(0).astype(int)
    review["expanded_theme_count"] = pd.to_numeric(review["expanded_theme_count"], errors="coerce").fillna(0).astype(int)

    actions: list[str] = []
    reasons: list[str] = []
    recommended_theme_ids: list[str] = []
    for row in review.itertuples(index=False):
        action_info = TOP40_ACTIONS.get(row.code)
        if action_info:
            actions.append(action_info["action"])
            reasons.append(action_info["reason"])
            recommended_theme_ids.append(str(action_info.get("theme_id", "")))
        elif row.current_theme_count > 0:
            actions.append("keep")
            reasons.append("Current mapping is aligned enough with the existing ETF theme taxonomy.")
            recommended_theme_ids.append(row.expanded_theme_ids)
        else:
            actions.append("no_fit")
            reasons.append("No clean fit within the current ETF theme taxonomy.")
            recommended_theme_ids.append("")
    review["review_action"] = actions
    review["review_reason"] = reasons
    review["recommended_theme_ids"] = recommended_theme_ids
    review["ranking_theme_none"] = ~_theme_presence(review["dominant_theme"])
    cols = [
        "bucket",
        "rank_final",
        "code",
        "name_x",
        "market",
        "sector",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "current_theme_ids",
        "current_theme_names",
        "current_theme_count",
        "expanded_theme_ids",
        "expanded_theme_names",
        "expanded_theme_count",
        "review_action",
        "recommended_theme_ids",
        "review_reason",
        "ranking_theme_none",
    ]
    out = review.loc[:, cols].rename(columns={"name_x": "name"})
    return out.sort_values(["rank_final", "code"]).reset_index(drop=True)


def build_gap_report(
    latest_rank: pd.DataFrame,
    base_map: pd.DataFrame,
    expanded_map: pd.DataFrame,
    current_theme_daily_latest: pd.DataFrame,
    projected_theme_daily_latest: pd.DataFrame,
    top40_review: pd.DataFrame,
    removed_notes: list[str],
    added_notes: list[str],
) -> str:
    latest_rank_date = latest_rank["date"].dropna().max()
    current_theme_date = current_theme_daily_latest["date"].dropna().max() if not current_theme_daily_latest.empty else "(missing)"
    projected_theme_date = projected_theme_daily_latest["date"].dropna().max() if not projected_theme_daily_latest.empty else "(missing)"
    ranking_none_ratio = 1.0 if latest_rank.empty else float((~_theme_presence(latest_rank["dominant_theme"])).mean())

    top40_codes = set(top40_review["code"])
    current_top40_theme = current_theme_daily_latest[current_theme_daily_latest["code"].isin(top40_codes)].copy()
    projected_top40_theme = projected_theme_daily_latest[projected_theme_daily_latest["code"].isin(top40_codes)].copy()
    top20_review = top40_review[top40_review["bucket"] == "top20"].copy()
    near_review = top40_review[top40_review["bucket"] == "near_top20"].copy()

    current_top20_themed = int(current_top40_theme[current_top40_theme["code"].isin(set(top20_review["code"]))]["dominant_theme"].astype(str).str.strip().ne("").sum())
    projected_top20_themed = int(projected_top40_theme[projected_top40_theme["code"].isin(set(top20_review["code"]))]["dominant_theme"].astype(str).str.strip().ne("").sum())
    current_near_themed = int(current_top40_theme[current_top40_theme["code"].isin(set(near_review["code"]))]["dominant_theme"].astype(str).str.strip().ne("").sum())
    projected_near_themed = int(projected_top40_theme[projected_top40_theme["code"].isin(set(near_review["code"]))]["dominant_theme"].astype(str).str.strip().ne("").sum())

    action_counts = top40_review["review_action"].value_counts().to_dict()
    no_fit_rows = top40_review[top40_review["review_action"].isin(["no_fit", "review_new_theme"])].copy()
    multi_theme = (
        expanded_map.groupby("code")
        .agg(name=("name", "first"), theme_count=("theme_id", "nunique"), themes=("theme_name", lambda s: "|".join(sorted(set(map(str, s))))))
        .reset_index()
    )
    multi_theme = multi_theme[multi_theme["theme_count"] > 1].sort_values(["theme_count", "code"], ascending=[False, True])
    theme_counts = (
        expanded_map.groupby(["theme_id", "theme_name"], dropna=False)["code"]
        .nunique()
        .reset_index(name="stock_count")
        .sort_values(["stock_count", "theme_id"], ascending=[False, True])
    )

    lines = [
        "# Theme Mapping Gap Report",
        "",
        "## Snapshot",
        f"- latest_ranking_date: {latest_rank_date}",
        f"- latest_stock_theme_daily_date: {current_theme_date}",
        f"- projected_theme_overlay_date: {projected_theme_date}",
        f"- ranking_final dominant_theme none_ratio: {ranking_none_ratio:.1%}",
        "- 해석: `ranking_final`의 `(none)` 100%는 매핑 누락만이 아니라 최신 ranking 일자와 theme overlay 일자 불일치도 함께 반영한다.",
        "",
        "## Mapping Structure",
        f"- base_map rows={len(base_map)}, unique_stocks={base_map['code'].nunique()}, multi_theme_stocks={int(base_map.groupby('code')['theme_id'].nunique().gt(1).sum())}",
        f"- expanded_map rows={len(expanded_map)}, unique_stocks={expanded_map['code'].nunique()}, multi_theme_stocks={int(expanded_map.groupby('code')['theme_id'].nunique().gt(1).sum())}",
        "- canonical ETF theme names: HBM, 반도체장비, AI서버기판, 전력설비, 방산, 2차전지, 조선, 바이오",
        "",
        "## Top40 Coverage",
        f"- current_top20_themed_on_available_theme_date: {current_top20_themed}/20",
        f"- projected_top20_themed_after_expansion: {projected_top20_themed}/20",
        f"- current_near_top20_themed_on_available_theme_date: {current_near_themed}/20",
        f"- projected_near_top20_themed_after_expansion: {projected_near_themed}/20",
        f"- top40 action_counts: {action_counts}",
        "",
        "## Added Mappings",
    ]
    for note in added_notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Removed / Cleaned Mappings"])
    for note in removed_notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Multi-theme Conflict Review"])
    for row in multi_theme.itertuples(index=False):
        comment = MULTI_THEME_REVIEW.get(row.code, "Review: multiple themes exist; primary/secondary ordering should be watched.")
        lines.append(f"- {row.code} {row.name}: themes={row.themes} ({comment})")
    lines.extend(["", "## No-fit / New-theme Candidates"])
    for row in no_fit_rows.itertuples(index=False):
        lines.append(f"- rank={int(row.rank_final)} {row.code} {row.name}: action={row.review_action}, reason={row.review_reason}")
    lines.extend(["", "## Theme Distribution After Expansion"])
    for row in theme_counts.itertuples(index=False):
        lines.append(f"- {row.theme_id} / {row.theme_name}: stock_count={int(row.stock_count)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ranking_path = Path(args.ranking_csv)
    stock_theme_map_path = Path(args.stock_theme_map)
    theme_etf_master_path = Path(args.theme_etf_master)
    theme_etf_daily_path = Path(args.theme_etf_daily)
    current_stock_theme_daily_path = Path(args.current_stock_theme_daily)
    expanded_map_csv = Path(args.expanded_map_csv)
    top20_review_csv = Path(args.top20_review_csv)
    gap_report_md = Path(args.gap_report_md)

    latest_rank = load_latest_ranking(ranking_path)
    base_map = standardize_stock_theme_map(pd.read_csv(stock_theme_map_path, dtype={"code": str, "theme_id": str}))
    expanded_map, removed_notes, added_notes = apply_expansion(base_map)
    current_theme_daily_latest = load_current_theme_daily(current_stock_theme_daily_path)
    projected_theme_daily_latest = build_projected_theme_daily(expanded_map, theme_etf_daily_path, theme_etf_master_path)
    top40_review = build_top40_review(latest_rank, base_map, expanded_map)
    report = build_gap_report(
        latest_rank,
        base_map,
        expanded_map,
        current_theme_daily_latest,
        projected_theme_daily_latest,
        top40_review,
        removed_notes,
        added_notes,
    )

    expanded_map_csv.parent.mkdir(parents=True, exist_ok=True)
    top20_review_csv.parent.mkdir(parents=True, exist_ok=True)
    gap_report_md.parent.mkdir(parents=True, exist_ok=True)
    expanded_map.to_csv(expanded_map_csv, index=False, encoding="utf-8-sig", columns=STOCK_THEME_MAP_COLUMNS)
    top40_review.to_csv(top20_review_csv, index=False, encoding="utf-8-sig")
    gap_report_md.write_text(report, encoding="utf-8")

    print(f"latest_ranking_date={latest_rank['date'].dropna().max()}")
    print(f"expanded_map_rows={len(expanded_map)}")
    print(f"top40_review_rows={len(top40_review)}")
    print(f"generated_files={[str(expanded_map_csv), str(top20_review_csv), str(gap_report_md)]}")


if __name__ == "__main__":
    main()
