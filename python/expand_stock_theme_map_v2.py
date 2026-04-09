import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
REPORT_MD = DATA_DIR / "stock_theme_map_expand_report_v2.md"

LOGGER = logging.getLogger("expand_stock_theme_map_v2")

NEW_MAPPINGS = [
    {
        "code": "011200",
        "theme_id": "SHIP",
        "theme_name": "조선",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.60,
        "theme_weight": 0.60,
        "source_note": "Container shipping cycle and vessel supply-demand move with shipbuilding sentiment.",
    },
    {
        "code": "226950",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 1.00,
        "theme_weight": 1.00,
        "source_note": "Oligonucleotide therapeutics developer with direct biotech identity.",
    },
    {
        "code": "140410",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.95,
        "theme_weight": 0.95,
        "source_note": "Clinical-stage biotech driven by core pipeline outcomes.",
    },
    {
        "code": "058470",
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.65,
        "theme_weight": 0.65,
        "source_note": "Probe and test sockets are not pure capex tools but trade with semiconductor equipment demand.",
    },
    {
        "code": "424870",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.95,
        "theme_weight": 0.95,
        "source_note": "Immuno-oncology pipeline name with direct biotech exposure.",
    },
    {
        "code": "036830",
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "theme_role": "primary",
        "match_type": "proxy",
        "mapping_weight": 0.45,
        "theme_weight": 0.45,
        "source_note": "Holdco proxy to semiconductor materials and process-equipment subsidiaries.",
    },
    {
        "code": "005490",
        "theme_id": "BATTERY",
        "theme_name": "2차전지",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.65,
        "theme_weight": 0.65,
        "source_note": "Lithium and battery-material investment exposure via group-level resource chain.",
    },
    {
        "code": "287840",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "ADC payload and biotech platform exposure is direct.",
    },
    {
        "code": "0126Z0",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Biopharma platform holding company anchored by biosimilar and biotech operations.",
    },
    {
        "code": "115180",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Clinical biotech with direct pipeline sensitivity.",
    },
    {
        "code": "030530",
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "theme_role": "primary",
        "match_type": "proxy",
        "mapping_weight": 0.50,
        "theme_weight": 0.50,
        "source_note": "Holdco proxy to semiconductor equipment and materials affiliates.",
    },
    {
        "code": "005935",
        "theme_id": "HBM",
        "theme_name": "HBM",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 1.00,
        "theme_weight": 1.00,
        "source_note": "Preferred share of Samsung Electronics with direct HBM/memory exposure.",
    },
    {
        "code": "267250",
        "theme_id": "SHIP",
        "theme_name": "조선",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.60,
        "theme_weight": 0.60,
        "source_note": "Group holding exposure to shipbuilding and marine equipment cycle.",
    },
    {
        "code": "214450",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.70,
        "theme_weight": 0.70,
        "source_note": "Healthcare and regenerative medicine franchise repeatedly trades with bio theme.",
    },
    {
        "code": "000100",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 1.00,
        "theme_weight": 1.00,
        "source_note": "Large-cap pharma name with direct drug pipeline and biotech linkage.",
    },
    {
        "code": "007390",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Stem-cell and regenerative medicine exposure is directly tied to bio-theme moves.",
    },
    {
        "code": "222800",
        "theme_id": "AIPCB",
        "theme_name": "AI서버기판",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 1.00,
        "theme_weight": 1.00,
        "source_note": "High-end package substrate and PCB supplier directly exposed to AI server board demand.",
    },
    {
        "code": "445680",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.85,
        "theme_weight": 0.85,
        "source_note": "Cell analysis and biotech instrumentation platform with direct sector exposure.",
    },
    {
        "code": "039200",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.95,
        "theme_weight": 0.95,
        "source_note": "Targeted oncology pipeline name with direct biotech identity.",
    },
    {
        "code": "468530",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.85,
        "theme_weight": 0.85,
        "source_note": "Protein analysis biotech platform with direct life-science exposure.",
    },
    {
        "code": "005930",
        "theme_id": "HBM",
        "theme_name": "HBM",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 1.00,
        "theme_weight": 1.00,
        "source_note": "Samsung Electronics is a direct HBM and high-bandwidth memory supplier.",
    },
    {
        "code": "950160",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Tissue engineering and regenerative medicine story is directly tied to bio-theme sentiment.",
    },
    {
        "code": "195940",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.75,
        "theme_weight": 0.75,
        "source_note": "Pharma/healthcare franchise repeatedly grouped with large-cap bio names.",
    },
    {
        "code": "028300",
        "theme_id": "BIO",
        "theme_name": "바이오",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Clinical biotech and healthcare platform with direct bio-theme identity.",
    },
    {
        "code": "232140",
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "theme_role": "primary",
        "match_type": "direct",
        "mapping_weight": 0.90,
        "theme_weight": 0.90,
        "source_note": "Semiconductor equipment and test-process exposure is direct.",
    },
    {
        "code": "098460",
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "theme_role": "primary",
        "match_type": "indirect",
        "mapping_weight": 0.55,
        "theme_weight": 0.55,
        "source_note": "Inspection and automation equipment trades with semiconductor equipment cycle.",
    },
    {
        "code": "000880",
        "theme_id": "DEFENSE",
        "theme_name": "방산",
        "theme_role": "primary",
        "match_type": "proxy",
        "mapping_weight": 0.45,
        "theme_weight": 0.45,
        "source_note": "Holdco proxy to Hanwha Aerospace and defense subsidiaries rather than direct pure-play.",
    },
]

REVIEW_CANDIDATES = [
    ("066570", "LG전자", "AI서버기판", "Consumer/IT hardware exposure is broad, but direct AI-server PCB linkage is weak."),
    ("204270", "제이앤티씨", "AI서버기판", "Precision component maker, but server-board directness is not clean enough."),
    ("065350", "신성델타테크", "2차전지", "Market theme reaction exists, but business directness is weaker than battery leaders."),
    ("486990", "노타", "AI서버기판", "AI software angle exists, but this theme is server PCB, not broad AI software."),
    ("028050", "삼성E&A", "전력설비", "Energy EPC exposure is real, but direct power-equipment identity is weak."),
    ("009830", "한화솔루션", "2차전지", "Chemical adjacency exists, but ranking priority is lower than direct battery/material names."),
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest_ranking() -> pd.DataFrame:
    ranking = pd.read_csv(RANKING_CSV, dtype={"code": str}, low_memory=False)
    ranking["date"] = ranking["date"].astype(str)
    latest_date = ranking["date"].max()
    latest = ranking.loc[ranking["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["rank_final"] = pd.to_numeric(latest["rank_final"], errors="coerce")
    latest["name"] = latest["name"].fillna("").astype(str)
    latest["sector"] = latest.get("sector", "").fillna("").astype(str)
    latest["market"] = latest.get("market", "").fillna("").astype(str)
    latest = latest.sort_values("rank_final").reset_index(drop=True)
    return latest


def load_map() -> pd.DataFrame:
    df = pd.read_csv(STOCK_THEME_MAP_CSV, dtype={"code": str, "theme_id": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["theme_role", "match_type", "theme_weight", "source_note", "mapping_source"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["theme_weight"] = pd.to_numeric(df["theme_weight"], errors="coerce")
    df["mapping_weight"] = pd.to_numeric(df["mapping_weight"], errors="coerce")
    return df


def build_additions(latest: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    latest_name = latest.set_index("code")["name"].to_dict()
    existing_keys = set(zip(existing["code"], existing["theme_id"].astype(str).str.upper()))
    existing_counts = existing.groupby("code")["theme_id"].nunique().to_dict()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for item in NEW_MAPPINGS:
        code = str(item["code"]).zfill(6)
        theme_id = str(item["theme_id"]).upper()
        if (code, theme_id) in existing_keys:
            continue
        if existing_counts.get(code, 0) >= 3:
            LOGGER.warning("skip code=%s theme_id=%s because max theme count already reached", code, theme_id)
            continue
        row = dict(item)
        row["code"] = code
        row["name"] = latest_name.get(code, "")
        row["theme_id"] = theme_id
        row["is_primary"] = row["theme_role"] == "primary"
        row["mapping_source"] = "manual_expand_v2"
        row["updated_at"] = now
        rows.append(row)
        existing_counts[code] = existing_counts.get(code, 0) + 1

    additions = pd.DataFrame(rows)
    return additions


def _coverage_stats(ranking: pd.DataFrame, codes: set[str], top_n: int | None = None) -> tuple[int, int]:
    sample = ranking.head(top_n) if top_n is not None else ranking
    total = int(len(sample))
    covered = int(sample["code"].isin(codes).sum())
    return covered, total


def merge_and_write(existing: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([existing, additions], ignore_index=True, sort=False)
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    merged["theme_id"] = merged["theme_id"].astype(str).str.upper()
    merged["mapping_weight"] = pd.to_numeric(merged["mapping_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["theme_weight"] = pd.to_numeric(merged["theme_weight"], errors="coerce").fillna(merged["mapping_weight"]).clip(0.0, 1.0)
    merged["is_primary"] = merged["is_primary"].fillna(False).astype(bool)
    merged["theme_role"] = merged["theme_role"].fillna("").astype(str)
    merged["match_type"] = merged["match_type"].fillna("").astype(str)
    merged["source_note"] = merged["source_note"].fillna("").astype(str)
    merged["mapping_source"] = merged["mapping_source"].fillna("manual_seed_v1").astype(str)
    merged["updated_at"] = merged["updated_at"].fillna("")
    merged = merged.drop_duplicates(subset=["code", "theme_id"], keep="last")
    merged = merged.sort_values(
        ["code", "is_primary", "theme_weight", "mapping_weight", "theme_id"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    merged.to_csv(STOCK_THEME_MAP_CSV, index=False, encoding="utf-8-sig")
    return merged


def build_report(
    latest: pd.DataFrame,
    before_map: pd.DataFrame,
    after_map: pd.DataFrame,
    additions: pd.DataFrame,
    actual_after_filled_count: int | None,
) -> None:
    before_codes = set(before_map["code"].astype(str).str.zfill(6).unique())
    after_codes = set(after_map["code"].astype(str).str.zfill(6).unique())

    before_cover, total_latest = _coverage_stats(latest, before_codes)
    after_cover, _ = _coverage_stats(latest, after_codes)
    before_top100_cover, total_top100 = _coverage_stats(latest, before_codes, top_n=100)
    after_top100_cover, _ = _coverage_stats(latest, after_codes, top_n=100)

    theme_add_counts = additions.groupby("theme_name")["code"].nunique().sort_values(ascending=False)
    match_dist = additions["match_type"].value_counts().to_dict() if not additions.empty else {}
    role_dist = additions["theme_role"].value_counts().to_dict() if not additions.empty else {}

    newly_covered_latest = latest.loc[
        latest["code"].isin(after_codes - before_codes),
        ["rank_final", "code", "name", "sector"],
    ].sort_values("rank_final")
    near_top20 = newly_covered_latest.loc[
        newly_covered_latest["rank_final"].between(21, 40, inclusive="both")
    ]

    lines = [
        "# Stock Theme Map Expansion Report V2",
        "",
        "## Summary",
        f"- total_mapping_rows_before={len(before_map)}",
        f"- total_mapping_rows_after={len(after_map)}",
        f"- unique_codes_before={before_map['code'].nunique()}",
        f"- unique_codes_after={after_map['code'].nunique()}",
        f"- latest_ranking_theme_covered_before={before_cover}",
        f"- latest_ranking_theme_covered_after={after_cover}",
    ]
    if actual_after_filled_count is not None:
        lines.append(f"- latest_ranking_theme_filled_after_refresh={actual_after_filled_count}")
    lines.extend(
        [
            f"- latest_top100_coverage_before={before_top100_cover}/{total_top100}",
            f"- latest_top100_coverage_after={after_top100_cover}/{total_top100}",
            "",
            "## Added / Changed",
            f"- added_or_updated_rows={len(additions)}",
            f"- newly_covered_latest_ranking_codes={len(after_codes - before_codes & set(latest['code']))}",
            f"- top100_coverage_delta={after_top100_cover - before_top100_cover}",
            "",
            "## Added Stocks By Theme",
        ]
    )
    if theme_add_counts.empty:
        lines.append("- no additions")
    else:
        for theme_name, count in theme_add_counts.items():
            lines.append(f"- {theme_name}: +{int(count)}")

    lines.extend(["", "## Match Type Distribution"])
    for key, value in match_dist.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Primary / Secondary Distribution"])
    for key, value in role_dist.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Newly Covered Near Top20 Candidates"])
    if near_top20.empty:
        lines.append("- none")
    else:
        for row in near_top20.itertuples(index=False):
            lines.append(f"- rank={int(row.rank_final)} {row.code} {row.name} / {row.sector}")

    lines.extend(["", "## Review Candidates"])
    for code, name, theme_name, reason in REVIEW_CANDIDATES:
        lines.append(f"- {code} {name} / candidate_theme={theme_name} / excluded_reason={reason}")

    lines.extend(
        [
            "",
            "## Next Commands",
            "- python python/build_stock_theme_daily.py",
            "- python python/ranking_builder.py",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    before_map = load_map()
    additions = build_additions(latest, before_map)
    after_map = merge_and_write(before_map, additions)
    build_report(latest, before_map, after_map, additions, actual_after_filled_count=None)
    LOGGER.info("updated stock theme map rows_before=%d rows_after=%d", len(before_map), len(after_map))
    print(f"added_or_updated_rows={len(additions)}")
    print(f"new_unique_codes={int(additions['code'].nunique()) if not additions.empty else 0}")
    print(f"generated_files={[str(STOCK_THEME_MAP_CSV), str(REPORT_MD)]}")
    print("example=python python\\expand_stock_theme_map_v2.py")


if __name__ == "__main__":
    main()
