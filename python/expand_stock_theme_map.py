import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RANKING_CSV = DATA_DIR / "ranking_final.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
REPORT_MD = DATA_DIR / "stock_theme_map_expand_report.md"

LOGGER = logging.getLogger("expand_stock_theme_map")

NEW_MAPPINGS = [
    {"code": "089030", "theme_id": "HBM", "theme_name": "HBM", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.65, "theme_weight": 0.65, "source_note": "HBM test/socket and advanced packaging exposure.", "is_primary": False},
    {"code": "089030", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "primary", "match_type": "indirect", "mapping_weight": 0.60, "theme_weight": 0.60, "source_note": "Semiconductor test equipment vendor with repeated equipment-theme reaction.", "is_primary": True},
    {"code": "140860", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 1.00, "theme_weight": 1.00, "source_note": "Metrology/inspection equipment directly tied to semiconductor capex.", "is_primary": True},
    {"code": "036930", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 1.00, "theme_weight": 1.00, "source_note": "Deposition/display equipment with direct semiconductor equipment exposure.", "is_primary": True},
    {"code": "056190", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.60, "theme_weight": 0.60, "source_note": "Factory automation beneficiary tied to secondary battery line investment.", "is_primary": False},
    {"code": "056190", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "secondary", "match_type": "proxy", "mapping_weight": 0.45, "theme_weight": 0.45, "source_note": "Automation/platform proxy for equipment spending cycle.", "is_primary": False},
    {"code": "034020", "theme_id": "POWER", "theme_name": "전력설비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.85, "theme_weight": 0.85, "source_note": "Generation and power infrastructure leader with repeated grid-cycle linkage.", "is_primary": True},
    {"code": "085660", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 1.00, "theme_weight": 1.00, "source_note": "Core biotech platform/operator recognized as direct bio name.", "is_primary": True},
    {"code": "389470", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.80, "theme_weight": 0.80, "source_note": "Drug delivery platform with direct biotech identity.", "is_primary": True},
    {"code": "067310", "theme_id": "HBM", "theme_name": "HBM", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.85, "theme_weight": 0.85, "source_note": "Memory package/test OSAT exposed to HBM supply chain.", "is_primary": True},
    {"code": "067310", "theme_id": "AIPCB", "theme_name": "AI서버기판", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.55, "theme_weight": 0.55, "source_note": "Advanced package substrate adjacency through server package demand.", "is_primary": False},
    {"code": "443060", "theme_id": "SHIP", "theme_name": "조선", "theme_role": "primary", "match_type": "indirect", "mapping_weight": 0.65, "theme_weight": 0.65, "source_note": "Marine lifecycle/services beneficiary that moves with shipbuilding cycle.", "is_primary": True},
    {"code": "086450", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.55, "theme_weight": 0.55, "source_note": "Healthcare/pharma franchise with repeated bio-theme market reaction.", "is_primary": False},
    {"code": "086900", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.90, "theme_weight": 0.90, "source_note": "Direct biotech/biopharma name with product-driven theme sensitivity.", "is_primary": True},
    {"code": "237690", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.95, "theme_weight": 0.95, "source_note": "CDMO/API growth name directly tied to biotech manufacturing demand.", "is_primary": True},
    {"code": "060370", "theme_id": "POWER", "theme_name": "전력설비", "theme_role": "primary", "match_type": "indirect", "mapping_weight": 0.65, "theme_weight": 0.65, "source_note": "Subsea cable and grid connection beneficiary tied to power infra cycle.", "is_primary": True},
    {"code": "348370", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.95, "theme_weight": 0.95, "source_note": "Electrolyte leader with direct secondary battery demand linkage.", "is_primary": True},
    {"code": "015760", "theme_id": "POWER", "theme_name": "전력설비", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.55, "theme_weight": 0.55, "source_note": "Utility policy and capex proxy for the grid investment cycle.", "is_primary": False},
    {"code": "161580", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.80, "theme_weight": 0.80, "source_note": "Equipment exposure to secondary battery and advanced process lines.", "is_primary": True},
    {"code": "128940", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.95, "theme_weight": 0.95, "source_note": "Major pharma/biotech franchise with direct clinical and pipeline sensitivity.", "is_primary": True},
    {"code": "137400", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.90, "theme_weight": 0.90, "source_note": "Winding and battery production equipment directly tied to cell capex.", "is_primary": True},
    {"code": "281740", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.65, "theme_weight": 0.65, "source_note": "Battery material/precursor adjacency with recurring market linkage.", "is_primary": False},
    {"code": "096770", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.70, "theme_weight": 0.70, "source_note": "Battery materials and cell ecosystem exposure through affiliates and capex cycle.", "is_primary": False},
    {"code": "403870", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.92, "theme_weight": 0.92, "source_note": "High-margin thermal process equipment directly tied to semiconductor fab investment.", "is_primary": True},
    {"code": "403870", "theme_id": "HBM", "theme_name": "HBM", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.60, "theme_weight": 0.60, "source_note": "Advanced memory process beneficiary through HBM-related fab intensity.", "is_primary": False},
    {"code": "183300", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.88, "theme_weight": 0.88, "source_note": "Semiconductor parts/coating service directly exposed to fab utilization and capex.", "is_primary": True},
    {"code": "083650", "theme_id": "POWER", "theme_name": "전력설비", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.80, "theme_weight": 0.80, "source_note": "Boiler/turbine and power EPC linkage with direct power equipment cycle sensitivity.", "is_primary": True},
    {"code": "086520", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.90, "theme_weight": 0.90, "source_note": "Core battery ecosystem name repeatedly repriced with secondary battery demand.", "is_primary": True},
    {"code": "000150", "theme_id": "POWER", "theme_name": "전력설비", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.60, "theme_weight": 0.60, "source_note": "Group-level exposure to power equipment and infra capex beneficiaries.", "is_primary": False},
    {"code": "000150", "theme_id": "DEFENSE", "theme_name": "방산", "theme_role": "secondary", "match_type": "proxy", "mapping_weight": 0.35, "theme_weight": 0.35, "source_note": "Holdco proxy to defense/aerospace subsidiaries, not a direct pure-play.", "is_primary": False},
    {"code": "096530", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.85, "theme_weight": 0.85, "source_note": "Diagnostics/biotech direct exposure recognized by the market as bio-theme.", "is_primary": True},
    {"code": "456160", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.88, "theme_weight": 0.88, "source_note": "Drug delivery biotech with direct therapeutic platform exposure.", "is_primary": True},
    {"code": "328130", "theme_id": "BIO", "theme_name": "바이오", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.60, "theme_weight": 0.60, "source_note": "AI-healthcare platform repeatedly trades with biotech/medical innovation theme.", "is_primary": False},
    {"code": "178320", "theme_id": "AIPCB", "theme_name": "AI서버기판", "theme_role": "primary", "match_type": "indirect", "mapping_weight": 0.75, "theme_weight": 0.75, "source_note": "Server/chassis/connectivity beneficiary repeatedly linked to AI server buildout.", "is_primary": True},
    {"code": "323280", "theme_id": "AIPCB", "theme_name": "AI서버기판", "theme_role": "primary", "match_type": "direct", "mapping_weight": 0.90, "theme_weight": 0.90, "source_note": "PCB/FCCL process exposure directly tied to high-end substrate demand.", "is_primary": True},
    {"code": "011070", "theme_id": "AIPCB", "theme_name": "AI서버기판", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.65, "theme_weight": 0.65, "source_note": "High-value component and server-board demand linkage through AI hardware mix.", "is_primary": False},
    {"code": "051910", "theme_id": "BATTERY", "theme_name": "2차전지", "theme_role": "secondary", "match_type": "indirect", "mapping_weight": 0.80, "theme_weight": 0.80, "source_note": "Battery materials and enterprise-level exposure keep it in recurring battery theme baskets.", "is_primary": False},
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest_ranking() -> pd.DataFrame:
    rank = pd.read_csv(RANKING_CSV, low_memory=False)
    rank["date"] = rank["date"].astype(str)
    latest = rank.loc[rank["date"] == rank["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["name"] = latest["name"].fillna("").astype(str)
    latest["rank_final"] = pd.to_numeric(latest.get("rank_final"), errors="coerce")
    return latest


def load_existing_map() -> pd.DataFrame:
    df = pd.read_csv(STOCK_THEME_MAP_CSV, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["theme_role", "match_type", "theme_weight", "source_note"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def build_additions(latest: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    latest_name_map = latest.set_index("code")["name"].to_dict()
    rows = []
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing_keys = set(zip(existing["code"], existing["theme_id"]))

    for item in NEW_MAPPINGS:
        code = str(item["code"]).zfill(6)
        theme_id = str(item["theme_id"]).upper()
        if (code, theme_id) in existing_keys:
            continue
        row = dict(item)
        row["code"] = code
        row["theme_id"] = theme_id
        row["name"] = latest_name_map.get(code, row.get("name", ""))
        row["mapping_source"] = "manual_expand_v1"
        row["updated_at"] = updated_at
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(existing: pd.DataFrame, additions: pd.DataFrame, latest: pd.DataFrame) -> None:
    merged = pd.concat([existing, additions], ignore_index=True, sort=False)
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    merged["theme_id"] = merged["theme_id"].astype(str).str.upper()
    merged["mapping_weight"] = pd.to_numeric(merged["mapping_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["theme_weight"] = pd.to_numeric(merged["theme_weight"], errors="coerce").fillna(merged["mapping_weight"]).clip(0.0, 1.0)
    merged["is_primary"] = merged["is_primary"].fillna(False)
    merged["theme_role"] = merged["theme_role"].fillna("")
    merged["match_type"] = merged["match_type"].fillna("")
    merged["source_note"] = merged["source_note"].fillna("")
    merged = merged.drop_duplicates(subset=["code", "theme_id"], keep="last")
    merged = merged.sort_values(["code", "is_primary", "mapping_weight", "theme_id"], ascending=[True, False, False, True])
    merged.to_csv(STOCK_THEME_MAP_CSV, index=False, encoding="utf-8-sig")

    added_codes = set(additions["code"].astype(str)) if not additions.empty else set()
    top100 = latest.sort_values("rank_final").head(100).copy()
    top100_new = top100.loc[top100["code"].isin(added_codes), ["code", "name", "rank_final"]].sort_values("rank_final")

    theme_counts = additions.groupby("theme_id")["code"].nunique().sort_values(ascending=False) if not additions.empty else pd.Series(dtype=int)
    match_dist = additions["match_type"].value_counts().to_dict() if not additions.empty else {}
    total_new_stocks = int(additions["code"].nunique()) if not additions.empty else 0

    lines = [
        "# Stock Theme Map Expansion Report",
        "",
        f"- added_stock_count={total_new_stocks}",
        f"- added_mapping_count={int(len(additions))}",
        f"- total_stock_count_after_expand={int(merged['code'].nunique())}",
        f"- top100_newly_covered_count={int(top100_new['code'].nunique())}",
        f"- match_type_distribution={match_dist}",
        "",
        "## Added Stocks By Theme",
    ]
    if theme_counts.empty:
        lines.append("- No new mappings were added.")
    else:
        for theme_id, count in theme_counts.items():
            lines.append(f"- {theme_id}: {int(count)}")

    lines.extend([
        "",
        "## Top-Ranked Newly Covered Stocks",
    ])
    if top100_new.empty:
        lines.append("- No newly covered stock appears in the latest top100 ranking slice.")
    else:
        for row in top100_new.itertuples(index=False):
            lines.append(f"- rank={int(row.rank_final)} {row.code} {row.name}")

    lines.extend([
        "",
        "## Theme Weight Integration",
        "- Current pipeline already uses `mapping_weight` in theme score aggregation.",
        "- `theme_weight` is now stored for schema expansion, but it is not consumed separately yet.",
        "- Next candidate task: switch stock theme merge logic to prefer `theme_weight` when present.",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("Updated stock theme map: %s rows=%d stocks=%d", STOCK_THEME_MAP_CSV.resolve(), len(merged), merged["code"].nunique())
    LOGGER.info("Saved expansion report: %s", REPORT_MD.resolve())


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    existing = load_existing_map()
    additions = build_additions(latest, existing)
    write_outputs(existing, additions, latest)
    print(f"added_stock_count={int(additions['code'].nunique()) if not additions.empty else 0}")
    print(f"added_mapping_count={int(len(additions))}")
    print(f"generated_files={[str(STOCK_THEME_MAP_CSV), str(REPORT_MD)]}")
    print("example=python python\\expand_stock_theme_map.py")


if __name__ == "__main__":
    main()
