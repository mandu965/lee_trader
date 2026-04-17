from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "theme_source_headroom.csv"
OUTPUT_MD = DATA_DIR / "theme_source_headroom_review.md"
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
THEME_ETF_DAILY_CSV = Path("output") / "theme_etf_daily.csv"
NEAR_TOP20_AUDIT_CSV = DATA_DIR / "theme_near_top20_mapping_audit.csv"


KNOWN_LOCAL_THEME_MAPPINGS = [
    {"theme_id": "BIO", "etf_code": "261070", "etf_name": "TIGER 코스닥150 바이오테크", "mapping_basis": "local_override"},
    {"theme_id": "BATTERY", "etf_code": "305720", "etf_name": "KODEX 2차전지산업", "mapping_basis": "local_override"},
    {"theme_id": "DEFENSE", "etf_code": "463250", "etf_name": "TIGER K방산&우주", "mapping_basis": "local_override"},
    {"theme_id": "AISOFT", "etf_code": "466950", "etf_name": "TIGER AI소프트웨어", "mapping_basis": "local_override"},
    {"theme_id": "HBM", "etf_code": "471760", "etf_name": "TIGER AI반도체핵심공정", "mapping_basis": "local_override"},
    {"theme_id": "SEMIEQP", "etf_code": "471760", "etf_name": "TIGER AI반도체핵심공정", "mapping_basis": "local_override"},
    {"theme_id": "AIPCB", "etf_code": "471760", "etf_name": "TIGER AI반도체핵심공정", "mapping_basis": "local_override"},
    {"theme_id": "POWER", "etf_code": "487240", "etf_name": "KODEX AI전력핵심설비", "mapping_basis": "local_override"},
    {"theme_id": "BROKER", "etf_code": "157500", "etf_name": "TIGER 증권", "mapping_basis": "local_override"},
    {"theme_id": "BANKRET", "etf_code": "091220", "etf_name": "TIGER 은행고배당플러스TOP10", "mapping_basis": "local_override"},
    {"theme_id": "FINPLAT", "etf_code": "091220", "etf_name": "TIGER 은행고배당플러스TOP10", "mapping_basis": "local_override"},
    {"theme_id": "FINPLAT", "etf_code": "365000", "etf_name": "TIGER 인터넷TOP10", "mapping_basis": "local_override"},
    {"theme_id": "PLATECO", "etf_code": "365000", "etf_name": "TIGER 인터넷TOP10", "mapping_basis": "local_override"},
]


def load_theme_etf_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"theme_id": str, "etf_code": str})
    df["theme_id"] = df["theme_id"].fillna("").astype(str).str.upper().str.strip()
    df["etf_code"] = df["etf_code"].fillna("").astype(str).str.zfill(6)
    df["etf_name"] = df.get("etf_name", "").fillna("").astype(str)
    return df


def load_theme_etf_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"theme_id": str, "etf_code": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["theme_id"] = df["theme_id"].fillna("").astype(str).str.upper().str.strip()
    df["etf_code"] = df["etf_code"].fillna("").astype(str).str.zfill(6)
    return df


def load_near_top20(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["rank_final"] = pd.to_numeric(df["rank_final"], errors="coerce").fillna(0).astype(int)
    return df.loc[df["rank_final"].between(21, 35)].copy()


def build_headroom_frame() -> pd.DataFrame:
    master = load_theme_etf_master(THEME_ETF_MASTER_CSV)
    daily = load_theme_etf_daily(THEME_ETF_DAILY_CSV)
    near_top20 = load_near_top20(NEAR_TOP20_AUDIT_CSV)
    latest_date = str(daily["date"].max())
    latest_daily = daily.loc[daily["date"] == latest_date].copy()

    local_map = pd.DataFrame(KNOWN_LOCAL_THEME_MAPPINGS)
    local_map["theme_id"] = local_map["theme_id"].astype(str).str.upper().str.strip()
    local_map["etf_code"] = local_map["etf_code"].astype(str).str.zfill(6)

    demand = (
        near_top20.loc[near_top20["dominant_theme"].fillna("").astype(str).str.strip() != ""]
        .groupby("dominant_theme")
        .agg(
            near_top20_stock_count=("code", "nunique"),
            near_top20_names=("name", lambda s: " | ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"dominant_theme": "theme_name"})
    )

    rows: list[dict[str, object]] = []
    for theme_id, grp in latest_daily.groupby("theme_id", dropna=False):
        theme_id = str(theme_id)
        theme_name = str(grp["theme_name"].iloc[0])
        current_codes = sorted(set(grp["etf_code"].astype(str)))
        current_source_names = sorted(set(grp.get("source_name", pd.Series(dtype=str)).fillna("").astype(str)))
        current_score = float(pd.to_numeric(grp["etf_theme_score"], errors="coerce").fillna(0.0).max())
        current_conf = float(pd.to_numeric(grp["etf_signal_confidence"], errors="coerce").fillna(0.0).max())
        local_candidates = local_map.loc[local_map["theme_id"] == theme_id].copy()
        extra_candidates = local_candidates.loc[~local_candidates["etf_code"].isin(current_codes)].copy()
        demand_row = demand.loc[demand["theme_name"] == theme_name]
        rows.append(
            {
                "theme_id": theme_id,
                "theme_name": theme_name,
                "latest_date": latest_date,
                "current_etf_count": len(current_codes),
                "current_etf_codes": "|".join(current_codes),
                "current_source_names": "|".join(current_source_names),
                "current_etf_theme_score": round(current_score, 4),
                "current_etf_signal_confidence": round(current_conf, 4),
                "near_top20_stock_count": int(demand_row["near_top20_stock_count"].iloc[0]) if not demand_row.empty else 0,
                "near_top20_names": str(demand_row["near_top20_names"].iloc[0]) if not demand_row.empty else "",
                "known_local_candidate_count": int(len(local_candidates)),
                "extra_local_candidate_count": int(len(extra_candidates)),
                "extra_local_etf_codes": "|".join(extra_candidates["etf_code"].astype(str).tolist()),
                "extra_local_etf_names": "|".join(extra_candidates["etf_name"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out["headroom_status"] = out.apply(classify_headroom, axis=1)
    return out.sort_values(
        ["near_top20_stock_count", "current_etf_theme_score", "theme_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def classify_headroom(row: pd.Series) -> str:
    score = float(row.get("current_etf_theme_score", 0.0) or 0.0)
    near_count = int(row.get("near_top20_stock_count", 0) or 0)
    extra_count = int(row.get("extra_local_candidate_count", 0) or 0)
    if near_count <= 0:
        return "non_priority"
    if score < 50.0 and extra_count <= 0:
        return "weak_source_no_local_headroom"
    if score < 50.0 and extra_count > 0:
        return "weak_source_with_local_headroom"
    if score >= 50.0 and extra_count > 0:
        return "strong_source_with_local_headroom"
    return "strong_source_no_local_headroom"


def write_markdown(df: pd.DataFrame) -> None:
    latest_date = str(df["latest_date"].max()) if not df.empty else ""
    priority = df.loc[df["near_top20_stock_count"] > 0].copy()
    weak_no_headroom = priority.loc[priority["headroom_status"] == "weak_source_no_local_headroom"].copy()
    weak_with_headroom = priority.loc[priority["headroom_status"] == "weak_source_with_local_headroom"].copy()
    lines = [
        "# Theme Source Headroom Review",
        "",
        f"- latest_date: {latest_date}",
        f"- priority themes with near-top20 demand: {len(priority)}",
        f"- weak_source_no_local_headroom: {len(weak_no_headroom)}",
        f"- weak_source_with_local_headroom: {len(weak_with_headroom)}",
        "",
        "## One-Line Verdict",
        "",
        "- The current bottleneck is not just weak source score; for the main weak themes, there is also almost no additional local ETF breadth to unlock inside the current universe.",
        "",
        "## Priority Findings",
        "",
    ]
    for row in priority.itertuples(index=False):
        lines.append(
            f"- `{row.theme_id}` {row.theme_name}: score={row.current_etf_theme_score}, near_top20={row.near_top20_stock_count}, "
            f"extra_local_candidate_count={row.extra_local_candidate_count}, status={row.headroom_status}, names={row.near_top20_names}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `BATTERY`, `BIO`, and `BROKER` are weak because the source theme score itself is low on the latest date.",
            "- Under the current local ETF universe, those weak themes also do not have enough additional ETF breadth to fix the problem by simple master expansion.",
            "- That means the next structural choices are:",
            "  1. strengthen theme taxonomy or proxy design for missing sectors, or",
            "  2. add more ETF universe coverage beyond the current local set.",
            "",
            "## Recommended Next Step",
            "",
            "- Do not spend another round only tuning ranking overlay weights.",
            "- First decide whether to add a new clean-fit theme axis for unmapped candidates, or expand local ETF proxy coverage for weak themes.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    df = build_headroom_frame()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    write_markdown(df)
    print(json.dumps(
        {
            "latest_date": str(df["latest_date"].max()) if not df.empty else None,
            "priority_theme_count": int((df["near_top20_stock_count"] > 0).sum()) if not df.empty else 0,
            "weak_source_no_local_headroom": int((df["headroom_status"] == "weak_source_no_local_headroom").sum()) if not df.empty else 0,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
