"""Utilities for bootstrapping ETF-based theme mapping CSV files.

ETF rows are used to measure theme strength.
Stock-theme rows are intentionally editable so mappings can be curated manually.
`mapping_weight` represents how directly a stock belongs to a theme, on a 0.0~1.0 scale.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"

THEME_ETF_MASTER_COLUMNS = [
    "theme_id",
    "theme_name",
    "etf_code",
    "etf_name",
    "market",
    "is_active",
    "priority",
    "note",
]

STOCK_THEME_MAP_COLUMNS = [
    "code",
    "name",
    "theme_id",
    "theme_name",
    "theme_role",
    "match_type",
    "mapping_weight",
    "theme_weight",
    "mapping_source",
    "source_note",
    "is_primary",
    "updated_at",
]

CANONICAL_THEME_NAME_BY_ID = {
    "HBM": "HBM",
    "SEMIEQP": "반도체장비",
    "AIPCB": "AI서버기판",
    "POWER": "전력설비",
    "DEFENSE": "방산",
    "BATTERY": "2차전지",
    "SHIP": "조선",
    "BIO": "바이오",
    "TELCO": "Digital Connectivity",
    "BROKER": "Brokerage Markets",
    "BANKRET": "Retail Finance Return",
    "FINPLAT": "Digital Finance Platform",
    "PLATECO": "Digital Platform Ecosystem",
    "GAMEIP": "Game IP Monetization",
    "AIRMOB": "Air Mobility Recovery",
    "AISOFT": "Enterprise AI Software",
    "ROBAUTO": "Robotics & Factory Automation",
}

SAMPLE_THEME_ETF_MASTER = [
    {
        "theme_id": "HBM",
        "theme_name": "HBM",
        "etf_code": "471760",
        "etf_name": "TIGER AI반도체핵심공정",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "HBM theme strength proxy using semiconductor core process ETF.",
    },
    {
        "theme_id": "SEMIEQP",
        "theme_name": "반도체장비",
        "etf_code": "471760",
        "etf_name": "TIGER AI반도체핵심공정",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "Semiconductor equipment theme proxy using core process ETF.",
    },
    {
        "theme_id": "AIPCB",
        "theme_name": "AI서버기판",
        "etf_code": "471760",
        "etf_name": "TIGER AI반도체핵심공정",
        "market": "KRX",
        "is_active": True,
        "priority": 2,
        "note": "AI server substrate and advanced packaging proxy using core process ETF.",
    },
    {
        "theme_id": "POWER",
        "theme_name": "전력설비",
        "etf_code": "487240",
        "etf_name": "KODEX AI전력핵심설비",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "Grid, transformer, switchgear, and power equipment theme proxy.",
    },
    {
        "theme_id": "DEFENSE",
        "theme_name": "방산",
        "etf_code": "463250",
        "etf_name": "TIGER K방산&우주",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "Defense and aerospace theme strength proxy.",
    },
    {
        "theme_id": "BATTERY",
        "theme_name": "2차전지",
        "etf_code": "305720",
        "etf_name": "KODEX 2차전지산업",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "Battery cell, material, and equipment theme proxy.",
    },
    {
        "theme_id": "SHIP",
        "theme_name": "조선",
        "etf_code": "000000",
        "etf_name": "TEMP_SHIP_THEME_BASKET",
        "market": "KRX",
        "is_active": False,
        "priority": 3,
        "note": "ETF 분리 후보",
    },
    {
        "theme_id": "BIO",
        "theme_name": "바이오",
        "etf_code": "261070",
        "etf_name": "TIGER 코스닥150 바이오테크",
        "market": "KRX",
        "is_active": True,
        "priority": 1,
        "note": "Biotech and healthcare platform theme proxy.",
    },
]

_UPDATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SAMPLE_STOCK_THEME_MAP = [
    {"code": "000660", "name": "SK하이닉스", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "042700", "name": "한미반도체", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 0.98, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "319660", "name": "피에스케이", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 0.82, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "240810", "name": "원익IPS", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 0.78, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "039030", "name": "이오테크닉스", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 0.69, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "091700", "name": "파트론", "theme_id": "HBM", "theme_name": "HBM", "mapping_weight": 0.45, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "240810", "name": "원익IPS", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.94, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "084370", "name": "유진테크", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.89, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "319660", "name": "피에스케이", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.87, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "039030", "name": "이오테크닉스", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.84, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "005290", "name": "동진쎄미켐", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.72, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "042700", "name": "한미반도체", "theme_id": "SEMIEQP", "theme_name": "반도체장비", "mapping_weight": 0.86, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "007660", "name": "이수페타시스", "theme_id": "AIPCB", "theme_name": "AI서버기판", "mapping_weight": 0.98, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "353200", "name": "대덕전자", "theme_id": "AIPCB", "theme_name": "AI서버기판", "mapping_weight": 0.88, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "009150", "name": "삼성전기", "theme_id": "AIPCB", "theme_name": "AI서버기판", "mapping_weight": 0.79, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "007810", "name": "코리아써키트", "theme_id": "AIPCB", "theme_name": "AI서버기판", "mapping_weight": 0.77, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "336200", "name": "두산퓨얼셀", "theme_id": "AIPCB", "theme_name": "AI서버기판", "mapping_weight": 0.42, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "267260", "name": "HD현대일렉트릭", "theme_id": "POWER", "theme_name": "전력설비", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "010120", "name": "LS ELECTRIC", "theme_id": "POWER", "theme_name": "전력설비", "mapping_weight": 0.97, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "006260", "name": "LS", "theme_id": "POWER", "theme_name": "전력설비", "mapping_weight": 0.86, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "017890", "name": "한국알콜", "theme_id": "POWER", "theme_name": "전력설비", "mapping_weight": 0.41, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "028260", "name": "삼성물산", "theme_id": "POWER", "theme_name": "전력설비", "mapping_weight": 0.55, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "012450", "name": "한화에어로스페이스", "theme_id": "DEFENSE", "theme_name": "방산", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "079550", "name": "LIG넥스원", "theme_id": "DEFENSE", "theme_name": "방산", "mapping_weight": 0.98, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "047810", "name": "한국항공우주", "theme_id": "DEFENSE", "theme_name": "방산", "mapping_weight": 0.92, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "272210", "name": "한화시스템", "theme_id": "DEFENSE", "theme_name": "방산", "mapping_weight": 0.91, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "064350", "name": "현대로템", "theme_id": "DEFENSE", "theme_name": "방산", "mapping_weight": 0.78, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "373220", "name": "LG에너지솔루션", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "006400", "name": "삼성SDI", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 0.96, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "247540", "name": "에코프로비엠", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 0.94, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "066970", "name": "엘앤에프", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 0.88, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "003670", "name": "포스코퓨처엠", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 0.86, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "278280", "name": "천보", "theme_id": "BATTERY", "theme_name": "2차전지", "mapping_weight": 0.71, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "329180", "name": "HD현대중공업", "theme_id": "SHIP", "theme_name": "조선", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "009540", "name": "HD한국조선해양", "theme_id": "SHIP", "theme_name": "조선", "mapping_weight": 0.98, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "042660", "name": "한화오션", "theme_id": "SHIP", "theme_name": "조선", "mapping_weight": 0.95, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "010140", "name": "삼성중공업", "theme_id": "SHIP", "theme_name": "조선", "mapping_weight": 0.90, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "010620", "name": "현대미포조선", "theme_id": "SHIP", "theme_name": "조선", "mapping_weight": 0.81, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},

    {"code": "207940", "name": "삼성바이오로직스", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 1.00, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "068270", "name": "셀트리온", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 0.95, "mapping_source": "manual_seed_v1", "is_primary": True, "updated_at": _UPDATED_AT},
    {"code": "196170", "name": "알테오젠", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 0.93, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "326030", "name": "SK바이오팜", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 0.83, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "145020", "name": "휴젤", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 0.78, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
    {"code": "214150", "name": "클래시스", "theme_id": "BIO", "theme_name": "바이오", "mapping_weight": 0.52, "mapping_source": "manual_seed_v1", "is_primary": False, "updated_at": _UPDATED_AT},
]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.loc[:, columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def get_canonical_theme_name(theme_id: str, fallback: str = "") -> str:
    theme_key = str(theme_id or "").strip().upper()
    if not theme_key:
        return str(fallback or "").strip()
    return CANONICAL_THEME_NAME_BY_ID.get(theme_key, str(fallback or "").strip())


def standardize_stock_theme_map(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STOCK_THEME_MAP_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["theme_id"] = out["theme_id"].fillna("").astype(str).str.upper().str.strip()
    out["theme_name"] = [
        get_canonical_theme_name(theme_id, fallback=theme_name)
        for theme_id, theme_name in zip(out["theme_id"], out["theme_name"], strict=False)
    ]
    out["mapping_weight"] = pd.to_numeric(out["mapping_weight"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_weight"] = pd.to_numeric(out["theme_weight"], errors="coerce").fillna(out["mapping_weight"]).clip(lower=0.0, upper=1.0)
    out["mapping_source"] = out["mapping_source"].fillna("").astype(str).str.strip()
    out["source_note"] = out["source_note"].fillna("").astype(str).str.strip()
    out["updated_at"] = out["updated_at"].fillna("").astype(str).str.strip()
    out["is_primary"] = (
        out["is_primary"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
    )
    out["theme_role"] = out["theme_role"].fillna("").astype(str).str.strip().str.lower()
    out["match_type"] = out["match_type"].fillna("").astype(str).str.strip().str.lower()
    out.loc[out["theme_role"] == "", "theme_role"] = out["is_primary"].map({True: "primary", False: "secondary"})
    out.loc[out["match_type"] == "", "match_type"] = "direct"
    out = out[out["theme_id"] != ""].copy()
    out = (
        out.sort_values(
            ["code", "theme_id", "is_primary", "theme_weight", "mapping_weight", "updated_at"],
            ascending=[True, True, False, False, False, False],
        )
        .drop_duplicates(subset=["code", "theme_id"], keep="first")
        .reset_index(drop=True)
    )
    # Keep the strongest 3 mappings at most per stock so the dominant theme remains interpretable.
    out = (
        out.sort_values(
            ["code", "is_primary", "theme_weight", "mapping_weight", "theme_id"],
            ascending=[True, False, False, False, True],
        )
        .groupby("code", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    # Enforce a single primary theme per stock for stable downstream tie-breaking.
    out["is_primary"] = False
    primary_idx = out.groupby("code", sort=False).head(1).index
    out.loc[primary_idx, "is_primary"] = True
    out["theme_role"] = out["is_primary"].map({True: "primary", False: "secondary"})
    return out.loc[:, STOCK_THEME_MAP_COLUMNS].copy()


def ensure_theme_mapping_files(*, force: bool = False, logger: logging.Logger | None = None) -> tuple[Path, Path]:
    log = logger or logging.getLogger("theme_mapping_utils")
    if force or not THEME_ETF_MASTER_CSV.exists():
        _write_csv(THEME_ETF_MASTER_CSV, SAMPLE_THEME_ETF_MASTER, THEME_ETF_MASTER_COLUMNS)
        log.info("Created theme ETF master CSV: %s", THEME_ETF_MASTER_CSV.resolve())
    if force or not STOCK_THEME_MAP_CSV.exists():
        _write_csv(STOCK_THEME_MAP_CSV, SAMPLE_STOCK_THEME_MAP, STOCK_THEME_MAP_COLUMNS)
        log.info("Created stock theme map CSV: %s", STOCK_THEME_MAP_CSV.resolve())
    return THEME_ETF_MASTER_CSV, STOCK_THEME_MAP_CSV


def load_theme_etf_master(*, ensure_exists: bool = True) -> pd.DataFrame:
    if ensure_exists:
        ensure_theme_mapping_files()
    df = pd.read_csv(THEME_ETF_MASTER_CSV, dtype={"theme_id": str, "etf_code": str})
    for col in THEME_ETF_MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["theme_id"] = df["theme_id"].fillna("").astype(str).str.upper()
    df["etf_code"] = df["etf_code"].astype(str).str.zfill(6)
    df["is_active"] = df["is_active"].fillna(True)
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(999).astype(int)
    return df.loc[:, THEME_ETF_MASTER_COLUMNS].copy()


def load_stock_theme_map(*, ensure_exists: bool = True) -> pd.DataFrame:
    if ensure_exists:
        ensure_theme_mapping_files()
    df = pd.read_csv(STOCK_THEME_MAP_CSV, dtype={"code": str, "theme_id": str})
    return standardize_stock_theme_map(df)


def validate_theme_mapping_files(*, ensure_exists: bool = True) -> dict[str, object]:
    theme_df = load_theme_etf_master(ensure_exists=ensure_exists)
    stock_df = load_stock_theme_map(ensure_exists=ensure_exists)
    duplicate_count = int(stock_df.duplicated(subset=["code", "theme_id"], keep=False).sum())
    theme_stock_counts = (
        stock_df.groupby(["theme_id", "theme_name"], dropna=False)["code"]
        .nunique()
        .reset_index(name="stock_count")
        .sort_values(["stock_count", "theme_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {
        "theme_count": int(theme_df["theme_id"].nunique()),
        "stock_count": int(stock_df["code"].nunique()),
        "mapping_count": int(len(stock_df)),
        "duplicate_mapping_count": duplicate_count,
        "theme_stock_counts": theme_stock_counts,
    }


def _print_validation_summary() -> None:
    summary = validate_theme_mapping_files()
    print(f"theme_count={summary['theme_count']}")
    print(f"stock_count={summary['stock_count']}")
    print(f"mapping_count={summary['mapping_count']}")
    print(f"duplicate_mapping_count={summary['duplicate_mapping_count']}")
    print("theme_stock_counts:")
    print(summary["theme_stock_counts"].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate ETF theme mapping bootstrap CSV files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files with sample data.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_theme_mapping_files(force=args.force)
    _print_validation_summary()
    print(f"theme_etf_master_path={THEME_ETF_MASTER_CSV}")
    print(f"stock_theme_map_path={STOCK_THEME_MAP_CSV}")


if __name__ == "__main__":
    main()
