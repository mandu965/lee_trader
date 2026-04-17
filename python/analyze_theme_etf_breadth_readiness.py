import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
PRICES_CSV = DATA_DIR / "prices_daily_raw.csv"
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
THEME_ETF_DAILY_CSV = OUTPUT_DIR / "theme_etf_daily.csv"

OUTPUT_CSV = DATA_DIR / "theme_etf_breadth_readiness.csv"
OUTPUT_MD = DATA_DIR / "theme_etf_breadth_readiness.md"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_readiness_frame() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV, dtype={"theme_id": str, "etf_code": str})
    master["theme_id"] = master["theme_id"].fillna("").astype(str).str.upper().str.strip()
    master["etf_code"] = master["etf_code"].fillna("").astype(str).str.zfill(6)
    master["is_active"] = master["is_active"].fillna(False).astype(bool)

    prices = pd.read_csv(PRICES_CSV, usecols=["code"], dtype={"code": str})
    available_codes = set(prices["code"].fillna("").astype(str).str.zfill(6).unique())

    stock_theme = pd.read_csv(STOCK_THEME_MAP_CSV, dtype={"code": str, "theme_id": str})
    stock_theme["theme_id"] = stock_theme["theme_id"].fillna("").astype(str).str.upper().str.strip()
    stock_counts = (
        stock_theme.groupby("theme_id", dropna=False)["code"]
        .nunique()
        .rename("mapped_stock_count")
        .reset_index()
    )

    latest_etf = pd.read_csv(THEME_ETF_DAILY_CSV, dtype={"theme_id": str, "etf_code": str})
    latest_etf["theme_id"] = latest_etf["theme_id"].fillna("").astype(str).str.upper().str.strip()
    latest_etf["date"] = pd.to_datetime(latest_etf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = latest_etf["date"].max()
    latest_etf = latest_etf.loc[latest_etf["date"] == latest_date].copy()
    latest_theme_counts = (
        latest_etf.groupby("theme_id", dropna=False)["etf_code"]
        .nunique()
        .rename("latest_theme_etf_count")
        .reset_index()
    )

    master["in_local_prices"] = master["etf_code"].isin(available_codes)
    readiness_rows: list[dict[str, object]] = []
    for (theme_id, theme_name), grp in master.groupby(["theme_id", "theme_name"], dropna=False):
        readiness_rows.append(
            {
                "theme_id": theme_id,
                "theme_name": theme_name,
                "master_etf_count": int(grp["etf_code"].nunique()),
                "active_etf_count": int(grp["is_active"].sum()),
                "local_price_match_count": int(grp["in_local_prices"].sum()),
                "etf_codes": json.dumps(sorted(set(grp["etf_code"])), ensure_ascii=False),
                "active_etf_codes": json.dumps(sorted(set(grp.loc[grp["is_active"], "etf_code"])), ensure_ascii=False),
            }
        )
    readiness = pd.DataFrame(readiness_rows)
    readiness = readiness.merge(stock_counts, on="theme_id", how="left")
    readiness = readiness.merge(latest_theme_counts, on="theme_id", how="left")
    readiness["mapped_stock_count"] = _to_numeric(readiness["mapped_stock_count"]).fillna(0).astype(int)
    readiness["latest_theme_etf_count"] = _to_numeric(readiness["latest_theme_etf_count"]).fillna(0).astype(int)
    readiness["breadth_ready"] = (
        (readiness["master_etf_count"] >= 2)
        & (readiness["local_price_match_count"] >= 1)
    )
    readiness["proxy_only_risk"] = readiness["local_price_match_count"] == 0
    readiness["comment"] = readiness.apply(
        lambda row: (
            "multi-etf ready"
            if bool(row["breadth_ready"])
            else "single ETF in master + no local ETF price support"
            if int(row["master_etf_count"]) <= 1 and int(row["local_price_match_count"]) == 0
            else "needs ETF master expansion or price coverage"
        ),
        axis=1,
    )
    return readiness.sort_values(
        ["breadth_ready", "mapped_stock_count", "theme_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_markdown(readiness: pd.DataFrame) -> None:
    total_themes = len(readiness)
    breadth_ready_count = int(readiness["breadth_ready"].sum())
    proxy_only_count = int(readiness["proxy_only_risk"].sum())
    single_etf_count = int((readiness["master_etf_count"] <= 1).sum())
    heavy_themes = readiness.sort_values("mapped_stock_count", ascending=False).head(8)

    lines = [
        "# Theme ETF Breadth Readiness",
        "",
        f"- total_themes: {total_themes}",
        f"- breadth_ready_count: {breadth_ready_count}",
        f"- proxy_only_theme_count: {proxy_only_count}",
        f"- single_etf_theme_count: {single_etf_count}",
        "",
        "## Summary",
        "",
        "- 현재는 모든 테마가 master 상 1 ETF 구조입니다.",
        "- `prices_daily_raw.csv` 기준 theme_etf_master ETF 코드와 매칭되는 로컬 ETF 가격 데이터는 0개입니다.",
        "- 따라서 지금 단계에서 theme_etf_master만 늘려도 실제 ETF breadth 효과보다 proxy 중복 효과가 커질 가능성이 높습니다.",
        "- 우선순위는 `ETF 가격/AUM/좌수/NAV 원천 확보` 또는 `proxy basket을 theme 내부에서 더 정교하게 다변화`하는 쪽입니다.",
        "",
        "## High Stock-Coverage Themes",
        "",
    ]
    for _, row in heavy_themes.iterrows():
        lines.append(
            f"- {row['theme_id']} / {row['theme_name']}: "
            f"stocks={int(row['mapped_stock_count'])}, master_etf_count={int(row['master_etf_count'])}, "
            f"local_price_match_count={int(row['local_price_match_count'])}, comment={row['comment']}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- 1차: ETF 로컬 가격 universe부터 확보해 `theme_etf_master` 코드가 실제 시계열과 연결되게 만들기",
            "- 2차: 테마당 복수 ETF를 넣되, 같은 proxy를 중복 복제하는 방식은 피하기",
            "- 3차: 그 다음 `build_stock_theme_daily.py` breadth 보정 강도를 실제 복수 ETF 구조에 맞춰 재조정",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    readiness = build_readiness_frame()
    readiness.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    write_markdown(readiness)
    print(f"saved_csv={OUTPUT_CSV}")
    print(f"saved_md={OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
