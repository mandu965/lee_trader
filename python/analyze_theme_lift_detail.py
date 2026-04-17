from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

THEME_LIFT_CSV = DATA_DIR / "theme_lift_analysis.csv"
RANKING_FINAL_CSV = DATA_DIR / "ranking_final.csv"
OUTPUT_CSV = DATA_DIR / "theme_lift_detail_report.csv"


def load_csv(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, dtype=dtype, low_memory=False)


def normalize_theme(series: pd.Series) -> pd.Series:
    return (
        series.fillna("(none)")
        .astype(str)
        .str.strip()
        .replace({"": "(none)", "nan": "(none)", "None": "(none)"})
    )


def build_report() -> pd.DataFrame:
    lift = load_csv(THEME_LIFT_CSV, dtype={"code": str})
    ranking = load_csv(RANKING_FINAL_CSV, dtype={"code": str})

    lift["code"] = lift["code"].astype(str).str.zfill(6)
    ranking["code"] = ranking["code"].astype(str).str.zfill(6)

    lift["theme"] = normalize_theme(lift.get("dominant_theme", pd.Series(dtype=str)))
    lift["score_delta_v3"] = pd.to_numeric(lift.get("score_delta_v3"), errors="coerce").fillna(0.0)

    ranking["date"] = pd.to_datetime(ranking["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = ranking["date"].dropna().max()
    latest = ranking.loc[ranking["date"] == latest_date].copy()
    latest["dominant_theme"] = normalize_theme(latest.get("dominant_theme", pd.Series(dtype=str)))
    latest["rank_final"] = pd.to_numeric(latest.get("rank_final"), errors="coerce")
    top20 = latest.loc[latest["rank_final"] <= 20, ["code", "name", "dominant_theme"]].copy()
    top20["in_top20"] = 1

    merged = lift.merge(
        top20.rename(columns={"dominant_theme": "top20_theme"}),
        on=["code", "name"],
        how="left",
    )
    merged["in_top20"] = pd.to_numeric(merged.get("in_top20"), errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for theme, grp in merged.groupby("theme", dropna=False):
        grp = grp.sort_values(["score_delta_v3", "code"], ascending=[False, True]).reset_index(drop=True)
        contributors = ", ".join(grp["name"].astype(str).head(3).tolist())
        rows.append(
            {
                "theme": theme,
                "avg_lift": float(grp["score_delta_v3"].mean()),
                "max_lift": float(grp["score_delta_v3"].max()),
                "top20_count": int(grp["in_top20"].sum()),
                "contributors": contributors,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["theme", "avg_lift", "max_lift", "top20_count", "contributors"])

    return out.sort_values(["top20_count", "avg_lift", "theme"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> None:
    report = build_report()
    report.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"generated_file={OUTPUT_CSV}")
    print(f"row_count={len(report)}")


if __name__ == "__main__":
    main()
