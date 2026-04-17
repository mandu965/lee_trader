from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RANKING_CSV = DATA_DIR / "ranking_final.csv"
QUALITY_CSV = DATA_DIR / "quality.csv"
OUT_JSON = OUTPUT_DIR / "quality_coverage_report.json"
OUT_MD = OUTPUT_DIR / "quality_coverage_report.md"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def load_latest_ranking() -> tuple[pd.DataFrame, str]:
    if not RANKING_CSV.exists():
        raise FileNotFoundError(f"ranking file not found: {RANKING_CSV}")
    df = pd.read_csv(RANKING_CSV, encoding="utf-8-sig")
    if df.empty or "date" not in df.columns:
        raise ValueError("ranking_final.csv is empty or missing date column")
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    latest_date = df["date"].max()
    latest = df.loc[df["date"] == latest_date].copy()
    if latest.empty:
        raise ValueError("no latest-date rows found in ranking_final.csv")
    return latest, latest_date


def load_latest_quality_date() -> str | None:
    if not QUALITY_CSV.exists():
        return None
    try:
        df = pd.read_csv(QUALITY_CSV, encoding="utf-8-sig", usecols=["date"])
    except Exception:
        return None
    if df.empty or "date" not in df.columns:
        return None
    dates = df["date"].astype(str).str.slice(0, 10)
    return dates.max() if not dates.empty else None


def build_payload(latest: pd.DataFrame, latest_date: str) -> dict[str, object]:
    out = latest.copy()
    out["live_rank"] = _num(out["live_rank"]) if "live_rank" in out.columns else pd.Series(pd.NA, index=out.index)
    out["quality_missing_ratio"] = _num(out["quality_missing_ratio"]) if "quality_missing_ratio" in out.columns else pd.Series(pd.NA, index=out.index)
    out["quality_factor_count"] = _num(out["quality_factor_count"]) if "quality_factor_count" in out.columns else pd.Series(pd.NA, index=out.index)
    out["quality_score_confidence"] = _num(out["quality_score_confidence"]) if "quality_score_confidence" in out.columns else pd.Series(pd.NA, index=out.index)
    out["qual_score"] = _num(out["qual_score"]) if "qual_score" in out.columns else pd.Series(pd.NA, index=out.index)
    out["final_score"] = _num(out["final_score"]) if "final_score" in out.columns else pd.Series(pd.NA, index=out.index)

    top20 = out.nsmallest(20, "live_rank") if out["live_rank"].notna().any() else out.head(20)

    low_conf_threshold = 60.0
    heavy_missing_threshold = 0.60
    low_qual_threshold = 35.0

    summary = {
        "asof_date": latest_date,
        "ranking_row_count": int(len(out)),
        "quality_latest_date": load_latest_quality_date(),
        "mean_quality_missing_ratio": round(float(out["quality_missing_ratio"].mean()), 4),
        "mean_quality_factor_count": round(float(out["quality_factor_count"].mean()), 2),
        "mean_quality_score_confidence": round(float(out["quality_score_confidence"].mean()), 2),
        "low_confidence_count": int((out["quality_score_confidence"] < low_conf_threshold).sum()),
        "heavy_missing_count": int((out["quality_missing_ratio"] >= heavy_missing_threshold).sum()),
        "top20_mean_quality_missing_ratio": round(float(top20["quality_missing_ratio"].mean()), 4),
        "top20_mean_quality_score_confidence": round(float(top20["quality_score_confidence"].mean()), 2),
        "top20_low_confidence_count": int((top20["quality_score_confidence"] < low_conf_threshold).sum()),
        "top20_low_qual_count": int((top20["qual_score"] < low_qual_threshold).sum()),
    }

    weakest = out.loc[
        (out["quality_score_confidence"] < low_conf_threshold) | (out["quality_missing_ratio"] >= heavy_missing_threshold)
    ].copy()
    weakest = weakest.sort_values(
        by=["quality_score_confidence", "quality_missing_ratio", "live_rank"],
        ascending=[True, False, True],
        na_position="last",
    ).head(10)

    return {
        "summary": summary,
        "coverage_flags": {
            "low_confidence_threshold": low_conf_threshold,
            "heavy_missing_threshold": heavy_missing_threshold,
            "low_qual_threshold": low_qual_threshold,
        },
        "top20_coverage_snapshot": top20[
            [col for col in ["code", "name", "live_rank", "final_score", "qual_score", "quality_factor_count", "quality_missing_ratio", "quality_score_confidence"] if col in top20.columns]
        ].to_dict(orient="records"),
        "weakest_coverage_names": weakest[
            [col for col in ["code", "name", "live_rank", "qual_score", "quality_factor_count", "quality_missing_ratio", "quality_score_confidence"] if col in weakest.columns]
        ].to_dict(orient="records"),
    }


def build_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    weakest = payload.get("weakest_coverage_names", []) if isinstance(payload, dict) else []

    lines = [
        "# Quality Coverage Report",
        "",
        f"- asof_date: {summary.get('asof_date', '-')}",
        f"- ranking_row_count: {summary.get('ranking_row_count', 0)}",
        f"- quality_latest_date: {summary.get('quality_latest_date', '-')}",
        f"- mean_quality_missing_ratio: {_fmt_num(summary.get('mean_quality_missing_ratio'), 4)}",
        f"- mean_quality_factor_count: {_fmt_num(summary.get('mean_quality_factor_count'), 2)} / 5",
        f"- mean_quality_score_confidence: {_fmt_num(summary.get('mean_quality_score_confidence'), 2)}",
        f"- low_confidence_count: {summary.get('low_confidence_count', 0)}",
        f"- heavy_missing_count: {summary.get('heavy_missing_count', 0)}",
        f"- top20_mean_quality_missing_ratio: {_fmt_num(summary.get('top20_mean_quality_missing_ratio'), 4)}",
        f"- top20_mean_quality_score_confidence: {_fmt_num(summary.get('top20_mean_quality_score_confidence'), 2)}",
        f"- top20_low_confidence_count: {summary.get('top20_low_confidence_count', 0)}",
        f"- top20_low_qual_count: {summary.get('top20_low_qual_count', 0)}",
        "",
        "## Interpretation",
        "",
        f"- 전체 평균 기준 quality factor 확보 수는 {_fmt_num(summary.get('mean_quality_factor_count'), 2)} / 5 수준입니다.",
        f"- 전체 평균 missing ratio는 {_fmt_pct(summary.get('mean_quality_missing_ratio'))}이며, Top20도 {_fmt_pct(summary.get('top20_mean_quality_missing_ratio'))}로 낮지 않습니다.",
        f"- Top20 안에서도 quality confidence 60 미만 종목이 {summary.get('top20_low_confidence_count', 0)}개 있어 quality 축을 강한 production 축으로 쓰기엔 아직 보수적 해석이 필요합니다.",
        "",
    ]

    if weakest:
        lines.extend(
            [
                "## Weakest Coverage Names",
                "",
                "| code | name | live_rank | qual_score | factor_count | missing_ratio | confidence |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in weakest:
            lines.append(
                f"| {row.get('code', '-')}"
                f" | {row.get('name', '-')}"
                f" | {_fmt_num(row.get('live_rank'), 0)}"
                f" | {_fmt_num(row.get('qual_score'), 1)}"
                f" | {_fmt_num(row.get('quality_factor_count'), 0)}"
                f" | {_fmt_num(row.get('quality_missing_ratio'), 2)}"
                f" | {_fmt_num(row.get('quality_score_confidence'), 1)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Action",
            "",
            "- 다음 단계는 quality_score_raw / quality_score_rank 역할을 분리하고, coverage 부족 종목에 대한 감쇠 규칙을 더 명확히 두는 것입니다.",
            "- production 승격 판단 전까지는 quality_score_confidence와 missing_ratio를 함께 보고 해석해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    latest, latest_date = load_latest_ranking()
    payload = build_payload(latest, latest_date)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")

    print(f"quality_coverage_report_json: {OUT_JSON}")
    print(f"quality_coverage_report_md: {OUT_MD}")
    print(f"asof_date: {payload['summary']['asof_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
