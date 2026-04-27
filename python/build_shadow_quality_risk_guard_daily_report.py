from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RANKING_CSV = DATA_DIR / "ranking_final.csv"
REPEATABILITY_JSON = OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json"
OUT_CSV = OUTPUT_DIR / "shadow_quality_risk_guard_daily_report.csv"
OUT_JSON = OUTPUT_DIR / "shadow_quality_risk_guard_daily_report.json"
OUT_MD = OUTPUT_DIR / "shadow_quality_risk_guard_daily_report.md"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _fmt_rank(value: object) -> str:
    return str(int(value)) if pd.notna(value) else "-"


def _fmt_num(value: object) -> str:
    return f"{float(value):.1f}" if pd.notna(value) else "-"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_latest_ranking() -> tuple[pd.DataFrame, str]:
    if not RANKING_CSV.exists():
        raise FileNotFoundError(f"ranking file not found: {RANKING_CSV}")

    df = pd.read_csv(RANKING_CSV, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("ranking_final.csv is empty")
    if "date" not in df.columns:
        raise ValueError("ranking_final.csv missing date column")

    df["date"] = df["date"].astype(str).str.slice(0, 10)
    latest_date = df["date"].max()
    latest = df.loc[df["date"] == latest_date].copy()
    if latest.empty:
        raise ValueError("no latest-date rows found in ranking_final.csv")
    return latest, latest_date


def load_repeatability_payload() -> dict[str, object]:
    if not REPEATABILITY_JSON.exists():
        return {}
    try:
        return json.loads(REPEATABILITY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_candidate_frame(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    out["live_rank"] = _to_num(_pick_first_existing(out, ["live_rank", "rank_final", "rank"]))
    out["live_score"] = _to_num(_pick_first_existing(out, ["live_score", "final_score", "score"]))
    out["shadow_rank_quality_risk_guard"] = _to_num(_pick_first_existing(out, ["shadow_rank_quality_risk_guard"]))
    out["shadow_final_score_quality_risk_guard"] = _to_num(
        _pick_first_existing(out, ["shadow_final_score_quality_risk_guard"])
    )
    out["shadow_quality_risk_guard_penalty"] = _to_num(
        _pick_first_existing(out, ["shadow_quality_risk_guard_penalty"])
    )
    out["shadow_rank_delta_quality_risk_guard"] = out["live_rank"] - out["shadow_rank_quality_risk_guard"]

    keep_cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "live_rank",
        "shadow_rank_quality_risk_guard",
        "shadow_rank_delta_quality_risk_guard",
        "live_score",
        "shadow_final_score_quality_risk_guard",
        "shadow_quality_risk_guard_penalty",
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "risk_penalty",
        "confidence_score",
        "buy_eligibility_status",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    out = out[keep_cols].copy()

    out = out.loc[
        out["shadow_rank_delta_quality_risk_guard"].notna()
        & (out["shadow_rank_delta_quality_risk_guard"] > 0)
    ].copy()
    if out.empty:
        return out

    out = out.sort_values(
        by=[
            "shadow_rank_delta_quality_risk_guard",
            "shadow_quality_risk_guard_penalty",
            "live_rank",
        ],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def build_payload(report_df: pd.DataFrame, latest_date: str) -> dict[str, object]:
    top5 = report_df.head(5).copy() if not report_df.empty else report_df.copy()
    summary = {
        "asof_date": latest_date,
        "candidate_count": int(len(report_df)),
        "top5_count": int(len(top5)),
        "max_rank_delta": int(top5["shadow_rank_delta_quality_risk_guard"].max()) if not top5.empty else 0,
        "mean_rank_delta": round(float(report_df["shadow_rank_delta_quality_risk_guard"].mean()), 2)
        if not report_df.empty
        else 0.0,
        "mean_penalty": round(float(report_df["shadow_quality_risk_guard_penalty"].mean()), 2)
        if not report_df.empty
        else 0.0,
    }
    return {
        "summary": summary,
        "top5_candidates": top5.to_dict(orient="records"),
        "all_candidates": report_df.to_dict(orient="records"),
    }


def build_markdown(payload: dict[str, object], repeatability_payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    rows = payload.get("top5_candidates", []) if isinstance(payload, dict) else []
    repeatability_summary = (
        repeatability_payload.get("summary", {}) if isinstance(repeatability_payload, dict) else {}
    )
    repeatability_top = (
        repeatability_payload.get("top_repeaters", []) if isinstance(repeatability_payload, dict) else []
    )

    lines = [
        "# Shadow Quality/Risk Guard Daily Report",
        "",
        f"- asof_date: {summary.get('asof_date', '-')}",
        f"- candidate_count: {summary.get('candidate_count', 0)}",
        f"- top5_count: {summary.get('top5_count', 0)}",
        f"- max_rank_delta: {summary.get('max_rank_delta', 0)}",
        f"- mean_rank_delta: {summary.get('mean_rank_delta', 0)}",
        f"- mean_penalty: {summary.get('mean_penalty', 0)}",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "## Interpretation",
                "",
                "- 오늘은 baseline 대비 순위가 뚜렷하게 개선되는 shadow 후보가 없습니다.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Top 5 Candidates",
                "",
                "| code | name | live_rank | shadow_rank | delta | penalty | confidence | buy_eligibility |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.get('code', '-')}"
                f" | {row.get('name', '-')}"
                f" | {_fmt_rank(row.get('live_rank'))}"
                f" | {_fmt_rank(row.get('shadow_rank_quality_risk_guard'))}"
                f" | {_fmt_rank(row.get('shadow_rank_delta_quality_risk_guard'))}"
                f" | {_fmt_num(row.get('shadow_quality_risk_guard_penalty'))}"
                f" | {_fmt_num(row.get('confidence_score'))}"
                f" | {row.get('buy_eligibility_status', '-') or '-'} |"
            )

    lines.extend(
        [
            "",
            "## Repeatability Snapshot",
            "",
            f"- judgment: {repeatability_summary.get('judgment', 'unavailable')}",
            f"- usable_snapshot_count: {repeatability_summary.get('usable_snapshot_count', 0)}",
            f"- repeated_candidate_count: {repeatability_summary.get('repeated_candidate_count', 0)}",
        ]
    )

    judgment = str(repeatability_summary.get("judgment", ""))
    if judgment == "insufficient_history":
        lines.append("- 아직 archived ranking snapshot에 shadow 컬럼 이력이 부족해 반복성 해석은 보류합니다.")
    elif judgment == "no_repeaters_yet":
        lines.append("- usable snapshot은 쌓였지만 같은 종목이 반복적으로 개선되는 패턴은 아직 약합니다.")
    elif repeatability_top:
        first = repeatability_top[0]
        lines.append(
            f"- 최근 반복 후보 선두는 {first.get('name', first.get('code', '-'))}이며, "
            f"{first.get('appearance_days', 0)}일 등장, 최근 {first.get('consecutive_recent_days', 0)}일 연속입니다."
        )
    else:
        lines.append("- repeatability 정보가 아직 생성되지 않았습니다.")

    if repeatability_top:
        lines.extend(
            [
                "",
                "## Top Repeaters",
                "",
                "| code | name | appearance_days | recent_streak | avg_delta | latest_delta |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in repeatability_top[:5]:
            lines.append(
                f"| {row.get('code', '-')}"
                f" | {row.get('name', '-')}"
                f" | {_fmt_rank(row.get('appearance_days'))}"
                f" | {_fmt_rank(row.get('consecutive_recent_days'))}"
                f" | {_fmt_num(row.get('avg_rank_delta'))}"
                f" | {_fmt_rank(row.get('latest_rank_delta'))} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 이 리포트는 production 순위를 바꾸지 않고, quality/risk guard shadow 기준에서 개선 여지가 큰 종목만 따로 보여줍니다.",
            "- `delta`가 클수록 baseline 순위보다 shadow 순위가 더 좋아진 종목입니다.",
            "- 아직 production 승격 전이므로 실제 운영 판단은 production 점수를 우선합니다.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    latest, latest_date = load_latest_ranking()
    report_df = build_candidate_frame(latest)
    repeatability_payload = load_repeatability_payload()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    payload = build_payload(report_df, latest_date)
    OUT_JSON.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    OUT_MD.write_text(build_markdown(payload, repeatability_payload), encoding="utf-8")

    print(f"shadow_daily_report_csv: {OUT_CSV}")
    print(f"shadow_daily_report_json: {OUT_JSON}")
    print(f"shadow_daily_report_md: {OUT_MD}")
    print(f"candidate_count: {payload['summary']['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
