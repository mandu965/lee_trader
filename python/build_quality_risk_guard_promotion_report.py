from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

VARIANT_METRICS_CSV = OUTPUT_DIR / "walkforward_weight_variant_metrics.csv"
REPEATABILITY_JSON = OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json"
QUALITY_COVERAGE_JSON = OUTPUT_DIR / "quality_coverage_report.json"

OUT_JSON = OUTPUT_DIR / "quality_risk_guard_promotion_report.json"
OUT_MD = OUTPUT_DIR / "quality_risk_guard_promotion_report.md"


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


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


def load_variant_metrics() -> tuple[dict[str, object], dict[str, object]]:
    if not VARIANT_METRICS_CSV.exists():
        return {}, {}
    df = pd.read_csv(VARIANT_METRICS_CSV)
    if df.empty:
        return {}, {}
    baseline = df.loc[df["variant"].astype(str).eq("baseline")].head(1)
    guard = df.loc[df["variant"].astype(str).eq("quality_risk_guard")].head(1)
    return (
        baseline.iloc[0].to_dict() if not baseline.empty else {},
        guard.iloc[0].to_dict() if not guard.empty else {},
    )


def build_payload() -> dict[str, object]:
    baseline, guard = load_variant_metrics()
    repeatability = _read_json(REPEATABILITY_JSON)
    coverage = _read_json(QUALITY_COVERAGE_JSON)

    repeatability_summary = repeatability.get("summary", {}) if isinstance(repeatability, dict) else {}
    coverage_summary = coverage.get("summary", {}) if isinstance(coverage, dict) else {}

    ordering_ok = bool(guard.get("ordering_ok")) if guard else False
    return_not_worse = (
        pd.notna(pd.to_numeric(guard.get("top20_avg_return"), errors="coerce"))
        and pd.notna(pd.to_numeric(baseline.get("top20_avg_return"), errors="coerce"))
        and float(guard.get("top20_avg_return")) >= float(baseline.get("top20_avg_return"))
    )
    mdd_not_worse = (
        pd.notna(pd.to_numeric(guard.get("top20_avg_mdd"), errors="coerce"))
        and pd.notna(pd.to_numeric(baseline.get("top20_avg_mdd"), errors="coerce"))
        and float(guard.get("top20_avg_mdd")) >= float(baseline.get("top20_avg_mdd"))
    )
    low_qual_improved = (
        pd.notna(pd.to_numeric(guard.get("top20_low_qual_count"), errors="coerce"))
        and pd.notna(pd.to_numeric(baseline.get("top20_low_qual_count"), errors="coerce"))
        and float(guard.get("top20_low_qual_count")) < float(baseline.get("top20_low_qual_count"))
    )
    high_risk_improved = (
        pd.notna(pd.to_numeric(guard.get("top20_high_risk_count"), errors="coerce"))
        and pd.notna(pd.to_numeric(baseline.get("top20_high_risk_count"), errors="coerce"))
        and float(guard.get("top20_high_risk_count")) < float(baseline.get("top20_high_risk_count"))
    )
    repeatability_ready = str(repeatability_summary.get("judgment", "")) == "emerging_repeatability"
    quality_coverage_ready = (
        pd.notna(pd.to_numeric(coverage_summary.get("top20_low_confidence_count"), errors="coerce"))
        and float(coverage_summary.get("top20_low_confidence_count")) <= 6.0
    )

    checks = [
        {
            "name": "ordering_ok",
            "passed": ordering_ok,
            "detail": "quality_risk_guard가 latest matured run에서 top20 > top50 > universe를 회복했는지",
        },
        {
            "name": "top20_return_not_worse",
            "passed": return_not_worse,
            "detail": "top20 평균 수익률이 baseline보다 악화되지 않았는지",
        },
        {
            "name": "top20_mdd_not_worse",
            "passed": mdd_not_worse,
            "detail": "top20 평균 MDD가 baseline보다 더 깊어지지 않았는지",
        },
        {
            "name": "low_quality_count_improved",
            "passed": low_qual_improved,
            "detail": "top20 저품질 종목 수가 baseline보다 줄었는지",
        },
        {
            "name": "high_risk_count_improved",
            "passed": high_risk_improved,
            "detail": "top20 고위험 종목 수가 baseline보다 줄었는지",
        },
        {
            "name": "repeatability_ready",
            "passed": repeatability_ready,
            "detail": "shadow 반복성 리포트가 emerging_repeatability까지 올라왔는지",
        },
        {
            "name": "quality_coverage_ready",
            "passed": quality_coverage_ready,
            "detail": "Top20 quality low-confidence 종목 수가 아직 과도하지 않은지",
        },
    ]

    if all(item["passed"] for item in checks):
        verdict = "READY_FOR_PROMOTION_REVIEW"
    elif ordering_ok and return_not_worse and mdd_not_worse and low_qual_improved and high_risk_improved:
        verdict = "HOLD_FOR_MORE_EVIDENCE"
    else:
        verdict = "DO_NOT_PROMOTE"

    return {
        "summary": {
            "verdict": verdict,
            "ordering_ok": ordering_ok,
            "repeatability_judgment": repeatability_summary.get("judgment"),
            "top20_low_confidence_count": coverage_summary.get("top20_low_confidence_count"),
        },
        "baseline_metrics": baseline,
        "quality_risk_guard_metrics": guard,
        "repeatability_summary": repeatability_summary,
        "quality_coverage_summary": coverage_summary,
        "checks": checks,
    }


def build_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    baseline = payload.get("baseline_metrics", {}) if isinstance(payload, dict) else {}
    guard = payload.get("quality_risk_guard_metrics", {}) if isinstance(payload, dict) else {}
    checks = payload.get("checks", []) if isinstance(payload, dict) else []

    lines = [
        "# Quality/Risk Guard Promotion Report",
        "",
        f"- verdict: {summary.get('verdict', '-')}",
        f"- ordering_ok: {summary.get('ordering_ok', False)}",
        f"- repeatability_judgment: {summary.get('repeatability_judgment', '-')}",
        f"- top20_low_confidence_count: {summary.get('top20_low_confidence_count', '-')}",
        "",
        "## Variant Comparison",
        "",
        "| metric | baseline | quality_risk_guard |",
        "| --- | ---: | ---: |",
        f"| top20_avg_return | {_fmt_pct(baseline.get('top20_avg_return'))} | {_fmt_pct(guard.get('top20_avg_return'))} |",
        f"| top50_avg_return | {_fmt_pct(baseline.get('top50_avg_return'))} | {_fmt_pct(guard.get('top50_avg_return'))} |",
        f"| universe_avg_return | {_fmt_pct(baseline.get('universe_avg_return'))} | {_fmt_pct(guard.get('universe_avg_return'))} |",
        f"| top20_avg_mdd | {_fmt_pct(baseline.get('top20_avg_mdd'))} | {_fmt_pct(guard.get('top20_avg_mdd'))} |",
        f"| score_return_corr | {_fmt_num(baseline.get('score_return_corr'), 4)} | {_fmt_num(guard.get('score_return_corr'), 4)} |",
        f"| top20_low_qual_count | {_fmt_num(baseline.get('top20_low_qual_count'), 0)} | {_fmt_num(guard.get('top20_low_qual_count'), 0)} |",
        f"| top20_high_risk_count | {_fmt_num(baseline.get('top20_high_risk_count'), 0)} | {_fmt_num(guard.get('top20_high_risk_count'), 0)} |",
        "",
        "## Promotion Checks",
        "",
    ]

    for item in checks:
        mark = "PASS" if item.get("passed") else "HOLD"
        lines.append(f"- {mark} {item.get('name')}: {item.get('detail')}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    verdict = str(summary.get("verdict", ""))
    if verdict == "READY_FOR_PROMOTION_REVIEW":
        lines.append("- 현재 기준으로는 production 승격 검토를 열 수 있습니다.")
    elif verdict == "HOLD_FOR_MORE_EVIDENCE":
        lines.append("- latest matured run 성능은 좋아졌지만, repeatability와 coverage 같은 운영 증거가 더 필요합니다.")
    else:
        lines.append("- 아직 production 승격 단계는 아니며, quality/risk guard는 shadow 관찰을 유지하는 편이 맞습니다.")

    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build_payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")
    print(f"quality_risk_guard_promotion_report_json: {OUT_JSON}")
    print(f"quality_risk_guard_promotion_report_md: {OUT_MD}")
    print(f"verdict: {payload.get('summary', {}).get('verdict', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
