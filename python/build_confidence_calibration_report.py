from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from build_walk_forward_score_validation import (  # noqa: E402
    MIN_RUN_ROWS,
    attach_realized_outcomes,
    build_rank_history,
)


OUTPUT_MD = ROOT / "outputs" / "confidence_calibration_report.md"
OUTPUT_CSV = ROOT / "outputs" / "confidence_bucket_stats.csv"
LIVE_GRADE_JSON = ROOT / "outputs" / "confidence_live_grade_map.json"
LIVE_GRADE_MD = ROOT / "outputs" / "confidence_live_grade_report.md"
HORIZONS = [20, 60, 90]
BUCKET_ORDER = ["0-20", "20-40", "40-60", "60-80", "80-100"]
BUCKET_MID = {"0-20": 10.0, "20-40": 30.0, "40-60": 50.0, "60-80": 70.0, "80-100": 90.0}


def _fmt(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(item) for item in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def bucketize_confidence(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series(pd.NA, index=series.index, dtype="object")
    bucket = bucket.mask(values.between(0, 20, inclusive="both"), "0-20")
    bucket = bucket.mask((values > 20) & (values <= 40), "20-40")
    bucket = bucket.mask((values > 40) & (values <= 60), "40-60")
    bucket = bucket.mask((values > 60) & (values <= 80), "60-80")
    bucket = bucket.mask((values > 80) & (values <= 100), "80-100")
    return bucket


def monotonic_judgment(summary: pd.DataFrame) -> str:
    valid = summary.dropna(subset=["bucket_mid", "avg_return", "hit_rate", "avg_mdd"]).copy()
    if len(valid) < 3:
        return "INSUFFICIENT"

    return_corr = valid["bucket_mid"].corr(valid["avg_return"], method="spearman")
    hit_corr = valid["bucket_mid"].corr(valid["hit_rate"], method="spearman")
    mdd_corr = valid["bucket_mid"].corr(valid["avg_mdd"], method="spearman")
    if pd.isna(return_corr) or pd.isna(hit_corr) or pd.isna(mdd_corr):
        return "INSUFFICIENT"

    if return_corr >= 0.5 and hit_corr >= 0.5 and mdd_corr >= 0.3:
        return "GOOD"
    if return_corr >= 0.2 and hit_corr >= 0.2:
        return "WEAK"
    return "NOT_MONOTONIC"


def correlation_judgment(conf_outcome_corr: float | None, final_outcome_corr: float | None) -> str:
    if conf_outcome_corr is None or pd.isna(conf_outcome_corr) or final_outcome_corr is None or pd.isna(final_outcome_corr):
        return "INSUFFICIENT"
    if abs(conf_outcome_corr - final_outcome_corr) <= 0.05:
        return "LIKELY_REPACKAGED"
    if conf_outcome_corr >= 0.15 and abs(conf_outcome_corr - final_outcome_corr) > 0.05:
        return "HAS_INCREMENTAL_SIGNAL"
    return "WEAK_INCREMENTAL_SIGNAL"


def build_horizon_report(ranked: pd.DataFrame, horizon_days: int) -> dict[str, object]:
    return_col = f"realized_return_{horizon_days}d"
    mdd_col = f"realized_mdd_{horizon_days}d"
    work = attach_realized_outcomes(ranked, horizon_days=horizon_days)
    matured = work.loc[pd.to_numeric(work[return_col], errors="coerce").notna()].copy()
    if matured.empty:
        return {"status": "unavailable", "horizon_days": horizon_days, "reason": "no matured rows available"}

    eligible_dates = matured.groupby("date")["code"].size()
    eligible_dates = eligible_dates[eligible_dates >= MIN_RUN_ROWS]
    matured = matured.loc[matured["date"].isin(eligible_dates.index)].copy()
    if matured.empty:
        return {
            "status": "unavailable",
            "horizon_days": horizon_days,
            "reason": f"no matured ranking dates with >= {MIN_RUN_ROWS} rows",
        }

    matured["confidence_score"] = pd.to_numeric(matured["confidence_score"], errors="coerce")
    matured["final_score"] = pd.to_numeric(matured["final_score"], errors="coerce")
    matured[return_col] = pd.to_numeric(matured[return_col], errors="coerce")
    matured[mdd_col] = pd.to_numeric(matured[mdd_col], errors="coerce")
    matured["confidence_bucket"] = bucketize_confidence(matured["confidence_score"])
    matured["hit"] = matured[return_col] > 0
    matured["top20_flag"] = matured.groupby("date")["rank_final"].transform(
        lambda s: pd.to_numeric(s, errors="coerce") <= 20
    )

    bucket_summary = (
        matured.groupby("confidence_bucket", dropna=False, observed=False)
        .agg(
            n=("code", "size"),
            avg_return=(return_col, "mean"),
            median_return=(return_col, "median"),
            hit_rate=("hit", "mean"),
            avg_mdd=(mdd_col, "mean"),
            top20_entry_rate=("top20_flag", "mean"),
            avg_final_score=("final_score", "mean"),
            avg_confidence_score=("confidence_score", "mean"),
        )
        .reset_index()
    )

    completed_rows: list[dict[str, object]] = []
    for bucket in BUCKET_ORDER:
        row = bucket_summary.loc[bucket_summary["confidence_bucket"] == bucket]
        if row.empty:
            completed_rows.append(
                {
                    "confidence_bucket": bucket,
                    "bucket_mid": BUCKET_MID[bucket],
                    "n": 0,
                    "avg_return": pd.NA,
                    "median_return": pd.NA,
                    "hit_rate": pd.NA,
                    "avg_mdd": pd.NA,
                    "top20_entry_rate": pd.NA,
                    "avg_final_score": pd.NA,
                    "avg_confidence_score": pd.NA,
                }
            )
        else:
            rec = row.iloc[0].to_dict()
            rec["bucket_mid"] = BUCKET_MID[bucket]
            completed_rows.append(rec)
    completed = pd.DataFrame(completed_rows)

    conf_outcome_corr = matured["confidence_score"].corr(matured[return_col])
    final_outcome_corr = matured["final_score"].corr(matured[return_col])
    conf_final_corr = matured["confidence_score"].corr(matured["final_score"])

    return {
        "status": "ok",
        "horizon_days": horizon_days,
        "rows": int(len(matured)),
        "eligible_date_count": int(matured["date"].nunique()),
        "latest_date": str(matured["date"].max()),
        "summary": completed,
        "monotonic_judgment": monotonic_judgment(completed),
        "confidence_outcome_corr": float(conf_outcome_corr) if pd.notna(conf_outcome_corr) else None,
        "final_outcome_corr": float(final_outcome_corr) if pd.notna(final_outcome_corr) else None,
        "confidence_final_corr": float(conf_final_corr) if pd.notna(conf_final_corr) else None,
        "correlation_judgment": correlation_judgment(
            float(conf_outcome_corr) if pd.notna(conf_outcome_corr) else None,
            float(final_outcome_corr) if pd.notna(final_outcome_corr) else None,
        ),
    }


def build_csv(reports: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report in reports:
        if report["status"] != "ok":
            rows.append(
                {
                    "horizon_days": report["horizon_days"],
                    "confidence_bucket": "unavailable",
                    "bucket_mid": pd.NA,
                    "n": 0,
                    "avg_return": pd.NA,
                    "median_return": pd.NA,
                    "hit_rate": pd.NA,
                    "avg_mdd": pd.NA,
                    "top20_entry_rate": pd.NA,
                    "avg_final_score": pd.NA,
                    "avg_confidence_score": pd.NA,
                    "eligible_date_count": 0,
                    "latest_date": pd.NA,
                    "monotonic_judgment": "INSUFFICIENT",
                    "confidence_outcome_corr": pd.NA,
                    "final_outcome_corr": pd.NA,
                    "confidence_final_corr": pd.NA,
                    "correlation_judgment": "INSUFFICIENT",
                    "status": report["status"],
                    "note": report.get("reason", ""),
                }
            )
            continue

        for _, row in report["summary"].iterrows():
            rows.append(
                {
                    "horizon_days": report["horizon_days"],
                    "confidence_bucket": row["confidence_bucket"],
                    "bucket_mid": row["bucket_mid"],
                    "n": row["n"],
                    "avg_return": row["avg_return"],
                    "median_return": row["median_return"],
                    "hit_rate": row["hit_rate"],
                    "avg_mdd": row["avg_mdd"],
                    "top20_entry_rate": row["top20_entry_rate"],
                    "avg_final_score": row["avg_final_score"],
                    "avg_confidence_score": row["avg_confidence_score"],
                    "eligible_date_count": report["eligible_date_count"],
                    "latest_date": report["latest_date"],
                    "monotonic_judgment": report["monotonic_judgment"],
                    "confidence_outcome_corr": report["confidence_outcome_corr"],
                    "final_outcome_corr": report["final_outcome_corr"],
                    "confidence_final_corr": report["confidence_final_corr"],
                    "correlation_judgment": report["correlation_judgment"],
                    "status": report["status"],
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def build_interpretation_lines(reports: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for report in reports:
        horizon = report["horizon_days"]
        if report["status"] != "ok":
            lines.append(f"- {horizon}d: {report.get('reason', 'unavailable')}.")
            continue

        if report["monotonic_judgment"] == "GOOD":
            lines.append(f"- {horizon}d: high confidence bucket일수록 수익률과 hit rate가 함께 개선됩니다.")
        elif report["monotonic_judgment"] == "WEAK":
            lines.append(f"- {horizon}d: confidence와 outcome의 관계는 약한 수준이며 일부 bucket에서만 개선됩니다.")
        else:
            lines.append(f"- {horizon}d: confidence bucket 성과가 단조적으로 개선되지 않아 예측 신뢰도 지표로는 약합니다.")

        lines.append(
            f"- {horizon}d: corr(confidence, final_score)={_fmt(report['confidence_final_corr'])} / correlation_judgment={report['correlation_judgment']}."
        )

    lines.append("- confidence_score 이력은 research 테이블에 저장돼 있지 않아 현재 공식을 과거 scored date에 재적용해 검증했습니다.")
    return lines


def build_markdown(reports: list[dict[str, object]], excluded_dates: list[str]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_rows: list[list[object]] = []
    for report in reports:
        if report["status"] != "ok":
            summary_rows.append(
                [report["horizon_days"], "unavailable", "NA", "NA", "NA", "NA", "NA", report.get("reason", "")]
            )
        else:
            summary_rows.append(
                [
                    report["horizon_days"],
                    report["eligible_date_count"],
                    report["latest_date"],
                    report["rows"],
                    report["monotonic_judgment"],
                    _fmt(report["confidence_outcome_corr"]),
                    _fmt(report["final_outcome_corr"]),
                    _fmt(report["confidence_final_corr"]),
                ]
            )

    lines = [
        "# Confidence Calibration Report",
        "",
        f"- generated_at: {generated_at}",
        "- source: current confidence_score formula reapplied to historical scored dates with matured outcomes",
        "- note: research history does not store confidence_score, so calibration is based on reconstructed historical scores",
        "- confidence_buckets: 0-20, 20-40, 40-60, 60-80, 80-100",
        f"- minimum_rows_per_run: {MIN_RUN_ROWS}",
        "- recomputed_from_current_code: true",
        "",
        "## Summary",
        _markdown_table(
            summary_rows,
            [
                "horizon_days",
                "eligible_dates",
                "latest_date",
                "rows",
                "monotonic_judgment",
                "corr(conf,outcome)",
                "corr(final,outcome)",
                "corr(conf,final)",
            ],
        ),
        "",
        "## Interpretation",
        *build_interpretation_lines(reports),
        "",
    ]

    if excluded_dates:
        lines.extend(["## Excluded Dates", *[f"- {item}" for item in excluded_dates], ""])

    for report in reports:
        horizon = report["horizon_days"]
        lines.append(f"## Horizon {horizon}d")
        if report["status"] != "ok":
            lines.append(f"- status: {report['status']}")
            lines.append(f"- reason: {report.get('reason', 'unknown')}")
            lines.append("")
            continue

        lines.append(f"- eligible_date_count: {report['eligible_date_count']}")
        lines.append(f"- latest_date: {report['latest_date']}")
        lines.append(f"- monotonic_judgment: {report['monotonic_judgment']}")
        lines.append(f"- corr(confidence_score, realized_return): {_fmt(report['confidence_outcome_corr'])}")
        lines.append(f"- corr(final_score, realized_return): {_fmt(report['final_outcome_corr'])}")
        lines.append(f"- corr(confidence_score, final_score): {_fmt(report['confidence_final_corr'])}")
        lines.append(f"- correlation_judgment: {report['correlation_judgment']}")
        lines.append("")

        table_rows: list[list[object]] = []
        for _, row in report["summary"].iterrows():
            table_rows.append(
                [
                    row["confidence_bucket"],
                    int(row["n"]),
                    _fmt_pct(row["avg_return"]),
                    _fmt_pct(row["median_return"]),
                    _fmt(row["hit_rate"]),
                    _fmt_pct(row["avg_mdd"]),
                    _fmt(row["top20_entry_rate"]),
                    _fmt(row["avg_final_score"]),
                ]
            )
        lines.append(
            _markdown_table(
                table_rows,
                [
                    "confidence_bucket",
                    "n",
                    "avg_return",
                    "median_return",
                    "hit_rate",
                    "avg_mdd",
                    "top20_entry_rate",
                    "avg_final_score",
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Calibration Judgment",
            "- high confidence bucket이 low confidence bucket보다 일관되게 좋아야 calibration이 유효하다고 봅니다.",
            "- monotonicity가 약하거나 corr(confidence_score, realized_return)가 낮으면 confidence 정의 개선이 필요합니다.",
            "- corr(confidence_score, final_score)가 지나치게 높으면 confidence가 별도 정보보다 점수 재포장에 가까울 수 있습니다.",
            "",
            "## Improvement Ideas",
            "- confidence_score 이력을 research.prediction_history 또는 ranking_history에 직접 저장해 재구성 오차를 없앱니다.",
            "- historical hit rate와 realized drawdown 안정성 같은 outcome-linked 축을 confidence 구성요소에 추가합니다.",
            "- low bucket 표본이 비어 있으면 절대 구간 bucket과 quantile bucket 점검을 함께 돌립니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_live_grade_markdown(path: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not path.exists():
        return "\n".join(
            [
                "# Live Confidence Grade Report",
                "",
                f"- generated_at: {generated_at}",
                "- status: unavailable",
                f"- reason: missing {path}",
                "",
            ]
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    rules = payload.get("rules", {}) if isinstance(payload.get("rules"), dict) else {}
    buckets = payload.get("buckets", []) if isinstance(payload.get("buckets"), list) else []
    grade_counts = summary.get("grade_counts", {}) if isinstance(summary.get("grade_counts"), dict) else {}
    bucket_rows: list[list[object]] = []
    for item in buckets:
        if not isinstance(item, dict):
            continue
        policy = item.get("execution_policy", {}) if isinstance(item.get("execution_policy"), dict) else {}
        bucket_rows.append(
            [
                item.get("bucket_label"),
                int(pd.to_numeric(item.get("sample_count"), errors="coerce") or 0),
                int(pd.to_numeric(item.get("performance_count"), errors="coerce") or 0),
                _fmt_pct(item.get("avg_return")),
                _fmt_pct(item.get("avg_excess_return")),
                _fmt_pct(item.get("recent_avg_return")),
                _fmt(item.get("hit_rate")),
                item.get("live_confidence_grade"),
                item.get("grade_reason"),
                "Y" if bool(policy.get("entry_allowed")) else "N",
                _fmt(policy.get("weight_scale"), 2),
                str(policy.get("mode") or ""),
            ]
        )

    lines = [
        "# Live Confidence Grade Report",
        "",
        f"- generated_at: {payload.get('generated_at') or generated_at}",
        f"- version: {payload.get('version')}",
        f"- source_table: {payload.get('source_table')}",
        f"- recent_trade_count: {payload.get('recent_trade_count')}",
        f"- min_bucket_rows: {payload.get('min_bucket_rows')}",
        f"- review_rows_with_confidence: {summary.get('review_rows_with_confidence')}",
        f"- review_rows_with_strategy_return: {summary.get('review_rows_with_strategy_return')}",
        f"- review_rows_with_excess_return: {summary.get('review_rows_with_excess_return')}",
        "",
        "## Rules",
        "",
        f"- sample_lt_20_max_grade: {rules.get('sample_lt_20_max_grade')}",
        f"- excess_return_lt_minus_1pct: {rules.get('excess_return_lt_minus_1pct')}",
        f"- recent_10_trade_return_lt_minus_2pct: {rules.get('recent_10_trade_return_lt_minus_2pct')}",
        f"- missing_performance_info: {rules.get('missing_performance_info')}",
        "",
        "## Grade Counts",
        "",
        _markdown_table([[key, value] for key, value in grade_counts.items()] or [["(none)", 0]], ["grade", "count"]),
        "",
        "## Bucket Detail",
        "",
        _markdown_table(
            bucket_rows or [["NA", 0, 0, "NA", "NA", "NA", "NA", "C", "no_bucket_data", "N", "0.20", "watch_only"]],
            ["bucket", "samples", "perf_rows", "avg_return", "avg_excess", "recent10_return", "hit_rate", "grade", "reason", "entry_allowed", "weight_scale", "mode"],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    try:
        ranked, excluded_dates = build_rank_history()
        reports = [build_horizon_report(ranked, horizon) for horizon in HORIZONS]
        csv_df = build_csv(reports)
        OUTPUT_MD.write_text(build_markdown(reports, excluded_dates), encoding="utf-8")
        csv_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"[ok] wrote {OUTPUT_MD}")
        print(f"[ok] wrote {OUTPUT_CSV}")
    except Exception as exc:
        OUTPUT_MD.write_text(
            "\n".join(
                [
                    "# Confidence Calibration Report",
                    "",
                    f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "- status: unavailable",
                    f"- reason: {exc}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"horizon_days": horizon, "confidence_bucket": "unavailable", "status": "unavailable", "note": str(exc)} for horizon in HORIZONS]
        ).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"[warn] wrote fallback {OUTPUT_MD}")
        print(f"[warn] wrote fallback {OUTPUT_CSV}")
    LIVE_GRADE_MD.parent.mkdir(parents=True, exist_ok=True)
    LIVE_GRADE_MD.write_text(build_live_grade_markdown(LIVE_GRADE_JSON), encoding="utf-8")
    print(f"[ok] wrote {LIVE_GRADE_MD}")


if __name__ == "__main__":
    main()
