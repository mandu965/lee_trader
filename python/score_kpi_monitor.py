from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from payload_store import upsert_json_payload
from build_walk_forward_score_validation_from_runs import (
    aggregate_selection_summary,
    cohort_selection_metrics,
    load_joined_rows,
    load_matured_run_ids,
)


INPUT_CSV = Path("data/ranking_final.csv")
CONFIDENCE_CALIBRATION_JSON = Path("data/confidence_calibration_map.json")
OUTPUT_MD = Path("outputs/score_kpi_monitor.md")
OUTPUT_JSON = Path("data/score_kpi_monitor.json")
TOP_N = 20
WALKFORWARD_HORIZON = 60


@dataclass(frozen=True)
class ThresholdBand:
    good: float
    watch: float
    direction: str


THRESHOLDS = {
    "overlap_final_ret_top20": ThresholdBand(good=0.50, watch=0.42, direction="higher_better"),
    "overlap_final_prob_top20": ThresholdBand(good=0.45, watch=0.35, direction="higher_better"),
    "overlap_final_tech_top20": ThresholdBand(good=0.25, watch=0.18, direction="higher_better"),
    "corr_final_risk_penalty_abs": ThresholdBand(good=0.50, watch=0.60, direction="lower_better"),
    "corr_final_safety_abs": ThresholdBand(good=0.30, watch=0.40, direction="lower_better"),
    "corr_final_ret_score": ThresholdBand(good=0.80, watch=0.70, direction="higher_better"),
    "corr_final_prob_score": ThresholdBand(good=0.65, watch=0.55, direction="higher_better"),
    "corr_final_tech_score": ThresholdBand(good=0.10, watch=0.03, direction="higher_better"),
    "top20_mean_confidence_score": ThresholdBand(good=80.0, watch=72.0, direction="higher_better"),
    "top20_top_driver_share": ThresholdBand(good=0.32, watch=0.40, direction="lower_better"),
    "walkforward_top20_avg_return_60d": ThresholdBand(good=0.20, watch=0.08, direction="higher_better"),
    "confidence_high_bucket_hit_rate_60d": ThresholdBand(good=0.60, watch=0.52, direction="higher_better"),
    "confidence_calibration_usable_bucket_count": ThresholdBand(good=2.0, watch=1.0, direction="higher_better"),
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_ranking(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"ranking CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def latest_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    latest_date = str(df["date"].dropna().max()) if "date" in df.columns and df["date"].notna().any() else "all_dates"
    latest = df.loc[df["date"] == latest_date].copy() if latest_date != "all_dates" else df.copy()
    latest["final_score"] = pd.to_numeric(latest["final_score"], errors="coerce")
    return latest.sort_values(["final_score"], ascending=[False]), latest_date


def safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    if left not in df.columns or right not in df.columns:
        return float("nan")
    sample = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sample) < 2:
        return float("nan")
    return float(sample[left].corr(sample[right]))


def overlap_ratio(df: pd.DataFrame, score_col: str, top_n: int = TOP_N) -> float:
    if score_col not in df.columns:
        return float("nan")
    final_set = set(df.sort_values(["final_score"], ascending=[False]).head(top_n)["code"].astype(str))
    comp_set = set(df.sort_values([score_col], ascending=[False]).head(top_n)["code"].astype(str))
    return len(final_set & comp_set) / float(top_n) if top_n else float("nan")


def dominant_driver_frequency(df: pd.DataFrame, top_n: int = TOP_N) -> tuple[dict[str, int], float]:
    top = df.head(top_n)
    values: list[str] = []
    for col in ["score_driver_1", "score_driver_2", "score_driver_3"]:
        if col in top.columns:
            values.extend(top[col].dropna().astype(str).tolist())
    counts = Counter(values)
    total = sum(counts.values())
    top_share = (max(counts.values()) / total) if total and counts else float("nan")
    return dict(counts.most_common(10)), top_share


def evaluate_status(value: float, band: ThresholdBand) -> str:
    if pd.isna(value):
        return "ALERT"
    if band.direction == "higher_better":
        if value >= band.good:
            return "GOOD"
        if value >= band.watch:
            return "WATCH"
        return "ALERT"
    if value <= band.good:
        return "GOOD"
    if value <= band.watch:
        return "WATCH"
    return "ALERT"


def format_value(metric: str, value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "NA"
    if "confidence_score" in metric or "bucket_count" in metric:
        return f"{value:.2f}"
    return f"{value:.4f}"


def calculate_ranking_kpis(latest: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    driver_counts, top_driver_share = dominant_driver_frequency(latest, TOP_N)
    metrics = {
        "overlap_final_ret_top20": overlap_ratio(latest, "ret_score"),
        "overlap_final_prob_top20": overlap_ratio(latest, "prob_score"),
        "overlap_final_tech_top20": overlap_ratio(latest, "tech_score"),
        "corr_final_risk_penalty_abs": abs(safe_corr(latest, "final_score", "risk_penalty")),
        "corr_final_safety_abs": abs(safe_corr(latest, "final_score", "safety_score")),
        "corr_final_ret_score": safe_corr(latest, "final_score", "ret_score"),
        "corr_final_prob_score": safe_corr(latest, "final_score", "prob_score"),
        "corr_final_tech_score": safe_corr(latest, "final_score", "tech_score"),
        "top20_mean_confidence_score": float(pd.to_numeric(latest.head(TOP_N).get("confidence_score"), errors="coerce").mean()),
        "top20_top_driver_share": top_driver_share,
    }

    rows: list[dict[str, object]] = []
    for name, value in metrics.items():
        band = THRESHOLDS[name]
        rows.append(
            {
                "category": "ranking",
                "metric": name,
                "value": value,
                "value_display": format_value(name, value),
                "good_threshold": f"{band.good:.4f}",
                "watch_threshold": f"{band.watch:.4f}",
                "direction": band.direction,
                "status": evaluate_status(value, band),
            }
        )
    return rows, driver_counts


def calculate_walkforward_kpis() -> list[dict[str, object]]:
    matured_runs = load_matured_run_ids()
    joined = load_joined_rows(matured_runs)
    required_columns = {"horizon_days", "run_id", "as_of_date", "rank", "final_score", "realized_return", "realized_mdd"}
    if joined.empty or not required_columns.issubset(set(joined.columns)):
        selection_rows = pd.DataFrame()
    else:
        selection_rows = pd.concat(
            [
                cohort_selection_metrics(joined, "top20", 20),
                cohort_selection_metrics(joined, "top50", 50),
                cohort_selection_metrics(joined, "universe", None),
            ],
            ignore_index=True,
        )
    selection_summary = aggregate_selection_summary(selection_rows)
    top20 = selection_summary.loc[
        (selection_summary["horizon_days"] == WALKFORWARD_HORIZON)
        & (selection_summary["selection"] == "top20")
    ]
    value = float(top20["avg_return"].iloc[0]) if not top20.empty else float("nan")
    band = THRESHOLDS["walkforward_top20_avg_return_60d"]
    return [
        {
            "category": "walkforward",
            "metric": "walkforward_top20_avg_return_60d",
            "value": value,
            "value_display": format_value("walkforward_top20_avg_return_60d", value),
            "good_threshold": f"{band.good:.4f}",
            "watch_threshold": f"{band.watch:.4f}",
            "direction": band.direction,
            "status": evaluate_status(value, band),
        }
    ]


def calculate_confidence_kpis() -> list[dict[str, object]]:
    value = float("nan")
    usable_bucket_count = float("nan")
    source_mode = "unknown"

    if CONFIDENCE_CALIBRATION_JSON.exists():
        payload = json.loads(CONFIDENCE_CALIBRATION_JSON.read_text(encoding="utf-8").replace("NaN", "null"))
        summary = payload.get("summary", {}) or {}
        bucket_rows = pd.DataFrame(payload.get("bucket_map", []))
        source_mode = str(summary.get("source_mode") or "unknown")
        usable_bucket_count = float(pd.to_numeric(summary.get("usable_bucket_count"), errors="coerce"))
        if not bucket_rows.empty:
            stable = bucket_rows.loc[bucket_rows["status"].isin(["stable", "thin_sample"])].copy()
            stable["avg_raw_confidence_score"] = pd.to_numeric(stable["avg_raw_confidence_score"], errors="coerce")
            stable["hit_rate"] = pd.to_numeric(stable["hit_rate"], errors="coerce")
            stable = stable.sort_values(["avg_raw_confidence_score"], ascending=[False])
            if not stable.empty:
                value = float(stable["hit_rate"].iloc[0])

    rows = []
    band = THRESHOLDS["confidence_high_bucket_hit_rate_60d"]
    rows.append(
        {
            "category": "confidence",
            "metric": "confidence_high_bucket_hit_rate_60d",
            "value": value,
            "value_display": format_value("confidence_high_bucket_hit_rate_60d", value),
            "good_threshold": f"{band.good:.4f}",
            "watch_threshold": f"{band.watch:.4f}",
            "direction": band.direction,
            "status": evaluate_status(value, band),
        }
    )
    band = THRESHOLDS["confidence_calibration_usable_bucket_count"]
    rows.append(
        {
            "category": "confidence",
            "metric": "confidence_calibration_usable_bucket_count",
            "value": usable_bucket_count,
            "value_display": format_value("confidence_calibration_usable_bucket_count", usable_bucket_count),
            "good_threshold": f"{band.good:.4f}",
            "watch_threshold": f"{band.watch:.4f}",
            "direction": band.direction,
            "status": evaluate_status(usable_bucket_count, band),
        }
    )
    rows.append(
        {
            "category": "confidence",
            "metric": "confidence_calibration_source_mode",
            "value": source_mode,
            "value_display": source_mode,
            "good_threshold": "-",
            "watch_threshold": "-",
            "direction": "informational",
            "status": "GOOD" if source_mode != "unknown" else "ALERT",
        }
    )
    return rows


def metadata(df: pd.DataFrame) -> dict[str, str]:
    latest_date = "NA"
    if "date" in df.columns and df["date"].notna().any():
        latest_date = str(df["date"].dropna().max())
    score_formula_version = "NA"
    if "score_formula_version" in df.columns and df["score_formula_version"].notna().any():
        score_formula_version = str(df["score_formula_version"].dropna().iloc[0])
    stat = INPUT_CSV.stat()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_date": latest_date,
        "score_formula_version": score_formula_version,
        "source_ranking_file": f"{INPUT_CSV.name}; rows={len(df)}; modified_at={datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        "recomputed_from_current_code": "true",
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_threshold_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": key,
                "good_threshold": f"{band.good:.4f}",
                "watch_threshold": f"{band.watch:.4f}",
                "direction": band.direction,
            }
            for key, band in THRESHOLDS.items()
        ]
    )


def build_driver_frequency_table(driver_counts: dict[str, int]) -> pd.DataFrame:
    rows = [{"driver": key, "count": value} for key, value in driver_counts.items()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["driver", "count"])


def collect_kpis(latest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows, driver_counts = calculate_ranking_kpis(latest)
    walkforward_rows = calculate_walkforward_kpis()
    confidence_rows = calculate_confidence_kpis()
    kpi_df = pd.DataFrame(ranking_rows + walkforward_rows + confidence_rows)
    driver_df = build_driver_frequency_table(driver_counts)
    return kpi_df, driver_df


def overall_status(kpi_df: pd.DataFrame) -> str:
    if (kpi_df["status"] == "ALERT").any():
        return "ALERT"
    if (kpi_df["status"] == "WATCH").any():
        return "WATCH"
    return "GOOD"


def build_markdown(df: pd.DataFrame, latest: pd.DataFrame, latest_date: str) -> str:
    meta = metadata(df)
    kpi_df, driver_df = collect_kpis(latest)
    threshold_df = build_threshold_table()
    status = overall_status(kpi_df)

    snapshot_df = kpi_df[
        ["category", "metric", "value_display", "good_threshold", "watch_threshold", "direction", "status"]
    ].rename(columns={"value_display": "value"})

    lines = [
        "# Score KPI Monitor",
        "",
        "## KPI Guide",
        "- GOOD: 현재 운영 신호가 기준 범위 안에 있습니다.",
        "- WATCH: 추가 점검이 필요한 경계 구간입니다.",
        "- ALERT: 즉시 원인 점검이 필요한 상태입니다.",
        "- `overlap/corr` 계열은 최신 ranking slice 기준입니다.",
        f"- `walkforward_top20_avg_return_60d`는 matured walk-forward run 기준 {WALKFORWARD_HORIZON}일 top20 평균 수익률입니다.",
        "- `confidence_high_bucket_hit_rate_60d`는 현재 calibration에서 가장 높은 usable bucket의 hit rate입니다.",
        "- `confidence_calibration_usable_bucket_count`는 현재 calibration에서 실제로 해석 가능한 bucket 수입니다.",
        "- `confidence_calibration_source_mode`는 현재 confidence 해석 소스입니다.",
        "",
        "## Metadata",
        *[f"- {key}: {value}" for key, value in meta.items()],
        "",
        "## Overall Status",
        f"- overall_status: {status}",
        f"- latest_date: {latest_date}",
        f"- top_n: {TOP_N}",
        "",
        "## Thresholds",
        dataframe_to_markdown(threshold_df),
        "",
        "## KPI Snapshot",
        dataframe_to_markdown(snapshot_df),
        "",
        "## Dominant Driver Frequency",
        dataframe_to_markdown(driver_df),
        "",
        "## Operator Notes",
        "- `corr_final_safety_abs`와 driver frequency는 방어 편향 조기 탐지용입니다.",
        "- `overlap_final_tech_top20`와 `corr_final_tech_score`는 tech 약화 탐지용입니다.",
        "- `top20_mean_confidence_score`, `confidence_high_bucket_hit_rate_60d`, `confidence_calibration_usable_bucket_count`는 confidence 무력화 탐지용입니다.",
        "- confidence source가 `walkforward_provisional`이면 정식 운영 calibration이 아니라 보조 해석 단계로 읽어야 합니다.",
        "- daily pipeline 이후 `python python/score_kpi_monitor.py`만 실행하면 현재 상태를 다시 점검할 수 있습니다.",
    ]
    return "\n".join(lines) + "\n"


def clean_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): clean_scalar(value) for key, value in row.items()})
    return records


def build_json_payload(df: pd.DataFrame, latest: pd.DataFrame, latest_date: str) -> dict[str, object]:
    meta = metadata(df)
    kpi_df, driver_df = collect_kpis(latest)
    threshold_df = build_threshold_table()
    status = overall_status(kpi_df)
    kpi_records = dataframe_records(kpi_df)
    metric_map = {str(item["metric"]): item for item in kpi_records}
    return {
        "summary": {
            "overall_status": status,
            "latest_date": latest_date,
            "top_n": TOP_N,
            "walkforward_horizon": WALKFORWARD_HORIZON,
        },
        "metadata": meta,
        "thresholds": dataframe_records(threshold_df),
        "kpis": kpi_records,
        "metric_map": metric_map,
        "dominant_driver_frequency": dataframe_records(driver_df),
    }


def main() -> None:
    setup_logging()
    df = load_ranking(INPUT_CSV)
    latest, latest_date = latest_slice(df)
    payload = build_json_payload(df, latest, latest_date)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_markdown(df, latest, latest_date), encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_json_payload(
        "score_kpi_monitor",
        payload,
        asof_date=payload.get("summary", {}).get("latest_date"),
        source_path=OUTPUT_JSON,
    )
    logging.info("Saved score KPI monitor: %s", OUTPUT_MD.resolve())


if __name__ == "__main__":
    main()
