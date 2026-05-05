from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine
from outcome_maturity import attach_forward_outcomes, load_price_history
from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
CURRENT_RANKING_CSV = DATA_DIR / "ranking_final.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
OUT_CSV = DATA_DIR / "confidence_calibration_operational.csv"
OUT_MD = OUTPUT_DIR / "confidence_calibration_operational_report.md"
OUT_LIVE_GRADE_JSON = OUTPUT_DIR / "confidence_live_grade_map.json"
PROVISIONAL_MAP_CSV = DATA_DIR / "confidence_calibration_map.csv"

HORIZONS = [5, 20, 60, 90]
MIN_BUCKET_ROWS = 20
RECENT_TRADE_COUNT = 10
BUCKETS = [
    (0, 20, "0-20"),
    (20, 40, "20-40"),
    (40, 60, "40-60"),
    (60, 80, "60-80"),
    (80, 100, "80-100"),
]
CONFIDENCE_CALIBRATION_VERSION = str(
    get_production_config_value(["metadata", "confidence_calibration_version"], "confidence_four_axis_v1")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate operational confidence_score using stored ranking snapshots.")
    p.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    p.add_argument("--ranking-current-csv", type=Path, default=CURRENT_RANKING_CSV)
    p.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    p.add_argument("--min-bucket-rows", type=int, default=int(get_production_config_value(["confidence_calibration", "min_bucket_rows_operational"], MIN_BUCKET_ROWS)))
    p.add_argument("--provisional-map-csv", type=Path, default=PROVISIONAL_MAP_CSV)
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--out-live-grade-json", type=Path, default=OUT_LIVE_GRADE_JSON)
    p.add_argument("--recent-trade-count", type=int, default=RECENT_TRADE_COUNT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


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
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def load_operational_history(history_dir: Path, current_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    snapshot_files: list[str] = []

    if history_dir.exists():
        for path in sorted(history_dir.glob("*_ranking_final.csv")):
            df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            df["snapshot_file"] = path.name
            frames.append(df)
            snapshot_files.append(path.name)

    if current_csv.exists():
        current = pd.read_csv(current_csv, dtype={"code": str}, low_memory=False)
        current["snapshot_file"] = current_csv.name
        frames.append(current)
        snapshot_files.append(current_csv.name)

    if not frames:
        raise FileNotFoundError("No operational ranking snapshots found.")

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["confidence_score"] = pd.to_numeric(df.get("confidence_score"), errors="coerce")
    df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
    df["rank_final"] = pd.to_numeric(df.get("rank_final"), errors="coerce")
    df["regime"] = df.get("regime", "").fillna("").astype(str)
    df = df.dropna(subset=["date", "code", "confidence_score"]).copy()
    df["snapshot_priority"] = df["snapshot_file"].eq(current_csv.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    meta = {
        "source_mode": "operational_snapshot_history",
        "source_label": "stored operational confidence from ranking snapshots",
        "history_row_count": int(len(df)),
        "history_date_count": int(df["date"].nunique()),
        "history_start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else "NA",
        "history_end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else "NA",
        "source_item_count": len(sorted(set(snapshot_files))),
        "source_items": sorted(set(snapshot_files)),
    }
    return df, meta


def attach_realized_returns(df: pd.DataFrame, prices_csv: Path) -> pd.DataFrame:
    prices = load_price_history(prices_csv=prices_csv)
    work = df.copy()
    for horizon in HORIZONS:
        outcome = attach_forward_outcomes(prices, horizon_days=horizon).rename(
            columns={
                "realized_return": f"realized_return_{horizon}d",
                "realized_mdd": f"realized_mdd_{horizon}d",
            }
        )
        outcome["date"] = pd.to_datetime(outcome["date"], errors="coerce").dt.normalize()
        work = work.merge(outcome, on=["code", "date"], how="left")
    return work


def bucketize(score: pd.Series) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce")
    out = pd.Series(pd.NA, index=score.index, dtype="object")
    for lo, hi, label in BUCKETS:
        if lo == 0:
            mask = values.between(lo, hi, inclusive="both")
        else:
            mask = (values > lo) & (values <= hi)
        out = out.mask(mask, label)
    return out


def _safe_float(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def load_live_trade_review() -> tuple[pd.DataFrame, dict[str, object]]:
    query = text(
        """
        SELECT
            review_date,
            confidence_score,
            strategy_return,
            excess_return,
            review_status
        FROM research.live_trade_review
        """
    )
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as exc:
        return pd.DataFrame(), {"available": False, "reason": str(exc)}

    if df.empty:
        return pd.DataFrame(), {"available": True, "row_count": 0, "performance_row_count": 0}

    work = df.copy()
    work["review_date"] = pd.to_datetime(work.get("review_date"), errors="coerce").dt.normalize()
    work["confidence_score"] = pd.to_numeric(work.get("confidence_score"), errors="coerce")
    work["strategy_return"] = pd.to_numeric(work.get("strategy_return"), errors="coerce")
    work["excess_return"] = pd.to_numeric(work.get("excess_return"), errors="coerce")
    work["review_status"] = work.get("review_status", "").fillna("").astype(str)
    work = work.dropna(subset=["confidence_score"]).copy()
    work["confidence_bucket"] = bucketize(work["confidence_score"])
    work = work.dropna(subset=["confidence_bucket"]).copy()
    meta = {
        "available": True,
        "row_count": int(len(work)),
        "performance_row_count": int(work["strategy_return"].notna().sum()),
        "excess_row_count": int(work["excess_return"].notna().sum()),
        "latest_review_date": work["review_date"].max().strftime("%Y-%m-%d") if work["review_date"].notna().any() else None,
    }
    return work.reset_index(drop=True), meta


def classify_live_grade(
    *,
    sample_count: int,
    performance_count: int,
    excess_count: int,
    avg_return: float | None,
    avg_excess_return: float | None,
    recent_avg_return: float | None,
    hit_rate: float | None,
    min_bucket_rows: int,
) -> tuple[str, str]:
    if avg_excess_return is not None and avg_excess_return < -0.01:
        return "D", "excess_return_below_minus_1pct"
    if recent_avg_return is not None and recent_avg_return < -0.02:
        return "D", "recent_10_trade_return_below_minus_2pct"
    if performance_count <= 0 or avg_return is None or hit_rate is None:
        return "C", "performance_data_missing"
    if sample_count < min_bucket_rows:
        return "C", "sample_below_20"
    if excess_count <= 0 or avg_excess_return is None:
        return "C", "excess_return_missing"
    if hit_rate >= 0.60 and avg_return >= 0.01 and avg_excess_return >= 0.005 and (recent_avg_return is None or recent_avg_return >= 0):
        return "A", "sample_sufficient_and_live_performance_strong"
    if hit_rate >= 0.52 and avg_return >= 0 and avg_excess_return >= 0 and (recent_avg_return is None or recent_avg_return >= -0.005):
        return "B", "sample_sufficient_and_live_performance_acceptable"
    return "C", "live_performance_unclear"


def grade_policy_for(grade: str) -> dict[str, object]:
    normalized = (grade or "C").upper()
    if normalized == "A":
        return {"entry_allowed": True, "weight_scale": 1.0, "mode": "standard"}
    if normalized == "B":
        return {"entry_allowed": True, "weight_scale": 0.5, "mode": "scaled"}
    if normalized == "D":
        return {"entry_allowed": False, "weight_scale": 0.0, "mode": "blocked"}
    return {"entry_allowed": False, "weight_scale": 0.2, "mode": "watch_only"}


def build_live_grade_map(review_df: pd.DataFrame, *, min_bucket_rows: int, recent_trade_count: int) -> dict[str, object]:
    bucket_rows: list[dict[str, object]] = []
    for lo, hi, label in BUCKETS:
        bucket = review_df.loc[review_df.get("confidence_bucket", pd.Series(index=review_df.index)).eq(label)].copy()
        perf = bucket.loc[pd.to_numeric(bucket.get("strategy_return"), errors="coerce").notna()].copy()
        recent_perf = perf.sort_values(["review_date"], ascending=False).head(recent_trade_count)
        sample_count = int(len(bucket))
        performance_count = int(len(perf))
        excess_count = int(bucket["excess_return"].notna().sum()) if "excess_return" in bucket.columns else 0
        avg_return = _safe_float(perf["strategy_return"].mean()) if not perf.empty else None
        avg_excess_return = _safe_float(bucket["excess_return"].mean()) if excess_count > 0 else None
        recent_avg_return = _safe_float(recent_perf["strategy_return"].mean()) if not recent_perf.empty else None
        hit_rate = _safe_float((perf["strategy_return"] > 0).mean()) if not perf.empty else None
        calibrated_confidence = float(hit_rate * 100.0) if hit_rate is not None else None
        grade, reason = classify_live_grade(
            sample_count=sample_count,
            performance_count=performance_count,
            excess_count=excess_count,
            avg_return=avg_return,
            avg_excess_return=avg_excess_return,
            recent_avg_return=recent_avg_return,
            hit_rate=hit_rate,
            min_bucket_rows=min_bucket_rows,
        )
        bucket_rows.append(
            {
                "bucket_label": label,
                "bucket_low": lo,
                "bucket_high": hi,
                "sample_count": sample_count,
                "performance_count": performance_count,
                "excess_count": excess_count,
                "avg_return": avg_return,
                "avg_excess_return": avg_excess_return,
                "recent_avg_return": recent_avg_return,
                "hit_rate": hit_rate,
                "calibrated_confidence": calibrated_confidence,
                "live_confidence_grade": grade,
                "grade_reason": reason,
                "execution_policy": grade_policy_for(grade),
            }
        )

    grade_counts = pd.Series([row["live_confidence_grade"] for row in bucket_rows]).value_counts().to_dict()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": CONFIDENCE_CALIBRATION_VERSION,
        "source_table": "research.live_trade_review",
        "recent_trade_count": int(recent_trade_count),
        "min_bucket_rows": int(min_bucket_rows),
        "rules": {
            "sample_lt_20_max_grade": "C",
            "excess_return_lt_minus_1pct": "D",
            "recent_10_trade_return_lt_minus_2pct": "D",
            "missing_performance_info": "C",
            "grade_A_policy": grade_policy_for("A"),
            "grade_B_policy": grade_policy_for("B"),
            "grade_C_policy": grade_policy_for("C"),
            "grade_D_policy": grade_policy_for("D"),
        },
        "default_bucket": {
            "live_confidence_grade": "C",
            "grade_reason": "no_bucket_data",
            "execution_policy": grade_policy_for("C"),
        },
        "summary": {
            "bucket_count": len(bucket_rows),
            "grade_counts": grade_counts,
            "review_rows_with_confidence": int(len(review_df)),
            "review_rows_with_strategy_return": int(review_df["strategy_return"].notna().sum()) if "strategy_return" in review_df.columns else 0,
            "review_rows_with_excess_return": int(review_df["excess_return"].notna().sum()) if "excess_return" in review_df.columns else 0,
        },
        "buckets": bucket_rows,
    }


def run_self_test() -> dict[str, object]:
    cases = [
        {
            "name": "zero_sample_defaults_to_c",
            "kwargs": dict(sample_count=0, performance_count=0, excess_count=0, avg_return=None, avg_excess_return=None, recent_avg_return=None, hit_rate=None, min_bucket_rows=20),
            "expected": ("C", "performance_data_missing"),
        },
        {
            "name": "thin_sample_caps_at_c",
            "kwargs": dict(sample_count=19, performance_count=19, excess_count=19, avg_return=0.02, avg_excess_return=0.01, recent_avg_return=0.01, hit_rate=0.7, min_bucket_rows=20),
            "expected": ("C", "sample_below_20"),
        },
        {
            "name": "excess_return_below_threshold_is_d",
            "kwargs": dict(sample_count=25, performance_count=25, excess_count=25, avg_return=0.01, avg_excess_return=-0.011, recent_avg_return=0.005, hit_rate=0.6, min_bucket_rows=20),
            "expected": ("D", "excess_return_below_minus_1pct"),
        },
        {
            "name": "recent_10_trade_return_below_threshold_is_d",
            "kwargs": dict(sample_count=25, performance_count=25, excess_count=25, avg_return=0.01, avg_excess_return=0.0, recent_avg_return=-0.021, hit_rate=0.6, min_bucket_rows=20),
            "expected": ("D", "recent_10_trade_return_below_minus_2pct"),
        },
        {
            "name": "strong_live_performance_is_a",
            "kwargs": dict(sample_count=30, performance_count=30, excess_count=30, avg_return=0.015, avg_excess_return=0.007, recent_avg_return=0.004, hit_rate=0.65, min_bucket_rows=20),
            "expected": ("A", "sample_sufficient_and_live_performance_strong"),
        },
    ]
    results: list[dict[str, object]] = []
    passed = True
    for case in cases:
        actual = classify_live_grade(**case["kwargs"])
        ok = actual == case["expected"]
        passed = passed and ok
        results.append({"name": case["name"], "expected": list(case["expected"]), "actual": list(actual), "passed": ok})
    return {"passed": passed, "scenario_count": len(results), "results": results}


def load_provisional_map(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        return pd.DataFrame(), {"available": False}
    df = pd.read_csv(path, low_memory=False)
    for col in ["bucket_label", "calibrated_confidence_score", "avg_return", "hit_rate", "rows"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df, {"available": True, "row_count": int(len(df))}


def build_bucket_rows(
    matured: pd.DataFrame,
    *,
    horizon_days: int,
    min_bucket_rows: int,
    provisional_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ret_col = f"realized_return_{horizon_days}d"
    mdd_col = f"realized_mdd_{horizon_days}d"
    matured = matured.loc[pd.to_numeric(matured[ret_col], errors="coerce").notna()].copy()
    matured["positive_hit"] = pd.to_numeric(matured[ret_col], errors="coerce") > 0
    matured["confidence_bucket"] = bucketize(matured["confidence_score"])

    rows: list[dict[str, object]] = []
    for lo, hi, label in BUCKETS:
        bucket = matured.loc[matured["confidence_bucket"] == label].copy()
        n = int(len(bucket))
        status = "stable" if n >= min_bucket_rows else ("thin_sample" if n > 0 else "empty")
        hit_rate = float(bucket["positive_hit"].mean()) if n else None
        avg_return = float(pd.to_numeric(bucket[ret_col], errors="coerce").mean()) if n else None
        avg_mdd = float(pd.to_numeric(bucket[mdd_col], errors="coerce").mean()) if n else None
        avg_conf = float(pd.to_numeric(bucket["confidence_score"], errors="coerce").mean()) if n else None
        median_return = float(pd.to_numeric(bucket[ret_col], errors="coerce").median()) if n else None
        calibrated = hit_rate * 100.0 if hit_rate is not None else None

        provisional_row = provisional_map.loc[provisional_map["bucket_label"] == label].head(1) if not provisional_map.empty else pd.DataFrame()
        provisional_calibrated = None
        provisional_hit = None
        provisional_rows = None
        if not provisional_row.empty:
            provisional_calibrated = pd.to_numeric(provisional_row["calibrated_confidence_score"], errors="coerce").iloc[0]
            provisional_hit = pd.to_numeric(provisional_row["hit_rate"], errors="coerce").iloc[0]
            provisional_rows = pd.to_numeric(provisional_row["rows"], errors="coerce").iloc[0]

        rows.append(
            {
                "source_mode": "operational",
                "confidence_calibration_version": CONFIDENCE_CALIBRATION_VERSION,
                "horizon_days": horizon_days,
                "bucket_label": label,
                "bucket_low": lo,
                "bucket_high": hi,
                "bucket_mid": float((lo + hi) / 2),
                "rows": n,
                "avg_raw_confidence_score": avg_conf,
                "avg_return": avg_return,
                "median_return": median_return,
                "hit_rate": hit_rate,
                "avg_mdd": avg_mdd,
                "calibrated_confidence_score": calibrated,
                "calibrated_confidence_band": (
                    "high" if calibrated is not None and calibrated >= 80
                    else "medium" if calibrated is not None and calibrated >= 60
                    else "low" if calibrated is not None
                    else "unknown"
                ),
                "status": status,
                "provisional_calibrated_confidence_score": provisional_calibrated,
                "provisional_hit_rate": provisional_hit,
                "provisional_rows": provisional_rows,
                "operational_minus_provisional": (
                    float(calibrated - provisional_calibrated)
                    if calibrated is not None and provisional_calibrated is not None and pd.notna(provisional_calibrated)
                    else None
                ),
            }
        )

    bucket_df = pd.DataFrame(rows)
    valid = bucket_df.loc[bucket_df["status"].isin(["stable", "thin_sample"])].copy()
    summary = {
        "horizon_days": horizon_days,
        "matured_row_count": int(len(matured)),
        "usable_bucket_count": int(len(valid)),
        "monotonic_hit_rate": bool(valid["hit_rate"].dropna().is_monotonic_increasing) if not valid.empty else False,
        "monotonic_avg_return": bool(valid["avg_return"].dropna().is_monotonic_increasing) if not valid.empty else False,
        "spearman_confidence_vs_return": float(matured["confidence_score"].corr(pd.to_numeric(matured[ret_col], errors="coerce"), method="spearman")) if len(matured) >= 2 else None,
        "spearman_confidence_vs_hit": float(matured["confidence_score"].corr(matured["positive_hit"].astype(int), method="spearman")) if len(matured) >= 2 else None,
        "bucket_sample_warning": bool((valid["rows"] < min_bucket_rows).any()) if not valid.empty else True,
        "judgment": (
            "insufficient_operational_history" if len(matured) == 0
            else "thin_operational_history" if int(len(valid)) < 2 or bool((valid["rows"] < min_bucket_rows).any())
            else "operational_monotonic" if bool(valid["hit_rate"].dropna().is_monotonic_increasing)
            else "operational_non_monotonic"
        ),
    }
    return bucket_df, summary


def build_calibration_table(history: pd.DataFrame, min_bucket_rows: int, provisional_map: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for horizon in HORIZONS:
        bucket_df, summary = build_bucket_rows(
            history,
            horizon_days=horizon,
            min_bucket_rows=min_bucket_rows,
            provisional_map=provisional_map,
        )
        frames.append(bucket_df)
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), summaries


def build_report(
    calibration_df: pd.DataFrame,
    summaries: list[dict[str, object]],
    meta: dict[str, object],
    provisional_meta: dict[str, object],
    min_bucket_rows: int,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Operational Confidence Calibration Report",
        "",
        f"- generated_at: {generated_at}",
        f"- source_mode: {meta.get('source_mode')}",
        f"- source_label: {meta.get('source_label')}",
        f"- confidence_calibration_version: {CONFIDENCE_CALIBRATION_VERSION}",
        f"- history_row_count: {meta.get('history_row_count')}",
        f"- history_date_count: {meta.get('history_date_count')}",
        f"- history_range: {meta.get('history_start_date')} ~ {meta.get('history_end_date')}",
        f"- min_bucket_rows: {min_bucket_rows}",
        f"- provisional_map_available: {provisional_meta.get('available', False)}",
        "",
        "## Existing Confidence Logic",
        "",
        "- `confidence_score` is an evidence-quality meta metric built from four axes in `ranking_builder.py`.",
        "- The current production formula is a weighted blend of `data_maturity_score`, `model_reliability_score`, `signal_agreement_score`, and `regime_fitness_score`.",
        "- This script does not reconstruct confidence. It validates the stored operational `confidence_score` from ranking snapshots against realized operational outcomes.",
        "",
    ]

    summary_rows = [
        [
            f"{int(item['horizon_days'])}d",
            int(item["matured_row_count"]),
            int(item["usable_bucket_count"]),
            "yes" if item["monotonic_hit_rate"] else "no",
            "yes" if item["monotonic_avg_return"] else "no",
            _fmt(item["spearman_confidence_vs_hit"]),
            _fmt(item["spearman_confidence_vs_return"]),
            item["judgment"],
        ]
        for item in summaries
    ]
    lines.extend(
        [
            "## Horizon Summary",
            "",
            _markdown_table(
                summary_rows,
                ["horizon", "matured_rows", "usable_buckets", "monotonic_hit", "monotonic_return", "spearman_conf_vs_hit", "spearman_conf_vs_return", "judgment"],
            ),
            "",
        ]
    )

    for horizon in HORIZONS:
        subset = calibration_df.loc[calibration_df["horizon_days"] == horizon].copy()
        rows = [
            [
                row.bucket_label,
                int(row.rows),
                _fmt(row.avg_raw_confidence_score, 2),
                _fmt_pct(row.avg_return),
                _fmt_pct(row.median_return),
                _fmt(row.hit_rate),
                _fmt_pct(row.avg_mdd),
                _fmt(row.calibrated_confidence_score, 2),
                _fmt(row.provisional_calibrated_confidence_score, 2),
                _fmt(row.operational_minus_provisional, 2),
                row.status,
            ]
            for row in subset.itertuples(index=False)
        ]
        lines.extend(
            [
                f"## {horizon}d Buckets",
                "",
                _markdown_table(
                    rows,
                    ["bucket", "rows", "avg_raw", "avg_return", "median_return", "hit_rate", "avg_mdd", "operational_cal", "provisional_cal", "delta_vs_provisional", "status"],
                ),
                "",
            ]
        )

    thin_horizons = [item["horizon_days"] for item in summaries if item["judgment"] in {"insufficient_operational_history", "thin_operational_history"}]
    if thin_horizons:
        lines.extend(
            [
                "## Sample Warning",
                "",
                f"- Buckets are too sparse for robust operational calibration on horizons: {', '.join(f'{h}d' for h in thin_horizons)}.",
                "- Keep the current score as provisional operational confidence until more matured snapshots accumulate.",
                "",
            ]
        )

    recommendation_lines = [
        "## Recommendation",
        "",
    ]
    if any(item["judgment"] == "operational_non_monotonic" for item in summaries):
        recommendation_lines.extend(
            [
                "- Operational monotonicity is weak on at least one horizon.",
                "- Prefer bucket mapping first because the sample is still small.",
                "- Consider isotonic regression only after each active bucket has comfortably more than the minimum row threshold.",
            ]
        )
    elif any(item["judgment"] == "operational_monotonic" for item in summaries):
        recommendation_lines.extend(
            [
                "- Operational buckets look monotonic on the matured horizon(s).",
                "- A simple bucket mapping from `confidence_score` to empirical hit rate is the lowest-risk first calibration.",
                "- If operational history deepens further, isotonic regression can replace bucket mapping for smoother calibration.",
            ]
        )
    else:
        recommendation_lines.extend(
            [
                "- Operational matured history is not yet sufficient for a trustworthy recalibration curve.",
                "- Keep the provisional mapping as the reference baseline and re-run this script as more operational snapshots mature.",
            ]
        )
    recommendation_lines.append("")
    lines.extend(recommendation_lines)

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    history, meta = load_operational_history(args.ranking_history_dir, args.ranking_current_csv)
    history = attach_realized_returns(history, args.prices_csv)
    provisional_map, provisional_meta = load_provisional_map(args.provisional_map_csv)
    calibration_df, summaries = build_calibration_table(history, args.min_bucket_rows, provisional_map)
    review_df, review_meta = load_live_trade_review()
    live_grade_map = build_live_grade_map(review_df, min_bucket_rows=args.min_bucket_rows, recent_trade_count=args.recent_trade_count)
    live_grade_map["review_meta"] = review_meta

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    report = build_report(calibration_df, summaries, meta, provisional_meta, args.min_bucket_rows)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    args.out_live_grade_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_live_grade_json.write_text(json.dumps(live_grade_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"history_rows: {len(history)}")
    print(f"calibration_rows: {len(calibration_df)}")
    print(f"report_path: {args.out_md}")
    print(f"csv_path: {args.out_csv}")
    print(f"live_grade_map_path: {args.out_live_grade_json}")
    print(json.dumps({"summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
