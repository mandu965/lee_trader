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
from ranking_builder import _compute_confidence_score, DEFAULT_SCORE_FORMULA_VERSION


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
CURRENT_RANKING_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_OUT_CSV = DATA_DIR / "confidence_calibration_map.csv"
DEFAULT_OUT_JSON = DATA_DIR / "confidence_calibration_map.json"
DEFAULT_OUT_MD = DATA_DIR / "confidence_calibration_report.md"
HORIZON_DAYS = 60
MIN_BUCKET_ROWS = 20
SOURCE_SNAPSHOT = "snapshot_operational"
SOURCE_WALKFORWARD = "walkforward_provisional"
SOURCE_AUTO = "auto"
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
    parser = argparse.ArgumentParser(description="Build confidence calibration map.")
    parser.add_argument("--source", choices=[SOURCE_AUTO, SOURCE_SNAPSHOT, SOURCE_WALKFORWARD], default=SOURCE_AUTO)
    parser.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    parser.add_argument("--ranking-current-csv", type=Path, default=CURRENT_RANKING_CSV)
    parser.add_argument("--prices-csv", type=Path, default=DATA_DIR / "prices_daily_adjusted.csv")
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    parser.add_argument("--min-bucket-rows", type=int, default=int(get_production_config_value(["confidence_calibration", "min_bucket_rows_snapshot"], MIN_BUCKET_ROWS)))
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


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


def load_ranking_history(history_dir: Path, current_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
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
        raise FileNotFoundError("No ranking history found for confidence calibration.")

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"] if "confidence_score" in df.columns else pd.Series(index=df.index), errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"] if "final_score" in df.columns else pd.Series(index=df.index), errors="coerce")
    df["rank_final"] = pd.to_numeric(df["rank_final"] if "rank_final" in df.columns else pd.Series(index=df.index), errors="coerce")
    df = df.dropna(subset=["date", "code", "confidence_score"]).copy()
    df["snapshot_priority"] = df["snapshot_file"].eq(current_csv.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    meta = {
        "source_mode": SOURCE_SNAPSHOT,
        "source_label": "daily ranking snapshots",
        "confidence_kind": "stored_operational_confidence",
        "source_item_count": len(sorted(set(snapshot_files))),
        "source_items": sorted(set(snapshot_files)),
    }
    return df, meta


def attach_realized_returns(df: pd.DataFrame, prices_csv: Path, horizon_days: int) -> pd.DataFrame:
    prices = load_price_history(prices_csv=prices_csv)
    outcomes = attach_forward_outcomes(prices, horizon_days=horizon_days).rename(
        columns={
            "realized_return": f"realized_return_{horizon_days}d",
            "realized_mdd": f"realized_mdd_{horizon_days}d",
        }
    )
    outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce").dt.normalize()
    merged = df.merge(outcomes, on=["code", "date"], how="left")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return merged


def load_walkforward_provisional_history(horizon_days: int) -> tuple[pd.DataFrame, dict[str, object]]:
    query = text(
        """
        SELECT
            r.run_id,
            r.as_of_date AS date,
            r.code,
            r.rank AS rank_final,
            COALESCE(r.final_score, p.final_score) AS final_score,
            p.ret_score,
            p.prob_score,
            p.qual_score,
            p.tech_score,
            p.risk_penalty,
            p.model_version,
            o.realized_return,
            o.realized_mdd
        FROM research.ranking_history r
        JOIN research.prediction_history p
          ON p.run_id = r.run_id
         AND p.as_of_date = r.as_of_date
         AND p.code = r.code
         AND p.horizon_days = r.horizon_days
        JOIN research.backtest_outcome o
          ON o.run_id = r.run_id
         AND o.as_of_date = r.as_of_date
         AND o.code = r.code
         AND o.horizon_days = r.horizon_days
        JOIN research.dim_model_run d
          ON d.run_id = r.run_id
        WHERE d.run_type = 'walkforward_backtest'
          AND r.horizon_days = :horizon_days
          AND o.realized_return IS NOT NULL
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"horizon_days": horizon_days}, parse_dates=["date"])

    if df.empty:
        raise RuntimeError("No matured walk-forward rows found for provisional calibration.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["rank_final"] = pd.to_numeric(df["rank_final"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["return_score"] = pd.to_numeric(df["ret_score"], errors="coerce").fillna(50.0)
    df["probability_score"] = pd.to_numeric(df["prob_score"], errors="coerce").fillna(50.0)
    df["technical_score"] = pd.to_numeric(df["tech_score"], errors="coerce").fillna(50.0)
    df["qual_score"] = pd.to_numeric(df["qual_score"], errors="coerce").fillna(50.0)
    df["valuation_score"] = 50.0
    df["risk_penalty"] = pd.to_numeric(df["risk_penalty"], errors="coerce").fillna(0.0)
    df["quality_score_confidence"] = 50.0
    df["fallback_count"] = 0.0
    df["tech_liquidity_guard"] = 1.0
    df["score_formula_version"] = DEFAULT_SCORE_FORMULA_VERSION
    df["regime"] = "defensive"
    for col in ["ret_score_missing", "prob_score_missing", "tech_score_missing", "qual_score_missing"]:
        df[col] = False
    df = _compute_confidence_score(df)
    df[f"realized_return_{horizon_days}d"] = pd.to_numeric(df["realized_return"], errors="coerce")
    df[f"realized_mdd_{horizon_days}d"] = pd.to_numeric(df["realized_mdd"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    meta = {
        "source_mode": SOURCE_WALKFORWARD,
        "source_label": "walk-forward backtest rows",
        "confidence_kind": "reconstructed_provisional_confidence",
        "source_item_count": int(df["run_id"].nunique()),
        "source_items": [str(v) for v in sorted(df["run_id"].astype(str).unique().tolist())[:20]],
        "source_items_truncated": int(df["run_id"].nunique()) > 20,
    }
    return df, meta


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


def calibrated_band(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    if value >= 80:
        return "high"
    if value >= 60:
        return "medium"
    return "low"


def build_bucket_map(df: pd.DataFrame, horizon_days: int, min_bucket_rows: int) -> tuple[pd.DataFrame, int]:
    ret_col = f"realized_return_{horizon_days}d"
    mdd_col = f"realized_mdd_{horizon_days}d"
    ret_source = df[ret_col] if ret_col in df.columns else pd.Series(index=df.index, dtype="float64")
    matured = df.loc[pd.to_numeric(ret_source, errors="coerce").notna()].copy()
    matured_rows = int(len(matured))

    if matured.empty:
        rows = [
            {
                "confidence_calibration_version": CONFIDENCE_CALIBRATION_VERSION,
                "bucket_label": label,
                "bucket_low": lo,
                "bucket_high": hi,
                "bucket_mid": float((lo + hi) / 2),
                "rows": 0,
                "avg_raw_confidence_score": None,
                "avg_return": None,
                "hit_rate": None,
                "avg_mdd": None,
                "top20_entry_rate": None,
                "calibrated_confidence_score": None,
                "calibrated_confidence_band": "unknown",
                "status": "insufficient_history",
            }
            for lo, hi, label in BUCKETS
        ]
        return pd.DataFrame(rows), matured_rows

    matured["confidence_bucket_raw"] = bucketize(matured["confidence_score"])
    matured["positive_hit"] = pd.to_numeric(matured[ret_col], errors="coerce") > 0
    matured["top20_flag"] = pd.to_numeric(matured["rank_final"], errors="coerce") <= 20

    rows: list[dict[str, object]] = []
    for lo, hi, label in BUCKETS:
        bucket = matured.loc[matured["confidence_bucket_raw"] == label].copy()
        n = int(len(bucket))
        avg_return = float(pd.to_numeric(bucket[ret_col], errors="coerce").mean()) if n else None
        hit_rate = float(bucket["positive_hit"].mean()) if n else None
        avg_mdd = float(pd.to_numeric(bucket[mdd_col], errors="coerce").mean()) if n else None
        top20_rate = float(bucket["top20_flag"].mean()) if n else None
        avg_raw_score = float(pd.to_numeric(bucket["confidence_score"], errors="coerce").mean()) if n else None
        midpoint = float((lo + hi) / 2)

        if n == 0:
            calibrated = None
            status = "empty"
        else:
            reliability_component = (hit_rate or 0.0) * 100.0
            top20_component = min(max((top20_rate or 0.0) * 100.0, 0.0), 100.0)
            calibrated = (reliability_component * 0.70) + (midpoint * 0.20) + (top20_component * 0.10)
            calibrated = max(0.0, min(100.0, calibrated))
            status = "stable" if n >= min_bucket_rows else "thin_sample"

        rows.append(
            {
                "confidence_calibration_version": CONFIDENCE_CALIBRATION_VERSION,
                "bucket_label": label,
                "bucket_low": lo,
                "bucket_high": hi,
                "bucket_mid": midpoint,
                "rows": n,
                "avg_raw_confidence_score": avg_raw_score,
                "avg_return": avg_return,
                "hit_rate": hit_rate,
                "avg_mdd": avg_mdd,
                "top20_entry_rate": top20_rate,
                "calibrated_confidence_score": calibrated,
                "calibrated_confidence_band": calibrated_band(calibrated),
                "status": status,
            }
        )

    return pd.DataFrame(rows), matured_rows


def build_summary(
    bucket_map: pd.DataFrame,
    history_df: pd.DataFrame,
    matured_rows: int,
    horizon_days: int,
    min_bucket_rows: int,
    meta: dict[str, object],
) -> dict[str, object]:
    valid = bucket_map.loc[bucket_map["status"].isin(["stable", "thin_sample"])].copy()
    monotonic_return = valid["avg_return"].dropna().is_monotonic_increasing if not valid.empty else False
    monotonic_hit = valid["hit_rate"].dropna().is_monotonic_increasing if not valid.empty else False
    monotonic_cal = valid["calibrated_confidence_score"].dropna().is_monotonic_increasing if not valid.empty else False
    date_values = pd.to_datetime(history_df["date"], errors="coerce")
    latest_date = date_values.max()
    earliest_date = date_values.min()
    days_span = int((latest_date - earliest_date).days) if pd.notna(latest_date) and pd.notna(earliest_date) else 0

    if matured_rows <= 0:
        judgment = "insufficient_history"
        recommendation = f"Need older evidence with at least {horizon_days} days of matured outcomes."
    elif monotonic_hit and monotonic_cal:
        judgment = "usable"
        recommendation = "Calibrated confidence is usable for stable buckets."
    else:
        judgment = "review_required"
        recommendation = "Keep raw confidence until more samples improve bucket monotonicity."

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_mode": meta["source_mode"],
        "source_label": meta["source_label"],
        "confidence_calibration_version": CONFIDENCE_CALIBRATION_VERSION,
        "confidence_kind": meta["confidence_kind"],
        "horizon_days": horizon_days,
        "min_bucket_rows": min_bucket_rows,
        "source_item_count": int(meta.get("source_item_count", 0)),
        "source_items": list(meta.get("source_items", [])),
        "source_items_truncated": bool(meta.get("source_items_truncated", False)),
        "history_row_count": int(len(history_df)),
        "history_date_count": int(date_values.nunique()),
        "history_start_date": earliest_date.strftime("%Y-%m-%d") if pd.notna(earliest_date) else None,
        "history_end_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else None,
        "history_span_days": days_span,
        "matured_row_count": int(matured_rows),
        "bucket_count": int(len(bucket_map)),
        "usable_bucket_count": int(bucket_map["status"].isin(["stable", "thin_sample"]).sum()),
        "monotonic_return": bool(monotonic_return),
        "monotonic_hit_rate": bool(monotonic_hit),
        "monotonic_calibrated_score": bool(monotonic_cal),
        "judgment": judgment,
        "recommendation": recommendation,
    }


def build_markdown(bucket_map: pd.DataFrame, summary: dict[str, object]) -> str:
    rows = [
        [
            row["bucket_label"],
            int(row["rows"]),
            _fmt(row["avg_raw_confidence_score"], 2),
            _fmt_pct(row["avg_return"]),
            _fmt(row["hit_rate"]),
            _fmt_pct(row["avg_mdd"]),
            _fmt(row["top20_entry_rate"]),
            _fmt(row["calibrated_confidence_score"], 2),
            row["calibrated_confidence_band"],
            row["status"],
        ]
        for _, row in bucket_map.iterrows()
    ]

    lines = [
        "# Confidence Calibration Report",
        "",
        "## Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- source_mode: {summary['source_mode']}",
        f"- source_label: {summary['source_label']}",
        f"- confidence_calibration_version: {summary['confidence_calibration_version']}",
        f"- confidence_kind: {summary['confidence_kind']}",
        f"- horizon_days: {summary['horizon_days']}",
        f"- min_bucket_rows: {summary['min_bucket_rows']}",
        f"- judgment: {summary['judgment']}",
        f"- recommendation: {summary['recommendation']}",
        f"- history_date_count: {summary['history_date_count']}",
        f"- history_window: {summary.get('history_start_date', 'NA')} -> {summary.get('history_end_date', 'NA')}",
        f"- history_span_days: {summary['history_span_days']}",
        f"- matured_row_count: {summary['matured_row_count']}",
        f"- usable_bucket_count: {summary['usable_bucket_count']}",
        "",
        "## Bucket Map",
        "",
        _markdown_table(
            rows,
            ["bucket", "rows", "avg_raw", "avg_return", "hit_rate", "avg_mdd", "top20_rate", "calibrated", "band", "status"],
        ),
        "",
        "## Interpretation",
        "",
    ]

    if summary["source_mode"] == SOURCE_WALKFORWARD:
        lines.extend(
            [
                "- This is a provisional calibration built from walk-forward backtest rows.",
                "- Confidence values are reconstructed from stored score components, not copied from original daily snapshots.",
                "- Use this for research and operator interpretation, not as a final operational calibration.",
            ]
        )
    elif summary["judgment"] == "insufficient_history":
        lines.extend(
            [
                "- Current daily ranking snapshots do not yet contain enough matured outcomes for this horizon.",
                "- Keep raw confidence in production until more daily snapshots accumulate.",
                "- Re-run the same script after snapshot readiness reaches READY.",
            ]
        )
    else:
        lines.extend(
            [
                "- Calibrated confidence is an interpretation layer, not a replacement for final_score.",
                "- Buckets marked thin_sample should be read conservatively.",
                "- If monotonicity weakens, accumulate more samples before promoting the mapping.",
            ]
        )

    if summary["source_items"]:
        lines.extend(["", "## Source Items", ""])
        lines.extend([f"- {name}" for name in summary["source_items"]])
        if summary.get("source_items_truncated"):
            lines.append("- ...")

    return "\n".join(lines) + "\n"


def build_source_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    if args.source == SOURCE_SNAPSHOT:
        ranking, meta = load_ranking_history(args.ranking_history_dir, args.ranking_current_csv)
        return attach_realized_returns(ranking, args.prices_csv, args.horizon_days), meta

    if args.source == SOURCE_WALKFORWARD:
        return load_walkforward_provisional_history(args.horizon_days)

    ranking, meta = load_ranking_history(args.ranking_history_dir, args.ranking_current_csv)
    ranking = attach_realized_returns(ranking, args.prices_csv, args.horizon_days)
    _, matured_rows = build_bucket_map(ranking, args.horizon_days, args.min_bucket_rows)
    if matured_rows > 0:
        return ranking, meta
    try:
        return load_walkforward_provisional_history(args.horizon_days)
    except RuntimeError:
        empty = ranking.iloc[0:0].copy()
        fallback_meta = {
            "source_mode": SOURCE_WALKFORWARD,
            "source_label": "walk-forward backtest rows (unavailable)",
            "confidence_kind": "reconstructed_provisional_confidence",
            "source_item_count": 0,
            "source_items": [],
            "source_items_truncated": False,
        }
        return empty, fallback_meta


def main() -> None:
    args = parse_args()
    history_df, meta = build_source_dataset(args)
    bucket_map, matured_rows = build_bucket_map(history_df, args.horizon_days, args.min_bucket_rows)
    summary = build_summary(bucket_map, history_df, matured_rows, args.horizon_days, args.min_bucket_rows, meta)

    bucket_map.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    args.out_json.write_text(
        json.dumps({"summary": summary, "bucket_map": bucket_map.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out_md.write_text(build_markdown(bucket_map, summary), encoding="utf-8")

    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")
    print(f"source_mode: {summary['source_mode']}")
    print(f"judgment: {summary['judgment']}")
    print(f"matured_row_count: {summary['matured_row_count']}")


if __name__ == "__main__":
    main()
