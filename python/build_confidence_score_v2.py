from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_RANKING = DATA_DIR / "ranking_final.csv"
INPUT_CONFIDENCE_OPERATIONAL = DATA_DIR / "confidence_calibration_operational.csv"
INPUT_WALKFORWARD_CSV = OUTPUT_DIR / "walk_forward_score_validation.csv"
INPUT_WALKFORWARD_MD = OUTPUT_DIR / "walk_forward_score_validation.md"
INPUT_REAL_DIFF = DATA_DIR / "real_vs_model_diff.csv"

OUT_CSV = DATA_DIR / "confidence_score_v2.csv"
OUT_JSON = DATA_DIR / "confidence_score_v2.json"
OUT_MD = OUTPUT_DIR / "confidence_score_v2.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build operational confidence score v2 for the latest ranking snapshot.")
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--confidence-operational-csv", type=Path, default=INPUT_CONFIDENCE_OPERATIONAL)
    parser.add_argument("--walkforward-csv", type=Path, default=INPUT_WALKFORWARD_CSV)
    parser.add_argument("--walkforward-md", type=Path, default=INPUT_WALKFORWARD_MD)
    parser.add_argument("--real-diff-csv", type=Path, default=INPUT_REAL_DIFF)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _clip_score(value: object, default: float = 50.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(min(100.0, max(0.0, numeric)))


def _percentile_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = numeric.rank(method="average", pct=True)
    return (ranks.fillna(0.5) * 100.0).clip(0, 100)


def _trading_value_score(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or numeric <= 0:
        return 0.0
    score = (math.log10(float(numeric)) - 9.0) / 2.0 * 100.0
    return float(min(100.0, max(0.0, score)))


def _effective_trading_value(row: pd.Series) -> float | None:
    direct = pd.to_numeric(row.get("trading_value"), errors="coerce")
    if pd.notna(direct) and float(direct) > 0:
        return float(direct)
    close = pd.to_numeric(row.get("close"), errors="coerce")
    volume = pd.to_numeric(row.get("volume"), errors="coerce")
    if pd.notna(close) and pd.notna(volume) and float(close) > 0 and float(volume) > 0:
        return float(close) * float(volume)
    return None


def _overheat_penalty(row: pd.Series) -> float:
    ret_5d = _clip_score((pd.to_numeric(row.get("ret_5d"), errors="coerce") or 0.0) * 100.0, 0.0)
    ret_10d = _clip_score((pd.to_numeric(row.get("ret_10d"), errors="coerce") or 0.0) * 100.0, 0.0)
    rsi = _clip_score(row.get("rsi_14"), 50.0)
    penalty = 0.0
    if ret_5d >= 12:
        penalty += 20.0
    elif ret_5d >= 8:
        penalty += 10.0
    if ret_10d >= 20:
        penalty += 20.0
    elif ret_10d >= 12:
        penalty += 10.0
    if rsi >= 80:
        penalty += 25.0
    elif rsi >= 70:
        penalty += 10.0
    return min(50.0, penalty)


def load_latest_ranking(path: Path) -> tuple[pd.DataFrame, str]:
    df = _read_csv(path, dtype={"code": str})
    if df.empty:
        raise FileNotFoundError(f"ranking csv not found or empty: {_resolve(path)}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    if df["date"].isna().all():
        raise ValueError("ranking csv has no usable date column")
    latest_date = df["date"].max()
    latest = df.loc[df["date"].eq(latest_date)].copy()
    latest["latest_rank_pct"] = _percentile_score(-pd.to_numeric(latest.get("rank_final"), errors="coerce"))
    for col in ["final_score", "confidence_score", "ret_score", "prob_score", "tech_score", "quality_score", "qual_score", "liquidity_score", "trading_value", "ret_5d", "ret_10d", "rsi_14"]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce")
    if "qual_score" not in latest.columns:
        latest["qual_score"] = pd.to_numeric(latest.get("quality_score"), errors="coerce")
    latest["name"] = latest.get("name", "").fillna("").astype(str)
    return latest.sort_values(["rank_final", "code"]).reset_index(drop=True), latest_date.strftime("%Y-%m-%d")


def load_confidence_operational_summary(path: Path) -> dict[str, object]:
    df = _read_csv(path)
    if df.empty:
        return {"available": False, "calibration_confidence": 45.0, "stable_bucket_count_5d": 0, "monotonic_hit_rate_5d": False}
    work = df.loc[df.get("source_mode", pd.Series(index=df.index)).astype(str).eq("operational") & pd.to_numeric(df.get("horizon_days"), errors="coerce").eq(5)].copy()
    if work.empty:
        return {"available": False, "calibration_confidence": 45.0, "stable_bucket_count_5d": 0, "monotonic_hit_rate_5d": False}
    work["rows"] = pd.to_numeric(work.get("rows"), errors="coerce")
    work["hit_rate"] = pd.to_numeric(work.get("hit_rate"), errors="coerce")
    work["bucket_mid"] = pd.to_numeric(work.get("bucket_mid"), errors="coerce")
    stable = work.loc[work.get("status", "").astype(str).eq("stable") & work["rows"].fillna(0).ge(20)].sort_values("bucket_mid")
    monotonic = bool(stable["hit_rate"].fillna(-1).is_monotonic_increasing) if len(stable) >= 2 else False
    calibration_confidence = 40.0 + min(20.0, float(len(stable)) * 10.0)
    calibration_confidence += 20.0 if monotonic else -10.0
    calibration_confidence = float(min(100.0, max(15.0, calibration_confidence)))
    return {"available": True, "calibration_confidence": calibration_confidence, "stable_bucket_count_5d": int(len(stable)), "monotonic_hit_rate_5d": monotonic}


def _parse_walkforward_markdown(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    text = resolved.read_text(encoding="utf-8")
    summary_rows: list[dict[str, object]] = []
    in_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "## Summary":
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary or not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 9 or parts[0] in {"horizon_days", "------------"}:
            continue
        summary_rows.append(
            {
                "horizon_days": pd.to_numeric(parts[0], errors="coerce"),
                "selection": parts[1],
                "run_dates": pd.to_numeric(parts[2], errors="coerce"),
                "avg_return": pd.to_numeric(parts[3].replace("%", ""), errors="coerce") / 100.0,
                "benchmark_return": pd.to_numeric(parts[4].replace("%", ""), errors="coerce") / 100.0,
                "excess_return": pd.to_numeric(parts[5].replace("%", ""), errors="coerce") / 100.0,
                "hit_rate": pd.to_numeric(parts[6], errors="coerce"),
                "avg_mdd": pd.to_numeric(parts[7].replace("%", ""), errors="coerce") / 100.0,
                "median_return": pd.to_numeric(parts[8].replace("%", ""), errors="coerce") / 100.0,
            }
        )
    return pd.DataFrame(summary_rows)


def load_walkforward_summary(csv_path: Path, md_path: Path) -> dict[str, object]:
    df = _read_csv(csv_path)
    if not df.empty:
        selection = df.loc[df.get("section", pd.Series(index=df.index)).astype(str).eq("selection_summary")].copy()
    else:
        selection = _parse_walkforward_markdown(md_path)
    if selection.empty:
        return {"available": False, "alpha_floor_boost": 0.0}
    selection["horizon_days"] = pd.to_numeric(selection.get("horizon_days"), errors="coerce")
    selection["avg_return"] = pd.to_numeric(selection.get("avg_return"), errors="coerce")
    selection["excess_return"] = pd.to_numeric(selection.get("excess_return"), errors="coerce")
    selection["hit_rate"] = pd.to_numeric(selection.get("hit_rate"), errors="coerce")
    selection["run_dates"] = pd.to_numeric(selection.get("run_dates"), errors="coerce")

    top20 = selection.loc[(selection["horizon_days"].eq(60)) & (selection["selection"].astype(str).eq("top20"))].head(1)
    top50 = selection.loc[(selection["horizon_days"].eq(60)) & (selection["selection"].astype(str).eq("top50"))].head(1)
    universe = selection.loc[(selection["horizon_days"].eq(60)) & (selection["selection"].astype(str).eq("universe"))].head(1)
    ordered = False
    alpha_floor_boost = 0.0
    if not top20.empty:
        top20_row = top20.iloc[0]
        alpha_floor_boost += 10.0 if pd.notna(top20_row["excess_return"]) and float(top20_row["excess_return"]) > 0 else -10.0
        alpha_floor_boost += 5.0 if pd.notna(top20_row["hit_rate"]) and float(top20_row["hit_rate"]) >= 0.55 else 0.0
        alpha_floor_boost += 5.0 if pd.notna(top20_row["run_dates"]) and float(top20_row["run_dates"]) >= 30 else 0.0
        if not top50.empty and not universe.empty:
            ordered = float(top20_row["avg_return"]) > float(top50.iloc[0]["avg_return"]) > float(universe.iloc[0]["avg_return"])
            alpha_floor_boost += 5.0 if ordered else -5.0
    alpha_floor_boost = float(min(15.0, max(-15.0, alpha_floor_boost)))
    return {
        "available": True,
        "alpha_floor_boost": alpha_floor_boost,
        "top20": top20.iloc[0].to_dict() if not top20.empty else None,
        "top50": top50.iloc[0].to_dict() if not top50.empty else None,
        "universe": universe.iloc[0].to_dict() if not universe.empty else None,
        "ordered_top20_top50_universe": ordered,
    }


def load_execution_summary(path: Path) -> dict[str, object]:
    df = _read_csv(path, dtype={"code": str})
    if df.empty:
        return {"available": False, "execution_quality_score": 55.0, "fill_ratio": None, "median_abs_slippage": None}
    entry_status = df.get("actual_entry_status", pd.Series(index=df.index)).fillna("").astype(str).str.upper()
    filled = entry_status.isin(["FILLED", "PARTIAL", "ENTERED"])
    slippage = pd.to_numeric(df.get("entry_slippage"), errors="coerce")
    if slippage.isna().all():
        slippage = pd.to_numeric(df.get("entry_price_vs_ref_return"), errors="coerce")
    fill_ratio = float(filled.mean()) if len(df) else None
    median_abs_slippage = float(slippage.abs().median()) if slippage.notna().any() else None
    score = 55.0
    if fill_ratio is not None:
        score += (fill_ratio - 0.5) * 40.0
    if median_abs_slippage is not None:
        score -= min(20.0, median_abs_slippage * 400.0)
    score = float(min(100.0, max(20.0, score)))
    return {"available": True, "execution_quality_score": score, "fill_ratio": fill_ratio, "median_abs_slippage": median_abs_slippage}


def build_confidence_frame(ranking: pd.DataFrame, calibration_summary: dict[str, object], walkforward_summary: dict[str, object], execution_summary: dict[str, object]) -> pd.DataFrame:
    work = ranking.copy()
    calibration_score = float(calibration_summary.get("calibration_confidence") or 45.0)
    execution_global = float(execution_summary.get("execution_quality_score") or 55.0)
    work["alpha_confidence"] = (0.45 * work["ret_score"].fillna(50.0) + 0.35 * work["prob_score"].fillna(50.0) + 0.20 * work["latest_rank_pct"].fillna(50.0) + float(walkforward_summary.get("alpha_floor_boost") or 0.0)).clip(0, 100)

    execution_scores = []
    stability_scores = []
    states = []
    reasons = []
    for _, row in work.iterrows():
        liquidity = _clip_score(row.get("liquidity_score"), 40.0)
        effective_trading_value = _effective_trading_value(row)
        trading_value_score = _trading_value_score(effective_trading_value)
        overheat_penalty = _overheat_penalty(row)
        execution_conf = min(100.0, max(0.0, 0.60 * liquidity + 0.25 * trading_value_score + 0.15 * execution_global - overheat_penalty))
        signal_agreement = max(0.0, 100.0 - abs(_clip_score(row.get("ret_score")) - _clip_score(row.get("prob_score"))))
        stability_conf = min(100.0, max(0.0, 0.45 * _clip_score(row.get("confidence_score")) + 0.30 * _clip_score(row.get("tech_score")) + 0.25 * signal_agreement))
        raw_conf = 0.40 * float(row["alpha_confidence"]) + 0.30 * execution_conf + 0.20 * stability_conf + 0.10 * calibration_score
        raw_conf = float(min(100.0, max(0.0, raw_conf)))

        state = "WEAK"
        reason = []
        if liquidity < 8.0 or (effective_trading_value is not None and effective_trading_value < 2_500_000_000.0):
            state = "BLOCKED"
            reason.append("execution_blocker")
        elif raw_conf >= 74.0 and calibration_score >= 50.0 and execution_conf >= 45.0 and float(row["alpha_confidence"]) >= 80.0:
            state = "TRUSTED"
            reason.append("all_four_axes_aligned")
        elif raw_conf >= 64.0 and execution_conf >= 28.0:
            state = "PROVISIONAL"
            reason.append("alpha_supported_but_not_fully_calibrated")
        else:
            reason.append("evidence_not_strong_enough")
        if overheat_penalty >= 20.0:
            reason.append("overheat_risk")
        if calibration_score < 60.0:
            reason.append("weak_operational_calibration")

        execution_scores.append(execution_conf)
        stability_scores.append(stability_conf)
        states.append(state)
        reasons.append(" / ".join(reason))

    work["execution_confidence"] = execution_scores
    work["stability_confidence"] = stability_scores
    work["calibration_confidence"] = calibration_score
    work["raw_confidence_v2"] = (0.40 * work["alpha_confidence"] + 0.30 * work["execution_confidence"] + 0.20 * work["stability_confidence"] + 0.10 * work["calibration_confidence"]).clip(0, 100)
    work["confidence_state_v2"] = states
    work["confidence_reason_v2"] = reasons
    return work


def build_payload(asof_date: str, frame: pd.DataFrame, calibration_summary: dict[str, object], walkforward_summary: dict[str, object], execution_summary: dict[str, object]) -> dict[str, object]:
    state_counts = frame["confidence_state_v2"].value_counts().to_dict()
    trusted_top20 = frame.head(20)["confidence_state_v2"].eq("TRUSTED").mean() if not frame.empty else 0.0
    provisional_or_better_top20 = frame.head(20)["confidence_state_v2"].isin(["TRUSTED", "PROVISIONAL"]).mean() if not frame.empty else 0.0
    items = []
    for _, row in frame.iterrows():
        items.append(
            {
                "code": row["code"],
                "name": row.get("name"),
                "rank_final": pd.to_numeric(row.get("rank_final"), errors="coerce"),
                "final_score": pd.to_numeric(row.get("final_score"), errors="coerce"),
                "confidence_score": pd.to_numeric(row.get("confidence_score"), errors="coerce"),
                "alpha_confidence": pd.to_numeric(row.get("alpha_confidence"), errors="coerce"),
                "execution_confidence": pd.to_numeric(row.get("execution_confidence"), errors="coerce"),
                "stability_confidence": pd.to_numeric(row.get("stability_confidence"), errors="coerce"),
                "calibration_confidence": pd.to_numeric(row.get("calibration_confidence"), errors="coerce"),
                "raw_confidence_v2": pd.to_numeric(row.get("raw_confidence_v2"), errors="coerce"),
                "confidence_state_v2": row.get("confidence_state_v2"),
                "confidence_reason_v2": row.get("confidence_reason_v2"),
            }
        )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof_date,
        "summary": {
            "row_count": int(len(frame)),
            "mean_raw_confidence_v2": float(pd.to_numeric(frame["raw_confidence_v2"], errors="coerce").mean()) if not frame.empty else None,
            "trusted_ratio_top20": float(trusted_top20),
            "provisional_or_better_ratio_top20": float(provisional_or_better_top20),
            "state_counts": state_counts,
            "calibration_summary": calibration_summary,
            "walkforward_summary": walkforward_summary,
            "execution_summary": execution_summary,
        },
        "items": items,
    }


def build_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    state_counts = summary.get("state_counts", {}) if isinstance(summary.get("state_counts"), dict) else {}
    top_items = payload.get("items", [])[:20] if isinstance(payload.get("items"), list) else []
    count_rows = [[key, value] for key, value in state_counts.items()]
    detail_rows = [
        [item.get("code"), item.get("name"), _fmt_num(item.get("rank_final"), 0), _fmt_num(item.get("raw_confidence_v2")), item.get("confidence_state_v2"), _fmt_num(item.get("alpha_confidence")), _fmt_num(item.get("execution_confidence")), _fmt_num(item.get("stability_confidence")), item.get("confidence_reason_v2")]
        for item in top_items
    ]
    lines = [
        "# Confidence Score V2",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- asof_date: {payload.get('asof_date')}",
        f"- mean_raw_confidence_v2: {_fmt_num(summary.get('mean_raw_confidence_v2'))}",
        f"- trusted_ratio_top20: {_fmt_pct(summary.get('trusted_ratio_top20'))}",
        f"- provisional_or_better_ratio_top20: {_fmt_pct(summary.get('provisional_or_better_ratio_top20'))}",
        "",
        "## State Counts",
        "",
        _markdown_table(count_rows or [["(none)", 0]], ["state", "count"]),
        "",
        "## Top20 Detail",
        "",
        _markdown_table(detail_rows or [["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]], ["code", "name", "rank", "raw_conf_v2", "state", "alpha", "execution", "stability", "reason"]),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ranking, asof_date = load_latest_ranking(args.ranking_csv)
    calibration_summary = load_confidence_operational_summary(args.confidence_operational_csv)
    walkforward_summary = load_walkforward_summary(args.walkforward_csv, args.walkforward_md)
    execution_summary = load_execution_summary(args.real_diff_csv)
    frame = build_confidence_frame(ranking, calibration_summary, walkforward_summary, execution_summary)

    out_csv = _resolve(args.out_csv)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    frame[["date", "code", "name", "rank_final", "final_score", "confidence_score", "alpha_confidence", "execution_confidence", "stability_confidence", "calibration_confidence", "raw_confidence_v2", "confidence_state_v2", "confidence_reason_v2"]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    payload = build_payload(asof_date, frame, calibration_summary, walkforward_summary, execution_summary)
    out_json.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(payload), encoding="utf-8")

    print(f"asof_date: {asof_date}")
    print(f"out_csv: {out_csv}")
    print(f"out_json: {out_json}")
    print(f"out_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
