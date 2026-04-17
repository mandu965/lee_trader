from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RANKING_CSV = DATA_DIR / "ranking_final.csv"
INVENTORY_CSV = DATA_DIR / "history" / "ranking_snapshot_inventory.csv"
FORWARD_CSV = OUTPUT_DIR / "operational_forward_return_by_day.csv"
BENCHMARK_CSV = OUTPUT_DIR / "benchmark_comparison.csv"
BUY_GATE_JSON = OUTPUT_DIR / "operational_buy_gate.json"
DAILY_CYCLE_JSON = OUTPUT_DIR / "operational_daily_cycle_status.json"
CONFIDENCE_CSV = DATA_DIR / "confidence_calibration_operational.csv"
SCORE_KPI_JSON = DATA_DIR / "score_kpi_monitor.json"
PAPER_NAV_CSV = DATA_DIR / "paper_trading_nav.csv"

OUT_MD = OUTPUT_DIR / "operational_weekly_review.md"
OUT_CSV = OUTPUT_DIR / "operational_weekly_review.csv"

SCORE_FORMULA_VERSION = str(get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v8_return_prob_tech_regime"))
GATE_VERSION = str(get_production_config_value(["metadata", "gate_version"], "operational_buy_gate_v1"))
PORTFOLIO_VERSION = str(get_production_config_value(["metadata", "portfolio_version"], "model_portfolio_constructor_v1"))
CONFIDENCE_VERSION = str(get_production_config_value(["metadata", "confidence_calibration_version"], "confidence_four_axis_v1"))


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8").replace("NaN", "null"))


def fnum(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    return "NA" if pd.isna(x) else f"{float(x):.{digits}f}"


def fpct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    return "NA" if pd.isna(x) else f"{float(x) * 100:.{digits}f}%"


def fdelta(value: object, digits: int = 2, pct: bool = False) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    sign = "+" if float(x) >= 0 else ""
    return f"{sign}{float(x) * 100:.{digits}f}%" if pct else f"{sign}{float(x):.{digits}f}"


def delta(left: object, right: object) -> float | None:
    l = pd.to_numeric(left, errors="coerce")
    r = pd.to_numeric(right, errors="coerce")
    return None if pd.isna(l) or pd.isna(r) else float(l - r)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows_"
    rows = df[cols].fillna("").astype(str).values.tolist()
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    out = [line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def latest_ranking() -> tuple[pd.DataFrame, pd.Timestamp]:
    df = read_csv(RANKING_CSV, dtype={"code": str})
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"].eq(latest_date)].copy() if pd.notna(latest_date) else pd.DataFrame()
    return latest, latest_date if pd.notna(latest_date) else pd.Timestamp(datetime.now().date())


def week_window(asof_date: pd.Timestamp) -> tuple[str, str]:
    start = asof_date - timedelta(days=asof_date.weekday())
    end = start + timedelta(days=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def concentration(top: pd.DataFrame) -> dict[str, object]:
    if top.empty:
        return {}
    work = top.copy()
    work["final_score"] = pd.to_numeric(work.get("final_score"), errors="coerce")
    work["confidence_score"] = pd.to_numeric(work.get("confidence_score"), errors="coerce")
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["dominant_theme"] = work.get("dominant_theme", "(none)").fillna("(none)").astype(str).replace({"": "(none)", "nan": "(none)"})
    work = work.sort_values(["final_score"], ascending=[False]).head(20)
    sec = work["sector"].value_counts(normalize=True)
    thm = work["dominant_theme"].value_counts(normalize=True)
    return {
        "top20_mean_final_score": float(work["final_score"].mean()) if not work["final_score"].dropna().empty else None,
        "top20_mean_confidence_score": float(work["confidence_score"].mean()) if not work["confidence_score"].dropna().empty else None,
        "sector_top_label": sec.index[0] if not sec.empty else None,
        "sector_top_share": float(sec.iloc[0]) if not sec.empty else None,
        "sector_hhi": float((sec.pow(2)).sum()) if not sec.empty else None,
        "theme_top_label": thm.index[0] if not thm.empty else None,
        "theme_top_share": float(thm.iloc[0]) if not thm.empty else None,
        "theme_hhi": float((thm.pow(2)).sum()) if not thm.empty else None,
        "no_theme_share": float(thm.get("(none)", 0.0)) if not thm.empty else None,
    }


def inventory_metrics() -> dict[str, object]:
    df = read_csv(INVENTORY_CSV)
    if df.empty:
        return {"snapshot_count": 0, "matured_snapshot_count_20d": 0, "matured_snapshot_count_60d": 0, "matured_snapshot_count_90d": 0}
    for col in ["matured_20d", "matured_60d", "matured_90d"]:
        df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return {
        "snapshot_count": int(len(df)),
        "matured_snapshot_count_20d": int(df["matured_20d"].sum()),
        "matured_snapshot_count_60d": int(df["matured_60d"].sum()),
        "matured_snapshot_count_90d": int(df["matured_90d"].sum()),
    }


def forward_metrics() -> dict[str, object]:
    df = read_csv(FORWARD_CSV)
    if df.empty:
        return {}
    df["top_n"] = pd.to_numeric(df.get("top_n"), errors="coerce")
    df["horizon_days"] = pd.to_numeric(df.get("horizon_days"), errors="coerce")
    df["matured_count"] = pd.to_numeric(df.get("matured_count"), errors="coerce")
    top5 = df.loc[df["top_n"].eq(5)].copy()
    if top5.empty:
        return {}
    latest = top5.sort_values(["asof_date", "horizon_days"]).iloc[-1]
    out = {"forward_top5_latest_maturity_state": latest.get("maturity_state")}
    for h in [5, 20, 60, 90]:
        row = top5.loc[top5["horizon_days"].eq(h), "matured_count"]
        out[f"forward_top5_matured_count_{h}d"] = int(row.iloc[-1]) if not row.empty and pd.notna(row.iloc[-1]) else None
    return out


def paper_nav_metrics() -> dict[str, object]:
    df = read_csv(PAPER_NAV_CSV)
    if df.empty:
        return {"paper_nav_available": False}
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    work = df.loc[df["strategy"].astype(str).eq("top5")].copy() if "strategy" in df.columns else df.copy()
    work = work.dropna(subset=["date"])
    if work.empty:
        return {"paper_nav_available": False}
    latest = work.sort_values(["date"]).iloc[-1]
    return {
        "paper_nav_available": True,
        "paper_nav_latest_date": latest["date"].strftime("%Y-%m-%d"),
        "paper_nav_latest": pd.to_numeric(latest.get("nav"), errors="coerce"),
        "paper_nav_cumulative_return": pd.to_numeric(latest.get("cumulative_return"), errors="coerce"),
        "paper_nav_drawdown": pd.to_numeric(latest.get("drawdown"), errors="coerce"),
    }


def benchmark_metrics() -> dict[str, object]:
    df = read_csv(BENCHMARK_CSV)
    if df.empty:
        return {}
    df["top_n"] = pd.to_numeric(df.get("top_n"), errors="coerce")
    df["dates_matured"] = pd.to_numeric(df.get("dates_matured"), errors="coerce")
    df["avg_excess_return"] = pd.to_numeric(df.get("avg_excess_return"), errors="coerce")
    top5 = df.loc[df["top_n"].eq(5)].copy()
    matured = top5.loc[top5["dates_matured"].fillna(0) > 0].copy()

    def avg_for(name: str) -> float | None:
        s = matured.loc[matured["benchmark_name"].astype(str).eq(name), "avg_excess_return"]
        return None if s.empty else float(s.mean())

    return {
        "benchmark_top5_matured_dates_max": int(top5["dates_matured"].fillna(0).max()) if not top5.empty else 0,
        "benchmark_kospi_avg_excess_return": avg_for("KOSPI"),
        "benchmark_baseline_avg_excess_return": avg_for("BASELINE_RANKING_V2"),
    }


def gate_metrics() -> dict[str, object]:
    gate = read_json(BUY_GATE_JSON)
    decisions = gate.get("decisions", []) if isinstance(gate.get("decisions"), list) else []
    primary_bucket = int(pd.to_numeric(gate.get("primary_bucket"), errors="coerce")) if pd.notna(pd.to_numeric(gate.get("primary_bucket"), errors="coerce")) else 5
    primary = next((d for d in decisions if int(pd.to_numeric(d.get("bucket"), errors="coerce")) == primary_bucket), {}) if decisions else {}
    return {
        "gate_asof_date": gate.get("asof_date"),
        "buy_gate_overall_status": gate.get("overall_status"),
        "buy_gate_primary_bucket": primary_bucket,
        "buy_gate_primary_status": primary.get("status"),
        "buy_gate_reason_summary": primary.get("reason_summary"),
        "buy_gate_top5_status": next((d.get("status") for d in decisions if d.get("bucket") == 5), None),
        "buy_gate_top8_status": next((d.get("status") for d in decisions if d.get("bucket") == 8), None),
        "buy_gate_top10_status": next((d.get("status") for d in decisions if d.get("bucket") == 10), None),
    }


def daily_wait_metrics() -> dict[str, object]:
    payload = read_json(DAILY_CYCLE_JSON)
    steps = payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
    reasons = [str(step.get("wait_reason")).strip() for step in steps if str(step.get("status") or "").upper() == "WAIT" and str(step.get("wait_reason") or "").strip()]
    reasons = list(dict.fromkeys(reasons))
    return {"daily_cycle_overall_status": payload.get("overall_status"), "daily_cycle_wait_reasons": " | ".join(reasons)}


def confidence_metrics() -> dict[str, object]:
    df = read_csv(CONFIDENCE_CSV)
    if df.empty:
        return {}
    work = df.loc[df["source_mode"].astype(str).eq("operational") & pd.to_numeric(df["horizon_days"], errors="coerce").eq(5)].copy()
    if work.empty:
        return {}
    work["rows"] = pd.to_numeric(work["rows"], errors="coerce")
    work["hit_rate"] = pd.to_numeric(work["hit_rate"], errors="coerce")
    work["bucket_mid"] = pd.to_numeric(work["bucket_mid"], errors="coerce")
    work["operational_minus_provisional"] = pd.to_numeric(work["operational_minus_provisional"], errors="coerce")
    work["calibrated_confidence_score"] = pd.to_numeric(work["calibrated_confidence_score"], errors="coerce")
    stable = work.loc[work["status"].astype(str).eq("stable") & work["rows"].fillna(0).ge(20)].sort_values("bucket_mid")
    monotonic = bool(stable["hit_rate"].fillna(-1).is_monotonic_increasing) if len(stable) >= 2 else False
    return {
        "confidence_stable_bucket_count_5d": int(len(stable)),
        "confidence_monotonic_hit_rate_5d": monotonic,
        "confidence_reliable_5d": bool(len(stable) >= 2 and monotonic),
        "confidence_avg_calibrated_5d": float(stable["calibrated_confidence_score"].mean()) if not stable.empty else None,
        "confidence_avg_delta_vs_provisional_5d": float(stable["operational_minus_provisional"].mean()) if not stable.empty else None,
    }


def score_kpi_metrics() -> dict[str, object]:
    payload = read_json(SCORE_KPI_JSON)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    metric_map = payload.get("metric_map", {}) if isinstance(payload.get("metric_map"), dict) else {}
    kpis = payload.get("kpis", []) if isinstance(payload.get("kpis"), list) else []
    return {
        "score_kpi_overall_status": summary.get("overall_status"),
        "score_kpi_alert_count": sum(1 for k in kpis if isinstance(k, dict) and k.get("status") == "ALERT"),
        "score_kpi_watch_count": sum(1 for k in kpis if isinstance(k, dict) and k.get("status") == "WATCH"),
        "score_kpi_overlap_final_ret_top20": pd.to_numeric((metric_map.get("overlap_final_ret_top20") or {}).get("value"), errors="coerce"),
        "score_kpi_overlap_final_prob_top20": pd.to_numeric((metric_map.get("overlap_final_prob_top20") or {}).get("value"), errors="coerce"),
        "score_kpi_top20_mean_confidence_score": pd.to_numeric((metric_map.get("top20_mean_confidence_score") or {}).get("value"), errors="coerce"),
        "score_kpi_confidence_usable_bucket_count": pd.to_numeric((metric_map.get("confidence_calibration_usable_bucket_count") or {}).get("value"), errors="coerce"),
    }


def current_row() -> dict[str, object]:
    latest, asof_date = latest_ranking()
    week_start, week_end = week_window(asof_date)
    row = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_asof_date": asof_date.strftime("%Y-%m-%d"),
        "review_week_start": week_start,
        "review_week_end": week_end,
        "score_formula_version": latest["score_formula_version"].dropna().iloc[0] if not latest.empty and "score_formula_version" in latest.columns and latest["score_formula_version"].notna().any() else SCORE_FORMULA_VERSION,
        "gate_version": GATE_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "confidence_calibration_version": latest["confidence_version"].dropna().iloc[0] if not latest.empty and "confidence_version" in latest.columns and latest["confidence_version"].notna().any() else CONFIDENCE_VERSION,
        "paper_nav_available": False,
        "paper_nav_latest_date": None,
        "paper_nav_latest": None,
        "paper_nav_cumulative_return": None,
        "paper_nav_drawdown": None,
        "benchmark_top5_matured_dates_max": 0,
        "benchmark_kospi_avg_excess_return": None,
        "benchmark_baseline_avg_excess_return": None,
        "buy_gate_overall_status": None,
        "buy_gate_primary_bucket": None,
        "buy_gate_primary_status": None,
        "buy_gate_reason_summary": None,
        "buy_gate_top5_status": None,
        "buy_gate_top8_status": None,
        "buy_gate_top10_status": None,
        "daily_cycle_overall_status": None,
        "daily_cycle_wait_reasons": None,
        "confidence_stable_bucket_count_5d": None,
        "confidence_monotonic_hit_rate_5d": None,
        "confidence_reliable_5d": None,
        "confidence_avg_calibrated_5d": None,
        "confidence_avg_delta_vs_provisional_5d": None,
        "score_kpi_overall_status": None,
        "score_kpi_alert_count": None,
        "score_kpi_watch_count": None,
        "score_kpi_overlap_final_ret_top20": None,
        "score_kpi_overlap_final_prob_top20": None,
        "score_kpi_top20_mean_confidence_score": None,
        "score_kpi_confidence_usable_bucket_count": None,
        "top20_mean_final_score": None,
        "top20_mean_confidence_score": None,
        "sector_top_label": None,
        "sector_top_share": None,
        "sector_hhi": None,
        "theme_top_label": None,
        "theme_top_share": None,
        "theme_hhi": None,
        "no_theme_share": None,
    }
    for block in [inventory_metrics(), forward_metrics(), paper_nav_metrics(), benchmark_metrics(), gate_metrics(), daily_wait_metrics(), concentration(latest), confidence_metrics(), score_kpi_metrics()]:
        row.update(block)
    return row


def repeat_count(history: pd.DataFrame, col: str, value: str) -> int:
    if history.empty or not value or col not in history.columns:
        return 1 if value else 0
    count = 1
    for item in reversed(history.sort_values(["review_week_end"])[col].fillna("").astype(str).tolist()):
        if item.strip() == value.strip():
            count += 1
        else:
            break
    return count


def enrich(history: pd.DataFrame, row: dict[str, object]) -> dict[str, object]:
    prev = history.sort_values(["review_week_end"]).iloc[-1] if not history.empty else None
    row["delta_snapshot_count"] = delta(row.get("snapshot_count"), prev.get("snapshot_count") if prev is not None else None)
    row["delta_paper_nav_latest"] = delta(row.get("paper_nav_latest"), prev.get("paper_nav_latest") if prev is not None else None)
    row["delta_benchmark_kospi_avg_excess_return"] = delta(row.get("benchmark_kospi_avg_excess_return"), prev.get("benchmark_kospi_avg_excess_return") if prev is not None else None)
    row["delta_sector_top_share"] = delta(row.get("sector_top_share"), prev.get("sector_top_share") if prev is not None else None)
    row["delta_theme_top_share"] = delta(row.get("theme_top_share"), prev.get("theme_top_share") if prev is not None else None)
    row["delta_confidence_stable_bucket_count_5d"] = delta(row.get("confidence_stable_bucket_count_5d"), prev.get("confidence_stable_bucket_count_5d") if prev is not None else None)
    row["delta_confidence_avg_calibrated_5d"] = delta(row.get("confidence_avg_calibrated_5d"), prev.get("confidence_avg_calibrated_5d") if prev is not None else None)
    row["delta_score_kpi_alert_count"] = delta(row.get("score_kpi_alert_count"), prev.get("score_kpi_alert_count") if prev is not None else None)
    row["delta_top20_mean_final_score"] = delta(row.get("top20_mean_final_score"), prev.get("top20_mean_final_score") if prev is not None else None)
    row["delta_top20_mean_confidence_score"] = delta(row.get("top20_mean_confidence_score"), prev.get("top20_mean_confidence_score") if prev is not None else None)

    notes: list[str] = []
    gate_reason = str(row.get("buy_gate_reason_summary") or "").strip()
    if str(row.get("buy_gate_overall_status") or "") in {"HOLD", "BLOCK"} and gate_reason:
        repeats = repeat_count(history, "buy_gate_reason_summary", gate_reason)
        if repeats >= 2:
            notes.append(f"buy gate reason repeated {repeats} reviews: {gate_reason}")
    wait_reason = str(row.get("daily_cycle_wait_reasons") or "").strip()
    if str(row.get("daily_cycle_overall_status") or "") == "WAIT" and wait_reason:
        repeats = repeat_count(history, "daily_cycle_wait_reasons", wait_reason)
        if repeats >= 2:
            notes.append(f"WAIT reasons repeated {repeats} reviews: {wait_reason}")
    if prev is not None and str(prev.get("buy_gate_overall_status") or "") != str(row.get("buy_gate_overall_status") or ""):
        notes.append(f"buy gate drift: {prev.get('buy_gate_overall_status')} -> {row.get('buy_gate_overall_status')}")
    if pd.notna(pd.to_numeric(row.get("delta_top20_mean_final_score"), errors="coerce")) and abs(float(row["delta_top20_mean_final_score"])) >= 3.0:
        notes.append(f"score drift: top20 mean final score {fdelta(row['delta_top20_mean_final_score'])}p")
    row["operator_note"] = " | ".join(notes) if notes else "no repeated WAIT/HOLD reason or material weekly drift detected"
    return row


def save_history(row: dict[str, object]) -> pd.DataFrame:
    history = read_csv(OUT_CSV)
    if not history.empty:
        history = history.loc[history["review_week_end"].astype(str).ne(str(row["review_week_end"]))].copy()
    row = enrich(history, row)
    out = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    out["review_week_end_dt"] = pd.to_datetime(out["review_week_end"], errors="coerce")
    out = out.sort_values(["review_week_end_dt", "review_asof_date"]).drop(columns=["review_week_end_dt"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return out


def build_markdown(history: pd.DataFrame) -> str:
    latest = history.iloc[-1]
    trend = history.tail(8).copy()
    for col in ["paper_nav_latest", "top20_mean_final_score", "top20_mean_confidence_score"]:
        trend[col] = trend[col].map(fnum)
    for col in ["benchmark_kospi_avg_excess_return", "sector_top_share", "theme_top_share"]:
        trend[col] = trend[col].map(fpct)

    summary = pd.DataFrame([
        {"item": "snapshot_count", "current": latest["snapshot_count"], "delta_vs_prev": fdelta(latest.get("delta_snapshot_count"), 0)},
        {"item": "matured_snapshot_count_20d", "current": latest["matured_snapshot_count_20d"], "delta_vs_prev": "NA"},
        {"item": "matured_snapshot_count_60d", "current": latest["matured_snapshot_count_60d"], "delta_vs_prev": "NA"},
        {"item": "matured_snapshot_count_90d", "current": latest["matured_snapshot_count_90d"], "delta_vs_prev": "NA"},
        {"item": "paper_nav_latest", "current": fnum(latest.get("paper_nav_latest")), "delta_vs_prev": fdelta(latest.get("delta_paper_nav_latest"))},
        {"item": "benchmark_kospi_avg_excess_return", "current": fpct(latest.get("benchmark_kospi_avg_excess_return")), "delta_vs_prev": fdelta(latest.get("delta_benchmark_kospi_avg_excess_return"), pct=True)},
        {"item": "buy_gate_overall_status", "current": latest.get("buy_gate_overall_status", "NA"), "delta_vs_prev": "see operator note"},
        {"item": "sector_top_share", "current": fpct(latest.get("sector_top_share")), "delta_vs_prev": fdelta(latest.get("delta_sector_top_share"), pct=True)},
        {"item": "theme_top_share", "current": fpct(latest.get("theme_top_share")), "delta_vs_prev": fdelta(latest.get("delta_theme_top_share"), pct=True)},
        {"item": "confidence_stable_bucket_count_5d", "current": latest.get("confidence_stable_bucket_count_5d", "NA"), "delta_vs_prev": fdelta(latest.get("delta_confidence_stable_bucket_count_5d"), 0)},
        {"item": "confidence_avg_calibrated_5d", "current": fnum(latest.get("confidence_avg_calibrated_5d")), "delta_vs_prev": fdelta(latest.get("delta_confidence_avg_calibrated_5d"))},
        {"item": "score_kpi_alert_count", "current": latest.get("score_kpi_alert_count", "NA"), "delta_vs_prev": fdelta(latest.get("delta_score_kpi_alert_count"), 0)},
        {"item": "top20_mean_final_score", "current": fnum(latest.get("top20_mean_final_score")), "delta_vs_prev": fdelta(latest.get("delta_top20_mean_final_score"))},
        {"item": "top20_mean_confidence_score", "current": fnum(latest.get("top20_mean_confidence_score")), "delta_vs_prev": fdelta(latest.get("delta_top20_mean_confidence_score"))},
    ])

    return "\n".join([
        "# Operational Weekly Review",
        "",
        f"- generated_at: {latest['generated_at']}",
        f"- review_week: {latest['review_week_start']} ~ {latest['review_week_end']}",
        f"- review_asof_date: {latest['review_asof_date']}",
        f"- score_formula_version: {latest['score_formula_version']}",
        f"- gate_version: {latest['gate_version']}",
        f"- portfolio_version: {latest['portfolio_version']}",
        f"- confidence_calibration_version: {latest['confidence_calibration_version']}",
        "",
        "## Operator Note",
        f"- {latest['operator_note']}",
        "",
        "## Weekly Summary",
        md_table(summary, ["item", "current", "delta_vs_prev"]),
        "",
        "## Buy Gate",
        f"- overall_status: {latest.get('buy_gate_overall_status', 'NA')}",
        f"- primary_bucket: top{latest.get('buy_gate_primary_bucket', 'NA')}",
        f"- primary_status: {latest.get('buy_gate_primary_status', 'NA')}",
        f"- reason_summary: {latest.get('buy_gate_reason_summary', 'NA')}",
        f"- daily_cycle_status: {latest.get('daily_cycle_overall_status', 'NA')}",
        f"- daily_cycle_wait_reasons: {latest.get('daily_cycle_wait_reasons', 'NA')}",
        "",
        "## Recent Trend",
        md_table(trend, ["review_week_end", "snapshot_count", "buy_gate_overall_status", "daily_cycle_overall_status", "paper_nav_latest", "benchmark_kospi_avg_excess_return", "sector_top_share", "theme_top_share", "score_kpi_overall_status"]),
        "",
    ]) + "\n"


def main() -> int:
    history = save_history(current_row())
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(history), encoding="utf-8")
    print(f"out_md: {OUT_MD}")
    print(f"out_csv: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
