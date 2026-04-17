from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from outcome_maturity import attach_forward_outcomes, load_price_history


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
RANKING_CURRENT_CSV = DATA_DIR / "ranking_final.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
BUY_GATE_JSON = OUTPUT_DIR / "operational_buy_gate.json"
BUY_GATE_HISTORY_CSV = DATA_DIR / "operational_buy_gate_history.csv"

OUT_FAILURE_CSV = DATA_DIR / "model_failure_cases.csv"
OUT_REPORT_MD = OUTPUT_DIR / "model_failure_analysis.md"

HORIZONS = [5, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and analyze model failure cases from ranking history.")
    parser.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    parser.add_argument("--ranking-current-csv", type=Path, default=RANKING_CURRENT_CSV)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--market-status-csv", type=Path, default=MARKET_STATUS_CSV)
    parser.add_argument("--buy-gate-json", type=Path, default=BUY_GATE_JSON)
    parser.add_argument("--buy-gate-history-csv", type=Path, default=BUY_GATE_HISTORY_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_FAILURE_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_REPORT_MD)
    parser.add_argument("--high-confidence-threshold", type=float, default=80.0)
    parser.add_argument("--low-liquidity-threshold", type=float, default=15.0)
    parser.add_argument("--high-risk-penalty-threshold", type=float, default=10.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def load_ranking_snapshots(history_dir: Path, current_csv: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    resolved_history = _resolve(history_dir)
    resolved_current = _resolve(current_csv)
    if resolved_history.exists():
        for path in sorted(resolved_history.glob("*_ranking_final.csv")):
            df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            df["snapshot_file"] = path.name
            frames.append(df)
    if resolved_current.exists():
        current = pd.read_csv(resolved_current, dtype={"code": str}, low_memory=False)
        current["snapshot_file"] = resolved_current.name
        frames.append(current)
    if not frames:
        raise FileNotFoundError("no ranking snapshots found")

    df = pd.concat(frames, ignore_index=True).copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["rank_final"] = pd.to_numeric(df.get("rank_final"), errors="coerce")
    df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
    df["confidence_score"] = pd.to_numeric(df.get("confidence_score"), errors="coerce")
    df["liquidity_score"] = pd.to_numeric(df.get("liquidity_score"), errors="coerce")
    df["ret_5d"] = pd.to_numeric(df.get("ret_5d"), errors="coerce")
    df["ret_10d"] = pd.to_numeric(df.get("ret_10d"), errors="coerce")
    df["mom_20"] = pd.to_numeric(df.get("mom_20"), errors="coerce")
    df["market"] = df.get("market", "").fillna("").astype(str)
    df["sector"] = df.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    df["dominant_theme"] = df.get("dominant_theme", "(none)").fillna("(none)").astype(str).replace({"": "(none)", "nan": "(none)"})
    df["regime"] = df.get("regime", "unknown").fillna("unknown").astype(str)
    df["snapshot_priority"] = df["snapshot_file"].eq(resolved_current.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    return df.dropna(subset=["date", "code", "rank_final"]).copy()


def build_current_gate_history(gate_json_path: Path, gate_history_path: Path) -> pd.DataFrame:
    history_path = _resolve(gate_history_path)
    if history_path.exists():
        history = pd.read_csv(history_path, low_memory=False)
    else:
        history = pd.DataFrame()

    if not history.empty:
        history["asof_date"] = pd.to_datetime(history["asof_date"], errors="coerce").dt.normalize()
        history["bucket"] = pd.to_numeric(history["bucket"], errors="coerce").astype("Int64")

    gate_json_file = _resolve(gate_json_path)
    if not gate_json_file.exists():
        return history

    payload = json.loads(gate_json_file.read_text(encoding="utf-8"))
    asof_date = pd.to_datetime(payload.get("asof_date"), errors="coerce")
    decisions = payload.get("decisions", [])
    rows: list[dict[str, object]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        rows.append(
            {
                "asof_date": asof_date,
                "bucket": pd.to_numeric(decision.get("bucket"), errors="coerce"),
                "status": decision.get("status"),
                "reason_summary": decision.get("reason_summary"),
                "generated_at": payload.get("generated_at"),
            }
        )

    if not rows:
        return history

    appended = pd.DataFrame(rows)
    full = pd.concat([history, appended], ignore_index=True) if not history.empty else appended
    full["asof_date"] = pd.to_datetime(full["asof_date"], errors="coerce").dt.normalize()
    full["bucket"] = pd.to_numeric(full["bucket"], errors="coerce").astype("Int64")
    full = full.sort_values(["asof_date", "bucket", "generated_at"]).drop_duplicates(["asof_date", "bucket"], keep="last").reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(history_path, index=False, encoding="utf-8-sig")
    return full


def build_forward_case_frame(ranking: pd.DataFrame, prices_csv: Path, horizons: list[int]) -> pd.DataFrame:
    prices = load_price_history(prices_csv=_resolve(prices_csv))
    enriched = ranking.copy()
    if "trading_value" not in enriched.columns:
        close = pd.to_numeric(enriched.get("close"), errors="coerce")
        volume = pd.to_numeric(enriched.get("volume"), errors="coerce")
        enriched["trading_value"] = close * volume
    for col in [
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "top_positive_factor",
        "top_positive_value",
        "top_negative_factor",
        "top_negative_value",
        "contrib_ret",
        "contrib_prob",
        "contrib_tech",
        "contrib_qual",
        "contrib_safety",
        "contrib_liquidity",
        "contrib_penalty",
        "explain_text",
    ]:
        if col not in enriched.columns:
            enriched[col] = pd.NA
    base_cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "dominant_theme",
        "regime",
        "rank_final",
        "final_score",
        "confidence_score",
        "liquidity_score",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "ret_score",
        "prob_score",
        "tech_score",
        "qual_score",
        "quality_score",
        "safety_score",
        "risk_penalty",
        "theme_score",
        "trading_value",
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "top_positive_factor",
        "top_positive_value",
        "top_negative_factor",
        "top_negative_value",
        "contrib_ret",
        "contrib_prob",
        "contrib_tech",
        "contrib_qual",
        "contrib_safety",
        "contrib_liquidity",
        "contrib_penalty",
        "explain_text",
    ]
    work = enriched[base_cols].copy()
    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        outcome = attach_forward_outcomes(prices, horizon_days=horizon).rename(
            columns={"date": "date", "realized_return": "realized_return", "realized_mdd": "realized_mdd_like"}
        )
        merged = work.merge(outcome, on=["code", "date"], how="left")
        merged["horizon_days"] = horizon
        frames.append(merged)
    detail = pd.concat(frames, ignore_index=True)
    detail["date"] = pd.to_datetime(detail["date"], errors="coerce").dt.normalize()
    detail["in_top5"] = detail["rank_final"].le(5)
    detail["in_top10"] = detail["rank_final"].le(10)
    detail["in_top20"] = detail["rank_final"].le(20)
    detail["top_bucket_smallest"] = 20
    detail.loc[detail["in_top10"], "top_bucket_smallest"] = 10
    detail.loc[detail["in_top5"], "top_bucket_smallest"] = 5
    return detail


def build_benchmarks(cases: pd.DataFrame, market_status_csv: Path) -> pd.DataFrame:
    market = pd.read_csv(_resolve(market_status_csv), low_memory=False)
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["code"] = "KOSPI"
    market["close"] = pd.to_numeric(market["kospi_close"], errors="coerce")
    market = market.dropna(subset=["date", "close"]).copy()

    kospi_frames: list[pd.DataFrame] = []
    for horizon in sorted(cases["horizon_days"].dropna().astype(int).unique().tolist()):
        bench = attach_forward_outcomes(market[["code", "date", "close"]], horizon_days=horizon).rename(
            columns={"realized_return": "benchmark_return_kospi", "realized_mdd": "benchmark_mdd_kospi"}
        )
        bench["horizon_days"] = horizon
        kospi_frames.append(bench[["date", "horizon_days", "benchmark_return_kospi", "benchmark_mdd_kospi"]])
    kospi = pd.concat(kospi_frames, ignore_index=True) if kospi_frames else pd.DataFrame()

    universe = (
        cases.groupby(["date", "horizon_days"], dropna=False)
        .agg(
            benchmark_return_universe=("realized_return", "mean"),
            benchmark_mdd_universe=("realized_mdd_like", "mean"),
        )
        .reset_index()
    )
    return kospi.merge(universe, on=["date", "horizon_days"], how="outer")


def assign_gate_status(cases: pd.DataFrame, gate_history: pd.DataFrame) -> pd.DataFrame:
    work = cases.copy()
    if gate_history.empty:
        work["buy_gate_status"] = "UNKNOWN"
        work["buy_gate_reason_summary"] = "gate_history_missing"
        work["buy_gate_evaluable"] = False
        return work

    gate_map = gate_history.copy()
    gate_map["gate_bucket"] = gate_map["bucket"].map(lambda x: 5 if int(x) == 5 else (10 if int(x) == 10 else pd.NA))
    gate_map = gate_map.dropna(subset=["gate_bucket"]).copy()
    gate_map["gate_bucket"] = gate_map["gate_bucket"].astype(int)

    work["gate_bucket"] = pd.NA
    work.loc[work["rank_final"].le(5), "gate_bucket"] = 5
    work.loc[work["rank_final"].gt(5) & work["rank_final"].le(10), "gate_bucket"] = 10
    work["buy_gate_evaluable"] = False

    merged = work.merge(
        gate_map[["asof_date", "gate_bucket", "status", "reason_summary"]].rename(columns={"asof_date": "date", "status": "buy_gate_status", "reason_summary": "buy_gate_reason_summary"}),
        on=["date", "gate_bucket"],
        how="left",
    )
    matched_gate = merged["buy_gate_status"].notna()
    merged["buy_gate_status"] = merged["buy_gate_status"].fillna("UNKNOWN")
    merged["buy_gate_reason_summary"] = merged["buy_gate_reason_summary"].fillna("gate_history_missing_for_snapshot")
    merged["buy_gate_evaluable"] = matched_gate
    return merged


def classify_failure_type(row: pd.Series, high_confidence_threshold: float, low_liquidity_threshold: float, high_risk_penalty_threshold: float) -> str:
    if bool(row.get("condition_gate_allowed_underperform", False)):
        return "gate_allowed_underperform"
    if bool(row.get("condition_high_confidence_failure", False)) and pd.to_numeric(row.get("quality_score"), errors="coerce") < 40:
        return "high_confidence_weak_quality"
    if bool(row.get("condition_high_confidence_failure", False)) and pd.to_numeric(row.get("ret_5d"), errors="coerce") > 0 and pd.to_numeric(row.get("mom_20"), errors="coerce") > 0:
        return "high_confidence_momentum_reversal"
    if bool(row.get("condition_high_confidence_failure", False)):
        return "high_confidence_failure"
    if pd.to_numeric(row.get("liquidity_score"), errors="coerce") < low_liquidity_threshold:
        return "low_liquidity_breakdown"
    if pd.to_numeric(row.get("risk_penalty"), errors="coerce") >= high_risk_penalty_threshold:
        return "risk_penalty_ignored"
    if str(row.get("dominant_theme", "(none)")) != "(none)":
        return "theme_exposure_failure"
    return "topn_loss"


def build_failure_cases(
    ranking: pd.DataFrame,
    prices_csv: Path,
    market_status_csv: Path,
    gate_history: pd.DataFrame,
    horizons: list[int],
    high_confidence_threshold: float,
    low_liquidity_threshold: float,
    high_risk_penalty_threshold: float,
) -> pd.DataFrame:
    cases = build_forward_case_frame(ranking, prices_csv=prices_csv, horizons=horizons)
    benchmarks = build_benchmarks(cases, market_status_csv=market_status_csv)
    cases = cases.merge(benchmarks, on=["date", "horizon_days"], how="left")
    cases = assign_gate_status(cases, gate_history)

    cases["excess_return_vs_kospi"] = pd.to_numeric(cases["realized_return"], errors="coerce") - pd.to_numeric(cases["benchmark_return_kospi"], errors="coerce")
    cases["excess_return_vs_universe"] = pd.to_numeric(cases["realized_return"], errors="coerce") - pd.to_numeric(cases["benchmark_return_universe"], errors="coerce")

    cases["condition_topn_loss"] = cases["in_top20"] & pd.to_numeric(cases["realized_return"], errors="coerce").lt(0)
    cases["condition_high_confidence_failure"] = cases["condition_topn_loss"] & pd.to_numeric(cases["confidence_score"], errors="coerce").ge(high_confidence_threshold)
    cases["condition_gate_allowed_underperform"] = (
        cases["buy_gate_status"].astype(str).eq("BUY_ALLOWED")
        & cases["buy_gate_evaluable"].fillna(False)
        & pd.to_numeric(cases["excess_return_vs_kospi"], errors="coerce").lt(0)
    )

    failures = cases.loc[
        cases["condition_topn_loss"] | cases["condition_high_confidence_failure"] | cases["condition_gate_allowed_underperform"]
    ].copy()

    failures["liquidity_band"] = pd.cut(
        pd.to_numeric(failures["liquidity_score"], errors="coerce"),
        bins=[-float("inf"), 10, 20, 40, float("inf")],
        labels=["very_low", "low", "mid", "high"],
    ).astype("string").fillna("unknown")
    failures["recent_5d_bucket"] = pd.cut(
        pd.to_numeric(failures["ret_5d"], errors="coerce"),
        bins=[-float("inf"), -0.05, 0.0, 0.05, float("inf")],
        labels=["down_big", "down", "flat_up", "up_big"],
    ).astype("string").fillna("unknown")
    failures["recent_20d_bucket"] = pd.cut(
        pd.to_numeric(failures["mom_20"], errors="coerce"),
        bins=[-float("inf"), -0.10, 0.0, 0.10, float("inf")],
        labels=["weak", "soft", "firm", "strong"],
    ).astype("string").fillna("unknown")
    failures["theme_or_sector_focus"] = failures["dominant_theme"].where(failures["dominant_theme"].ne("(none)"), failures["sector"])
    failures["failure_type"] = failures.apply(
        classify_failure_type,
        axis=1,
        high_confidence_threshold=high_confidence_threshold,
        low_liquidity_threshold=low_liquidity_threshold,
        high_risk_penalty_threshold=high_risk_penalty_threshold,
    )
    failures["cluster_key"] = (
        failures["failure_type"].astype(str)
        + "|"
        + failures["regime"].astype(str)
        + "|"
        + failures["liquidity_band"].astype(str)
        + "|"
        + failures["recent_5d_bucket"].astype(str)
        + "|"
        + failures["recent_20d_bucket"].astype(str)
        + "|"
        + failures["theme_or_sector_focus"].astype(str)
    )
    cluster_sizes = failures["cluster_key"].value_counts()
    failures["cluster_size"] = failures["cluster_key"].map(cluster_sizes)
    failures["cluster_id"] = "FCL-" + failures["cluster_key"].map({key: f"{idx:03d}" for idx, key in enumerate(cluster_sizes.index, start=1)})

    ordered = [
        "date",
        "code",
        "name",
        "horizon_days",
        "rank_final",
        "top_bucket_smallest",
        "in_top5",
        "in_top10",
        "in_top20",
        "final_score",
        "confidence_score",
        "ret_score",
        "prob_score",
        "tech_score",
        "quality_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
        "contrib_ret",
        "contrib_prob",
        "contrib_tech",
        "contrib_qual",
        "contrib_safety",
        "contrib_liquidity",
        "contrib_penalty",
        "sector",
        "dominant_theme",
        "regime",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "trading_value",
        "buy_gate_status",
        "buy_gate_reason_summary",
        "buy_gate_evaluable",
        "realized_return",
        "realized_mdd_like",
        "benchmark_return_kospi",
        "benchmark_return_universe",
        "excess_return_vs_kospi",
        "excess_return_vs_universe",
        "condition_topn_loss",
        "condition_high_confidence_failure",
        "condition_gate_allowed_underperform",
        "failure_type",
        "liquidity_band",
        "recent_5d_bucket",
        "recent_20d_bucket",
        "theme_or_sector_focus",
        "cluster_id",
        "cluster_key",
        "cluster_size",
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "top_positive_factor",
        "top_positive_value",
        "top_negative_factor",
        "top_negative_value",
        "explain_text",
    ]
    return failures[ordered].sort_values(["date", "horizon_days", "rank_final", "code"]).reset_index(drop=True)


def build_report(failures: pd.DataFrame, gate_history: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = int(len(failures))
    topn_loss_count = int(failures["condition_topn_loss"].fillna(False).sum()) if not failures.empty else 0
    high_conf_count = int(failures["condition_high_confidence_failure"].fillna(False).sum()) if not failures.empty else 0
    gate_underperform_count = int(failures["condition_gate_allowed_underperform"].fillna(False).sum()) if not failures.empty else 0

    failure_type_table = pd.DataFrame()
    cluster_table = pd.DataFrame()
    regime_table = pd.DataFrame()
    sector_table = pd.DataFrame()
    examples = pd.DataFrame()

    if not failures.empty:
        failure_type_table = (
            failures.groupby("failure_type", dropna=False)
            .agg(
                cases=("code", "size"),
                avg_return=("realized_return", "mean"),
                avg_excess_kospi=("excess_return_vs_kospi", "mean"),
                avg_confidence=("confidence_score", "mean"),
            )
            .reset_index()
            .sort_values(["cases", "avg_return"], ascending=[False, True])
        )
        failure_type_table["avg_return"] = failure_type_table["avg_return"].map(_fmt_pct)
        failure_type_table["avg_excess_kospi"] = failure_type_table["avg_excess_kospi"].map(_fmt_pct)
        failure_type_table["avg_confidence"] = failure_type_table["avg_confidence"].map(_fmt_num)

        cluster_table = (
            failures.groupby(["cluster_id", "failure_type", "regime", "liquidity_band", "recent_5d_bucket", "recent_20d_bucket", "theme_or_sector_focus"], dropna=False)
            .agg(cases=("code", "size"), avg_return=("realized_return", "mean"), avg_excess_kospi=("excess_return_vs_kospi", "mean"))
            .reset_index()
            .sort_values(["cases", "avg_return"], ascending=[False, True])
            .head(10)
        )
        cluster_table["avg_return"] = cluster_table["avg_return"].map(_fmt_pct)
        cluster_table["avg_excess_kospi"] = cluster_table["avg_excess_kospi"].map(_fmt_pct)

        regime_table = (
            failures.groupby("regime", dropna=False)
            .agg(cases=("code", "size"), avg_return=("realized_return", "mean"), avg_confidence=("confidence_score", "mean"))
            .reset_index()
            .sort_values("cases", ascending=False)
        )
        regime_table["avg_return"] = regime_table["avg_return"].map(_fmt_pct)
        regime_table["avg_confidence"] = regime_table["avg_confidence"].map(_fmt_num)

        sector_table = (
            failures.groupby(["sector", "dominant_theme"], dropna=False)
            .agg(cases=("code", "size"), avg_return=("realized_return", "mean"))
            .reset_index()
            .sort_values(["cases", "avg_return"], ascending=[False, True])
            .head(10)
        )
        sector_table["avg_return"] = sector_table["avg_return"].map(_fmt_pct)

        examples = failures.sort_values(["condition_high_confidence_failure", "condition_gate_allowed_underperform", "realized_return"], ascending=[False, False, True]).head(10).copy()
        for col in ["realized_return", "excess_return_vs_kospi"]:
            examples[col] = examples[col].map(_fmt_pct)
        examples["confidence_score"] = examples["confidence_score"].map(_fmt_num)

    gate_coverage_dates = int(pd.to_datetime(gate_history.get("asof_date"), errors="coerce").dropna().nunique()) if not gate_history.empty else 0

    lines = [
        "# Model Failure Analysis",
        "",
        f"- generated_at: {generated_at}",
        f"- failure_case_rows: {total}",
        f"- topn_loss_cases: {topn_loss_count}",
        f"- high_confidence_failure_cases: {high_conf_count}",
        f"- gate_allowed_underperform_cases: {gate_underperform_count}",
        f"- gate_history_dates_available: {gate_coverage_dates}",
        "",
        "## Failure Type Breakdown",
        _markdown_table(failure_type_table, ["failure_type", "cases", "avg_return", "avg_excess_kospi", "avg_confidence"]),
        "",
        "## Pattern Clusters",
        _markdown_table(cluster_table, ["cluster_id", "failure_type", "regime", "liquidity_band", "recent_5d_bucket", "recent_20d_bucket", "theme_or_sector_focus", "cases", "avg_return", "avg_excess_kospi"]),
        "",
        "## Regime Concentration",
        _markdown_table(regime_table, ["regime", "cases", "avg_return", "avg_confidence"]),
        "",
        "## Sector / Theme Concentration",
        _markdown_table(sector_table, ["sector", "dominant_theme", "cases", "avg_return"]),
        "",
        "## Representative Failures",
        _markdown_table(examples, ["date", "code", "name", "horizon_days", "rank_final", "failure_type", "confidence_score", "realized_return", "excess_return_vs_kospi", "sector", "dominant_theme", "regime"]),
        "",
        "## Notes",
        "- `condition_gate_allowed_underperform` is exact only for dates that exist in `data/operational_buy_gate_history.csv`.",
        "- Historical gate rows are appended automatically from the current `outputs/operational_buy_gate.json`, so this condition becomes more complete over time.",
        "- Clustering is deterministic signature clustering over failure type, regime, liquidity band, recent move buckets, and sector/theme focus.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ranking = load_ranking_snapshots(args.ranking_history_dir, args.ranking_current_csv)
    gate_history = build_current_gate_history(args.buy_gate_json, args.buy_gate_history_csv)
    failures = build_failure_cases(
        ranking=ranking,
        prices_csv=args.prices_csv,
        market_status_csv=args.market_status_csv,
        gate_history=gate_history,
        horizons=sorted(set(int(x) for x in args.horizons)),
        high_confidence_threshold=float(args.high_confidence_threshold),
        low_liquidity_threshold=float(args.low_liquidity_threshold),
        high_risk_penalty_threshold=float(args.high_risk_penalty_threshold),
    )

    out_csv = _resolve(args.out_csv)
    out_md = _resolve(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_md.write_text(build_report(failures, gate_history), encoding="utf-8")

    print(f"model_failure_cases_csv: {out_csv}")
    print(f"model_failure_analysis_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
