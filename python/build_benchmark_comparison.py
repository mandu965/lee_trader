from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from outcome_maturity import (
    attach_forward_outcomes,
    build_price_reference,
    evaluate_prediction_maturity_rows,
    load_price_history,
)
from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

ARCHIVE_CSV = DATA_DIR / "ranking_snapshot_archive.csv"
RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
RANKING_CURRENT_CSV = DATA_DIR / "ranking_final.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
BENCHMARK_CSV = DATA_DIR / "market_status.csv"
OUTPUT_MD = OUTPUT_DIR / "benchmark_comparison_report.md"
OUTPUT_CSV = OUTPUT_DIR / "benchmark_comparison.csv"

HORIZONS = [5, 20, 60, 90]
TOP_BUCKETS = [20, 10, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare operational recommendation returns against multiple benchmarks.")
    p.add_argument("--archive-csv", type=Path, default=ARCHIVE_CSV)
    p.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    p.add_argument("--ranking-current-csv", type=Path, default=RANKING_CURRENT_CSV)
    p.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    p.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_CSV)
    p.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    p.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    return p.parse_args()


def _fmt(value: object, digits: int = 4) -> str:
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
    rendered = [[str(item) for item in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def load_archive(archive_csv: Path) -> pd.DataFrame:
    if not archive_csv.exists():
        raise FileNotFoundError(f"archive not found: {archive_csv}")
    df = pd.read_csv(archive_csv, dtype={"code": str}, low_memory=False)
    if df.empty:
        raise ValueError("archive is empty")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.normalize()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df.dropna(subset=["asof_date", "code", "rank"]).copy()


def load_ranking_snapshots(history_dir: Path, current_csv: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if history_dir.exists():
        for path in sorted(history_dir.glob("*_ranking_final.csv")):
            df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            df["snapshot_file"] = path.name
            frames.append(df)
    if current_csv.exists():
        current = pd.read_csv(current_csv, dtype={"code": str}, low_memory=False)
        current["snapshot_file"] = current_csv.name
        frames.append(current)
    if not frames:
        raise FileNotFoundError("no ranking snapshots found")

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["rank_final"] = pd.to_numeric(df.get("rank_final"), errors="coerce")
    df["rank_v2"] = pd.to_numeric(df.get("rank_v2"), errors="coerce")
    df["market"] = df.get("market", "").fillna("").astype(str)
    df["regime"] = df.get("regime", "").fillna("").astype(str)
    df = df.dropna(subset=["date", "code"]).copy()
    df["snapshot_priority"] = df["snapshot_file"].eq(current_csv.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    return df


def attach_forward_metrics(df: pd.DataFrame, *, date_col: str, prices_csv: Path) -> pd.DataFrame:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    price_history = load_price_history(prices_csv=prices_csv)
    price_reference = build_price_reference(price_history[["code", "date"]].copy())

    for horizon in HORIZONS:
        maturity = evaluate_prediction_maturity_rows(
            work[["code", date_col]].rename(columns={date_col: "asof_date"}),
            price_reference=price_reference,
            horizon_days=horizon,
            as_of_col="asof_date",
            code_col="code",
        ).rename(
            columns={
                "asof_date": date_col,
                "available_future_trading_days": f"available_future_trading_days_{horizon}d",
                "required_future_trading_days": f"required_future_trading_days_{horizon}d",
                "is_matured": f"is_matured_{horizon}d",
                "maturity_status": f"maturity_status_{horizon}d",
                "last_available_price_date": f"last_available_price_date_{horizon}d",
            }
        )
        outcome = attach_forward_outcomes(price_history, horizon_days=horizon).rename(
            columns={
                "date": date_col,
                "realized_return": f"forward_return_{horizon}d",
                "realized_mdd": f"forward_mdd_like_{horizon}d",
            }
        )
        work = work.merge(maturity, on=["code", date_col], how="left")
        work = work.merge(outcome, on=["code", date_col], how="left")
        mature_mask = work.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
        work.loc[~mature_mask, [f"forward_return_{horizon}d", f"forward_mdd_like_{horizon}d"]] = pd.NA
    return work


def load_index_series(benchmark_csv: Path, column_name: str, label: str) -> pd.DataFrame:
    if not benchmark_csv.exists():
        return pd.DataFrame(columns=["code", "date", "close", "benchmark_name"])
    df = pd.read_csv(benchmark_csv)
    if "date" not in df.columns or column_name not in df.columns:
        return pd.DataFrame(columns=["code", "date", "close", "benchmark_name"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df[column_name], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["code", "date", "close", "benchmark_name"])
    df["code"] = label
    df["benchmark_name"] = label
    return df[["code", "date", "close", "benchmark_name"]].reset_index(drop=True)


def build_index_benchmark_returns(benchmark_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, column_name in [("KOSPI", "kospi_close"), ("KOSDAQ", "kosdaq_close")]:
        hist = load_index_series(benchmark_csv, column_name, label)
        if hist.empty:
            for horizon in HORIZONS:
                rows.append({"asof_date": pd.NaT, "horizon_days": horizon, "benchmark_name": label, "benchmark_available": False})
            continue
        for horizon in HORIZONS:
            outcome = attach_forward_outcomes(hist[["code", "date", "close"]], horizon_days=horizon).rename(
                columns={
                    "date": "asof_date",
                    "realized_return": "benchmark_return",
                    "realized_mdd": "benchmark_mdd_like",
                }
            )
            outcome["horizon_days"] = horizon
            outcome["benchmark_name"] = label
            outcome["benchmark_available"] = True
            outcome["benchmark_hit_ratio"] = pd.NA
            outcome["benchmark_win_flag"] = pd.to_numeric(outcome["benchmark_return"], errors="coerce").gt(0)
            rows.extend(
                outcome[["asof_date", "horizon_days", "benchmark_name", "benchmark_return", "benchmark_mdd_like", "benchmark_hit_ratio", "benchmark_win_flag", "benchmark_available"]]
                .to_dict(orient="records")
            )
    out = pd.DataFrame(rows)
    if "asof_date" in out.columns:
        out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce").dt.normalize()
    return out


def build_universe_equal_weight_benchmark(full_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_dates = sorted(pd.to_datetime(full_detail["date"], errors="coerce").dropna().unique().tolist())
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        mdd_col = f"forward_mdd_like_{horizon}d"
        matured_mask = full_detail.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
        matured = full_detail.loc[matured_mask].copy()
        for asof_date in all_dates:
            day_df = matured.loc[matured["date"] == asof_date].copy()
            returns = pd.to_numeric(day_df[ret_col], errors="coerce")
            rows.append(
                {
                    "asof_date": pd.to_datetime(asof_date),
                    "horizon_days": horizon,
                    "benchmark_name": "UNIVERSE_EQUAL_WEIGHT",
                    "benchmark_return": float(returns.mean()) if returns.notna().any() else None,
                    "benchmark_mdd_like": float(pd.to_numeric(day_df[mdd_col], errors="coerce").mean()) if day_df[mdd_col].notna().any() else None,
                    "benchmark_hit_ratio": float((returns > 0).mean()) if returns.notna().any() else None,
                    "benchmark_win_flag": bool(float(returns.mean()) > 0) if returns.notna().any() else pd.NA,
                    "benchmark_available": True,
                }
            )
    return pd.DataFrame(rows)


def build_baseline_ranking_benchmark(full_detail: pd.DataFrame) -> pd.DataFrame:
    rank_source = "rank_v2" if "rank_v2" in full_detail.columns and pd.to_numeric(full_detail["rank_v2"], errors="coerce").notna().any() else ""
    if not rank_source:
        rank_source = "simple_factor_rank"
        baseline = full_detail.copy()
        factor_cols = ["ret_score", "prob_score", "tech_score", "qual_score", "safety_score"]
        baseline["simple_factor_score"] = baseline[factor_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        baseline["simple_factor_rank"] = baseline.groupby("date")["simple_factor_score"].rank(method="first", ascending=False)
    else:
        baseline = full_detail.copy()
    baseline[rank_source] = pd.to_numeric(baseline[rank_source], errors="coerce")

    rows: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        selected = baseline.loc[baseline[rank_source] <= top_n].copy()
        all_dates = sorted(pd.to_datetime(selected["date"], errors="coerce").dropna().unique().tolist())
        for horizon in HORIZONS:
            ret_col = f"forward_return_{horizon}d"
            mdd_col = f"forward_mdd_like_{horizon}d"
            matured_mask = selected.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
            matured = selected.loc[matured_mask].copy()
            for asof_date in all_dates:
                day_df = matured.loc[matured["date"] == asof_date].copy()
                returns = pd.to_numeric(day_df[ret_col], errors="coerce")
                rows.append(
                    {
                        "asof_date": pd.to_datetime(asof_date),
                        "top_n": int(top_n),
                        "horizon_days": horizon,
                        "benchmark_name": "BASELINE_RANKING_V2" if rank_source == "rank_v2" else "SIMPLE_FACTOR_BASELINE",
                        "benchmark_return": float(returns.mean()) if returns.notna().any() else None,
                        "benchmark_mdd_like": float(pd.to_numeric(day_df[mdd_col], errors="coerce").mean()) if day_df[mdd_col].notna().any() else None,
                        "benchmark_hit_ratio": float((returns > 0).mean()) if returns.notna().any() else None,
                        "benchmark_win_flag": bool(float(returns.mean()) > 0) if returns.notna().any() else pd.NA,
                        "benchmark_available": True,
                    }
                )
    return pd.DataFrame(rows)


def build_recommendation_daily(detail: pd.DataFrame, full_snapshots: pd.DataFrame) -> pd.DataFrame:
    regime_by_date = (
        full_snapshots.sort_values(["date", "code"])
        .groupby("date")["regime"]
        .agg(lambda s: str(s.dropna().iloc[0]) if not s.dropna().empty else "")
        .reset_index()
        .rename(columns={"date": "asof_date"})
    )

    rows: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        bucket = detail.loc[detail["rank"] <= top_n].copy()
        for asof_date, day_df in bucket.groupby("asof_date", sort=True):
            for horizon in HORIZONS:
                ret_col = f"forward_return_{horizon}d"
                mdd_col = f"forward_mdd_like_{horizon}d"
                matured_mask = day_df.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
                matured = day_df.loc[matured_mask].copy()
                returns = pd.to_numeric(matured[ret_col], errors="coerce")
                rows.append(
                    {
                        "asof_date": pd.to_datetime(asof_date),
                        "top_n": int(top_n),
                        "horizon_days": int(horizon),
                        "snapshot_count": int(len(day_df)),
                        "matured_count": int(len(matured)),
                        "portfolio_return": float(returns.mean()) if returns.notna().any() else None,
                        "portfolio_median_return": float(returns.median()) if returns.notna().any() else None,
                        "portfolio_hit_ratio": float((returns > 0).mean()) if returns.notna().any() else None,
                        "portfolio_mdd_like": float(pd.to_numeric(matured[mdd_col], errors="coerce").mean()) if matured[mdd_col].notna().any() else None,
                        "portfolio_win_flag": bool(float(returns.mean()) > 0) if returns.notna().any() else pd.NA,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.merge(regime_by_date, on="asof_date", how="left")
        out["regime"] = out.get("regime", "").fillna("").astype(str)
    return out


def expand_index_benchmarks(index_benchmarks: pd.DataFrame, recommendation_daily: pd.DataFrame) -> pd.DataFrame:
    if recommendation_daily.empty:
        return pd.DataFrame()
    keys = recommendation_daily[["asof_date", "top_n", "horizon_days"]].drop_duplicates().copy()
    expanded_frames: list[pd.DataFrame] = []
    for benchmark_name in ["KOSPI", "KOSDAQ"]:
        bench = index_benchmarks.loc[index_benchmarks["benchmark_name"] == benchmark_name].copy()
        if bench.empty:
            temp = keys.copy()
            temp["benchmark_name"] = benchmark_name
            temp["benchmark_return"] = pd.NA
            temp["benchmark_mdd_like"] = pd.NA
            temp["benchmark_hit_ratio"] = pd.NA
            temp["benchmark_win_flag"] = pd.NA
            temp["benchmark_available"] = False
            expanded_frames.append(temp)
            continue
        temp = keys.merge(
            bench[["asof_date", "horizon_days", "benchmark_name", "benchmark_return", "benchmark_mdd_like", "benchmark_hit_ratio", "benchmark_win_flag", "benchmark_available"]],
            on=["asof_date", "horizon_days"],
            how="left",
        )
        temp["benchmark_name"] = benchmark_name
        temp["benchmark_available"] = temp["benchmark_available"].where(temp["benchmark_available"].notna(), False).astype(bool)
        expanded_frames.append(temp)
    return pd.concat(expanded_frames, ignore_index=True) if expanded_frames else pd.DataFrame()


def build_comparison_rows(recommendation_daily: pd.DataFrame, benchmark_daily: pd.DataFrame) -> pd.DataFrame:
    compare = recommendation_daily.merge(
        benchmark_daily,
        on=["asof_date", "top_n", "horizon_days"],
        how="left",
    )
    compare["benchmark_name"] = compare["benchmark_name"].fillna("UNKNOWN")
    compare["benchmark_available"] = compare["benchmark_available"].where(compare["benchmark_available"].notna(), False).astype(bool)
    compare["excess_return"] = pd.to_numeric(compare["portfolio_return"], errors="coerce") - pd.to_numeric(compare["benchmark_return"], errors="coerce")
    compare["hit_ratio_diff"] = pd.to_numeric(compare["portfolio_hit_ratio"], errors="coerce") - pd.to_numeric(compare["benchmark_hit_ratio"], errors="coerce")
    compare["excess_win_flag"] = pd.to_numeric(compare["portfolio_return"], errors="coerce") > pd.to_numeric(compare["benchmark_return"], errors="coerce")
    return compare


def summarize_comparison(compare: pd.DataFrame, *, regime_split: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["top_n", "horizon_days", "benchmark_name"]
    if regime_split:
        group_cols.append("regime")
    grouped = compare.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        key_values = list(keys) if isinstance(keys, tuple) else [keys]
        row = dict(zip(group_cols, key_values))
        available = group.loc[group["benchmark_available"].fillna(False)].copy()
        matured = available.loc[pd.to_numeric(available["portfolio_return"], errors="coerce").notna()].copy()
        row.update(
            {
                "dates_total": int(group["asof_date"].nunique()),
                "dates_benchmark_available": int(available["asof_date"].nunique()),
                "dates_matured": int(matured["asof_date"].nunique()),
                "avg_portfolio_return": float(pd.to_numeric(matured["portfolio_return"], errors="coerce").mean()) if not matured.empty else None,
                "median_portfolio_return": float(pd.to_numeric(matured["portfolio_median_return"], errors="coerce").mean()) if not matured.empty else None,
                "portfolio_win_rate": float(pd.to_numeric(matured["portfolio_win_flag"], errors="coerce").mean()) if not matured.empty else None,
                "portfolio_hit_ratio": float(pd.to_numeric(matured["portfolio_hit_ratio"], errors="coerce").mean()) if not matured.empty else None,
                "avg_portfolio_mdd_like": float(pd.to_numeric(matured["portfolio_mdd_like"], errors="coerce").mean()) if not matured.empty else None,
                "avg_benchmark_return": float(pd.to_numeric(matured["benchmark_return"], errors="coerce").mean()) if not matured.empty else None,
                "benchmark_win_rate": float(pd.to_numeric(matured["benchmark_win_flag"], errors="coerce").mean()) if not matured.empty else None,
                "benchmark_hit_ratio": float(pd.to_numeric(matured["benchmark_hit_ratio"], errors="coerce").mean()) if not matured.empty else None,
                "avg_benchmark_mdd_like": float(pd.to_numeric(matured["benchmark_mdd_like"], errors="coerce").mean()) if not matured.empty else None,
                "avg_excess_return": float(pd.to_numeric(matured["excess_return"], errors="coerce").mean()) if not matured.empty else None,
                "median_excess_return": float(pd.to_numeric(matured["excess_return"], errors="coerce").median()) if not matured.empty else None,
                "excess_hit_ratio": float(pd.to_numeric(matured["excess_win_flag"], errors="coerce").mean()) if not matured.empty else None,
                "avg_hit_ratio_diff": float(pd.to_numeric(matured["hit_ratio_diff"], errors="coerce").mean()) if not matured.empty else None,
                "benchmark_available": bool(group["benchmark_available"].fillna(False).any()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_markdown(summary: pd.DataFrame, regime_summary: pd.DataFrame, compare: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Benchmark Comparison Report",
        "",
        f"- generated_at: {generated_at}",
        f"- tracked_horizons: {', '.join(f'{x}d' for x in HORIZONS)}",
        f"- tracked_buckets: {', '.join(f'Top{x}' for x in TOP_BUCKETS)}",
        f"- benchmark_set: KOSPI, KOSDAQ, UNIVERSE_EQUAL_WEIGHT, BASELINE_RANKING_V2/SIMPLE_FACTOR_BASELINE",
        "",
    ]

    availability = (
        summary[["benchmark_name", "benchmark_available"]]
        .drop_duplicates()
        .sort_values("benchmark_name")
    )
    availability_rows = [[row.benchmark_name, "yes" if bool(row.benchmark_available) else "no"] for row in availability.itertuples(index=False)]
    lines.extend(
        [
            "## Benchmark Availability",
            "",
            _markdown_table(availability_rows, ["benchmark", "available"]),
            "",
        ]
    )

    headline = summary.copy()
    headline_rows = [
        [
            f"Top{int(row.top_n)}",
            f"{int(row.horizon_days)}d",
            row.benchmark_name,
            int(row.dates_matured),
            _fmt_pct(row.avg_portfolio_return),
            _fmt_pct(row.avg_benchmark_return),
            _fmt_pct(row.avg_excess_return),
            _fmt(row.portfolio_win_rate),
            _fmt(row.benchmark_win_rate),
            _fmt(row.excess_hit_ratio),
        ]
        for row in headline.sort_values(["top_n", "horizon_days", "benchmark_name"]).itertuples(index=False)
    ]
    lines.extend(
        [
            "## Overall Comparison",
            "",
            _markdown_table(
                headline_rows,
                ["bucket", "horizon", "benchmark", "dates_matured", "avg_portfolio_return", "avg_benchmark_return", "avg_excess_return", "portfolio_win_rate", "benchmark_win_rate", "excess_hit_ratio"],
            ),
            "",
        ]
    )

    if not regime_summary.empty:
        regime_rows = [
            [
                f"Top{int(row.top_n)}",
                f"{int(row.horizon_days)}d",
                row.benchmark_name,
                row.regime or "(blank)",
                int(row.dates_matured),
                _fmt_pct(row.avg_excess_return),
                _fmt(row.excess_hit_ratio),
            ]
            for row in regime_summary.sort_values(["regime", "top_n", "horizon_days", "benchmark_name"]).itertuples(index=False)
        ]
        lines.extend(
            [
                "## Regime Split",
                "",
                _markdown_table(regime_rows, ["bucket", "horizon", "benchmark", "regime", "dates_matured", "avg_excess_return", "excess_hit_ratio"]),
                "",
            ]
        )

    immature_count = int(pd.to_numeric(compare["portfolio_return"], errors="coerce").isna().sum()) if not compare.empty else 0
    if immature_count > 0:
        lines.extend(
            [
                "## Note",
                "",
                f"- Some comparison rows are still immature because the snapshot archive does not yet have enough future trading rows.",
                f"- immature_comparison_rows: {immature_count}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    archive = load_archive(args.archive_csv)

    calibration_start = str(
        get_production_config_value(["confidence_calibration", "calibration_start_date"], "") or ""
    ).strip()
    if calibration_start:
        start_dt = pd.to_datetime(calibration_start, errors="coerce")
        if pd.notna(start_dt):
            before = len(archive)
            archive = archive.loc[archive["asof_date"] >= start_dt].copy()
            print(f"calibration_start_date={calibration_start}: archive filtered {before} -> {len(archive)} rows")

    if archive.empty:
        empty_summary = pd.DataFrame(columns=["top_n", "horizon_days", "benchmark_name", "dates_matured", "avg_portfolio_return", "avg_benchmark_return", "avg_excess_return"])
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        empty_summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            f"# Benchmark Comparison Report\n\n- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- calibration_start_date: {calibration_start}\n"
            "- status: no_archive_data_yet (fresh start, awaiting new model data)\n\n"
            "No ranked snapshot data available after the calibration start date.\n"
            "Benchmark comparison will populate once D+20 periods mature for new model entries.\n",
            encoding="utf-8",
        )
        print(f"comparison_rows: 0 (no archive data after {calibration_start})")
        print(f"report_path: {args.out_md}")
        print(f"csv_path: {args.out_csv}")
        return 0

    full_snapshots = load_ranking_snapshots(args.ranking_history_dir, args.ranking_current_csv)

    archive_detail = attach_forward_metrics(archive, date_col="asof_date", prices_csv=args.prices_csv)
    full_detail = attach_forward_metrics(full_snapshots, date_col="date", prices_csv=args.prices_csv)

    recommendation_daily = build_recommendation_daily(archive_detail, full_snapshots)
    index_benchmarks = build_index_benchmark_returns(args.benchmark_csv)
    index_benchmark_daily = expand_index_benchmarks(index_benchmarks, recommendation_daily)
    universe_benchmark_daily = build_universe_equal_weight_benchmark(full_detail)
    universe_benchmark_daily = recommendation_daily[["asof_date", "top_n", "horizon_days"]].drop_duplicates().merge(
        universe_benchmark_daily,
        on=["asof_date", "horizon_days"],
        how="left",
    )
    baseline_benchmark_daily = build_baseline_ranking_benchmark(full_detail)

    benchmark_frames = [frame for frame in [index_benchmark_daily, universe_benchmark_daily, baseline_benchmark_daily] if not frame.empty]
    benchmark_daily = pd.concat(benchmark_frames, ignore_index=True, sort=False) if benchmark_frames else pd.DataFrame()

    compare = build_comparison_rows(recommendation_daily, benchmark_daily)
    summary = summarize_comparison(compare, regime_split=False)
    regime_summary = summarize_comparison(compare.loc[compare["regime"].astype(str).str.len() > 0].copy(), regime_split=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    report = build_markdown(summary, regime_summary, compare)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")

    print(f"comparison_rows: {len(compare)}")
    print(f"summary_rows: {len(summary)}")
    print(f"report_path: {args.out_md}")
    print(f"csv_path: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
