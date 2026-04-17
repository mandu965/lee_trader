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


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

ARCHIVE_CSV = DATA_DIR / "ranking_snapshot_archive.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
BENCHMARK_CSV = DATA_DIR / "market_status.csv"
OUTPUT_MD = OUTPUT_DIR / "operational_forward_return_report.md"
OUTPUT_BY_DAY_CSV = OUTPUT_DIR / "operational_forward_return_by_day.csv"

HORIZONS = [5, 20, 60, 90]
TOP_BUCKETS = [20, 10, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build forward return tracking report from ranking snapshot archive.")
    p.add_argument("--archive-csv", type=Path, default=ARCHIVE_CSV)
    p.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    p.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_CSV)
    p.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    p.add_argument("--out-by-day-csv", type=Path, default=OUTPUT_BY_DAY_CSV)
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
        raise ValueError("ranking snapshot archive is empty")
    required = {"asof_date", "rank", "code", "final_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"archive missing required columns: {sorted(missing)}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.normalize()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df = df.dropna(subset=["asof_date", "rank", "code"]).copy()
    df["rank"] = df["rank"].round().astype(int)
    return df


def load_benchmark_history(benchmark_csv: Path) -> pd.DataFrame:
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"benchmark csv not found: {benchmark_csv}")
    df = pd.read_csv(benchmark_csv)
    if "date" not in df.columns or "kospi_close" not in df.columns:
        raise ValueError("benchmark csv must contain date and kospi_close")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["kospi_close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["date"]).copy()
    df["code"] = "KOSPI"
    return df[["code", "date", "close"]].reset_index(drop=True)


def attach_forward_metrics(archive: pd.DataFrame, prices_csv: Path, benchmark_csv: Path) -> pd.DataFrame:
    work = archive.copy()
    price_history = load_price_history(prices_csv=prices_csv)
    price_reference = build_price_reference(price_history[["code", "date"]].copy())
    work["asof_date"] = pd.to_datetime(work["asof_date"], errors="coerce").dt.normalize()

    benchmark_history = load_benchmark_history(benchmark_csv)
    benchmark_frames = []
    for horizon in HORIZONS:
        maturity = evaluate_prediction_maturity_rows(
            work[["code", "asof_date"]],
            price_reference=price_reference,
            horizon_days=horizon,
            as_of_col="asof_date",
            code_col="code",
        )
        maturity = maturity.rename(
            columns={
                "available_future_trading_days": f"available_future_trading_days_{horizon}d",
                "required_future_trading_days": f"required_future_trading_days_{horizon}d",
                "is_matured": f"is_matured_{horizon}d",
                "maturity_status": f"maturity_status_{horizon}d",
                "last_available_price_date": f"last_available_price_date_{horizon}d",
            }
        )
        outcome = attach_forward_outcomes(price_history, horizon_days=horizon).rename(
            columns={
                "date": "asof_date",
                "realized_return": f"forward_return_{horizon}d",
                "realized_mdd": f"forward_mdd_like_{horizon}d",
            }
        )
        work = work.merge(maturity, on=["code", "asof_date"], how="left")
        work = work.merge(outcome, on=["code", "asof_date"], how="left")
        mature_mask = work.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
        work.loc[~mature_mask, [f"forward_return_{horizon}d", f"forward_mdd_like_{horizon}d"]] = pd.NA

        bench = attach_forward_outcomes(benchmark_history, horizon_days=horizon).rename(
            columns={
                "date": "asof_date",
                "realized_return": f"benchmark_return_{horizon}d",
                "realized_mdd": f"benchmark_mdd_like_{horizon}d",
            }
        )
        benchmark_frames.append(bench[["asof_date", f"benchmark_return_{horizon}d", f"benchmark_mdd_like_{horizon}d"]])

    benchmark_join = benchmark_frames[0]
    for frame in benchmark_frames[1:]:
        benchmark_join = benchmark_join.merge(frame, on="asof_date", how="outer")
    work = work.merge(benchmark_join, on="asof_date", how="left")

    for horizon in HORIZONS:
        work[f"excess_return_{horizon}d"] = (
            pd.to_numeric(work[f"forward_return_{horizon}d"], errors="coerce")
            - pd.to_numeric(work[f"benchmark_return_{horizon}d"], errors="coerce")
        )

    work["maturity_state"] = "immature"
    for horizon in HORIZONS:
        matured_mask = work.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
        work.loc[matured_mask, "maturity_state"] = f"matured_{horizon}d"
    return work


def summarize_by_day(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        bucket = detail.loc[detail["rank"] <= top_n].copy()
        for asof_date, day_df in bucket.groupby("asof_date", sort=True):
            for horizon in HORIZONS:
                ret_col = f"forward_return_{horizon}d"
                mdd_col = f"forward_mdd_like_{horizon}d"
                excess_col = f"excess_return_{horizon}d"
                bench_col = f"benchmark_return_{horizon}d"
                matured_mask = day_df.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
                matured = day_df.loc[matured_mask].copy()
                rows.append(
                    {
                        "asof_date": pd.to_datetime(asof_date).strftime("%Y-%m-%d"),
                        "top_n": int(top_n),
                        "horizon_days": int(horizon),
                        "snapshot_count": int(len(day_df)),
                        "matured_count": int(len(matured)),
                        "maturity_state": "immature" if matured.empty else f"matured_{horizon}d",
                        "avg_return": float(pd.to_numeric(matured[ret_col], errors="coerce").mean()) if not matured.empty else None,
                        "median_return": float(pd.to_numeric(matured[ret_col], errors="coerce").median()) if not matured.empty else None,
                        "win_rate": float((pd.to_numeric(matured[ret_col], errors="coerce") > 0).mean()) if not matured.empty else None,
                        "avg_mdd_like": float(pd.to_numeric(matured[mdd_col], errors="coerce").mean()) if not matured.empty else None,
                        "median_mdd_like": float(pd.to_numeric(matured[mdd_col], errors="coerce").median()) if not matured.empty else None,
                        "benchmark_return": float(pd.to_numeric(day_df[bench_col], errors="coerce").dropna().iloc[0]) if pd.to_numeric(day_df[bench_col], errors="coerce").notna().any() else None,
                        "avg_excess_return": float(pd.to_numeric(matured[excess_col], errors="coerce").mean()) if not matured.empty else None,
                        "median_excess_return": float(pd.to_numeric(matured[excess_col], errors="coerce").median()) if not matured.empty else None,
                    }
                )
    return pd.DataFrame(rows).sort_values(["asof_date", "top_n", "horizon_days"]).reset_index(drop=True)


def summarize_overall(by_day: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        bucket = by_day.loc[by_day["top_n"] == top_n].copy()
        for horizon in HORIZONS:
            horizon_df = bucket.loc[bucket["horizon_days"] == horizon].copy()
            matured = horizon_df.loc[horizon_df["matured_count"] > 0].copy()
            rows.append(
                {
                    "top_n": int(top_n),
                    "horizon_days": int(horizon),
                    "dates_total": int(len(horizon_df)),
                    "dates_matured": int(len(matured)),
                    "avg_return": float(pd.to_numeric(matured["avg_return"], errors="coerce").mean()) if not matured.empty else None,
                    "median_return": float(pd.to_numeric(matured["median_return"], errors="coerce").mean()) if not matured.empty else None,
                    "win_rate": float(pd.to_numeric(matured["win_rate"], errors="coerce").mean()) if not matured.empty else None,
                    "avg_mdd_like": float(pd.to_numeric(matured["avg_mdd_like"], errors="coerce").mean()) if not matured.empty else None,
                    "benchmark_return": float(pd.to_numeric(matured["benchmark_return"], errors="coerce").mean()) if not matured.empty else None,
                    "avg_excess_return": float(pd.to_numeric(matured["avg_excess_return"], errors="coerce").mean()) if not matured.empty else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_maturity(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        bucket = detail.loc[detail["rank"] <= top_n].copy()
        dates_total = int(bucket["asof_date"].nunique())
        for horizon in HORIZONS:
            matured_dates = (
                bucket.groupby("asof_date")[f"is_matured_{horizon}d"]
                .apply(lambda s: bool(pd.Series(s).fillna(False).all()))
                if not bucket.empty
                else pd.Series(dtype=bool)
            )
            rows.append(
                {
                    "top_n": int(top_n),
                    "horizon_days": int(horizon),
                    "dates_total": dates_total,
                    "dates_fully_matured": int(matured_dates.sum()) if not matured_dates.empty else 0,
                    "latest_snapshot_state": "immature"
                    if bucket.empty
                    else str(
                        bucket.sort_values(["asof_date", "rank"])
                        .groupby("asof_date")["maturity_state"]
                        .last()
                        .iloc[-1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_markdown(detail: pd.DataFrame, by_day: pd.DataFrame, overall: pd.DataFrame, maturity: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_dates = sorted(pd.to_datetime(detail["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist())
    latest_date = snapshot_dates[-1] if snapshot_dates else "NA"

    lines = [
        "# Operational Forward Return Report",
        "",
        f"- generated_at: {generated_at}",
        f"- snapshot_dates: {len(snapshot_dates)}",
        f"- latest_snapshot_date: {latest_date}",
        f"- tracked_horizons: {', '.join(f'{x}d' for x in HORIZONS)}",
        f"- tracked_buckets: {', '.join(f'Top{x}' for x in TOP_BUCKETS)}",
        "",
    ]

    maturity_rows = [
        [
            f"Top{int(row.top_n)}",
            f"{int(row.horizon_days)}d",
            int(row.dates_total),
            int(row.dates_fully_matured),
            row.latest_snapshot_state,
        ]
        for row in maturity.itertuples(index=False)
    ]
    lines.extend(
        [
            "## Maturity Overview",
            "",
            _markdown_table(maturity_rows, ["bucket", "horizon", "dates_total", "dates_fully_matured", "latest_snapshot_state"]),
            "",
        ]
    )

    overall_rows = [
        [
            f"Top{int(row.top_n)}",
            f"{int(row.horizon_days)}d",
            int(row.dates_total),
            int(row.dates_matured),
            _fmt_pct(row.avg_return),
            _fmt_pct(row.median_return),
            _fmt(row.win_rate),
            _fmt_pct(row.avg_mdd_like),
            _fmt_pct(row.benchmark_return),
            _fmt_pct(row.avg_excess_return),
        ]
        for row in overall.itertuples(index=False)
    ]
    lines.extend(
        [
            "## Overall Aggregation",
            "",
            _markdown_table(
                overall_rows,
                ["bucket", "horizon", "dates_total", "dates_matured", "avg_return", "median_return", "win_rate", "avg_mdd_like", "benchmark_return", "avg_excess_return"],
            ),
            "",
        ]
    )

    latest_by_day = by_day.sort_values(["asof_date", "top_n", "horizon_days"], ascending=[False, True, True]).head(12)
    latest_rows = [
        [
            row.asof_date,
            f"Top{int(row.top_n)}",
            f"{int(row.horizon_days)}d",
            row.maturity_state,
            int(row.snapshot_count),
            int(row.matured_count),
            _fmt_pct(row.avg_return),
            _fmt_pct(row.benchmark_return),
            _fmt_pct(row.avg_excess_return),
        ]
        for row in latest_by_day.itertuples(index=False)
    ]
    lines.extend(
        [
            "## Latest Daily Rows",
            "",
            _markdown_table(
                latest_rows,
                ["asof_date", "bucket", "horizon", "maturity_state", "snapshot_count", "matured_count", "avg_return", "benchmark_return", "avg_excess_return"],
            ),
            "",
        ]
    )

    immature_rows = int((by_day["maturity_state"] == "immature").sum()) if not by_day.empty else 0
    if immature_rows > 0:
        lines.extend(
            [
                "## Note",
                "",
                f"- `immature` rows mean the archive snapshot does not yet have enough future trading rows for the requested horizon.",
                f"- immature_by_day_rows: {immature_rows}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    archive = load_archive(args.archive_csv)
    detail = attach_forward_metrics(archive, args.prices_csv, args.benchmark_csv)
    by_day = summarize_by_day(detail)
    overall = summarize_overall(by_day)
    maturity = summarize_maturity(detail)

    args.out_by_day_csv.parent.mkdir(parents=True, exist_ok=True)
    by_day.to_csv(args.out_by_day_csv, index=False, encoding="utf-8-sig")

    report = build_markdown(detail, by_day, overall, maturity)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")

    print(f"detail_rows: {len(detail)}")
    print(f"by_day_rows: {len(by_day)}")
    print(f"report_path: {args.out_md}")
    print(f"by_day_csv_path: {args.out_by_day_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
