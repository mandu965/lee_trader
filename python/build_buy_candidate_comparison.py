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

RANKING_CSV = DATA_DIR / "ranking_final.csv"
BUY_TOP5_CSV = DATA_DIR / "buy_candidates_top5.csv"
BUY_TOP8_CSV = DATA_DIR / "buy_candidates_top8.csv"
BUY_TOP10_CSV = DATA_DIR / "buy_candidates_top10.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"

OUTPUT_CSV = OUTPUT_DIR / "buy_candidate_comparison.csv"
OUTPUT_MD = OUTPUT_DIR / "buy_candidate_comparison_report.md"

DATE_CANDIDATE_COLUMNS = ["as_of_date", "date", "trade_date", "snapshot_date"]
HORIZONS = [5, 20, 60, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw ranking top20 against operational buy candidate lists.")
    parser.add_argument("--ranking-csv", type=Path, default=RANKING_CSV)
    parser.add_argument("--buy-top5-csv", type=Path, default=BUY_TOP5_CSV)
    parser.add_argument("--buy-top8-csv", type=Path, default=BUY_TOP8_CSV)
    parser.add_argument("--buy-top10-csv", type=Path, default=BUY_TOP10_CSV)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--soft-surge-ret5d", type=float, default=0.12)
    parser.add_argument("--soft-surge-ret10d", type=float, default=0.20)
    parser.add_argument("--soft-surge-rsi", type=float, default=70.0)
    parser.add_argument("--hard-surge-ret5d", type=float, default=0.20)
    parser.add_argument("--hard-surge-ret10d", type=float, default=0.35)
    parser.add_argument("--hard-surge-rsi", type=float, default=80.0)
    return parser.parse_args()


def _fmt(value: object, digits: int = 2) -> str:
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
    headers = columns
    widths = [len(header) for header in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(val.ljust(widths[idx]) for idx, val in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def resolve_latest_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    for column in DATE_CANDIDATE_COLUMNS:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            latest = parsed.max().normalize()
            return df.loc[parsed.dt.normalize().eq(latest)].copy(), latest.strftime("%Y-%m-%d"), column
    raise ValueError("could not resolve latest date from ranking csv")


def normalize_ranking(df: pd.DataFrame, *, asof_date: str, args: argparse.Namespace) -> pd.DataFrame:
    work = df.copy()
    work["asof_date"] = asof_date
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["dominant_theme"] = (
        work.get("dominant_theme", "(none)")
        .fillna("(none)")
        .astype(str)
        .replace({"": "(none)", "nan": "(none)"})
    )
    for column in ["final_score", "confidence_score", "ret_score", "tech_score", "ret_5d", "ret_10d", "rsi_14"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    work["rank_source"] = pd.to_numeric(work.get("rank_final"), errors="coerce")
    if work["rank_source"].isna().all():
        work["rank_source"] = (
            pd.to_numeric(work["final_score"], errors="coerce")
            .rank(method="first", ascending=False)
            .astype(float)
        )
    work["rank_source"] = work["rank_source"].round().astype("Int64")
    work["overheat_soft_flag"] = (
        work["ret_5d"].ge(args.soft_surge_ret5d).fillna(False)
        | work["ret_10d"].ge(args.soft_surge_ret10d).fillna(False)
        | work["rsi_14"].ge(args.soft_surge_rsi).fillna(False)
    )
    work["overheat_hard_flag"] = (
        work["ret_5d"].ge(args.hard_surge_ret5d).fillna(False)
        | work["ret_10d"].ge(args.hard_surge_ret10d).fillna(False)
        | work["rsi_14"].ge(args.hard_surge_rsi).fillna(False)
    )
    return work.sort_values(["rank_source", "final_score", "code"], ascending=[True, False, True]).reset_index(drop=True)


def load_buy_candidates(path: Path, cohort_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"buy candidate csv not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["cohort_name"] = cohort_name
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def build_cohorts(ranking_latest: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], str]:
    latest_norm = normalize_ranking(ranking_latest, asof_date=ranking_latest["asof_date"].iloc[0], args=args)
    raw_top20 = latest_norm.head(20).copy()
    raw_top20["cohort_name"] = "raw_top20"

    top5 = load_buy_candidates(args.buy_top5_csv if args.buy_top5_csv.is_absolute() else ROOT / args.buy_top5_csv, "buy_top5")
    top8 = load_buy_candidates(args.buy_top8_csv if args.buy_top8_csv.is_absolute() else ROOT / args.buy_top8_csv, "buy_top8")
    top10 = load_buy_candidates(args.buy_top10_csv if args.buy_top10_csv.is_absolute() else ROOT / args.buy_top10_csv, "buy_top10")

    asof_date = str(raw_top20["asof_date"].iloc[0])
    cohorts: dict[str, pd.DataFrame] = {"raw_top20": raw_top20}
    for cohort_name, buy_df in [("buy_top5", top5), ("buy_top8", top8), ("buy_top10", top10)]:
        cohort = latest_norm.merge(
            buy_df[["code"]].drop_duplicates(),
            on="code",
            how="inner",
        ).copy()
        cohort["cohort_name"] = cohort_name
        cohorts[cohort_name] = cohort
    return cohorts, asof_date


def concentration_stats(series: pd.Series) -> dict[str, object]:
    cleaned = series.fillna("(missing)").astype(str).replace({"": "(missing)", "nan": "(missing)"})
    if cleaned.empty:
        return {"unique_count": 0, "top_label": "NA", "top_share": None, "hhi": None}
    counts = cleaned.value_counts(dropna=False)
    shares = counts / counts.sum()
    return {
        "unique_count": int(counts.size),
        "top_label": str(counts.index[0]),
        "top_share": float(shares.iloc[0]),
        "hhi": float((shares**2).sum()),
    }


def attach_forward_metrics(frame: pd.DataFrame, prices_csv: Path) -> pd.DataFrame:
    work = frame.copy()
    work["asof_date"] = pd.to_datetime(work["asof_date"], errors="coerce").dt.normalize()
    prices = load_price_history(prices_csv=prices_csv)
    price_reference = build_price_reference(prices[["code", "date"]].copy())

    for horizon in HORIZONS:
        maturity = evaluate_prediction_maturity_rows(
            work[["code", "asof_date"]],
            price_reference=price_reference,
            horizon_days=horizon,
            as_of_col="asof_date",
            code_col="code",
        ).rename(
            columns={
                "is_matured": f"is_matured_{horizon}d",
                "maturity_status": f"maturity_status_{horizon}d",
                "available_future_trading_days": f"available_future_trading_days_{horizon}d",
            }
        )
        outcome = attach_forward_outcomes(prices, horizon_days=horizon).rename(
            columns={
                "date": "asof_date",
                "realized_return": f"forward_return_{horizon}d",
                "realized_mdd": f"forward_mdd_like_{horizon}d",
            }
        )
        work = work.merge(
            maturity[
                [
                    "code",
                    "asof_date",
                    f"is_matured_{horizon}d",
                    f"maturity_status_{horizon}d",
                    f"available_future_trading_days_{horizon}d",
                ]
            ],
            on=["code", "asof_date"],
            how="left",
        )
        work = work.merge(
            outcome[["code", "asof_date", f"forward_return_{horizon}d", f"forward_mdd_like_{horizon}d"]],
            on=["code", "asof_date"],
            how="left",
        )
        mature_mask = work.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)
        work.loc[~mature_mask, [f"forward_return_{horizon}d", f"forward_mdd_like_{horizon}d"]] = pd.NA
    return work


def summarize_cohort(frame: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {
        "cohort_name": str(frame["cohort_name"].iloc[0]),
        "asof_date": pd.to_datetime(frame["asof_date"].iloc[0], errors="coerce").strftime("%Y-%m-%d"),
        "row_count": int(len(frame)),
        "avg_final_score": float(pd.to_numeric(frame["final_score"], errors="coerce").mean()) if not frame.empty else None,
        "avg_confidence_score": float(pd.to_numeric(frame["confidence_score"], errors="coerce").mean()) if not frame.empty else None,
        "avg_ret_score": float(pd.to_numeric(frame["ret_score"], errors="coerce").mean()) if not frame.empty else None,
        "avg_tech_score": float(pd.to_numeric(frame["tech_score"], errors="coerce").mean()) if not frame.empty else None,
        "overheat_soft_ratio": float(frame["overheat_soft_flag"].fillna(False).mean()) if not frame.empty else None,
        "overheat_hard_ratio": float(frame["overheat_hard_flag"].fillna(False).mean()) if not frame.empty else None,
    }

    sector_stats = concentration_stats(frame["sector"])
    theme_stats = concentration_stats(frame["dominant_theme"])
    out["sector_unique_count"] = sector_stats["unique_count"]
    out["sector_top_label"] = sector_stats["top_label"]
    out["sector_top_share"] = sector_stats["top_share"]
    out["sector_hhi"] = sector_stats["hhi"]
    out["theme_unique_count"] = theme_stats["unique_count"]
    out["theme_top_label"] = theme_stats["top_label"]
    out["theme_top_share"] = theme_stats["top_share"]
    out["theme_hhi"] = theme_stats["hhi"]

    for horizon in HORIZONS:
        matured = frame.loc[frame.get(f"is_matured_{horizon}d", False).fillna(False).astype(bool)].copy()
        ret_col = f"forward_return_{horizon}d"
        mdd_col = f"forward_mdd_like_{horizon}d"
        out[f"matured_count_{horizon}d"] = int(len(matured))
        out[f"avg_forward_return_{horizon}d"] = float(pd.to_numeric(matured[ret_col], errors="coerce").mean()) if not matured.empty else None
        out[f"median_forward_return_{horizon}d"] = float(pd.to_numeric(matured[ret_col], errors="coerce").median()) if not matured.empty else None
        out[f"hit_rate_{horizon}d"] = float((pd.to_numeric(matured[ret_col], errors="coerce") > 0).mean()) if not matured.empty else None
        out[f"avg_forward_mdd_like_{horizon}d"] = float(pd.to_numeric(matured[mdd_col], errors="coerce").mean()) if not matured.empty else None
        out[f"maturity_status_{horizon}d"] = "immature" if matured.empty else f"matured_{horizon}d"
    return out


def build_comparison_csv(cohorts: dict[str, pd.DataFrame], prices_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_name, cohort in cohorts.items():
        enriched = attach_forward_metrics(cohort, prices_csv=prices_csv)
        rows.append(summarize_cohort(enriched))

    result = pd.DataFrame(rows).sort_values("cohort_name").reset_index(drop=True)
    raw_row = result.loc[result["cohort_name"] == "raw_top20"]
    if not raw_row.empty:
        for metric in [
            "avg_final_score",
            "avg_confidence_score",
            "avg_ret_score",
            "avg_tech_score",
            "sector_hhi",
            "theme_hhi",
            "overheat_soft_ratio",
            "overheat_hard_ratio",
        ]:
            base = pd.to_numeric(raw_row.iloc[0][metric], errors="coerce")
            result[f"delta_vs_raw_top20_{metric}"] = pd.to_numeric(result[metric], errors="coerce") - base
        for horizon in HORIZONS:
            metric = f"avg_forward_return_{horizon}d"
            hit_metric = f"hit_rate_{horizon}d"
            base_ret = pd.to_numeric(raw_row.iloc[0][metric], errors="coerce")
            base_hit = pd.to_numeric(raw_row.iloc[0][hit_metric], errors="coerce")
            result[f"delta_vs_raw_top20_{metric}"] = pd.to_numeric(result[metric], errors="coerce") - base_ret
            result[f"delta_vs_raw_top20_{hit_metric}"] = pd.to_numeric(result[hit_metric], errors="coerce") - base_hit
    return result


def build_report(comparison: pd.DataFrame, asof_date: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    static_table = comparison[
        [
            "cohort_name",
            "row_count",
            "avg_final_score",
            "avg_confidence_score",
            "avg_ret_score",
            "avg_tech_score",
            "sector_unique_count",
            "sector_top_label",
            "sector_top_share",
            "sector_hhi",
            "theme_unique_count",
            "theme_top_label",
            "theme_top_share",
            "theme_hhi",
            "overheat_soft_ratio",
            "overheat_hard_ratio",
        ]
    ].copy()
    for col in ["avg_final_score", "avg_confidence_score", "avg_ret_score", "avg_tech_score", "sector_hhi", "theme_hhi"]:
        static_table[col] = static_table[col].map(_fmt)
    for col in ["sector_top_share", "theme_top_share", "overheat_soft_ratio", "overheat_hard_ratio"]:
        static_table[col] = static_table[col].map(_fmt_pct)

    forward_rows: list[dict[str, object]] = []
    for _, row in comparison.iterrows():
        for horizon in HORIZONS:
            forward_rows.append(
                {
                    "cohort_name": row["cohort_name"],
                    "horizon": f"{horizon}d",
                    "matured_count": int(row[f"matured_count_{horizon}d"]),
                    "maturity_status": row[f"maturity_status_{horizon}d"],
                    "avg_forward_return": _fmt_pct(row[f"avg_forward_return_{horizon}d"]),
                    "median_forward_return": _fmt_pct(row[f"median_forward_return_{horizon}d"]),
                    "hit_rate": _fmt_pct(row[f"hit_rate_{horizon}d"]),
                    "avg_forward_mdd_like": _fmt_pct(row[f"avg_forward_mdd_like_{horizon}d"]),
                    "delta_vs_raw_top20_avg_forward_return": _fmt_pct(row.get(f"delta_vs_raw_top20_avg_forward_return_{horizon}d")),
                    "delta_vs_raw_top20_hit_rate": _fmt_pct(row.get(f"delta_vs_raw_top20_hit_rate_{horizon}d")),
                }
            )
    forward_table = pd.DataFrame(forward_rows)

    lines = [
        "# Buy Candidate Comparison Report",
        "",
        f"- generated_at: {generated_at}",
        f"- asof_date: {asof_date}",
        "- compared cohorts: raw_top20, buy_top5, buy_top8, buy_top10",
        "",
        "## Static Quality Comparison",
        _markdown_table(
            static_table,
            [
                "cohort_name",
                "row_count",
                "avg_final_score",
                "avg_confidence_score",
                "avg_ret_score",
                "avg_tech_score",
                "sector_unique_count",
                "sector_top_label",
                "sector_top_share",
                "sector_hhi",
                "theme_unique_count",
                "theme_top_label",
                "theme_top_share",
                "theme_hhi",
                "overheat_soft_ratio",
                "overheat_hard_ratio",
            ],
        ),
        "",
        "## Forward Return Comparison",
        _markdown_table(
            forward_table,
            [
                "cohort_name",
                "horizon",
                "matured_count",
                "maturity_status",
                "avg_forward_return",
                "median_forward_return",
                "hit_rate",
                "avg_forward_mdd_like",
                "delta_vs_raw_top20_avg_forward_return",
                "delta_vs_raw_top20_hit_rate",
            ],
        ),
    ]

    immature_rows = forward_table.loc[forward_table["matured_count"].eq(0)]
    if len(immature_rows) == len(forward_table):
        lines.extend(
            [
                "",
                "## Note",
                "- All forward-return comparison rows are currently immature, so forward metrics are NA. This is a data-maturity limitation, not a script failure.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ranking_path = args.ranking_csv if args.ranking_csv.is_absolute() else ROOT / args.ranking_csv
    raw = pd.read_csv(ranking_path, dtype={"code": str}, low_memory=False)
    ranking_latest, asof_date, _ = resolve_latest_slice(raw)
    ranking_latest["asof_date"] = asof_date

    cohorts, asof_date = build_cohorts(ranking_latest, args)
    comparison = build_comparison_csv(cohorts, prices_csv=args.prices_csv if args.prices_csv.is_absolute() else ROOT / args.prices_csv)

    out_csv = args.out_csv if args.out_csv.is_absolute() else ROOT / args.out_csv
    out_md = args.out_md if args.out_md.is_absolute() else ROOT / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_md.write_text(build_report(comparison, asof_date=asof_date), encoding="utf-8")

    print(f"asof_date: {asof_date}")
    print(f"comparison_rows: {len(comparison)}")
    print(f"out_csv: {out_csv}")
    print(f"out_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
