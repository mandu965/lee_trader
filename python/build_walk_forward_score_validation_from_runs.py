from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from check_walkforward_outcome_coverage import (  # noqa: E402
    build_run_coverage_summary,
    load_backtest_outcome_rows,
    load_run_metadata,
    load_walkforward_predictions,
)
from db import get_engine  # noqa: E402


OUTPUT_MD = ROOT / "outputs" / "walk_forward_score_validation.md"
OUTPUT_CSV = ROOT / "outputs" / "walk_forward_score_validation.csv"
HORIZONS = [20, 60, 90]
MIN_RUNS_PER_GROUP = 3


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


def load_matured_run_ids() -> pd.DataFrame:
    preds = load_walkforward_predictions()
    run_metadata = load_run_metadata()
    outcome_counts = load_backtest_outcome_rows()
    run_summary, _ = build_run_coverage_summary(
        preds,
        outcome_counts,
        run_metadata,
        min_runs_per_group=MIN_RUNS_PER_GROUP,
    )
    matured = run_summary.loc[
        run_summary["run_status"] == "matured",
        ["run_id", "horizon_days", "model_version", "score_formula_version"],
    ].copy()
    if matured.empty:
        return matured
    matured["run_id"] = matured["run_id"].astype(int)
    matured["horizon_days"] = matured["horizon_days"].astype(int)
    return matured


def load_joined_rows(matured_runs: pd.DataFrame) -> pd.DataFrame:
    if matured_runs.empty:
        return pd.DataFrame()
    eng = get_engine()
    query = text(
        """
        SELECT
            r.run_id,
            r.as_of_date,
            r.code,
            r.horizon_days,
            r.rank,
            r.final_score,
            o.realized_return,
            o.realized_mdd
        FROM research.ranking_history r
        JOIN research.backtest_outcome o
          ON o.run_id = r.run_id
         AND o.as_of_date = r.as_of_date
         AND o.code = r.code
         AND o.horizon_days = r.horizon_days
        JOIN research.dim_model_run d
          ON d.run_id = r.run_id
        WHERE d.run_type = 'walkforward_backtest'
          AND o.realized_return IS NOT NULL
        """
    )
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["as_of_date"])
    if df.empty:
        return df
    df["run_id"] = df["run_id"].astype(int)
    df["horizon_days"] = df["horizon_days"].astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["realized_return"] = pd.to_numeric(df["realized_return"], errors="coerce")
    df["realized_mdd"] = pd.to_numeric(df["realized_mdd"], errors="coerce")
    df = df.merge(matured_runs, on=["run_id", "horizon_days"], how="inner")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.strftime("%Y-%m-%d")
    return df


def cohort_selection_metrics(df: pd.DataFrame, selection_name: str, rank_cap: int | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, run_id, as_of_date), group in df.groupby(["horizon_days", "run_id", "as_of_date"], sort=True):
        universe = group.sort_values(["rank", "final_score"], ascending=[True, False]).copy()
        selected = universe if rank_cap is None else universe.loc[universe["rank"] <= rank_cap].copy()
        if selected.empty:
            continue
        universe_avg = float(universe["realized_return"].mean())
        rows.append(
            {
                "section": "selection_summary",
                "horizon_days": int(horizon),
                "selection": selection_name,
                "run_id": int(run_id),
                "as_of_date": as_of_date,
                "n": int(len(selected)),
                "avg_return": float(selected["realized_return"].mean()),
                "median_return": float(selected["realized_return"].median()),
                "hit_rate": float((selected["realized_return"] > 0).mean()),
                "avg_mdd": float(selected["realized_mdd"].mean()),
                "benchmark_return": universe_avg,
                "excess_return": float(selected["realized_return"].mean() - universe_avg),
                "status": "OK",
                "decile": pd.NA,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def cohort_decile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, run_id, as_of_date), group in df.groupby(["horizon_days", "run_id", "as_of_date"], sort=True):
        ordered = group.sort_values(["rank", "final_score"], ascending=[True, False]).copy()
        if len(ordered) < 10:
            continue
        ordered["rank_order"] = range(1, len(ordered) + 1)
        ordered["decile"] = pd.qcut(ordered["rank_order"], 10, labels=list(range(1, 11)))
        for decile, bucket in ordered.groupby("decile", observed=False):
            if bucket.empty:
                continue
            rows.append(
                {
                    "section": "decile_summary",
                    "horizon_days": int(horizon),
                    "selection": "decile",
                    "run_id": int(run_id),
                    "as_of_date": as_of_date,
                    "n": int(len(bucket)),
                    "avg_return": float(bucket["realized_return"].mean()),
                    "median_return": float(bucket["realized_return"].median()),
                    "hit_rate": float((bucket["realized_return"] > 0).mean()),
                    "avg_mdd": float(bucket["realized_mdd"].mean()),
                    "benchmark_return": float(ordered["realized_return"].mean()),
                    "excess_return": float(bucket["realized_return"].mean() - ordered["realized_return"].mean()),
                    "status": "OK",
                    "decile": int(decile),
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def aggregate_selection_summary(selection_rows: pd.DataFrame) -> pd.DataFrame:
    if selection_rows.empty:
        return pd.DataFrame(
            columns=[
                "horizon_days",
                "selection",
                "run_dates",
                "avg_return",
                "median_return",
                "benchmark_return",
                "excess_return",
                "hit_rate",
                "avg_mdd",
            ]
        )
    return (
        selection_rows.groupby(["horizon_days", "selection"], as_index=False)
        .agg(
            run_dates=("as_of_date", "nunique"),
            avg_return=("avg_return", "mean"),
            median_return=("median_return", "mean"),
            benchmark_return=("benchmark_return", "mean"),
            excess_return=("excess_return", "mean"),
            hit_rate=("hit_rate", "mean"),
            avg_mdd=("avg_mdd", "mean"),
        )
        .sort_values(["horizon_days", "selection"])
        .reset_index(drop=True)
    )


def aggregate_decile_summary(decile_rows: pd.DataFrame) -> pd.DataFrame:
    if decile_rows.empty:
        return pd.DataFrame(
            columns=[
                "horizon_days",
                "decile",
                "run_dates",
                "avg_return",
                "median_return",
                "excess_return",
                "hit_rate",
                "avg_mdd",
            ]
        )
    return (
        decile_rows.groupby(["horizon_days", "decile"], as_index=False)
        .agg(
            run_dates=("as_of_date", "nunique"),
            avg_return=("avg_return", "mean"),
            median_return=("median_return", "mean"),
            excess_return=("excess_return", "mean"),
            hit_rate=("hit_rate", "mean"),
            avg_mdd=("avg_mdd", "mean"),
        )
        .sort_values(["horizon_days", "decile"])
        .reset_index(drop=True)
    )


def build_interpretation(selection_summary: pd.DataFrame, decile_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for horizon in HORIZONS:
        summary = selection_summary.loc[selection_summary["horizon_days"] == horizon].copy()
        if summary.empty:
            lines.append(f"- {horizon}d: matured walk-forward run이 부족해 unavailable입니다.")
            continue

        top20 = summary.loc[summary["selection"] == "top20"]
        top50 = summary.loc[summary["selection"] == "top50"]
        universe = summary.loc[summary["selection"] == "universe"]
        if top20.empty or top50.empty or universe.empty:
            lines.append(f"- {horizon}d: 비교 cohort가 충분하지 않아 해석을 보류합니다.")
            continue

        top20_row = top20.iloc[0]
        top50_row = top50.iloc[0]
        universe_row = universe.iloc[0]

        dec = decile_summary.loc[decile_summary["horizon_days"] == horizon].copy()
        decile_signal = False
        if not dec.empty:
            dec1 = dec.loc[dec["decile"] == 1]
            dec10 = dec.loc[dec["decile"] == 10]
            decile_signal = (
                not dec1.empty
                and not dec10.empty
                and float(dec1["avg_return"].iloc[0]) > float(dec10["avg_return"].iloc[0])
            )

        if float(top20_row["avg_return"]) > float(universe_row["avg_return"]) and decile_signal:
            lines.append(
                f"- {horizon}d: top20 평균 수익률 {_fmt_pct(top20_row['avg_return'])}, 초과수익 {_fmt_pct(top20_row['excess_return'])}로 점수 상위 그룹이 universe보다 우세합니다."
            )
        else:
            lines.append(
                f"- {horizon}d: top20 우위가 약하거나 decile monotonicity가 불안정합니다. top20={_fmt_pct(top20_row['avg_return'])}, universe={_fmt_pct(universe_row['avg_return'])}."
            )

        if float(top20_row["avg_return"]) > float(top50_row["avg_return"]) > float(universe_row["avg_return"]):
            lines.append(f"- {horizon}d: top20 > top50 > universe 순서가 유지돼 점수 선별력이 비교적 일관됩니다.")
        else:
            lines.append(f"- {horizon}d: top20 / top50 / universe의 순위 구조가 단조롭지 않아 과대해석을 피해야 합니다.")

    lines.append("- 현재 데이터는 60d matured walk-forward run 중심이라, 20d와 90d는 unavailable일 수 있습니다.")
    lines.append("- run 수가 아직 제한적이어서 특정 기간 성과를 일반 성능으로 확대 해석하면 안 됩니다.")
    return lines


def build_markdown(selection_summary: pd.DataFrame, decile_summary: pd.DataFrame, matured_runs: pd.DataFrame) -> str:
    selection_rows: list[list[object]] = []
    for horizon in HORIZONS:
        subset = selection_summary.loc[selection_summary["horizon_days"] == horizon]
        if subset.empty:
            selection_rows.append([horizon, "unavailable", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
            continue
        for _, row in subset.iterrows():
            selection_rows.append(
                [
                    int(row["horizon_days"]),
                    row["selection"],
                    int(row["run_dates"]),
                    _fmt_pct(row["avg_return"]),
                    _fmt_pct(row["benchmark_return"]),
                    _fmt_pct(row["excess_return"]),
                    _fmt(row["hit_rate"]),
                    _fmt_pct(row["avg_mdd"]),
                    _fmt_pct(row["median_return"]),
                ]
            )

    lines = [
        "# Walk-Forward Score Validation",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- source: matured walkforward_backtest runs from research.ranking_history + research.backtest_outcome",
        f"- matured_run_count: {len(matured_runs)}",
        "- benchmark_definition: same run-date scored universe average realized return",
        "- recomputed_from_current_code: true",
        "",
        "## Summary",
        _markdown_table(
            selection_rows,
            [
                "horizon_days",
                "selection",
                "run_dates",
                "avg_return",
                "benchmark_return",
                "excess_return",
                "hit_rate",
                "avg_mdd",
                "median_return",
            ],
        ),
        "",
        "## Interpretation",
        *build_interpretation(selection_summary, decile_summary),
        "",
    ]

    for horizon in HORIZONS:
        subset = selection_summary.loc[selection_summary["horizon_days"] == horizon]
        lines.append(f"## Horizon {horizon}d")
        if subset.empty:
            lines.append("- status: unavailable")
            lines.append("- reason: matured walk-forward run 부족")
            lines.append("")
            continue

        lines.append("### Top20 / Top50 / Universe")
        table_rows = [
            [
                row["selection"],
                int(row["run_dates"]),
                _fmt_pct(row["avg_return"]),
                _fmt_pct(row["benchmark_return"]),
                _fmt_pct(row["excess_return"]),
                _fmt(row["hit_rate"]),
                _fmt_pct(row["avg_mdd"]),
                _fmt_pct(row["median_return"]),
            ]
            for _, row in subset.iterrows()
        ]
        lines.append(
            _markdown_table(
                table_rows,
                [
                    "selection",
                    "run_dates",
                    "avg_return",
                    "benchmark_return",
                    "excess_return",
                    "hit_rate",
                    "avg_mdd",
                    "median_return",
                ],
            )
        )
        lines.append("")

        dec = decile_summary.loc[decile_summary["horizon_days"] == horizon]
        lines.append("### Decile Performance")
        if dec.empty:
            lines.append("- unavailable")
            lines.append("")
            continue

        dec_rows = [
            [
                int(row["decile"]),
                int(row["run_dates"]),
                _fmt_pct(row["avg_return"]),
                _fmt_pct(row["excess_return"]),
                _fmt(row["hit_rate"]),
                _fmt_pct(row["avg_mdd"]),
                _fmt_pct(row["median_return"]),
            ]
            for _, row in dec.iterrows()
        ]
        lines.append(
            _markdown_table(
                dec_rows,
                ["decile", "run_dates", "avg_return", "excess_return", "hit_rate", "avg_mdd", "median_return"],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Cautions",
            "- benchmark는 동일 run-date scored universe 평균이라 외부 지수 benchmark와는 다릅니다.",
            "- matured run이 적은 horizon은 unavailable로 두고, 수치를 일반화하지 않습니다.",
            "- decile 성과가 단조롭지 않으면 final_score를 선형적인 미래 수익 서열로 해석하면 안 됩니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_csv(selection_summary: pd.DataFrame, decile_summary: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not selection_summary.empty:
        selection_csv = selection_summary.copy()
        selection_csv["section"] = "selection_summary"
        selection_csv["decile"] = pd.NA
        selection_csv["status"] = "OK"
        selection_csv["note"] = ""
        frames.append(
            selection_csv[
                [
                    "section",
                    "horizon_days",
                    "selection",
                    "decile",
                    "run_dates",
                    "avg_return",
                    "benchmark_return",
                    "excess_return",
                    "hit_rate",
                    "avg_mdd",
                    "median_return",
                    "status",
                    "note",
                ]
            ]
        )

    if not decile_summary.empty:
        decile_csv = decile_summary.copy()
        decile_csv["section"] = "decile_summary"
        decile_csv["selection"] = "decile"
        decile_csv["benchmark_return"] = pd.NA
        decile_csv["status"] = "OK"
        decile_csv["note"] = ""
        frames.append(
            decile_csv[
                [
                    "section",
                    "horizon_days",
                    "selection",
                    "decile",
                    "run_dates",
                    "avg_return",
                    "benchmark_return",
                    "excess_return",
                    "hit_rate",
                    "avg_mdd",
                    "median_return",
                    "status",
                    "note",
                ]
            ]
        )

    available_horizons = set(selection_summary["horizon_days"].astype(int).tolist()) if not selection_summary.empty else set()
    unavailable_rows = []
    for horizon in HORIZONS:
        if horizon in available_horizons:
            continue
        unavailable_rows.append(
            {
                "section": "selection_summary",
                "horizon_days": horizon,
                "selection": "unavailable",
                "decile": pd.NA,
                "run_dates": 0,
                "avg_return": pd.NA,
                "benchmark_return": pd.NA,
                "excess_return": pd.NA,
                "hit_rate": pd.NA,
                "avg_mdd": pd.NA,
                "median_return": pd.NA,
                "status": "unavailable",
                "note": "matured walk-forward run 부족",
            }
        )
    if unavailable_rows:
        frames.append(pd.DataFrame(unavailable_rows))

    if not frames:
        return pd.DataFrame(
            columns=[
                "section",
                "horizon_days",
                "selection",
                "decile",
                "run_dates",
                "avg_return",
                "benchmark_return",
                "excess_return",
                "hit_rate",
                "avg_mdd",
                "median_return",
                "status",
                "note",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    matured_runs = load_matured_run_ids()
    joined = load_joined_rows(matured_runs)
    selection_rows = pd.concat(
        [
            cohort_selection_metrics(joined, "top20", 20),
            cohort_selection_metrics(joined, "top50", 50),
            cohort_selection_metrics(joined, "universe", None),
        ],
        ignore_index=True,
    )
    decile_rows = cohort_decile_metrics(joined)

    selection_summary = aggregate_selection_summary(selection_rows)
    decile_summary = aggregate_decile_summary(decile_rows)

    csv_df = build_csv(selection_summary, decile_summary)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(selection_summary, decile_summary, matured_runs), encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_MD}")
    print(f"[ok] wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
