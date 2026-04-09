from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from outcome_maturity import attach_forward_outcomes, load_price_history  # noqa: E402
from ranking_builder import (  # noqa: E402
    _attach_component_integrity_flags,
    _attach_market_columns,
    _compute_risk_penalty,
    apply_default_ranking_scores,
    compute_component_scores,
)


DB_PATH = ROOT / "data" / "lee_trader.db"
OUTPUT_MD = ROOT / "outputs" / "walk_forward_score_validation.md"
MIN_RUN_ROWS = 50
TOP_N = 20
BOTTOM_N = 20
HORIZONS = [20, 60, 90]


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


def load_historical_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    predictions = pd.read_sql_query(
        "SELECT date, code, pred_return_60d, pred_return_90d, pred_mdd_60d, pred_mdd_90d, prob_top20_60d, prob_top20_90d, score FROM predictions",
        conn,
    )
    features = pd.read_sql_query("SELECT * FROM features WHERE date IN (SELECT DISTINCT date FROM predictions)", conn)
    stocks = pd.read_sql_query("SELECT code, name, market, sector FROM stocks", conn)
    market_status = pd.read_sql_query("SELECT date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up FROM market_status", conn)
    conn.close()

    for frame in [predictions, features, stocks]:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for frame in [predictions, features, market_status]:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return predictions, features, stocks, market_status


def build_rank_history() -> tuple[pd.DataFrame, list[str]]:
    predictions, features, stocks, market_status = load_historical_inputs()
    base = predictions.merge(features, on=["date", "code"], how="left", suffixes=("", "_feat"))
    base = base.merge(stocks, on="code", how="left", suffixes=("", "_stk"))
    if "score" in base.columns and "score_score" not in base.columns:
        base["score_score"] = pd.to_numeric(base["score"], errors="coerce")

    market_history = market_status.copy()
    market_history["date"] = pd.to_datetime(market_history["date"], errors="coerce")

    ranked_frames: list[pd.DataFrame] = []
    excluded_dates: list[str] = []
    for date_value, group in base.groupby("date", sort=True):
        if len(group) < MIN_RUN_ROWS:
            excluded_dates.append(f"{date_value}: rows={len(group)} < {MIN_RUN_ROWS}")
            continue
        market_row = market_status.loc[market_status["date"] == date_value]
        if market_row.empty:
            excluded_dates.append(f"{date_value}: market_status missing")
            continue
        market_info = market_row.iloc[-1].to_dict()
        hist = market_history.loc[market_history["date"] <= pd.to_datetime(date_value), ["date", "kospi_close"]].copy()
        work = group.copy()
        work = compute_component_scores(work)
        work = _attach_component_integrity_flags(work)
        work = _compute_risk_penalty(work)
        work = _attach_market_columns(work, bool(market_info.get("market_up")), market_info, hist)
        # Match the current production operating final_score axes.
        required_cols = ["ret_score", "prob_score", "tech_score", "qual_score", "risk_penalty"]
        valid_mask = work[required_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        dropped = int((~valid_mask).sum())
        if dropped:
            excluded_dates.append(f"{date_value}: dropped_rows_with_missing_components={dropped}")
        work = work.loc[valid_mask].copy()
        if len(work) < MIN_RUN_ROWS:
            excluded_dates.append(f"{date_value}: valid rows after component filter={len(work)} < {MIN_RUN_ROWS}")
            continue
        work = apply_default_ranking_scores(work)
        ranked_frames.append(work)

    ranked = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    return ranked, excluded_dates


def attach_realized_outcomes(ranked: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    price_history = load_price_history()
    outcomes = attach_forward_outcomes(price_history, horizon_days=horizon_days)
    outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = ranked.merge(
        outcomes.rename(columns={"realized_return": f"realized_return_{horizon_days}d", "realized_mdd": f"realized_mdd_{horizon_days}d"}),
        on=["code", "date"],
        how="left",
    )
    return merged


def confidence_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [values >= 80.0, values >= 60.0],
            ["high", "mid"],
            default="low",
        ),
        index=series.index,
    )


def top_vs_bottom_summary(df: pd.DataFrame, return_col: str, mdd_col: str) -> dict[str, object]:
    top = df.sort_values("final_score", ascending=False).head(TOP_N)
    bottom = df.sort_values("final_score", ascending=True).head(BOTTOM_N)
    return {
        "top_avg_return": float(pd.to_numeric(top[return_col], errors="coerce").mean()),
        "bottom_avg_return": float(pd.to_numeric(bottom[return_col], errors="coerce").mean()),
        "spread": float(pd.to_numeric(top[return_col], errors="coerce").mean() - pd.to_numeric(bottom[return_col], errors="coerce").mean()),
        "top_avg_mdd": float(pd.to_numeric(top[mdd_col], errors="coerce").mean()),
        "bottom_avg_mdd": float(pd.to_numeric(bottom[mdd_col], errors="coerce").mean()),
    }


def selection_snapshot(
    df: pd.DataFrame,
    selection: str,
    top_n: int,
    return_col: str,
    mdd_col: str,
    benchmark_return: float,
    run_dates: int,
) -> dict[str, object]:
    if selection == "universe":
        sample = df.copy()
    else:
        sample = df.sort_values("final_score", ascending=False).head(top_n).copy()
    returns = pd.to_numeric(sample[return_col], errors="coerce")
    mdds = pd.to_numeric(sample[mdd_col], errors="coerce")
    avg_return = float(returns.mean()) if returns.notna().any() else float("nan")
    return {
        "selection": selection,
        "run_dates": int(run_dates),
        "avg_return": avg_return,
        "benchmark_return": float(benchmark_return),
        "excess_return": float(avg_return - benchmark_return) if pd.notna(avg_return) and pd.notna(benchmark_return) else float("nan"),
        "hit_rate": float((returns > 0).mean()) if returns.notna().any() else float("nan"),
        "avg_mdd": float(mdds.mean()) if mdds.notna().any() else float("nan"),
        "median_return": float(returns.median()) if returns.notna().any() else float("nan"),
    }


def build_horizon_report(ranked: pd.DataFrame, horizon_days: int) -> dict[str, object]:
    return_col = f"realized_return_{horizon_days}d"
    mdd_col = f"realized_mdd_{horizon_days}d"
    if return_col not in ranked.columns:
        work = attach_realized_outcomes(ranked, horizon_days)
    else:
        work = ranked.copy()

    matured = work.loc[pd.to_numeric(work[return_col], errors="coerce").notna()].copy()
    if matured.empty:
        return {
            "status": "INSUFFICIENT",
            "horizon_days": horizon_days,
            "reason": "no matured rows available",
        }

    eligible_dates = matured.groupby("date")["code"].size()
    eligible_dates = eligible_dates[eligible_dates >= MIN_RUN_ROWS]
    matured = matured.loc[matured["date"].isin(eligible_dates.index)].copy()
    if matured.empty:
        return {
            "status": "INSUFFICIENT",
            "horizon_days": horizon_days,
            "reason": f"no matured ranking dates with >= {MIN_RUN_ROWS} rows",
        }

    matured["benchmark_return"] = matured.groupby("date")[return_col].transform("mean")
    matured["hit"] = pd.to_numeric(matured[return_col], errors="coerce") > 0
    matured["confidence_bucket"] = confidence_bucket(matured["confidence_score"])

    latest_matured_date = str(matured["date"].max())
    latest_run = matured.loc[matured["date"] == latest_matured_date].copy()
    top20 = latest_run.sort_values("final_score", ascending=False).head(TOP_N)
    benchmark_return = float(pd.to_numeric(top20["benchmark_return"], errors="coerce").iloc[0]) if not top20.empty else float("nan")
    selection_rows = [
        selection_snapshot(latest_run, "top20", TOP_N, return_col, mdd_col, benchmark_return, int(len(eligible_dates))),
        selection_snapshot(latest_run, "top50", 50, return_col, mdd_col, benchmark_return, int(len(eligible_dates))),
        selection_snapshot(latest_run, "universe", len(latest_run), return_col, mdd_col, benchmark_return, int(len(eligible_dates))),
    ]

    decile_frame = latest_run.sort_values("final_score", ascending=False).copy()
    if len(decile_frame) >= 10:
        decile_frame["rank_decile"] = pd.qcut(decile_frame["final_score"].rank(method="first", ascending=False), 10, labels=list(range(1, 11)))
        decile_perf = (
            decile_frame.groupby("rank_decile", observed=False)
            .agg(
                n=("code", "size"),
                avg_return=(return_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
                hit_rate=("hit", "mean"),
                avg_mdd=(mdd_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
            .reset_index()
        )
        decile_rows = [
            [int(row["rank_decile"]), int(row["n"]), _fmt_pct(row["avg_return"]), _fmt(row["hit_rate"]), _fmt_pct(row["avg_mdd"])]
            for _, row in decile_perf.iterrows()
        ]
    else:
        decile_rows = [["NA", len(decile_frame), "NA", "NA", "NA"]]

    bucket_perf = (
        latest_run.groupby("confidence_bucket", observed=False)
        .agg(
            n=("code", "size"),
            avg_return=(return_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            hit_rate=("hit", "mean"),
            avg_mdd=(mdd_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )
    bucket_rows = [
        [row["confidence_bucket"], int(row["n"]), _fmt_pct(row["avg_return"]), _fmt(row["hit_rate"]), _fmt_pct(row["avg_mdd"])]
        for _, row in bucket_perf.iterrows()
    ]

    spread = top_vs_bottom_summary(latest_run, return_col, mdd_col)

    return {
        "status": "OK",
        "horizon_days": horizon_days,
        "latest_matured_date": latest_matured_date,
        "eligible_dates": list(eligible_dates.index.astype(str)),
        "eligible_date_count": int(len(eligible_dates)),
        "rows": int(len(matured)),
        "latest_run_rows": int(len(latest_run)),
        "top20_avg_return": float(pd.to_numeric(top20[return_col], errors="coerce").mean()),
        "benchmark_return": benchmark_return,
        "excess_return": float(pd.to_numeric(top20[return_col], errors="coerce").mean() - benchmark_return),
        "hit_rate": float(top20["hit"].mean()),
        "avg_mdd": float(pd.to_numeric(top20[mdd_col], errors="coerce").mean()),
        "selection_rows": selection_rows,
        "spread": spread,
        "decile_rows": decile_rows,
        "bucket_rows": bucket_rows,
    }


def build_markdown(horizon_reports: list[dict[str, object]], excluded_dates: list[str]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_summary_rows: list[list[object]] = []
    interpretation_lines: list[str] = []

    for report in horizon_reports:
        if report["status"] != "OK":
            top_summary_rows.append([report["horizon_days"], "top20", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
            interpretation_lines.append(f"- {report['horizon_days']}d: {report.get('reason', 'insufficient data')}.")
            continue

        for item in report.get("selection_rows", []):
            top_summary_rows.append(
                [
                    report["horizon_days"],
                    item["selection"],
                    item["run_dates"],
                    _fmt_pct(item["avg_return"]),
                    _fmt_pct(item["benchmark_return"]),
                    _fmt_pct(item["excess_return"]),
                    _fmt(item["hit_rate"]),
                    _fmt_pct(item["avg_mdd"]),
                    _fmt_pct(item["median_return"]),
                ]
            )
        if report["excess_return"] > 0 and report["hit_rate"] >= 0.5:
            interpretation_lines.append(
                f"- {report['horizon_days']}d: top20이 scored-universe benchmark 대비 초과수익 {_fmt_pct(report['excess_return'])}를 기록했고 hit rate도 {_fmt(report['hit_rate'])}로 양호합니다."
            )
        else:
            interpretation_lines.append(
                f"- {report['horizon_days']}d: top20 우위가 약합니다. excess_return={_fmt_pct(report['excess_return'])}, hit_rate={_fmt(report['hit_rate'])}."
            )

    lines = [
        "# Walk-Forward Score Validation",
        "",
        f"- generated_at: {generated_at}",
        "- score_source: current final_score formula reapplied to stored historical prediction dates",
        "- operating_score_axes: ret_score, prob_score(60d), tech_score, qual_score, risk_penalty",
        "- diagnostic_only_axes: valuation_score, safety_score, liquidity_score",
        "- benchmark_definition: same-date scored universe equal-weight realized return",
        f"- minimum_rows_per_run: {MIN_RUN_ROWS}",
        "- recomputed_from_current_code: true",
        "",
        "## Summary",
        _markdown_table(
            top_summary_rows,
            ["horizon_days", "selection", "run_dates", "avg_return", "benchmark_return", "excess_return", "hit_rate", "avg_mdd", "median_return"],
        ),
        "",
        "## Interpretation",
        *interpretation_lines,
        "",
    ]

    if excluded_dates:
        lines.extend([
            "## Excluded Dates",
            *[f"- {item}" for item in excluded_dates],
            "",
        ])

    for report in horizon_reports:
        lines.append(f"## Horizon {report['horizon_days']}d")
        if report["status"] != "OK":
            lines.append(f"- status: {report['status']}")
            lines.append(f"- reason: {report.get('reason', 'unknown')}")
            lines.append("")
            continue

        lines.append(f"- latest_matured_date: {report['latest_matured_date']}")
        lines.append(f"- eligible_dates: {', '.join(report['eligible_dates'])}")
        lines.append(f"- benchmark_definition: same-date scored universe equal-weight realized return")
        lines.append("")
        lines.append("### Top20 vs Benchmark")
        lines.append(
            _markdown_table(
                [[
                    _fmt_pct(report["top20_avg_return"]),
                    _fmt_pct(report["benchmark_return"]),
                    _fmt_pct(report["excess_return"]),
                    _fmt(report["hit_rate"]),
                    _fmt_pct(report["avg_mdd"]),
                ]],
                ["top20_avg_return", "benchmark_return", "excess_return", "hit_rate", "avg_mdd"],
            )
        )
        lines.append("")
        lines.append("### Top vs Bottom Group")
        lines.append(
            _markdown_table(
                [[
                    _fmt_pct(report["spread"]["top_avg_return"]),
                    _fmt_pct(report["spread"]["bottom_avg_return"]),
                    _fmt_pct(report["spread"]["spread"]),
                    _fmt_pct(report["spread"]["top_avg_mdd"]),
                    _fmt_pct(report["spread"]["bottom_avg_mdd"]),
                ]],
                ["top20_avg_return", "bottom20_avg_return", "return_spread", "top20_avg_mdd", "bottom20_avg_mdd"],
            )
        )
        lines.append("")
        lines.append("### Rank Decile Performance")
        lines.append(_markdown_table(report["decile_rows"], ["decile", "n", "avg_return", "hit_rate", "avg_mdd"]))
        lines.append("")
        lines.append("### Confidence Bucket Performance")
        lines.append(_markdown_table(report["bucket_rows"], ["confidence_bucket", "n", "avg_return", "hit_rate", "avg_mdd"]))
        lines.append("")

    lines.extend([
        "## Notes",
        "- `hit_rate`는 realized return > 0 비율입니다.",
        "- `avg_mdd`는 같은 horizon의 realized forward MDD 평균입니다.",
        "- 90d는 mature coverage가 부족하면 `INSUFFICIENT`로 남깁니다.",
        "- 현재 로컬 데이터에는 연속 benchmark index 시계열이 없어서, benchmark는 scored universe 평균으로 정의했습니다.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ranked, excluded_dates = build_rank_history()
    horizon_reports = [build_horizon_report(ranked, horizon) for horizon in HORIZONS]
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_markdown(horizon_reports, excluded_dates), encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
