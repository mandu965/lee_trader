"""
Model trust report.

Uses prediction_history + backtest_outcome (DB) or backfill CSVs to summarize:
- rank correlation (Spearman/Pearson)
- top-N hit rate (precision vs realized top-N)
- quantile return spread (top vs bottom)

Example (CSV backfill):
  python python/research/model_trust_report.py ^
    --source csv ^
    --pred-csv data/prediction_history_backfill.csv ^
    --outcome-csv data/backtest_outcome_backfill.csv ^
    --horizon-days 60 ^
    --out-dir outputs/trust_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from db import get_engine
except Exception:
    get_engine = None


def _load_predictions_db(run_id: int, start: Optional[str], end: Optional[str], horizon: int) -> pd.DataFrame:
    if get_engine is None:
        raise RuntimeError("DB engine not available")
    sql = """
        SELECT
          as_of_date AS date,
          code,
          model_version,
          horizon_days,
          pred_return_30d,
          pred_return_60d,
          pred_return_90d,
          pred_mdd_30d,
          pred_mdd_60d,
          pred_mdd_90d,
          prob_top20_30d,
          prob_top20_60d,
          prob_top20_90d
        FROM research.prediction_history
        WHERE run_id = %(run_id)s
    """
    params = {"run_id": run_id}
    if start:
        sql += " AND as_of_date >= %(start)s"
        params["start"] = start
    if end:
        sql += " AND as_of_date <= %(end)s"
        params["end"] = end
    if horizon:
        sql += " AND horizon_days = %(horizon)s"
        params["horizon"] = horizon
    eng = get_engine()
    df = pd.read_sql(sql, eng, params=params)
    return df


def _load_outcome_db(run_id: int, start: Optional[str], end: Optional[str], horizon: int) -> pd.DataFrame:
    if get_engine is None:
        raise RuntimeError("DB engine not available")
    sql = """
        SELECT
          as_of_date AS date,
          code,
          horizon_days,
          realized_return,
          realized_mdd
        FROM research.backtest_outcome
        WHERE run_id = %(run_id)s
    """
    params = {"run_id": run_id}
    if start:
        sql += " AND as_of_date >= %(start)s"
        params["start"] = start
    if end:
        sql += " AND as_of_date <= %(end)s"
        params["end"] = end
    if horizon:
        sql += " AND horizon_days = %(horizon)s"
        params["horizon"] = horizon
    eng = get_engine()
    df = pd.read_sql(sql, eng, params=params)
    return df


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_preds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "as_of_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"as_of_date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["code"] = df["code"].astype(str)
    return df


def _normalize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "as_of_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"as_of_date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["code"] = df["code"].astype(str)
    return df


def _assign_quantile(series: pd.Series, q: int) -> pd.Series:
    order = series.rank(method="first", ascending=False)
    n = len(series)
    if n <= 0:
        return pd.Series([], dtype=int)
    return ((order - 1) * q / n).astype(int) + 1


def _topn_hit_rate(df: pd.DataFrame, pred_field: str, top_n: int) -> pd.DataFrame:
    rows = []
    for date, grp in df.groupby("date"):
        pred_top = set(grp.nlargest(top_n, pred_field)["code"])
        real_top = set(grp.nlargest(top_n, "realized_return")["code"])
        if top_n <= 0:
            hit = float("nan")
        else:
            hit = len(pred_top & real_top) / float(top_n)
        rows.append({"date": date, "hit_rate": hit, "n": len(grp)})
    return pd.DataFrame(rows)


def _quantile_returns(df: pd.DataFrame, pred_field: str, q: int) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    df["quantile"] = df.groupby("date")[pred_field].transform(lambda s: _assign_quantile(s, q))
    per_date = (
        df.groupby(["date", "quantile"])["realized_return"]
        .mean()
        .reset_index()
    )
    summary = (
        per_date.groupby("quantile")["realized_return"]
        .mean()
        .reset_index()
        .rename(columns={"realized_return": "avg_realized_return"})
        .sort_values("quantile")
    )
    top = summary.loc[summary["quantile"] == 1, "avg_realized_return"]
    bottom = summary.loc[summary["quantile"] == q, "avg_realized_return"]
    spread = float(top.values[0] - bottom.values[0]) if not top.empty and not bottom.empty else float("nan")
    return summary, spread


def _corr(df: pd.DataFrame, pred_field: str) -> Tuple[float, float]:
    sub = df[[pred_field, "realized_return"]].dropna()
    if sub.empty:
        return float("nan"), float("nan")
    spearman = sub.corr(method="spearman").iloc[0, 1]
    pearson = sub.corr(method="pearson").iloc[0, 1]
    return float(spearman), float(pearson)


def _topn_return(df: pd.DataFrame, pred_field: str, top_n: int) -> Tuple[float, float]:
    rows = []
    for date, grp in df.groupby("date"):
        if grp.empty:
            continue
        top = grp.nlargest(top_n, pred_field)["realized_return"].mean()
        all_mean = grp["realized_return"].mean()
        rows.append({"date": date, "topn_mean": top, "all_mean": all_mean})
    if not rows:
        return float("nan"), float("nan")
    d = pd.DataFrame(rows)
    return float(d["topn_mean"].mean()), float((d["topn_mean"] - d["all_mean"]).mean())


def compute_report(
    df: pd.DataFrame,
    pred_field: str,
    top_n: int,
    q: int,
) -> Dict[str, object]:
    spearman, pearson = _corr(df, pred_field)
    quantile_summary, spread = _quantile_returns(df, pred_field, q)
    hit = _topn_hit_rate(df, pred_field, top_n)
    topn_mean, topn_spread = _topn_return(df, pred_field, top_n)
    report = {
        "n_rows": int(df.shape[0]),
        "n_dates": int(df["date"].nunique()),
        "pred_field": pred_field,
        "spearman_ic": spearman,
        "pearson_ic": pearson,
        "topn_hit_rate_mean": float(hit["hit_rate"].mean()) if not hit.empty else float("nan"),
        "topn_return_mean": topn_mean,
        "topn_return_spread_vs_all": topn_spread,
        "quantile_spread_top_minus_bottom": spread,
    }
    return report, quantile_summary, hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "db", "csv"], default="auto")
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--pred-csv", type=Path, default=Path("data/prediction_history_backfill.csv"))
    ap.add_argument("--outcome-csv", type=Path, default=Path("data/backtest_outcome_backfill.csv"))
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    ap.add_argument("--horizon-days", type=int, default=60)
    ap.add_argument("--pred-field", type=str, default="pred_return_60d")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/trust_report"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    use_db = args.source in ("auto", "db") and args.run_id is not None and get_engine is not None
    if args.source == "db" and not use_db:
        raise SystemExit("DB source requested but DB is not available or run-id missing.")

    if use_db:
        preds = _load_predictions_db(args.run_id, args.start_date, args.end_date, args.horizon_days)
        outs = _load_outcome_db(args.run_id, args.start_date, args.end_date, args.horizon_days)
    else:
        preds = _load_csv(args.pred_csv)
        outs = _load_csv(args.outcome_csv)

    preds = _normalize_preds(preds)
    outs = _normalize_outcomes(outs)

    if "horizon_days" in preds.columns:
        preds = preds[preds["horizon_days"] == args.horizon_days]
    if "horizon_days" in outs.columns:
        outs = outs[outs["horizon_days"] == args.horizon_days]

    if args.start_date:
        start = pd.to_datetime(args.start_date).date()
        preds = preds[preds["date"] >= start]
        outs = outs[outs["date"] >= start]
    if args.end_date:
        end = pd.to_datetime(args.end_date).date()
        preds = preds[preds["date"] <= end]
        outs = outs[outs["date"] <= end]

    if args.pred_field not in preds.columns:
        raise SystemExit(f"pred_field not found in predictions: {args.pred_field}")

    merged = preds.merge(
        outs[["date", "code", "realized_return"]],
        on=["date", "code"],
        how="inner",
    )
    merged = merged.dropna(subset=[args.pred_field, "realized_return"]).copy()

    if merged.empty:
        raise SystemExit("No merged rows available for report.")

    reports = []
    quantile_tables = []
    hit_tables = []

    overall_report, q_summary, hit = compute_report(merged, args.pred_field, args.top_n, args.quantiles)
    overall_report["group"] = "overall"
    reports.append(overall_report)
    q_summary.insert(0, "group", "overall")
    quantile_tables.append(q_summary)
    hit["group"] = "overall"
    hit_tables.append(hit)

    if "model_version" in merged.columns:
        for mv, grp in merged.groupby("model_version"):
            if grp.empty:
                continue
            r, q_s, h = compute_report(grp, args.pred_field, args.top_n, args.quantiles)
            r["group"] = str(mv)
            reports.append(r)
            q_s.insert(0, "group", str(mv))
            quantile_tables.append(q_s)
            h["group"] = str(mv)
            hit_tables.append(h)

    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.concat(quantile_tables, ignore_index=True).to_csv(
        args.out_dir / "quantile_returns.csv", index=False, encoding="utf-8"
    )
    pd.concat(hit_tables, ignore_index=True).to_csv(
        args.out_dir / "topn_hit_rate.csv", index=False, encoding="utf-8"
    )

    print(f"[trust_report] rows={merged.shape[0]} dates={merged['date'].nunique()}")
    print(f"[trust_report] summary -> {report_path}")


if __name__ == "__main__":
    main()
