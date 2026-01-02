"""
Baselines runner (equal-weight universe, benchmark, random topN).

This is a lightweight helper built on top of research/data_loader + outcomes.
Assumptions:
  - price_source=db or csv provides date, code, close (adj_close accepted in loader)
  - horizon_days determines the forward return window
  - No prediction scores are needed; we compute realized_return directly from prices.

Outputs:
  - baseline_equal_weight.csv
  - baseline_benchmark.csv (if benchmark available)
  - baseline_random_topN_seed<N>.csv (one per seed)
  - summary_baselines.csv

Example:
  python python/research/run_baselines.py `
    --start-date 2024-01-01 --end-date 2024-03-31 `
    --horizon-days 60 --top-n 20 `
    --price-source db `
    --benchmark KOSPI `
    --random-seeds 20 `
    --output-dir outputs/baselines_q1
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_loader import load_prices, load_benchmark
from outcomes import compute_realized_outcomes
from metrics import summarize_strategy_performance


def compute_mdd(cum_returns: pd.Series) -> float | None:
    if cum_returns.empty:
        return None
    equity = 1 + cum_returns
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min()) if not dd.empty else None


def summarize_portfolio(df: pd.DataFrame, label: str, tx_cost_bps: float) -> dict:
    r = df["portfolio_return"].dropna()
    perf = summarize_strategy_performance(df.rename(columns={"portfolio_return": "portfolio_return"}))
    out = {
        "name": label,
        "tx_cost_bps": tx_cost_bps,
        "avg_return": perf.get("avg_return"),
        "hit_rate": perf.get("hit_rate"),
        "sharpe_like": perf.get("sharpe_like"),
        "num_periods": perf.get("num_periods"),
        "mdd": compute_mdd(df.get("cum_return", pd.Series(dtype=float)).dropna()),
        "turnover_avg": df.get("turnover", pd.Series(dtype=float)).mean() if "turnover" in df.columns else None,
    }
    return out


def build_realized_returns(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    # Build minimal preds-like frame to reuse compute_realized_outcomes
    dummy_cfg = BacktestConfig(
        start_date=prices["date"].min(),
        end_date=prices["date"].max(),
        model_versions=["baseline"],
        horizon_days=horizon_days,
        top_n=0,
    )
    preds = prices[["date", "code"]].copy()
    preds["model_version"] = "baseline"
    preds["ret_score"] = 0.0
    preds["prob_score"] = 0.0
    preds["qual_score"] = 0.0
    preds["tech_score"] = 0.0
    preds["risk_penalty"] = 0.0
    preds["final_score_custom"] = 0.0

    realized = compute_realized_outcomes(preds, prices, dummy_cfg)
    realized = realized[["date", "code", "model_version", "realized_return"]].copy()
    return realized


def _compute_turnover(prev_w: Dict[str, float], curr_w: Dict[str, float]) -> float:
    """Turnover = 0.5 * L1 difference between successive weight vectors."""
    all_codes = set(prev_w) | set(curr_w)
    diff = 0.0
    for c in all_codes:
        diff += abs(curr_w.get(c, 0.0) - prev_w.get(c, 0.0))
    return diff * 0.5


def run_equal_weight(realized: pd.DataFrame) -> pd.DataFrame:
    records = []
    prev_w: Dict[str, float] = {}
    for dt, grp in realized.groupby("date"):
        k = len(grp)
        if k == 0:
            continue
        w = 1.0 / k
        weights = {code: w for code in grp["code"]}
        ret = grp["realized_return"].mean()
        turnover = _compute_turnover(prev_w, weights)
        records.append({"date": dt, "portfolio_return": ret, "turnover": turnover})
        prev_w = weights
    port = pd.DataFrame(records).sort_values("date")
    port["cum_return"] = (1 + port["portfolio_return"]).cumprod() - 1
    return port


def run_random_topn(realized: pd.DataFrame, top_n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    prev_w: Dict[str, float] = {}
    for dt, grp in realized.groupby("date"):
        codes = grp["code"].tolist()
        if not codes:
            continue
        k = min(top_n, len(codes))
        chosen = rng.choice(codes, size=k, replace=False)
        sub = grp[grp["code"].isin(chosen)]
        ret = sub["realized_return"].mean()
        w = 1.0 / k
        weights = {code: w for code in chosen}
        turnover = _compute_turnover(prev_w, weights)
        records.append({"date": dt, "portfolio_return": ret, "turnover": turnover})
        prev_w = weights
    port = pd.DataFrame(records).sort_values("date")
    port["cum_return"] = (1 + port["portfolio_return"]).cumprod() - 1
    return port


def run_benchmark(bench: pd.DataFrame) -> pd.DataFrame:
    if bench.empty:
        return pd.DataFrame()
    port = bench[["date", "return"]].rename(columns={"return": "portfolio_return"}).copy()
    port["cum_return"] = (1 + port["portfolio_return"]).cumprod() - 1
    port["turnover"] = 0.0
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, type=str)
    ap.add_argument("--end-date", required=True, type=str)
    ap.add_argument("--horizon-days", type=int, default=60)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--price-source", choices=["db", "csv"], default="db")
    ap.add_argument("--prices-csv", type=Path, default=Path("data/prices_daily_adjusted.csv"))
    ap.add_argument("--benchmark", type=str, default=None, help="benchmark code (e.g., KOSPI)")
    ap.add_argument("--benchmark-csv", type=Path, default=None)
    ap.add_argument("--random-seeds", type=int, default=10, help="number of random topN runs")
    ap.add_argument("--tx-cost-bps", type=float, default=0.0, help="per-period transaction cost in bps (applied once per rebalance)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = BacktestConfig(
        start_date=pd.to_datetime(args.start_date).date(),
        end_date=pd.to_datetime(args.end_date).date(),
        model_versions=["baseline"],
        horizon_days=args.horizon_days,
        top_n=args.top_n,
        price_source=args.price_source,
        prices_csv=str(args.prices_csv),
        benchmark_code=args.benchmark,
        benchmark_csv=str(args.benchmark_csv) if args.benchmark_csv else None,
    )

    prices = load_prices(cfg)
    realized = build_realized_returns(prices, args.horizon_days)

    summaries: List[dict] = []
    cost = float(args.tx_cost_bps) / 10000.0

    # Equal-weight universe
    eq = run_equal_weight(realized)
    if cost != 0:
        eq["portfolio_return"] = eq["portfolio_return"] - cost
    eq_path = args.output_dir / "baseline_equal_weight.csv"
    eq.to_csv(eq_path, index=False)
    summaries.append(summarize_portfolio(eq, "equal_weight", args.tx_cost_bps))

    # Benchmark
    bench = load_benchmark(cfg)
    if not bench.empty:
        bdf = run_benchmark(bench)
        if cost != 0:
            bdf["portfolio_return"] = bdf["portfolio_return"] - cost
        bdf.to_csv(args.output_dir / "baseline_benchmark.csv", index=False)
        summaries.append(summarize_portfolio(bdf, "benchmark", args.tx_cost_bps))

    # Random TopN
    for i, seed in enumerate(range(args.random_seeds), start=1):
        rnd = run_random_topn(realized, args.top_n, seed=seed)
        if cost != 0:
            rnd["portfolio_return"] = rnd["portfolio_return"] - cost
        rnd.to_csv(args.output_dir / f"baseline_random_top{args.top_n}_seed{seed}.csv", index=False)
        summaries.append(summarize_portfolio(rnd, f"random_top{args.top_n}_seed{seed}", args.tx_cost_bps))

    # Summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "summary_baselines.csv", index=False)
    print(f"[baselines] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
