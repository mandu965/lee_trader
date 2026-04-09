from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from outcome_maturity import attach_forward_outcomes, load_price_history


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
RANKING_CURRENT_CSV = DATA_DIR / "ranking_final.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
OUT_REPORT_MD = OUTPUT_DIR / "model_edge_report.md"

TOP_BUCKETS = [5, 8, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether the model has statistical edge versus benchmark and random portfolios.")
    parser.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    parser.add_argument("--ranking-current-csv", type=Path, default=RANKING_CURRENT_CSV)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--market-status-csv", type=Path, default=MARKET_STATUS_CSV)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--random-portfolios", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--out-md", type=Path, default=OUT_REPORT_MD)
    return parser.parse_args()


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _fmt_num(value: object, digits: int = 3) -> str:
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
    df["snapshot_priority"] = df["snapshot_file"].eq(resolved_current.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    return df.dropna(subset=["date", "code", "rank_final"]).copy()


def build_case_frame(ranking: pd.DataFrame, prices_csv: Path, horizon_days: int) -> pd.DataFrame:
    prices = load_price_history(prices_csv=_resolve(prices_csv))
    outcomes = attach_forward_outcomes(prices, horizon_days=horizon_days).rename(
        columns={"realized_return": "forward_return", "realized_mdd": "forward_mdd_like"}
    )
    work = ranking[["date", "code", "rank_final", "name", "market"]].copy()
    work = work.merge(outcomes[["date", "code", "forward_return", "forward_mdd_like"]], on=["date", "code"], how="left")
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    return work


def build_benchmark_frame(market_status_csv: Path, horizon_days: int) -> pd.DataFrame:
    market = pd.read_csv(_resolve(market_status_csv), low_memory=False)
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["code"] = "KOSPI"
    market["close"] = pd.to_numeric(market["kospi_close"], errors="coerce")
    market = market.dropna(subset=["date", "close"]).copy()
    outcome = attach_forward_outcomes(market[["code", "date", "close"]], horizon_days=horizon_days).rename(
        columns={"realized_return": "benchmark_return", "realized_mdd": "benchmark_mdd_like"}
    )
    return outcome[["date", "benchmark_return", "benchmark_mdd_like"]]


def compute_sharpe(returns: pd.Series, horizon_days: int) -> float | None:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if len(series) < 2:
        return None
    std = float(series.std(ddof=1))
    if std <= 0:
        return None
    annualization = math.sqrt(252.0 / max(1.0, float(horizon_days)))
    return float(series.mean() / std * annualization)


def compute_path_mdd(returns: pd.Series) -> float | None:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return None
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def empirical_p_value(random_distribution: np.ndarray, observed: float) -> float | None:
    if random_distribution.size == 0 or np.isnan(observed):
        return None
    return float((np.sum(random_distribution >= observed) + 1.0) / (random_distribution.size + 1.0))


def infer_confidence_level(p_value: float | None, sample_count: int) -> str:
    if p_value is None or sample_count < 10:
        return "LOW"
    if p_value <= 0.01 and sample_count >= 20:
        return "HIGH"
    if p_value <= 0.05 and sample_count >= 10:
        return "MEDIUM"
    return "LOW"


def infer_edge_decision(*, p_value: float | None, model_mean: float | None, benchmark_mean: float | None, sample_count: int) -> str:
    if p_value is None or model_mean is None or benchmark_mean is None or sample_count < 10:
        return "UNCERTAIN"
    if p_value <= 0.05 and model_mean > benchmark_mean:
        return "YES"
    if p_value > 0.20 or model_mean <= benchmark_mean:
        return "NO"
    return "UNCERTAIN"


def estimate_min_sample_size(effect: float | None, sigma: float | None, alpha: float = 0.05, power: float = 0.80) -> float | None:
    if effect is None or sigma is None:
        return None
    effect = float(effect)
    sigma = float(sigma)
    if sigma <= 0 or effect <= 0:
        return None
    z_alpha = 1.96
    z_beta = 0.84
    return float(((z_alpha + z_beta) * sigma / effect) ** 2)


def evaluate_bucket(case_frame: pd.DataFrame, benchmark_frame: pd.DataFrame, *, top_n: int, random_portfolios: int, random_seed: int, horizon_days: int) -> tuple[dict[str, object], pd.DataFrame]:
    subset = case_frame.loc[case_frame["rank_final"].le(top_n)].copy()
    by_date = subset.groupby("date", sort=True)
    universe_by_date = case_frame.groupby("date", sort=True)

    model_rows: list[dict[str, object]] = []
    random_daily_rows: list[dict[str, object]] = []

    rng = np.random.default_rng(random_seed + top_n)
    date_list = []
    model_returns = []
    benchmark_returns = []

    for date, day_df in by_date:
        date = pd.Timestamp(date).normalize()
        day_returns = pd.to_numeric(day_df["forward_return"], errors="coerce")
        if day_returns.notna().sum() < top_n:
            continue
        universe_df = universe_by_date.get_group(date)
        universe_returns = pd.to_numeric(universe_df["forward_return"], errors="coerce")
        eligible = universe_df.loc[universe_returns.notna()].copy()
        if len(eligible) < top_n:
            continue

        model_ret = float(day_returns.dropna().mean())
        model_mdd = float(pd.to_numeric(day_df["forward_mdd_like"], errors="coerce").dropna().mean())
        bench_row = benchmark_frame.loc[benchmark_frame["date"].eq(date)].copy()
        bench_ret = float(pd.to_numeric(bench_row["benchmark_return"], errors="coerce").dropna().iloc[0]) if not bench_row.empty and pd.to_numeric(bench_row["benchmark_return"], errors="coerce").notna().any() else np.nan
        bench_mdd = float(pd.to_numeric(bench_row["benchmark_mdd_like"], errors="coerce").dropna().iloc[0]) if not bench_row.empty and pd.to_numeric(bench_row["benchmark_mdd_like"], errors="coerce").notna().any() else np.nan

        model_rows.append(
            {
                "date": date,
                "top_n": top_n,
                "model_return": model_ret,
                "model_mdd_like": model_mdd,
                "benchmark_return": bench_ret,
                "benchmark_mdd_like": bench_mdd,
            }
        )
        date_list.append(date)
        model_returns.append(model_ret)
        benchmark_returns.append(bench_ret)

        eligible_returns = pd.to_numeric(eligible["forward_return"], errors="coerce").to_numpy(dtype=float)
        for sim in range(random_portfolios):
            picked = rng.choice(eligible_returns, size=top_n, replace=False)
            random_daily_rows.append(
                {
                    "date": date,
                    "top_n": top_n,
                    "simulation_id": sim,
                    "random_return": float(np.mean(picked)),
                }
            )

    model_daily = pd.DataFrame(model_rows)
    random_daily = pd.DataFrame(random_daily_rows)

    if model_daily.empty or random_daily.empty:
        return {
            "top_n": top_n,
            "sample_count": 0,
            "model_mean_return": None,
            "benchmark_mean_return": None,
            "random_mean_return": None,
            "p_value_vs_random": None,
            "sharpe_model": None,
            "sharpe_benchmark": None,
            "mdd_model": None,
            "mdd_benchmark": None,
            "edge_decision": "UNCERTAIN",
            "confidence_level": "LOW",
            "min_required_samples": None,
        }, model_daily

    random_summary = (
        random_daily.groupby("simulation_id", sort=True)["random_return"]
        .mean()
        .to_numpy(dtype=float)
    )

    observed_model_mean = float(pd.to_numeric(model_daily["model_return"], errors="coerce").mean())
    observed_benchmark_mean = float(pd.to_numeric(model_daily["benchmark_return"], errors="coerce").mean())
    random_mean = float(np.mean(random_summary)) if random_summary.size else None
    p_value = empirical_p_value(random_summary, observed_model_mean)
    model_sharpe = compute_sharpe(model_daily["model_return"], horizon_days=horizon_days)
    benchmark_sharpe = compute_sharpe(model_daily["benchmark_return"], horizon_days=horizon_days)
    model_mdd = compute_path_mdd(model_daily["model_return"])
    benchmark_mdd = compute_path_mdd(model_daily["benchmark_return"])
    effect = observed_model_mean - (random_mean if random_mean is not None else float("nan"))
    random_sigma = float(np.std(random_summary, ddof=1)) if random_summary.size > 1 else None
    min_required = estimate_min_sample_size(effect if effect > 0 else None, random_sigma)
    sample_count = int(len(model_daily))

    edge_decision = infer_edge_decision(
        p_value=p_value,
        model_mean=observed_model_mean,
        benchmark_mean=observed_benchmark_mean,
        sample_count=sample_count,
    )
    confidence_level = infer_confidence_level(p_value=p_value, sample_count=sample_count)

    return {
        "top_n": top_n,
        "sample_count": sample_count,
        "model_mean_return": observed_model_mean,
        "benchmark_mean_return": observed_benchmark_mean,
        "random_mean_return": random_mean,
        "p_value_vs_random": p_value,
        "sharpe_model": model_sharpe,
        "sharpe_benchmark": benchmark_sharpe,
        "mdd_model": model_mdd,
        "mdd_benchmark": benchmark_mdd,
        "edge_decision": edge_decision,
        "confidence_level": confidence_level,
        "min_required_samples": min_required,
    }, model_daily


def build_report(summary_df: pd.DataFrame, horizon_days: int, random_portfolios: int) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table = summary_df.copy()
    for col in ["model_mean_return", "benchmark_mean_return", "random_mean_return", "mdd_model", "mdd_benchmark"]:
        table[col] = table[col].map(_fmt_pct)
    for col in ["p_value_vs_random", "sharpe_model", "sharpe_benchmark", "min_required_samples"]:
        table[col] = table[col].map(_fmt_num)

    overall = "UNCERTAIN"
    if not summary_df.empty:
        yes = int(summary_df["edge_decision"].astype(str).eq("YES").sum())
        no = int(summary_df["edge_decision"].astype(str).eq("NO").sum())
        min_samples = int(pd.to_numeric(summary_df["sample_count"], errors="coerce").min()) if pd.to_numeric(summary_df["sample_count"], errors="coerce").notna().any() else 0
        if min_samples < 10:
            overall = "UNCERTAIN"
        elif yes >= 2:
            overall = "YES"
        elif no >= 2:
            overall = "NO"

    lines = [
        "# Model Edge Report",
        "",
        f"- generated_at: {generated_at}",
        f"- horizon_days: {horizon_days}",
        f"- random_portfolios_per_date: {random_portfolios}",
        f"- overall_edge_decision: {overall}",
        "",
        "## Summary",
        _markdown_table(
            table,
            [
                "top_n",
                "sample_count",
                "model_mean_return",
                "benchmark_mean_return",
                "random_mean_return",
                "p_value_vs_random",
                "sharpe_model",
                "sharpe_benchmark",
                "mdd_model",
                "mdd_benchmark",
                "edge_decision",
                "confidence_level",
                "min_required_samples",
            ],
        ),
        "",
        "## Interpretation Rule",
        "- `YES`: empirical p-value <= 0.05, sample_count >= 10, and model mean return > benchmark mean return.",
        "- `NO`: sample_count >= 10 and either model mean return <= benchmark mean return or empirical p-value > 0.20.",
        "- `UNCERTAIN`: everything in between, usually because sample size is still too small.",
        "",
        "## Notes",
        "- Random portfolios are sampled from the same-date ranking universe with the same portfolio size as the tested bucket.",
        "- p-value is empirical: fraction of random simulations whose mean return is at least as large as the model mean return.",
        "- Sharpe is annualized with `sqrt(252 / horizon_days)`.",
        "- Minimum required sample size is a rough power estimate using observed effect vs random-distribution sigma.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ranking = load_ranking_snapshots(args.ranking_history_dir, args.ranking_current_csv)
    case_frame = build_case_frame(ranking, prices_csv=args.prices_csv, horizon_days=int(args.horizon_days))
    benchmark_frame = build_benchmark_frame(args.market_status_csv, horizon_days=int(args.horizon_days))

    summaries: list[dict[str, object]] = []
    for top_n in TOP_BUCKETS:
        summary, _ = evaluate_bucket(
            case_frame,
            benchmark_frame,
            top_n=top_n,
            random_portfolios=int(args.random_portfolios),
            random_seed=int(args.random_seed),
            horizon_days=int(args.horizon_days),
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values("top_n").reset_index(drop=True)
    out_md = _resolve(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_report(summary_df, horizon_days=int(args.horizon_days), random_portfolios=int(args.random_portfolios)), encoding="utf-8")

    print(f"model_edge_report_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
