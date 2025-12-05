"""
Benchmark comparison for strategy vs KOSPI/KOSDAQ ETFs.

- Reads ETF CSVs (KODEX200 069500, KODEX KOSDAQ150 229200)
- Loads strategy trades, converts rebalance returns to daily series via 60d interpolation
- Computes cumulative return, CAGR, volatility, Sharpe, and Alpha
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DEFAULT_KOSPI = Path("data/benchmarks/kospi_069500.csv")
DEFAULT_KOSDAQ = Path("data/benchmarks/kosdaq_229200.csv")
DEFAULT_TRADES_PRIORITY = [
    Path("data/backtest_walkforward_combined_90_10_top10_trades.csv"),
    Path("data/backtest_walkforward_60d_30d_combined_top10_trades.csv"),
]


def load_etf(csv_path: Path) -> pd.DataFrame:
    """
    Load ETF CSV and return columns: date (datetime), close (float), ret (daily), cum_return.
    Handles both cleaned (date, close) and Naver-style (날짜, 종가, ...) formats.
    """
    df = pd.read_csv(csv_path)
    if "date" in df.columns and "close" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
    elif "날짜" in df.columns and "종가" in df.columns:
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["날짜"], errors="coerce").dt.normalize(),
                "close": pd.to_numeric(df["종가"].astype(str).str.replace(",", ""), errors="coerce"),
            }
        )
    else:
        raise ValueError(f"Unsupported ETF CSV format: {csv_path}")
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    out["ret"] = out["close"].pct_change()
    out = out.dropna(subset=["ret"])
    out["cum_return"] = (1 + out["ret"]).cumprod()
    return out


def compute_perf_stats(df: pd.DataFrame, freq_per_year: int) -> dict:
    """
    df: expects columns ['date', 'return', 'cum_return']
    freq_per_year: e.g., 24 for biweekly strategy, 252 for daily ETF
    """
    if df.empty:
        return dict(
            start_date=None,
            end_date=None,
            num_periods=0,
            total_return=np.nan,
            cagr=np.nan,
            vol=np.nan,
            sharpe=np.nan,
        )
    start_date = df["date"].min()
    end_date = df["date"].max()
    num_periods = df["return"].dropna().shape[0]
    total_return = df["cum_return"].iloc[-1] - 1
    years = max((end_date - start_date).days / 365.0, 1e-6)
    cagr = (1 + total_return) ** (1 / years) - 1
    period_ret = df["return"].dropna()
    vol = period_ret.std(ddof=1) * np.sqrt(freq_per_year)
    mean_ret_annual = period_ret.mean() * freq_per_year
    sharpe = mean_ret_annual / (vol + 1e-9)
    return dict(
        start_date=start_date,
        end_date=end_date,
        num_periods=num_periods,
        total_return=total_return,
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
    )


def load_strategy_returns() -> pd.DataFrame:
    """
    Load strategy trades CSV and build daily return series via 60d interpolation.
    Priority:
      1) data/backtest_walkforward_combined_90_10_top10_trades.csv
      2) data/backtest_walkforward_60d_30d_combined_top10_trades.csv
    Returns daily DataFrame columns: date, return, cum_return
    """
    trades_path = None
    for cand in DEFAULT_TRADES_PRIORITY:
        if cand.exists():
            trades_path = cand
            break
    if trades_path is None:
        raise FileNotFoundError("Strategy trades CSV not found in default priority list.")

    df = pd.read_csv(trades_path)
    if "rebalance_date" not in df.columns or "realized_return_60d" not in df.columns:
        raise ValueError("Trades CSV must have rebalance_date and realized_return_60d columns.")
    df["date"] = pd.to_datetime(df["rebalance_date"]).dt.normalize()
    rets = df.groupby("date")["realized_return_60d"].mean().reset_index()
    rets = rets.sort_values("date").reset_index(drop=True)
    rets = rets.rename(columns={"realized_return_60d": "return"})

    # daily interpolation over holding days (assume 60d horizon)
    holding = 60
    daily_records = []
    for _, row in rets.iterrows():
        r = row["return"]
        start = row["date"]
        daily_r = (1 + r) ** (1 / holding) - 1
        for offset in range(holding):
            daily_records.append({"date": start + pd.Timedelta(days=offset), "return": daily_r})
    daily_df = pd.DataFrame(daily_records)
    daily_df = daily_df.groupby("date")["return"].mean().reset_index().sort_values("date")
    daily_df["cum_return"] = (1 + daily_df["return"]).cumprod()
    return daily_df


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison for strategy vs KOSPI/KOSDAQ ETFs")
    parser.add_argument("--kospi-csv", type=Path, default=DEFAULT_KOSPI)
    parser.add_argument("--kosdaq-csv", type=Path, default=DEFAULT_KOSDAQ)
    parser.add_argument("--trades-csv", type=Path, default=None, help="Strategy trades CSV (realized_return_60d)")
    parser.add_argument("--out-csv", type=Path, default=Path("data/benchmarks/benchmark_comparison_summary.csv"))
    args = parser.parse_args()

    # strategy daily returns (already interpolated to daily inside load_strategy_returns)
    strat_daily = load_strategy_returns()
    strat_stats = compute_perf_stats(strat_daily, freq_per_year=252)

    # load benchmarks
    kospi = load_etf(args.kospi_csv)
    kosdaq = load_etf(args.kosdaq_csv)

    # align dates (overlapping window)
    start_date = max(strat_daily["date"].min(), kospi["date"].min(), kosdaq["date"].min())
    end_date = min(strat_daily["date"].max(), kospi["date"].max(), kosdaq["date"].max())
    kospi = kospi[(kospi["date"] >= start_date) & (kospi["date"] <= end_date)]
    kosdaq = kosdaq[(kosdaq["date"] >= start_date) & (kosdaq["date"] <= end_date)]
    strat_daily = strat_daily[(strat_daily["date"] >= start_date) & (strat_daily["date"] <= end_date)]

    # compute ETF stats (convert column names to match compute_perf_stats expectations)
    kp_df = kospi.rename(columns={"ret": "return", "cum_return": "cum_return", "date": "date"})
    kq_df = kosdaq.rename(columns={"ret": "return", "cum_return": "cum_return", "date": "date"})
    kp_stats = compute_perf_stats(kp_df, freq_per_year=252)
    kq_stats = compute_perf_stats(kq_df, freq_per_year=252)

    # alpha: strategy annual return minus benchmark annual return (CAGR)
    alpha_kp = strat_stats["cagr"] - kp_stats["cagr"] if np.isfinite(strat_stats["cagr"]) and np.isfinite(kp_stats["cagr"]) else np.nan
    alpha_kq = strat_stats["cagr"] - kq_stats["cagr"] if np.isfinite(strat_stats["cagr"]) and np.isfinite(kq_stats["cagr"]) else np.nan

    summary = pd.DataFrame(
        [
            {
                "name": "strategy",
                "cum_return": strat_stats["total_return"],
                "CAGR": strat_stats["cagr"],
                "vol_annual": strat_stats["vol"],
                "sharpe": strat_stats["sharpe"],
                "alpha_vs_kospi": alpha_kp,
                "alpha_vs_kosdaq": alpha_kq,
            },
            {
                "name": "KOSPI(069500)",
                "cum_return": kp_stats["total_return"],
                "CAGR": kp_stats["cagr"],
                "vol_annual": kp_stats["vol"],
                "sharpe": kp_stats["sharpe"],
                "alpha_vs_kospi": 0.0,
                "alpha_vs_kosdaq": np.nan,
            },
            {
                "name": "KOSDAQ(229200)",
                "cum_return": kq_stats["total_return"],
                "CAGR": kq_stats["cagr"],
                "vol_annual": kq_stats["vol"],
                "sharpe": kq_stats["sharpe"],
                "alpha_vs_kospi": np.nan,
                "alpha_vs_kosdaq": 0.0,
            },
        ]
    )

    print(summary)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved summary -> {args.out_csv}")


if __name__ == "__main__":
    main()
