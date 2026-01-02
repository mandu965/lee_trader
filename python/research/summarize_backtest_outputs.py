"""
Quick aggregator for backtest output folders.

Reads summary.csv / portfolio_*.csv under --input-dir and emits a concise CSV/JSON.
Designed to work with run_research_backtest.py outputs.

Usage:
  python python/research/summarize_backtest_outputs.py `
    --input-dir outputs/backtest_run4_q1 `
    --output outputs/backtest_run4_q1/summary_agg.csv
"""
import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def compute_mdd(cum_returns: pd.Series) -> float | None:
    if cum_returns.empty:
        return None
    # Convert cum_return to equity curve (1 + r)
    equity = 1 + cum_returns
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    mdd = drawdown.min()
    return float(mdd) if pd.notna(mdd) else None


def summarize_portfolio(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {"file": str(path)}
    if "portfolio_return" in df.columns:
        r = df["portfolio_return"].dropna()
        out["avg_return"] = r.mean()
        out["hit_rate"] = (r > 0).mean() if not r.empty else None
        out["num_periods"] = len(r)
    if "cum_return" in df.columns:
        mdd = compute_mdd(df["cum_return"].dropna())
        out["mdd"] = mdd
    return out


def run(input_dir: Path, output: Path | None) -> None:
    summaries: List[dict] = []

    # 1) Backtest summary.csv (if exists)
    summary_csv = input_dir / "summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df.insert(0, "file", str(summary_csv))
        summaries.extend(df.to_dict(orient="records"))

    # 2) portfolio_*.csv (strategy-level)
    for p in sorted(input_dir.glob("portfolio_*.csv")):
        summaries.append(summarize_portfolio(p))

    if not summaries:
        raise SystemExit(f"No summary/portfolio files found under {input_dir}")

    out_df = pd.DataFrame(summaries)
    out_df.to_csv(output, index=False)

    # Also emit JSON next to CSV for quick inspection
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[summary] wrote {output}")
    print(f"[summary] wrote {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="Backtest output folder")
    ap.add_argument("--output", type=Path, required=True, help="Path to write aggregated CSV")
    args = ap.parse_args()

    run(args.input_dir, args.output)


if __name__ == "__main__":
    main()
