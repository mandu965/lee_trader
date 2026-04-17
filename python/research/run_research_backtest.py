# run_research_backtest.py
"""
CLI helper to run backtest_research with prediction_source="research".

Example:
  python python/research/run_research_backtest.py ^
    --run-id 4 ^
    --start-date 2024-01-01 --end-date 2024-03-31 ^
    --model-version v1 ^
    --horizon-days 60 ^
    --top-n 20 ^
    --price-source db ^
    --output-dir outputs/backtest_run4
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project/python paths are importable when run as a script
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "research"))

from config import BacktestConfig
from backtest_research import run_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run research backtest using prediction_history (run_id)")
    p.add_argument("--run-id", type=int, required=True, help="prediction_history run_id")
    p.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--model-version", action="append", required=True, help="model_version (repeatable)")
    p.add_argument("--horizon-days", type=int, default=60)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--price-source", choices=["db", "csv"], default="db")
    p.add_argument("--prices-csv", type=Path, default=Path("data/prices_daily_adjusted.csv"))
    p.add_argument("--prediction-source", choices=["research"], default="research")
    p.add_argument("--benchmark-code", type=str, default=None)
    p.add_argument("--benchmark-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/backtest_research"))
    return p.parse_args()


def to_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        start_date=to_date(args.start_date),
        end_date=to_date(args.end_date),
        model_versions=args.model_version,
        run_id=args.run_id,
        horizon_days=args.horizon_days,
        top_n=args.top_n,
        prediction_source=args.prediction_source,
        price_source=args.price_source,
        prices_csv=str(args.prices_csv),
        benchmark_code=args.benchmark_code,
        benchmark_csv=str(args.benchmark_csv) if args.benchmark_csv else None,
        output_dir=str(args.output_dir),
    )
    run_backtest(cfg)


if __name__ == "__main__":
    main()
