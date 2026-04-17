# run_grid_backtests.py
"""
Run backtests for multiple weight combinations (AB grid) using prediction_history (run_id).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure local imports work when run as script
import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "research"))

from config import BacktestConfig, ScoreWeights  # noqa: E402
from backtest_research import run_backtest  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run backtests over a weight grid")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--model-version", action="append", default=None)
    p.add_argument("--horizon-days", type=int, default=60)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--prediction-source", choices=["research", "csv", "db"], default="research")
    p.add_argument("--predictions-csv", type=Path, default=Path("data/predictions.csv"))
    p.add_argument("--price-source", choices=["db", "csv"], default="db")
    p.add_argument("--prices-csv", type=Path, default=Path("data/prices_daily_adjusted.csv"))
    p.add_argument("--weights-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    return p.parse_args()


def load_weights(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("weights-json must contain a list")
    cleaned = []
    for i, w in enumerate(data):
        name = w.get("name") or f"w{i}"
        cleaned.append(
            {
                "name": name,
                "w_ret": float(w.get("w_ret", w.get("ret", 0.0))),
                "w_prob": float(w.get("w_prob", w.get("prob", 0.0))),
                "w_qual": float(w.get("w_qual", w.get("qual", 0.0))),
                "w_tech": float(w.get("w_tech", w.get("tech", 0.0))),
                "w_risk": float(w.get("w_risk", w.get("risk", 1.0))),
            }
        )
    return cleaned


def to_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    args = parse_args()
    weights_list = load_weights(args.weights_json)
    model_versions = args.model_version or ["v1"]
    # de-duplicate while preserving order
    seen = set()
    mv_list = []
    for mv in model_versions:
        if mv not in seen:
            seen.add(mv)
            mv_list.append(mv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for w in weights_list:
        weight_name = w["name"]
        subdir = args.output_dir / weight_name
        subdir.mkdir(parents=True, exist_ok=True)

        cfg = BacktestConfig(
            start_date=to_date(args.start_date),
            end_date=to_date(args.end_date),
            model_versions=mv_list,
            run_id=args.run_id,
            horizon_days=args.horizon_days,
            top_n=args.top_n,
            prediction_source=args.prediction_source,
            predictions_csv=str(args.predictions_csv),
            price_source=args.price_source,
            prices_csv=str(args.prices_csv),
            output_dir=str(subdir),
            weights=ScoreWeights(
                w_ret=w["w_ret"],
                w_prob=w["w_prob"],
                w_qual=w["w_qual"],
                w_tech=w["w_tech"],
                w_risk=w["w_risk"],
            ),
        )

        df = run_backtest(cfg)
        df.insert(0, "weight_name", weight_name)
        df.insert(1, "w_ret", w["w_ret"])
        df.insert(2, "w_prob", w["w_prob"])
        df.insert(3, "w_qual", w["w_qual"])
        df.insert(4, "w_tech", w["w_tech"])
        df.insert(5, "w_risk", w["w_risk"])
        summaries.append(df)

    if summaries:
        grid_df = pd.concat(summaries, ignore_index=True)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(args.summary_out, index=False)
        print(f"[grid] wrote summary to {args.summary_out}")
    else:
        print("[grid] no weight runs executed")


if __name__ == "__main__":
    main()
