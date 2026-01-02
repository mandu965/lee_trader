# apply_weights_and_rank.py
"""
Apply weight sets to prediction_history (run_id) and export weighted scores.

This script keeps everything file-based (no DB mutation by default) to stay safe.
It produces per-weight CSVs with final_score_custom recomputed.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow running as a standalone script
import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "research"))

from config import BacktestConfig, ScoreWeights  # noqa: E402
from data_loader import load_predictions  # noqa: E402
from scoring import apply_score_weights  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Apply weight grid to predictions and export weighted scores")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--horizon-days", type=int, default=60)
    p.add_argument("--model-version", action="append", default=["v1"])
    p.add_argument("--weights-json", type=Path, required=True)
    p.add_argument("--score-field-out", type=str, default="final_score_custom")
    p.add_argument("--output-dir", type=Path, required=True)
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


def main():
    args = parse_args()
    weights_list = load_weights(args.weights_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for w in weights_list:
        sw = ScoreWeights(
            w_ret=w["w_ret"],
            w_prob=w["w_prob"],
            w_qual=w["w_qual"],
            w_tech=w["w_tech"],
            w_risk=w["w_risk"],
        )
        cfg = BacktestConfig(
            start_date=datetime.strptime(args.start_date, "%Y-%m-%d").date(),
            end_date=datetime.strptime(args.end_date, "%Y-%m-%d").date(),
            model_versions=args.model_version,
            run_id=args.run_id,
            horizon_days=args.horizon_days,
            top_n=20,
            prediction_source="research",
            price_source="db",
            output_dir=str(args.output_dir),
            weights=sw,
        )
        preds = load_predictions(cfg)
        preds_weighted = apply_score_weights(preds, cfg)
        out_path = args.output_dir / f"predictions_{w['name']}.csv"
        preds_weighted.to_csv(out_path, index=False)
        print(f"[weights] {w['name']} rows={len(preds_weighted)} -> {out_path}")


if __name__ == "__main__":
    main()
