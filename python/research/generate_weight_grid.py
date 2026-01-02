# generate_weight_grid.py
"""
Create a small deterministic grid of score weights around a base allocation.

Output: JSON list of weight dictionaries:
  [
    {"name": "w0_base", "w_ret": 0.45, "w_prob": 0.35, "w_qual": 0.10, "w_tech": 0.10, "w_risk": 1.00},
    ...
  ]
"""
import argparse
import json
import ast
from pathlib import Path
from typing import Dict, List


def normalize(w: Dict[str, float]) -> Dict[str, float]:
    """Normalize return/prob/qual/tech to sum to 1.0; keep risk as-is."""
    parts = ["w_ret", "w_prob", "w_qual", "w_tech"]
    total = sum(max(w.get(k, 0.0), 0.0) for k in parts)
    if total <= 0:
        raise ValueError("Non-risk weights sum to 0; cannot normalize")
    for k in parts:
        w[k] = max(w.get(k, 0.0), 0.0) / total
    return w


def to_weight_dict(base: Dict[str, float]) -> Dict[str, float]:
    """Map possible short keys to ScoreWeights keys."""
    mapped = {}
    for k, v in base.items():
        if k in {"ret", "w_ret"}:
            mapped["w_ret"] = float(v)
        elif k in {"prob", "w_prob"}:
            mapped["w_prob"] = float(v)
        elif k in {"qual", "w_qual"}:
            mapped["w_qual"] = float(v)
        elif k in {"tech", "w_tech"}:
            mapped["w_tech"] = float(v)
        elif k in {"risk", "w_risk"}:
            mapped["w_risk"] = float(v)
    return mapped


def make_grid(base: Dict[str, float], n: int) -> List[Dict[str, float]]:
    """
    Build a deterministic set of weight variations around base.
    """
    b = to_weight_dict(base)
    b.setdefault("w_risk", 1.0)
    b = normalize(b)

    variations = [
        ("w0_base", {}),
        ("w1_ret_up", {"w_ret": b["w_ret"] + 0.05, "w_prob": b["w_prob"] - 0.05}),
        ("w2_prob_up", {"w_prob": b["w_prob"] + 0.05, "w_ret": b["w_ret"] - 0.05}),
        ("w3_qual_up", {"w_qual": b["w_qual"] + 0.05, "w_ret": b["w_ret"] - 0.05}),
        ("w4_tech_up", {"w_tech": b["w_tech"] + 0.05, "w_prob": b["w_prob"] - 0.05}),
        ("w5_ret_high", {"w_ret": b["w_ret"] + 0.10, "w_prob": b["w_prob"] - 0.10}),
        ("w6_prob_high", {"w_prob": b["w_prob"] + 0.10, "w_ret": b["w_ret"] - 0.10}),
        ("w7_qual_high", {"w_qual": b["w_qual"] + 0.10, "w_prob": b["w_prob"] - 0.05, "w_ret": b["w_ret"] - 0.05}),
        ("w8_tech_high", {"w_tech": b["w_tech"] + 0.10, "w_prob": b["w_prob"] - 0.05, "w_ret": b["w_ret"] - 0.05}),
        ("w9_risk_low", {"w_risk": b["w_risk"] * 0.8}),
        ("w10_risk_high", {"w_risk": b["w_risk"] * 1.2}),
        ("w11_balanced_plus", {"w_ret": b["w_ret"] + 0.05, "w_prob": b["w_prob"] + 0.05, "w_qual": b["w_qual"] - 0.05, "w_tech": b["w_tech"] - 0.05}),
    ]

    grid: List[Dict[str, float]] = []
    for name, delta in variations:
        w = {**b, **delta}
        risk = w.get("w_risk", b["w_risk"])
        w = normalize(w)
        w["name"] = name
        w["w_risk"] = float(risk)
        grid.append(w)
        if len(grid) >= n:
            break
    return grid


def parse_args():
    p = argparse.ArgumentParser(description="Generate a grid of weight combinations")
    p.add_argument("--base", type=str, required=False, help='Base weights JSON string, e.g. {"ret":0.45,"prob":0.35,"qual":0.1,"tech":0.1,"risk":1.0}')
    p.add_argument("--base-file", type=Path, required=False, help="Optional JSON file containing base weights object")
    p.add_argument("--n", type=int, default=12, help="Number of grid points to emit")
    p.add_argument("--output", type=Path, required=True, help="Output JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    if args.base_file:
        base = json.loads(Path(args.base_file).read_text(encoding="utf-8"))
    elif args.base:
        try:
            base = json.loads(args.base)
        except Exception:
            base = ast.literal_eval(args.base)
    else:
        raise ValueError("Either --base or --base-file must be provided")
    grid = make_grid(base, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    print(f"[grid] wrote {len(grid)} weights to {args.output}")


if __name__ == "__main__":
    main()
