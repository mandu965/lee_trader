"""
Verify that the snapshot top20 matches re-sorted ranking for a given date.

Usage:
  python scripts/verify_snapshot_top20.py 20251216
  python scripts/verify_snapshot_top20.py          # defaults to today (YYYYMMDD)

Outputs a short report of mismatches (if any).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "outputs" / "snapshots"


def main(date_str: str) -> int:
    snapshot_dir = SNAPSHOT_ROOT / date_str
    ranking_path = snapshot_dir / f"ranking_final_{date_str}.csv"
    top20_path = snapshot_dir / f"top20_{date_str}.csv"

    if not ranking_path.exists() or not top20_path.exists():
        print(f"[FAIL] Missing files: {ranking_path.exists()=}, {top20_path.exists()=}")
        return 1

    ranking = pd.read_csv(ranking_path)
    top_saved = pd.read_csv(top20_path)
    if "code" in ranking.columns:
        ranking["code"] = ranking["code"].astype(str).str.zfill(6)
    if "code" in top_saved.columns:
        top_saved["code"] = top_saved["code"].astype(str).str.zfill(6)

    score_col = "final_score" if "final_score" in ranking.columns else "score"
    recomputed = (
        ranking.sort_values(score_col, ascending=False)
        .head(len(top_saved))
        [["code", score_col]]
        .reset_index(drop=True)
    )

    # Compare codes
    mismatch_codes = []
    for i, (code_saved, code_re) in enumerate(zip(top_saved["code"], recomputed["code"])):
        if str(code_saved) != str(code_re):
            mismatch_codes.append((i, code_saved, code_re))

    # Compare scores (optional tolerance)
    mismatch_scores = []
    for i, (s_saved, s_re) in enumerate(
        zip(top_saved.get(score_col, top_saved.iloc[:, 0]), recomputed[score_col])
    ):
        if abs(float(s_saved) - float(s_re)) > 1e-9:
            mismatch_scores.append((i, s_saved, s_re))

    if not mismatch_codes and not mismatch_scores:
        print(f"[OK] Top{len(top_saved)} matches snapshot for {date_str} (sorted by {score_col}).")
        return 0

    if mismatch_codes:
        print(f"[WARN] Code mismatch ({len(mismatch_codes)} rows):")
        for i, s, r in mismatch_codes:
            print(f"  idx {i}: saved={s}, recomputed={r}")
    if mismatch_scores:
        print(f"[WARN] Score mismatch ({len(mismatch_scores)} rows):")
        for i, s, r in mismatch_scores:
            print(f"  idx {i}: saved={s}, recomputed={r}")
    return 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else datetime.today().strftime("%Y%m%d")
    raise SystemExit(main(arg))
