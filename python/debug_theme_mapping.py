from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RANKING_FINAL_CSV = BASE_DIR / "data" / "ranking_final.csv"


def main() -> int:
    if not RANKING_FINAL_CSV.exists():
        print(f"FILE_ERROR: Required file not found: {RANKING_FINAL_CSV}")
        return 1

    try:
        df = pd.read_csv(RANKING_FINAL_CSV, low_memory=False)
    except Exception as exc:
        print(f"PARSE_ERROR: Failed to read ranking_final.csv ({exc})")
        return 1

    total_stocks = len(df)
    dominant_theme = df.get("dominant_theme", pd.Series(["(none)"] * total_stocks)).fillna("(none)").astype(str)
    theme_confidence = pd.to_numeric(df.get("theme_confidence", pd.Series([0.0] * total_stocks)), errors="coerce").fillna(0.0)
    theme_score = pd.to_numeric(df.get("theme_score", pd.Series([0.0] * total_stocks)), errors="coerce").fillna(0.0)

    has_theme_mask = dominant_theme.str.strip().ne("(none)") & dominant_theme.str.strip().ne("")
    with_theme = int(has_theme_mask.sum())
    with_theme_ratio = (with_theme / total_stocks * 100.0) if total_stocks else 0.0

    rank_col = "rank_final" if "rank_final" in df.columns else ("rank_v2" if "rank_v2" in df.columns else None)
    if rank_col:
        top50 = df.copy()
        top50[rank_col] = pd.to_numeric(top50[rank_col], errors="coerce")
        top50 = top50.sort_values(rank_col, ascending=True).head(50)
    else:
        top50 = df.head(50).copy()

    top50_theme = top50.get("dominant_theme", pd.Series(["(none)"] * len(top50))).fillna("(none)").astype(str)
    top50_with_theme = int((top50_theme.str.strip().ne("(none)") & top50_theme.str.strip().ne("")).sum())

    print(f"total stocks: {total_stocks}")
    print(f"with theme: {with_theme} ({with_theme_ratio:.1f}%)")
    print(f"avg confidence: {theme_confidence.mean():.2f}")
    print(f"avg theme_score: {theme_score.mean():.2f}")
    print(f"top50 theme coverage: {top50_with_theme}/{len(top50)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
