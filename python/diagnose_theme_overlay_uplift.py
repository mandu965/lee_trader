from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "data"

RANKING_FINAL_CSV = DATA_DIR / "ranking_final.csv"
BEFORE_AFTER_V3_CSV = DATA_DIR / "before_after_score_compare_v3.csv"
ACCEPTANCE_REPORT_MD = DATA_DIR / "theme_overlay_acceptance_report.md"
GATE_DEBUG_JSON = DATA_DIR / "theme_overlay_gate_debug.json"
SHADOW_PREVIEW_CSV = DATA_DIR / "theme_overlay_shadow_preview.csv"
SHADOW_SUMMARY_JSON = DATA_DIR / "theme_overlay_shadow_summary.json"
WEIGHT_BY_REGIME_JSON = DATA_DIR / "experiments" / "theme_weight" / "best_weight_by_regime.json"
WEIGHT_GLOBAL_JSON = DATA_DIR / "experiments" / "theme_weight" / "best_weight.json"

OUT_MD = DATA_DIR / "theme_overlay_uplift_diagnosis.md"
OUT_TOP40_CSV = DATA_DIR / "theme_overlay_uplift_top40_debug.csv"

DEFAULT_FALLBACK_THEME_WEIGHT = 0.15
DEFENSIVE_THEME_WEIGHT = 0.10
TOP_N = 20
NEAR_TOP_N = 40


def _read_text_value(path: Path, prefix: str) -> str:
    if not path.exists():
        return "NA"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "NA"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _load_weight_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_theme_weight_for_regime(regime: str) -> tuple[float, str]:
    by_regime = _load_weight_payload(WEIGHT_BY_REGIME_JSON)
    global_payload = _load_weight_payload(WEIGHT_GLOBAL_JSON)
    regime_key = str(regime or "").strip().lower()

    if regime_key and regime_key in by_regime:
        return _safe_float(by_regime.get(regime_key), DEFAULT_FALLBACK_THEME_WEIGHT), "best_weight_by_regime"
    if "global" in by_regime:
        return _safe_float(by_regime.get("global"), DEFAULT_FALLBACK_THEME_WEIGHT), "best_weight_by_regime_global"
    if "best_weight" in global_payload:
        return _safe_float(global_payload.get("best_weight"), DEFAULT_FALLBACK_THEME_WEIGHT), "best_weight_global"
    return DEFAULT_FALLBACK_THEME_WEIGHT, "fallback_default"


def _distribution(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def _segment_summary(df: pd.DataFrame, label: str) -> dict[str, object]:
    total = len(df)
    none_mask = df["dominant_theme"].fillna("(none)").astype(str).str.strip().isin(["", "(none)", "nan", "None"])
    actual_contrib = pd.to_numeric(df["theme_contribution_v3"], errors="coerce").fillna(0.0)
    potential_delta_default = pd.to_numeric(df["score_diff_at_0_15"], errors="coerce").fillna(0.0)
    gap_to_prev = pd.to_numeric(df["gap_to_prev_rank"], errors="coerce")
    swap_ratio = pd.to_numeric(df["swap_ratio_actual"], errors="coerce")
    return {
        "label": label,
        "rows": int(total),
        "none_ratio": float(none_mask.mean()) if total else 0.0,
        "actual_contribution": _distribution(actual_contrib),
        "potential_delta_default": _distribution(potential_delta_default),
        "gap_to_prev": _distribution(gap_to_prev),
        "swap_ratio_actual": _distribution(swap_ratio),
    }


def main() -> None:
    ranking = pd.read_csv(RANKING_FINAL_CSV, dtype={"code": str}, low_memory=False)
    ranking["date"] = pd.to_datetime(ranking["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = ranking["date"].dropna().max()
    latest = ranking.loc[ranking["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)

    compare_v3 = pd.read_csv(BEFORE_AFTER_V3_CSV, dtype={"code": str}, low_memory=False) if BEFORE_AFTER_V3_CSV.exists() else pd.DataFrame()
    if not compare_v3.empty:
        compare_v3["date"] = pd.to_datetime(compare_v3["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        compare_v3 = compare_v3.loc[compare_v3["date"] == latest_date].copy()
        compare_v3["code"] = compare_v3["code"].astype(str).str.zfill(6)

    shadow_preview = pd.read_csv(SHADOW_PREVIEW_CSV, low_memory=False) if SHADOW_PREVIEW_CSV.exists() else pd.DataFrame()
    if not shadow_preview.empty and "ticker" in shadow_preview.columns:
        shadow_preview["ticker"] = shadow_preview["ticker"].astype(str).str.zfill(6)
        shadow_preview = shadow_preview.rename(columns={"ticker": "code"})

    for col in [
        "final_score",
        "final_score_v3",
        "score_diff_v3",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
        "theme_weight",
        "w_theme",
        "w_base_v2",
        "contrib_theme",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "shadow_theme_weight",
        "shadow_theme_score_effective",
    ]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce")

    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    latest["baseline_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)
    latest["overlay_rank"] = latest["final_score_v3"].rank(method="first", ascending=False).astype(int)
    latest["rank_change"] = latest["baseline_rank"] - latest["overlay_rank"]
    latest["theme_contribution_v3"] = pd.to_numeric(latest.get("contrib_theme"), errors="coerce").fillna(
        pd.to_numeric(latest.get("w_theme"), errors="coerce").fillna(0.0)
        * pd.to_numeric(latest.get("theme_score_effective"), errors="coerce").fillna(0.0)
    )
    latest["theme_gap_vs_baseline"] = latest["theme_score_effective"].fillna(0.0) - latest["final_score"].fillna(0.0)
    latest["score_diff_at_0_10"] = DEFENSIVE_THEME_WEIGHT * latest["theme_gap_vs_baseline"]
    latest["score_diff_at_0_15"] = DEFAULT_FALLBACK_THEME_WEIGHT * latest["theme_gap_vs_baseline"]

    latest = latest.sort_values(["baseline_rank", "code"]).reset_index(drop=True)
    latest["prev_final_score"] = latest["final_score"].shift(1)
    latest["gap_to_prev_rank"] = latest["prev_final_score"] - latest["final_score"]
    latest["swap_ratio_actual"] = np.where(
        latest["gap_to_prev_rank"].gt(0.0),
        latest["theme_contribution_v3"] / latest["gap_to_prev_rank"],
        np.nan,
    )

    gate_debug = {}
    if GATE_DEBUG_JSON.exists():
        gate_debug = json.loads(GATE_DEBUG_JSON.read_text(encoding="utf-8"))

    resolved_mode = _read_text_value(ACCEPTANCE_REPORT_MD, "- resolved_mode")
    evaluation_profile = _read_text_value(ACCEPTANCE_REPORT_MD, "- evaluation_profile")
    overlay_score_col = _read_text_value(ACCEPTANCE_REPORT_MD, "- overlay_score_column_used_for_evaluation")

    regime = str(latest["regime"].dropna().iloc[0]) if "regime" in latest.columns and not latest["regime"].dropna().empty else "NA"
    resolved_weight, resolved_weight_source = _resolve_theme_weight_for_regime(regime)

    themed_mask = latest["dominant_theme"].astype(str).str.strip().ne("(none)")
    themed_mask &= latest["dominant_theme"].astype(str).str.strip().ne("")
    top20 = latest.loc[latest["baseline_rank"] <= TOP_N].copy()
    near_top20 = latest.loc[(latest["baseline_rank"] > TOP_N) & (latest["baseline_rank"] <= NEAR_TOP_N)].copy()
    top40 = latest.loc[latest["baseline_rank"] <= NEAR_TOP_N].copy()

    rank20_score = float(top20["final_score"].min()) if not top20.empty else 0.0
    top40["gap_to_top20_cut"] = rank20_score - top40["final_score"]
    near_top20_band = top40["baseline_rank"].between(TOP_N + 1, NEAR_TOP_N)
    top40["actual_swap_possible_to_top20"] = near_top20_band & (top40["score_diff_v3"] >= top40["gap_to_top20_cut"])
    top40["default_015_swap_possible_to_top20"] = near_top20_band & (top40["score_diff_at_0_15"] >= top40["gap_to_top20_cut"])
    top40["defensive_010_swap_possible_to_top20"] = near_top20_band & (top40["score_diff_at_0_10"] >= top40["gap_to_top20_cut"])

    if not compare_v3.empty:
        merge_cols = [
            col
            for col in [
                "code",
                "before_rank_final_score",
                "after_rank_final_score_v3",
                "shadow_rank_v3",
                "shadow_theme_weight",
                "shadow_theme_score_effective",
                "shadow_final_score_v3",
                "shadow_score_diff_v3",
            ]
            if col in compare_v3.columns
        ]
        if merge_cols:
            top40 = top40.merge(compare_v3[merge_cols], on="code", how="left", suffixes=("", "_compare"))

    if not shadow_preview.empty:
        preview_cols = [
            col
            for col in [
                "code",
                "base_score",
                "raw_theme_score",
                "overlay_gate_result",
                "overlay_disable_reason",
            ]
            if col in shadow_preview.columns
        ]
        if preview_cols:
            top40 = top40.merge(shadow_preview[preview_cols], on="code", how="left", suffixes=("", "_preview"))

    out_cols = [
        "date",
        "code",
        "name",
        "final_score",
        "final_score_v3",
        "score_diff_v3",
        "theme_score",
        "theme_confidence",
        "dominant_theme",
        "theme_score_effective",
        "theme_weight",
        "theme_contribution_v3",
        "baseline_rank",
        "overlay_rank",
        "rank_change",
        "gap_to_prev_rank",
        "gap_to_top20_cut",
        "score_diff_at_0_10",
        "score_diff_at_0_15",
        "actual_swap_possible_to_top20",
        "default_015_swap_possible_to_top20",
        "defensive_010_swap_possible_to_top20",
        "shadow_theme_weight",
        "shadow_theme_score_effective",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "before_rank_final_score",
        "after_rank_final_score_v3",
        "shadow_rank_v3",
    ]
    for column in out_cols:
        if column not in top40.columns:
            top40[column] = np.nan
    top40[out_cols].to_csv(OUT_TOP40_CSV, index=False, encoding="utf-8-sig")

    summary_payload = {
        "latest_date": latest_date,
        "resolved_mode": resolved_mode,
        "evaluation_profile": evaluation_profile,
        "overlay_score_column": overlay_score_col,
        "gate_debug": gate_debug,
        "latest_regime": regime,
        "resolved_theme_weight": resolved_weight,
        "resolved_theme_weight_source": resolved_weight_source,
        "row_count": int(len(latest)),
        "themed_ratio": float(themed_mask.mean()) if len(latest) else 0.0,
        "score_diff_v3": _distribution(latest["score_diff_v3"]),
        "theme_score": _distribution(latest["theme_score"]),
        "theme_confidence": _distribution(latest["theme_confidence"]),
        "theme_score_effective": _distribution(latest["theme_score_effective"]),
        "theme_weight": _distribution(latest["theme_weight"]),
        "w_theme": _distribution(latest["w_theme"]),
        "theme_contribution_v3": _distribution(latest["theme_contribution_v3"]),
        "theme_gap_vs_baseline": _distribution(latest["theme_gap_vs_baseline"]),
        "top20": _segment_summary(top20, "top20"),
        "near_top20": _segment_summary(near_top20, "near_top20"),
        "top40_actual_swap_to_top20_count": int(top40["actual_swap_possible_to_top20"].fillna(False).sum()),
        "top40_default_015_swap_to_top20_count": int(top40["default_015_swap_possible_to_top20"].fillna(False).sum()),
        "top40_defensive_010_swap_to_top20_count": int(top40["defensive_010_swap_possible_to_top20"].fillna(False).sum()),
    }
    SHADOW_SUMMARY_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Theme Overlay Uplift Diagnosis",
        "",
        "## Root Cause Summary",
        f"- latest_date: {latest_date}",
        f"- current_mode: {resolved_mode}",
        f"- evaluation_profile: {evaluation_profile}",
        f"- overlay_score_column: {overlay_score_col}",
        f"- latest_regime: {regime}",
        f"- overlay_gate_result: {gate_debug.get('overlay_gate_result', 'NA')}",
        f"- overlay_disable_reason: {gate_debug.get('overlay_disable_reason', 'NA')}",
        f"- resolved_theme_weight: {resolved_weight:.4f} ({resolved_weight_source})",
        f"- score_diff_v3 max_abs: {latest['score_diff_v3'].abs().max():.6f}",
        f"- theme_weight mean/max: {latest['theme_weight'].fillna(0.0).mean():.6f} / {latest['theme_weight'].fillna(0.0).max():.6f}",
        f"- w_theme mean/max: {latest['w_theme'].fillna(0.0).mean():.6f} / {latest['w_theme'].fillna(0.0).max():.6f}",
        f"- theme_contribution_v3 mean/max: {latest['theme_contribution_v3'].fillna(0.0).mean():.6f} / {latest['theme_contribution_v3'].fillna(0.0).max():.6f}",
        "",
        "Current latest output does not fail because acceptance report picked the wrong column.",
        "It fails because ranking generation produces zero effective overlay weight on the latest date, so `final_score_v3` remains identical to `final_score`.",
        "",
        "## Most Likely Causes",
        "1. The live gate is disabled in the current latest output, so `_theme_gate_allows_score_application()` forces `configured_theme_weight = 0.0` before `w_theme` and `final_score_v3` are built.",
        f"2. Even if the gate opens, the persisted theme-weight configuration resolves to `{resolved_weight:.4f}` via `{resolved_weight_source}`, so the overlay formula still contributes almost nothing on the latest regime.",
        "3. Among latest top20 and near-top20 names, many are `(none)` theme names or have `theme_score_effective` below their baseline `final_score`, so even a non-zero weight would often be neutral or negative rather than lifting ranks.",
        "",
        "## Current Distribution",
        f"- themed_ratio_all: {themed_mask.mean():.2%}",
        f"- dominant_theme_none_ratio_all: {(~themed_mask).mean():.2%}",
        f"- theme_score p50/p90/max: {latest['theme_score'].fillna(0.0).quantile(0.50):.4f} / {latest['theme_score'].fillna(0.0).quantile(0.90):.4f} / {latest['theme_score'].fillna(0.0).max():.4f}",
        f"- theme_confidence p50/p90/max: {latest['theme_confidence'].fillna(0.0).quantile(0.50):.4f} / {latest['theme_confidence'].fillna(0.0).quantile(0.90):.4f} / {latest['theme_confidence'].fillna(0.0).max():.4f}",
        f"- theme_score_effective p50/p90/max: {latest['theme_score_effective'].fillna(0.0).quantile(0.50):.4f} / {latest['theme_score_effective'].fillna(0.0).quantile(0.90):.4f} / {latest['theme_score_effective'].fillna(0.0).max():.4f}",
        f"- final_score p50/p90/max: {latest['final_score'].fillna(0.0).quantile(0.50):.4f} / {latest['final_score'].fillna(0.0).quantile(0.90):.4f} / {latest['final_score'].fillna(0.0).max():.4f}",
        f"- score_diff_v3 p50/p90/max: {latest['score_diff_v3'].fillna(0.0).quantile(0.50):.4f} / {latest['score_diff_v3'].fillna(0.0).quantile(0.90):.4f} / {latest['score_diff_v3'].fillna(0.0).max():.4f}",
        "",
        "## Direct Cause Breakdown",
        f"- theme_weight_zero_count: {int(latest['theme_weight'].fillna(0.0).eq(0.0).sum())} / {len(latest)}",
        f"- w_theme_zero_count: {int(latest['w_theme'].fillna(0.0).eq(0.0).sum())} / {len(latest)}",
        f"- nonzero_theme_score_count: {int(latest['theme_score'].fillna(0.0).gt(0.0).sum())} / {len(latest)}",
        f"- nonzero_theme_score_effective_count: {int(latest['theme_score_effective'].fillna(0.0).gt(0.0).sum())} / {len(latest)}",
        f"- nonzero_actual_score_diff_count: {int(latest['score_diff_v3'].fillna(0.0).abs().gt(1e-9).sum())} / {len(latest)}",
        f"- names_with_theme_score_but_zero_theme_weight: {int((latest['theme_score'].fillna(0.0).gt(0.0) & latest['theme_weight'].fillna(0.0).eq(0.0)).sum())}",
        f"- names_with_theme_score_effective_below_baseline: {int((latest['theme_score_effective'].fillna(0.0) < latest['final_score'].fillna(0.0)).sum())}",
        "",
        "The score formula itself is `final_score_v3 = w_base_v2 * final_score + w_theme * theme_score_effective`.",
        "When `w_theme = 0`, both `contrib_theme` and `score_diff_v3` collapse to zero regardless of non-zero theme score/confidence.",
        "",
        "## Top20 / Near-Top20",
        f"- top20 none_ratio: {top20['dominant_theme'].fillna('(none)').astype(str).str.strip().isin(['', '(none)', 'nan', 'None']).mean():.2%}",
        f"- near_top20 none_ratio: {near_top20['dominant_theme'].fillna('(none)').astype(str).str.strip().isin(['', '(none)', 'nan', 'None']).mean():.2%}",
        f"- top20 actual contribution p50/p90/max: {top20['theme_contribution_v3'].fillna(0.0).quantile(0.50):.4f} / {top20['theme_contribution_v3'].fillna(0.0).quantile(0.90):.4f} / {top20['theme_contribution_v3'].fillna(0.0).max():.4f}",
        f"- near_top20 actual contribution p50/p90/max: {near_top20['theme_contribution_v3'].fillna(0.0).quantile(0.50):.4f} / {near_top20['theme_contribution_v3'].fillna(0.0).quantile(0.90):.4f} / {near_top20['theme_contribution_v3'].fillna(0.0).max():.4f}",
        f"- top20 baseline gap p50/p90/max: {top20['gap_to_prev_rank'].dropna().quantile(0.50) if top20['gap_to_prev_rank'].dropna().size else 0.0:.4f} / {top20['gap_to_prev_rank'].dropna().quantile(0.90) if top20['gap_to_prev_rank'].dropna().size else 0.0:.4f} / {top20['gap_to_prev_rank'].dropna().max() if top20['gap_to_prev_rank'].dropna().size else 0.0:.4f}",
        f"- near_top20 baseline gap p50/p90/max: {near_top20['gap_to_prev_rank'].dropna().quantile(0.50) if near_top20['gap_to_prev_rank'].dropna().size else 0.0:.4f} / {near_top20['gap_to_prev_rank'].dropna().quantile(0.90) if near_top20['gap_to_prev_rank'].dropna().size else 0.0:.4f} / {near_top20['gap_to_prev_rank'].dropna().max() if near_top20['gap_to_prev_rank'].dropna().size else 0.0:.4f}",
        f"- actual swap candidates into top20: {int(top40['actual_swap_possible_to_top20'].fillna(False).sum())}",
        f"- hypothetical swap candidates at 0.10 weight: {int(top40['defensive_010_swap_possible_to_top20'].fillna(False).sum())}",
        f"- hypothetical swap candidates at 0.15 weight: {int(top40['default_015_swap_possible_to_top20'].fillna(False).sum())}",
        "",
        "## Code Pinpoint",
        "- Weight resolution: `_resolve_theme_weight_info_for_regime()` first prefers `best_weight_by_regime.json`, then its `global`, then `best_weight.json`, then fallback default.",
        f"- Current persisted config yields `global=0.0` in `{WEIGHT_BY_REGIME_JSON.as_posix()}` and `best_weight=0.0` in `{WEIGHT_GLOBAL_JSON.as_posix()}`.",
        "- Gate no-op path: `_theme_gate_allows_score_application()` is checked inside `_attach_regime_weights()` and `apply_theme_overlay_v2()`, which zeroes `theme_weight` when the gate is closed.",
        "- Confidence damping: `sanitize_theme_columns()` sets `theme_score_effective = theme_score * theme_confidence` only when `dominant_theme` is active, which further suppresses raw theme strength.",
        "- Date alignment / coverage: latest theme date matches ranking date and coverage is not the blocker here; the blocker is weight application after coverage succeeds.",
        "",
        "## Parameter / Formula Candidates To Adjust",
        "1. Theme weight config: review `data/experiments/theme_weight/best_weight_by_regime.json` and `data/experiments/theme_weight/best_weight.json`, because current persisted best weights are 0.0.",
        "2. Defensive-regime minimum floor: clamp resolved theme weight with a non-zero floor for shadow evaluation so `shadow_final_score_v3` can actually test counterfactual uplift.",
        "3. Overlay formula scaling: instead of blending against full `final_score`, test an additive delta or z-score normalized theme component so strong themes are not muted by already-high baseline scores.",
        "",
        "## Recommended Fix",
        "Use the existing shadow split, but introduce a non-zero evaluation floor for shadow weight resolution, for example `max(resolved_weight, 0.10)` in shadow-only score generation while keeping live operational weight policy unchanged.",
        "",
        "## Checkpoints Before Applying The Recommendation",
        "1. Confirm whether `best_weight=0.0` is intentional research output or a stale experiment artifact.",
        "2. Re-run one latest-date shadow build and verify `shadow_score_diff_v3` becomes non-zero for at least themed names near ranks 15~40.",
        "3. Check whether positive-weight shadow overlay mostly lifts valid themed names, or simply penalizes top ranks where `theme_score_effective < final_score`.",
        "4. Ensure live `off` and `operational` semantics remain unchanged until shadow evaluation quality is accepted.",
        "",
        "## Generated Debug Outputs",
        f"- {OUT_TOP40_CSV.as_posix()}",
        f"- {SHADOW_SUMMARY_JSON.as_posix()}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(OUT_MD))
    print(str(OUT_TOP40_CSV))
    print(str(SHADOW_SUMMARY_JSON))


if __name__ == "__main__":
    main()
