from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

RANKING_FINAL_CSV = DATA_DIR / "ranking_final.csv"
RANKING_FINAL_V3_CSV = DATA_DIR / "ranking_final_v3.csv"
STOCK_THEME_DAILY_CSV = OUTPUT_DIR / "stock_theme_daily.csv"
MODE_RESOLUTION_MD = DATA_DIR / "theme_overlay_mode_resolution.md"
MODE_NOTE_MD = DATA_DIR / "theme_overlay_acceptance_mode_note.md"

COMPARE_V3_CANDIDATES = [
    DATA_DIR / "before_after_score_compare_v3.csv",
    OUTPUT_DIR / "before_after_score_compare_v3.csv",
]

TOP20_CHURN_CSV = DATA_DIR / "top20_churn_analysis.csv"
NO_THEME_RETENTION_CSV = DATA_DIR / "no_theme_retention.csv"
THEME_CONCENTRATION_CSV = DATA_DIR / "theme_concentration.csv"
THEME_LIFT_ANALYSIS_CSV = DATA_DIR / "theme_lift_analysis.csv"
NEW_ENTRY_QUALITY_CSV = DATA_DIR / "new_entry_quality_report.csv"
ACCEPTANCE_REPORT_MD = DATA_DIR / "theme_overlay_acceptance_report.md"

TOP_N = 20
MINIMAL_SCORE_DELTA = 0.05
LARGE_NEGATIVE_SCORE_DELTA = -1.0


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _status_from_thresholds(value: float, *, pass_cond: bool, warn_cond: bool) -> str:
    if pass_cond:
        return "PASS"
    if warn_cond:
        return "WARN"
    return "FAIL"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _theme_is_none(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("(none)").astype(str).str.strip()
    return cleaned.isin(["", "(none)", "nan", "None"])


def _extract_mode_from_resolution_md() -> str:
    if not MODE_RESOLUTION_MD.exists():
        return "off"
    for line in MODE_RESOLUTION_MD.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("- resolved_execution_mode:"):
            return line.split(":", 1)[1].strip() or "off"
    return "off"


def _merge_latest_columns(base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    merge_cols = [column for column in columns if column in extra.columns]
    required = {"date", "code"}
    if not required.issubset(extra.columns) or not merge_cols:
        return base
    return base.drop(columns=merge_cols, errors="ignore").merge(
        extra[["date", "code", *merge_cols]],
        on=["date", "code"],
        how="left",
    )


def _load_shadow_compare_overlay_scores(latest_date: str) -> tuple[pd.DataFrame | None, str]:
    for path in COMPARE_V3_CANDIDATES:
        if not path.exists():
            continue
        compare = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if compare.empty or "date" not in compare.columns:
            continue
        compare["date"] = pd.to_datetime(compare["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        compare = compare.loc[compare["date"] == latest_date].copy()
        if compare.empty:
            continue
        compare["code"] = compare["code"].astype(str).str.zfill(6)
        shadow_cols = [
            column
            for column in [
                "theme_overlay_formula",
                "theme_delta_vs_base",
                "theme_positive_part",
                "theme_negative_part",
                "theme_uplift_applied",
                "theme_penalty_applied",
                "shadow_theme_weight_raw",
                "shadow_theme_weight_effective",
                "shadow_floor_applied",
                "shadow_theme_weight",
                "shadow_base_weight",
                "shadow_theme_score_effective",
                "shadow_final_score_v3",
                "shadow_score_diff_v3",
                "shadow_rank_v3",
            ]
            if column in compare.columns
        ]
        if shadow_cols:
            return compare[["date", "code", *shadow_cols]].copy(), str(path)
    return None, "NA"


def _resolve_overlay_score_column(latest: pd.DataFrame) -> dict[str, object]:
    resolved_mode = _extract_mode_from_resolution_md()
    if resolved_mode == "operational":
        overlay_score_column = "final_score_v3"
        evaluation_profile = "operational"
        no_op_expected = False
    elif resolved_mode == "shadow":
        overlay_score_column = "shadow_final_score_v3" if "shadow_final_score_v3" in latest.columns else "final_score_v3"
        evaluation_profile = "shadow_counterfactual"
        no_op_expected = False
    else:
        resolved_mode = "off"
        overlay_score_column = "final_score"
        evaluation_profile = "off_no_op"
        no_op_expected = True

    if overlay_score_column not in latest.columns:
        overlay_score_column = "final_score"
        no_op_expected = True
        if resolved_mode == "shadow":
            evaluation_profile = "shadow_counterfactual_fallback_to_baseline"

    return {
        "resolved_mode": resolved_mode,
        "overlay_score_column": overlay_score_column,
        "evaluation_profile": evaluation_profile,
        "no_op_expected": no_op_expected,
    }


def load_latest_ranking() -> tuple[pd.DataFrame, str, dict[str, object]]:
    base = pd.read_csv(RANKING_FINAL_CSV, dtype={"code": str}, low_memory=False)
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = base["date"].dropna().max()
    latest = base.loc[base["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)

    v3_source = f"{RANKING_FINAL_CSV} (embedded final_score_v3)"
    if RANKING_FINAL_V3_CSV.exists():
        v3 = pd.read_csv(RANKING_FINAL_V3_CSV, dtype={"code": str}, low_memory=False)
        v3["date"] = pd.to_datetime(v3["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        v3 = v3.loc[v3["date"] == latest_date].copy()
        v3["code"] = v3["code"].astype(str).str.zfill(6)
        latest = _merge_latest_columns(latest, v3, ["final_score_v3"])
        v3_source = str(RANKING_FINAL_V3_CSV)

    shadow_source = f"{RANKING_FINAL_CSV} (embedded shadow columns)"
    if "shadow_final_score_v3" not in latest.columns:
        shadow_compare, compare_source = _load_shadow_compare_overlay_scores(latest_date)
        if shadow_compare is not None:
            latest = _merge_latest_columns(
                latest,
                shadow_compare,
                [
                    "theme_overlay_formula",
                    "theme_delta_vs_base",
                    "theme_positive_part",
                    "theme_negative_part",
                    "theme_uplift_applied",
                    "theme_penalty_applied",
                    "shadow_theme_weight_raw",
                    "shadow_theme_weight_effective",
                    "shadow_floor_applied",
                    "shadow_theme_weight",
                    "shadow_base_weight",
                    "shadow_theme_score_effective",
                    "shadow_final_score_v3",
                    "shadow_score_diff_v3",
                    "shadow_rank_v3",
                ],
            )
            shadow_source = compare_source
        else:
            shadow_source = "NA"

    latest["final_score"] = _to_num(latest.get("final_score")).fillna(0.0)
    latest["final_score_v3"] = _to_num(latest.get("final_score_v3")).fillna(latest["final_score"])
    latest["shadow_theme_weight"] = _to_num(latest.get("shadow_theme_weight"))
    latest["shadow_theme_weight_raw"] = _to_num(latest.get("shadow_theme_weight_raw"))
    latest["shadow_theme_weight_effective"] = _to_num(latest.get("shadow_theme_weight_effective"))
    latest["shadow_base_weight"] = _to_num(latest.get("shadow_base_weight"))
    latest["shadow_theme_score_effective"] = _to_num(
        latest.get("shadow_theme_score_effective", latest.get("theme_score_effective"))
    ).fillna(0.0)
    latest["shadow_final_score_v3"] = _to_num(latest.get("shadow_final_score_v3"))
    latest["shadow_score_diff_v3"] = _to_num(latest.get("shadow_score_diff_v3"))
    latest["shadow_rank_v3"] = _to_num(latest.get("shadow_rank_v3"))
    latest["theme_delta_vs_base"] = _to_num(latest.get("theme_delta_vs_base"))
    latest["theme_positive_part"] = _to_num(latest.get("theme_positive_part"))
    latest["theme_negative_part"] = _to_num(latest.get("theme_negative_part"))
    latest["theme_overlay_formula"] = latest.get("theme_overlay_formula", pd.Series(pd.NA, index=latest.index)).astype("string")
    latest["theme_uplift_applied"] = latest.get("theme_uplift_applied", False).fillna(False).astype(bool)
    latest["theme_penalty_applied"] = latest.get("theme_penalty_applied", False).fillna(False).astype(bool)
    latest["shadow_floor_applied"] = latest.get("shadow_floor_applied", False).fillna(False).astype(bool)
    latest["theme_score_effective"] = _to_num(latest.get("theme_score_effective")).fillna(0.0)
    latest["theme_confidence"] = _to_num(latest.get("theme_confidence")).fillna(0.0).clip(lower=0.0, upper=1.0)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    latest["explain_text"] = latest.get("explain_text", "").fillna("").astype(str)
    latest["baseline_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)

    overlay_meta = _resolve_overlay_score_column(latest)
    overlay_score_column = str(overlay_meta["overlay_score_column"])
    latest["overlay_score_eval"] = _to_num(latest.get(overlay_score_column)).fillna(latest["final_score"])
    latest["overlay_rank"] = latest["overlay_score_eval"].rank(method="first", ascending=False).astype(int)

    if overlay_meta["resolved_mode"] == "off":
        latest["score_delta_v3"] = 0.0
    else:
        latest["score_delta_v3"] = latest["overlay_score_eval"] - latest["final_score"]
    latest["rank_change_shadow"] = latest["baseline_rank"] - latest["overlay_rank"]

    latest["is_no_theme"] = _theme_is_none(latest["dominant_theme"])
    latest["has_theme_explain"] = latest.apply(
        lambda row: (str(row["dominant_theme"]).strip() not in {"", "(none)"})
        and (str(row["dominant_theme"]).strip() in str(row["explain_text"]))
        or ("theme=" in str(row["explain_text"]).lower()),
        axis=1,
    )
    overlay_meta["v3_source"] = v3_source
    overlay_meta["shadow_source"] = shadow_source
    return latest, v3_source, overlay_meta


def build_shadow_effect_diagnostics(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    diagnostics = latest.copy()
    diagnostics["score_delta_v3"] = _to_num(diagnostics.get("score_delta_v3")).fillna(0.0)
    diagnostics["rank_change_shadow"] = _to_num(diagnostics.get("rank_change_shadow")).fillna(0.0)
    diagnostics["direct_uplift"] = diagnostics["score_delta_v3"].gt(0.0) & diagnostics["rank_change_shadow"].gt(0.0)
    diagnostics["indirect_rank_gain"] = diagnostics["rank_change_shadow"].gt(0.0) & diagnostics["score_delta_v3"].le(MINIMAL_SCORE_DELTA)
    diagnostics["large_negative_displacement"] = diagnostics["rank_change_shadow"].le(-5) | diagnostics["score_delta_v3"].le(LARGE_NEGATIVE_SCORE_DELTA)
    diagnostics["direct_uplift_top20"] = diagnostics["direct_uplift"] & diagnostics["overlay_rank"].le(TOP_N)
    cols = [
        "date",
        "code",
        "name",
        "baseline_rank",
        "overlay_rank",
        "rank_change_shadow",
        "final_score",
        "overlay_score_eval",
        "score_delta_v3",
        "dominant_theme",
        "theme_confidence",
        "theme_score_effective",
        "theme_overlay_formula",
        "theme_uplift_applied",
        "theme_penalty_applied",
        "shadow_floor_applied",
        "direct_uplift",
        "indirect_rank_gain",
        "large_negative_displacement",
    ]
    out = diagnostics.loc[:, [column for column in cols if column in diagnostics.columns]].sort_values(
        ["overlay_rank", "baseline_rank", "code"]
    ).reset_index(drop=True)
    return out, {
        "direct_uplift_count": int(diagnostics["direct_uplift"].sum()),
        "direct_uplift_top20_count": int(diagnostics["direct_uplift_top20"].sum()),
        "indirect_rank_gain_count": int(diagnostics["indirect_rank_gain"].sum()),
        "large_negative_displacement_count": int(diagnostics["large_negative_displacement"].sum()),
        "minimal_score_delta_threshold": MINIMAL_SCORE_DELTA,
        "large_negative_score_delta_threshold": LARGE_NEGATIVE_SCORE_DELTA,
    }


def load_latest_stock_theme_date() -> str:
    path = STOCK_THEME_DAILY_CSV if STOCK_THEME_DAILY_CSV.exists() else DATA_DIR / "stock_theme_daily.csv"
    if not path.exists():
        return "NA"
    df = pd.read_csv(path, usecols=["date"], low_memory=False)
    if df.empty:
        return "NA"
    return pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().max() or "NA"


def build_top20_churn(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    base_top20 = latest.loc[latest["baseline_rank"] <= TOP_N].copy()
    overlay_top20 = latest.loc[latest["overlay_rank"] <= TOP_N].copy()
    base_set = set(base_top20["code"])
    overlay_set = set(overlay_top20["code"])
    intersection = base_set & overlay_set
    entered = overlay_set - base_set
    exited = base_set - overlay_set
    churn_ratio = _safe_ratio(len(entered) + len(exited), TOP_N * 2)
    overlap_ratio = _safe_ratio(len(intersection), TOP_N)
    status = _status_from_thresholds(
        churn_ratio,
        pass_cond=churn_ratio <= 0.40,
        warn_cond=churn_ratio <= 0.60,
    )
    rows = []
    for _, row in latest.loc[latest["code"].isin(base_set | overlay_set)].sort_values(
        ["overlay_rank", "baseline_rank", "code"]
    ).iterrows():
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "name": row.get("name", ""),
                "baseline_rank": int(row["baseline_rank"]),
                "overlay_rank": int(row["overlay_rank"]),
                "in_baseline_top20": bool(row["code"] in base_set),
                "in_overlay_top20": bool(row["code"] in overlay_set),
                "top20_status": "kept" if row["code"] in intersection else ("entered" if row["code"] in entered else "exited"),
                "score_delta_v3": float(row["score_delta_v3"]),
                "dominant_theme": row["dominant_theme"],
                "theme_confidence": float(row["theme_confidence"]),
                "explain_text": row["explain_text"],
            }
        )
    return pd.DataFrame(rows), {
        "status": status,
        "base_count": len(base_set),
        "overlay_count": len(overlay_set),
        "intersection_count": len(intersection),
        "entered_count": len(entered),
        "exited_count": len(exited),
        "churn_ratio": churn_ratio,
        "overlap_ratio": overlap_ratio,
        "entered_names": overlay_top20.loc[overlay_top20["code"].isin(entered), "name"].astype(str).tolist(),
        "exited_names": base_top20.loc[base_top20["code"].isin(exited), "name"].astype(str).tolist(),
    }


def build_no_theme_retention(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    base_no_theme = latest.loc[(latest["baseline_rank"] <= TOP_N) & latest["is_no_theme"]].copy()
    overlay_top20_codes = set(latest.loc[latest["overlay_rank"] <= TOP_N, "code"])
    base_no_theme["retained_in_overlay_top20"] = base_no_theme["code"].isin(overlay_top20_codes)
    retention_ratio = _safe_ratio(int(base_no_theme["retained_in_overlay_top20"].sum()), len(base_no_theme))
    status = _status_from_thresholds(
        retention_ratio,
        pass_cond=retention_ratio >= 0.70,
        warn_cond=retention_ratio >= 0.50,
    )
    out = base_no_theme.loc[
        :,
        [
            "date",
            "code",
            "name",
            "baseline_rank",
            "overlay_rank",
            "dominant_theme",
            "theme_confidence",
            "retained_in_overlay_top20",
            "explain_text",
        ],
    ].sort_values(["baseline_rank", "overlay_rank", "code"]).reset_index(drop=True)
    return out, {
        "status": status,
        "baseline_no_theme_top20_count": int(len(base_no_theme)),
        "retained_count": int(base_no_theme["retained_in_overlay_top20"].sum()),
        "retention_ratio": retention_ratio,
    }


def build_theme_concentration(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    top20 = latest.loc[latest["overlay_rank"] <= TOP_N].copy()
    top20["dominant_theme"] = top20["dominant_theme"].fillna("(none)").replace("", "(none)")
    grouped = (
        top20.groupby("dominant_theme", as_index=False)
        .agg(
            stock_count=("code", "count"),
            avg_theme_confidence=("theme_confidence", "mean"),
            avg_theme_score_effective=("theme_score_effective", "mean"),
        )
        .sort_values(["stock_count", "dominant_theme"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["top20_share"] = grouped["stock_count"] / max(len(top20), 1)
    max_share = float(grouped["top20_share"].max()) if not grouped.empty else 0.0
    max_theme = str(grouped.iloc[0]["dominant_theme"]) if not grouped.empty else "(none)"
    status = _status_from_thresholds(
        max_share,
        pass_cond=max_share <= 0.40,
        warn_cond=max_share <= 0.60,
    )
    return grouped, {
        "status": status,
        "max_theme": max_theme,
        "max_share": max_share,
        "max_count": int(grouped.iloc[0]["stock_count"]) if not grouped.empty else 0,
    }


def build_new_entry_quality(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    base_top20 = set(latest.loc[latest["baseline_rank"] <= TOP_N, "code"])
    overlay_entries = latest.loc[(latest["overlay_rank"] <= TOP_N) & ~latest["code"].isin(base_top20)].copy()
    overlay_entries["explainable"] = (
        overlay_entries["has_theme_explain"]
        & overlay_entries["theme_confidence"].gt(0.5)
        & ~overlay_entries["is_no_theme"]
    )
    explainable_ratio = _safe_ratio(int(overlay_entries["explainable"].sum()), len(overlay_entries))
    status = _status_from_thresholds(
        explainable_ratio,
        pass_cond=explainable_ratio >= 0.60,
        warn_cond=explainable_ratio >= 0.40,
    )
    out = overlay_entries.loc[
        :,
        [
            "date",
            "code",
            "name",
            "baseline_rank",
            "overlay_rank",
            "score_delta_v3",
            "dominant_theme",
            "theme_confidence",
            "theme_score_effective",
            "has_theme_explain",
            "explainable",
            "explain_text",
        ],
    ].sort_values(["overlay_rank", "baseline_rank", "code"]).reset_index(drop=True)
    return out, {
        "status": status,
        "entry_count": int(len(overlay_entries)),
        "explainable_count": int(overlay_entries["explainable"].sum()),
        "explainable_ratio": explainable_ratio,
    }


def build_theme_lift(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    lifted = latest.loc[latest["overlay_rank"] < latest["baseline_rank"]].copy()
    high_effective_threshold = float(lifted["theme_score_effective"].quantile(0.60)) if not lifted.empty else 0.0
    lifted["high_theme_score_effective"] = lifted["theme_score_effective"] >= high_effective_threshold
    high_ratio = _safe_ratio(int(lifted["high_theme_score_effective"].sum()), len(lifted))
    status = "PASS" if high_ratio >= 0.60 else "WARN"
    out = lifted.loc[
        :,
        [
            "date",
            "code",
            "name",
            "baseline_rank",
            "overlay_rank",
            "score_delta_v3",
            "dominant_theme",
            "theme_confidence",
            "theme_score_effective",
            "high_theme_score_effective",
            "explain_text",
        ],
    ].sort_values(["score_delta_v3", "baseline_rank"], ascending=[False, True]).reset_index(drop=True)
    return out, {
        "status": status,
        "lifted_count": int(len(lifted)),
        "high_effective_count": int(lifted["high_theme_score_effective"].sum()),
        "high_effective_ratio": high_ratio,
        "high_effective_threshold": high_effective_threshold,
    }


def build_explain_consistency_sample(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = latest.sort_values(["overlay_rank", "baseline_rank", "code"]).head(20).copy()
    sample["manual_review_flag"] = sample["is_no_theme"] | ~sample["has_theme_explain"] | sample["theme_confidence"].lt(0.5)
    sample["review_point"] = sample.apply(
        lambda row: (
            "Check theme-to-explain linkage manually."
            if bool(row["manual_review_flag"])
            else "Explain/theme linkage looks mechanically consistent."
        ),
        axis=1,
    )
    return sample.loc[
        :,
        [
            "date",
            "code",
            "name",
            "overlay_rank",
            "dominant_theme",
            "theme_confidence",
            "has_theme_explain",
            "manual_review_flag",
            "review_point",
            "explain_text",
        ],
    ].reset_index(drop=True), {
        "review_count": int(sample["manual_review_flag"].sum()),
        "sample_count": int(len(sample)),
    }


def final_decision(metric_statuses: list[str]) -> str:
    fail_count = sum(1 for status in metric_statuses if status == "FAIL")
    warn_count = sum(1 for status in metric_statuses if status == "WARN")
    if fail_count >= 1:
        return "DO NOT PROMOTE"
    if warn_count >= 2:
        return "NEED MORE VALIDATION"
    return "READY FOR OPERATIONAL PROMOTION"


def write_mode_note(overlay_meta: dict[str, object]) -> None:
    resolved_mode = str(overlay_meta["resolved_mode"])
    evaluation_profile = str(overlay_meta["evaluation_profile"])
    overlay_score_column = str(overlay_meta["overlay_score_column"])
    note_lines = [
        "# Theme Overlay Acceptance Mode Note",
        "",
        "## Mode Rules",
        "- operational: uplift is evaluated with `final_score_v3` and its derived overlay rank.",
        "- shadow: uplift is evaluated with `shadow_final_score_v3` when available, as counterfactual overlay rank.",
        "- off: report is treated as no-op; overlay uplift is not expected to be meaningful.",
        "",
        "## Current Evaluation",
        f"- resolved_mode: {resolved_mode}",
        f"- evaluation_profile: {evaluation_profile}",
        f"- overlay_score_column_used: {overlay_score_column}",
        f"- final_score_baseline_column: final_score",
        f"- operational_score_delta_rule: final_score_v3 - final_score",
        f"- shadow_score_delta_rule: shadow_final_score_v3 - final_score",
        f"- off_score_delta_rule: 0.0 (no-op baseline)",
        f"- final_score_v3_source: {overlay_meta.get('v3_source', 'NA')}",
        f"- shadow_score_source: {overlay_meta.get('shadow_source', 'NA')}",
    ]
    MODE_NOTE_MD.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> None:
    latest, v3_source, overlay_meta = load_latest_ranking()
    latest_theme_date = load_latest_stock_theme_date()

    churn_df, churn_meta = build_top20_churn(latest)
    no_theme_df, no_theme_meta = build_no_theme_retention(latest)
    concentration_df, concentration_meta = build_theme_concentration(latest)
    new_entry_df, new_entry_meta = build_new_entry_quality(latest)
    lift_df, lift_meta = build_theme_lift(latest)
    shadow_effect_df, shadow_effect_meta = build_shadow_effect_diagnostics(latest)
    explain_sample_df, explain_meta = build_explain_consistency_sample(latest)

    TOP20_CHURN_CSV.parent.mkdir(parents=True, exist_ok=True)
    churn_df.to_csv(TOP20_CHURN_CSV, index=False, encoding="utf-8")
    no_theme_df.to_csv(NO_THEME_RETENTION_CSV, index=False, encoding="utf-8")
    concentration_df.to_csv(THEME_CONCENTRATION_CSV, index=False, encoding="utf-8")
    lift_df.to_csv(THEME_LIFT_ANALYSIS_CSV, index=False, encoding="utf-8")
    new_entry_df.to_csv(NEW_ENTRY_QUALITY_CSV, index=False, encoding="utf-8")
    write_mode_note(overlay_meta)

    statuses = [
        churn_meta["status"],
        no_theme_meta["status"],
        concentration_meta["status"],
        new_entry_meta["status"],
        lift_meta["status"],
    ]
    decision = final_decision(statuses)

    mode_notice = ""
    if overlay_meta["resolved_mode"] == "off":
        mode_notice = "mode=off, no effective overlay uplift expected."
    elif overlay_meta["resolved_mode"] == "shadow":
        mode_notice = "mode=shadow, report is a counterfactual overlay evaluation."
    else:
        mode_notice = "mode=operational, report reflects live overlay scoring."

    lines = [
        "# Theme Overlay Acceptance Report",
        "",
        "## Snapshot",
        f"- latest_ranking_date: {latest['date'].max() if not latest.empty else 'NA'}",
        f"- ranking_final_source: {RANKING_FINAL_CSV}",
        f"- overlay_score_source: {v3_source}",
        f"- stock_theme_daily_latest_date: {latest_theme_date}",
        f"- row_count_latest: {len(latest)}",
        f"- resolved_mode: {overlay_meta['resolved_mode']}",
        f"- evaluation_profile: {overlay_meta['evaluation_profile']}",
        f"- overlay_score_column_used_for_evaluation: {overlay_meta['overlay_score_column']}",
        f"- shadow_score_source: {overlay_meta.get('shadow_source', 'NA')}",
        f"- mode_notice: {mode_notice}",
        "",
        "## Final Decision",
        f"- decision: {decision}",
        f"- metric_statuses: {statuses}",
        "",
        "## 1. Top20 Churn Stability",
        f"- status: {churn_meta['status']}",
        f"- intersection_ratio: {churn_meta['overlap_ratio']:.2%}",
        f"- churn_ratio: {churn_meta['churn_ratio']:.2%}",
        f"- entered_count: {churn_meta['entered_count']}",
        f"- exited_count: {churn_meta['exited_count']}",
        f"- interpretation: overlap ratio is {churn_meta['overlap_ratio']:.2%} and churn ratio is {churn_meta['churn_ratio']:.2%}.",
        "",
        "## 2. No-Theme Retention",
        f"- status: {no_theme_meta['status']}",
        f"- baseline_no_theme_top20_count: {no_theme_meta['baseline_no_theme_top20_count']}",
        f"- retained_count: {no_theme_meta['retained_count']}",
        f"- retention_ratio: {no_theme_meta['retention_ratio']:.2%}",
        f"- interpretation: baseline top20 no-theme names retained at {no_theme_meta['retention_ratio']:.2%}.",
        "",
        "## 3. Theme Concentration",
        f"- status: {concentration_meta['status']}",
        f"- max_theme: {concentration_meta['max_theme']}",
        f"- max_share: {concentration_meta['max_share']:.2%}",
        f"- max_count: {concentration_meta['max_count']}",
        f"- interpretation: the largest dominant theme share in overlay top20 is {concentration_meta['max_share']:.2%}.",
        "",
        "## 4. Near-Top20 Entry Quality",
        f"- status: {new_entry_meta['status']}",
        f"- entry_count: {new_entry_meta['entry_count']}",
        f"- explainable_count: {new_entry_meta['explainable_count']}",
        f"- explainable_ratio: {new_entry_meta['explainable_ratio']:.2%}",
        f"- interpretation: explainable new entries account for {new_entry_meta['explainable_ratio']:.2%}.",
        "",
        "## 5. Theme Lift Effect",
        f"- status: {lift_meta['status']}",
        f"- lifted_count: {lift_meta['lifted_count']}",
        f"- high_effective_count: {lift_meta['high_effective_count']}",
        f"- high_effective_ratio: {lift_meta['high_effective_ratio']:.2%}",
        f"- high_effective_threshold: {lift_meta['high_effective_threshold']:.4f}",
        f"- interpretation: high theme-effective names among lifted stocks account for {lift_meta['high_effective_ratio']:.2%}.",
        "",
        "## 6. Shadow Effect Diagnostics",
        f"- direct_uplift_count: {shadow_effect_meta['direct_uplift_count']}",
        f"- direct_uplift_top20_count: {shadow_effect_meta['direct_uplift_top20_count']}",
        f"- indirect_rank_gain_count: {shadow_effect_meta['indirect_rank_gain_count']}",
        f"- large_negative_displacement_count: {shadow_effect_meta['large_negative_displacement_count']}",
        f"- minimal_score_delta_threshold: {shadow_effect_meta['minimal_score_delta_threshold']:.2f}",
        f"- large_negative_score_delta_threshold: {shadow_effect_meta['large_negative_score_delta_threshold']:.2f}",
        "- interpretation: direct uplift means positive score delta with positive rank gain; indirect rank gain means rank gain with score delta near zero; large negative displacement tracks strong losers.",
        "",
        "## 7. Explain Consistency",
        f"- sample_count: {explain_meta['sample_count']}",
        f"- manual_review_count: {explain_meta['review_count']}",
        "- interpretation: top overlay names are sampled for theme/explain consistency review.",
        "",
        "## Review Pointers",
        "- Check whether names with `(none)` dominant_theme still contain theme wording in explain_text.",
        "- Check whether low-confidence new entries are being over-promoted.",
        "- Check whether a single theme is dominating top20 beyond the intended cap.",
        "",
        "## Output Files",
        f"- {TOP20_CHURN_CSV}",
        f"- {NO_THEME_RETENTION_CSV}",
        f"- {THEME_CONCENTRATION_CSV}",
        f"- {THEME_LIFT_ANALYSIS_CSV}",
        f"- {NEW_ENTRY_QUALITY_CSV}",
        f"- {MODE_NOTE_MD}",
        "",
        "## Explain Sample",
        "```json",
        explain_sample_df.head(20).to_json(orient="records", force_ascii=False, indent=2),
        "```",
    ]
    ACCEPTANCE_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        "generated_files="
        + str(
            [
                str(ACCEPTANCE_REPORT_MD),
                str(TOP20_CHURN_CSV),
                str(NO_THEME_RETENTION_CSV),
                str(THEME_CONCENTRATION_CSV),
                str(THEME_LIFT_ANALYSIS_CSV),
                str(NEW_ENTRY_QUALITY_CSV),
                str(MODE_NOTE_MD),
            ]
        )
    )


if __name__ == "__main__":
    main()
