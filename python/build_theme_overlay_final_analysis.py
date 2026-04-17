from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RANKING_CSV = DATA_DIR / "ranking_final.csv"
BEFORE_AFTER_V3_CSV = DATA_DIR / "before_after_score_compare_v3.csv"
TOP20_BEFORE_AFTER_V3_CSV = DATA_DIR / "top20_before_after_compare_v3.csv"
THEME_CONCENTRATION_CSV = DATA_DIR / "theme_concentration_report.csv"
NEAR_TOP20_LIFT_CSV = DATA_DIR / "near_top20_theme_lift_report.csv"
NO_THEME_DISPLACEMENT_MD = DATA_DIR / "no_theme_displacement_report.md"
ACCEPTANCE_SUMMARY_MD = DATA_DIR / "theme_overlay_acceptance_summary.md"
FINAL_ANALYSIS_MD = DATA_DIR / "theme_overlay_final_analysis.md"

TOP_N = 20
NEAR_TOP_N = 40
CHURN_WARNING_THRESHOLD = 6
NO_THEME_EXIT_WARNING_THRESHOLD = 4
TOP1_CONCENTRATION_WARNING = 8
TOP2_CONCENTRATION_WARNING = 12

LOGGER = logging.getLogger("build_theme_overlay_final_analysis")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def load_latest_ranking(path: Path = RANKING_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = df["date"].dropna().max()
    latest = df[df["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["name"] = latest.get("name", "").fillna("").astype(str)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str).str.strip()
    latest["regime"] = latest.get("regime", "").fillna("").astype(str).str.strip().str.lower()

    numeric_columns = [
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
        "ret_score",
        "prob_score",
        "tech_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
    ]
    for col in numeric_columns:
        latest[col] = _to_numeric(latest.get(col)).fillna(0.0)

    latest["theme_score_effective"] = latest["theme_score_effective"].fillna(latest["theme_score"] * latest["theme_confidence"])
    latest["base_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)
    latest["v2_rank"] = latest["final_score_v2"].rank(method="first", ascending=False).astype(int)
    latest["v3_rank"] = latest["final_score_v3"].rank(method="first", ascending=False).astype(int)
    latest["is_themed"] = latest["dominant_theme"].ne("") & latest["theme_score"].gt(0.0)
    latest["is_no_theme"] = ~latest["is_themed"]
    return latest.sort_values(["base_rank", "code"]).reset_index(drop=True)


def build_compare_tables(latest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = latest.copy()
    work["score_diff_v2"] = work["final_score_v2"] - work["final_score"]
    work["score_diff_v3"] = work["final_score_v3"] - work["final_score"]
    work["v3_vs_v2_diff"] = work["final_score_v3"] - work["final_score_v2"]
    work["rank_shift_vs_base"] = work["base_rank"] - work["v3_rank"]
    work["rank_shift_vs_v2"] = work["v2_rank"] - work["v3_rank"]
    work["theme_label"] = work["dominant_theme"].where(work["dominant_theme"].ne(""), "(none)")
    work["overlay_direction"] = work["rank_shift_vs_base"].map(
        lambda value: "lift" if value > 0 else ("drag" if value < 0 else "flat")
    )
    work["explain_ko"] = work.apply(
        lambda row: (
            f"{row['name']}은/는 baseline rank {int(row['base_rank'])}에서 v3 rank {int(row['v3_rank'])}로 "
            f"{'상승' if row['rank_shift_vs_base'] > 0 else ('하락' if row['rank_shift_vs_base'] < 0 else '유지')}했다. "
            f"theme={row['theme_label']}, theme_score={float(row['theme_score']):.1f}, "
            f"theme_confidence={float(row['theme_confidence']):.2f}, score_diff_v3={float(row['score_diff_v3']):+.2f}."
        ),
        axis=1,
    )
    work["explain_en"] = work.apply(
        lambda row: (
            f"{row['name']} moved from baseline rank {int(row['base_rank'])} to v3 rank {int(row['v3_rank'])}. "
            f"theme={row['theme_label']}, theme_score={float(row['theme_score']):.1f}, "
            f"theme_confidence={float(row['theme_confidence']):.2f}, score_diff_v3={float(row['score_diff_v3']):+.2f}."
        ),
        axis=1,
    )
    compare_cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "regime",
        "dominant_theme",
        "theme_label",
        "is_themed",
        "base_rank",
        "v2_rank",
        "v3_rank",
        "rank_shift_vs_base",
        "rank_shift_vs_v2",
        "overlay_direction",
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "score_diff_v2",
        "score_diff_v3",
        "v3_vs_v2_diff",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
        "ret_score",
        "prob_score",
        "tech_score",
        "qual_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
        "explain_ko",
        "explain_en",
    ]
    compare_df = work.loc[:, compare_cols].sort_values(["v3_rank", "base_rank", "code"]).reset_index(drop=True)

    top_union = work[
        work["base_rank"].le(TOP_N) | work["v2_rank"].le(TOP_N) | work["v3_rank"].le(TOP_N)
    ].copy()
    top_union["in_base_top20"] = top_union["base_rank"].le(TOP_N)
    top_union["in_v2_top20"] = top_union["v2_rank"].le(TOP_N)
    top_union["in_v3_top20"] = top_union["v3_rank"].le(TOP_N)
    top_union = top_union.sort_values(["v3_rank", "base_rank", "code"]).reset_index(drop=True)
    top20_compare_df = top_union.loc[:, compare_cols + ["in_base_top20", "in_v2_top20", "in_v3_top20"]]
    return compare_df, top20_compare_df


def build_theme_concentration_report(latest: pd.DataFrame) -> pd.DataFrame:
    top20_v3 = latest[latest["v3_rank"].le(TOP_N)].copy()
    top20_v3["theme_label"] = top20_v3["dominant_theme"].where(top20_v3["dominant_theme"].ne(""), "(none)")
    concentration = (
        top20_v3.groupby("theme_label", as_index=False)
        .agg(
            stock_count=("code", "count"),
            avg_theme_score=("theme_score", "mean"),
            avg_theme_confidence=("theme_confidence", "mean"),
            avg_rank_shift_vs_base=("base_rank", lambda s: float((top20_v3.loc[s.index, "base_rank"] - top20_v3.loc[s.index, "v3_rank"]).mean())),
        )
        .sort_values(["stock_count", "avg_theme_score", "theme_label"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    concentration["top20_share"] = concentration["stock_count"] / max(len(top20_v3), 1)
    concentration["concentration_flag"] = concentration["stock_count"].ge(TOP1_CONCENTRATION_WARNING)
    concentration["explain_ko"] = concentration.apply(
        lambda row: f"{row['theme_label']} 테마는 v3 top20에서 {int(row['stock_count'])}종목({row['top20_share']:.1%})을 차지한다.",
        axis=1,
    )
    concentration["explain_en"] = concentration.apply(
        lambda row: f"{row['theme_label']} accounts for {int(row['stock_count'])} names ({row['top20_share']:.1%}) in the v3 top20.",
        axis=1,
    )
    return concentration


def build_near_top20_lift_report(latest: pd.DataFrame) -> pd.DataFrame:
    near = latest[
        latest["base_rank"].between(TOP_N + 1, NEAR_TOP_N)
        & latest["v3_rank"].le(TOP_N)
        & latest["is_themed"]
    ].copy()
    cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "base_rank",
        "v2_rank",
        "v3_rank",
        "lift_size",
        "lift_band",
        "final_score",
        "final_score_v3",
        "score_diff_v3",
        "explain_ko",
        "explain_en",
    ]
    if near.empty:
        return pd.DataFrame(columns=cols)
    near["lift_size"] = near["base_rank"] - near["v3_rank"]
    near["lift_band"] = near["lift_size"].map(lambda value: "strong" if value >= 10 else ("moderate" if value >= 5 else "mild"))
    near["explain_ko"] = near.apply(
        lambda row: (
            f"{row['name']}은/는 near-top20 구간(base_rank={int(row['base_rank'])})에서 "
            f"theme overlay 후 top20(v3_rank={int(row['v3_rank'])})에 진입했다."
        ),
        axis=1,
    )
    near["explain_en"] = near.apply(
        lambda row: (
            f"{row['name']} entered the v3 top20 from the near-top20 band "
            f"(base_rank={int(row['base_rank'])}, v3_rank={int(row['v3_rank'])})."
        ),
        axis=1,
    )
    return near.loc[:, cols].sort_values(["v3_rank", "lift_size", "code"], ascending=[True, False, True]).reset_index(drop=True)


def build_no_theme_displacement_report(latest: pd.DataFrame) -> str:
    base_top20 = latest[latest["base_rank"].le(TOP_N)].copy()
    v3_top20 = latest[latest["v3_rank"].le(TOP_N)].copy()
    base_no_theme = base_top20[base_top20["is_no_theme"]].copy()
    v3_no_theme = v3_top20[v3_top20["is_no_theme"]].copy()
    displaced = base_no_theme[~base_no_theme["code"].isin(set(v3_no_theme["code"]))].copy()
    replaced_by_theme = v3_top20[~v3_top20["code"].isin(set(base_top20["code"])) & v3_top20["is_themed"]].copy()
    displaced = displaced.sort_values(["base_rank", "code"]).reset_index(drop=True)
    replaced_by_theme = replaced_by_theme.sort_values(["v3_rank", "code"]).reset_index(drop=True)

    lines = [
        "# No-Theme Displacement Report",
        "",
        f"- base_top20_no_theme_count: {len(base_no_theme)}",
        f"- v3_top20_no_theme_count: {len(v3_no_theme)}",
        f"- displaced_no_theme_count: {len(displaced)}",
        f"- warning_threshold: {NO_THEME_EXIT_WARNING_THRESHOLD}",
        "",
        "## 한국어 요약",
        (
            "- no-theme 종목 이탈이 경고 수준이다."
            if len(displaced) >= NO_THEME_EXIT_WARNING_THRESHOLD
            else "- no-theme 종목 이탈은 관리 가능한 범위다."
        ),
        (
            f"- baseline top20의 no-theme 종목 {len(base_no_theme)}개 중 {len(displaced)}개가 v3 top20에서 이탈했다."
            if len(base_no_theme)
            else "- baseline top20에 no-theme 종목이 없었다."
        ),
        "",
        "## English Summary",
        (
            "- No-theme displacement is above the warning threshold."
            if len(displaced) >= NO_THEME_EXIT_WARNING_THRESHOLD
            else "- No-theme displacement remains within the acceptable range."
        ),
        (
            f"- {len(displaced)} of {len(base_no_theme)} no-theme baseline top20 names were displaced in v3."
            if len(base_no_theme)
            else "- There were no no-theme names in the baseline top20."
        ),
        "",
        "## Displaced No-Theme Names",
    ]
    if displaced.empty:
        lines.append("- none")
    else:
        for row in displaced.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, v3_rank={int(row.v3_rank)}, "
                f"score_diff_v3={float(row.final_score_v3 - row.final_score):+.2f}"
            )
    lines.extend(["", "## Themed Replacements"])
    if replaced_by_theme.empty:
        lines.append("- none")
    else:
        for row in replaced_by_theme.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: v3_rank={int(row.v3_rank)}, theme={row.dominant_theme}, "
                f"theme_score={float(row.theme_score):.1f}, theme_confidence={float(row.theme_confidence):.2f}"
            )
    return "\n".join(lines) + "\n"


def build_acceptance_summary(latest: pd.DataFrame, concentration_df: pd.DataFrame, near_lift_df: pd.DataFrame) -> str:
    base_top20 = latest[latest["base_rank"].le(TOP_N)].copy()
    v2_top20 = latest[latest["v2_rank"].le(TOP_N)].copy()
    v3_top20 = latest[latest["v3_rank"].le(TOP_N)].copy()

    base_set = set(base_top20["code"])
    v3_set = set(v3_top20["code"])
    top20_churn = len(base_set.symmetric_difference(v3_set))

    base_no_theme = base_top20[base_top20["is_no_theme"]].copy()
    v3_no_theme = v3_top20[v3_top20["is_no_theme"]].copy()
    no_theme_displaced = base_no_theme[~base_no_theme["code"].isin(v3_no_theme["code"])].copy()

    themed_concentration = concentration_df[concentration_df["theme_label"] != "(none)"].copy()
    top1_count = int(themed_concentration["stock_count"].iloc[0]) if not themed_concentration.empty else 0
    top2_count = int(themed_concentration["stock_count"].head(2).sum()) if not themed_concentration.empty else 0
    top1_theme = str(themed_concentration["theme_label"].iloc[0]) if not themed_concentration.empty else "(none)"
    top2_themes = themed_concentration["theme_label"].head(2).tolist() if not themed_concentration.empty else []

    warnings: list[str] = []
    if top20_churn > CHURN_WARNING_THRESHOLD:
        warnings.append(f"top20 churn warning: {top20_churn} names moved across the top20 boundary.")
    if len(no_theme_displaced) >= NO_THEME_EXIT_WARNING_THRESHOLD:
        warnings.append(f"no-theme displacement warning: {len(no_theme_displaced)} baseline no-theme names left the top20.")
    if top1_count >= TOP1_CONCENTRATION_WARNING or top2_count >= TOP2_CONCENTRATION_WARNING:
        warnings.append(
            f"theme concentration warning: top1={top1_theme} ({top1_count}), top2={top2_themes} ({top2_count} combined)."
        )

    status = "HOLD"
    if not warnings and not near_lift_df.empty:
        status = "PASS"
    elif not warnings:
        status = "CONDITIONAL PASS"

    lines = [
        "# Theme Overlay Acceptance Summary",
        "",
        "## Acceptance Summary",
        f"- decision_status: {status}",
        f"- latest_date: {latest['date'].iloc[0] if not latest.empty else 'NA'}",
        f"- baseline_top20_count: {len(base_top20)}",
        f"- v2_top20_count: {len(v2_top20)}",
        f"- v3_top20_count: {len(v3_top20)}",
        f"- top20_churn_count: {top20_churn}",
        f"- no_theme_displaced_count: {len(no_theme_displaced)}",
        f"- near_top20_theme_entries: {len(near_lift_df)}",
        f"- top1_theme_count: {top1_count}",
        f"- top2_theme_count: {top2_count}",
        "",
        "## 판정 해석",
        f"- 한국어: 현재 판정은 `{status}`이다. top20 churn, no-theme displacement, top-theme concentration을 함께 반영했다.",
        f"- English: Current decision is `{status}` based on top20 churn, no-theme displacement, and dominant-theme concentration.",
        "",
        "## Rule Check",
        f"- top20 churn rule: count={top20_churn}, threshold={CHURN_WARNING_THRESHOLD}, result={'WARN' if top20_churn > CHURN_WARNING_THRESHOLD else 'OK'}",
        f"- no-theme displacement rule: count={len(no_theme_displaced)}, threshold={NO_THEME_EXIT_WARNING_THRESHOLD}, result={'WARN' if len(no_theme_displaced) >= NO_THEME_EXIT_WARNING_THRESHOLD else 'OK'}",
        f"- theme concentration rule: top1={top1_count}, top2={top2_count}, thresholds=({TOP1_CONCENTRATION_WARNING}, {TOP2_CONCENTRATION_WARNING}), result={'WARN' if (top1_count >= TOP1_CONCENTRATION_WARNING or top2_count >= TOP2_CONCENTRATION_WARNING) else 'OK'}",
        "",
        "## Warnings",
    ]
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Top20 Delta",
        f"- Entered v3 top20: {', '.join(v3_top20.loc[~v3_top20['code'].isin(base_set), 'name'].tolist()) if len(v3_top20.loc[~v3_top20['code'].isin(base_set)]) else 'none'}",
        f"- Left baseline top20: {', '.join(base_top20.loc[~base_top20['code'].isin(v3_set), 'name'].tolist()) if len(base_top20.loc[~base_top20['code'].isin(v3_set)]) else 'none'}",
    ])
    return "\n".join(lines) + "\n"


def build_final_analysis_markdown(
    latest: pd.DataFrame,
    compare_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    near_lift_df: pd.DataFrame,
) -> str:
    top_lifters = compare_df.sort_values(["rank_shift_vs_base", "score_diff_v3"], ascending=[False, False]).head(10)
    top_drags = compare_df.sort_values(["rank_shift_vs_base", "score_diff_v3"], ascending=[True, True]).head(10)
    lines = [
        "# Theme Overlay Final Analysis",
        "",
        "## Summary",
        f"- latest_date: {latest['date'].iloc[0] if not latest.empty else 'NA'}",
        f"- total_stocks: {len(latest)}",
        f"- themed_stocks: {int(latest['is_themed'].sum())}",
        f"- top20_themed_count_v3: {int(latest['v3_rank'].le(TOP_N).mul(latest['is_themed']).sum())}",
        f"- near_top20_theme_entries: {len(near_lift_df)}",
        "",
        "## Top Lifters",
    ]
    for row in top_lifters.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, v3_rank={int(row.v3_rank)}, "
            f"rank_shift={int(row.rank_shift_vs_base)}, theme={row.theme_label}, score_diff_v3={float(row.score_diff_v3):+.2f}"
        )
    lines.extend(["", "## Top Drags"])
    for row in top_drags.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, v3_rank={int(row.v3_rank)}, "
            f"rank_shift={int(row.rank_shift_vs_base)}, theme={row.theme_label}, score_diff_v3={float(row.score_diff_v3):+.2f}"
        )
    lines.extend(["", "## Theme Concentration Snapshot"])
    for row in concentration_df.head(10).itertuples(index=False):
        lines.append(
            f"- {row.theme_label}: stock_count={int(row.stock_count)}, top20_share={float(row.top20_share):.1%}, "
            f"avg_theme_confidence={float(row.avg_theme_confidence):.2f}"
        )
    lines.extend(["", "## Near-Top20 Lift Candidates"])
    if near_lift_df.empty:
        lines.append("- none")
    else:
        for row in near_lift_df.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: base_rank={int(row.base_rank)}, v3_rank={int(row.v3_rank)}, "
                f"theme={row.dominant_theme}, lift_size={int(row.lift_size)}"
            )
    return "\n".join(lines) + "\n"


def save_outputs(
    compare_df: pd.DataFrame,
    top20_compare_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    near_lift_df: pd.DataFrame,
    no_theme_md: str,
    acceptance_md: str,
    final_analysis_md: str,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(BEFORE_AFTER_V3_CSV, index=False, encoding="utf-8-sig")
    top20_compare_df.to_csv(TOP20_BEFORE_AFTER_V3_CSV, index=False, encoding="utf-8-sig")
    concentration_df.to_csv(THEME_CONCENTRATION_CSV, index=False, encoding="utf-8-sig")
    near_lift_df.to_csv(NEAR_TOP20_LIFT_CSV, index=False, encoding="utf-8-sig")
    NO_THEME_DISPLACEMENT_MD.write_text(no_theme_md, encoding="utf-8")
    ACCEPTANCE_SUMMARY_MD.write_text(acceptance_md, encoding="utf-8")
    FINAL_ANALYSIS_MD.write_text(final_analysis_md, encoding="utf-8")


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    compare_df, top20_compare_df = build_compare_tables(latest)
    concentration_df = build_theme_concentration_report(latest)
    near_lift_df = build_near_top20_lift_report(latest)
    no_theme_md = build_no_theme_displacement_report(latest)
    acceptance_md = build_acceptance_summary(latest, concentration_df, near_lift_df)
    final_analysis_md = build_final_analysis_markdown(latest, compare_df, concentration_df, near_lift_df)
    save_outputs(
        compare_df,
        top20_compare_df,
        concentration_df,
        near_lift_df,
        no_theme_md,
        acceptance_md,
        final_analysis_md,
    )
    LOGGER.info("Saved %s", BEFORE_AFTER_V3_CSV.resolve())
    LOGGER.info("Saved %s", TOP20_BEFORE_AFTER_V3_CSV.resolve())
    LOGGER.info("Saved %s", THEME_CONCENTRATION_CSV.resolve())
    LOGGER.info("Saved %s", NEAR_TOP20_LIFT_CSV.resolve())
    LOGGER.info("Saved %s", NO_THEME_DISPLACEMENT_MD.resolve())
    LOGGER.info("Saved %s", ACCEPTANCE_SUMMARY_MD.resolve())
    LOGGER.info("Saved %s", FINAL_ANALYSIS_MD.resolve())
    print(
        "generated_files="
        + str(
            [
                str(BEFORE_AFTER_V3_CSV),
                str(TOP20_BEFORE_AFTER_V3_CSV),
                str(THEME_CONCENTRATION_CSV),
                str(NEAR_TOP20_LIFT_CSV),
                str(NO_THEME_DISPLACEMENT_MD),
                str(ACCEPTANCE_SUMMARY_MD),
                str(FINAL_ANALYSIS_MD),
            ]
        )
    )


if __name__ == "__main__":
    main()
