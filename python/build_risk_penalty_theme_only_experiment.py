import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUT_CSV = DATA_DIR / "ranking_final_risk_penalty_theme_only.csv"
OUT_COMPARE_CSV = DATA_DIR / "risk_penalty_theme_only_compare.csv"
OUT_MD = DATA_DIR / "risk_penalty_theme_only_experiment.md"

LOGGER = logging.getLogger("build_risk_penalty_theme_only_experiment")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest() -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV, dtype={"code": str}, low_memory=False)
    df["date"] = df["date"].astype(str)
    latest = df.loc[df["date"] == df["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    numeric_cols = [
        "ret_score",
        "prob_score",
        "tech_score",
        "qual_score",
        "valuation_score",
        "risk_penalty",
        "theme_score",
        "theme_confidence",
        "w_ret_base",
        "w_prob_base",
        "w_tech_base",
        "w_qual_base",
        "w_valuation_base",
        "w_risk_penalty",
        "w_theme",
        "final_score_v3",
        "rank_final",
    ]
    for col in numeric_cols:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce").fillna(0.0)
    latest["is_themed"] = latest["dominant_theme"].str.strip().ne("")
    latest["is_neutral"] = latest["regime"].astype(str).str.lower().eq("neutral")
    return latest.sort_values("rank_final").reset_index(drop=True)


def simulate(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    base_core = (
        out["w_ret_base"] * out["ret_score"]
        + out["w_prob_base"] * out["prob_score"]
        + out["w_tech_base"] * out["tech_score"]
        + out["w_qual_base"] * out["qual_score"]
        + out["w_valuation_base"] * out["valuation_score"]
    )

    # Baseline
    out["baseline_core_score"] = base_core - out["w_risk_penalty"] * out["risk_penalty"]
    out["theme_score_effective_exp"] = out["theme_score"] * out["theme_confidence"]
    out["baseline_v3_recalc"] = (1.0 - out["w_theme"]) * out["baseline_core_score"] + out["w_theme"] * out["theme_score_effective_exp"]

    # Global soft
    global_soft_weight = out["w_risk_penalty"].where(~out["is_neutral"], 0.45)
    out["global_soft_weight"] = global_soft_weight
    out["global_soft_core_score"] = base_core - global_soft_weight * out["risk_penalty"]
    out["global_soft_v3"] = (1.0 - out["w_theme"]) * out["global_soft_core_score"] + out["w_theme"] * out["theme_score_effective_exp"]

    # Theme-only soft
    theme_soft_weight = out["w_risk_penalty"].copy()
    mask = out["is_neutral"] & out["is_themed"]
    theme_soft_weight.loc[mask] = 0.45
    out["theme_only_soft_weight"] = theme_soft_weight
    out["theme_only_soft_core_score"] = base_core - theme_soft_weight * out["risk_penalty"]
    out["theme_only_soft_v3"] = (1.0 - out["w_theme"]) * out["theme_only_soft_core_score"] + out["w_theme"] * out["theme_score_effective_exp"]

    out["baseline_rank_v3"] = out["final_score_v3"].rank(method="first", ascending=False).astype(int)
    out["global_soft_rank_v3"] = out["global_soft_v3"].rank(method="first", ascending=False).astype(int)
    out["theme_only_rank_v3"] = out["theme_only_soft_v3"].rank(method="first", ascending=False).astype(int)
    out["theme_only_rank_change"] = out["baseline_rank_v3"] - out["theme_only_rank_v3"]
    out["theme_only_score_diff"] = out["theme_only_soft_v3"] - out["final_score_v3"]
    return out.sort_values("theme_only_rank_v3").reset_index(drop=True)


def export_compare(df: pd.DataFrame) -> pd.DataFrame:
    compare = df.loc[:, [
        "date",
        "code",
        "name",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "risk_penalty",
        "w_risk_penalty",
        "global_soft_weight",
        "theme_only_soft_weight",
        "final_score_v3",
        "global_soft_v3",
        "theme_only_soft_v3",
        "baseline_rank_v3",
        "global_soft_rank_v3",
        "theme_only_rank_v3",
        "theme_only_rank_change",
        "theme_only_score_diff",
    ]].copy()
    compare.to_csv(OUT_COMPARE_CSV, index=False, encoding="utf-8-sig")
    return compare


def build_markdown(df: pd.DataFrame) -> str:
    baseline_top20 = df.loc[df["baseline_rank_v3"] <= 20].copy()
    global_soft_top20 = df.loc[df["global_soft_rank_v3"] <= 20].copy()
    theme_only_top20 = df.loc[df["theme_only_rank_v3"] <= 20].copy()

    baseline_theme_top20 = int(baseline_top20["is_themed"].sum())
    global_theme_top20 = int(global_soft_top20["is_themed"].sum())
    theme_only_theme_top20 = int(theme_only_top20["is_themed"].sum())

    entrants_theme_only = df.loc[(df["baseline_rank_v3"] > 20) & (df["theme_only_rank_v3"] <= 20)].copy()
    exits_theme_only = df.loc[(df["baseline_rank_v3"] <= 20) & (df["theme_only_rank_v3"] > 20)].copy()
    near_top20_themed = df.loc[(df["baseline_rank_v3"].between(21, 40)) & (df["is_themed"])].copy()
    near_to_top20 = near_top20_themed.loc[near_top20_themed["theme_only_rank_v3"] <= 20].copy()
    top_improvements = df.sort_values(["theme_only_rank_change", "theme_only_score_diff"], ascending=[False, False]).head(15)
    non_theme_side_effects = df.loc[(~df["is_themed"]) & (df["theme_only_rank_change"] > 0)].copy()

    lines = [
        "# Risk Penalty Theme-Only Experiment",
        "",
        "## Setup",
        "- baseline: current `final_score_v3`",
        "- global_soft: neutral regime `w_risk_penalty 0.65 -> 0.45` for all names",
        "- theme_only_soft: same relaxation, but only for rows with `dominant_theme != ''`",
        "- production ranking is unchanged; this is an experiment only.",
        "",
        "## Top20 Comparison",
        f"- baseline_top20_theme_count={baseline_theme_top20}",
        f"- global_soft_top20_theme_count={global_theme_top20}",
        f"- theme_only_top20_theme_count={theme_only_theme_top20}",
        "",
        "## Theme-Only Top20 Entries",
    ]
    if entrants_theme_only.empty:
        lines.append("- none")
    else:
        for row in entrants_theme_only.sort_values("theme_only_rank_v3").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: baseline_rank={int(row.baseline_rank_v3)}, exp_rank={int(row.theme_only_rank_v3)}, "
                f"theme={row.dominant_theme}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Theme-Only Top20 Exits"])
    if exits_theme_only.empty:
        lines.append("- none")
    else:
        for row in exits_theme_only.sort_values("baseline_rank_v3").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: baseline_rank={int(row.baseline_rank_v3)}, exp_rank={int(row.theme_only_rank_v3)}, "
                f"theme={row.dominant_theme or '(none)'}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Near-Top20 Themed Winners"])
    if near_to_top20.empty:
        lines.append("- none")
    else:
        for row in near_to_top20.sort_values("theme_only_rank_v3").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: {int(row.baseline_rank_v3)} -> {int(row.theme_only_rank_v3)}, "
                f"theme={row.dominant_theme}, theme_score={float(row.theme_score):.2f}, "
                f"theme_confidence={float(row.theme_confidence):.3f}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Largest Theme-Only Rank Improvements"])
    for row in top_improvements.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: {int(row.baseline_rank_v3)} -> {int(row.theme_only_rank_v3)}, "
            f"rank_change={int(row.theme_only_rank_change)}, theme={row.dominant_theme or '(none)'}, "
            f"score_diff={float(row.theme_only_score_diff):.4f}, risk_penalty={float(row.risk_penalty):.2f}"
        )

    lines.extend([
        "",
        "## Non-Theme Side Effects",
        f"- non_theme_positive_rank_change_count={int(len(non_theme_side_effects))}",
        "- theme_only_soft는 비테마 종목에는 risk_penalty weight를 바꾸지 않으므로, 직접적인 부작용은 거의 없어야 한다.",
        "",
        "## Interpretation",
    ])
    if len(near_to_top20) > 0:
        lines.append("- theme 종목 한정 완화는 전체 랭킹 왜곡을 줄이면서 near-top20 테마 후보를 상단으로 밀어 올리는 데 더 적합하다.")
    else:
        lines.append("- theme 종목 한정 완화만으로는 top20 구조 변화가 부족하다. 이 경우 base score gap이 더 근본 원인이다.")
    if theme_only_theme_top20 >= baseline_theme_top20:
        lines.append("- 최소한 테마 종목 비중을 훼손하지는 않는다.")

    lines.extend([
        "",
        "## Recommendation",
        "- 운영 반영 후보는 global soft보다 theme-only soft가 더 안전하다. 다음 단계는 이 버전의 top20/near-top20 변화가 충분한지 검토하는 것이다.",
    ])
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    latest = load_latest()
    experiment = simulate(latest)
    experiment.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    export_compare(experiment)
    OUT_MD.write_text(build_markdown(experiment), encoding="utf-8")
    LOGGER.info("saved %s", OUT_CSV)
    LOGGER.info("saved %s", OUT_COMPARE_CSV)
    LOGGER.info("saved %s", OUT_MD)
    print(f"generated_files={[str(OUT_CSV), str(OUT_COMPARE_CSV), str(OUT_MD)]}")
    print("example=python python\\build_risk_penalty_theme_only_experiment.py")


if __name__ == "__main__":
    main()
