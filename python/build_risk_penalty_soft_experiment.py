import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUT_CSV = DATA_DIR / "ranking_final_risk_penalty_soft.csv"
OUT_COMPARE_CSV = DATA_DIR / "risk_penalty_soft_compare.csv"
OUT_MD = DATA_DIR / "risk_penalty_soft_experiment.md"

LOGGER = logging.getLogger("build_risk_penalty_soft_experiment")


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
    return latest.sort_values("rank_final").reset_index(drop=True)


def simulate_soft_penalty(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    base_core = (
        out["w_ret_base"] * out["ret_score"]
        + out["w_prob_base"] * out["prob_score"]
        + out["w_tech_base"] * out["tech_score"]
        + out["w_qual_base"] * out["qual_score"]
        + out["w_valuation_base"] * out["valuation_score"]
    )
    risk_weight_soft = out["w_risk_penalty"].where(~out["regime"].astype(str).str.lower().eq("neutral"), 0.45)
    out["risk_penalty_weight_exp"] = risk_weight_soft
    out["final_score_soft_penalty"] = base_core - risk_weight_soft * out["risk_penalty"]
    out["theme_score_effective_exp"] = out["theme_score"] * out["theme_confidence"]
    out["final_score_v3_soft_penalty"] = (1.0 - out["w_theme"]) * out["final_score_soft_penalty"] + out["w_theme"] * out["theme_score_effective_exp"]
    out["rank_final_soft_penalty"] = out["final_score_v3_soft_penalty"].rank(method="first", ascending=False).astype(int)
    out["rank_change_soft_penalty"] = out["rank_final"] - out["rank_final_soft_penalty"]
    out["score_diff_soft_penalty"] = out["final_score_v3_soft_penalty"] - out["final_score_v3"]
    return out.sort_values("rank_final_soft_penalty").reset_index(drop=True)


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
        "risk_penalty_weight_exp",
        "final_score_v3",
        "final_score_v3_soft_penalty",
        "score_diff_soft_penalty",
        "rank_final",
        "rank_final_soft_penalty",
        "rank_change_soft_penalty",
    ]].copy()
    compare.to_csv(OUT_COMPARE_CSV, index=False, encoding="utf-8-sig")
    return compare


def build_markdown(df: pd.DataFrame) -> str:
    baseline_top20 = df.loc[df["rank_final"] <= 20].copy()
    soft_top20 = df.loc[df["rank_final_soft_penalty"] <= 20].copy()
    entrants = df.loc[(df["rank_final"] > 20) & (df["rank_final_soft_penalty"] <= 20)].copy()
    exits = df.loc[(df["rank_final"] <= 20) & (df["rank_final_soft_penalty"] > 20)].copy()
    themed_top20_before = int(baseline_top20["dominant_theme"].str.strip().ne("").sum())
    themed_top20_after = int(soft_top20["dominant_theme"].str.strip().ne("").sum())
    near_top20_themed = df.loc[(df["rank_final"].between(21, 40)) & (df["dominant_theme"].str.strip().ne(""))].copy()
    near_to_top20 = near_top20_themed.loc[near_top20_themed["rank_final_soft_penalty"] <= 20].copy()
    top_lifters = df.sort_values(["rank_change_soft_penalty", "score_diff_soft_penalty"], ascending=[False, False]).head(15)

    lines = [
        "# Risk Penalty Soft Experiment",
        "",
        "## Setup",
        "- baseline: current `final_score_v3`",
        "- experiment: neutral regime `w_risk_penalty 0.65 -> 0.45`",
        "- production file is unchanged; this is a sidecar experiment only.",
        "",
        "## Top20 Comparison",
        f"- baseline_top20_theme_count={themed_top20_before}",
        f"- experiment_top20_theme_count={themed_top20_after}",
        f"- new_top20_entries={len(entrants)}",
        f"- top20_exits={len(exits)}",
        "",
        "## New Top20 Entries",
    ]
    if entrants.empty:
        lines.append("- none")
    else:
        for row in entrants.sort_values("rank_final_soft_penalty").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: baseline_rank={int(row.rank_final)}, exp_rank={int(row.rank_final_soft_penalty)}, "
                f"theme={row.dominant_theme or '(none)'}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Top20 Exits"])
    if exits.empty:
        lines.append("- none")
    else:
        for row in exits.sort_values("rank_final").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: baseline_rank={int(row.rank_final)}, exp_rank={int(row.rank_final_soft_penalty)}, "
                f"theme={row.dominant_theme or '(none)'}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Near-Top20 Themed Winners"])
    if near_to_top20.empty:
        lines.append("- none")
    else:
        for row in near_to_top20.sort_values("rank_final_soft_penalty").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: {int(row.rank_final)} -> {int(row.rank_final_soft_penalty)}, "
                f"theme={row.dominant_theme}, theme_score={float(row.theme_score):.2f}, "
                f"theme_confidence={float(row.theme_confidence):.3f}, risk_penalty={float(row.risk_penalty):.2f}"
            )

    lines.extend(["", "## Largest Rank Improvements"])
    for row in top_lifters.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: {int(row.rank_final)} -> {int(row.rank_final_soft_penalty)}, "
            f"rank_change={int(row.rank_change_soft_penalty)}, theme={row.dominant_theme or '(none)'}, "
            f"score_diff={float(row.score_diff_soft_penalty):.4f}, risk_penalty={float(row.risk_penalty):.2f}"
        )

    lines.extend([
        "",
        "## Interpretation",
        "- 이 실험은 테마 종목에만 유리하지 않다. high-penalty 비테마 종목도 같이 올라온다.",
        "- 따라서 바로 운영 반영하기보다, top20 변화가 원하는 방향인지 먼저 검토해야 한다.",
        "- 그래도 near-top20 테마 종목의 top20 진입 가능성을 확인하는 목적에는 유효하다.",
        "",
        "## Recommendation",
        "- 다음 단계는 이 실험 결과를 운영 랭킹과 나란히 검토한 뒤, `neutral regime only`로 제한 반영할지 결정하는 것이다.",
    ])
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    latest = load_latest()
    experiment = simulate_soft_penalty(latest)
    experiment.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    export_compare(experiment)
    OUT_MD.write_text(build_markdown(experiment), encoding="utf-8")
    LOGGER.info("saved %s", OUT_CSV)
    LOGGER.info("saved %s", OUT_COMPARE_CSV)
    LOGGER.info("saved %s", OUT_MD)
    print(f"generated_files={[str(OUT_CSV), str(OUT_COMPARE_CSV), str(OUT_MD)]}")
    print("example=python python\\build_risk_penalty_soft_experiment.py")


if __name__ == "__main__":
    main()
