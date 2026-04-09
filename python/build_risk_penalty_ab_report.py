import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUTPUT_MD = DATA_DIR / "risk_penalty_ab_report.md"


LOGGER = logging.getLogger("build_risk_penalty_ab_report")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest() -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV, dtype={"code": str}, low_memory=False)
    df["date"] = df["date"].astype(str)
    latest = df.loc[df["date"] == df["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    for col in [
        "rank_final",
        "final_score",
        "final_score_v3",
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
        "w_base_v2",
    ]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce").fillna(0.0)
    return latest.sort_values("rank_final").reset_index(drop=True)


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_core = (
        out["w_ret_base"] * out["ret_score"]
        + out["w_prob_base"] * out["prob_score"]
        + out["w_tech_base"] * out["tech_score"]
        + out["w_qual_base"] * out["qual_score"]
        + out["w_valuation_base"] * out["valuation_score"]
    )
    out["baseline_recalc"] = base_core - out["w_risk_penalty"] * out["risk_penalty"]

    soft_w = out["w_risk_penalty"].where(~out["regime"].astype(str).str.lower().eq("neutral"), 0.45)
    out["ab_soft_penalty"] = base_core - soft_w * out["risk_penalty"]

    capped_penalty = out["risk_penalty"].clip(upper=12.0)
    out["ab_cap12_penalty"] = base_core - out["w_risk_penalty"] * capped_penalty

    theme_effective = out["theme_score"] * out["theme_confidence"]
    out["ab_soft_penalty_v3"] = (1.0 - out["w_theme"]) * out["ab_soft_penalty"] + out["w_theme"] * theme_effective
    out["ab_cap12_penalty_v3"] = (1.0 - out["w_theme"]) * out["ab_cap12_penalty"] + out["w_theme"] * theme_effective
    out["baseline_rank_v3"] = out["final_score_v3"].rank(method="first", ascending=False).astype(int)
    out["ab_soft_rank_v3"] = out["ab_soft_penalty_v3"].rank(method="first", ascending=False).astype(int)
    out["ab_cap12_rank_v3"] = out["ab_cap12_penalty_v3"].rank(method="first", ascending=False).astype(int)
    out["soft_rank_change"] = out["baseline_rank_v3"] - out["ab_soft_rank_v3"]
    out["cap12_rank_change"] = out["baseline_rank_v3"] - out["ab_cap12_rank_v3"]
    out["is_themed"] = out["dominant_theme"].str.strip().ne("")
    return out


def build_markdown(df: pd.DataFrame) -> str:
    themed_near = df.loc[(df["baseline_rank_v3"].between(21, 40)) & (df["is_themed"])].copy()
    top20_baseline = df.loc[df["baseline_rank_v3"] <= 20].copy()
    top20_soft = df.loc[df["ab_soft_rank_v3"] <= 20].copy()
    top20_cap = df.loc[df["ab_cap12_rank_v3"] <= 20].copy()

    soft_lifters = df.sort_values(["soft_rank_change", "theme_score"], ascending=[False, False]).head(10)
    cap_lifters = df.sort_values(["cap12_rank_change", "theme_score"], ascending=[False, False]).head(10)

    soft_near_into_top20 = themed_near.loc[themed_near["ab_soft_rank_v3"] <= 20].copy()
    cap_near_into_top20 = themed_near.loc[themed_near["ab_cap12_rank_v3"] <= 20].copy()

    lines = [
        "# Risk Penalty A/B Report",
        "",
        "## Setup",
        "- baseline: current ranking_final final_score_v3",
        "- A/B #1: neutral regime `w_risk_penalty` 0.65 -> 0.45",
        "- A/B #2: keep weight, but clamp `risk_penalty` to max 12.0",
        "- note: `safety_score` is not a direct positive term in current final_score. A safety-floor tweak alone does not change ranking unless score structure changes.",
        "",
        "## Key Finding",
    ]

    if not soft_near_into_top20.empty or not cap_near_into_top20.empty:
        lines.append("- risk_penalty 완화만으로도 일부 near-top20 테마 종목의 top20 진입 가능성이 생긴다.")
    else:
        lines.append("- risk_penalty를 완화해도 top20 구조 변화는 제한적이다. 즉 penalty는 병목 중 하나지만 단독 원인은 아니다.")

    lines.extend([
        "",
        "## Near-Top20 Themed Candidates",
    ])
    if themed_near.empty:
        lines.append("- none")
    else:
        for row in themed_near[["code", "name", "dominant_theme", "baseline_rank_v3", "ab_soft_rank_v3", "ab_cap12_rank_v3", "risk_penalty", "theme_score", "theme_confidence"]].sort_values("baseline_rank_v3").itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: theme={row.dominant_theme}, "
                f"baseline={int(row.baseline_rank_v3)}, soft={int(row.ab_soft_rank_v3)}, cap12={int(row.ab_cap12_rank_v3)}, "
                f"risk_penalty={float(row.risk_penalty):.2f}, theme_score={float(row.theme_score):.2f}, theme_confidence={float(row.theme_confidence):.3f}"
            )

    lines.extend([
        "",
        "## Top 10 Rank Improvements: Soft Weight",
    ])
    for row in soft_lifters[["code", "name", "dominant_theme", "baseline_rank_v3", "ab_soft_rank_v3", "soft_rank_change", "risk_penalty"]].itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: theme={row.dominant_theme or '(none)'}, "
            f"{int(row.baseline_rank_v3)} -> {int(row.ab_soft_rank_v3)}, "
            f"rank_change={int(row.soft_rank_change)}, risk_penalty={float(row.risk_penalty):.2f}"
        )

    lines.extend([
        "",
        "## Top 10 Rank Improvements: Cap 12",
    ])
    for row in cap_lifters[["code", "name", "dominant_theme", "baseline_rank_v3", "ab_cap12_rank_v3", "cap12_rank_change", "risk_penalty"]].itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: theme={row.dominant_theme or '(none)'}, "
            f"{int(row.baseline_rank_v3)} -> {int(row.ab_cap12_rank_v3)}, "
            f"rank_change={int(row.cap12_rank_change)}, risk_penalty={float(row.risk_penalty):.2f}"
        )

    lines.extend([
        "",
        "## Top20 Comparison",
        f"- baseline_top20_theme_count={int(top20_baseline['is_themed'].sum())}",
        f"- soft_weight_top20_theme_count={int(top20_soft['is_themed'].sum())}",
        f"- cap12_top20_theme_count={int(top20_cap['is_themed'].sum())}",
        "",
        "## Interpretation",
        "- `safety_score` 하한 보정은 현재 구조에서 직접 효과가 없다. safety는 참고 진단 지표이지, final_score 가산 항목이 아니다.",
        "- 실질 조정 포인트는 `risk_penalty`다.",
        "- 다만 risk_penalty만 완화해도 모든 문제가 해결되지는 않는다. ret/prob/quality gap이 여전히 남는다.",
        "",
        "## Recommendation",
        "- 다음 실험은 `neutral regime risk_penalty 완화`를 우선하고, safety는 구조 개편이 필요할 때만 별도 다룰 것.",
    ])
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    latest = load_latest()
    simulated = simulate(latest)
    OUTPUT_MD.write_text(build_markdown(simulated), encoding="utf-8")
    LOGGER.info("saved %s", OUTPUT_MD)
    print(f"generated_files={[str(OUTPUT_MD)]}")
    print("example=python python\\build_risk_penalty_ab_report.py")


if __name__ == "__main__":
    main()
