import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUTPUT_MD = DATA_DIR / "quality_safety_penalty_diagnostic.md"

LOGGER = logging.getLogger("build_quality_safety_penalty_diagnostic")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_latest() -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV, dtype={"code": str}, low_memory=False)
    df["date"] = df["date"].astype(str)
    latest = df.loc[df["date"] == df["date"].max()].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    numeric_cols = [
        "rank_final",
        "quality_score",
        "qual_score",
        "quality_factor_count",
        "quality_missing_ratio",
        "quality_score_confidence",
        "vol_20_pct",
        "vol_60_pct",
        "safety_score",
        "pred_mdd_mix",
        "risk_penalty",
        "ret_score",
        "prob_score",
        "theme_score",
        "theme_confidence",
    ]
    for col in numeric_cols:
        df_col = latest.get(col)
        latest[col] = pd.to_numeric(df_col, errors="coerce").fillna(0.0)
    return latest.sort_values("rank_final").reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    cols = [
        "quality_score",
        "qual_score",
        "quality_factor_count",
        "quality_missing_ratio",
        "quality_score_confidence",
        "vol_20_pct",
        "vol_60_pct",
        "safety_score",
        "pred_mdd_mix",
        "risk_penalty",
        "ret_score",
        "prob_score",
        "theme_score",
        "theme_confidence",
    ]
    out = {c: float(df[c].mean()) if not df.empty else 0.0 for c in cols}
    out["count"] = int(len(df))
    return out


def build_markdown(latest: pd.DataFrame) -> str:
    non_theme_top20 = latest.loc[(latest["rank_final"] <= 20) & (latest["dominant_theme"].str.strip() == "")].copy()
    themed_21_40 = latest.loc[
        (latest["rank_final"] >= 21) & (latest["rank_final"] <= 40) & (latest["dominant_theme"].str.strip() != "")
    ].copy()

    top = describe(non_theme_top20)
    near = describe(themed_21_40)

    lines = [
        "# Quality / Safety / Risk Penalty Diagnostic",
        "",
        "## Scope",
        f"- non_theme_top20_count={top['count']}",
        f"- themed_rank21_40_count={near['count']}",
        "",
        "## Average Comparison",
        f"- quality_score: top20={top['quality_score']:.2f}, themed_21_40={near['quality_score']:.2f}, gap={top['quality_score']-near['quality_score']:+.2f}",
        f"- qual_score: top20={top['qual_score']:.2f}, themed_21_40={near['qual_score']:.2f}, gap={top['qual_score']-near['qual_score']:+.2f}",
        f"- quality_factor_count: top20={top['quality_factor_count']:.2f}, themed_21_40={near['quality_factor_count']:.2f}",
        f"- quality_missing_ratio: top20={top['quality_missing_ratio']:.2f}, themed_21_40={near['quality_missing_ratio']:.2f}",
        f"- quality_score_confidence: top20={top['quality_score_confidence']:.2f}, themed_21_40={near['quality_score_confidence']:.2f}",
        f"- vol_20_pct: top20={top['vol_20_pct']:.2f}, themed_21_40={near['vol_20_pct']:.2f}",
        f"- vol_60_pct: top20={top['vol_60_pct']:.2f}, themed_21_40={near['vol_60_pct']:.2f}",
        f"- safety_score: top20={top['safety_score']:.2f}, themed_21_40={near['safety_score']:.2f}, gap={top['safety_score']-near['safety_score']:+.2f}",
        f"- pred_mdd_mix: top20={top['pred_mdd_mix']:.4f}, themed_21_40={near['pred_mdd_mix']:.4f}, gap={top['pred_mdd_mix']-near['pred_mdd_mix']:+.4f}",
        f"- risk_penalty: top20={top['risk_penalty']:.2f}, themed_21_40={near['risk_penalty']:.2f}, gap={top['risk_penalty']-near['risk_penalty']:+.2f}",
        "",
        "## Formula Reading",
        "- `qual_score`는 `quality_score`의 날짜별 percentile rank다. 지금 차이는 missing 문제보다 원본 `quality_score` 수준 차이에서 나온다.",
        "- `safety_score`는 `100 - vol_20_pct`, `100 - vol_60_pct` 평균이다. 즉 변동성이 높은 성장/테마주는 구조적으로 불리하다.",
        "- `risk_penalty`는 `pred_mdd_mix = 0.6*|pred_mdd_60d| + 0.4*|pred_mdd_90d|` 기반 절대 패널티다.",
        "- 현재 neutral 체계에서는 `risk_penalty`가 가중치 0.65로 final_score에서 차감된다.",
        "",
        "## Diagnosis",
    ]

    if top["quality_score"] - near["quality_score"] > 15:
        lines.append("- `qual_score`는 테마 성장주에 다소 불리하게 작동한다. 다만 missing 때문이 아니라 원본 quality 자체가 낮다.")
    if top["safety_score"] - near["safety_score"] > 25:
        lines.append("- `safety_score`는 성장/테마주 변동성을 강하게 벌점화한다. 현재 gap이 매우 크다.")
    if near["risk_penalty"] - top["risk_penalty"] > 3:
        lines.append("- `risk_penalty`도 near-top20 테마주에 더 무겁게 적용된다. drawdown 예측이 theme overlay 효과를 상당 부분 상쇄한다.")
    if near["quality_missing_ratio"] <= top["quality_missing_ratio"]:
        lines.append("- quality 쪽은 데이터 결측 문제보다 점수 정의 문제가 더 크다.")

    lines.extend([
        "",
        "## Practical Conclusion",
        "- 지금은 `theme_weight`를 더 올리기보다, `safety_score`와 `risk_penalty`가 성장/테마주를 과도하게 누르는지 먼저 완화 실험을 하는 편이 맞다.",
        "- 특히 점검 우선순위는 `risk_penalty 구간함수`와 `safety_score의 상대 percentile 구조`다.",
        "",
        "## Next Action",
        "- neutral regime에서 `risk_penalty` 완화안과 `safety_score` 하한 보정안의 A/B 실험 리포트를 만들 것.",
    ])
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    latest = load_latest()
    OUTPUT_MD.write_text(build_markdown(latest), encoding="utf-8")
    LOGGER.info("saved %s", OUTPUT_MD)
    print(f"generated_files={[str(OUTPUT_MD)]}")
    print("example=python python\\build_quality_safety_penalty_diagnostic.py")


if __name__ == "__main__":
    main()
