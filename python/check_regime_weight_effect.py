import logging
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("data/ranking_final.csv")
OUTPUT_MD = Path("outputs/regime_weight_effect.md")
TOP_N = 20


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _fmt(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def load_ranking() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"ranking CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    setup_logging()
    df = load_ranking()
    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"] == latest_date].copy().sort_values(["final_score"], ascending=[False])
    top20 = latest.head(TOP_N).copy()

    regime = str(latest["regime"].dropna().iloc[0]) if "regime" in latest.columns and latest["regime"].notna().any() else "NA"
    regime_reason = str(latest["regime_reason"].dropna().iloc[0]) if "regime_reason" in latest.columns and latest["regime_reason"].notna().any() else "NA"
    weight_profile = str(latest["weight_profile"].dropna().iloc[0]) if "weight_profile" in latest.columns and latest["weight_profile"].notna().any() else "NA"

    contribution_cols = [
        "score_contribution_ret",
        "score_contribution_prob",
        "score_contribution_tech",
        "score_contribution_qual",
        "score_contribution_safety",
        "score_contribution_liquidity",
        "score_contribution_risk",
    ]
    top20_mean = top20[contribution_cols].apply(pd.to_numeric, errors="coerce").mean()
    offensive = top20_mean[["score_contribution_ret", "score_contribution_prob", "score_contribution_tech"]].sum()
    defensive = top20_mean[["score_contribution_qual", "score_contribution_safety", "score_contribution_liquidity"]].sum()

    lines = []
    lines.append("# Regime Weight Effect")
    lines.append("")
    lines.append("## 요약")
    lines.append(f"- latest_date: {latest_date}")
    lines.append(f"- regime: {regime}")
    lines.append(f"- weight_profile: {weight_profile}")
    lines.append("")
    lines.append("## detected regime")
    lines.append(f"- {regime}")
    lines.append("")
    lines.append("## regime reason")
    lines.append(f"- {regime_reason}")
    lines.append("")
    lines.append("## weight profile")
    lines.append(f"- {weight_profile}")
    lines.append("")
    lines.append("## top20 contribution summary")
    for col in contribution_cols:
        lines.append(f"- {col}: {_fmt(top20_mean[col])}")
    lines.append("")
    lines.append("## offensive vs defensive contribution balance")
    lines.append(f"- offensive_contribution_sum(ret+prob+tech): {_fmt(offensive)}")
    lines.append(f"- defensive_contribution_sum(qual+safety+liquidity): {_fmt(defensive)}")
    lines.append(f"- risk_contribution_mean: {_fmt(top20_mean['score_contribution_risk'])}")
    lines.append("")
    lines.append("## interpretation")
    if offensive > defensive:
        lines.append("- 현재 regime 가중치는 top20에서 공격형 기여가 방어형 기여보다 크게 작동했습니다.")
    else:
        lines.append("- 현재 regime 가중치는 top20에서 방어형 기여가 공격형 기여보다 크게 작동했습니다.")
    lines.append("")
    lines.append("## remaining limitations")
    lines.append("- 1차 regime는 bull / defensive 2분기만 지원합니다.")
    lines.append("- breadth_20d는 현재 단면의 close_over_ma20 비율 프록시를 사용합니다.")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved regime weight effect report: %s", OUTPUT_MD.resolve())


if __name__ == "__main__":
    main()
