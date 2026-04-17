import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("data/ranking_final.csv")
TOP20_CSV = Path("outputs/top20_score_breakdown.csv")
SECTOR_CSV = Path("outputs/sector_score_summary.csv")
REPORT_MD = Path("outputs/ranking_trend_alignment.md")
CONFIDENCE_MD = Path("outputs/confidence_anomaly_report.md")
RULE_NOTE_MD = Path("outputs/interpretation_rule_note.md")
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


def _safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    sample = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sample) < 2:
        return float("nan")
    return float(sample[left].corr(sample[right]))


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_ranking() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"ranking CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def latest_slice(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"] == latest_date].copy()
    return latest.sort_values(["final_score", "rank_final"], ascending=[False, True])


def build_top20_breakdown(latest: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "code",
        "name",
        "sector",
        "market",
        "final_score",
        "rank_final",
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
        "confidence_score",
        "confidence_grade",
        "regime",
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "score_explain_summary",
    ]
    cols = [col for col in cols if col in latest.columns]
    return latest.head(TOP_N)[cols]


def classify_sector_style(sector: object) -> str:
    text = str(sector or "").strip().lower()
    if any(key in text for key in ["은행", "보험", "증권", "카드", "통신", "금융", "지주"]):
        return "defensive"
    if any(key in text for key in ["반도체", "인터넷", "게임", "엔터", "바이오", "소프트웨어", "2차전지", "it", "미디어"]):
        return "growth"
    if any(key in text for key in ["화학", "철강", "조선", "건설", "운송", "자동차", "기계", "에너지"]):
        return "cyclical"
    return "neutral"


def build_sector_summary(latest: pd.DataFrame, top20: pd.DataFrame) -> pd.DataFrame:
    universe_counts = latest["sector"].fillna("Unknown").value_counts(dropna=False)
    top20_counts = top20["sector"].fillna("Unknown").value_counts(dropna=False)
    sectors = sorted(set(universe_counts.index).union(set(top20_counts.index)))
    rows = []
    for sector in sectors:
        all_count = int(universe_counts.get(sector, 0))
        top_count = int(top20_counts.get(sector, 0))
        all_ratio = all_count / len(latest) if len(latest) else 0.0
        top_ratio = top_count / len(top20) if len(top20) else 0.0
        rows.append(
            {
                "sector": sector,
                "sector_style": classify_sector_style(sector),
                "universe_count": all_count,
                "universe_ratio": all_ratio,
                "top20_count": top_count,
                "top20_ratio": top_ratio,
                "overweight_ratio": top_ratio - all_ratio,
            }
        )
    return pd.DataFrame(rows).sort_values(["top20_count", "overweight_ratio", "sector"], ascending=[False, False, True])


def overlap_ratio(df: pd.DataFrame, score_col: str, top_n: int = TOP_N) -> float:
    final_set = set(df.sort_values(["final_score"], ascending=[False]).head(top_n)["code"])
    comp_set = set(df.sort_values([score_col], ascending=[False]).head(top_n)["code"])
    return len(final_set & comp_set) / float(top_n) if top_n else 0.0


def _metadata(df: pd.DataFrame) -> dict[str, str]:
    stat = INPUT_CSV.stat()
    latest_date = str(df["date"].dropna().max()) if "date" in df.columns and df["date"].notna().any() else "NA"
    score_formula_version = "NA"
    if "score_formula_version" in df.columns and df["score_formula_version"].notna().any():
        score_formula_version = str(df["score_formula_version"].dropna().iloc[0])
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_date": latest_date,
        "score_formula_version": score_formula_version,
        "source_ranking_file": f"{INPUT_CSV.name}; rows={len(df)}; modified_at={datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        "recomputed_from_current_code": "true",
    }


def determine_interpretation(metrics: dict[str, float]) -> tuple[str, list[str]]:
    ret_corr = metrics["ret_corr"]
    prob_corr = metrics["prob_corr"]
    tech_corr = metrics["tech_corr"]
    safety_corr_abs = abs(metrics["safety_corr"])
    risk_corr_abs = abs(metrics["risk_corr"])
    overlap_ret = metrics["overlap_ret"]
    overlap_prob = metrics["overlap_prob"]
    overlap_tech = metrics["overlap_tech"]

    reasons: list[str] = []

    conservative_balanced = (
        ret_corr >= 0.82
        and prob_corr >= 0.65
        and 0.40 <= risk_corr_abs <= 0.58
        and safety_corr_abs <= 0.38
        and overlap_ret >= 0.45
        and overlap_prob >= 0.40
    )
    if conservative_balanced:
        reasons.append("ret/prob corr가 높고 risk_penalty 상관은 중간 수준이며 safety 상관은 보조 수준입니다.")
        return "보수적 밸런스형, ret/prob 주도", reasons

    trend_driven = (
        ret_corr >= 0.82
        and tech_corr >= 0.10
        and overlap_ret >= 0.50
        and overlap_tech >= 0.25
        and risk_corr_abs < 0.45
    )
    if trend_driven:
        reasons.append("ret/tech 정렬이 강하고 penalty 지배력이 낮아 상방 추세 축 반영이 상대적으로 좋습니다.")
        return "상방 추세형, ret/tech 동반 주도", reasons

    defensive_bias = (
        safety_corr_abs >= 0.45
        and risk_corr_abs >= 0.58
        and overlap_ret < 0.45
        and overlap_prob < 0.40
    )
    if defensive_bias:
        reasons.append("safety 및 risk_penalty 축 영향이 과하고 ret/prob 정렬이 약합니다.")
        return "방어형 편향", reasons

    reasons.append("ret/prob/tech와 방어 축이 혼합돼 있어 어느 한쪽으로 단정하기 어렵습니다.")
    return "혼합형", reasons


def build_rule_note() -> str:
    rules = pd.DataFrame(
        [
            {
                "interpretation": "보수적 밸런스형, ret/prob 주도",
                "rule": "ret_corr >= 0.82, prob_corr >= 0.65, 0.40 <= abs(risk_corr) <= 0.58, abs(safety_corr) <= 0.38, overlap_ret >= 0.45, overlap_prob >= 0.40",
                "meaning": "ret/prob 축이 랭킹을 주도하지만 penalty와 safety가 완충 역할을 하는 상태",
            },
            {
                "interpretation": "상방 추세형, ret/tech 동반 주도",
                "rule": "ret_corr >= 0.82, tech_corr >= 0.10, overlap_ret >= 0.50, overlap_tech >= 0.25, abs(risk_corr) < 0.45",
                "meaning": "ret와 tech가 함께 살아 있고 penalty 영향이 상대적으로 낮은 상태",
            },
            {
                "interpretation": "방어형 편향",
                "rule": "abs(safety_corr) >= 0.45, abs(risk_corr) >= 0.58, overlap_ret < 0.45, overlap_prob < 0.40",
                "meaning": "방어 축과 penalty가 과도하게 강해 ret/prob 반영력이 낮은 상태",
            },
            {
                "interpretation": "혼합형",
                "rule": "위 세 조건 어디에도 명확히 속하지 않음",
                "meaning": "방향성이 섞여 있어 추가 점검이 필요한 상태",
            },
        ]
    )
    lines = [
        "# Interpretation Rule Note",
        "",
        "## 목적",
        "- `ranking_trend_alignment.md`의 interpretation 문구를 수치 조건 기반으로 판정하기 위한 규칙입니다.",
        "- 단순 문구 치환이 아니라 corr / overlap / penalty 민감도를 함께 봅니다.",
        "",
        "## Rule Table",
        _dataframe_to_markdown(rules),
        "",
        "## Notes",
        "- `보수적 밸런스형, ret/prob 주도`는 현재 운영 상태처럼 ret/prob가 주도하지만 risk_penalty와 safety가 보조적으로 작동하는 구간을 위해 추가한 판정입니다.",
        "- `tech_corr`가 소폭 개선돼도 overlap이 낮으면 여전히 `상방 추세형`으로 올리지 않습니다.",
        "- `risk_corr`는 절대값 기준으로 보고, 0.5 전후는 보수적 완충 수준으로 해석합니다.",
    ]
    return "\n".join(lines) + "\n"


def confidence_report(df: pd.DataFrame) -> str:
    confidence = pd.to_numeric(df["confidence_score"], errors="coerce")
    grade_dist = df["confidence_grade"].fillna("NA").astype(str).value_counts(dropna=False)
    reason_dist = df["confidence_reason"].fillna("NA").astype(str).value_counts(dropna=False).head(10)

    mostly_zero = float((confidence.fillna(0.0) <= 0.0).mean()) >= 0.8
    mostly_low = float(df["confidence_grade"].fillna("NA").astype(str).isin(["D", "E", "LOW", "low"]).mean()) >= 0.8
    lines = []
    lines.append("# Confidence Anomaly Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- confidence_score_min: {_fmt(confidence.min())}")
    lines.append(f"- confidence_score_max: {_fmt(confidence.max())}")
    lines.append(f"- confidence_score_mean: {_fmt(confidence.mean())}")
    lines.append(f"- confidence_score_null_ratio: {_fmt(confidence.isna().mean())}")
    lines.append("")
    lines.append("## Grade Distribution")
    for key, count in grade_dist.items():
        lines.append(f"- {key}: {int(count)}")
    lines.append("")
    lines.append("## Top Reasons")
    for key, count in reason_dist.items():
        lines.append(f"- {key}: {int(count)}")
    lines.append("")
    lines.append("## Diagnosis")
    if mostly_zero or mostly_low:
        lines.append("- Data-side anomaly detected: confidence_score or confidence_grade is collapsed for most names.")
        lines.append("- Candidate causes: missing flag generation order, fillna applied too early, confidence scaling compression.")
    else:
        lines.append("- Data-side confidence columns are not collapsed to 0.0 / LOW.")
        lines.append("- If the screen still shows 0.0 or LOW broadly, the first suspect is UI mapping or stale client-side interpretation.")
    return "\n".join(lines) + "\n"


def build_report(latest: pd.DataFrame, top20: pd.DataFrame, sector_summary: pd.DataFrame) -> str:
    components = ["ret_score", "prob_score", "qual_score", "tech_score", "safety_score", "liquidity_score", "risk_penalty", "confidence_score"]
    corr_lines = []
    corr_map = {}
    for col in components:
        corr_value = _safe_corr(latest, "final_score", col)
        corr_map[col] = corr_value
        corr_lines.append(f"- corr(final_score, {col}): {_fmt(corr_value)}")

    top20_means = top20[components].apply(pd.to_numeric, errors="coerce").mean()
    all_means = latest[components].apply(pd.to_numeric, errors="coerce").mean()
    mean_delta = (top20_means - all_means).sort_values(ascending=False)

    overlap_ret = overlap_ratio(latest, "ret_score")
    overlap_tech = overlap_ratio(latest, "tech_score")
    overlap_prob = overlap_ratio(latest, "prob_score")

    driver_counts = pd.concat([top20[col].dropna().astype(str) for col in ["score_driver_1", "score_driver_2", "score_driver_3"] if col in top20.columns], ignore_index=True).value_counts()
    drag_counts = pd.concat([top20[col].dropna().astype(str) for col in ["score_drag_1", "score_drag_2"] if col in top20.columns], ignore_index=True).value_counts()

    defensive_share = float((sector_summary.loc[sector_summary["sector_style"] == "defensive", "top20_count"].sum() / len(top20))) if len(top20) else 0.0
    growth_share = float((sector_summary.loc[sector_summary["sector_style"] == "growth", "top20_count"].sum() / len(top20))) if len(top20) else 0.0
    interpretation, interpretation_reasons = determine_interpretation(
        {
            "ret_corr": corr_map["ret_score"],
            "prob_corr": corr_map["prob_score"],
            "tech_corr": corr_map["tech_score"],
            "safety_corr": corr_map["safety_score"],
            "risk_corr": corr_map["risk_penalty"],
            "overlap_ret": overlap_ret,
            "overlap_prob": overlap_prob,
            "overlap_tech": overlap_tech,
        }
    )
    meta = _metadata(latest)

    lines = []
    lines.append("## Metadata")
    lines.extend(f"- {key}: {value}" for key, value in meta.items())
    lines.append("")
    lines.append("# Ranking Trend Alignment")
    lines.append("")
    lines.append("## 요약")
    lines.append(f"- latest_date: {latest['date'].max() if len(latest) else 'NA'}")
    lines.append(f"- interpretation: {interpretation}")
    lines.append(f"- top20 mean final_score: {_fmt(pd.to_numeric(top20['final_score'], errors='coerce').mean())}")
    lines.append(f"- top20 mean confidence_score: {_fmt(pd.to_numeric(top20['confidence_score'], errors='coerce').mean())}")
    lines.append("")
    lines.append("## top20 score breakdown 해석")
    for _, row in top20.head(10).iterrows():
        lines.append(
            f"- {row.get('code', 'NA')} rank={row.get('rank_final', 'NA')} final={_fmt(row.get('final_score'))} "
            f"drivers={[row.get('score_driver_1'), row.get('score_driver_2'), row.get('score_driver_3')]} "
            f"drags={[row.get('score_drag_1'), row.get('score_drag_2')]} "
            f"summary={row.get('score_explain_summary', '')}"
        )
    lines.append("")
    lines.append("## component dominance")
    lines.extend(corr_lines)
    lines.append("- top20 minus universe mean deltas")
    for key, value in mean_delta.items():
        lines.append(f"  - {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## final_score vs ret/prob/tech overlap")
    lines.append(f"- overlap(final_score top20, ret_score top20): {_fmt(overlap_ret)}")
    lines.append(f"- overlap(final_score top20, tech_score top20): {_fmt(overlap_tech)}")
    lines.append(f"- overlap(final_score top20, prob_score top20): {_fmt(overlap_prob)}")
    lines.append("")
    lines.append("## sector concentration summary")
    for _, row in sector_summary.head(12).iterrows():
        lines.append(
            f"- {row['sector']} style={row['sector_style']} top20_count={int(row['top20_count'])} "
            f"top20_ratio={_fmt(row['top20_ratio'])} overweight={_fmt(row['overweight_ratio'])}"
        )
    lines.append(f"- defensive_style_share_top20: {_fmt(defensive_share)}")
    lines.append(f"- growth_style_share_top20: {_fmt(growth_share)}")
    lines.append("")
    lines.append("## interpretation diagnostics")
    lines.append(f"- safety_score top20 mean vs universe mean: {_fmt(top20_means['safety_score'])} vs {_fmt(all_means['safety_score'])}")
    lines.append(f"- liquidity_score top20 mean vs universe mean: {_fmt(top20_means['liquidity_score'])} vs {_fmt(all_means['liquidity_score'])}")
    lines.append(f"- risk_penalty top20 mean vs universe mean: {_fmt(top20_means['risk_penalty'])} vs {_fmt(all_means['risk_penalty'])}")
    lines.append(f"- ret_score top20 mean vs universe mean: {_fmt(top20_means['ret_score'])} vs {_fmt(all_means['ret_score'])}")
    lines.append(f"- prob_score top20 mean vs universe mean: {_fmt(top20_means['prob_score'])} vs {_fmt(all_means['prob_score'])}")
    lines.append(f"- tech_score top20 mean vs universe mean: {_fmt(top20_means['tech_score'])} vs {_fmt(all_means['tech_score'])}")
    lines.append(f"- interpretation rationale: {' '.join(interpretation_reasons)}")
    if len(driver_counts):
        lines.append(f"- top20 dominant driver codes: {driver_counts.head(5).to_dict()}")
    if len(drag_counts):
        lines.append(f"- top20 dominant drag codes: {drag_counts.head(5).to_dict()}")
    lines.append("")
    lines.append("## confidence anomaly summary")
    lines.append(f"- confidence_score min/max/mean: {_fmt(pd.to_numeric(latest['confidence_score'], errors='coerce').min())} / {_fmt(pd.to_numeric(latest['confidence_score'], errors='coerce').max())} / {_fmt(pd.to_numeric(latest['confidence_score'], errors='coerce').mean())}")
    lines.append(f"- confidence_grade distribution: {latest['confidence_grade'].fillna('NA').astype(str).value_counts(dropna=False).to_dict()}")
    lines.append("")
    lines.append("## conclusion")
    if interpretation == "방어형 편향":
        lines.append("- 현재 랭킹은 내부 점수 구조 기준으로 방어형/리스크 회피형 성향이 강합니다.")
    elif interpretation == "상방 추세형, ret/tech 동반 주도":
        lines.append("- 현재 랭킹은 ret/tech 축의 상방 신호가 상대적으로 잘 반영된 상태입니다.")
    elif interpretation == "보수적 밸런스형, ret/prob 주도":
        lines.append("- 현재 랭킹은 ret/prob가 주도하고 risk_penalty와 safety가 완충하는 보수적 밸런스형에 가깝습니다.")
    else:
        lines.append("- 현재 랭킹은 상방 축과 방어 축이 혼합된 상태로, 특정 성격으로 단정하기는 이릅니다.")
    lines.append("")
    lines.append("## recommended next actions")
    lines.append("- ret_score 및 tech_score 상위와 final_score 상위의 overlap 임계치를 지속 모니터링합니다.")
    lines.append("- interpretation이 `보수적 밸런스형, ret/prob 주도`일 때는 tech overlap 개선 실험을 별도 추적합니다.")
    lines.append("- 방어형 편향이 다시 커지면 defensive 가중치와 risk_penalty 조합을 재점검합니다.")
    lines.append("- explain driver 분포가 과도하게 단조로운지 정기적으로 확인합니다.")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    df = load_ranking()
    latest = latest_slice(df)
    top20 = build_top20_breakdown(latest)
    sector_summary = build_sector_summary(latest, top20)

    TOP20_CSV.parent.mkdir(parents=True, exist_ok=True)
    top20.to_csv(TOP20_CSV, index=False, encoding="utf-8")
    sector_summary.to_csv(SECTOR_CSV, index=False, encoding="utf-8")
    REPORT_MD.write_text(build_report(latest, top20, sector_summary), encoding="utf-8")
    CONFIDENCE_MD.write_text(confidence_report(df), encoding="utf-8")
    RULE_NOTE_MD.write_text(build_rule_note(), encoding="utf-8")

    logging.info("Saved top20 breakdown: %s", TOP20_CSV.resolve())
    logging.info("Saved sector summary: %s", SECTOR_CSV.resolve())
    logging.info("Saved ranking trend report: %s", REPORT_MD.resolve())
    logging.info("Saved confidence anomaly report: %s", CONFIDENCE_MD.resolve())
    logging.info("Saved interpretation rule note: %s", RULE_NOTE_MD.resolve())


if __name__ == "__main__":
    main()
