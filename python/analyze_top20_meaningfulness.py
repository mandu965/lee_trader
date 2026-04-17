from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from payload_store import upsert_json_payload


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_RANKING_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_OUT_CSV = OUTPUT_DIR / "top20_meaningfulness_report.csv"
DEFAULT_OUT_MD = OUTPUT_DIR / "top20_meaningfulness_report.md"
DEFAULT_OUT_JSON = OUTPUT_DIR / "top20_meaningfulness_report.json"
DEFAULT_DATA_JSON = DATA_DIR / "top20_meaningfulness_report.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze whether latest Top20 names are meaningful and investable.")
    p.add_argument("--ranking-csv", type=Path, default=DEFAULT_RANKING_CSV)
    p.add_argument("--date", default="latest", help="YYYY-MM-DD or 'latest' (default).")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    return p.parse_args()


def _num(value: object) -> float | None:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return None
    return float(x)


def _fmt(value: object, digits: int = 2) -> str:
    x = _num(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [render_line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(render_line(row) for row in rendered)
    return "\n".join(lines)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def load_top_n(path: Path, target_date: str, top_n: int) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    if df["date"].isna().all():
        raise ValueError("ranking csv has no usable date column")

    latest_date = df["date"].max().strftime("%Y-%m-%d")
    use_date = latest_date if target_date == "latest" else target_date
    filtered = df.loc[df["date"].dt.strftime("%Y-%m-%d") == use_date].copy()
    if filtered.empty:
        raise ValueError(f"no rows found for date={use_date}")

    mixed_dates = sorted(d.strftime("%Y-%m-%d") for d in df["date"].dropna().dt.date.unique())
    score_col = "final_score_v3" if "final_score_v3" in filtered.columns else ("final_score_v2" if "final_score_v2" in filtered.columns else "final_score")
    filtered[score_col] = pd.to_numeric(filtered[score_col], errors="coerce")
    top = filtered.sort_values([score_col, "code"], ascending=[False, True]).head(top_n).copy()
    top["analysis_score_col"] = score_col

    meta = {
        "latest_date_in_file": latest_date,
        "analysis_date": use_date,
        "row_count_for_date": int(len(filtered)),
        "mixed_dates_detected": len(mixed_dates) > 1,
        "date_list_in_file": mixed_dates,
        "score_col": score_col,
    }
    return top, meta


def build_positive_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if (_num(row.get("ret_score")) or 0) >= 80:
        reasons.append("high_return_signal")
    if (_num(row.get("prob_score")) or 0) >= 80:
        reasons.append("high_top20_probability")
    if (_num(row.get("qual_score")) or 0) >= 75:
        reasons.append("healthy_quality")
    if (_num(row.get("safety_score")) or 0) >= 70:
        reasons.append("healthy_safety")
    if (_num(row.get("liquidity_score")) or 0) >= 60:
        reasons.append("healthy_liquidity")
    if (_num(row.get("tech_score")) or 0) >= 60:
        reasons.append("healthy_technical_trend")
    if (_num(row.get("theme_score")) or 0) >= 60 and (_num(row.get("theme_confidence")) or 0) >= 0.6:
        reasons.append("theme_supported")
    return reasons[:3]


def build_caution_reasons(row: pd.Series) -> list[str]:
    cautions: list[str] = []
    if (_num(row.get("qual_score")) or 100) < 35:
        cautions.append("low_quality")
    if (_num(row.get("safety_score")) or 100) < 35:
        cautions.append("low_safety")
    if (_num(row.get("liquidity_score")) or 100) < 20:
        cautions.append("low_liquidity")
    if (_num(row.get("tech_score")) or 100) < 25:
        cautions.append("weak_technical_trend")
    if (_num(row.get("risk_penalty")) or 0) >= 9:
        cautions.append("high_risk_penalty")
    if str(row.get("confidence_label") or "").strip().lower() in {"medium", "low", "experimental"}:
        cautions.append("confidence_needs_caution")
    return cautions[:3]


def meaningfulness_score(row: pd.Series) -> float:
    score = 0.0

    def add_component(value: float | None, *, strong: float, good: float, weak: float, very_weak: float, high_bonus: float, mid_bonus: float) -> None:
        nonlocal score
        if value is None:
            return
        if value >= strong:
            score += high_bonus
        elif value >= good:
            score += mid_bonus
        elif value < very_weak:
            score -= 2.0
        elif value < weak:
            score -= 1.0

    add_component(_num(row.get("ret_score")), strong=85, good=65, weak=45, very_weak=30, high_bonus=2.0, mid_bonus=1.0)
    add_component(_num(row.get("prob_score")), strong=85, good=65, weak=45, very_weak=30, high_bonus=2.0, mid_bonus=1.0)
    add_component(_num(row.get("qual_score")), strong=75, good=55, weak=50, very_weak=35, high_bonus=1.5, mid_bonus=0.75)
    add_component(_num(row.get("safety_score")), strong=75, good=55, weak=50, very_weak=35, high_bonus=1.5, mid_bonus=0.75)
    add_component(_num(row.get("liquidity_score")), strong=60, good=40, weak=35, very_weak=20, high_bonus=1.5, mid_bonus=0.75)
    add_component(_num(row.get("tech_score")), strong=60, good=45, weak=35, very_weak=25, high_bonus=1.0, mid_bonus=0.5)

    risk_penalty = _num(row.get("risk_penalty"))
    if risk_penalty is not None:
        if risk_penalty < 4:
            score += 1.0
        elif risk_penalty < 8:
            score += 0.5
        elif risk_penalty >= 12:
            score -= 2.0
        elif risk_penalty >= 8:
            score -= 1.0

    confidence_label = str(row.get("confidence_label") or "").strip().lower()
    if confidence_label == "high":
        score += 0.75
    elif confidence_label in {"medium", "low", "experimental"}:
        score -= 0.5

    theme_score = _num(row.get("theme_score"))
    theme_conf = _num(row.get("theme_confidence"))
    if theme_score is not None and theme_score >= 60 and theme_conf is not None and theme_conf >= 0.6:
        score += 0.5

    return round(score, 2)


def label_meaningfulness(row: pd.Series) -> tuple[str, str]:
    score = _num(row.get("meaningfulness_score")) or 0.0
    severe = {
        "low_quality": (_num(row.get("qual_score")) or 100) < 35,
        "low_safety": (_num(row.get("safety_score")) or 100) < 35,
        "low_liquidity": (_num(row.get("liquidity_score")) or 100) < 20,
    }
    severe_count = sum(bool(v) for v in severe.values())

    if score >= 6.0 and severe_count == 0:
        return "A", "core_candidate"
    if score >= 3.5 and severe_count == 0:
        return "B", "explainable_candidate"
    if score >= 1.0 and severe_count <= 1:
        return "C", "conditional_candidate"
    return "D", "caution_candidate"


def classify_style(row: pd.Series) -> str:
    ret_score = _num(row.get("ret_score")) or 0.0
    prob_score = _num(row.get("prob_score")) or 0.0
    qual_score = _num(row.get("qual_score")) or 0.0
    safety_score = _num(row.get("safety_score")) or 0.0
    liquidity_score = _num(row.get("liquidity_score")) or 0.0
    theme_score = _num(row.get("theme_score")) or 0.0

    if ret_score >= 80 and prob_score >= 80 and qual_score >= 65 and safety_score >= 60:
        return "balanced_leader"
    if ret_score >= 80 and prob_score >= 80 and (qual_score < 35 or safety_score < 35):
        return "aggressive_momentum"
    if qual_score >= 80 and safety_score >= 70 and liquidity_score >= 50:
        return "quality_defensive"
    if theme_score >= 60:
        return "theme_supported"
    return "mixed_profile"


def enrich(top: pd.DataFrame) -> pd.DataFrame:
    out = top.copy()
    out["meaningfulness_score"] = out.apply(meaningfulness_score, axis=1)
    labels = out.apply(label_meaningfulness, axis=1, result_type="expand")
    labels.columns = ["meaningfulness_grade", "meaningfulness_label"]
    out[labels.columns] = labels
    out["style_bucket"] = out.apply(classify_style, axis=1)
    out["positive_reasons"] = out.apply(lambda row: " / ".join(build_positive_reasons(row)), axis=1)
    out["caution_reasons"] = out.apply(lambda row: " / ".join(build_caution_reasons(row)), axis=1)
    out["theme_supported"] = out.apply(
        lambda row: bool((_num(row.get("theme_score")) or 0) >= 60 and (_num(row.get("theme_confidence")) or 0) >= 0.6),
        axis=1,
    )
    return out


def build_markdown(report: pd.DataFrame, meta: dict[str, object]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grade_counts = report["meaningfulness_grade"].value_counts().to_dict()
    label_counts = report["meaningfulness_label"].value_counts().to_dict()
    style_counts = report["style_bucket"].value_counts().to_dict()
    caution_counts = (
        report["caution_reasons"]
        .str.split(" / ")
        .explode()
        .loc[lambda s: s.notna() & s.ne("")]
        .value_counts()
        .head(10)
        .to_dict()
    )

    summary_rows = [
        ["analysis_date", meta["analysis_date"]],
        ["latest_date_in_file", meta["latest_date_in_file"]],
        ["row_count_for_date", meta["row_count_for_date"]],
        ["mixed_dates_detected", meta["mixed_dates_detected"]],
        ["top_n", len(report)],
        ["mean_meaningfulness_score", _fmt(report["meaningfulness_score"].mean())],
    ]

    result_rows = [
        [
            row["code"],
            row["name"],
            row["meaningfulness_grade"],
            row["meaningfulness_label"],
            _fmt(row["meaningfulness_score"]),
            row["style_bucket"],
            row["positive_reasons"] or "-",
            row["caution_reasons"] or "-",
        ]
        for _, row in report.iterrows()
    ]

    count_rows = [[k, v] for k, v in grade_counts.items()]
    style_rows = [[k, v] for k, v in style_counts.items()]
    caution_rows = [[k, v] for k, v in caution_counts.items()]

    lines = [
        "# Top20 Meaningfulness Report",
        "",
        f"- generated_at: {generated_at}",
        f"- ranking_csv: {DEFAULT_RANKING_CSV}",
        f"- analysis_date: {meta['analysis_date']}",
        "",
        "## Summary",
        "",
        _markdown_table(summary_rows, ["metric", "value"]),
        "",
        "## Grade Counts",
        "",
        _markdown_table(count_rows or [["(none)", 0]], ["grade", "count"]),
        "",
        "## Style Counts",
        "",
        _markdown_table(style_rows or [["(none)", 0]], ["style", "count"]),
        "",
    ]

    if caution_rows:
        lines.extend(
            [
                "## Main Cautions",
                "",
                _markdown_table(caution_rows, ["caution", "count"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Top20 Detail",
            "",
            _markdown_table(
                result_rows,
                ["code", "name", "grade", "label", "score", "style", "positive_reasons", "caution_reasons"],
            ),
            "",
            "## Interpretation",
            "",
            f"- grade distribution: {label_counts}",
            "- `A`: strong ret/prob with no major quality, safety, or liquidity damage",
            "- `B`: explainable setup with limited but non-zero caveats",
            "- `C`: workable score, but needs conditional review before action",
            "- `D`: caution bucket due to weak quality, weak safety, low liquidity, or similar issues",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    top, meta = load_top_n(args.ranking_csv, args.date, args.top_n)
    report = enrich(top)

    select_cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        str(meta["score_col"]),
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "safety_score",
        "liquidity_score",
        "risk_penalty",
        "theme_score",
        "dominant_theme",
        "theme_confidence",
        "confidence_label",
        "action_note",
        "meaningfulness_score",
        "meaningfulness_grade",
        "meaningfulness_label",
        "style_bucket",
        "theme_supported",
        "positive_reasons",
        "caution_reasons",
        "score_explain_summary",
    ]
    select_cols = [col for col in select_cols if col in report.columns]
    final_report = report[select_cols].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.data_json.parent.mkdir(parents=True, exist_ok=True)

    final_report.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    args.out_md.write_text(build_markdown(final_report, meta), encoding="utf-8")
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta,
        "grade_counts": final_report["meaningfulness_grade"].value_counts().to_dict(),
        "style_counts": final_report["style_bucket"].value_counts().to_dict(),
        "rows": final_report.to_dict(orient="records"),
    }
    serialized = json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False)
    args.out_json.write_text(serialized, encoding="utf-8")
    args.data_json.write_text(serialized, encoding="utf-8")
    upsert_json_payload(
        "top20_meaningfulness_report",
        json.loads(serialized),
        asof_date=meta.get("analysis_date"),
        generated_at=payload.get("generated_at"),
        source_path=args.out_json,
    )

    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.out_json}")
    print(f"saved: {args.data_json}")


if __name__ == "__main__":
    main()
