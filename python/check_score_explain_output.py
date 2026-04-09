import json
import logging
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("data/ranking_final.csv")
OUTPUT_MD = Path("outputs/score_explain_diagnostics.md")
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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def ensure_columns(df: pd.DataFrame) -> None:
    required = [
        "score_explain_summary",
        "score_explain_strengths",
        "score_explain_risks",
        "score_explain_confidence",
        "score_explain_regime",
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "score_explain_json",
        "final_score",
        "confidence_score",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")


def _value_counts(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    values = pd.concat([df[col].dropna().astype(str) for col in cols if col in df.columns], ignore_index=True)
    if values.empty:
        return pd.Series(dtype=int)
    return values.value_counts()


def _json_valid_ratio(series: pd.Series) -> float:
    valid = 0
    total = 0
    for value in series.fillna(""):
        text = str(value).strip()
        if not text:
            total += 1
            continue
        total += 1
        try:
            json.loads(text)
            valid += 1
        except Exception:
            pass
    return 0.0 if total == 0 else valid / total


def build_markdown(df: pd.DataFrame) -> str:
    driver_dist = _value_counts(df, ["score_driver_1", "score_driver_2", "score_driver_3"])
    drag_dist = _value_counts(df, ["score_drag_1", "score_drag_2"])
    top20 = df.sort_values(["final_score"], ascending=[False]).head(TOP_N)
    mismatch = df.loc[(pd.to_numeric(df["final_score"], errors="coerce") >= 60.0) & (pd.to_numeric(df["confidence_score"], errors="coerce") < 55.0)].copy()
    mismatch = mismatch.sort_values(["final_score", "confidence_score"], ascending=[False, True]).head(10)

    lines: list[str] = []
    lines.append("# Score Explain Diagnostics")
    lines.append("")
    lines.append("## 요약")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- explain_summary_null_ratio: {_fmt(df['score_explain_summary'].isna().mean())}")
    lines.append(f"- explain_json_valid_ratio: {_fmt(_json_valid_ratio(df['score_explain_json']))}")
    lines.append("")
    lines.append("## explain column coverage")
    for col in [
        "score_explain_summary",
        "score_explain_strengths",
        "score_explain_risks",
        "score_explain_confidence",
        "score_explain_regime",
        "score_driver_1",
        "score_driver_2",
        "score_driver_3",
        "score_drag_1",
        "score_drag_2",
        "score_explain_json",
    ]:
        lines.append(f"- {col}: null_ratio={_fmt(df[col].isna().mean())}")
    lines.append("")
    lines.append("## driver distribution")
    if len(driver_dist):
        for key, count in driver_dist.head(15).items():
            lines.append(f"- {key}: {int(count)}")
    else:
        lines.append("- no drivers found")
    lines.append("")
    lines.append("## drag distribution")
    if len(drag_dist):
        for key, count in drag_dist.head(15).items():
            lines.append(f"- {key}: {int(count)}")
    else:
        lines.append("- no drags found")
    lines.append("")
    lines.append("## confidence-related explanation summary")
    lines.append(f"- confidence_explain_null_ratio: {_fmt(df['score_explain_confidence'].isna().mean())}")
    lines.append(f"- low_confidence_drag_count: {int((df[['score_drag_1', 'score_drag_2']].fillna('') == 'low_confidence').sum().sum())}")
    lines.append("")
    lines.append("## top20 explain sample")
    for _, row in top20.iterrows():
        lines.append(
            f"- {row.get('date', 'NA')} {row.get('code', 'NA')} final={_fmt(row.get('final_score'))} "
            f"summary={row.get('score_explain_summary', '')} "
            f"drivers={[row.get('score_driver_1'), row.get('score_driver_2'), row.get('score_driver_3')]} "
            f"drags={[row.get('score_drag_1'), row.get('score_drag_2')]}"
        )
    lines.append("")
    lines.append("## mismatch case review")
    if len(mismatch):
        for _, row in mismatch.iterrows():
            lines.append(
                f"- {row.get('date', 'NA')} {row.get('code', 'NA')} final={_fmt(row.get('final_score'))} "
                f"confidence={_fmt(row.get('confidence_score'))} summary={row.get('score_explain_summary', '')}"
            )
    else:
        lines.append("- no high-final low-confidence explain cases found in the current output")
    lines.append("")
    lines.append("## interpretation")
    lines.append("- Explain text is generated from actual component scores, risk penalty, confidence, and regime fields.")
    lines.append("- Driver and drag codes are intended for UI/API consumption, while summary fields are human-readable.")
    lines.append("")
    lines.append("## remaining limitations")
    lines.append("- Rules are deterministic and intentionally simple, so nuance beyond the tracked components is not described.")
    lines.append("- Summary phrasing depends on threshold buckets and may need tuning after UI review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    df = load_ranking()
    ensure_columns(df)
    report = build_markdown(df)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    logging.info("Saved score explain diagnostics: %s", OUTPUT_MD.resolve())


if __name__ == "__main__":
    main()
