from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RANKING_FINAL_CSV = BASE_DIR / "data" / "ranking_final.csv"
OUTPUT_CSV = BASE_DIR / "data" / "theme_contribution_report.csv"
DEBUG_TOP20_CSV = BASE_DIR / "data" / "theme_contribution_debug_top20.csv"
VALIDATION_REPORT_MD = BASE_DIR / "data" / "theme_contribution_validation_report.md"
TOP_N = 20


def _to_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _safe_parse_explain(value: object) -> tuple[dict[str, object], bool]:
    if isinstance(value, dict):
        return value, True
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}, True

    text = str(value).strip()
    if not text:
        return {}, True

    try:
        parsed = json.loads(text)
        return (parsed, True) if isinstance(parsed, dict) else ({}, False)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return (parsed, True) if isinstance(parsed, dict) else ({}, False)
    except Exception:
        return {}, False


def _resolve_score_column(df: pd.DataFrame) -> str:
    for column in ("final_score_v2", "final_score", "score"):
        if column in df.columns:
            return column
    raise ValueError("missing all ranking score columns: final_score_v2, final_score, score")


def _resolve_base_score(row: pd.Series, explain_payload: dict[str, object]) -> tuple[float | None, str]:
    value = _to_float(row.get("base_score"))
    if value is not None:
        return value, "base_score_column"

    value = _to_float(explain_payload.get("base_score"))
    if value is not None:
        return value, "explain.base_score"

    final_score = _to_float(row.get("final_score"))
    final_score_v2 = _to_float(row.get("final_score_v2"))
    theme_score = _to_float(row.get("theme_score"))
    theme_weight = _to_float(row.get("theme_weight"))
    if final_score_v2 is not None and theme_score is not None and theme_weight is not None and 0.0 <= theme_weight < 1.0:
        reconstructed = (final_score_v2 - (theme_weight * theme_score)) / (1.0 - theme_weight)
        return reconstructed, "reconstructed_from_final_score_v2"
    if final_score is not None:
        return final_score, "final_score_fallback"
    return None, "unavailable"


def _resolve_applied_theme_weight(row: pd.Series, explain_payload: dict[str, object]) -> tuple[float, str]:
    value = _to_float(row.get("theme_weight"))
    if value is not None:
        return value, "theme_weight_column"
    value = _to_float(explain_payload.get("theme_weight"))
    if value is not None:
        return value, "explain.theme_weight"
    return 0.0, "zero_fallback"


def _extract_contribution(row: pd.Series, explain_payload: dict[str, object]) -> tuple[float | None, str]:
    value = _to_float(row.get("theme_contribution"))
    if value is not None:
        return value, "theme_contribution_column"

    theme_payload = explain_payload.get("theme")
    if isinstance(theme_payload, dict):
        value = _to_float(theme_payload.get("contribution"))
        if value is not None:
            return value, "explain.theme.contribution"

    weight, _ = _resolve_applied_theme_weight(row, explain_payload)
    theme_score = _to_float(row.get("theme_score"))
    if theme_score is not None:
        return float(weight) * float(theme_score), "fallback_weight_times_theme_score"

    base_score, _ = _resolve_base_score(row, explain_payload)
    final_score = _to_float(row.get("final_score_v2"))
    if final_score is None:
        final_score = _to_float(row.get("final_score"))
    if final_score is not None and base_score is not None:
        return float(final_score) - ((1.0 - float(weight)) * float(base_score)), "fallback_final_minus_base_component"

    return None, "unavailable"


def main() -> int:
    if not RANKING_FINAL_CSV.exists():
        print(f"FILE_ERROR: required input not found: {RANKING_FINAL_CSV}")
        return 1

    try:
        df = pd.read_csv(RANKING_FINAL_CSV, low_memory=False)
    except Exception as exc:
        print(f"PARSE_ERROR: failed to read ranking file ({exc})")
        return 1

    try:
        score_column = _resolve_score_column(df)
    except ValueError as exc:
        print(f"PARSE_ERROR: {exc}")
        return 1

    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    out["ticker"] = out.get("ticker", out.get("code", "")).astype(str).str.zfill(6)
    out["name"] = out.get("name", "").fillna("").astype(str)
    out["dominant_theme"] = out.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    out.loc[out["dominant_theme"].str.strip() == "", "dominant_theme"] = "(none)"
    out[score_column] = pd.to_numeric(out.get(score_column), errors="coerce")
    latest_date = out["date"].max()
    if pd.notna(latest_date):
        out = out[out["date"] == latest_date].copy()
    out = out.sort_values([score_column, "ticker"], ascending=[False, True]).head(TOP_N).reset_index(drop=True)
    out["rank"] = out.index + 1

    explain_parse_failures = 0
    fallback_usage_count = 0
    rows: list[dict[str, object]] = []
    for row in out.itertuples(index=False):
        series = pd.Series(row._asdict())
        explain_payload, parse_ok = _safe_parse_explain(series.get("explain"))
        explain_parse_failures += int(not parse_ok)
        base_score, base_source = _resolve_base_score(series, explain_payload)
        applied_theme_weight, weight_source = _resolve_applied_theme_weight(series, explain_payload)
        extracted_contribution, source_used = _extract_contribution(series, explain_payload)
        fallback_usage_count += int(source_used.startswith("fallback"))
        theme_score = _to_float(series.get("theme_score")) or 0.0
        final_score = _to_float(series.get(score_column))
        expected_theme_contribution = applied_theme_weight * theme_score
        if extracted_contribution is None and final_score is not None and base_score is not None:
            extracted_contribution = final_score - ((1.0 - applied_theme_weight) * base_score)

        rows.append(
            {
                "rank": int(series.get("rank")),
                "ticker": str(series.get("ticker")),
                "name": str(series.get("name")),
                "final_score": final_score,
                "base_score": base_score,
                "theme_score": theme_score,
                "applied_theme_weight": applied_theme_weight,
                "expected_theme_contribution": expected_theme_contribution,
                "extracted_theme_contribution": extracted_contribution,
                "contribution_source_used": source_used,
                "base_score_source": base_source,
                "theme_weight_source": weight_source,
                "dominant_theme": str(series.get("dominant_theme")),
            }
        )

    debug_df = pd.DataFrame(rows)
    positive_count = int(pd.to_numeric(debug_df["extracted_theme_contribution"], errors="coerce").fillna(0.0).gt(0).sum())
    zero_count = int(pd.to_numeric(debug_df["extracted_theme_contribution"], errors="coerce").fillna(0.0).eq(0).sum())
    source_breakdown = debug_df["contribution_source_used"].value_counts(dropna=False).to_dict()

    DEBUG_TOP20_CSV.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(DEBUG_TOP20_CSV, index=False, encoding="utf-8-sig")

    report_df = (
        debug_df.groupby("dominant_theme", dropna=False)
        .agg(
            top20_count=("ticker", "size"),
            avg_contribution=("extracted_theme_contribution", "mean"),
            total_contribution=("extracted_theme_contribution", "sum"),
            tickers=("ticker", lambda s: ",".join(pd.Series(s).astype(str).tolist())),
            source_used=("contribution_source_used", lambda s: ",".join(sorted(set(pd.Series(s).astype(str).tolist())))),
        )
        .reset_index()
        .rename(columns={"dominant_theme": "theme"})
        .sort_values(["top20_count", "total_contribution", "theme"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    report_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# Theme Contribution Validation",
        "",
        f"- top20 rows loaded: {len(debug_df)}",
        f"- positive contribution row count: {positive_count}",
        f"- zero contribution row count: {zero_count}",
        f"- explain parse failures: {explain_parse_failures}",
        f"- fallback usage count: {fallback_usage_count}",
        f"- source breakdown: {source_breakdown}",
        "",
        "## Files",
        f"- debug_top20: {DEBUG_TOP20_CSV}",
        f"- contribution_report: {OUTPUT_CSV}",
    ]
    VALIDATION_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"top20 rows loaded: {len(debug_df)}")
    print(f"positive contribution row count: {positive_count}")
    print(f"zero contribution row count: {zero_count}")
    print(f"source breakdown: {source_breakdown}")
    print(f"fallback usage count: {fallback_usage_count}")
    print(f"saved file paths: {OUTPUT_CSV}, {DEBUG_TOP20_CSV}, {VALIDATION_REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
