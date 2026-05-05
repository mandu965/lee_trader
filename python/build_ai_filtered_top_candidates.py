from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_INPUT_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_ENTRY_QUALITY_CSV = DATA_DIR / "ai_entry_quality_score.csv"
DEFAULT_OUT_CSV = DATA_DIR / "ai_filtered_top_candidates.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "ai_filtered_top_candidates.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "ai_filtered_top_candidates_report.md"

TOP_N = 10
FINAL_PICK_N = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI filtered top candidates from ranking_final.csv.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--entry-quality-csv", type=Path, default=DEFAULT_ENTRY_QUALITY_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt_num(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    work = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


def load_csv(path: Path, *, label: str) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        raise ValueError(f"{label} is empty: {resolved}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_ranking(path: Path) -> pd.DataFrame:
    df = load_csv(path, label="ranking csv")
    required = ["code", "name", "final_score", "liquidity_score", "risk_penalty", "confidence_score", "ret_10d"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")
    work = df.copy()
    work["name"] = work.get("name", "").fillna("").astype(str)
    for col in [
        "final_score",
        "liquidity_score",
        "risk_penalty",
        "confidence_score",
        "ret_10d",
        "ret_5d",
        "rank_final",
    ]:
        if col in work.columns:
            work[col] = _to_float(work[col])
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        work["date"] = ""
    return work


def build_filter_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    liquidity_score = pd.to_numeric(row.get("liquidity_score"), errors="coerce")
    risk_penalty = pd.to_numeric(row.get("risk_penalty"), errors="coerce")
    confidence_score = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    ret_10d = pd.to_numeric(row.get("ret_10d"), errors="coerce")

    if pd.isna(liquidity_score):
        reasons.append("liquidity_score missing")
    elif float(liquidity_score) < 50.0:
        reasons.append(f"liquidity_score {float(liquidity_score):.2f} < 50")

    if pd.isna(risk_penalty):
        reasons.append("risk_penalty missing")
    elif float(risk_penalty) >= 17.0:
        reasons.append(f"risk_penalty {float(risk_penalty):.2f} >= 17")

    if pd.isna(confidence_score):
        reasons.append("confidence_score missing")
    elif float(confidence_score) < 70.0:
        reasons.append(f"confidence_score {float(confidence_score):.2f} < 70")

    if pd.isna(ret_10d):
        reasons.append("ret_10d missing")
    elif float(ret_10d) >= 0.20:
        reasons.append(f"ret_10d {_fmt_pct(ret_10d)} >= 20.00%")

    return reasons


def merge_entry_quality(top10: pd.DataFrame, entry_quality_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    entry_quality = load_csv(entry_quality_path, label="entry quality csv")
    required = ["code", "entry_quality_score", "entry_quality_grade", "entry_quality_status", "entry_quality_reasons"]
    missing = [col for col in required if col not in entry_quality.columns]
    if missing:
        raise ValueError(f"entry quality columns missing: {', '.join(missing)}")

    keep_cols = [
        "code",
        "entry_quality_score",
        "entry_quality_grade",
        "entry_quality_status",
        "entry_quality_reasons",
        "ret_5d",
        "ret_10d",
        "liquidity_score",
        "risk_penalty",
    ]
    available_keep = [col for col in keep_cols if col in entry_quality.columns]
    merged = top10.merge(entry_quality.loc[:, available_keep].drop_duplicates("code"), on="code", how="left", suffixes=("", "_entry"))

    filled_from_entry: list[str] = []
    for col in ["ret_5d", "ret_10d", "liquidity_score", "risk_penalty"]:
        entry_col = f"{col}_entry"
        if entry_col in merged.columns:
            if col in merged.columns:
                before_missing = merged[col].isna().sum()
                merged[col] = pd.to_numeric(merged[col], errors="coerce").where(
                    pd.to_numeric(merged[col], errors="coerce").notna(),
                    pd.to_numeric(merged[entry_col], errors="coerce"),
                )
                after_missing = merged[col].isna().sum()
                if after_missing < before_missing:
                    filled_from_entry.append(col)
            else:
                merged[col] = pd.to_numeric(merged[entry_col], errors="coerce")
                filled_from_entry.append(col)
            merged = merged.drop(columns=[entry_col])

    for col in ["entry_quality_score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["entry_quality_status"] = merged.get("entry_quality_status", "").fillna("").astype(str)
    merged["entry_quality_grade"] = merged.get("entry_quality_grade", "").fillna("").astype(str)
    merged["entry_quality_reasons"] = merged.get("entry_quality_reasons", "").fillna("").astype(str)
    meta = {
        "entry_quality_rows": int(len(entry_quality)),
        "top10_missing_entry_quality_count": int(merged["entry_quality_status"].eq("").sum()),
        "filled_from_entry_quality": filled_from_entry,
    }
    return merged, meta


def build_adjusted_score(row: pd.Series) -> tuple[float, list[str], dict[str, bool]]:
    final_score = pd.to_numeric(row.get("final_score"), errors="coerce")
    entry_quality_score = pd.to_numeric(row.get("entry_quality_score"), errors="coerce")
    liquidity_score = pd.to_numeric(row.get("liquidity_score"), errors="coerce")
    risk_penalty = pd.to_numeric(row.get("risk_penalty"), errors="coerce")
    ret_10d = pd.to_numeric(row.get("ret_10d"), errors="coerce")
    ret_5d = pd.to_numeric(row.get("ret_5d"), errors="coerce")

    missing_flags = {
        "missing_final_score": pd.isna(final_score),
        "missing_entry_quality_score": pd.isna(entry_quality_score),
        "missing_liquidity_score": pd.isna(liquidity_score),
        "missing_risk_penalty": pd.isna(risk_penalty),
        "missing_ret_10d": pd.isna(ret_10d),
        "missing_ret_5d": pd.isna(ret_5d),
    }

    final_val = 0.0 if pd.isna(final_score) else float(final_score)
    entry_quality_val = 0.0 if pd.isna(entry_quality_score) else float(entry_quality_score)
    liquidity_val = 0.0 if pd.isna(liquidity_score) else float(liquidity_score)
    risk_penalty_val = 0.0 if pd.isna(risk_penalty) else float(risk_penalty)
    ret_10d_val = 0.0 if pd.isna(ret_10d) else float(ret_10d)
    ret_5d_val = 0.0 if pd.isna(ret_5d) else float(ret_5d)

    adjusted = (
        final_val
        + entry_quality_val * 0.20
        + liquidity_val * 0.10
        - risk_penalty_val * 0.50
        - ret_10d_val * 0.30
        - ret_5d_val * 0.20
    )

    reasons = [
        f"final_score={final_val:.2f}",
        f"+ entry_quality_score*0.20={entry_quality_val * 0.20:.2f}",
        f"+ liquidity_score*0.10={liquidity_val * 0.10:.2f}",
        f"- risk_penalty*0.50={risk_penalty_val * 0.50:.2f}",
        f"- ret_10d*0.30={ret_10d_val * 0.30:.4f}",
        f"- ret_5d*0.20={ret_5d_val * 0.20:.4f}",
    ]
    for flag_name, is_missing in missing_flags.items():
        if is_missing:
            reasons.append(f"{flag_name}=true -> treated as 0")
    return round(adjusted, 4), reasons, missing_flags


def select_top_candidates(ranking: pd.DataFrame, entry_quality_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    top10 = (
        ranking.sort_values(["final_score", "code"], ascending=[False, True], na_position="last")
        .head(TOP_N)
        .copy()
        .reset_index(drop=True)
    )
    top10["ai_top10_rank"] = range(1, len(top10) + 1)
    top10["filter_reason_list"] = top10.apply(build_filter_reasons, axis=1)
    top10["filter_passed"] = top10["filter_reason_list"].map(lambda items: len(items) == 0)
    top10["filter_reasons"] = top10["filter_reason_list"].map(lambda items: "PASS" if not items else "; ".join(items))
    top10["selected_for_ai_top5"] = False
    top10["ai_filtered_rank"] = pd.NA
    top10["fallback_selected"] = False
    top10["selection_note"] = ""
    top10["ai_adjusted_score"] = pd.NA
    top10["ai_adjusted_rank"] = pd.NA
    top10["score_adjustment_reasons"] = ""

    top10, entry_meta = merge_entry_quality(top10, entry_quality_path)

    adjusted_scores: list[float] = []
    adjustment_reasons: list[str] = []
    missing_flag_rows: list[dict[str, bool]] = []
    for _, row in top10.iterrows():
        adjusted, reasons, missing_flags = build_adjusted_score(row)
        adjusted_scores.append(adjusted)
        adjustment_reasons.append("; ".join(reasons))
        missing_flag_rows.append(missing_flags)
    top10["ai_adjusted_score"] = adjusted_scores
    top10["score_adjustment_reasons"] = adjustment_reasons
    for key in missing_flag_rows[0].keys() if missing_flag_rows else []:
        top10[key] = [flags[key] for flags in missing_flag_rows]

    eligible = top10.loc[top10["entry_quality_status"].astype(str).str.upper() != "BLOCK"].copy()
    eligible = eligible.sort_values(
        ["ai_adjusted_score", "final_score", "ai_top10_rank", "code"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    eligible["ai_adjusted_rank"] = range(1, len(eligible) + 1)

    for _, row in eligible.iterrows():
        mask = top10["code"].astype(str) == str(row["code"])
        top10.loc[mask, "ai_adjusted_rank"] = int(row["ai_adjusted_rank"])

    selected = eligible.head(FINAL_PICK_N).copy()
    selected_codes = set(selected["code"].astype(str).tolist())

    for idx, row in top10.iterrows():
        code = str(row["code"])
        status = str(row.get("entry_quality_status") or "").upper()
        if status == "BLOCK":
            top10.at[idx, "selection_note"] = "excluded_by_entry_quality_block"
            continue
        if code in selected_codes:
            top10.at[idx, "selected_for_ai_top5"] = True
            rank_value = pd.to_numeric(top10.at[idx, "ai_adjusted_rank"], errors="coerce")
            top10.at[idx, "ai_filtered_rank"] = int(rank_value) if pd.notna(rank_value) else pd.NA
            if not bool(top10.at[idx, "filter_passed"]):
                top10.at[idx, "fallback_selected"] = True
                top10.at[idx, "selection_note"] = "selected_despite_filter_fail_because_not_blocked_and_high_adjusted_score"
            else:
                top10.at[idx, "selection_note"] = "selected_by_ai_adjusted_score"
        else:
            top10.at[idx, "selection_note"] = "not_selected_after_ai_adjusted_rerank"

    top10["filter_passed"] = top10["filter_passed"].astype(bool)
    top10["selected_for_ai_top5"] = top10["selected_for_ai_top5"].astype(bool)
    top10["fallback_selected"] = top10["fallback_selected"].astype(bool)

    meta = {
        "entry_quality": entry_meta,
        "eligible_after_block_exclusion": int(len(eligible)),
        "selected_count": int(top10["selected_for_ai_top5"].sum()),
        "block_excluded_count": int(top10["entry_quality_status"].astype(str).str.upper().eq("BLOCK").sum()),
    }
    return top10, meta


def build_json_payload(result: pd.DataFrame, meta: dict[str, object]) -> dict[str, object]:
    selected = result.loc[result["selected_for_ai_top5"]].copy()
    selected = selected.sort_values(["ai_filtered_rank", "ai_adjusted_rank", "ai_top10_rank"], ascending=[True, True, True], na_position="last")
    failed = result.loc[~result["filter_passed"]].copy().sort_values(["ai_top10_rank"], ascending=[True])

    def _row_payload(row: pd.Series) -> dict[str, object]:
        return {
            "code": str(row.get("code") or "").zfill(6),
            "name": str(row.get("name") or ""),
            "date": str(row.get("date") or ""),
            "final_score": None if pd.isna(row.get("final_score")) else float(row.get("final_score")),
            "ai_top10_rank": None if pd.isna(row.get("ai_top10_rank")) else int(row.get("ai_top10_rank")),
            "filter_passed": _as_bool(row.get("filter_passed")),
            "filter_reasons": str(row.get("filter_reasons") or ""),
            "entry_quality_status": str(row.get("entry_quality_status") or ""),
            "entry_quality_score": None if pd.isna(row.get("entry_quality_score")) else float(row.get("entry_quality_score")),
            "selected_for_ai_top5": _as_bool(row.get("selected_for_ai_top5")),
            "ai_filtered_rank": None if pd.isna(row.get("ai_filtered_rank")) else int(row.get("ai_filtered_rank")),
            "ai_adjusted_score": None if pd.isna(row.get("ai_adjusted_score")) else float(row.get("ai_adjusted_score")),
            "ai_adjusted_rank": None if pd.isna(row.get("ai_adjusted_rank")) else int(row.get("ai_adjusted_rank")),
            "fallback_selected": _as_bool(row.get("fallback_selected")),
            "selection_note": str(row.get("selection_note") or ""),
            "score_adjustment_reasons": str(row.get("score_adjustment_reasons") or ""),
            "liquidity_score": None if pd.isna(row.get("liquidity_score")) else float(row.get("liquidity_score")),
            "risk_penalty": None if pd.isna(row.get("risk_penalty")) else float(row.get("risk_penalty")),
            "confidence_score": None if pd.isna(row.get("confidence_score")) else float(row.get("confidence_score")),
            "ret_5d": None if pd.isna(row.get("ret_5d")) else float(row.get("ret_5d")),
            "ret_10d": None if pd.isna(row.get("ret_10d")) else float(row.get("ret_10d")),
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": str(DEFAULT_INPUT_CSV),
        "entry_quality_csv": str(DEFAULT_ENTRY_QUALITY_CSV),
        "selection_policy": {
            "top_n": TOP_N,
            "final_pick_n": FINAL_PICK_N,
            "filters": {
                "liquidity_score_gte": 50,
                "risk_penalty_lt": 17,
                "confidence_score_gte": 70,
                "ret_10d_lt": 0.20,
            },
            "adjusted_score_formula": "final_score + entry_quality_score*0.20 + liquidity_score*0.10 - risk_penalty*0.50 - ret_10d*0.30 - ret_5d*0.20",
            "block_rule": "entry_quality_status=BLOCK excluded before reranking",
        },
        "summary": {
            "top10_count": int(len(result)),
            "filter_passed_count": int(result["filter_passed"].sum()),
            "filter_failed_count": int((~result["filter_passed"]).sum()),
            "selected_count": int(result["selected_for_ai_top5"].sum()),
            "fallback_selected_count": int(result["fallback_selected"].sum()),
            "block_excluded_count": int(result["entry_quality_status"].astype(str).str.upper().eq("BLOCK").sum()),
            "eligible_after_block_exclusion": int(meta.get("eligible_after_block_exclusion", 0)),
        },
        "selected_top5": [_row_payload(row) for _, row in selected.iterrows()],
        "failed_candidates": [_row_payload(row) for _, row in failed.iterrows()],
        "top10_candidates": [_row_payload(row) for _, row in result.sort_values(["ai_top10_rank"]).iterrows()],
        "meta": meta,
    }


def build_markdown_report(result: pd.DataFrame, payload: dict[str, object], meta: dict[str, object]) -> str:
    passed_count = int(result["filter_passed"].sum())
    failed = result.loc[~result["filter_passed"]].copy().sort_values(["ai_top10_rank"])
    selected = result.loc[result["selected_for_ai_top5"]].copy().sort_values(["ai_filtered_rank", "ai_adjusted_rank", "ai_top10_rank"])
    block_rows = result.loc[result["entry_quality_status"].astype(str).str.upper() == "BLOCK"].copy().sort_values(["ai_top10_rank"])

    display = result.copy()
    for col in ["final_score", "liquidity_score", "risk_penalty", "confidence_score", "entry_quality_score", "ai_adjusted_score"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: _fmt_num(x, 2))
    for col in ["ret_10d", "ret_5d"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: _fmt_pct(x, 2))
    display["filter_passed"] = display["filter_passed"].map(lambda x: "Y" if _as_bool(x) else "N")
    display["selected_for_ai_top5"] = display["selected_for_ai_top5"].map(lambda x: "Y" if _as_bool(x) else "N")
    display["fallback_selected"] = display["fallback_selected"].map(lambda x: "Y" if _as_bool(x) else "N")
    for col in ["ai_filtered_rank", "ai_adjusted_rank"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(int(x)))

    lines: list[str] = [
        "# AI Filtered Top Candidates Report",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- source_csv: {payload.get('source_csv')}",
        f"- entry_quality_csv: {payload.get('entry_quality_csv')}",
        f"- top10_count: {len(result)}",
        f"- filter_passed_count: {passed_count}",
        f"- filter_failed_count: {len(failed)}",
        f"- block_excluded_count: {len(block_rows)}",
        f"- eligible_after_block_exclusion: {meta.get('eligible_after_block_exclusion')}",
        f"- selected_top5_count: {int(result['selected_for_ai_top5'].sum())}",
        f"- fallback_selected_count: {int(result['fallback_selected'].sum())}",
        "",
        "## Filter Rules",
        "",
        "- liquidity_score >= 50",
        "- risk_penalty < 17",
        "- confidence_score >= 70",
        "- ret_10d < 20%",
        "- entry_quality_status=BLOCK rows are excluded before final selection",
        "",
        "## Adjusted Score Formula",
        "",
        "- ai_adjusted_score = final_score + entry_quality_score*0.20 + liquidity_score*0.10 - risk_penalty*0.50 - ret_10d*0.30 - ret_5d*0.20",
        "- ret_10d and ret_5d use the actual ranking columns when present",
        "- missing inputs are treated as 0 and recorded in score_adjustment_reasons",
        "",
        "## Top10 Evaluation",
        "",
        _markdown_table(
            display,
            [
                "ai_top10_rank",
                "ai_adjusted_rank",
                "ai_filtered_rank",
                "code",
                "name",
                "final_score",
                "entry_quality_status",
                "entry_quality_score",
                "liquidity_score",
                "risk_penalty",
                "ret_5d",
                "ret_10d",
                "ai_adjusted_score",
                "filter_passed",
                "filter_reasons",
                "selected_for_ai_top5",
                "fallback_selected",
                "selection_note",
            ],
        ),
        "",
        "## Rank Comparison",
        "",
        _markdown_table(
            display.sort_values(["ai_top10_rank"]),
            [
                "ai_top10_rank",
                "code",
                "name",
                "final_score",
                "ai_adjusted_score",
                "entry_quality_status",
                "ai_adjusted_rank",
                "selected_for_ai_top5",
            ],
        ),
        "",
        "## Filter Failed Candidates",
        "",
    ]
    if failed.empty:
        lines.extend(["- none", ""])
    else:
        for _, row in failed.iterrows():
            lines.append(
                f"- top10_rank={int(row['ai_top10_rank'])} {str(row['code']).zfill(6)} {row['name']}: {row['filter_reasons']}"
            )
        lines.append("")

    lines.extend(["## BLOCK Excluded Candidates", ""])
    if block_rows.empty:
        lines.extend(["- none", ""])
    else:
        for _, row in block_rows.iterrows():
            lines.append(
                f"- top10_rank={int(row['ai_top10_rank'])} {str(row['code']).zfill(6)} {row['name']}: "
                f"entry_quality_status=BLOCK, reason={row.get('entry_quality_reasons') or ''}"
            )
        lines.append("")

    lines.extend(["## Final AI Top5", ""])
    if selected.empty:
        lines.extend(["- none", ""])
    else:
        for _, row in selected.iterrows():
            note = "filter_failed_but_not_blocked" if _as_bool(row["fallback_selected"]) else "filter_passed"
            lines.append(
                f"- rank={int(row['ai_filtered_rank'])} adjusted_rank={int(row['ai_adjusted_rank'])} "
                f"top10_rank={int(row['ai_top10_rank'])} {str(row['code']).zfill(6)} {row['name']}: "
                f"ai_adjusted_score={_fmt_num(row['ai_adjusted_score'])}, selection={note}"
            )
        lines.append("")
    return "\n".join(lines)


def save_outputs(result: pd.DataFrame, out_csv: Path, out_json: Path, out_md: Path, meta: dict[str, object]) -> dict[str, object]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    export = result.drop(columns=["filter_reason_list"], errors="ignore").copy()
    export.to_csv(out_csv, index=False, encoding="utf-8-sig")

    payload = build_json_payload(result, meta)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown_report(result, payload, meta), encoding="utf-8")
    return payload


def print_summary(result: pd.DataFrame, payload: dict[str, object], meta: dict[str, object]) -> None:
    print(f"top10_count={len(result)}")
    print(f"filter_passed_count={int(result['filter_passed'].sum())}")
    print(f"filter_failed_count={int((~result['filter_passed']).sum())}")
    print(f"block_excluded_count={int(result['entry_quality_status'].astype(str).str.upper().eq('BLOCK').sum())}")
    print(f"eligible_after_block_exclusion={int(meta.get('eligible_after_block_exclusion', 0))}")
    print(f"selected_count={int(result['selected_for_ai_top5'].sum())}")

    selected = result.loc[result["selected_for_ai_top5"]].copy().sort_values(["ai_filtered_rank", "ai_adjusted_rank", "ai_top10_rank"])
    if selected.empty:
        print("selected_top5=none")
    else:
        print("selected_top5:")
        for _, row in selected.iterrows():
            note = "filter_failed_but_not_blocked" if _as_bool(row["fallback_selected"]) else "filter_passed"
            print(
                f" - rank={int(row['ai_filtered_rank'])} adjusted_rank={int(row['ai_adjusted_rank'])} "
                f"top10_rank={int(row['ai_top10_rank'])} {str(row['code']).zfill(6)} {row['name']} "
                f"score={_fmt_num(row['ai_adjusted_score'])} ({note})"
            )

    print(f"entry_quality_meta={payload.get('meta', {}).get('entry_quality')}")
    print(f"csv={DEFAULT_OUT_CSV}")
    print(f"json={DEFAULT_OUT_JSON}")
    print(f"md={DEFAULT_OUT_MD}")


def main() -> None:
    args = parse_args()
    ranking = load_ranking(args.input_csv)
    result, meta = select_top_candidates(ranking, args.entry_quality_csv)
    payload = save_outputs(result, _resolve(args.out_csv), _resolve(args.out_json), _resolve(args.out_md), meta)
    print_summary(result, payload, meta)


if __name__ == "__main__":
    main()
