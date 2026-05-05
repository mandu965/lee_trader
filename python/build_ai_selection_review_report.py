from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_RANKING_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_FILTERED_CSV = DATA_DIR / "ai_filtered_top_candidates.csv"
DEFAULT_ENTRY_QUALITY_CSV = DATA_DIR / "ai_entry_quality_score.csv"
DEFAULT_PREVIEW_JSON = OUTPUT_DIR / "order_requests_preview.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "ai_selection_review_report.md"
DEFAULT_OUT_JSON = OUTPUT_DIR / "ai_selection_review_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI selection review report.")
    parser.add_argument("--ranking-csv", type=Path, default=DEFAULT_RANKING_CSV)
    parser.add_argument("--filtered-csv", type=Path, default=DEFAULT_FILTERED_CSV)
    parser.add_argument("--entry-quality-csv", type=Path, default=DEFAULT_ENTRY_QUALITY_CSV)
    parser.add_argument("--preview-json", type=Path, default=DEFAULT_PREVIEW_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, dtype={"code": str}, low_memory=False)


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


def _reason_list(text: object) -> list[str]:
    raw = str(text or "").strip()
    if not raw or raw.upper() == "PASS":
        return []
    return [item.strip() for item in raw.split(";") if item.strip()]


def _is_broken_text(text: object) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    return "?" in raw or "�" in raw


def _pick_display_name(*values: object) -> str:
    candidates = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not candidates:
        return ""
    for candidate in candidates:
        if not _is_broken_text(candidate):
            return candidate
    return candidates[0]


def load_csv(path: Path, *, label: str) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    frame = _load_csv_with_fallback(resolved)
    if frame.empty:
        raise ValueError(f"{label} is empty: {resolved}")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "name" in frame.columns:
        frame["name"] = frame["name"].fillna("").astype(str)
    else:
        frame["name"] = ""
    return frame


def load_preview_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"preview json not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def build_preview_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in payload.get("items") or []:
        if str(item.get("side") or "").upper() != "BUY":
            continue
        code = str(item.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        lookup[code] = item
    return lookup


def prepare_top10(ranking: pd.DataFrame) -> pd.DataFrame:
    work = ranking.copy()
    for col in ["final_score", "rank_final", "buy_rank"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work.get(col), errors="coerce")
    top10 = (
        work.sort_values(["final_score", "code"], ascending=[False, True], na_position="last")
        .head(10)
        .copy()
        .reset_index(drop=True)
    )
    top10["original_final_rank"] = range(1, len(top10) + 1)
    return top10


def build_report_frame(
    ranking_top10: pd.DataFrame,
    filtered: pd.DataFrame,
    entry_quality: pd.DataFrame,
    preview_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    base = ranking_top10.loc[:, ["code", "name", "final_score", "original_final_rank"]].copy()
    filtered_keep = [
        "code",
        "name",
        "ai_top10_rank",
        "filter_passed",
        "filter_reasons",
        "selected_for_ai_top5",
        "ai_filtered_rank",
        "fallback_selected",
        "selection_note",
        "entry_quality_status",
        "entry_quality_score",
        "ai_adjusted_score",
        "ai_adjusted_rank",
        "score_adjustment_reasons",
        "ret_5d",
        "ret_10d",
        "liquidity_score",
        "risk_penalty",
        "confidence_score",
    ]
    filtered_view = filtered.loc[:, [col for col in filtered_keep if col in filtered.columns]].drop_duplicates("code", keep="first")
    entry_keep = ["code", "name", "entry_quality_grade", "entry_quality_status", "entry_quality_reasons", "entry_quality_score"]
    entry_view = entry_quality.loc[:, [col for col in entry_keep if col in entry_quality.columns]].drop_duplicates("code", keep="first")

    merged = base.merge(filtered_view, on="code", how="left", suffixes=("", "_filtered"))
    merged = merged.merge(entry_view, on="code", how="left", suffixes=("", "_entry"))

    if "entry_quality_status_entry" in merged.columns:
        merged["entry_quality_status"] = merged["entry_quality_status"].where(
            merged["entry_quality_status"].notna() & merged["entry_quality_status"].astype(str).ne(""),
            merged["entry_quality_status_entry"],
        )
        merged = merged.drop(columns=["entry_quality_status_entry"])
    if "entry_quality_score_entry" in merged.columns:
        merged["entry_quality_score"] = pd.to_numeric(merged["entry_quality_score"], errors="coerce").where(
            pd.to_numeric(merged["entry_quality_score"], errors="coerce").notna(),
            pd.to_numeric(merged["entry_quality_score_entry"], errors="coerce"),
        )
        merged = merged.drop(columns=["entry_quality_score_entry"])

    merged["name"] = merged.apply(
        lambda row: _pick_display_name(row.get("name_filtered"), row.get("name_entry"), row.get("name")),
        axis=1,
    )
    merged = merged.drop(columns=[col for col in ["name_filtered", "name_entry"] if col in merged.columns])

    merged["filter_passed"] = merged.get("filter_passed", False).astype(str).str.strip().str.lower().isin(["1", "true", "yes", "on"])
    merged["selected_for_ai_top5"] = merged.get("selected_for_ai_top5", False).astype(str).str.strip().str.lower().isin(["1", "true", "yes", "on"])
    merged["fallback_selected"] = merged.get("fallback_selected", False).astype(str).str.strip().str.lower().isin(["1", "true", "yes", "on"])

    for col in ["ai_top10_rank", "ai_filtered_rank", "ai_adjusted_rank"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged.get(col), errors="coerce")
    for col in ["final_score", "entry_quality_score", "ai_adjusted_score", "ret_5d", "ret_10d", "liquidity_score", "risk_penalty", "confidence_score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    merged["preview_side"] = ""
    merged["preview_executable_now"] = False
    merged["preview_blocked_reason"] = ""
    merged["preview_expected_hold_reason"] = ""
    for idx, row in merged.iterrows():
        item = preview_lookup.get(str(row["code"]).zfill(6))
        if not item:
            continue
        merged.at[idx, "name"] = _pick_display_name(item.get("name"), merged.at[idx, "name"])
        merged.at[idx, "preview_side"] = str(item.get("side") or "")
        merged.at[idx, "preview_executable_now"] = bool(item.get("executable_now"))
        merged.at[idx, "preview_blocked_reason"] = str(item.get("blocked_reason") or "")
        merged.at[idx, "preview_expected_hold_reason"] = str(item.get("expected_hold_reason") or "")

    merged["selection_reasons"] = merged.apply(
        lambda row: "; ".join(
            [
                part
                for part in [
                    str(row.get("selection_note") or "").strip(),
                    "filter_passed" if bool(row.get("filter_passed")) else "",
                    f"entry_quality_status={row.get('entry_quality_status')}" if str(row.get("entry_quality_status") or "").strip() else "",
                    f"ai_adjusted_rank={int(row['ai_adjusted_rank'])}" if pd.notna(pd.to_numeric(row.get("ai_adjusted_rank"), errors="coerce")) else "",
                ]
                if part
            ]
        ),
        axis=1,
    )
    merged["deduction_reasons"] = merged.apply(
        lambda row: "; ".join(
            list(dict.fromkeys(_reason_list(row.get("filter_reasons")) + _reason_list(row.get("entry_quality_reasons"))))
        ),
        axis=1,
    )
    return merged.sort_values(["original_final_rank", "code"], ascending=[True, True], na_position="last").reset_index(drop=True)


def build_summary(frame: pd.DataFrame, preview_payload: dict[str, Any]) -> dict[str, Any]:
    filter_passed_count = int(frame["filter_passed"].sum())
    filter_failed_count = int((~frame["filter_passed"]).sum())
    entry_counts = Counter(frame.get("entry_quality_status", pd.Series(dtype="object")).fillna("UNKNOWN").astype(str).tolist())

    original_top5 = frame.sort_values(["original_final_rank"]).head(5)
    adjusted_top5 = frame.loc[frame["selected_for_ai_top5"]].sort_values(["ai_filtered_rank", "ai_adjusted_rank"], na_position="last")
    original_codes = [str(code).zfill(6) for code in original_top5["code"].tolist()]
    adjusted_codes = [str(code).zfill(6) for code in adjusted_top5["code"].tolist()]

    new_entries = [code for code in adjusted_codes if code not in original_codes]
    dropped = [code for code in original_codes if code not in adjusted_codes]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": str(preview_payload.get("asof_date") or frame.get("date", pd.Series([""])).iloc[0] if not frame.empty else ""),
        "preview_gate_status": preview_payload.get("gate_status"),
        "preview_request_count": int((preview_payload.get("summary") or {}).get("request_count") or 0),
        "preview_buy_count": int((preview_payload.get("summary") or {}).get("buy_count") or 0),
        "filter_passed_count": filter_passed_count,
        "filter_failed_count": filter_failed_count,
        "entry_quality_status_counts": {str(k): int(v) for k, v in entry_counts.items()},
        "original_top5_codes": original_codes,
        "adjusted_top5_codes": adjusted_codes,
        "new_entries_vs_original_top5": new_entries,
        "dropped_vs_original_top5": dropped,
        "final_ai_buy_candidates": [
            {
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name") or ""),
                "ai_filtered_rank": int(row["ai_filtered_rank"]) if pd.notna(row.get("ai_filtered_rank")) else None,
                "ai_adjusted_rank": int(row["ai_adjusted_rank"]) if pd.notna(row.get("ai_adjusted_rank")) else None,
                "entry_quality_status": str(row.get("entry_quality_status") or ""),
                "preview_executable_now": bool(row.get("preview_executable_now")),
            }
            for _, row in adjusted_top5.iterrows()
        ],
    }


def build_markdown(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    top10 = frame.sort_values(["original_final_rank"]).copy()
    original_top5 = top10.head(5).copy()
    adjusted_top5 = frame.loc[frame["selected_for_ai_top5"]].copy().sort_values(["ai_filtered_rank", "ai_adjusted_rank"], na_position="last")
    excluded = frame.loc[~frame["selected_for_ai_top5"]].copy().sort_values(["original_final_rank"], na_position="last")

    display_top10 = top10.copy()
    for col in ["final_score", "ai_adjusted_score", "entry_quality_score", "liquidity_score", "risk_penalty", "confidence_score"]:
        if col in display_top10.columns:
            display_top10[col] = display_top10[col].map(lambda x: _fmt_num(x, 2))
    for col in ["ret_5d", "ret_10d"]:
        if col in display_top10.columns:
            display_top10[col] = display_top10[col].map(lambda x: _fmt_pct(x, 2))
    for col in ["filter_passed", "selected_for_ai_top5", "fallback_selected", "preview_executable_now"]:
        if col in display_top10.columns:
            display_top10[col] = display_top10[col].map(lambda x: "Y" if bool(x) else "N")
    for col in ["original_final_rank", "ai_filtered_rank", "ai_adjusted_rank", "ai_top10_rank"]:
        if col in display_top10.columns:
            display_top10[col] = display_top10[col].map(lambda x: "" if pd.isna(x) else str(int(x)))

    lines: list[str] = [
        "# AI Selection Review Report",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- 기준일: {summary.get('asof_date')}",
        f"- preview_gate_status: {summary.get('preview_gate_status') or 'NA'}",
        f"- preview_buy_count: {summary.get('preview_buy_count')}",
        f"- preview_request_count: {summary.get('preview_request_count')}",
        "",
        "> 본 리포트는 AI 자동매매 후보 검토용이며, RULE 전략 운영과는 별도입니다.",
        "",
        "## 1. 기존 final_score Top10",
        "",
        _markdown_table(
            display_top10,
            [
                "original_final_rank",
                "code",
                "name",
                "final_score",
                "ai_adjusted_score",
                "filter_passed",
                "entry_quality_status",
                "selected_for_ai_top5",
            ],
        ),
        "",
        "## 2. 필터 통과/탈락 현황",
        "",
        f"- filter_passed_count: {summary.get('filter_passed_count')}",
        f"- filter_failed_count: {summary.get('filter_failed_count')}",
        "",
    ]

    failed = frame.loc[~frame["filter_passed"]].copy().sort_values(["original_final_rank"])
    if failed.empty:
        lines.append("- filter failed candidates: none")
    else:
        for _, row in failed.iterrows():
            lines.append(f"- {str(row['code']).zfill(6)} {row.get('name') or ''}: {row.get('filter_reasons') or ''}")

    lines.extend(
        [
            "",
            "## 3. entry_quality_status 분포",
            "",
            f"- {summary.get('entry_quality_status_counts')}",
            "",
            "## 4. 기존 Top5 vs Adjusted Top5 비교",
            "",
            f"- original_top5_codes: {summary.get('original_top5_codes')}",
            f"- adjusted_top5_codes: {summary.get('adjusted_top5_codes')}",
            f"- new_entries_vs_original_top5: {summary.get('new_entries_vs_original_top5')}",
            f"- dropped_vs_original_top5: {summary.get('dropped_vs_original_top5')}",
            "",
            _markdown_table(
                pd.concat(
                    [
                        original_top5.assign(group="original_top5"),
                        adjusted_top5.assign(group="adjusted_top5"),
                    ],
                    ignore_index=True,
                ),
                [
                    "group",
                    "original_final_rank",
                    "ai_filtered_rank",
                    "ai_adjusted_rank",
                    "code",
                    "name",
                    "final_score",
                    "ai_adjusted_score",
                    "entry_quality_status",
                ],
            ),
            "",
            "## 5. 최종 AI 매수 후보",
            "",
        ]
    )

    if adjusted_top5.empty:
        lines.append("- none")
    else:
        for _, row in adjusted_top5.iterrows():
            lines.append(
                f"- {str(row['code']).zfill(6)} {row.get('name') or ''}: "
                f"ai_filtered_rank={_fmt_num(row.get('ai_filtered_rank'), 0)}, "
                f"ai_adjusted_rank={_fmt_num(row.get('ai_adjusted_rank'), 0)}, "
                f"entry_quality_status={row.get('entry_quality_status') or 'NA'}, "
                f"preview_executable_now={'Y' if bool(row.get('preview_executable_now')) else 'N'}"
            )

    lines.extend(["", "## 6. 후보별 선정 사유", ""])
    if adjusted_top5.empty:
        lines.append("- none")
    else:
        for _, row in adjusted_top5.iterrows():
            lines.append(f"- {str(row['code']).zfill(6)} {row.get('name') or ''}: {row.get('selection_reasons') or '확인 필요'}")

    lines.extend(["", "## 7. 후보별 차단/감점 사유", ""])
    if excluded.empty:
        lines.append("- none")
    else:
        for _, row in excluded.iterrows():
            reasons = str(row.get("deduction_reasons") or row.get("entry_quality_reasons") or row.get("filter_reasons") or "확인 필요")
            lines.append(f"- {str(row['code']).zfill(6)} {row.get('name') or ''}: {reasons}")

    lines.extend(["", "## 8. Preview 기준 실주문 가능 여부", ""])
    if frame.empty:
        lines.append("- none")
    else:
        for _, row in frame.sort_values(["selected_for_ai_top5", "original_final_rank"], ascending=[False, True]).iterrows():
            lines.append(
                f"- {str(row['code']).zfill(6)} {row.get('name') or ''}: "
                f"selected={'Y' if bool(row.get('selected_for_ai_top5')) else 'N'}, "
                f"preview_executable_now={'Y' if bool(row.get('preview_executable_now')) else 'N'}, "
                f"preview_blocked_reason={row.get('preview_blocked_reason') or '-'}, "
                f"expected_hold_reason={row.get('preview_expected_hold_reason') or '-'}"
            )

    lines.extend(["", "## 9. 제외 후보", ""])
    if excluded.empty:
        lines.append("- none")
    else:
        lines.append(
            _markdown_table(
                excluded,
                [
                    "original_final_rank",
                    "code",
                    "name",
                    "final_score",
                    "entry_quality_status",
                    "filter_reasons",
                    "entry_quality_reasons",
                    "selection_note",
                ],
            )
        )

    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], markdown: str, out_json: Path, out_md: Path) -> None:
    out_json = _resolve(out_json)
    out_md = _resolve(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")


def print_summary(summary: dict[str, Any], frame: pd.DataFrame) -> None:
    print(f"asof_date={summary.get('asof_date')}")
    print(f"filter_passed_count={summary.get('filter_passed_count')}")
    print(f"filter_failed_count={summary.get('filter_failed_count')}")
    print(f"entry_quality_status_counts={summary.get('entry_quality_status_counts')}")
    print(f"final_candidate_count={len(summary.get('final_ai_buy_candidates') or [])}")
    final_codes = [item.get("code") for item in summary.get("final_ai_buy_candidates") or []]
    print(f"final_candidate_codes={final_codes}")
    excluded_codes = frame.loc[~frame["selected_for_ai_top5"], "code"].astype(str).str.zfill(6).tolist()
    print(f"excluded_candidate_codes={excluded_codes}")
    print(f"md={_resolve(DEFAULT_OUT_MD)}")
    print(f"json={_resolve(DEFAULT_OUT_JSON)}")


def main() -> None:
    args = parse_args()
    ranking = load_csv(args.ranking_csv, label="ranking csv")
    filtered = load_csv(args.filtered_csv, label="filtered csv")
    entry_quality = load_csv(args.entry_quality_csv, label="entry quality csv")
    preview_payload = load_preview_json(args.preview_json)
    preview_lookup = build_preview_lookup(preview_payload)
    ranking_top10 = prepare_top10(ranking)
    frame = build_report_frame(ranking_top10, filtered, entry_quality, preview_lookup)
    summary = build_summary(frame, preview_payload)
    markdown = build_markdown(frame, summary)
    write_outputs(summary, markdown, args.out_json, args.out_md)
    print_summary(summary, frame)


if __name__ == "__main__":
    main()
