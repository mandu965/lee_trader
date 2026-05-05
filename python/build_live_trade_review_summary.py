from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from payload_store import upsert_json_payload


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_trade_review_summary.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_trade_review_summary.md"

RETURN_RE = re.compile(r"d(?P<horizon>\d+)_signed_return=(?P<value>[-+]?\d+(?:\.\d+)?)%")
RANK_RE = re.compile(r"rank=(?P<value>\d+)")
CONFIDENCE_RE = re.compile(r"confidence=(?P<value>[-+]?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize live trade review history for strategy feedback.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--reviewer", default="auto_review")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _extract_return(note: str) -> tuple[str | None, float | None]:
    match = RETURN_RE.search(str(note or ""))
    if not match:
        return None, None
    return f"d{match.group('horizon')}", float(match.group("value")) / 100.0


def _extract_int(note: str, regex: re.Pattern[str]) -> int | None:
    match = regex.search(str(note or ""))
    return int(match.group("value")) if match else None


def _extract_float(note: str, regex: re.Pattern[str]) -> float | None:
    match = regex.search(str(note or ""))
    return float(match.group("value")) if match else None


def _rank_bucket(value: Any) -> str:
    rank = pd.to_numeric(value, errors="coerce")
    if pd.isna(rank):
        return "rank_unknown"
    rank = int(rank)
    if rank <= 3:
        return "rank_1_3"
    if rank <= 8:
        return "rank_4_8"
    if rank <= 20:
        return "rank_9_20"
    return "rank_21_plus"


def _confidence_bucket(value: Any) -> str:
    confidence = pd.to_numeric(value, errors="coerce")
    if pd.isna(confidence):
        return "confidence_unknown"
    confidence = float(confidence)
    if confidence >= 90:
        return "confidence_90_plus"
    if confidence >= 80:
        return "confidence_80_90"
    if confidence >= 70:
        return "confidence_70_80"
    return "confidence_under_70"


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def load_review_rows(reviewer: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    v.review_id,
                    v.intent_id,
                    v.request_id,
                    v.code,
                    v.review_date,
                    v.pre_tags,
                    v.post_tags,
                    v.outcome_label,
                    v.review_note,
                    v.next_action_note,
                    v.reviewer,
                    v.created_at,
                    v.updated_at,
                    r.side,
                    r.intent_type,
                    r.ranking_rank,
                    r.confidence_score,
                    r.risk_penalty,
                    r.final_score,
                    v.engine_type,
                    v.strategy_id,
                    v.run_mode,
                    v.entry_gate_status,
                    v.entry_gate_reason,
                    v.review_status,
                    v.strategy_return,
                    v.holding_days
                FROM research.live_trade_review v
                LEFT JOIN research.live_order_request r
                  ON r.request_id = v.request_id
                WHERE (:reviewer = '' OR v.reviewer = :reviewer)
                ORDER BY v.review_date DESC, v.review_id DESC
                """
            ),
            {"reviewer": reviewer},
        ).mappings().all()
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return df

    extracted = df["review_note"].fillna("").map(_extract_return)
    df["return_horizon"] = extracted.map(lambda item: item[0])
    df["signed_return"] = extracted.map(lambda item: item[1])
    df["parsed_rank"] = df["review_note"].fillna("").map(lambda note: _extract_int(note, RANK_RE))
    df["parsed_confidence"] = df["review_note"].fillna("").map(lambda note: _extract_float(note, CONFIDENCE_RE))
    df["rank_for_bucket"] = df["ranking_rank"].where(df["ranking_rank"].notna(), df["parsed_rank"])
    df["confidence_for_bucket"] = df["confidence_score"].where(df["confidence_score"].notna(), df["parsed_confidence"])
    df["rank_bucket"] = df["rank_for_bucket"].map(_rank_bucket)
    df["confidence_bucket"] = df["confidence_for_bucket"].map(_confidence_bucket)
    df["engine_type"] = df["engine_type"].fillna("")
    df["entry_gate_status"] = df["entry_gate_status"].fillna("")
    df["review_status"] = df["review_status"].fillna("")
    df["side"] = df["side"].fillna("")
    df["intent_type"] = df["intent_type"].fillna("")
    return df


def _aggregate(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    grouped = (
        df.groupby(columns, dropna=False)
        .agg(
            count=("review_id", "count"),
            observed_count=("signed_return", lambda s: int(s.notna().sum())),
            avg_signed_return=("signed_return", "mean"),
            win_rate=("signed_return", lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else None),
            positive_count=("outcome_label", lambda s: int((s == "positive").sum())),
            negative_count=("outcome_label", lambda s: int((s == "negative").sum())),
            pending_count=("outcome_label", lambda s: int((s == "pending_price_data").sum())),
        )
        .reset_index()
        .sort_values(["observed_count", "count"], ascending=[False, False])
    )
    return grouped.to_dict(orient="records")


def _group_label(row: dict[str, Any], keys: list[str]) -> str:
    return " / ".join(str(row.get(key) or "-") for key in keys)


def _recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    overview = report["overview"]
    observed_count = int(overview.get("observed_count") or 0)
    if observed_count < 20:
        recommendations.append(
            {
                "level": "info",
                "topic": "sample_size",
                "message": "Observed live-review sample is still small; use as monitoring signal, not as a parameter-change trigger.",
            }
        )

    for group_name, rows, keys in (
        ("intent", report["by_intent"], ["intent_type"]),
        ("rank", report["by_rank_bucket"], ["rank_bucket"]),
        ("confidence", report["by_confidence_bucket"], ["confidence_bucket"]),
    ):
        for row in rows:
            observed = int(row.get("observed_count") or 0)
            avg_return = row.get("avg_signed_return")
            win_rate = row.get("win_rate")
            if observed < 3 or avg_return is None or win_rate is None:
                continue
            if float(avg_return) <= -0.01 and float(win_rate) < 0.35:
                recommendations.append(
                    {
                        "level": "watch",
                        "topic": group_name,
                        "group": _group_label(row, keys),
                        "message": "Underperforming live-review segment; review sizing, entry timing, or gate conditions before increasing exposure.",
                    }
                )
            elif float(avg_return) >= 0.01 and float(win_rate) >= 0.55:
                recommendations.append(
                    {
                        "level": "candidate",
                        "topic": group_name,
                        "group": _group_label(row, keys),
                        "message": "Positive segment candidate; keep monitoring until sample size is large enough for a policy change.",
                    }
                )
    return recommendations


def build_report(reviewer: str) -> dict[str, Any]:
    df = load_review_rows(reviewer)
    if df.empty:
        latest_review_date = None
        observed = pd.DataFrame()
    else:
        latest_review_date = str(df["review_date"].max())
        observed = df[df["signed_return"].notna()].copy()

    overview = {
        "review_count": int(len(df)),
        "observed_count": int(len(observed)),
        "latest_review_date": latest_review_date,
        "latest_date": latest_review_date,
        "avg_signed_return": float(observed["signed_return"].mean()) if not observed.empty else None,
        "win_rate": float((observed["signed_return"] > 0).mean()) if not observed.empty else None,
        "pending_count": int((df["outcome_label"] == "pending_price_data").sum()) if not df.empty else 0,
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reviewer": reviewer or "all",
        "overview": overview,
        "by_intent": _aggregate(df, ["intent_type"]) if not df.empty else [],
        "by_side": _aggregate(df, ["side"]) if not df.empty else [],
        "by_outcome": _aggregate(df, ["outcome_label"]) if not df.empty else [],
        "by_rank_bucket": _aggregate(df, ["rank_bucket"]) if not df.empty else [],
        "by_confidence_bucket": _aggregate(df, ["confidence_bucket"]) if not df.empty else [],
        "by_engine_type": _aggregate(df, ["engine_type"]) if not df.empty else [],
        "by_entry_gate_status": _aggregate(df, ["entry_gate_status"]) if not df.empty else [],
        "by_review_status": _aggregate(df, ["review_status"]) if not df.empty else [],
    }
    report["recommendations"] = _recommendations(report)
    return _clean_json(report)


def _pct(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) * 100:.2f}%"


def _render_group(title: str, rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = ["", f"## {title}", "", "| group | count | observed | avg_return | win_rate | pending |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        group = " / ".join(str(row.get(key) or "-") for key in keys)
        lines.append(
            f"| {group} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_pct(row.get('avg_signed_return'))} | {_pct(row.get('win_rate'))} | {row.get('pending_count') or 0} |"
        )
    if not rows:
        lines.append("| - | 0 | 0 | - | - | 0 |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    lines = [
        "# Live Trade Review Summary",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- reviewer: `{report['reviewer']}`",
        f"- latest_review_date: `{overview.get('latest_review_date') or '-'}`",
        f"- review_count: `{overview['review_count']}`",
        f"- observed_count: `{overview['observed_count']}`",
        f"- avg_signed_return: `{_pct(overview.get('avg_signed_return'))}`",
        f"- win_rate: `{_pct(overview.get('win_rate'))}`",
        f"- pending_count: `{overview['pending_count']}`",
        "",
        "## Recommendations",
        "",
    ]
    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.extend(
            f"- [{item.get('level')}] {item.get('topic')}: {item.get('group') or '-'} - {item.get('message')}"
            for item in recommendations
        )
    else:
        lines.append("- None")
    lines.extend(_render_group("By Intent", report["by_intent"], ["intent_type"]))
    lines.extend(_render_group("By Side", report["by_side"], ["side"]))
    lines.extend(_render_group("By Outcome", report["by_outcome"], ["outcome_label"]))
    lines.extend(_render_group("By Rank Bucket", report["by_rank_bucket"], ["rank_bucket"]))
    lines.extend(_render_group("By Confidence Bucket", report["by_confidence_bucket"], ["confidence_bucket"]))
    lines.extend(_render_group("By Engine Type", report["by_engine_type"], ["engine_type"]))
    lines.extend(_render_group("By Entry Gate Status", report["by_entry_gate_status"], ["entry_gate_status"]))
    lines.extend(_render_group("By Review Status", report["by_review_status"], ["review_status"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = build_report(args.reviewer)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    upsert_json_payload(
        "live_trade_review_summary",
        report,
        asof_date=report["overview"].get("latest_review_date"),
        generated_at=report["generated_at"],
        source_path=out_json,
    )
    print(f"live_trade_review_summary_json: {out_json}")
    print(f"live_trade_review_summary_md: {out_md}")
    print(f"review_count: {report['overview']['review_count']}")
    print(f"observed_count: {report['overview']['observed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
