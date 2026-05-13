from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import logging
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_rank_report_config, load_us_rule_ranking_config, parse_iso_date
from python.us.us_db import get_us_engine


LOGGER = logging.getLogger("us_rank_validate")


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    report_cfg = load_us_rank_report_config()
    parser = argparse.ArgumentParser(description="Validate daily US stock rank rows.")
    parser.add_argument("--trade-date", required=True, help="Trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=None, help="Optional limit for validation subset.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero exit code if validation errors exist.")
    parser.add_argument("--output", choices=["console", "markdown"], default="console", help="Validation output format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=report_cfg.output_dir,
        help="Directory used for markdown validation output.",
    )
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: object, digits: int = 1) -> str:
    num = _safe_float(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def _parse_score_detail(value: object) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "score_detail_json missing"
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            return None, f"score_detail_json parse failed: {exc}"
        if isinstance(loaded, dict):
            return loaded, None
    return None, f"score_detail_json has unsupported type {type(value).__name__}"


def _expected_grade(total_score: float, *, cfg) -> str:
    if total_score >= cfg.strong_buy_score:
        return "STRONG_BUY"
    if total_score >= cfg.buy_score:
        return "BUY"
    if total_score >= cfg.watch_score:
        return "WATCH"
    if total_score >= cfg.hold_score:
        return "HOLD"
    return "EXCLUDE"


def validate_rank_result(row: dict[str, object], *, cfg) -> list[str]:
    messages: list[str] = []
    detail, detail_error = _parse_score_detail(row.get("score_detail_json"))
    if detail_error:
        messages.append(f"ERROR: {detail_error}")

    score_ranges = {
        "momentum_score": (0.0, 25.0),
        "relative_strength_score": (0.0, 20.0),
        "fundamental_score": (0.0, 20.0),
        "growth_score": (0.0, 15.0),
        "valuation_score": (0.0, 10.0),
        "risk_score": (-10.0, 0.0),
        "total_score": (0.0, 100.0),
    }
    for field, (lower, upper) in score_ranges.items():
        value = _safe_float(row.get(field))
        if value is None:
            messages.append(f"WARNING: {field} missing")
            continue
        if value < lower or value > upper:
            messages.append(f"ERROR: {field} out of range {value:.4f} not in [{lower}, {upper}]")

    grade = str(row.get("recommend_grade") or "").strip().upper()
    total_score = _safe_float(row.get("total_score")) or 0.0
    exclude_reason = str(row.get("exclude_reason") or "").strip()
    expected_grade = _expected_grade(total_score, cfg=cfg)
    if grade == "EXCLUDE":
        if not exclude_reason:
            messages.append("ERROR: EXCLUDE row missing exclude_reason")
    elif grade and grade != expected_grade:
        messages.append(f"ERROR: recommend_grade {grade} inconsistent with total_score {total_score:.1f} expected {expected_grade}")

    reason_summary = str(row.get("reason_summary") or "").strip()
    if not reason_summary:
        messages.append("ERROR: reason_summary missing")

    if detail:
        meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
        reason_category = meta.get("reason_category") if isinstance(meta, dict) else None
        reason_tags = meta.get("reason_tags") if isinstance(meta, dict) else None
        if not reason_category:
            messages.append("WARNING: reason_category missing in score_detail_json.meta")
        if not isinstance(reason_tags, list) or not reason_tags:
            messages.append("WARNING: reason_tags missing in score_detail_json.meta")

    feature_quality = _safe_float(row.get("feature_quality_score"))
    is_etf = bool(row.get("is_etf"))
    risk_score = _safe_float(row.get("risk_score")) or 0.0
    valuation_score = _safe_float(row.get("valuation_score")) or 0.0
    return_20d = None
    volatility_20d = None
    trailing_pe = None
    price_to_book = None
    debt_to_equity = None
    if detail:
        momentum_inputs = (detail.get("momentum") or {}).get("inputs") if isinstance(detail.get("momentum"), dict) else {}
        risk_inputs = (detail.get("risk") or {}).get("inputs") if isinstance(detail.get("risk"), dict) else {}
        valuation_inputs = (detail.get("valuation") or {}).get("inputs") if isinstance(detail.get("valuation"), dict) else {}
        fundamental_inputs = (detail.get("fundamental") or {}).get("inputs") if isinstance(detail.get("fundamental"), dict) else {}
        return_20d = _safe_float((momentum_inputs or {}).get("return_20d"))
        volatility_20d = _safe_float((risk_inputs or {}).get("volatility_20d"))
        trailing_pe = _safe_float((valuation_inputs or {}).get("trailing_pe"))
        price_to_book = _safe_float((valuation_inputs or {}).get("price_to_book"))
        debt_to_equity = _safe_float((fundamental_inputs or {}).get("debt_to_equity"))

    if total_score >= 95:
        messages.append("WARNING: total_score at or above 95")
    if risk_score <= -8 and grade in {"BUY", "STRONG_BUY"}:
        messages.append("WARNING: high risk penalty with BUY-or-better grade")
    if valuation_score <= 0 and grade == "STRONG_BUY":
        messages.append("WARNING: STRONG_BUY despite zero valuation score")
    if not is_etf and (feature_quality or 0.0) < 40 and grade in {"WATCH", "BUY", "STRONG_BUY"}:
        messages.append("WARNING: non-ETF has low feature quality but still ranks WATCH-or-better")
    if return_20d is not None and return_20d > 0.50:
        messages.append("WARNING: 20d return above 50%")
    if volatility_20d is not None and volatility_20d > 0.10:
        messages.append("WARNING: 20d volatility above 10%")
    if trailing_pe is not None and trailing_pe > 300:
        messages.append("WARNING: trailing PE above 300")
    if price_to_book is not None and price_to_book > 100:
        messages.append("WARNING: price-to-book above 100")
    if debt_to_equity is not None and debt_to_equity > 10:
        messages.append("WARNING: debt_to_equity unusually large")

    return messages


def fetch_rank_rows(*, trade_date: date, top_n: int | None = None) -> list[dict[str, object]]:
    limit_clause = ""
    params: dict[str, object] = {"trade_date": trade_date}
    if top_n is not None:
        limit_clause = "LIMIT :limit_n"
        params["limit_n"] = top_n
    stmt = text(
        f"""
        SELECT
            trade_date,
            rank_no,
            symbol,
            company_name,
            recommend_grade,
            total_score,
            momentum_score,
            relative_strength_score,
            fundamental_score,
            growth_score,
            valuation_score,
            risk_score,
            feature_quality_score,
            is_etf,
            data_status,
            exclude_reason,
            reason_summary,
            score_detail_json
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
        ORDER BY rank_no ASC NULLS LAST, symbol ASC
        {limit_clause}
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def summarize_validation(rows: list[dict[str, object]], *, cfg, top_n: int = 20) -> dict[str, object]:
    summary = {
        "total_checked": len(rows),
        "valid_count": 0,
        "warning_count": 0,
        "error_count": 0,
        "missing_reason_count": 0,
        "invalid_json_count": 0,
        "score_range_error_count": 0,
        "exclude_without_reason_count": 0,
        "low_quality_data_count": 0,
        "high_risk_top_count": 0,
        "expensive_valuation_top_count": 0,
    }
    row_results: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        messages = validate_rank_result(row, cfg=cfg)
        warnings = [msg for msg in messages if msg.startswith("WARNING:")]
        errors = [msg for msg in messages if msg.startswith("ERROR:")]
        if not messages:
            summary["valid_count"] += 1
        summary["warning_count"] += len(warnings)
        summary["error_count"] += len(errors)
        if any("reason_summary missing" in msg for msg in messages):
            summary["missing_reason_count"] += 1
        if any("score_detail_json" in msg for msg in messages):
            summary["invalid_json_count"] += 1
        if any("out of range" in msg for msg in messages):
            summary["score_range_error_count"] += 1
        if any("EXCLUDE row missing exclude_reason" in msg for msg in messages):
            summary["exclude_without_reason_count"] += 1
        if (_safe_float(row.get("feature_quality_score")) or 0.0) < 40:
            summary["low_quality_data_count"] += 1
        if index <= top_n and (_safe_float(row.get("risk_score")) or 0.0) <= -7:
            summary["high_risk_top_count"] += 1
        if index <= top_n and (_safe_float(row.get("valuation_score")) or 0.0) <= 2:
            summary["expensive_valuation_top_count"] += 1
        row_results.append({"row": row, "messages": messages})
    summary["row_results"] = row_results
    return summary


def build_validation_summary_text(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "[Validation Summary]",
            f"Total Checked: {summary.get('total_checked', 0)}",
            f"Valid: {summary.get('valid_count', 0)}",
            f"Warnings: {summary.get('warning_count', 0)}",
            f"Errors: {summary.get('error_count', 0)}",
            f"Missing Reason: {summary.get('missing_reason_count', 0)}",
            f"Invalid JSON: {summary.get('invalid_json_count', 0)}",
            f"Score Range Errors: {summary.get('score_range_error_count', 0)}",
            f"EXCLUDE without Reason: {summary.get('exclude_without_reason_count', 0)}",
            f"Low Quality Data: {summary.get('low_quality_data_count', 0)}",
            f"High Risk in Top 20: {summary.get('high_risk_top_count', 0)}",
            f"Expensive Valuation in Top 20: {summary.get('expensive_valuation_top_count', 0)}",
        ]
    )


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def build_validation_markdown(trade_date: date, summary: dict[str, object]) -> str:
    issue_rows: list[dict[str, object]] = []
    for item in summary.get("row_results", [])[:]:
        row = item["row"]
        messages = item["messages"]
        if not messages:
            continue
        issue_rows.append(
            {
                "symbol": row.get("symbol"),
                "grade": row.get("recommend_grade"),
                "total_score": _fmt_num(row.get("total_score")),
                "messages": "; ".join(messages[:3]),
            }
        )
    lines = [
        f"# US Stock Rank Validation: {trade_date.isoformat()}",
        "",
        build_validation_summary_text(summary),
        "",
        "## Issue Rows",
        "",
        _markdown_table(issue_rows[:50], [("symbol", "Symbol"), ("grade", "Grade"), ("total_score", "Total"), ("messages", "Messages")]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_cfg = load_us_rank_report_config()
    setup_logging(report_cfg.log_level)
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    if trade_date is None:
        raise SystemExit("trade_date is required.")

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_RANK_VALIDATE] DB connection failed: {exc}") from exc

    rows = fetch_rank_rows(trade_date=trade_date, top_n=args.top_n)
    if not rows:
        LOGGER.info("[US_RANK_VALIDATE] No rank rows found for %s", trade_date.isoformat())
        return 1

    cfg = load_us_rule_ranking_config()
    summary = summarize_validation(rows, cfg=cfg)

    if args.output == "console":
        print(build_validation_summary_text(summary))
        issue_counter = 0
        for item in summary["row_results"]:
            if not item["messages"]:
                continue
            issue_counter += 1
            if issue_counter > 20:
                break
            row = item["row"]
            print(f"- {row.get('symbol')} ({row.get('recommend_grade')}): {'; '.join(item['messages'][:3])}")
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else Path(__file__).resolve().parents[2] / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"us_stock_rank_validation_{trade_date.isoformat()}.md"
        rendered = build_validation_markdown(trade_date, summary)
        output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        print(f"markdown_path: {output_path}")

    if args.fail_on_error and int(summary.get("error_count", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
