from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import logging
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.calculate_us_stock_rule_scores import calculate_rule_scores
from python.us.us_config import (
    load_us_rank_report_config,
    load_us_rule_ranking_config,
    parse_iso_date,
)
from python.us.us_db import get_us_engine
from python.us.validate_us_stock_rank_daily import build_validation_summary_text, summarize_validation


LOGGER = logging.getLogger("us_rank_report")
SUPPORTED_FORMATS = {"console", "markdown", "csv"}
CSV_COLUMNS = [
    "trade_date",
    "rank_no",
    "symbol",
    "company_name",
    "sector",
    "industry",
    "recommend_grade",
    "total_score",
    "momentum_score",
    "relative_strength_score",
    "fundamental_score",
    "growth_score",
    "valuation_score",
    "risk_score",
    "feature_quality_score",
    "reason_category",
    "reason_tags",
    "data_status",
    "exclude_reason",
    "reason_summary",
]


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    report_cfg = load_us_rank_report_config()
    parser = argparse.ArgumentParser(description="Build/read US stock top-rank report outputs.")
    parser.add_argument("--trade-date", required=True, help="Requested trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=report_cfg.top_n, help="Top N rows for non-symbol report.")
    parser.add_argument("--grade", default=None, help="Optional grade filter such as BUY or STRONG_BUY.")
    parser.add_argument("--symbol", default=None, help="Optional symbol detail lookup such as NVDA.")
    parser.add_argument("--show-excluded", action="store_true", help="Show excluded rows instead of Top-N eligible rows.")
    parser.add_argument("--limit", type=int, default=50, help="Row limit for excluded list mode.")
    parser.add_argument(
        "--format",
        default="console",
        choices=sorted(SUPPORTED_FORMATS),
        help="Output format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=report_cfg.output_dir,
        help="Directory used for markdown/csv outputs.",
    )
    parser.add_argument(
        "--auto-calculate",
        action="store_true",
        help="If the requested trade date has no rank rows, calculate scores first.",
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


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _truncate(text_value: object, limit: int = 100) -> str:
    text = str(text_value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalize_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _parse_score_detail(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            LOGGER.warning("[US_RANK_REPORT] score_detail_json parse failed: %s", exc)
            return None
        if isinstance(loaded, dict):
            return loaded
    LOGGER.warning("[US_RANK_REPORT] score_detail_json type unsupported: %s", type(value).__name__)
    return None


def _extract_reason_meta(row: dict[str, object]) -> tuple[str | None, list[str]]:
    detail = _parse_score_detail(row.get("score_detail_json")) or {}
    meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
    category = meta.get("reason_category") if isinstance(meta, dict) else None
    tags = meta.get("reason_tags") if isinstance(meta, dict) and isinstance(meta.get("reason_tags"), list) else []
    return (str(category) if category else None), [str(tag) for tag in tags]


def _resolve_output_path(
    *,
    output_dir: Path,
    trade_date: date,
    fmt: str,
    top_n: int,
    grade: str | None,
    symbol: str | None,
) -> Path:
    suffix = "md" if fmt == "markdown" else "csv"
    if symbol:
        name = f"us_stock_{symbol.lower()}_{trade_date.isoformat()}.{suffix}"
    elif grade:
        name = f"us_stock_{grade.lower()}_{trade_date.isoformat()}.{suffix}"
    else:
        name = f"us_stock_top{top_n}_{trade_date.isoformat()}.{suffix}"
    return output_dir / name


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        headers = [header for _, header in columns]
        divider = ["---" if not header.endswith(")") else "---" for header in headers]
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(divider) + " |",
            ]
        )
    rendered: list[list[str]] = []
    for row in rows:
        rendered.append([str(row.get(key, "")) for key, _ in columns])
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in rendered:
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _query_any_rows_for_date(trade_date: date) -> int:
    stmt = text(
        """
        SELECT COUNT(*) AS row_count
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
        """
    )
    with get_us_engine().connect() as conn:
        return int(conn.execute(stmt, {"trade_date": trade_date}).scalar_one() or 0)


def _query_top_rows(*, trade_date: date, top_n: int, grade: str | None) -> list[dict[str, object]]:
    clauses = [
        "trade_date = :trade_date",
        "recommend_grade <> 'EXCLUDE'",
        "rank_no IS NOT NULL",
    ]
    params: dict[str, object] = {"trade_date": trade_date, "limit_n": top_n}
    if grade:
        clauses.append("recommend_grade = :grade")
        params["grade"] = grade

    stmt = text(
        f"""
        SELECT
            trade_date,
            rank_no,
            symbol,
            company_name,
            sector,
            industry,
            recommend_grade,
            total_score,
            momentum_score,
            relative_strength_score,
            fundamental_score,
            growth_score,
            valuation_score,
            risk_score,
            feature_quality_score,
            reason_summary,
            score_detail_json,
            source
        FROM recommend.us_stock_rank_daily
        WHERE {' AND '.join(clauses)}
        ORDER BY rank_no ASC
        LIMIT :limit_n
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def _query_all_rows(trade_date: date) -> list[dict[str, object]]:
    stmt = text(
        """
        SELECT
            trade_date,
            rank_no,
            symbol,
            company_name,
            sector,
            industry,
            recommend_grade,
            total_score,
            momentum_score,
            relative_strength_score,
            fundamental_score,
            growth_score,
            valuation_score,
            risk_score,
            feature_quality_score,
            reason_summary,
            score_detail_json,
            source,
            data_status,
            exclude_reason,
            is_etf
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
        ORDER BY rank_no ASC NULLS LAST, symbol ASC
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, {"trade_date": trade_date}).mappings().all()
    return [dict(row) for row in rows]


def _query_excluded_rows(*, trade_date: date, limit_n: int) -> list[dict[str, object]]:
    stmt = text(
        """
        SELECT
            trade_date,
            rank_no,
            symbol,
            company_name,
            sector,
            industry,
            recommend_grade,
            total_score,
            risk_score,
            feature_quality_score,
            reason_summary,
            score_detail_json,
            source,
            data_status,
            exclude_reason,
            is_etf
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
          AND recommend_grade = 'EXCLUDE'
        ORDER BY rank_no ASC NULLS LAST, symbol ASC
        LIMIT :limit_n
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, {"trade_date": trade_date, "limit_n": limit_n}).mappings().all()
    return [dict(row) for row in rows]


def _query_symbol_row(*, trade_date: date, symbol: str) -> dict[str, object] | None:
    stmt = text(
        """
        SELECT
            trade_date,
            rank_no,
            symbol,
            company_name,
            sector,
            industry,
            recommend_grade,
            total_score,
            momentum_score,
            relative_strength_score,
            fundamental_score,
            growth_score,
            valuation_score,
            risk_score,
            feature_quality_score,
            reason_summary,
            score_detail_json,
            source,
            data_status,
            exclude_reason
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
          AND symbol = :symbol
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT 1
        """
    )
    with get_us_engine().connect() as conn:
        row = conn.execute(stmt, {"trade_date": trade_date, "symbol": symbol}).mappings().first()
    return None if row is None else dict(row)


def _query_summary_stats(trade_date: date) -> dict[str, object]:
    stmt = text(
        """
        SELECT
            trade_date,
            COUNT(*)::integer AS total_ranked,
            COUNT(*) FILTER (WHERE recommend_grade <> 'EXCLUDE')::integer AS eligible_count,
            COUNT(*) FILTER (WHERE recommend_grade = 'STRONG_BUY')::integer AS strong_buy_count,
            COUNT(*) FILTER (WHERE recommend_grade = 'BUY')::integer AS buy_count,
            COUNT(*) FILTER (WHERE recommend_grade = 'WATCH')::integer AS watch_count,
            COUNT(*) FILTER (WHERE recommend_grade = 'HOLD')::integer AS hold_count,
            COUNT(*) FILTER (WHERE recommend_grade = 'EXCLUDE')::integer AS exclude_count,
            AVG(total_score) AS avg_total_score,
            MAX(total_score) AS max_total_score,
            MIN(total_score) AS min_total_score,
            AVG(feature_quality_score) AS avg_feature_quality_score,
            AVG(momentum_score) AS avg_momentum_score,
            AVG(relative_strength_score) AS avg_relative_strength_score,
            AVG(fundamental_score) AS avg_fundamental_score,
            AVG(risk_score) AS avg_risk_score
        FROM recommend.us_stock_rank_daily
        WHERE trade_date = :trade_date
        GROUP BY trade_date
        """
    )
    with get_us_engine().connect() as conn:
        row = conn.execute(stmt, {"trade_date": trade_date}).mappings().first()
    return {} if row is None else dict(row)


def _prepare_top_rows(rows: list[dict[str, object]], *, console_reason_limit: int) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for row in rows:
        category, tags = _extract_reason_meta(row)
        prepared.append(
            {
                "Rank": row.get("rank_no"),
                "Symbol": row.get("symbol"),
                "Grade": row.get("recommend_grade"),
                "Total": _fmt_num(row.get("total_score")),
                "Category": category or "-",
                "Tags": _truncate(", ".join(tags), limit=40) if tags else "-",
                "Mom": _fmt_num(row.get("momentum_score")),
                "RS": _fmt_num(row.get("relative_strength_score")),
                "Fund": _fmt_num(row.get("fundamental_score")),
                "Growth": _fmt_num(row.get("growth_score")),
                "Val": _fmt_num(row.get("valuation_score")),
                "Risk": _fmt_num(row.get("risk_score")),
                "Reason": _truncate(row.get("reason_summary"), limit=console_reason_limit),
            }
        )
    return prepared


def _fixed_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = " | ".join(str(column).ljust(widths[column]) for column in columns)
    divider = "-|-".join("-" * widths[column] for column in columns)
    body = [" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns) for row in rows]
    return "\n".join([header, divider, *body])


def build_summary_text(stats: dict[str, object]) -> str:
    return "\n".join(
        [
            "[Summary]",
            f"Trade Date: {_format_date(stats.get('trade_date'))}",
            f"Total Ranked: {stats.get('total_ranked', 0)}",
            f"Eligible: {stats.get('eligible_count', 0)}",
            f"STRONG_BUY: {stats.get('strong_buy_count', 0)}",
            f"BUY: {stats.get('buy_count', 0)}",
            f"WATCH: {stats.get('watch_count', 0)}",
            f"HOLD: {stats.get('hold_count', 0)}",
            f"EXCLUDE: {stats.get('exclude_count', 0)}",
            f"Avg Total Score: {_fmt_num(stats.get('avg_total_score'))}",
            f"Max Total Score: {_fmt_num(stats.get('max_total_score'))}",
            f"Min Total Score: {_fmt_num(stats.get('min_total_score'))}",
            f"Avg Feature Quality: {_fmt_num(stats.get('avg_feature_quality_score'))}",
            f"Avg Momentum: {_fmt_num(stats.get('avg_momentum_score'))}",
            f"Avg Relative Strength: {_fmt_num(stats.get('avg_relative_strength_score'))}",
            f"Avg Fundamental: {_fmt_num(stats.get('avg_fundamental_score'))}",
            f"Avg Risk: {_fmt_num(stats.get('avg_risk_score'))}",
        ]
    )


def build_console_report(*, trade_date: date, rows: list[dict[str, object]], stats: dict[str, object], top_n: int) -> str:
    table_rows = _prepare_top_rows(rows, console_reason_limit=100)
    title = f"[US Stock Top {top_n} Ranking]"
    source = str(rows[0].get("source") or "unknown") if rows else "unknown"
    lines = [
        title,
        f"Trade Date: {trade_date.isoformat()}",
        f"Source: {source}",
        "",
        _fixed_table(table_rows, ["Rank", "Symbol", "Grade", "Total", "Category", "Tags", "Mom", "RS", "Fund", "Growth", "Val", "Risk", "Reason"]),
        "",
        build_summary_text(stats),
    ]
    return "\n".join(lines).strip() + "\n"


def build_markdown_report(
    *,
    trade_date: date,
    rows: list[dict[str, object]],
    stats: dict[str, object],
    top_n: int,
    grade: str | None,
) -> str:
    source = str(rows[0].get("source") or "unknown") if rows else "unknown"
    summary_rows: list[dict[str, object]] = []
    reason_sections: list[str] = []
    for row in rows:
        category, tags = _extract_reason_meta(row)
        summary_rows.append(
            {
                "rank_no": row.get("rank_no"),
                "symbol": row.get("symbol"),
                "company_name": row.get("company_name") or "",
                "recommend_grade": row.get("recommend_grade") or "",
                "total_score": _fmt_num(row.get("total_score")),
                "reason_category": category or "-",
                "reason_tags": ", ".join(tags) if tags else "-",
                "momentum_score": _fmt_num(row.get("momentum_score")),
                "relative_strength_score": _fmt_num(row.get("relative_strength_score")),
                "fundamental_score": _fmt_num(row.get("fundamental_score")),
                "growth_score": _fmt_num(row.get("growth_score")),
                "valuation_score": _fmt_num(row.get("valuation_score")),
                "risk_score": _fmt_num(row.get("risk_score")),
            }
        )
        reason_sections.extend(
            [
                f"### {row.get('rank_no')}. {row.get('symbol')}",
                "",
                f"- company: {row.get('company_name') or '-'}",
                f"- grade: {row.get('recommend_grade') or '-'}",
                f"- total_score: {_fmt_num(row.get('total_score'))}",
                f"- reason_category: {category or '-'}",
                f"- reason_tags: {', '.join(tags) if tags else '-'}",
                f"- feature_quality_score: {_fmt_num(row.get('feature_quality_score'))}",
                f"- risk_score: {_fmt_num(row.get('risk_score'))}",
                f"- data_status: {row.get('data_status') or '-'}",
                f"- key_reason: {row.get('reason_summary') or '-'}",
                "",
            ]
        )

    filter_line = grade if grade else f"Top {top_n}"
    lines = [
        f"# US Stock {filter_line} Report",
        "",
        f"- trade_date: {trade_date.isoformat()}",
        f"- source: {source}",
        "- target: recommend.us_stock_rank_daily",
        "- default_exclusions: recommend_grade=EXCLUDE, leveraged ETF, inverse ETF",
        "",
        "## Top Summary",
        "",
        _markdown_table(
            summary_rows,
            [
                ("rank_no", "Rank"),
                ("symbol", "Symbol"),
                ("company_name", "Company"),
                ("recommend_grade", "Grade"),
                ("total_score", "Total"),
                ("reason_category", "Category"),
                ("reason_tags", "Tags"),
                ("momentum_score", "Momentum"),
                ("relative_strength_score", "Relative Strength"),
                ("fundamental_score", "Fundamental"),
                ("growth_score", "Growth"),
                ("valuation_score", "Valuation"),
                ("risk_score", "Risk"),
            ],
        ),
        "",
        "## Recommendation Reasons",
        "",
        *reason_sections,
        "## Check Points",
        "",
        f"- STRONG_BUY count: {stats.get('strong_buy_count', 0)}",
        f"- BUY count: {stats.get('buy_count', 0)}",
        f"- WATCH count: {stats.get('watch_count', 0)}",
        f"- HOLD count: {stats.get('hold_count', 0)}",
        f"- EXCLUDE count: {stats.get('exclude_count', 0)}",
        f"- average total_score: {_fmt_num(stats.get('avg_total_score'))}",
        f"- average feature_quality_score: {_fmt_num(stats.get('avg_feature_quality_score'))}",
        f"- average momentum_score: {_fmt_num(stats.get('avg_momentum_score'))}",
        f"- average relative_strength_score: {_fmt_num(stats.get('avg_relative_strength_score'))}",
        f"- average fundamental_score: {_fmt_num(stats.get('avg_fundamental_score'))}",
        f"- average risk_score: {_fmt_num(stats.get('avg_risk_score'))}",
        "",
    ]
    return "\n".join(lines)


def build_detail_console_report(*, trade_date: date, row: dict[str, object]) -> str:
    detail = _parse_score_detail(row.get("score_detail_json")) or {}
    meta = detail.get("meta", {}) if isinstance(detail.get("meta"), dict) else {}
    category = meta.get("reason_category") if isinstance(meta, dict) else None
    tags = meta.get("reason_tags") if isinstance(meta, dict) and isinstance(meta.get("reason_tags"), list) else []
    section_lines = []
    section_specs = [
        ("Momentum", "momentum", "max_score"),
        ("Relative Strength", "relative_strength", "max_score"),
        ("Fundamental", "fundamental", "max_score"),
        ("Growth", "growth", "max_score"),
        ("Valuation", "valuation", "max_score"),
        ("Risk", "risk", "min_score"),
    ]
    for label, key, bound_key in section_specs:
        section = detail.get(key, {}) if isinstance(detail.get(key), dict) else {}
        bound = section.get(bound_key)
        if key == "risk" and bound is not None:
            bound_text = f"{bound}~0"
        else:
            bound_text = str(bound) if bound is not None else "?"
        section_lines.append(f"{label}: {_fmt_num(section.get('score') if section else row.get(f'{key}_score'))} / {bound_text}")
        for reason in list(section.get("reasons") or [])[:3]:
            section_lines.append(f"- {reason}")
        missing_fields = list(section.get("missing_fields") or [])
        if missing_fields:
            section_lines.append(f"- missing_fields: {', '.join(str(item) for item in missing_fields[:5])}")

    detail_json_text = json.dumps(detail, ensure_ascii=False, indent=2) if detail else str(row.get("score_detail_json") or "")
    detail_preview = detail_json_text if len(detail_json_text) <= 1200 else detail_json_text[:1200].rstrip() + "\n..."
    return "\n".join(
        [
            "[US Stock Ranking Detail]",
            f"Trade Date: {trade_date.isoformat()}",
            f"Symbol: {row.get('symbol')}",
            f"Company: {row.get('company_name') or '-'}",
            f"Rank: {row.get('rank_no') or '-'}",
            f"Grade: {row.get('recommend_grade') or '-'}",
            f"Category: {category or '-'}",
            f"Tags: {', '.join(str(tag) for tag in tags) if tags else '-'}",
            f"Total Score: {_fmt_num(row.get('total_score'))}",
            f"Data Status: {meta.get('data_status') or row.get('data_status') or '-'}",
            "",
            "[Score Breakdown]",
            *section_lines,
            "",
            "[Reason]",
            str(row.get("reason_summary") or "-"),
            "",
            "[Detail JSON]",
            detail_preview,
        ]
    ).strip() + "\n"


def build_detail_markdown_report(*, trade_date: date, row: dict[str, object]) -> str:
    detail = _parse_score_detail(row.get("score_detail_json")) or {}
    meta = detail.get("meta", {}) if isinstance(detail.get("meta"), dict) else {}
    category = meta.get("reason_category") if isinstance(meta, dict) else None
    tags = meta.get("reason_tags") if isinstance(meta, dict) and isinstance(meta.get("reason_tags"), list) else []
    pretty_json = json.dumps(detail, ensure_ascii=False, indent=2) if detail else str(row.get("score_detail_json") or "")
    return "\n".join(
        [
            f"# US Stock Ranking Detail: {row.get('symbol')}",
            "",
            f"- trade_date: {trade_date.isoformat()}",
            f"- company: {row.get('company_name') or '-'}",
            f"- rank: {row.get('rank_no') or '-'}",
            f"- grade: {row.get('recommend_grade') or '-'}",
            f"- reason_category: {category or '-'}",
            f"- reason_tags: {', '.join(str(tag) for tag in tags) if tags else '-'}",
            f"- total_score: {_fmt_num(row.get('total_score'))}",
            f"- feature_quality_score: {_fmt_num(row.get('feature_quality_score'))}",
            f"- data_status: {row.get('data_status') or '-'}",
            f"- exclude_reason: {row.get('exclude_reason') or '-'}",
            "",
            "## Score Breakdown",
            "",
            f"- momentum_score: {_fmt_num(row.get('momentum_score'))}",
            f"- relative_strength_score: {_fmt_num(row.get('relative_strength_score'))}",
            f"- fundamental_score: {_fmt_num(row.get('fundamental_score'))}",
            f"- growth_score: {_fmt_num(row.get('growth_score'))}",
            f"- valuation_score: {_fmt_num(row.get('valuation_score'))}",
            f"- risk_score: {_fmt_num(row.get('risk_score'))}",
            "",
            "## Section Notes",
            "",
            *(f"- momentum: {reason}" for reason in list((detail.get("momentum") or {}).get("reasons") or [])[:3]),
            *(f"- relative_strength: {reason}" for reason in list((detail.get("relative_strength") or {}).get("reasons") or [])[:3]),
            *(f"- fundamental: {reason}" for reason in list((detail.get("fundamental") or {}).get("reasons") or [])[:3]),
            *(f"- growth: {reason}" for reason in list((detail.get("growth") or {}).get("reasons") or [])[:3]),
            *(f"- valuation: {reason}" for reason in list((detail.get("valuation") or {}).get("reasons") or [])[:3]),
            *(f"- risk: {reason}" for reason in list((detail.get("risk") or {}).get("reasons") or [])[:3]),
            "",
            "## Reason",
            "",
            str(row.get("reason_summary") or "-"),
            "",
            "## Detail JSON",
            "",
            "```json",
            pretty_json,
            "```",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            category, tags = _extract_reason_meta(row)
            payload = {key: row.get(key) for key in CSV_COLUMNS}
            payload["reason_category"] = category
            payload["reason_tags"] = ", ".join(tags) if tags else None
            writer.writerow(payload)


def build_validation_text_for_report(trade_date: date, all_rows: list[dict[str, object]]) -> str:
    cfg = load_us_rule_ranking_config()
    summary = summarize_validation(all_rows, cfg=cfg)
    return build_validation_summary_text(summary)


def build_excluded_console_report(*, trade_date: date, rows: list[dict[str, object]], limit_n: int) -> str:
    prepared: list[dict[str, object]] = []
    for row in rows:
        category, tags = _extract_reason_meta(row)
        prepared.append(
            {
                "Symbol": row.get("symbol"),
                "Company": _truncate(row.get("company_name"), limit=20),
                "Grade": row.get("recommend_grade"),
                "Total": _fmt_num(row.get("total_score")),
                "Category": category or "-",
                "Tags": _truncate(", ".join(tags), limit=35) if tags else "-",
                "Status": row.get("data_status") or "-",
                "FQ": _fmt_num(row.get("feature_quality_score")),
                "Exclude Reason": _truncate(row.get("exclude_reason"), limit=70),
            }
        )
    return "\n".join(
        [
            f"[US Stock Excluded List]",
            f"Trade Date: {trade_date.isoformat()}",
            f"Limit: {limit_n}",
            "",
            _fixed_table(prepared, ["Symbol", "Company", "Grade", "Total", "Category", "Tags", "Status", "FQ", "Exclude Reason"]),
        ]
    ).strip() + "\n"


def maybe_notify(*, enabled: bool, title: str, message: str, details: dict[str, object]) -> None:
    if not enabled:
        return
    try:
        from python.notifier import notify_info
    except Exception as exc:
        LOGGER.warning("[US_RANK_REPORT] notifier import failed: %s", exc)
        return
    notify_info(title, message, details)


def maybe_auto_calculate(*, trade_date: date) -> date:
    rule_cfg = load_us_rule_ranking_config()
    result = calculate_rule_scores(
        trade_date=trade_date,
        symbols=None,
        dry_run=False,
        top_n=load_us_rank_report_config().top_n,
        source=rule_cfg.source,
        cfg=rule_cfg,
    )
    LOGGER.info(
        "[US_RANK_REPORT] auto_calculate finished requested_trade_date=%s effective_trade_date=%s written=%s",
        trade_date.isoformat(),
        result.trade_date.isoformat(),
        result.written_row_count,
    )
    return result.trade_date


def _ensure_db() -> None:
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_RANK_REPORT] DB connection failed: {exc}") from exc


def run_top_report(
    *,
    trade_date: date,
    top_n: int,
    grade: str | None,
    fmt: str,
    output_dir: Path,
    email_enabled: bool,
) -> int:
    stats = _query_summary_stats(trade_date)
    rows = _query_top_rows(trade_date=trade_date, top_n=top_n, grade=grade)
    all_rows = _query_all_rows(trade_date)
    if not rows:
        if stats:
            LOGGER.info("[US_RANK_REPORT] Top-N result is empty. Check EXCLUDE conditions or the grade filter.")
        else:
            LOGGER.info(
                "[US_RANK_REPORT] No ranking rows found for %s. Run scripts/calculate_us_stock_rule_scores.py first.",
                trade_date.isoformat(),
            )
        return 1

    if fmt == "console":
        print(build_console_report(trade_date=trade_date, rows=rows, stats=stats, top_n=top_n), end="")
        print()
        print(build_validation_text_for_report(trade_date, all_rows))
        return 0

    output_path = _resolve_output_path(
        output_dir=output_dir,
        trade_date=trade_date,
        fmt=fmt,
        top_n=top_n,
        grade=grade,
        symbol=None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "markdown":
        rendered = build_markdown_report(trade_date=trade_date, rows=rows, stats=stats, top_n=top_n, grade=grade)
        rendered += "\n## Validation Summary\n\n" + build_validation_text_for_report(trade_date, all_rows) + "\n"
        output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    else:
        write_csv(output_path, rows)
        print(build_summary_text(stats))
        print()
        print(build_validation_text_for_report(trade_date, all_rows))
        print(f"csv_path: {output_path}")

    maybe_notify(
        enabled=email_enabled,
        title="US rank report generated",
        message=f"trade_date={trade_date.isoformat()} format={fmt}",
        details={"path": str(output_path), "top_n": top_n, "grade": grade},
    )
    return 0


def run_excluded_report(
    *,
    trade_date: date,
    limit_n: int,
    fmt: str,
    output_dir: Path,
    email_enabled: bool,
) -> int:
    rows = _query_excluded_rows(trade_date=trade_date, limit_n=limit_n)
    if not rows:
        LOGGER.info("[US_RANK_REPORT] No excluded rows found for %s", trade_date.isoformat())
        return 1
    if fmt == "console":
        print(build_excluded_console_report(trade_date=trade_date, rows=rows, limit_n=limit_n), end="")
        return 0
    output_path = _resolve_output_path(
        output_dir=output_dir,
        trade_date=trade_date,
        fmt=fmt,
        top_n=limit_n,
        grade="excluded",
        symbol=None,
    )
    if fmt == "markdown":
        lines = [
            f"# US Stock Excluded Report",
            "",
            f"- trade_date: {trade_date.isoformat()}",
            f"- limit: {limit_n}",
            "",
        ]
        table_rows: list[dict[str, object]] = []
        for row in rows:
            category, tags = _extract_reason_meta(row)
            table_rows.append(
                {
                    "symbol": row.get("symbol"),
                    "company_name": row.get("company_name") or "",
                    "grade": row.get("recommend_grade") or "",
                    "total_score": _fmt_num(row.get("total_score")),
                    "category": category or "-",
                    "tags": ", ".join(tags) if tags else "-",
                    "exclude_reason": row.get("exclude_reason") or "-",
                    "data_status": row.get("data_status") or "-",
                    "feature_quality_score": _fmt_num(row.get("feature_quality_score")),
                }
            )
        lines.append(
            _markdown_table(
                table_rows,
                [
                    ("symbol", "Symbol"),
                    ("company_name", "Company"),
                    ("grade", "Grade"),
                    ("total_score", "Total"),
                    ("category", "Category"),
                    ("tags", "Tags"),
                    ("exclude_reason", "Exclude Reason"),
                    ("data_status", "Data Status"),
                    ("feature_quality_score", "Feature Quality"),
                ],
            )
        )
        rendered = "\n".join(lines) + "\n"
        output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    else:
        write_csv(output_path, rows)
        print(f"csv_path: {output_path}")
    maybe_notify(
        enabled=email_enabled,
        title="US excluded report generated",
        message=f"trade_date={trade_date.isoformat()} format={fmt}",
        details={"path": str(output_path), "limit": limit_n},
    )
    return 0


def run_symbol_report(
    *,
    trade_date: date,
    symbol: str,
    fmt: str,
    output_dir: Path,
    email_enabled: bool,
) -> int:
    row = _query_symbol_row(trade_date=trade_date, symbol=symbol)
    if row is None:
        LOGGER.info(
            "[US_RANK_REPORT] No ranking row for symbol=%s trade_date=%s. The symbol may be outside the universe or ranking is not generated.",
            symbol,
            trade_date.isoformat(),
        )
        return 1

    if fmt == "console":
        print(build_detail_console_report(trade_date=trade_date, row=row), end="")
        return 0

    output_path = _resolve_output_path(
        output_dir=output_dir,
        trade_date=trade_date,
        fmt=fmt,
        top_n=1,
        grade=None,
        symbol=symbol,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        rendered = build_detail_markdown_report(trade_date=trade_date, row=row)
        output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    else:
        write_csv(output_path, [row])
        print(f"csv_path: {output_path}")

    maybe_notify(
        enabled=email_enabled,
        title="US rank detail generated",
        message=f"trade_date={trade_date.isoformat()} symbol={symbol} format={fmt}",
        details={"path": str(output_path), "symbol": symbol},
    )
    return 0


def main() -> int:
    args = parse_args()
    report_cfg = load_us_rank_report_config()
    setup_logging(report_cfg.log_level)
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    if trade_date is None:
        raise SystemExit("trade_date is required.")

    grade = str(args.grade or "").strip().upper() or None
    symbol = str(args.symbol or "").strip().upper() or None
    fmt = str(args.format or "console").strip().lower()
    output_dir = _normalize_output_dir(args.output_dir)
    _ensure_db()

    any_rows = _query_any_rows_for_date(trade_date)
    if any_rows == 0 and args.auto_calculate:
        trade_date = maybe_auto_calculate(trade_date=trade_date)
        any_rows = _query_any_rows_for_date(trade_date)

    if symbol:
        return run_symbol_report(
            trade_date=trade_date,
            symbol=symbol,
            fmt=fmt,
            output_dir=output_dir,
            email_enabled=report_cfg.email_enabled,
        )

    if args.show_excluded:
        return run_excluded_report(
            trade_date=trade_date,
            limit_n=max(1, int(args.limit)),
            fmt=fmt,
            output_dir=output_dir,
            email_enabled=report_cfg.email_enabled,
        )

    return run_top_report(
        trade_date=trade_date,
        top_n=max(1, int(args.top_n)),
        grade=grade,
        fmt=fmt,
        output_dir=output_dir,
        email_enabled=report_cfg.email_enabled,
    )


if __name__ == "__main__":
    raise SystemExit(main())
