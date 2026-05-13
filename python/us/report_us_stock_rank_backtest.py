from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_backtest_report_config, parse_iso_date
from python.us.us_db import get_us_engine


SUPPORTED_FORMATS = {"console", "markdown", "csv"}


def setup_logging(level_name: str) -> None:
    import logging

    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_backtest_report_config()
    parser = argparse.ArgumentParser(description="Report Project C US stock backtest performance.")
    parser.add_argument("--backtest-id", required=True, help="Backtest ID to report.")
    parser.add_argument("--format", default=cfg.default_format, choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--strategy", default=None, help="Optional strategy filter such as US_RANK_TOP20.")
    parser.add_argument("--holding-days", type=int, default=None, help="Optional holding-day filter.")
    parser.add_argument("--start-date", default=None, help="Optional summary/result trade-date lower bound.")
    parser.add_argument("--end-date", default=None, help="Optional summary/result trade-date upper bound.")
    parser.add_argument("--symbol", default=None, help="Optional symbol detail lookup.")
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _fmt_pct(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric * 100:.2f}%"


def _fmt_num(value: object, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.{digits}f}"


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _normalize_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _mean(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return float(statistics.fmean(numbers))


def _median(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return float(statistics.median(numbers))


def _std(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if len(numbers) < 2:
        return None
    return float(statistics.pstdev(numbers))


def _rate(values: list[float | int | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _query_summary_rows(
    *,
    backtest_id: str,
    strategy: str | None,
    holding_days: int | None,
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, object]]:
    clauses = ["backtest_id = :backtest_id"]
    params: dict[str, object] = {"backtest_id": backtest_id}
    if strategy:
        clauses.append("strategy_name = :strategy")
        params["strategy"] = strategy
    if holding_days is not None:
        clauses.append("holding_days = :holding_days")
        params["holding_days"] = holding_days
    if start_date is not None:
        clauses.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        clauses.append("trade_date <= :end_date")
        params["end_date"] = end_date

    stmt = text(
        f"""
        SELECT *
        FROM research.us_stock_rank_backtest_summary
        WHERE {' AND '.join(clauses)}
        ORDER BY trade_date, strategy_name, holding_days
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def _query_result_rows(
    *,
    backtest_id: str,
    strategy: str | None,
    holding_days: int | None,
    start_date: date | None,
    end_date: date | None,
    symbol: str | None,
) -> list[dict[str, object]]:
    clauses = ["r.backtest_id = :backtest_id"]
    params: dict[str, object] = {"backtest_id": backtest_id}
    if strategy:
        clauses.append("r.strategy_name = :strategy")
        params["strategy"] = strategy
    if holding_days is not None:
        clauses.append("r.holding_days = :holding_days")
        params["holding_days"] = holding_days
    if start_date is not None:
        clauses.append("r.trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        clauses.append("r.trade_date <= :end_date")
        params["end_date"] = end_date
    if symbol:
        clauses.append("r.symbol = :symbol")
        params["symbol"] = symbol

    stmt = text(
        f"""
        SELECT
            r.*
        FROM research.us_stock_rank_backtest_result r
        WHERE {' AND '.join(clauses)}
        ORDER BY r.trade_date, r.strategy_name, r.holding_days, r.rank_no, r.symbol
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def _aggregate_strategy_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row.get("strategy_name") or ""), str(row.get("selection_rule") or ""), int(row.get("holding_days") or 0))].append(row)

    aggregates: list[dict[str, object]] = []
    for (strategy_name, selection_rule, holding_days), rows in grouped.items():
        valid_return_rows = [row for row in rows if _safe_float(row.get("avg_return_pct")) is not None]
        avg_excess_spy_values = [_safe_float(row.get("avg_excess_return_vs_spy")) for row in valid_return_rows]
        avg_excess_qqq_values = [_safe_float(row.get("avg_excess_return_vs_qqq")) for row in valid_return_rows]
        avg_return_values = [_safe_float(row.get("avg_return_pct")) for row in valid_return_rows]
        avg_win_spy_values = [_safe_float(row.get("win_rate_vs_spy")) for row in valid_return_rows]
        best_day_row = max(valid_return_rows, key=lambda item: float(item["avg_return_pct"])) if valid_return_rows else None
        worst_day_row = min(valid_return_rows, key=lambda item: float(item["avg_return_pct"])) if valid_return_rows else None
        selected_avg = _mean([_safe_float(row.get("selected_count")) for row in rows])
        return_std = _std(avg_return_values)
        avg_return = _mean(avg_return_values)
        sharpe_like_ratio = None
        if avg_return is not None and return_std not in {None, 0.0}:
            sharpe_like_ratio = avg_return / return_std
        positive_excess_spy_days = sum(1 for value in avg_excess_spy_values if value is not None and value > 0)
        positive_excess_qqq_days = sum(1 for value in avg_excess_qqq_values if value is not None and value > 0)
        aggregates.append(
            {
                "backtest_id": rows[0].get("backtest_id"),
                "strategy_name": strategy_name,
                "selection_rule": selection_rule,
                "holding_days": holding_days,
                "test_days": len(rows),
                "selected_count_avg": round(selected_avg, 4) if selected_avg is not None else None,
                "avg_return_pct": round(avg_return, 6) if avg_return is not None else None,
                "median_return_pct": round(_median(avg_return_values), 6) if _median(avg_return_values) is not None else None,
                "win_rate": round(_mean([_safe_float(row.get("win_rate")) for row in valid_return_rows]), 6) if valid_return_rows else None,
                "avg_excess_return_vs_spy": round(_mean(avg_excess_spy_values), 6) if _mean(avg_excess_spy_values) is not None else None,
                "avg_excess_return_vs_qqq": round(_mean(avg_excess_qqq_values), 6) if _mean(avg_excess_qqq_values) is not None else None,
                "avg_excess_return_vs_universe": round(_mean([_safe_float(row.get("avg_excess_return_vs_universe")) for row in valid_return_rows]), 6) if valid_return_rows and _mean([_safe_float(row.get("avg_excess_return_vs_universe")) for row in valid_return_rows]) is not None else None,
                "win_rate_vs_spy": round(_mean(avg_win_spy_values), 6) if _mean(avg_win_spy_values) is not None else None,
                "win_rate_vs_qqq": round(_mean([_safe_float(row.get("win_rate_vs_qqq")) for row in valid_return_rows]), 6) if valid_return_rows and _mean([_safe_float(row.get("win_rate_vs_qqq")) for row in valid_return_rows]) is not None else None,
                "win_rate_vs_universe": round(_mean([_safe_float(row.get("win_rate_vs_universe")) for row in valid_return_rows]), 6) if valid_return_rows and _mean([_safe_float(row.get("win_rate_vs_universe")) for row in valid_return_rows]) is not None else None,
                "best_day": best_day_row.get("trade_date") if best_day_row else None,
                "worst_day": worst_day_row.get("trade_date") if worst_day_row else None,
                "best_symbol": _most_common_value([row.get("best_symbol") for row in rows]),
                "worst_symbol": _most_common_value([row.get("worst_symbol") for row in rows]),
                "positive_excess_spy_days": positive_excess_spy_days,
                "positive_excess_qqq_days": positive_excess_qqq_days,
                "return_std": round(return_std, 6) if return_std is not None else None,
                "sharpe_like_ratio": round(sharpe_like_ratio, 6) if sharpe_like_ratio is not None else None,
                "max_daily_strategy_loss": round(min(avg_return_values), 6) if avg_return_values else None,
                "max_daily_strategy_gain": round(max(avg_return_values), 6) if avg_return_values else None,
                "data_status_counter": dict(Counter(str(row.get("data_status") or "UNKNOWN") for row in rows)),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            int(row.get("holding_days") or 0),
            -(_safe_float(row.get("avg_excess_return_vs_spy")) or -999),
            -(_safe_float(row.get("avg_excess_return_vs_qqq")) or -999),
            -(_safe_float(row.get("win_rate_vs_spy")) or -999),
            -(_safe_float(row.get("avg_return_pct")) or -999),
            str(row.get("strategy_name") or ""),
        ),
    )


def _most_common_value(values: list[object]) -> str | None:
    cleaned = [str(value) for value in values if str(value or "").strip()]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _build_data_quality_summary(summary_rows: list[dict[str, object]], result_rows: list[dict[str, object]]) -> dict[str, object]:
    result_status_counter = Counter(str(row.get("data_status") or "UNKNOWN") for row in result_rows)
    summary_status_counter = Counter(str(row.get("data_status") or "UNKNOWN") for row in summary_rows)
    total_result_rows = len(result_rows)
    missing_rows = total_result_rows - result_status_counter.get("OK", 0)
    holding_missing: dict[int, float] = {}
    strategy_missing: dict[str, float] = {}

    grouped_by_hd: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_by_strategy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result_rows:
        grouped_by_hd[int(row.get("holding_days") or 0)].append(row)
        grouped_by_strategy[str(row.get("strategy_name") or "")].append(row)

    for hd, rows in grouped_by_hd.items():
        if rows:
            holding_missing[hd] = sum(1 for row in rows if str(row.get("data_status")) != "OK") / len(rows)
    for strategy_name, rows in grouped_by_strategy.items():
        if rows:
            strategy_missing[strategy_name] = sum(1 for row in rows if str(row.get("data_status")) != "OK") / len(rows)

    return {
        "total_result_rows": total_result_rows,
        "ok_rows": result_status_counter.get("OK", 0),
        "not_enough_forward_data_rows": result_status_counter.get("NOT_ENOUGH_FORWARD_DATA", 0),
        "missing_entry_price_rows": result_status_counter.get("MISSING_ENTRY_PRICE", 0),
        "missing_exit_price_rows": result_status_counter.get("MISSING_EXIT_PRICE", 0),
        "missing_benchmark_rows": result_status_counter.get("PARTIAL_BENCHMARK_DATA", 0),
        "no_rank_data_days": summary_status_counter.get("NO_RANK_DATA", 0),
        "no_selection_days": summary_status_counter.get("NO_SELECTION", 0),
        "missing_rate": (missing_rows / total_result_rows) if total_result_rows else 0.0,
        "holding_days_missing_rate": holding_missing,
        "strategy_missing_rate": strategy_missing,
    }


def _select_best_candidate(aggregate_rows: list[dict[str, object]]) -> dict[str, object] | None:
    valid = [row for row in aggregate_rows if row.get("test_days", 0) and _safe_float(row.get("avg_excess_return_vs_spy")) is not None]
    return valid[0] if valid else None


def _build_interpretation_lines(
    aggregate_rows: list[dict[str, object]],
    *,
    quality: dict[str, object],
    cfg,
) -> list[str]:
    lines: list[str] = []
    best = _select_best_candidate(aggregate_rows)
    if best is None:
        lines.append("현재 집계 기준으로 비교 가능한 전략 성과가 충분하지 않습니다.")
        if quality.get("no_selection_days", 0):
            lines.append("선택 종목이 없는 날짜가 많아 추가 데이터 적재와 rank snapshot 누적이 필요합니다.")
        return lines

    if (_safe_float(best.get("avg_excess_return_vs_spy")) or 0.0) > 0 and (_safe_float(best.get("win_rate_vs_spy")) or 0.0) > 0.5:
        lines.append("SPY 대비 초과성과가 관찰됩니다.")
    else:
        lines.append("SPY 대비 초과성과는 아직 명확하지 않습니다.")

    if int(best.get("test_days") or 0) < cfg.min_test_days_warning:
        lines.append("테스트 일수가 부족하므로 해석에 주의가 필요합니다.")

    if float(quality.get("missing_rate") or 0.0) > cfg.missing_rate_warning:
        lines.append("데이터 누락률이 높아 결과 신뢰도가 낮을 수 있습니다.")

    same_hd = [row for row in aggregate_rows if int(row.get("holding_days") or 0) == int(best.get("holding_days") or 0)]
    top5 = next((row for row in same_hd if str(row.get("strategy_name")) == "US_RANK_TOP5"), None)
    top20 = next((row for row in same_hd if str(row.get("strategy_name")) == "US_RANK_TOP20"), None)
    if top5 and top20:
        top5_spy = _safe_float(top5.get("avg_excess_return_vs_spy"))
        top20_spy = _safe_float(top20.get("avg_excess_return_vs_spy"))
        if top5_spy is not None and top20_spy is not None:
            if top5_spy > top20_spy + 0.0025:
                lines.append("상위 랭킹 집중 전략이 더 유리한 경향이 관찰됩니다.")
            elif top20_spy > top5_spy + 0.0025:
                lines.append("분산된 후보군 전략이 더 안정적일 가능성이 있습니다.")

    lines.append("이 결과는 추가 검증 후보를 고르는 참고 자료이며 자동매매 판단 근거가 아닙니다.")
    return lines


def _fixed_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = "  ".join(str(column).ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = ["  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns) for row in rows]
    return "\n".join([header, divider, *body])


def build_console_report(
    *,
    backtest_id: str,
    summary_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    quality: dict[str, object],
    interpretation_lines: list[str],
) -> str:
    period_start = min((row.get("trade_date") for row in summary_rows if isinstance(row.get("trade_date"), date)), default=None)
    period_end = max((row.get("trade_date") for row in summary_rows if isinstance(row.get("trade_date"), date)), default=None)
    rendered_rows = []
    for row in aggregate_rows:
        rendered_rows.append(
            {
                "Strategy": row.get("strategy_name"),
                "HD": row.get("holding_days"),
                "Days": row.get("test_days"),
                "AvgRet": _fmt_pct(row.get("avg_return_pct")),
                "ExcessSPY": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                "ExcessQQQ": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                "WinRate": _fmt_pct(row.get("win_rate")),
                "WinSPY": _fmt_pct(row.get("win_rate_vs_spy")),
                "WinQQQ": _fmt_pct(row.get("win_rate_vs_qqq")),
            }
        )
    best = _select_best_candidate(aggregate_rows)
    lines = [
        "[US Stock Rank Backtest Report]",
        f"Backtest ID: {backtest_id}",
        f"Period: {_format_date(period_start)} ~ {_format_date(period_end)}",
        "",
        "[Strategy Summary]",
        "",
        _fixed_table(rendered_rows, ["Strategy", "HD", "Days", "AvgRet", "ExcessSPY", "ExcessQQQ", "WinRate", "WinSPY", "WinQQQ"]),
        "",
    ]
    if best:
        lines.extend(
            [
                "[Best Candidate]",
                f"Holding Days: {best.get('holding_days')}",
                f"Strategy: {best.get('strategy_name')}",
                f"Avg Return: {_fmt_pct(best.get('avg_return_pct'))}",
                f"Avg Excess vs SPY: {_fmt_pct(best.get('avg_excess_return_vs_spy'))}",
                f"Avg Excess vs QQQ: {_fmt_pct(best.get('avg_excess_return_vs_qqq'))}",
                f"Win Rate vs SPY: {_fmt_pct(best.get('win_rate_vs_spy'))}",
                "",
            ]
        )
    lines.extend(
        [
            "[Data Quality Summary]",
            f"Total Result Rows: {quality.get('total_result_rows', 0)}",
            f"OK: {quality.get('ok_rows', 0)}",
            f"NOT_ENOUGH_FORWARD_DATA: {quality.get('not_enough_forward_data_rows', 0)}",
            f"MISSING_ENTRY_PRICE: {quality.get('missing_entry_price_rows', 0)}",
            f"MISSING_EXIT_PRICE: {quality.get('missing_exit_price_rows', 0)}",
            f"MISSING_BENCHMARK: {quality.get('missing_benchmark_rows', 0)}",
            f"NO_RANK_DATA Days: {quality.get('no_rank_data_days', 0)}",
            f"Missing Rate: {_fmt_pct(quality.get('missing_rate'))}",
            "",
            "[Interpretation]",
            *[f"- {line}" for line in interpretation_lines],
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def _daily_extremes(summary_rows: list[dict[str, object]], *, strategy: str, holding_days: int, limit_n: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    filtered = [
        row for row in summary_rows
        if str(row.get("strategy_name")) == strategy and int(row.get("holding_days") or 0) == holding_days and _safe_float(row.get("avg_return_pct")) is not None
    ]
    filtered.sort(key=lambda row: row.get("trade_date") or date.min)
    best_days = sorted(filtered, key=lambda row: float(row["avg_return_pct"]), reverse=True)[:limit_n]
    worst_days = sorted(filtered, key=lambda row: float(row["avg_return_pct"]))[:limit_n]
    recent_days = filtered[-limit_n:]
    return best_days, worst_days, recent_days


def build_markdown_report(
    *,
    backtest_id: str,
    summary_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    quality: dict[str, object],
    interpretation_lines: list[str],
    cfg,
) -> str:
    period_start = min((row.get("trade_date") for row in summary_rows if isinstance(row.get("trade_date"), date)), default=None)
    period_end = max((row.get("trade_date") for row in summary_rows if isinstance(row.get("trade_date"), date)), default=None)
    holding_day_values = sorted({int(row.get("holding_days") or 0) for row in aggregate_rows})
    summary_table_rows = []
    for row in aggregate_rows:
        summary_table_rows.append(
            {
                "strategy": row.get("strategy_name"),
                "holding_days": row.get("holding_days"),
                "test_days": row.get("test_days"),
                "avg_return_pct": _fmt_pct(row.get("avg_return_pct")),
                "avg_excess_return_vs_spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                "avg_excess_return_vs_qqq": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                "win_rate": _fmt_pct(row.get("win_rate")),
                "win_rate_vs_spy": _fmt_pct(row.get("win_rate_vs_spy")),
                "win_rate_vs_qqq": _fmt_pct(row.get("win_rate_vs_qqq")),
            }
        )

    best = _select_best_candidate(aggregate_rows)
    lines = [
        "# 미국주식 랭킹 백테스트 리포트",
        "",
        "## 1. 개요",
        "",
        f"- Backtest ID: {backtest_id}",
        f"- 기간: {_format_date(period_start)} ~ {_format_date(period_end)}",
        f"- 대상 전략 수: {len({str(row.get('strategy_name')) for row in aggregate_rows})}",
        f"- Holding Days: {', '.join(str(value) for value in holding_day_values) or '-'}",
        "- Benchmark:",
        "- SPY",
        "- QQQ",
        "- Universe 평균",
        "",
        "## 2. 전략별 성과 요약",
        "",
        _markdown_table(
            summary_table_rows,
            [
                ("strategy", "Strategy"),
                ("holding_days", "Holding Days"),
                ("test_days", "Test Days"),
                ("avg_return_pct", "Avg Return"),
                ("avg_excess_return_vs_spy", "Excess vs SPY"),
                ("avg_excess_return_vs_qqq", "Excess vs QQQ"),
                ("win_rate", "Win Rate"),
                ("win_rate_vs_spy", "Win vs SPY"),
                ("win_rate_vs_qqq", "Win vs QQQ"),
            ],
        ),
        "",
        "## 3. Holding Days별 비교",
        "",
    ]
    for holding_days in sorted({int(row.get("holding_days") or 0) for row in aggregate_rows}):
        lines.append(f"### Holding Days {holding_days}")
        lines.append("")
        holding_rows = [row for row in aggregate_rows if int(row.get("holding_days") or 0) == holding_days]
        lines.append(
            _markdown_table(
                [
                    {
                        "strategy": row.get("strategy_name"),
                        "avg_return": _fmt_pct(row.get("avg_return_pct")),
                        "spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                        "qqq": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                        "win_spy": _fmt_pct(row.get("win_rate_vs_spy")),
                    }
                    for row in holding_rows
                ],
                [
                    ("strategy", "Strategy"),
                    ("avg_return", "Avg Return"),
                    ("spy", "Excess vs SPY"),
                    ("qqq", "Excess vs QQQ"),
                    ("win_spy", "Win vs SPY"),
                ],
            )
        )
        lines.append("")

    lines.extend(["## 4. Best Candidate", ""])
    if best:
        lines.extend(
            [
                f"- Strategy: {best.get('strategy_name')}",
                f"- Holding Days: {best.get('holding_days')}",
                f"- Avg Return: {_fmt_pct(best.get('avg_return_pct'))}",
                f"- Avg Excess vs SPY: {_fmt_pct(best.get('avg_excess_return_vs_spy'))}",
                f"- Avg Excess vs QQQ: {_fmt_pct(best.get('avg_excess_return_vs_qqq'))}",
                f"- Win Rate vs SPY: {_fmt_pct(best.get('win_rate_vs_spy'))}",
                "",
            ]
        )
    else:
        lines.extend(["- 비교 가능한 후보 없음", ""])

    lines.extend(["## 5. 전략별 상세", ""])
    for row in aggregate_rows:
        lines.extend(
            [
                f"### {row.get('strategy_name')} / {row.get('holding_days')}D",
                "",
                f"- test_days: {row.get('test_days')}",
                f"- selected_count_avg: {_fmt_num(row.get('selected_count_avg'))}",
                f"- avg_return_pct: {_fmt_pct(row.get('avg_return_pct'))}",
                f"- median_return_pct: {_fmt_pct(row.get('median_return_pct'))}",
                f"- avg_excess_return_vs_spy: {_fmt_pct(row.get('avg_excess_return_vs_spy'))}",
                f"- avg_excess_return_vs_qqq: {_fmt_pct(row.get('avg_excess_return_vs_qqq'))}",
                f"- avg_excess_return_vs_universe: {_fmt_pct(row.get('avg_excess_return_vs_universe'))}",
                f"- win_rate: {_fmt_pct(row.get('win_rate'))}",
                f"- win_rate_vs_spy: {_fmt_pct(row.get('win_rate_vs_spy'))}",
                f"- best_day: {_format_date(row.get('best_day'))}",
                f"- worst_day: {_format_date(row.get('worst_day'))}",
                f"- best_symbol: {row.get('best_symbol') or 'N/A'}",
                f"- worst_symbol: {row.get('worst_symbol') or 'N/A'}",
                "",
            ]
        )

    lines.extend(["## 6. 일자별 성과 추이", ""])
    if best:
        best_days, worst_days, recent_days = _daily_extremes(summary_rows, strategy=str(best.get("strategy_name")), holding_days=int(best.get("holding_days") or 0), limit_n=cfg.best_worst_limit)
        for title, rows in [("Best 10 Days", best_days), ("Worst 10 Days", worst_days), ("Recent 10 Days", recent_days)]:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(
                _markdown_table(
                    [
                        {
                            "trade_date": _format_date(row.get("trade_date")),
                            "avg_return_pct": _fmt_pct(row.get("avg_return_pct")),
                            "avg_excess_return_vs_spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                            "win_rate": _fmt_pct(row.get("win_rate")),
                            "data_status": row.get("data_status") or "N/A",
                        }
                        for row in rows
                    ],
                    [
                        ("trade_date", "Trade Date"),
                        ("avg_return_pct", "Avg Return"),
                        ("avg_excess_return_vs_spy", "Excess SPY"),
                        ("win_rate", "Win Rate"),
                        ("data_status", "Data Status"),
                    ],
                )
            )
            lines.append("")

    lines.extend(["## 7. Best/Worst 종목", ""])
    lines.append(
        _markdown_table(
            [
                {
                    "strategy": row.get("strategy_name"),
                    "holding_days": row.get("holding_days"),
                    "best_symbol": row.get("best_symbol") or "N/A",
                    "worst_symbol": row.get("worst_symbol") or "N/A",
                }
                for row in aggregate_rows
            ],
            [
                ("strategy", "Strategy"),
                ("holding_days", "Holding Days"),
                ("best_symbol", "Best Symbol"),
                ("worst_symbol", "Worst Symbol"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 8. 데이터 품질 및 누락 현황",
            "",
            f"- 전체 result rows: {quality.get('total_result_rows', 0)}",
            f"- 정상 계산 rows: {quality.get('ok_rows', 0)}",
            f"- NOT_ENOUGH_FORWARD_DATA rows: {quality.get('not_enough_forward_data_rows', 0)}",
            f"- MISSING_ENTRY_PRICE rows: {quality.get('missing_entry_price_rows', 0)}",
            f"- MISSING_EXIT_PRICE rows: {quality.get('missing_exit_price_rows', 0)}",
            f"- MISSING_BENCHMARK rows: {quality.get('missing_benchmark_rows', 0)}",
            f"- NO_RANK_DATA days: {quality.get('no_rank_data_days', 0)}",
            f"- missing rate: {_fmt_pct(quality.get('missing_rate'))}",
            "",
            "## 9. 해석 시 주의사항",
            "",
            "- 이 결과는 백테스트 결과입니다.",
            "- 실매매 성과를 보장하지 않습니다.",
            "- Phase 4 결과만으로 자동매매를 시작하지 않습니다.",
        ]
    )
    lines.extend([f"- {line}" for line in interpretation_lines])
    lines.append("")
    return "\n".join(lines)


def _resolve_output_path(output_dir: Path, *, backtest_id: str, kind: str, suffix: str) -> Path:
    name = {
        "markdown": f"report_{backtest_id}.{suffix}",
        "summary": f"backtest_summary_{backtest_id}.{suffix}",
        "daily": f"backtest_daily_{backtest_id}.{suffix}",
        "symbol": f"backtest_symbol_{backtest_id}.{suffix}",
    }[kind]
    return output_dir / name


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name) for name in fieldnames}
            writer.writerow(payload)


def build_symbol_console_report(*, backtest_id: str, symbol: str, result_rows: list[dict[str, object]]) -> str:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in result_rows:
        grouped[int(row.get("holding_days") or 0)].append(row)
    lines = [
        "[Symbol Backtest Detail]",
        f"Backtest ID: {backtest_id}",
        f"Symbol: {symbol}",
        "",
    ]
    table_rows = []
    for holding_days in sorted(grouped):
        rows = grouped[holding_days]
        table_rows.append(
            {
                "HD": holding_days,
                "Count": len(rows),
                "AvgRank": _fmt_num(_mean([_safe_float(row.get("rank_no")) for row in rows]), 1),
                "AvgScore": _fmt_num(_mean([_safe_float(row.get("total_score")) for row in rows]), 1),
                "AvgRet": _fmt_pct(_mean([_safe_float(row.get("return_pct")) for row in rows])),
                "ExcessSPY": _fmt_pct(_mean([_safe_float(row.get("excess_return_vs_spy")) for row in rows])),
                "WinRate": _fmt_pct(_rate([1 if (_safe_float(row.get("return_pct")) or 0.0) > 0 else 0 for row in rows if _safe_float(row.get("return_pct")) is not None])),
            }
        )
    lines.append(_fixed_table(table_rows, ["HD", "Count", "AvgRank", "AvgScore", "AvgRet", "ExcessSPY", "WinRate"]))
    return "\n".join(lines).strip() + "\n"


def build_strategy_detail_console_report(
    *,
    strategy: str,
    holding_days: int,
    summary_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    quality: dict[str, object],
    cfg,
) -> str:
    target = next((row for row in aggregate_rows if str(row.get("strategy_name")) == strategy and int(row.get("holding_days") or 0) == holding_days), None)
    best_days, worst_days, _ = _daily_extremes(summary_rows, strategy=strategy, holding_days=holding_days, limit_n=cfg.best_worst_limit)
    lines = [
        "[Strategy Backtest Detail]",
        f"Strategy: {strategy}",
        f"Holding Days: {holding_days}",
        "",
    ]
    if target:
        lines.extend(
            [
                f"Test Days: {target.get('test_days')}",
                f"Avg Return: {_fmt_pct(target.get('avg_return_pct'))}",
                f"Avg Excess vs SPY: {_fmt_pct(target.get('avg_excess_return_vs_spy'))}",
                f"Avg Excess vs QQQ: {_fmt_pct(target.get('avg_excess_return_vs_qqq'))}",
                f"Win Rate: {_fmt_pct(target.get('win_rate'))}",
                f"Best Symbol: {target.get('best_symbol') or 'N/A'}",
                f"Worst Symbol: {target.get('worst_symbol') or 'N/A'}",
                "",
            ]
        )
    lines.extend(["[Best Days]", ""])
    lines.append(
        _fixed_table(
            [
                {
                    "TradeDate": _format_date(row.get("trade_date")),
                    "AvgRet": _fmt_pct(row.get("avg_return_pct")),
                    "ExcessSPY": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "WinRate": _fmt_pct(row.get("win_rate")),
                    "DataStatus": row.get("data_status") or "N/A",
                }
                for row in best_days
            ],
            ["TradeDate", "AvgRet", "ExcessSPY", "WinRate", "DataStatus"],
        )
    )
    lines.extend(["", "[Worst Days]", ""])
    lines.append(
        _fixed_table(
            [
                {
                    "TradeDate": _format_date(row.get("trade_date")),
                    "AvgRet": _fmt_pct(row.get("avg_return_pct")),
                    "ExcessSPY": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "WinRate": _fmt_pct(row.get("win_rate")),
                    "DataStatus": row.get("data_status") or "N/A",
                }
                for row in worst_days
            ],
            ["TradeDate", "AvgRet", "ExcessSPY", "WinRate", "DataStatus"],
        )
    )
    lines.extend(
        [
            "",
            "[Data Quality Summary]",
            f"Missing Rate: {_fmt_pct(quality.get('missing_rate'))}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    cfg = load_us_backtest_report_config()
    setup_logging(cfg.log_level)
    start_date = parse_iso_date(args.start_date, field_name="start_date") if args.start_date else None
    end_date = parse_iso_date(args.end_date, field_name="end_date") if args.end_date else None
    strategy = str(args.strategy or "").strip() or None
    symbol = str(args.symbol or "").strip().upper() or None
    output_dir = _normalize_output_dir(args.output_dir)

    summary_rows = _query_summary_rows(
        backtest_id=args.backtest_id,
        strategy=strategy,
        holding_days=args.holding_days,
        start_date=start_date,
        end_date=end_date,
    )
    if not summary_rows:
        print(f"[US_BACKTEST_REPORT] No summary rows found for backtest_id={args.backtest_id}.")
        return 1

    result_rows = _query_result_rows(
        backtest_id=args.backtest_id,
        strategy=strategy,
        holding_days=args.holding_days,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )
    aggregate_rows = _aggregate_strategy_summary(summary_rows)
    quality = _build_data_quality_summary(summary_rows, result_rows)
    interpretation_lines = _build_interpretation_lines(aggregate_rows, quality=quality, cfg=cfg)

    if symbol:
        if not result_rows:
            print(f"[US_BACKTEST_REPORT] No symbol result rows found for backtest_id={args.backtest_id} symbol={symbol}.")
            return 1
        print(build_symbol_console_report(backtest_id=args.backtest_id, symbol=symbol, result_rows=result_rows), end="")
        return 0

    if args.format == "console":
        if strategy and args.holding_days is not None:
            print(
                build_strategy_detail_console_report(
                    strategy=strategy,
                    holding_days=args.holding_days,
                    summary_rows=summary_rows,
                    aggregate_rows=aggregate_rows,
                    quality=quality,
                    cfg=cfg,
                ),
                end="",
            )
        else:
            print(
                build_console_report(
                    backtest_id=args.backtest_id,
                    summary_rows=summary_rows,
                    aggregate_rows=aggregate_rows,
                    quality=quality,
                    interpretation_lines=interpretation_lines,
                ),
                end="",
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        rendered = build_markdown_report(
            backtest_id=args.backtest_id,
            summary_rows=summary_rows,
            aggregate_rows=aggregate_rows,
            result_rows=result_rows,
            quality=quality,
            interpretation_lines=interpretation_lines,
            cfg=cfg,
        )
        path = _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="markdown", suffix="md")
        path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    summary_csv_rows = aggregate_rows
    daily_csv_rows = [
        {
            "backtest_id": row.get("backtest_id"),
            "trade_date": row.get("trade_date"),
            "strategy_name": row.get("strategy_name"),
            "selection_rule": row.get("selection_rule"),
            "holding_days": row.get("holding_days"),
            "selected_count": row.get("selected_count"),
            "avg_return_pct": row.get("avg_return_pct"),
            "median_return_pct": row.get("median_return_pct"),
            "win_rate": row.get("win_rate"),
            "avg_excess_return_vs_spy": row.get("avg_excess_return_vs_spy"),
            "avg_excess_return_vs_qqq": row.get("avg_excess_return_vs_qqq"),
            "avg_excess_return_vs_universe": row.get("avg_excess_return_vs_universe"),
            "data_status": row.get("data_status"),
        }
        for row in summary_rows
    ]
    symbol_csv_rows = result_rows
    _write_csv(
        _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="summary", suffix="csv"),
        summary_csv_rows,
        [
            "backtest_id",
            "strategy_name",
            "selection_rule",
            "holding_days",
            "test_days",
            "selected_count_avg",
            "avg_return_pct",
            "median_return_pct",
            "win_rate",
            "avg_excess_return_vs_spy",
            "avg_excess_return_vs_qqq",
            "avg_excess_return_vs_universe",
            "win_rate_vs_spy",
            "win_rate_vs_qqq",
            "win_rate_vs_universe",
            "best_symbol",
            "best_return_pct",
            "worst_symbol",
            "worst_return_pct",
        ],
    )
    _write_csv(
        _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="daily", suffix="csv"),
        daily_csv_rows,
        [
            "backtest_id",
            "trade_date",
            "strategy_name",
            "selection_rule",
            "holding_days",
            "selected_count",
            "avg_return_pct",
            "median_return_pct",
            "win_rate",
            "avg_excess_return_vs_spy",
            "avg_excess_return_vs_qqq",
            "avg_excess_return_vs_universe",
            "data_status",
        ],
    )
    _write_csv(
        _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="symbol", suffix="csv"),
        symbol_csv_rows,
        [
            "backtest_id",
            "trade_date",
            "symbol",
            "strategy_name",
            "selection_rule",
            "holding_days",
            "rank_no",
            "recommend_grade",
            "total_score",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "return_pct",
            "excess_return_vs_spy",
            "excess_return_vs_qqq",
            "excess_return_vs_universe",
            "data_status",
        ],
    )
    print(f"summary_csv: {_resolve_output_path(output_dir, backtest_id=args.backtest_id, kind='summary', suffix='csv')}")
    print(f"daily_csv: {_resolve_output_path(output_dir, backtest_id=args.backtest_id, kind='daily', suffix='csv')}")
    print(f"symbol_csv: {_resolve_output_path(output_dir, backtest_id=args.backtest_id, kind='symbol', suffix='csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
