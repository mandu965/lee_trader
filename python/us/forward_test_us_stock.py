from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.backtest_us_stock_rank_strategy import (
    BENCHMARKS,
    StrategySpec,
    _build_price_lookup,
    _compute_return,
    _parse_int_csv,
    resolve_strategy_specs,
    select_strategy_rows,
)
from python.us.us_config import load_us_forward_test_config, load_us_rule_ranking_config, parse_iso_date
from python.us.us_db import (
    ensure_us_forward_test_tables,
    fetch_market_regime_rows_between,
    fetch_mixed_price_rows_for_tickers_between,
    fetch_rank_component_rows_between,
    fetch_us_forward_test_rows,
    fetch_us_forward_test_summary_rows,
    upsert_us_forward_test_rows,
    upsert_us_forward_test_summary_rows,
)


SUPPORTED_FORMATS = {"console", "markdown", "csv"}


def setup_logging(level_name: str) -> None:
    import logging

    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return float(statistics.fmean(nums))


def _median(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return float(statistics.median(nums))


def _rate(values: list[int | None]) -> float | None:
    nums = [int(value) for value in values if value is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _fmt_pct(value: object) -> str:
    num = _safe_float(value)
    if num is None:
        return "N/A"
    return f"{num * 100:.2f}%"


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _normalize_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _normalize_strategy_aliases(raw: str | None) -> tuple[str, ...]:
    parts = [part.strip().upper() for part in str(raw or "").split(",")]
    return tuple(part for part in parts if part)


def _resolve_forward_strategy_specs(raw: str | None) -> list[StrategySpec]:
    wanted = set(_normalize_strategy_aliases(raw))
    specs = resolve_strategy_specs(custom_top_n=20, strategy_filter=None)
    if not wanted:
        return specs
    selected: list[StrategySpec] = []
    for spec in specs:
        alias = spec.strategy_name.removeprefix("US_RANK_")
        if alias in wanted or spec.strategy_name in wanted:
            selected.append(spec)
    return selected


def _build_market_calendar(price_lookup: dict[str, list[dict[str, object]]]) -> list[date]:
    calendar_dates: set[date] = set()
    for ticker in BENCHMARKS:
        for row in price_lookup.get(ticker, []):
            trade_date = row.get("trade_date")
            if isinstance(trade_date, date):
                calendar_dates.add(trade_date)
    if not calendar_dates:
        for rows in price_lookup.values():
            for row in rows:
                trade_date = row.get("trade_date")
                if isinstance(trade_date, date):
                    calendar_dates.add(trade_date)
    return sorted(calendar_dates)


def next_trade_date(trade_date: date, market_calendar: list[date]) -> date | None:
    for candidate in market_calendar:
        if candidate > trade_date:
            return candidate
    return None


def target_exit_date(entry_date: date | None, holding_days: int, market_calendar: list[date]) -> date | None:
    if entry_date is None:
        return None
    try:
        entry_index = market_calendar.index(entry_date)
    except ValueError:
        return None
    exit_index = entry_index + holding_days
    if exit_index >= len(market_calendar):
        return None
    return market_calendar[exit_index]


def _find_price_for_date(price_lookup: dict[str, list[dict[str, object]]], ticker: str, target_date: date | None) -> float | None:
    if target_date is None:
        return None
    for row in price_lookup.get(ticker.upper(), []):
        if row.get("trade_date") == target_date:
            return _safe_float(row.get("price"))
    return None


def _load_rank_rows_for_trade_date(*, trade_date: date, source: str) -> list[dict[str, object]]:
    return fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source=source)


def _load_regime_map(*, start_date: date, end_date: date) -> dict[date, dict[str, object]]:
    rows = fetch_market_regime_rows_between(start_date=start_date, end_date=end_date)
    return {row["trade_date"]: row for row in rows if isinstance(row.get("trade_date"), date)}


def build_forward_test_rows(
    *,
    forward_test_id: str,
    trade_date: date,
    holding_days: list[int],
    strategies: list[StrategySpec],
    source: str,
    weight_config_id: str = "RULE_V1_BASELINE",
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rank_rows = _load_rank_rows_for_trade_date(trade_date=trade_date, source=source)
    if not rank_rows:
        return [], {"selected_rows": 0, "strategy_rows": 0, "rank_rows": 0}

    symbols = sorted({str(row.get("symbol") or "").upper() for row in rank_rows if str(row.get("symbol") or "").strip()} | set(BENCHMARKS))
    price_rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=symbols,
        start_date=trade_date - timedelta(days=5),
        end_date=trade_date + timedelta(days=max(holding_days) * 4 + 45),
    )
    price_lookup = _build_price_lookup(price_rows)
    market_calendar = _build_market_calendar(price_lookup)
    regime_map = _load_regime_map(start_date=trade_date, end_date=trade_date)
    regime = regime_map.get(trade_date, {})

    output: list[dict[str, object]] = []
    strategy_rows = 0
    for spec in strategies:
        selected_rows = select_strategy_rows(rank_rows, spec)
        strategy_rows += len(selected_rows)
        for row in selected_rows:
            entry_dt = next_trade_date(trade_date, market_calendar)
            for hd in holding_days:
                output.append(
                    {
                        "forward_test_id": forward_test_id,
                        "trade_date": trade_date,
                        "symbol": str(row.get("symbol") or "").upper(),
                        "holding_days": hd,
                        "strategy_name": spec.strategy_name,
                        "selection_rule": spec.selection_rule,
                        "rank_no": row.get("rank_no"),
                        "recommend_grade": row.get("recommend_grade"),
                        "total_score": row.get("total_score"),
                        "company_name": row.get("company_name"),
                        "sector": row.get("sector"),
                        "industry": row.get("industry"),
                        "weight_config_id": weight_config_id,
                        "source": source,
                        "entry_date": entry_dt,
                        "entry_price": None,
                        "target_exit_date": None,
                        "exit_date": None,
                        "exit_price": None,
                        "return_pct": None,
                        "spy_entry_price": None,
                        "spy_exit_price": None,
                        "spy_return_pct": None,
                        "qqq_entry_price": None,
                        "qqq_exit_price": None,
                        "qqq_return_pct": None,
                        "excess_return_vs_spy": None,
                        "excess_return_vs_qqq": None,
                        "win_flag": None,
                        "win_vs_spy_flag": None,
                        "win_vs_qqq_flag": None,
                        "market_regime": regime.get("market_regime"),
                        "spy_regime": regime.get("spy_regime"),
                        "qqq_regime": regime.get("qqq_regime"),
                        "vol_regime": regime.get("vol_regime"),
                        "status": "PENDING_ENTRY",
                        "data_status": "OK" if entry_dt is not None else "MISSING_ENTRY_PRICE",
                        "exclude_reason": row.get("exclude_reason"),
                    }
                )
    return output, {"selected_rows": len(output), "strategy_rows": strategy_rows, "rank_rows": len(rank_rows)}


def apply_entry_updates(rows: list[dict[str, object]], *, as_of_date: date, price_lookup: dict[str, list[dict[str, object]]], market_calendar: list[date]) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    for row in rows:
        status = str(row.get("status") or "")
        entry_date = row.get("entry_date")
        if status != "PENDING_ENTRY" or not isinstance(entry_date, date) or entry_date > as_of_date:
            updated.append(dict(row))
            continue

        next_row = dict(row)
        symbol = str(row.get("symbol") or "").upper()
        entry_price = _find_price_for_date(price_lookup, symbol, entry_date)
        if entry_price is None:
            next_row["data_status"] = "MISSING_ENTRY_PRICE"
            next_row["status"] = "PENDING_ENTRY"
            updated.append(next_row)
            continue
        if entry_price <= 0:
            next_row["data_status"] = "INVALID_PRICE"
            next_row["status"] = "ERROR"
            updated.append(next_row)
            continue

        spy_entry = _find_price_for_date(price_lookup, "SPY", entry_date)
        qqq_entry = _find_price_for_date(price_lookup, "QQQ", entry_date)
        exit_target = target_exit_date(entry_date, int(row.get("holding_days") or 0), market_calendar)

        next_row["entry_price"] = round(entry_price, 6)
        next_row["spy_entry_price"] = round(spy_entry, 6) if spy_entry is not None else None
        next_row["qqq_entry_price"] = round(qqq_entry, 6) if qqq_entry is not None else None
        next_row["target_exit_date"] = exit_target
        next_row["status"] = "ACTIVE"
        next_row["data_status"] = "MISSING_BENCHMARK" if spy_entry is None or qqq_entry is None else "OK"
        updated.append(next_row)
    return updated


def apply_exit_updates(rows: list[dict[str, object]], *, as_of_date: date, price_lookup: dict[str, list[dict[str, object]]], market_calendar: list[date]) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    for row in rows:
        status = str(row.get("status") or "")
        if status not in {"ACTIVE", "PENDING_EXIT"}:
            updated.append(dict(row))
            continue

        next_row = dict(row)
        entry_date = row.get("entry_date")
        if not isinstance(entry_date, date):
            next_row["status"] = "PENDING_ENTRY"
            next_row["data_status"] = "MISSING_ENTRY_PRICE"
            updated.append(next_row)
            continue

        entry_price = _safe_float(row.get("entry_price"))
        if entry_price is None or entry_price <= 0:
            next_row["status"] = "ERROR"
            next_row["data_status"] = "INVALID_PRICE"
            updated.append(next_row)
            continue

        exit_target = row.get("target_exit_date")
        if not isinstance(exit_target, date):
            exit_target = target_exit_date(entry_date, int(row.get("holding_days") or 0), market_calendar)
            next_row["target_exit_date"] = exit_target
        if not isinstance(exit_target, date):
            next_row["status"] = "ACTIVE"
            next_row["data_status"] = "NOT_ENOUGH_FORWARD_DATA"
            updated.append(next_row)
            continue
        if exit_target > as_of_date:
            next_row["status"] = "ACTIVE"
            updated.append(next_row)
            continue

        symbol = str(row.get("symbol") or "").upper()
        exit_price = _find_price_for_date(price_lookup, symbol, exit_target)
        if exit_price is None:
            next_row["status"] = "PENDING_EXIT"
            next_row["data_status"] = "MISSING_EXIT_PRICE"
            updated.append(next_row)
            continue
        if exit_price <= 0:
            next_row["status"] = "ERROR"
            next_row["data_status"] = "INVALID_PRICE"
            updated.append(next_row)
            continue

        spy_entry = _safe_float(row.get("spy_entry_price"))
        qqq_entry = _safe_float(row.get("qqq_entry_price"))
        spy_exit = _find_price_for_date(price_lookup, "SPY", exit_target)
        qqq_exit = _find_price_for_date(price_lookup, "QQQ", exit_target)

        return_pct = _compute_return(entry_price, exit_price)
        spy_return = _compute_return(spy_entry, spy_exit)
        qqq_return = _compute_return(qqq_entry, qqq_exit)
        excess_spy = None if return_pct is None or spy_return is None else float(return_pct - spy_return)
        excess_qqq = None if return_pct is None or qqq_return is None else float(return_pct - qqq_return)

        next_row["exit_date"] = exit_target
        next_row["exit_price"] = round(exit_price, 6)
        next_row["return_pct"] = round(return_pct, 6) if return_pct is not None else None
        next_row["spy_exit_price"] = round(spy_exit, 6) if spy_exit is not None else None
        next_row["qqq_exit_price"] = round(qqq_exit, 6) if qqq_exit is not None else None
        next_row["spy_return_pct"] = round(spy_return, 6) if spy_return is not None else None
        next_row["qqq_return_pct"] = round(qqq_return, 6) if qqq_return is not None else None
        next_row["excess_return_vs_spy"] = round(excess_spy, 6) if excess_spy is not None else None
        next_row["excess_return_vs_qqq"] = round(excess_qqq, 6) if excess_qqq is not None else None
        next_row["win_flag"] = 1 if return_pct is not None and return_pct > 0 else 0 if return_pct is not None else None
        next_row["win_vs_spy_flag"] = 1 if excess_spy is not None and excess_spy > 0 else 0 if excess_spy is not None else None
        next_row["win_vs_qqq_flag"] = 1 if excess_qqq is not None and excess_qqq > 0 else 0 if excess_qqq is not None else None
        next_row["status"] = "COMPLETED"
        next_row["data_status"] = "MISSING_BENCHMARK" if spy_return is None or qqq_return is None else "OK"
        updated.append(next_row)
    return updated


def build_forward_summary_rows(rows: list[dict[str, object]], *, forward_test_id: str) -> list[dict[str, object]]:
    grouped: dict[tuple[date, str, int], list[dict[str, object]]] = {}
    for row in rows:
        trade_date = row.get("trade_date")
        strategy_name = str(row.get("strategy_name") or "")
        holding_days = int(row.get("holding_days") or 0)
        if isinstance(trade_date, date) and strategy_name and holding_days > 0:
            grouped.setdefault((trade_date, strategy_name, holding_days), []).append(row)

    output: list[dict[str, object]] = []
    for (trade_date, strategy_name, holding_days), group_rows in sorted(grouped.items()):
        completed = [row for row in group_rows if str(row.get("status") or "") == "COMPLETED" and _safe_float(row.get("return_pct")) is not None]
        active_count = sum(1 for row in group_rows if str(row.get("status") or "") == "ACTIVE")
        pending_count = sum(1 for row in group_rows if str(row.get("status") or "") in {"PENDING_ENTRY", "PENDING_EXIT"})
        error_count = sum(1 for row in group_rows if str(row.get("status") or "") in {"ERROR", "SKIPPED"})
        best_row = max(completed, key=lambda item: float(item["return_pct"])) if completed else None
        worst_row = min(completed, key=lambda item: float(item["return_pct"])) if completed else None
        if completed and len(completed) == len(group_rows):
            status = "COMPLETED"
        elif completed:
            status = "PARTIAL"
        elif active_count or pending_count:
            status = "ACTIVE"
        elif error_count == len(group_rows):
            status = "ERROR"
        else:
            status = "PENDING"
        data_status = "OK" if completed else "NO_COMPLETED_RETURNS"
        output.append(
            {
                "forward_test_id": forward_test_id,
                "trade_date": trade_date,
                "strategy_name": strategy_name,
                "holding_days": holding_days,
                "selected_count": len(group_rows),
                "completed_count": len(completed),
                "active_count": active_count,
                "pending_count": pending_count,
                "error_count": error_count,
                "avg_return_pct": round(_mean([_safe_float(row.get("return_pct")) for row in completed]), 6) if completed and _mean([_safe_float(row.get("return_pct")) for row in completed]) is not None else None,
                "median_return_pct": round(_median([_safe_float(row.get("return_pct")) for row in completed]), 6) if completed and _median([_safe_float(row.get("return_pct")) for row in completed]) is not None else None,
                "win_rate": round(_rate([row.get("win_flag") for row in completed]), 6) if completed and _rate([row.get("win_flag") for row in completed]) is not None else None,
                "avg_spy_return_pct": round(_mean([_safe_float(row.get("spy_return_pct")) for row in completed]), 6) if completed and _mean([_safe_float(row.get("spy_return_pct")) for row in completed]) is not None else None,
                "avg_qqq_return_pct": round(_mean([_safe_float(row.get("qqq_return_pct")) for row in completed]), 6) if completed and _mean([_safe_float(row.get("qqq_return_pct")) for row in completed]) is not None else None,
                "avg_excess_return_vs_spy": round(_mean([_safe_float(row.get("excess_return_vs_spy")) for row in completed]), 6) if completed and _mean([_safe_float(row.get("excess_return_vs_spy")) for row in completed]) is not None else None,
                "avg_excess_return_vs_qqq": round(_mean([_safe_float(row.get("excess_return_vs_qqq")) for row in completed]), 6) if completed and _mean([_safe_float(row.get("excess_return_vs_qqq")) for row in completed]) is not None else None,
                "win_rate_vs_spy": round(_rate([row.get("win_vs_spy_flag") for row in completed]), 6) if completed and _rate([row.get("win_vs_spy_flag") for row in completed]) is not None else None,
                "win_rate_vs_qqq": round(_rate([row.get("win_vs_qqq_flag") for row in completed]), 6) if completed and _rate([row.get("win_vs_qqq_flag") for row in completed]) is not None else None,
                "best_symbol": best_row.get("symbol") if best_row else None,
                "best_return_pct": round(float(best_row["return_pct"]), 6) if best_row else None,
                "worst_symbol": worst_row.get("symbol") if worst_row else None,
                "worst_return_pct": round(float(worst_row["return_pct"]), 6) if worst_row else None,
                "status": status,
                "data_status": data_status,
            }
        )
    return output


def _query_forward_rows(
    *,
    forward_test_id: str,
    trade_date: date | None = None,
    status: str | None = None,
    strategy_name: str | None = None,
    holding_days: int | None = None,
) -> list[dict[str, object]]:
    return fetch_us_forward_test_rows(
        forward_test_id=forward_test_id,
        trade_date=trade_date,
        strategy_name=strategy_name,
        holding_days=holding_days,
        status=status,
    )


def _query_forward_summary_rows(
    *,
    forward_test_id: str,
    trade_date: date | None = None,
    strategy_name: str | None = None,
    holding_days: int | None = None,
) -> list[dict[str, object]]:
    return fetch_us_forward_test_summary_rows(
        forward_test_id=forward_test_id,
        trade_date=trade_date,
        strategy_name=strategy_name,
        holding_days=holding_days,
    )


def build_console_report(*, forward_test_id: str, detail_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> str:
    lines = [
        "[US Stock Forward Test Report]",
        f"Forward Test ID: {forward_test_id}",
        "",
        "[Progress Summary]",
        f"Total Rows: {len(detail_rows)}",
        f"Completed: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'COMPLETED')}",
        f"Active: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'ACTIVE')}",
        f"Pending Entry: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'PENDING_ENTRY')}",
        f"Pending Exit: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'PENDING_EXIT')}",
        f"Error: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'ERROR')}",
        "",
        "[Performance Summary - Completed Only]",
        "",
        "Strategy              HD   Trades  AvgRet   ExcessSPY  ExcessQQQ  WinRate  WinSPY  WinQQQ",
        "--------------------  ---  ------  -------  ---------  ---------  -------  ------  ------",
    ]
    for row in sorted(summary_rows, key=lambda item: (str(item.get("strategy_name") or ""), int(item.get("holding_days") or 0), item.get("trade_date"))):
        if not int(row.get("completed_count") or 0):
            continue
        lines.append(
            f"{str(row.get('strategy_name') or '')[:20]:20}  "
            f"{int(row.get('holding_days') or 0):>3}  "
            f"{int(row.get('completed_count') or 0):>6}  "
            f"{_fmt_pct(row.get('avg_return_pct')):>7}  "
            f"{_fmt_pct(row.get('avg_excess_return_vs_spy')):>9}  "
            f"{_fmt_pct(row.get('avg_excess_return_vs_qqq')):>9}  "
            f"{_fmt_pct(row.get('win_rate')):>7}  "
            f"{_fmt_pct(row.get('win_rate_vs_spy')):>6}  "
            f"{_fmt_pct(row.get('win_rate_vs_qqq')):>6}"
        )
    if all(not int(row.get("completed_count") or 0) for row in summary_rows):
        lines.append("No completed forward-test rows yet.")
    return "\n".join(lines)


def build_markdown_report(*, forward_test_id: str, detail_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> str:
    recent_rows = sorted(detail_rows, key=lambda row: (row.get("trade_date") or date.min, str(row.get("strategy_name") or ""), int(row.get("holding_days") or 0)), reverse=True)[:20]
    completed_rows = [row for row in detail_rows if str(row.get("status") or "") == "COMPLETED"]
    lines = [
        "# 미국주식 Forward Test 리포트",
        "",
        "## 1. 개요",
        "",
        f"- Forward Test ID: {forward_test_id}",
        f"- 등록 시작일: {_format_date(min((row.get('trade_date') for row in detail_rows if isinstance(row.get('trade_date'), date)), default=None))}",
        f"- 완료 건수: {len(completed_rows)}",
        f"- 진행 중 건수: {sum(1 for row in detail_rows if str(row.get('status') or '') in {'ACTIVE', 'PENDING_ENTRY', 'PENDING_EXIT'})}",
        "",
        "## 2. 진행 상황 요약",
        "",
        f"- Total Rows: {len(detail_rows)}",
        f"- Completed: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'COMPLETED')}",
        f"- Active: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'ACTIVE')}",
        f"- Pending Entry: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'PENDING_ENTRY')}",
        f"- Pending Exit: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'PENDING_EXIT')}",
        f"- Error: {sum(1 for row in detail_rows if str(row.get('status') or '') == 'ERROR')}",
        "",
        "## 3. 전략별 완료 성과",
        "",
        "| Strategy | Trade Date | HD | Completed | Avg Return | Excess vs SPY | Excess vs QQQ | Win Rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: (item.get("trade_date") or date.min, str(item.get("strategy_name") or ""), int(item.get("holding_days") or 0))):
        lines.append(
            f"| {row.get('strategy_name') or ''} | {_format_date(row.get('trade_date'))} | {int(row.get('holding_days') or 0)} | {int(row.get('completed_count') or 0)} | {_fmt_pct(row.get('avg_return_pct'))} | {_fmt_pct(row.get('avg_excess_return_vs_spy'))} | {_fmt_pct(row.get('avg_excess_return_vs_qqq'))} | {_fmt_pct(row.get('win_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 최근 등록 추천 목록",
            "",
            "| Trade Date | Strategy | Symbol | HD | Status | Rank | Grade | Total Score | Market Regime |",
            "|---|---|---|---:|---|---:|---|---:|---|",
        ]
    )
    for row in recent_rows:
        lines.append(
            f"| {_format_date(row.get('trade_date'))} | {row.get('strategy_name') or ''} | {row.get('symbol') or ''} | {int(row.get('holding_days') or 0)} | {row.get('status') or ''} | {row.get('rank_no') or ''} | {row.get('recommend_grade') or ''} | {row.get('total_score') or ''} | {row.get('market_regime') or ''} |"
        )
    lines.extend(
        [
            "",
            "## 5. 해석 시 주의사항",
            "",
            "- Forward Test는 실매매가 아닙니다.",
            "- Forward Test는 Paper Trading도 아닙니다.",
            "- 성과가 좋더라도 Paper Trading 전 별도 검증이 필요합니다.",
        ]
    )
    return "\n".join(lines)


def write_csv_outputs(*, output_dir: Path, forward_test_id: str, detail_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"forward_summary_{forward_test_id}.csv"
    detail_path = output_dir / f"forward_detail_{forward_test_id}.csv"
    summary_fields = [
        "forward_test_id", "trade_date", "strategy_name", "holding_days", "selected_count", "completed_count",
        "active_count", "pending_count", "error_count", "avg_return_pct", "median_return_pct", "win_rate",
        "avg_spy_return_pct", "avg_qqq_return_pct", "avg_excess_return_vs_spy", "avg_excess_return_vs_qqq",
        "win_rate_vs_spy", "win_rate_vs_qqq", "best_symbol", "best_return_pct", "worst_symbol", "worst_return_pct",
        "status", "data_status",
    ]
    detail_fields = [
        "forward_test_id", "trade_date", "symbol", "holding_days", "strategy_name", "selection_rule", "rank_no",
        "recommend_grade", "total_score", "company_name", "sector", "industry", "weight_config_id", "source",
        "entry_date", "entry_price", "target_exit_date", "exit_date", "exit_price", "return_pct",
        "spy_entry_price", "spy_exit_price", "spy_return_pct", "qqq_entry_price", "qqq_exit_price", "qqq_return_pct",
        "excess_return_vs_spy", "excess_return_vs_qqq", "win_flag", "win_vs_spy_flag", "win_vs_qqq_flag",
        "market_regime", "spy_regime", "qqq_regime", "vol_regime", "status", "data_status", "exclude_reason",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with detail_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    return [summary_path, detail_path]


def parse_register_args() -> argparse.Namespace:
    cfg = load_us_forward_test_config()
    parser = argparse.ArgumentParser(description="Register US stock rank rows into forward test tracking.")
    parser.add_argument("--trade-date", required=True, help="Ranking trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--forward-test-id", default=cfg.forward_test_id)
    parser.add_argument("--strategies", default=",".join(cfg.strategies))
    parser.add_argument("--holding-days", default=",".join(str(value) for value in cfg.holding_days))
    parser.add_argument("--source", default=load_us_rule_ranking_config().source)
    parser.add_argument("--weight-config-id", default="RULE_V1_BASELINE")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_entry_args() -> argparse.Namespace:
    cfg = load_us_forward_test_config()
    parser = argparse.ArgumentParser(description="Update forward-test entry prices.")
    parser.add_argument("--as-of-date", required=True, help="Evaluation date. Format: YYYY-MM-DD.")
    parser.add_argument("--forward-test-id", default=cfg.forward_test_id)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_exit_args() -> argparse.Namespace:
    cfg = load_us_forward_test_config()
    parser = argparse.ArgumentParser(description="Update forward-test exit prices and returns.")
    parser.add_argument("--as-of-date", required=True, help="Evaluation date. Format: YYYY-MM-DD.")
    parser.add_argument("--forward-test-id", default=cfg.forward_test_id)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_summary_args() -> argparse.Namespace:
    cfg = load_us_forward_test_config()
    parser = argparse.ArgumentParser(description="Update forward-test summary rows.")
    parser.add_argument("--forward-test-id", default=cfg.forward_test_id)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_report_args() -> argparse.Namespace:
    cfg = load_us_forward_test_config()
    parser = argparse.ArgumentParser(description="Report US stock forward-test progress and performance.")
    parser.add_argument("--forward-test-id", default=cfg.forward_test_id)
    parser.add_argument("--format", default="console", choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--holding-days", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    return parser.parse_args()


def main_register() -> int:
    cfg = load_us_forward_test_config()
    setup_logging(cfg.log_level)
    args = parse_register_args()
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    strategies = _resolve_forward_strategy_specs(args.strategies)
    holding_days = _parse_int_csv(args.holding_days)
    rows, counts = build_forward_test_rows(
        forward_test_id=args.forward_test_id,
        trade_date=trade_date,
        holding_days=holding_days,
        strategies=strategies,
        source=args.source,
        weight_config_id=args.weight_config_id,
    )
    print(f"[FORWARD_REGISTER] trade_date={trade_date.isoformat()} rank_rows={counts['rank_rows']} rows={len(rows)} selected={counts['strategy_rows']}")
    if counts["rank_rows"] == 0:
        print("[FORWARD_REGISTER] No ranking snapshot found for the requested trade_date.")
    elif counts["strategy_rows"] == 0:
        print("[FORWARD_REGISTER] Ranking snapshot exists but no rows matched the selected strategies. Check EXCLUDE grades or strategy filters.")
    if not args.dry_run:
        ensure_us_forward_test_tables()
        upsert_us_forward_test_rows(rows)
    return 0


def _load_forward_price_context(rows: list[dict[str, object]], as_of_date: date) -> tuple[dict[str, list[dict[str, object]]], list[date]]:
    symbols = sorted({str(row.get("symbol") or "").upper() for row in rows if str(row.get("symbol") or "").strip()} | set(BENCHMARKS))
    min_trade_date = min((row.get("trade_date") for row in rows if isinstance(row.get("trade_date"), date)), default=as_of_date)
    price_rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=symbols,
        start_date=min_trade_date - timedelta(days=5),
        end_date=as_of_date + timedelta(days=5),
    )
    price_lookup = _build_price_lookup(price_rows)
    return price_lookup, _build_market_calendar(price_lookup)


def main_update_entry() -> int:
    cfg = load_us_forward_test_config()
    setup_logging(cfg.log_level)
    args = parse_entry_args()
    as_of_date = parse_iso_date(args.as_of_date, field_name="as_of_date")
    rows = _query_forward_rows(forward_test_id=args.forward_test_id, status="PENDING_ENTRY")
    if not rows:
        print(f"[FORWARD_ENTRY] no pending-entry rows for {args.forward_test_id}")
        return 0
    price_lookup, market_calendar = _load_forward_price_context(rows, as_of_date)
    updated_rows = apply_entry_updates(rows, as_of_date=as_of_date, price_lookup=price_lookup, market_calendar=market_calendar)
    changed = sum(1 for old, new in zip(rows, updated_rows) if old != new)
    print(f"[FORWARD_ENTRY] as_of_date={as_of_date.isoformat()} rows={len(rows)} changed={changed}")
    if not args.dry_run:
        ensure_us_forward_test_tables()
        upsert_us_forward_test_rows(updated_rows)
    return 0


def main_update_exit() -> int:
    cfg = load_us_forward_test_config()
    setup_logging(cfg.log_level)
    args = parse_exit_args()
    as_of_date = parse_iso_date(args.as_of_date, field_name="as_of_date")
    rows = _query_forward_rows(forward_test_id=args.forward_test_id)
    rows = [row for row in rows if str(row.get("status") or "") in {"ACTIVE", "PENDING_EXIT"}]
    if not rows:
        print(f"[FORWARD_EXIT] no active/pending-exit rows for {args.forward_test_id}")
        return 0
    price_lookup, market_calendar = _load_forward_price_context(rows, as_of_date)
    updated_rows = apply_exit_updates(rows, as_of_date=as_of_date, price_lookup=price_lookup, market_calendar=market_calendar)
    changed = sum(1 for old, new in zip(rows, updated_rows) if old != new)
    print(f"[FORWARD_EXIT] as_of_date={as_of_date.isoformat()} rows={len(rows)} changed={changed}")
    if not args.dry_run:
        ensure_us_forward_test_tables()
        upsert_us_forward_test_rows(updated_rows)
    return 0


def main_update_summary() -> int:
    cfg = load_us_forward_test_config()
    setup_logging(cfg.log_level)
    args = parse_summary_args()
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date") if args.trade_date else None
    rows = _query_forward_rows(forward_test_id=args.forward_test_id, trade_date=trade_date)
    summary_rows = build_forward_summary_rows(rows, forward_test_id=args.forward_test_id)
    print(f"[FORWARD_SUMMARY] forward_test_id={args.forward_test_id} summary_rows={len(summary_rows)}")
    if not args.dry_run:
        ensure_us_forward_test_tables()
        upsert_us_forward_test_summary_rows(summary_rows)
    return 0


def main_report() -> int:
    cfg = load_us_forward_test_config()
    setup_logging(cfg.log_level)
    args = parse_report_args()
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date") if args.trade_date else None
    detail_rows = _query_forward_rows(
        forward_test_id=args.forward_test_id,
        trade_date=trade_date,
        status=args.status,
        strategy_name=args.strategy,
        holding_days=args.holding_days,
    )
    summary_rows = _query_forward_summary_rows(
        forward_test_id=args.forward_test_id,
        trade_date=trade_date,
        strategy_name=args.strategy,
        holding_days=args.holding_days,
    )
    if not detail_rows and not summary_rows:
        print(f"[FORWARD_REPORT] no forward-test rows found for {args.forward_test_id}")
        return 0
    output_dir = _normalize_output_dir(args.output_dir)
    if args.format == "console":
        print(build_console_report(forward_test_id=args.forward_test_id, detail_rows=detail_rows, summary_rows=summary_rows))
        return 0
    if args.format == "markdown":
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"forward_report_{args.forward_test_id}.md"
        path.write_text(build_markdown_report(forward_test_id=args.forward_test_id, detail_rows=detail_rows, summary_rows=summary_rows), encoding="utf-8")
        print(str(path))
        return 0
    paths = write_csv_outputs(output_dir=output_dir, forward_test_id=args.forward_test_id, detail_rows=detail_rows, summary_rows=summary_rows)
    for path in paths:
        print(str(path))
    return 0
