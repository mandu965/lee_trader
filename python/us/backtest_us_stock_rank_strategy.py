from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
import re
import statistics
import sys
from typing import Callable

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_db import (
    ensure_us_rank_backtest_tables,
    fetch_price_rows_for_tickers_between,
    fetch_rank_rows_between,
    get_us_engine,
    upsert_us_rank_backtest_result_rows,
    upsert_us_rank_backtest_summary_rows,
)
from python.us.us_config import load_us_rule_ranking_config, parse_iso_date


LOGGER = logging.getLogger("us_rank_backtest")
BENCHMARKS = ("SPY", "QQQ")


@dataclass(frozen=True)
class StrategySpec:
    strategy_name: str
    selection_rule: str
    predicate: Callable[[dict[str, object]], bool]


@dataclass(frozen=True)
class ForwardWindow:
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    data_status: str


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Project C US stock rank strategies.")
    parser.add_argument("--start-date", required=True, help="Backtest start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Backtest end date. Format: YYYY-MM-DD.")
    parser.add_argument("--top-n", type=int, default=20, help="Custom Top-N strategy size. Default: 20.")
    parser.add_argument(
        "--holding-days",
        default="5,20,60",
        help="Comma-separated holding-day list. Default: 5,20,60.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional strategy filter: TOP5, TOP10, TOP20, BUY_OR_BETTER, STRONG_BUY, or full US_RANK_* name.",
    )
    parser.add_argument("--backtest-id", default=None, help="Optional fixed backtest ID.")
    parser.add_argument("--source", default=None, help="Ranking source. Default: US_RANKING_DEFAULT_SOURCE.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summaries without DB writes.")
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    values = sorted(set(value for value in values if value > 0))
    if not values:
        raise ValueError("holding_days must contain at least one positive integer.")
    return values


def _normalize_source_tag(source: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(source or "").strip().upper()).strip("_")
    return token or "RULE_V1"


def build_backtest_id(*, start_date: date, end_date: date, holding_days: list[int], source: str) -> str:
    hd_token = "_".join(str(value) for value in holding_days)
    return f"US_RANK_{_normalize_source_tag(source)}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_HD{hd_token}"


def _fmt_pct(value: object) -> str:
    num = _safe_float(value)
    if num is None:
        return "-"
    return f"{num * 100:.2f}%"


def _fmt_num(value: object, digits: int = 4) -> str:
    num = _safe_float(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def _group_rows_by_trade_date(rows: list[dict[str, object]]) -> dict[date, list[dict[str, object]]]:
    grouped: dict[date, list[dict[str, object]]] = {}
    for row in rows:
        trade_date = row.get("trade_date")
        if isinstance(trade_date, date):
            grouped.setdefault(trade_date, []).append(row)
    return grouped


def _build_strategy_specs(*, custom_top_n: int) -> dict[str, StrategySpec]:
    return {
        "TOP5": StrategySpec("US_RANK_TOP5", "rank_no <= 5", lambda row: _safe_float(row.get("rank_no")) is not None and int(row["rank_no"]) <= 5 and str(row.get("recommend_grade")) != "EXCLUDE"),
        "TOP10": StrategySpec("US_RANK_TOP10", "rank_no <= 10", lambda row: _safe_float(row.get("rank_no")) is not None and int(row["rank_no"]) <= 10 and str(row.get("recommend_grade")) != "EXCLUDE"),
        "TOP20": StrategySpec("US_RANK_TOP20", "rank_no <= 20", lambda row: _safe_float(row.get("rank_no")) is not None and int(row["rank_no"]) <= 20 and str(row.get("recommend_grade")) != "EXCLUDE"),
        "BUY_OR_BETTER": StrategySpec("US_RANK_BUY_OR_BETTER", "recommend_grade in ('STRONG_BUY', 'BUY')", lambda row: str(row.get("recommend_grade")) in {"STRONG_BUY", "BUY"}),
        "STRONG_BUY": StrategySpec("US_RANK_STRONG_BUY", "recommend_grade = 'STRONG_BUY'", lambda row: str(row.get("recommend_grade")) == "STRONG_BUY"),
        f"TOP{custom_top_n}": StrategySpec(
            f"US_RANK_TOP{custom_top_n}",
            f"rank_no <= {custom_top_n}",
            lambda row, n=custom_top_n: _safe_float(row.get("rank_no")) is not None and int(row["rank_no"]) <= n and str(row.get("recommend_grade")) != "EXCLUDE",
        ),
    }


def resolve_strategy_specs(*, custom_top_n: int, strategy_filter: str | None) -> list[StrategySpec]:
    specs = _build_strategy_specs(custom_top_n=custom_top_n)
    if not strategy_filter:
        ordered_keys = ["TOP5", "TOP10", "TOP20", "BUY_OR_BETTER", "STRONG_BUY"]
        custom_key = f"TOP{custom_top_n}"
        if custom_key not in ordered_keys:
            ordered_keys.append(custom_key)
        return [specs[key] for key in ordered_keys]

    normalized = str(strategy_filter).strip().upper()
    if normalized.startswith("US_RANK_"):
        for spec in specs.values():
            if spec.strategy_name == normalized:
                return [spec]
    if normalized in specs:
        return [specs[normalized]]
    raise ValueError(f"Unsupported strategy '{strategy_filter}'.")


def _price_value(row: dict[str, object]) -> float | None:
    return _safe_float(row.get("adj_close_price")) or _safe_float(row.get("close_price"))


def _build_price_lookup(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        trade_date = row.get("trade_date")
        if not ticker or not isinstance(trade_date, date):
            continue
        price = _price_value(row)
        output.setdefault(ticker, []).append(
            {
                "trade_date": trade_date,
                "price": price,
            }
        )
    for ticker_rows in output.values():
        ticker_rows.sort(key=lambda item: item["trade_date"])
    return output


def resolve_forward_window(price_rows: list[dict[str, object]], *, trade_date: date, holding_days: int) -> ForwardWindow:
    if not price_rows:
        return ForwardWindow(None, None, None, None, "MISSING_ENTRY_PRICE")

    date_list = [item["trade_date"] for item in price_rows if isinstance(item.get("trade_date"), date)]
    entry_index = bisect_right(date_list, trade_date)
    if entry_index >= len(price_rows):
        return ForwardWindow(None, None, None, None, "MISSING_ENTRY_PRICE")

    entry_row = price_rows[entry_index]
    entry_price = _safe_float(entry_row.get("price"))
    if entry_price is None or entry_price <= 0:
        return ForwardWindow(entry_row["trade_date"], None, None, None, "MISSING_ENTRY_PRICE")

    exit_index = entry_index + holding_days
    if exit_index >= len(price_rows):
        return ForwardWindow(entry_row["trade_date"], entry_price, None, None, "NOT_ENOUGH_FORWARD_DATA")

    exit_row = price_rows[exit_index]
    exit_price = _safe_float(exit_row.get("price"))
    if exit_price is None or exit_price <= 0:
        return ForwardWindow(entry_row["trade_date"], entry_price, exit_row["trade_date"], None, "MISSING_EXIT_PRICE")

    return ForwardWindow(entry_row["trade_date"], entry_price, exit_row["trade_date"], exit_price, "OK")


def _compute_return(entry_price: float | None, exit_price: float | None) -> float | None:
    if entry_price is None or exit_price is None or entry_price <= 0:
        return None
    return float((exit_price - entry_price) / entry_price)


def _mean(values: list[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(statistics.fmean(numbers))


def _median(values: list[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(statistics.median(numbers))


def _rate(flags: list[int | None]) -> float | None:
    numbers = [int(value) for value in flags if value is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def select_strategy_rows(rows: list[dict[str, object]], spec: StrategySpec) -> list[dict[str, object]]:
    selected = [row for row in rows if spec.predicate(row)]
    return sorted(selected, key=lambda row: (int(row.get("rank_no") or 999999), str(row.get("symbol") or "")))


def _compute_benchmark_return(
    price_lookup: dict[str, list[dict[str, object]]],
    benchmark: str,
    *,
    trade_date: date,
    holding_days: int,
) -> tuple[float | None, str | None]:
    window = resolve_forward_window(price_lookup.get(benchmark, []), trade_date=trade_date, holding_days=holding_days)
    if window.data_status != "OK":
        return None, window.data_status
    return _compute_return(window.entry_price, window.exit_price), None


def _decorate_universe_average(result_rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[date, int], list[dict[str, object]]] = {}
    for row in result_rows:
        trade_date = row.get("trade_date")
        holding_days = row.get("holding_days")
        if isinstance(trade_date, date) and isinstance(holding_days, int):
            grouped.setdefault((trade_date, holding_days), []).append(row)

    for key, rows in grouped.items():
        avg_return = _mean([_safe_float(row.get("return_pct")) for row in rows])
        for row in rows:
            row["universe_avg_return_pct"] = round(avg_return, 6) if avg_return is not None else None
            return_pct = _safe_float(row.get("return_pct"))
            if return_pct is not None and avg_return is not None:
                row["excess_return_vs_universe"] = round(return_pct - avg_return, 6)
                row["win_vs_universe_flag"] = 1 if (return_pct - avg_return) > 0 else 0
            else:
                row["excess_return_vs_universe"] = None
                row["win_vs_universe_flag"] = None


def build_summary_row(
    *,
    backtest_id: str,
    trade_date: date,
    holding_days: int,
    spec: StrategySpec,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    valid_rows = [row for row in rows if _safe_float(row.get("return_pct")) is not None]
    best_row = max(valid_rows, key=lambda row: float(row["return_pct"])) if valid_rows else None
    worst_row = min(valid_rows, key=lambda row: float(row["return_pct"])) if valid_rows else None

    benchmark_missing = any(str(row.get("data_status")) == "PARTIAL_BENCHMARK_DATA" for row in rows)
    if not rows:
        data_status = "NO_SELECTION"
    elif not valid_rows:
        data_status = "NO_VALID_RETURNS"
    elif benchmark_missing:
        data_status = "PARTIAL_BENCHMARK_DATA"
    elif len(valid_rows) < len(rows):
        data_status = "PARTIAL_FORWARD_DATA"
    else:
        data_status = "OK"

    return {
        "backtest_id": backtest_id,
        "trade_date": trade_date,
        "strategy_name": spec.strategy_name,
        "selection_rule": spec.selection_rule,
        "holding_days": holding_days,
        "selected_count": len(rows),
        "avg_return_pct": round(_mean([_safe_float(row.get("return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "median_return_pct": round(_median([_safe_float(row.get("return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "win_rate": round(_rate([row.get("win_flag") for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_spy_return_pct": round(_mean([_safe_float(row.get("spy_return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_qqq_return_pct": round(_mean([_safe_float(row.get("qqq_return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_universe_return_pct": round(_mean([_safe_float(row.get("universe_avg_return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_excess_return_vs_spy": round(_mean([_safe_float(row.get("excess_return_vs_spy")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_excess_return_vs_qqq": round(_mean([_safe_float(row.get("excess_return_vs_qqq")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "avg_excess_return_vs_universe": round(_mean([_safe_float(row.get("excess_return_vs_universe")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "win_rate_vs_spy": round(_rate([row.get("win_vs_spy_flag") for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "win_rate_vs_qqq": round(_rate([row.get("win_vs_qqq_flag") for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "win_rate_vs_universe": round(_rate([row.get("win_vs_universe_flag") for row in valid_rows]) or 0.0, 6) if valid_rows else None,
        "best_symbol": best_row.get("symbol") if best_row else None,
        "best_return_pct": round(float(best_row["return_pct"]), 6) if best_row and _safe_float(best_row.get("return_pct")) is not None else None,
        "worst_symbol": worst_row.get("symbol") if worst_row else None,
        "worst_return_pct": round(float(worst_row["return_pct"]), 6) if worst_row and _safe_float(worst_row.get("return_pct")) is not None else None,
        "data_status": data_status,
    }


def _print_console_summary(*, backtest_id: str, start_date: date, end_date: date, summary_rows: list[dict[str, object]]) -> None:
    print("[US Stock Rank Backtest Summary]")
    print(f"Backtest ID: {backtest_id}")
    print(f"Period: {start_date.isoformat()} ~ {end_date.isoformat()}")
    print("")
    for row in sorted(summary_rows, key=lambda item: (str(item.get("strategy_name")), int(item.get("holding_days") or 0), item.get("trade_date"))):
        print(f"Strategy: {row.get('strategy_name')}")
        print(f"Holding Days: {row.get('holding_days')}")
        print(f"Trade Date: {row.get('trade_date')}")
        print(f"Selected Count: {row.get('selected_count')}")
        print(f"Avg Return: {_fmt_pct(row.get('avg_return_pct'))}")
        print(f"Avg Excess vs SPY: {_fmt_pct(row.get('avg_excess_return_vs_spy'))}")
        print(f"Avg Excess vs QQQ: {_fmt_pct(row.get('avg_excess_return_vs_qqq'))}")
        print(f"Win Rate: {_fmt_pct(row.get('win_rate'))}")
        print(f"Win Rate vs SPY: {_fmt_pct(row.get('win_rate_vs_spy'))}")
        print(f"Best Symbol: {row.get('best_symbol') or '-'}")
        print(f"Worst Symbol: {row.get('worst_symbol') or '-'}")
        print(f"Data Status: {row.get('data_status') or '-'}")
        print("")


def run_backtest(
    *,
    start_date: date,
    end_date: date,
    holding_days: list[int],
    custom_top_n: int,
    strategy_filter: str | None,
    backtest_id: str,
    source: str,
    dry_run: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rank_rows = fetch_rank_rows_between(start_date=start_date, end_date=end_date, source=source)
    if not rank_rows:
        LOGGER.info("[US_RANK_BACKTEST] No rank rows found for %s ~ %s source=%s", start_date.isoformat(), end_date.isoformat(), source)
        return [], []

    grouped_rank_rows = _group_rows_by_trade_date(rank_rows)
    max_holding = max(holding_days)
    price_end_date = end_date + timedelta(days=max_holding * 3 + 30)
    symbols = sorted({str(row.get("symbol") or "").upper() for row in rank_rows if str(row.get("symbol") or "").strip()} | set(BENCHMARKS))
    price_rows = fetch_price_rows_for_tickers_between(tickers=symbols, start_date=start_date, end_date=price_end_date)
    price_lookup = _build_price_lookup(price_rows)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    strategy_specs = resolve_strategy_specs(custom_top_n=custom_top_n, strategy_filter=strategy_filter)

    for trade_day in sorted(grouped_rank_rows):
        daily_rows = grouped_rank_rows[trade_day]
        universe_bench_cache = {
            hd: {
                "SPY": _compute_benchmark_return(price_lookup, "SPY", trade_date=trade_day, holding_days=hd),
                "QQQ": _compute_benchmark_return(price_lookup, "QQQ", trade_date=trade_day, holding_days=hd),
            }
            for hd in holding_days
        }

        for spec in strategy_specs:
            selected_rows = select_strategy_rows(daily_rows, spec)
            for hd in holding_days:
                strategy_detail_rows: list[dict[str, object]] = []
                for rank_row in selected_rows:
                    symbol = str(rank_row.get("symbol") or "").upper()
                    window = resolve_forward_window(price_lookup.get(symbol, []), trade_date=trade_day, holding_days=hd)
                    return_pct = _compute_return(window.entry_price, window.exit_price) if window.data_status == "OK" else None
                    spy_return_pct, spy_status = universe_bench_cache[hd]["SPY"]
                    qqq_return_pct, qqq_status = universe_bench_cache[hd]["QQQ"]
                    data_status = window.data_status
                    if data_status == "OK" and (spy_status is not None or qqq_status is not None):
                        data_status = "PARTIAL_BENCHMARK_DATA"

                    row = {
                        "backtest_id": backtest_id,
                        "trade_date": trade_day,
                        "strategy_name": spec.strategy_name,
                        "selection_rule": spec.selection_rule,
                        "symbol": symbol,
                        "rank_no": rank_row.get("rank_no"),
                        "recommend_grade": rank_row.get("recommend_grade"),
                        "total_score": rank_row.get("total_score"),
                        "holding_days": hd,
                        "entry_date": window.entry_date,
                        "entry_price": round(window.entry_price, 6) if window.entry_price is not None else None,
                        "exit_date": window.exit_date,
                        "exit_price": round(window.exit_price, 6) if window.exit_price is not None else None,
                        "return_pct": round(return_pct, 6) if return_pct is not None else None,
                        "spy_return_pct": round(spy_return_pct, 6) if spy_return_pct is not None else None,
                        "qqq_return_pct": round(qqq_return_pct, 6) if qqq_return_pct is not None else None,
                        "universe_avg_return_pct": None,
                        "excess_return_vs_spy": round(return_pct - spy_return_pct, 6) if return_pct is not None and spy_return_pct is not None else None,
                        "excess_return_vs_qqq": round(return_pct - qqq_return_pct, 6) if return_pct is not None and qqq_return_pct is not None else None,
                        "excess_return_vs_universe": None,
                        "win_flag": 1 if return_pct is not None and return_pct > 0 else (0 if return_pct is not None else None),
                        "win_vs_spy_flag": 1 if return_pct is not None and spy_return_pct is not None and (return_pct - spy_return_pct) > 0 else (0 if return_pct is not None and spy_return_pct is not None else None),
                        "win_vs_qqq_flag": 1 if return_pct is not None and qqq_return_pct is not None and (return_pct - qqq_return_pct) > 0 else (0 if return_pct is not None and qqq_return_pct is not None else None),
                        "win_vs_universe_flag": None,
                        "data_status": data_status,
                        "exclude_reason": rank_row.get("exclude_reason"),
                        "source": f"rank_{source}",
                    }
                    strategy_detail_rows.append(row)

                _decorate_universe_average(strategy_detail_rows)
                detail_rows.extend(strategy_detail_rows)
                summary_rows.append(
                    build_summary_row(
                        backtest_id=backtest_id,
                        trade_date=trade_day,
                        holding_days=hd,
                        spec=spec,
                        rows=strategy_detail_rows,
                    )
                )

    if not dry_run:
        ensure_us_rank_backtest_tables()
        upsert_us_rank_backtest_result_rows(detail_rows)
        upsert_us_rank_backtest_summary_rows(summary_rows)

    return detail_rows, summary_rows


def _ensure_db() -> None:
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_RANK_BACKTEST] DB connection failed: {exc}") from exc


def main() -> int:
    args = parse_args()
    cfg = load_us_rule_ranking_config()
    setup_logging(cfg.log_level)
    start_date = parse_iso_date(args.start_date, field_name="start_date")
    end_date = parse_iso_date(args.end_date, field_name="end_date")
    if start_date is None or end_date is None:
        raise SystemExit("start_date and end_date are required.")
    if start_date > end_date:
        raise SystemExit("start_date must be on or before end_date.")

    holding_days = _parse_int_csv(args.holding_days)
    source = str(args.source or cfg.source).strip() or cfg.source
    backtest_id = str(args.backtest_id or build_backtest_id(start_date=start_date, end_date=end_date, holding_days=holding_days, source=source)).strip()

    _ensure_db()
    detail_rows, summary_rows = run_backtest(
        start_date=start_date,
        end_date=end_date,
        holding_days=holding_days,
        custom_top_n=max(1, int(args.top_n)),
        strategy_filter=args.strategy,
        backtest_id=backtest_id,
        source=source,
        dry_run=bool(args.dry_run),
    )
    if not summary_rows:
        LOGGER.info("[US_RANK_BACKTEST] No backtest summaries were generated.")
        return 1

    _print_console_summary(backtest_id=backtest_id, start_date=start_date, end_date=end_date, summary_rows=summary_rows)
    LOGGER.info(
        "[US_RANK_BACKTEST] finished backtest_id=%s detail_rows=%s summary_rows=%s dry_run=%s",
        backtest_id,
        len(detail_rows),
        len(summary_rows),
        str(bool(args.dry_run)).lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
