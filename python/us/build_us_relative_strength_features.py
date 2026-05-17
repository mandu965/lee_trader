from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_config import USRelativeStrengthConfig, load_us_relative_strength_config
from python.us.us_db import (
    fetch_active_tickers,
    fetch_price_history_for_tickers,
    get_us_engine,
    upsert_relative_strength_rows,
)


LOGGER = logging.getLogger("us_relative_strength")
SUPPORTED_BENCHMARKS = ("SPY", "QQQ")
SUPPORTED_SOURCE_TABLES = {"market.us_stock_daily_price"}
SUPPORTED_TARGET_TABLES = {"feature.us_stock_relative_strength_daily"}
SUPPORTED_WRITE_MODE = "upsert"
SUPPORTED_PRICE_COLUMNS = {"auto"}


@dataclass(frozen=True)
class RelativeStrengthBuildResult:
    source_price_rows: int
    target_ticker_count: int
    spy_coverage_rows: int
    qqq_coverage_rows: int
    built_rows: int
    upserted_rows: int
    skipped_rows: int
    null_ratio_summary: dict[str, float]
    duplicate_key_count: int
    price_column_used: str


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_relative_strength_config()
    parser = argparse.ArgumentParser(description="Build US relative strength features versus SPY/QQQ.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag used to load active tickers.")
    parser.add_argument("--ticker", action="append", default=None, help="Build for explicit ticker(s).")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for test runs.")
    return parser.parse_args()


def _safe_return(current: Any, previous: Any) -> float | None:
    try:
        cur = None if pd.isna(current) else float(current)
        prev = None if pd.isna(previous) else float(previous)
    except (TypeError, ValueError):
        return None
    if cur is None or prev in {None, 0.0}:
        return None
    value = cur / prev - 1.0
    if pd.isna(value) or value in {float("inf"), float("-inf")}:
        return None
    return float(value)


def _safe_diff(left: Any, right: Any) -> float | None:
    try:
        first = None if pd.isna(left) else float(left)
        second = None if pd.isna(right) else float(right)
    except (TypeError, ValueError):
        return None
    if first is None or second is None:
        return None
    value = first - second
    if pd.isna(value) or value in {float("inf"), float("-inf")}:
        return None
    return float(value)


def _normalize_benchmarks(values: tuple[str, ...]) -> tuple[str, ...]:
    benchmarks = tuple(str(value).strip().upper() for value in values if str(value).strip())
    if set(benchmarks) != set(SUPPORTED_BENCHMARKS):
        raise ValueError(f"Unsupported benchmarks={benchmarks}. Only SPY,QQQ is supported in Phase 2-4.")
    return SUPPORTED_BENCHMARKS


def _normalize_windows(values: tuple[int, ...]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values if int(value) > 0)
    if not windows:
        raise ValueError("No relative strength windows configured.")
    return windows


def prepare_relative_strength_frame(price_rows: list[dict[str, object]], *, windows: tuple[int, ...]) -> tuple[pd.DataFrame, str]:
    if not price_rows:
        return pd.DataFrame(), "adj_close_or_close"

    frame = pd.DataFrame(price_rows)
    for column in ["close_price", "adj_close_price"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["ticker", "trade_date"]).copy()
    frame["base_price"] = frame["adj_close_price"].where(frame["adj_close_price"].notna(), frame["close_price"])
    frame.loc[frame["base_price"] <= 0, "base_price"] = pd.NA
    frame["price_column_used"] = frame["adj_close_price"].notna().map(lambda x: "adj_close_price" if x else "close_price")
    frame = frame.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    built_groups: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", sort=False):
        ordered = group.sort_values("trade_date").reset_index(drop=True).copy()
        for window in windows:
            ordered[f"ret_{window}d"] = [
                _safe_return(cur, prev)
                for cur, prev in zip(ordered["base_price"], ordered["base_price"].shift(window), strict=False)
            ]
        built_groups.append(ordered)

    out = pd.concat(built_groups, ignore_index=True) if built_groups else frame.iloc[0:0].copy()
    return out, "adj_close_or_close"


def apply_benchmark_relative_strength(
    feature_df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    benchmarks: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if feature_df.empty:
        return feature_df.copy(), {"SPY": 0, "QQQ": 0}

    out = feature_df.copy()
    coverage: dict[str, int] = {}
    benchmark_frames: dict[str, pd.DataFrame] = {}
    for benchmark in benchmarks:
        bench = out.loc[out["ticker"] == benchmark, ["trade_date"] + [f"ret_{window}d" for window in windows]].copy()
        bench = bench.rename(columns={f"ret_{window}d": f"{benchmark.lower()}_ret_{window}d" for window in windows})
        coverage[benchmark] = len(bench)
        benchmark_frames[benchmark] = bench
        if bench.empty:
            LOGGER.warning("[US_RS] Missing benchmark data benchmark=%s. Related features will be null.", benchmark)
        else:
            out = out.merge(bench, on="trade_date", how="left")

    for benchmark in benchmarks:
        lower = benchmark.lower()
        for window in windows:
            bench_col = f"{lower}_ret_{window}d"
            if bench_col not in out.columns:
                out[bench_col] = pd.NA
            out[f"rs_{lower}_{window}d"] = [
                _safe_diff(left, right)
                for left, right in zip(out[f"ret_{window}d"], out[bench_col], strict=False)
            ]

    return out, coverage


def _frame_to_rows(frame: pd.DataFrame, *, source: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    columns = [
        "ticker",
        "trade_date",
        "price_column_used",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "ret_252d",
        "spy_ret_5d",
        "spy_ret_20d",
        "spy_ret_60d",
        "spy_ret_120d",
        "spy_ret_252d",
        "qqq_ret_5d",
        "qqq_ret_20d",
        "qqq_ret_60d",
        "qqq_ret_120d",
        "qqq_ret_252d",
        "rs_spy_5d",
        "rs_spy_20d",
        "rs_spy_60d",
        "rs_spy_120d",
        "rs_spy_252d",
        "rs_qqq_5d",
        "rs_qqq_20d",
        "rs_qqq_60d",
        "rs_qqq_120d",
        "rs_qqq_252d",
        "rs_spy_20d_rank_pct",
        "rs_spy_60d_rank_pct",
        "rs_qqq_20d_rank_pct",
        "rs_qqq_60d_rank_pct",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    rows: list[dict[str, object]] = []
    for record in frame[columns].to_dict(orient="records"):
        row: dict[str, object] = {
            "market": "US",
            "source": source,
        }
        for key, value in record.items():
            if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
                row[key] = None
            else:
                row[key] = value
        rows.append(row)
    return rows


def _null_ratio_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    summary: dict[str, float] = {}
    for column in [
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "ret_252d",
        "rs_spy_20d",
        "rs_spy_60d",
        "rs_qqq_20d",
        "rs_qqq_60d",
    ]:
        if column in frame.columns:
            summary[column] = round(float(frame[column].isna().mean()), 4)
    return summary


def build_us_relative_strength_features(
    *,
    cfg: USRelativeStrengthConfig,
    universe_tag: str,
    explicit_tickers: list[str] | None = None,
    limit: int | None = None,
    price_fetcher=fetch_price_history_for_tickers,
    row_writer=upsert_relative_strength_rows,
) -> RelativeStrengthBuildResult:
    if not cfg.enabled:
        LOGGER.info("[US_RS] US_RELATIVE_STRENGTH_BUILD_ENABLED=0. Skip builder.")
        return RelativeStrengthBuildResult(0, 0, 0, 0, 0, 0, 0, {}, 0, "adj_close_or_close")

    benchmarks = _normalize_benchmarks(cfg.benchmarks)
    windows = _normalize_windows(cfg.windows)
    if cfg.source_table not in SUPPORTED_SOURCE_TABLES:
        raise ValueError(f"Unsupported source table '{cfg.source_table}'.")
    if cfg.target_table not in SUPPORTED_TARGET_TABLES:
        raise ValueError(f"Unsupported target table '{cfg.target_table}'.")
    if cfg.write_mode != SUPPORTED_WRITE_MODE:
        raise ValueError(f"Unsupported write mode '{cfg.write_mode}'. Only upsert is supported.")
    if cfg.price_column not in SUPPORTED_PRICE_COLUMNS:
        raise ValueError(f"Unsupported price column mode '{cfg.price_column}'. Only auto is supported.")

    target_tickers = (
        [str(ticker).strip().upper() for ticker in explicit_tickers if str(ticker).strip()]
        if explicit_tickers
        else fetch_active_tickers(universe_tag)
    )
    if limit is not None:
        target_tickers = target_tickers[:limit]
    if not target_tickers:
        LOGGER.info("[US_RS] No active tickers found. universe=%s", universe_tag)
        return RelativeStrengthBuildResult(0, 0, 0, 0, 0, 0, 0, {}, 0, "adj_close_or_close")

    read_tickers = sorted(set(target_tickers) | set(benchmarks))
    price_rows = price_fetcher(read_tickers)
    LOGGER.info("[US_RS] relative strength build started")
    LOGGER.info("[US_RS] source_table=%s", cfg.source_table)
    LOGGER.info("[US_RS] target_table=%s", cfg.target_table)
    LOGGER.info("[US_RS] target_tickers=%s benchmarks=%s", len(target_tickers), ",".join(benchmarks))

    prepared, price_column_used = prepare_relative_strength_frame(price_rows, windows=windows)
    duplicate_key_count = max(0, len(price_rows) - len(prepared))
    enriched, coverage = apply_benchmark_relative_strength(prepared, windows=windows, benchmarks=benchmarks)
    result_frame = enriched.loc[enriched["ticker"].isin(target_tickers)].copy()
    for metric in ["rs_spy_20d", "rs_spy_60d", "rs_qqq_20d", "rs_qqq_60d"]:
        if metric in result_frame.columns:
            result_frame[f"{metric}_rank_pct"] = result_frame.groupby("trade_date")[metric].rank(
                pct=True,
                method="average",
            )
            result_frame.loc[result_frame[metric].isna(), f"{metric}_rank_pct"] = pd.NA
    rows = _frame_to_rows(result_frame, source=cfg.source_table)
    upserted_rows = row_writer(rows)
    null_summary = _null_ratio_summary(result_frame)

    LOGGER.info("[US_RS] price_column_used=%s", price_column_used)
    LOGGER.info("[US_RS] benchmark_coverage spy=%s qqq=%s", coverage.get("SPY", 0), coverage.get("QQQ", 0))
    LOGGER.info("[US_RS] built_rows=%s upserted_rows=%s skipped_rows=%s", len(result_frame), upserted_rows, 0)
    LOGGER.info("[US_RS] null_ratio_by_window=%s", null_summary)
    LOGGER.info("[US_RS] duplicate_key_count=%s", duplicate_key_count)
    LOGGER.info("[US_RS] relative strength build finished")

    return RelativeStrengthBuildResult(
        source_price_rows=len(price_rows),
        target_ticker_count=len(target_tickers),
        spy_coverage_rows=coverage.get("SPY", 0),
        qqq_coverage_rows=coverage.get("QQQ", 0),
        built_rows=len(result_frame),
        upserted_rows=upserted_rows,
        skipped_rows=0,
        null_ratio_summary=null_summary,
        duplicate_key_count=duplicate_key_count,
        price_column_used=price_column_used,
    )


def main() -> None:
    args = parse_args()
    cfg = load_us_relative_strength_config()
    setup_logging(cfg.log_level)
    runtime_cfg = USRelativeStrengthConfig(
        enabled=cfg.enabled,
        source_table=cfg.source_table,
        target_table=cfg.target_table,
        benchmarks=cfg.benchmarks,
        windows=cfg.windows,
        price_column=cfg.price_column,
        write_mode=cfg.write_mode,
        log_level=cfg.log_level,
        universe=str(args.universe or cfg.universe).strip().upper() or cfg.universe,
    )
    tickers = [str(name).strip().upper() for name in (args.ticker or []) if str(name).strip()] or None

    if not runtime_cfg.enabled:
        build_us_relative_strength_features(
            cfg=runtime_cfg,
            universe_tag=runtime_cfg.universe,
            explicit_tickers=tickers,
            limit=args.limit,
        )
        return

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_RS] DB connection failed: {exc}") from exc

    try:
        build_us_relative_strength_features(
            cfg=runtime_cfg,
            universe_tag=runtime_cfg.universe,
            explicit_tickers=tickers,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        LOGGER.info("[US_RS] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_RS] Failed error=%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
