from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import USLabelConfig, load_us_label_config
from python.us.us_db import fetch_active_tickers, fetch_price_history_for_tickers, get_us_engine, upsert_label_rows


LOGGER = logging.getLogger("us_label")
SUPPORTED_SOURCE_TABLES = {"market.us_stock_daily_price"}
SUPPORTED_TARGET_TABLES = {"label.us_stock_label_daily"}
SUPPORTED_PRICE_COLUMNS = {"auto"}
SUPPORTED_WRITE_MODE = "upsert"


@dataclass(frozen=True)
class USLabelBuildResult:
    source_price_rows: int
    label_rows: int
    upserted_rows: int
    skipped_rows: int
    ticker_count: int
    trade_date_min: str | None
    trade_date_max: str | None
    null_ratio_summary: dict[str, float]
    label_distribution: dict[str, dict[str, float]]
    duplicate_key_count: int
    price_column_used: str


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_label_config()
    parser = argparse.ArgumentParser(description="Build US stock labels from daily price history.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag used to load active tickers.")
    parser.add_argument("--ticker", action="append", default=None, help="Build labels for explicit ticker(s).")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for test runs.")
    return parser.parse_args()


def _safe_forward_return(current: Any, future: Any) -> float | None:
    try:
        cur = None if pd.isna(current) else float(current)
        fut = None if pd.isna(future) else float(future)
    except (TypeError, ValueError):
        return None
    if cur in {None, 0.0} or fut is None:
        return None
    value = fut / cur - 1.0
    if pd.isna(value) or value in {float("inf"), float("-inf")}:
        return None
    return float(value)


def prepare_label_frame(price_rows: list[dict[str, object]], *, windows: tuple[int, ...]) -> tuple[pd.DataFrame, str]:
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
    for _, group in frame.groupby("ticker", sort=False):
        ordered = group.sort_values("trade_date").reset_index(drop=True).copy()
        for window in windows:
            ordered[f"future_ret_{window}d"] = [
                _safe_forward_return(cur, fut)
                for cur, fut in zip(ordered["base_price"], ordered["base_price"].shift(-window), strict=False)
            ]
        built_groups.append(ordered)

    out = pd.concat(built_groups, ignore_index=True) if built_groups else frame.iloc[0:0].copy()
    return out, "adj_close_or_close"


def add_label_columns(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    top_percentile: float,
    min_universe_size: int,
    exclude_benchmarks: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    if 20 in windows:
        out["label_positive_20d"] = out["future_ret_20d"].map(
            lambda x: None if pd.isna(x) else (1 if float(x) > 0 else 0)
        )
    if 60 in windows:
        out["label_positive_60d"] = out["future_ret_60d"].map(
            lambda x: None if pd.isna(x) else (1 if float(x) > 0 else 0)
        )

    universe_mask = ~out["ticker"].isin(exclude_benchmarks)
    for horizon in [20, 60]:
        future_col = f"future_ret_{horizon}d"
        rank_col = f"future_ret_{horizon}d_rank_pct"
        label_col = f"label_top20_{horizon}d"
        if future_col not in out.columns:
            out[rank_col] = pd.NA
            out[label_col] = pd.NA
            continue

        rank_source = out.loc[universe_mask, ["ticker", "trade_date", future_col]].copy()
        universe_counts = rank_source.groupby("trade_date")["ticker"].transform("count")
        rank_source[rank_col] = rank_source.groupby("trade_date")[future_col].rank(pct=True, method="average")
        rank_source[label_col] = rank_source.apply(
            lambda row: None
            if pd.isna(row[future_col]) or universe_counts.loc[row.name] < min_universe_size
            else (1 if float(row[rank_col]) >= (1.0 - top_percentile) else 0),
            axis=1,
        )
        out = out.merge(rank_source[["ticker", "trade_date", rank_col, label_col]], on=["ticker", "trade_date"], how="left")
        out.loc[~universe_mask, rank_col] = pd.NA
        out.loc[~universe_mask, label_col] = pd.NA

    return out


def _frame_to_rows(frame: pd.DataFrame, *, source: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    columns = [
        "ticker",
        "trade_date",
        "price_column_used",
        "future_ret_5d",
        "future_ret_20d",
        "future_ret_60d",
        "future_ret_20d_rank_pct",
        "future_ret_60d_rank_pct",
        "label_positive_20d",
        "label_positive_60d",
        "label_top20_20d",
        "label_top20_60d",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc)
    for record in frame[columns].to_dict(orient="records"):
        row: dict[str, object] = {"market": "US", "source": source, "label_created_at": now}
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
        "future_ret_5d",
        "future_ret_20d",
        "future_ret_60d",
        "label_positive_20d",
        "label_positive_60d",
        "label_top20_20d",
        "label_top20_60d",
    ]:
        if column in frame.columns:
            summary[column] = round(float(frame[column].isna().mean()), 4)
    return summary


def _label_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if frame.empty:
        return out
    for column in ["label_positive_20d", "label_positive_60d", "label_top20_20d", "label_top20_60d"]:
        if column in frame.columns:
            valid = frame[column].dropna()
            if valid.empty:
                out[column] = {"ones_ratio": 0.0, "zeros_ratio": 0.0}
            else:
                out[column] = {
                    "ones_ratio": round(float((valid == 1).mean()), 4),
                    "zeros_ratio": round(float((valid == 0).mean()), 4),
                }
    return out


def build_us_stock_labels(
    *,
    cfg: USLabelConfig,
    universe_tag: str,
    explicit_tickers: list[str] | None = None,
    limit: int | None = None,
    ticker_loader=fetch_active_tickers,
    price_fetcher=fetch_price_history_for_tickers,
    row_writer=upsert_label_rows,
) -> USLabelBuildResult:
    if not cfg.enabled:
        LOGGER.info("[US_LABEL] US_LABEL_BUILD_ENABLED=0. Skip builder.")
        return USLabelBuildResult(0, 0, 0, 0, 0, None, None, {}, {}, 0, "adj_close_or_close")
    if cfg.source_price_table not in SUPPORTED_SOURCE_TABLES:
        raise ValueError(f"Unsupported source price table '{cfg.source_price_table}'.")
    if cfg.target_table not in SUPPORTED_TARGET_TABLES:
        raise ValueError(f"Unsupported target table '{cfg.target_table}'.")
    if cfg.price_column not in SUPPORTED_PRICE_COLUMNS:
        raise ValueError(f"Unsupported price column mode '{cfg.price_column}'. Only auto is supported.")
    if cfg.write_mode != SUPPORTED_WRITE_MODE:
        raise ValueError(f"Unsupported write mode '{cfg.write_mode}'. Only upsert is supported.")

    tickers = (
        [str(ticker).strip().upper() for ticker in explicit_tickers if str(ticker).strip()]
        if explicit_tickers
        else ticker_loader(universe_tag)
    )
    if limit is not None:
        tickers = tickers[:limit]
    if not tickers:
        LOGGER.info("[US_LABEL] No active tickers found. universe=%s", universe_tag)
        return USLabelBuildResult(0, 0, 0, 0, 0, None, None, {}, {}, 0, "adj_close_or_close")

    read_tickers = sorted(set(tickers) | set(cfg.exclude_benchmarks))
    price_rows = price_fetcher(read_tickers)
    LOGGER.info("[US_LABEL] label build started")
    LOGGER.info("[US_LABEL] source_price_table=%s", cfg.source_price_table)
    LOGGER.info("[US_LABEL] target_label_table=%s", cfg.target_table)
    LOGGER.info("[US_LABEL] target_tickers=%s windows=%s", len(tickers), ",".join(map(str, cfg.windows)))

    prepared, price_column_used = prepare_label_frame(price_rows, windows=cfg.windows)
    duplicate_key_count = max(0, len(price_rows) - len(prepared))
    labeled = add_label_columns(
        prepared,
        windows=cfg.windows,
        top_percentile=cfg.top_percentile,
        min_universe_size=cfg.min_universe_size,
        exclude_benchmarks=cfg.exclude_benchmarks,
    )
    result_frame = labeled.loc[labeled["ticker"].isin(tickers)].copy()
    rows = _frame_to_rows(result_frame, source=cfg.source_price_table)
    upserted_rows = row_writer(rows)
    null_summary = _null_ratio_summary(result_frame)
    distribution = _label_distribution(result_frame)

    trade_date_min = None if result_frame.empty else str(result_frame["trade_date"].min())
    trade_date_max = None if result_frame.empty else str(result_frame["trade_date"].max())
    LOGGER.info("[US_LABEL] price_column_used=%s", price_column_used)
    LOGGER.info("[US_LABEL] trade_date_range=%s..%s", trade_date_min, trade_date_max)
    LOGGER.info("[US_LABEL] label_rows_built=%s label_rows_upserted=%s skipped_rows=%s", len(result_frame), upserted_rows, 0)
    LOGGER.info("[US_LABEL] future_ret_null_ratio_by_window=%s", null_summary)
    LOGGER.info("[US_LABEL] label_distribution=%s", distribution)
    LOGGER.info("[US_LABEL] duplicate_key_count=%s", duplicate_key_count)
    LOGGER.info("[US_LABEL] label build finished")

    return USLabelBuildResult(
        source_price_rows=len(price_rows),
        label_rows=len(result_frame),
        upserted_rows=upserted_rows,
        skipped_rows=0,
        ticker_count=len(tickers),
        trade_date_min=trade_date_min,
        trade_date_max=trade_date_max,
        null_ratio_summary=null_summary,
        label_distribution=distribution,
        duplicate_key_count=duplicate_key_count,
        price_column_used=price_column_used,
    )


def main() -> None:
    args = parse_args()
    cfg = load_us_label_config()
    setup_logging(cfg.log_level)
    runtime_cfg = USLabelConfig(
        enabled=cfg.enabled,
        source_price_table=cfg.source_price_table,
        target_table=cfg.target_table,
        price_column=cfg.price_column,
        windows=cfg.windows,
        top_percentile=cfg.top_percentile,
        min_universe_size=cfg.min_universe_size,
        exclude_benchmarks=cfg.exclude_benchmarks,
        write_mode=cfg.write_mode,
        log_level=cfg.log_level,
        universe=str(args.universe or cfg.universe).strip().upper() or cfg.universe,
    )
    tickers = [str(name).strip().upper() for name in (args.ticker or []) if str(name).strip()] or None

    if not runtime_cfg.enabled:
        build_us_stock_labels(cfg=runtime_cfg, universe_tag=runtime_cfg.universe, explicit_tickers=tickers, limit=args.limit)
        return
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_LABEL] DB connection failed: {exc}") from exc
    try:
        build_us_stock_labels(cfg=runtime_cfg, universe_tag=runtime_cfg.universe, explicit_tickers=tickers, limit=args.limit)
    except KeyboardInterrupt:
        LOGGER.info("[US_LABEL] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_LABEL] Failed error=%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
