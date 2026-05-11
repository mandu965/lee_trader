from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_stock_config, parse_iso_date
from python.us.us_db import fetch_active_tickers, fetch_price_history, get_us_engine, upsert_feature_rows


LOGGER = logging.getLogger("us_feature")
MIN_HISTORY_ROWS = 60


@dataclass(frozen=True)
class FeatureBuildResult:
    universe_tag: str
    processed_ticker_count: int
    success_count: int
    skipped_count: int
    failed_count: int
    total_rows: int


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_stock_config()
    parser = argparse.ArgumentParser(description="Build baseline US stock daily features.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag. Default: env US_STOCK_UNIVERSE.")
    parser.add_argument("--start-date", default=None, help="Feature save start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Feature save end date. Format: YYYY-MM-DD.")
    parser.add_argument("--ticker", default=None, help="Build features for a single ticker only.")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for testing.")
    parser.add_argument("--verbose", action="store_true", help="Print per-ticker processing details.")
    return parser.parse_args()


def _safe_flag(price: float | None, moving_average: float | None) -> str | None:
    if price is None or moving_average is None or pd.isna(price) or pd.isna(moving_average):
        return None
    return "Y" if float(price) > float(moving_average) else "N"


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature_date"] = pd.to_datetime(out["trade_date"]).dt.date
    for column in ["open_price", "high_price", "low_price", "close_price", "adj_close_price", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["base_price"] = out["adj_close_price"].where(out["adj_close_price"].notna(), out["close_price"])
    out.loc[out["base_price"] <= 0, "base_price"] = pd.NA
    out["volume_clean"] = pd.to_numeric(out["volume"], errors="coerce")
    out.loc[out["volume_clean"] < 0, "volume_clean"] = pd.NA

    out["ret_5d"] = out["base_price"] / out["base_price"].shift(5) - 1.0
    out["ret_10d"] = out["base_price"] / out["base_price"].shift(10) - 1.0
    out["ret_20d"] = out["base_price"] / out["base_price"].shift(20) - 1.0
    out["ret_60d"] = out["base_price"] / out["base_price"].shift(60) - 1.0

    out["volume_avg_20d"] = out["volume_clean"].rolling(window=20, min_periods=20).mean()
    out["daily_ret_1d"] = out["base_price"].pct_change()
    out["volatility_20d"] = out["daily_ret_1d"].rolling(window=20, min_periods=20).std()
    out["ma_20"] = out["base_price"].rolling(window=20, min_periods=20).mean()
    out["ma_60"] = out["base_price"].rolling(window=60, min_periods=60).mean()
    out["price_above_ma20_flag"] = [
        _safe_flag(price, ma) for price, ma in zip(out["base_price"], out["ma_20"], strict=False)
    ]
    out["price_above_ma60_flag"] = [
        _safe_flag(price, ma) for price, ma in zip(out["base_price"], out["ma_60"], strict=False)
    ]
    return out


def _frame_to_feature_rows(df: pd.DataFrame, *, ticker: str, start_date: date | None, end_date: date | None) -> list[dict[str, object]]:
    eligible = df.loc[df["ma_60"].notna()].copy()
    if start_date is not None:
        eligible = eligible.loc[eligible["feature_date"] >= start_date]
    if end_date is not None:
        eligible = eligible.loc[eligible["feature_date"] <= end_date]

    if eligible.empty:
        return []

    rows: list[dict[str, object]] = []
    for row in eligible.itertuples(index=False):
        rows.append(
            {
                "feature_date": row.feature_date,
                "ticker": ticker,
                "ret_5d": None if pd.isna(row.ret_5d) else float(row.ret_5d),
                "ret_10d": None if pd.isna(row.ret_10d) else float(row.ret_10d),
                "ret_20d": None if pd.isna(row.ret_20d) else float(row.ret_20d),
                "ret_60d": None if pd.isna(row.ret_60d) else float(row.ret_60d),
                "volume_avg_20d": None if pd.isna(row.volume_avg_20d) else float(row.volume_avg_20d),
                "volatility_20d": None if pd.isna(row.volatility_20d) else float(row.volatility_20d),
                "ma_20": None if pd.isna(row.ma_20) else float(row.ma_20),
                "ma_60": None if pd.isna(row.ma_60) else float(row.ma_60),
                "price_above_ma20_flag": row.price_above_ma20_flag,
                "price_above_ma60_flag": row.price_above_ma60_flag,
            }
        )
    return rows


def _select_tickers(*, universe_tag: str, only_ticker: str | None, limit: int | None) -> list[str]:
    tickers = fetch_active_tickers(universe_tag)
    if only_ticker:
        ticker = str(only_ticker).strip().upper()
        tickers = [name for name in tickers if name == ticker]
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def build_us_features(
    *,
    universe_tag: str,
    start_date: date | None = None,
    end_date: date | None = None,
    ticker: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> FeatureBuildResult:
    tickers = _select_tickers(universe_tag=universe_tag, only_ticker=ticker, limit=limit)
    LOGGER.info("[US_FEATURE] Start feature build universe=%s", universe_tag)
    LOGGER.info("[US_FEATURE] Active tickers=%s", len(tickers))
    if not tickers:
        LOGGER.info("[US_FEATURE] No active tickers found. universe=%s", universe_tag)
        return FeatureBuildResult(
            universe_tag=universe_tag,
            processed_ticker_count=0,
            success_count=0,
            skipped_count=0,
            failed_count=0,
            total_rows=0,
        )

    success = 0
    skipped = 0
    failed = 0
    total_rows = 0

    for name in tickers:
        try:
            history = fetch_price_history(name, end_date=end_date)
            if not history:
                skipped += 1
                LOGGER.info("[US_FEATURE] %s SKIPPED reason=no price data rows=0", name)
                continue

            frame = pd.DataFrame(history)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame = frame.sort_values("trade_date").reset_index(drop=True)
            frame = frame.loc[
                frame["ticker"].notna()
                & frame["trade_date"].notna()
                & frame["close_price"].notna()
            ].copy()
            if len(frame) < MIN_HISTORY_ROWS:
                skipped += 1
                LOGGER.info("[US_FEATURE] %s SKIPPED reason=not enough history rows=%s", name, len(frame))
                continue

            feature_frame = _compute_features(frame)
            feature_rows = _frame_to_feature_rows(feature_frame, ticker=name, start_date=start_date, end_date=end_date)
            if not feature_rows:
                skipped += 1
                LOGGER.info("[US_FEATURE] %s SKIPPED reason=no eligible feature rows rows=0", name)
                continue

            saved = upsert_feature_rows(feature_rows)
            success += 1
            total_rows += saved
            if verbose:
                LOGGER.info(
                    "[US_FEATURE] %s SUCCESS rows=%s feature_date_range=%s..%s",
                    name,
                    saved,
                    feature_rows[0]["feature_date"],
                    feature_rows[-1]["feature_date"],
                )
            else:
                LOGGER.info("[US_FEATURE] %s SUCCESS rows=%s", name, saved)
        except Exception as exc:
            failed += 1
            LOGGER.info("[US_FEATURE] %s FAILED error=%s", name, exc)

    LOGGER.info("[US_FEATURE] Finished success=%s skipped=%s failed=%s total_rows=%s", success, skipped, failed, total_rows)
    return FeatureBuildResult(
        universe_tag=universe_tag,
        processed_ticker_count=len(tickers),
        success_count=success,
        skipped_count=skipped,
        failed_count=failed,
        total_rows=total_rows,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_us_stock_config()
    universe_tag = str(args.universe or cfg.universe).strip().upper() or "NASDAQ100"
    start_date = parse_iso_date(args.start_date, field_name="start_date")
    end_date = parse_iso_date(args.end_date, field_name="end_date")
    if start_date and end_date and start_date > end_date:
        raise SystemExit(f"Invalid date window: start_date {start_date.isoformat()} is after end_date {end_date.isoformat()}.")

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_FEATURE] DB connection failed: {exc}") from exc

    build_us_features(
        universe_tag=universe_tag,
        start_date=start_date,
        end_date=end_date,
        ticker=args.ticker,
        limit=args.limit,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
