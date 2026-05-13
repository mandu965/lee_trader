from __future__ import annotations

import argparse
from datetime import date, timedelta
import logging
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_market_regime_config, parse_iso_date
from python.us.us_db import (
    ensure_us_market_regime_tables,
    get_us_engine,
    upsert_us_market_regime_rows,
)


LOGGER = logging.getLogger("us_market_regime")
BENCHMARKS = ("SPY", "QQQ")
LOOKBACK_BUFFER_DAYS = 120


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily US market regime rows from SPY/QQQ price data.")
    parser.add_argument("--trade-date", default=None, help="Single trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--start-date", default=None, help="Start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="End date. Format: YYYY-MM-DD.")
    parser.add_argument("--dry-run", action="store_true", help="Compute regime rows without DB writes.")
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _price_value(row: dict[str, object]) -> float | None:
    adj = _safe_float(row.get("adj_close_price"))
    close = _safe_float(row.get("close_price"))
    return adj if adj is not None and adj > 0 else close


def _load_price_frame(*, start_date: date, end_date: date) -> pd.DataFrame:
    fetch_start = start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)
    stmt = text(
        """
        SELECT
            trade_date,
            ticker,
            close_price AS close_price,
            adj_close_price AS adj_close_price
        FROM market.us_stock_daily_price
        WHERE ticker = ANY(:tickers)
          AND trade_date BETWEEN :start_date AND :end_date
        UNION ALL
        SELECT
            trade_date,
            ticker,
            close AS close_price,
            adj_close AS adj_close_price
        FROM market.us_etf_daily_price
        WHERE ticker = ANY(:tickers)
          AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY ticker, trade_date
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(
            stmt,
            {"tickers": list(BENCHMARKS), "start_date": fetch_start, "end_date": end_date},
        ).mappings().all()
    payload: list[dict[str, object]] = []
    for row in rows:
        trade_date = row.get("trade_date")
        ticker = str(row.get("ticker") or "").upper()
        price = _price_value(row)
        if isinstance(trade_date, date) and ticker in BENCHMARKS and price is not None and price > 0:
            payload.append({"trade_date": trade_date, "ticker": ticker, "price": price})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def _derive_trend_regime(*, close: float | None, ret60: float | None, ma60: float | None, bull_value: str, bear_value: str, sideways_value: str) -> str:
    if close is None or ret60 is None or ma60 is None:
        return "UNKNOWN"
    if ret60 > 0 and close > ma60:
        return bull_value
    if ret60 < 0 and close < ma60:
        return bear_value
    return sideways_value


def _derive_vol_regime(*, spy_vol20: float | None, qqq_vol20: float | None, cfg) -> str:
    spy_high = spy_vol20 is not None and spy_vol20 >= cfg.spy_vol20_high_threshold
    qqq_high = qqq_vol20 is not None and qqq_vol20 >= cfg.qqq_vol20_high_threshold
    if spy_vol20 is None and qqq_vol20 is None:
        return "UNKNOWN"
    return "HIGH_VOL" if spy_high or qqq_high else "LOW_VOL"


def _derive_market_regime(*, spy_regime: str, vol_regime: str) -> str:
    if spy_regime == "BULL" and vol_regime == "HIGH_VOL":
        return "BULL_HIGH_VOL"
    if spy_regime == "BULL" and vol_regime == "LOW_VOL":
        return "BULL_LOW_VOL"
    if spy_regime == "BEAR" and vol_regime == "HIGH_VOL":
        return "BEAR_HIGH_VOL"
    if spy_regime == "BEAR" and vol_regime == "LOW_VOL":
        return "BEAR_LOW_VOL"
    if spy_regime == "SIDEWAYS" and vol_regime == "HIGH_VOL":
        return "SIDEWAYS_HIGH_VOL"
    if spy_regime == "SIDEWAYS" and vol_regime == "LOW_VOL":
        return "SIDEWAYS_LOW_VOL"
    return "UNKNOWN"


def compute_market_regime_rows(*, start_date: date, end_date: date, cfg) -> list[dict[str, object]]:
    frame = _load_price_frame(start_date=start_date, end_date=end_date)
    if frame.empty:
        return []

    output_frames: list[pd.DataFrame] = []
    for ticker in BENCHMARKS:
        ticker_frame = frame[frame["ticker"] == ticker].copy()
        if ticker_frame.empty:
            continue
        ticker_frame["return_20d"] = ticker_frame["price"] / ticker_frame["price"].shift(20) - 1.0
        ticker_frame["return_60d"] = ticker_frame["price"] / ticker_frame["price"].shift(60) - 1.0
        ticker_frame["ma20"] = ticker_frame["price"].rolling(window=20, min_periods=20).mean()
        ticker_frame["ma60"] = ticker_frame["price"].rolling(window=60, min_periods=60).mean()
        ticker_frame["daily_ret_1d"] = ticker_frame["price"].pct_change()
        ticker_frame["volatility_20d"] = ticker_frame["daily_ret_1d"].rolling(window=20, min_periods=20).std()
        rename_map = {
            "price": f"{ticker.lower()}_close",
            "return_20d": f"{ticker.lower()}_return_20d",
            "return_60d": f"{ticker.lower()}_return_60d",
            "ma20": f"{ticker.lower()}_ma20",
            "ma60": f"{ticker.lower()}_ma60",
            "volatility_20d": f"{ticker.lower()}_volatility_20d",
        }
        ticker_frame = ticker_frame[["trade_date", *rename_map.keys()]].rename(columns=rename_map)
        output_frames.append(ticker_frame)

    if not output_frames:
        return []

    merged = output_frames[0]
    for extra in output_frames[1:]:
        merged = merged.merge(extra, on="trade_date", how="outer")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged = merged[
        (merged["trade_date"] >= pd.Timestamp(start_date)) &
        (merged["trade_date"] <= pd.Timestamp(end_date))
    ].copy()

    rows: list[dict[str, object]] = []
    for _, rec in merged.iterrows():
        trade_date_value = rec["trade_date"]
        if pd.isna(trade_date_value):
            continue
        spy_close = _safe_float(rec.get("spy_close"))
        spy_ret60 = _safe_float(rec.get("spy_return_60d"))
        spy_ma60 = _safe_float(rec.get("spy_ma60"))
        qqq_close = _safe_float(rec.get("qqq_close"))
        qqq_ret60 = _safe_float(rec.get("qqq_return_60d"))
        qqq_ma60 = _safe_float(rec.get("qqq_ma60"))
        spy_regime = _derive_trend_regime(close=spy_close, ret60=spy_ret60, ma60=spy_ma60, bull_value="BULL", bear_value="BEAR", sideways_value="SIDEWAYS")
        qqq_regime = _derive_trend_regime(close=qqq_close, ret60=qqq_ret60, ma60=qqq_ma60, bull_value="QQQ_BULL", bear_value="QQQ_BEAR", sideways_value="QQQ_SIDEWAYS")
        vol_regime = _derive_vol_regime(
            spy_vol20=_safe_float(rec.get("spy_volatility_20d")),
            qqq_vol20=_safe_float(rec.get("qqq_volatility_20d")),
            cfg=cfg,
        )
        market_regime = _derive_market_regime(spy_regime=spy_regime, vol_regime=vol_regime)
        data_status = "OK"
        if spy_regime == "UNKNOWN" or vol_regime == "UNKNOWN":
            data_status = "INSUFFICIENT_LOOKBACK"
        if spy_close is None:
            data_status = "MISSING_SPY_PRICE"
            spy_regime = "UNKNOWN"
            market_regime = "UNKNOWN"
        elif qqq_close is None:
            data_status = "MISSING_QQQ_PRICE"
            qqq_regime = "UNKNOWN"
        rows.append(
            {
                "trade_date": trade_date_value.date(),
                "spy_close": round(spy_close, 6) if spy_close is not None else None,
                "spy_return_20d": round(_safe_float(rec.get("spy_return_20d")), 6) if _safe_float(rec.get("spy_return_20d")) is not None else None,
                "spy_return_60d": round(spy_ret60, 6) if spy_ret60 is not None else None,
                "spy_ma20": round(_safe_float(rec.get("spy_ma20")), 6) if _safe_float(rec.get("spy_ma20")) is not None else None,
                "spy_ma60": round(spy_ma60, 6) if spy_ma60 is not None else None,
                "spy_volatility_20d": round(_safe_float(rec.get("spy_volatility_20d")), 6) if _safe_float(rec.get("spy_volatility_20d")) is not None else None,
                "qqq_close": round(qqq_close, 6) if qqq_close is not None else None,
                "qqq_return_20d": round(_safe_float(rec.get("qqq_return_20d")), 6) if _safe_float(rec.get("qqq_return_20d")) is not None else None,
                "qqq_return_60d": round(qqq_ret60, 6) if qqq_ret60 is not None else None,
                "qqq_ma20": round(_safe_float(rec.get("qqq_ma20")), 6) if _safe_float(rec.get("qqq_ma20")) is not None else None,
                "qqq_ma60": round(qqq_ma60, 6) if qqq_ma60 is not None else None,
                "qqq_volatility_20d": round(_safe_float(rec.get("qqq_volatility_20d")), 6) if _safe_float(rec.get("qqq_volatility_20d")) is not None else None,
                "spy_regime": spy_regime,
                "qqq_regime": qqq_regime,
                "vol_regime": vol_regime,
                "market_regime": market_regime,
                "data_status": data_status,
            }
        )
    return rows


def _ensure_db() -> None:
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_MARKET_REGIME] DB connection failed: {exc}") from exc


def main() -> int:
    args = parse_args()
    cfg = load_us_market_regime_config()
    setup_logging(cfg.log_level)
    if args.trade_date:
        trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
        if trade_date is None:
            raise SystemExit("trade_date is required.")
        start_date = trade_date
        end_date = trade_date
    else:
        start_date = parse_iso_date(args.start_date, field_name="start_date") if args.start_date else None
        end_date = parse_iso_date(args.end_date, field_name="end_date") if args.end_date else None
        if start_date is None or end_date is None:
            raise SystemExit("Provide --trade-date or both --start-date and --end-date.")
    if start_date > end_date:
        raise SystemExit("start_date must be on or before end_date.")

    _ensure_db()
    rows = compute_market_regime_rows(start_date=start_date, end_date=end_date, cfg=cfg)
    if not rows:
        LOGGER.info("[US_MARKET_REGIME] No SPY/QQQ price rows found for %s ~ %s", start_date.isoformat(), end_date.isoformat())
        return 1
    LOGGER.info(
        "[US_MARKET_REGIME] computed rows=%s period=%s~%s dry_run=%s",
        len(rows),
        start_date.isoformat(),
        end_date.isoformat(),
        str(bool(args.dry_run)).lower(),
    )
    if not args.dry_run:
        ensure_us_market_regime_tables()
        written = upsert_us_market_regime_rows(rows)
        LOGGER.info("[US_MARKET_REGIME] upserted rows=%s", written)
    for row in rows[: min(5, len(rows))]:
        LOGGER.info(
            "[US_MARKET_REGIME] trade_date=%s market_regime=%s spy_regime=%s qqq_regime=%s vol_regime=%s status=%s",
            row["trade_date"],
            row["market_regime"],
            row["spy_regime"],
            row["qqq_regime"],
            row["vol_regime"],
            row["data_status"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
