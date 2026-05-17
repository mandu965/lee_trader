from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_config import load_us_rule_ranking_config, parse_iso_date
from python.us.us_db import (
    delete_us_rank_rows,
    fetch_latest_daily_feature_snapshots,
    fetch_latest_financial_feature_snapshots,
    fetch_latest_macro_snapshot,
    fetch_latest_relative_strength_snapshots,
    fetch_meta_us_universe_rows,
    fetch_price_history_for_tickers,
    get_us_engine,
    upsert_us_rank_rows,
)


LOGGER = logging.getLogger("us_rule_rank")
BENCHMARKS = ("SPY", "QQQ")
PRICE_LOOKBACK_MIN_ROWS = 60


@dataclass(frozen=True)
class ScoreComponentResult:
    score: float
    detail: dict[str, object]


@dataclass(frozen=True)
class RuleRankingRunResult:
    trade_date: date
    source: str
    evaluated_symbol_count: int
    written_row_count: int
    top_rows: list[dict[str, object]]
    dry_run: bool


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Rule-based US stock ranking scores.")
    parser.add_argument("--trade-date", required=True, help="Ranking trade date. Format: YYYY-MM-DD.")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols to score.")
    parser.add_argument("--dry-run", action="store_true", help="Calculate scores without writing DB rows.")
    parser.add_argument("--top-n", type=int, default=20, help="How many rows to print in the summary output.")
    parser.add_argument("--source", default=None, help="Override ranking source tag. Default: env rule_v1.")
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_ratio(value: object) -> float | None:
    raw = _safe_float(value)
    if raw is None:
        return None
    if abs(raw) > 3.0:
        return raw / 100.0
    return raw


def _normalize_debt_to_equity(value: object) -> float | None:
    raw = _safe_float(value)
    if raw is None:
        return None
    # Some vendors store 42 meaning 42%, while the scoring rules expect 0.42.
    if abs(raw) > 10.0:
        return raw / 100.0
    return raw


def _bool_flag(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().upper()
    if text_value in {"Y", "YES", "TRUE", "1"}:
        return True
    if text_value in {"N", "NO", "FALSE", "0"}:
        return False
    return None


def _json_default(value: object) -> object:
    if isinstance(value, (date,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    numeric = _safe_float(value)
    if numeric is not None and not isinstance(value, str):
        return numeric
    return str(value)


def _cap(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_symbols_arg(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    symbols = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
    return symbols or None


def _compute_history_snapshot(history_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    if not history_rows:
        return {}
    frame = pd.DataFrame(history_rows)
    if frame.empty:
        return {}
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["ticker", "trade_date"]).copy()
    if frame.empty:
        return {}

    for column in ["close_price", "adj_close_price", "volume"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["base_price"] = frame["adj_close_price"].where(frame["adj_close_price"].notna(), frame["close_price"])
    frame.loc[frame["base_price"] <= 0, "base_price"] = pd.NA
    frame["volume_clean"] = frame["volume"]
    frame.loc[frame["volume_clean"] < 0, "volume_clean"] = pd.NA
    frame = frame.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    output: dict[str, dict[str, object]] = {}
    for ticker, group in frame.groupby("ticker", sort=False):
        ordered = group.sort_values("trade_date").reset_index(drop=True).copy()
        ordered["ret_20d"] = ordered["base_price"] / ordered["base_price"].shift(20) - 1.0
        ordered["ret_60d"] = ordered["base_price"] / ordered["base_price"].shift(60) - 1.0
        ordered["ret_120d"] = ordered["base_price"] / ordered["base_price"].shift(120) - 1.0
        ordered["daily_ret_1d"] = ordered["base_price"].pct_change()
        ordered["volatility_20d"] = ordered["daily_ret_1d"].rolling(window=20, min_periods=20).std()
        ordered["volatility_60d"] = ordered["daily_ret_1d"].rolling(window=60, min_periods=60).std()
        ordered["ma_20"] = ordered["base_price"].rolling(window=20, min_periods=20).mean()
        ordered["ma_60"] = ordered["base_price"].rolling(window=60, min_periods=60).mean()
        ordered["avg_volume_20d"] = ordered["volume_clean"].rolling(window=20, min_periods=20).mean()
        latest = ordered.iloc[-1]
        close = _safe_float(latest.get("base_price"))
        ma_20 = _safe_float(latest.get("ma_20"))
        ma_60 = _safe_float(latest.get("ma_60"))
        output[str(ticker).upper()] = {
            "trade_date": latest.get("trade_date"),
            "close": close,
            "ret_20d": _safe_float(latest.get("ret_20d")),
            "ret_60d": _safe_float(latest.get("ret_60d")),
            "ret_120d": _safe_float(latest.get("ret_120d")),
            "volatility_20d": _safe_float(latest.get("volatility_20d")),
            "volatility_60d": _safe_float(latest.get("volatility_60d")),
            "ma_20": ma_20,
            "ma_60": ma_60,
            "price_above_ma20_flag": None if close is None or ma_20 is None else ("Y" if close > ma_20 else "N"),
            "price_above_ma60_flag": None if close is None or ma_60 is None else ("Y" if close > ma_60 else "N"),
            "avg_volume_20d": _safe_float(latest.get("avg_volume_20d")),
            "price_row_count": int(len(ordered)),
        }
    return output


def _derive_feature_quality_score(*, daily_feature: dict[str, object] | None, relative_strength: dict[str, object] | None, financial: dict[str, object] | None, price_snapshot: dict[str, object] | None, universe_row: dict[str, object]) -> float:
    meta_quality = _safe_float(universe_row.get("feature_quality_score"))
    if meta_quality is not None:
        return _cap(meta_quality, lower=0.0, upper=100.0)

    score = 0.0
    if daily_feature or price_snapshot:
        score += 40.0
    if relative_strength:
        score += 30.0
    elif price_snapshot and price_snapshot.get("ret_20d") is not None and price_snapshot.get("ret_60d") is not None:
        score += 30.0
    if financial:
        score += 30.0
    return _cap(score, lower=0.0, upper=100.0)


def _build_joined_row(
    *,
    symbol: str,
    universe_row: dict[str, object],
    daily_feature: dict[str, object] | None,
    relative_strength: dict[str, object] | None,
    financial: dict[str, object] | None,
    price_snapshot: dict[str, object] | None,
    benchmark_snapshot: dict[str, dict[str, object]],
    macro_snapshot: dict[str, object] | None,
) -> dict[str, object]:
    price = price_snapshot or {}
    row: dict[str, object] = {
        "symbol": symbol,
        "company_name": universe_row.get("company_name"),
        "sector": universe_row.get("sector"),
        "industry": universe_row.get("industry"),
        "universe_group": universe_row.get("universe_group"),
        "is_active": bool(universe_row.get("is_active")) if universe_row.get("is_active") is not None else False,
        "is_etf": bool(universe_row.get("is_etf")) if universe_row.get("is_etf") is not None else False,
        "is_leveraged": bool(universe_row.get("is_leveraged")) if universe_row.get("is_leveraged") is not None else False,
        "is_inverse": bool(universe_row.get("is_inverse")) if universe_row.get("is_inverse") is not None else False,
        "exclude_reason": universe_row.get("exclude_reason"),
        "close": price.get("close"),
        "price_trade_date": price.get("trade_date"),
        "price_row_count": price.get("price_row_count"),
    }

    row["ret_20d"] = _safe_float((daily_feature or {}).get("ret_20d")) if daily_feature else None
    row["ret_60d"] = _safe_float((daily_feature or {}).get("ret_60d")) if daily_feature else None
    row["volatility_20d"] = _safe_float((daily_feature or {}).get("volatility_20d")) if daily_feature else None
    row["ma_20"] = _safe_float((daily_feature or {}).get("ma_20")) if daily_feature else None
    row["ma_60"] = _safe_float((daily_feature or {}).get("ma_60")) if daily_feature else None
    row["price_above_ma20_flag"] = (daily_feature or {}).get("price_above_ma20_flag") if daily_feature else None
    row["price_above_ma60_flag"] = (daily_feature or {}).get("price_above_ma60_flag") if daily_feature else None
    row["avg_volume"] = _safe_float((daily_feature or {}).get("volume_avg_20d")) if daily_feature else None

    for field in ["ret_20d", "ret_60d", "volatility_20d", "ma_20", "ma_60", "price_above_ma20_flag", "price_above_ma60_flag", "avg_volume"]:
        if row.get(field) is None:
            row[field] = price.get(field if field != "avg_volume" else "avg_volume_20d")

    row["ret_120d"] = _safe_float((relative_strength or {}).get("ret_120d")) if relative_strength else None
    if row["ret_120d"] is None:
        row["ret_120d"] = price.get("ret_120d")
    row["volatility_60d"] = price.get("volatility_60d")

    for metric in ["rs_spy_20d", "rs_spy_60d", "rs_qqq_20d", "rs_qqq_60d"]:
        row[metric] = _safe_float((relative_strength or {}).get(metric)) if relative_strength else None

    spy = benchmark_snapshot.get("SPY", {})
    qqq = benchmark_snapshot.get("QQQ", {})
    if row["rs_spy_20d"] is None and row.get("ret_20d") is not None and spy.get("ret_20d") is not None:
        row["rs_spy_20d"] = float(row["ret_20d"]) - float(spy["ret_20d"])
    if row["rs_spy_60d"] is None and row.get("ret_60d") is not None and spy.get("ret_60d") is not None:
        row["rs_spy_60d"] = float(row["ret_60d"]) - float(spy["ret_60d"])
    if row["rs_qqq_20d"] is None and row.get("ret_20d") is not None and qqq.get("ret_20d") is not None:
        row["rs_qqq_20d"] = float(row["ret_20d"]) - float(qqq["ret_20d"])
    if row["rs_qqq_60d"] is None and row.get("ret_60d") is not None and qqq.get("ret_60d") is not None:
        row["rs_qqq_60d"] = float(row["ret_60d"]) - float(qqq["ret_60d"])

    # Technical features — new signals
    row["rsi_14"] = _safe_float((daily_feature or {}).get("rsi_14")) if daily_feature else None
    row["high_52w_ratio"] = _safe_float((daily_feature or {}).get("high_52w_ratio")) if daily_feature else None
    row["bb_position"] = _safe_float((daily_feature or {}).get("bb_position")) if daily_feature else None
    row["price_vs_ma200"] = _safe_float((daily_feature or {}).get("price_vs_ma200")) if daily_feature else None
    row["ret_252d"] = _safe_float((daily_feature or {}).get("ret_252d")) if daily_feature else None
    row["sector_rank_pct"] = _safe_float((daily_feature or {}).get("sector_rank_pct")) if daily_feature else None
    row["volume_ratio_20d"] = _safe_float((daily_feature or {}).get("volume_ratio_20d")) if daily_feature else None

    row["roe"] = _normalize_ratio((financial or {}).get("roe")) if financial else None
    row["operating_margin"] = _normalize_ratio((financial or {}).get("operating_margin")) if financial else None
    row["profit_margin"] = _normalize_ratio((financial or {}).get("net_margin")) if financial else None
    row["debt_to_equity"] = _normalize_debt_to_equity((financial or {}).get("debt_to_equity")) if financial else None
    rev_growth_yoy = (financial or {}).get("revenue_growth_yoy") if financial else None
    rev_growth_qoq = (financial or {}).get("revenue_growth_qoq") if financial else None
    # Quarterly records often lack YoY (needs 4 prior quarters); fall back to QoQ
    row["revenue_growth"] = _normalize_ratio(rev_growth_yoy if rev_growth_yoy is not None else rev_growth_qoq) if financial else None
    earnings_growth = (financial or {}).get("net_income_growth_yoy") if financial else None
    if earnings_growth is None and financial:
        earnings_growth = financial.get("eps_growth_yoy")
    if earnings_growth is None and financial:
        earnings_growth = financial.get("net_income_growth_qoq") or financial.get("eps_growth_qoq")
    row["earnings_growth"] = _normalize_ratio(earnings_growth) if financial else None
    row["trailing_pe"] = _safe_float((financial or {}).get("per")) if financial else None
    row["forward_pe"] = _safe_float((financial or {}).get("forward_pe")) if financial else None
    row["price_to_book"] = _safe_float((financial or {}).get("pbr")) if financial else None
    row["market_cap"] = _safe_float((financial or {}).get("market_cap")) if financial else _safe_float(universe_row.get("market_cap"))

    # Fundamental features — new signals
    row["fcf_yield"] = _safe_float((financial or {}).get("fcf_yield")) if financial else None
    row["roic_approx"] = _safe_float((financial or {}).get("roic_approx")) if financial else None
    row["analyst_recommendation"] = _safe_float((financial or {}).get("analyst_recommendation")) if financial else None
    row["analyst_target_upside"] = _safe_float((financial or {}).get("analyst_target_upside")) if financial else None

    # Macro context — market regime and VIX
    row["market_regime"] = str((macro_snapshot or {}).get("market_regime") or "neutral").lower()
    row["vix_close"] = _safe_float((macro_snapshot or {}).get("vix_close")) if macro_snapshot else None

    row["feature_quality_score"] = _derive_feature_quality_score(
        daily_feature=daily_feature,
        relative_strength=relative_strength,
        financial=financial,
        price_snapshot=price_snapshot,
        universe_row=universe_row,
    )
    return row


def calculate_momentum_score(row: dict[str, object]) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    missing_fields: list[str] = []
    inputs = {
        "return_20d": _safe_float(row.get("ret_20d")),
        "return_60d": _safe_float(row.get("ret_60d")),
        "return_120d": _safe_float(row.get("ret_120d")),
        "return_252d": _safe_float(row.get("ret_252d")),
        "close": _safe_float(row.get("close")),
        "ma20": _safe_float(row.get("ma_20")),
        "ma60": _safe_float(row.get("ma_60")),
        "rsi_14": _safe_float(row.get("rsi_14")),
        "high_52w_ratio": _safe_float(row.get("high_52w_ratio")),
        "price_vs_ma200": _safe_float(row.get("price_vs_ma200")),
        "sector_rank_pct": _safe_float(row.get("sector_rank_pct")),
    }

    for key in ["return_20d", "return_60d", "return_120d", "close", "ma20", "ma60"]:
        if inputs[key] is None:
            missing_fields.append(key)

    if inputs["return_20d"] is not None and inputs["return_20d"] > 0:
        score += 5
        reasons.append("20d return positive")
    if inputs["return_60d"] is not None and inputs["return_60d"] > 0:
        score += 5
        reasons.append("60d return positive")
    if inputs["return_120d"] is not None and inputs["return_120d"] > 0:
        score += 5
        reasons.append("120d return positive")

    above_ma20 = _bool_flag(row.get("price_above_ma20_flag"))
    if above_ma20 is None and inputs["close"] is not None and inputs["ma20"] is not None:
        above_ma20 = bool(inputs["close"] > inputs["ma20"])
    if above_ma20:
        score += 3
        reasons.append("price above 20d moving average")

    above_ma60 = _bool_flag(row.get("price_above_ma60_flag"))
    if above_ma60 is None and inputs["close"] is not None and inputs["ma60"] is not None:
        above_ma60 = bool(inputs["close"] > inputs["ma60"])
    if above_ma60:
        score += 4
        reasons.append("price above 60d moving average")

    if inputs["ma20"] is not None and inputs["ma60"] is not None and inputs["ma20"] > inputs["ma60"]:
        score += 3
        reasons.append("20d moving average above 60d moving average")

    # Extended signals — additional paths to reach max_score
    if inputs["rsi_14"] is not None and 40.0 <= inputs["rsi_14"] <= 75.0:
        score += 2
        reasons.append("RSI in healthy momentum zone 40-75")
    if inputs["high_52w_ratio"] is not None and inputs["high_52w_ratio"] >= 0.85:
        score += 2
        reasons.append("price near 52-week high (>=85%)")
    if inputs["price_vs_ma200"] is not None and inputs["price_vs_ma200"] > 1.0:
        score += 2
        reasons.append("price above 200d moving average")
    if inputs["sector_rank_pct"] is not None and inputs["sector_rank_pct"] >= 0.60:
        score += 2
        reasons.append("top 40% within sector by 20d momentum")
    if inputs["return_252d"] is not None and inputs["return_252d"] > 0:
        score += 2
        reasons.append("1-year return positive")

    detail = {
        "score": float(min(score, 25.0)),
        "max_score": 25,
        "inputs": inputs,
        "missing_fields": missing_fields,
        "reasons": reasons,
    }
    return float(min(score, 25.0)), detail


def calculate_relative_strength_score(row: dict[str, object]) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    missing_fields: list[str] = []
    inputs = {
        "spy_relative_return_20d": _safe_float(row.get("rs_spy_20d")),
        "spy_relative_return_60d": _safe_float(row.get("rs_spy_60d")),
        "qqq_relative_return_20d": _safe_float(row.get("rs_qqq_20d")),
        "qqq_relative_return_60d": _safe_float(row.get("rs_qqq_60d")),
    }
    for key, value in inputs.items():
        if value is None:
            missing_fields.append(key)
        elif value > 0:
            score += 5

    if inputs["spy_relative_return_20d"] is not None and inputs["spy_relative_return_20d"] > 0:
        reasons.append("SPY 20d relative strength positive")
    if inputs["spy_relative_return_60d"] is not None and inputs["spy_relative_return_60d"] > 0:
        reasons.append("SPY 60d relative strength positive")
    if inputs["qqq_relative_return_20d"] is not None and inputs["qqq_relative_return_20d"] > 0:
        reasons.append("QQQ 20d relative strength positive")
    if inputs["qqq_relative_return_60d"] is not None and inputs["qqq_relative_return_60d"] > 0:
        reasons.append("QQQ 60d relative strength positive")

    detail = {
        "score": float(score),
        "max_score": 20,
        "benchmark_basis": "SPY and QQQ",
        "inputs": inputs,
        "missing_fields": missing_fields,
        "reasons": reasons,
    }
    return float(score), detail


def calculate_fundamental_score(row: dict[str, object]) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    missing_fields: list[str] = []
    inputs = {
        "return_on_equity": _safe_float(row.get("roe")),
        "operating_margin": _safe_float(row.get("operating_margin")),
        "profit_margin": _safe_float(row.get("profit_margin")),
        "debt_to_equity": _safe_float(row.get("debt_to_equity")),
        "feature_quality_score": _safe_float(row.get("feature_quality_score")),
        "fcf_yield": _safe_float(row.get("fcf_yield")),
        "roic_approx": _safe_float(row.get("roic_approx")),
        "analyst_recommendation": _safe_float(row.get("analyst_recommendation")),
        "analyst_target_upside": _safe_float(row.get("analyst_target_upside")),
    }
    for key in ["return_on_equity", "operating_margin", "profit_margin", "debt_to_equity", "feature_quality_score"]:
        if inputs[key] is None:
            missing_fields.append(key)

    if inputs["return_on_equity"] is not None and inputs["return_on_equity"] > 0.15:
        score += 5
        reasons.append("ROE above 15%")
    if inputs["operating_margin"] is not None and inputs["operating_margin"] > 0.15:
        score += 5
        reasons.append("operating margin above 15%")
    if inputs["profit_margin"] is not None and inputs["profit_margin"] > 0.10:
        score += 4
        reasons.append("profit margin above 10%")

    debt_to_equity = inputs["debt_to_equity"]
    if debt_to_equity is not None:
        if debt_to_equity <= 0.5:
            score += 3
            reasons.append("debt to equity at or below 0.5")
        elif debt_to_equity <= 1.0:
            score += 2
            reasons.append("debt to equity at or below 1.0")
        elif debt_to_equity <= 2.0:
            score += 1
            reasons.append("debt to equity at or below 2.0")

    if inputs["feature_quality_score"] is not None and inputs["feature_quality_score"] >= 60:
        score += 3
        reasons.append("feature quality at or above 60")

    # Extended signals — quality depth
    if inputs["fcf_yield"] is not None and inputs["fcf_yield"] > 0:
        score += 2
        reasons.append("free cash flow yield positive")
    if inputs["roic_approx"] is not None and inputs["roic_approx"] > 0.15:
        score += 2
        reasons.append("ROIC above 15%")
    if inputs["analyst_recommendation"] is not None and inputs["analyst_recommendation"] <= 2.0:
        score += 2
        reasons.append("analyst consensus buy or strong buy")
    if inputs["analyst_target_upside"] is not None and inputs["analyst_target_upside"] > 0.10:
        score += 1
        reasons.append("analyst target upside above 10%")

    detail = {
        "score": float(min(score, 20.0)),
        "max_score": 20,
        "inputs": inputs,
        "missing_fields": missing_fields,
        "reasons": reasons,
        "notes": ["debt_to_equity values above 10 are normalized as percentages"],
    }
    return float(min(score, 20.0)), detail


def calculate_growth_score(row: dict[str, object]) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    missing_fields: list[str] = []
    revenue_growth = _safe_float(row.get("revenue_growth"))
    earnings_growth = _safe_float(row.get("earnings_growth"))
    inputs = {
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
    }
    for key, value in inputs.items():
        if value is None:
            missing_fields.append(key)

    abnormal_flags: list[str] = []
    if revenue_growth is not None:
        if revenue_growth > 0.10:
            score += 5
            reasons.append("revenue growth above 10%")
        if revenue_growth > 0.20:
            score += 2
            reasons.append("revenue growth above 20%")
        if abs(revenue_growth) > 3.0:
            abnormal_flags.append("revenue_growth_extreme")

    if earnings_growth is not None:
        if earnings_growth > 0.10:
            score += 5
            reasons.append("earnings growth above 10%")
        if earnings_growth > 0.20:
            score += 2
            reasons.append("earnings growth above 20%")
        if abs(earnings_growth) > 3.0:
            abnormal_flags.append("earnings_growth_extreme")

    if revenue_growth is not None and earnings_growth is not None and revenue_growth > 0 and earnings_growth > 0:
        score += 1
        reasons.append("revenue and earnings growth both positive")

    detail = {
        "score": float(min(score, 15.0)),
        "max_score": 15,
        "inputs": inputs,
        "missing_fields": missing_fields,
        "reasons": reasons,
        "abnormal_flags": abnormal_flags,
    }
    return float(min(score, 15.0)), detail


def calculate_valuation_score(row: dict[str, object]) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    missing_fields: list[str] = []
    inputs = {
        "trailing_pe": _safe_float(row.get("trailing_pe")),
        "forward_pe": _safe_float(row.get("forward_pe")),
        "price_to_book": _safe_float(row.get("price_to_book")),
    }
    for key, value in inputs.items():
        if value is None:
            missing_fields.append(key)

    trailing_pe = inputs["trailing_pe"]
    if trailing_pe is not None and 0 < trailing_pe <= 20:
        score += 4
        reasons.append("trailing PE at or below 20")

    forward_pe = inputs["forward_pe"]
    if forward_pe is not None and 0 < forward_pe <= 25:
        score += 3
        reasons.append("forward PE at or below 25")

    price_to_book = inputs["price_to_book"]
    if price_to_book is not None and 0 < price_to_book <= 5:
        score += 3
        reasons.append("price to book at or below 5")

    detail = {
        "score": float(score),
        "max_score": 10,
        "inputs": inputs,
        "missing_fields": missing_fields,
        "reasons": reasons,
    }
    return float(score), detail


def calculate_risk_score(row: dict[str, object], *, cfg) -> tuple[float, dict[str, object]]:
    score = 0.0
    reasons: list[str] = []
    inputs = {
        "volatility_20d": _safe_float(row.get("volatility_20d")),
        "volatility_60d": _safe_float(row.get("volatility_60d")),
        "return_20d": _safe_float(row.get("ret_20d")),
        "feature_quality_score": _safe_float(row.get("feature_quality_score")),
        "price_row_count": row.get("price_row_count"),
        "rsi_14": _safe_float(row.get("rsi_14")),
        "bb_position": _safe_float(row.get("bb_position")),
        "market_regime": str(row.get("market_regime") or "neutral").lower(),
        "vix_close": _safe_float(row.get("vix_close")),
    }

    if inputs["volatility_20d"] is not None and inputs["volatility_20d"] > cfg.volatility_20d_threshold:
        score -= 3
        reasons.append("20d volatility above threshold")
    if inputs["volatility_60d"] is not None and inputs["volatility_60d"] > cfg.volatility_60d_threshold:
        score -= 3
        reasons.append("60d volatility above threshold")
    if inputs["return_20d"] is not None and inputs["return_20d"] > cfg.return_20d_overheat_threshold:
        score -= 2
        reasons.append("20d return overheat threshold exceeded")
    if inputs["feature_quality_score"] is not None and inputs["feature_quality_score"] < cfg.min_feature_quality_score:
        score -= 2
        reasons.append("feature quality below minimum threshold")
    if int(inputs["price_row_count"] or 0) < PRICE_LOOKBACK_MIN_ROWS:
        score -= 5
        reasons.append("insufficient price history")

    # Technical overbought penalties
    if inputs["rsi_14"] is not None and inputs["rsi_14"] > 82.0:
        score -= 2
        reasons.append("RSI overbought above 82")
    if inputs["bb_position"] is not None and inputs["bb_position"] > 0.95:
        score -= 1
        reasons.append("price at Bollinger Band upper extreme")

    # Macro regime penalties
    if inputs["market_regime"] == "bear":
        score -= 2
        reasons.append("market regime is bear (SPY below MA200 and declining)")
    if inputs["vix_close"] is not None and inputs["vix_close"] > 35.0:
        score -= 1
        reasons.append("VIX elevated above 35 (extreme fear)")

    score = _cap(score, lower=-10.0, upper=0.0)
    detail = {
        "score": float(score),
        "min_score": -10,
        "inputs": inputs,
        "reasons": reasons,
        "thresholds": {
            "volatility_20d": cfg.volatility_20d_threshold,
            "volatility_60d": cfg.volatility_60d_threshold,
            "return_20d_overheat": cfg.return_20d_overheat_threshold,
            "feature_quality_min": cfg.min_feature_quality_score,
            "rsi_overbought": 82.0,
            "bb_extreme": 0.95,
            "vix_extreme": 35.0,
        },
    }
    return float(score), detail


def _resolve_grade(total_score: float, *, exclude: bool, cfg) -> str:
    if exclude:
        return "EXCLUDE"
    if total_score >= cfg.strong_buy_score:
        return "STRONG_BUY"
    if total_score >= cfg.buy_score:
        return "BUY"
    if total_score >= cfg.watch_score:
        return "WATCH"
    if total_score >= cfg.hold_score:
        return "HOLD"
    return "EXCLUDE"


def _has_missing(detail: dict[str, object]) -> bool:
    missing_fields = detail.get("missing_fields") if isinstance(detail, dict) else None
    return bool(missing_fields)


def _derive_data_status(
    *,
    row: dict[str, object],
    trade_date: date,
    cfg,
) -> tuple[str, str | None, dict[str, bool]]:
    is_active = bool(row.get("is_active"))
    is_etf = bool(row.get("is_etf"))
    quality_score = _safe_float(row.get("feature_quality_score")) or 0.0
    exact_price_date = row.get("price_trade_date") == trade_date
    base_exclude_reason = str(row.get("exclude_reason") or "").strip() or None

    flags = {
        "missing_price_feature": any(
            row.get(field) is None
            for field in ["close", "ret_20d", "ret_60d", "ma_20", "ma_60", "volatility_20d"]
        )
        or not exact_price_date,
        "missing_relative_strength": all(
            row.get(field) is None for field in ["rs_spy_20d", "rs_spy_60d", "rs_qqq_20d", "rs_qqq_60d"]
        ),
        "missing_fundamental": (not is_etf)
        and all(
            row.get(field) is None
            for field in ["roe", "operating_margin", "profit_margin", "debt_to_equity", "revenue_growth", "earnings_growth"]
        ),
        "low_feature_quality": quality_score < cfg.min_feature_quality_score,
        "insufficient_price_history": int(row.get("price_row_count") or 0) < PRICE_LOOKBACK_MIN_ROWS,
    }

    if not is_active:
        return "EXCLUDED", base_exclude_reason or "Universe row is inactive.", flags
    if bool(row.get("is_leveraged")):
        return "EXCLUDED", base_exclude_reason or "Leveraged ETF excluded from ranking.", flags
    if bool(row.get("is_inverse")):
        return "EXCLUDED", base_exclude_reason or "Inverse ETF excluded from ranking.", flags
    if flags["missing_price_feature"]:
        return "MISSING_PRICE_FEATURE", "Price-derived features are missing for the effective trade date.", flags
    if flags["low_feature_quality"] and (not is_etf or cfg.apply_fundamental_quality_to_etf):
        return "LOW_FEATURE_QUALITY", f"Feature quality score below minimum threshold {cfg.min_feature_quality_score:.0f}.", flags
    if flags["insufficient_price_history"]:
        return "PARTIAL_DATA", "Insufficient price history for stable scoring.", flags

    missing_optional_layers = int(flags["missing_relative_strength"]) + int(flags["missing_fundamental"])
    if missing_optional_layers >= 2:
        return "PARTIAL_DATA", None, flags
    if flags["missing_fundamental"]:
        return "MISSING_FUNDAMENTAL", None, flags
    if flags["missing_relative_strength"]:
        return "MISSING_RELATIVE_STRENGTH", None, flags
    return "OK", None, flags


def _build_reason_tags(
    *,
    row: dict[str, object],
    total_score: float,
    component_scores: dict[str, float],
    data_status: str,
    detail_sections: dict[str, dict[str, object]],
) -> list[str]:
    tags: list[str] = []
    if bool(row.get("is_etf")):
        tags.append("etf")
    if bool(row.get("is_leveraged")):
        tags.append("leveraged_etf_excluded")
    if bool(row.get("is_inverse")):
        tags.append("inverse_etf_excluded")

    if component_scores["momentum_score"] >= 18:
        tags.append("strong_momentum")
    elif component_scores["momentum_score"] <= 5:
        tags.append("weak_momentum")

    if _safe_float(row.get("rs_spy_20d")) is not None and _safe_float(row.get("rs_spy_20d")) > 0:
        tags.append("spy_outperform")
    if _safe_float(row.get("rs_qqq_20d")) is not None and _safe_float(row.get("rs_qqq_20d")) > 0:
        tags.append("qqq_outperform")
    if component_scores["relative_strength_score"] <= 5 and data_status in {"MISSING_RELATIVE_STRENGTH", "PARTIAL_DATA"}:
        tags.append("missing_relative_strength")

    if _safe_float(row.get("roe")) is not None and float(row["roe"]) >= 0.15:
        tags.append("high_roe")
    if _safe_float(row.get("operating_margin")) is not None and float(row["operating_margin"]) >= 0.15:
        tags.append("high_margin")
    if component_scores["fundamental_score"] <= 3 and data_status in {"MISSING_FUNDAMENTAL", "PARTIAL_DATA"}:
        tags.append("missing_fundamental")

    if (_safe_float(row.get("revenue_growth")) or 0) > 0.10 or (_safe_float(row.get("earnings_growth")) or 0) > 0.10:
        tags.append("positive_growth")

    trailing_pe = _safe_float(row.get("trailing_pe"))
    price_to_book = _safe_float(row.get("price_to_book"))
    if component_scores["valuation_score"] >= 6:
        tags.append("cheap_valuation")
    elif (trailing_pe is not None and trailing_pe > 25) or (price_to_book is not None and price_to_book > 5) or component_scores["valuation_score"] <= 2:
        tags.append("expensive_valuation")

    volatility_20d = _safe_float(row.get("volatility_20d"))
    if volatility_20d is not None and volatility_20d <= 0.025 and component_scores["risk_score"] >= -1:
        tags.append("low_volatility")
    if volatility_20d is not None and volatility_20d > 0.05:
        tags.append("high_volatility")

    if (_safe_float(row.get("feature_quality_score")) or 0.0) < 60:
        tags.append("low_quality_data")

    if total_score >= 95:
        tags.append("score_outlier_high")
    if component_scores["risk_score"] <= -8:
        tags.append("risk_extreme")

    market_regime = str(row.get("market_regime") or "neutral").lower()
    if market_regime == "bear":
        tags.append("bear_market_regime")
    elif market_regime == "bull":
        tags.append("bull_market_regime")
    vix = _safe_float(row.get("vix_close"))
    if vix is not None and vix > 35.0:
        tags.append("vix_extreme")

    for section_key in ["momentum", "relative_strength", "fundamental", "growth", "valuation"]:
        section = detail_sections.get(section_key) or {}
        if _has_missing(section):
            tags.append(f"{section_key}_partial")

    return list(dict.fromkeys(tags))


def _derive_reason_category(
    *,
    row: dict[str, object],
    recommend_grade: str,
    data_status: str,
    component_scores: dict[str, float],
    reason_tags: list[str],
) -> str:
    if recommend_grade == "EXCLUDE":
        if data_status in {"MISSING_PRICE_FEATURE", "MISSING_FUNDAMENTAL", "MISSING_RELATIVE_STRENGTH", "LOW_FEATURE_QUALITY", "PARTIAL_DATA", "ERROR"}:
            return "DATA_INSUFFICIENT"
        if component_scores["risk_score"] <= -7:
            return "RISK_HIGH"
        return "EXCLUDED"
    if bool(row.get("is_etf")):
        return "ETF_CORE"
    if recommend_grade in {"WATCH", "HOLD"}:
        return "WATCHLIST"
    if component_scores["momentum_score"] >= 18 and component_scores["relative_strength_score"] >= 10:
        return "MOMENTUM_LEADER"
    if component_scores["relative_strength_score"] >= 15:
        return "RELATIVE_STRENGTH_LEADER"
    if component_scores["fundamental_score"] >= 14 and component_scores["risk_score"] >= -2:
        return "QUALITY_COMPOUNDER"
    if component_scores["growth_score"] >= 10:
        return "GROWTH_LEADER"
    if component_scores["valuation_score"] >= 6:
        return "VALUE_CANDIDATE"
    if component_scores["risk_score"] <= -7 or "high_volatility" in reason_tags:
        return "RISK_HIGH"
    return "WATCHLIST"


def _describe_grade_rationale(recommend_grade: str, *, total_score: float, cfg, forced_exclude_reason: str | None) -> str:
    if recommend_grade == "STRONG_BUY":
        return f"Total score cleared the STRONG_BUY cutoff ({cfg.strong_buy_score:.0f})."
    if recommend_grade == "BUY":
        return f"Total score cleared the BUY cutoff ({cfg.buy_score:.0f}) but stayed below STRONG_BUY."
    if recommend_grade == "WATCH":
        return f"Total score cleared the WATCH cutoff ({cfg.watch_score:.0f}) but stayed below BUY."
    if recommend_grade == "HOLD":
        return f"Total score cleared the HOLD cutoff ({cfg.hold_score:.0f}) but stayed below WATCH."
    if forced_exclude_reason:
        return f"Recommendation was excluded because {forced_exclude_reason.rstrip('.') .lower()}."
    return f"Total score {total_score:.1f} stayed below the HOLD cutoff ({cfg.hold_score:.0f})."


def _component_labels(score_map: dict[str, float]) -> list[str]:
    labels = {
        "momentum_score": "momentum",
        "relative_strength_score": "relative strength",
        "fundamental_score": "fundamental quality",
        "growth_score": "growth",
        "valuation_score": "valuation",
    }
    ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    return [labels[name] for name, value in ranked if value > 0]


def build_reason_summary(result_row: dict[str, object]) -> str:
    detail = result_row.get("_detail") or {}
    meta = detail.get("meta") or {}
    reason_tags = list(meta.get("reason_tags") or [])
    recommend_grade = str(result_row.get("recommend_grade") or "")
    is_etf = bool(result_row.get("is_etf"))
    data_status = str(result_row.get("data_status") or meta.get("data_status") or "OK")
    total_score = float(result_row.get("total_score") or 0.0)

    strengths: list[str] = []
    if "strong_momentum" in reason_tags:
        strengths.append("strong momentum")
    if "qqq_outperform" in reason_tags:
        strengths.append("strong QQQ-relative strength")
    elif "spy_outperform" in reason_tags:
        strengths.append("solid SPY-relative strength")
    if "high_roe" in reason_tags:
        strengths.append("strong profitability quality")
    if "high_margin" in reason_tags:
        strengths.append("healthy operating margin")
    if "positive_growth" in reason_tags:
        strengths.append("positive growth")
    if "cheap_valuation" in reason_tags:
        strengths.append("supportive valuation")
    if is_etf and not strengths:
        strengths.append("ETF market exposure metrics remained investable")

    weaknesses: list[str] = []
    if "expensive_valuation" in reason_tags:
        weaknesses.append("valuation remains demanding")
    if "high_volatility" in reason_tags:
        weaknesses.append("short-term volatility is elevated")
    if "low_quality_data" in reason_tags:
        weaknesses.append("feature quality is limited")
    if "missing_fundamental" in reason_tags and not is_etf:
        weaknesses.append("fundamental coverage is incomplete")
    if "missing_relative_strength" in reason_tags:
        weaknesses.append("relative-strength coverage is incomplete")

    if is_etf:
        lead = " and ".join(strengths[:2]) if strengths else "ETF-relative signals stayed investable"
        message_1 = f"As an ETF, company-level fundamentals were only partially applied, and {lead}."
    else:
        lead = " and ".join(strengths[:2]) if strengths else "clear strengths were limited"
        if lead == "clear strengths were limited":
            message_1 = "Clear strengths were limited."
        else:
            message_1 = f"{lead.capitalize()} supported the ranking."

    message_2_parts: list[str] = []
    if weaknesses:
        message_2_parts.append(f"However, {weaknesses[0]}")
    if data_status not in {"OK", "EXCLUDED"}:
        status_text = data_status.replace("_", " ").lower()
        connector = "Data status is" if not message_2_parts else "Data status remains"
        message_2_parts.append(f"{connector} {status_text}")
    grade_rationale = str(meta.get("grade_rationale") or "").strip()
    if grade_rationale:
        if message_2_parts:
            message_2 = " ".join(part.rstrip(".") + "." for part in message_2_parts)
            message_2 = f"{message_2} {grade_rationale}"
        else:
            message_2 = grade_rationale
    else:
        message_2 = ""

    if recommend_grade == "EXCLUDE" and not message_2:
        message_2 = f"총점 {total_score:.1f}로 추천 등급은 EXCLUDE입니다."
    elif not message_2:
        message_2 = f"총점 {total_score:.1f}로 {recommend_grade} 등급으로 분류되었습니다."

    return " ".join(part.strip() for part in [message_1.strip(), message_2.strip()] if part.strip())


def calculate_total_score(row: dict[str, object], *, trade_date: date, cfg) -> dict[str, object]:
    momentum_score, momentum_detail = calculate_momentum_score(row)
    relative_strength_score, rs_detail = calculate_relative_strength_score(row)
    fundamental_score, fundamental_detail = calculate_fundamental_score(row)
    growth_score, growth_detail = calculate_growth_score(row)
    valuation_score, valuation_detail = calculate_valuation_score(row)
    risk_score, risk_detail = calculate_risk_score(row, cfg=cfg)

    raw_total = momentum_score + relative_strength_score + fundamental_score + growth_score + valuation_score + risk_score
    total_score = _cap(raw_total, lower=0.0, upper=100.0)

    quality_score = _safe_float(row.get("feature_quality_score")) or 0.0
    data_status, forced_exclude_reason, data_flags = _derive_data_status(row=row, trade_date=trade_date, cfg=cfg)
    force_exclude = forced_exclude_reason is not None and data_status in {"EXCLUDED", "MISSING_PRICE_FEATURE", "LOW_FEATURE_QUALITY"}

    score_detail = {
        "meta": {
            "symbol": row["symbol"],
            "trade_date": trade_date,
            "raw_total_score": float(raw_total),
            "capped_total_score": float(total_score),
            "feature_quality_score": float(quality_score),
            "data_status": data_status,
            "exclude_reason": forced_exclude_reason,
        },
        "momentum": momentum_detail,
        "relative_strength": rs_detail,
        "fundamental": fundamental_detail,
        "growth": growth_detail,
        "valuation": valuation_detail,
        "risk": risk_detail,
    }

    recommend_grade = _resolve_grade(total_score, exclude=force_exclude, cfg=cfg)
    if recommend_grade == "EXCLUDE" and forced_exclude_reason is None:
        forced_exclude_reason = f"Total score below HOLD threshold {cfg.hold_score:.0f}."
    component_scores = {
        "momentum_score": float(momentum_score),
        "relative_strength_score": float(relative_strength_score),
        "fundamental_score": float(fundamental_score),
        "growth_score": float(growth_score),
        "valuation_score": float(valuation_score),
        "risk_score": float(risk_score),
    }
    reason_tags = _build_reason_tags(
        row=row,
        total_score=float(total_score),
        component_scores=component_scores,
        data_status=data_status,
        detail_sections={
            "momentum": momentum_detail,
            "relative_strength": rs_detail,
            "fundamental": fundamental_detail,
            "growth": growth_detail,
            "valuation": valuation_detail,
        },
    )
    reason_category = _derive_reason_category(
        row=row,
        recommend_grade=recommend_grade,
        data_status=data_status,
        component_scores=component_scores,
        reason_tags=reason_tags,
    )
    score_detail["meta"].update(
        {
            "reason_category": reason_category,
            "reason_tags": reason_tags,
            "grade_rationale": _describe_grade_rationale(
                recommend_grade,
                total_score=float(total_score),
                cfg=cfg,
                forced_exclude_reason=forced_exclude_reason,
            ),
            "data_flags": data_flags,
        }
    )
    result = {
        "trade_date": trade_date,
        "symbol": row["symbol"],
        "rank_no": None,
        "recommend_grade": recommend_grade,
        "total_score": round(float(total_score), 4),
        "momentum_score": round(float(momentum_score), 4),
        "relative_strength_score": round(float(relative_strength_score), 4),
        "fundamental_score": round(float(fundamental_score), 4),
        "growth_score": round(float(growth_score), 4),
        "valuation_score": round(float(valuation_score), 4),
        "risk_score": round(float(risk_score), 4),
        "feature_quality_score": round(float(quality_score), 4),
        "universe_group": row.get("universe_group"),
        "company_name": row.get("company_name"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "market_cap": row.get("market_cap"),
        "avg_volume": row.get("avg_volume"),
        "is_etf": row.get("is_etf"),
        "is_active": row.get("is_active"),
        "data_status": data_status,
        "exclude_reason": forced_exclude_reason,
        "reason_summary": None,
        "score_detail_json": json.dumps(score_detail, ensure_ascii=True, default=_json_default, sort_keys=True),
        "_detail": score_detail,
    }
    result["reason_summary"] = build_reason_summary(result)
    return result


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda item: (
            -float(item["total_score"]),
            -float(item["momentum_score"]),
            -float(item["relative_strength_score"]),
            -float(item["fundamental_score"]),
            str(item["symbol"]),
        ),
    )
    for idx, row in enumerate(ordered, start=1):
        row["rank_no"] = idx
    return ordered


def _log_missing_summary(rows: list[dict[str, object]]) -> None:
    counter: Counter[str] = Counter()
    for row in rows:
        detail = row.get("_detail") or {}
        for section_name in ["momentum", "relative_strength", "fundamental", "growth", "valuation"]:
            section = detail.get(section_name) or {}
            for field in section.get("missing_fields") or []:
                counter[f"{section_name}.{field}"] += 1
    if counter:
        top_items = ", ".join(f"{name}={count}" for name, count in counter.most_common(10))
        LOGGER.info("[US_RULE_RANK] missing_field_summary=%s", top_items)


def _resolve_effective_trade_date(price_snapshots: dict[str, dict[str, object]], *, requested_trade_date: date) -> date:
    dates = [snapshot.get("trade_date") for snapshot in price_snapshots.values() if snapshot.get("trade_date") is not None]
    if not dates:
        return requested_trade_date
    return max(dates)


def _fetch_join_inputs(*, trade_date: date, symbols: list[str]) -> tuple[date, dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object] | None]:
    history_rows = fetch_price_history_for_tickers(sorted(set(symbols) | set(BENCHMARKS)), end_date=trade_date)
    price_snapshots = _compute_history_snapshot(history_rows)
    effective_trade_date = _resolve_effective_trade_date(price_snapshots, requested_trade_date=trade_date)
    daily_features = fetch_latest_daily_feature_snapshots(symbols, trade_date=effective_trade_date)
    relative_strength = fetch_latest_relative_strength_snapshots(symbols, trade_date=effective_trade_date)
    financial = fetch_latest_financial_feature_snapshots(symbols, trade_date=effective_trade_date)
    macro = fetch_latest_macro_snapshot(trade_date=effective_trade_date)
    return effective_trade_date, daily_features, relative_strength, financial, price_snapshots, macro


def calculate_rule_scores(
    *,
    trade_date: date,
    symbols: list[str] | None,
    dry_run: bool,
    top_n: int,
    source: str,
    cfg,
) -> RuleRankingRunResult:
    if not cfg.enabled:
        raise SystemExit("[US_RULE_RANK] US_RULE_RANKING_ENABLED=0. Scoring is disabled.")

    universe_rows = fetch_meta_us_universe_rows()
    active_universe = [row for row in universe_rows if str(row.get("symbol") or "").strip()]
    if symbols:
        symbol_set = {name.upper() for name in symbols}
        active_universe = [row for row in active_universe if str(row.get("symbol")).upper() in symbol_set]

    if not active_universe:
        LOGGER.info("[US_RULE_RANK] No universe rows selected.")
        return RuleRankingRunResult(trade_date, source, 0, 0, [], dry_run)

    symbol_list = [str(row["symbol"]).upper() for row in active_universe]
    effective_trade_date, daily_features, relative_strength, financial, price_snapshots, macro = _fetch_join_inputs(
        trade_date=trade_date,
        symbols=symbol_list,
    )
    if effective_trade_date != trade_date:
        LOGGER.info(
            "[US_RULE_RANK] requested_trade_date=%s effective_trade_date=%s",
            trade_date.isoformat(),
            effective_trade_date.isoformat(),
        )
    if macro:
        LOGGER.info(
            "[US_RULE_RANK] macro market_regime=%s vix=%.1f",
            macro.get("market_regime", "n/a"),
            float(macro.get("vix_close") or 0.0),
        )
    benchmark_snapshot = {name: price_snapshots.get(name, {}) for name in BENCHMARKS}

    scored_rows: list[dict[str, object]] = []
    for universe_row in active_universe:
        symbol = str(universe_row["symbol"]).upper()
        joined = _build_joined_row(
            symbol=symbol,
            universe_row=universe_row,
            daily_feature=daily_features.get(symbol),
            relative_strength=relative_strength.get(symbol),
            financial=financial.get(symbol),
            price_snapshot=price_snapshots.get(symbol),
            benchmark_snapshot=benchmark_snapshot,
            macro_snapshot=macro,
        )
        scored_rows.append(calculate_total_score(joined, trade_date=effective_trade_date, cfg=cfg))

    ranked_rows = _rank_rows(scored_rows)
    _log_missing_summary(ranked_rows)

    display_rows = ranked_rows[: max(top_n, 0)]
    LOGGER.info(
        "[US_RULE_RANK] trade_date=%s evaluated=%s write_mode=%s",
        trade_date.isoformat(),
        len(ranked_rows),
        "dry-run" if dry_run else "upsert",
    )
    for row in display_rows:
        LOGGER.info(
            "[US_RULE_RANK] rank=%s symbol=%s grade=%s total=%.2f momentum=%.2f rs=%.2f fundamental=%.2f growth=%.2f valuation=%.2f risk=%.2f status=%s",
            row["rank_no"],
            row["symbol"],
            row["recommend_grade"],
            float(row["total_score"]),
            float(row["momentum_score"]),
            float(row["relative_strength_score"]),
            float(row["fundamental_score"]),
            float(row["growth_score"]),
            float(row["valuation_score"]),
            float(row["risk_score"]),
            row["data_status"],
        )

    write_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in ranked_rows]
    written = 0
    if not dry_run:
        if symbols is None:
            delete_us_rank_rows(trade_date=effective_trade_date, source=source)
        for row in write_rows:
            row["source"] = source
        written = upsert_us_rank_rows(write_rows)

    return RuleRankingRunResult(
        trade_date=effective_trade_date,
        source=source,
        evaluated_symbol_count=len(ranked_rows),
        written_row_count=written,
        top_rows=display_rows,
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()
    cfg = load_us_rule_ranking_config()
    setup_logging(cfg.log_level)
    trade_date = parse_iso_date(args.trade_date, field_name="trade_date")
    if trade_date is None:
        raise SystemExit("trade_date is required.")

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_RULE_RANK] DB connection failed: {exc}") from exc

    result = calculate_rule_scores(
        trade_date=trade_date,
        symbols=_parse_symbols_arg(args.symbols),
        dry_run=bool(args.dry_run),
        top_n=int(args.top_n),
        source=str(args.source or cfg.source).strip() or cfg.source,
        cfg=cfg,
    )
    LOGGER.info(
        "[US_RULE_RANK] finished trade_date=%s evaluated=%s written=%s dry_run=%s",
        result.trade_date.isoformat(),
        result.evaluated_symbol_count,
        result.written_row_count,
        str(result.dry_run).lower(),
    )


if __name__ == "__main__":
    main()
