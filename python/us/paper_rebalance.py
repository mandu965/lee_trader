from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_paper_trading_config, load_us_paper_trading_profiles
from python.us.us_db import fetch_mixed_price_rows_for_tickers_between


MIN_ORDER_AMOUNT_DEFAULT = 100.0


@dataclass(frozen=True)
class PaperStrategyPolicy:
    selection_rule: str
    buy_grades: tuple[str, ...]
    sell_grades: tuple[str, ...]
    max_rank_no: int
    max_positions: int
    max_position_weight: float
    max_sector_weight: float
    min_cash_weight: float
    max_daily_new_buys: int
    allow_fractional_shares: bool
    min_order_amount: float
    source: str
    sell_first: bool
    allow_rebuy_same_day: bool
    min_rebalance_amount: float
    min_weight_diff: float
    full_sell_on_rank_exit: bool
    full_sell_on_grade_downgrade: bool
    rebalance_frequency: str


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def position_value(row: dict[str, object]) -> float:
    market_value = safe_float(row.get("market_value"))
    if market_value is not None:
        return max(0.0, market_value)
    qty = safe_float(row.get("qty")) or 0.0
    last_price = safe_float(row.get("last_price")) or safe_float(row.get("avg_price")) or 0.0
    return max(0.0, qty * last_price)


def sector_of(symbol: str, rank_map: dict[str, dict[str, object]]) -> str:
    sector = str(rank_map.get(symbol, {}).get("sector") or "").strip()
    return sector or "UNKNOWN"


def round_qty(qty: float, *, allow_fractional: bool) -> float:
    if allow_fractional:
        return round(max(0.0, qty), 6)
    return float(max(0, math.floor(qty)))


def load_policy(account_id: str) -> PaperStrategyPolicy:
    cfg = load_us_paper_trading_config(account_id=account_id)
    profiles = load_us_paper_trading_profiles()
    profile = profiles.get(account_id, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    strategy = profile.get("strategy", {}) if isinstance(profile.get("strategy"), dict) else {}
    rebalance = profile.get("rebalance", {}) if isinstance(profile.get("rebalance"), dict) else {}
    risk = profile.get("risk", {}) if isinstance(profile.get("risk"), dict) else {}
    selection_rule = str(strategy.get("selection_rule") or cfg.max_rank_no or "TOP20").strip().upper() or "TOP20"
    raw_buy_grades = str(strategy.get("buy_grades") or ",".join(cfg.buy_grades or ("STRONG_BUY", "BUY")))
    buy_grades = tuple(part.strip().upper() for part in raw_buy_grades.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",") if part.strip())
    raw_sell_grades = str(strategy.get("sell_grades") or ",".join(cfg.sell_grades or ("HOLD", "EXCLUDE")))
    sell_grades = tuple(part.strip().upper() for part in raw_sell_grades.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",") if part.strip())
    max_rank_no = int(strategy.get("max_rank_no") or cfg.max_rank_no or (selection_rule.replace("TOP", "") if selection_rule.startswith("TOP") and selection_rule[3:].isdigit() else 20))
    return PaperStrategyPolicy(
        selection_rule=selection_rule,
        buy_grades=buy_grades or ("STRONG_BUY", "BUY"),
        sell_grades=sell_grades or ("HOLD", "EXCLUDE"),
        max_rank_no=max_rank_no,
        max_positions=int(risk.get("max_positions", cfg.max_positions) or cfg.max_positions),
        max_position_weight=float(risk.get("max_position_weight", cfg.max_position_weight) or cfg.max_position_weight),
        max_sector_weight=float(risk.get("max_sector_weight", cfg.max_sector_weight) or cfg.max_sector_weight),
        min_cash_weight=float(risk.get("min_cash_weight", cfg.min_cash_weight) or cfg.min_cash_weight),
        max_daily_new_buys=int(risk.get("max_daily_new_buys", cfg.max_daily_new_buys) or cfg.max_daily_new_buys),
        allow_fractional_shares=bool(risk.get("allow_fractional_shares", cfg.allow_fractional_shares)),
        min_order_amount=float(os.environ.get("US_PAPER_MIN_ORDER_AMOUNT", profile.get("min_order_amount", MIN_ORDER_AMOUNT_DEFAULT) or MIN_ORDER_AMOUNT_DEFAULT)),
        source=str(strategy.get("source_rank_table") or "recommend.us_stock_rank_daily"),
        sell_first=bool(cfg.rebalance_sell_first),
        allow_rebuy_same_day=bool(cfg.rebalance_allow_rebuy_same_day),
        min_rebalance_amount=float(cfg.rebalance_min_amount),
        min_weight_diff=float(cfg.rebalance_min_weight_diff),
        full_sell_on_rank_exit=bool(cfg.rebalance_full_sell_on_rank_exit),
        full_sell_on_grade_downgrade=bool(cfg.rebalance_full_sell_on_grade_downgrade),
        rebalance_frequency=str(cfg.rebalance_frequency).upper() or "DAILY",
    )


def order_price_lookup(*, symbols: list[str], trade_date: date) -> dict[str, float]:
    rows = fetch_mixed_price_rows_for_tickers_between(
        tickers=sorted(set(symbols)),
        start_date=trade_date - timedelta(days=10),
        end_date=trade_date,
    )
    latest: dict[str, tuple[date, float]] = {}
    for row in rows:
        symbol = str(row.get("ticker") or "").upper()
        row_date = row.get("trade_date")
        price = safe_float(row.get("adj_close_price")) or safe_float(row.get("close_price"))
        if not symbol or not isinstance(row_date, date) or price is None:
            continue
        current = latest.get(symbol)
        if current is None or row_date > current[0]:
            latest[symbol] = (row_date, price)
    return {symbol: value[1] for symbol, value in latest.items()}


def natural_order_key(row: dict[str, object]) -> tuple[str, object, str, str, str]:
    return (
        str(row.get("account_id") or ""),
        row.get("trade_date"),
        str(row.get("symbol") or "").upper(),
        str(row.get("side") or "").upper(),
        str(row.get("strategy_name") or ""),
    )


def build_order_id(*, account_id: str, trade_date: date, strategy_name: str, side: str, symbol: str) -> str:
    return f"USPO_{account_id}_{trade_date:%Y%m%d}_{strategy_name}_{side}_{symbol}"


def order_reason_buy(rank_row: dict[str, object], target_weight: float, current_weight: float, current_value: float) -> str:
    return (
        f"BUY candidate: rank_no={rank_row.get('rank_no')}, grade={rank_row.get('recommend_grade')}, "
        f"total_score={rank_row.get('total_score')}. Target weight {target_weight * 100:.1f}% exceeds current weight "
        f"{current_weight * 100:.1f}% and current value {current_value:.2f}."
    )


def order_reason_sell(rank_row: dict[str, object] | None, detail: str) -> str:
    if rank_row:
        return f"SELL candidate: {detail}. rank_no={rank_row.get('rank_no')}, grade={rank_row.get('recommend_grade')}."
    return f"SELL candidate: {detail}."


def order_reason_reject(message: str) -> str:
    return f"Order rejected: {message}"

