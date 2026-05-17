"""
rule_portfolio_backtest.py

RULE_TREND_LIQUIDITY_V1 전략의 3년 day-by-day 포트폴리오 백테스트.

기존 rule_backtest.py 는 신호→고정 horizon 수익률만 분석한다.
이 스크립트는 실제 운영 규칙(stop-loss, trailing-stop, 보유기간 제한,
섹터 비중, cooldown, 현금 비중)을 그대로 시뮬레이션한다.

실행 흐름 (하루 단위):
  T-1 장마감 → 신호 확인 → 매수/매도 후보 결정
  T   장시작  → D+1 open 에 주문 체결
  T   장마감  → close 기준으로 stop/trailing/보유기간 판단

입력:
  data/rule_signals.csv           (모든 과거 날짜의 신호)
  data/prices_daily_adjusted.csv  (OHLCV)

출력:
  outputs/rule_portfolio_backtest_equity.csv
  outputs/rule_portfolio_backtest_trades.csv
  outputs/rule_portfolio_backtest_report.json
  outputs/rule_portfolio_backtest_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day-by-day portfolio backtest for RULE_TREND_LIQUIDITY_V1"
    )
    p.add_argument("--signals-csv", type=Path, default=DATA_DIR / "rule_signals.csv")
    p.add_argument("--prices-csv", type=Path, default=DATA_DIR / "prices_daily_adjusted.csv")
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--max-positions", type=int, default=int(os.getenv("RULE_MAX_POSITIONS", "5")))
    p.add_argument("--new-entry-weight", type=float, default=float(os.getenv("RULE_NEW_ENTRY_WEIGHT", "0.10")))
    p.add_argument("--max-position-weight", type=float, default=float(os.getenv("RULE_MAX_POSITION_WEIGHT", "0.20")))
    p.add_argument("--min-cash-weight", type=float, default=float(os.getenv("RULE_MIN_CASH_WEIGHT", "0.20")))
    p.add_argument("--max-sector-weight", type=float, default=float(os.getenv("RULE_MAX_SECTOR_WEIGHT", "0.35")))
    p.add_argument("--cooldown-days", type=int, default=int(os.getenv("RULE_COOLDOWN_DAYS", "5")))
    p.add_argument("--max-holding-days", type=int, default=int(os.getenv("RULE_MAX_HOLDING_DAYS", "10")))
    p.add_argument("--max-holding-profit-buffer", type=float, default=float(os.getenv("RULE_MAX_HOLDING_DAYS_PROFIT_BUFFER", "0.02")))
    p.add_argument("--stop-loss-pct", type=float, default=float(os.getenv("RULE_STOP_LOSS_PCT", "0.05")))
    p.add_argument("--trailing-stop-pct", type=float, default=float(os.getenv("RULE_TRAILING_STOP_PCT", "0.04")))
    p.add_argument("--trailing-stop-min-profit-pct", type=float, default=float(os.getenv("RULE_TRAILING_STOP_MIN_PROFIT_PCT", "0.03")))
    p.add_argument("--signal-col", type=str, default="strong_entry_signal",
                   help="Signal column to use: entry_signal or strong_entry_signal")
    p.add_argument("--out-equity-csv", type=Path, default=OUTPUT_DIR / "rule_portfolio_backtest_equity.csv")
    p.add_argument("--out-trades-csv", type=Path, default=OUTPUT_DIR / "rule_portfolio_backtest_trades.csv")
    p.add_argument("--out-report-json", type=Path, default=OUTPUT_DIR / "rule_portfolio_backtest_report.json")
    p.add_argument("--out-report-md", type=Path, default=OUTPUT_DIR / "rule_portfolio_backtest_report.md")
    # ── 거래비용 / 슬리피지 ───────────────────────────────────────────────
    p.add_argument("--buy-commission", type=float, default=0.00015,
                   help="매수 수수료 (default 0.015%%)")
    p.add_argument("--sell-commission", type=float, default=0.00015,
                   help="매도 수수료 (default 0.015%%)")
    p.add_argument("--sell-tax", type=float, default=0.0018,
                   help="증권거래세 (default 0.18%%)")
    p.add_argument("--buy-slippage", type=float, default=0.001,
                   help="매수 슬리피지 (default 0.1%%)")
    p.add_argument("--sell-slippage", type=float, default=0.001,
                   help="매도 슬리피지 (default 0.1%%)")
    p.add_argument("--no-cost", action="store_true",
                   help="비용/슬리피지 미반영 (기존 방식 — 비교용)")
    # ── 벤치마크 ─────────────────────────────────────────────────────────
    p.add_argument("--market-csv", type=Path, default=DATA_DIR / "market_status.csv",
                   help="KOSPI 벤치마크 소스 (market_status.csv)")
    # ── IS / OOS 분리 ────────────────────────────────────────────────────
    p.add_argument("--is-end-date", type=str, default=None,
                   help="IS 종료일 YYYY-MM-DD. 지정 시 IS/OOS 성과 분리 출력.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["entry_signal", "strong_entry_signal", "gap_risk_blocked",
                "market_defensive_mode", "trading_value_pass", "market_entry_allowed"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df[col] = False
    for col in ["rule_score", "rule_score_v2", "liquidity_score", "vol_20",
                "trading_value_ma_20"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "code"]).sort_values(["date", "code"])


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["open"] = pd.to_numeric(df.get("adj_open", df.get("open")), errors="coerce")
    df["high"] = pd.to_numeric(df.get("adj_high", df.get("high")), errors="coerce")
    df["close"] = pd.to_numeric(df.get("adj_close", df.get("close")), errors="coerce")
    return df.dropna(subset=["date", "code", "open", "close"]).sort_values(["date", "code"])


# ---------------------------------------------------------------------------
# 포지션 상태
# ---------------------------------------------------------------------------

@dataclass
class Position:
    code: str
    name: str
    sector: str
    entry_date: pd.Timestamp
    entry_price: float          # 체결 시가 (stop-loss 기준)
    qty: int
    effective_entry_price: float = 0.0  # 수수료·슬리피지 반영 실질 매수가 (PnL 기준)
    highest_price: float = 0.0
    highest_return: float = 0.0
    is_reduced: bool = False  # reduce 1회 후 플래그 → 이후 전량 청산만 허용

    @property
    def cost(self) -> float:
        return self.entry_price * self.qty

    def current_value(self, price: float) -> float:
        return price * self.qty

    def current_return(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return price / self.entry_price - 1.0

    def update_high(self, price: float) -> None:
        if price > self.highest_price:
            self.highest_price = price
            self.highest_return = self.current_return(price)

    def holding_days(self, today: pd.Timestamp) -> int:
        return max(int((today.normalize() - self.entry_date.normalize()).days), 0)


@dataclass
class PendingOrder:
    code: str
    name: str
    sector: str
    signal_date: pd.Timestamp
    target_amount: float  # 목표 투자금액


@dataclass
class CompletedTrade:
    code: str
    name: str
    sector: str
    entry_date: pd.Timestamp
    entry_price: float           # 체결 시가 (raw)
    exit_date: pd.Timestamp
    exit_price: float            # 청산 시가 (raw)
    qty: int
    exit_reason: str
    effective_entry_price: float = 0.0  # 매수 수수료·슬리피지 반영
    effective_exit_price: float = 0.0   # 매도 수수료·세금·슬리피지 반영

    @property
    def gross_return(self) -> float:
        """비용 미반영 수익률 (raw prices)."""
        if self.entry_price <= 0:
            return 0.0
        return self.exit_price / self.entry_price - 1.0

    @property
    def realized_return(self) -> float:
        """비용 반영 실질 수익률 (effective prices)."""
        ep = self.effective_entry_price if self.effective_entry_price > 0 else self.entry_price
        xp = self.effective_exit_price if self.effective_exit_price > 0 else self.exit_price
        if ep <= 0:
            return 0.0
        return xp / ep - 1.0

    @property
    def holding_days(self) -> int:
        return max(int((self.exit_date.normalize() - self.entry_date.normalize()).days), 0)

    @property
    def pnl(self) -> float:
        ep = self.effective_entry_price if self.effective_entry_price > 0 else self.entry_price
        xp = self.effective_exit_price if self.effective_exit_price > 0 else self.exit_price
        return (xp - ep) * self.qty

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "sector": self.sector,
            "entry_date": self.entry_date.date().isoformat(),
            "exit_date": self.exit_date.date().isoformat(),
            "holding_days": self.holding_days,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "qty": self.qty,
            "gross_return": round(self.gross_return, 6),
            "realized_return": round(self.realized_return, 6),
            "pnl": round(self.pnl, 2),
            "exit_reason": self.exit_reason,
        }


# ---------------------------------------------------------------------------
# 시뮬레이터
# ---------------------------------------------------------------------------

@dataclass
class PendingSell:
    code: str
    reason: str
    qty: int       # 청산 수량 (전량 or 50%)
    is_partial: bool = False  # True이면 reduce (부분 청산)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending_buys: list[PendingOrder] = field(default_factory=list)
    pending_sells: list[PendingSell] = field(default_factory=list)
    recent_exits: list[tuple[str, pd.Timestamp]] = field(default_factory=list)
    trade_log: list[CompletedTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)


def _bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def _float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _load_benchmark(market_csv: Path) -> dict[pd.Timestamp, float]:
    """market_status.csv에서 KOSPI 종가를 {date: close} 딕셔너리로 반환."""
    if not market_csv.exists():
        return {}
    try:
        mkt = pd.read_csv(market_csv, low_memory=False)
        if "date" not in mkt.columns or "kospi_close" not in mkt.columns:
            return {}
        mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
        mkt = mkt.dropna(subset=["date", "kospi_close"])
        return {pd.Timestamp(r.date): float(r.kospi_close) for r in mkt.itertuples()}
    except Exception:
        return {}


def run_simulation(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: argparse.Namespace,
) -> PortfolioState:
    # 비용 계수 (no-cost 모드이면 0)
    buy_cost_rate = 0.0 if getattr(cfg, "no_cost", False) else (
        getattr(cfg, "buy_slippage", 0.001) + getattr(cfg, "buy_commission", 0.00015)
    )
    sell_cost_rate = 0.0 if getattr(cfg, "no_cost", False) else (
        getattr(cfg, "sell_slippage", 0.001)
        + getattr(cfg, "sell_commission", 0.00015)
        + getattr(cfg, "sell_tax", 0.0018)
    )

    # KOSPI 벤치마크 데이터 로드
    benchmark_idx: dict[pd.Timestamp, float] = {}
    if hasattr(cfg, "market_csv") and cfg.market_csv:
        benchmark_idx = _load_benchmark(Path(cfg.market_csv))

    # 가격 인덱스: {date -> {code -> {open, high, close}}}
    price_idx: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    for row in prices.itertuples(index=False):
        d = pd.Timestamp(row.date)
        if d not in price_idx:
            price_idx[d] = {}
        price_idx[d][row.code] = {
            "open": _float(row.open),
            "close": _float(row.close),
        }

    # 신호 인덱스: {date -> {code -> row_dict}}
    signal_idx: dict[pd.Timestamp, dict[str, dict]] = {}
    for row in signals.itertuples(index=False):
        d = pd.Timestamp(row.date)
        if d not in signal_idx:
            signal_idx[d] = {}
        signal_idx[d][row.code] = row._asdict()

    all_dates = sorted(set(price_idx.keys()) | set(signal_idx.keys()))

    # 날짜 필터
    if cfg.start_date:
        cutoff_start = pd.Timestamp(cfg.start_date)
        all_dates = [d for d in all_dates if d >= cutoff_start]
    if cfg.end_date:
        cutoff_end = pd.Timestamp(cfg.end_date)
        all_dates = [d for d in all_dates if d <= cutoff_end]

    state = PortfolioState(cash=cfg.initial_capital)

    for today in all_dates:
        prices_today = price_idx.get(today, {})
        signals_today = signal_idx.get(today, {})

        # ------------------------------------------------------------------
        # 1. 전날 결정된 매도 체결 (오늘 open) — 전량 or 부분(reduce)
        # ------------------------------------------------------------------
        fully_exited: list[str] = []
        for sell in state.pending_sells:
            code = sell.code
            if code not in state.positions:
                continue
            pos = state.positions[code]
            px = prices_today.get(code, {})
            exit_price = px.get("open", pos.entry_price)
            if exit_price <= 0:
                exit_price = pos.entry_price

            sell_qty = min(sell.qty, pos.qty)
            effective_exit_price = exit_price * (1.0 - sell_cost_rate)
            state.cash += effective_exit_price * sell_qty
            trade = CompletedTrade(
                code=pos.code,
                name=pos.name,
                sector=pos.sector,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=today,
                exit_price=exit_price,
                qty=sell_qty,
                exit_reason=sell.reason,
                effective_entry_price=pos.effective_entry_price,
                effective_exit_price=effective_exit_price,
            )
            state.trade_log.append(trade)

            remaining_qty = pos.qty - sell_qty
            if remaining_qty <= 0:
                # 전량 청산
                state.recent_exits.append((code, today))
                fully_exited.append(code)
            else:
                # 부분 청산: 남은 수량 유지, reduce 플래그 설정
                pos.qty = remaining_qty
                pos.is_reduced = True

        for code in fully_exited:
            state.positions.pop(code, None)
        state.pending_sells.clear()

        # ------------------------------------------------------------------
        # 2. 전날 결정된 매수 체결 (오늘 open)
        # ------------------------------------------------------------------
        executed_buy_codes: set[str] = set()
        for order in state.pending_buys:
            if order.code in state.positions:
                continue
            px = prices_today.get(order.code, {})
            buy_price = px.get("open", 0.0)
            if buy_price <= 0:
                continue
            effective_buy_price = buy_price * (1.0 + buy_cost_rate)
            qty = int(order.target_amount // effective_buy_price)
            if qty <= 0:
                continue
            actual_cost = effective_buy_price * qty
            if actual_cost > state.cash + 1:  # 1원 오차 허용
                qty = int(state.cash // effective_buy_price)
                actual_cost = effective_buy_price * qty
            if qty <= 0:
                continue
            state.cash -= actual_cost
            state.positions[order.code] = Position(
                code=order.code,
                name=order.name,
                sector=order.sector,
                entry_date=today,
                entry_price=buy_price,
                effective_entry_price=effective_buy_price,
                qty=qty,
                highest_price=buy_price,
                highest_return=0.0,
            )
            executed_buy_codes.add(order.code)
        state.pending_buys.clear()

        # ------------------------------------------------------------------
        # 3. 보유 포지션 가격 갱신 + 청산 조건 판단 (오늘 close 기준)
        # ------------------------------------------------------------------
        for code, pos in state.positions.items():
            px = prices_today.get(code, {})
            close = px.get("close", 0.0)
            if close <= 0:
                continue

            pos.update_high(close)
            cur_ret = pos.current_return(close)
            hold_days = pos.holding_days(today)
            drawdown_from_high = (close / pos.highest_price - 1.0) if pos.highest_price > 0 else 0.0

            exit_reason: str | None = None
            is_partial = False

            if cur_ret <= -abs(cfg.stop_loss_pct):
                # 손절: 항상 전량 청산
                exit_reason = "stop_loss"
            elif (
                pos.highest_return >= cfg.trailing_stop_min_profit_pct
                and drawdown_from_high <= -abs(cfg.trailing_stop_pct)
            ):
                if cur_ret <= 0 or pos.is_reduced:
                    # 수익 반납 후 하락, 또는 이미 reduce됐으면 전량 청산
                    exit_reason = "trailing_stop_exit"
                else:
                    # 아직 수익 중 + 최초 reduce: 50% 부분 청산
                    exit_reason = "trailing_stop_reduce"
                    is_partial = True
            elif hold_days > cfg.max_holding_days and cur_ret < cfg.max_holding_profit_buffer:
                if cur_ret <= 0 or pos.is_reduced:
                    # 손실 or 본전, 또는 이미 reduce됐으면 전량 청산
                    exit_reason = "max_holding_days_exit"
                else:
                    # 소폭 수익 중 + 최초 reduce: 50% 부분 청산
                    exit_reason = "max_holding_days_reduce"
                    is_partial = True
            else:
                sig = signals_today.get(code, {})
                rule_score_v2 = _float(sig.get("rule_score_v2"), 100.0)
                gap_blocked = _bool(sig.get("gap_risk_blocked", False))
                if rule_score_v2 < 35.0 or gap_blocked:
                    exit_reason = "rule_exit_condition"

            if exit_reason:
                sell_qty = max(pos.qty // 2, 1) if is_partial else pos.qty
                state.pending_sells.append(
                    PendingSell(code=code, reason=exit_reason, qty=sell_qty, is_partial=is_partial)
                )

        # ------------------------------------------------------------------
        # 4. 신규 진입 후보 선택 (오늘 신호 → 내일 매수)
        # ------------------------------------------------------------------
        cooldown_set: set[str] = {
            code for code, dt in state.recent_exits
            if (today - dt).days <= cfg.cooldown_days
        }
        held_codes = set(state.positions.keys())
        # 전량 청산 예정인 코드만 active 에서 제외 (partial reduce는 계속 보유)
        full_exit_codes = {s.code for s in state.pending_sells if not s.is_partial}
        active_codes = held_codes - full_exit_codes

        # 현재 시장 방어 모드 판단
        defensive_values = [_bool(s.get("market_defensive_mode", False)) for s in signals_today.values()]
        market_defensive = any(defensive_values) if defensive_values else False

        if not market_defensive and signals_today:
            # 현재 equity 추정 (entry 결정을 위한 비중 계산 기준)
            position_value = sum(
                pos.qty * _float(prices_today.get(code, {}).get("close", pos.entry_price), pos.entry_price)
                for code, pos in state.positions.items()
                if code not in full_exit_codes
            )
            total_equity = state.cash + position_value
            if total_equity <= 0:
                total_equity = cfg.initial_capital

            # 섹터 비중 계산 (전량 청산 예정 포지션 제외)
            sector_weight: dict[str, float] = {}
            for code, pos in state.positions.items():
                if code in full_exit_codes:
                    continue
                pos_val = pos.qty * _float(prices_today.get(code, {}).get("close", pos.entry_price), pos.entry_price)
                w = pos_val / total_equity
                sector_weight[pos.sector] = sector_weight.get(pos.sector, 0.0) + w

            # 후보 정렬: rule_score_v2 ↓, rule_score ↓, liquidity_score ↓, vol_20 ↑
            signal_col = cfg.signal_col
            candidates = [
                s for s in signals_today.values()
                if (
                    _bool(s.get(signal_col, False))
                    and s.get("code") not in active_codes
                    and s.get("code") not in cooldown_set
                )
            ]
            candidates.sort(
                key=lambda s: (
                    -_float(s.get("rule_score_v2"), 0.0),
                    -_float(s.get("rule_score"), 0.0),
                    -_float(s.get("liquidity_score"), 0.0),
                    _float(s.get("vol_20"), 999.0),
                )
            )

            for sig in candidates:
                code = str(sig.get("code", "")).zfill(6)
                sector = str(sig.get("sector") or "(none)")
                n_open = len(active_codes) + len(state.pending_buys)
                if n_open >= cfg.max_positions:
                    break

                # 섹터 비중 확인
                sector_after = sector_weight.get(sector, 0.0) + cfg.new_entry_weight
                if sector_after > cfg.max_sector_weight:
                    continue

                # 현금 비중 확인
                allocated = len(state.pending_buys) * cfg.new_entry_weight * total_equity
                cash_after_buy = state.cash - allocated - cfg.new_entry_weight * total_equity
                if cash_after_buy / total_equity < cfg.min_cash_weight:
                    continue

                target_amount = total_equity * cfg.new_entry_weight
                state.pending_buys.append(
                    PendingOrder(
                        code=code,
                        name=str(sig.get("name") or ""),
                        sector=sector,
                        signal_date=today,
                        target_amount=target_amount,
                    )
                )
                active_codes.add(code)
                sector_weight[sector] = sector_weight.get(sector, 0.0) + cfg.new_entry_weight

        # ------------------------------------------------------------------
        # 5. 오늘 equity 기록
        # ------------------------------------------------------------------
        position_value_eod = sum(
            pos.qty * _float(prices_today.get(code, {}).get("close", pos.entry_price), pos.entry_price)
            for code, pos in state.positions.items()
        )
        equity = state.cash + position_value_eod
        state.equity_curve.append({
            "date": today.date().isoformat(),
            "equity": round(equity, 2),
            "cash": round(state.cash, 2),
            "position_value": round(position_value_eod, 2),
            "n_positions": len(state.positions),
            "n_pending_buys": len(state.pending_buys),
            "kospi_close": benchmark_idx.get(today),
        })

    return state


# ---------------------------------------------------------------------------
# 성과 지표 계산
# ---------------------------------------------------------------------------

def _period_metrics(
    equity_df: pd.DataFrame,
    trades: list,
    initial_equity: float,
) -> dict[str, Any]:
    """주어진 equity_df / trades 구간의 성과 지표를 계산한다."""
    if equity_df.empty or initial_equity <= 0:
        return {}

    eq = equity_df["equity"]
    final_equity = float(eq.iloc[-1])
    total_return = final_equity / initial_equity - 1.0

    n_days = (equity_df.index[-1] - equity_df.index[0]).days
    years = max(n_days / 365.25, 1 / 365.25)
    cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0

    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    mdd = float(drawdown.min())

    daily_ret = eq.pct_change().dropna()
    rf_daily = 0.02 / 252
    excess = daily_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0

    # KOSPI 벤치마크 수익률
    kospi_col = equity_df.get("kospi_close") if isinstance(equity_df, pd.DataFrame) else None
    if kospi_col is not None:
        kospi_valid = kospi_col.dropna()
    else:
        kospi_valid = pd.Series(dtype=float)
    if len(kospi_valid) >= 2:
        kospi_return = float(kospi_valid.iloc[-1] / kospi_valid.iloc[0] - 1.0)
        alpha = total_return - kospi_return
    else:
        kospi_return = None
        alpha = None

    # 거래 통계
    n_trades = len(trades)
    if n_trades > 0:
        returns = [t.realized_return for t in trades]
        gross_returns = [t.gross_return for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        win_rate = len(wins) / n_trades
        avg_return = float(np.mean(returns))
        avg_gross = float(np.mean(gross_returns))
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else None
        avg_holding = float(np.mean([t.holding_days for t in trades]))
        exit_reasons: dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    else:
        win_rate = avg_return = avg_gross = avg_win = avg_loss = payoff = avg_holding = 0.0
        exit_reasons = {}

    yearly: dict[str, float] = {}
    for year, grp in equity_df.groupby(equity_df.index.year):
        y_start = float(grp["equity"].iloc[0])
        y_end = float(grp["equity"].iloc[-1])
        yearly[str(year)] = round(y_end / y_start - 1.0, 6)

    return {
        "period": {
            "start": equity_df.index[0].date().isoformat(),
            "end": equity_df.index[-1].date().isoformat(),
            "days": n_days,
            "years": round(years, 2),
        },
        "capital": {
            "initial": round(initial_equity, 2),
            "final": round(final_equity, 2),
        },
        "returns": {
            "total_return": round(total_return, 6),
            "cagr": round(cagr, 6),
            "mdd": round(mdd, 6),
            "sharpe": round(sharpe, 4),
            "kospi_return": round(kospi_return, 6) if kospi_return is not None else None,
            "alpha": round(alpha, 6) if alpha is not None else None,
        },
        "trades": {
            "total": n_trades,
            "win_rate": round(win_rate, 4) if n_trades else None,
            "avg_return_net": round(avg_return, 6) if n_trades else None,
            "avg_return_gross": round(avg_gross, 6) if n_trades else None,
            "avg_win": round(avg_win, 6) if n_trades else None,
            "avg_loss": round(avg_loss, 6) if n_trades else None,
            "payoff_ratio": round(payoff, 4) if payoff is not None else None,
            "avg_holding_days": round(avg_holding, 1) if n_trades else None,
            "exit_reasons": exit_reasons,
        },
        "yearly_returns": yearly,
    }


def compute_metrics(
    state: PortfolioState,
    initial_capital: float,
    is_end_date: str | None = None,
) -> dict[str, Any]:
    equity_df = pd.DataFrame(state.equity_curve)
    if equity_df.empty:
        return {}

    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df = equity_df.set_index("date").sort_index()

    full = _period_metrics(equity_df, state.trade_log, initial_capital)

    result: dict[str, Any] = {"full": full}

    if is_end_date:
        is_cut = pd.Timestamp(is_end_date)
        eq_is = equity_df[equity_df.index <= is_cut]
        eq_oos = equity_df[equity_df.index > is_cut]
        trades_is = [t for t in state.trade_log if t.exit_date <= is_cut]
        trades_oos = [t for t in state.trade_log if t.exit_date > is_cut]

        if not eq_is.empty:
            result["IS"] = _period_metrics(eq_is, trades_is, initial_capital)
        if not eq_oos.empty:
            oos_initial = float(eq_oos["equity"].iloc[0])
            result["OOS"] = _period_metrics(eq_oos, trades_oos, oos_initial)

    return result


# ---------------------------------------------------------------------------
# 마크다운 리포트
# ---------------------------------------------------------------------------

def fmt_pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "NA"


def fmt_num(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):,.{decimals}f}"
    except (TypeError, ValueError):
        return "NA"


def _render_period_section(title: str, m: dict) -> list[str]:
    p = m.get("period", {})
    c = m.get("capital", {})
    r = m.get("returns", {})
    t = m.get("trades", {})
    yr = m.get("yearly_returns", {})

    no_cost = r.get("kospi_return") is None  # 벤치마크 없으면 비용도 없을 수 있음

    lines = [
        f"## {title}",
        "",
        f"| 항목 | 값 |",
        f"| --- | --- |",
        f"| 기간 | {p.get('start')} ~ {p.get('end')} ({p.get('days')}일, {p.get('years')}년) |",
        f"| 초기 자산 | {fmt_num(c.get('initial'))} 원 |",
        f"| 최종 자산 | {fmt_num(c.get('final'))} 원 |",
        f"| 총 수익률 (net) | {fmt_pct(r.get('total_return'))} |",
        f"| CAGR | {fmt_pct(r.get('cagr'))} |",
        f"| MDD | {fmt_pct(r.get('mdd'))} |",
        f"| Sharpe (rf=2%) | {fmt_num(r.get('sharpe'), 3)} |",
    ]
    if r.get("kospi_return") is not None:
        lines.append(f"| KOSPI 수익률 | {fmt_pct(r.get('kospi_return'))} |")
        lines.append(f"| **Alpha** | **{fmt_pct(r.get('alpha'))}** |")

    lines += [
        "",
        f"| 거래 항목 | 값 |",
        f"| --- | --- |",
        f"| 총 거래 수 | {t.get('total', 0)} |",
        f"| 승률 | {fmt_pct(t.get('win_rate'))} |",
        f"| 평균 수익률 (net) | {fmt_pct(t.get('avg_return_net'))} |",
        f"| 평균 수익률 (gross) | {fmt_pct(t.get('avg_return_gross'))} |",
        f"| 평균 수익 (win) | {fmt_pct(t.get('avg_win'))} |",
        f"| 평균 손실 (loss) | {fmt_pct(t.get('avg_loss'))} |",
        f"| Payoff ratio | {fmt_num(t.get('payoff_ratio'), 3)} |",
        f"| 평균 보유기간 | {fmt_num(t.get('avg_holding_days'), 1)}일 |",
        "",
        "| 청산 사유 | 건수 |",
        "| --- | ---: |",
    ]
    for reason, cnt in sorted((t.get("exit_reasons") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {cnt} |")

    if yr:
        lines += ["", "| 연도 | 수익률 |", "| --- | ---: |"]
        for y, ret in sorted(yr.items()):
            lines.append(f"| {y} | {fmt_pct(ret)} |")

    lines.append("")
    return lines


def render_md(metrics: dict, cfg: argparse.Namespace) -> str:
    no_cost = getattr(cfg, "no_cost", False)
    buy_total = (getattr(cfg, "buy_slippage", 0.001) + getattr(cfg, "buy_commission", 0.00015)) if not no_cost else 0.0
    sell_total = (getattr(cfg, "sell_slippage", 0.001) + getattr(cfg, "sell_commission", 0.00015) + getattr(cfg, "sell_tax", 0.0018)) if not no_cost else 0.0

    lines = [
        "# Rule Portfolio Backtest Report",
        "",
        f"전략: **RULE_TREND_LIQUIDITY_V1** | 신호: `{cfg.signal_col}`",
        "",
        "## 파라미터",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| 초기 자본 | {fmt_num(cfg.initial_capital)} 원 |",
        f"| 최대 보유 종목 수 | {cfg.max_positions} |",
        f"| 신규 진입 비중 | {fmt_pct(cfg.new_entry_weight)} |",
        f"| 최소 현금 비중 | {fmt_pct(cfg.min_cash_weight)} |",
        f"| 최대 섹터 비중 | {fmt_pct(cfg.max_sector_weight)} |",
        f"| 최대 보유기간 | {cfg.max_holding_days}일 |",
        f"| 쿨다운 | {cfg.cooldown_days}일 |",
        f"| Stop-loss | {fmt_pct(cfg.stop_loss_pct)} |",
        f"| Trailing stop | {fmt_pct(cfg.trailing_stop_pct)} (최소 수익 {fmt_pct(cfg.trailing_stop_min_profit_pct)} 이상 시) |",
        f"| 비용 모드 | {'미반영 (no-cost)' if no_cost else f'매수 {fmt_pct(buy_total)} / 매도 {fmt_pct(sell_total)}'} |",
        f"| IS 종료일 | {getattr(cfg, 'is_end_date', None) or '미설정'} |",
        "",
    ]

    if "full" in metrics:
        lines += _render_period_section("전체 구간 성과", metrics["full"])
    if "IS" in metrics:
        lines += _render_period_section("IS (In-Sample) 성과", metrics["IS"])
    if "OOS" in metrics:
        lines += _render_period_section("OOS (Out-of-Sample) 성과", metrics["OOS"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    print("신호 데이터 로딩 중...")
    signals = load_signals(cfg.signals_csv)
    print(f"  신호 날짜: {signals['date'].min().date()} ~ {signals['date'].max().date()}, "
          f"종목×날짜 {len(signals):,}행")

    print("가격 데이터 로딩 중...")
    prices = load_prices(cfg.prices_csv)
    print(f"  가격 날짜: {prices['date'].min().date()} ~ {prices['date'].max().date()}, "
          f"{len(prices):,}행")

    print(f"\n시뮬레이션 시작 (초기 자본: {cfg.initial_capital:,.0f}원, 신호: {cfg.signal_col})")
    state = run_simulation(signals, prices, cfg)

    print(f"시뮬레이션 완료: {len(state.trade_log)}건 거래, {len(state.equity_curve)}일 기록")

    metrics = compute_metrics(state, cfg.initial_capital, getattr(cfg, "is_end_date", None))

    # 출력 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame(state.equity_curve)
    equity_df.to_csv(cfg.out_equity_csv, index=False, encoding="utf-8-sig")
    print(f"equity curve → {cfg.out_equity_csv}")

    trades_df = pd.DataFrame([t.to_dict() for t in state.trade_log])
    trades_df.to_csv(cfg.out_trades_csv, index=False, encoding="utf-8-sig")
    print(f"trade log    → {cfg.out_trades_csv}")

    cfg.out_report_json.write_text(
        json.dumps({"metrics": metrics, "config": {
            "signal_col": cfg.signal_col,
            "initial_capital": cfg.initial_capital,
            "max_positions": cfg.max_positions,
            "new_entry_weight": cfg.new_entry_weight,
            "stop_loss_pct": cfg.stop_loss_pct,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "trailing_stop_min_profit_pct": cfg.trailing_stop_min_profit_pct,
            "max_holding_days": cfg.max_holding_days,
            "cooldown_days": cfg.cooldown_days,
        }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report JSON  → {cfg.out_report_json}")

    cfg.out_report_md.write_text(render_md(metrics, cfg), encoding="utf-8")
    print(f"report MD    → {cfg.out_report_md}")

    # 콘솔 요약
    full = metrics.get("full", {})
    r = full.get("returns", {})
    t = full.get("trades", {})
    print("\n" + "=" * 55)
    print(f"총 수익률 (net) : {fmt_pct(r.get('total_return'))}")
    print(f"KOSPI 수익률    : {fmt_pct(r.get('kospi_return'))}")
    print(f"Alpha           : {fmt_pct(r.get('alpha'))}")
    print(f"CAGR            : {fmt_pct(r.get('cagr'))}")
    print(f"MDD             : {fmt_pct(r.get('mdd'))}")
    print(f"Sharpe          : {fmt_num(r.get('sharpe'), 3)}")
    print(f"총 거래         : {t.get('total', 0)}건")
    print(f"승률            : {fmt_pct(t.get('win_rate'))}")
    if "OOS" in metrics:
        oos = metrics["OOS"].get("returns", {})
        print(f"OOS 수익률      : {fmt_pct(oos.get('total_return'))}")
        print(f"OOS Alpha       : {fmt_pct(oos.get('alpha'))}")
    print("=" * 55)


if __name__ == "__main__":
    main()
