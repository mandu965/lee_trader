"""
US 자동매매 전략 포트폴리오 백테스트

현재 운영 중인 US_TRADE_SCHEDULER 전략을 과거 구간에 그대로 재현.
명세서: doc/modules/Lee_trader_us/20260519_US 자동매매 전략 백테스트 명세서.md

체결 방식:
  - 진입/청산 신호: 결정일 종가로 즉시 체결 (paper fill 방식과 동일하게 보수적 가정)
  - 슬리피지: 매수 +0.05%, 매도 -0.05%

실행:
  cd d:/ai/lee_trader
  .venv/Scripts/python.exe python/us/run_portfolio_backtest.py [--start 2023-01-01] [--end 2026-05-18]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text
from python.us.us_db import get_us_engine

LOGGER = logging.getLogger("us_portfolio_backtest")

# =============================================================================
# 전략 파라미터 (현재 운영 기준)
# =============================================================================
INITIAL_CAPITAL = 100_000.0
BUDGET_PER_SYMBOL = 5_000.0
MAX_DAILY_NEW_BUYS = 2
TOP_N = 5
MIN_GRADE = "BUY"
MIN_SCORE = 70.0
MIN_PROB = 0.0
STOP_LOSS = -0.08
TAKE_PROFIT = 0.15
TRAILING_STOP_PCT = 0.10
MAX_HOLDING_DAYS = 60
RANK_EXIT_THRESHOLD = 30
MIN_PROB_HOLD = 0.0
COOLDOWN_DAYS = 10
SLIPPAGE_BPS = 5.0          # 편도 슬리피지 (bps)
COMMISSION_BPS = 1.0        # 편도 수수료 (bps)
MAX_OPEN_POSITIONS = 10

RANKING_SOURCE = "ml_v1"

# =============================================================================
# 데이터 클래스
# =============================================================================

@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    entry_date: date
    peak_price: float
    cost_amount: float
    cooldown_until: Optional[date] = None

    def pnl_pct(self, current_price: float) -> float:
        if self.avg_price <= 0:
            return 0.0
        return current_price / self.avg_price - 1.0

    def peak_drawdown(self, current_price: float) -> float:
        if self.peak_price <= 0:
            return 0.0
        return current_price / self.peak_price - 1.0


@dataclass
class Trade:
    trade_date: date
    symbol: str
    side: str
    qty: int
    price: float
    amount: float
    reason: str
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0
    holding_days: int = 0


@dataclass
class DailySnapshot:
    snapshot_date: date
    cash: float
    position_value: float
    equity: float
    n_positions: int
    spy_close: float
    qqq_close: float


# =============================================================================
# DB 조회
# =============================================================================

def load_rankings(engine, source: str = "ml_v1") -> Dict[date, List[dict]]:
    sql = text("""
        SELECT trade_date, symbol, rank_no, total_score, recommend_grade,
               score_detail_json
        FROM recommend.us_stock_rank_daily
        WHERE source = :source
        ORDER BY trade_date, rank_no ASC NULLS LAST
    """)
    result: Dict[date, List[dict]] = defaultdict(list)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"source": source}).fetchall()
    for row in rows:
        d = dict(row._mapping)
        sd = d.get("score_detail_json") or {}
        d["prob"] = sd.get("prob_top20_20d") or sd.get("prob_top20_60d") or 0.0
        result[d["trade_date"]].append(d)
    LOGGER.info("Loaded rankings: %d trading days, source=%s", len(result), source)
    return result


def load_prices(engine) -> Dict[Tuple[str, date], dict]:
    sql = text("""
        SELECT trade_date, ticker AS symbol, open_price, close_price, adj_close_price
        FROM market.us_stock_daily_price
        ORDER BY trade_date, ticker
    """)
    prices: Dict[Tuple[str, date], dict] = {}
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    for row in rows:
        d = dict(row._mapping)
        prices[(d["symbol"], d["trade_date"])] = d
    LOGGER.info("Loaded stock prices: %d rows", len(prices))
    return prices


def load_benchmark_prices(engine) -> Dict[Tuple[str, date], float]:
    sql = text("""
        SELECT trade_date, ticker AS symbol, close AS close_price
        FROM market.us_etf_daily_price
        WHERE ticker IN ('SPY', 'QQQ')
        ORDER BY trade_date
    """)
    bm: Dict[Tuple[str, date], float] = {}
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    for row in rows:
        d = dict(row._mapping)
        bm[(d["symbol"], d["trade_date"])] = float(d["close_price"])
    LOGGER.info("Loaded benchmark prices: %d rows", len(bm))
    return bm


# =============================================================================
# 유틸리티
# =============================================================================

def _grade_passes(grade: str, min_grade: str = "BUY") -> bool:
    order = {"BUY": 0, "STRONG_BUY": -1, "HOLD": 1, "SELL": 2}
    return order.get(str(grade).upper(), 99) <= order.get(min_grade.upper(), 0)


def _fill_price(close: float, side: str, slippage_bps: float = SLIPPAGE_BPS,
                commission_bps: float = COMMISSION_BPS) -> float:
    rate = (slippage_bps + commission_bps) / 10000.0
    return close * (1 + rate) if side == "BUY" else close * (1 - rate)


# =============================================================================
# 백테스트 엔진
# =============================================================================

def run_backtest(
    rankings: Dict[date, List[dict]],
    prices: Dict[Tuple[str, date], dict],
    bm_prices: Dict[Tuple[str, date], float],
    start_date: date,
    end_date: date,
) -> Tuple[List[Trade], List[DailySnapshot], dict]:

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    cooldowns: Dict[str, date] = {}
    trades: List[Trade] = []
    daily_snapshots: List[DailySnapshot] = []

    trading_days = sorted(d for d in rankings if start_date <= d <= end_date)
    LOGGER.info("Backtest range: %s ~ %s (%d days)", start_date, end_date, len(trading_days))

    for today in trading_days:
        ranks = rankings[today]
        price_map: Dict[str, float] = {}
        for sym in {r["symbol"] for r in ranks} | set(positions.keys()):
            p = prices.get((sym, today))
            if p and p.get("close_price"):
                price_map[sym] = float(p["close_price"])

        # ── 1. 포지션 업데이트 (peak_price) ──────────────────────────────
        for sym, pos in positions.items():
            cp = price_map.get(sym)
            if cp and cp > pos.peak_price:
                pos.peak_price = cp

        # ── 2. 매도 결정 ─────────────────────────────────────────────────
        rank_map = {r["symbol"]: r for r in ranks}
        to_sell: List[Tuple[str, str]] = []  # (symbol, reason)

        for sym, pos in list(positions.items()):
            cp = price_map.get(sym)
            if cp is None:
                continue
            pnl = pos.pnl_pct(cp)
            trail = pos.peak_drawdown(cp)
            hdays = (today - pos.entry_date).days
            rank_info = rank_map.get(sym)
            cur_rank = int(rank_info["rank_no"]) if rank_info and rank_info.get("rank_no") else 9999
            cur_prob = float(rank_info["prob"]) if rank_info else 0.0

            reason = None
            if pnl <= STOP_LOSS:
                reason = f"STOP_LOSS({pnl:.1%})"
            elif pnl >= TAKE_PROFIT:
                reason = f"TAKE_PROFIT({pnl:.1%})"
            elif trail <= -TRAILING_STOP_PCT:
                reason = f"TRAILING_STOP(trail={trail:.1%}, peak={pos.peak_price:.2f})"
            elif hdays >= MAX_HOLDING_DAYS:
                reason = f"MAX_HOLDING_DAYS({hdays}d)"
            elif cur_rank > RANK_EXIT_THRESHOLD:
                reason = f"RANK_EXIT(rank={cur_rank})"
            elif MIN_PROB_HOLD > 0 and cur_prob < MIN_PROB_HOLD:
                reason = f"PROB_HOLD({cur_prob:.3f}<{MIN_PROB_HOLD})"

            if reason:
                to_sell.append((sym, reason))

        # ── 3. 매도 체결 ─────────────────────────────────────────────────
        for sym, reason in to_sell:
            pos = positions.pop(sym)
            cp = price_map.get(sym, pos.avg_price)
            fill = _fill_price(cp, "SELL")
            pnl_amt = (fill - pos.avg_price) * pos.qty
            pnl_pct = fill / pos.avg_price - 1.0
            proceeds = fill * pos.qty
            cash += proceeds
            hdays = (today - pos.entry_date).days
            trades.append(Trade(
                trade_date=today, symbol=sym, side="SELL", qty=pos.qty,
                price=fill, amount=proceeds, reason=reason,
                pnl_pct=pnl_pct, pnl_amount=pnl_amt, holding_days=hdays,
            ))
            cooldowns[sym] = today + timedelta(days=COOLDOWN_DAYS)
            LOGGER.debug("SELL %s @ %.2f (%s) pnl=%.1f%%", sym, fill, reason, pnl_pct * 100)

        # ── 4. 매수 후보 선정 ────────────────────────────────────────────
        daily_new_buys = 0
        candidate_ranks = [
            r for r in sorted(ranks, key=lambda x: int(x.get("rank_no") or 9999))
            if int(r.get("rank_no") or 9999) <= TOP_N
        ]

        for r in candidate_ranks:
            if daily_new_buys >= MAX_DAILY_NEW_BUYS:
                break
            if len(positions) >= MAX_OPEN_POSITIONS:
                break

            sym = r["symbol"]
            grade = str(r.get("recommend_grade") or "").upper()
            score = float(r.get("total_score") or 0.0)
            prob = float(r.get("prob") or 0.0)

            # 차단 조건
            if sym in positions:
                continue
            if cooldowns.get(sym, date.min) > today:
                continue
            if not _grade_passes(grade, MIN_GRADE):
                continue
            if score < MIN_SCORE:
                continue
            if MIN_PROB > 0 and prob < MIN_PROB:
                continue

            cp = price_map.get(sym)
            if cp is None:
                continue

            # 정수 수량 계산 ($5,000 기준)
            qty = int(BUDGET_PER_SYMBOL / cp)
            if qty <= 0:
                continue
            cost = _fill_price(cp, "BUY") * qty
            if cost > cash:
                continue

            # 매수 체결
            fill = _fill_price(cp, "BUY")
            cash -= fill * qty
            positions[sym] = Position(
                symbol=sym, qty=qty, avg_price=fill, entry_date=today,
                peak_price=cp, cost_amount=fill * qty,
            )
            trades.append(Trade(
                trade_date=today, symbol=sym, side="BUY", qty=qty,
                price=fill, amount=fill * qty, reason=f"RANK_{r['rank_no']}",
            ))
            daily_new_buys += 1
            LOGGER.debug("BUY %s @ %.2f qty=%d", sym, fill, qty)

        # ── 5. 일별 스냅샷 ───────────────────────────────────────────────
        pos_value = sum(
            (price_map.get(sym, pos.avg_price)) * pos.qty
            for sym, pos in positions.items()
        )
        equity = cash + pos_value
        spy_close = bm_prices.get(("SPY", today), 0.0)
        qqq_close = bm_prices.get(("QQQ", today), 0.0)
        daily_snapshots.append(DailySnapshot(
            snapshot_date=today, cash=cash, position_value=pos_value,
            equity=equity, n_positions=len(positions),
            spy_close=spy_close, qqq_close=qqq_close,
        ))

    # ── 최종 청산 (보유 포지션 마지막 날 종가 체결) ──────────────────────
    last_day = trading_days[-1] if trading_days else end_date
    for sym, pos in list(positions.items()):
        cp = price_map.get(sym, pos.avg_price)
        fill = _fill_price(cp, "SELL")
        pnl_amt = (fill - pos.avg_price) * pos.qty
        pnl_pct = fill / pos.avg_price - 1.0
        hdays = (last_day - pos.entry_date).days
        trades.append(Trade(
            trade_date=last_day, symbol=sym, side="SELL", qty=pos.qty,
            price=fill, amount=fill * pos.qty, reason="BACKTEST_END",
            pnl_pct=pnl_pct, pnl_amount=pnl_amt, holding_days=hdays,
        ))

    return trades, daily_snapshots, {}


# =============================================================================
# 성과 집계
# =============================================================================

def compute_summary(
    trades: List[Trade],
    daily_snapshots: List[DailySnapshot],
    bm_prices: Dict[Tuple[str, date], float],
    start_date: date,
    end_date: date,
    source: str,
) -> dict:
    if not daily_snapshots:
        return {}

    equities = [s.equity for s in daily_snapshots]
    dates = [s.snapshot_date for s in daily_snapshots]
    start_equity = INITIAL_CAPITAL
    end_equity = equities[-1]
    total_return = end_equity / start_equity - 1.0

    # MDD
    peak = start_equity
    mdd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)

    # 거래 통계
    sell_trades = [t for t in trades if t.side == "SELL" and t.reason != "BACKTEST_END"]
    wins = [t for t in sell_trades if t.pnl_pct > 0]
    losses = [t for t in sell_trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0.0
    avg_pnl = sum(t.pnl_pct for t in sell_trades) / len(sell_trades) if sell_trades else 0.0
    avg_hold = sum(t.holding_days for t in sell_trades) / len(sell_trades) if sell_trades else 0.0

    # Benchmark 비교 (SPY/QQQ)
    spy_start = bm_prices.get(("SPY", start_date))
    spy_end = bm_prices.get(("SPY", end_date))
    qqq_start = bm_prices.get(("QQQ", start_date))
    qqq_end = bm_prices.get(("QQQ", end_date))
    spy_return = (spy_end / spy_start - 1.0) if spy_start and spy_end else None
    qqq_return = (qqq_end / qqq_start - 1.0) if qqq_start and qqq_end else None

    days = (end_date - start_date).days
    years = days / 365.25
    cagr = (end_equity / start_equity) ** (1 / years) - 1.0 if years > 0 else 0.0

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "initial_capital": INITIAL_CAPITAL,
        "end_equity": round(end_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "total_trades": len(sell_trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_pnl_pct": round(avg_pnl * 100, 2),
        "avg_holding_days": round(avg_hold, 1),
        "spy_return_pct": round(spy_return * 100, 2) if spy_return is not None else None,
        "qqq_return_pct": round(qqq_return * 100, 2) if qqq_return is not None else None,
        "excess_vs_spy_pct": round((total_return - (spy_return or 0)) * 100, 2),
        "excess_vs_qqq_pct": round((total_return - (qqq_return or 0)) * 100, 2),
        "strategy": {
            "source": source,
            "top_n": TOP_N,
            "min_grade": MIN_GRADE,
            "min_score": MIN_SCORE,
            "budget_per_symbol": BUDGET_PER_SYMBOL,
            "max_daily_new_buys": MAX_DAILY_NEW_BUYS,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "trailing_stop": TRAILING_STOP_PCT,
            "max_holding_days": MAX_HOLDING_DAYS,
            "rank_exit_threshold": RANK_EXIT_THRESHOLD,
            "cooldown_days": COOLDOWN_DAYS,
        },
    }


# =============================================================================
# 결과 출력
# =============================================================================

def save_results(
    output_dir: Path,
    trades: List[Trade],
    daily_snapshots: List[DailySnapshot],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    (output_dir / "portfolio_backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # equity_curve.csv
    with (output_dir / "portfolio_backtest_equity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "cash", "position_value", "equity", "n_positions",
                    "spy_close", "qqq_close"])
        for s in daily_snapshots:
            w.writerow([s.snapshot_date, round(s.cash, 2), round(s.position_value, 2),
                        round(s.equity, 2), s.n_positions, s.spy_close, s.qqq_close])

    # trades.csv
    with (output_dir / "portfolio_backtest_trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "side", "qty", "price", "amount",
                    "reason", "pnl_pct", "pnl_amount", "holding_days"])
        for t in trades:
            w.writerow([t.trade_date, t.symbol, t.side, t.qty,
                        round(t.price, 4), round(t.amount, 2), t.reason,
                        round(t.pnl_pct * 100, 2), round(t.pnl_amount, 2), t.holding_days])

    LOGGER.info("Results saved to %s", output_dir)


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("  US 포트폴리오 백테스트 결과")
    print("=" * 55)
    print(f"  기간:           {summary['start_date']} ~ {summary['end_date']}")
    print(f"  초기 자본:      ${summary['initial_capital']:,.0f}")
    print(f"  최종 자산:      ${summary['end_equity']:,.0f}")
    print(f"  누적 수익률:    {summary['total_return_pct']:+.2f}%")
    print(f"  CAGR:           {summary['cagr_pct']:+.2f}%")
    print(f"  최대 낙폭:      {summary['max_drawdown_pct']:.2f}%")
    print(f"  총 거래:        {summary['total_trades']}건")
    print(f"  승률:           {summary['win_rate_pct']:.1f}%")
    print(f"  평균 수익:      {summary['avg_pnl_pct']:+.2f}%")
    print(f"  평균 보유일:    {summary['avg_holding_days']:.1f}일")
    if summary.get("spy_return_pct") is not None:
        print(f"\n  SPY 수익률:     {summary['spy_return_pct']:+.2f}%")
        print(f"  QQQ 수익률:     {summary['qqq_return_pct']:+.2f}%")
        print(f"  초과수익(SPY):  {summary['excess_vs_spy_pct']:+.2f}%p")
        print(f"  초과수익(QQQ):  {summary['excess_vs_qqq_pct']:+.2f}%p")
    print("=" * 55 + "\n")


# =============================================================================
# 진입점
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US 포트폴리오 백테스트")
    p.add_argument("--start", default="2023-01-03",
                   help="백테스트 시작일 (YYYY-MM-DD, 기본: 2023-01-03)")
    p.add_argument("--end", default="2026-05-18",
                   help="백테스트 종료일 (YYYY-MM-DD, 기본: 2026-05-18)")
    p.add_argument("--source", default=RANKING_SOURCE,
                   help=f"랭킹 소스 (기본: {RANKING_SOURCE})")
    p.add_argument("--output-dir", default="outputs/us_portfolio_backtest",
                   help="결과 출력 디렉토리")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)

    LOGGER.info("백테스트 시작: %s ~ %s (source=%s)", start_date, end_date, args.source)

    engine = get_us_engine()
    rankings = load_rankings(engine, source=args.source)
    prices = load_prices(engine)
    bm_prices = load_benchmark_prices(engine)

    trades, daily_snapshots, _ = run_backtest(
        rankings=rankings,
        prices=prices,
        bm_prices=bm_prices,
        start_date=start_date,
        end_date=end_date,
    )

    summary = compute_summary(
        trades=trades,
        daily_snapshots=daily_snapshots,
        bm_prices=bm_prices,
        start_date=start_date,
        end_date=end_date,
        source=args.source,
    )

    save_results(output_dir, trades, daily_snapshots, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
