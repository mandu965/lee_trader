"""
backtest_us_macro_overlay.py

Phase 4 — US Macro Overlay Backtest Verification

Compares two strategies over the available ranking history period:
  Strategy A (baseline) : existing ranking-based TopN selection
  Strategy B (overlay)  : same selection with US macro overlay applied
                          (buy_blocked excluded, scores adjusted)

Holding-period returns are computed for 1D / 5D / 20D / 60D.

⚠ BACKTEST / REPORT ONLY — 실제 주문/추천/RULE 실행 로직과 기존 추천 점수에는
  영향을 주지 않습니다.

Usage:
    python backtest_us_macro_overlay.py
    python backtest_us_macro_overlay.py --start-date 2026-04-01 --end-date 2026-05-08
    python backtest_us_macro_overlay.py --top-n 5,10 --holding-days 1,5,20,60
    python backtest_us_macro_overlay.py --no-db   # skip DB write
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
from db import get_engine
from compute_us_macro_overlay_shadow import _apply_overlay_rules, _Cfg as OverlayCfg

LOGGER = logging.getLogger("backtest_us_macro_overlay")

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
UNIVERSE_CSV = ROOT / "data" / "universe.csv"

# ─────────────────────────────────────────────────────────────────────────────
# DB DDL (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_RESULT_TABLE_SQL = text("""
CREATE TABLE IF NOT EXISTS research.us_macro_overlay_backtest_result (
    run_id             VARCHAR(100) NOT NULL,
    run_date           DATE         NOT NULL,
    start_date         DATE         NOT NULL,
    end_date           DATE         NOT NULL,
    strategy_name      VARCHAR(100) NOT NULL,

    top_n              INTEGER,
    holding_days       INTEGER,

    total_return       NUMERIC(18,6),
    avg_return         NUMERIC(18,6),
    median_return      NUMERIC(18,6),
    win_rate           NUMERIC(10,4),
    mdd                NUMERIC(18,6),
    volatility         NUMERIC(18,6),

    trade_count        INTEGER,
    signal_dates       INTEGER,
    blocked_count      INTEGER,
    avoided_loss       NUMERIC(18,6),
    missed_gain        NUMERIC(18,6),

    win_rate_1d        NUMERIC(10,4),
    win_rate_5d        NUMERIC(10,4),
    win_rate_20d       NUMERIC(10,4),
    win_rate_60d       NUMERIC(10,4),

    avg_return_1d      NUMERIC(18,6),
    avg_return_5d      NUMERIC(18,6),
    avg_return_20d     NUMERIC(18,6),
    avg_return_60d     NUMERIC(18,6),

    summary            TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (run_id, strategy_name, top_n, holding_days)
)
""")

_UPSERT_RESULT_SQL = text("""
INSERT INTO research.us_macro_overlay_backtest_result (
    run_id, run_date, start_date, end_date, strategy_name,
    top_n, holding_days,
    total_return, avg_return, median_return, win_rate, mdd, volatility,
    trade_count, signal_dates, blocked_count, avoided_loss, missed_gain,
    win_rate_1d, win_rate_5d, win_rate_20d, win_rate_60d,
    avg_return_1d, avg_return_5d, avg_return_20d, avg_return_60d,
    summary
) VALUES (
    :run_id, :run_date, :start_date, :end_date, :strategy_name,
    :top_n, :holding_days,
    :total_return, :avg_return, :median_return, :win_rate, :mdd, :volatility,
    :trade_count, :signal_dates, :blocked_count, :avoided_loss, :missed_gain,
    :win_rate_1d, :win_rate_5d, :win_rate_20d, :win_rate_60d,
    :avg_return_1d, :avg_return_5d, :avg_return_20d, :avg_return_60d,
    :summary
)
ON CONFLICT (run_id, strategy_name, top_n, holding_days) DO UPDATE SET
    total_return  = EXCLUDED.total_return,
    avg_return    = EXCLUDED.avg_return,
    summary       = EXCLUDED.summary,
    created_at    = CURRENT_TIMESTAMP
""")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_ranking_history(engine: Any, start_date: date, end_date: date) -> pd.DataFrame:
    """Load research.ranking_history for the backtest period."""
    sql = text("""
        SELECT as_of_date, code, final_score, rank
        FROM research.ranking_history
        WHERE as_of_date BETWEEN :start_date AND :end_date
          AND horizon_days = 60
        ORDER BY as_of_date, rank
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start_date": start_date, "end_date": end_date})
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
    if df.empty:
        return df
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df


def _load_macro_features(engine: Any, start_date: date, end_date: date) -> pd.DataFrame:
    """Load signal.us_macro_feature_daily mapped to kr_apply_date."""
    # kr_apply_date = the Korean date this US signal applies to
    # We extend lookback slightly for calendar edge cases
    extended_start = start_date - timedelta(days=5)
    sql = text("""
        SELECT kr_apply_date, us_trade_date,
               spy_ret_1d, qqq_ret_1d, semiconductor_ret_1d,
               vix_ret_1d, dxy_ret_1d, tnx_ret_1d,
               sector_breadth, top_sector, bottom_sector,
               risk_on_flag, risk_off_flag, vix_spike_flag,
               semiconductor_strength_flag,
               macro_status, macro_summary
        FROM signal.us_macro_feature_daily
        WHERE kr_apply_date BETWEEN :start_date AND :end_date
        ORDER BY kr_apply_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start_date": extended_start, "end_date": end_date + timedelta(days=5)})
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
    if df.empty:
        return df
    df["kr_apply_date"] = pd.to_datetime(df["kr_apply_date"]).dt.date
    df["us_trade_date"] = pd.to_datetime(df["us_trade_date"]).dt.date
    for col in ["spy_ret_1d", "qqq_ret_1d", "semiconductor_ret_1d",
                "vix_ret_1d", "dxy_ret_1d", "tnx_ret_1d", "sector_breadth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_prices(engine: Any, codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Load public.prices_adjusted for the given codes and date range."""
    # Extend end_date to cover 60D holding period returns
    extended_end = end_date + timedelta(days=100)
    sql = text("""
        SELECT date, code, adj_close
        FROM public.prices_adjusted
        WHERE code = ANY(:codes)
          AND date BETWEEN :start_date AND :end_date
        ORDER BY date, code
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {
            "codes": codes,
            "start_date": start_date,
            "end_date": extended_end,
        })
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    return df


def _load_universe_sector() -> dict[str, str]:
    """Return {code: sector} mapping from universe.csv."""
    if not UNIVERSE_CSV.exists():
        return {}
    df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
    if "sector" not in df.columns:
        return {}
    return dict(zip(df["code"].str.zfill(6), df["sector"].fillna("")))


# ─────────────────────────────────────────────────────────────────────────────
# Return computation
# ─────────────────────────────────────────────────────────────────────────────

def _build_price_pivot(price_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot price data: rows=date, cols=code, values=adj_close."""
    return price_df.pivot(index="date", columns="code", values="adj_close").sort_index()


def _holding_return(price_pivot: pd.DataFrame, code: str, signal_date: date, holding_days: int) -> float | None:
    """Compute N-day return: price[signal_date + N trading days] / price[signal_date] - 1."""
    if code not in price_pivot.columns:
        return None
    dates_after = [d for d in price_pivot.index if d >= signal_date]
    if len(dates_after) < holding_days + 1:
        return None
    entry_date = dates_after[0]
    exit_date = dates_after[holding_days] if holding_days < len(dates_after) else None
    if exit_date is None:
        return None
    p0 = price_pivot.loc[entry_date, code]
    p1 = price_pivot.loc[exit_date, code]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return float(p1 / p0 - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay application (per signal date)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overlay_to_candidates(
    candidates: pd.DataFrame,
    macro_row: dict | None,
    sector_map: dict[str, str],
    cfg: OverlayCfg,
) -> pd.DataFrame:
    """Apply macro overlay rules to a day's candidates.

    Returns DataFrame with extra columns:
      macro_adjustment, adjusted_score, buy_blocked_flag, overlay_reason
    """
    rows = []
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        sector = sector_map.get(code, "")
        original_score = float(row["final_score"]) if pd.notna(row["final_score"]) else None

        if macro_row is None:
            result = {
                "code": code,
                "macro_adjustment": 0.0,
                "adjusted_score": original_score,
                "buy_blocked_flag": False,
                "overlay_reason": "[NO_MACRO_DATA]",
            }
        else:
            r = _apply_overlay_rules(
                code=code,
                name=None,
                sector=sector,
                original_score=original_score,
                is_buy_candidate=True,
                macro=macro_row,
                cfg=cfg,
            )
            result = {
                "code": code,
                "macro_adjustment": r["macro_adjustment"],
                "adjusted_score": r["adjusted_score"],
                "buy_blocked_flag": r["buy_blocked_flag"],
                "overlay_reason": r["overlay_reason"],
            }
        rows.append(result)

    overlay_df = pd.DataFrame(rows)
    return candidates.merge(overlay_df, on="code", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics calculation
# ─────────────────────────────────────────────────────────────────────────────

def _calc_mdd(returns_series: list[float]) -> float:
    """Compute max drawdown from a list of per-period returns."""
    if not returns_series:
        return 0.0
    cum = np.cumprod([1 + r for r in returns_series])
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    return float(np.min(drawdowns))


def _aggregate_metrics(trade_returns: list[float], blocked_returns: list[float]) -> dict:
    """Build metrics dict from per-trade return lists."""
    n = len(trade_returns)
    if n == 0:
        return {
            "avg_return": None, "median_return": None, "total_return": None,
            "win_rate": None, "mdd": None, "volatility": None,
            "trade_count": 0, "avoided_loss": None, "missed_gain": None,
        }
    arr = np.array(trade_returns)
    avg_r = float(np.mean(arr))
    compound = float(np.prod(1 + arr) - 1)
    mdd = _calc_mdd(trade_returns)
    vol = float(np.std(arr)) if n > 1 else 0.0

    avoided_loss = sum(r for r in blocked_returns if r < 0) if blocked_returns else 0.0
    missed_gain = sum(r for r in blocked_returns if r >= 0) if blocked_returns else 0.0

    return {
        "avg_return": avg_r,
        "median_return": float(np.median(arr)),
        "total_return": compound,
        "win_rate": float(np.mean(arr > 0)),
        "mdd": mdd,
        "volatility": vol,
        "trade_count": n,
        "avoided_loss": round(avoided_loss, 6),
        "missed_gain": round(missed_gain, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    ranking_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    sector_map: dict[str, str],
    top_n_list: list[int],
    holding_days_list: list[int],
) -> dict[str, Any]:
    """Run Strategy A and B for all (top_n, holding_days) combinations.

    Returns nested dict: results[top_n][holding_days][strategy] = metrics
    """
    cfg = OverlayCfg()

    # Build macro lookup by kr_apply_date
    macro_by_date: dict[date, dict] = {}
    for _, row in macro_df.iterrows():
        macro_by_date[row["kr_apply_date"]] = row.to_dict()

    signal_dates = sorted(ranking_df["as_of_date"].unique())
    LOGGER.info("Backtest: %d signal dates, %d top-N settings, %d holding periods",
                len(signal_dates), len(top_n_list), len(holding_days_list))

    # Pre-compute overlay for all signal dates
    overlay_cache: dict[date, pd.DataFrame] = {}
    for sig_date in signal_dates:
        day_candidates = ranking_df[ranking_df["as_of_date"] == sig_date].copy()
        macro_row = macro_by_date.get(sig_date)

        # Find nearest macro (within 3 days) if exact date missing
        if macro_row is None:
            for delta in range(1, 4):
                macro_row = macro_by_date.get(sig_date - timedelta(days=delta))
                if macro_row is not None:
                    break

        overlaid = _apply_overlay_to_candidates(day_candidates, macro_row, sector_map, cfg)
        overlay_cache[sig_date] = overlaid

    results: dict = {}

    for top_n in top_n_list:
        results[top_n] = {}
        for holding_days in holding_days_list:
            LOGGER.info("  Computing top_n=%d holding=%d ...", top_n, holding_days)

            a_returns: list[float] = []
            b_returns: list[float] = []
            blocked_actual_returns: list[float] = []  # what blocked stocks actually did
            blocked_count_total = 0
            valid_signal_dates = 0

            per_holding: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}

            for sig_date in signal_dates:
                overlaid = overlay_cache[sig_date]

                # Strategy A: top-N by original final_score
                top_a = (
                    overlaid
                    .sort_values("final_score", ascending=False)
                    .head(top_n)
                )

                # Strategy B: exclude blocked, then top-N by adjusted_score
                not_blocked = overlaid[~overlaid["buy_blocked_flag"].fillna(False)]
                blocked_stocks = overlaid[overlaid["buy_blocked_flag"].fillna(False)]
                top_b = (
                    not_blocked
                    .sort_values("adjusted_score", ascending=False)
                    .head(top_n)
                )

                blocked_count_total += len(blocked_stocks)

                # Compute returns for this holding period
                a_day_returns = []
                for _, row in top_a.iterrows():
                    r = _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, holding_days)
                    if r is not None:
                        a_day_returns.append(r)

                b_day_returns = []
                for _, row in top_b.iterrows():
                    r = _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, holding_days)
                    if r is not None:
                        b_day_returns.append(r)

                # Blocked stocks actual returns
                for _, row in blocked_stocks.iterrows():
                    r = _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, holding_days)
                    if r is not None:
                        blocked_actual_returns.append(r)

                if a_day_returns:
                    avg_a = float(np.mean(a_day_returns))
                    a_returns.append(avg_a)
                    valid_signal_dates += 1

                if b_day_returns:
                    avg_b = float(np.mean(b_day_returns))
                    b_returns.append(avg_b)

                # Collect per-holding sub-metrics (for all-holding aggregate)
                for hd in [1, 5, 20, 60]:
                    if hd == holding_days:
                        if a_day_returns:
                            per_holding[hd].extend(a_day_returns)

            metrics_a = _aggregate_metrics(a_returns, [])
            metrics_b = _aggregate_metrics(b_returns, blocked_actual_returns)
            metrics_b["blocked_count"] = blocked_count_total

            results[top_n][holding_days] = {
                "A": {**metrics_a, "blocked_count": 0, "signal_dates": valid_signal_dates},
                "B": {**metrics_b, "signal_dates": valid_signal_dates},
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-holding report (cross-holding summary)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cross_holding_metrics(
    ranking_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    sector_map: dict[str, str],
    top_n: int,
    all_holding_days: list[int],
) -> dict[str, dict[str, float]]:
    """For each strategy, compute avg/win_rate across all holding periods."""
    cfg = OverlayCfg()
    macro_by_date: dict[date, dict] = {}
    for _, row in macro_df.iterrows():
        macro_by_date[row["kr_apply_date"]] = row.to_dict()

    signal_dates = sorted(ranking_df["as_of_date"].unique())

    summary: dict[str, dict] = {"A": {}, "B": {}}
    for hd in all_holding_days:
        a_rets, b_rets = [], []
        for sig_date in signal_dates:
            day_cands = ranking_df[ranking_df["as_of_date"] == sig_date].copy()
            macro_row = macro_by_date.get(sig_date)
            if macro_row is None:
                for delta in range(1, 4):
                    macro_row = macro_by_date.get(sig_date - timedelta(days=delta))
                    if macro_row is not None:
                        break
            overlaid = _apply_overlay_to_candidates(day_cands, macro_row, sector_map, cfg)

            top_a = overlaid.sort_values("final_score", ascending=False).head(top_n)
            top_b = overlaid[~overlaid["buy_blocked_flag"].fillna(False)].sort_values("adjusted_score", ascending=False).head(top_n)

            a_day = [r for _, row in top_a.iterrows() if (r := _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, hd)) is not None]
            b_day = [r for _, row in top_b.iterrows() if (r := _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, hd)) is not None]

            if a_day:
                a_rets.append(float(np.mean(a_day)))
            if b_day:
                b_rets.append(float(np.mean(b_day)))

        summary["A"][hd] = {"avg": np.mean(a_rets) if a_rets else None, "win_rate": np.mean(np.array(a_rets) > 0) if a_rets else None}
        summary["B"][hd] = {"avg": np.mean(b_rets) if b_rets else None, "win_rate": np.mean(np.array(b_rets) > 0) if b_rets else None}

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Risk-Off block analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_risk_off_blocks(
    ranking_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict:
    """Analyze what actually happened to blocked stocks (1D / 5D / 20D / 60D)."""
    cfg = OverlayCfg()
    macro_by_date = {row["kr_apply_date"]: row.to_dict() for _, row in macro_df.iterrows()}
    signal_dates = sorted(ranking_df["as_of_date"].unique())

    blocked_returns: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}
    block_days = []

    for sig_date in signal_dates:
        day_cands = ranking_df[ranking_df["as_of_date"] == sig_date].copy()
        macro_row = macro_by_date.get(sig_date)
        if macro_row is None:
            for delta in range(1, 4):
                macro_row = macro_by_date.get(sig_date - timedelta(days=delta))
                if macro_row is not None:
                    break
        overlaid = _apply_overlay_to_candidates(day_cands, macro_row, sector_map, cfg)
        blocked = overlaid[overlaid["buy_blocked_flag"].fillna(False)]
        if len(blocked) > 0:
            block_days.append(sig_date)
            for _, row in blocked.iterrows():
                for hd in [1, 5, 20, 60]:
                    r = _holding_return(price_pivot, str(row["code"]).zfill(6), sig_date, hd)
                    if r is not None:
                        blocked_returns[hd].append(r)

    result = {"block_days": len(block_days), "total_blocked_stocks": len(blocked_returns[1])}
    for hd in [1, 5, 20, 60]:
        rets = blocked_returns[hd]
        if rets:
            arr = np.array(rets)
            result[f"blocked_{hd}d_avg"] = float(np.mean(arr))
            result[f"blocked_{hd}d_down_pct"] = float(np.mean(arr < 0))
            result[f"blocked_{hd}d_up_pct"] = float(np.mean(arr >= 0))
            result[f"blocked_{hd}d_avoided_loss"] = float(sum(r for r in rets if r < 0))
            result[f"blocked_{hd}d_missed_gain"] = float(sum(r for r in rets if r >= 0))
        else:
            result[f"blocked_{hd}d_avg"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Semiconductor analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_semiconductor(
    ranking_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict:
    cfg = OverlayCfg()
    macro_by_date = {row["kr_apply_date"]: row.to_dict() for _, row in macro_df.iterrows()}
    signal_dates = sorted(ranking_df["as_of_date"].unique())

    semi_positive_returns: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}
    semi_negative_returns: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}
    semi_positive_count = 0
    semi_negative_count = 0

    for sig_date in signal_dates:
        day_cands = ranking_df[ranking_df["as_of_date"] == sig_date].copy()
        macro_row = macro_by_date.get(sig_date)
        if macro_row is None:
            for delta in range(1, 4):
                macro_row = macro_by_date.get(sig_date - timedelta(days=delta))
                if macro_row is not None:
                    break
        overlaid = _apply_overlay_to_candidates(day_cands, macro_row, sector_map, cfg)

        for _, row in overlaid.iterrows():
            code = str(row["code"]).zfill(6)
            reason = str(row.get("overlay_reason", ""))
            adj = row.get("macro_adjustment", 0) or 0

            if "SEMI_POSITIVE" in reason and adj > 0:
                semi_positive_count += 1
                for hd in [1, 5, 20, 60]:
                    r = _holding_return(price_pivot, code, sig_date, hd)
                    if r is not None:
                        semi_positive_returns[hd].append(r)

            elif "SEMI_NEGATIVE" in reason and adj < 0:
                semi_negative_count += 1
                for hd in [1, 5, 20, 60]:
                    r = _holding_return(price_pivot, code, sig_date, hd)
                    if r is not None:
                        semi_negative_returns[hd].append(r)

    result = {
        "semi_positive_count": semi_positive_count,
        "semi_negative_count": semi_negative_count,
    }
    for hd in [1, 5, 20, 60]:
        pos = semi_positive_returns[hd]
        neg = semi_negative_returns[hd]
        result[f"semi_positive_{hd}d_avg"] = float(np.mean(pos)) if pos else None
        result[f"semi_negative_{hd}d_avg"] = float(np.mean(neg)) if neg else None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float | None, pct: bool = True, dec: int = 2) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v*100:+.{dec}f}%"
    return f"{v:.{dec}f}"


def _fmt_plain(v: float | None, dec: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{dec}f}"


def generate_markdown_report(
    run_id: str,
    start_date: date,
    end_date: date,
    results: dict,
    risk_off_analysis: dict,
    semi_analysis: dict,
    cross_holding: dict,
    data_stats: dict,
    top_n_list: list[int],
    holding_days_list: list[int],
) -> str:
    today_str = date.today().isoformat()
    lines = [
        "# US Macro Overlay Backtest Report",
        "",
        "## 1. 실행 정보",
        "",
        f"- **실행일**: {today_str}",
        f"- **Run ID**: {run_id}",
        f"- **분석 기간**: {start_date} ~ {end_date}",
        f"- **사용 데이터**: research.ranking_history, signal.us_macro_feature_daily, public.prices_adjusted",
        f"- **TopN 기준**: {', '.join(map(str, top_n_list))}",
        f"- **보유 기간**: {', '.join(f'{d}D' for d in holding_days_list)}",
        "",
        "---",
        "",
        "## 2. 데이터 정합성 점검",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| us_macro_feature_daily 건수 | {data_stats.get('macro_rows', 'n/a')} |",
        f"| 국내 ranking_history 건수 | {data_stats.get('ranking_rows', 'n/a')} |",
        f"| 신호 날짜 수 | {data_stats.get('signal_dates', 'n/a')} |",
        f"| 매크로 매핑된 날짜 수 | {data_stats.get('macro_matched_dates', 'n/a')} |",
        f"| 가격 데이터 종목 수 | {data_stats.get('price_codes', 'n/a')} |",
        f"| 가격 데이터 기간 | {data_stats.get('price_start', 'n/a')} ~ {data_stats.get('price_end', 'n/a')} |",
        f"| 매크로 누락 날짜 수 | {data_stats.get('missing_macro_dates', 0)} |",
        "",
        "> ⚠ **제약**: ranking_history는 {start_date} 이후 데이터만 존재합니다. "
        "백테스트 기간이 짧을 수 있습니다.",
        "",
        "---",
    ]

    for top_n in top_n_list:
        lines += [
            "",
            f"## 3. Top{top_n} 전략 비교",
            "",
        ]
        for holding_days in holding_days_list:
            res = results.get(top_n, {}).get(holding_days, {})
            a = res.get("A", {})
            b = res.get("B", {})

            lines += [
                f"### Top{top_n} | 보유기간 {holding_days}D",
                "",
                "| 지표 | 기존 전략 (A) | Overlay 전략 (B) | 차이 (B-A) |",
                "|---|---|---|---|",
            ]

            def diff(va, vb):
                if va is None or vb is None:
                    return "n/a"
                return f"{(vb-va)*100:+.2f}%p"

            metrics = [
                ("누적 수익률", "total_return", True),
                ("평균 수익률", "avg_return", True),
                ("중앙값 수익률", "median_return", True),
                ("승률", "win_rate", True),
                ("MDD", "mdd", True),
                ("변동성", "volatility", True),
            ]
            for label, key, is_pct in metrics:
                va = a.get(key)
                vb = b.get(key)
                lines.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {diff(va, vb)} |")

            lines += [
                f"| 거래 횟수 | {a.get('trade_count', 'n/a')} | {b.get('trade_count', 'n/a')} | - |",
                f"| 신호 날짜 수 | {a.get('signal_dates', 'n/a')} | - | - |",
                f"| 매수 차단 횟수 | - | {b.get('blocked_count', 0)} | - |",
                f"| 피한 손실 (차단 종목 합계) | - | {_fmt(b.get('avoided_loss'))} | - |",
                f"| 놓친 수익 (차단 종목 합계) | - | {_fmt(b.get('missed_gain'))} | - |",
                "",
            ]

    # Cross-holding summary
    lines += [
        "---",
        "",
        "## 4. 보유기간별 평균 수익률 / 승률 요약",
        "",
    ]
    for top_n in top_n_list:
        ch = cross_holding.get(top_n, {})
        lines += [f"### Top{top_n}", "", "| 보유기간 | 전략A 평균수익 | 전략A 승률 | 전략B 평균수익 | 전략B 승률 |", "|---|---|---|---|---|"]
        for hd in holding_days_list:
            a_ch = ch.get("A", {}).get(hd, {})
            b_ch = ch.get("B", {}).get(hd, {})
            lines.append(f"| {hd}D | {_fmt(a_ch.get('avg'))} | {_fmt(a_ch.get('win_rate'))} | {_fmt(b_ch.get('avg'))} | {_fmt(b_ch.get('win_rate'))} |")
        lines.append("")

    # Risk-Off analysis
    lines += [
        "---",
        "",
        "## 5. Risk-Off 차단 효과 분석",
        "",
        f"- **차단 발생 날짜 수**: {risk_off_analysis.get('block_days', 0)}",
        f"- **총 차단 종목 수**: {risk_off_analysis.get('total_blocked_stocks', 0)}",
        "",
        "| 보유기간 | 차단종목 평균수익 | 하락 비율 | 상승 비율 | 피한 손실 | 놓친 수익 |",
        "|---|---|---|---|---|---|",
    ]
    for hd in [1, 5, 20, 60]:
        avg = risk_off_analysis.get(f"blocked_{hd}d_avg")
        down = risk_off_analysis.get(f"blocked_{hd}d_down_pct")
        up = risk_off_analysis.get(f"blocked_{hd}d_up_pct")
        al = risk_off_analysis.get(f"blocked_{hd}d_avoided_loss")
        mg = risk_off_analysis.get(f"blocked_{hd}d_missed_gain")
        lines.append(f"| {hd}D | {_fmt(avg)} | {_fmt(down)} | {_fmt(up)} | {_fmt(al)} | {_fmt(mg)} |")

    lines += [
        "",
        "### Risk-Off 차단 판단",
        "",
    ]
    total_blocked = risk_off_analysis.get("total_blocked_stocks", 0)
    avg_1d = risk_off_analysis.get("blocked_1d_avg")
    down_1d = risk_off_analysis.get("blocked_1d_down_pct")
    if total_blocked == 0:
        lines += ["> 차단 발생 없음 — Risk-Off 조건이 분석 기간 내 미충족", ""]
    elif avg_1d is not None and avg_1d < 0:
        lines += [f"> 차단 종목 평균 1D 수익률 {_fmt(avg_1d)} (하락 비율 {_fmt(down_1d)}) → 차단이 손실 회피에 기여했을 가능성 있음", ""]
    else:
        lines += [f"> 차단 종목 평균 1D 수익률 {_fmt(avg_1d)} → 차단으로 인한 기회손실 발생 가능성 있음", ""]

    # Semiconductor analysis
    lines += [
        "---",
        "",
        "## 6. 반도체 가산 / 감점 효과 분석",
        "",
        f"- **반도체 가산 적용 종목 수**: {semi_analysis.get('semi_positive_count', 0)}",
        f"- **반도체 감점 적용 종목 수**: {semi_analysis.get('semi_negative_count', 0)}",
        "",
        "| | 1D 평균수익 | 5D 평균수익 | 20D 평균수익 | 60D 평균수익 |",
        "|---|---|---|---|---|",
    ]
    lines.append("| 반도체 가산 후보 | " + " | ".join(_fmt(semi_analysis.get(f"semi_positive_{hd}d_avg")) for hd in [1, 5, 20, 60]) + " |")
    lines.append("| 반도체 감점 후보 | " + " | ".join(_fmt(semi_analysis.get(f"semi_negative_{hd}d_avg")) for hd in [1, 5, 20, 60]) + " |")
    lines.append("")

    # Core questions
    lines += [
        "---",
        "",
        "## 7. 핵심 검증 질문 답변",
        "",
    ]

    # Q1: Did Risk-Off block save losses?
    avg_1d_val = risk_off_analysis.get("blocked_1d_avg")
    down_pct_val = risk_off_analysis.get("blocked_1d_down_pct")
    if total_blocked == 0:
        q1 = "분석 기간 내 Risk-Off 차단 발생 없음. 조건 임계값 검토 필요."
    elif avg_1d_val is not None and avg_1d_val < -0.01:
        q1 = f"차단 종목 평균 1D 수익률 {_fmt(avg_1d_val)}. **손실 회피에 기여한 것으로 추정됨.**"
    elif avg_1d_val is not None and avg_1d_val > 0.01:
        q1 = f"차단 종목 평균 1D 수익률 {_fmt(avg_1d_val)}. **기회손실 발생 가능성 높음.** 차단 기준 재검토 권장."
    else:
        q1 = f"차단 종목 평균 1D 수익률 {_fmt(avg_1d_val)}. 효과가 미미하거나 분석 기간이 짧아 판단 어려움."

    # Compare Strategy A vs B total returns
    first_top_n = top_n_list[0]
    primary_hd = 20  # use 20D as primary comparison
    if primary_hd not in holding_days_list:
        primary_hd = holding_days_list[0]
    res_ref = results.get(first_top_n, {}).get(primary_hd, {})
    a_total = res_ref.get("A", {}).get("total_return")
    b_total = res_ref.get("B", {}).get("total_return")
    a_mdd = res_ref.get("A", {}).get("mdd")
    b_mdd = res_ref.get("B", {}).get("mdd")

    if a_total is not None and b_total is not None:
        ret_diff = b_total - a_total
        ret_q = f"B {_fmt(b_total)} vs A {_fmt(a_total)} → {'개선' if ret_diff > 0 else '감소'} ({_fmt(ret_diff)})"
    else:
        ret_q = "데이터 부족으로 비교 불가"

    if a_mdd is not None and b_mdd is not None:
        mdd_diff = b_mdd - a_mdd
        mdd_q = f"B {_fmt(b_mdd)} vs A {_fmt(a_mdd)} → MDD {'개선' if mdd_diff > 0 else '악화'} ({_fmt(mdd_diff)})"
    else:
        mdd_q = "데이터 부족으로 비교 불가"

    questions = [
        ("1", "Risk-Off 차단이 실제 손실을 줄였는가?", q1),
        ("2", "Risk-Off 차단으로 인해 좋은 매수 기회를 놓친 경우는?", f"놓친 수익 합계: {_fmt(res_ref.get('B', {}).get('missed_gain'))}"),
        ("3", "Macro Overlay 적용 후 MDD가 개선되었는가?", f"Top{first_top_n} {primary_hd}D 기준: {mdd_q}"),
        ("4", "Macro Overlay 적용 후 누적 수익률이 개선되었는가?", f"Top{first_top_n} {primary_hd}D 기준: {ret_q}"),
        ("5", "매매 횟수만 줄고 수익도 같이 줄어든 건 아닌가?", "위 표의 '거래 횟수' vs '누적 수익률' 비교 참조"),
        ("6", "반도체 가산 룰이 실제 수익률 개선으로 이어졌는가?", f"반도체 가산 종목 {semi_analysis.get('semi_positive_count', 0)}건, 1D 평균 {_fmt(semi_analysis.get('semi_positive_1d_avg'))}"),
        ("7", "반도체 감점 룰이 실제 손실 회피에 도움이 되었는가?", f"반도체 감점 종목 {semi_analysis.get('semi_negative_count', 0)}건, 1D 평균 {_fmt(semi_analysis.get('semi_negative_1d_avg'))}"),
        ("8", "Overlay가 기존 추천 랭킹을 과도하게 왜곡하지 않았는가?", "blocked_count 대비 전체 후보 비율 확인 권장"),
        ("9", "통계적으로 의미 있는 결과인가?", f"신호 날짜 수: {data_stats.get('signal_dates', 'n/a')} — 기간이 짧을수록 결론의 신뢰도 낮음"),
        ("10", "실반영해도 될 만큼 충분한 검증이 되었는가?", "최소 2~3개월 Shadow 운영 후 판단 권장 (현재 기간 부족)"),
    ]
    for num, q, ans in questions:
        lines += [f"**Q{num}. {q}**", f"> {ans}", ""]

    # Conclusion
    lines += [
        "---",
        "",
        "## 8. 결론",
        "",
        "### 실반영 가능 여부",
        "",
    ]
    if data_stats.get("signal_dates", 0) < 20:
        lines.append("> ⚠ **분석 기간이 너무 짧습니다 (신호 날짜 {}일).** 최소 60일 이상 Shadow Mode 운영 후 재평가 권장.".format(data_stats.get("signal_dates", 0)))
    elif total_blocked == 0:
        lines.append("> Risk-Off 조건이 한 번도 발동되지 않았습니다. 임계값 재검토 또는 기간 확장 필요.")
    else:
        lines.append("> 현 기간 기준 결과를 참고하되, Phase 5 실반영 전 2~3개월 추가 Shadow 운영을 권장합니다.")

    lines += [
        "",
        "### 추가 검증 필요 사항",
        "",
        "- [ ] ranking_history 기간 확장 후 재실행 (최소 60영업일)",
        "- [ ] RULE 후보 기반 백테스트 추가 (rule_signals.csv 연동)",
        "- [ ] Risk-Off 임계값 조정 후 재검증",
        "- [ ] 반도체 섹터 매핑 정확도 검증",
        "",
        "## 9. 다음 단계 (Phase 5 실반영 전)",
        "",
        "1. Shadow Mode 2~3개월 운영하며 `signal.kr_macro_overlay_log` 데이터 축적",
        "2. 누적 데이터로 이 스크립트 재실행 → 결과 비교",
        "3. MDD 개선 + 손실 거래 감소 + 기회손실 허용 범위 내 확인",
        "4. `US_MACRO_ALLOW_REAL_APPLY=true` 설정 전 운영팀 검토",
        "",
        "---",
        "",
        "```",
        "이번 Phase 4 작업은 미국 macro overlay의 효과를 백테스트로 검증하는 작업이며,",
        "실제 주문 생성/실행 로직과 기존 추천 점수에는 영향을 주지 않습니다.",
        "```",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CSV generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_csv_rows(
    run_id: str,
    start_date: date,
    end_date: date,
    results: dict,
    top_n_list: list[int],
    holding_days_list: list[int],
) -> list[dict]:
    rows = []
    for top_n in top_n_list:
        for holding_days in holding_days_list:
            res = results.get(top_n, {}).get(holding_days, {})
            for strategy, metrics in res.items():
                rows.append({
                    "run_id": run_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "strategy": strategy,
                    "top_n": top_n,
                    "holding_days": holding_days,
                    "total_return": metrics.get("total_return"),
                    "avg_return": metrics.get("avg_return"),
                    "median_return": metrics.get("median_return"),
                    "win_rate": metrics.get("win_rate"),
                    "mdd": metrics.get("mdd"),
                    "volatility": metrics.get("volatility"),
                    "trade_count": metrics.get("trade_count"),
                    "signal_dates": metrics.get("signal_dates"),
                    "blocked_count": metrics.get("blocked_count", 0),
                    "avoided_loss": metrics.get("avoided_loss"),
                    "missed_gain": metrics.get("missed_gain"),
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DB write
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_result_table(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(_CREATE_RESULT_TABLE_SQL)


def _write_results_to_db(
    engine: Any,
    run_id: str,
    run_date: date,
    start_date: date,
    end_date: date,
    results: dict,
    top_n_list: list[int],
    holding_days_list: list[int],
) -> int:
    rows = []
    for top_n in top_n_list:
        for holding_days in holding_days_list:
            res = results.get(top_n, {}).get(holding_days, {})
            for strategy_name, metrics in res.items():
                rows.append({
                    "run_id": run_id,
                    "run_date": run_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "strategy_name": strategy_name,
                    "top_n": top_n,
                    "holding_days": holding_days,
                    "total_return": metrics.get("total_return"),
                    "avg_return": metrics.get("avg_return"),
                    "median_return": metrics.get("median_return"),
                    "win_rate": metrics.get("win_rate"),
                    "mdd": metrics.get("mdd"),
                    "volatility": metrics.get("volatility"),
                    "trade_count": metrics.get("trade_count"),
                    "signal_dates": metrics.get("signal_dates"),
                    "blocked_count": metrics.get("blocked_count", 0),
                    "avoided_loss": metrics.get("avoided_loss"),
                    "missed_gain": metrics.get("missed_gain"),
                    "win_rate_1d": None,
                    "win_rate_5d": None,
                    "win_rate_20d": None,
                    "win_rate_60d": None,
                    "avg_return_1d": None,
                    "avg_return_5d": None,
                    "avg_return_20d": None,
                    "avg_return_60d": None,
                    "summary": f"Top{top_n} | {holding_days}D | Strategy {strategy_name}",
                })
    if rows:
        with engine.begin() as conn:
            conn.execute(_UPSERT_RESULT_SQL, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI & orchestration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4: US Macro Overlay Backtest"
    )
    p.add_argument("--start-date", default=None, help="Backtest start date YYYY-MM-DD. Default: 1 year ago (or earliest available)")
    p.add_argument("--end-date", default=None, help="Backtest end date YYYY-MM-DD. Default: today")
    p.add_argument("--top-n", default="5,10", help="Comma-separated TopN list (default: 5,10)")
    p.add_argument("--holding-days", default="1,5,20,60", help="Comma-separated holding days (default: 1,5,20,60)")
    p.add_argument("--no-db", action="store_true", help="Skip writing results to DB")
    p.add_argument("--report-dir", default=str(REPORTS_DIR), help="Directory for output reports")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = date.fromisoformat(args.start_date) if args.start_date else (today - timedelta(days=365))
    top_n_list = [int(x.strip()) for x in args.top_n.split(",")]
    holding_days_list = [int(x.strip()) for x in args.holding_days.split(",")]
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"bt_{today.isoformat().replace('-', '')}_{str(uuid.uuid4())[:8]}"
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 4 Backtest | run_id=%s | %s ~ %s", run_id, start_date, end_date)
    LOGGER.info("TopN: %s | HoldingDays: %s", top_n_list, holding_days_list)
    LOGGER.info("=" * 60)

    engine = get_engine()

    # ── Load data ──
    LOGGER.info("Loading ranking history ...")
    ranking_df = _load_ranking_history(engine, start_date, end_date)
    if ranking_df.empty:
        LOGGER.error("No ranking data found for %s ~ %s. Aborting.", start_date, end_date)
        return

    signal_dates = sorted(ranking_df["as_of_date"].unique())
    actual_start = signal_dates[0]
    actual_end = signal_dates[-1]
    LOGGER.info("Ranking data: %d rows across %d signal dates (%s ~ %s)",
                len(ranking_df), len(signal_dates), actual_start, actual_end)

    LOGGER.info("Loading macro features ...")
    macro_df = _load_macro_features(engine, actual_start, actual_end)
    LOGGER.info("Macro feature rows: %d", len(macro_df))

    macro_dates_set = set(macro_df["kr_apply_date"].unique()) if not macro_df.empty else set()
    missing_macro = [d for d in signal_dates if d not in macro_dates_set and
                     not any(d - timedelta(days=i) in macro_dates_set for i in range(1, 4))]
    LOGGER.info("Signal dates without macro data: %d", len(missing_macro))

    codes = ranking_df["code"].str.zfill(6).unique().tolist()
    LOGGER.info("Loading price data for %d codes ...", len(codes))
    price_df = _load_prices(engine, codes, actual_start, actual_end)
    LOGGER.info("Price data: %d rows", len(price_df))

    price_pivot = _build_price_pivot(price_df)
    sector_map = _load_universe_sector()
    LOGGER.info("Sector map loaded: %d codes", len(sector_map))

    data_stats = {
        "macro_rows": len(macro_df),
        "ranking_rows": len(ranking_df),
        "signal_dates": len(signal_dates),
        "macro_matched_dates": len(signal_dates) - len(missing_macro),
        "missing_macro_dates": len(missing_macro),
        "price_codes": len(price_pivot.columns),
        "price_start": str(price_pivot.index.min()) if not price_pivot.empty else "n/a",
        "price_end": str(price_pivot.index.max()) if not price_pivot.empty else "n/a",
    }

    # ── Run backtest ──
    LOGGER.info("Running backtest ...")
    results = run_backtest(
        ranking_df, macro_df, price_pivot, sector_map,
        top_n_list, holding_days_list,
    )

    # ── Risk-Off analysis ──
    LOGGER.info("Analyzing Risk-Off blocks ...")
    risk_off_analysis = _analyze_risk_off_blocks(ranking_df, macro_df, price_pivot, sector_map)
    LOGGER.info("Risk-Off blocks: %d days, %d stocks", risk_off_analysis["block_days"], risk_off_analysis["total_blocked_stocks"])

    # ── Semiconductor analysis ──
    LOGGER.info("Analyzing semiconductor rules ...")
    semi_analysis = _analyze_semiconductor(ranking_df, macro_df, price_pivot, sector_map)

    # ── Cross-holding summary ──
    cross_holding: dict = {}
    for top_n in top_n_list:
        cross_holding[top_n] = _compute_cross_holding_metrics(
            ranking_df, macro_df, price_pivot, sector_map, top_n, holding_days_list
        )

    # ── Save to DB ──
    if not args.no_db:
        LOGGER.info("Ensuring DB table exists ...")
        _ensure_result_table(engine)
        n_written = _write_results_to_db(
            engine, run_id, today, actual_start, actual_end,
            results, top_n_list, holding_days_list,
        )
        LOGGER.info("Wrote %d result rows to research.us_macro_overlay_backtest_result", n_written)

    # ── Generate reports ──
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = report_dir / f"us_macro_overlay_backtest_{ts}.md"
    csv_path = report_dir / f"us_macro_overlay_backtest_{ts}.csv"

    LOGGER.info("Generating Markdown report → %s", md_path)
    md_content = generate_markdown_report(
        run_id, actual_start, actual_end,
        results, risk_off_analysis, semi_analysis, cross_holding,
        data_stats, top_n_list, holding_days_list,
    )
    md_path.write_text(md_content, encoding="utf-8")

    LOGGER.info("Generating CSV report → %s", csv_path)
    csv_rows = generate_csv_rows(run_id, actual_start, actual_end, results, top_n_list, holding_days_list)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    LOGGER.info("=" * 60)
    LOGGER.info("Phase 4 Backtest Complete")
    LOGGER.info("  MD report : %s", md_path)
    LOGGER.info("  CSV report: %s", csv_path)
    LOGGER.info("")
    LOGGER.info("이번 Phase 4 작업은 미국 macro overlay의 효과를 백테스트로 검증하는 작업이며,")
    LOGGER.info("실제 주문 생성/실행 로직과 기존 추천 점수에는 영향을 주지 않습니다.")
    LOGGER.info("=" * 60)

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"[Phase 4 Backtest] {actual_start} ~ {actual_end} ({len(signal_dates)} signal dates)")
    for top_n in top_n_list:
        for hd in holding_days_list:
            res = results.get(top_n, {}).get(hd, {})
            a = res.get("A", {})
            b = res.get("B", {})
            a_ret = _fmt(a.get("total_return"))
            b_ret = _fmt(b.get("total_return"))
            a_mdd = _fmt(a.get("mdd"))
            b_mdd = _fmt(b.get("mdd"))
            print(f"  Top{top_n} {hd}D | A: ret={a_ret} mdd={a_mdd} | B: ret={b_ret} mdd={b_mdd} | blocked={b.get('blocked_count',0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
