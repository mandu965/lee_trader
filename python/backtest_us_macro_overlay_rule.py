"""
backtest_us_macro_overlay_rule.py

Phase 4 — US Macro Overlay × RULE 전략 백테스트 (3년치)

RULE_TREND_LIQUIDITY_V1 신호(entry_signal / strong_entry_signal)를 기반으로
미국 매크로 overlay를 적용했을 때 성과가 개선되는지 검증합니다.

데이터 소스:
  - data/rule_signals.csv          : RULE 신호 (2023-04-14 ~)
  - data/prices_daily_adjusted.csv : 국내 종목 가격
  - signal.us_macro_feature_daily  : 미국 매크로 피처 (DB)

비교 전략:
  Strategy A: entry_signal=True 후보 → rule_score_v2 기준 TopN 매수
  Strategy B: 동일 후보 중 buy_blocked_flag=Y 제외 → adjusted_score 기준 TopN 매수

⚠ BACKTEST ONLY — 실제 주문/추천/RULE 실행 로직과 기존 추천 점수에는 영향 없음.

Usage:
    python backtest_us_macro_overlay_rule.py
    python backtest_us_macro_overlay_rule.py --start-date 2023-04-14 --end-date 2026-05-08
    python backtest_us_macro_overlay_rule.py --signal-col strong_entry_signal --top-n 5,10
    python backtest_us_macro_overlay_rule.py --no-db
"""

from __future__ import annotations

import argparse
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

LOGGER = logging.getLogger("backtest_us_macro_overlay_rule")

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_SIGNALS_CSV = ROOT / "data" / "rule_signals.csv"
DEFAULT_PRICES_CSV = ROOT / "data" / "prices_daily_adjusted.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_rule_signals(csv_path: Path, signal_col: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load RULE signals for the backtest period.

    Returns rows where signal_col=True within the date range.
    Columns used: date, code, rule_score_v2, sector, entry_signal, strong_entry_signal
    """
    df = pd.read_csv(csv_path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["code"] = df["code"].str.zfill(6)

    # Sanitize signal column
    if signal_col not in df.columns:
        LOGGER.warning("Signal column '%s' not found, falling back to 'entry_signal'", signal_col)
        signal_col = "entry_signal"
    df[signal_col] = df[signal_col].astype(str).str.lower().isin(["true", "1", "yes"])

    df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    LOGGER.info("Rule signals loaded: %d rows, date range %s ~ %s",
                len(df), df["date"].min(), df["date"].max())
    return df


def _load_prices_csv(csv_path: Path, start_date: date, end_date: date) -> pd.DataFrame:
    """Load price data and build pivot (date × code → adj_close)."""
    # Extended end_date to cover 60D holding period returns
    df = pd.read_csv(csv_path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["code"] = df["code"].str.zfill(6)
    df["adj_close"] = pd.to_numeric(df.get("adj_close", df.get("close")), errors="coerce")

    df = df[df["date"] >= start_date].copy()
    LOGGER.info("Price data loaded: %d rows, date range %s ~ %s",
                len(df), df["date"].min(), df["date"].max())
    return df


def _load_macro_features_db(engine: Any, start_date: date, end_date: date) -> pd.DataFrame:
    """Load signal.us_macro_feature_daily for the backtest period."""
    sql = text("""
        SELECT kr_apply_date, us_trade_date,
               spy_ret_1d, qqq_ret_1d, semiconductor_ret_1d,
               vix_ret_1d, dxy_ret_1d, sector_breadth,
               risk_on_flag, risk_off_flag, vix_spike_flag,
               semiconductor_strength_flag,
               macro_status, macro_summary
        FROM signal.us_macro_feature_daily
        WHERE kr_apply_date BETWEEN :start_date AND :end_date
        ORDER BY kr_apply_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {
            "start_date": start_date - timedelta(days=5),
            "end_date": end_date + timedelta(days=3),
        })
        df = pd.DataFrame(rows.fetchall(), columns=rows.keys())
    if df.empty:
        return df
    df["kr_apply_date"] = pd.to_datetime(df["kr_apply_date"]).dt.date
    for col in ["spy_ret_1d", "qqq_ret_1d", "semiconductor_ret_1d",
                "vix_ret_1d", "dxy_ret_1d", "sector_breadth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    LOGGER.info("Macro features loaded: %d rows, %s ~ %s",
                len(df), df["kr_apply_date"].min(), df["kr_apply_date"].max())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Return computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_forward_returns(price_df: pd.DataFrame, holding_days_list: list[int]) -> pd.DataFrame:
    """For each (date, code), compute N-day forward returns.

    Uses adj_close[d+N] / adj_close[d] - 1.
    Returns DataFrame with columns: date, code, ret_Nd for each N.
    """
    pivot = price_df.pivot(index="date", columns="code", values="adj_close").sort_index()
    date_arr = pivot.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(date_arr)}

    records = []
    for code in pivot.columns:
        series = pivot[code].dropna()
        if series.empty:
            continue
        for sig_date in series.index:
            idx = date_to_idx.get(sig_date)
            if idx is None:
                continue
            row: dict = {"date": sig_date, "code": code}
            for hd in holding_days_list:
                exit_idx = idx + hd
                if exit_idx < len(date_arr):
                    exit_date = date_arr[exit_idx]
                    p0 = pivot.at[sig_date, code]
                    p1 = pivot.at[exit_date, code] if exit_date in pivot.index else np.nan
                    if p0 and p1 and not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                        row[f"fwd_{hd}d"] = float(p1 / p0 - 1)
                    else:
                        row[f"fwd_{hd}d"] = np.nan
                else:
                    row[f"fwd_{hd}d"] = np.nan
            records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay application
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_macro(macro_by_date: dict[date, dict], sig_date: date, max_lag: int = 3) -> dict | None:
    for lag in range(max_lag + 1):
        m = macro_by_date.get(sig_date - timedelta(days=lag))
        if m is not None:
            return m
    return None


def _apply_overlay_to_df(
    candidates: pd.DataFrame,
    macro_row: dict | None,
    cfg: OverlayCfg,
) -> pd.DataFrame:
    """Add macro_adjustment, adjusted_score, buy_blocked_flag to candidates."""
    rows = []
    base_score_col = "rule_score_v2" if "rule_score_v2" in candidates.columns else "rule_score"

    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        sector = str(row.get("sector", "") or "")
        original_score = float(row[base_score_col]) if pd.notna(row.get(base_score_col)) else None

        if macro_row is None:
            rows.append({
                "code": code,
                "macro_adjustment": 0.0,
                "adjusted_score": original_score,
                "buy_blocked_flag": False,
                "overlay_reason": "[NO_MACRO]",
            })
        else:
            r = _apply_overlay_rules(
                code=code, name=None, sector=sector,
                original_score=original_score,
                is_buy_candidate=True,
                macro=macro_row, cfg=cfg,
            )
            rows.append({
                "code": code,
                "macro_adjustment": r["macro_adjustment"],
                "adjusted_score": r["adjusted_score"],
                "buy_blocked_flag": r["buy_blocked_flag"],
                "overlay_reason": r["overlay_reason"],
            })

    overlay_df = pd.DataFrame(rows)
    return candidates.merge(overlay_df, on="code", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mdd(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cum = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cum)
    return float(np.min(cum / running_max - 1))


def _metrics(trade_rets: list[float], blocked_rets: list[float] | None = None) -> dict:
    if not trade_rets:
        return {k: None for k in ["total_return", "avg_return", "median_return",
                                   "win_rate", "mdd", "volatility",
                                   "trade_count", "avoided_loss", "missed_gain"]}
    arr = np.array(trade_rets)
    bl = blocked_rets or []
    return {
        "total_return": float(np.clip(np.prod(1 + arr) - 1, -1.0, 50.0)),
        "avg_return": float(np.mean(arr)),
        "median_return": float(np.median(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "mdd": _mdd(trade_rets),
        "volatility": float(np.std(arr)) if len(arr) > 1 else 0.0,
        "trade_count": len(arr),
        "avoided_loss": sum(r for r in bl if r < 0),
        "missed_gain": sum(r for r in bl if r >= 0),
    }


def _sharpe(rets: list[float]) -> float | None:
    if len(rets) < 2:
        return None
    arr = np.array(rets)
    std = np.std(arr)
    return float(np.mean(arr) / std * np.sqrt(252)) if std > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_rule_backtest(
    signals_df: pd.DataFrame,
    forward_ret_df: pd.DataFrame,
    macro_by_date: dict[date, dict],
    signal_col: str,
    top_n_list: list[int],
    holding_days_list: list[int],
) -> dict[str, Any]:
    """Run Strategy A and B over RULE signals.

    Returns results[top_n][holding_days] = {A: metrics, B: metrics}
    """
    cfg = OverlayCfg()
    base_score_col = "rule_score_v2" if "rule_score_v2" in signals_df.columns else "rule_score"

    # Merge forward returns into signals
    merged = signals_df.merge(forward_ret_df, on=["date", "code"], how="left")
    signal_dates = sorted(merged["date"].unique())
    LOGGER.info("Signal dates with forward returns: %d", len(signal_dates))

    results: dict = {}
    for top_n in top_n_list:
        results[top_n] = {}
        for hd in holding_days_list:
            ret_col = f"fwd_{hd}d"
            if ret_col not in merged.columns:
                LOGGER.warning("Column %s not found, skipping", ret_col)
                results[top_n][hd] = {"A": _metrics([]), "B": _metrics([])}
                continue

            LOGGER.info("  top_n=%d holding=%dD ...", top_n, hd)

            a_portfolio_rets: list[float] = []
            b_portfolio_rets: list[float] = []
            blocked_actual_rets: list[float] = []
            blocked_count_total = 0
            valid_dates = 0
            no_macro_dates = 0

            for sig_date in signal_dates:
                day = merged[
                    (merged["date"] == sig_date) & (merged[signal_col])
                ].copy()

                if day.empty:
                    continue

                macro_row = _nearest_macro(macro_by_date, sig_date)
                if macro_row is None:
                    no_macro_dates += 1

                day_overlaid = _apply_overlay_to_df(day, macro_row, cfg)

                # Strategy A: top-N by rule_score_v2, no filter
                top_a = day_overlaid.nlargest(top_n, base_score_col)
                a_rets = top_a[ret_col].dropna().tolist()

                # Strategy B: exclude blocked, top-N by adjusted_score
                not_blocked = day_overlaid[~day_overlaid["buy_blocked_flag"].fillna(False)]
                blocked = day_overlaid[day_overlaid["buy_blocked_flag"].fillna(False)]
                top_b = not_blocked.nlargest(top_n, "adjusted_score")
                b_rets = top_b[ret_col].dropna().tolist()

                blocked_count_total += len(blocked)
                b_actual = blocked[ret_col].dropna().tolist()
                blocked_actual_rets.extend(b_actual)

                if a_rets:
                    a_portfolio_rets.append(float(np.mean(a_rets)))
                    valid_dates += 1

                if b_rets:
                    b_portfolio_rets.append(float(np.mean(b_rets)))

            LOGGER.info("    valid signal dates: %d | no-macro dates: %d | blocked total: %d",
                        valid_dates, no_macro_dates, blocked_count_total)

            m_a = _metrics(a_portfolio_rets)
            m_b = _metrics(b_portfolio_rets, blocked_actual_rets)
            m_a["signal_dates"] = valid_dates
            m_a["no_macro_dates"] = no_macro_dates
            m_a["blocked_count"] = 0
            m_b["signal_dates"] = valid_dates
            m_b["no_macro_dates"] = no_macro_dates
            m_b["blocked_count"] = blocked_count_total
            m_a["sharpe"] = _sharpe(a_portfolio_rets)
            m_b["sharpe"] = _sharpe(b_portfolio_rets)

            results[top_n][hd] = {"A": m_a, "B": m_b}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Risk-Off & Semiconductor analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_risk_off(
    signals_df: pd.DataFrame,
    forward_ret_df: pd.DataFrame,
    macro_by_date: dict[date, dict],
    signal_col: str,
) -> dict:
    cfg = OverlayCfg()
    merged = signals_df.merge(forward_ret_df, on=["date", "code"], how="left")

    block_days = 0
    blocked_rets: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}

    for sig_date in sorted(merged["date"].unique()):
        day = merged[(merged["date"] == sig_date) & (merged[signal_col])].copy()
        if day.empty:
            continue
        macro_row = _nearest_macro(macro_by_date, sig_date)
        overlaid = _apply_overlay_to_df(day, macro_row, cfg)
        blocked = overlaid[overlaid["buy_blocked_flag"].fillna(False)]
        if len(blocked) > 0:
            block_days += 1
            for hd in [1, 5, 20, 60]:
                col = f"fwd_{hd}d"
                if col in blocked.columns:
                    blocked_rets[hd].extend(blocked[col].dropna().tolist())

    result = {"block_days": block_days, "total_blocked": sum(len(v) for v in blocked_rets.values()) // 4}
    for hd in [1, 5, 20, 60]:
        rets = blocked_rets[hd]
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


def _analyze_semiconductor(
    signals_df: pd.DataFrame,
    forward_ret_df: pd.DataFrame,
    macro_by_date: dict[date, dict],
    signal_col: str,
) -> dict:
    cfg = OverlayCfg()
    merged = signals_df.merge(forward_ret_df, on=["date", "code"], how="left")

    semi_pos: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}
    semi_neg: dict[int, list[float]] = {1: [], 5: [], 20: [], 60: []}
    pos_count = neg_count = 0

    for sig_date in sorted(merged["date"].unique()):
        day = merged[(merged["date"] == sig_date) & (merged[signal_col])].copy()
        if day.empty:
            continue
        macro_row = _nearest_macro(macro_by_date, sig_date)
        overlaid = _apply_overlay_to_df(day, macro_row, cfg)

        for _, row in overlaid.iterrows():
            reason = str(row.get("overlay_reason", ""))
            adj = row.get("macro_adjustment", 0) or 0
            if "SEMI_POSITIVE" in reason and adj > 0:
                pos_count += 1
                for hd in [1, 5, 20, 60]:
                    v = row.get(f"fwd_{hd}d")
                    if pd.notna(v):
                        semi_pos[hd].append(float(v))
            elif "SEMI_NEGATIVE" in reason and adj < 0:
                neg_count += 1
                for hd in [1, 5, 20, 60]:
                    v = row.get(f"fwd_{hd}d")
                    if pd.notna(v):
                        semi_neg[hd].append(float(v))

    result = {"semi_positive_count": pos_count, "semi_negative_count": neg_count}
    for hd in [1, 5, 20, 60]:
        result[f"semi_positive_{hd}d_avg"] = float(np.mean(semi_pos[hd])) if semi_pos[hd] else None
        result[f"semi_negative_{hd}d_avg"] = float(np.mean(semi_neg[hd])) if semi_neg[hd] else None
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, pct=True, dec=2):
    if v is None:
        return "n/a"
    return f"{v*100:+.{dec}f}%" if pct else f"{v:.{dec}f}"


def _diff(va, vb, pct=True, dec=2):
    if va is None or vb is None:
        return "n/a"
    d = vb - va
    return f"{d*100:+.{dec}f}%p" if pct else f"{d:+.{dec}f}"


def generate_report(
    run_id: str,
    start_date: date,
    end_date: date,
    signal_col: str,
    results: dict,
    risk_off: dict,
    semi: dict,
    data_stats: dict,
    top_n_list: list[int],
    holding_days_list: list[int],
) -> str:
    today = date.today().isoformat()
    lines = [
        "# US Macro Overlay × RULE 전략 백테스트 리포트",
        "",
        "## 1. 실행 정보",
        "",
        f"- **실행일**: {today}",
        f"- **Run ID**: {run_id}",
        f"- **분석 기간**: {start_date} ~ {end_date}",
        f"- **신호 기준**: `{signal_col}`",
        f"- **데이터**: rule_signals.csv × prices_daily_adjusted.csv × signal.us_macro_feature_daily",
        f"- **TopN**: {', '.join(map(str, top_n_list))}",
        f"- **보유기간**: {', '.join(f'{d}D' for d in holding_days_list)}",
        "",
        "---",
        "",
        "## 2. 데이터 정합성 점검",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| RULE 신호 전체 건수 | {data_stats.get('signal_total', 'n/a')} |",
        f"| 매수신호 발생 날짜 수 | {data_stats.get('signal_dates', 'n/a')} |",
        f"| 매크로 피처 보유 날짜 수 | {data_stats.get('macro_rows', 'n/a')} |",
        f"| 매크로 매핑 누락 날짜 수 | {data_stats.get('no_macro_dates', 'n/a')} |",
        f"| 가격 데이터 날짜 수 | {data_stats.get('price_dates', 'n/a')} |",
        "",
        "---",
    ]

    # Per top-N / holding-days tables
    for top_n in top_n_list:
        lines += ["", f"## 3. Top{top_n} 전략 비교", ""]
        for hd in holding_days_list:
            res = results.get(top_n, {}).get(hd, {})
            a = res.get("A", {})
            b = res.get("B", {})
            lines += [
                f"### Top{top_n} | 보유기간 {hd}D",
                "",
                "| 지표 | 전략 A (기존 RULE) | 전략 B (RULE + Overlay) | 차이 (B-A) |",
                "|---|---|---|---|",
                f"| 누적 수익률 | {_fmt(a.get('total_return'))} | {_fmt(b.get('total_return'))} | {_diff(a.get('total_return'), b.get('total_return'))} |",
                f"| 평균 거래수익률 | {_fmt(a.get('avg_return'))} | {_fmt(b.get('avg_return'))} | {_diff(a.get('avg_return'), b.get('avg_return'))} |",
                f"| 중앙값 수익률 | {_fmt(a.get('median_return'))} | {_fmt(b.get('median_return'))} | {_diff(a.get('median_return'), b.get('median_return'))} |",
                f"| 승률 | {_fmt(a.get('win_rate'))} | {_fmt(b.get('win_rate'))} | {_diff(a.get('win_rate'), b.get('win_rate'))} |",
                f"| MDD | {_fmt(a.get('mdd'))} | {_fmt(b.get('mdd'))} | {_diff(a.get('mdd'), b.get('mdd'))} |",
                f"| 변동성 | {_fmt(a.get('volatility'))} | {_fmt(b.get('volatility'))} | {_diff(a.get('volatility'), b.get('volatility'))} |",
                f"| Sharpe (연율화) | {_fmt(a.get('sharpe'), pct=False)} | {_fmt(b.get('sharpe'), pct=False)} | {_diff(a.get('sharpe'), b.get('sharpe'), pct=False)} |",
                f"| 거래 횟수 (날짜 수) | {a.get('trade_count', 'n/a')} | {b.get('trade_count', 'n/a')} | - |",
                f"| 신호 날짜 수 | {a.get('signal_dates', 'n/a')} | - | - |",
                f"| 매크로 미적용 날짜 | {a.get('no_macro_dates', 'n/a')} | - | - |",
                f"| 매수 차단 횟수 (누계) | - | {b.get('blocked_count', 0)} | - |",
                f"| 피한 손실 합계 | - | {_fmt(b.get('avoided_loss'))} | - |",
                f"| 놓친 수익 합계 | - | {_fmt(b.get('missed_gain'))} | - |",
                "",
            ]

    # Risk-Off section
    lines += [
        "---",
        "",
        "## 4. Risk-Off 차단 효과 분석",
        "",
        f"- **차단 발생 날짜 수**: {risk_off.get('block_days', 0)}",
        f"- **총 차단 종목 수 (1D 기준)**: {risk_off.get('total_blocked', 0)}",
        "",
        "| 보유기간 | 차단종목 평균수익 | 하락 비율 | 상승 비율 | 피한 손실 | 놓친 수익 |",
        "|---|---|---|---|---|---|",
    ]
    for hd in [1, 5, 20, 60]:
        lines.append(
            f"| {hd}D | "
            f"{_fmt(risk_off.get(f'blocked_{hd}d_avg'))} | "
            f"{_fmt(risk_off.get(f'blocked_{hd}d_down_pct'))} | "
            f"{_fmt(risk_off.get(f'blocked_{hd}d_up_pct'))} | "
            f"{_fmt(risk_off.get(f'blocked_{hd}d_avoided_loss'))} | "
            f"{_fmt(risk_off.get(f'blocked_{hd}d_missed_gain'))} |"
        )

    block_days = risk_off.get("block_days", 0)
    avg_1d = risk_off.get("blocked_1d_avg")
    down_1d = risk_off.get("blocked_1d_down_pct")
    if block_days == 0:
        verdict = "> Risk-Off 조건이 분석 기간 내 발동되지 않음. 임계값 또는 기간 재검토 필요."
    elif avg_1d is not None and avg_1d < -0.005:
        verdict = f"> **차단 종목 평균 1D 수익률 {_fmt(avg_1d)} (하락 비율 {_fmt(down_1d)}) → 차단이 손실 회피에 기여.**"
    elif avg_1d is not None and avg_1d > 0.005:
        verdict = f"> 차단 종목 평균 1D 수익률 {_fmt(avg_1d)} → **기회손실 발생. 차단 임계값 재검토 권장.**"
    else:
        verdict = f"> 차단 종목 평균 1D 수익률 {_fmt(avg_1d)} → 효과 미미 또는 혼재."
    lines += ["", verdict, ""]

    # Semiconductor section
    lines += [
        "---",
        "",
        "## 5. 반도체 가산 / 감점 효과 분석",
        "",
        f"- **반도체 가산 적용 종목**: {semi.get('semi_positive_count', 0)}건",
        f"- **반도체 감점 적용 종목**: {semi.get('semi_negative_count', 0)}건",
        "",
        "| | 1D 평균수익 | 5D 평균수익 | 20D 평균수익 | 60D 평균수익 |",
        "|---|---|---|---|---|",
        "| 반도체 가산 후보 | " + " | ".join(_fmt(semi.get(f"semi_positive_{hd}d_avg")) for hd in [1, 5, 20, 60]) + " |",
        "| 반도체 감점 후보 | " + " | ".join(_fmt(semi.get(f"semi_negative_{hd}d_avg")) for hd in [1, 5, 20, 60]) + " |",
        "",
    ]

    # Key questions
    ref_top = top_n_list[0]
    ref_hd = 20 if 20 in holding_days_list else holding_days_list[-1]
    ra = results.get(ref_top, {}).get(ref_hd, {}).get("A", {})
    rb = results.get(ref_top, {}).get(ref_hd, {}).get("B", {})
    a_wr = ra.get("win_rate")
    b_wr = rb.get("win_rate")
    a_mdd = ra.get("mdd")
    b_mdd = rb.get("mdd")
    a_ret = ra.get("total_return")
    b_ret = rb.get("total_return")

    lines += [
        "---",
        "",
        "## 6. 핵심 검증 질문 답변",
        "",
        f"**Q1. Risk-Off 차단이 실제 손실을 줄였는가?**",
        f"> {verdict.lstrip('> ')}",
        "",
        f"**Q2. Overlay 후 MDD가 개선되었는가?** (Top{ref_top} {ref_hd}D 기준)",
        f"> A: {_fmt(a_mdd)} → B: {_fmt(b_mdd)} | 차이: {_diff(a_mdd, b_mdd)} ({'개선' if a_mdd is not None and b_mdd is not None and b_mdd > a_mdd else '변화없음/악화'})",
        "",
        f"**Q3. Overlay 후 누적 수익률이 개선되었는가?** (Top{ref_top} {ref_hd}D 기준)",
        f"> A: {_fmt(a_ret)} → B: {_fmt(b_ret)} | 차이: {_diff(a_ret, b_ret)}",
        "",
        f"**Q4. Overlay 후 승률이 개선되었는가?** (Top{ref_top} {ref_hd}D 기준)",
        f"> A: {_fmt(a_wr)} → B: {_fmt(b_wr)} | 차이: {_diff(a_wr, b_wr)}",
        "",
        f"**Q5. 반도체 가산 룰이 실제 수익률 개선으로 이어졌는가?**",
        f"> 가산 {semi.get('semi_positive_count', 0)}건 | 1D 평균 {_fmt(semi.get('semi_positive_1d_avg'))} / 5D {_fmt(semi.get('semi_positive_5d_avg'))}",
        "",
        f"**Q6. 반도체 감점 룰이 실제 손실 회피에 도움이 되었는가?**",
        f"> 감점 {semi.get('semi_negative_count', 0)}건 | 1D 평균 {_fmt(semi.get('semi_negative_1d_avg'))} / 5D {_fmt(semi.get('semi_negative_5d_avg'))}",
        "",
        f"**Q7. 분석 기간이 통계적으로 충분한가?**",
        f"> 신호 날짜 {data_stats.get('signal_dates', 0)}일 — {'3년치 충분한 샘플' if data_stats.get('signal_dates', 0) > 200 else '기간 부족, 추가 검증 필요'}",
        "",
    ]

    # Conclusion
    lines += [
        "---",
        "",
        "## 7. 결론 및 실반영 가능 여부",
        "",
    ]
    if block_days == 0:
        lines.append("> ⚠ **Risk-Off 차단이 발동된 적이 없습니다.** 임계값 재검토 또는 다른 차단 조건 고려 필요.")
    elif b_mdd is not None and a_mdd is not None and b_mdd > a_mdd and b_ret is not None and a_ret is not None:
        lines.append("> **MDD 개선 확인.** 수익률도 확인 필요. Phase 5 실반영 검토 시작 가능.")
    else:
        lines.append("> 추가 검증 필요. Shadow Mode 운영 데이터 축적 후 재판단 권장.")

    lines += [
        "",
        "### Phase 5 실반영 전 필수 조건",
        "",
        "- [ ] Risk-Off 차단 효과가 손실 회피 방향으로 확인됨",
        "- [ ] MDD 개선 or 동등 + 수익률 유지",
        "- [ ] 기회손실이 허용 범위(예: +2% 이내) 이내",
        "- [ ] `US_MACRO_ALLOW_REAL_APPLY=true` 설정 전 운영팀 검토",
        "- [ ] Shadow Mode 최소 2개월 운영 후 재실행",
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
# DB write
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = text("""
CREATE TABLE IF NOT EXISTS research.us_macro_overlay_backtest_result (
    run_id             VARCHAR(100) NOT NULL,
    run_date           DATE         NOT NULL,
    start_date         DATE         NOT NULL,
    end_date           DATE         NOT NULL,
    strategy_name      VARCHAR(100) NOT NULL,
    top_n              INTEGER,
    holding_days       INTEGER,
    total_return       DOUBLE PRECISION,
    avg_return         DOUBLE PRECISION,
    median_return      DOUBLE PRECISION,
    win_rate           DOUBLE PRECISION,
    mdd                DOUBLE PRECISION,
    volatility         DOUBLE PRECISION,
    trade_count        INTEGER,
    signal_dates       INTEGER,
    blocked_count      INTEGER,
    avoided_loss       DOUBLE PRECISION,
    missed_gain        DOUBLE PRECISION,
    win_rate_1d        DOUBLE PRECISION,
    win_rate_5d        DOUBLE PRECISION,
    win_rate_20d       DOUBLE PRECISION,
    win_rate_60d       DOUBLE PRECISION,
    avg_return_1d      DOUBLE PRECISION,
    avg_return_5d      DOUBLE PRECISION,
    avg_return_20d     DOUBLE PRECISION,
    avg_return_60d     DOUBLE PRECISION,
    summary            TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, strategy_name, top_n, holding_days)
)
""")

_UPSERT_SQL = text("""
INSERT INTO research.us_macro_overlay_backtest_result (
    run_id, run_date, start_date, end_date, strategy_name,
    top_n, holding_days, total_return, avg_return, median_return, win_rate,
    mdd, volatility, trade_count, signal_dates, blocked_count,
    avoided_loss, missed_gain,
    win_rate_1d, win_rate_5d, win_rate_20d, win_rate_60d,
    avg_return_1d, avg_return_5d, avg_return_20d, avg_return_60d, summary
) VALUES (
    :run_id, :run_date, :start_date, :end_date, :strategy_name,
    :top_n, :holding_days, :total_return, :avg_return, :median_return, :win_rate,
    :mdd, :volatility, :trade_count, :signal_dates, :blocked_count,
    :avoided_loss, :missed_gain,
    :win_rate_1d, :win_rate_5d, :win_rate_20d, :win_rate_60d,
    :avg_return_1d, :avg_return_5d, :avg_return_20d, :avg_return_60d, :summary
)
ON CONFLICT (run_id, strategy_name, top_n, holding_days) DO UPDATE SET
    total_return = EXCLUDED.total_return,
    avg_return   = EXCLUDED.avg_return,
    summary      = EXCLUDED.summary,
    created_at   = CURRENT_TIMESTAMP
""")


def _write_db(engine: Any, run_id: str, run_date: date, start_date: date, end_date: date,
              results: dict, top_n_list: list[int], holding_days_list: list[int]) -> int:
    with engine.begin() as conn:
        conn.execute(_CREATE_TABLE_SQL)
    rows = []
    for top_n in top_n_list:
        for hd in holding_days_list:
            res = results.get(top_n, {}).get(hd, {})
            for sname, m in res.items():
                rows.append({
                    "run_id": run_id, "run_date": run_date,
                    "start_date": start_date, "end_date": end_date,
                    "strategy_name": f"RULE_{sname}",
                    "top_n": top_n, "holding_days": hd,
                    "total_return": m.get("total_return"),
                    "avg_return": m.get("avg_return"),
                    "median_return": m.get("median_return"),
                    "win_rate": m.get("win_rate"),
                    "mdd": m.get("mdd"),
                    "volatility": m.get("volatility"),
                    "trade_count": m.get("trade_count"),
                    "signal_dates": m.get("signal_dates"),
                    "blocked_count": m.get("blocked_count", 0),
                    "avoided_loss": m.get("avoided_loss"),
                    "missed_gain": m.get("missed_gain"),
                    "win_rate_1d": None, "win_rate_5d": None,
                    "win_rate_20d": None, "win_rate_60d": None,
                    "avg_return_1d": None, "avg_return_5d": None,
                    "avg_return_20d": None, "avg_return_60d": None,
                    "summary": f"RULE Top{top_n} {hd}D Strategy {sname}",
                })
    if rows:
        with engine.begin() as conn:
            conn.execute(_UPSERT_SQL, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: RULE × US Macro Overlay Backtest")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--signals-csv", default=str(DEFAULT_SIGNALS_CSV))
    p.add_argument("--prices-csv", default=str(DEFAULT_PRICES_CSV))
    p.add_argument("--signal-col", default="entry_signal",
                   help="Signal column: entry_signal or strong_entry_signal")
    p.add_argument("--top-n", default="5,10")
    p.add_argument("--holding-days", default="1,5,20,60")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--report-dir", default=str(REPORTS_DIR))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args = parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = date.fromisoformat(args.start_date) if args.start_date else date(2023, 4, 14)
    top_n_list = [int(x.strip()) for x in args.top_n.split(",")]
    holding_days_list = [int(x.strip()) for x in args.holding_days.split(",")]
    signal_col = args.signal_col
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"rule_bt_{today.isoformat().replace('-', '')}_{str(uuid.uuid4())[:8]}"
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 4 RULE Backtest | run_id=%s", run_id)
    LOGGER.info("%s ~ %s | signal_col=%s", start_date, end_date, signal_col)
    LOGGER.info("=" * 60)

    # Load data
    engine = get_engine()

    LOGGER.info("Loading rule signals ...")
    signals_df = _load_rule_signals(Path(args.signals_csv), signal_col, start_date, end_date)
    if signals_df.empty:
        LOGGER.error("No rule signals found. Abort.")
        return

    signal_dates = sorted(signals_df[signals_df[signal_col]]["date"].unique())
    actual_start = signal_dates[0] if signal_dates else start_date
    actual_end = signal_dates[-1] if signal_dates else end_date

    LOGGER.info("Loading price data ...")
    price_df = _load_prices_csv(Path(args.prices_csv), actual_start, actual_end)

    LOGGER.info("Computing forward returns (this may take a moment) ...")
    fwd_df = _build_forward_returns(price_df, holding_days_list)
    LOGGER.info("Forward return rows: %d", len(fwd_df))

    LOGGER.info("Loading macro features ...")
    macro_df = _load_macro_features_db(engine, actual_start, actual_end)
    macro_by_date: dict[date, dict] = {}
    if not macro_df.empty:
        for _, row in macro_df.iterrows():
            macro_by_date[row["kr_apply_date"]] = row.to_dict()

    no_macro = sum(1 for d in signal_dates if _nearest_macro(macro_by_date, d) is None)
    LOGGER.info("Signal dates: %d | No-macro dates: %d", len(signal_dates), no_macro)

    data_stats = {
        "signal_total": len(signals_df[signals_df[signal_col]]),
        "signal_dates": len(signal_dates),
        "macro_rows": len(macro_df),
        "no_macro_dates": no_macro,
        "price_dates": price_df["date"].nunique(),
    }

    # Run backtest
    LOGGER.info("Running backtest ...")
    results = run_rule_backtest(
        signals_df, fwd_df, macro_by_date, signal_col, top_n_list, holding_days_list
    )

    LOGGER.info("Analyzing Risk-Off blocks ...")
    risk_off = _analyze_risk_off(signals_df, fwd_df, macro_by_date, signal_col)
    LOGGER.info("Risk-Off: %d block days, %d blocked stocks", risk_off["block_days"], risk_off.get("total_blocked", 0))

    LOGGER.info("Analyzing semiconductor rules ...")
    semi = _analyze_semiconductor(signals_df, fwd_df, macro_by_date, signal_col)

    # DB write
    if not args.no_db:
        n = _write_db(engine, run_id, today, actual_start, actual_end,
                      results, top_n_list, holding_days_list)
        LOGGER.info("Wrote %d rows to research.us_macro_overlay_backtest_result", n)

    # Reports
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = report_dir / f"us_macro_overlay_backtest_rule_{ts}.md"
    csv_path = report_dir / f"us_macro_overlay_backtest_rule_{ts}.csv"

    md = generate_report(
        run_id, actual_start, actual_end, signal_col,
        results, risk_off, semi, data_stats, top_n_list, holding_days_list,
    )
    md_path.write_text(md, encoding="utf-8")
    LOGGER.info("MD report → %s", md_path)

    csv_rows = []
    for top_n in top_n_list:
        for hd in holding_days_list:
            res = results.get(top_n, {}).get(hd, {})
            for sname, m in res.items():
                csv_rows.append({"strategy": sname, "top_n": top_n, "holding_days": hd,
                                 **{k: m.get(k) for k in ["total_return", "avg_return",
                                    "median_return", "win_rate", "mdd", "volatility",
                                    "trade_count", "signal_dates", "blocked_count",
                                    "avoided_loss", "missed_gain", "sharpe"]}})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info("CSV report → %s", csv_path)

    # Console summary
    print("\n" + "=" * 60)
    print(f"[RULE Backtest] {actual_start} ~ {actual_end} ({len(signal_dates)} signal dates)")
    print(f"Signal: {signal_col} | Risk-Off block days: {risk_off['block_days']}")
    for top_n in top_n_list:
        for hd in holding_days_list:
            res = results.get(top_n, {}).get(hd, {})
            a, b = res.get("A", {}), res.get("B", {})
            def s(v): return _fmt(v) if v is not None else "n/a"
            print(f"  Top{top_n} {hd}D | A: ret={s(a.get('total_return'))} wr={s(a.get('win_rate'))} mdd={s(a.get('mdd'))}"
                  f" | B: ret={s(b.get('total_return'))} wr={s(b.get('win_rate'))} mdd={s(b.get('mdd'))}"
                  f" | blocked={b.get('blocked_count', 0)}")
    print("=" * 60)
    LOGGER.info("이번 Phase 4 작업은 백테스트 검증 전용이며, 실제 주문/추천/RULE 실행 로직에 영향을 주지 않습니다.")


if __name__ == "__main__":
    main()
