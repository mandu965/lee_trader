from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rule_signal_builder import DEFAULT_PRICES, ROOT, resolve


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SIGNALS = DATA_DIR / "rule_signals.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "rule_strategy_backtest_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "rule_strategy_backtest_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest RULE_TREND_LIQUIDITY_V1 signals using D+1 open.")
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def read_signals(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule signals not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["entry_signal", "strong_entry_signal"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df[col] = False
    return df.dropna(subset=["date", "code"]).sort_values(["code", "date"])


def read_prices(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"prices csv not found for backtest: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["open"] = pd.to_numeric(df["adj_open"] if "adj_open" in df.columns else df.get("open"), errors="coerce")
    df["close"] = pd.to_numeric(df["adj_close"] if "adj_close" in df.columns else df.get("close"), errors="coerce")
    return df.dropna(subset=["date", "code", "open", "close"]).sort_values(["code", "date"]).reset_index(drop=True)


def build_trade_rows(signals: pd.DataFrame, prices: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    selected = signals.loc[signals[signal_col].fillna(False)].copy()
    if selected.empty:
        return pd.DataFrame()

    price_rows: list[pd.DataFrame] = []
    for code, g in prices.groupby("code", sort=False):
        pg = g.sort_values("date").reset_index(drop=True).copy()
        pg["entry_date"] = pg["date"].shift(-1)
        pg["entry_price"] = pg["open"].shift(-1)
        for horizon in [5, 20, 60]:
            pg[f"exit_date_{horizon}"] = pg["date"].shift(-horizon)
            pg[f"exit_close_{horizon}"] = pg["close"].shift(-horizon)
        price_rows.append(pg)
    forward = pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()
    keep = ["date", "code", "entry_date", "entry_price"]
    for horizon in [5, 20, 60]:
        keep.extend([f"exit_date_{horizon}", f"exit_close_{horizon}"])
    merged = selected.merge(forward[keep], on=["date", "code"], how="left")
    for horizon in [5, 20, 60]:
        merged[f"ret_d{horizon}"] = pd.to_numeric(merged[f"exit_close_{horizon}"], errors="coerce") / pd.to_numeric(merged["entry_price"], errors="coerce") - 1.0
    merged["signal_type"] = signal_col
    return merged.dropna(subset=["entry_price"])


def max_drawdown(returns: pd.Series) -> float | None:
    vals = pd.to_numeric(returns, errors="coerce").dropna()
    if vals.empty:
        return None
    vals = vals.clip(lower=-0.99)
    log_equity = np.log1p(vals).cumsum()
    log_drawdown = log_equity - log_equity.cummax()
    return float(np.expm1(log_drawdown.min()))


def max_drawdown_by_signal_date(frame: pd.DataFrame, ret_col: str = "ret_d20") -> float | None:
    if frame.empty or "date" not in frame.columns or ret_col not in frame.columns:
        return None
    daily = (
        frame.assign(_ret=pd.to_numeric(frame[ret_col], errors="coerce"))
        .dropna(subset=["date", "_ret"])
        .groupby("date")["_ret"]
        .mean()
        .sort_index()
    )
    return max_drawdown(daily)


def metrics_for(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "avg_return_d5": None,
            "avg_return_d20": None,
            "avg_return_d60": None,
            "win_rate_d20": None,
            "payoff_ratio_d20": None,
            "mdd_d20_equal_trade": None,
            "mdd_d20_daily_avg": None,
            "avg_holding_days": None,
        }
    ret20 = pd.to_numeric(frame["ret_d20"], errors="coerce")
    wins = ret20[ret20 > 0]
    losses = ret20[ret20 < 0]
    payoff = None
    if not wins.empty and not losses.empty and abs(float(losses.mean())) > 0:
        payoff = float(wins.mean() / abs(losses.mean()))
    return {
        "trade_count": int(len(frame)),
        "avg_return_d5": _float(frame["ret_d5"].mean()),
        "avg_return_d20": _float(frame["ret_d20"].mean()),
        "avg_return_d60": _float(frame["ret_d60"].mean()),
        "win_rate_d20": _float((ret20 > 0).mean()),
        "payoff_ratio_d20": payoff,
        "mdd_d20_equal_trade": max_drawdown(ret20.sort_index()),
        "mdd_d20_daily_avg": max_drawdown_by_signal_date(frame, "ret_d20"),
        "avg_holding_days": 20.0,
    }


def portfolio_equity_metrics_for(frame: pd.DataFrame, max_positions: int = 5) -> dict[str, Any]:
    if frame.empty:
        return {
            "rebalance_count": 0,
            "candidate_signal_date_count": 0,
            "avg_positions_per_signal_date": None,
            "avg_return_d5": None,
            "avg_return_d20": None,
            "avg_return_d60": None,
            "win_rate_d20": None,
            "mdd_d20_portfolio_equity": None,
            "final_return_d20_portfolio_equity": None,
        }
    sort_cols = [col for col in ["date", "rule_score_v2", "rule_score", "liquidity_score", "vol_20"] if col in frame.columns]
    ascending = [True]
    for col in sort_cols[1:]:
        ascending.append(col == "vol_20")
    selected = frame.sort_values(sort_cols, ascending=ascending).groupby("date", group_keys=False).head(max_positions)
    daily = selected.groupby("date").agg(
        ret_d5=("ret_d5", "mean"),
        ret_d20=("ret_d20", "mean"),
        ret_d60=("ret_d60", "mean"),
        exit_date_20=("exit_date_20", "max"),
        position_count=("code", "count"),
    ).sort_index()
    rebalance_rows = []
    next_available_date = pd.Timestamp.min
    for signal_date, row in daily.iterrows():
        if pd.Timestamp(signal_date) < next_available_date:
            continue
        exit_date = pd.to_datetime(row.get("exit_date_20"), errors="coerce")
        if pd.isna(exit_date):
            continue
        rebalance_rows.append(row)
        next_available_date = exit_date
    rebalance = pd.DataFrame(rebalance_rows)
    ret20 = pd.to_numeric(rebalance.get("ret_d20"), errors="coerce").dropna()
    if ret20.empty:
        mdd = None
        final_return = None
        win_rate = None
    else:
        clipped = ret20.clip(lower=-0.99)
        log_equity = np.log1p(clipped).cumsum()
        final_return = float(np.expm1(log_equity.iloc[-1]))
        mdd = max_drawdown(ret20)
        win_rate = _float((ret20 > 0).mean())
    return {
        "rebalance_count": int(len(rebalance)),
        "candidate_signal_date_count": int(len(daily)),
        "avg_positions_per_signal_date": _float(rebalance["position_count"].mean()) if not rebalance.empty else None,
        "avg_return_d5": _float(rebalance["ret_d5"].mean()) if not rebalance.empty else None,
        "avg_return_d20": _float(rebalance["ret_d20"].mean()) if not rebalance.empty else None,
        "avg_return_d60": _float(rebalance["ret_d60"].mean()) if not rebalance.empty else None,
        "win_rate_d20": win_rate,
        "mdd_d20_portfolio_equity": mdd,
        "final_return_d20_portfolio_equity": final_return,
    }


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def grouped_metrics(frame: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    if frame.empty or col not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(frame[col].fillna("(none)").astype(str)):
        item = {"group": value}
        item.update(metrics_for(group))
        rows.append(item)
    rows.sort(key=lambda item: item.get("trade_count", 0), reverse=True)
    return rows


def numeric_col(frame: pd.DataFrame, col: str, default: float | None = None) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def bool_col(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(default).astype(bool)
    return raw.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def core_entry_conditions(frame: pd.DataFrame) -> pd.Series:
    rsi = numeric_col(frame, "rsi_14")
    close_over_ma20 = numeric_col(frame, "close_over_ma20")
    return (
        (numeric_col(frame, "close") > numeric_col(frame, "ma_20"))
        & (numeric_col(frame, "ma_20") >= numeric_col(frame, "ma_60"))
        & (numeric_col(frame, "ret_5d") > 0)
        & (numeric_col(frame, "mom_20") > 0)
        & rsi.between(45, 75)
        & (numeric_col(frame, "volume_ratio_20d") >= 1.2)
        & (numeric_col(frame, "trading_value_ratio_20d") >= 1.1)
        & bool_col(frame, "trading_value_pass", True)
        & (~bool_col(frame, "gap_risk_blocked", False))
        & bool_col(frame, "market_entry_allowed", True)
        & (~((rsi > 80) | (close_over_ma20 > 0.20) | (numeric_col(frame, "ret_5d") > 0.25)))
    ).fillna(False)


def build_report(signals: pd.DataFrame, prices: pd.DataFrame) -> dict[str, Any]:
    trades_entry = build_trade_rows(signals, prices, "entry_signal")
    trades_strong = build_trade_rows(signals, prices, "strong_entry_signal")
    latest_date = signals["date"].max()
    date_count = max(int(signals["date"].nunique()), 1)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_signal_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "summary": {
            "entry_signal": metrics_for(trades_entry),
            "strong_entry_signal": metrics_for(trades_strong),
            "turnover_entry_per_signal_date": _float(len(trades_entry) / date_count),
            "turnover_strong_per_signal_date": _float(len(trades_strong) / date_count),
        },
        "portfolio_equity_curve": {
            "method": "Select up to 5 candidates by rule_score_v2/rule_score/liquidity_score/vol_20, hold the equal-weight basket for 20 trading days, then rebalance on the next eligible signal date.",
            "entry_signal": portfolio_equity_metrics_for(trades_entry, max_positions=5),
            "strong_entry_signal": portfolio_equity_metrics_for(trades_strong, max_positions=5),
        },
        "test_summary_2024_plus": {
            "entry_signal": metrics_for(trades_entry[trades_entry["date"] >= pd.Timestamp("2024-01-01")]),
            "strong_entry_signal": metrics_for(trades_strong[trades_strong["date"] >= pd.Timestamp("2024-01-01")]),
        },
        "sector": {
            "entry_signal": grouped_metrics(trades_entry, "sector"),
            "strong_entry_signal": grouped_metrics(trades_strong, "sector"),
        },
        "regime": {
            "entry_signal": grouped_metrics(trades_entry, "regime"),
            "strong_entry_signal": grouped_metrics(trades_strong, "regime"),
        },
        "candidate_counts": {
            "all_signal_rows": int(len(signals)),
            "entry_signal_rows": int(signals["entry_signal"].sum()),
            "strong_entry_signal_rows": int(signals["strong_entry_signal"].sum()),
            "gap_risk_blocked_rows": int(signals.get("gap_risk_blocked", pd.Series(False, index=signals.index)).sum()),
            "trading_value_failed_rows": int((~signals.get("trading_value_pass", pd.Series(True, index=signals.index))).sum()),
            "defensive_mode_rows": int(signals.get("market_defensive_mode", pd.Series(False, index=signals.index)).sum()),
        },
        "threshold_sensitivity": threshold_sensitivity(signals, prices),
        "rule_score_v2_sensitivity": rule_score_v2_sensitivity(signals, prices),
        "score_distribution": score_distribution(signals),
        "trading_value_distribution": trading_value_distribution(signals),
    }
    return report


def score_distribution(signals: pd.DataFrame) -> dict[str, Any]:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    out: dict[str, Any] = {}
    for col in ["rule_score", "rule_score_v2"]:
        if col not in latest.columns:
            out[col] = {}
            continue
        vals = pd.to_numeric(latest[col], errors="coerce").dropna()
        out[col] = {
            "count": int(vals.count()),
            "mean": _float(vals.mean()),
            "p10": _float(vals.quantile(0.10)) if not vals.empty else None,
            "p25": _float(vals.quantile(0.25)) if not vals.empty else None,
            "p50": _float(vals.quantile(0.50)) if not vals.empty else None,
            "p75": _float(vals.quantile(0.75)) if not vals.empty else None,
            "p90": _float(vals.quantile(0.90)) if not vals.empty else None,
            "max": _float(vals.max()) if not vals.empty else None,
        }
    return out


def trading_value_distribution(signals: pd.DataFrame) -> dict[str, Any]:
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date].copy()
    vals = pd.to_numeric(latest.get("trading_value_ma_20"), errors="coerce").dropna()
    thresholds = [500_000_000, 1_000_000_000, 2_000_000_000]
    return {
        "count": int(vals.count()),
        "mean": _float(vals.mean()),
        "p50": _float(vals.quantile(0.50)) if not vals.empty else None,
        "p75": _float(vals.quantile(0.75)) if not vals.empty else None,
        "p90": _float(vals.quantile(0.90)) if not vals.empty else None,
        "thresholds": [
            {
                "threshold": threshold,
                "pass_count": int((vals >= threshold).sum()),
                "pass_rate": _float((vals >= threshold).mean()) if not vals.empty else None,
            }
            for threshold in thresholds
        ],
    }


def threshold_sensitivity(signals: pd.DataFrame, prices: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_thresholds = [65, 70, 75]
    trading_value_thresholds = [500_000_000, 1_000_000_000, 2_000_000_000]
    base = signals.copy()
    base_conditions = core_entry_conditions(base)
    for score_threshold in score_thresholds:
        for tv_threshold in trading_value_thresholds:
            tmp = base.copy()
            tmp["sensitivity_signal"] = (
                base_conditions
                & (pd.to_numeric(tmp.get("rule_score"), errors="coerce") >= score_threshold)
                & (pd.to_numeric(tmp.get("trading_value_ma_20"), errors="coerce") >= tv_threshold)
            )
            trades = build_trade_rows(tmp, prices, "sensitivity_signal")
            metrics = metrics_for(trades)
            rows.append(
                {
                    "rule_score_threshold": score_threshold,
                    "trading_value_ma20_threshold": tv_threshold,
                    "candidate_count": int(tmp["sensitivity_signal"].sum()),
                    "trade_count": metrics["trade_count"],
                    "avg_return_d20": metrics["avg_return_d20"],
                    "win_rate_d20": metrics["win_rate_d20"],
                    "mdd_d20_daily_avg": metrics["mdd_d20_daily_avg"],
                }
            )
    return rows


def rule_score_v2_sensitivity(signals: pd.DataFrame, prices: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v2_thresholds = [35, 40, 45, 50, 55, 60]
    trading_value_thresholds = [500_000_000, 1_000_000_000, 2_000_000_000]
    base = signals.copy()
    base_conditions = core_entry_conditions(base) & (numeric_col(base, "rule_score") >= 70)
    for v2_threshold in v2_thresholds:
        for tv_threshold in trading_value_thresholds:
            tmp = base.copy()
            tmp["v2_sensitivity_signal"] = (
                base_conditions
                & (numeric_col(tmp, "rule_score_v2") >= v2_threshold)
                & (numeric_col(tmp, "trading_value_ma_20") >= tv_threshold)
            )
            trades = build_trade_rows(tmp, prices, "v2_sensitivity_signal")
            metrics = metrics_for(trades)
            rows.append(
                {
                    "rule_score_min": 70,
                    "rule_score_v2_threshold": v2_threshold,
                    "trading_value_ma20_threshold": tv_threshold,
                    "candidate_count": int(tmp["v2_sensitivity_signal"].sum()),
                    "trade_count": metrics["trade_count"],
                    "avg_return_d20": metrics["avg_return_d20"],
                    "win_rate_d20": metrics["win_rate_d20"],
                    "mdd_d20_daily_avg": metrics["mdd_d20_daily_avg"],
                }
            )
    return rows


def fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}%"


def render_md(report: dict[str, Any]) -> str:
    def section_metrics(name: str, metrics: dict[str, Any]) -> list[str]:
        return [
            f"### {name}",
            "",
            f"- trade_count: `{metrics.get('trade_count', 0)}`",
            f"- avg_return_d5: `{fmt_pct(metrics.get('avg_return_d5'))}`",
            f"- avg_return_d20: `{fmt_pct(metrics.get('avg_return_d20'))}`",
            f"- avg_return_d60: `{fmt_pct(metrics.get('avg_return_d60'))}`",
            f"- win_rate_d20: `{fmt_pct(metrics.get('win_rate_d20'))}`",
            f"- payoff_ratio_d20: `{metrics.get('payoff_ratio_d20') if metrics.get('payoff_ratio_d20') is not None else 'NA'}`",
            f"- mdd_d20_daily_avg: `{fmt_pct(metrics.get('mdd_d20_daily_avg'))}`",
            f"- mdd_d20_equal_trade: `{fmt_pct(metrics.get('mdd_d20_equal_trade'))}`",
            "",
        ]

    lines = [
        "# RULE Strategy Backtest Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- latest_signal_date: `{report.get('latest_signal_date') or 'NA'}`",
        "",
        "## Overall",
        "",
    ]
    lines.extend(section_metrics("entry_signal", report["summary"]["entry_signal"]))
    lines.extend(section_metrics("strong_entry_signal", report["summary"]["strong_entry_signal"]))
    portfolio = report.get("portfolio_equity_curve") or {}
    lines.extend(
        [
            "## Portfolio Equity Curve",
            "",
            f"- method: `{portfolio.get('method', 'NA')}`",
            "",
            "| signal | rebalances | signal_dates | avg_positions | avg_return_d20 | win_rate_d20 | mdd_d20 | final_return_d20 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ["entry_signal", "strong_entry_signal"]:
        metrics = portfolio.get(name) or {}
        lines.append(
            "| {name} | {dates} | {signal_dates} | {avg_pos} | {avg_d20} | {win} | {mdd} | {final} |".format(
                name=name,
                dates=metrics.get("rebalance_count", 0),
                signal_dates=metrics.get("candidate_signal_date_count", 0),
                avg_pos="NA" if metrics.get("avg_positions_per_signal_date") is None else f"{float(metrics['avg_positions_per_signal_date']):.2f}",
                avg_d20=fmt_pct(metrics.get("avg_return_d20")),
                win=fmt_pct(metrics.get("win_rate_d20")),
                mdd=fmt_pct(metrics.get("mdd_d20_portfolio_equity")),
                final=fmt_pct(metrics.get("final_return_d20_portfolio_equity")),
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Test 2024+",
            "",
        ]
    )
    lines.extend(section_metrics("entry_signal test", report["test_summary_2024_plus"]["entry_signal"]))
    lines.extend(section_metrics("strong_entry_signal test", report["test_summary_2024_plus"]["strong_entry_signal"]))
    counts = report["candidate_counts"]
    lines.extend(
        [
            "## Candidate Counts",
            "",
            f"- all_signal_rows: `{counts['all_signal_rows']}`",
            f"- entry_signal_rows: `{counts['entry_signal_rows']}`",
            f"- strong_entry_signal_rows: `{counts['strong_entry_signal_rows']}`",
            f"- gap_risk_blocked_rows: `{counts['gap_risk_blocked_rows']}`",
            f"- trading_value_failed_rows: `{counts['trading_value_failed_rows']}`",
            f"- defensive_mode_rows: `{counts['defensive_mode_rows']}`",
            "",
        ]
    )
    lines.extend(["## Score Distribution", ""])
    for col, stats in report.get("score_distribution", {}).items():
        lines.extend(
            [
                f"### {col}",
                "",
                f"- mean: `{stats.get('mean') if stats.get('mean') is not None else 'NA'}`",
                f"- p50: `{stats.get('p50') if stats.get('p50') is not None else 'NA'}`",
                f"- p75: `{stats.get('p75') if stats.get('p75') is not None else 'NA'}`",
                f"- p90: `{stats.get('p90') if stats.get('p90') is not None else 'NA'}`",
                f"- max: `{stats.get('max') if stats.get('max') is not None else 'NA'}`",
                "",
            ]
        )
    tv = report.get("trading_value_distribution", {})
    lines.extend(
        [
            "## Trading Value Filter",
            "",
            f"- p50: `{tv.get('p50') if tv.get('p50') is not None else 'NA'}`",
            f"- p75: `{tv.get('p75') if tv.get('p75') is not None else 'NA'}`",
            f"- p90: `{tv.get('p90') if tv.get('p90') is not None else 'NA'}`",
            "",
            "| threshold | pass_count | pass_rate |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in tv.get("thresholds", []):
        lines.append(f"| {row['threshold']:,.0f} | {row['pass_count']} | {fmt_pct(row.get('pass_rate'))} |")
    lines.extend(["", "## Threshold Sensitivity", "", "| rule_score | trading_value_ma20 | candidates | trades | avg_return_d20 | win_rate_d20 | mdd_d20_daily_avg |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report.get("threshold_sensitivity", []):
        lines.append(
            f"| {row['rule_score_threshold']} | {row['trading_value_ma20_threshold']:,.0f} | {row['candidate_count']} | {row['trade_count']} | {fmt_pct(row.get('avg_return_d20'))} | {fmt_pct(row.get('win_rate_d20'))} | {fmt_pct(row.get('mdd_d20_daily_avg'))} |"
        )
    lines.extend(["", "## rule_score_v2 Sensitivity", "", "| rule_score_min | rule_score_v2_min | trading_value_ma20 | candidates | trades | avg_return_d20 | win_rate_d20 | mdd_d20_daily_avg |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report.get("rule_score_v2_sensitivity", []):
        lines.append(
            f"| {row['rule_score_min']} | {row['rule_score_v2_threshold']} | {row['trading_value_ma20_threshold']:,.0f} | {row['candidate_count']} | {row['trade_count']} | {fmt_pct(row.get('avg_return_d20'))} | {fmt_pct(row.get('win_rate_d20'))} | {fmt_pct(row.get('mdd_d20_daily_avg'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    signals = read_signals(args.signals_csv)
    prices = read_prices(args.prices_csv)
    report = build_report(signals, prices)
    out_json = resolve(args.out_json)
    out_md = resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(f"saved {out_json}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
