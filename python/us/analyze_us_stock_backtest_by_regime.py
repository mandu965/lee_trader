from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
import logging
import math
from pathlib import Path
import statistics
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_market_regime_config
from python.us.us_db import (
    ensure_us_market_regime_tables,
    get_us_engine,
    upsert_us_rank_backtest_regime_summary_rows,
)


LOGGER = logging.getLogger("us_regime_analysis")
SUPPORTED_FORMATS = {"console", "markdown", "csv"}


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_market_regime_config()
    parser = argparse.ArgumentParser(description="Analyze US rank backtest performance by market regime.")
    parser.add_argument("--backtest-id", required=True)
    parser.add_argument("--format", default=cfg.report_default_format, choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--holding-days", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=cfg.report_output_dir)
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _fmt_pct(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric * 100:.2f}%"


def _format_date(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _normalize_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _mean(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return float(statistics.fmean(numbers))


def _median(values: list[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return float(statistics.median(numbers))


def _query_joined_rows(*, backtest_id: str, strategy: str | None, holding_days: int | None) -> list[dict[str, object]]:
    clauses = ["s.backtest_id = :backtest_id"]
    params: dict[str, object] = {"backtest_id": backtest_id}
    if strategy:
        clauses.append("s.strategy_name = :strategy")
        params["strategy"] = strategy
    if holding_days is not None:
        clauses.append("s.holding_days = :holding_days")
        params["holding_days"] = holding_days
    stmt = text(
        f"""
        SELECT
            s.*,
            r.spy_regime,
            r.qqq_regime,
            r.vol_regime,
            r.market_regime,
            r.data_status AS regime_data_status
        FROM research.us_stock_rank_backtest_summary s
        LEFT JOIN research.us_market_regime_daily r
          ON r.trade_date = s.trade_date
        WHERE {' AND '.join(clauses)}
        ORDER BY s.trade_date, s.strategy_name, s.holding_days
        """
    )
    with get_us_engine().connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def _period_key(trade_date: date, period_type: str) -> str:
    if period_type == "MONTH":
        return trade_date.strftime("%Y-%m")
    if period_type == "QUARTER":
        quarter = ((trade_date.month - 1) // 3) + 1
        return f"{trade_date.year}-Q{quarter}"
    if period_type == "YEAR":
        return f"{trade_date.year}"
    raise ValueError(period_type)


def _aggregate_group(
    *,
    backtest_id: str,
    strategy_name: str,
    selection_rule: str,
    holding_days: int,
    regime_type: str,
    regime_value: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    valid_rows = [row for row in rows if _safe_float(row.get("avg_return_pct")) is not None]
    best_row = max(valid_rows, key=lambda row: float(row["avg_return_pct"])) if valid_rows else None
    worst_row = min(valid_rows, key=lambda row: float(row["avg_return_pct"])) if valid_rows else None
    if not rows:
        data_status = "NO_DATA"
    elif regime_value == "UNKNOWN":
        data_status = "UNKNOWN_REGIME"
    elif len(valid_rows) < len(rows):
        data_status = "PARTIAL_METRICS"
    else:
        data_status = "OK"

    selected_count_avg = _mean([_safe_float(row.get("selected_count")) for row in rows])
    avg_return_pct = _mean([_safe_float(row.get("avg_return_pct")) for row in valid_rows])
    median_return_pct = _median([_safe_float(row.get("avg_return_pct")) for row in valid_rows])
    win_rate = _mean([_safe_float(row.get("win_rate")) for row in valid_rows])
    avg_excess_return_vs_spy = _mean([_safe_float(row.get("avg_excess_return_vs_spy")) for row in valid_rows])
    avg_excess_return_vs_qqq = _mean([_safe_float(row.get("avg_excess_return_vs_qqq")) for row in valid_rows])
    avg_excess_return_vs_universe = _mean([_safe_float(row.get("avg_excess_return_vs_universe")) for row in valid_rows])
    win_rate_vs_spy = _mean([_safe_float(row.get("win_rate_vs_spy")) for row in valid_rows])
    win_rate_vs_qqq = _mean([_safe_float(row.get("win_rate_vs_qqq")) for row in valid_rows])
    win_rate_vs_universe = _mean([_safe_float(row.get("win_rate_vs_universe")) for row in valid_rows])

    return {
        "backtest_id": backtest_id,
        "strategy_name": strategy_name,
        "selection_rule": selection_rule,
        "holding_days": holding_days,
        "regime_type": regime_type,
        "regime_value": regime_value,
        "test_days": len(rows),
        "selected_count_avg": round(selected_count_avg, 6) if selected_count_avg is not None else None,
        "avg_return_pct": round(avg_return_pct, 6) if avg_return_pct is not None else None,
        "median_return_pct": round(median_return_pct, 6) if median_return_pct is not None else None,
        "win_rate": round(win_rate, 6) if win_rate is not None else None,
        "avg_excess_return_vs_spy": round(avg_excess_return_vs_spy, 6) if avg_excess_return_vs_spy is not None else None,
        "avg_excess_return_vs_qqq": round(avg_excess_return_vs_qqq, 6) if avg_excess_return_vs_qqq is not None else None,
        "avg_excess_return_vs_universe": round(avg_excess_return_vs_universe, 6) if avg_excess_return_vs_universe is not None else None,
        "win_rate_vs_spy": round(win_rate_vs_spy, 6) if win_rate_vs_spy is not None else None,
        "win_rate_vs_qqq": round(win_rate_vs_qqq, 6) if win_rate_vs_qqq is not None else None,
        "win_rate_vs_universe": round(win_rate_vs_universe, 6) if win_rate_vs_universe is not None else None,
        "best_trade_date": best_row.get("trade_date") if best_row else None,
        "best_avg_return_pct": round(float(best_row["avg_return_pct"]), 6) if best_row else None,
        "worst_trade_date": worst_row.get("trade_date") if worst_row else None,
        "worst_avg_return_pct": round(float(worst_row["avg_return_pct"]), 6) if worst_row else None,
        "data_status": data_status,
    }


def aggregate_regime_rows(joined_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in joined_rows:
        strategy_name = str(row.get("strategy_name") or "")
        selection_rule = str(row.get("selection_rule") or "")
        holding_days = int(row.get("holding_days") or 0)
        trade_date = row.get("trade_date")
        if not isinstance(trade_date, date):
            continue
        regime_pairs = [
            ("MARKET_REGIME", str(row.get("market_regime") or "UNKNOWN")),
            ("SPY_REGIME", str(row.get("spy_regime") or "UNKNOWN")),
            ("QQQ_REGIME", str(row.get("qqq_regime") or "UNKNOWN")),
            ("VOL_REGIME", str(row.get("vol_regime") or "UNKNOWN")),
            ("MONTH", _period_key(trade_date, "MONTH")),
            ("QUARTER", _period_key(trade_date, "QUARTER")),
            ("YEAR", _period_key(trade_date, "YEAR")),
        ]
        for regime_type, regime_value in regime_pairs:
            grouped[(strategy_name, selection_rule, holding_days, regime_type, regime_value)].append(row)

    results: list[dict[str, object]] = []
    for (strategy_name, selection_rule, holding_days, regime_type, regime_value), rows in grouped.items():
        results.append(
            _aggregate_group(
                backtest_id=str(rows[0].get("backtest_id") or ""),
                strategy_name=strategy_name,
                selection_rule=selection_rule,
                holding_days=holding_days,
                regime_type=regime_type,
                regime_value=regime_value,
                rows=rows,
            )
        )
    return results


def _filter_regime_rows(
    rows: list[dict[str, object]],
    *,
    regime_type: str,
    strategy: str | None = None,
    holding_days: int | None = None,
) -> list[dict[str, object]]:
    filtered = [row for row in rows if row.get("regime_type") == regime_type]
    if strategy:
        filtered = [row for row in filtered if str(row.get("strategy_name")) == strategy]
    if holding_days is not None:
        filtered = [row for row in filtered if int(row.get("holding_days") or 0) == holding_days]
    return filtered


def _select_best_regime(rows: list[dict[str, object]], *, strategy: str, holding_days: int) -> dict[str, object] | None:
    candidates = [
        row for row in rows
        if row.get("regime_type") == "MARKET_REGIME"
        and str(row.get("strategy_name")) == strategy
        and int(row.get("holding_days") or 0) == holding_days
        and row.get("regime_value") != "UNKNOWN"
        and _safe_float(row.get("avg_excess_return_vs_spy")) is not None
    ]
    candidates.sort(
        key=lambda row: (
            -(_safe_float(row.get("avg_excess_return_vs_spy")) or -999.0),
            -(_safe_float(row.get("avg_excess_return_vs_qqq")) or -999.0),
            -(_safe_float(row.get("win_rate_vs_spy")) or -999.0),
            -(_safe_float(row.get("avg_return_pct")) or -999.0),
        )
    )
    return candidates[0] if candidates else None


def _select_worst_regime(rows: list[dict[str, object]], *, strategy: str, holding_days: int) -> dict[str, object] | None:
    candidates = [
        row for row in rows
        if row.get("regime_type") == "MARKET_REGIME"
        and str(row.get("strategy_name")) == strategy
        and int(row.get("holding_days") or 0) == holding_days
        and row.get("regime_value") != "UNKNOWN"
        and _safe_float(row.get("avg_excess_return_vs_spy")) is not None
    ]
    candidates.sort(
        key=lambda row: (
            (_safe_float(row.get("avg_excess_return_vs_spy")) or 999.0),
            (_safe_float(row.get("avg_excess_return_vs_qqq")) or 999.0),
            (_safe_float(row.get("win_rate_vs_spy")) or 999.0),
            (_safe_float(row.get("avg_return_pct")) or 999.0),
        )
    )
    return candidates[0] if candidates else None


def _build_quality_summary(joined_rows: list[dict[str, object]], cfg) -> dict[str, object]:
    unknown_regime_rows = sum(1 for row in joined_rows if str(row.get("market_regime") or "UNKNOWN") == "UNKNOWN")
    missing_regime_rows = sum(1 for row in joined_rows if row.get("market_regime") is None)
    spy_missing_rows = sum(1 for row in joined_rows if row.get("spy_regime") is None)
    qqq_missing_rows = sum(1 for row in joined_rows if row.get("qqq_regime") is None)
    volatility_unavailable_rows = sum(1 for row in joined_rows if row.get("vol_regime") in {None, "UNKNOWN"})

    grouped = Counter(
        (str(row.get("strategy_name") or ""), int(row.get("holding_days") or 0), str(row.get("market_regime") or "UNKNOWN"))
        for row in joined_rows
    )
    insufficient_samples: list[str] = []
    for (strategy_name, holding_days, regime_value), count in sorted(grouped.items()):
        if regime_value != "UNKNOWN" and count < cfg.min_test_days_warning:
            insufficient_samples.append(f"{strategy_name}/{holding_days}D/{regime_value}: {count} days")

    return {
        "joined_rows": len(joined_rows),
        "unknown_regime_rows": unknown_regime_rows,
        "missing_regime_rows": missing_regime_rows,
        "spy_missing_rows": spy_missing_rows,
        "qqq_missing_rows": qqq_missing_rows,
        "volatility_unavailable_rows": volatility_unavailable_rows,
        "insufficient_samples": insufficient_samples,
    }


def _build_interpretation_lines(aggregate_rows: list[dict[str, object]], cfg) -> list[str]:
    del cfg
    lines: list[str] = []
    market_rows = [row for row in aggregate_rows if row.get("regime_type") == "MARKET_REGIME"]
    bull_low_vol = [row for row in market_rows if row.get("regime_value") == "BULL_LOW_VOL"]
    bear_high_vol = [row for row in market_rows if row.get("regime_value") == "BEAR_HIGH_VOL"]
    high_vol_rows = [row for row in market_rows if str(row.get("regime_value")).endswith("HIGH_VOL")]
    bull_rows = [row for row in market_rows if str(row.get("regime_value")).startswith("BULL")]
    bear_rows = [row for row in market_rows if str(row.get("regime_value")).startswith("BEAR")]
    qqq_bull_rows = [row for row in aggregate_rows if row.get("regime_type") == "QQQ_REGIME" and row.get("regime_value") == "QQQ_BULL"]

    if any((_safe_float(row.get("avg_excess_return_vs_spy")) or 0.0) > 0 for row in bull_low_vol):
        lines.append("상승 저변동 구간에서 전략의 SPY 대비 초과성과가 관찰됩니다.")
    if any((_safe_float(row.get("avg_return_pct")) or 0.0) < 0 for row in bear_high_vol):
        lines.append("하락 고변동 구간에서는 절대수익률 방어가 취약할 가능성이 있습니다.")
    if any((_safe_float(row.get("win_rate_vs_spy")) or 0.0) < 0.5 for row in high_vol_rows):
        lines.append("고변동 구간에서 시장 대비 승률이 낮아 리스크 필터 보강이 필요할 수 있습니다.")
    if bull_rows and bear_rows:
        bull_avg = _mean([_safe_float(row.get("avg_excess_return_vs_spy")) for row in bull_rows])
        bear_avg = _mean([_safe_float(row.get("avg_excess_return_vs_spy")) for row in bear_rows])
        if bull_avg is not None and bear_avg is not None and bull_avg > 0 and bear_avg < 0:
            lines.append("상승장에는 강하고 하락장에는 약한 추세추종 성격일 가능성이 있습니다.")
    if any((_safe_float(row.get("avg_excess_return_vs_qqq")) or 0.0) <= 0 for row in qqq_bull_rows):
        lines.append("기술주 강세장에서는 QQQ 대비 우위가 제한적일 수 있습니다.")
    return lines or ["현재 표본만으로는 국면별 우위를 단정하기 어렵습니다. 추가 누적 관찰이 필요합니다."]


def _build_rule_improvement_candidates(aggregate_rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    market_rows = [row for row in aggregate_rows if row.get("regime_type") == "MARKET_REGIME"]
    if any((_safe_float(row.get("avg_excess_return_vs_spy")) or 0.0) < 0 for row in market_rows if row.get("regime_value") == "BEAR_HIGH_VOL"):
        lines.append("BEAR_HIGH_VOL 구간 성과가 약하면 하락 고변동 구간에서 Top N 축소를 검토할 수 있습니다.")
    if any((_safe_float(row.get("avg_return_pct")) or 0.0) < 0 for row in market_rows if str(row.get("regime_value")).endswith("HIGH_VOL")):
        lines.append("HIGH_VOL 구간 손실이 크면 risk_score 가중치 확대를 검토할 수 있습니다.")
    if any((_safe_float(row.get("avg_excess_return_vs_qqq")) or 0.0) < 0 for row in aggregate_rows if row.get("regime_type") == "QQQ_REGIME" and row.get("regime_value") == "QQQ_BEAR"):
        lines.append("QQQ_BEAR 구간에서 약하면 QQQ 상대강도 기준 강화 후보로 볼 수 있습니다.")
    if any((_safe_float(row.get("avg_excess_return_vs_spy")) or 0.0) < 0 for row in market_rows if str(row.get("regime_value")).startswith("SIDEWAYS")):
        lines.append("SIDEWAYS 구간 성과가 약하면 momentum_score 비중 축소를 검토할 수 있습니다.")
    return lines or ["현재 표본에서는 뚜렷한 Rule 개선 후보를 확정하기 어렵습니다."]


def _fixed_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = "  ".join(str(column).ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = ["  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns) for row in rows]
    return "\n".join([header, divider, *body])


def build_console_report(
    *,
    backtest_id: str,
    joined_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    quality: dict[str, object],
    interpretation: list[str],
    improvement_candidates: list[str],
    strategy: str | None,
    holding_days: int | None,
    cfg,
) -> str:
    market_rows = _filter_regime_rows(aggregate_rows, regime_type="MARKET_REGIME", strategy=strategy, holding_days=holding_days)
    period_start = min((row.get("trade_date") for row in joined_rows if isinstance(row.get("trade_date"), date)), default=None)
    period_end = max((row.get("trade_date") for row in joined_rows if isinstance(row.get("trade_date"), date)), default=None)

    best_regime = None
    worst_regime = None
    if strategy and holding_days is not None:
        best_regime = _select_best_regime(aggregate_rows, strategy=strategy, holding_days=holding_days)
        worst_regime = _select_worst_regime(aggregate_rows, strategy=strategy, holding_days=holding_days)

    lines = [
        "[Regime Performance Summary]",
        f"Backtest ID: {backtest_id}",
        f"Period: {_format_date(period_start)} ~ {_format_date(period_end)}",
        f"Strategy: {strategy or 'ALL'}",
        f"Holding Days: {holding_days or 'ALL'}",
        "",
        _fixed_table(
            [
                {
                    "Regime": row.get("regime_value"),
                    "Days": row.get("test_days"),
                    "AvgRet": _fmt_pct(row.get("avg_return_pct")),
                    "ExcessSPY": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "ExcessQQQ": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                    "WinRate": _fmt_pct(row.get("win_rate")),
                    "WinSPY": _fmt_pct(row.get("win_rate_vs_spy")),
                    "WinQQQ": _fmt_pct(row.get("win_rate_vs_qqq")),
                }
                for row in sorted(market_rows, key=lambda item: (str(item.get("strategy_name")), int(item.get("holding_days") or 0), str(item.get("regime_value"))))
            ],
            ["Regime", "Days", "AvgRet", "ExcessSPY", "ExcessQQQ", "WinRate", "WinSPY", "WinQQQ"],
        ),
        "",
    ]
    if best_regime:
        lines.extend(
            [
                "[Best Regime]",
                f"Strategy: {best_regime.get('strategy_name')}",
                f"Holding Days: {best_regime.get('holding_days')}",
                f"Regime: {best_regime.get('regime_value')}",
                f"Avg Excess vs SPY: {_fmt_pct(best_regime.get('avg_excess_return_vs_spy'))}",
                f"Win Rate vs SPY: {_fmt_pct(best_regime.get('win_rate_vs_spy'))}",
                "",
            ]
        )
    if worst_regime:
        lines.extend(
            [
                "[Worst Regime]",
                f"Strategy: {worst_regime.get('strategy_name')}",
                f"Holding Days: {worst_regime.get('holding_days')}",
                f"Regime: {worst_regime.get('regime_value')}",
                f"Avg Excess vs SPY: {_fmt_pct(worst_regime.get('avg_excess_return_vs_spy'))}",
                f"Win Rate vs SPY: {_fmt_pct(worst_regime.get('win_rate_vs_spy'))}",
                "",
            ]
        )
    lines.extend(
        [
            "[Data Quality Summary]",
            f"Joined Rows: {quality.get('joined_rows', 0)}",
            f"UNKNOWN Regime Rows: {quality.get('unknown_regime_rows', 0)}",
            f"Missing Regime Rows: {quality.get('missing_regime_rows', 0)}",
            f"SPY Missing Rows: {quality.get('spy_missing_rows', 0)}",
            f"QQQ Missing Rows: {quality.get('qqq_missing_rows', 0)}",
            f"Volatility Unavailable Rows: {quality.get('volatility_unavailable_rows', 0)}",
            "",
            "[Interpretation]",
            *[f"- {line}" for line in interpretation],
            "",
            "[Rule Improvement Candidates]",
            *[f"- {line}" for line in improvement_candidates],
        ]
    )
    if quality.get("insufficient_samples"):
        lines.extend(["", "[Warnings]"])
        for item in quality["insufficient_samples"][: max(10, cfg.min_test_days_warning)]:
            lines.append(f"- Low sample regime: {item}")
    return "\n".join(lines).strip() + "\n"


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def build_markdown_report(
    *,
    backtest_id: str,
    joined_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    quality: dict[str, object],
    interpretation: list[str],
    improvement_candidates: list[str],
    strategy: str | None,
    holding_days: int | None,
) -> str:
    period_start = min((row.get("trade_date") for row in joined_rows if isinstance(row.get("trade_date"), date)), default=None)
    period_end = max((row.get("trade_date") for row in joined_rows if isinstance(row.get("trade_date"), date)), default=None)
    market_rows = _filter_regime_rows(aggregate_rows, regime_type="MARKET_REGIME", strategy=strategy, holding_days=holding_days)
    monthly_rows = _filter_regime_rows(aggregate_rows, regime_type="MONTH", strategy=strategy, holding_days=holding_days)
    quarterly_rows = _filter_regime_rows(aggregate_rows, regime_type="QUARTER", strategy=strategy, holding_days=holding_days)

    regime_dist_counter = Counter(str(row.get("market_regime") or "UNKNOWN") for row in joined_rows)
    total_dist = sum(regime_dist_counter.values()) or 1
    best_regime = None
    worst_regime = None
    if strategy and holding_days is not None:
        best_regime = _select_best_regime(aggregate_rows, strategy=strategy, holding_days=holding_days)
        worst_regime = _select_worst_regime(aggregate_rows, strategy=strategy, holding_days=holding_days)

    lines = [
        "# 미국주식 랭킹 백테스트 시장국면별 분석 리포트",
        "",
        "## 1. 개요",
        "",
        f"- Backtest ID: {backtest_id}",
        f"- 분석 기간: {_format_date(period_start)} ~ {_format_date(period_end)}",
        f"- 대상 전략: {strategy or 'ALL'}",
        f"- Holding Days: {holding_days or 'ALL'}",
        "- 시장국면 기준:",
        "- SPY 60일 수익률",
        "- SPY 60일 이동평균",
        "- SPY 20일 변동성",
        "- QQQ 기준 보조 판단",
        "",
        "## 2. 시장국면 분포",
        "",
        _markdown_table(
            [
                {"regime": regime, "days": count, "weight": f"{count / total_dist * 100:.1f}%"}
                for regime, count in sorted(regime_dist_counter.items())
            ],
            [("regime", "Market Regime"), ("days", "Days"), ("weight", "Weight")],
        ),
        "",
        "## 3. 전략별/국면별 성과",
        "",
        _markdown_table(
            [
                {
                    "strategy": row.get("strategy_name"),
                    "holding_days": row.get("holding_days"),
                    "regime": row.get("regime_value"),
                    "days": row.get("test_days"),
                    "avg_return": _fmt_pct(row.get("avg_return_pct")),
                    "spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "qqq": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                    "win_rate": _fmt_pct(row.get("win_rate")),
                }
                for row in market_rows
            ],
            [
                ("strategy", "Strategy"),
                ("holding_days", "Holding Days"),
                ("regime", "Regime"),
                ("days", "Days"),
                ("avg_return", "Avg Return"),
                ("spy", "Excess vs SPY"),
                ("qqq", "Excess vs QQQ"),
                ("win_rate", "Win Rate"),
            ],
        ),
        "",
        "## 4. 월별 성과",
        "",
        _markdown_table(
            [
                {
                    "period": row.get("regime_value"),
                    "strategy": row.get("strategy_name"),
                    "holding_days": row.get("holding_days"),
                    "days": row.get("test_days"),
                    "avg_return": _fmt_pct(row.get("avg_return_pct")),
                    "spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "qqq": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                    "win_rate": _fmt_pct(row.get("win_rate")),
                }
                for row in monthly_rows
            ],
            [
                ("period", "Month"),
                ("strategy", "Strategy"),
                ("holding_days", "Holding Days"),
                ("days", "Days"),
                ("avg_return", "Avg Return"),
                ("spy", "Excess vs SPY"),
                ("qqq", "Excess vs QQQ"),
                ("win_rate", "Win Rate"),
            ],
        ),
        "",
        "## 5. 분기별 성과",
        "",
        _markdown_table(
            [
                {
                    "period": row.get("regime_value"),
                    "strategy": row.get("strategy_name"),
                    "holding_days": row.get("holding_days"),
                    "days": row.get("test_days"),
                    "avg_return": _fmt_pct(row.get("avg_return_pct")),
                    "spy": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                    "qqq": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                    "win_rate": _fmt_pct(row.get("win_rate")),
                }
                for row in quarterly_rows
            ],
            [
                ("period", "Quarter"),
                ("strategy", "Strategy"),
                ("holding_days", "Holding Days"),
                ("days", "Days"),
                ("avg_return", "Avg Return"),
                ("spy", "Excess vs SPY"),
                ("qqq", "Excess vs QQQ"),
                ("win_rate", "Win Rate"),
            ],
        ),
        "",
        "## 6. Best Regime / Worst Regime",
        "",
    ]
    if best_regime:
        lines.extend(
            [
                f"- Best Regime: {best_regime.get('regime_value')} / {best_regime.get('strategy_name')} / {best_regime.get('holding_days')}D",
                f"- Avg Excess vs SPY: {_fmt_pct(best_regime.get('avg_excess_return_vs_spy'))}",
                f"- Win Rate vs SPY: {_fmt_pct(best_regime.get('win_rate_vs_spy'))}",
            ]
        )
    if worst_regime:
        lines.extend(
            [
                f"- Worst Regime: {worst_regime.get('regime_value')} / {worst_regime.get('strategy_name')} / {worst_regime.get('holding_days')}D",
                f"- Avg Excess vs SPY: {_fmt_pct(worst_regime.get('avg_excess_return_vs_spy'))}",
                f"- Win Rate vs SPY: {_fmt_pct(worst_regime.get('win_rate_vs_spy'))}",
            ]
        )
    if not best_regime and not worst_regime:
        lines.append("- Best/Worst Regime is unavailable for the current filter.")
    lines.extend(
        [
            "",
            "## 7. 해석 요약",
            "",
            *[f"- {line}" for line in interpretation],
            "",
            "## 8. 데이터 품질 및 누락 현황",
            "",
            f"- 시장국면이 없는 trade_date 수: {quality.get('missing_regime_rows', 0)}",
            f"- UNKNOWN regime 수: {quality.get('unknown_regime_rows', 0)}",
            f"- SPY 가격 누락 일수: {quality.get('spy_missing_rows', 0)}",
            f"- QQQ 가격 누락 일수: {quality.get('qqq_missing_rows', 0)}",
            f"- 변동성 계산 불가 일수: {quality.get('volatility_unavailable_rows', 0)}",
            "",
            "## 9. 주의사항",
            "",
            "- 이 결과는 백테스트 국면 분석입니다.",
            "- 실매매 성과를 보장하지 않습니다.",
            "- 특정 국면 성과만으로 자동매매를 시작하지 않습니다.",
            "- Rule 개선 후보는 Phase 4-4에서 다룹니다.",
            "",
            "## 10. Rule 개선 후보",
            "",
            *[f"- {line}" for line in improvement_candidates],
            "",
        ]
    )
    if quality.get("insufficient_samples"):
        lines.extend(["## 11. 표본 수 Warning", ""])
        for item in quality["insufficient_samples"][:20]:
            lines.append(f"- 주의: {item} 는 표본 수가 적어 해석 신뢰도가 낮을 수 있습니다.")
        lines.append("")
    return "\n".join(lines)


def _resolve_output_path(output_dir: Path, *, backtest_id: str, kind: str, suffix: str) -> Path:
    name = {
        "markdown": f"regime_report_{backtest_id}.{suffix}",
        "regime": f"regime_summary_{backtest_id}.{suffix}",
        "monthly": f"monthly_summary_{backtest_id}.{suffix}",
        "quarterly": f"quarterly_summary_{backtest_id}.{suffix}",
    }[kind]
    return output_dir / name


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> int:
    args = parse_args()
    cfg = load_us_market_regime_config()
    setup_logging(cfg.log_level)
    output_dir = _normalize_output_dir(args.output_dir)

    ensure_us_market_regime_tables()
    joined_rows = _query_joined_rows(backtest_id=args.backtest_id, strategy=args.strategy, holding_days=args.holding_days)
    if not joined_rows:
        print(f"[US_REGIME_ANALYSIS] No backtest summary rows found for backtest_id={args.backtest_id}.")
        return 1

    aggregate_rows = aggregate_regime_rows(joined_rows)
    quality = _build_quality_summary(joined_rows, cfg)
    interpretation = _build_interpretation_lines(aggregate_rows, cfg)
    improvement_candidates = _build_rule_improvement_candidates(aggregate_rows)
    upsert_us_rank_backtest_regime_summary_rows(aggregate_rows)

    if args.format == "console":
        print(
            build_console_report(
                backtest_id=args.backtest_id,
                joined_rows=joined_rows,
                aggregate_rows=aggregate_rows,
                quality=quality,
                interpretation=interpretation,
                improvement_candidates=improvement_candidates,
                strategy=args.strategy,
                holding_days=args.holding_days,
                cfg=cfg,
            ),
            end="",
        )
        if quality.get("missing_regime_rows") == len(joined_rows):
            print("[US_REGIME_ANALYSIS] Market regime rows are missing. Run build_us_market_regime_daily.py first.")
        return 0

    if args.format == "markdown":
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered = build_markdown_report(
            backtest_id=args.backtest_id,
            joined_rows=joined_rows,
            aggregate_rows=aggregate_rows,
            quality=quality,
            interpretation=interpretation,
            improvement_candidates=improvement_candidates,
            strategy=args.strategy,
            holding_days=args.holding_days,
        )
        path = _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="markdown", suffix="md")
        path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    fieldnames = [
        "backtest_id",
        "strategy_name",
        "selection_rule",
        "holding_days",
        "regime_type",
        "regime_value",
        "test_days",
        "selected_count_avg",
        "avg_return_pct",
        "median_return_pct",
        "win_rate",
        "avg_excess_return_vs_spy",
        "avg_excess_return_vs_qqq",
        "avg_excess_return_vs_universe",
        "win_rate_vs_spy",
        "win_rate_vs_qqq",
        "win_rate_vs_universe",
        "best_trade_date",
        "worst_trade_date",
        "data_status",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    regime_path = _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="regime", suffix="csv")
    monthly_path = _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="monthly", suffix="csv")
    quarterly_path = _resolve_output_path(output_dir, backtest_id=args.backtest_id, kind="quarterly", suffix="csv")
    _write_csv(regime_path, [row for row in aggregate_rows if row.get("regime_type") in {"MARKET_REGIME", "SPY_REGIME", "QQQ_REGIME", "VOL_REGIME"}], fieldnames)
    _write_csv(monthly_path, [row for row in aggregate_rows if row.get("regime_type") == "MONTH"], fieldnames)
    _write_csv(quarterly_path, [row for row in aggregate_rows if row.get("regime_type") == "QUARTER"], fieldnames)
    print(f"regime_csv: {regime_path}")
    print(f"monthly_csv: {monthly_path}")
    print(f"quarterly_csv: {quarterly_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
