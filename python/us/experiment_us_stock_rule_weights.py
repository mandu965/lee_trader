from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
from pathlib import Path
import re
import statistics
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.backtest_us_stock_rank_strategy import (
    StrategySpec,
    _build_price_lookup,
    _compute_benchmark_return,
    _compute_return,
    _decorate_universe_average,
    _parse_int_csv,
    build_summary_row,
    resolve_forward_window,
    resolve_strategy_specs,
    select_strategy_rows,
)
from python.us.calculate_us_stock_rule_scores import (
    _cap,
    _describe_grade_rationale,
    _rank_rows,
    _resolve_grade,
    build_reason_summary,
)
from python.us.us_config import (
    load_us_rule_ranking_config,
    load_us_weight_experiment_config,
    parse_iso_date,
)
from python.us.us_db import (
    ensure_us_weight_experiment_tables,
    fetch_market_regime_rows_between,
    fetch_price_rows_for_tickers_between,
    fetch_rank_component_rows_between,
    fetch_us_weight_config_rows,
    get_us_engine,
    upsert_us_rank_weight_experiment_result_rows,
    upsert_us_rule_weight_config_rows,
    upsert_us_weight_experiment_backtest_summary_rows,
)


LOGGER = logging.getLogger("us_weight_experiment")
BASE_MAX_SCORES = {
    "momentum_score": 25.0,
    "relative_strength_score": 20.0,
    "fundamental_score": 20.0,
    "growth_score": 15.0,
    "valuation_score": 10.0,
    "risk_score": 10.0,
}
STRUCTURAL_EXCLUDE_STATUSES = {"EXCLUDED", "MISSING_PRICE_FEATURE", "LOW_FEATURE_QUALITY"}


@dataclass(frozen=True)
class WeightConfig:
    weight_config_id: str
    description: str
    momentum_weight: float
    relative_strength_weight: float
    fundamental_weight: float
    growth_weight: float
    valuation_weight: float
    risk_penalty_weight: float
    is_active: bool = True
    is_baseline: bool = False


def default_weight_configs() -> list[WeightConfig]:
    return [
        WeightConfig("RULE_V1_BASELINE", "Baseline Rule v1 weights", 25, 20, 20, 15, 10, 10, True, True),
        WeightConfig("RULE_V1_MOMENTUM_PLUS", "Momentum and relative-strength emphasis", 30, 25, 15, 15, 5, 10),
        WeightConfig("RULE_V1_QUALITY_PLUS", "Quality emphasis for drawdown defense", 20, 15, 30, 15, 10, 10),
        WeightConfig("RULE_V1_GROWTH_PLUS", "Growth emphasis candidate", 20, 20, 15, 25, 10, 10),
        WeightConfig("RULE_V1_RISK_DEFENSIVE", "Stronger risk penalty candidate", 20, 15, 20, 15, 10, 20),
        WeightConfig("RULE_V1_VALUE_BALANCED", "Valuation-balanced candidate", 20, 20, 20, 10, 20, 10),
    ]


def setup_logging(level_name: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level_name).upper(), logging.INFO), format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_weight_experiment_config()
    parser = argparse.ArgumentParser(description="Run US stock Rule weight experiments on stored ranking snapshots.")
    parser.add_argument("--start-date", required=True, help="Experiment start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Experiment end date. Format: YYYY-MM-DD.")
    parser.add_argument("--weight-configs", default="ALL", help="Comma-separated config IDs or ALL.")
    parser.add_argument(
        "--holding-days",
        default=",".join(str(value) for value in cfg.default_holding_days),
        help="Comma-separated holding-day list.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(cfg.default_strategies),
        help="Comma-separated strategy aliases such as TOP5,TOP10,TOP20,BUY_OR_BETTER.",
    )
    parser.add_argument("--experiment-id", default=None, help="Optional fixed experiment ID.")
    parser.add_argument("--source", default=None, help="Source rank snapshot tag. Default: rule_v1.")
    parser.add_argument("--top-n", type=int, default=20, help="Custom Top-N size used by strategy resolver.")
    parser.add_argument("--dry-run", action="store_true", help="Compute experiments without DB writes.")
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return float(statistics.fmean(nums))


def _median(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return float(statistics.median(nums))


def _build_experiment_id(*, start_date: date, end_date: date, config_ids: list[str]) -> str:
    token = "_".join(re.sub(r"[^A-Z0-9]+", "_", item.upper()).strip("_") for item in config_ids[:6])
    if len(config_ids) > 6:
        token = f"{token}_PLUS{len(config_ids) - 6}"
    return f"US_RULE_WEIGHT_EXP_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{token}"


def _weight_config_to_row(item: WeightConfig) -> dict[str, object]:
    return {
        "weight_config_id": item.weight_config_id,
        "description": item.description,
        "momentum_weight": item.momentum_weight,
        "relative_strength_weight": item.relative_strength_weight,
        "fundamental_weight": item.fundamental_weight,
        "growth_weight": item.growth_weight,
        "valuation_weight": item.valuation_weight,
        "risk_penalty_weight": item.risk_penalty_weight,
        "is_active": item.is_active,
        "is_baseline": item.is_baseline,
    }


def _load_weight_configs(selected: str, *, persist: bool) -> list[WeightConfig]:
    defaults = default_weight_configs()
    if persist:
        ensure_us_weight_experiment_tables()
        upsert_us_rule_weight_config_rows([_weight_config_to_row(item) for item in defaults])
        rows = fetch_us_weight_config_rows()
        configs = [
            WeightConfig(
                weight_config_id=str(row.get("weight_config_id") or "").upper(),
                description=str(row.get("description") or ""),
                momentum_weight=float(row.get("momentum_weight") or 0.0),
                relative_strength_weight=float(row.get("relative_strength_weight") or 0.0),
                fundamental_weight=float(row.get("fundamental_weight") or 0.0),
                growth_weight=float(row.get("growth_weight") or 0.0),
                valuation_weight=float(row.get("valuation_weight") or 0.0),
                risk_penalty_weight=float(row.get("risk_penalty_weight") or 0.0),
                is_active=bool(row.get("is_active")),
                is_baseline=bool(row.get("is_baseline")),
            )
            for row in rows
            if str(row.get("weight_config_id") or "").strip()
        ]
    else:
        configs = defaults
    if str(selected).strip().upper() == "ALL":
        return [item for item in configs if item.is_active]
    wanted = {part.strip().upper() for part in str(selected).split(",") if part.strip()}
    return [item for item in configs if item.weight_config_id in wanted]


def _score_contributions(source_row: dict[str, object], weight_config: WeightConfig) -> tuple[dict[str, float], float]:
    momentum = (_safe_float(source_row.get("momentum_score")) or 0.0) * weight_config.momentum_weight / BASE_MAX_SCORES["momentum_score"]
    rs = (_safe_float(source_row.get("relative_strength_score")) or 0.0) * weight_config.relative_strength_weight / BASE_MAX_SCORES["relative_strength_score"]
    fundamental = (_safe_float(source_row.get("fundamental_score")) or 0.0) * weight_config.fundamental_weight / BASE_MAX_SCORES["fundamental_score"]
    growth = (_safe_float(source_row.get("growth_score")) or 0.0) * weight_config.growth_weight / BASE_MAX_SCORES["growth_score"]
    valuation = (_safe_float(source_row.get("valuation_score")) or 0.0) * weight_config.valuation_weight / BASE_MAX_SCORES["valuation_score"]
    risk = (_safe_float(source_row.get("risk_score")) or 0.0) * weight_config.risk_penalty_weight / BASE_MAX_SCORES["risk_score"]
    contributions = {
        "momentum_score": round(momentum, 4),
        "relative_strength_score": round(rs, 4),
        "fundamental_score": round(fundamental, 4),
        "growth_score": round(growth, 4),
        "valuation_score": round(valuation, 4),
        "risk_score": round(risk, 4),
    }
    raw_total = sum(contributions.values())
    return contributions, _cap(float(raw_total), lower=0.0, upper=100.0)


def _parse_score_detail_json(raw: object) -> dict[str, object]:
    if raw is None:
        return {}
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {"meta": {"score_detail_parse_warning": True}}
    return payload if isinstance(payload, dict) else {}


def _is_structural_exclude(source_row: dict[str, object]) -> bool:
    status = str(source_row.get("data_status") or "")
    return status in STRUCTURAL_EXCLUDE_STATUSES and bool(source_row.get("exclude_reason"))


def _build_experiment_rank_row(
    *,
    experiment_id: str,
    weight_config: WeightConfig,
    source_row: dict[str, object],
    cfg,
) -> dict[str, object]:
    contributions, total_score = _score_contributions(source_row, weight_config)
    force_exclude = _is_structural_exclude(source_row)
    recommend_grade = _resolve_grade(total_score, exclude=force_exclude, cfg=cfg)
    exclude_reason = source_row.get("exclude_reason") if force_exclude else None
    if recommend_grade == "EXCLUDE" and not exclude_reason:
        exclude_reason = f"Total score below HOLD threshold {cfg.hold_score:.0f}."

    detail = _parse_score_detail_json(source_row.get("score_detail_json"))
    meta = detail.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        detail["meta"] = meta
    meta.update(
        {
            "experiment_id": experiment_id,
            "weight_config_id": weight_config.weight_config_id,
            "weight_config_description": weight_config.description,
            "baseline_total_score": _safe_float(source_row.get("total_score")),
            "weighted_total_score": round(float(total_score), 4),
            "weighted_component_scores": contributions,
            "weight_config": {
                "momentum_weight": weight_config.momentum_weight,
                "relative_strength_weight": weight_config.relative_strength_weight,
                "fundamental_weight": weight_config.fundamental_weight,
                "growth_weight": weight_config.growth_weight,
                "valuation_weight": weight_config.valuation_weight,
                "risk_penalty_weight": weight_config.risk_penalty_weight,
            },
            "grade_rationale": _describe_grade_rationale(
                recommend_grade,
                total_score=float(total_score),
                cfg=cfg,
                forced_exclude_reason=str(exclude_reason) if exclude_reason else None,
            ),
            "baseline_source": source_row.get("source"),
        }
    )

    result = {
        "experiment_id": experiment_id,
        "weight_config_id": weight_config.weight_config_id,
        "trade_date": source_row.get("trade_date"),
        "symbol": source_row.get("symbol"),
        "rank_no": None,
        "recommend_grade": recommend_grade,
        "total_score": round(float(total_score), 4),
        "momentum_score": contributions["momentum_score"],
        "relative_strength_score": contributions["relative_strength_score"],
        "fundamental_score": contributions["fundamental_score"],
        "growth_score": contributions["growth_score"],
        "valuation_score": contributions["valuation_score"],
        "risk_score": contributions["risk_score"],
        "reason_summary": str(source_row.get("reason_summary") or ""),
        "score_detail_json": json.dumps(detail, ensure_ascii=True, sort_keys=True),
        "data_status": source_row.get("data_status"),
        "exclude_reason": exclude_reason,
        "is_etf": source_row.get("is_etf"),
        "_detail": detail,
    }
    result["reason_summary"] = build_reason_summary(result)
    return result


def _group_rows_by_date(rows: list[dict[str, object]]) -> dict[date, list[dict[str, object]]]:
    grouped: dict[date, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        trade_date = row.get("trade_date")
        if isinstance(trade_date, date):
            grouped[trade_date].append(row)
    return grouped


def _group_rows_by_config_and_date(rows: list[dict[str, object]]) -> dict[tuple[str, date], list[dict[str, object]]]:
    grouped: dict[tuple[str, date], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        trade_date = row.get("trade_date")
        config_id = str(row.get("weight_config_id") or "")
        if isinstance(trade_date, date) and config_id:
            grouped[(config_id, trade_date)].append(row)
    return grouped


def _build_selected_strategy_specs(custom_top_n: int, strategy_aliases: tuple[str, ...]) -> list[StrategySpec]:
    wanted = {item.strip().upper() for item in strategy_aliases if item.strip()}
    default_specs = resolve_strategy_specs(custom_top_n=custom_top_n, strategy_filter=None)
    selected = [
        spec
        for spec in default_specs
        if spec.strategy_name.replace("US_RANK_", "") in wanted or spec.strategy_name in wanted
    ]
    if not selected:
        raise ValueError("No strategy specs selected for the experiment.")
    return selected


def _aggregate_experiment_summaries(
    *,
    experiment_id: str,
    daily_summary_rows: list[dict[str, object]],
    regime_lookup: dict[date, dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in daily_summary_rows:
        grouped[(str(row["weight_config_id"]), str(row["strategy_name"]), int(row["holding_days"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (weight_config_id, strategy_name, holding_days), rows in grouped.items():
        selection_rule = str(rows[0].get("selection_rule") or "")
        valid_rows = [row for row in rows if _safe_float(row.get("avg_return_pct")) is not None]
        bull_rows = [row for row in valid_rows if str((regime_lookup.get(row["trade_date"]) or {}).get("market_regime") or "").startswith("BULL")]
        bear_rows = [row for row in valid_rows if str((regime_lookup.get(row["trade_date"]) or {}).get("market_regime") or "").startswith("BEAR")]
        high_vol_rows = [row for row in valid_rows if str((regime_lookup.get(row["trade_date"]) or {}).get("vol_regime") or "") == "HIGH_VOL"]
        if not rows:
            data_status = "NO_DATA"
        elif not valid_rows:
            data_status = "NO_VALID_RETURNS"
        elif any(str(row.get("data_status")) != "OK" for row in rows):
            data_status = "PARTIAL_DATA"
        else:
            data_status = "OK"
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "weight_config_id": weight_config_id,
                "strategy_name": strategy_name,
                "selection_rule": selection_rule,
                "holding_days": holding_days,
                "test_days": len(rows),
                "selected_count_avg": round(_mean([_safe_float(row.get("selected_count")) for row in rows]) or 0.0, 6) if rows else None,
                "avg_return_pct": round(_mean([_safe_float(row.get("avg_return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "median_return_pct": round(_median([_safe_float(row.get("avg_return_pct")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "win_rate": round(_mean([_safe_float(row.get("win_rate")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "avg_excess_return_vs_spy": round(_mean([_safe_float(row.get("avg_excess_return_vs_spy")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "avg_excess_return_vs_qqq": round(_mean([_safe_float(row.get("avg_excess_return_vs_qqq")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "avg_excess_return_vs_universe": round(_mean([_safe_float(row.get("avg_excess_return_vs_universe")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "win_rate_vs_spy": round(_mean([_safe_float(row.get("win_rate_vs_spy")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "win_rate_vs_qqq": round(_mean([_safe_float(row.get("win_rate_vs_qqq")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "win_rate_vs_universe": round(_mean([_safe_float(row.get("win_rate_vs_universe")) for row in valid_rows]) or 0.0, 6) if valid_rows else None,
                "avg_return_bull": round(_mean([_safe_float(row.get("avg_return_pct")) for row in bull_rows]) or 0.0, 6) if bull_rows else None,
                "avg_return_bear": round(_mean([_safe_float(row.get("avg_return_pct")) for row in bear_rows]) or 0.0, 6) if bear_rows else None,
                "avg_return_high_vol": round(_mean([_safe_float(row.get("avg_return_pct")) for row in high_vol_rows]) or 0.0, 6) if high_vol_rows else None,
                "score_rank": None,
                "risk_adjusted_rank": None,
                "data_status": data_status,
            }
        )
    return summary_rows


def _assign_summary_ranks(summary_rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["strategy_name"]), int(row["holding_days"]))].append(row)

    def metric_rank(rows: list[dict[str, object]], key: str) -> dict[str, int]:
        ordered = sorted(
            [row for row in rows if _safe_float(row.get(key)) is not None],
            key=lambda item: (-float(item[key]), str(item["weight_config_id"])),
        )
        return {str(row["weight_config_id"]): idx for idx, row in enumerate(ordered, start=1)}

    for group_rows in groups.values():
        ranks_by_metric = {
            "avg_excess_return_vs_spy": metric_rank(group_rows, "avg_excess_return_vs_spy"),
            "avg_excess_return_vs_qqq": metric_rank(group_rows, "avg_excess_return_vs_qqq"),
            "win_rate_vs_spy": metric_rank(group_rows, "win_rate_vs_spy"),
            "win_rate_vs_qqq": metric_rank(group_rows, "win_rate_vs_qqq"),
            "avg_return_bear": metric_rank(group_rows, "avg_return_bear"),
            "avg_return_high_vol": metric_rank(group_rows, "avg_return_high_vol"),
        }
        risk_ranks = {
            "avg_return_bear": ranks_by_metric["avg_return_bear"],
            "avg_return_high_vol": ranks_by_metric["avg_return_high_vol"],
            "win_rate_vs_spy": ranks_by_metric["win_rate_vs_spy"],
        }
        for row in group_rows:
            config_id = str(row["weight_config_id"])
            component_ranks = [mapping.get(config_id) for mapping in ranks_by_metric.values() if mapping.get(config_id) is not None]
            risk_component_ranks = [mapping.get(config_id) for mapping in risk_ranks.values() if mapping.get(config_id) is not None]
            row["score_rank"] = int(sum(component_ranks)) if component_ranks else None
            row["risk_adjusted_rank"] = int(sum(risk_component_ranks)) if risk_component_ranks else None


def run_experiment(
    *,
    start_date: date,
    end_date: date,
    weight_configs: list[WeightConfig],
    holding_days: list[int],
    strategy_aliases: tuple[str, ...],
    experiment_id: str,
    source: str,
    custom_top_n: int,
    dry_run: bool,
    cfg,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_rows = fetch_rank_component_rows_between(start_date=start_date, end_date=end_date, source=source)
    if not source_rows:
        LOGGER.info("[US_WEIGHT_EXPERIMENT] No source rank rows found for %s ~ %s source=%s", start_date.isoformat(), end_date.isoformat(), source)
        return [], []

    experiment_rows: list[dict[str, object]] = []
    for weight_config in weight_configs:
        local_rows = [
            _build_experiment_rank_row(
                experiment_id=experiment_id,
                weight_config=weight_config,
                source_row=row,
                cfg=cfg,
            )
            for row in source_rows
        ]
        grouped = _group_rows_by_date(local_rows)
        for _, rows in grouped.items():
            _rank_rows(rows)
        experiment_rows.extend(local_rows)

    strategy_specs = _build_selected_strategy_specs(custom_top_n=custom_top_n, strategy_aliases=strategy_aliases)
    grouped_rows = _group_rows_by_config_and_date(experiment_rows)
    symbols = sorted({str(row.get("symbol") or "").upper() for row in experiment_rows if str(row.get("symbol") or "").strip()} | {"SPY", "QQQ"})
    max_holding = max(holding_days)
    price_end_date = end_date + timedelta(days=max_holding * 3 + 30)
    price_rows = fetch_price_rows_for_tickers_between(tickers=symbols, start_date=start_date, end_date=price_end_date)
    price_lookup = _build_price_lookup(price_rows)
    regime_lookup = {row["trade_date"]: row for row in fetch_market_regime_rows_between(start_date=start_date, end_date=end_date) if isinstance(row.get("trade_date"), date)}

    daily_summary_rows: list[dict[str, object]] = []
    for (weight_config_id, trade_day), rows in sorted(grouped_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        bench_cache = {
            hd: {
                "SPY": _compute_benchmark_return(price_lookup, "SPY", trade_date=trade_day, holding_days=hd),
                "QQQ": _compute_benchmark_return(price_lookup, "QQQ", trade_date=trade_day, holding_days=hd),
            }
            for hd in holding_days
        }
        for spec in strategy_specs:
            selected_rows = select_strategy_rows(rows, spec)
            for hd in holding_days:
                detail_rows: list[dict[str, object]] = []
                for rank_row in selected_rows:
                    symbol = str(rank_row.get("symbol") or "").upper()
                    window = resolve_forward_window(price_lookup.get(symbol, []), trade_date=trade_day, holding_days=hd)
                    return_pct = _compute_return(window.entry_price, window.exit_price) if window.data_status == "OK" else None
                    spy_return_pct, spy_status = bench_cache[hd]["SPY"]
                    qqq_return_pct, qqq_status = bench_cache[hd]["QQQ"]
                    data_status = window.data_status
                    if data_status == "OK" and (spy_status is not None or qqq_status is not None):
                        data_status = "PARTIAL_BENCHMARK_DATA"
                    detail_rows.append(
                        {
                            "trade_date": trade_day,
                            "selected_count": len(selected_rows),
                            "symbol": symbol,
                            "return_pct": round(return_pct, 6) if return_pct is not None else None,
                            "spy_return_pct": round(spy_return_pct, 6) if spy_return_pct is not None else None,
                            "qqq_return_pct": round(qqq_return_pct, 6) if qqq_return_pct is not None else None,
                            "universe_avg_return_pct": None,
                            "excess_return_vs_spy": round(return_pct - spy_return_pct, 6) if return_pct is not None and spy_return_pct is not None else None,
                            "excess_return_vs_qqq": round(return_pct - qqq_return_pct, 6) if return_pct is not None and qqq_return_pct is not None else None,
                            "excess_return_vs_universe": None,
                            "win_flag": 1 if return_pct is not None and return_pct > 0 else (0 if return_pct is not None else None),
                            "win_vs_spy_flag": 1 if return_pct is not None and spy_return_pct is not None and (return_pct - spy_return_pct) > 0 else (0 if return_pct is not None and spy_return_pct is not None else None),
                            "win_vs_qqq_flag": 1 if return_pct is not None and qqq_return_pct is not None and (return_pct - qqq_return_pct) > 0 else (0 if return_pct is not None and qqq_return_pct is not None else None),
                            "win_vs_universe_flag": None,
                            "data_status": data_status,
                        }
                    )
                _decorate_universe_average(detail_rows)
                summary = build_summary_row(
                    backtest_id=experiment_id,
                    trade_date=trade_day,
                    holding_days=hd,
                    spec=spec,
                    rows=detail_rows,
                )
                summary["weight_config_id"] = weight_config_id
                daily_summary_rows.append(summary)

    summary_rows = _aggregate_experiment_summaries(
        experiment_id=experiment_id,
        daily_summary_rows=daily_summary_rows,
        regime_lookup=regime_lookup,
    )
    _assign_summary_ranks(summary_rows)

    if not dry_run:
        ensure_us_weight_experiment_tables()
        write_rows = [{key: value for key, value in row.items() if not key.startswith("_") and key not in {"is_etf"}} for row in experiment_rows]
        upsert_us_rank_weight_experiment_result_rows(write_rows)
        upsert_us_weight_experiment_backtest_summary_rows(summary_rows)

    return experiment_rows, summary_rows


def _ensure_db() -> None:
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_WEIGHT_EXPERIMENT] DB connection failed: {exc}") from exc


def main() -> int:
    args = parse_args()
    env_cfg = load_us_weight_experiment_config()
    rank_cfg = load_us_rule_ranking_config()
    setup_logging(env_cfg.log_level)
    start_date = parse_iso_date(args.start_date, field_name="start_date")
    end_date = parse_iso_date(args.end_date, field_name="end_date")
    if start_date is None or end_date is None:
        raise SystemExit("start_date and end_date are required.")
    if start_date > end_date:
        raise SystemExit("start_date must be on or before end_date.")

    weight_configs = _load_weight_configs(args.weight_configs, persist=not bool(args.dry_run))
    if not weight_configs:
        raise SystemExit("No weight configurations selected.")
    holding_days = _parse_int_csv(args.holding_days)
    strategy_aliases = tuple(part.strip().upper() for part in str(args.strategies).split(",") if part.strip())
    experiment_id = str(args.experiment_id or _build_experiment_id(start_date=start_date, end_date=end_date, config_ids=[item.weight_config_id for item in weight_configs])).strip()
    source = str(args.source or rank_cfg.source).strip() or rank_cfg.source

    _ensure_db()
    experiment_rows, summary_rows = run_experiment(
        start_date=start_date,
        end_date=end_date,
        weight_configs=weight_configs,
        holding_days=holding_days,
        strategy_aliases=strategy_aliases,
        experiment_id=experiment_id,
        source=source,
        custom_top_n=max(1, int(args.top_n)),
        dry_run=bool(args.dry_run),
        cfg=rank_cfg,
    )
    LOGGER.info(
        "[US_WEIGHT_EXPERIMENT] finished experiment_id=%s weight_configs=%s rank_rows=%s summary_rows=%s dry_run=%s",
        experiment_id,
        len(weight_configs),
        len(experiment_rows),
        len(summary_rows),
        str(bool(args.dry_run)).lower(),
    )
    if summary_rows:
        for row in sorted(summary_rows, key=lambda item: (str(item["weight_config_id"]), str(item["strategy_name"]), int(item["holding_days"])) )[:20]:
            LOGGER.info(
                "[US_WEIGHT_EXPERIMENT] config=%s strategy=%s hd=%s avg_ret=%s excess_spy=%s score_rank=%s status=%s",
                row["weight_config_id"],
                row["strategy_name"],
                row["holding_days"],
                row.get("avg_return_pct"),
                row.get("avg_excess_return_vs_spy"),
                row.get("score_rank"),
                row.get("data_status"),
            )
    return 0 if experiment_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
