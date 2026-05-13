from __future__ import annotations

import argparse
import csv
from datetime import date
import math
from pathlib import Path
import statistics
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import load_us_weight_experiment_config
from python.us.us_db import fetch_us_weight_config_rows, fetch_us_weight_experiment_summary_rows, get_us_engine


SUPPORTED_FORMATS = {"console", "markdown", "csv"}


def parse_args() -> argparse.Namespace:
    cfg = load_us_weight_experiment_config()
    parser = argparse.ArgumentParser(description="Report US stock Rule weight experiment results.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--format", default="console", choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--strategy", default="US_RANK_TOP20")
    parser.add_argument("--holding-days", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
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


def _normalize_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _query_experiment_period(experiment_id: str) -> tuple[date | None, date | None]:
    stmt = text(
        """
        SELECT MIN(trade_date) AS start_date, MAX(trade_date) AS end_date
        FROM research.us_stock_rank_weight_experiment_result
        WHERE experiment_id = :experiment_id
        """
    )
    with get_us_engine().connect() as conn:
        row = conn.execute(stmt, {"experiment_id": experiment_id}).mappings().first()
    if not row:
        return None, None
    return row.get("start_date"), row.get("end_date")


def _query_experiment_row_counts(experiment_id: str) -> tuple[int, int]:
    stmt = text(
        """
        SELECT
            (SELECT COUNT(*) FROM research.us_stock_rank_weight_experiment_result WHERE experiment_id = :experiment_id) AS rank_count,
            (SELECT COUNT(*) FROM research.us_stock_weight_experiment_backtest_summary WHERE experiment_id = :experiment_id) AS summary_count
        """
    )
    with get_us_engine().connect() as conn:
        row = conn.execute(stmt, {"experiment_id": experiment_id}).mappings().first()
    if not row:
        return 0, 0
    return int(row.get("rank_count") or 0), int(row.get("summary_count") or 0)


def _weight_config_lookup() -> dict[str, dict[str, object]]:
    return {str(row.get("weight_config_id") or ""): row for row in fetch_us_weight_config_rows()}


def _filter_target_rows(rows: list[dict[str, object]], *, strategy: str, holding_days: int) -> list[dict[str, object]]:
    return [
        row for row in rows
        if str(row.get("strategy_name") or "") == strategy
        and int(row.get("holding_days") or 0) == holding_days
    ]


def _delta(current: object, baseline: object) -> float | None:
    left = _safe_float(current)
    right = _safe_float(baseline)
    if left is None or right is None:
        return None
    return float(left - right)


def _candidate_rank_key(row: dict[str, object]) -> tuple[float, float, float]:
    score_rank = _safe_float(row.get("score_rank"))
    risk_rank = _safe_float(row.get("risk_adjusted_rank"))
    excess_spy = _safe_float(row.get("avg_excess_return_vs_spy"))
    return (
        score_rank if score_rank is not None else 9999.0,
        risk_rank if risk_rank is not None else 9999.0,
        -(excess_spy if excess_spy is not None else -9999.0),
    )


def _select_best_candidate(rows: list[dict[str, object]], baseline_id: str) -> dict[str, object] | None:
    candidates = [row for row in rows if str(row.get("weight_config_id") or "") != baseline_id]
    candidates = [row for row in candidates if row.get("score_rank") is not None]
    candidates.sort(key=_candidate_rank_key)
    return candidates[0] if candidates else None


def _promote_status(row: dict[str, object], baseline_row: dict[str, object] | None, min_test_days: int) -> str:
    if baseline_row is None:
        return "WATCH_CANDIDATE"
    test_days = int(row.get("test_days") or 0)
    delta_spy = _delta(row.get("avg_excess_return_vs_spy"), baseline_row.get("avg_excess_return_vs_spy"))
    delta_win_spy = _delta(row.get("win_rate_vs_spy"), baseline_row.get("win_rate_vs_spy"))
    delta_bear = _delta(row.get("avg_return_bear"), baseline_row.get("avg_return_bear"))
    delta_high_vol = _delta(row.get("avg_return_high_vol"), baseline_row.get("avg_return_high_vol"))
    if (
        delta_spy is not None and delta_spy > 0
        and delta_win_spy is not None and delta_win_spy > 0
        and (delta_bear is None or delta_bear >= 0)
        and (delta_high_vol is None or delta_high_vol >= 0)
        and test_days >= min_test_days
    ):
        return "PROMOTE_CANDIDATE"
    if delta_spy is None or delta_win_spy is None or test_days < min_test_days:
        return "WATCH_CANDIDATE"
    if delta_spy <= 0 and delta_win_spy <= 0 and ((delta_bear is not None and delta_bear < 0) or (delta_high_vol is not None and delta_high_vol < 0)):
        return "REJECT_CANDIDATE"
    return "WATCH_CANDIDATE"


def _pros_cons(row: dict[str, object], baseline_row: dict[str, object] | None) -> tuple[str, str]:
    if baseline_row is None:
        return ("Baseline comparison is unavailable.", "Sample is insufficient for a stronger conclusion.")
    delta_spy = _delta(row.get("avg_excess_return_vs_spy"), baseline_row.get("avg_excess_return_vs_spy"))
    delta_qqq = _delta(row.get("avg_excess_return_vs_qqq"), baseline_row.get("avg_excess_return_vs_qqq"))
    delta_bear = _delta(row.get("avg_return_bear"), baseline_row.get("avg_return_bear"))
    delta_high_vol = _delta(row.get("avg_return_high_vol"), baseline_row.get("avg_return_high_vol"))
    if (delta_spy or 0.0) > 0 and (delta_qqq or 0.0) > 0:
        pro = "SPY/QQQ 대비 초과성과 개선 가능성이 관찰됩니다."
    elif (delta_bear or 0.0) > 0 or (delta_high_vol or 0.0) > 0:
        pro = "하락장 또는 고변동 구간 방어력 개선 가능성이 있습니다."
    else:
        pro = "일부 지표에서 baseline 대비 유지 또는 소폭 개선이 관찰됩니다."
    if (delta_bear or 0.0) < 0 and (delta_high_vol or 0.0) < 0:
        con = "BEAR/HIGH_VOL 구간 방어력이 baseline보다 약해질 수 있습니다."
    elif (delta_spy or 0.0) < 0 and (delta_qqq or 0.0) < 0:
        con = "평균 초과수익률이 baseline보다 악화될 수 있습니다."
    else:
        con = "표본 수와 기간이 제한적이어서 추가 검증이 필요합니다."
    return pro, con


def build_console_report(
    *,
    experiment_id: str,
    target_rows: list[dict[str, object]],
    baseline_row: dict[str, object] | None,
    baseline_id: str,
    period: tuple[date | None, date | None],
    min_test_days: int,
) -> str:
    lines = [
        "[US Stock Rule Weight Experiment Report]",
        f"Experiment ID: {experiment_id}",
        f"Period: {(period[0].isoformat() if period[0] else '')} ~ {(period[1].isoformat() if period[1] else '')}",
        "",
        "[Candidate Summary]",
        "",
    ]
    headers = ["Weight Config", "AvgRet", "ExcessSPY", "ExcessQQQ", "WinSPY", "WinQQQ", "BearRet", "HighVolRet", "RankScore"]
    rows = []
    for row in target_rows:
        rows.append(
            {
                "Weight Config": row.get("weight_config_id"),
                "AvgRet": _fmt_pct(row.get("avg_return_pct")),
                "ExcessSPY": _fmt_pct(row.get("avg_excess_return_vs_spy")),
                "ExcessQQQ": _fmt_pct(row.get("avg_excess_return_vs_qqq")),
                "WinSPY": _fmt_pct(row.get("win_rate_vs_spy")),
                "WinQQQ": _fmt_pct(row.get("win_rate_vs_qqq")),
                "BearRet": _fmt_pct(row.get("avg_return_bear")),
                "HighVolRet": _fmt_pct(row.get("avg_return_high_vol")),
                "RankScore": row.get("score_rank") if row.get("score_rank") is not None else "N/A",
            }
        )
    if rows:
        widths = {name: max(len(name), max(len(str(row[name])) for row in rows)) for name in headers}
        lines.append("  ".join(name.ljust(widths[name]) for name in headers))
        lines.append("  ".join("-" * widths[name] for name in headers))
        for row in rows:
            lines.append("  ".join(str(row[name]).ljust(widths[name]) for name in headers))
    else:
        lines.append("(no rows)")

    best = _select_best_candidate(target_rows, baseline_id)
    lines.extend(["", "[Best Candidate]"])
    if best is None:
        lines.append("No candidate could be promoted from the current sample.")
    else:
        status = _promote_status(best, baseline_row, min_test_days)
        pro, con = _pros_cons(best, baseline_row)
        lines.extend(
            [
                str(best.get("weight_config_id")),
                "",
                "해석:",
                pro,
                con,
                f"판정: {status}",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_markdown_report(
    *,
    experiment_id: str,
    all_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    config_lookup: dict[str, dict[str, object]],
    baseline_row: dict[str, object] | None,
    baseline_id: str,
    period: tuple[date | None, date | None],
    min_test_days: int,
) -> str:
    best = _select_best_candidate(target_rows, baseline_id)
    lines = [
        "# 미국주식 Rule 가중치 실험 리포트",
        "",
        "## 1. 개요",
        "",
        f"- Experiment ID: {experiment_id}",
        f"- 기간: {(period[0].isoformat() if period[0] else '')} ~ {(period[1].isoformat() if period[1] else '')}",
        f"- Baseline: {baseline_id}",
        "",
        "## 2. 가중치 후보 목록",
        "",
        "| Weight Config | Momentum | Relative Strength | Fundamental | Growth | Valuation | Risk Penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_id, cfg_row in sorted(config_lookup.items()):
        lines.append(
            f"| {config_id} | {cfg_row.get('momentum_weight')} | {cfg_row.get('relative_strength_weight')} | "
            f"{cfg_row.get('fundamental_weight')} | {cfg_row.get('growth_weight')} | {cfg_row.get('valuation_weight')} | {cfg_row.get('risk_penalty_weight')} |"
        )
    lines.extend(["", "## 3. 후보별 성과 요약", "", "| Weight Config | Strategy | HD | Avg Return | Excess vs SPY | Excess vs QQQ | Win vs SPY | Win vs QQQ | Rank Score |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in target_rows:
        lines.append(
            f"| {row.get('weight_config_id')} | {row.get('strategy_name')} | {row.get('holding_days')} | {_fmt_pct(row.get('avg_return_pct'))} | "
            f"{_fmt_pct(row.get('avg_excess_return_vs_spy'))} | {_fmt_pct(row.get('avg_excess_return_vs_qqq'))} | "
            f"{_fmt_pct(row.get('win_rate_vs_spy'))} | {_fmt_pct(row.get('win_rate_vs_qqq'))} | {row.get('score_rank') if row.get('score_rank') is not None else 'N/A'} |"
        )
    lines.extend(["", "## 4. Baseline 대비 개선 여부", ""])
    if baseline_row is None:
        lines.append("- Baseline row is unavailable for the current filter.")
    else:
        for row in target_rows:
            if str(row.get("weight_config_id")) == baseline_id:
                continue
            lines.extend(
                [
                    f"### {row.get('weight_config_id')} vs {baseline_id}",
                    "",
                    f"- avg_excess_return_vs_spy_delta: {_fmt_pct(_delta(row.get('avg_excess_return_vs_spy'), baseline_row.get('avg_excess_return_vs_spy')))}",
                    f"- avg_excess_return_vs_qqq_delta: {_fmt_pct(_delta(row.get('avg_excess_return_vs_qqq'), baseline_row.get('avg_excess_return_vs_qqq')))}",
                    f"- win_rate_vs_spy_delta: {_fmt_pct(_delta(row.get('win_rate_vs_spy'), baseline_row.get('win_rate_vs_spy')))}",
                    f"- win_rate_vs_qqq_delta: {_fmt_pct(_delta(row.get('win_rate_vs_qqq'), baseline_row.get('win_rate_vs_qqq')))}",
                    f"- bear_regime_return_delta: {_fmt_pct(_delta(row.get('avg_return_bear'), baseline_row.get('avg_return_bear')))}",
                    f"- high_vol_return_delta: {_fmt_pct(_delta(row.get('avg_return_high_vol'), baseline_row.get('avg_return_high_vol')))}",
                    "",
                ]
            )
    lines.extend(["## 5. 시장국면별 비교", "", "| Weight Config | Bull | Bear | High Vol |", "| --- | ---: | ---: | ---: |"])
    for row in target_rows:
        lines.append(
            f"| {row.get('weight_config_id')} | {_fmt_pct(_safe_float(row.get('avg_return_pct')))} | {_fmt_pct(row.get('avg_return_bear'))} | {_fmt_pct(row.get('avg_return_high_vol'))} |"
        )
    lines.extend(["", "## 6. Best Candidate", ""])
    if best is None:
        lines.append("- Best Candidate is unavailable because score comparison inputs are insufficient.")
    else:
        status = _promote_status(best, baseline_row, min_test_days)
        lines.append(f"- Best Candidate: {best.get('weight_config_id')}")
        lines.append(f"- Status: {status}")
        pro, con = _pros_cons(best, baseline_row)
        lines.append(f"- 장점: {pro}")
        lines.append(f"- 단점: {con}")
    lines.extend(["", "## 7. 후보별 장단점", ""])
    for row in target_rows:
        if baseline_row is None or str(row.get("weight_config_id")) == baseline_id:
            continue
        pro, con = _pros_cons(row, baseline_row)
        lines.append(f"### {row.get('weight_config_id')}")
        lines.append(f"- 장점: {pro}")
        lines.append(f"- 단점: {con}")
        lines.append("")
    lines.extend(
        [
            "## 8. 운영 반영 여부 판단",
            "",
            "- PROMOTE_CANDIDATE: baseline 대비 초과수익률/승률 개선과 방어력 유지가 동시에 관찰되고 표본 수가 충분한 경우",
            "- WATCH_CANDIDATE: 일부 개선은 있으나 표본 수 부족 또는 약점이 남는 경우",
            "- REJECT_CANDIDATE: 주요 지표와 방어력이 함께 악화되는 경우",
            "",
            "## 9. 주의사항",
            "",
            "- 이 결과는 실험용 백테스트입니다.",
            "- 운영 Rule을 자동으로 변경하지 않습니다.",
            "- 실매매 성과를 보장하지 않습니다.",
            "- Forward Test 이전에는 PROMOTE_CANDIDATE도 운영 반영 대상이 아닙니다.",
            "- 특정 기간에만 좋은 후보를 바로 채택하지 않습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> int:
    args = parse_args()
    cfg = load_us_weight_experiment_config()
    output_dir = _normalize_output_dir(args.output_dir)
    all_rows = fetch_us_weight_experiment_summary_rows(experiment_id=args.experiment_id)
    if not all_rows:
        print(f"[US_WEIGHT_EXPERIMENT_REPORT] No summary rows found for experiment_id={args.experiment_id}.")
        return 1

    config_lookup = _weight_config_lookup()
    target_rows = _filter_target_rows(all_rows, strategy=str(args.strategy), holding_days=int(args.holding_days))
    baseline_row = next((row for row in target_rows if str(row.get("weight_config_id") or "") == cfg.baseline_weight_config_id), None)
    period = _query_experiment_period(args.experiment_id)
    rank_count, summary_count = _query_experiment_row_counts(args.experiment_id)

    if args.format == "console":
        print(
            build_console_report(
                experiment_id=args.experiment_id,
                target_rows=target_rows,
                baseline_row=baseline_row,
                baseline_id=cfg.baseline_weight_config_id,
                period=period,
                min_test_days=cfg.min_test_days,
            ),
            end="",
        )
        print(f"Rank Rows: {rank_count}")
        print(f"Summary Rows: {summary_count}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        rendered = build_markdown_report(
            experiment_id=args.experiment_id,
            all_rows=all_rows,
            target_rows=target_rows,
            config_lookup=config_lookup,
            baseline_row=baseline_row,
            baseline_id=cfg.baseline_weight_config_id,
            period=period,
            min_test_days=cfg.min_test_days,
        )
        path = output_dir / f"report_{args.experiment_id}.md"
        path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    _write_csv(
        output_dir / f"summary_{args.experiment_id}.csv",
        all_rows,
        [
            "experiment_id",
            "weight_config_id",
            "strategy_name",
            "selection_rule",
            "holding_days",
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
            "avg_return_bull",
            "avg_return_bear",
            "avg_return_high_vol",
            "score_rank",
            "risk_adjusted_rank",
            "data_status",
        ],
    )
    _write_csv(
        output_dir / f"target_{args.experiment_id}_{args.strategy}_{args.holding_days}d.csv",
        target_rows,
        [
            "experiment_id",
            "weight_config_id",
            "strategy_name",
            "selection_rule",
            "holding_days",
            "test_days",
            "selected_count_avg",
            "avg_return_pct",
            "avg_excess_return_vs_spy",
            "avg_excess_return_vs_qqq",
            "win_rate_vs_spy",
            "win_rate_vs_qqq",
            "avg_return_bear",
            "avg_return_high_vol",
            "score_rank",
            "risk_adjusted_rank",
            "data_status",
        ],
    )
    print(f"summary_csv: {output_dir / f'summary_{args.experiment_id}.csv'}")
    print(f"target_csv: {output_dir / f'target_{args.experiment_id}_{args.strategy}_{args.holding_days}d.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
