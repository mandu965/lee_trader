from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from payload_store import upsert_json_payload
from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

BUY_TOP5_CSV = DATA_DIR / "buy_candidates_top5.csv"
BUY_TOP8_CSV = DATA_DIR / "buy_candidates_top8.csv"
BUY_TOP10_CSV = DATA_DIR / "buy_candidates_top10.csv"
BENCHMARK_CSV = OUTPUT_DIR / "benchmark_comparison.csv"
FORWARD_CSV = OUTPUT_DIR / "operational_forward_return_by_day.csv"
CONF_CAL_CSV = DATA_DIR / "confidence_calibration_operational.csv"
BUY_COMPARE_CSV = OUTPUT_DIR / "buy_candidate_comparison.csv"
DAILY_STATUS_JSON = OUTPUT_DIR / "operational_daily_cycle_status.json"
THEME_ACCEPTANCE_MD = OUTPUT_DIR / "theme_overlay_acceptance_operational.md"
CONFIDENCE_V2_JSON = DATA_DIR / "confidence_score_v2.json"
TOP20_BUYABILITY_JSON = OUTPUT_DIR / "top20_buyability_report.json"
WALKFORWARD_ACCEPTANCE_JSON = OUTPUT_DIR / "walkforward_acceptance.json"
MARKET_STATUS_VALIDATION_JSON = OUTPUT_DIR / "market_status_validation_report.json"

OUTPUT_MD = OUTPUT_DIR / "operational_buy_gate_report.md"
OUTPUT_JSON = OUTPUT_DIR / "operational_buy_gate.json"
GATE_VERSION = str(
    get_production_config_value(["metadata", "gate_version"], "operational_buy_gate_v1")
)
SCORE_FORMULA_VERSION = str(
    get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v8_return_prob_tech_regime")
)
PORTFOLIO_VERSION = str(
    get_production_config_value(["metadata", "portfolio_version"], "model_portfolio_constructor_v1")
)

STATUS_BUY_ALLOWED = "BUY_ALLOWED"
STATUS_WATCH = "WATCH"
STATUS_HOLD = "HOLD"
STATUS_BLOCK = "BLOCK"

STATUS_PRIORITY = {
    STATUS_BUY_ALLOWED: 0,
    STATUS_WATCH: 1,
    STATUS_HOLD: 2,
    STATUS_BLOCK: 3,
}

BENCHMARK_MAP = {5: 5, 8: 10, 10: 10}
RELEVANT_BENCHMARKS = ["KOSPI", "UNIVERSE_EQUAL_WEIGHT", "BASELINE_RANKING_V2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build operational buy gate from daily operational diagnostics.")
    parser.add_argument("--buy-top5-csv", type=Path, default=BUY_TOP5_CSV)
    parser.add_argument("--buy-top8-csv", type=Path, default=BUY_TOP8_CSV)
    parser.add_argument("--buy-top10-csv", type=Path, default=BUY_TOP10_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_CSV)
    parser.add_argument("--forward-csv", type=Path, default=FORWARD_CSV)
    parser.add_argument("--confidence-calibration-csv", type=Path, default=CONF_CAL_CSV)
    parser.add_argument("--buy-comparison-csv", type=Path, default=BUY_COMPARE_CSV)
    parser.add_argument("--daily-status-json", type=Path, default=DAILY_STATUS_JSON)
    parser.add_argument("--theme-acceptance-md", type=Path, default=THEME_ACCEPTANCE_MD)
    parser.add_argument("--confidence-v2-json", type=Path, default=CONFIDENCE_V2_JSON)
    parser.add_argument("--top20-buyability-json", type=Path, default=TOP20_BUYABILITY_JSON)
    parser.add_argument("--walkforward-acceptance-json", type=Path, default=WALKFORWARD_ACCEPTANCE_JSON)
    parser.add_argument("--market-status-validation-json", type=Path, default=MARKET_STATUS_VALIDATION_JSON)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--primary-bucket", type=int, default=int(get_production_config_value(["buy_gate", "primary_bucket"], 5)), choices=[5, 8, 10])
    parser.add_argument("--min-matured-benchmark-dates", type=int, default=int(get_production_config_value(["buy_gate", "min_matured_benchmark_dates"], 3)))
    parser.add_argument("--max-liquidity-risk-ratio", type=float, default=float(get_production_config_value(["buy_gate", "max_liquidity_risk_ratio"], 0.30)))
    parser.add_argument("--max-very-low-liquidity-ratio", type=float, default=float(get_production_config_value(["buy_gate", "max_very_low_liquidity_ratio"], 0.20)))
    parser.add_argument("--max-relaxed-selection-ratio", type=float, default=float(get_production_config_value(["buy_gate", "max_relaxed_selection_ratio"], 0.60)))
    parser.add_argument("--max-sector-top-share", type=float, default=float(get_production_config_value(["buy_gate", "max_sector_top_share"], 0.40)))
    parser.add_argument("--max-overheat-ratio", type=float, default=float(get_production_config_value(["buy_gate", "max_overheat_ratio"], 0.20)))
    parser.add_argument("--min-confidence-score", type=float, default=float(get_production_config_value(["buy_gate", "min_confidence_score"], 82.0)))
    parser.add_argument("--min-quality-delta-vs-raw", type=float, default=float(get_production_config_value(["buy_gate", "min_quality_delta_vs_raw"], 0.0)))
    parser.add_argument("--block_negative_excess_return", type=float, default=float(get_production_config_value(["buy_gate", "block_negative_excess_return"], -0.01)))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def safe_read_json(path: Path) -> dict[str, object]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def market_regime_diagnostics(payload: dict[str, object]) -> dict[str, object]:
    latest = payload.get("latest") if isinstance(payload.get("latest"), dict) else {}
    diagnosis = payload.get("latest_diagnosis") if isinstance(payload.get("latest_diagnosis"), list) else []
    return {
        "available": bool(latest),
        "latest_date": latest.get("date") if isinstance(latest, dict) else None,
        "regime": latest.get("regime") if isinstance(latest, dict) else None,
        "true_count": latest.get("true_count") if isinstance(latest, dict) else None,
        "close_gt_ma20": latest.get("close_gt_ma20") if isinstance(latest, dict) else None,
        "ma20_gt_ma60": latest.get("ma20_gt_ma60") if isinstance(latest, dict) else None,
        "recent_20d_return_gt_0.03": latest.get("recent_20d_return_gt_0.03") if isinstance(latest, dict) else None,
        "breadth_20d_gt_0.55": latest.get("breadth_20d_gt_0.55") if isinstance(latest, dict) else None,
        "volatility_risk_flag": latest.get("volatility_risk_flag") if isinstance(latest, dict) else None,
        "breadth_20d": latest.get("breadth_20d") if isinstance(latest, dict) else None,
        "recent_20d_return": latest.get("recent_20d_return") if isinstance(latest, dict) else None,
        "volatility_5d": latest.get("volatility_5d") if isinstance(latest, dict) else None,
        "reason": latest.get("regime_reason") if isinstance(latest, dict) else None,
        "diagnosis": diagnosis[:4],
    }


def load_candidates(path: Path, bucket: int) -> pd.DataFrame:
    df = safe_read_csv(path, dtype={"code": str})
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["target_bucket"] = bucket
    return df


def parse_theme_churn_gate(path: Path) -> tuple[str | None, str | None]:
    resolved = _resolve(path)
    if not resolved.exists():
        return None, None
    text = resolved.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| top20_churn_not_excessive"):
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) >= 2:
                return parts[1], line.strip()
    return None, None


def actual_theme_top_share(candidate_df: pd.DataFrame) -> float:
    if candidate_df.empty or "dominant_theme" not in candidate_df.columns:
        return 0.0
    themes = (
        candidate_df["dominant_theme"]
        .fillna("(none)")
        .astype(str)
        .replace({"": "(none)", "nan": "(none)"})
    )
    actual = themes.loc[themes.ne("(none)")]
    if actual.empty:
        return 0.0
    return float(actual.value_counts(normalize=True).iloc[0])


def sector_top_share(candidate_df: pd.DataFrame) -> float:
    if candidate_df.empty or "sector" not in candidate_df.columns:
        return 0.0
    sector = candidate_df["sector"].fillna("(unknown)").astype(str)
    if sector.empty:
        return 0.0
    return float(sector.value_counts(normalize=True).iloc[0])


def candidate_static_metrics(candidate_df: pd.DataFrame) -> dict[str, object]:
    if candidate_df.empty:
        return {
            "row_count": 0,
            "avg_final_score": None,
            "avg_confidence_score": None,
            "avg_liquidity_score": None,
            "liquidity_risk_ratio": None,
            "very_low_liquidity_ratio": None,
            "relaxed_selection_ratio": None,
            "sector_top_share": None,
            "theme_top_share_actual": None,
        }

    liquidity_score = pd.to_numeric(candidate_df.get("liquidity_score"), errors="coerce")
    liquidity_gate_source = candidate_df.get("liquidity_gate_source", pd.Series("", index=candidate_df.index)).fillna("").astype(str)
    selection_stage = candidate_df.get("selection_stage", pd.Series("", index=candidate_df.index)).fillna("").astype(str)

    liquidity_risk = liquidity_score.lt(15).fillna(False) | liquidity_gate_source.eq("trading_value_only")
    very_low_liquidity = liquidity_score.lt(10).fillna(False)
    relaxed_selection = selection_stage.ne("strict").fillna(False)

    return {
        "row_count": int(len(candidate_df)),
        "avg_final_score": float(pd.to_numeric(candidate_df.get("final_score"), errors="coerce").mean()),
        "avg_confidence_score": float(pd.to_numeric(candidate_df.get("confidence_score"), errors="coerce").mean()),
        "avg_liquidity_score": float(liquidity_score.mean()),
        "liquidity_risk_ratio": float(liquidity_risk.mean()),
        "very_low_liquidity_ratio": float(very_low_liquidity.mean()),
        "relaxed_selection_ratio": float(relaxed_selection.mean()),
        "sector_top_share": sector_top_share(candidate_df),
        "theme_top_share_actual": actual_theme_top_share(candidate_df),
    }


def benchmark_metrics(benchmark_df: pd.DataFrame, bucket: int) -> dict[str, object]:
    if benchmark_df.empty:
        return {
            "benchmark_bucket": BENCHMARK_MAP[bucket],
            "matured_dates_max": 0,
            "available_benchmarks": 0,
            "avg_excess_return_kospi": None,
            "avg_excess_return_baseline": None,
            "excess_confirmed": False,
            "negative_excess_block": False,
        }

    benchmark_bucket = BENCHMARK_MAP[bucket]
    subset = benchmark_df.loc[
        benchmark_df["top_n"].eq(benchmark_bucket)
        & benchmark_df["benchmark_name"].isin(RELEVANT_BENCHMARKS)
    ].copy()
    if subset.empty:
        return {
            "benchmark_bucket": benchmark_bucket,
            "matured_dates_max": 0,
            "available_benchmarks": 0,
            "avg_excess_return_kospi": None,
            "avg_excess_return_baseline": None,
            "excess_confirmed": False,
            "negative_excess_block": False,
        }

    subset["dates_matured"] = pd.to_numeric(subset["dates_matured"], errors="coerce")
    subset["avg_excess_return"] = pd.to_numeric(subset["avg_excess_return"], errors="coerce")
    matured = subset.loc[subset["dates_matured"].fillna(0) > 0].copy()

    def _metric(name: str) -> float | None:
        match = matured.loc[matured["benchmark_name"].eq(name), "avg_excess_return"]
        return None if match.empty else float(match.iloc[0])

    kospi = _metric("KOSPI")
    baseline = _metric("BASELINE_RANKING_V2")
    universe = _metric("UNIVERSE_EQUAL_WEIGHT")
    excess_confirmed = (
        kospi is not None
        and baseline is not None
        and kospi > 0
        and baseline > 0
    )
    negative_excess_block = any(
        metric is not None and metric <= args_global.block_negative_excess_return
        for metric in [kospi, baseline, universe]
    )

    return {
        "benchmark_bucket": benchmark_bucket,
        "matured_dates_max": int(matured["dates_matured"].max()) if not matured.empty else 0,
        "available_benchmarks": int(len(subset)),
        "avg_excess_return_kospi": kospi,
        "avg_excess_return_baseline": baseline,
        "avg_excess_return_universe": universe,
        "excess_confirmed": excess_confirmed,
        "negative_excess_block": negative_excess_block,
    }


def forward_metrics(forward_df: pd.DataFrame, bucket: int) -> dict[str, object]:
    if forward_df.empty:
        return {
            "forward_bucket": BENCHMARK_MAP[bucket],
            "matured_count_max": 0,
            "latest_maturity_state": None,
            "avg_return_latest": None,
            "avg_excess_return_latest": None,
        }

    forward_bucket = BENCHMARK_MAP[bucket]
    subset = forward_df.loc[forward_df["top_n"].eq(forward_bucket)].copy()
    if subset.empty:
        return {
            "forward_bucket": forward_bucket,
            "matured_count_max": 0,
            "latest_maturity_state": None,
            "avg_return_latest": None,
            "avg_excess_return_latest": None,
        }

    subset["matured_count"] = pd.to_numeric(subset["matured_count"], errors="coerce")
    latest = subset.sort_values(["asof_date", "horizon_days"]).iloc[-1]
    return {
        "forward_bucket": forward_bucket,
        "matured_count_max": int(subset["matured_count"].fillna(0).max()),
        "latest_maturity_state": str(latest.get("maturity_state")) if pd.notna(latest.get("maturity_state")) else None,
        "avg_return_latest": pd.to_numeric(latest.get("avg_return"), errors="coerce"),
        "avg_excess_return_latest": pd.to_numeric(latest.get("avg_excess_return"), errors="coerce"),
    }


def confidence_reliability(conf_df: pd.DataFrame) -> dict[str, object]:
    if conf_df.empty:
        return {
            "reliable": False,
            "stable_bucket_count_5d": 0,
            "monotonic_hit_rate_5d": False,
            "avg_operational_minus_provisional_5d": None,
            "avg_calibrated_confidence_5d": None,
        }

    work = conf_df.loc[
        conf_df["source_mode"].astype(str).eq("operational")
        & pd.to_numeric(conf_df["horizon_days"], errors="coerce").eq(5)
    ].copy()
    if work.empty:
        return {
            "reliable": False,
            "stable_bucket_count_5d": 0,
            "monotonic_hit_rate_5d": False,
            "avg_operational_minus_provisional_5d": None,
            "avg_calibrated_confidence_5d": None,
        }

    work["rows"] = pd.to_numeric(work["rows"], errors="coerce")
    work["hit_rate"] = pd.to_numeric(work["hit_rate"], errors="coerce")
    work["bucket_mid"] = pd.to_numeric(work["bucket_mid"], errors="coerce")
    work["operational_minus_provisional"] = pd.to_numeric(work["operational_minus_provisional"], errors="coerce")
    work["calibrated_confidence_score"] = pd.to_numeric(work["calibrated_confidence_score"], errors="coerce")

    stable = work.loc[work["status"].astype(str).eq("stable") & work["rows"].fillna(0).ge(20)].sort_values("bucket_mid")
    monotonic = bool(stable["hit_rate"].fillna(-1).is_monotonic_increasing) if len(stable) >= 2 else False
    reliable = len(stable) >= 2 and monotonic
    return {
        "reliable": reliable,
        "stable_bucket_count_5d": int(len(stable)),
        "monotonic_hit_rate_5d": monotonic,
        "avg_operational_minus_provisional_5d": float(stable["operational_minus_provisional"].mean()) if not stable.empty else None,
        "avg_calibrated_confidence_5d": float(stable["calibrated_confidence_score"].mean()) if not stable.empty else None,
    }


def comparison_metrics(compare_df: pd.DataFrame, bucket: int) -> dict[str, object]:
    if compare_df.empty:
        return {
            "avg_final_score_delta_vs_raw": None,
            "avg_confidence_delta_vs_raw": None,
            "avg_ret_score_delta_vs_raw": None,
            "avg_tech_score_delta_vs_raw": None,
            "overheat_soft_ratio": None,
            "sector_top_share": None,
        }

    cohort = compare_df.loc[compare_df["cohort_name"].eq(f"buy_top{bucket}")].copy()
    if cohort.empty:
        return {
            "avg_final_score_delta_vs_raw": None,
            "avg_confidence_delta_vs_raw": None,
            "avg_ret_score_delta_vs_raw": None,
            "avg_tech_score_delta_vs_raw": None,
            "overheat_soft_ratio": None,
            "sector_top_share": None,
        }
    row = cohort.iloc[0]
    return {
        "avg_final_score_delta_vs_raw": pd.to_numeric(row.get("delta_vs_raw_top20_avg_final_score"), errors="coerce"),
        "avg_confidence_delta_vs_raw": pd.to_numeric(row.get("delta_vs_raw_top20_avg_confidence_score"), errors="coerce"),
        "avg_ret_score_delta_vs_raw": pd.to_numeric(row.get("delta_vs_raw_top20_avg_ret_score"), errors="coerce"),
        "avg_tech_score_delta_vs_raw": pd.to_numeric(row.get("delta_vs_raw_top20_avg_tech_score"), errors="coerce"),
        "overheat_soft_ratio": pd.to_numeric(row.get("overheat_soft_ratio"), errors="coerce"),
        "sector_top_share": pd.to_numeric(row.get("sector_top_share"), errors="coerce"),
    }


def confidence_v2_metrics(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "available": bool(payload),
        "trusted_ratio_top20": pd.to_numeric(summary.get("trusted_ratio_top20"), errors="coerce"),
        "provisional_or_better_ratio_top20": pd.to_numeric(summary.get("provisional_or_better_ratio_top20"), errors="coerce"),
        "mean_raw_confidence_v2": pd.to_numeric(summary.get("mean_raw_confidence_v2"), errors="coerce"),
    }


def buyability_metrics(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    counts = summary.get("buyability_counts") if isinstance(summary.get("buyability_counts"), dict) else {}
    top_n = int(summary.get("top_n") or 0)
    buy_now = int(counts.get("BUY_NOW", 0))
    paper_only = int(counts.get("PAPER_ONLY", 0))
    blocked = int(counts.get("BLOCK", 0))
    watchlist = int(counts.get("WATCHLIST", 0))
    return {
        "available": bool(payload),
        "top_n": top_n,
        "buy_now_count": buy_now,
        "paper_only_count": paper_only,
        "blocked_count": blocked,
        "watchlist_count": watchlist,
        "buy_now_ratio": float(buy_now / top_n) if top_n else None,
        "blocked_ratio": float(blocked / top_n) if top_n else None,
    }


def walkforward_acceptance_metrics(payload: dict[str, object]) -> dict[str, object]:
    return {
        "available": bool(payload),
        "status": payload.get("status"),
        "reason_codes": payload.get("reason_codes") if isinstance(payload.get("reason_codes"), list) else [],
    }


def build_interpretation(
    primary: dict[str, object],
    overall_status: str,
    daily_status: dict[str, object],
) -> dict[str, object]:
    benchmark = primary.get("benchmark", {}) if isinstance(primary.get("benchmark"), dict) else {}
    confidence = primary.get("confidence", {}) if isinstance(primary.get("confidence"), dict) else {}
    buyability = primary.get("buyability", {}) if isinstance(primary.get("buyability"), dict) else {}
    walkforward = primary.get("walkforward_acceptance", {}) if isinstance(primary.get("walkforward_acceptance"), dict) else {}
    market_regime = primary.get("market_regime", {}) if isinstance(primary.get("market_regime"), dict) else {}

    critical_reasons: list[str] = []
    matured_dates = int(benchmark.get("matured_dates_max") or 0)
    if matured_dates <= 0:
        critical_reasons.append("benchmark matured dates가 아직 0이라 실전 excess return 근거가 없습니다.")
    if walkforward.get("status") == "REJECTED":
        critical_reasons.append("walk-forward acceptance가 REJECTED라 production 매수 허용 조건을 충족하지 못했습니다.")
    if confidence.get("reliable") is False:
        critical_reasons.append("operational confidence calibration이 아직 reliable 상태가 아닙니다.")
    blocked_ratio = pd.to_numeric(buyability.get("blocked_ratio"), errors="coerce")
    if pd.notna(blocked_ratio) and float(blocked_ratio) >= 0.30:
        critical_reasons.append(f"top20 blocked ratio가 {_fmt_pct(blocked_ratio)}로 높습니다.")
    regime = str(market_regime.get("regime") or "").strip()
    if regime:
        critical_reasons.append(
            f"market regime는 {regime}이며 breadth={_fmt_num(market_regime.get('breadth_20d'), 3)}, vol_5d={_fmt_num(market_regime.get('volatility_5d'), 4)} 상태입니다."
        )

    if overall_status == STATUS_BLOCK:
        summary_decision = "오늘은 실매수 진행 금지"
        summary_reason = "walk-forward rejection 또는 실전 차단 신호가 남아 있어 score 상위 종목이라도 매수 승인 상태가 아닙니다."
        action_guide = "신규 실매수는 멈추고, walk-forward rejection 원인과 blocked top20 사유를 먼저 점검합니다."
    elif overall_status == STATUS_HOLD:
        summary_decision = "오늘은 보류"
        summary_reason = "랭킹 품질은 볼 수 있지만 matured operational evidence가 아직 부족합니다."
        action_guide = "신규 매수보다 benchmark maturation과 calibration 업데이트를 우선 확인합니다."
    elif overall_status == STATUS_WATCH:
        summary_decision = "조건부 관찰"
        summary_reason = "즉시 차단 수준은 아니지만 benchmark 확인이나 calibration 신뢰도가 충분하지 않습니다."
        action_guide = "watchlist 중심으로만 운영하고, live buy 전 마지막 blocker 해소 여부를 확인합니다."
    else:
        summary_decision = "실매수 가능 구간"
        summary_reason = "benchmark excess, calibration, liquidity, concentration 조건이 모두 허용 구간입니다."
        action_guide = "buy_now 후보만 대상으로 주문 전 최종 체결/유동성 체크를 진행합니다."

    if str(daily_status.get("overall_status") or "").upper() == "WAIT":
        action_guide = "daily cycle이 WAIT 상태라 신규 액션보다 파이프라인 완료 여부를 먼저 확인합니다."

    return {
        "summary_decision": summary_decision,
        "summary_reason": summary_reason,
        "action_guide": action_guide,
        "critical_reasons": critical_reasons[:6],
    }


def bucket_decision(
    bucket: int,
    candidate_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    forward_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    confidence_v2_payload: dict[str, object],
    buyability_payload: dict[str, object],
    walkforward_acceptance_payload: dict[str, object],
    market_regime_payload: dict[str, object],
    theme_churn_status: str | None,
    daily_status: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    reasons: list[str] = []
    static = candidate_static_metrics(candidate_df)
    bench = benchmark_metrics(benchmark_df, bucket)
    forward = forward_metrics(forward_df, bucket)
    conf = confidence_reliability(conf_df)
    compare = comparison_metrics(compare_df, bucket)
    conf_v2 = confidence_v2_metrics(confidence_v2_payload)
    buyability = buyability_metrics(buyability_payload)
    acceptance = walkforward_acceptance_metrics(walkforward_acceptance_payload)

    status = STATUS_BUY_ALLOWED
    if candidate_df.empty:
        status = STATUS_HOLD
        reasons.append("buy candidate file is missing or empty")
    else:
        if static["liquidity_risk_ratio"] is not None and static["liquidity_risk_ratio"] >= args.max_liquidity_risk_ratio:
            status = STATUS_BLOCK
            reasons.append(
                f"liquidity risk ratio {_fmt_pct(static['liquidity_risk_ratio'])} exceeds {_fmt_pct(args.max_liquidity_risk_ratio)}"
            )
        if static["very_low_liquidity_ratio"] is not None and static["very_low_liquidity_ratio"] >= args.max_very_low_liquidity_ratio:
            status = STATUS_BLOCK
            reasons.append(
                f"very low liquidity ratio {_fmt_pct(static['very_low_liquidity_ratio'])} exceeds {_fmt_pct(args.max_very_low_liquidity_ratio)}"
            )
        if static["sector_top_share"] is not None and static["sector_top_share"] > args.max_sector_top_share:
            status = STATUS_BLOCK
            reasons.append(
                f"sector concentration {_fmt_pct(static['sector_top_share'])} exceeds {_fmt_pct(args.max_sector_top_share)}"
            )
        if bench["negative_excess_block"]:
            status = STATUS_BLOCK
            reasons.append("benchmark excess return is materially negative on available matured evidence")

        if status != STATUS_BLOCK:
            matured_ok = bench["matured_dates_max"] >= args.min_matured_benchmark_dates
            if not matured_ok:
                status = STATUS_HOLD
                reasons.append(
                    f"matured benchmark dates {bench['matured_dates_max']} are below required {args.min_matured_benchmark_dates}"
                )
            elif not bench["excess_confirmed"]:
                status = STATUS_WATCH
                reasons.append("benchmark-relative excess return is not confirmed, so BUY_ALLOWED is not permitted")

            if not conf["reliable"]:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append("operational confidence calibration is not reliable enough yet")

            if conf_v2["available"] and conf_v2["trusted_ratio_top20"] is not None and conf_v2["trusted_ratio_top20"] < 0.20:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append("confidence v2 trusted ratio in top20 is still low")

            if static["avg_confidence_score"] is not None and static["avg_confidence_score"] < args.min_confidence_score:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append(
                    f"average confidence {_fmt_num(static['avg_confidence_score'])} is below {_fmt_num(args.min_confidence_score)}"
                )

            if compare["avg_final_score_delta_vs_raw"] is not None and compare["avg_final_score_delta_vs_raw"] < args.min_quality_delta_vs_raw:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append("buy candidate quality uplift versus raw top20 is not positive")

            if compare["overheat_soft_ratio"] is not None and compare["overheat_soft_ratio"] > args.max_overheat_ratio:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append(
                    f"overheat ratio {_fmt_pct(compare['overheat_soft_ratio'])} exceeds {_fmt_pct(args.max_overheat_ratio)}"
                )

            if static["relaxed_selection_ratio"] is not None and static["relaxed_selection_ratio"] > args.max_relaxed_selection_ratio:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append(
                    f"relaxed-selection ratio {_fmt_pct(static['relaxed_selection_ratio'])} is high"
                )

            if acceptance["available"]:
                if acceptance["status"] == "REJECTED":
                    status = STATUS_BLOCK
                    reasons.append("walk-forward acceptance is rejected")
                elif acceptance["status"] == "CONDITIONAL" and status == STATUS_BUY_ALLOWED:
                    status = STATUS_HOLD if not matured_ok else STATUS_WATCH
                    reasons.append("walk-forward acceptance is conditional")

            if buyability["available"] and buyability["blocked_ratio"] is not None and buyability["blocked_ratio"] >= 0.30:
                if status == STATUS_BUY_ALLOWED:
                    status = STATUS_WATCH
                reasons.append("too many top20 names are operationally blocked")

    if theme_churn_status == "FAIL" and status != STATUS_BLOCK:
        status = STATUS_WATCH if STATUS_PRIORITY[status] < STATUS_PRIORITY[STATUS_WATCH] else status
        reasons.append("theme overlay churn diagnostic is not stable")

    daily_overall = str(daily_status.get("overall_status", "")).upper()
    if daily_overall == "WAIT" and status == STATUS_BUY_ALLOWED:
        status = STATUS_HOLD
        reasons.append("daily operational cycle is still in WAIT state")

    market_regime = market_regime_diagnostics(market_regime_payload)
    if market_regime["available"] and market_regime.get("regime"):
        reasons.append(
            f"market regime is {market_regime['regime']} "
            f"(true_count={market_regime.get('true_count')}, breadth={_fmt_num(market_regime.get('breadth_20d'), 3)}, vol_5d={_fmt_num(market_regime.get('volatility_5d'), 4)})"
        )

    if not reasons:
        reasons.append("matured excess return, calibration, liquidity, and concentration checks all passed")

    return {
        "bucket": bucket,
        "status": status,
        "reason_summary": "; ".join(reasons),
        "static": static,
        "benchmark": bench,
        "forward": forward,
        "confidence": conf,
        "confidence_v2": conf_v2,
        "comparison": compare,
        "buyability": buyability,
        "walkforward_acceptance": acceptance,
        "market_regime": market_regime,
    }


def build_markdown(
    *,
    asof_date: str,
    primary_bucket: int,
    overall_status: str,
    decisions: list[dict[str, object]],
    theme_churn_status: str | None,
    theme_churn_line: str | None,
    daily_status: dict[str, object],
    market_regime_payload: dict[str, object],
    interpretation: dict[str, object],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_rows = []
    for item in decisions:
        summary_rows.append(
            {
                "bucket": f"top{item['bucket']}",
                "status": item["status"],
                "avg_final_score": _fmt_num(item["static"]["avg_final_score"]),
                "avg_confidence": _fmt_num(item["static"]["avg_confidence_score"]),
                "liquidity_risk_ratio": _fmt_pct(item["static"]["liquidity_risk_ratio"]),
                "sector_top_share": _fmt_pct(item["static"]["sector_top_share"]),
                "benchmark_matured_dates": item["benchmark"]["matured_dates_max"],
                "excess_confirmed": item["benchmark"]["excess_confirmed"],
                "confidence_reliable": item["confidence"]["reliable"],
                "wf_acceptance": item["walkforward_acceptance"]["status"],
                "reason_summary": item["reason_summary"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    lines = [
        "# Operational Buy Gate Report",
        "",
        f"- generated_at: {generated_at}",
        f"- asof_date: {asof_date}",
        f"- gate_version: {GATE_VERSION}",
        f"- score_formula_version: {SCORE_FORMULA_VERSION}",
        f"- portfolio_version: {PORTFOLIO_VERSION}",
        f"- primary_bucket: top{primary_bucket}",
        f"- overall_status: {overall_status}",
        f"- daily_cycle_status: {daily_status.get('overall_status', 'NA')}",
        f"- theme_churn_status: {theme_churn_status or 'NA'}",
        "",
        "## Executive Summary",
        f"- today_decision: {interpretation.get('summary_decision', 'NA')}",
        f"- why: {interpretation.get('summary_reason', 'NA')}",
        f"- action: {interpretation.get('action_guide', 'NA')}",
        "",
    ]

    critical_reasons = interpretation.get("critical_reasons") if isinstance(interpretation, dict) else []
    if critical_reasons:
        lines.extend(
            [
                "## Key Blockers",
                *[f"- {item}" for item in critical_reasons],
                "",
            ]
        )

    lines.extend([
        "## Bucket Decisions",
        _markdown_table(
            summary_df,
            [
                "bucket",
                "status",
                "avg_final_score",
                "avg_confidence",
                "liquidity_risk_ratio",
                "sector_top_share",
                "benchmark_matured_dates",
                "excess_confirmed",
                "confidence_reliable",
                "wf_acceptance",
                "reason_summary",
            ],
        ),
    ])

    market_regime = market_regime_diagnostics(market_regime_payload)
    if market_regime["available"]:
        lines.extend(
            [
                "",
                "## Market Regime",
                f"- latest_date: {market_regime.get('latest_date') or 'NA'}",
                f"- regime: {market_regime.get('regime') or 'NA'}",
                f"- true_count: {market_regime.get('true_count') if market_regime.get('true_count') is not None else 'NA'} / 5",
                f"- breadth_20d: {_fmt_num(market_regime.get('breadth_20d'), 3)}",
                f"- recent_20d_return: {_fmt_pct(market_regime.get('recent_20d_return'))}",
                f"- volatility_5d: {_fmt_num(market_regime.get('volatility_5d'), 4)}",
                f"- close_gt_ma20: {market_regime.get('close_gt_ma20')}",
                f"- ma20_gt_ma60: {market_regime.get('ma20_gt_ma60')}",
                f"- recent_20d_return_gt_0.03: {market_regime.get('recent_20d_return_gt_0.03')}",
                f"- breadth_20d_gt_0.55: {market_regime.get('breadth_20d_gt_0.55')}",
                f"- volatility_risk_flag: {market_regime.get('volatility_risk_flag')}",
            ]
        )
        for item in market_regime.get("diagnosis") or []:
            lines.append(f"- {item}")

    if theme_churn_line:
        lines.extend(
            [
                "",
                "## Theme Diagnostic",
                f"- source: {theme_churn_line}",
            ]
        )

    for item in decisions:
        lines.extend(
            [
                "",
                f"## Top{item['bucket']} Detail",
                f"- status: {item['status']}",
                f"- reason_summary: {item['reason_summary']}",
                f"- avg_final_score: {_fmt_num(item['static']['avg_final_score'])}",
                f"- avg_confidence_score: {_fmt_num(item['static']['avg_confidence_score'])}",
                f"- avg_liquidity_score: {_fmt_num(item['static']['avg_liquidity_score'])}",
                f"- liquidity_risk_ratio: {_fmt_pct(item['static']['liquidity_risk_ratio'])}",
                f"- very_low_liquidity_ratio: {_fmt_pct(item['static']['very_low_liquidity_ratio'])}",
                f"- relaxed_selection_ratio: {_fmt_pct(item['static']['relaxed_selection_ratio'])}",
                f"- sector_top_share: {_fmt_pct(item['static']['sector_top_share'])}",
                f"- actual_theme_top_share: {_fmt_pct(item['static']['theme_top_share_actual'])}",
                f"- benchmark_bucket_used: top{item['benchmark']['benchmark_bucket']}",
                f"- benchmark_matured_dates_max: {item['benchmark']['matured_dates_max']}",
                f"- kospi_excess_return: {_fmt_pct(item['benchmark']['avg_excess_return_kospi'])}",
                f"- baseline_excess_return: {_fmt_pct(item['benchmark']['avg_excess_return_baseline'])}",
                f"- universe_excess_return: {_fmt_pct(item['benchmark']['avg_excess_return_universe'])}",
                f"- excess_confirmed: {item['benchmark']['excess_confirmed']}",
                f"- forward_bucket_used: top{item['forward']['forward_bucket']}",
                f"- forward_matured_count_max: {item['forward']['matured_count_max']}",
                f"- latest_forward_maturity_state: {item['forward']['latest_maturity_state'] or 'NA'}",
                f"- confidence_reliable: {item['confidence']['reliable']}",
                f"- confidence_stable_bucket_count_5d: {item['confidence']['stable_bucket_count_5d']}",
                f"- confidence_monotonic_hit_rate_5d: {item['confidence']['monotonic_hit_rate_5d']}",
                f"- avg_operational_minus_provisional_5d: {_fmt_num(item['confidence']['avg_operational_minus_provisional_5d'])}",
                f"- confidence_v2_trusted_ratio_top20: {_fmt_pct(item['confidence_v2']['trusted_ratio_top20'])}",
                f"- confidence_v2_provisional_or_better_ratio_top20: {_fmt_pct(item['confidence_v2']['provisional_or_better_ratio_top20'])}",
                f"- quality_delta_vs_raw_final_score: {_fmt_num(item['comparison']['avg_final_score_delta_vs_raw'])}",
                f"- quality_delta_vs_raw_confidence: {_fmt_num(item['comparison']['avg_confidence_delta_vs_raw'])}",
                f"- quality_delta_vs_raw_ret_score: {_fmt_num(item['comparison']['avg_ret_score_delta_vs_raw'])}",
                f"- quality_delta_vs_raw_tech_score: {_fmt_num(item['comparison']['avg_tech_score_delta_vs_raw'])}",
                f"- overheat_soft_ratio: {_fmt_pct(item['comparison']['overheat_soft_ratio'])}",
                f"- top20_buy_now_ratio: {_fmt_pct(item['buyability']['buy_now_ratio'])}",
                f"- top20_blocked_ratio: {_fmt_pct(item['buyability']['blocked_ratio'])}",
                f"- walkforward_acceptance_status: {item['walkforward_acceptance']['status'] or 'NA'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- `BUY_ALLOWED`: excess return is confirmed against benchmark baselines, calibration is reliable, and liquidity/concentration risk is contained.",
            "- `WATCH`: some evidence exists, but calibration or benchmark confirmation is not strong enough for live buy authorization.",
            "- `HOLD`: matured operational evidence is insufficient, so buying is deferred even if ranking quality looks good.",
            "- `BLOCK`: risk diagnostics are severe enough that buying should be stopped regardless of model score.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    global args_global
    args = parse_args()
    args_global = args

    candidates = {
        5: load_candidates(args.buy_top5_csv, 5),
        8: load_candidates(args.buy_top8_csv, 8),
        10: load_candidates(args.buy_top10_csv, 10),
    }
    benchmark_df = safe_read_csv(args.benchmark_csv)
    forward_df = safe_read_csv(args.forward_csv)
    conf_df = safe_read_csv(args.confidence_calibration_csv)
    compare_df = safe_read_csv(args.buy_comparison_csv)
    daily_status = safe_read_json(args.daily_status_json)
    confidence_v2_payload = safe_read_json(args.confidence_v2_json)
    buyability_payload = safe_read_json(args.top20_buyability_json)
    walkforward_acceptance_payload = safe_read_json(args.walkforward_acceptance_json)
    market_status_validation_payload = safe_read_json(args.market_status_validation_json)
    theme_churn_status, theme_churn_line = parse_theme_churn_gate(args.theme_acceptance_md)

    asof_candidates = [df["asof_date"].astype(str).iloc[0] for df in candidates.values() if not df.empty and "asof_date" in df.columns]
    asof_date = max(asof_candidates) if asof_candidates else "NA"

    decisions = [
        bucket_decision(
            bucket=bucket,
            candidate_df=candidates[bucket],
            benchmark_df=benchmark_df,
            forward_df=forward_df,
            conf_df=conf_df,
            compare_df=compare_df,
            confidence_v2_payload=confidence_v2_payload,
            buyability_payload=buyability_payload,
            walkforward_acceptance_payload=walkforward_acceptance_payload,
            market_regime_payload=market_status_validation_payload,
            theme_churn_status=theme_churn_status,
            daily_status=daily_status,
            args=args,
        )
        for bucket in [5, 8, 10]
    ]

    primary = next(item for item in decisions if item["bucket"] == args.primary_bucket)
    overall_status = primary["status"]
    interpretation = build_interpretation(primary, overall_status, daily_status)
    output_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof_date": asof_date,
        "gate_version": GATE_VERSION,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "primary_bucket": args.primary_bucket,
        "overall_status": overall_status,
        "theme_churn_status": theme_churn_status,
        "daily_cycle_status": daily_status.get("overall_status"),
        "market_regime": market_regime_diagnostics(market_status_validation_payload),
        "interpretation": interpretation,
        "decisions": decisions,
    }

    out_md = _resolve(args.out_md)
    out_json = _resolve(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        build_markdown(
            asof_date=asof_date,
            primary_bucket=args.primary_bucket,
            overall_status=overall_status,
            decisions=decisions,
            theme_churn_status=theme_churn_status,
            theme_churn_line=theme_churn_line,
            daily_status=daily_status,
            market_regime_payload=market_status_validation_payload,
            interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    out_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_json_payload(
        "operational_buy_gate",
        output_payload,
        asof_date=output_payload.get("asof_date"),
        generated_at=output_payload.get("generated_at"),
        source_path=args.out_json,
    )

    print(f"overall_status: {overall_status}")
    print(f"out_md: {out_md}")
    print(f"out_json: {out_json}")
    return 0


args_global: argparse.Namespace


if __name__ == "__main__":
    raise SystemExit(main())
