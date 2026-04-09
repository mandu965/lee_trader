from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apply_execution_policy import compute_target_weights
from payload_store import upsert_json_payload
from production_config import get_production_config_value, get_production_versions


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"

INPUT_CANDIDATES = DATA_DIR / "buy_candidates_top5.csv"
INPUT_GATE = OUTPUT_DIR / "operational_buy_gate.json"
INPUT_MANIFEST = OUTPUT_DIR / "production_v1_manifest.json"
INPUT_BENCHMARK = OUTPUT_DIR / "benchmark_comparison.csv"
INPUT_WEEKLY_REVIEW = OUTPUT_DIR / "operational_weekly_review.csv"
INPUT_SCORE_KPI = DATA_DIR / "score_kpi_monitor.json"
INPUT_CONFIDENCE_MAP = DATA_DIR / "confidence_calibration_map.json"
INPUT_CONFIDENCE_V2 = DATA_DIR / "confidence_score_v2.json"
INPUT_TOP20_BUYABILITY = OUTPUT_DIR / "top20_buyability_report.json"
INPUT_WALKFORWARD_ACCEPTANCE = OUTPUT_DIR / "walkforward_acceptance.json"
INPUT_PORTFOLIO_TOP5 = DATA_DIR / "model_portfolio_top5.csv"
INPUT_NAV = DATA_DIR / "paper_trading_nav.csv"
INPUT_SNAPSHOT_ARCHIVE = DATA_DIR / "ranking_snapshot_archive.csv"
INPUT_RANKING = DATA_DIR / "ranking_final.csv"
INPUT_LIVE_HOLDINGS = DATA_DIR / "live_account_holdings.csv"
INPUT_LIVE_SUMMARY = OUTPUT_DIR / "live_account_balance_summary.json"
INPUT_LIVE_ORDER_PREVIEW = OUTPUT_DIR / "live_order_preview.json"

OUT_DAILY = SERVING_DIR / "daily_recommendations.json"
OUT_GATE = SERVING_DIR / "buy_gate_status.json"
OUT_PORTFOLIO = SERVING_DIR / "model_portfolio.json"
OUT_PERFORMANCE = SERVING_DIR / "performance_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export operational outputs into serving-ready JSON payloads.")
    parser.add_argument("--candidates-csv", type=Path, default=INPUT_CANDIDATES)
    parser.add_argument("--gate-json", type=Path, default=INPUT_GATE)
    parser.add_argument("--manifest-json", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--benchmark-csv", type=Path, default=INPUT_BENCHMARK)
    parser.add_argument("--weekly-review-csv", type=Path, default=INPUT_WEEKLY_REVIEW)
    parser.add_argument("--score-kpi-json", type=Path, default=INPUT_SCORE_KPI)
    parser.add_argument("--confidence-map-json", type=Path, default=INPUT_CONFIDENCE_MAP)
    parser.add_argument("--confidence-v2-json", type=Path, default=INPUT_CONFIDENCE_V2)
    parser.add_argument("--top20-buyability-json", type=Path, default=INPUT_TOP20_BUYABILITY)
    parser.add_argument("--walkforward-acceptance-json", type=Path, default=INPUT_WALKFORWARD_ACCEPTANCE)
    parser.add_argument("--portfolio-top5-csv", type=Path, default=INPUT_PORTFOLIO_TOP5)
    parser.add_argument("--paper-nav-csv", type=Path, default=INPUT_NAV)
    parser.add_argument("--snapshot-archive-csv", type=Path, default=INPUT_SNAPSHOT_ARCHIVE)
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--live-holdings-csv", type=Path, default=INPUT_LIVE_HOLDINGS)
    parser.add_argument("--live-summary-json", type=Path, default=INPUT_LIVE_SUMMARY)
    parser.add_argument("--live-order-preview-json", type=Path, default=INPUT_LIVE_ORDER_PREVIEW)
    parser.add_argument("--out-daily", type=Path, default=OUT_DAILY)
    parser.add_argument("--out-gate", type=Path, default=OUT_GATE)
    parser.add_argument("--out-portfolio", type=Path, default=OUT_PORTFOLIO)
    parser.add_argument("--out-performance", type=Path, default=OUT_PERFORMANCE)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    read_kwargs = {"low_memory": False, "encoding": "utf-8-sig"}
    read_kwargs.update(kwargs)
    return pd.read_csv(resolved, **read_kwargs)


def read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def normalize_theme(series: pd.Series) -> pd.Series:
    return series.fillna("(none)").astype(str).replace({"": "(none)", "nan": "(none)"})


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["dominant_theme"] = normalize_theme(work.get("dominant_theme", pd.Series("(none)", index=work.index)))
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["name"] = work.get("name", "").fillna("").astype(str)
    for col in [
        "buy_rank",
        "rank_source",
        "final_score",
        "confidence_score",
        "liquidity_score",
        "theme_score",
        "trading_value",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
    ]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["recent_surge_soft_flag"] = work.get("recent_surge_soft_flag", False).astype(str).str.lower().isin(["true", "1"])
    work["selection_stage"] = work.get("selection_stage", "").fillna("").astype(str)
    work["selection_notes"] = work.get("selection_notes", "").fillna("").astype(str)
    work["explain_text"] = work.get("explain_text", "").fillna("").astype(str)
    work["asof_date"] = work.get("asof_date", "").fillna("").astype(str)
    return work.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)


def load_latest_snapshot(path: Path) -> pd.DataFrame:
    df = read_csv(path, dtype={"code": str})
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.normalize()
    latest_date = df["asof_date"].dropna().max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    latest = df.loc[df["asof_date"].eq(latest_date)].copy()
    latest["rank"] = pd.to_numeric(latest.get("rank"), errors="coerce")
    latest["final_score"] = pd.to_numeric(latest.get("final_score"), errors="coerce")
    latest["confidence_score"] = pd.to_numeric(latest.get("confidence_score"), errors="coerce")
    latest["dominant_theme"] = normalize_theme(latest.get("dominant_theme", pd.Series("(none)", index=latest.index)))
    return latest.sort_values(["rank", "code"]).reset_index(drop=True)


def load_latest_ranking_details(path: Path) -> pd.DataFrame:
    df = read_csv(path, dtype={"code": str})
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    latest_date = df["date"].dropna().max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    latest = df.loc[df["date"].eq(latest_date)].copy()
    for col in [
        "pred_return_60d",
        "prob_top20_60d",
        "pred_mdd_60d",
        "confidence_score",
        "final_score",
        "ret_score",
        "prob_score",
    ]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce")
    latest["regime"] = latest.get("regime", "").fillna("").astype(str)
    latest["regime_reason"] = latest.get("regime_reason", "").fillna("").astype(str)
    return latest.sort_values(["final_score", "code"], ascending=[False, True]).reset_index(drop=True)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scaled_score(value: float | None, low: float, high: float) -> float | None:
    if value is None or pd.isna(value):
        return None
    if high <= low:
        return None
    return _clip01((float(value) - low) / (high - low)) * 100.0


def _scaled_reverse_score(value: float | None, good_max: float, bad_min: float) -> float | None:
    if value is None or pd.isna(value):
        return None
    if bad_min <= good_max:
        return None
    return _clip01((bad_min - float(value)) / (bad_min - good_max)) * 100.0


def compute_buy_eligibility(detail: dict[str, Any], gate_payload: dict[str, Any]) -> dict[str, Any]:
    pred_return_60d = pd.to_numeric(detail.get("pred_return_60d"), errors="coerce")
    prob_top20_60d = pd.to_numeric(detail.get("prob_top20_60d"), errors="coerce")
    pred_mdd_60d = pd.to_numeric(detail.get("pred_mdd_60d"), errors="coerce")
    confidence_score = pd.to_numeric(detail.get("confidence_score"), errors="coerce")
    regime = str(detail.get("regime") or "").lower()
    gate_overall_status = str(gate_payload.get("overall_status") or "").upper()

    pred_return_score = _scaled_score(pred_return_60d, 0.04, 0.18)
    probability_score = _scaled_score(prob_top20_60d, 0.10, 0.30)
    mdd_score = _scaled_reverse_score(abs(pred_mdd_60d) if pd.notna(pred_mdd_60d) else None, 0.12, 0.30)
    confidence_gate_score = _scaled_score(confidence_score, 55.0, 85.0)

    regime_multiplier = {
        "bull": 1.00,
        "neutral": 0.85,
        "defensive": 0.60,
    }.get(regime, 0.75)

    component_pairs = [
        ("pred_return_60d", pred_return_score, 0.35),
        ("prob_top20_60d", probability_score, 0.30),
        ("pred_mdd_60d", mdd_score, 0.15),
        ("confidence_score", confidence_gate_score, 0.20),
    ]
    valid_pairs = [(name, score, weight) for name, score, weight in component_pairs if score is not None]
    if valid_pairs:
        total_weight = sum(weight for _, _, weight in valid_pairs)
        base_score = sum(score * weight for _, score, weight in valid_pairs) / total_weight
        buy_eligibility_score = round(base_score * regime_multiplier, 2)
    else:
        buy_eligibility_score = None

    hard_block_reasons: list[str] = []
    caution_reasons: list[str] = []

    if pd.notna(confidence_score) and confidence_score < 55.0:
        hard_block_reasons.append("confidence_score below 55")
    elif pd.notna(confidence_score) and confidence_score < 70.0:
        caution_reasons.append("confidence_score below preferred 70")

    if pd.notna(pred_return_60d) and pred_return_60d < 0.04:
        hard_block_reasons.append("pred_return_60d below 4%")
    elif pd.notna(pred_return_60d) and pred_return_60d < 0.08:
        caution_reasons.append("pred_return_60d below preferred 8%")

    if pd.notna(prob_top20_60d) and prob_top20_60d < 0.10:
        hard_block_reasons.append("prob_top20_60d below 10%")
    elif pd.notna(prob_top20_60d) and prob_top20_60d < 0.18:
        caution_reasons.append("prob_top20_60d below preferred 18%")

    if pd.notna(pred_mdd_60d) and pred_mdd_60d <= -0.30:
        hard_block_reasons.append("pred_mdd_60d worse than -30%")
    elif pd.notna(pred_mdd_60d) and pred_mdd_60d <= -0.20:
        caution_reasons.append("pred_mdd_60d worse than preferred -20%")

    if regime == "defensive":
        caution_reasons.append("market regime defensive")
    elif regime == "neutral":
        caution_reasons.append("market regime neutral")

    if gate_overall_status in {"HOLD", "BLOCK"}:
        caution_reasons.append(f"portfolio gate {gate_overall_status.lower()}")

    if hard_block_reasons:
        status = "BLOCK"
    elif buy_eligibility_score is not None and buy_eligibility_score >= 70.0 and gate_overall_status == "BUY_ALLOWED":
        status = "BUY_ALLOWED"
    elif buy_eligibility_score is not None and buy_eligibility_score >= 55.0:
        status = "WATCH"
    else:
        status = "BLOCK"

    summary_parts = []
    if pd.notna(pred_return_60d):
        summary_parts.append(f"return {float(pred_return_60d) * 100:.1f}%")
    if pd.notna(prob_top20_60d):
        summary_parts.append(f"prob {float(prob_top20_60d) * 100:.1f}%")
    if pd.notna(pred_mdd_60d):
        summary_parts.append(f"mdd {float(pred_mdd_60d) * 100:.1f}%")
    if pd.notna(confidence_score):
        summary_parts.append(f"confidence {float(confidence_score):.1f}")
    if regime:
        summary_parts.append(f"regime {regime}")

    return {
        "status": status,
        "score": buy_eligibility_score,
        "regime_multiplier": regime_multiplier,
        "hard_block_reasons": hard_block_reasons,
        "caution_reasons": caution_reasons,
        "component_scores": {
            "pred_return_60d": pred_return_score,
            "prob_top20_60d": probability_score,
            "pred_mdd_60d": mdd_score,
            "confidence_score": confidence_gate_score,
        },
        "summary": ", ".join(summary_parts) if summary_parts else None,
    }


def build_explanation(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {"summary_text": None, "highlights": []}
    parts = [part.strip() for part in cleaned.split(". ") if part.strip()]
    summary = parts[0]
    highlights = []
    for part in parts[:4]:
        sentence = part if part.endswith(".") else f"{part}."
        highlights.append(sentence)
    return {"summary_text": summary, "highlights": highlights}


def daily_payload(
    candidates: pd.DataFrame,
    gate_payload: dict[str, Any],
    manifest: dict[str, Any],
    latest_snapshot: pd.DataFrame,
    latest_ranking_details: pd.DataFrame,
    confidence_v2_payload: dict[str, Any],
    buyability_payload: dict[str, Any],
    walkforward_acceptance_payload: dict[str, Any],
) -> dict[str, Any]:
    candidate_asof = str(candidates["asof_date"].iloc[0]) if not candidates.empty else None
    latest_snapshot_date = latest_snapshot["asof_date"].iloc[0].strftime("%Y-%m-%d") if not latest_snapshot.empty else None
    is_stale = bool(candidate_asof and latest_snapshot_date and candidate_asof < latest_snapshot_date)
    latest_lookup = latest_snapshot.set_index("code") if not latest_snapshot.empty else pd.DataFrame()
    latest_detail_lookup = latest_ranking_details.set_index("code") if not latest_ranking_details.empty else pd.DataFrame()
    confidence_lookup = {
        str(item.get("code", "")).zfill(6): item
        for item in confidence_v2_payload.get("items", [])
        if isinstance(item, dict)
    }
    buyability_lookup = {
        str(item.get("code", "")).zfill(6): item
        for item in buyability_payload.get("items", [])
        if isinstance(item, dict)
    }

    items: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        latest_rank = latest_lookup.at[code, "rank"] if not latest_lookup.empty and code in latest_lookup.index else None
        latest_final_score = latest_lookup.at[code, "final_score"] if not latest_lookup.empty and code in latest_lookup.index else None
        latest_confidence = latest_lookup.at[code, "confidence_score"] if not latest_lookup.empty and code in latest_lookup.index else None
        detail_row = latest_detail_lookup.loc[code].to_dict() if not latest_detail_lookup.empty and code in latest_detail_lookup.index else {}
        confidence_v2 = confidence_lookup.get(code, {})
        buyability = buyability_lookup.get(code, {})
        buy_eligibility = compute_buy_eligibility(detail_row, gate_payload)
        item = {
            "recommendation_id": f"{candidate_asof or 'na'}:{int(row['target_bucket'])}:{int(row['buy_rank'])}:{code}",
            "asof_date": candidate_asof,
            "target_bucket": int(row["target_bucket"]),
            "buy_rank": int(row["buy_rank"]),
            "rank_source": int(row["rank_source"]) if pd.notna(row["rank_source"]) else None,
            "security": {
                "code": code,
                "name": row["name"],
                "market": row.get("market"),
                "sector": row["sector"],
                "dominant_theme": row["dominant_theme"],
            },
            "scores": {
                "final_score": row["final_score"],
                "confidence_score": row["confidence_score"],
                "liquidity_score": row["liquidity_score"],
                "theme_score": row["theme_score"],
                "raw_confidence_v2": pd.to_numeric(confidence_v2.get("raw_confidence_v2"), errors="coerce"),
                "alpha_confidence": pd.to_numeric(confidence_v2.get("alpha_confidence"), errors="coerce"),
                "execution_confidence": pd.to_numeric(confidence_v2.get("execution_confidence"), errors="coerce"),
                "stability_confidence": pd.to_numeric(confidence_v2.get("stability_confidence"), errors="coerce"),
                "calibration_confidence": pd.to_numeric(confidence_v2.get("calibration_confidence"), errors="coerce"),
                "confidence_state_v2": confidence_v2.get("confidence_state_v2"),
                "latest_snapshot_final_score": latest_final_score,
                "latest_snapshot_confidence_score": latest_confidence,
                "score_drift_vs_latest_snapshot": (latest_final_score - row["final_score"]) if latest_final_score is not None and pd.notna(row["final_score"]) else None,
            },
            "market_signals": {
                "trading_value": row["trading_value"],
                "ret_5d": row["ret_5d"],
                "ret_10d": row["ret_10d"],
                "mom_20": row["mom_20"],
                "rsi_14": row["rsi_14"],
                "pred_return_60d": pd.to_numeric(detail_row.get("pred_return_60d"), errors="coerce"),
                "prob_top20_60d": pd.to_numeric(detail_row.get("prob_top20_60d"), errors="coerce"),
                "pred_mdd_60d": pd.to_numeric(detail_row.get("pred_mdd_60d"), errors="coerce"),
                "regime": detail_row.get("regime") or None,
                "regime_reason": detail_row.get("regime_reason") or None,
            },
            "buy_eligibility": buy_eligibility,
            "selection": {
                "selection_stage": row["selection_stage"],
                "selection_notes": row["selection_notes"],
                "recent_surge_soft_flag": bool(row["recent_surge_soft_flag"]),
                "entry_rule_pass": bool(
                    pd.notna(row["confidence_score"])
                    and row["confidence_score"] >= float(get_production_config_value(["buy_candidate", "min_confidence"], 80.0))
                    and pd.notna(row["liquidity_score"])
                    and row["liquidity_score"] >= float(get_production_config_value(["buy_candidate", "min_liquidity_score"], 15.0))
                    and pd.notna(row["trading_value"])
                    and row["trading_value"] >= float(get_production_config_value(["buy_candidate", "min_trading_value"], 5_000_000_000.0))
                ),
                "latest_snapshot_rank": latest_rank,
                "buyability_status": buyability.get("buyability_status"),
                "buyability_watchlist_tier": buyability.get("watchlist_tier"),
                "buyability_promotion_readiness_score": pd.to_numeric(buyability.get("promotion_readiness_score"), errors="coerce"),
                "buyability_expected_action": buyability.get("expected_action"),
                "buyability_supporting_reasons": buyability.get("supporting_reasons") or [],
                "buyability_blocking_reasons": buyability.get("blocking_reasons") or [],
            },
            "score_explanations": build_explanation(row["explain_text"]),
        }
        items.append(item)

    return {
        "entity": "daily_recommendations",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": candidate_asof,
        "source_status": "stale" if is_stale else "current",
        "source_detail": {
            "candidates_path": str(INPUT_CANDIDATES.relative_to(ROOT)),
            "latest_snapshot_date": latest_snapshot_date,
        },
        "versions": manifest.get("versions") or get_production_versions(),
        "gate_overall_status": gate_payload.get("overall_status"),
        "walkforward_acceptance_status": walkforward_acceptance_payload.get("status"),
        "count": len(items),
        "items": items,
    }


def gate_status_payload(gate_payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    decisions = []
    for decision in gate_payload.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        decisions.append(
            {
                "bucket": decision.get("bucket"),
                "status": decision.get("status"),
                "reason_summary": decision.get("reason_summary"),
                "candidate_diagnostics": decision.get("static"),
                "benchmark_diagnostics": decision.get("benchmark"),
                "forward_diagnostics": decision.get("forward"),
                "confidence_diagnostics": decision.get("confidence"),
                "comparison_diagnostics": decision.get("comparison"),
                "confidence_v2_diagnostics": decision.get("confidence_v2"),
                "buyability_diagnostics": decision.get("buyability"),
                "walkforward_acceptance_diagnostics": decision.get("walkforward_acceptance"),
                "market_regime_diagnostics": decision.get("market_regime"),
            }
        )
    return {
        "entity": "buy_gate_status",
        "generated_at": gate_payload.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": gate_payload.get("asof_date"),
        "versions": manifest.get("versions") or get_production_versions(),
        "primary_bucket": gate_payload.get("primary_bucket"),
        "overall_status": gate_payload.get("overall_status"),
        "theme_churn_status": gate_payload.get("theme_churn_status"),
        "daily_cycle_status": gate_payload.get("daily_cycle_status"),
        "market_regime": gate_payload.get("market_regime"),
        "decisions": decisions,
    }


def model_portfolio_payload(candidates: pd.DataFrame, manifest: dict[str, Any], input_path: Path) -> dict[str, Any]:
    weights_source_status = "derived_preview"
    source_path = str(input_path.relative_to(ROOT))
    portfolio_df = read_csv(input_path, dtype={"code": str})

    if portfolio_df.empty:
        class Args:
            cash_minimum = float(get_production_config_value(["portfolio", "cash_buffer"], 0.05))
            max_position_weight = float(get_production_config_value(["portfolio", "max_weight_top5"], 0.24))
            sector_cap = float(get_production_config_value(["portfolio", "sector_cap"], 0.35))
            theme_cap = float(get_production_config_value(["portfolio", "theme_cap"], 0.35))
            confidence_block_below = float(get_production_config_value(["execution_policy", "confidence_block_below"], 55.0))
            confidence_reduced_below = float(get_production_config_value(["execution_policy", "confidence_reduced_below"], 70.0))
            confidence_standard_below = float(get_production_config_value(["execution_policy", "confidence_standard_below"], 85.0))
            confidence_reduced_weight_scale = float(get_production_config_value(["execution_policy", "confidence_reduced_weight_scale"], 0.45))
            confidence_standard_weight_scale = float(get_production_config_value(["execution_policy", "confidence_standard_weight_scale"], 1.00))
            confidence_expanded_weight_scale = float(get_production_config_value(["execution_policy", "confidence_expanded_weight_scale"], 1.15))
            confidence_reduced_position_cap_scale = float(get_production_config_value(["execution_policy", "confidence_reduced_position_cap_scale"], 0.50))
            confidence_standard_position_cap_scale = float(get_production_config_value(["execution_policy", "confidence_standard_position_cap_scale"], 1.00))
            confidence_expanded_position_cap_scale = float(get_production_config_value(["execution_policy", "confidence_expanded_position_cap_scale"], 1.15))

        weighted = compute_target_weights(candidates.copy(), Args())
        portfolio_df = weighted.copy()
    else:
        weights_source_status = "model_portfolio_top5"
        source_path = str(_resolve(input_path).relative_to(ROOT))

    portfolio_df["code"] = portfolio_df["code"].astype(str).str.zfill(6)
    if "target_weight" not in portfolio_df.columns:
        if "weight" in portfolio_df.columns:
            portfolio_df["target_weight"] = pd.to_numeric(portfolio_df["weight"], errors="coerce")
        else:
            portfolio_df["target_weight"] = pd.to_numeric(portfolio_df.get("suggested_weight"), errors="coerce")

    holdings = []
    for _, row in portfolio_df.iterrows():
        holdings.append(
            {
                "code": row["code"],
                "name": row.get("name"),
                "target_weight": pd.to_numeric(row.get("target_weight"), errors="coerce"),
                "buy_rank": pd.to_numeric(row.get("buy_rank"), errors="coerce"),
                "sector": row.get("sector"),
                "dominant_theme": row.get("dominant_theme"),
                "final_score": pd.to_numeric(row.get("final_score"), errors="coerce"),
                "confidence_score": pd.to_numeric(row.get("confidence_score"), errors="coerce"),
                "liquidity_score": pd.to_numeric(row.get("liquidity_score"), errors="coerce"),
                "selection_stage": row.get("selection_stage"),
            }
        )

    return {
        "entity": "model_portfolio",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": str(candidates["asof_date"].iloc[0]) if not candidates.empty else None,
        "versions": manifest.get("versions") or get_production_versions(),
        "source_status": weights_source_status,
        "source_path": source_path,
        "constraints": {
            "target_bucket": 5,
            "cash_buffer": float(get_production_config_value(["portfolio", "cash_buffer"], 0.05)),
            "max_position_weight": float(get_production_config_value(["portfolio", "max_weight_top5"], 0.24)),
            "sector_cap": float(get_production_config_value(["portfolio", "sector_cap"], 0.35)),
            "theme_cap": float(get_production_config_value(["portfolio", "theme_cap"], 0.35)),
            "no_theme_cap": float(get_production_config_value(["portfolio", "no_theme_cap"], 0.60)),
        },
        "cash_target": float(get_production_config_value(["portfolio", "cash_buffer"], 0.05)),
        "holding_count": len(holdings),
        "holdings": holdings,
    }


def performance_payload(
    manifest: dict[str, Any],
    benchmark_csv: Path,
    weekly_review_csv: Path,
    score_kpi_json: Path,
    confidence_map_json: Path,
    confidence_v2_json: Path,
    top20_buyability_json: Path,
    walkforward_acceptance_json: Path,
    paper_nav_csv: Path,
    live_holdings_csv: Path,
    live_summary_json: Path,
    live_order_preview_json: Path,
) -> dict[str, Any]:
    benchmark_df = read_csv(benchmark_csv)
    weekly_df = read_csv(weekly_review_csv)
    score_kpi = read_json(score_kpi_json)
    confidence_map = read_json(confidence_map_json)
    confidence_v2 = read_json(confidence_v2_json)
    top20_buyability = read_json(top20_buyability_json)
    walkforward_acceptance = read_json(walkforward_acceptance_json)
    nav_df = read_csv(paper_nav_csv)
    live_holdings_df = read_csv(live_holdings_csv, dtype={"code": str})
    live_summary = read_json(live_summary_json)
    live_order_preview = read_json(live_order_preview_json)

    benchmark_items = []
    if not benchmark_df.empty:
        work = benchmark_df.copy()
        work["top_n"] = pd.to_numeric(work["top_n"], errors="coerce")
        work["horizon_days"] = pd.to_numeric(work["horizon_days"], errors="coerce")
        work["dates_matured"] = pd.to_numeric(work["dates_matured"], errors="coerce")
        work["avg_excess_return"] = pd.to_numeric(work.get("avg_excess_return"), errors="coerce")
        subset = work.loc[work["top_n"].eq(5)].sort_values(["horizon_days", "benchmark_name"])
        for _, row in subset.iterrows():
            benchmark_items.append(
                {
                    "top_n": int(row["top_n"]),
                    "horizon_days": int(row["horizon_days"]),
                    "benchmark_name": row["benchmark_name"],
                    "dates_total": pd.to_numeric(row.get("dates_total"), errors="coerce"),
                    "dates_matured": pd.to_numeric(row.get("dates_matured"), errors="coerce"),
                    "avg_portfolio_return": pd.to_numeric(row.get("avg_portfolio_return"), errors="coerce"),
                    "avg_benchmark_return": pd.to_numeric(row.get("avg_benchmark_return"), errors="coerce"),
                    "avg_excess_return": pd.to_numeric(row.get("avg_excess_return"), errors="coerce"),
                    "excess_hit_ratio": pd.to_numeric(row.get("excess_hit_ratio"), errors="coerce"),
                    "benchmark_available": row.get("benchmark_available"),
                }
            )

    weekly_latest: dict[str, Any] | None = None
    if not weekly_df.empty:
        review = weekly_df.copy()
        weekly_latest = review.iloc[-1].to_dict()

    paper_trading: dict[str, Any]
    if nav_df.empty:
        paper_trading = {
            "available": False,
            "latest_date": None,
            "latest_nav": None,
            "cumulative_return": None,
            "drawdown": None,
        }
    else:
        nav = nav_df.copy()
        latest = nav.iloc[-1]
        paper_trading = {
            "available": True,
            "latest_date": latest.get("date"),
            "latest_nav": pd.to_numeric(latest.get("nav"), errors="coerce"),
            "cumulative_return": pd.to_numeric(latest.get("cumulative_return"), errors="coerce"),
            "drawdown": pd.to_numeric(latest.get("drawdown"), errors="coerce"),
        }

    live_account: dict[str, Any]
    if live_holdings_df.empty and not live_summary:
        live_account = {
            "available": False,
            "env_dv": None,
            "holding_count": 0,
            "cash_summary": None,
            "order_preview_count": 0,
        }
    else:
        live_account = {
            "available": True,
            "generated_at": live_summary.get("generated_at"),
            "env_dv": live_summary.get("env_dv"),
            "holding_count": int(live_summary.get("holding_count") or len(live_holdings_df)),
            "cash_summary": live_summary.get("cash_summary"),
            "holdings_path": str(_resolve(live_holdings_csv).relative_to(ROOT)),
            "summary_path": str(_resolve(live_summary_json).relative_to(ROOT)),
            "order_preview_path": str(_resolve(live_order_preview_json).relative_to(ROOT))
            if _resolve(live_order_preview_json).exists()
            else None,
            "order_preview_count": len(live_order_preview.get("items") or []),
            "gate_status_for_preview": live_order_preview.get("gate_status"),
        }

    return {
        "entity": "performance_summary",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versions": manifest.get("versions") or get_production_versions(),
        "paper_trading": paper_trading,
        "live_account": live_account,
        "benchmark_summary": {
            "source_path": str(_resolve(benchmark_csv).relative_to(ROOT)),
            "items": benchmark_items,
        },
        "weekly_review": weekly_latest,
        "score_kpi_monitor": {
            "summary": score_kpi.get("summary"),
            "top_metrics": (score_kpi.get("kpis") or [])[:12],
        },
        "confidence_calibration": {
            "summary": confidence_map.get("summary"),
            "bucket_map": confidence_map.get("bucket_map"),
        },
        "confidence_score_v2": confidence_v2.get("summary"),
        "top20_buyability": top20_buyability.get("summary"),
        "walkforward_acceptance": walkforward_acceptance,
    }


def main() -> None:
    args = parse_args()
    candidates = normalize_candidates(read_csv(args.candidates_csv, dtype={"code": str}))
    gate_payload = read_json(args.gate_json)
    manifest = read_json(args.manifest_json)
    latest_snapshot = load_latest_snapshot(args.snapshot_archive_csv)
    latest_ranking_details = load_latest_ranking_details(args.ranking_csv)
    confidence_v2_payload = read_json(args.confidence_v2_json)
    top20_buyability_payload = read_json(args.top20_buyability_json)
    walkforward_acceptance_payload = read_json(args.walkforward_acceptance_json)

    daily = daily_payload(
        candidates,
        gate_payload,
        manifest,
        latest_snapshot,
        latest_ranking_details,
        confidence_v2_payload,
        top20_buyability_payload,
        walkforward_acceptance_payload,
    )
    gate = gate_status_payload(gate_payload, manifest)
    portfolio = model_portfolio_payload(candidates, manifest, args.portfolio_top5_csv)
    performance = performance_payload(
        manifest=manifest,
        benchmark_csv=args.benchmark_csv,
        weekly_review_csv=args.weekly_review_csv,
        score_kpi_json=args.score_kpi_json,
        confidence_map_json=args.confidence_map_json,
        confidence_v2_json=args.confidence_v2_json,
        top20_buyability_json=args.top20_buyability_json,
        walkforward_acceptance_json=args.walkforward_acceptance_json,
        paper_nav_csv=args.paper_nav_csv,
        live_holdings_csv=args.live_holdings_csv,
        live_summary_json=args.live_summary_json,
        live_order_preview_json=args.live_order_preview_json,
    )

    write_json(args.out_daily, daily)
    write_json(args.out_gate, gate)
    write_json(args.out_portfolio, portfolio)
    write_json(args.out_performance, performance)
    upsert_json_payload(
        "daily_recommendations",
        daily,
        asof_date=daily.get("asof_date"),
        generated_at=daily.get("generated_at"),
        source_path=args.out_daily,
    )
    upsert_json_payload(
        "buy_gate_status",
        gate,
        asof_date=gate.get("asof_date"),
        generated_at=gate.get("generated_at"),
        source_path=args.out_gate,
    )
    upsert_json_payload(
        "model_portfolio",
        portfolio,
        asof_date=portfolio.get("asof_date"),
        generated_at=portfolio.get("generated_at"),
        source_path=args.out_portfolio,
    )
    upsert_json_payload(
        "performance_summary",
        performance,
        asof_date=daily.get("asof_date"),
        generated_at=performance.get("generated_at"),
        source_path=args.out_performance,
    )


if __name__ == "__main__":
    main()
