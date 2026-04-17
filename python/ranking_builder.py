"""
Single source of truth for daily ranking score construction.

This file is responsible for the full ranking build pipeline:
1. Load and merge predictions, features, universe, technical scores, and market status.
2. Compute per-stock component scores.
3. Reflect market regime (bull / neutral / defensive) in the final weighted score.
4. Reflect drawdown-based risk penalty.
5. Produce final_score, final_score_v2, and rank outputs.

Final score formula summary
---------------------------
- Inputs:
  predictions.csv    -> model return / MDD / probability outputs
  scores_final.csv   -> technical score source (composite or score_score)
  features.csv       -> quality / volatility / liquidity inputs
  universe.csv       -> metadata such as name / sector / market
  market_status.csv  -> market regime inputs
- Component scores:
  tech_score, ret_score, pred_score, prob_score_raw, prob_score, qual_score,
  valuation_score, safety_score, liquidity_score, risk_penalty
- Prediction-score policy:
  ret_score  -> primary production prediction score used in final_score
  pred_score -> legacy / research-only comparison metric
- Probability-score policy:
  prob_score_raw -> raw 60d probability converted to a 0~100 absolute score
  prob_score     -> per-date relative operating score from prob_top20_60d
  prob_top20_90d -> stored / research-only auxiliary probability signal
- Operating-score policy:
  final_score directly uses ret_score, prob_score, tech_score, qual_score,
  and risk_penalty. valuation_score, safety_score, and liquidity_score are
  retained as compatibility / diagnostic columns and are not direct operating axes.
- Regime-aware weighted score:
  bull      -> return/probability/technical weights are relatively higher
  neutral   -> baseline balanced mix
  defensive -> quality weight is relatively higher
- Final output:
  final_score    -> regime-aware weighted score after soft risk penalty
  final_score_v2 -> legacy fixed-weight reference score after soft risk penalty
  rank_final     -> per-date rank by final_score
  rank_v2        -> per-date rank by final_score_v2 (comparison-only reference)
"""
import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from production_config import (
    allow_experimental_runtime_features,
    get_production_config_value,
    is_operational_runtime_mode,
)
from scoring.final_score import (
    apply_baseline_final_score as shared_apply_baseline_final_score,
    attach_market_columns as shared_attach_market_columns,
    attach_operational_score_aliases as shared_attach_operational_score_aliases,
    baseline_risk_penalty_from_mix,
    compute_component_scores as shared_compute_component_scores,
    compute_risk_penalty as shared_compute_risk_penalty,
    compute_score_explain as shared_compute_score_explain,
    detect_market_regime as shared_detect_market_regime,
    ensure_regime_column as shared_ensure_regime_column,
    resolve_core_weight_profile,
)
from score_explainer import attach_score_explain_columns

try:
    from db import ensure_unique_keys, get_engine, replace_table_rows_pg, replace_table_rows_sqlite, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    ensure_unique_keys = None
    replace_table_rows_pg = None
    replace_table_rows_sqlite = None
    def use_sqlite_fallback_writes() -> bool:
        return False

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
COMPARE_OUTPUT_DIR = Path("output")

PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
SCORES_CSV = DATA_DIR / "scores_final.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
STOCK_THEME_DAILY_CSV = COMPARE_OUTPUT_DIR / "stock_theme_daily.csv"

OUT_CSV = DATA_DIR / "ranking_final.csv"
QUALITY_GATE_SHADOW_CSV = DATA_DIR / "quality_gate_shadow.csv"
SCORE_BREAKDOWN_DEBUG_CSV = OUTPUT_DIR / "score_breakdown_debug.csv"
CONFIDENCE_DIAGNOSTICS_CSV = OUTPUT_DIR / "confidence_diagnostics_snapshot.csv"
THEME_IMPACT_COMPARE_CSV = OUTPUT_DIR / "theme_score_impact_compare.csv"
BEFORE_AFTER_SCORE_COMPARE_CSV = COMPARE_OUTPUT_DIR / "before_after_score_compare.csv"
TOP20_BEFORE_AFTER_COMPARE_CSV = COMPARE_OUTPUT_DIR / "top20_before_after_compare.csv"
BEFORE_AFTER_SCORE_COMPARE_V3_CSV = COMPARE_OUTPUT_DIR / "before_after_score_compare_v3.csv"
TOP20_BEFORE_AFTER_COMPARE_V3_CSV = COMPARE_OUTPUT_DIR / "top20_before_after_compare_v3.csv"
THEME_CONFIDENCE_OVERLAY_VALIDATION_MD = COMPARE_OUTPUT_DIR / "theme_confidence_overlay_validation.md"
RANKING_THEME_RISK_SOFT_CSV = DATA_DIR / "ranking_final_theme_risk_soft.csv"
THEME_RISK_SOFT_COMPARE_CSV = DATA_DIR / "theme_risk_soft_compare.csv"
THEME_RISK_SOFT_VALIDATION_MD = DATA_DIR / "theme_risk_soft_validation.md"
THEME_RISK_CURVE_COMPARE_CSV = DATA_DIR / "theme_risk_curve_compare.csv"
THEME_RISK_CURVE_VALIDATION_MD = DATA_DIR / "theme_risk_curve_validation.md"
THEME_RISK_CURVE_NEAR_TOP20_CSV = DATA_DIR / "theme_risk_curve_near_top20.csv"
FEATURE_CANDIDATE_EXP_B_CSV = DATA_DIR / "feature_candidate_exp_b.csv"
FEATURE_CANDIDATE_EXP_B_TOP20_DIFF_CSV = DATA_DIR / "feature_candidate_exp_b_top20_diff.csv"
FEATURE_CANDIDATE_EXP_B_NEAR_TOP20_CSV = DATA_DIR / "feature_candidate_exp_b_near_top20.csv"
FEATURE_CANDIDATE_EXP_B_SUMMARY_MD = DATA_DIR / "feature_candidate_exp_b_summary.md"
THEME_GUARD_REPORT_MD = DATA_DIR / "ranking_builder_theme_guard_report.md"
THEME_OVERLAY_GATE_DEBUG_JSON = DATA_DIR / "theme_overlay_gate_debug.json"
THEME_OVERLAY_GATE_DEBUG_MD = DATA_DIR / "theme_overlay_gate_debug.md"
THEME_OVERLAY_MODE_RESOLUTION_MD = DATA_DIR / "theme_overlay_mode_resolution.md"
DEBUG_THEME_TOP50_CSV = DATA_DIR / "debug_theme_top50.csv"
DEBUG_THEME_SUMMARY_TXT = DATA_DIR / "debug_theme_summary.txt"
THEME_OVERLAY_SHADOW_PREVIEW_CSV = DATA_DIR / "theme_overlay_shadow_preview.csv"
THEME_OVERLAY_SHADOW_SUMMARY_JSON = DATA_DIR / "theme_overlay_shadow_summary.json"
THEME_OVERLAY_SHADOW_MODE_UPDATE_MD = DATA_DIR / "theme_overlay_shadow_mode_update.md"
DB_PATH = DATA_DIR / "lee_trader.db"
DEFAULT_SCORE_FORMULA_VERSION = str(
    get_production_config_value(
        ["metadata", "score_formula_version"],
        "ranking_builder_v8_return_prob_tech_regime",
    )
)
QUALITY_GATE_FEATURE_CANDIDATE = True
QUALITY_GATE_FEATURE_ENABLED = False
QUALITY_GATE_ALLOWED_REGIMES = {"defensive"}
QUALITY_GATE_EXPERIMENT = "v2"
DEFAULT_CONFIDENCE_VERSION = str(
    get_production_config_value(
        ["metadata", "confidence_calibration_version"],
        "confidence_four_axis_v1",
    )
)
DEFAULT_THEME_FACTOR_VERSION = str(
    get_production_config_value(["ranking", "theme_factor_version"], "theme_factor_v1")
)
THEME_OVERLAY_OFF = "off"
THEME_OVERLAY_SHADOW = "shadow"
THEME_OVERLAY_OPERATIONAL = "operational"
THEME_OVERLAY_ALLOWED_MODES = {
    THEME_OVERLAY_OFF,
    THEME_OVERLAY_SHADOW,
    THEME_OVERLAY_OPERATIONAL,
}
THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT = 0.35
RISK_PENALTY_THEME_ONLY_SOFT_DEFAULT = False
RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT = 0.85
RISK_PENALTY_THEME_MIN_SCORE_DEFAULT = 70.0
RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT = 0.60
RISK_CURVE_EXPERIMENT_DEFAULT = False
EXP_A_THRESHOLD_DEFAULT = 0.25
EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT = 0.50
EXP_B_DELAYED_REACH_FACTOR_DEFAULT = 1.20
PENALTY_CAP_DEFAULT = 18.0
RISK_CURVE_FEATURE_CANDIDATE_DEFAULT = "none"
RISK_CURVE_FEATURE_CANDIDATE_ENABLED_DEFAULT = False
EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT = 1.20
EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT = 0.70
EXP_B_DELAYED_CAP_APPLY_REGIMES_DEFAULT = ""
EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT = True
EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT = 60.0
EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT = 0.80
LAST_THEME_GUARD_STATUS: dict[str, object] = {
    "overlay_enabled": False,
    "mode": THEME_OVERLAY_OFF,
    "operational": False,
    "applied": False,
    "disable_reason": "mode_mismatch",
    "coverage_ratio": 0.0,
    "coverage_threshold": THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT,
    "matched_rows": 0,
    "base_rows": 0,
    "theme_row_count": 0,
    "ranking_latest_date": "NA",
    "theme_latest_date": "NA",
    "available_theme_dates": [],
    "source": "none",
}
LAST_THEME_GATE_DEBUG: dict[str, object] = {
    "enable_theme_overlay_raw": "0",
    "theme_overlay_mode_requested": THEME_OVERLAY_OFF,
    "current_execution_mode": THEME_OVERLAY_OFF,
    "requested_execution_mode": THEME_OVERLAY_OFF,
    "resolved_execution_mode": THEME_OVERLAY_OFF,
    "fallback_applied": False,
    "fallback_reason": "(none)",
    "enable_theme_validation_raw": "0",
    "overlay_gate_result": "disabled",
    "overlay_disable_reason": "disabled_by_flag",
    "theme_weight_source_priority": [
        "best_weight_by_regime",
        "best_weight_global",
        "fallback_default",
    ],
    "theme_weight_config_paths": {
        "best_weight_by_regime": str(BEST_THEME_WEIGHT_BY_REGIME_JSON) if "BEST_THEME_WEIGHT_BY_REGIME_JSON" in globals() else "data/experiments/theme_weight/best_weight_by_regime.json",
        "best_weight_global": str(BEST_THEME_WEIGHT_JSON) if "BEST_THEME_WEIGHT_JSON" in globals() else "data/experiments/theme_weight/best_weight.json",
    },
}
THEME_RISK_SOFT_EXPERIMENT_COLUMNS = [
    "risk_penalty_base",
    "risk_penalty_effective",
    "risk_penalty_soft_delta",
    "theme_risk_soft_enabled",
    "theme_risk_soft_applied",
    "theme_risk_soft_reason",
    "final_score_baseline",
    "final_score_theme_risk_soft",
    "rank_baseline",
    "rank_theme_risk_soft",
    "rank_change_theme_risk_soft",
    "theme_risk_soft_explain_append",
]
THEME_RISK_CURVE_EXPERIMENT_COLUMNS = [
    "risk_penalty_base",
    "risk_penalty_exp_a",
    "risk_penalty_exp_b",
    "risk_penalty_delta_exp_a",
    "risk_penalty_delta_exp_b",
    "final_score_baseline",
    "final_score_exp_a",
    "final_score_exp_b",
    "rank_baseline",
    "rank_exp_a",
    "rank_exp_b",
    "rank_change_exp_a",
    "rank_change_exp_b",
    "explain_base",
    "explain_exp_a",
    "explain_exp_b",
]
FEATURE_CANDIDATE_COLUMNS = [
    "has_theme_flag",
    "candidate_feature_name",
    "candidate_enabled",
    "candidate_applied_flag",
    "candidate_reason",
    "candidate_apply_regimes",
    "candidate_baseline_final_score",
    "candidate_final_score",
    "candidate_score_delta",
    "candidate_baseline_rank",
    "candidate_rank",
    "candidate_rank_delta",
    "candidate_baseline_risk_penalty",
    "candidate_risk_penalty",
    "candidate_penalty_delta",
    "candidate_explain",
    "near_top20_band",
    "top20_status",
]
CORE_COMPONENT_COLUMNS = [
    "ret_score",
    "prob_score",
    "qual_score",
    "tech_score",
    "safety_score",
    "liquidity_score",
    "theme_score",
]
DAILY_RANKING_STORE_COLUMNS = [
    "date",
    "code",
    "close",
    "pred_return_60d",
    "pred_return_90d",
    "pred_mdd_60d",
    "pred_mdd_90d",
    "prob_top20_60d",
    "prob_top20_90d",
    "prob_score_raw",
    "prob_rank_pct",
    "score",
    "score_score",
    "composite",
    "quality_score",
    "quality_factor_count",
    "quality_missing_ratio",
    "quality_score_confidence",
    "vol_20",
    "vol_60",
    "vol_ma_20",
    "volume",
    "mom_20",
    "close_over_ma20",
    "rsi_14",
    "vol_ratio_20",
    "name",
    "market",
    "sector",
    "regime",
    "regime_reason",
    "weight_profile",
    "tech_source",
    "tech_trend_score",
    "tech_momentum_score",
    "tech_stability_score",
    "tech_volume_score",
    "tech_liquidity_guard",
    "tech_score",
    "pred_score_60",
    "ret_rank_60",
    "pred_return_60d_pct01",
    "pred_score_90",
    "ret_rank_90",
    "pred_return_90d_pct01",
    "pred_score",
    "ret_score",
    "return_score",
    "ret_score_v11",
    "prob_score",
    "probability_score",
    "qual_score",
    "quality_flag",
    "quality_gate_applied",
    "quality_penalty_ratio",
    "shadow_quality_gate_applied",
    "shadow_quality_penalty_ratio",
    "shadow_final_score_quality_gate",
    "shadow_rank_quality_gate",
    "shadow_quality_risk_guard_penalty",
    "shadow_quality_risk_guard_applied",
    "shadow_final_score_quality_risk_guard",
    "shadow_rank_quality_risk_guard",
    "quality_gate_experiment",
    "technical_score",
    "valuation_score",
    "ret_score_missing",
    "prob_score_missing",
    "qual_score_missing",
    "tech_score_missing",
    "safety_score_missing",
    "liquidity_score_missing",
    "ret_score_fallback_used",
    "prob_score_fallback_used",
    "qual_score_fallback_used",
    "tech_score_fallback_used",
    "safety_score_fallback_used",
    "liquidity_score_fallback_used",
    "fallback_count",
    "vol_20_pct",
    "vol_60_pct",
    "safety_score",
    "liquidity_score",
    "theme_score",
    "dominant_theme",
    "theme_confidence",
    "theme_score_effective",
    "raw_theme_score",
    "filtered_theme_score",
    "theme_threshold",
    "theme_applied_flag",
    "theme_debug_reason",
    "pred_mdd_mix",
    "final_score_raw",
    "final_score_before_theme",
    "final_score_v2_before_theme",
    "final_score",
    "final_score_v2",
    "final_score_v3",
    "live_score",
    "live_score_source",
    "score_diff_v2",
    "score_diff_v3",
    "v3_vs_v2_diff",
    "theme_overlay_mode",
    "theme_overlay_anchor",
    "theme_delta_raw",
    "theme_overlay_formula",
    "theme_delta_vs_base",
    "theme_delta_positive",
    "theme_positive_part",
    "theme_negative_part",
    "theme_overlay_gain",
    "theme_overlay_cap",
    "theme_overlay_signed_component",
    "theme_overlay_positive_component",
    "theme_overlay_negative_component",
    "theme_overlay_applied",
    "theme_overlay_capped",
    "theme_overlay_soft_conf_gate",
    "theme_uplift_applied",
    "theme_penalty_applied",
    "shadow_theme_weight_raw",
    "shadow_theme_weight",
    "shadow_theme_weight_effective",
    "shadow_base_weight",
    "shadow_floor_applied",
    "shadow_theme_score_effective",
    "shadow_final_score_v3",
    "shadow_score_diff_v3",
    "shadow_rank_v3",
    "shadow_explain",
    "live_rank",
    "rank_final",
    "rank_v2",
    "risk_penalty",
    "score_contribution_ret",
    "score_contribution_prob",
    "score_contribution_tech",
    "score_contribution_qual",
    "score_contribution_safety",
    "score_contribution_liquidity",
    "score_contribution_theme",
    "score_contribution_risk",
    "contrib_tech",
    "contrib_ret",
    "contrib_prob",
    "contrib_qual",
    "contrib_safety",
    "contrib_liquidity",
    "contrib_theme",
    "contrib_penalty",
    "top_positive_factor",
    "top_positive_value",
    "top_negative_factor",
    "top_negative_value",
    "explain_text",
    "explain",
    "score_explain_summary",
    "score_explain_strengths",
    "score_explain_risks",
    "score_explain_confidence",
    "score_explain_regime",
    "score_driver_1",
    "score_driver_2",
    "score_driver_3",
    "score_drag_1",
    "score_drag_2",
    "top_driver_1",
    "top_driver_2",
    "top_driver_3",
    "risk_factor_1",
    "risk_factor_2",
    "action_note",
    "score_explain_json",
    "confidence_version",
    "data_maturity_score",
    "model_reliability_score",
    "signal_agreement_score",
    "regime_fitness_score",
    "component_coverage_ratio",
    "confidence_score_research",
    "confidence_score_operational",
    "confidence_score",
    "confidence_label_research",
    "confidence_label_operational",
    "confidence_label",
    "confidence_grade",
    "confidence_reason",
    "confidence_explain_text",
    "market_up",
    "market_status_date",
    "market_kospi_close",
    "market_kospi_ma20",
    "market_vol_5d",
    "market_foreign_5d",
    "generated_at",
    "model_version",
    "score_formula_version",
]
DAILY_RANKING_PK = ["date", "code"]

# Fixed-weight legacy reference score.
# Note: the legacy reference still uses ret_score as its return / prediction axis.
WEIGHT_TECH = 0.15
WEIGHT_PRED = 0.30
WEIGHT_PROB = 0.25
WEIGHT_SAFETY = 0.15
WEIGHT_QUAL = 0.10
WEIGHT_LIQUIDITY = 0.05

# Regime-aware production final_score weights.
# All production score components are expected to be on a 0~100 scale.
# The positive-side weights sum to 1.0. Safety-oriented behavior is handled
# through risk_penalty rather than a direct safety bonus in final_score.
BULL_WEIGHT_PROFILE = {
    "profile": "bull_service_growth_lead",
    "ret": 0.35,
    "prob": 0.26,
    "tech": 0.29,
    "qual": 0.06,
    "valuation": 0.04,
    "risk_penalty": 0.40,
}

NEUTRAL_WEIGHT_PROFILE_BASELINE = {
    "profile": "neutral_service_balanced",
    "ret": 0.30,
    "prob": 0.26,
    "tech": 0.26,
    "qual": 0.10,
    "valuation": 0.08,
    "risk_penalty": 0.65,
}

NEUTRAL_WEIGHT_PROFILE_EXPERIMENTAL = {
    "profile": "neutral_service_tech_tilt_experimental",
    "ret": 0.29,
    "prob": 0.25,
    "tech": 0.29,
    "qual": 0.10,
    "valuation": 0.08,
    "risk_penalty": 0.65,
}

NEUTRAL_WEIGHT_PROFILE = NEUTRAL_WEIGHT_PROFILE_BASELINE

DEFENSIVE_WEIGHT_PROFILE = {
    "profile": "defensive_service_carry",
    "ret": 0.27,
    "prob": 0.24,
    "tech": 0.15,
    "qual": 0.19,
    "valuation": 0.15,
    "risk_penalty": 0.80,
}

# Risk penalty is handled as a subtraction from the weighted score.
# The current policy is intentionally softer than the previous version to
# reduce over-dominance of drawdown estimates in final_score ordering.
RISK_MDD_THRESHOLD = 0.15
RISK_PENALTY_SCALE = 100.0

# Rebalance score weights reuse the same component columns built here.
REBALANCE_WEIGHT_RET = 0.28
REBALANCE_WEIGHT_PROB = 0.25
REBALANCE_WEIGHT_QUAL = 0.20
REBALANCE_WEIGHT_TECH = 0.17
REBALANCE_WEIGHT_PRED = 0.10
REBALANCE_PRED_SCORE_DEFAULT = 60.0
EXPLAIN_FACTOR_LABELS = {}
DISPLAY_FACTOR_LABELS = {
    "contrib_tech": "Tech flow score",
    "contrib_ret": "Primary prediction score",
    "contrib_prob": "Top-bucket probability score",
    "contrib_qual": "Financial quality score",
    "contrib_valuation": "Valuation score",
    "contrib_theme": "Theme support score",
    "contrib_penalty": "Risk penalty",
}
THEME_WEIGHT_DEFAULT = float(os.environ.get("THEME_WEIGHT_DEFAULT", "0.15"))
THEME_WEIGHT_BULL = float(os.environ.get("THEME_WEIGHT_BULL", "0.20"))
THEME_WEIGHT_NEUTRAL = float(os.environ.get("THEME_WEIGHT_NEUTRAL", str(THEME_WEIGHT_DEFAULT)))
THEME_WEIGHT_DEFENSIVE = float(os.environ.get("THEME_WEIGHT_DEFENSIVE", "0.10"))
SHADOW_THEME_WEIGHT_FLOOR = float(os.environ.get("SHADOW_THEME_WEIGHT_FLOOR", "0.10"))
SHADOW_THEME_OVERLAY_FORMULA = str(
    os.environ.get(
        "THEME_OVERLAY_SHADOW_MODE",
        os.environ.get("SHADOW_THEME_OVERLAY_FORMULA", "symmetric_floor"),
    )
).strip().lower()
SHADOW_THEME_NEGATIVE_PENALTY_RATIO = float(os.environ.get("SHADOW_THEME_NEGATIVE_PENALTY_RATIO", "0.20"))
SHADOW_THEME_UPLIFT_THRESHOLD = float(os.environ.get("SHADOW_THEME_UPLIFT_THRESHOLD", "3.0"))
SHADOW_THEME_OVERLAY_GAIN = float(os.environ.get("THEME_OVERLAY_SHADOW_GAIN", "0.10"))
SHADOW_THEME_OVERLAY_CAP = float(os.environ.get("THEME_OVERLAY_SHADOW_CAP", "6.0"))
SHADOW_THEME_OVERLAY_BASELINE_ANCHOR = str(os.environ.get("THEME_OVERLAY_SHADOW_BASELINE_ANCHOR", "baseline_score")).strip().lower()
SHADOW_THEME_OVERLAY_SOFT_CONF_ENABLED_RAW = str(os.environ.get("THEME_OVERLAY_SHADOW_SOFT_CONF_ENABLED", "1")).strip()
THEME_V2_BASE_WEIGHT = 0.85
THEME_V2_THEME_WEIGHT = 0.15
THEME_WEIGHT_EXPERIMENT_DIR = DATA_DIR / "experiments" / "theme_weight"
BEST_THEME_WEIGHT_JSON = THEME_WEIGHT_EXPERIMENT_DIR / "best_weight.json"
BEST_THEME_WEIGHT_BY_REGIME_JSON = THEME_WEIGHT_EXPERIMENT_DIR / "best_weight_by_regime.json"

SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR = "symmetric_floor"
SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY = "asymmetric_positive_only"
SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_CAPPED = "asymmetric_positive_only_capped"
SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_SOFT_CONF = "asymmetric_positive_only_soft_conf"
SHADOW_THEME_OVERLAY_FORMULA_SOFT_PENALTY = "asymmetric_soft_penalty"
SHADOW_THEME_OVERLAY_FORMULA_THRESHOLD = "asymmetric_threshold"
SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_THRESHOLD = "asymmetric_positive_only_with_threshold"


def _clip_theme_weight(value: float) -> float:
    return float(min(max(value, 0.0), 0.30))


def _load_theme_weight_by_regime_payload() -> dict[str, object]:
    if BEST_THEME_WEIGHT_BY_REGIME_JSON.exists():
        try:
            return json.loads(BEST_THEME_WEIGHT_BY_REGIME_JSON.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to load theme weight by regime config: %s", BEST_THEME_WEIGHT_BY_REGIME_JSON)
    return {}


def _load_theme_weight_global_payload() -> dict[str, object]:
    if BEST_THEME_WEIGHT_JSON.exists():
        try:
            return json.loads(BEST_THEME_WEIGHT_JSON.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to load theme weight config: %s", BEST_THEME_WEIGHT_JSON)
    return {}


def _resolve_best_weight_from_payload(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return _clip_theme_weight(float(value))
    except (TypeError, ValueError):
        return None


def _resolve_theme_weight_info_for_regime(regime: str) -> dict[str, object]:
    by_regime_payload = _load_theme_weight_by_regime_payload()
    global_payload = _load_theme_weight_global_payload()
    regime_key = str(regime or "").strip().lower()

    by_regime_exact = _resolve_best_weight_from_payload(by_regime_payload, regime_key) if regime_key else None
    if by_regime_exact is not None:
        return {
            "theme_weight": by_regime_exact,
            "weight_source": "best_weight_by_regime",
            "regime_applied": regime_key,
        }

    by_regime_global = _resolve_best_weight_from_payload(by_regime_payload, "global")
    if by_regime_global is not None:
        return {
            "theme_weight": by_regime_global,
            "weight_source": "best_weight_by_regime_global",
            "regime_applied": "global",
        }

    global_best_weight = _resolve_best_weight_from_payload(global_payload, "best_weight")
    if global_best_weight is not None:
        return {
            "theme_weight": global_best_weight,
            "weight_source": "best_weight_global",
            "regime_applied": regime_key or "global",
        }

    return {
        "theme_weight": _clip_theme_weight(THEME_V2_THEME_WEIGHT),
        "weight_source": "fallback_default",
        "regime_applied": regime_key or "global",
    }


def _resolve_theme_weight_for_regime(regime: str) -> float:
    return float(_resolve_theme_weight_info_for_regime(regime).get("theme_weight", THEME_V2_THEME_WEIGHT))


def _resolve_theme_weight_series(base: pd.DataFrame) -> pd.Series:
    regime_series = base.get("regime")
    if regime_series is None:
        return pd.Series(THEME_V2_THEME_WEIGHT, index=base.index, dtype="float64")
    mapped = regime_series.astype(str).str.lower().map(_resolve_theme_weight_for_regime)
    return pd.to_numeric(mapped, errors="coerce").fillna(THEME_V2_THEME_WEIGHT).astype(float)


def _resolve_theme_weight_metadata_frame(base: pd.DataFrame) -> pd.DataFrame:
    regime_series = base.get("regime")
    if regime_series is None:
        return pd.DataFrame(
            {
                "theme_weight": pd.Series(THEME_V2_THEME_WEIGHT, index=base.index, dtype="float64"),
                "weight_source": pd.Series("fallback_default", index=base.index, dtype="object"),
                "regime_applied": pd.Series("global", index=base.index, dtype="object"),
            }
        )
    info_series = regime_series.astype(str).str.lower().apply(_resolve_theme_weight_info_for_regime)
    frame = pd.DataFrame(info_series.tolist(), index=base.index)
    frame["theme_weight"] = pd.to_numeric(frame.get("theme_weight"), errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
    frame["weight_source"] = frame.get("weight_source", "fallback_default").fillna("fallback_default").astype(str)
    frame["regime_applied"] = frame.get("regime_applied", "global").fillna("global").astype(str)
    return frame


def _parse_bool_like(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text_value in {"0", "false", "f", "no", "n", "off"}:
        return False
    logging.warning("Invalid bool value=%r; using default=%s", value, default)
    return default


def _parse_float_like(value, default: float, minimum: float | None = None, maximum: float | None = None, name: str = "value") -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logging.warning("Invalid %s=%r; using default=%s", name, value, default)
        parsed = float(default)
    if minimum is not None and parsed < minimum:
        logging.warning("%s=%s below minimum=%s; using default=%s", name, parsed, minimum, default)
        parsed = float(default)
    if maximum is not None and parsed > maximum:
        logging.warning("%s=%s above maximum=%s; using default=%s", name, parsed, maximum, default)
        parsed = float(default)
    return float(parsed)


def resolve_shadow_theme_overlay_config(args: argparse.Namespace | None = None) -> dict[str, object]:
    mode_raw = getattr(args, "theme_overlay_shadow_mode", None) if args is not None else None
    if mode_raw is None:
        mode_raw = SHADOW_THEME_OVERLAY_FORMULA
    mode = str(mode_raw or SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR).strip().lower()
    allowed_modes = {
        SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR,
        SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY,
        SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_CAPPED,
        SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_SOFT_CONF,
        SHADOW_THEME_OVERLAY_FORMULA_SOFT_PENALTY,
        SHADOW_THEME_OVERLAY_FORMULA_THRESHOLD,
        SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_THRESHOLD,
    }
    if mode not in allowed_modes:
        logging.warning("Invalid THEME_OVERLAY_SHADOW_MODE=%r; falling back to %s", mode_raw, SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR)
        mode = SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR

    gain_raw = getattr(args, "theme_overlay_shadow_gain", None) if args is not None else None
    if gain_raw is None:
        gain_raw = SHADOW_THEME_OVERLAY_GAIN
    cap_raw = getattr(args, "theme_overlay_shadow_cap", None) if args is not None else None
    if cap_raw is None:
        cap_raw = SHADOW_THEME_OVERLAY_CAP
    anchor_raw = getattr(args, "theme_overlay_shadow_baseline_anchor", None) if args is not None else None
    if anchor_raw is None:
        anchor_raw = SHADOW_THEME_OVERLAY_BASELINE_ANCHOR
    soft_conf_raw = getattr(args, "theme_overlay_shadow_soft_conf_enabled", None) if args is not None else None
    if soft_conf_raw is None:
        soft_conf_raw = SHADOW_THEME_OVERLAY_SOFT_CONF_ENABLED_RAW

    anchor = str(anchor_raw or "baseline_score").strip().lower()
    if anchor not in {"baseline_score", "final_score"}:
        logging.warning("Invalid THEME_OVERLAY_SHADOW_BASELINE_ANCHOR=%r; falling back to baseline_score", anchor_raw)
        anchor = "baseline_score"

    config = {
        "mode": mode,
        "gain": _parse_float_like(gain_raw, 0.10, minimum=0.0, maximum=1.0, name="THEME_OVERLAY_SHADOW_GAIN"),
        "cap": _parse_float_like(cap_raw, 6.0, minimum=0.0, maximum=100.0, name="THEME_OVERLAY_SHADOW_CAP"),
        "baseline_anchor": anchor,
        "soft_conf_enabled": _parse_bool_like(soft_conf_raw, True),
        "floor": float(_clip_theme_weight(SHADOW_THEME_WEIGHT_FLOOR)),
        "negative_penalty_ratio": float(max(SHADOW_THEME_NEGATIVE_PENALTY_RATIO, 0.0)),
        "uplift_threshold": float(max(SHADOW_THEME_UPLIFT_THRESHOLD, 0.0)),
    }
    logging.info(
        "shadow theme overlay config: mode=%s gain=%.3f cap=%.3f baseline_anchor=%s soft_conf_enabled=%s floor=%.3f",
        config["mode"],
        config["gain"],
        config["cap"],
        config["baseline_anchor"],
        config["soft_conf_enabled"],
        config["floor"],
    )
    return config


def apply_shadow_theme_overlay_config(config: dict[str, object] | None = None) -> dict[str, object]:
    global SHADOW_THEME_OVERLAY_FORMULA
    global SHADOW_THEME_OVERLAY_GAIN
    global SHADOW_THEME_OVERLAY_CAP
    global SHADOW_THEME_OVERLAY_BASELINE_ANCHOR
    global SHADOW_THEME_OVERLAY_SOFT_CONF_ENABLED_RAW

    resolved = config or resolve_shadow_theme_overlay_config()
    SHADOW_THEME_OVERLAY_FORMULA = str(resolved.get("mode") or SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR).strip().lower()
    SHADOW_THEME_OVERLAY_GAIN = float(resolved.get("gain", SHADOW_THEME_OVERLAY_GAIN))
    SHADOW_THEME_OVERLAY_CAP = float(resolved.get("cap", SHADOW_THEME_OVERLAY_CAP))
    SHADOW_THEME_OVERLAY_BASELINE_ANCHOR = str(resolved.get("baseline_anchor") or "baseline_score").strip().lower()
    SHADOW_THEME_OVERLAY_SOFT_CONF_ENABLED_RAW = "1" if bool(resolved.get("soft_conf_enabled", True)) else "0"
    return resolved


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--theme-overlay-shadow-mode", type=str, default=None, help="Shadow overlay mode: symmetric_floor, asymmetric_positive_only, asymmetric_positive_only_capped, asymmetric_positive_only_soft_conf.")
    parser.add_argument("--theme-overlay-shadow-gain", type=float, default=None, help="Gain applied to positive shadow delta.")
    parser.add_argument("--theme-overlay-shadow-cap", type=float, default=None, help="Maximum positive overlay contribution for capped shadow modes.")
    parser.add_argument("--theme-overlay-shadow-baseline-anchor", type=str, default=None, help="Shadow baseline anchor. default=baseline_score.")
    parser.add_argument("--theme-overlay-shadow-soft-conf-enabled", type=str, default=None, help="Enable soft confidence gate for shadow overlay. Accepts 0/1/true/false.")
    parser.add_argument("--theme-risk-soft", action="store_true", help="Enable theme-only soft risk penalty experiment.")
    parser.add_argument("--theme-risk-soft-factor", type=float, default=None, help="Soft factor applied to risk penalty for eligible themed names.")
    parser.add_argument("--theme-risk-min-score", type=float, default=None, help="Minimum theme_score to apply soft risk penalty.")
    parser.add_argument("--theme-risk-min-confidence", type=float, default=None, help="Minimum theme_confidence to apply soft risk penalty.")
    parser.add_argument("--risk-curve-experiment", action="store_true", help="Enable sidecar risk penalty curve experiments.")
    parser.add_argument("--risk-curve-exp-a-threshold", type=float, default=None, help="Threshold above which exp_a softens the penalty curve.")
    parser.add_argument("--risk-curve-exp-a-slope-ratio", type=float, default=None, help="Slope ratio applied above threshold for exp_a.")
    parser.add_argument("--risk-curve-exp-b-delayed-reach-factor", type=float, default=None, help="Delayed reach factor for exp_b.")
    parser.add_argument("--risk-curve-feature-candidate", type=str, default=None, help="Feature candidate name. Use exp_b_delayed_cap or none.")
    parser.add_argument("--risk-curve-feature-candidate-enabled", type=str, default=None, help="Enable feature candidate sidecar path. Accepts 0/1/true/false.")
    parser.add_argument("--exp-b-delayed-cap-reach-factor", type=float, default=None, help="Feature candidate exp_b delayed reach factor.")
    parser.add_argument("--exp-b-delayed-cap-max-penalty-ratio", type=float, default=None, help="Floor ratio versus baseline penalty for feature candidate exp_b.")
    parser.add_argument("--exp-b-delayed-cap-apply-regimes", type=str, default=None, help="Comma-separated allowed regimes for feature candidate exp_b.")
    parser.add_argument("--exp-b-delayed-cap-theme-only", type=str, default=None, help="Limit feature candidate exp_b to themed names only. Accepts 0/1/true/false.")
    parser.add_argument("--exp-b-delayed-cap-min-theme-score", type=float, default=None, help="Minimum theme_score required for feature candidate exp_b.")
    parser.add_argument("--exp-b-delayed-cap-min-theme-confidence", type=float, default=None, help="Minimum theme_confidence required for feature candidate exp_b.")
    return parser.parse_args()


def resolve_theme_risk_soft_config(args: argparse.Namespace | None = None) -> dict:
    if is_operational_runtime_mode() and not allow_experimental_runtime_features(False):
        config = {
            "enabled": False,
            "soft_factor": float(RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT),
            "min_score": float(RISK_PENALTY_THEME_MIN_SCORE_DEFAULT),
            "min_confidence": float(RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT),
        }
        logging.info("theme risk soft config forced off by operational runtime")
        return config

    env_enabled = _parse_bool_like(os.environ.get("RISK_PENALTY_THEME_ONLY_SOFT"), RISK_PENALTY_THEME_ONLY_SOFT_DEFAULT)
    enabled = env_enabled
    if args is not None and getattr(args, "theme_risk_soft", False):
        enabled = True

    factor_raw = getattr(args, "theme_risk_soft_factor", None) if args is not None else None
    if factor_raw is None:
        factor_raw = os.environ.get("RISK_PENALTY_THEME_SOFT_FACTOR", RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT)
    min_score_raw = getattr(args, "theme_risk_min_score", None) if args is not None else None
    if min_score_raw is None:
        min_score_raw = os.environ.get("RISK_PENALTY_THEME_MIN_SCORE", RISK_PENALTY_THEME_MIN_SCORE_DEFAULT)
    min_conf_raw = getattr(args, "theme_risk_min_confidence", None) if args is not None else None
    if min_conf_raw is None:
        min_conf_raw = os.environ.get("RISK_PENALTY_THEME_MIN_CONFIDENCE", RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT)

    factor = _parse_float_like(factor_raw, RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT, minimum=0.05, maximum=1.0, name="RISK_PENALTY_THEME_SOFT_FACTOR")
    min_score = _parse_float_like(min_score_raw, RISK_PENALTY_THEME_MIN_SCORE_DEFAULT, minimum=0.0, maximum=100.0, name="RISK_PENALTY_THEME_MIN_SCORE")
    min_conf = _parse_float_like(min_conf_raw, RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT, minimum=0.0, maximum=1.0, name="RISK_PENALTY_THEME_MIN_CONFIDENCE")

    config = {
        "enabled": bool(enabled),
        "soft_factor": float(factor),
        "min_score": float(min_score),
        "min_confidence": float(min_conf),
    }
    logging.info(
        "theme risk soft config: enabled=%s factor=%.3f min_score=%.2f min_confidence=%.2f",
        config["enabled"],
        config["soft_factor"],
        config["min_score"],
        config["min_confidence"],
    )
    return config


def resolve_risk_curve_experiment_config(args: argparse.Namespace | None = None) -> dict:
    if is_operational_runtime_mode() and not allow_experimental_runtime_features(False):
        config = {
            "enabled": False,
            "exp_a_threshold": float(EXP_A_THRESHOLD_DEFAULT),
            "exp_a_slope_ratio": float(EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT),
            "exp_b_delayed_reach_factor": float(EXP_B_DELAYED_REACH_FACTOR_DEFAULT),
            "penalty_cap": float(PENALTY_CAP_DEFAULT),
        }
        logging.info("risk curve experiment config forced off by operational runtime")
        return config

    env_enabled = _parse_bool_like(os.environ.get("RISK_CURVE_EXPERIMENT"), RISK_CURVE_EXPERIMENT_DEFAULT)
    enabled = env_enabled
    if args is not None and getattr(args, "risk_curve_experiment", False):
        enabled = True

    threshold_raw = getattr(args, "risk_curve_exp_a_threshold", None) if args is not None else None
    if threshold_raw is None:
        threshold_raw = os.environ.get("RISK_CURVE_EXP_A_THRESHOLD", EXP_A_THRESHOLD_DEFAULT)
    slope_raw = getattr(args, "risk_curve_exp_a_slope_ratio", None) if args is not None else None
    if slope_raw is None:
        slope_raw = os.environ.get("RISK_CURVE_EXP_A_SLOPE_RATIO", EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT)
    delayed_raw = getattr(args, "risk_curve_exp_b_delayed_reach_factor", None) if args is not None else None
    if delayed_raw is None:
        delayed_raw = os.environ.get("RISK_CURVE_EXP_B_DELAYED_REACH_FACTOR", EXP_B_DELAYED_REACH_FACTOR_DEFAULT)

    config = {
        "enabled": bool(enabled),
        "exp_a_threshold": _parse_float_like(
            threshold_raw,
            EXP_A_THRESHOLD_DEFAULT,
            minimum=0.10,
            maximum=0.50,
            name="RISK_CURVE_EXP_A_THRESHOLD",
        ),
        "exp_a_slope_ratio": _parse_float_like(
            slope_raw,
            EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT,
            minimum=0.10,
            maximum=1.0,
            name="RISK_CURVE_EXP_A_SLOPE_RATIO",
        ),
        "exp_b_delayed_reach_factor": _parse_float_like(
            delayed_raw,
            EXP_B_DELAYED_REACH_FACTOR_DEFAULT,
            minimum=1.01,
            maximum=2.0,
            name="RISK_CURVE_EXP_B_DELAYED_REACH_FACTOR",
        ),
        "penalty_cap": PENALTY_CAP_DEFAULT,
    }
    logging.info(
        "risk curve experiment config: enabled=%s exp_a_threshold=%.3f exp_a_slope_ratio=%.3f exp_b_delayed_reach_factor=%.3f penalty_cap=%.2f",
        config["enabled"],
        config["exp_a_threshold"],
        config["exp_a_slope_ratio"],
        config["exp_b_delayed_reach_factor"],
        config["penalty_cap"],
    )
    return config


def normalize_apply_regimes(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_parts = [str(v).strip().lower() for v in value]
    else:
        raw_parts = [part.strip().lower() for part in str(value).split(",")]
    mapped: list[str] = []
    alias = {
        "sideways": "neutral",
        "sideway": "neutral",
    }
    for part in raw_parts:
        if not part:
            continue
        normalized = alias.get(part, part)
        if normalized in {"bull", "neutral", "defensive"} and normalized not in mapped:
            mapped.append(normalized)
    return mapped


def is_feature_candidate_enabled(config: dict | None) -> bool:
    if not config:
        return False
    return bool(config.get("enabled", False)) and str(config.get("candidate", "none")).strip().lower() != "none"


def resolve_feature_candidate_config(args: argparse.Namespace | None = None) -> dict:
    if is_operational_runtime_mode() and not allow_experimental_runtime_features(False):
        config = {
            "candidate": "none",
            "enabled": False,
            "exp_b_delayed_cap_reach_factor": float(EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT),
            "exp_b_delayed_cap_max_penalty_ratio": float(EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT),
            "exp_b_delayed_cap_apply_regimes": normalize_apply_regimes(EXP_B_DELAYED_CAP_APPLY_REGIMES_DEFAULT),
            "exp_b_delayed_cap_theme_only": bool(EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT),
            "exp_b_delayed_cap_min_theme_score": float(EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT),
            "exp_b_delayed_cap_min_theme_confidence": float(EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT),
            "penalty_cap": float(PENALTY_CAP_DEFAULT),
        }
        logging.info("feature candidate config forced off by operational runtime")
        return config

    candidate_raw = getattr(args, "risk_curve_feature_candidate", None) if args is not None else None
    if candidate_raw is None:
        candidate_raw = os.environ.get("RISK_CURVE_FEATURE_CANDIDATE", RISK_CURVE_FEATURE_CANDIDATE_DEFAULT)
    candidate = str(candidate_raw).strip().lower() or RISK_CURVE_FEATURE_CANDIDATE_DEFAULT
    if candidate not in {"none", "exp_b_delayed_cap"}:
        logging.warning("Invalid RISK_CURVE_FEATURE_CANDIDATE=%r; using none", candidate_raw)
        candidate = "none"

    enabled_raw = getattr(args, "risk_curve_feature_candidate_enabled", None) if args is not None else None
    if enabled_raw is None:
        enabled_raw = os.environ.get("RISK_CURVE_FEATURE_CANDIDATE_ENABLED", RISK_CURVE_FEATURE_CANDIDATE_ENABLED_DEFAULT)
    enabled = _parse_bool_like(enabled_raw, RISK_CURVE_FEATURE_CANDIDATE_ENABLED_DEFAULT)

    reach_raw = getattr(args, "exp_b_delayed_cap_reach_factor", None) if args is not None else None
    if reach_raw is None:
        reach_raw = os.environ.get("EXP_B_DELAYED_CAP_REACH_FACTOR", EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT)
    max_ratio_raw = getattr(args, "exp_b_delayed_cap_max_penalty_ratio", None) if args is not None else None
    if max_ratio_raw is None:
        max_ratio_raw = os.environ.get("EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO", EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT)
    regimes_raw = getattr(args, "exp_b_delayed_cap_apply_regimes", None) if args is not None else None
    if regimes_raw is None:
        regimes_raw = os.environ.get("EXP_B_DELAYED_CAP_APPLY_REGIMES", EXP_B_DELAYED_CAP_APPLY_REGIMES_DEFAULT)
    theme_only_raw = getattr(args, "exp_b_delayed_cap_theme_only", None) if args is not None else None
    if theme_only_raw is None:
        theme_only_raw = os.environ.get("EXP_B_DELAYED_CAP_THEME_ONLY", EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT)
    min_theme_score_raw = getattr(args, "exp_b_delayed_cap_min_theme_score", None) if args is not None else None
    if min_theme_score_raw is None:
        min_theme_score_raw = os.environ.get("EXP_B_DELAYED_CAP_MIN_THEME_SCORE", EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT)
    min_theme_confidence_raw = getattr(args, "exp_b_delayed_cap_min_theme_confidence", None) if args is not None else None
    if min_theme_confidence_raw is None:
        min_theme_confidence_raw = os.environ.get("EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE", EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT)

    config = {
        "candidate": candidate,
        "enabled": bool(enabled),
        "exp_b_delayed_cap_reach_factor": _parse_float_like(
            reach_raw,
            EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT,
            minimum=1.01,
            maximum=2.0,
            name="EXP_B_DELAYED_CAP_REACH_FACTOR",
        ),
        "exp_b_delayed_cap_max_penalty_ratio": _parse_float_like(
            max_ratio_raw,
            EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT,
            minimum=0.10,
            maximum=1.0,
            name="EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO",
        ),
        "exp_b_delayed_cap_apply_regimes": normalize_apply_regimes(regimes_raw),
        "exp_b_delayed_cap_theme_only": _parse_bool_like(theme_only_raw, EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT),
        "exp_b_delayed_cap_min_theme_score": _parse_float_like(
            min_theme_score_raw,
            EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT,
            minimum=0.0,
            maximum=100.0,
            name="EXP_B_DELAYED_CAP_MIN_THEME_SCORE",
        ),
        "exp_b_delayed_cap_min_theme_confidence": _parse_float_like(
            min_theme_confidence_raw,
            EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT,
            minimum=0.0,
            maximum=1.0,
            name="EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE",
        ),
        "penalty_cap": PENALTY_CAP_DEFAULT,
    }
    logging.info(
        "feature candidate config: candidate=%s enabled=%s reach_factor=%.3f max_penalty_ratio=%.3f apply_regimes=%s theme_only=%s min_theme_score=%.2f min_theme_confidence=%.2f",
        config["candidate"],
        config["enabled"],
        config["exp_b_delayed_cap_reach_factor"],
        config["exp_b_delayed_cap_max_penalty_ratio"],
        ",".join(config["exp_b_delayed_cap_apply_regimes"]) if config["exp_b_delayed_cap_apply_regimes"] else "(all)",
        config["exp_b_delayed_cap_theme_only"],
        config["exp_b_delayed_cap_min_theme_score"],
        config["exp_b_delayed_cap_min_theme_confidence"],
    )
    return config


THEME_WEIGHT_BULL = _clip_theme_weight(THEME_WEIGHT_BULL)
THEME_WEIGHT_NEUTRAL = _clip_theme_weight(THEME_WEIGHT_NEUTRAL)
THEME_WEIGHT_DEFENSIVE = _clip_theme_weight(THEME_WEIGHT_DEFENSIVE)
THEME_OVERLAY_WEIGHTS = {
    "bull": {
        "base_weight": 1.0 - THEME_WEIGHT_BULL,
        "theme_weight": THEME_WEIGHT_BULL,
    },
    "neutral": {
        "base_weight": 1.0 - THEME_WEIGHT_NEUTRAL,
        "theme_weight": THEME_WEIGHT_NEUTRAL,
    },
    "defensive": {
        "base_weight": 1.0 - THEME_WEIGHT_DEFENSIVE,
        "theme_weight": THEME_WEIGHT_DEFENSIVE,
    },
}


def resolve_score_formula_version() -> str:
    if is_operational_runtime_mode():
        value = DEFAULT_SCORE_FORMULA_VERSION
    else:
        value = str(os.environ.get("SCORE_FORMULA_VERSION", DEFAULT_SCORE_FORMULA_VERSION)).strip()
    return value or DEFAULT_SCORE_FORMULA_VERSION


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _set_theme_gate_debug(**kwargs: object) -> None:
    global LAST_THEME_GATE_DEBUG
    payload = dict(LAST_THEME_GATE_DEBUG)
    payload.update(kwargs)
    LAST_THEME_GATE_DEBUG = payload


def resolve_theme_overlay_mode(enable_theme_overlay_raw: str | None = None, requested_mode_raw: str | None = None) -> dict[str, object]:
    config_enable_default = "1" if bool(get_production_config_value(["ranking", "theme_overlay", "enabled"], False)) else "0"
    config_mode_default = str(get_production_config_value(["ranking", "theme_overlay", "mode"], THEME_OVERLAY_OFF)).strip().lower() or THEME_OVERLAY_OFF
    if is_operational_runtime_mode() and not allow_experimental_runtime_features(False):
        enable_raw = str(enable_theme_overlay_raw if enable_theme_overlay_raw is not None else config_enable_default).strip()
        requested_raw = str(requested_mode_raw if requested_mode_raw is not None else config_mode_default).strip().lower()
    else:
        enable_raw = str(enable_theme_overlay_raw if enable_theme_overlay_raw is not None else os.environ.get("ENABLE_THEME_OVERLAY", config_enable_default)).strip()
        requested_raw = str(requested_mode_raw if requested_mode_raw is not None else os.environ.get("THEME_OVERLAY_MODE", config_mode_default)).strip().lower()
    requested_execution_mode = requested_raw or THEME_OVERLAY_OFF

    fallback_applied = False
    fallback_reason = "(none)"

    if enable_raw != "1":
        resolved_execution_mode = THEME_OVERLAY_OFF
        fallback_applied = requested_execution_mode != THEME_OVERLAY_OFF
        fallback_reason = "disabled_by_flag"
    elif requested_execution_mode == THEME_OVERLAY_SHADOW:
        resolved_execution_mode = THEME_OVERLAY_SHADOW
    elif requested_execution_mode == THEME_OVERLAY_OPERATIONAL:
        resolved_execution_mode = THEME_OVERLAY_OPERATIONAL
    elif requested_execution_mode == THEME_OVERLAY_OFF:
        resolved_execution_mode = THEME_OVERLAY_OFF
    else:
        resolved_execution_mode = THEME_OVERLAY_OFF
        fallback_applied = True
        fallback_reason = "invalid_mode"

    return {
        "requested_execution_mode": requested_execution_mode,
        "resolved_execution_mode": resolved_execution_mode,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
    }


def _resolve_theme_overlay_runtime() -> dict[str, object]:
    config_enable_default = "1" if bool(get_production_config_value(["ranking", "theme_overlay", "enabled"], False)) else "0"
    config_mode_default = str(get_production_config_value(["ranking", "theme_overlay", "mode"], THEME_OVERLAY_OFF)).strip().lower() or THEME_OVERLAY_OFF
    config_validation_default = "1" if bool(get_production_config_value(["ranking", "theme_overlay", "validation_enabled"], False)) else "0"
    if is_operational_runtime_mode() and not allow_experimental_runtime_features(False):
        enable_theme_overlay_raw = config_enable_default
        enable_theme_validation_raw = config_validation_default
        requested_mode = config_mode_default
    else:
        enable_theme_overlay_raw = str(os.environ.get("ENABLE_THEME_OVERLAY", config_enable_default)).strip()
        enable_theme_validation_raw = str(os.environ.get("ENABLE_THEME_VALIDATION", config_validation_default)).strip()
        requested_mode = os.environ.get("THEME_OVERLAY_MODE", config_mode_default)
    mode_info = resolve_theme_overlay_mode(enable_theme_overlay_raw, requested_mode)
    by_regime_exists = BEST_THEME_WEIGHT_BY_REGIME_JSON.exists()
    global_exists = BEST_THEME_WEIGHT_JSON.exists()
    current_execution_mode = str(mode_info["resolved_execution_mode"])
    overlay_enabled = current_execution_mode != THEME_OVERLAY_OFF
    operational = current_execution_mode == THEME_OVERLAY_OPERATIONAL
    overlay_disable_reason = ""
    if current_execution_mode == THEME_OVERLAY_OFF and enable_theme_overlay_raw != "1":
        overlay_disable_reason = "disabled_by_flag"
    elif current_execution_mode == THEME_OVERLAY_SHADOW:
        overlay_disable_reason = "mode_mismatch"
    elif bool(mode_info.get("fallback_applied")):
        overlay_disable_reason = str(mode_info.get("fallback_reason") or "unknown")
    _set_theme_gate_debug(
        enable_theme_overlay_raw=enable_theme_overlay_raw,
        theme_overlay_mode_requested=str(requested_mode).strip().lower() or THEME_OVERLAY_OFF,
        current_execution_mode=current_execution_mode,
        requested_execution_mode=mode_info["requested_execution_mode"],
        resolved_execution_mode=mode_info["resolved_execution_mode"],
        fallback_applied=bool(mode_info["fallback_applied"]),
        fallback_reason=str(mode_info["fallback_reason"]),
        enable_theme_validation_raw=enable_theme_validation_raw,
        overlay_gate_result="enabled" if operational else "disabled",
        overlay_disable_reason=overlay_disable_reason or "(none)",
        theme_weight_source_priority=[
            "best_weight_by_regime",
            "best_weight_by_regime_global",
            "best_weight_global",
            "fallback_default",
        ],
        theme_weight_config_paths={
            "best_weight_by_regime": str(BEST_THEME_WEIGHT_BY_REGIME_JSON),
            "best_weight_global": str(BEST_THEME_WEIGHT_JSON),
        },
        theme_weight_config_available={
            "best_weight_by_regime": bool(by_regime_exists),
            "best_weight_global": bool(global_exists),
        },
    )
    return {
        "overlay_enabled": overlay_enabled,
        "mode": current_execution_mode,
        "operational": operational,
        "shadow": current_execution_mode == THEME_OVERLAY_SHADOW,
        "coverage_threshold": THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT,
        "enable_theme_overlay_raw": enable_theme_overlay_raw,
        "enable_theme_validation_raw": enable_theme_validation_raw,
        "current_execution_mode": current_execution_mode,
        "overlay_disable_reason": overlay_disable_reason or "(none)",
        "requested_execution_mode": mode_info["requested_execution_mode"],
        "resolved_execution_mode": mode_info["resolved_execution_mode"],
        "fallback_applied": bool(mode_info["fallback_applied"]),
        "fallback_reason": str(mode_info["fallback_reason"]),
    }


def _set_theme_guard_status(**kwargs: object) -> None:
    global LAST_THEME_GUARD_STATUS
    status = dict(LAST_THEME_GUARD_STATUS)
    status.update(kwargs)
    LAST_THEME_GUARD_STATUS = status


def _theme_gate_allows_score_application() -> bool:
    return bool(LAST_THEME_GUARD_STATUS.get("applied", False))


def _resolve_theme_overlay_runtime_flags() -> dict[str, object]:
    mode = str(LAST_THEME_GUARD_STATUS.get("mode", THEME_OVERLAY_OFF) or THEME_OVERLAY_OFF)
    applied = bool(LAST_THEME_GUARD_STATUS.get("applied", False))
    operational = mode == THEME_OVERLAY_OPERATIONAL
    shadow = mode == THEME_OVERLAY_SHADOW
    off = mode == THEME_OVERLAY_OFF
    live_uses_theme = operational and applied
    return {
        "mode": mode,
        "applied": applied,
        "operational": operational,
        "shadow": shadow,
        "off": off,
        "live_uses_theme": live_uses_theme,
        "shadow_score_enabled": shadow or live_uses_theme,
    }


def _is_active_theme_label(value: object) -> bool:
    cleaned = str(value or "").strip()
    return cleaned not in {"", "(none)"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def _load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input CSV not found: {path}")
        logging.warning("Optional input CSV not found: %s", path)
        return pd.DataFrame()
    read_kwargs: dict[str, object] = {"low_memory": False}
    if path.name.lower() in {"predictions.csv", "scores_final.csv", "features.csv", "universe.csv"}:
        read_kwargs["dtype"] = {"code": str}
    df = pd.read_csv(path, **read_kwargs)
    logging.info("Loaded %s (rows=%d)", path, len(df))
    return df


def _empty_theme_overlay_payload(source: str = "none") -> dict[str, object]:
    return {
        "df": pd.DataFrame(),
        "source": source,
        "latest_theme_date": "NA",
        "available_dates": [],
        "theme_row_count": 0,
        "load_error_reason": "missing_theme_input",
    }


def _load_theme_overlay_from_csv(dates: list[str]) -> dict[str, object]:
    if not dates or not STOCK_THEME_DAILY_CSV.exists():
        return _empty_theme_overlay_payload("csv")
    try:
        df = pd.read_csv(STOCK_THEME_DAILY_CSV, dtype={"code": str})
        if df.empty:
            return _empty_theme_overlay_payload("csv")
        required_cols = ["date", "code", "dominant_theme", "theme_score", "theme_confidence", "theme_score_raw"]
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        if missing_required_cols:
            payload = _empty_theme_overlay_payload("csv")
            payload["load_error_reason"] = "missing_required_columns"
            logging.warning("Theme overlay CSV missing required columns: %s", missing_required_cols)
            return payload
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["code"] = df["code"].astype(str).str.zfill(6)
        latest_theme_date = df["date"].dropna().astype(str).max() if "date" in df.columns else "NA"
        filtered = df[df["date"].isin(dates)].copy()
        if filtered.empty:
            logging.info("Theme overlay CSV exists but has no rows for ranking dates=%s", dates[:5])
            payload = _empty_theme_overlay_payload("csv")
            payload["latest_theme_date"] = latest_theme_date or "NA"
            return payload
        filtered["theme_score"] = pd.to_numeric(filtered["theme_score"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
        filtered["theme_confidence"] = pd.to_numeric(filtered["theme_confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        filtered["theme_raw_score"] = pd.to_numeric(filtered.get("theme_score_raw"), errors="coerce")
        filtered["dominant_theme"] = filtered["dominant_theme"].fillna("").astype(str)
        filtered = (
            filtered.sort_values(["date", "code", "theme_score", "theme_confidence"], ascending=[True, True, False, False])
            .drop_duplicates(subset=["date", "code"], keep="first")
            .reset_index(drop=True)
        )
        logging.info("Loaded theme overlay from CSV rows=%d path=%s", len(filtered), STOCK_THEME_DAILY_CSV)
        return {
            "df": filtered[["date", "code", "dominant_theme", "theme_score", "theme_confidence", "theme_raw_score"]],
            "source": "csv",
            "latest_theme_date": latest_theme_date or "NA",
            "available_dates": sorted(filtered["date"].dropna().astype(str).unique().tolist()),
            "theme_row_count": int(len(filtered)),
            "load_error_reason": "",
        }
    except Exception:
        logging.exception("Failed to load theme overlay from CSV: %s", STOCK_THEME_DAILY_CSV)
        payload = _empty_theme_overlay_payload("csv")
        payload["load_error_reason"] = "unknown"
        return payload


def _load_theme_overlay_from_db(dates: list[str]) -> dict[str, object]:
    if not get_engine or not dates:
        return _empty_theme_overlay_payload("db")
    try:
        eng = get_engine()
        query = text(
            """
            WITH exposure AS (
                SELECT
                    as_of_date,
                    stock_code,
                    theme_code,
                    exposure_score,
                    exposure_weight,
                    supporting_etf_count
                FROM stock_theme_exposure_daily
                WHERE as_of_date IN :dates
            ),
            dominant AS (
                SELECT DISTINCT ON (as_of_date, stock_code)
                    as_of_date,
                    stock_code,
                    theme_code AS dominant_theme,
                    exposure_score,
                    exposure_weight,
                    supporting_etf_count
                FROM exposure
                ORDER BY as_of_date, stock_code, exposure_score DESC, theme_code
            ),
            totals AS (
                SELECT
                    as_of_date,
                    stock_code,
                    SUM(ABS(COALESCE(exposure_score, 0))) AS total_exposure_score
                FROM exposure
                GROUP BY as_of_date, stock_code
            )
            SELECT
                d.as_of_date AS date,
                d.stock_code AS code,
                d.dominant_theme,
                ts.theme_score,
                d.exposure_score,
                d.exposure_weight,
                d.supporting_etf_count,
                t.total_exposure_score,
                ts.signal_score AS theme_raw_score
            FROM dominant d
            LEFT JOIN totals t
              ON d.as_of_date = t.as_of_date
             AND d.stock_code = t.stock_code
            LEFT JOIN theme_score_daily ts
              ON d.as_of_date = ts.as_of_date
             AND d.dominant_theme = ts.theme_code
            """
        ).bindparams(bindparam("dates", expanding=True))
        with eng.connect() as conn:
            rows = conn.execute(query, {"dates": list(dates)}).mappings().all()
        if not rows:
            logging.info("Theme overlay not found for ranking dates=%s", dates[:5])
            return _empty_theme_overlay_payload("db")
        df = pd.DataFrame(rows)
        for col in [
            "theme_score",
            "exposure_score",
            "exposure_weight",
            "supporting_etf_count",
            "total_exposure_score",
            "theme_raw_score",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        dominance_ratio = np.where(
            df["total_exposure_score"].fillna(0.0) > 0.0,
            df["exposure_score"].fillna(0.0) / df["total_exposure_score"].replace(0.0, np.nan),
            0.0,
        )
        etf_support_ratio = (df["supporting_etf_count"].fillna(0.0) / 3.0).clip(lower=0.0, upper=1.0)
        theme_score_pct = (df["theme_score"].fillna(0.0) * 100.0).clip(lower=0.0, upper=100.0)
        df["theme_score"] = theme_score_pct
        df["theme_confidence"] = (
            0.50 * (theme_score_pct / 100.0)
            + 0.30 * pd.Series(dominance_ratio, index=df.index).fillna(0.0).clip(lower=0.0, upper=1.0)
            + 0.20 * etf_support_ratio
        ).clip(lower=0.0, upper=1.0)
        logging.info("Loaded theme overlay rows=%d", len(df))
        return {
            "df": df[["date", "code", "dominant_theme", "theme_score", "theme_confidence", "theme_raw_score"]],
            "source": "db",
            "latest_theme_date": df["date"].dropna().astype(str).max() if "date" in df.columns else "NA",
            "available_dates": sorted(df["date"].dropna().astype(str).unique().tolist()),
            "theme_row_count": int(len(df)),
            "load_error_reason": "",
        }
    except Exception:
        logging.exception("Failed to load theme overlay from DB; theme factor disabled")
        payload = _empty_theme_overlay_payload("db")
        payload["load_error_reason"] = "unknown"
        return payload


def _load_theme_overlay(dates: list[str]) -> dict[str, object]:
    if STOCK_THEME_DAILY_CSV.exists():
        return _load_theme_overlay_from_csv(dates)
    db_overlay = _load_theme_overlay_from_db(dates)
    if not db_overlay["df"].empty:
        return db_overlay
    return _empty_theme_overlay_payload("none")


def _clip01(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.astype(float).clip(lower=lower, upper=upper)


def _percentile_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a per-date percentile score in the 0~100 range."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True, method="average") * 100.0

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def _percentile01_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a per-date percentile score in the 0~1 range."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, ascending=True, method="average")

    return df.groupby("date", group_keys=False)[col].transform(_rank)


def _winsorize_by_date(df: pd.DataFrame, col: str, *, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _clip(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        valid = s.dropna()
        if valid.empty:
            return s
        lo = valid.quantile(lower_q)
        hi = valid.quantile(upper_q)
        return s.clip(lower=lo, upper=hi)

    return df.groupby("date", group_keys=False)[col].transform(_clip)


def _average_available(parts: list[pd.Series], index: pd.Index, *, fill_value: float = np.nan) -> pd.Series:
    valid_parts = [pd.to_numeric(part, errors="coerce") for part in parts if part is not None]
    if not valid_parts:
        return pd.Series(fill_value, index=index, dtype=float)
    frame = pd.concat(valid_parts, axis=1)
    out = frame.mean(axis=1, skipna=True)
    if pd.isna(fill_value):
        return out
    return out.fillna(fill_value)


def _log_tech_feature_availability(base: pd.DataFrame) -> None:
    feature_cols = [
        "composite",
        "score_score",
        "close",
        "ma_5",
        "ma_20",
        "ma_60",
        "close_over_ma20",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
        "vol_20",
        "vol_60",
        "vol_ma_20",
        "volume",
        "vol_ratio_20",
    ]
    availability = {}
    for col in feature_cols:
        if col not in base.columns:
            availability[col] = "missing"
            continue
        nonnull = int(base[col].notna().sum())
        availability[col] = f"{nonnull}/{len(base)}"
    logging.info("Tech feature availability: %s", availability)


def _is_usable_legacy_tech_source(base: pd.DataFrame, col: str) -> bool:
    if col not in base.columns:
        return False
    series = pd.to_numeric(base[col], errors="coerce")
    nonnull_ratio = float(series.notna().mean()) if len(series) else 0.0
    unique_count = int(series.dropna().nunique())
    usable = nonnull_ratio >= 0.80 and unique_count >= 10
    logging.info(
        "Legacy tech source check: col=%s nonnull_ratio=%.3f unique_count=%d usable=%s",
        col,
        nonnull_ratio,
        unique_count,
        usable,
    )
    return usable


def _compute_feature_based_tech_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    index = base.index

    for col in ["close", "ma_5", "ma_20", "ma_60", "rsi_14", "vol_ratio_20", "vol_ma_20", "volume"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if "rsi_14" in base.columns:
        base["rsi_14"] = base["rsi_14"].clip(lower=0.0, upper=100.0)
    if "vol_ratio_20" in base.columns:
        base["vol_ratio_20"] = base["vol_ratio_20"].clip(lower=0.0, upper=3.0)

    trend_position = None
    if "close_over_ma20" in base.columns:
        base["close_over_ma20_win"] = _winsorize_by_date(base, "close_over_ma20")
        trend_position = _percentile_by_date(base, "close_over_ma20_win")

    ma_ready = all(col in base.columns for col in ["close", "ma_5", "ma_20", "ma_60"])
    if ma_ready:
        cond_100 = (base["close"] >= base["ma_5"]) & (base["ma_5"] >= base["ma_20"]) & (base["ma_20"] >= base["ma_60"])
        cond_75 = (base["close"] >= base["ma_5"]) & (base["ma_5"] >= base["ma_20"])
        cond_55 = base["close"] >= base["ma_20"]
        cond_35 = (base["close"] < base["ma_20"]) & (base["ma_20"] >= base["ma_60"])
        base["tech_trend_alignment"] = np.select(
            [cond_100, cond_75, cond_55, cond_35],
            [100.0, 75.0, 55.0, 35.0],
            default=15.0,
        )
    else:
        base["tech_trend_alignment"] = 50.0

    base["tech_trend_score"] = _average_available(
        [trend_position, base["tech_trend_alignment"]],
        index,
        fill_value=50.0,
    )

    momentum_parts = []
    for src_col, out_col in [("ret_5d", "tech_ret_5d_win"), ("ret_10d", "tech_ret_10d_win"), ("mom_20", "tech_mom_20_win")]:
        if src_col in base.columns:
            base[out_col] = _winsorize_by_date(base, src_col)
            momentum_parts.append(_percentile_by_date(base, out_col))
    if "rsi_14" in base.columns:
        base["tech_rsi_score"] = (100.0 - ((base["rsi_14"] - 60.0).abs() * 2.0)).clip(lower=0.0, upper=100.0)
    else:
        base["tech_rsi_score"] = 50.0
    momentum_parts.append(base["tech_rsi_score"])
    base["tech_momentum_score"] = _average_available(momentum_parts, index, fill_value=50.0)

    stability_parts = []
    if "vol_20" in base.columns:
        base["tech_vol_20_win"] = _winsorize_by_date(base, "vol_20")
        stability_parts.append(100.0 - _percentile_by_date(base, "tech_vol_20_win"))
    if "vol_60" in base.columns:
        base["tech_vol_60_win"] = _winsorize_by_date(base, "vol_60")
        stability_parts.append(100.0 - _percentile_by_date(base, "tech_vol_60_win"))
    base["tech_stability_score"] = _average_available(stability_parts, index, fill_value=50.0)

    volume_parts = []
    if "vol_ma_20" in base.columns:
        base["tech_vol_ma_20_win"] = _winsorize_by_date(base, "vol_ma_20")
        volume_parts.append(_percentile_by_date(base, "tech_vol_ma_20_win"))
    if "vol_ratio_20" in base.columns:
        volume_parts.append(_percentile_by_date(base, "vol_ratio_20"))
    if "volume" in base.columns:
        base["tech_volume_win"] = _winsorize_by_date(base, "volume")
        volume_parts.append(_percentile_by_date(base, "tech_volume_win"))
    base["tech_volume_score"] = _average_available(volume_parts, index, fill_value=50.0)

    if "vol_ma_20" in base.columns:
        base["tech_liquidity_pct"] = _percentile_by_date(base, "tech_vol_ma_20_win" if "tech_vol_ma_20_win" in base.columns else "vol_ma_20")
    elif "volume" in base.columns:
        base["tech_liquidity_pct"] = _percentile_by_date(base, "tech_volume_win" if "tech_volume_win" in base.columns else "volume")
    else:
        base["tech_liquidity_pct"] = 50.0

    base["tech_liquidity_guard"] = np.select(
        [base["tech_liquidity_pct"] < 10.0, base["tech_liquidity_pct"] < 20.0],
        [0.72, 0.90],
        default=1.00,
    )

    base["tech_score"] = (
        0.30 * base["tech_trend_score"].fillna(50.0)
        + 0.35 * base["tech_momentum_score"].fillna(50.0)
        + 0.20 * base["tech_stability_score"].fillna(50.0)
        + 0.15 * base["tech_volume_score"].fillna(50.0)
    )
    base["tech_score"] = (
        base["tech_score"].fillna(50.0) * pd.to_numeric(base["tech_liquidity_guard"], errors="coerce").fillna(1.0)
    ).clip(lower=0.0, upper=100.0)
    base["tech_source"] = "feature_v1"

    logging.info(
        "Feature-based tech score built: rows=%d trend_nonnull=%d momentum_nonnull=%d stability_nonnull=%d volume_nonnull=%d tech_nonnull=%d",
        len(base),
        int(base["tech_trend_score"].notna().sum()),
        int(base["tech_momentum_score"].notna().sum()),
        int(base["tech_stability_score"].notna().sum()),
        int(base["tech_volume_score"].notna().sum()),
        int(base["tech_score"].notna().sum()),
    )
    return base


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _normalize_code_columns(*dfs: pd.DataFrame) -> None:
    for df in dfs:
        if df is not None and not df.empty and "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)


def _load_market_status() -> tuple[bool, dict, pd.DataFrame]:
    if not MARKET_STATUS_CSV.exists():
        info = {
            "fallback_reason": "market_status_missing",
            "fallback_mode": "conservative_market_up_false",
        }
        logging.warning(
            "market_status.csv not found; applying conservative fallback "
            "(market_up=False, reason=%s)",
            info["fallback_reason"],
        )
        return False, info, pd.DataFrame()

    try:
        df = pd.read_csv(MARKET_STATUS_CSV)
    except Exception:
        info = {
            "fallback_reason": "market_status_read_failed",
            "fallback_mode": "conservative_market_up_false",
        }
        logging.exception(
            "Failed to read market_status.csv; applying conservative fallback "
            "(market_up=False, reason=%s)",
            info["fallback_reason"],
        )
        return False, info, pd.DataFrame()

    if df.empty or "market_up" not in df.columns:
        info = {
            "fallback_reason": "market_status_empty_or_missing_market_up",
            "fallback_mode": "conservative_market_up_false",
        }
        logging.warning(
            "market_status.csv is empty or missing market_up; applying conservative fallback "
            "(market_up=False, reason=%s)",
            info["fallback_reason"],
        )
        return False, info, df

    last = df.iloc[-1]
    raw = last["market_up"]
    if isinstance(raw, bool):
        market_up = raw
    else:
        market_up = str(raw).strip().lower() in {"true", "1", "t", "y", "yes"}

    info = {}
    for col in ["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d"]:
        if col in last.index:
            info[col] = last[col]

    logging.info(
        "Loaded market status: market_up=%s, info=%s",
        market_up,
        {k: info.get(k) for k in ["date", "kospi_close", "kospi_ma20", "volatility_5d", "foreign_net_5d"]},
    )
    return market_up, info, df


def detect_market_regime(df: pd.DataFrame, market_info: dict, market_history: pd.DataFrame | None = None) -> tuple[str, str]:
    regime, regime_reason = shared_detect_market_regime(df, market_info, market_history)
    logging.info("Detected market regime=%s reason=%s", regime, regime_reason)
    return regime, regime_reason


def _attach_market_columns(
    df: pd.DataFrame,
    market_up: bool,
    market_info: dict,
    market_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return shared_attach_market_columns(
        df,
        market_up=market_up,
        market_info=market_info,
        market_history=market_history,
        log_distribution=True,
        log_prefix="_attach_market_columns",
    )


def _ensure_regime_column(
    df: pd.DataFrame,
    *,
    log_distribution: bool = False,
    log_prefix: str = "regime",
) -> pd.DataFrame:
    return shared_ensure_regime_column(df, log_distribution=log_distribution, log_prefix=log_prefix)


def _compute_tech_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    tech_score
    - Input columns: composite or score_score from scores_final.csv
    - Purpose: reflect chart / technical composite strength
    - Interpretation: higher means stronger technical profile on that date
    - Score type: relative score (per-date percentile, 0~100)
    """
    base = base.copy()
    _log_tech_feature_availability(base)
    if _is_usable_legacy_tech_source(base, "composite"):
        base["tech_score"] = _percentile_by_date(base, "composite")
        base["tech_source"] = "scores_final.composite"
    elif _is_usable_legacy_tech_source(base, "score_score"):
        base["tech_score"] = _percentile_by_date(base, "score_score")
        base["tech_source"] = "scores_final.score_score"
    else:
        logging.warning("Legacy technical source unavailable or low-variance; falling back to feature-based tech score.")
        base = _compute_feature_based_tech_score(base)
        return base

    for col in [
        "tech_trend_score",
        "tech_momentum_score",
        "tech_stability_score",
        "tech_volume_score",
    ]:
        if col not in base.columns:
            base[col] = np.nan
    if "tech_liquidity_guard" not in base.columns:
        base["tech_liquidity_guard"] = 1.0
    return base


def _compute_ret_and_pred_scores(base: pd.DataFrame) -> pd.DataFrame:
    """
    ret_score
    - Input columns: pred_return_60d, pred_return_90d
    - Purpose: represent the primary production prediction axis for final_score
    - Interpretation: higher means stronger predicted return rank on that date
    - Score type: relative score (per-date percentile blend, 0~100)
    - Definition: a 60d/90d predicted-return blend used as the operating prediction score

    pred_score
    - Input columns: pred_return_60d, pred_return_90d
    - Purpose: retain a legacy model percentile reference for research comparison
    - Interpretation: higher means better model return rank
    - Score type: relative score (per-date percentile blend, 0~100)
    - Usage: research only, not the primary production prediction score
    """
    base = base.copy()
    pred_60 = None
    pred_90 = None

    if "pred_return_60d" in base.columns:
        base["pred_score_60"] = _percentile_by_date(base, "pred_return_60d")
        pred_60 = base["pred_score_60"]
        base["ret_rank_60"] = _percentile01_by_date(base, "pred_return_60d")
        base["pred_return_60d_pct01"] = base["ret_rank_60"]
    else:
        base["ret_rank_60"] = np.nan
        base["pred_return_60d_pct01"] = np.nan

    if "pred_return_90d" in base.columns:
        base["pred_score_90"] = _percentile_by_date(base, "pred_return_90d")
        pred_90 = base["pred_score_90"]
        base["ret_rank_90"] = _percentile01_by_date(base, "pred_return_90d")
        base["pred_return_90d_pct01"] = base["ret_rank_90"]
    else:
        base["ret_rank_90"] = np.nan
        base["pred_return_90d_pct01"] = np.nan

    base["ret_score"] = 100.0 * (
        0.7 * base["ret_rank_60"].fillna(0)
        + 0.3 * base["ret_rank_90"].fillna(0)
    )
    base["ret_score_v11"] = base["ret_score"]

    if (pred_60 is not None) and (pred_90 is not None):
        base["pred_score"] = 0.6 * pred_60 + 0.4 * pred_90
    elif pred_60 is not None:
        base["pred_score"] = pred_60
    elif pred_90 is not None:
        base["pred_score"] = pred_90
    else:
        logging.warning("No 'pred_return_60d' or 'pred_return_90d' columns; pred_score will be NaN.")
        base["pred_score"] = np.nan

    return base


def _compute_prob_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    prob_score_raw / prob_score
    - Operating input column: prob_top20_60d
    - Research / stored auxiliary column: prob_top20_90d (not blended here)
    - Purpose:
      prob_score_raw preserves the absolute probability conversion, while
      prob_score is the operating same-date relative score used by final_score.
    - Interpretation:
      prob_score_raw -> raw 60d probability converted to 0~100
      prob_score     -> same-date percentile rank converted to 0~100
    - Missing handling:
      keep prob_score_missing for observability, then apply a neutral same-date
      fallback instead of forcing missing names to 0.
    - Tie handling:
      percentile rank uses pandas rank(method="average"), so tied names receive
      the same averaged percentile within a date slice.
    - Policy:
      the operating probability axis intentionally uses the 60d horizon only.
      prob_top20_90d is preserved for storage, diagnostics, and research, but
      it does not change the production operating prob_score.
    """
    base = base.copy()
    if "prob_top20_60d" in base.columns:
        prob = pd.to_numeric(base["prob_top20_60d"], errors="coerce")
        base["prob_score_missing"] = prob.isna()
        base["prob_score_raw"] = _clip01(prob.fillna(0.0) * 100.0, 0.0, 100.0)
        base["prob_rank_pct"] = _percentile01_by_date(base.assign(prob_top20_60d=prob), "prob_top20_60d")
        base["prob_rank_pct"] = (
            base.groupby("date", group_keys=False)["prob_rank_pct"]
            .transform(lambda s: s.fillna(s.median()))
        )
        base["prob_rank_pct"] = pd.to_numeric(base["prob_rank_pct"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
        base["prob_score"] = (base["prob_rank_pct"] * 100.0).clip(lower=0.0, upper=100.0)
    else:
        logging.warning("'prob_top20_60d' column not found; using neutral fallback for probability scores.")
        base["prob_score_missing"] = True
        base["prob_score_raw"] = np.nan
        base["prob_rank_pct"] = 0.5
        base["prob_score"] = 50.0
    return base


def _compute_qual_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    qual_score
    - Input columns: quality_score
    - Purpose: reflect cross-sectional business / fundamental quality
    - Interpretation: higher means stronger quality rank on that date
    - Score type: relative score (per-date percentile, 0~100)
    """
    base = base.copy()
    if "quality_score" in base.columns:
        base["qual_score"] = _percentile_by_date(base, "quality_score")
    else:
        logging.warning("'quality_score' column not found; qual_score will be NaN.")
        base["qual_score"] = np.nan
    return base


def _compute_safety_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    safety_score
    - Input columns: vol_20, vol_60
    - Purpose: reward lower-volatility names
    - Interpretation: higher means relatively safer / lower volatility on that date
    - Score type: relative score (inverse percentile average, 0~100)
    """
    base = base.copy()
    safety_parts = []
    if "vol_20" in base.columns:
        base["vol_20_pct"] = _percentile_by_date(base, "vol_20")
        safety_parts.append(100.0 - base["vol_20_pct"])
    if "vol_60" in base.columns:
        base["vol_60_pct"] = _percentile_by_date(base, "vol_60")
        safety_parts.append(100.0 - base["vol_60_pct"])

    if safety_parts:
        base["safety_score"] = sum(safety_parts) / len(safety_parts)
    else:
        logging.info("No vol_20 / vol_60 columns; safety_score will be NaN.")
        base["safety_score"] = np.nan
    return base


def _compute_liquidity_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    liquidity_score
    - Input columns: vol_ma_20 or volume
    - Purpose: favor names with stronger trading liquidity
    - Interpretation: higher means more liquid on that date
    - Score type: relative score (per-date percentile, 0~100)
    """
    base = base.copy()
    if "vol_ma_20" in base.columns:
        base["liquidity_score"] = _percentile_by_date(base, "vol_ma_20")
    elif "volume" in base.columns:
        base["liquidity_score"] = _percentile_by_date(base, "volume")
    else:
        logging.info("No vol_ma_20 / volume columns; liquidity_score will be NaN.")
        base["liquidity_score"] = np.nan
    return base


def _compute_valuation_score(base: pd.DataFrame) -> pd.DataFrame:
    """
    valuation_score
    - Purpose: retain a valuation-flavored diagnostic / compatibility column.
      It is not a direct axis in the current production operating final_score.
    - Preferred inputs: explicit valuation metrics if they exist in features.
    - Current fallback: if no valuation source exists, use a neutral 50.0 so
      diagnostics and sidecar experiments do not hallucinate valuation edge
      from unrelated columns.
    """
    base = base.copy()
    valuation_candidates: list[pd.Series] = []
    lower_is_better = [
        "per",
        "pe",
        "pbr",
        "pb",
        "psr",
        "ps",
        "ev_ebitda",
        "ev_to_ebitda",
        "price_to_book",
        "price_to_sales",
        "price_to_earnings",
    ]
    higher_is_better = [
        "earnings_yield",
        "book_to_price",
        "free_cash_flow_yield",
        "dividend_yield",
    ]

    for col in lower_is_better:
        if col in base.columns:
            score = 100.0 - _percentile_by_date(base, col)
            valuation_candidates.append(score)
    for col in higher_is_better:
        if col in base.columns:
            score = _percentile_by_date(base, col)
            valuation_candidates.append(score)

    if valuation_candidates:
        base["valuation_score"] = sum(valuation_candidates) / len(valuation_candidates)
    else:
        logging.warning("No valuation input columns found; valuation_score falls back to neutral 50.0.")
        base["valuation_score"] = 50.0
    return base


def _attach_operational_score_aliases(base: pd.DataFrame) -> pd.DataFrame:
    return shared_attach_operational_score_aliases(base)


def compute_component_scores(base: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all component scores used by the final ranking score.

    This is the main score-construction block inside ranking_builder.py.
    ret_score is the primary production prediction score used by final_score.
    prob_score is the operating same-date relative probability score used by final_score.
    prob_score_raw is kept only as the raw probability conversion reference.
    pred_score is preserved only as a legacy / research comparison field.
    The production final score is driven by return / probability / technical /
    quality, with risk behavior expressed through risk_penalty instead of a
    direct safety bonus. valuation_score is retained as a compatibility /
    diagnostic column and is not a direct operating axis.
    """
    base = shared_compute_component_scores(base.copy())

    tech = pd.to_numeric(base["tech_score"], errors="coerce")
    diag = {
        "rows": int(len(base)),
        "nonnull": int(tech.notna().sum()),
        "unique": int(tech.dropna().nunique()),
        "mean": float(tech.mean()) if tech.notna().any() else None,
        "std": float(tech.std(ddof=0)) if tech.notna().any() else None,
        "source_counts": base["tech_source"].fillna("NA").value_counts(dropna=False).to_dict() if "tech_source" in base.columns else {},
        "guard_lt_1_count": int((pd.to_numeric(base.get("tech_liquidity_guard"), errors="coerce").fillna(1.0) < 1.0).sum()) if "tech_liquidity_guard" in base.columns else 0,
    }
    logging.info("Tech score diagnostics: %s", diag)
    if diag["unique"] <= 3:
        logging.warning("tech_score has very low variance (unique=%d); technical signal may be ineffective", diag["unique"])

    return base


def baseline_risk_penalty_from_mix(mix_like, penalty_cap: float = PENALTY_CAP_DEFAULT) -> pd.Series:
    mix = pd.to_numeric(pd.Series(mix_like), errors="coerce").fillna(0.0).abs()
    penalty = np.select(
        [mix <= 0.10, mix <= 0.15, mix <= 0.20],
        [0.0, (mix - 0.10) * 40.0, 2.0 + (mix - 0.15) * 80.0],
        default=6.0 + (mix - 0.20) * 120.0,
    )
    return pd.to_numeric(pd.Series(penalty, index=mix.index), errors="coerce").fillna(0.0).clip(lower=0.0, upper=penalty_cap)


def _compute_risk_penalty(base: pd.DataFrame) -> pd.DataFrame:
    return shared_compute_risk_penalty(base)


def _resolve_component_weights(base: pd.DataFrame) -> pd.DataFrame:
    base = resolve_core_weight_profile(base.copy())
    weight_meta = _resolve_theme_weight_metadata_frame(base)
    configured_theme_weight = pd.to_numeric(weight_meta["theme_weight"], errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
    if not _theme_gate_allows_score_application():
        configured_theme_weight = pd.Series(0.0, index=base.index, dtype="float64")
    configured_base_weight = 1.0 - configured_theme_weight
    has_theme_score = pd.to_numeric(base.get("theme_score"), errors="coerce").fillna(0.0).ne(0.0)
    base["theme_weight"] = configured_theme_weight
    base["weight_source"] = weight_meta["weight_source"].astype(str)
    base["regime_applied"] = weight_meta["regime_applied"].astype(str)
    base["w_theme"] = np.where(has_theme_score, configured_theme_weight, 0.0)
    base["w_base_v2"] = np.where(has_theme_score, configured_base_weight, 1.0)
    return base


def _extract_factor_extremes(row: pd.Series) -> pd.Series:
    factor_items = []
    for col in DISPLAY_FACTOR_LABELS:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            factor_items.append((col, float(value)))

    if not factor_items:
        return pd.Series(
            {
                "top_positive_factor": None,
                "top_positive_value": np.nan,
                "top_negative_factor": None,
                "top_negative_value": np.nan,
            }
        )

    positive_items = [(key, value) for key, value in factor_items if value > 0]
    negative_items = [(key, value) for key, value in factor_items if value < 0]

    top_positive = max(positive_items, key=lambda item: item[1]) if positive_items else (None, np.nan)
    top_negative = min(negative_items, key=lambda item: item[1]) if negative_items else (None, np.nan)

    return pd.Series(
        {
            "top_positive_factor": DISPLAY_FACTOR_LABELS.get(top_positive[0]) if top_positive[0] else None,
            "top_positive_value": top_positive[1],
            "top_negative_factor": DISPLAY_FACTOR_LABELS.get(top_negative[0]) if top_negative[0] else None,
            "top_negative_value": top_negative[1],
        }
    )


def _attach_component_integrity_flags(base: pd.DataFrame) -> pd.DataFrame:
    """
    Track missingness and fallback usage before score columns are filled for ranking.

    Missing and low scores must remain distinguishable. This helper is called
    before final_score construction so confidence_score can reflect evidence
    quality rather than the post-fill numeric score surface.
    """
    base = base.copy()
    for col in CORE_COMPONENT_COLUMNS:
        missing_col = f"{col}_missing"
        if col == "prob_score" and "prob_score_missing" in base.columns:
            base[missing_col] = base["prob_score_missing"].fillna(True).astype(bool)
        else:
            source = base[col] if col in base.columns else pd.Series(index=base.index, dtype="float64")
            base[missing_col] = pd.to_numeric(source, errors="coerce").isna()

    fallback_map = {
        "ret_score_fallback_used": "ret_score_missing",
        "prob_score_fallback_used": "prob_score_missing",
        "qual_score_fallback_used": "qual_score_missing",
        "tech_score_fallback_used": "tech_score_missing",
        "safety_score_fallback_used": "safety_score_missing",
        "liquidity_score_fallback_used": "liquidity_score_missing",
    }
    for fallback_col, missing_col in fallback_map.items():
        base[fallback_col] = base.get(missing_col, False)

    fallback_cols = list(fallback_map.keys())
    base["fallback_count"] = (
        pd.DataFrame({col: base[col].fillna(False).astype(bool) for col in fallback_cols}, index=base.index)
        .sum(axis=1)
        .astype(int)
    )
    return base


def _score_band_text(value: float) -> str:
    if pd.isna(value):
        return "not available"
    if value >= 80:
        return "very strong"
    if value >= 60:
        return "strong"
    if value >= 40:
        return "neutral"
    return "weak"


def _format_score_or_na(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _resolve_dominant_driver(row: pd.Series) -> tuple[str, str, float]:
    driver_candidates = [
        ("ret", "Primary prediction score", pd.to_numeric(row.get("contrib_ret"), errors="coerce")),
        ("prob", "Top-bucket probability score", pd.to_numeric(row.get("contrib_prob"), errors="coerce")),
        ("tech", "Tech flow score", pd.to_numeric(row.get("contrib_tech"), errors="coerce")),
    ]
    valid = [(code, label, float(value)) for code, label, value in driver_candidates if pd.notna(value)]
    if not valid:
        return "NA", "NA", float("nan")
    return max(valid, key=lambda item: item[2])


def _collect_key_driver_lines(row: pd.Series) -> list[str]:
    items = [
        ("ret", "ret_score", "Primary prediction score", pd.to_numeric(row.get("contrib_ret"), errors="coerce"), pd.to_numeric(row.get("ret_score"), errors="coerce")),
        ("prob", "prob_score", "Top-bucket probability score", pd.to_numeric(row.get("contrib_prob"), errors="coerce"), pd.to_numeric(row.get("prob_score"), errors="coerce")),
        ("tech", "tech_score", "Tech flow score", pd.to_numeric(row.get("contrib_tech"), errors="coerce"), pd.to_numeric(row.get("tech_score"), errors="coerce")),
        ("qual", "qual_score", "Financial quality score", pd.to_numeric(row.get("contrib_qual"), errors="coerce"), pd.to_numeric(row.get("qual_score"), errors="coerce")),
    ]
    ranked = [
        (code, field_name, label, float(contrib), raw_score)
        for code, field_name, label, contrib, raw_score in items
        if pd.notna(contrib) and float(contrib) > 0.0
    ]
    ranked.sort(key=lambda item: item[3], reverse=True)
    return [
        f"{label} ({field_name}) contribution {contrib:.1f}, score {_format_score_or_na(raw_score)}"
        for _, field_name, label, contrib, raw_score in ranked[:3]
    ]


def _collect_risk_lines(row: pd.Series) -> list[str]:
    risk_lines: list[str] = []
    top_negative_factor = row.get("top_negative_factor")
    top_negative_value = pd.to_numeric(row.get("top_negative_value"), errors="coerce")
    quality_gate_applied = row.get("quality_gate_applied")
    penalty_ratio = pd.to_numeric(row.get("quality_penalty_ratio"), errors="coerce")

    if top_negative_factor and pd.notna(top_negative_value):
        risk_lines.append(f"top_negative_factor={top_negative_factor} ({top_negative_value:.1f})")
    if bool(quality_gate_applied):
        risk_lines.append(f"quality_penalty applied (ratio={_format_score_or_na(penalty_ratio, digits=2)})")

    risk_penalty = pd.to_numeric(row.get("risk_penalty"), errors="coerce")
    qual_score = pd.to_numeric(row.get("qual_score"), errors="coerce")
    fallback_count = pd.to_numeric(row.get("fallback_count"), errors="coerce")

    if pd.notna(risk_penalty):
        risk_lines.append(f"risk_penalty {_format_score_or_na(risk_penalty)}")
    if pd.notna(qual_score) and float(qual_score) < 40.0:
        risk_lines.append(f"qual_score low ({qual_score:.1f})")
    if pd.notna(fallback_count) and float(fallback_count) > 0.0:
        risk_lines.append(f"fallback_count {int(fallback_count)}")

    deduped: list[str] = []
    for item in risk_lines:
        if item not in deduped:
            deduped.append(item)
    return deduped[:2]


def _build_quality_gate_explain(row: pd.Series) -> str:
    regime = str(row.get("regime") or "neutral").strip().lower()
    experiment = str(row.get("quality_gate_experiment") or QUALITY_GATE_EXPERIMENT).strip().lower()
    qual_score = pd.to_numeric(row.get("qual_score"), errors="coerce")
    penalty_ratio = pd.to_numeric(row.get("quality_penalty_ratio"), errors="coerce")
    quality_flag = bool(row.get("quality_flag"))
    quality_gate_applied = bool(row.get("quality_gate_applied"))

    is_gate_candidate = regime == "defensive" and pd.notna(qual_score) and float(qual_score) < 40.0
    alpha_protected = is_gate_candidate and should_protect_alpha(row) and not quality_gate_applied

    if quality_gate_applied:
        return f"quality downside reflected (penalty_ratio={_format_score_or_na(penalty_ratio, digits=2)})"
    if alpha_protected and experiment == "v2":
        return "quality penalty skipped: alpha protection triggered (prob/ret/confidence threshold met)"
    if experiment == "v3" and quality_flag:
        return "quality gate warning only: flag recorded, score unchanged (v3)"
    return ""


def _build_explain_text(row: pd.Series) -> str:
    live_score = pd.to_numeric(row.get("live_score"), errors="coerce")
    final_score = live_score if pd.notna(live_score) else pd.to_numeric(row.get("final_score"), errors="coerce")
    live_score_source = str(row.get("live_score_source") or "final_score").strip()
    regime = str(row.get("regime") or "defensive")
    top_positive_factor = row.get("top_positive_factor")
    top_positive_value = pd.to_numeric(row.get("top_positive_value"), errors="coerce")
    top_negative_factor = row.get("top_negative_factor")
    top_negative_value = pd.to_numeric(row.get("top_negative_value"), errors="coerce")
    ret_score = pd.to_numeric(row.get("ret_score"), errors="coerce")
    prob_score = pd.to_numeric(row.get("prob_score"), errors="coerce")
    qual_score = pd.to_numeric(row.get("qual_score"), errors="coerce")
    tech_score = pd.to_numeric(row.get("tech_score"), errors="coerce")
    pred_score = pd.to_numeric(row.get("pred_score"), errors="coerce")
    theme_score = pd.to_numeric(row.get("theme_score"), errors="coerce")
    theme_confidence = pd.to_numeric(row.get("theme_confidence"), errors="coerce")
    dominant_theme = str(row.get("dominant_theme") or "").strip()
    confidence_score = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    confidence_label = str(row.get("confidence_label") or "Experimental").strip()
    confidence_reason = str(row.get("confidence_reason") or "").strip()

    dominant_driver_code, dominant_driver_label, dominant_driver_value = _resolve_dominant_driver(row)
    key_driver_lines = _collect_key_driver_lines(row)
    risk_lines = _collect_risk_lines(row)
    quality_gate_note = _build_quality_gate_explain(row)

    theme_flags = _resolve_theme_overlay_runtime_flags()
    if bool(theme_flags.get("live_uses_theme", False)):
        theme_status = (
            "theme_overlay: enabled (operational config)"
            f", dominant_theme={dominant_theme if _is_active_theme_label(dominant_theme) else '(none)'}"
            f", theme_score_effective={_format_score_or_na(row.get('theme_score_effective'))}"
        )
    else:
        theme_status = "theme_overlay: disabled (operational config)"

    confidence_status = (
        "confidence: 모델 메타 점수 (calibration 미완)"
        f", score={_format_score_or_na(confidence_score)}"
        f", label={confidence_label or 'Experimental'}"
        ", live_score 비반영"
        ", research/reference 용도"
    )
    if confidence_reason:
        confidence_status += f", reason={confidence_reason}"

    summary_bits = [
        f"요약: live_score {_format_score_or_na(final_score)}",
        f"live_score_source={live_score_source}",
        f"regime={regime}",
        f"dominant_driver={dominant_driver_code} ({dominant_driver_label} { _format_score_or_na(dominant_driver_value) })",
    ]
    if top_positive_factor and pd.notna(top_positive_value):
        summary_bits.append(f"top_positive_factor={top_positive_factor} ({top_positive_value:.1f})")
    if top_negative_factor and pd.notna(top_negative_value):
        summary_bits.append(f"top_negative_factor={top_negative_factor} ({top_negative_value:.1f})")

    if not key_driver_lines:
        key_driver_lines = ["actual positive score contribution not available"]
    if not risk_lines:
        risk_lines = ["top_negative_factor not available"]
    if quality_gate_note:
        risk_lines = [quality_gate_note] + [item for item in risk_lines if item != quality_gate_note]

    lines = [
        " | ".join(summary_bits),
        "핵심 driver: " + "; ".join(key_driver_lines[:3]),
        "리스크 요인: " + "; ".join(risk_lines[:2]),
        "운영 상태: " + theme_status + "; " + confidence_status,
    ]
    return "\n".join(lines)


def build_theme_explain(row: pd.Series) -> str:
    dominant_theme = str(row.get("dominant_theme") or "").strip()
    if not _is_active_theme_label(dominant_theme):
        return ""
    theme_score = pd.to_numeric(row.get("theme_score"), errors="coerce")
    theme_confidence = pd.to_numeric(row.get("theme_confidence"), errors="coerce")
    if pd.isna(theme_score):
        theme_score = 0.0
    if pd.isna(theme_confidence):
        theme_confidence = 0.0
    return (
        f"theme={dominant_theme}, theme_score={theme_score:.1f}, "
        f"theme_confidence={theme_confidence:.2f}"
    )


def _build_explain_json(row: pd.Series) -> str:
    base_score = pd.to_numeric(row.get("final_score"), errors="coerce")
    theme_score = pd.to_numeric(row.get("theme_score"), errors="coerce")
    theme_confidence = pd.to_numeric(row.get("theme_confidence"), errors="coerce")
    final_score_v2 = pd.to_numeric(row.get("final_score_v2"), errors="coerce")
    theme_weight = pd.to_numeric(row.get("theme_weight"), errors="coerce")
    weight_source = str(row.get("weight_source") or "").strip()
    regime_applied = str(row.get("regime_applied") or "").strip()
    dominant_theme = str(row.get("dominant_theme") or "").strip()

    if pd.isna(base_score):
        base_score = 0.0
    if pd.isna(theme_score):
        theme_score = 0.0
    if pd.isna(theme_confidence):
        theme_confidence = 0.0
    if pd.isna(theme_weight):
        weight_info = _resolve_theme_weight_info_for_regime(str(row.get("regime") or ""))
        theme_weight = float(weight_info.get("theme_weight", THEME_V2_THEME_WEIGHT))
        if not weight_source:
            weight_source = str(weight_info.get("weight_source") or "")
        if not regime_applied:
            regime_applied = str(weight_info.get("regime_applied") or "")
    if pd.isna(final_score_v2):
        final_score_v2 = (1.0 - float(theme_weight)) * float(base_score) + float(theme_weight) * float(theme_score)

    payload = {
        "base_score": round(float(base_score), 1),
        "theme_weight": round(float(theme_weight), 2),
        "weight_source": weight_source or "fallback_default",
        "regime_applied": regime_applied or str(row.get("regime") or "global"),
        "theme": {
            "name": dominant_theme if _is_active_theme_label(dominant_theme) else "(none)",
            "confidence": round(float(theme_confidence), 2),
            "contribution": round(float(final_score_v2) - float(base_score), 1),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def apply_theme_overlay_v2(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["final_score_v2_before_theme"] = pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
    weight_meta = _resolve_theme_weight_metadata_frame(out)
    out["theme_weight"] = pd.to_numeric(weight_meta["theme_weight"], errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
    out["weight_source"] = weight_meta["weight_source"].astype(str)
    out["regime_applied"] = weight_meta["regime_applied"].astype(str)
    if not _theme_gate_allows_score_application():
        out["theme_weight"] = 0.0
    base_weight = 1.0 - pd.to_numeric(out.get("theme_weight"), errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
    out["final_score_v2"] = (
        base_weight * out["final_score_v2_before_theme"]
        + pd.to_numeric(out.get("theme_weight"), errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
        * pd.to_numeric(out.get("theme_score"), errors="coerce").fillna(0.0)
    )
    out["score_diff_v2"] = out["final_score_v2"] - pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
    return out


def compute_theme_overlay_anchor(base: pd.DataFrame, anchor_name: str) -> pd.Series:
    anchor_key = str(anchor_name or "baseline_score").strip().lower()
    if anchor_key in {"baseline_score", "final_score"}:
        return pd.to_numeric(base.get("final_score"), errors="coerce").fillna(0.0)
    logging.warning("Unsupported shadow baseline anchor=%r; falling back to final_score", anchor_name)
    return pd.to_numeric(base.get("final_score"), errors="coerce").fillna(0.0)


def compute_symmetric_floor_overlay(
    anchor: pd.Series,
    theme_score_effective: pd.Series,
    theme_weight_effective: pd.Series,
) -> dict[str, pd.Series]:
    base_weight = 1.0 - theme_weight_effective
    final_score = base_weight * anchor + theme_weight_effective * theme_score_effective
    signed_component = final_score - anchor
    return {
        "final_score": final_score,
        "signed_component": signed_component,
        "positive_component": signed_component.clip(lower=0.0),
        "negative_component": signed_component.clip(upper=0.0),
        "applied_component": signed_component,
        "capped": pd.Series(False, index=anchor.index, dtype="boolean"),
        "soft_conf_gate": pd.Series(1.0, index=anchor.index, dtype="float64"),
    }


def compute_asymmetric_positive_only_overlay(
    anchor: pd.Series,
    theme_delta_positive: pd.Series,
    gain: float,
    cap: float | None = None,
    soft_conf_gate: pd.Series | None = None,
) -> dict[str, pd.Series]:
    overlay_raw = pd.to_numeric(theme_delta_positive, errors="coerce").fillna(0.0) * float(max(gain, 0.0))
    capped_mask = pd.Series(False, index=anchor.index, dtype="boolean")
    overlay_capped = overlay_raw
    if cap is not None:
        capped_mask = overlay_raw.gt(float(max(cap, 0.0))).astype("boolean")
        overlay_capped = overlay_raw.clip(upper=float(max(cap, 0.0)))
    gate = pd.Series(1.0, index=anchor.index, dtype="float64") if soft_conf_gate is None else pd.to_numeric(soft_conf_gate, errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    applied_component = overlay_capped * gate
    final_score = anchor + applied_component
    return {
        "final_score": final_score,
        "signed_component": applied_component,
        "positive_component": applied_component.clip(lower=0.0),
        "negative_component": pd.Series(0.0, index=anchor.index, dtype="float64"),
        "applied_component": applied_component,
        "capped": capped_mask,
        "soft_conf_gate": gate,
    }


def build_theme_overlay_debug_fields(
    anchor: pd.Series,
    theme_score_effective: pd.Series,
    theme_weight_raw: pd.Series,
    theme_weight_effective: pd.Series,
    overlay_result: dict[str, pd.Series],
    shadow_formula: str,
    shadow_config: dict[str, object],
) -> dict[str, pd.Series]:
    anchor_name = str(shadow_config.get("baseline_anchor") or "baseline_score")
    gain = float(shadow_config.get("gain", SHADOW_THEME_OVERLAY_GAIN))
    cap = float(shadow_config.get("cap", SHADOW_THEME_OVERLAY_CAP))
    theme_delta_raw = pd.to_numeric(theme_score_effective, errors="coerce").fillna(0.0) - pd.to_numeric(anchor, errors="coerce").fillna(0.0)
    return {
        "theme_overlay_anchor": pd.Series(anchor_name, index=anchor.index, dtype="object"),
        "theme_delta_raw": theme_delta_raw,
        "theme_delta_vs_base": theme_delta_raw,
        "theme_delta_positive": theme_delta_raw.clip(lower=0.0),
        "theme_positive_part": theme_delta_raw.clip(lower=0.0),
        "theme_negative_part": (-theme_delta_raw).clip(lower=0.0),
        "theme_overlay_mode": pd.Series(shadow_formula, index=anchor.index, dtype="object"),
        "theme_overlay_formula": pd.Series(shadow_formula, index=anchor.index, dtype="object"),
        "theme_overlay_gain": pd.Series(gain, index=anchor.index, dtype="float64"),
        "theme_overlay_cap": pd.Series(cap, index=anchor.index, dtype="float64"),
        "theme_overlay_signed_component": pd.to_numeric(overlay_result["signed_component"], errors="coerce").fillna(0.0),
        "theme_overlay_positive_component": pd.to_numeric(overlay_result["positive_component"], errors="coerce").fillna(0.0),
        "theme_overlay_negative_component": pd.to_numeric(overlay_result["negative_component"], errors="coerce").fillna(0.0),
        "theme_overlay_applied": pd.to_numeric(overlay_result["applied_component"], errors="coerce").fillna(0.0),
        "theme_overlay_capped": overlay_result["capped"].fillna(False).astype("boolean"),
        "theme_overlay_soft_conf_gate": pd.to_numeric(overlay_result["soft_conf_gate"], errors="coerce").fillna(1.0),
        "shadow_theme_weight_raw": pd.to_numeric(theme_weight_raw, errors="coerce").fillna(0.0),
        "shadow_theme_weight": pd.to_numeric(theme_weight_effective, errors="coerce").fillna(0.0),
        "shadow_theme_weight_effective": pd.to_numeric(theme_weight_effective, errors="coerce").fillna(0.0),
        "shadow_base_weight": 1.0 - pd.to_numeric(theme_weight_effective, errors="coerce").fillna(0.0),
    }


def _build_shadow_theme_score_frame(base: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_theme_columns(base.copy())
    flags = _resolve_theme_overlay_runtime_flags()
    shadow_score_enabled = bool(flags.get("shadow_score_enabled", False))
    shadow_config = resolve_shadow_theme_overlay_config()
    final_score = pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
    anchor = compute_theme_overlay_anchor(out, str(shadow_config.get("baseline_anchor") or "baseline_score"))
    theme_score_effective = pd.to_numeric(out.get("theme_score_effective"), errors="coerce").fillna(0.0)
    theme_confidence = pd.to_numeric(out.get("theme_confidence"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    dominant_theme = out.get("dominant_theme", "").fillna("").astype(str)
    has_theme_signal = dominant_theme.str.strip().apply(_is_active_theme_label) & theme_score_effective.gt(0.0)
    shadow_theme_weight_floor = float(shadow_config.get("floor", _clip_theme_weight(SHADOW_THEME_WEIGHT_FLOOR)))
    shadow_formula = str(shadow_config.get("mode") or SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR).strip().lower() or SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR

    if not shadow_score_enabled:
        out["theme_overlay_formula"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out["theme_overlay_mode"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out["theme_overlay_anchor"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out["theme_delta_raw"] = np.nan
        out["theme_delta_vs_base"] = np.nan
        out["theme_delta_positive"] = np.nan
        out["theme_positive_part"] = np.nan
        out["theme_negative_part"] = np.nan
        out["theme_overlay_gain"] = np.nan
        out["theme_overlay_cap"] = np.nan
        out["theme_overlay_signed_component"] = np.nan
        out["theme_overlay_positive_component"] = np.nan
        out["theme_overlay_negative_component"] = np.nan
        out["theme_overlay_applied"] = np.nan
        out["theme_overlay_capped"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out["theme_overlay_soft_conf_gate"] = np.nan
        out["theme_uplift_applied"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out["theme_penalty_applied"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out["shadow_theme_weight_raw"] = np.nan
        out["shadow_theme_weight"] = np.nan
        out["shadow_theme_weight_effective"] = np.nan
        out["shadow_base_weight"] = np.nan
        out["shadow_floor_applied"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out["shadow_theme_score_effective"] = np.nan
        out["shadow_final_score_v3"] = np.nan
        out["shadow_score_diff_v3"] = np.nan
        out["shadow_rank_v3"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out

    weight_meta = _resolve_theme_weight_metadata_frame(out)
    configured_theme_weight_raw = pd.to_numeric(weight_meta["theme_weight"], errors="coerce").fillna(THEME_V2_THEME_WEIGHT)
    configured_theme_weight_raw = np.where(has_theme_signal, configured_theme_weight_raw, 0.0)
    configured_theme_weight_raw = pd.Series(configured_theme_weight_raw, index=out.index, dtype="float64")
    configured_theme_weight_effective = np.where(
        has_theme_signal,
        np.maximum(configured_theme_weight_raw, shadow_theme_weight_floor),
        0.0,
    )
    configured_theme_weight_effective = pd.Series(configured_theme_weight_effective, index=out.index, dtype="float64")
    theme_delta_raw = theme_score_effective - anchor
    theme_positive_part = theme_delta_raw.clip(lower=0.0)
    theme_negative_part = (-theme_delta_raw).clip(lower=0.0)
    penalty_ratio = float(shadow_config.get("negative_penalty_ratio", max(SHADOW_THEME_NEGATIVE_PENALTY_RATIO, 0.0)))
    uplift_threshold = float(shadow_config.get("uplift_threshold", max(SHADOW_THEME_UPLIFT_THRESHOLD, 0.0)))
    gain = float(shadow_config.get("gain", SHADOW_THEME_OVERLAY_GAIN))
    cap = float(shadow_config.get("cap", SHADOW_THEME_OVERLAY_CAP))
    soft_conf_gate = 0.5 + 0.5 * theme_confidence if bool(shadow_config.get("soft_conf_enabled", True)) else pd.Series(1.0, index=out.index, dtype="float64")

    if shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY:
        overlay_result = compute_asymmetric_positive_only_overlay(
            anchor=anchor,
            theme_delta_positive=theme_positive_part,
            gain=gain,
        )
    elif shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_CAPPED:
        overlay_result = compute_asymmetric_positive_only_overlay(
            anchor=anchor,
            theme_delta_positive=theme_positive_part,
            gain=gain,
            cap=cap,
        )
    elif shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_SOFT_CONF:
        overlay_result = compute_asymmetric_positive_only_overlay(
            anchor=anchor,
            theme_delta_positive=theme_positive_part,
            gain=gain,
            cap=cap,
            soft_conf_gate=soft_conf_gate,
        )
    elif shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_THRESHOLD:
        threshold_positive = (theme_delta_raw - uplift_threshold).clip(lower=0.0)
        overlay_result = compute_asymmetric_positive_only_overlay(
            anchor=anchor,
            theme_delta_positive=threshold_positive,
            gain=gain,
        )
        theme_positive_part = threshold_positive
    elif shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_SOFT_PENALTY:
        applied_component = configured_theme_weight_effective * theme_positive_part - (configured_theme_weight_effective * penalty_ratio) * theme_negative_part
        overlay_result = {
            "final_score": anchor + applied_component,
            "signed_component": applied_component,
            "positive_component": applied_component.clip(lower=0.0),
            "negative_component": applied_component.clip(upper=0.0),
            "applied_component": applied_component,
            "capped": pd.Series(False, index=out.index, dtype="boolean"),
            "soft_conf_gate": pd.Series(1.0, index=out.index, dtype="float64"),
        }
    elif shadow_formula == SHADOW_THEME_OVERLAY_FORMULA_THRESHOLD:
        threshold_positive = (theme_delta_raw - uplift_threshold).clip(lower=0.0)
        overlay_result = compute_asymmetric_positive_only_overlay(
            anchor=anchor,
            theme_delta_positive=threshold_positive,
            gain=gain,
        )
        theme_positive_part = threshold_positive
    else:
        shadow_formula = SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR
        overlay_result = compute_symmetric_floor_overlay(
            anchor=anchor,
            theme_score_effective=theme_score_effective,
            theme_weight_effective=configured_theme_weight_effective,
        )

    shadow_final_score_v3 = pd.to_numeric(overlay_result["final_score"], errors="coerce").fillna(anchor).clip(lower=0.0, upper=100.0)
    debug_fields = build_theme_overlay_debug_fields(
        anchor=anchor,
        theme_score_effective=theme_score_effective,
        theme_weight_raw=configured_theme_weight_raw,
        theme_weight_effective=configured_theme_weight_effective,
        overlay_result=overlay_result,
        shadow_formula=shadow_formula,
        shadow_config=shadow_config,
    )
    for col, value in debug_fields.items():
        out[col] = value

    out["theme_uplift_applied"] = (has_theme_signal & pd.to_numeric(out["theme_overlay_positive_component"], errors="coerce").fillna(0.0).gt(0.0)).astype("boolean")
    out["theme_penalty_applied"] = (has_theme_signal & pd.to_numeric(out["theme_overlay_negative_component"], errors="coerce").fillna(0.0).lt(0.0)).astype("boolean")
    out["shadow_floor_applied"] = (
        has_theme_signal
        & configured_theme_weight_effective.gt(configured_theme_weight_raw)
    ).astype("boolean")
    out["shadow_theme_score_effective"] = np.where(has_theme_signal, theme_score_effective, 0.0)
    out["shadow_final_score_v3"] = shadow_final_score_v3
    out["shadow_score_diff_v3"] = out["shadow_final_score_v3"] - anchor
    out["shadow_explain"] = np.where(
        has_theme_signal,
        "theme overlay mode="
        + out["theme_overlay_mode"].astype(str)
        + ", positive_delta="
        + pd.to_numeric(out["theme_delta_positive"], errors="coerce").fillna(0.0).round(2).astype(str)
        + ", gain="
        + pd.to_numeric(out["theme_overlay_gain"], errors="coerce").fillna(0.0).round(3).astype(str)
        + ", cap="
        + pd.to_numeric(out["theme_overlay_cap"], errors="coerce").fillna(0.0).round(2).astype(str)
        + ", conf_gate="
        + pd.to_numeric(out["theme_overlay_soft_conf_gate"], errors="coerce").fillna(1.0).round(2).astype(str)
        + ", applied="
        + pd.to_numeric(out["theme_overlay_applied"], errors="coerce").fillna(0.0).round(3).astype(str),
        "",
    )
    out["shadow_rank_v3"] = (
        out.groupby("date")["shadow_final_score_v3"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return out


def apply_theme_overlay_v3(base: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_theme_columns(base)
    out["final_score_v3"] = (
        pd.to_numeric(out.get("w_base_v2"), errors="coerce").fillna(1.0) * pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_theme"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("theme_score_effective"), errors="coerce").fillna(0.0)
    )
    out["score_diff_v3"] = out["final_score_v3"] - pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
    out["v3_vs_v2_diff"] = out["final_score_v3"] - pd.to_numeric(out.get("final_score_v2"), errors="coerce").fillna(0.0)
    return out


def _apply_shadow_theme_overlay_v3(base: pd.DataFrame) -> pd.DataFrame:
    return _build_shadow_theme_score_frame(base)


def apply_theme_risk_soft_experiment(base: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Experimental sidecar path for theme-driven risk softening.

    This path intentionally remains separate from the production operating
    final_score. It may reuse legacy compatibility columns such as
    valuation_score when present so historical experiment outputs stay stable.
    """
    out = sanitize_theme_columns(base.copy())
    cfg = config or {
        "enabled": False,
        "soft_factor": RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT,
        "min_score": RISK_PENALTY_THEME_MIN_SCORE_DEFAULT,
        "min_confidence": RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT,
    }

    out["risk_penalty_base"] = pd.to_numeric(out.get("risk_penalty"), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["final_score_baseline"] = pd.to_numeric(out.get("final_score_v3"), errors="coerce").fillna(0.0)
    out["theme_risk_soft_enabled"] = bool(cfg.get("enabled", False))
    out["theme_risk_soft_applied"] = False
    out["theme_risk_soft_reason"] = "disabled"
    out["risk_penalty_effective"] = out["risk_penalty_base"]
    out["risk_penalty_soft_delta"] = 0.0
    out["final_score_theme_risk_soft"] = out["final_score_baseline"]
    out["rank_baseline"] = out.groupby("date")["final_score_baseline"].rank(method="first", ascending=False).astype(int)
    out["rank_theme_risk_soft"] = out["rank_baseline"]
    out["rank_change_theme_risk_soft"] = 0
    out["theme_risk_soft_explain_append"] = ""

    if not bool(cfg.get("enabled", False)):
        return out

    factor = float(cfg.get("soft_factor", RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT))
    min_score = float(cfg.get("min_score", RISK_PENALTY_THEME_MIN_SCORE_DEFAULT))
    min_conf = float(cfg.get("min_confidence", RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT))

    theme_name = out.get("dominant_theme", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    theme_score = pd.to_numeric(out.get("theme_score"), errors="coerce").fillna(0.0)
    theme_conf = pd.to_numeric(out.get("theme_confidence"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    is_theme = theme_name.ne("") & theme_name.ne("(none)")

    out.loc[~is_theme, "theme_risk_soft_reason"] = "no_theme"
    out.loc[is_theme & theme_score.lt(min_score), "theme_risk_soft_reason"] = "low_theme_score"
    out.loc[is_theme & theme_score.ge(min_score) & theme_conf.lt(min_conf), "theme_risk_soft_reason"] = "low_theme_confidence"

    apply_mask = is_theme & theme_score.ge(min_score) & theme_conf.ge(min_conf)
    out.loc[apply_mask, "theme_risk_soft_reason"] = "applied"
    out.loc[apply_mask, "theme_risk_soft_applied"] = True
    out.loc[apply_mask, "risk_penalty_effective"] = out.loc[apply_mask, "risk_penalty_base"] * factor
    out["risk_penalty_effective"] = pd.to_numeric(out["risk_penalty_effective"], errors="coerce").fillna(out["risk_penalty_base"]).clip(lower=0.0)
    out["risk_penalty_soft_delta"] = out["risk_penalty_base"] - out["risk_penalty_effective"]

    base_core = (
        pd.to_numeric(out.get("w_ret_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("ret_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_prob_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("prob_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_tech_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("tech_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_qual_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("qual_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_valuation_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("valuation_score"), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("w_risk_penalty"), errors="coerce").fillna(0.0) * out["risk_penalty_effective"]
    )
    out["final_score_theme_risk_soft"] = (
        pd.to_numeric(out.get("w_base_v2"), errors="coerce").fillna(1.0) * base_core
        + pd.to_numeric(out.get("w_theme"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("theme_score_effective"), errors="coerce").fillna(0.0)
    )
    out["final_score_theme_risk_soft"] = pd.to_numeric(out["final_score_theme_risk_soft"], errors="coerce").fillna(out["final_score_baseline"]).clip(lower=0.0, upper=100.0)
    out["rank_theme_risk_soft"] = out.groupby("date")["final_score_theme_risk_soft"].rank(method="first", ascending=False).astype(int)
    out["rank_change_theme_risk_soft"] = out["rank_baseline"] - out["rank_theme_risk_soft"]
    out["theme_risk_soft_explain_append"] = np.where(
        out["theme_risk_soft_applied"],
        "theme risk soft applied (theme_score="
        + theme_score.round(1).astype(str)
        + ", confidence="
        + theme_conf.round(2).astype(str)
        + ")",
        np.where(
            out["theme_risk_soft_reason"].eq("low_theme_confidence"),
            "theme risk soft not applied: confidence below threshold",
            np.where(
                out["theme_risk_soft_reason"].eq("low_theme_score"),
                "theme risk soft not applied: theme_score below threshold",
                np.where(out["theme_risk_soft_reason"].eq("no_theme"), "theme risk soft not applied: no_theme", "theme risk soft disabled"),
            ),
        ),
    )
    return out


def _compute_sidecar_final_score_from_penalty(df: pd.DataFrame, penalty_col: str, out_col: str) -> pd.DataFrame:
    """
    Build comparison-only sidecar scores for risk-curve experiments.

    These outputs are not the live operating final_score and may retain
    compatibility terms such as valuation_score for experiment continuity.
    """
    out = df.copy()
    base_core = (
        pd.to_numeric(out.get("w_ret_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("ret_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_prob_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("prob_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_tech_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("tech_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_qual_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("qual_score"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("w_valuation_base"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("valuation_score"), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("w_risk_penalty"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get(penalty_col), errors="coerce").fillna(0.0)
    )
    out[out_col] = (
        pd.to_numeric(out.get("w_base_v2"), errors="coerce").fillna(1.0) * base_core
        + pd.to_numeric(out.get("w_theme"), errors="coerce").fillna(0.0) * pd.to_numeric(out.get("theme_score_effective"), errors="coerce").fillna(0.0)
    )
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(pd.to_numeric(out.get("final_score_v3"), errors="coerce").fillna(0.0)).clip(lower=0.0, upper=100.0)
    return out


def apply_risk_curve_experiments(base: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    out = sanitize_theme_columns(base.copy())
    cfg = config or {
        "enabled": False,
        "exp_a_threshold": EXP_A_THRESHOLD_DEFAULT,
        "exp_a_slope_ratio": EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT,
        "exp_b_delayed_reach_factor": EXP_B_DELAYED_REACH_FACTOR_DEFAULT,
        "penalty_cap": PENALTY_CAP_DEFAULT,
    }

    out["risk_penalty_base"] = pd.to_numeric(out.get("risk_penalty"), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["final_score_baseline"] = pd.to_numeric(out.get("final_score_v3"), errors="coerce").fillna(0.0)
    out["rank_baseline"] = out.groupby("date")["final_score_baseline"].rank(method="first", ascending=False).astype(int)
    out["risk_penalty_exp_a"] = out["risk_penalty_base"]
    out["risk_penalty_exp_b"] = out["risk_penalty_base"]
    out["risk_penalty_delta_exp_a"] = 0.0
    out["risk_penalty_delta_exp_b"] = 0.0
    out["final_score_exp_a"] = out["final_score_baseline"]
    out["final_score_exp_b"] = out["final_score_baseline"]
    out["rank_exp_a"] = out["rank_baseline"]
    out["rank_exp_b"] = out["rank_baseline"]
    out["rank_change_exp_a"] = 0
    out["rank_change_exp_b"] = 0
    out["explain_base"] = "baseline risk curve"
    out["explain_exp_a"] = "baseline risk curve"
    out["explain_exp_b"] = "baseline risk curve"

    if not bool(cfg.get("enabled", False)):
        return out

    threshold = float(cfg.get("exp_a_threshold", EXP_A_THRESHOLD_DEFAULT))
    slope_ratio = float(cfg.get("exp_a_slope_ratio", EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT))
    delayed_factor = float(cfg.get("exp_b_delayed_reach_factor", EXP_B_DELAYED_REACH_FACTOR_DEFAULT))
    penalty_cap = float(cfg.get("penalty_cap", PENALTY_CAP_DEFAULT))

    mix = pd.to_numeric(out.get("pred_mdd_mix"), errors="coerce").fillna(0.0).abs()
    baseline_curve = baseline_risk_penalty_from_mix(mix, penalty_cap=penalty_cap)
    threshold_penalty = float(baseline_risk_penalty_from_mix(pd.Series([threshold]), penalty_cap=penalty_cap).iloc[0])
    incremental = (baseline_curve - threshold_penalty).clip(lower=0.0)
    exp_a = pd.Series(baseline_curve, index=out.index, dtype=float)
    above_mask = mix.gt(threshold)
    exp_a.loc[above_mask] = (threshold_penalty + incremental.loc[above_mask] * slope_ratio)
    exp_a = pd.to_numeric(exp_a, errors="coerce").fillna(baseline_curve).clip(lower=0.0, upper=penalty_cap)

    effective_mix_b = (mix / delayed_factor).clip(lower=0.0)
    exp_b = baseline_risk_penalty_from_mix(effective_mix_b, penalty_cap=penalty_cap)
    exp_b = pd.to_numeric(exp_b, errors="coerce").fillna(baseline_curve).clip(lower=0.0, upper=penalty_cap)

    out["risk_penalty_exp_a"] = exp_a
    out["risk_penalty_exp_b"] = exp_b
    out["risk_penalty_delta_exp_a"] = (out["risk_penalty_base"] - out["risk_penalty_exp_a"]).clip(lower=0.0)
    out["risk_penalty_delta_exp_b"] = (out["risk_penalty_base"] - out["risk_penalty_exp_b"]).clip(lower=0.0)

    out = _compute_sidecar_final_score_from_penalty(out, "risk_penalty_exp_a", "final_score_exp_a")
    out = _compute_sidecar_final_score_from_penalty(out, "risk_penalty_exp_b", "final_score_exp_b")
    out["rank_exp_a"] = out.groupby("date")["final_score_exp_a"].rank(method="first", ascending=False).astype(int)
    out["rank_exp_b"] = out.groupby("date")["final_score_exp_b"].rank(method="first", ascending=False).astype(int)
    out["rank_change_exp_a"] = out["rank_baseline"] - out["rank_exp_a"]
    out["rank_change_exp_b"] = out["rank_baseline"] - out["rank_exp_b"]
    out["explain_exp_a"] = f"exp_a softened above pred_mdd_mix {threshold:.2f}"
    out["explain_exp_b"] = f"exp_b delayed cap reach factor {delayed_factor:.2f}"
    return out


def apply_feature_candidate_sidecar(base: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    out = sanitize_theme_columns(base.copy())
    cfg = config or {
        "candidate": "none",
        "enabled": False,
        "exp_b_delayed_cap_reach_factor": EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT,
        "exp_b_delayed_cap_max_penalty_ratio": EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT,
        "exp_b_delayed_cap_apply_regimes": [],
        "exp_b_delayed_cap_theme_only": EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT,
        "exp_b_delayed_cap_min_theme_score": EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT,
        "exp_b_delayed_cap_min_theme_confidence": EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT,
        "penalty_cap": PENALTY_CAP_DEFAULT,
    }

    dominant_theme = out.get("dominant_theme", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    theme_score = pd.to_numeric(out.get("theme_score"), errors="coerce").fillna(0.0)
    theme_confidence = pd.to_numeric(out.get("theme_confidence"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    has_theme_flag = dominant_theme.ne("") & dominant_theme.ne("(none)") & (theme_score.gt(0.0) | theme_confidence.gt(0.0))
    regime = out.get("regime", pd.Series("", index=out.index)).fillna("").astype(str).str.strip().str.lower()

    out["has_theme_flag"] = has_theme_flag.astype(int)
    out["candidate_feature_name"] = str(cfg.get("candidate", "none"))
    out["candidate_enabled"] = is_feature_candidate_enabled(cfg)
    out["candidate_applied_flag"] = False
    out["candidate_reason"] = "candidate_disabled"
    out["candidate_baseline_final_score"] = pd.to_numeric(out.get("final_score_v3"), errors="coerce").fillna(0.0)
    out["candidate_final_score"] = out["candidate_baseline_final_score"]
    out["candidate_score_delta"] = 0.0
    out["candidate_baseline_rank"] = out.groupby("date")["candidate_baseline_final_score"].rank(method="first", ascending=False).astype(int)
    out["candidate_rank"] = out["candidate_baseline_rank"]
    out["candidate_rank_delta"] = 0
    out["candidate_baseline_risk_penalty"] = pd.to_numeric(out.get("risk_penalty"), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["candidate_risk_penalty"] = out["candidate_baseline_risk_penalty"]
    out["candidate_penalty_delta"] = 0.0
    out["candidate_explain"] = "candidate disabled"
    out["candidate_apply_regimes"] = ",".join(cfg.get("exp_b_delayed_cap_apply_regimes", [])) if cfg.get("exp_b_delayed_cap_apply_regimes") else "(all)"
    out["near_top20_band"] = "outside"
    out["top20_status"] = "outside"

    if not is_feature_candidate_enabled(cfg) or str(cfg.get("candidate", "none")).strip().lower() != "exp_b_delayed_cap":
        return out

    allowed_regimes = cfg.get("exp_b_delayed_cap_apply_regimes", [])
    theme_only = bool(cfg.get("exp_b_delayed_cap_theme_only", EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT))
    min_theme_score = float(cfg.get("exp_b_delayed_cap_min_theme_score", EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT))
    min_theme_confidence = float(
        cfg.get("exp_b_delayed_cap_min_theme_confidence", EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT)
    )
    delayed_factor = float(cfg.get("exp_b_delayed_cap_reach_factor", EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT))
    max_penalty_ratio = float(cfg.get("exp_b_delayed_cap_max_penalty_ratio", EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT))
    penalty_cap = float(cfg.get("penalty_cap", PENALTY_CAP_DEFAULT))

    regime_allowed = pd.Series(True, index=out.index)
    if allowed_regimes:
        regime_allowed = regime.isin(allowed_regimes)

    score_gate = theme_score.ge(min_theme_score)
    confidence_gate = theme_confidence.ge(min_theme_confidence)

    out.loc[~regime_allowed, "candidate_reason"] = "regime_not_allowed"
    if theme_only:
        out.loc[regime_allowed & ~has_theme_flag, "candidate_reason"] = "no_theme_filtered"
        out.loc[regime_allowed & has_theme_flag & ~score_gate, "candidate_reason"] = "low_theme_score"
        out.loc[regime_allowed & has_theme_flag & score_gate & ~confidence_gate, "candidate_reason"] = "low_theme_confidence"
    else:
        out.loc[regime_allowed & ~score_gate, "candidate_reason"] = "low_theme_score"
        out.loc[regime_allowed & score_gate & ~confidence_gate, "candidate_reason"] = "low_theme_confidence"
        out.loc[regime_allowed & score_gate & confidence_gate, "candidate_reason"] = "delayed_cap_applied"

    eligible_mask = regime_allowed.copy()
    if theme_only:
        eligible_mask &= has_theme_flag
    eligible_mask &= score_gate
    eligible_mask &= confidence_gate

    mix = pd.to_numeric(out.get("pred_mdd_mix"), errors="coerce").fillna(0.0).abs()
    baseline_penalty = out["candidate_baseline_risk_penalty"]
    candidate_raw = baseline_risk_penalty_from_mix(mix / delayed_factor, penalty_cap=penalty_cap)
    penalty_floor = baseline_penalty * max_penalty_ratio
    candidate_penalty = pd.concat([candidate_raw, penalty_floor], axis=1).max(axis=1).clip(lower=0.0, upper=penalty_cap)

    capped_mask = eligible_mask & candidate_penalty.round(10).eq(penalty_floor.round(10)) & candidate_penalty.lt(baseline_penalty)
    applied_mask = eligible_mask & candidate_penalty.lt(baseline_penalty)
    out.loc[eligible_mask, "candidate_reason"] = "delayed_cap_applied"
    out.loc[capped_mask, "candidate_reason"] = "max_penalty_ratio_capped"
    out.loc[applied_mask, "candidate_applied_flag"] = True
    out.loc[eligible_mask, "candidate_risk_penalty"] = candidate_penalty.loc[eligible_mask]
    out["candidate_penalty_delta"] = (out["candidate_baseline_risk_penalty"] - out["candidate_risk_penalty"]).clip(lower=0.0)

    out = _compute_sidecar_final_score_from_penalty(out, "candidate_risk_penalty", "candidate_final_score")
    out["candidate_score_delta"] = out["candidate_final_score"] - out["candidate_baseline_final_score"]
    out["candidate_rank"] = out.groupby("date")["candidate_final_score"].rank(method="first", ascending=False).astype(int)
    out["candidate_rank_delta"] = out["candidate_baseline_rank"] - out["candidate_rank"]
    out["candidate_explain"] = np.where(
        out["candidate_applied_flag"],
        "exp_b_delayed_cap applied: delayed reach factor="
        + pd.Series(delayed_factor, index=out.index).round(2).astype(str)
        + ", max penalty ratio="
        + pd.Series(max_penalty_ratio, index=out.index).round(2).astype(str),
        np.where(
            out["candidate_reason"].eq("regime_not_allowed"),
            "exp_b_delayed_cap not applied: regime not allowed",
            np.where(
                out["candidate_reason"].eq("no_theme_filtered"),
                "exp_b_delayed_cap not applied: theme absent",
                np.where(
                    out["candidate_reason"].eq("low_theme_score"),
                    "exp_b_delayed_cap not applied: theme score below threshold",
                    np.where(
                        out["candidate_reason"].eq("low_theme_confidence"),
                        "exp_b_delayed_cap not applied: theme confidence below threshold",
                        "exp_b_delayed_cap not applied",
                    ),
                ),
            ),
        ),
    )
    return out


def _compute_score_explain(base: pd.DataFrame) -> pd.DataFrame:
    base = shared_compute_score_explain(_resolve_component_weights(base.copy()))
    base["contrib_theme"] = (
        pd.to_numeric(base["w_theme"], errors="coerce").fillna(0.0)
        * pd.to_numeric(base.get("theme_score_effective"), errors="coerce").fillna(0.0)
    )
    base["score_contribution_theme"] = base["contrib_theme"]
    base["final_score_raw"] = pd.to_numeric(base["final_score_raw"], errors="coerce").fillna(0.0) + base["contrib_theme"].fillna(0.0)

    extremes = base.apply(_extract_factor_extremes, axis=1)
    for col in extremes.columns:
        base[col] = extremes[col]
    if "explain_text" not in base.columns:
        base["explain_text"] = ""
    base["explain_text"] = base.apply(_build_explain_text, axis=1)
    base["explain"] = base.apply(_build_explain_json, axis=1)
    return base


def _confidence_label_text(value: float) -> str:
    if pd.isna(value):
        return "Experimental"
    if value >= 80:
        return "High"
    if value >= 60:
        return "Medium"
    if value >= 40:
        return "Low"
    return "Experimental"


def _compute_data_maturity_score(base: pd.DataFrame) -> pd.Series:
    coverage = pd.to_numeric(base.get("component_coverage_ratio"), errors="coerce").fillna(0.0)
    quality_conf = pd.to_numeric(base.get("quality_score_confidence"), errors="coerce").fillna(50.0) / 100.0
    fallback_count = pd.to_numeric(base.get("fallback_count"), errors="coerce").fillna(0.0)
    score = 100.0 * (0.65 * coverage + 0.35 * quality_conf) - fallback_count * 6.0
    return pd.to_numeric(score, errors="coerce").clip(lower=0.0, upper=100.0)


def _compute_model_reliability_score(base: pd.DataFrame) -> pd.Series:
    ret_present = (~base.get("ret_score_missing", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(float)
    prob_present = (~base.get("prob_score_missing", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(float)
    tech_present = (~base.get("tech_score_missing", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(float)
    tech_guard = pd.to_numeric(base.get("tech_liquidity_guard"), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    version = base.get("score_formula_version")
    if version is None:
        version_bonus = pd.Series(65.0, index=base.index, dtype=float)
    else:
        version_clean = version.astype(str).str.strip().str.lower()
        version_bonus = pd.Series(np.where(version_clean.eq(DEFAULT_SCORE_FORMULA_VERSION.lower()), 80.0, 65.0), index=base.index, dtype=float)
    score = 100.0 * (0.25 * ret_present + 0.25 * prob_present + 0.20 * tech_present + 0.15 * tech_guard) + 0.15 * version_bonus
    return pd.to_numeric(score, errors="coerce").clip(lower=0.0, upper=100.0)


def _compute_signal_agreement_score(base: pd.DataFrame) -> pd.Series:
    components = pd.DataFrame(
        {
            "ret": pd.to_numeric(base.get("return_score"), errors="coerce"),
            "prob": pd.to_numeric(base.get("probability_score"), errors="coerce"),
            "tech": pd.to_numeric(base.get("technical_score"), errors="coerce"),
            "qual": pd.to_numeric(base.get("qual_score"), errors="coerce"),
        },
        index=base.index,
    )
    high_share = components.ge(60.0).mean(axis=1)
    weak_share = components.le(35.0).mean(axis=1)
    spread = (components.max(axis=1) - components.min(axis=1)).fillna(100.0)
    spread_score = (100.0 - spread).clip(lower=0.0, upper=100.0)
    score = 100.0 * (0.45 * high_share + 0.25 * (1.0 - weak_share) + 0.30 * (spread_score / 100.0))
    return pd.to_numeric(score, errors="coerce").clip(lower=0.0, upper=100.0)


def _compute_regime_fitness_score(base: pd.DataFrame) -> pd.Series:
    regime = base.get("regime", pd.Series("defensive", index=base.index)).astype(str).str.lower()
    ret = pd.to_numeric(base.get("return_score"), errors="coerce").fillna(0.0)
    prob = pd.to_numeric(base.get("probability_score"), errors="coerce").fillna(0.0)
    tech = pd.to_numeric(base.get("technical_score"), errors="coerce").fillna(0.0)
    qual = pd.to_numeric(base.get("qual_score"), errors="coerce").fillna(0.0)
    val = pd.to_numeric(base.get("valuation_score"), errors="coerce").fillna(50.0)
    risk = pd.to_numeric(base.get("risk_penalty"), errors="coerce").fillna(0.0)
    risk_score = (100.0 - (risk / 18.0 * 100.0)).clip(lower=0.0, upper=100.0)

    bull_score = 0.35 * ret + 0.30 * prob + 0.25 * tech + 0.10 * risk_score
    neutral_score = 0.28 * ret + 0.24 * prob + 0.20 * tech + 0.16 * qual + 0.12 * risk_score
    defensive_score = 0.22 * ret + 0.16 * prob + 0.12 * tech + 0.25 * qual + 0.15 * val + 0.10 * risk_score

    score = np.select(
        [regime.eq("bull"), regime.eq("neutral")],
        [bull_score, neutral_score],
        default=defensive_score,
    )
    return pd.Series(score, index=base.index, dtype=float).clip(lower=0.0, upper=100.0)


def _build_confidence_reason(
    row: pd.Series,
    *,
    include_model_reliability: bool = True,
) -> str:
    axis_pairs = [
        ("data_maturity_score", "data maturity is thin"),
        ("signal_agreement_score", "core signals disagree"),
        ("regime_fitness_score", "current regime fit is weak"),
    ]
    if include_model_reliability:
        axis_pairs.insert(1, ("model_reliability_score", "model reliability inputs are incomplete"))
    weak = []
    strong = []
    for col, text in axis_pairs:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(value):
            continue
        if value < 55.0:
            weak.append(text)
        elif value >= 75.0:
            strong.append(text.replace(" is thin", " is strong").replace(" are incomplete", " is strong").replace(" disagree", " align").replace(" is weak", " is strong"))

    reasons: list[str] = []
    fallback_count = pd.to_numeric(row.get("fallback_count"), errors="coerce")
    if pd.notna(fallback_count) and fallback_count > 0:
        reasons.append(f"fallback_count={int(fallback_count)}")
    if weak:
        reasons.extend(weak[:2])
    elif strong:
        reasons.append(strong[0])
    else:
        reasons.append("confidence inputs are broadly stable")
    return "; ".join(reasons[:2])


def _build_confidence_explain_text(row: pd.Series) -> str:
    score = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    label = str(row.get("confidence_label") or "Experimental")
    data_maturity = pd.to_numeric(row.get("data_maturity_score"), errors="coerce")
    model_reliability = pd.to_numeric(row.get("model_reliability_score"), errors="coerce")
    signal_agreement = pd.to_numeric(row.get("signal_agreement_score"), errors="coerce")
    regime_fitness = pd.to_numeric(row.get("regime_fitness_score"), errors="coerce")
    reason = str(row.get("confidence_reason") or "").strip()
    return (
        f"Confidence score is {0.0 if pd.isna(score) else score:.1f} ({label}). "
        f"data_maturity_score={0.0 if pd.isna(data_maturity) else data_maturity:.1f}, "
        f"model_reliability_score={0.0 if pd.isna(model_reliability) else model_reliability:.1f}, "
        f"signal_agreement_score={0.0 if pd.isna(signal_agreement) else signal_agreement:.1f}, "
        f"regime_fitness_score={0.0 if pd.isna(regime_fitness) else regime_fitness:.1f}. "
        f"Reason: {reason or 'none'}."
    )


def _confidence_label_operational_text(value: float) -> str:
    if pd.isna(value):
        return "Low"
    if value >= 70:
        return "High"
    if value >= 50:
        return "Medium"
    return "Low"


def _compute_confidence_score(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["confidence_version"] = DEFAULT_CONFIDENCE_VERSION
    component_present = pd.DataFrame(
        {
            col: (~base.get(f"{col}_missing", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(float)
            for col in CORE_COMPONENT_COLUMNS
        },
        index=base.index,
    )
    base["component_coverage_ratio"] = component_present.mean(axis=1).clip(lower=0.0, upper=1.0)
    base["data_maturity_score"] = _compute_data_maturity_score(base)
    base["model_reliability_score"] = _compute_model_reliability_score(base)
    base["signal_agreement_score"] = _compute_signal_agreement_score(base)
    base["regime_fitness_score"] = _compute_regime_fitness_score(base)
    base["confidence_score_research"] = (
        0.30 * pd.to_numeric(base["data_maturity_score"], errors="coerce").fillna(50.0)
        + 0.30 * pd.to_numeric(base["model_reliability_score"], errors="coerce").fillna(50.0)
        + 0.25 * pd.to_numeric(base["signal_agreement_score"], errors="coerce").fillna(50.0)
        + 0.15 * pd.to_numeric(base["regime_fitness_score"], errors="coerce").fillna(50.0)
    ).clip(lower=0.0, upper=100.0)
    base["confidence_score_operational"] = (
        0.40 * pd.to_numeric(base["data_maturity_score"], errors="coerce").fillna(50.0)
        + 0.40 * pd.to_numeric(base["signal_agreement_score"], errors="coerce").fillna(50.0)
        + 0.20 * pd.to_numeric(base["regime_fitness_score"], errors="coerce").fillna(50.0)
    ).clip(lower=0.0, upper=100.0)

    # UI/API should consume confidence_score_operational only.
    # confidence_score_research preserves the previous research-oriented meta score.
    base["confidence_score"] = base["confidence_score_operational"]
    base["confidence_label_research"] = base["confidence_score_research"].apply(_confidence_label_text)
    base["confidence_label_operational"] = base["confidence_score_operational"].apply(_confidence_label_operational_text)
    base["confidence_label"] = base["confidence_label_operational"]
    base["confidence_grade"] = base["confidence_label_operational"]
    base["confidence_reason"] = base.apply(
        lambda row: _build_confidence_reason(row, include_model_reliability=False),
        axis=1,
    )
    base["confidence_explain_text"] = base.apply(_build_confidence_explain_text, axis=1)
    return base


def _compute_quality_gate_penalty_ratio(
    base: pd.DataFrame,
    allowed_regimes: set[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    qual_score = pd.to_numeric(base.get("qual_score"), errors="coerce").fillna(100.0)
    regime = (
        base.get("regime", pd.Series("neutral", index=base.index))
        .fillna("neutral")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    normalized_allowed = {str(item).strip().lower() for item in (allowed_regimes or set())}
    regime_allowed = regime.isin(normalized_allowed) if normalized_allowed else pd.Series(True, index=base.index)

    penalty_ratio = pd.Series(1.0, index=base.index, dtype="float64")
    low20 = qual_score.lt(20.0)
    low40 = qual_score.ge(20.0) & qual_score.lt(40.0)
    quality_flag = (low20 | low40) & regime_allowed

    experiment = str(QUALITY_GATE_EXPERIMENT).strip().lower()
    if experiment == "v1":
        penalty_ratio.loc[low20 & regime.eq("defensive")] = 0.94
        penalty_ratio.loc[low40 & regime.eq("defensive")] = 0.97
        penalty_ratio = penalty_ratio.where(regime_allowed, 1.0)
        gate_applied = penalty_ratio.lt(1.0)
        return penalty_ratio, gate_applied, quality_flag

    if experiment == "v2":
        penalty_ratio.loc[low20 & regime.eq("defensive")] = 0.94
        penalty_ratio.loc[low40 & regime.eq("defensive")] = 0.97
        penalty_ratio = penalty_ratio.where(regime_allowed, 1.0)
        alpha_protected = base.apply(should_protect_alpha, axis=1)
        penalty_ratio = penalty_ratio.where(~alpha_protected, 1.0)
        gate_applied = penalty_ratio.lt(1.0)
        quality_flag = quality_flag & ~alpha_protected
        return penalty_ratio, gate_applied, quality_flag

    if experiment == "v3":
        gate_applied = pd.Series(False, index=base.index)
        penalty_ratio = pd.Series(1.0, index=base.index, dtype="float64")
        return penalty_ratio, gate_applied, quality_flag

    raise ValueError(f"Unsupported QUALITY_GATE_EXPERIMENT: {QUALITY_GATE_EXPERIMENT}")


def should_protect_alpha(row: pd.Series) -> bool:
    prob_score = pd.to_numeric(row.get("prob_score"), errors="coerce")
    ret_score = pd.to_numeric(row.get("ret_score"), errors="coerce")
    confidence_score = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    return bool(
        (pd.notna(prob_score) and prob_score >= 85.0)
        or (pd.notna(ret_score) and ret_score >= 75.0)
        or (pd.notna(confidence_score) and confidence_score >= 85.0)
    )


def apply_quality_downside_gate(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    regime = (
        out.get("regime", pd.Series("neutral", index=out.index))
        .fillna("neutral")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    is_defensive = regime.eq("defensive")
    penalty_ratio = pd.Series(1.0, index=out.index, dtype="float64")
    quality_flag = pd.Series(False, index=out.index)

    if is_defensive.any():
        defensive_penalty_ratio, defensive_gate_applied, defensive_quality_flag = _compute_quality_gate_penalty_ratio(
            out.loc[is_defensive].copy(),
            allowed_regimes={"defensive"},
        )
        penalty_ratio.loc[is_defensive] = defensive_penalty_ratio
        gate_applied = pd.Series(False, index=out.index)
        gate_applied.loc[is_defensive] = defensive_gate_applied
        quality_flag.loc[is_defensive] = defensive_quality_flag
    else:
        gate_applied = pd.Series(False, index=out.index)

    out["quality_flag"] = quality_flag
    out["quality_penalty_ratio"] = penalty_ratio
    out["quality_gate_applied"] = gate_applied
    out["final_score"] = (
        pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
        * out["quality_penalty_ratio"]
    ).clip(lower=0.0, upper=100.0)
    out["final_score_before_theme"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
    return out


def attach_quality_gate_shadow(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    penalty_ratio, gate_applied, quality_flag = _compute_quality_gate_penalty_ratio(
        out,
        allowed_regimes=QUALITY_GATE_ALLOWED_REGIMES,
    )
    out["quality_flag"] = quality_flag
    out["shadow_quality_penalty_ratio"] = penalty_ratio
    out["shadow_quality_gate_applied"] = gate_applied
    out["shadow_final_score_quality_gate"] = (
        pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
        * out["shadow_quality_penalty_ratio"]
    ).clip(lower=0.0, upper=100.0)
    out["shadow_rank_quality_gate"] = (
        out.groupby("date")["shadow_final_score_quality_gate"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["quality_gate_experiment"] = QUALITY_GATE_EXPERIMENT
    return out


def attach_quality_risk_guard_shadow(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    qual_score = pd.to_numeric(out.get("qual_score"), errors="coerce")
    risk_penalty = pd.to_numeric(out.get("risk_penalty"), errors="coerce")

    extra_penalty = pd.Series(0.0, index=out.index, dtype="float64")
    extra_penalty = extra_penalty + np.where(qual_score < 20.0, 6.0, 0.0)
    extra_penalty = extra_penalty + np.where(risk_penalty >= 12.0, 4.0, 0.0)
    extra_penalty = pd.Series(extra_penalty, index=out.index, dtype="float64")

    out["shadow_quality_risk_guard_penalty"] = extra_penalty
    out["shadow_quality_risk_guard_applied"] = extra_penalty.gt(0.0)
    out["shadow_final_score_quality_risk_guard"] = (
        pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
        - out["shadow_quality_risk_guard_penalty"]
    ).clip(lower=0.0, upper=100.0)
    out["shadow_rank_quality_risk_guard"] = (
        out.groupby("date")["shadow_final_score_quality_risk_guard"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def apply_default_ranking_scores(base: pd.DataFrame) -> pd.DataFrame:
    """
    final_score
    - Input columns:
      technical_score, return_score, probability_score, quality_score,
      risk_penalty, regime
    - Primary return axis: return_score (alias of ret_score)
    - Operational probability axis: probability_score (alias of prob_score)
    - Technical axis: technical_score (alias of tech_score)
    - Quality axis: quality_score / qual_score
    - Purpose:
      produce the final production ranking score with return / probability /
      technical components as the main ranking drivers
    - Interpretation:
      higher means a stronger candidate after regime weighting and risk deduction
    - Score type:
      weighted composite score, clipped to 0~100
    - Confidence separation:
      confidence_score is computed separately as an evidence-quality meta metric
      and is not multiplied into or subtracted from final_score.

    Weighting logic
    - bull profile expands return / probability / technical exposure
    - neutral profile uses the baseline balanced mix
    - defensive profile increases quality weight while keeping return alive
    - safety is not a direct positive term in final_score; safety behavior is
      expressed through risk_penalty
    - valuation_score is retained as a compatibility / diagnostic column and is
      not a direct positive axis in the operating final_score
    """
    base = base.copy()
    base = _ensure_regime_column(base, log_distribution=True, log_prefix="apply_default_ranking_scores")
    base = _attach_component_integrity_flags(base)
    base = _attach_operational_score_aliases(base)

    for col in [
        "tech_score",
        "ret_score",
        "prob_score",
        "qual_score",
        "valuation_score",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
    ]:
        source = base[col] if col in base.columns else pd.Series(index=base.index, dtype="float64")
        base[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    return_source = base["return_score"] if "return_score" in base.columns else pd.Series(index=base.index, dtype="float64")
    prob_source = base["probability_score"] if "probability_score" in base.columns else pd.Series(index=base.index, dtype="float64")
    tech_source = base["technical_score"] if "technical_score" in base.columns else pd.Series(index=base.index, dtype="float64")
    risk_source = base["risk_penalty"] if "risk_penalty" in base.columns else pd.Series(index=base.index, dtype="float64")
    base["return_score"] = pd.to_numeric(return_source, errors="coerce").fillna(base["ret_score"])
    base["probability_score"] = pd.to_numeric(prob_source, errors="coerce").fillna(base["prob_score"])
    base["technical_score"] = pd.to_numeric(tech_source, errors="coerce").fillna(base["tech_score"])
    base["risk_penalty"] = pd.to_numeric(risk_source, errors="coerce").fillna(0.0)

    component_scale_diag = {
        col: {
            "min": float(base[col].min()),
            "max": float(base[col].max()),
            "mean": float(base[col].mean()),
        }
        for col in ["ret_score", "prob_score", "tech_score", "qual_score"]
    }
    logging.info("final_score component scale diagnostics: %s", component_scale_diag)

    base = shared_apply_baseline_final_score(base, fill_score_columns=False, include_explain=False)
    if QUALITY_GATE_FEATURE_ENABLED:
        base = apply_quality_downside_gate(base)
    else:
        base["quality_penalty_ratio"] = 1.0
        base["quality_gate_applied"] = False
    if QUALITY_GATE_FEATURE_CANDIDATE:
        base = attach_quality_gate_shadow(base)
    base = _resolve_component_weights(base)
    logging.info(
        "weight profile distribution: %s",
        base["weight_profile"].fillna("NA").value_counts(dropna=False).to_dict(),
    )
    base["final_score_before_theme"] = pd.to_numeric(base["final_score_before_theme"], errors="coerce").clip(lower=0.0, upper=100.0)
    base["final_score"] = pd.to_numeric(base["final_score"], errors="coerce").clip(lower=0.0, upper=100.0)
    base = apply_theme_overlay_v2(base)
    base["final_score_v2_before_theme"] = pd.to_numeric(base["final_score_v2_before_theme"], errors="coerce").clip(lower=0.0, upper=100.0)
    base["final_score_v2"] = pd.to_numeric(base["final_score_v2"], errors="coerce").clip(lower=0.0, upper=100.0)
    base = apply_theme_overlay_v3(base)
    base = _apply_shadow_theme_overlay_v3(base)
    base = attach_quality_risk_guard_shadow(base)
    base["final_score_v3"] = pd.to_numeric(base["final_score_v3"], errors="coerce").clip(lower=0.0, upper=100.0)
    base["score_diff_v2"] = pd.to_numeric(base["score_diff_v2"], errors="coerce").fillna(0.0)
    base["score_diff_v3"] = pd.to_numeric(base["score_diff_v3"], errors="coerce").fillna(0.0)
    base["v3_vs_v2_diff"] = pd.to_numeric(base["v3_vs_v2_diff"], errors="coerce").fillna(0.0)
    base["shadow_theme_weight_raw"] = pd.to_numeric(base.get("shadow_theme_weight_raw"), errors="coerce")
    base["shadow_theme_weight"] = pd.to_numeric(base.get("shadow_theme_weight"), errors="coerce")
    base["shadow_theme_weight_effective"] = pd.to_numeric(base.get("shadow_theme_weight_effective"), errors="coerce")
    base["shadow_base_weight"] = pd.to_numeric(base.get("shadow_base_weight"), errors="coerce")
    if "shadow_floor_applied" in base.columns:
        base["shadow_floor_applied"] = base["shadow_floor_applied"].fillna(False).astype(bool)
    base["shadow_theme_score_effective"] = pd.to_numeric(base.get("shadow_theme_score_effective"), errors="coerce")
    base["shadow_final_score_v3"] = pd.to_numeric(base.get("shadow_final_score_v3"), errors="coerce")
    base["shadow_score_diff_v3"] = pd.to_numeric(base.get("shadow_score_diff_v3"), errors="coerce")
    if "shadow_rank_v3" in base.columns:
        base["shadow_rank_v3"] = pd.to_numeric(base["shadow_rank_v3"], errors="coerce").round().astype("Int64")
    base["rank_before_theme"] = (
        base.groupby("date")["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    sample_cols = [
        col for col in [
            "date",
            "code",
            "regime",
            "tech_score",
            "ret_score",
            "prob_score",
            "qual_score",
            "valuation_score",
            "theme_score",
            "dominant_theme",
            "theme_confidence",
            "theme_score_effective",
            "risk_penalty",
            "final_score_before_theme",
            "final_score",
            "final_score_v2",
            "final_score_v3",
            "shadow_final_score_v3",
        ]
        if col in base.columns
    ]
    if sample_cols:
        logging.info(
            "final_score sample rows:\n%s",
            base[sample_cols].head(5).to_string(index=False),
        )
    live_score_col = "final_score_v3" if bool(_resolve_theme_overlay_runtime_flags().get("live_uses_theme", False)) else "final_score"
    base["live_score_source"] = live_score_col
    base["live_score"] = pd.to_numeric(base.get(live_score_col), errors="coerce").fillna(
        pd.to_numeric(base.get("final_score"), errors="coerce").fillna(0.0)
    )
    base["live_rank"] = (
        base.groupby("date")[live_score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    base["rank_final"] = base["live_rank"]
    base = _compute_score_explain(base)
    base = _compute_confidence_score(base)
    base = attach_score_explain_columns(base)
    base["rank_v2"] = (
        base.groupby("date")["final_score_v2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    if "score_formula_version" in base.columns:
        formula_series = base["score_formula_version"].astype(str)
    else:
        formula_series = pd.Series(resolve_score_formula_version(), index=base.index, dtype=str)
    if bool(LAST_THEME_GUARD_STATUS.get("applied", False)):
        base["score_formula_version"] = formula_series + f"+{DEFAULT_THEME_FACTOR_VERSION}"
    else:
        base["score_formula_version"] = formula_series
    return base


def compute_rebalance_score(
    df: pd.DataFrame,
    *,
    score_col: str = "final_score_custom",
    w_ret: float = REBALANCE_WEIGHT_RET,
    w_prob: float = REBALANCE_WEIGHT_PROB,
    w_qual: float = REBALANCE_WEIGHT_QUAL,
    w_tech: float = REBALANCE_WEIGHT_TECH,
    w_pred: float = REBALANCE_WEIGHT_PRED,
    pred_score_default: float = REBALANCE_PRED_SCORE_DEFAULT,
) -> pd.DataFrame:
    """Compute rebalance-only custom score from the component scores built here."""
    df = df.copy()
    df[score_col] = (
        w_ret * df["ret_score"].fillna(0)
        + w_prob * df["prob_score"].fillna(0)
        + w_qual * df["qual_score"].fillna(0)
        + w_tech * df["tech_score"].fillna(0)
        + w_pred * pred_score_default
    )
    if "risk_penalty" in df.columns:
        df[score_col] = df[score_col] * df["risk_penalty"].fillna(1.0)
    return df


def _load_base_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = _load_csv(PREDICTIONS_CSV, required=True)
    scores = _load_csv(SCORES_CSV, required=False)
    feats = _load_csv(FEATURES_CSV, required=True)
    universe = _load_csv(UNIVERSE_CSV, required=False)

    preds = _normalize_date(preds)
    scores = _normalize_date(scores)
    feats = _normalize_date(feats)

    for df, name in [(preds, "predictions"), (feats, "features")]:
        if df.empty:
            raise RuntimeError(f"{name} is empty ??cannot build ranking.")

    if "date" in preds.columns and preds["date"].notna().any():
        latest_pred_date = preds["date"].max()
        stale_pred_mask = preds["date"].lt(latest_pred_date)
        stale_pred_count = int(stale_pred_mask.sum())
        latest_pred_date_text = (
            latest_pred_date.strftime("%Y-%m-%d")
            if hasattr(latest_pred_date, "strftime")
            else str(latest_pred_date)
        )
        if stale_pred_count > 0:
            stale_codes = preds.loc[stale_pred_mask, "code"].astype(str).head(10).tolist()
            logging.warning(
                "Predictions contain stale rows; filtering to latest prediction date=%s stale_rows=%d sample_codes=%s",
                latest_pred_date_text,
                stale_pred_count,
                ",".join(stale_codes) if stale_codes else "NA",
            )
            preds = preds.loc[~stale_pred_mask].copy()
            if not scores.empty and "date" in scores.columns:
                score_dates = pd.to_datetime(scores["date"], errors="coerce")
                latest_pred_ts = pd.to_datetime(latest_pred_date, errors="coerce")
                scores = scores.loc[score_dates.eq(latest_pred_ts)].copy()
        else:
            logging.info(
                "Using latest prediction snapshot date=%s rows=%d",
                latest_pred_date_text,
                len(preds),
            )

    if scores.empty:
        logging.warning("scores_final.csv missing/empty -> using feature_v1 tech-score fallback path")
        scores = preds[["date", "code"]].copy()
        scores["score"] = 0.0

    _normalize_code_columns(preds, scores, feats, universe)
    return preds, scores, feats, universe


def _merge_inputs(
    preds: pd.DataFrame,
    scores: pd.DataFrame,
    feats: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    base = preds.merge(
        scores,
        on=["date", "code"],
        how="left",
        suffixes=("", "_score"),
    )

    feat_cols = ["date", "code", "close"]
    for col in ["quality_score", "quality_factor_count", "quality_missing_ratio", "quality_score_confidence"]:
        if col in feats.columns:
            feat_cols.append(col)
    for col in ["vol_20", "vol_60", "vol_ma_20", "volume", "mom_20", "close_over_ma20", "rsi_14", "vol_ratio_20", "ma_5", "ma_20", "ma_60", "ret_5d", "ret_10d"]:
        if col in feats.columns:
            feat_cols.append(col)

    base = base.merge(
        feats[feat_cols],
        on=["date", "code"],
        how="left",
        suffixes=("", "_feat"),
    )
    logging.info("Base merged shape (preds + scores + features): %s", base.shape)

    if universe is not None and not universe.empty and "code" in universe.columns:
        base = base.merge(
            universe,
            on="code",
            how="left",
            suffixes=("", "_univ"),
        )
        logging.info("After universe merge shape: %s", base.shape)

    if base.empty:
        raise RuntimeError("No rows after merging predictions/scores/features ??cannot build ranking.")

    return base


def sanitize_theme_columns(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    theme_score_source = out["theme_score"] if "theme_score" in out.columns else pd.Series(index=out.index, dtype="float64")
    out["theme_score"] = pd.to_numeric(theme_score_source, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dominant_theme = out["dominant_theme"] if "dominant_theme" in out.columns else None
    if dominant_theme is None:
        out["dominant_theme"] = ""
    else:
        out["dominant_theme"] = dominant_theme.fillna("").astype(str)
    confidence_source = out["theme_confidence"] if "theme_confidence" in out.columns else pd.Series(index=out.index, dtype="float64")
    confidence = pd.to_numeric(confidence_source, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["theme_confidence"] = confidence.clip(lower=0.0, upper=1.0)
    theme_raw_source = out["theme_raw_score"] if "theme_raw_score" in out.columns else pd.Series(index=out.index, dtype="float64")
    out["theme_raw_score"] = pd.to_numeric(theme_raw_source, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_theme_source = out["raw_theme_score"] if "raw_theme_score" in out.columns else out["theme_raw_score"]
    out["raw_theme_score"] = pd.to_numeric(raw_theme_source, errors="coerce").replace([np.inf, -np.inf], np.nan)
    theme_threshold_source = out["theme_threshold"] if "theme_threshold" in out.columns else pd.Series(index=out.index, dtype="float64")
    out["theme_threshold"] = pd.to_numeric(theme_threshold_source, errors="coerce").fillna(0.30)
    filtered_theme_source = out["filtered_theme_score"] if "filtered_theme_score" in out.columns else out["theme_score"]
    out["filtered_theme_score"] = pd.to_numeric(filtered_theme_source, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    theme_debug_source = out["theme_debug_reason"] if "theme_debug_reason" in out.columns else pd.Series("", index=out.index, dtype="object")
    out["theme_debug_reason"] = theme_debug_source.fillna("").astype(str)
    out["theme_score_effective"] = np.where(
        out["dominant_theme"].astype(str).str.strip().ne(""),
        out["theme_score"].fillna(0.0) * out["theme_confidence"].fillna(0.0),
        0.0,
    )
    out["theme_score_effective"] = pd.to_numeric(out["theme_score_effective"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["theme_applied_flag"] = (
        out["dominant_theme"].astype(str).str.strip().apply(_is_active_theme_label)
        & out["theme_score_effective"].gt(0.0)
    )
    return out


def _attach_theme_preview_frame(
    base: pd.DataFrame,
    theme_overlay: pd.DataFrame,
    *,
    runtime: dict[str, object],
    reason: str,
    theme_latest_date: str,
    source: str,
    theme_row_count: int,
    available_dates: list[str] | None = None,
) -> pd.DataFrame:
    ranking_dates = sorted(base["date"].dropna().astype(str).unique().tolist()) if "date" in base.columns else []
    latest_ranking_date = max(ranking_dates) if ranking_dates else "NA"
    merged = base.merge(theme_overlay, on=["date", "code"], how="left") if not theme_overlay.empty else base.copy()
    matched_rows = int(merged["theme_score"].notna().sum()) if "theme_score" in merged.columns else 0
    base_rows = int(len(merged))
    coverage_ratio = float(matched_rows / base_rows) if base_rows else 0.0

    _set_theme_guard_status(
        overlay_enabled=bool(runtime.get("overlay_enabled", False)),
        mode=str(runtime.get("mode", THEME_OVERLAY_OFF)),
        operational=False,
        applied=False,
        disable_reason=reason,
        coverage_ratio=coverage_ratio,
        coverage_threshold=float(runtime.get("coverage_threshold", THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT)),
        matched_rows=matched_rows,
        base_rows=base_rows,
        theme_row_count=int(theme_row_count),
        ranking_latest_date=latest_ranking_date,
        theme_latest_date=theme_latest_date,
        available_theme_dates=list(available_dates or []),
        source=source,
    )
    logging.info("[theme] overlay_enabled: %s", False)
    logging.info("[theme] disable_reason: %s", reason)
    logging.info("[theme] coverage_ratio: %.4f", coverage_ratio)
    logging.info("[theme] theme_date vs ranking_date: %s vs %s", theme_latest_date, latest_ranking_date)

    if "theme_score" not in merged.columns:
        merged["theme_score"] = 0.0
    if "dominant_theme" not in merged.columns:
        merged["dominant_theme"] = "(none)"
    if "theme_confidence" not in merged.columns:
        merged["theme_confidence"] = 0.0
    if "theme_raw_score" not in merged.columns:
        merged["theme_raw_score"] = np.nan
    merged["filtered_theme_score"] = pd.to_numeric(merged.get("theme_score"), errors="coerce").fillna(0.0)
    merged["raw_theme_score"] = pd.to_numeric(merged.get("theme_raw_score"), errors="coerce")
    merged["theme_threshold"] = 0.30
    merged["theme_debug_reason"] = np.where(
        merged.get("theme_score").isna() & merged.get("theme_raw_score").isna(),
        "mapping_missing",
        "",
    )
    return sanitize_theme_columns(merged)


def _force_zero_theme_frame(base: pd.DataFrame, *, runtime: dict[str, object], reason: str, coverage_ratio: float = 0.0, theme_latest_date: str = "NA", source: str = "none", theme_row_count: int = 0, available_dates: list[str] | None = None) -> pd.DataFrame:
    ranking_dates = sorted(base["date"].dropna().astype(str).unique().tolist()) if "date" in base.columns else []
    latest_ranking_date = max(ranking_dates) if ranking_dates else "NA"
    logging.info("[theme] overlay_enabled: %s", bool(runtime.get("operational", False)))
    logging.info("[theme] disable_reason: %s", reason)
    logging.info("[theme] coverage_ratio: %.4f", float(coverage_ratio))
    logging.info("[theme] theme_date vs ranking_date: %s vs %s", theme_latest_date, latest_ranking_date)
    _set_theme_guard_status(
        overlay_enabled=bool(runtime.get("overlay_enabled", False)),
        mode=str(runtime.get("mode", THEME_OVERLAY_OFF)),
        operational=bool(runtime.get("operational", False)),
        applied=False,
        disable_reason=reason,
        coverage_ratio=float(coverage_ratio),
        coverage_threshold=float(runtime.get("coverage_threshold", THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT)),
        matched_rows=0,
        base_rows=int(len(base)),
        theme_row_count=int(theme_row_count),
        ranking_latest_date=latest_ranking_date,
        theme_latest_date=theme_latest_date,
        available_theme_dates=list(available_dates or []),
        source=source,
    )
    out = base.copy()
    out["theme_score"] = 0.0
    out["dominant_theme"] = "(none)"
    out["theme_confidence"] = 0.0
    out["theme_raw_score"] = np.nan
    out["raw_theme_score"] = np.nan
    out["filtered_theme_score"] = 0.0
    out["theme_threshold"] = 0.30
    out["theme_applied_flag"] = False
    out["theme_debug_reason"] = reason
    out["theme_score_effective"] = 0.0
    return sanitize_theme_columns(out)


def apply_theme_overlay(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    dates = sorted(base["date"].dropna().astype(str).unique().tolist()) if "date" in base.columns else []
    runtime = _resolve_theme_overlay_runtime()
    if not bool(runtime.get("operational")):
        preview_payload = _load_theme_overlay(dates)
        preview_reason = str(runtime.get("overlay_disable_reason", "mode_mismatch") or "mode_mismatch")
        _set_theme_gate_debug(overlay_gate_result="disabled", overlay_disable_reason=preview_reason)
        if not preview_payload["df"].empty:
            return _attach_theme_preview_frame(
                base,
                preview_payload["df"],
                runtime=runtime,
                reason=preview_reason,
                theme_latest_date=str(preview_payload.get("latest_theme_date") or "NA"),
                source=str(preview_payload.get("source") or "none"),
                theme_row_count=int(preview_payload.get("theme_row_count") or 0),
                available_dates=list(preview_payload.get("available_dates") or []),
            )
        return _force_zero_theme_frame(base, runtime=runtime, reason=preview_reason)

    payload = _load_theme_overlay(dates)
    theme_overlay = payload["df"]
    latest_ranking_date = max(dates) if dates else "NA"
    latest_theme_date = str(payload.get("latest_theme_date") or "NA")
    available_dates = list(payload.get("available_dates") or [])
    source = str(payload.get("source") or "none")
    theme_row_count = int(payload.get("theme_row_count") or 0)
    load_error_reason = str(payload.get("load_error_reason") or "")

    logging.info("[theme] overlay_enabled: %s", True)
    logging.info("[theme] theme_date vs ranking_date: %s vs %s", latest_theme_date, latest_ranking_date)

    if theme_overlay.empty:
        disable_reason = load_error_reason or "missing_theme_input"
        _set_theme_gate_debug(overlay_gate_result="disabled", overlay_disable_reason=disable_reason)
        return _force_zero_theme_frame(
            base,
            runtime=runtime,
            reason=disable_reason,
            theme_latest_date=latest_theme_date,
            source=source,
            theme_row_count=theme_row_count,
            available_dates=available_dates,
        )
    if latest_theme_date != latest_ranking_date:
        _set_theme_gate_debug(overlay_gate_result="disabled", overlay_disable_reason="mode_mismatch")
        return _force_zero_theme_frame(
            base,
            runtime=runtime,
            reason="stale_date",
            theme_latest_date=latest_theme_date,
            source=source,
            theme_row_count=theme_row_count,
            available_dates=available_dates,
        )
    if set(available_dates) != set(dates):
        _set_theme_gate_debug(overlay_gate_result="disabled", overlay_disable_reason="mode_mismatch")
        return _force_zero_theme_frame(
            base,
            runtime=runtime,
            reason="stale_date",
            theme_latest_date=latest_theme_date,
            source=source,
            theme_row_count=theme_row_count,
            available_dates=available_dates,
        )

    merged = base.merge(theme_overlay, on=["date", "code"], how="left")
    merged["raw_theme_score"] = pd.to_numeric(merged.get("theme_raw_score"), errors="coerce")
    merged["filtered_theme_score"] = pd.to_numeric(merged.get("theme_score"), errors="coerce").fillna(0.0)
    merged["theme_threshold"] = 0.30
    merged["theme_debug_reason"] = np.where(
        merged["theme_score"].isna() & merged["theme_raw_score"].isna(),
        "mapping_missing",
        np.where(
            pd.to_numeric(merged.get("theme_confidence"), errors="coerce").fillna(0.0).lt(0.30),
            "low_theme_confidence",
            np.where(pd.to_numeric(merged.get("theme_score"), errors="coerce").fillna(0.0).le(0.0), "filtered_out", ""),
        ),
    )
    matched_rows = int(merged["theme_score"].notna().sum()) if "theme_score" in merged.columns else 0
    base_rows = int(len(merged))
    coverage_ratio = float(matched_rows / base_rows) if base_rows else 0.0
    logging.info("[theme] disable_reason: (none)")
    logging.info("[theme] coverage_ratio: %.4f", coverage_ratio)
    if coverage_ratio < float(runtime.get("coverage_threshold", THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT)):
        _set_theme_gate_debug(overlay_gate_result="disabled", overlay_disable_reason="validation_failed")
        return _force_zero_theme_frame(
            base,
            runtime=runtime,
            reason="low_coverage",
            coverage_ratio=coverage_ratio,
            theme_latest_date=latest_theme_date,
            source=source,
            theme_row_count=theme_row_count,
            available_dates=available_dates,
        )

    _set_theme_guard_status(
        overlay_enabled=bool(runtime.get("overlay_enabled", False)),
        mode=str(runtime.get("mode", THEME_OVERLAY_OFF)),
        operational=True,
        applied=True,
        disable_reason="",
        coverage_ratio=coverage_ratio,
        coverage_threshold=float(runtime.get("coverage_threshold", THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT)),
        matched_rows=matched_rows,
        base_rows=base_rows,
        theme_row_count=theme_row_count,
        ranking_latest_date=latest_ranking_date,
        theme_latest_date=latest_theme_date,
        available_theme_dates=available_dates,
        source=source,
    )
    _set_theme_gate_debug(overlay_gate_result="enabled", overlay_disable_reason="(none)")
    return sanitize_theme_columns(merged)


def _attach_theme_columns(base: pd.DataFrame) -> pd.DataFrame:
    return apply_theme_overlay(base)


def build_ranking(
    theme_risk_soft_config: dict | None = None,
    risk_curve_experiment_config: dict | None = None,
    feature_candidate_config: dict | None = None,
) -> pd.DataFrame:
    preds, scores, feats, universe = _load_base_inputs()
    base = _merge_inputs(preds, scores, feats, universe)
    base = apply_theme_overlay(base)
    score_formula_version = resolve_score_formula_version()

    market_up, mkt_info, market_history = _load_market_status()
    base = _attach_market_columns(base, market_up, mkt_info, market_history)
    base = compute_component_scores(base)
    base["score_formula_version"] = score_formula_version
    base = apply_default_ranking_scores(base)

    base["date"] = pd.to_datetime(base["date"])
    live_sort_col = "final_score_v3" if bool(_resolve_theme_overlay_runtime_flags().get("live_uses_theme", False)) else "final_score"
    base = base.sort_values(["date", live_sort_col], ascending=[False, False])
    base["date"] = base["date"].dt.strftime("%Y-%m-%d")
    base = apply_theme_risk_soft_experiment(base, theme_risk_soft_config)
    base = apply_risk_curve_experiments(base, risk_curve_experiment_config)
    base = apply_feature_candidate_sidecar(base, feature_candidate_config)
    base = _attach_market_columns(base, market_up, mkt_info, market_history)
    base["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "score_formula_version" not in base.columns:
        base["score_formula_version"] = score_formula_version

    if "score" in base.columns:
        base = base.drop(columns=["score"])

    return base


def _get_pg_table_columns(table: str) -> list[str]:
    if not get_engine:
        return []
    try:
        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table
                    ORDER BY ordinal_position
                    """
                ),
                {"table": table},
            ).fetchall()
        return [row[0] for row in rows]
    except Exception:
        logging.exception("Failed to inspect Postgres columns for %s", table)
        return []


def _get_sqlite_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        logging.exception("Failed to inspect sqlite columns for %s", table)
        return []
    return [row[1] for row in rows]


def _prepare_db_rows(df: pd.DataFrame, actual_columns: list[str]) -> pd.DataFrame:
    use_columns = [col for col in DAILY_RANKING_STORE_COLUMNS if col in actual_columns]
    out = df.copy()
    for col in use_columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[use_columns]


def _save_score_breakdown_debug(df: pd.DataFrame) -> None:
    ensure_data_dir()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    breakdown_cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "regime",
        "regime_reason",
        "weight_profile",
        "rank_final",
        "live_rank",
        "rank_before_theme",
        "final_score_before_theme",
        "final_score_v2_before_theme",
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "live_score",
        "live_score_source",
        "shadow_final_score_v3",
        "quality_flag",
        "quality_gate_applied",
        "quality_penalty_ratio",
        "shadow_quality_gate_applied",
        "shadow_quality_penalty_ratio",
        "shadow_final_score_quality_gate",
        "shadow_rank_quality_gate",
        "shadow_quality_risk_guard_penalty",
        "shadow_quality_risk_guard_applied",
        "shadow_final_score_quality_risk_guard",
        "shadow_rank_quality_risk_guard",
        "quality_gate_experiment",
        "return_score",
        "probability_score",
        "technical_score",
        "quality_score",
        "valuation_score",
        "theme_score",
        "theme_score_effective",
        "theme_overlay_mode",
        "theme_overlay_anchor",
        "theme_delta_raw",
        "theme_delta_positive",
        "theme_overlay_gain",
        "theme_overlay_cap",
        "theme_overlay_signed_component",
        "theme_overlay_positive_component",
        "theme_overlay_negative_component",
        "theme_overlay_applied",
        "theme_overlay_capped",
        "theme_overlay_soft_conf_gate",
        "shadow_theme_weight_raw",
        "shadow_theme_weight",
        "shadow_theme_weight_effective",
        "shadow_base_weight",
        "shadow_floor_applied",
        "shadow_theme_score_effective",
        "shadow_score_diff_v3",
        "shadow_rank_v3",
        "shadow_explain",
        "dominant_theme",
        "theme_confidence",
        "risk_penalty",
        "score_contribution_theme",
        "score_contribution_ret",
        "score_contribution_prob",
        "score_contribution_tech",
        "score_contribution_qual",
        "score_contribution_risk",
    ]
    breakdown_cols = [col for col in breakdown_cols if col in df.columns]
    df[breakdown_cols].to_csv(SCORE_BREAKDOWN_DEBUG_CSV, index=False, encoding="utf-8")
    logging.info("Saved score breakdown debug CSV: %s (rows=%d)", SCORE_BREAKDOWN_DEBUG_CSV.resolve(), len(df))


def _save_confidence_diagnostics(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    cols = [
        "date",
        "code",
        "name",
        "regime",
        "weight_profile",
        "live_score",
        "live_rank",
        "live_score_source",
        "final_score",
        "confidence_score_research",
        "confidence_score_operational",
        "confidence_score",
        "confidence_label_research",
        "confidence_label_operational",
        "confidence_label",
        "confidence_reason",
        "data_maturity_score",
        "model_reliability_score",
        "signal_agreement_score",
        "regime_fitness_score",
        "component_coverage_ratio",
        "fallback_count",
        "quality_score_confidence",
        "theme_score",
        "dominant_theme",
        "theme_confidence",
    ]
    cols = [col for col in cols if col in df.columns]
    df[cols].to_csv(CONFIDENCE_DIAGNOSTICS_CSV, index=False, encoding="utf-8")
    logging.info("Saved confidence diagnostics CSV: %s (rows=%d)", CONFIDENCE_DIAGNOSTICS_CSV.resolve(), len(df))


def _save_theme_impact_compare(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if "final_score_before_theme" not in df.columns or "final_score" not in df.columns:
        return
    compare = df.copy()
    compare["theme_score_delta"] = (
        pd.to_numeric(compare["final_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(compare["final_score_before_theme"], errors="coerce").fillna(0.0)
    )
    compare["theme_score_delta_v2"] = (
        pd.to_numeric(compare.get("final_score_v2"), errors="coerce").fillna(0.0)
        - pd.to_numeric(compare.get("final_score_v2_before_theme"), errors="coerce").fillna(0.0)
    )
    cols = [
        "date",
        "code",
        "name",
        "market",
        "sector",
        "regime",
        "weight_profile",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "final_score_before_theme",
        "final_score",
        "theme_score_delta",
        "final_score_v2_before_theme",
        "final_score_v2",
        "theme_score_delta_v2",
        "final_score_v3",
        "score_diff_v3",
        "v3_vs_v2_diff",
        "theme_overlay_mode",
        "theme_overlay_anchor",
        "theme_delta_raw",
        "theme_overlay_formula",
        "theme_delta_vs_base",
        "theme_delta_positive",
        "theme_positive_part",
        "theme_negative_part",
        "theme_overlay_gain",
        "theme_overlay_cap",
        "theme_overlay_signed_component",
        "theme_overlay_positive_component",
        "theme_overlay_negative_component",
        "theme_overlay_applied",
        "theme_overlay_capped",
        "theme_overlay_soft_conf_gate",
        "theme_uplift_applied",
        "theme_penalty_applied",
        "shadow_theme_weight_raw",
        "shadow_theme_weight",
        "shadow_theme_weight_effective",
        "shadow_base_weight",
        "shadow_floor_applied",
        "shadow_theme_score_effective",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "shadow_rank_v3",
        "shadow_explain",
        "rank_before_theme",
        "rank_final",
        "rank_v2",
    ]
    cols = [col for col in cols if col in compare.columns]
    compare[cols].to_csv(THEME_IMPACT_COMPARE_CSV, index=False, encoding="utf-8")
    logging.info("Saved theme impact compare CSV: %s (rows=%d)", THEME_IMPACT_COMPARE_CSV.resolve(), len(compare))


def _save_quality_gate_shadow(df: pd.DataFrame) -> None:
    if not QUALITY_GATE_FEATURE_CANDIDATE:
        return
    cols = [
        "date",
        "code",
        "name",
        "regime",
        "final_score",
        "shadow_final_score_quality_gate",
        "quality_flag",
        "quality_penalty_ratio",
        "quality_gate_applied",
        "shadow_quality_penalty_ratio",
        "shadow_quality_gate_applied",
        "quality_gate_experiment",
        "shadow_quality_risk_guard_penalty",
        "shadow_quality_risk_guard_applied",
        "shadow_final_score_quality_risk_guard",
        "shadow_rank_quality_risk_guard",
        "top_positive_factor",
        "top_negative_factor",
        "explain_text",
        "score_explain_summary",
        "score_explain_strengths",
        "score_explain_risks",
        "score_explain_confidence",
        "score_explain_regime",
    ]
    export = df.loc[:, [col for col in cols if col in df.columns]].copy()
    export.to_csv(QUALITY_GATE_SHADOW_CSV, index=False, encoding="utf-8-sig")
    logging.info("Saved quality gate shadow CSV: %s (rows=%d)", QUALITY_GATE_SHADOW_CSV.resolve(), len(export))


def export_before_after_comparison(df: pd.DataFrame) -> None:
    COMPARE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if "final_score" not in df.columns or "final_score_v2" not in df.columns:
        return

    compare = df.copy()
    compare["theme_score"] = pd.to_numeric(compare.get("theme_score"), errors="coerce").fillna(0.0)
    compare["theme_confidence"] = pd.to_numeric(compare.get("theme_confidence"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    compare["theme_score_effective"] = pd.to_numeric(compare.get("theme_score_effective"), errors="coerce").fillna(compare["theme_score"] * compare["theme_confidence"])
    dominant_theme = compare.get("dominant_theme")
    if dominant_theme is None:
        compare["dominant_theme"] = ""
    else:
        compare["dominant_theme"] = dominant_theme.fillna("").astype(str)
    compare["final_score"] = pd.to_numeric(compare.get("final_score"), errors="coerce").fillna(0.0)
    compare["final_score_v2"] = pd.to_numeric(compare.get("final_score_v2"), errors="coerce").fillna(0.0)
    compare["final_score_v3"] = pd.to_numeric(compare.get("final_score_v3"), errors="coerce").fillna(0.0)
    compare["theme_overlay_mode"] = compare.get("theme_overlay_mode", compare.get("theme_overlay_formula", pd.Series(pd.NA, index=compare.index))).astype("string")
    compare["theme_overlay_anchor"] = compare.get("theme_overlay_anchor", pd.Series(pd.NA, index=compare.index)).astype("string")
    compare["theme_delta_raw"] = pd.to_numeric(compare.get("theme_delta_raw"), errors="coerce")
    compare["theme_overlay_formula"] = compare.get("theme_overlay_formula", pd.Series(pd.NA, index=compare.index)).astype("string")
    compare["theme_delta_vs_base"] = pd.to_numeric(compare.get("theme_delta_vs_base"), errors="coerce")
    compare["theme_delta_positive"] = pd.to_numeric(compare.get("theme_delta_positive"), errors="coerce")
    compare["theme_positive_part"] = pd.to_numeric(compare.get("theme_positive_part"), errors="coerce")
    compare["theme_negative_part"] = pd.to_numeric(compare.get("theme_negative_part"), errors="coerce")
    compare["theme_overlay_gain"] = pd.to_numeric(compare.get("theme_overlay_gain"), errors="coerce")
    compare["theme_overlay_cap"] = pd.to_numeric(compare.get("theme_overlay_cap"), errors="coerce")
    compare["theme_overlay_signed_component"] = pd.to_numeric(compare.get("theme_overlay_signed_component"), errors="coerce")
    compare["theme_overlay_positive_component"] = pd.to_numeric(compare.get("theme_overlay_positive_component"), errors="coerce")
    compare["theme_overlay_negative_component"] = pd.to_numeric(compare.get("theme_overlay_negative_component"), errors="coerce")
    compare["theme_overlay_applied"] = pd.to_numeric(compare.get("theme_overlay_applied"), errors="coerce")
    compare["theme_overlay_capped"] = compare.get("theme_overlay_capped", False).fillna(False).astype(bool)
    compare["theme_overlay_soft_conf_gate"] = pd.to_numeric(compare.get("theme_overlay_soft_conf_gate"), errors="coerce")
    compare["theme_uplift_applied"] = compare.get("theme_uplift_applied", False).fillna(False).astype(bool)
    compare["theme_penalty_applied"] = compare.get("theme_penalty_applied", False).fillna(False).astype(bool)
    compare["shadow_theme_weight_raw"] = pd.to_numeric(compare.get("shadow_theme_weight_raw"), errors="coerce")
    compare["shadow_final_score_v3"] = pd.to_numeric(compare.get("shadow_final_score_v3"), errors="coerce")
    compare["shadow_score_diff_v3"] = pd.to_numeric(compare.get("shadow_score_diff_v3"), errors="coerce")
    compare["shadow_rank_v3"] = pd.to_numeric(compare.get("shadow_rank_v3"), errors="coerce")
    compare["shadow_theme_weight"] = pd.to_numeric(compare.get("shadow_theme_weight"), errors="coerce")
    compare["shadow_theme_weight_effective"] = pd.to_numeric(compare.get("shadow_theme_weight_effective"), errors="coerce")
    compare["shadow_base_weight"] = pd.to_numeric(compare.get("shadow_base_weight"), errors="coerce")
    compare["shadow_floor_applied"] = compare.get("shadow_floor_applied", False).fillna(False).astype(bool)
    compare["shadow_theme_score_effective"] = pd.to_numeric(compare.get("shadow_theme_score_effective"), errors="coerce")
    compare["shadow_explain"] = compare.get("shadow_explain", pd.Series("", index=compare.index)).fillna("").astype(str)
    compare["score_diff"] = compare["final_score_v2"] - compare["final_score"]
    compare["score_diff_v2"] = pd.to_numeric(compare.get("score_diff_v2"), errors="coerce").fillna(compare["final_score_v2"] - compare["final_score"])
    compare["score_diff_v3"] = pd.to_numeric(compare.get("score_diff_v3"), errors="coerce").fillna(compare["final_score_v3"] - compare["final_score"])
    compare["v3_vs_v2_diff"] = pd.to_numeric(compare.get("v3_vs_v2_diff"), errors="coerce").fillna(compare["final_score_v3"] - compare["final_score_v2"])
    compare["before_rank"] = compare.groupby("date")["final_score"].rank(method="first", ascending=False).astype(int)
    compare["before_rank_v2"] = compare.groupby("date")["final_score_v2"].rank(method="first", ascending=False).astype(int)
    compare["after_rank"] = compare.groupby("date")["final_score_v2"].rank(method="first", ascending=False).astype(int)
    compare["after_rank_v3"] = compare.groupby("date")["final_score_v3"].rank(method="first", ascending=False).astype(int)
    compare["baseline_rank"] = compare["before_rank"]
    compare["rank_change_shadow"] = compare["before_rank"] - compare["shadow_rank_v3"]
    compare["rank_shift"] = compare["before_rank"] - compare["after_rank"]
    compare["rank_shift_vs_base"] = compare["before_rank"] - compare["after_rank_v3"]
    compare["rank_shift_vs_v2"] = compare["before_rank_v2"] - compare["after_rank_v3"]

    compare_cols = [
        "date",
        "code",
        "name",
        "regime",
        "final_score",
        "theme_score",
        "final_score_v2",
        "score_diff",
        "dominant_theme",
        "theme_confidence",
        "before_rank",
        "after_rank",
        "rank_shift",
    ]
    compare_cols = [col for col in compare_cols if col in compare.columns]
    compare.loc[:, compare_cols].to_csv(BEFORE_AFTER_SCORE_COMPARE_CSV, index=False, encoding="utf-8")
    logging.info("Saved before/after score compare CSV: %s (rows=%d)", BEFORE_AFTER_SCORE_COMPARE_CSV.resolve(), len(compare))

    compare_v3_cols = [
        "date",
        "code",
        "name",
        "regime",
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "theme_overlay_mode",
        "theme_overlay_anchor",
        "theme_delta_raw",
        "theme_overlay_formula",
        "theme_delta_vs_base",
        "theme_delta_positive",
        "theme_positive_part",
        "theme_negative_part",
        "theme_overlay_gain",
        "theme_overlay_cap",
        "theme_overlay_signed_component",
        "theme_overlay_positive_component",
        "theme_overlay_negative_component",
        "theme_overlay_applied",
        "theme_overlay_capped",
        "theme_overlay_soft_conf_gate",
        "theme_uplift_applied",
        "theme_penalty_applied",
        "shadow_theme_weight_raw",
        "shadow_theme_weight",
        "shadow_theme_weight_effective",
        "shadow_base_weight",
        "shadow_floor_applied",
        "shadow_theme_score_effective",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "shadow_rank_v3",
        "shadow_explain",
        "baseline_rank",
        "before_rank",
        "rank_change_shadow",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
        "score_diff_v2",
        "score_diff_v3",
        "v3_vs_v2_diff",
        "dominant_theme",
    ]
    compare_v3_cols = [col for col in compare_v3_cols if col in compare.columns]
    compare.loc[:, compare_v3_cols].to_csv(BEFORE_AFTER_SCORE_COMPARE_V3_CSV, index=False, encoding="utf-8")
    logging.info("Saved before/after score compare v3 CSV: %s (rows=%d)", BEFORE_AFTER_SCORE_COMPARE_V3_CSV.resolve(), len(compare))

    latest_compare_date = compare["date"].dropna().astype(str).max() if "date" in compare.columns else None
    if latest_compare_date:
        top20_source = compare.loc[compare["date"].astype(str) == latest_compare_date].copy()
    else:
        top20_source = compare.copy()

    top20_before = top20_source.loc[top20_source["before_rank"] <= 20, ["date", "code"]].drop_duplicates()
    top20_after = top20_source.loc[top20_source["after_rank"] <= 20, ["date", "code"]].drop_duplicates()
    top20_after_v3 = top20_source.loc[top20_source["after_rank_v3"] <= 20, ["date", "code"]].drop_duplicates()
    top20_keys = pd.concat([top20_before, top20_after, top20_after_v3], ignore_index=True).drop_duplicates()
    top20_compare = compare.merge(top20_keys, on=["date", "code"], how="inner")
    top20_compare = top20_compare.sort_values(["date", "after_rank", "before_rank", "code"], ascending=[False, True, True, True])
    top20_cols = [
        "date",
        "before_rank",
        "after_rank",
        "rank_shift",
        "code",
        "name",
        "final_score",
        "final_score_v2",
        "score_diff",
        "dominant_theme",
        "regime",
    ]
    top20_cols = [col for col in top20_cols if col in top20_compare.columns]
    top20_compare.loc[:, top20_cols].to_csv(TOP20_BEFORE_AFTER_COMPARE_CSV, index=False, encoding="utf-8")
    logging.info("Saved top20 before/after compare CSV: %s (rows=%d)", TOP20_BEFORE_AFTER_COMPARE_CSV.resolve(), len(top20_compare))

    top20_v3_cols = [
        "date",
        "before_rank",
        "before_rank_v2",
        "after_rank_v3",
        "rank_shift_vs_base",
        "rank_shift_vs_v2",
        "code",
        "name",
        "regime",
        "final_score",
        "final_score_v2",
        "final_score_v3",
        "theme_overlay_mode",
        "theme_overlay_anchor",
        "theme_delta_raw",
        "theme_overlay_formula",
        "theme_delta_vs_base",
        "theme_delta_positive",
        "theme_positive_part",
        "theme_negative_part",
        "theme_overlay_gain",
        "theme_overlay_cap",
        "theme_overlay_signed_component",
        "theme_overlay_positive_component",
        "theme_overlay_negative_component",
        "theme_overlay_applied",
        "theme_overlay_capped",
        "theme_overlay_soft_conf_gate",
        "theme_uplift_applied",
        "theme_penalty_applied",
        "shadow_theme_weight_raw",
        "shadow_theme_weight",
        "shadow_theme_weight_effective",
        "shadow_base_weight",
        "shadow_floor_applied",
        "shadow_theme_score_effective",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "shadow_rank_v3",
        "shadow_explain",
        "baseline_rank",
        "rank_change_shadow",
        "theme_score",
        "theme_confidence",
        "theme_score_effective",
        "dominant_theme",
    ]
    rename_map = {
        "before_rank": "before_rank_final_score",
        "before_rank_v2": "before_rank_final_score_v2",
        "after_rank_v3": "after_rank_final_score_v3",
    }
    top20_v3 = top20_compare.loc[:, [col for col in top20_v3_cols if col in top20_compare.columns]].rename(columns=rename_map)
    top20_v3.to_csv(TOP20_BEFORE_AFTER_COMPARE_V3_CSV, index=False, encoding="utf-8")
    logging.info("Saved top20 before/after compare v3 CSV: %s (rows=%d)", TOP20_BEFORE_AFTER_COMPARE_V3_CSV.resolve(), len(top20_v3))


def export_theme_validation_report(df: pd.DataFrame) -> None:
    COMPARE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if df.empty:
        return
    report_df = sanitize_theme_columns(df)
    report_df["final_score"] = pd.to_numeric(report_df.get("final_score"), errors="coerce").fillna(0.0)
    report_df["final_score_v2"] = pd.to_numeric(report_df.get("final_score_v2"), errors="coerce").fillna(0.0)
    report_df["final_score_v3"] = pd.to_numeric(report_df.get("final_score_v3"), errors="coerce").fillna(0.0)
    report_df["score_diff_v3"] = pd.to_numeric(report_df.get("score_diff_v3"), errors="coerce").fillna(report_df["final_score_v3"] - report_df["final_score"])
    report_df["before_rank_final_score"] = report_df.groupby("date")["final_score"].rank(method="first", ascending=False).astype(int)
    report_df["before_rank_final_score_v2"] = report_df.groupby("date")["final_score_v2"].rank(method="first", ascending=False).astype(int)
    report_df["after_rank_final_score_v3"] = report_df.groupby("date")["final_score_v3"].rank(method="first", ascending=False).astype(int)
    report_df["rank_shift_vs_base"] = report_df["before_rank_final_score"] - report_df["after_rank_final_score_v3"]
    report_df["rank_shift_vs_v2"] = report_df["before_rank_final_score_v2"] - report_df["after_rank_final_score_v3"]
    latest_date = report_df["date"].dropna().astype(str).max() if "date" in report_df.columns else "NA"
    latest = report_df.loc[report_df["date"].astype(str) == latest_date].copy() if latest_date != "NA" else report_df.head(0).copy()

    stats = report_df["score_diff_v3"].agg(["mean", "median", "min", "max", "std"]).to_dict()
    low_conf = latest.loc[latest["theme_confidence"] < 0.30].copy()
    high_conf = latest.loc[latest["theme_confidence"] >= 0.30].copy()
    low_conf_delta = float(low_conf["score_diff_v3"].mean()) if not low_conf.empty else 0.0
    high_conf_delta = float(high_conf["score_diff_v3"].mean()) if not high_conf.empty else 0.0
    theme_bias = (
        latest.assign(dominant_theme=latest["dominant_theme"].replace("", "(none)"))
        .groupby("dominant_theme", as_index=False)
        .agg(
            top20_count=("after_rank_final_score_v3", lambda s: int((pd.to_numeric(s, errors="coerce") <= 20).sum())),
            mean_final_score_v3=("final_score_v3", "mean"),
            mean_theme_confidence=("theme_confidence", "mean"),
        )
        .sort_values(["top20_count", "mean_final_score_v3"], ascending=[False, False])
        .reset_index(drop=True)
    )
    max_top20_theme = int(theme_bias["top20_count"].max()) if not theme_bias.empty else 0

    lines = [
        "# Theme Confidence Overlay Validation",
        "",
        "## 변경 목적",
        "- final_score, final_score_v2를 유지하면서 theme_confidence를 반영한 final_score_v3 영향이 과도하지 않은지 확인한다.",
        "",
        "## 계산식",
        "- final_score_v2 = base_weight * final_score + theme_weight * theme_score",
        "- theme_score_effective = theme_score * theme_confidence",
        "- final_score_v3 = base_weight * final_score + theme_weight * theme_score_effective",
        "",
        "## regime별 가중치",
    ]
    for regime, weights in THEME_OVERLAY_WEIGHTS.items():
        lines.append(f"- {regime}: base_weight={weights['base_weight']:.2f}, theme_weight={weights['theme_weight']:.2f}")
    lines.extend([
        "",
        "## final_score / final_score_v2 / final_score_v3 차이 설명",
        "- final_score는 기존 체계를 그대로 유지한다.",
        "- final_score_v2는 theme_score를 직접 반영한다.",
        "- final_score_v3는 theme_confidence가 낮으면 theme_score_effective가 줄어들도록 설계했다.",
        "",
        "## score_diff_v3 요약 통계",
        f"- mean: {stats.get('mean', float('nan')):.4f}",
        f"- median: {stats.get('median', float('nan')):.4f}",
        f"- min: {stats.get('min', float('nan')):.4f}",
        f"- max: {stats.get('max', float('nan')):.4f}",
        f"- std: {stats.get('std', float('nan')):.4f}",
        "",
        "## top20 변동 요약",
    ])
    if not latest.empty:
        base_top20 = set(latest.loc[latest["before_rank_final_score"] <= 20, "code"].astype(str))
        v3_top20 = set(latest.loc[latest["after_rank_final_score_v3"] <= 20, "code"].astype(str))
        lines.append(f"- latest_date: {latest_date}")
        lines.append(f"- 신규 진입 수(v3 vs base): {len(v3_top20 - base_top20)}")
        lines.append(f"- 이탈 수(v3 vs base): {len(base_top20 - v3_top20)}")
    else:
        lines.append("- latest_date unavailable")
    lines.extend([
        "",
        "## theme_confidence가 낮은 종목 억제 여부",
        f"- low_confidence_mean_score_diff_v3 (<0.30): {low_conf_delta:.4f}",
        f"- high_confidence_mean_score_diff_v3 (>=0.30): {high_conf_delta:.4f}",
    ])
    if low_conf_delta <= high_conf_delta:
        lines.append("- 낮은 confidence 종목의 평균 overlay 효과가 더 작거나 같아서 억제 방향은 대체로 맞다.")
    else:
        lines.append("- 낮은 confidence 종목 억제 효과가 약하다. confidence scaling 또는 theme_weight 조정이 필요할 수 있다.")
    lines.extend([
        "",
        "## 특정 테마 쏠림 여부 간단 평가",
        f"- max_top20_count_by_theme: {max_top20_theme}",
    ])
    if max_top20_theme > 8:
        lines.append("- 특정 dominant_theme 쏠림이 높다. theme cap 또는 winsorize를 검토한다.")
    else:
        lines.append("- 현재 기준으로 extreme theme crowding 신호는 제한적이다.")
    if not theme_bias.empty:
        lines.append("")
        lines.append("## dominant_theme top summary")
        for _, row in theme_bias.head(10).iterrows():
            lines.append(
                f"- {row['dominant_theme']}: top20_count={int(row['top20_count'])}, "
                f"mean_final_score_v3={float(row['mean_final_score_v3']):.4f}, "
                f"mean_theme_confidence={float(row['mean_theme_confidence']):.4f}"
            )
    THEME_CONFIDENCE_OVERLAY_VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved theme confidence overlay validation report: %s", THEME_CONFIDENCE_OVERLAY_VALIDATION_MD.resolve())


def export_theme_guard_report(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    status = dict(LAST_THEME_GUARD_STATUS)
    latest = df.copy()
    if "date" in latest.columns and not latest.empty:
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        latest_date = latest["date"].dropna().astype(str).max()
        latest = latest.loc[latest["date"].astype(str) == latest_date].copy() if latest_date else latest.head(0).copy()
    latest = sanitize_theme_columns(latest)
    has_theme = latest["dominant_theme"].apply(_is_active_theme_label) if not latest.empty else pd.Series(dtype=bool)
    coverage_live = float(has_theme.mean()) if len(has_theme) else 0.0
    lines = [
        "# Ranking Builder Theme Guard Report",
        "",
        "## Gate Status",
        f"- overlay_enabled_env: {bool(status.get('overlay_enabled', False))}",
        f"- mode: {status.get('mode', THEME_OVERLAY_OFF)}",
        f"- operational_gate: {bool(status.get('operational', False))}",
        f"- applied: {bool(status.get('applied', False))}",
        f"- disable_reason: {status.get('disable_reason', '') or '(none)'}",
        "",
        "## Coverage Stats",
        f"- coverage_ratio_guard: {float(status.get('coverage_ratio', 0.0)):.4f}",
        f"- coverage_ratio_live_latest: {coverage_live:.4f}",
        f"- coverage_threshold: {float(status.get('coverage_threshold', THEME_OVERLAY_MIN_COVERAGE_RATIO_DEFAULT)):.4f}",
        f"- matched_rows: {int(status.get('matched_rows', 0))}",
        f"- base_rows: {int(status.get('base_rows', 0))}",
        f"- theme_row_count: {int(status.get('theme_row_count', 0))}",
        "",
        "## Date Check",
        f"- ranking_latest_date: {status.get('ranking_latest_date', 'NA')}",
        f"- theme_latest_date: {status.get('theme_latest_date', 'NA')}",
        f"- available_theme_dates: {status.get('available_theme_dates', [])}",
        f"- date_mismatch_test: {'PASS' if status.get('ranking_latest_date', 'NA') == status.get('theme_latest_date', 'NA') and bool(status.get('applied', False)) else 'DISABLE'}",
        "",
        "## Disable Case Tests",
        "- mode_mismatch: hard disable implemented",
        "- stale_date: hard disable implemented",
        "- low_coverage: hard disable implemented",
        "- empty_data: hard disable implemented",
        "",
        "## Runtime Notes",
        f"- source: {status.get('source', 'none')}",
        "- final_score baseline is unchanged; theme guard only controls overlay inputs used by final_score_v2/final_score_v3.",
        "- explain_text includes theme text only when an active dominant_theme exists.",
    ]
    THEME_GUARD_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved theme guard report: %s", THEME_GUARD_REPORT_MD.resolve())


def export_theme_debug_outputs(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    latest = df.copy()
    if latest.empty or "date" not in latest.columns:
        return

    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.loc[latest["date"] == latest["date"].max()].copy()
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest = sanitize_theme_columns(latest)

    rank_col = "live_rank" if "live_rank" in latest.columns else ("rank_final" if "rank_final" in latest.columns else "rank_v2")
    latest[rank_col] = pd.to_numeric(latest.get(rank_col), errors="coerce")
    top50 = latest.sort_values(rank_col, ascending=True).head(50).copy()
    top50["theme_before"] = np.where(pd.to_numeric(top50.get("raw_theme_score"), errors="coerce").notna(), "raw_present", "raw_missing")
    top50["theme_after"] = np.where(top50["theme_applied_flag"].fillna(False), "applied", "not_applied")
    top50_cols = [
        "date",
        "code",
        "name",
        rank_col,
        "dominant_theme",
        "raw_theme_score",
        "filtered_theme_score",
        "theme_confidence",
        "theme_threshold",
        "theme_applied_flag",
        "theme_debug_reason",
        "theme_before",
        "theme_after",
        "live_score",
        "live_score_source",
        "final_score",
        "final_score_v2",
        "final_score_v3",
    ]
    top50.loc[:, [c for c in top50_cols if c in top50.columns]].to_csv(DEBUG_THEME_TOP50_CSV, index=False, encoding="utf-8-sig")

    total_count = int(len(latest))
    none_ratio = float((~latest["dominant_theme"].astype(str).str.strip().apply(_is_active_theme_label)).mean()) if total_count else 0.0
    applied_ratio = float(latest["theme_applied_flag"].fillna(False).astype(float).mean()) if total_count else 0.0
    avg_theme_score = float(pd.to_numeric(latest.get("filtered_theme_score"), errors="coerce").fillna(0.0).mean()) if total_count else 0.0
    theme_threshold = float(pd.to_numeric(latest.get("theme_threshold"), errors="coerce").fillna(0.30).iloc[0]) if total_count else 0.30
    reason_counts = latest["theme_debug_reason"].replace("", "applied_or_not_filtered").value_counts(dropna=False).to_dict()

    lines = [
        f"latest_date: {latest['date'].max() if not latest.empty else 'NA'}",
        f"theme_applied_ratio: {applied_ratio:.4f}",
        f"none_ratio: {none_ratio:.4f}",
        f"avg_theme_score: {avg_theme_score:.4f}",
        f"theme_threshold: {theme_threshold:.2f}",
        "reason_counts:",
    ]
    for key, value in reason_counts.items():
        lines.append(f"- {key}: {int(value)}")
    DEBUG_THEME_SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logging.info("[theme-debug] theme_applied_ratio: %.2f%%", applied_ratio * 100.0)
    logging.info("[theme-debug] none_ratio: %.2f%%", none_ratio * 100.0)
    logging.info("[theme-debug] avg_theme_score: %.4f", avg_theme_score)
    logging.info("[theme-debug] theme_threshold: %.2f", theme_threshold)
    for key, value in reason_counts.items():
        logging.info("[theme-debug] reason=%s count=%d", key, int(value))


def export_theme_overlay_gate_debug() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    payload = dict(LAST_THEME_GATE_DEBUG)
    THEME_OVERLAY_GATE_DEBUG_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    overlay_requested = str(payload.get("enable_theme_overlay_raw", "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    validation_enabled = str(payload.get("enable_theme_validation_raw", "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    lines = [
        "# Theme Overlay Gate Debug",
        "",
        f"- overlay requested: {'true' if overlay_requested else 'false'}",
        f"- requested mode: {payload.get('theme_overlay_mode_requested', THEME_OVERLAY_OFF)}",
        f"- current mode: {payload.get('current_execution_mode', 'standalone')}",
        f"- requested execution mode: {payload.get('requested_execution_mode', THEME_OVERLAY_OFF)}",
        f"- resolved execution mode: {payload.get('resolved_execution_mode', THEME_OVERLAY_OFF)}",
        f"- fallback applied: {str(bool(payload.get('fallback_applied', False))).lower()}",
        f"- fallback reason: {payload.get('fallback_reason', '(none)')}",
        f"- validation enabled: {'true' if validation_enabled else 'false'}",
        f"- final gate result: {payload.get('overlay_gate_result', 'disabled')}",
        f"- disable reason: {payload.get('overlay_disable_reason', 'unknown')}",
        f"- theme weight source priority: {', '.join(payload.get('theme_weight_source_priority', []))}",
        f"- theme weight config available: {payload.get('theme_weight_config_available', {})}",
        "",
        "## Raw Values",
        f"- enable_theme_overlay_raw: {payload.get('enable_theme_overlay_raw', '0')}",
        f"- enable_theme_validation_raw: {payload.get('enable_theme_validation_raw', '0')}",
        f"- theme_weight_config_paths: {payload.get('theme_weight_config_paths', {})}",
    ]
    THEME_OVERLAY_GATE_DEBUG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved theme overlay gate debug: %s | %s", THEME_OVERLAY_GATE_DEBUG_JSON.resolve(), THEME_OVERLAY_GATE_DEBUG_MD.resolve())


def export_theme_overlay_mode_resolution() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    payload = dict(LAST_THEME_GATE_DEBUG)
    lines = [
        "# Theme Overlay Mode Resolution",
        "",
        "## Mode Definitions",
        f"- {THEME_OVERLAY_OFF}: theme calculation and score reflection are both disabled.",
        f"- {THEME_OVERLAY_SHADOW}: theme calculation is allowed, but final_score_v2/final_score_v3 do not reflect theme.",
        f"- {THEME_OVERLAY_OPERATIONAL}: theme calculation and score reflection are both enabled.",
        "",
        "## Environment Rules",
        "- ENABLE_THEME_OVERLAY != 1 -> resolved mode is off",
        "- THEME_OVERLAY_MODE=shadow -> resolved mode is shadow",
        "- THEME_OVERLAY_MODE=operational -> resolved mode is operational",
        "- Any other THEME_OVERLAY_MODE value -> invalid_mode fallback to off",
        "",
        "## Current Resolution",
        f"- requested_execution_mode: {payload.get('requested_execution_mode', THEME_OVERLAY_OFF)}",
        f"- resolved_execution_mode: {payload.get('resolved_execution_mode', THEME_OVERLAY_OFF)}",
        f"- fallback_applied: {bool(payload.get('fallback_applied', False))}",
        f"- fallback_reason: {payload.get('fallback_reason', '(none)')}",
        f"- current_execution_mode: {payload.get('current_execution_mode', THEME_OVERLAY_OFF)}",
        f"- overlay_gate_result: {payload.get('overlay_gate_result', 'disabled')}",
        f"- overlay_disable_reason: {payload.get('overlay_disable_reason', 'unknown')}",
    ]
    THEME_OVERLAY_MODE_RESOLUTION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Saved theme overlay mode resolution: %s", THEME_OVERLAY_MODE_RESOLUTION_MD.resolve())


def export_theme_overlay_shadow_preview(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    latest = df.copy()
    if latest.empty or "date" not in latest.columns:
        return
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.loc[latest["date"] == latest["date"].max()].copy()
    latest = sanitize_theme_columns(latest)
    latest["overlay_gate_result"] = str(LAST_THEME_GATE_DEBUG.get("overlay_gate_result", "disabled"))
    latest["overlay_disable_reason"] = str(LAST_THEME_GATE_DEBUG.get("overlay_disable_reason", "unknown"))
    latest["theme_overlay_mode"] = latest.get("theme_overlay_mode", latest.get("theme_overlay_formula", pd.Series(pd.NA, index=latest.index))).astype("string")
    out = latest.rename(columns={"code": "ticker", "final_score": "base_score"})
    cols = [
        "ticker",
        "name",
        "base_score",
        "raw_theme_score",
        "theme_score",
        "dominant_theme",
        "theme_confidence",
        "theme_applied_flag",
        "theme_overlay_mode",
        "theme_overlay_anchor",
        "theme_delta_raw",
        "theme_overlay_formula",
        "theme_delta_vs_base",
        "theme_delta_positive",
        "theme_positive_part",
        "theme_negative_part",
        "theme_overlay_gain",
        "theme_overlay_cap",
        "theme_overlay_signed_component",
        "theme_overlay_positive_component",
        "theme_overlay_negative_component",
        "theme_overlay_applied",
        "theme_overlay_capped",
        "theme_overlay_soft_conf_gate",
        "theme_uplift_applied",
        "theme_penalty_applied",
        "shadow_theme_weight_raw",
        "shadow_theme_weight_effective",
        "shadow_floor_applied",
        "shadow_theme_score_effective",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "shadow_rank_v3",
        "shadow_explain",
        "overlay_gate_result",
        "overlay_disable_reason",
    ]
    out.loc[:, [c for c in cols if c in out.columns]].to_csv(THEME_OVERLAY_SHADOW_PREVIEW_CSV, index=False, encoding="utf-8-sig")
    logging.info("Saved theme overlay shadow preview: %s", THEME_OVERLAY_SHADOW_PREVIEW_CSV.resolve())

    shadow_diff = pd.to_numeric(latest.get("shadow_score_diff_v3"), errors="coerce")
    mode_value = latest.get("theme_overlay_mode", pd.Series([SHADOW_THEME_OVERLAY_FORMULA])).iloc[0]
    if pd.isna(mode_value):
        mode_name = SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR
    else:
        mode_name = str(mode_value).strip().lower() or SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR
    mode_suffix = mode_name.replace("-", "_").replace(" ", "_")
    preview_mode_path = DATA_DIR / f"theme_overlay_shadow_preview_{mode_suffix}.csv"
    debug_mode_path = DATA_DIR / f"theme_overlay_debug_{mode_suffix}.csv"
    summary_mode_md = DATA_DIR / f"theme_overlay_mode_summary_{mode_suffix}.md"
    out.loc[:, [c for c in cols if c in out.columns]].to_csv(preview_mode_path, index=False, encoding="utf-8-sig")
    out.loc[:, [c for c in cols if c in out.columns]].to_csv(debug_mode_path, index=False, encoding="utf-8-sig")
    payload = {
        "latest_date": latest["date"].max().strftime("%Y-%m-%d") if not latest.empty else "NA",
        "mode": LAST_THEME_GUARD_STATUS.get("mode", THEME_OVERLAY_OFF),
        "shadow_formula": mode_name,
        "shadow_theme_weight_floor": float(_clip_theme_weight(SHADOW_THEME_WEIGHT_FLOOR)),
        "shadow_gain": float(SHADOW_THEME_OVERLAY_GAIN),
        "shadow_cap": float(SHADOW_THEME_OVERLAY_CAP),
        "shadow_baseline_anchor": str(SHADOW_THEME_OVERLAY_BASELINE_ANCHOR),
        "shadow_soft_conf_enabled": _parse_bool_like(SHADOW_THEME_OVERLAY_SOFT_CONF_ENABLED_RAW, True),
        "shadow_negative_penalty_ratio": float(max(SHADOW_THEME_NEGATIVE_PENALTY_RATIO, 0.0)),
        "shadow_uplift_threshold": float(max(SHADOW_THEME_UPLIFT_THRESHOLD, 0.0)),
        "overlay_gate_result": str(LAST_THEME_GATE_DEBUG.get("overlay_gate_result", "disabled")),
        "overlay_disable_reason": str(LAST_THEME_GATE_DEBUG.get("overlay_disable_reason", "unknown")),
        "row_count": int(len(latest)),
        "shadow_signal_count": int(pd.to_numeric(latest.get("shadow_theme_score_effective"), errors="coerce").fillna(0.0).gt(0.0).sum()),
        "shadow_floor_applied_count": int(latest.get("shadow_floor_applied", False).fillna(False).astype(bool).sum()) if "shadow_floor_applied" in latest.columns else 0,
        "shadow_rank_changed_count": int(pd.to_numeric(latest.get("shadow_rank_v3"), errors="coerce").fillna(pd.to_numeric(latest.get("rank_final"), errors="coerce")).ne(pd.to_numeric(latest.get("rank_final"), errors="coerce")).sum()) if "shadow_rank_v3" in latest.columns else 0,
        "overlay_applied_rows": int(pd.to_numeric(latest.get("theme_overlay_applied"), errors="coerce").fillna(0.0).ne(0.0).sum()) if "theme_overlay_applied" in latest.columns else 0,
        "positive_overlay_rows": int(pd.to_numeric(latest.get("theme_overlay_positive_component"), errors="coerce").fillna(0.0).gt(0.0).sum()) if "theme_overlay_positive_component" in latest.columns else 0,
        "negative_overlay_rows": int(pd.to_numeric(latest.get("theme_overlay_negative_component"), errors="coerce").fillna(0.0).lt(0.0).sum()) if "theme_overlay_negative_component" in latest.columns else 0,
        "capped_rows": int(latest.get("theme_overlay_capped", False).fillna(False).astype(bool).sum()) if "theme_overlay_capped" in latest.columns else 0,
        "shadow_uplift_applied_count": int(latest.get("theme_uplift_applied", False).fillna(False).astype(bool).sum()) if "theme_uplift_applied" in latest.columns else 0,
        "shadow_penalty_applied_count": int(latest.get("theme_penalty_applied", False).fillna(False).astype(bool).sum()) if "theme_penalty_applied" in latest.columns else 0,
        "avg_overlay": float(pd.to_numeric(latest.get("theme_overlay_applied"), errors="coerce").fillna(0.0).mean()) if "theme_overlay_applied" in latest.columns else 0.0,
        "p95_overlay": float(pd.to_numeric(latest.get("theme_overlay_applied"), errors="coerce").fillna(0.0).quantile(0.95)) if "theme_overlay_applied" in latest.columns else 0.0,
        "top20_affected_rows": int(latest.loc[pd.to_numeric(latest.get("rank_final"), errors="coerce").le(20), "theme_overlay_applied"].pipe(pd.to_numeric, errors="coerce").fillna(0.0).ne(0.0).sum()) if "rank_final" in latest.columns and "theme_overlay_applied" in latest.columns else 0,
        "near_top20_affected_rows": int(latest.loc[pd.to_numeric(latest.get("rank_final"), errors="coerce").between(21, 40), "theme_overlay_applied"].pipe(pd.to_numeric, errors="coerce").fillna(0.0).ne(0.0).sum()) if "rank_final" in latest.columns and "theme_overlay_applied" in latest.columns else 0,
        "shadow_uplift_p50": float(shadow_diff.quantile(0.50)) if shadow_diff is not None and shadow_diff.notna().any() else 0.0,
        "shadow_uplift_p90": float(shadow_diff.quantile(0.90)) if shadow_diff is not None and shadow_diff.notna().any() else 0.0,
        "shadow_uplift_max": float(shadow_diff.max()) if shadow_diff is not None and shadow_diff.notna().any() else 0.0,
    }
    THEME_OVERLAY_SHADOW_SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved theme overlay shadow summary: %s", THEME_OVERLAY_SHADOW_SUMMARY_JSON.resolve())
    summary_lines = [
        "# Theme Overlay Shadow Mode Update",
        "",
        "## Current Shadow Mode",
        f"- mode: {mode_name}",
        f"- baseline_anchor: {payload['shadow_baseline_anchor']}",
        f"- gain: {payload['shadow_gain']:.3f}",
        f"- cap: {payload['shadow_cap']:.3f}",
        f"- soft_conf_enabled: {payload['shadow_soft_conf_enabled']}",
        "",
        "## Intent",
        f"- {SHADOW_THEME_OVERLAY_FORMULA_SYMMETRIC_FLOOR}: signed blend, useful as current baseline but can create negative displacement.",
        f"- {SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY}: positive uplift only, no negative penalty.",
        f"- {SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_CAPPED}: positive uplift only with a hard cap.",
        f"- {SHADOW_THEME_OVERLAY_FORMULA_POSITIVE_ONLY_SOFT_CONF}: capped positive uplift with confidence soft gate.",
        "",
        "## Acceptance Priority",
        "- First watch `large_negative_displacement_count` and `indirect_rank_gain_count` to confirm side-effects are shrinking.",
        "- Then watch `direct_uplift_count`, `Theme Lift Effect`, and `Near-top20 Entry Quality`.",
        "",
        "## Latest Summary",
        f"- row_count: {payload['row_count']}",
        f"- overlay_applied_rows: {payload['overlay_applied_rows']}",
        f"- positive_overlay_rows: {payload['positive_overlay_rows']}",
        f"- negative_overlay_rows: {payload['negative_overlay_rows']}",
        f"- capped_rows: {payload['capped_rows']}",
        f"- shadow_rank_changed_count: {payload['shadow_rank_changed_count']}",
        f"- top20_affected_rows: {payload['top20_affected_rows']}",
        f"- near_top20_affected_rows: {payload['near_top20_affected_rows']}",
        f"- shadow_uplift_p90: {payload['shadow_uplift_p90']:.4f}",
        f"- shadow_uplift_max: {payload['shadow_uplift_max']:.4f}",
    ]
    THEME_OVERLAY_SHADOW_MODE_UPDATE_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    summary_mode_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def export_theme_risk_soft_outputs(df: pd.DataFrame, config: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config or {
        "enabled": False,
        "soft_factor": RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT,
        "min_score": RISK_PENALTY_THEME_MIN_SCORE_DEFAULT,
        "min_confidence": RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT,
    }
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in [
        "theme_score",
        "theme_confidence",
        "risk_penalty_base",
        "risk_penalty_effective",
        "risk_penalty_soft_delta",
        "final_score_baseline",
        "final_score_theme_risk_soft",
        "rank_baseline",
        "rank_theme_risk_soft",
        "rank_change_theme_risk_soft",
    ]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["theme_risk_soft_enabled"] = out.get("theme_risk_soft_enabled", False)
    out["theme_risk_soft_applied"] = out.get("theme_risk_soft_applied", False)
    out["theme_risk_soft_reason"] = out.get("theme_risk_soft_reason", "disabled")
    out["dominant_theme"] = out.get("dominant_theme", "").fillna("").astype(str)

    soft_ranked = out.sort_values(["date", "rank_theme_risk_soft", "code"], ascending=[False, True, True]).copy()
    soft_ranked.to_csv(RANKING_THEME_RISK_SOFT_CSV, index=False, encoding="utf-8")

    compare_cols = [
        "date",
        "code",
        "name",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "risk_penalty_base",
        "risk_penalty_effective",
        "risk_penalty_soft_delta",
        "final_score_baseline",
        "final_score_theme_risk_soft",
        "rank_baseline",
        "rank_theme_risk_soft",
        "rank_change_theme_risk_soft",
        "theme_risk_soft_applied",
        "theme_risk_soft_reason",
    ]
    compare_cols = [col for col in compare_cols if col in out.columns]
    out.loc[:, compare_cols].to_csv(THEME_RISK_SOFT_COMPARE_CSV, index=False, encoding="utf-8")

    latest_date = out["date"].astype(str).max() if "date" in out.columns and not out.empty else "NA"
    latest = out.loc[out["date"].astype(str) == latest_date].copy() if latest_date != "NA" else out.head(0).copy()
    applied = latest.loc[latest["theme_risk_soft_applied"].fillna(False).astype(bool)].copy()
    avg_penalty_delta = float(applied["risk_penalty_soft_delta"].mean()) if not applied.empty else 0.0
    avg_score_gain = float((applied["final_score_theme_risk_soft"] - applied["final_score_baseline"]).mean()) if not applied.empty else 0.0
    top_lifters = latest.sort_values(["rank_change_theme_risk_soft", "risk_penalty_soft_delta"], ascending=[False, False]).head(10)
    near_top20 = latest.loc[
        latest["rank_baseline"].between(21, 40)
        & latest["rank_change_theme_risk_soft"].gt(0)
        & latest["dominant_theme"].str.strip().ne("")
    ].sort_values(["rank_theme_risk_soft", "rank_change_theme_risk_soft"], ascending=[True, False])
    base_top20 = set(latest.loc[latest["rank_baseline"] <= 20, "code"].astype(str))
    exp_top20 = set(latest.loc[latest["rank_theme_risk_soft"] <= 20, "code"].astype(str))
    theme_summary = (
        latest.assign(dominant_theme=latest["dominant_theme"].replace("", "(none)"))
        .groupby("dominant_theme", as_index=False)
        .agg(avg_rank_change=("rank_change_theme_risk_soft", "mean"), stock_count=("code", "count"))
        .sort_values(["avg_rank_change", "stock_count"], ascending=[False, False])
    )
    max_abs_rank_change = float(latest["rank_change_theme_risk_soft"].abs().max()) if not latest.empty else 0.0
    aggressive_flag = max_abs_rank_change > 15 or len(exp_top20 - base_top20) > 3

    lines = [
        "# Theme Risk Soft Validation",
        "",
        "## Scope",
        "- This report covers an experiment-only sidecar score.",
        "- It is not the live production `final_score` used in daily operations.",
        "- Compatibility columns such as `valuation_score` may still appear here for experiment continuity.",
        "",
        "## Config",
        f"- feature_flag={bool(cfg.get('enabled', False))}",
        f"- soft_factor={float(cfg.get('soft_factor', RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT)):.3f}",
        f"- min_score={float(cfg.get('min_score', RISK_PENALTY_THEME_MIN_SCORE_DEFAULT)):.2f}",
        f"- min_confidence={float(cfg.get('min_confidence', RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT)):.2f}",
        "",
        "## Latest Summary",
        f"- latest_date={latest_date}",
        f"- applied_stock_count={int(len(applied))}",
        f"- average_risk_penalty_delta={avg_penalty_delta:.4f}",
        f"- average_score_gain={avg_score_gain:.4f}",
        f"- top20_new_entries={len(exp_top20 - base_top20)}",
        f"- top20_exits={len(base_top20 - exp_top20)}",
        "",
        "## Top10 Rank Lifters",
    ]
    for row in top_lifters.itertuples(index=False):
        lines.append(
            f"- {row.code} {row.name}: baseline_rank={int(row.rank_baseline)}, exp_rank={int(row.rank_theme_risk_soft)}, "
            f"rank_change={int(row.rank_change_theme_risk_soft)}, theme={row.dominant_theme or '(none)'}, "
            f"risk_delta={float(row.risk_penalty_soft_delta):.4f}"
        )
    lines.extend(["", "## Near Top20 Movers"])
    if near_top20.empty:
        lines.append("- none")
    else:
        for row in near_top20.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: baseline_rank={int(row.rank_baseline)}, exp_rank={int(row.rank_theme_risk_soft)}, "
                f"rank_change={int(row.rank_change_theme_risk_soft)}, theme={row.dominant_theme}"
            )
    lines.extend(["", "## Dominant Theme Average Rank Change"])
    for row in theme_summary.head(12).itertuples(index=False):
        lines.append(f"- {row.dominant_theme}: avg_rank_change={float(row.avg_rank_change):.2f}, stock_count={int(row.stock_count)}")
    lines.extend([
        "",
        "## Assessment",
        "- " + ("experiment looks aggressive" if aggressive_flag else "experiment looks controlled"),
    ])
    THEME_RISK_SOFT_VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(
        "Theme risk soft: enabled=%s factor=%.3f min_score=%.2f min_conf=%.2f applied_count=%d new_top20=%d compare_csv=%s",
        bool(cfg.get("enabled", False)),
        float(cfg.get("soft_factor", RISK_PENALTY_THEME_SOFT_FACTOR_DEFAULT)),
        float(cfg.get("min_score", RISK_PENALTY_THEME_MIN_SCORE_DEFAULT)),
        float(cfg.get("min_confidence", RISK_PENALTY_THEME_MIN_CONFIDENCE_DEFAULT)),
        int(len(applied)),
        len(exp_top20 - base_top20),
        THEME_RISK_SOFT_COMPARE_CSV.resolve(),
    )


def export_risk_curve_experiment_outputs(df: pd.DataFrame, config: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config or {
        "enabled": False,
        "exp_a_threshold": EXP_A_THRESHOLD_DEFAULT,
        "exp_a_slope_ratio": EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT,
        "exp_b_delayed_reach_factor": EXP_B_DELAYED_REACH_FACTOR_DEFAULT,
        "penalty_cap": PENALTY_CAP_DEFAULT,
    }
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = [
        "pred_mdd_mix",
        "theme_score",
        "theme_confidence",
        "risk_penalty_base",
        "risk_penalty_exp_a",
        "risk_penalty_exp_b",
        "risk_penalty_delta_exp_a",
        "risk_penalty_delta_exp_b",
        "final_score_baseline",
        "final_score_exp_a",
        "final_score_exp_b",
        "rank_baseline",
        "rank_exp_a",
        "rank_exp_b",
        "rank_change_exp_a",
        "rank_change_exp_b",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["dominant_theme"] = out.get("dominant_theme", "").fillna("").astype(str)

    compare_cols = [
        "date",
        "code",
        "name",
        "pred_mdd_mix",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "risk_penalty_base",
        "risk_penalty_exp_a",
        "risk_penalty_exp_b",
        "risk_penalty_delta_exp_a",
        "risk_penalty_delta_exp_b",
        "final_score_baseline",
        "final_score_exp_a",
        "final_score_exp_b",
        "rank_baseline",
        "rank_exp_a",
        "rank_exp_b",
        "rank_change_exp_a",
        "rank_change_exp_b",
        "explain_base",
        "explain_exp_a",
        "explain_exp_b",
    ]
    compare_cols = [col for col in compare_cols if col in out.columns]
    out.loc[:, compare_cols].to_csv(THEME_RISK_CURVE_COMPARE_CSV, index=False, encoding="utf-8")

    latest_date = out["date"].astype(str).max() if "date" in out.columns and not out.empty else "NA"
    latest = out.loc[out["date"].astype(str) == latest_date].copy() if latest_date != "NA" else out.head(0).copy()
    tol = 1e-9

    def _cap18_count(series: pd.Series) -> int:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return int((s >= (float(cfg.get("penalty_cap", PENALTY_CAP_DEFAULT)) - tol)).sum())

    def _unique_count(series: pd.Series) -> int:
        return int(pd.to_numeric(series, errors="coerce").fillna(0.0).round(6).nunique())

    def _near_top20(df_in: pd.DataFrame, rank_col: str, change_col: str) -> pd.DataFrame:
        return df_in.loc[
            df_in["rank_baseline"].between(21, 40)
            & pd.to_numeric(df_in[change_col], errors="coerce").fillna(0.0).gt(0)
        ].sort_values([rank_col, change_col], ascending=[True, False])

    near_a = _near_top20(latest, "rank_exp_a", "rank_change_exp_a")
    near_b = _near_top20(latest, "rank_exp_b", "rank_change_exp_b")
    near_union = pd.concat(
        [
            near_a.assign(experiment="exp_a"),
            near_b.assign(experiment="exp_b"),
        ],
        ignore_index=True,
    )
    near_cols = [
        "experiment",
        "date",
        "code",
        "name",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "rank_baseline",
        "rank_exp_a",
        "rank_exp_b",
        "rank_change_exp_a",
        "rank_change_exp_b",
    ]
    near_union.loc[:, [c for c in near_cols if c in near_union.columns]].to_csv(
        THEME_RISK_CURVE_NEAR_TOP20_CSV, index=False, encoding="utf-8"
    )

    base_top20 = set(latest.loc[latest["rank_baseline"] <= 20, "code"].astype(str))
    exp_a_top20 = set(latest.loc[latest["rank_exp_a"] <= 20, "code"].astype(str))
    exp_b_top20 = set(latest.loc[latest["rank_exp_b"] <= 20, "code"].astype(str))

    theme_avg = (
        latest.assign(dominant_theme=latest["dominant_theme"].replace("", "(none)"))
        .groupby("dominant_theme", as_index=False)
        .agg(
            avg_rank_change_exp_a=("rank_change_exp_a", "mean"),
            avg_rank_change_exp_b=("rank_change_exp_b", "mean"),
            stock_count=("code", "count"),
        )
        .sort_values(["stock_count", "dominant_theme"], ascending=[False, True])
    )

    def _top_movers(df_in: pd.DataFrame, rank_col: str, change_col: str) -> list[str]:
        lines = []
        cols = ["code", "name", "dominant_theme", "rank_baseline", rank_col, change_col, "theme_score", "theme_confidence"]
        for row in df_in.sort_values([change_col, rank_col], ascending=[False, True]).head(10)[cols].itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: theme={row.dominant_theme or '(none)'}, baseline_rank={int(row.rank_baseline)}, "
                f"exp_rank={int(getattr(row, rank_col))}, rank_change={int(getattr(row, change_col))}, "
                f"theme_score={float(row.theme_score):.2f}, theme_confidence={float(row.theme_confidence):.2f}"
            )
        return lines or ["- none"]

    lines = [
        "# Theme Risk Curve Validation",
        "",
        "## 1. Baseline Summary",
        f"- latest_date={latest_date}",
        f"- ranking_row_count={int(len(latest))}",
        f"- baseline_cap18_count={_cap18_count(latest['risk_penalty_base'])}",
        f"- baseline_unique_penalty_count={_unique_count(latest['risk_penalty_base'])}",
        f"- baseline_near_top20_mover_count=0",
        "",
        "## 2. Exp-A Summary",
        f"- threshold={float(cfg.get('exp_a_threshold', EXP_A_THRESHOLD_DEFAULT)):.2f}",
        f"- softened_slope_ratio={float(cfg.get('exp_a_slope_ratio', EXP_A_SOFTENED_SLOPE_RATIO_DEFAULT)):.2f}",
        f"- exp_a_cap18_count={_cap18_count(latest['risk_penalty_exp_a'])}",
        f"- exp_a_unique_penalty_count={_unique_count(latest['risk_penalty_exp_a'])}",
        f"- exp_a_near_top20_mover_count={int(len(near_a))}",
        f"- exp_a_top20_new_entries={len(exp_a_top20 - base_top20)}",
        f"- exp_a_top20_exits={len(base_top20 - exp_a_top20)}",
        f"- exp_a_avg_rank_change={float(pd.to_numeric(latest['rank_change_exp_a'], errors='coerce').fillna(0.0).mean()):.4f}",
        f"- exp_a_median_rank_change={float(pd.to_numeric(latest['rank_change_exp_a'], errors='coerce').fillna(0.0).median()):.4f}",
        "",
        "## 3. Exp-B Summary",
        f"- delayed_reach_factor={float(cfg.get('exp_b_delayed_reach_factor', EXP_B_DELAYED_REACH_FACTOR_DEFAULT)):.2f}",
        f"- exp_b_cap18_count={_cap18_count(latest['risk_penalty_exp_b'])}",
        f"- exp_b_unique_penalty_count={_unique_count(latest['risk_penalty_exp_b'])}",
        f"- exp_b_near_top20_mover_count={int(len(near_b))}",
        f"- exp_b_top20_new_entries={len(exp_b_top20 - base_top20)}",
        f"- exp_b_top20_exits={len(base_top20 - exp_b_top20)}",
        f"- exp_b_avg_rank_change={float(pd.to_numeric(latest['rank_change_exp_b'], errors='coerce').fillna(0.0).mean()):.4f}",
        f"- exp_b_median_rank_change={float(pd.to_numeric(latest['rank_change_exp_b'], errors='coerce').fillna(0.0).median()):.4f}",
        "",
        "## 4. Cap 18 clustering comparison",
        f"- baseline_cap18_count={_cap18_count(latest['risk_penalty_base'])}",
        f"- exp_a_cap18_count={_cap18_count(latest['risk_penalty_exp_a'])}",
        f"- exp_b_cap18_count={_cap18_count(latest['risk_penalty_exp_b'])}",
        "- Interpretation: lower cap18_count with stable top20 change is preferred.",
        "",
        "## 5. Near Top20 Movers",
        "### Exp-A",
        *_top_movers(near_a, 'rank_exp_a', 'rank_change_exp_a'),
        "### Exp-B",
        *_top_movers(near_b, 'rank_exp_b', 'rank_change_exp_b'),
        "",
        "## 6. Top 20 impact",
        f"- exp_a_new_entries={len(exp_a_top20 - base_top20)} exp_a_exits={len(base_top20 - exp_a_top20)}",
        f"- exp_b_new_entries={len(exp_b_top20 - base_top20)} exp_b_exits={len(base_top20 - exp_b_top20)}",
        "- Excessive top20 churn is a negative sign.",
        "",
        "## 7. Diagnosis",
        f"- cap18_count relaxed: baseline={_cap18_count(latest['risk_penalty_base'])}, exp_a={_cap18_count(latest['risk_penalty_exp_a'])}, exp_b={_cap18_count(latest['risk_penalty_exp_b'])}",
        f"- near-top20 resolution: exp_a={int(len(near_a))}, exp_b={int(len(near_b))}",
        f"- most stable candidate is {'exp_a' if (len(exp_a_top20 - base_top20) <= len(exp_b_top20 - base_top20) and len(near_a) >= len(near_b)) else 'exp_b'} based on current sidecar outputs",
        "",
        "## 8. Final Recommendation",
    ]

    exp_a_cap = _cap18_count(latest["risk_penalty_exp_a"])
    exp_b_cap = _cap18_count(latest["risk_penalty_exp_b"])
    baseline_cap = _cap18_count(latest["risk_penalty_base"])
    exp_a_new = len(exp_a_top20 - base_top20)
    exp_a_exit = len(base_top20 - exp_a_top20)
    exp_b_new = len(exp_b_top20 - base_top20)
    exp_b_exit = len(base_top20 - exp_b_top20)
    if exp_b_cap < baseline_cap and exp_b_new == 0 and exp_b_exit == 0:
        recommendation = "promote exp_b to feature-flag candidate"
    elif exp_a_cap < baseline_cap and exp_a_new <= 1 and exp_a_exit <= 1:
        recommendation = "promote exp_a to feature-flag candidate"
    elif exp_a_cap == baseline_cap and exp_b_cap == baseline_cap:
        recommendation = "redesign needed before promotion"
    else:
        recommendation = "keep baseline"
    lines.append(f"- {recommendation}")
    lines.append("")
    lines.append("## Dominant Theme Average Rank Change")
    for row in theme_avg.head(12).itertuples(index=False):
        lines.append(
            f"- {row.dominant_theme}: avg_rank_change_exp_a={float(row.avg_rank_change_exp_a):.2f}, "
            f"avg_rank_change_exp_b={float(row.avg_rank_change_exp_b):.2f}, stock_count={int(row.stock_count)}"
        )

    THEME_RISK_CURVE_VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(
        "Risk curve experiment: enabled=%s exp_a_cap18=%d exp_b_cap18=%d exp_a_near_top20=%d exp_b_near_top20=%d compare_csv=%s",
        bool(cfg.get("enabled", False)),
        _cap18_count(latest["risk_penalty_exp_a"]),
        _cap18_count(latest["risk_penalty_exp_b"]),
        int(len(near_a)),
        int(len(near_b)),
        THEME_RISK_CURVE_COMPARE_CSV.resolve(),
    )


def export_feature_candidate_reports(df: pd.DataFrame, config: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config or {
        "candidate": "none",
        "enabled": False,
        "exp_b_delayed_cap_reach_factor": EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT,
        "exp_b_delayed_cap_max_penalty_ratio": EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT,
        "exp_b_delayed_cap_apply_regimes": [],
        "exp_b_delayed_cap_theme_only": EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT,
    }
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["dominant_theme"] = out.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    out["regime"] = out.get("regime", "").fillna("").astype(str)
    out["theme_score"] = pd.to_numeric(out.get("theme_score"), errors="coerce").fillna(0.0)
    out["theme_confidence"] = pd.to_numeric(out.get("theme_confidence"), errors="coerce").fillna(0.0)
    out["has_theme_flag"] = pd.to_numeric(out.get("has_theme_flag"), errors="coerce").fillna(0).astype(int)

    latest_date = out["date"].astype(str).max() if "date" in out.columns and not out.empty else "NA"
    latest = out.loc[out["date"].astype(str) == latest_date].copy() if latest_date != "NA" else out.head(0).copy()

    latest["symbol"] = latest.get("code", "").astype(str)
    latest["baseline_final_score"] = pd.to_numeric(latest.get("candidate_baseline_final_score"), errors="coerce").fillna(0.0)
    latest["candidate_final_score"] = pd.to_numeric(latest.get("candidate_final_score"), errors="coerce").fillna(0.0)
    latest["score_delta"] = pd.to_numeric(latest.get("candidate_score_delta"), errors="coerce").fillna(0.0)
    latest["baseline_rank"] = pd.to_numeric(latest.get("candidate_baseline_rank"), errors="coerce").fillna(0).astype(int)
    latest["candidate_rank"] = pd.to_numeric(latest.get("candidate_rank"), errors="coerce").fillna(0).astype(int)
    latest["rank_delta"] = pd.to_numeric(latest.get("candidate_rank_delta"), errors="coerce").fillna(0).astype(int)
    latest["baseline_risk_penalty"] = pd.to_numeric(latest.get("candidate_baseline_risk_penalty"), errors="coerce").fillna(0.0)
    latest["candidate_risk_penalty"] = pd.to_numeric(latest.get("candidate_risk_penalty"), errors="coerce").fillna(0.0)
    latest["penalty_delta"] = pd.to_numeric(latest.get("candidate_penalty_delta"), errors="coerce").fillna(0.0)
    latest["candidate_applied_flag"] = latest.get("candidate_applied_flag", False).fillna(False).astype(bool)
    latest["candidate_reason"] = latest.get("candidate_reason", "candidate_disabled").fillna("candidate_disabled").astype(str)

    base_top20 = set(latest.loc[latest["baseline_rank"] <= 20, "symbol"].astype(str))
    cand_top20 = set(latest.loc[latest["candidate_rank"] <= 20, "symbol"].astype(str))

    latest["top20_status"] = np.where(
        latest["symbol"].astype(str).isin(cand_top20 - base_top20),
        "entered",
        np.where(
            latest["symbol"].astype(str).isin(base_top20 - cand_top20),
            "exited",
            np.where(
                latest["symbol"].astype(str).isin(base_top20 & cand_top20),
                "kept",
                "outside",
            ),
        ),
    )

    baseline_band = latest["baseline_rank"].between(15, 30)
    candidate_band = latest["candidate_rank"].between(15, 30)
    latest["near_top20_band"] = np.select(
        [baseline_band & candidate_band, baseline_band, candidate_band],
        ["both", "baseline_only", "candidate_only"],
        default="outside",
    )

    main_cols = [
        "symbol",
        "name",
        "regime",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "has_theme_flag",
        "baseline_final_score",
        "candidate_final_score",
        "score_delta",
        "baseline_rank",
        "candidate_rank",
        "rank_delta",
        "baseline_risk_penalty",
        "candidate_risk_penalty",
        "penalty_delta",
        "candidate_applied_flag",
        "candidate_reason",
        "candidate_explain",
    ]
    latest.loc[:, [c for c in main_cols if c in latest.columns]].to_csv(FEATURE_CANDIDATE_EXP_B_CSV, index=False, encoding="utf-8")

    top20_cols = [
        "symbol",
        "baseline_rank",
        "candidate_rank",
        "top20_status",
        "score_delta",
        "rank_delta",
        "has_theme_flag",
        "dominant_theme",
        "candidate_applied_flag",
        "candidate_reason",
    ]
    latest.loc[:, [c for c in top20_cols if c in latest.columns]].to_csv(FEATURE_CANDIDATE_EXP_B_TOP20_DIFF_CSV, index=False, encoding="utf-8")

    near_df = latest.loc[latest["near_top20_band"].ne("outside")].copy()
    near_cols = [
        "symbol",
        "baseline_rank",
        "candidate_rank",
        "near_top20_band",
        "score_delta",
        "rank_delta",
        "has_theme_flag",
        "dominant_theme",
        "theme_score",
        "theme_confidence",
        "candidate_applied_flag",
        "candidate_reason",
    ]
    near_df.loc[:, [c for c in near_cols if c in near_df.columns]].to_csv(FEATURE_CANDIDATE_EXP_B_NEAR_TOP20_CSV, index=False, encoding="utf-8")

    def _ratio(no_theme_count: int, total_count: int) -> float:
        return float(no_theme_count / total_count) if total_count else 0.0

    baseline_near = latest.loc[baseline_band].copy()
    candidate_near = latest.loc[candidate_band].copy()
    baseline_top20_df = latest.loc[latest["baseline_rank"] <= 20].copy()
    candidate_top20_df = latest.loc[latest["candidate_rank"] <= 20].copy()
    entered_df = latest.loc[latest["top20_status"].eq("entered")].copy()
    exited_df = latest.loc[latest["top20_status"].eq("exited")].copy()
    no_theme_df = latest.loc[latest["has_theme_flag"] == 0].copy()
    no_theme_positive_rank_delta_count = int((pd.to_numeric(no_theme_df["rank_delta"], errors="coerce").fillna(0.0) > 0).sum())
    no_theme_positive_rank_delta_ratio = _ratio(no_theme_positive_rank_delta_count, len(no_theme_df))
    baseline_near_symbols = set(baseline_near["symbol"].astype(str))
    candidate_near_symbols = set(candidate_near["symbol"].astype(str))
    entered_near_symbols = candidate_near_symbols - baseline_near_symbols
    entered_near_df = latest.loc[latest["symbol"].astype(str).isin(entered_near_symbols)].copy()
    entered_near_no_theme_count = int((entered_near_df["has_theme_flag"] == 0).sum())
    entered_near_no_theme_ratio = _ratio(entered_near_no_theme_count, len(entered_near_df))
    displaced_top20_no_theme_count = int(
        (
            latest["symbol"].astype(str).isin(candidate_near_symbols)
            & latest["symbol"].astype(str).isin(set(baseline_top20_df["symbol"].astype(str)))
            & (latest["has_theme_flag"] == 0)
        ).sum()
    )

    baseline_near_no_theme = int((baseline_near["has_theme_flag"] == 0).sum())
    candidate_near_no_theme = int((candidate_near["has_theme_flag"] == 0).sum())
    baseline_top20_no_theme = int((baseline_top20_df["has_theme_flag"] == 0).sum())
    candidate_top20_no_theme = int((candidate_top20_df["has_theme_flag"] == 0).sum())
    entered_no_theme = int((entered_df["has_theme_flag"] == 0).sum())
    exited_no_theme = int((exited_df["has_theme_flag"] == 0).sum())

    avg_rank_delta = float(latest["rank_delta"].mean()) if not latest.empty else 0.0
    median_rank_delta = float(latest["rank_delta"].median()) if not latest.empty else 0.0
    avg_score_delta = float(latest["score_delta"].mean()) if not latest.empty else 0.0
    no_theme_near_ratio_delta = _ratio(candidate_near_no_theme, len(candidate_near)) - _ratio(baseline_near_no_theme, len(baseline_near))

    promote_guard_ok = (
        no_theme_positive_rank_delta_count == 0
        and entered_no_theme == 0
        and entered_near_no_theme_count <= 1
        and int(latest["candidate_applied_flag"].sum()) > 0
    )

    if not is_feature_candidate_enabled(cfg):
        recommendation = "hold_for_more_review"
    elif promote_guard_ok:
        recommendation = "promote_to_feature_flag_candidate"
    elif no_theme_near_ratio_delta > 0.10:
        recommendation = "reject_candidate"
    else:
        recommendation = "hold_for_more_review"

    lines = [
        "# Feature Candidate Exp-B Summary",
        "",
        "## Overview",
        "- Candidate path keeps baseline ranking untouched and exports sidecar-only outputs.",
        "",
        "## Candidate Configuration",
        f"- candidate={cfg.get('candidate', 'none')}",
        f"- enabled={bool(cfg.get('enabled', False))}",
        f"- reach_factor={float(cfg.get('exp_b_delayed_cap_reach_factor', EXP_B_DELAYED_CAP_REACH_FACTOR_DEFAULT)):.3f}",
        f"- max_penalty_ratio={float(cfg.get('exp_b_delayed_cap_max_penalty_ratio', EXP_B_DELAYED_CAP_MAX_PENALTY_RATIO_DEFAULT)):.3f}",
        f"- apply_regimes={','.join(cfg.get('exp_b_delayed_cap_apply_regimes', [])) if cfg.get('exp_b_delayed_cap_apply_regimes') else '(all)'}",
        f"- theme_only={bool(cfg.get('exp_b_delayed_cap_theme_only', EXP_B_DELAYED_CAP_THEME_ONLY_DEFAULT))}",
        f"- min_theme_score={float(cfg.get('exp_b_delayed_cap_min_theme_score', EXP_B_DELAYED_CAP_MIN_THEME_SCORE_DEFAULT)):.2f}",
        f"- min_theme_confidence={float(cfg.get('exp_b_delayed_cap_min_theme_confidence', EXP_B_DELAYED_CAP_MIN_THEME_CONFIDENCE_DEFAULT)):.2f}",
        "",
        "## Score / Rank Shift Summary",
        f"- latest_date={latest_date}",
        f"- row_count={int(len(latest))}",
        f"- candidate_applied_count={int(latest['candidate_applied_flag'].sum())}",
        f"- average_score_delta={avg_score_delta:.4f}",
        f"- average_rank_delta={avg_rank_delta:.4f}",
        f"- median_rank_delta={median_rank_delta:.4f}",
        "",
        "## Top20 Membership Changes",
        f"- baseline_top20_count={int(len(baseline_top20_df))}",
        f"- candidate_top20_count={int(len(candidate_top20_df))}",
        f"- entered_count={int(len(entered_df))}",
        f"- exited_count={int(len(exited_df))}",
        "",
        "## Near-Top20 No-Theme Analysis",
        f"- baseline_total={int(len(baseline_near))}",
        f"- baseline_no_theme_count={baseline_near_no_theme}",
        f"- baseline_no_theme_ratio={_ratio(baseline_near_no_theme, len(baseline_near)):.4f}",
        f"- candidate_total={int(len(candidate_near))}",
        f"- candidate_no_theme_count={candidate_near_no_theme}",
        f"- candidate_no_theme_ratio={_ratio(candidate_near_no_theme, len(candidate_near)):.4f}",
        f"- ratio_delta={no_theme_near_ratio_delta:.4f}",
        f"- no_theme_positive_rank_delta_count={no_theme_positive_rank_delta_count}",
        f"- no_theme_positive_rank_delta_ratio={no_theme_positive_rank_delta_ratio:.4f}",
        f"- entered_near_count={int(len(entered_near_df))}",
        f"- entered_near_no_theme_count={entered_near_no_theme_count}",
        f"- entered_near_no_theme_ratio={entered_near_no_theme_ratio:.4f}",
        f"- displaced_top20_no_theme_count={displaced_top20_no_theme_count}",
        "",
        "## Top20 No-Theme Analysis",
        f"- baseline_total={int(len(baseline_top20_df))}",
        f"- baseline_no_theme_count={baseline_top20_no_theme}",
        f"- baseline_no_theme_ratio={_ratio(baseline_top20_no_theme, len(baseline_top20_df)):.4f}",
        f"- candidate_total={int(len(candidate_top20_df))}",
        f"- candidate_no_theme_count={candidate_top20_no_theme}",
        f"- candidate_no_theme_ratio={_ratio(candidate_top20_no_theme, len(candidate_top20_df)):.4f}",
        f"- ratio_delta={(_ratio(candidate_top20_no_theme, len(candidate_top20_df)) - _ratio(baseline_top20_no_theme, len(baseline_top20_df))):.4f}",
        f"- entered_count={int(len(entered_df))}",
        f"- entered_no_theme_count={entered_no_theme}",
        f"- entered_no_theme_ratio={_ratio(entered_no_theme, len(entered_df)):.4f}",
        f"- exited_count={int(len(exited_df))}",
        f"- exited_no_theme_count={exited_no_theme}",
        f"- exited_no_theme_ratio={_ratio(exited_no_theme, len(exited_df)):.4f}",
        "",
        "## Interpretation",
        "- Use the no-theme ratios as guard rails, but interpret them alongside no_theme_positive_rank_delta_count.",
        "- If near-top20 no-theme ratio rises while no_theme_positive_rank_delta_count stays near zero, the change is likely caused by displaced top20 no-theme names rather than no-theme candidate uplift.",
        "- In the current run, entered_near_no_theme_count should be read as a monitoring signal, not direct no-theme uplift, when candidate_applied_flag remains false for those names.",
        "",
        "## Operational Recommendation",
        f"- {recommendation}",
    ]
    FEATURE_CANDIDATE_EXP_B_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(
        "Feature candidate exp_b: enabled=%s applied=%d near_top20_no_theme_ratio=%.4f top20_no_theme_ratio=%.4f promote_guard_ok=%s recommendation=%s summary=%s",
        is_feature_candidate_enabled(cfg),
        int(latest["candidate_applied_flag"].sum()),
        _ratio(candidate_near_no_theme, len(candidate_near)),
        _ratio(candidate_top20_no_theme, len(candidate_top20_df)),
        promote_guard_ok,
        recommendation,
        FEATURE_CANDIDATE_EXP_B_SUMMARY_MD.resolve(),
    )


def strip_theme_risk_soft_experiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.copy()
    drop_cols = [
        col for col in THEME_RISK_SOFT_EXPERIMENT_COLUMNS + THEME_RISK_CURVE_EXPERIMENT_COLUMNS + FEATURE_CANDIDATE_COLUMNS
        if col in baseline.columns
    ]
    if drop_cols:
        baseline = baseline.drop(columns=drop_cols)
    return baseline


def _ensure_pg_daily_ranking_columns() -> None:
    if not get_engine:
        return
    statements = [
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS rank_final INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS live_rank INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_factor_count INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_missing_ratio DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_score_confidence DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS prob_score_raw DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS prob_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS prob_rank_pct DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS ret_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS qual_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS tech_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS safety_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS liquidity_score_missing BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS ret_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS prob_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS qual_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS tech_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS safety_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS liquidity_score_fallback_used BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS fallback_count INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS component_coverage_ratio DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS data_maturity_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS model_reliability_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS signal_agreement_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS regime_fitness_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_label TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_reason TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_score_research DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_score_operational DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_label_research TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS confidence_label_operational TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_flag BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_gate_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_penalty_ratio DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_quality_gate_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_quality_penalty_ratio DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_final_score_quality_gate DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_rank_quality_gate INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_quality_risk_guard_penalty DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_quality_risk_guard_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_final_score_quality_risk_guard DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_rank_quality_risk_guard INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS quality_gate_experiment TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_summary TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_strengths TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_risks TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_confidence TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_regime TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS explain TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_driver_1 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_driver_2 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_driver_3 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_drag_1 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_drag_2 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS top_driver_1 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS top_driver_2 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS top_driver_3 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS risk_factor_1 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS risk_factor_2 TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS action_note TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_explain_json TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_ret DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_prob DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_tech DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_qual DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_safety DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_liquidity DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_theme DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_contribution_risk DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS return_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS probability_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS technical_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS valuation_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS dominant_theme TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_confidence DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_score_effective DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS final_score_before_theme DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS final_score_v2_before_theme DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS final_score_v3 DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS live_score DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS live_score_source TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_diff_v2 DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS score_diff_v3 DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS v3_vs_v2_diff DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_mode TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_anchor TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_delta_raw DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_formula TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_delta_vs_base DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_delta_positive DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_positive_part DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_negative_part DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_gain DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_cap DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_signed_component DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_positive_component DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_negative_component DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_applied DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_capped BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_overlay_soft_conf_gate DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_uplift_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS theme_penalty_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_theme_weight_raw DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_theme_weight DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_theme_weight_effective DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_base_weight DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_floor_applied BOOLEAN",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_theme_score_effective DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_final_score_v3 DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_score_diff_v3 DOUBLE PRECISION",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_rank_v3 INTEGER",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS shadow_explain TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS regime_reason TEXT",
        "ALTER TABLE daily_ranking ADD COLUMN IF NOT EXISTS weight_profile TEXT",
    ]
    eng = get_engine()
    with eng.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))


def _ensure_sqlite_daily_ranking_columns(conn: sqlite3.Connection) -> None:
    existing = set(_get_sqlite_table_columns(conn, "daily_ranking"))
    alter_specs = [
        ("rank_final", "INTEGER"),
        ("live_rank", "INTEGER"),
        ("quality_factor_count", "INTEGER"),
        ("quality_missing_ratio", "REAL"),
        ("quality_score_confidence", "REAL"),
        ("prob_score_raw", "REAL"),
        ("prob_score_missing", "INTEGER"),
        ("prob_rank_pct", "REAL"),
        ("ret_score_missing", "INTEGER"),
        ("qual_score_missing", "INTEGER"),
        ("tech_score_missing", "INTEGER"),
        ("safety_score_missing", "INTEGER"),
        ("liquidity_score_missing", "INTEGER"),
        ("ret_score_fallback_used", "INTEGER"),
        ("prob_score_fallback_used", "INTEGER"),
        ("qual_score_fallback_used", "INTEGER"),
        ("tech_score_fallback_used", "INTEGER"),
        ("safety_score_fallback_used", "INTEGER"),
        ("liquidity_score_fallback_used", "INTEGER"),
        ("fallback_count", "INTEGER"),
        ("component_coverage_ratio", "REAL"),
        ("data_maturity_score", "REAL"),
        ("model_reliability_score", "REAL"),
        ("signal_agreement_score", "REAL"),
        ("regime_fitness_score", "REAL"),
        ("confidence_score_research", "REAL"),
        ("confidence_score_operational", "REAL"),
        ("confidence_label_research", "TEXT"),
        ("confidence_label_operational", "TEXT"),
        ("quality_flag", "INTEGER"),
        ("quality_gate_applied", "INTEGER"),
        ("quality_penalty_ratio", "REAL"),
        ("shadow_quality_gate_applied", "INTEGER"),
        ("shadow_quality_penalty_ratio", "REAL"),
        ("shadow_final_score_quality_gate", "REAL"),
        ("shadow_rank_quality_gate", "INTEGER"),
        ("shadow_quality_risk_guard_penalty", "REAL"),
        ("shadow_quality_risk_guard_applied", "INTEGER"),
        ("shadow_final_score_quality_risk_guard", "REAL"),
        ("shadow_rank_quality_risk_guard", "INTEGER"),
        ("quality_gate_experiment", "TEXT"),
        ("confidence_label", "TEXT"),
        ("confidence_reason", "TEXT"),
        ("score_explain_summary", "TEXT"),
        ("score_explain_strengths", "TEXT"),
        ("score_explain_risks", "TEXT"),
        ("score_explain_confidence", "TEXT"),
        ("score_explain_regime", "TEXT"),
        ("explain", "TEXT"),
        ("score_driver_1", "TEXT"),
        ("score_driver_2", "TEXT"),
        ("score_driver_3", "TEXT"),
        ("score_drag_1", "TEXT"),
        ("score_drag_2", "TEXT"),
        ("top_driver_1", "TEXT"),
        ("top_driver_2", "TEXT"),
        ("top_driver_3", "TEXT"),
        ("risk_factor_1", "TEXT"),
        ("risk_factor_2", "TEXT"),
        ("action_note", "TEXT"),
        ("score_explain_json", "TEXT"),
        ("score_contribution_ret", "REAL"),
        ("score_contribution_prob", "REAL"),
        ("score_contribution_tech", "REAL"),
        ("score_contribution_qual", "REAL"),
        ("score_contribution_safety", "REAL"),
        ("score_contribution_liquidity", "REAL"),
        ("score_contribution_theme", "REAL"),
        ("score_contribution_risk", "REAL"),
        ("return_score", "REAL"),
        ("probability_score", "REAL"),
        ("technical_score", "REAL"),
        ("valuation_score", "REAL"),
        ("theme_score", "REAL"),
        ("dominant_theme", "TEXT"),
        ("theme_confidence", "REAL"),
        ("theme_score_effective", "REAL"),
        ("final_score_before_theme", "REAL"),
        ("final_score_v2_before_theme", "REAL"),
        ("final_score_v3", "REAL"),
        ("live_score", "REAL"),
        ("live_score_source", "TEXT"),
        ("score_diff_v2", "REAL"),
        ("score_diff_v3", "REAL"),
        ("v3_vs_v2_diff", "REAL"),
        ("theme_overlay_mode", "TEXT"),
        ("theme_overlay_anchor", "TEXT"),
        ("theme_delta_raw", "REAL"),
        ("theme_overlay_formula", "TEXT"),
        ("theme_delta_vs_base", "REAL"),
        ("theme_delta_positive", "REAL"),
        ("theme_positive_part", "REAL"),
        ("theme_negative_part", "REAL"),
        ("theme_overlay_gain", "REAL"),
        ("theme_overlay_cap", "REAL"),
        ("theme_overlay_signed_component", "REAL"),
        ("theme_overlay_positive_component", "REAL"),
        ("theme_overlay_negative_component", "REAL"),
        ("theme_overlay_applied", "REAL"),
        ("theme_overlay_capped", "INTEGER"),
        ("theme_overlay_soft_conf_gate", "REAL"),
        ("theme_uplift_applied", "INTEGER"),
        ("theme_penalty_applied", "INTEGER"),
        ("shadow_theme_weight_raw", "REAL"),
        ("shadow_theme_weight", "REAL"),
        ("shadow_theme_weight_effective", "REAL"),
        ("shadow_base_weight", "REAL"),
        ("shadow_floor_applied", "INTEGER"),
        ("shadow_theme_score_effective", "REAL"),
        ("shadow_final_score_v3", "REAL"),
        ("shadow_score_diff_v3", "REAL"),
        ("shadow_rank_v3", "INTEGER"),
        ("shadow_explain", "TEXT"),
        ("regime_reason", "TEXT"),
        ("weight_profile", "TEXT"),
    ]
    for col, col_type in alter_specs:
        if col not in existing:
            conn.execute(f"ALTER TABLE daily_ranking ADD COLUMN {col} {col_type}")


def save_ranking(
    df: pd.DataFrame,
    theme_risk_soft_config: dict | None = None,
    risk_curve_experiment_config: dict | None = None,
    feature_candidate_config: dict | None = None,
) -> None:
    ensure_data_dir()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    COMPARE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    full_out = df.copy()
    full_out["date"] = pd.to_datetime(full_out["date"]).dt.strftime("%Y-%m-%d")
    if "quality_factor_count" in full_out.columns:
        full_out["quality_factor_count"] = pd.to_numeric(full_out["quality_factor_count"], errors="coerce").round().astype("Int64")
    if "fallback_count" in full_out.columns:
        full_out["fallback_count"] = pd.to_numeric(full_out["fallback_count"], errors="coerce").round().astype("Int64")
    if "shadow_rank_v3" in full_out.columns:
        full_out["shadow_rank_v3"] = pd.to_numeric(full_out["shadow_rank_v3"], errors="coerce").round().astype("Int64")
    if "theme_uplift_applied" in full_out.columns:
        full_out["theme_uplift_applied"] = full_out["theme_uplift_applied"].fillna(False).astype(bool)
    if "theme_penalty_applied" in full_out.columns:
        full_out["theme_penalty_applied"] = full_out["theme_penalty_applied"].fillna(False).astype(bool)
    if "shadow_floor_applied" in full_out.columns:
        full_out["shadow_floor_applied"] = full_out["shadow_floor_applied"].fillna(False).astype(bool)
    if "model_version" not in full_out.columns:
        full_out["model_version"] = None

    df_out = strip_theme_risk_soft_experiment_columns(full_out)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    _save_score_breakdown_debug(df_out)
    _save_confidence_diagnostics(df_out)
    _save_theme_impact_compare(df_out)
    _save_quality_gate_shadow(df_out)
    export_before_after_comparison(df_out)
    export_theme_validation_report(df_out)
    export_theme_guard_report(df_out)
    export_theme_overlay_gate_debug()
    export_theme_overlay_mode_resolution()
    export_theme_overlay_shadow_preview(df_out)
    export_theme_debug_outputs(df_out)
    export_theme_risk_soft_outputs(full_out, theme_risk_soft_config)
    export_risk_curve_experiment_outputs(full_out, risk_curve_experiment_config)
    export_feature_candidate_reports(full_out, feature_candidate_config)
    logging.info("Saved ranking: %s (rows=%d)", OUT_CSV.resolve(), len(df_out))
    logging.info(
        "Theme overlay outputs: %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s",
        BEFORE_AFTER_SCORE_COMPARE_V3_CSV.resolve(),
        TOP20_BEFORE_AFTER_COMPARE_V3_CSV.resolve(),
        THEME_CONFIDENCE_OVERLAY_VALIDATION_MD.resolve(),
        THEME_GUARD_REPORT_MD.resolve(),
        THEME_RISK_SOFT_COMPARE_CSV.resolve(),
        THEME_RISK_CURVE_COMPARE_CSV.resolve(),
        THEME_RISK_CURVE_NEAR_TOP20_CSV.resolve(),
        FEATURE_CANDIDATE_EXP_B_CSV.resolve(),
        FEATURE_CANDIDATE_EXP_B_TOP20_DIFF_CSV.resolve(),
        FEATURE_CANDIDATE_EXP_B_NEAR_TOP20_CSV.resolve(),
        FEATURE_CANDIDATE_EXP_B_SUMMARY_MD.resolve(),
        OUT_CSV.resolve(),
    )

    try:
        if replace_table_rows_pg:
            _ensure_pg_daily_ranking_columns()
            pg_columns = _get_pg_table_columns("daily_ranking")
            db_out = _prepare_db_rows(df_out, pg_columns or DAILY_RANKING_STORE_COLUMNS)
            if ensure_unique_keys:
                ensure_unique_keys(db_out, DAILY_RANKING_PK, "daily_ranking")
            replace_table_rows_pg("daily_ranking", db_out, columns=list(db_out.columns))
            logging.info("Replaced daily_ranking rows in Postgres (rows=%d)", len(db_out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to sqlite")

    if not use_sqlite_fallback_writes():
        logging.info("Skipping sqlite fallback for daily_ranking (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_ranking (
                date               DATE NOT NULL,
                code               TEXT NOT NULL,
                close              REAL,
                pred_return_60d    REAL,
                pred_return_90d    REAL,
                pred_mdd_60d       REAL,
                pred_mdd_90d       REAL,
                prob_top20_60d     REAL,
                prob_top20_90d     REAL,
                prob_score_raw     REAL,
                prob_score_missing BOOLEAN,
                prob_rank_pct      REAL,
                score              REAL,
                score_score        REAL,
                composite          REAL,
                quality_score      REAL,
                quality_factor_count INTEGER,
                quality_missing_ratio REAL,
                quality_score_confidence REAL,
                vol_20             REAL,
                vol_60             REAL,
                vol_ma_20          REAL,
                volume             REAL,
                mom_20             REAL,
                close_over_ma20    REAL,
                rsi_14             REAL,
                vol_ratio_20       REAL,
                name               TEXT,
                market             TEXT,
                sector             TEXT,
                tech_source        TEXT,
                regime_reason      TEXT,
                weight_profile     TEXT,
                tech_trend_score   REAL,
                tech_momentum_score REAL,
                tech_stability_score REAL,
                tech_volume_score  REAL,
                tech_liquidity_guard REAL,
                tech_score         REAL,
                pred_score         REAL,
                ret_score          REAL,
                return_score       REAL,
                prob_score         REAL,
                probability_score  REAL,
                qual_score         REAL,
                technical_score    REAL,
                valuation_score    REAL,
                ret_score_missing  BOOLEAN,
                qual_score_missing BOOLEAN,
                tech_score_missing BOOLEAN,
                safety_score_missing BOOLEAN,
                liquidity_score_missing BOOLEAN,
                ret_score_fallback_used BOOLEAN,
                prob_score_fallback_used BOOLEAN,
                qual_score_fallback_used BOOLEAN,
                tech_score_fallback_used BOOLEAN,
                safety_score_fallback_used BOOLEAN,
                liquidity_score_fallback_used BOOLEAN,
                fallback_count     INTEGER,
                safety_score       REAL,
                liquidity_score    REAL,
                theme_score        REAL,
                dominant_theme     TEXT,
                theme_confidence   REAL,
                theme_score_effective REAL,
                final_score_raw    REAL,
                final_score_before_theme REAL,
                final_score_v2_before_theme REAL,
                final_score        REAL,
                final_score_v2     REAL,
                final_score_v3     REAL,
                live_score         REAL,
                live_score_source  TEXT,
                score_diff_v2      REAL,
                score_diff_v3      REAL,
                v3_vs_v2_diff      REAL,
                theme_overlay_mode TEXT,
                theme_overlay_anchor TEXT,
                theme_delta_raw REAL,
                theme_overlay_formula TEXT,
                theme_delta_vs_base REAL,
                theme_delta_positive REAL,
                theme_positive_part REAL,
                theme_negative_part REAL,
                theme_overlay_gain REAL,
                theme_overlay_cap REAL,
                theme_overlay_signed_component REAL,
                theme_overlay_positive_component REAL,
                theme_overlay_negative_component REAL,
                theme_overlay_applied REAL,
                theme_overlay_capped BOOLEAN,
                theme_overlay_soft_conf_gate REAL,
                theme_uplift_applied BOOLEAN,
                theme_penalty_applied BOOLEAN,
                shadow_theme_weight_raw REAL,
                shadow_theme_weight REAL,
                shadow_theme_weight_effective REAL,
                shadow_base_weight REAL,
                shadow_floor_applied BOOLEAN,
                shadow_theme_score_effective REAL,
                shadow_final_score_v3 REAL,
                shadow_score_diff_v3 REAL,
                shadow_rank_v3    INTEGER,
                shadow_explain    TEXT,
                rank_final         INTEGER,
                live_rank          INTEGER,
                rank_v2            INTEGER,
                score_contribution_ret REAL,
                score_contribution_prob REAL,
                score_contribution_tech REAL,
                score_contribution_qual REAL,
                score_contribution_safety REAL,
                score_contribution_liquidity REAL,
                score_contribution_theme REAL,
                score_contribution_risk REAL,
                contrib_tech       REAL,
                contrib_ret        REAL,
                contrib_prob       REAL,
                contrib_qual       REAL,
                contrib_safety     REAL,
                contrib_liquidity  REAL,
                contrib_theme      REAL,
                contrib_penalty    REAL,
                top_positive_factor TEXT,
                top_positive_value REAL,
                top_negative_factor TEXT,
                top_negative_value REAL,
                explain_text       TEXT,
                explain            TEXT,
                score_explain_summary TEXT,
                score_explain_strengths TEXT,
                score_explain_risks TEXT,
                score_explain_confidence TEXT,
                score_explain_regime TEXT,
                score_driver_1     TEXT,
                score_driver_2     TEXT,
                score_driver_3     TEXT,
                score_drag_1       TEXT,
                score_drag_2       TEXT,
                top_driver_1       TEXT,
                top_driver_2       TEXT,
                top_driver_3       TEXT,
                risk_factor_1      TEXT,
                risk_factor_2      TEXT,
                action_note        TEXT,
                score_explain_json TEXT,
                confidence_version TEXT,
                data_maturity_score REAL,
                model_reliability_score REAL,
                signal_agreement_score REAL,
                regime_fitness_score REAL,
                confidence_score_research REAL,
                confidence_score_operational REAL,
                component_coverage_ratio REAL,
                confidence_score   REAL,
                confidence_label_research TEXT,
                confidence_label_operational TEXT,
                quality_flag BOOLEAN,
                quality_gate_applied BOOLEAN,
                quality_penalty_ratio REAL,
                shadow_quality_gate_applied BOOLEAN,
                shadow_quality_penalty_ratio REAL,
                shadow_final_score_quality_gate REAL,
                shadow_rank_quality_gate INTEGER,
                shadow_quality_risk_guard_penalty REAL,
                shadow_quality_risk_guard_applied BOOLEAN,
                shadow_final_score_quality_risk_guard REAL,
                shadow_rank_quality_risk_guard INTEGER,
                quality_gate_experiment TEXT,
                confidence_label   TEXT,
                confidence_grade   TEXT,
                confidence_reason  TEXT,
                confidence_explain_text TEXT,
                risk_penalty       REAL,
                market_up          BOOLEAN,
                market_status_date DATE,
                market_kospi_close REAL,
                market_kospi_ma20  REAL,
                market_vol_5d      REAL,
                market_foreign_5d  REAL,
                generated_at       TEXT,
                model_version      TEXT,
                score_formula_version TEXT,
                PRIMARY KEY (date, code)
            );
            """
        )
        _ensure_sqlite_daily_ranking_columns(conn)
        sqlite_columns = _get_sqlite_table_columns(conn, "daily_ranking")
        db_out = _prepare_db_rows(df_out, sqlite_columns or DAILY_RANKING_STORE_COLUMNS)
        if ensure_unique_keys:
            ensure_unique_keys(db_out, DAILY_RANKING_PK, "daily_ranking")
        if replace_table_rows_sqlite:
            replace_table_rows_sqlite(conn, "daily_ranking", db_out)
        conn.commit()
        logging.info("Saved ranking to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(df_out))
    except Exception:
        logging.exception("Failed to save ranking to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def main() -> None:
    setup_logging()
    args = parse_cli_args()
    shadow_overlay_config = apply_shadow_theme_overlay_config(resolve_shadow_theme_overlay_config(args))
    theme_risk_soft_config = resolve_theme_risk_soft_config(args)
    risk_curve_experiment_config = resolve_risk_curve_experiment_config(args)
    feature_candidate_config = resolve_feature_candidate_config(args)
    ranking = build_ranking(
        theme_risk_soft_config=theme_risk_soft_config,
        risk_curve_experiment_config=risk_curve_experiment_config,
        feature_candidate_config=feature_candidate_config,
    )
    save_ranking(
        ranking,
        theme_risk_soft_config=theme_risk_soft_config,
        risk_curve_experiment_config=risk_curve_experiment_config,
        feature_candidate_config=feature_candidate_config,
    )
    print("generated_files=" + str([
        str(OUT_CSV),
        str(BEFORE_AFTER_SCORE_COMPARE_CSV),
        str(TOP20_BEFORE_AFTER_COMPARE_CSV),
        str(BEFORE_AFTER_SCORE_COMPARE_V3_CSV),
        str(TOP20_BEFORE_AFTER_COMPARE_V3_CSV),
        str(THEME_CONFIDENCE_OVERLAY_VALIDATION_MD),
        str(THEME_GUARD_REPORT_MD),
        str(THEME_OVERLAY_GATE_DEBUG_JSON),
        str(THEME_OVERLAY_GATE_DEBUG_MD),
        str(THEME_OVERLAY_MODE_RESOLUTION_MD),
        str(THEME_OVERLAY_SHADOW_PREVIEW_CSV),
        str(THEME_OVERLAY_SHADOW_MODE_UPDATE_MD),
        str(DEBUG_THEME_TOP50_CSV),
        str(DEBUG_THEME_SUMMARY_TXT),
        str(RANKING_THEME_RISK_SOFT_CSV),
        str(THEME_RISK_SOFT_COMPARE_CSV),
        str(THEME_RISK_SOFT_VALIDATION_MD),
        str(THEME_RISK_CURVE_COMPARE_CSV),
        str(THEME_RISK_CURVE_VALIDATION_MD),
        str(THEME_RISK_CURVE_NEAR_TOP20_CSV),
        str(FEATURE_CANDIDATE_EXP_B_CSV),
        str(FEATURE_CANDIDATE_EXP_B_TOP20_DIFF_CSV),
        str(FEATURE_CANDIDATE_EXP_B_NEAR_TOP20_CSV),
        str(FEATURE_CANDIDATE_EXP_B_SUMMARY_MD),
    ]))
    print("shadow_overlay_config=" + str(shadow_overlay_config))
    print("example=python python\\ranking_builder.py")


if __name__ == "__main__":
    main()
