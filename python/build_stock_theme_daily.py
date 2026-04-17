import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from theme_mapping_utils import standardize_stock_theme_map


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
STOCK_THEME_MAP_CSV = DATA_DIR / "stock_theme_map.csv"
STOCK_THEME_MAP_OVERRIDE_CSV = DATA_DIR / "stock_theme_map_overrides.csv"
THEME_ETF_DAILY_CSV = OUTPUT_DIR / "theme_etf_daily.csv"
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
OUTPUT_CSV = OUTPUT_DIR / "stock_theme_daily.csv"
SUMMARY_CSV = OUTPUT_DIR / "stock_theme_daily_summary.csv"
THEME_LEVEL_DEBUG_CSV = DATA_DIR / "theme_level_aggregation_debug.csv"
STOCK_THEME_TOPK_DEBUG_CSV = DATA_DIR / "stock_theme_topk_debug.csv"
SUMMARY_MD = DATA_DIR / "build_stock_theme_daily_topk_summary.md"
TOP_K_THEME_ETF = int(os.environ.get("THEME_TOPK_ETF", "3"))
THEME_MIN_ETF_SCORE = float(os.environ.get("THEME_MIN_ETF_SCORE", "50"))
THEME_MIN_ETF_CONF = float(os.environ.get("THEME_MIN_ETF_CONF", "0.45"))
STOCK_THEME_TRANSMISSION_MODE = str(os.environ.get("STOCK_THEME_TRANSMISSION_MODE", "baseline")).strip().lower()
STOCK_THEME_MAPPING_FLOOR = float(os.environ.get("STOCK_THEME_MAPPING_FLOOR", "0.0"))
STOCK_THEME_CONFIDENCE_FLOOR = float(os.environ.get("STOCK_THEME_CONFIDENCE_FLOOR", "0.0"))
STOCK_THEME_COMPONENT_BLEND_FLOOR = float(os.environ.get("STOCK_THEME_COMPONENT_BLEND_FLOOR", "0.0"))
STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN = float(os.environ.get("STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN", "85.0"))
STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN = float(os.environ.get("STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN", "0.55"))
STOCK_THEME_STRONG_SOURCE_CONF_FLOOR = float(os.environ.get("STOCK_THEME_STRONG_SOURCE_CONF_FLOOR", "0.74"))

OUTPUT_COLUMNS = [
    "date",
    "code",
    "name",
    "theme_score_raw",
    "theme_score",
    "dominant_theme",
    "theme_confidence",
    "theme_source_count",
    "theme_score_theme_level",
    "theme_breadth",
    "theme_consistency",
    "theme_topk_count",
    "theme_total_etf_count",
    "theme_concentration",
    "theme_signal_confidence",
    "theme_etf_count",
    "theme_etf_breadth",
    "theme_signal_regime",
    "theme_flow_source",
    "dominant_theme_source_type",
    "theme_transmission_mode",
    "theme_weight_effective",
    "theme_confidence_raw",
    "theme_confidence_effective",
    "theme_component_blend_floor",
    "strong_source_gate_passed",
    "component_floor_applied",
    "mapping_floor_applied",
    "confidence_floor_applied",
    "signal_retention_ratio",
    "mapping_quality",
    "theme_breadth_confidence",
    "stock_theme_alignment",
    "theme_trend_score",
    "theme_activity_score",
    "theme_flow_score",
    "theme_stability_score",
    "theme_explain",
    "theme_rank_within_stock",
    "theme_detail_json",
]

SUMMARY_COLUMNS = [
    "date",
    "total_stocks",
    "themed_stocks",
    "avg_theme_score",
    "avg_theme_confidence",
    "top_theme_1",
    "top_theme_1_count",
    "top_theme_2",
    "top_theme_2_count",
    "top_theme_3",
    "top_theme_3_count",
]

REGIME_CONFIDENCE_MAP = {
    "strong": 0.90,
    "neutral": 0.65,
    "weak": 0.35,
}

LOGGER = logging.getLogger("build_stock_theme_daily")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily stock dominant theme overlay from stock-theme map and ETF theme signals.")
    parser.add_argument("--stock-theme-map", default=str(STOCK_THEME_MAP_CSV), help="Input stock-theme mapping CSV path.")
    parser.add_argument("--stock-theme-map-override", default=str(STOCK_THEME_MAP_OVERRIDE_CSV), help="Optional override CSV merged on top of stock-theme map.")
    parser.add_argument("--theme-etf-daily", default=str(THEME_ETF_DAILY_CSV), help="Input theme ETF daily CSV path.")
    parser.add_argument("--theme-etf-master", default=str(THEME_ETF_MASTER_CSV), help="Input theme ETF master CSV path.")
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV), help="Output stock theme daily CSV path.")
    parser.add_argument("--summary-csv", default=str(SUMMARY_CSV), help="Output stock theme summary CSV path.")
    parser.add_argument("--theme-topk-etf", type=int, default=TOP_K_THEME_ETF, help="Number of ETFs to aggregate per date/theme.")
    parser.add_argument("--theme-min-etf-score", type=float, default=THEME_MIN_ETF_SCORE, help="Minimum ETF theme score threshold for top-k selection.")
    parser.add_argument("--theme-min-etf-conf", type=float, default=THEME_MIN_ETF_CONF, help="Minimum ETF signal confidence threshold for top-k selection.")
    parser.add_argument("--stock-theme-transmission-mode", type=str, default=STOCK_THEME_TRANSMISSION_MODE, help="Transmission mode: baseline, signal_priority, or high_conviction.")
    parser.add_argument("--stock-theme-mapping-floor", type=float, default=STOCK_THEME_MAPPING_FLOOR, help="Minimum mapping weight used by experimental transmission modes.")
    parser.add_argument("--stock-theme-confidence-floor", type=float, default=STOCK_THEME_CONFIDENCE_FLOOR, help="Minimum theme confidence used by experimental transmission modes.")
    parser.add_argument("--stock-theme-component-blend-floor", type=float, default=STOCK_THEME_COMPONENT_BLEND_FLOOR, help="Minimum retained share of theme-level score for strong-source experimental transmission.")
    parser.add_argument("--stock-theme-strong-source-theme-level-min", type=float, default=STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN, help="Minimum theme-level score to qualify for strong-source transmission treatment.")
    parser.add_argument("--stock-theme-strong-source-signal-conf-min", type=float, default=STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN, help="Minimum theme signal confidence to qualify for strong-source transmission treatment.")
    parser.add_argument("--stock-theme-strong-source-conf-floor", type=float, default=STOCK_THEME_STRONG_SOURCE_CONF_FLOOR, help="Confidence floor applied only to strong-source transmission candidates.")
    return parser.parse_args()


def _read_stock_theme_map_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"stock_theme_map.csv not found: {path}")

    df = pd.read_csv(path, dtype={"code": str, "theme_id": str})
    required = ["code", "name", "theme_id", "theme_name", "mapping_weight", "mapping_source", "is_primary", "updated_at"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"stock_theme_map.csv missing required columns: {missing}")

    df = standardize_stock_theme_map(df)
    return df


def load_stock_theme_map(path: Path = STOCK_THEME_MAP_CSV, override_path: Path | None = None) -> pd.DataFrame:
    base_df = _read_stock_theme_map_csv(path)
    merged_paths = [str(path)]

    if override_path is not None and str(override_path).strip():
        if override_path.exists():
            override_df = _read_stock_theme_map_csv(override_path)
            base_df = pd.concat([base_df, override_df], ignore_index=True)
            base_df = standardize_stock_theme_map(base_df)
            merged_paths.append(str(override_path))
            LOGGER.info(
                "Merged stock theme override rows=%d path=%s",
                len(override_df),
                override_path,
            )
        else:
            LOGGER.info("Stock theme override not found -> skip path=%s", override_path)

    df = base_df.copy()
    df["is_primary"] = df["is_primary"].astype(int)
    df = df[df["theme_id"] != ""].copy()
    LOGGER.info("Loaded stock theme map rows=%d paths=%s", len(df), merged_paths)
    return df


def load_theme_etf_daily(path: Path = THEME_ETF_DAILY_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"theme_etf_daily.csv not found: {path}")

    df = pd.read_csv(path, dtype={"theme_id": str, "etf_code": str})
    required = ["date", "theme_id", "theme_name", "etf_code", "etf_name", "etf_theme_score", "theme_regime"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"theme_etf_daily.csv missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["theme_id"] = df["theme_id"].fillna("").astype(str).str.upper().str.strip()
    df["theme_name"] = df["theme_name"].fillna("").astype(str)
    df["etf_code"] = df["etf_code"].astype(str).str.zfill(6)
    df["etf_name"] = df["etf_name"].fillna("").astype(str)
    df["theme_regime"] = df["theme_regime"].fillna("").astype(str).str.lower().str.strip()
    df["etf_theme_score"] = _to_numeric(df["etf_theme_score"]).fillna(0.0).clip(lower=0.0, upper=100.0)
    for col in [
        "etf_signal_confidence",
        "trend_score",
        "activity_score",
        "flow_score",
        "flow_proxy_score",
        "stability_score",
        "trading_value",
        "flow_data_available",
        "flow_source",
        "signal_regime",
        "explain_etf_theme",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    df["etf_signal_confidence"] = _to_numeric(df["etf_signal_confidence"]).fillna(df.apply(derive_etf_signal_confidence, axis=1)).clip(lower=0.0, upper=1.0)
    for col in ["trend_score", "activity_score", "flow_score", "flow_proxy_score", "stability_score", "trading_value"]:
        df[col] = _to_numeric(df[col]).fillna(0.0)
    df["flow_data_available"] = df["flow_data_available"].fillna(False).astype(bool)
    df["flow_source"] = df["flow_source"].fillna("unknown").astype(str)
    df["signal_regime"] = df["signal_regime"].fillna(df["theme_regime"]).astype(str).str.lower().str.strip()
    df["explain_etf_theme"] = df["explain_etf_theme"].fillna("").astype(str)
    df = df[df["theme_id"] != ""].copy()
    LOGGER.info("Loaded theme ETF daily rows=%d path=%s", len(df), path)
    return df


def _load_theme_priority_map(path: Path = THEME_ETF_MASTER_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["theme_id", "theme_priority"])
    df = pd.read_csv(path, dtype={"theme_id": str})
    if "theme_id" not in df.columns:
        return pd.DataFrame(columns=["theme_id", "theme_priority"])
    if "priority" not in df.columns:
        df["priority"] = 999
    out = df.loc[:, ["theme_id", "priority"]].copy()
    out["theme_id"] = out["theme_id"].fillna("").astype(str).str.upper().str.strip()
    out["theme_priority"] = _to_numeric(out["priority"]).fillna(999).astype(int)
    return out.loc[:, ["theme_id", "theme_priority"]].drop_duplicates(subset=["theme_id"])


def derive_etf_signal_confidence(row: pd.Series) -> float:
    regime = str(row.get("theme_regime") or "").strip().lower()
    if regime in REGIME_CONFIDENCE_MAP:
        return float(REGIME_CONFIDENCE_MAP[regime])
    score = float(_to_numeric(pd.Series([row.get("etf_theme_score")])).fillna(0.0).iloc[0])
    return max(0.0, min(1.0, score / 100.0))


def compute_theme_confidence(mapping_weight: Any, etf_signal_confidence: Any) -> float:
    mapping = float(_to_numeric(pd.Series([mapping_weight])).fillna(0.0).iloc[0])
    signal = float(_to_numeric(pd.Series([etf_signal_confidence])).fillna(0.0).iloc[0])
    confidence = 0.7 * mapping + 0.3 * signal
    return max(0.0, min(1.0, confidence))


def resolve_stock_theme_transmission_config(args: argparse.Namespace | None = None) -> dict[str, float | str]:
    mode = str(getattr(args, "stock_theme_transmission_mode", STOCK_THEME_TRANSMISSION_MODE) or "baseline").strip().lower()
    if mode not in {"baseline", "signal_priority", "high_conviction"}:
        LOGGER.warning("Invalid stock theme transmission mode=%r; falling back to baseline", mode)
        mode = "baseline"
    mapping_floor = float(getattr(args, "stock_theme_mapping_floor", STOCK_THEME_MAPPING_FLOOR))
    confidence_floor = float(getattr(args, "stock_theme_confidence_floor", STOCK_THEME_CONFIDENCE_FLOOR))
    component_blend_floor = float(getattr(args, "stock_theme_component_blend_floor", STOCK_THEME_COMPONENT_BLEND_FLOOR))
    strong_source_theme_level_min = float(getattr(args, "stock_theme_strong_source_theme_level_min", STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN))
    strong_source_signal_conf_min = float(getattr(args, "stock_theme_strong_source_signal_conf_min", STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN))
    strong_source_conf_floor = float(getattr(args, "stock_theme_strong_source_conf_floor", STOCK_THEME_STRONG_SOURCE_CONF_FLOOR))
    mapping_floor = max(0.0, min(1.0, mapping_floor))
    confidence_floor = max(0.0, min(1.0, confidence_floor))
    component_blend_floor = max(0.0, min(1.0, component_blend_floor))
    strong_source_theme_level_min = max(0.0, min(100.0, strong_source_theme_level_min))
    strong_source_signal_conf_min = max(0.0, min(1.0, strong_source_signal_conf_min))
    strong_source_conf_floor = max(0.0, min(1.0, strong_source_conf_floor))
    return {
        "mode": mode,
        "mapping_floor": mapping_floor,
        "confidence_floor": confidence_floor,
        "component_blend_floor": component_blend_floor,
        "strong_source_theme_level_min": strong_source_theme_level_min,
        "strong_source_signal_conf_min": strong_source_signal_conf_min,
        "strong_source_conf_floor": strong_source_conf_floor,
    }


def apply_stock_theme_transmission(
    merged: pd.DataFrame,
    config: dict[str, float | str],
) -> pd.DataFrame:
    out = merged.copy()
    mode = str(config.get("mode", "baseline"))
    mapping_floor = float(config.get("mapping_floor", 0.0))
    confidence_floor = float(config.get("confidence_floor", 0.0))
    component_blend_floor = float(config.get("component_blend_floor", 0.0))
    strong_source_theme_level_min = float(config.get("strong_source_theme_level_min", 85.0))
    strong_source_signal_conf_min = float(config.get("strong_source_signal_conf_min", 0.55))
    strong_source_conf_floor = float(config.get("strong_source_conf_floor", 0.74))

    out["theme_transmission_mode"] = mode
    out["theme_weight_raw"] = _to_numeric(out.get("theme_weight")).fillna(_to_numeric(out["mapping_weight"]).fillna(0.0)).clip(lower=0.0, upper=1.0)
    out["theme_confidence_raw"] = (
        0.25 * out["mapping_quality"]
        + 0.45 * out["theme_signal_confidence"]
        + 0.20 * out["theme_breadth_confidence"]
        + 0.10 * out["stock_theme_alignment"]
    ).clip(lower=0.0, upper=1.0)

    strong_source_mask = (
        out.get("theme_source_type", pd.Series(index=out.index, dtype=str)).fillna("none").astype(str).isin(["etf_price_only", "mixed"])
        & _to_numeric(out.get("theme_score_theme_level")).fillna(0.0).ge(strong_source_theme_level_min)
        & _to_numeric(out.get("theme_signal_confidence")).fillna(0.0).ge(strong_source_signal_conf_min)
    )

    if mode == "signal_priority":
        out["theme_weight_effective"] = np.maximum(out["theme_weight_raw"], mapping_floor)
        signal_priority_conf = (
            0.20 * out["mapping_quality"]
            + 0.55 * out["theme_signal_confidence"]
            + 0.15 * out["theme_breadth_confidence"]
            + 0.10 * out["stock_theme_alignment"]
        ).clip(lower=0.0, upper=1.0)
        out["theme_confidence_effective"] = np.maximum(out["theme_confidence_raw"], signal_priority_conf)
        out["theme_confidence_effective"] = np.maximum(out["theme_confidence_effective"], confidence_floor)
    elif mode == "high_conviction":
        out["theme_weight_effective"] = out["theme_weight_raw"]
        out.loc[strong_source_mask, "theme_weight_effective"] = np.maximum(
            out.loc[strong_source_mask, "theme_weight_raw"],
            mapping_floor,
        )
        high_conviction_conf = (
            0.18 * out["mapping_quality"]
            + 0.57 * out["theme_signal_confidence"]
            + 0.15 * out["theme_breadth_confidence"]
            + 0.10 * out["stock_theme_alignment"]
        ).clip(lower=0.0, upper=1.0)
        out["theme_confidence_effective"] = np.maximum(out["theme_confidence_raw"], confidence_floor)
        out.loc[strong_source_mask, "theme_confidence_effective"] = np.maximum(
            np.maximum(out.loc[strong_source_mask, "theme_confidence_raw"], high_conviction_conf.loc[strong_source_mask]),
            strong_source_conf_floor,
        )
    else:
        out["theme_weight_effective"] = out["theme_weight_raw"]
        out["theme_confidence_effective"] = np.maximum(out["theme_confidence_raw"], confidence_floor)

    raw_component = out["theme_score_theme_level"] * out["theme_weight_effective"]
    blended_component_floor = out["theme_score_theme_level"] * component_blend_floor
    out["theme_component_blend_floor"] = 0.0
    out.loc[strong_source_mask, "theme_component_blend_floor"] = component_blend_floor
    out["mapping_floor_applied"] = out["theme_weight_effective"].gt(out["theme_weight_raw"])
    out["confidence_floor_applied"] = out["theme_confidence_effective"].gt(out["theme_confidence_raw"])
    out["strong_source_gate_passed"] = strong_source_mask
    out["stock_theme_component"] = raw_component
    if mode == "high_conviction" and component_blend_floor > 0.0:
        out.loc[strong_source_mask, "stock_theme_component"] = np.maximum(
            raw_component.loc[strong_source_mask],
            blended_component_floor.loc[strong_source_mask],
        )
    out["component_floor_applied"] = out["stock_theme_component"].gt(raw_component + 1e-12)
    out["theme_confidence_candidate"] = out["theme_confidence_effective"]
    out["signal_retention_ratio"] = np.where(
        out["theme_score_theme_level"].gt(0.0),
        (out["stock_theme_component"] * out["theme_confidence_candidate"]) / out["theme_score_theme_level"],
        0.0,
    )
    return out


def _weighted_average(series: pd.Series, weights: pd.Series) -> float:
    values = _to_numeric(series).fillna(0.0)
    weight_values = _to_numeric(weights).fillna(0.0).clip(lower=0.0)
    total_weight = float(weight_values.sum())
    if total_weight <= 0:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weight_values).sum() / total_weight)


def select_topk_theme_etfs(theme_group: pd.DataFrame, top_k: int, min_score: float, min_conf: float) -> pd.DataFrame:
    grp = theme_group.copy()
    qualified = grp.loc[
        (_to_numeric(grp["etf_theme_score"]).fillna(0.0) >= float(min_score))
        & (_to_numeric(grp["etf_signal_confidence"]).fillna(0.0) >= float(min_conf))
    ].copy()
    if qualified.empty:
        qualified = grp.head(1).copy()
        qualified["selection_fallback"] = "top1_fallback"
    else:
        qualified = qualified.head(max(int(top_k), 1)).copy()
        qualified["selection_fallback"] = "qualified_topk"
    qualified["etf_rank_within_theme"] = range(1, len(qualified) + 1)
    return qualified


def compute_theme_breadth(selected: pd.DataFrame, total_count: int, min_score: float, min_conf: float) -> float:
    if total_count <= 0:
        return 0.0
    qualified_count = int(
        (
            (_to_numeric(selected["etf_theme_score"]).fillna(0.0) >= float(min_score))
            & (_to_numeric(selected["etf_signal_confidence"]).fillna(0.0) >= float(min_conf))
        ).sum()
    )
    return max(0.0, min(1.0, qualified_count / float(total_count)))


def compute_theme_consistency(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.5
    score_std = float(_to_numeric(selected["etf_theme_score"]).fillna(0.0).std(ddof=0))
    conf_std = float(_to_numeric(selected["etf_signal_confidence"]).fillna(0.0).std(ddof=0))
    regime_mode_share = float(selected["signal_regime"].value_counts(normalize=True, dropna=False).iloc[0]) if "signal_regime" in selected.columns and not selected["signal_regime"].empty else 0.5
    score_consistency = max(0.0, 1.0 - min(score_std / 25.0, 1.0))
    conf_consistency = max(0.0, 1.0 - min(conf_std / 0.25, 1.0))
    return max(0.0, min(1.0, 0.45 * score_consistency + 0.20 * conf_consistency + 0.35 * regime_mode_share))


def compute_theme_concentration(weights: pd.Series) -> float:
    weight_values = _to_numeric(weights).fillna(0.0).clip(lower=0.0)
    total = float(weight_values.sum())
    if total <= 0:
        return 1.0
    return float((weight_values / total).max())


def _resolve_theme_source_type(selected: pd.DataFrame) -> str:
    source_set = set(selected.get("source_name", pd.Series(dtype=str)).fillna("unknown").astype(str).tolist())
    source_set.discard("")
    if not source_set:
        return "none"
    if source_set == {"etf_price"}:
        return "etf_price_only"
    if source_set == {"theme_proxy"}:
        return "theme_proxy_only"
    return "mixed"


def aggregate_theme_signal(
    theme_etf_df: pd.DataFrame,
    top_k: int,
    min_score: float,
    min_conf: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if theme_etf_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    sort_cols = ["date", "theme_id", "etf_theme_score", "etf_signal_confidence", "trading_value", "etf_code"]
    work = theme_etf_df.sort_values(sort_cols, ascending=[True, True, False, False, False, True]).copy()

    for (as_of_date, theme_id), grp in work.groupby(["date", "theme_id"], dropna=False):
        grp = grp.reset_index(drop=True)
        total_count = int(len(grp))
        selected = select_topk_theme_etfs(grp, top_k=top_k, min_score=min_score, min_conf=min_conf)
        selected["rank_weight"] = [1.0 / idx for idx in selected["etf_rank_within_theme"]]
        selected["aggregate_weight"] = selected["rank_weight"] * (0.55 + 0.45 * _to_numeric(selected["etf_signal_confidence"]).fillna(0.0))

        weighted_mean_topk = _weighted_average(selected["etf_theme_score"], selected["aggregate_weight"])
        breadth = compute_theme_breadth(grp, total_count=total_count, min_score=min_score, min_conf=min_conf)
        consistency = compute_theme_consistency(selected)
        concentration = compute_theme_concentration(selected["aggregate_weight"])
        theme_signal_confidence = max(
            0.0,
            min(
                1.0,
                0.60 * _weighted_average(selected["etf_signal_confidence"], selected["aggregate_weight"])
                + 0.25 * breadth
                + 0.15 * (1.0 - concentration),
            ),
        )
        theme_score_theme_level = (
            0.55 * weighted_mean_topk
            + 0.25 * (breadth * 100.0)
            + 0.20 * (consistency * 100.0)
        )
        source_type = _resolve_theme_source_type(selected)
        dominant_flow_source = selected["flow_source"].mode().iloc[0] if selected["flow_source"].notna().any() else "unknown"
        regime = selected["signal_regime"].mode().iloc[0] if selected["signal_regime"].notna().any() else str(selected.iloc[0].get("theme_regime") or "")
        detail = []
        for _, etf_row in selected.iterrows():
            detail.append(
                {
                    "etf_code": str(etf_row.get("etf_code") or "").zfill(6),
                    "etf_name": str(etf_row.get("etf_name") or ""),
                    "rank": int(etf_row.get("etf_rank_within_theme") or 0),
                    "etf_theme_score": round(float(etf_row.get("etf_theme_score") or 0.0), 4),
                    "etf_signal_confidence": round(float(etf_row.get("etf_signal_confidence") or 0.0), 4),
                    "source_name": str(etf_row.get("source_name") or ""),
                    "trend_score": round(float(etf_row.get("trend_score") or 0.0), 4),
                    "activity_score": round(float(etf_row.get("activity_score") or 0.0), 4),
                    "flow_score": round(float(etf_row.get("flow_score") or 0.0), 4),
                    "stability_score": round(float(etf_row.get("stability_score") or 0.0), 4),
                    "aggregate_weight": round(float(etf_row.get("aggregate_weight") or 0.0), 4),
                    "selection_fallback": str(etf_row.get("selection_fallback") or ""),
                }
            )

        rows.append(
            {
                "date": as_of_date,
                "theme_id": str(theme_id).upper(),
                "theme_name": str(selected.iloc[0].get("theme_name") or ""),
                "theme_topk_count": int(len(selected)),
                "theme_total_etf_count": total_count,
                "theme_etf_count": total_count,
                "theme_breadth": breadth,
                "theme_etf_breadth": breadth,
                "theme_consistency": consistency,
                "theme_concentration": concentration,
                "theme_signal_confidence": theme_signal_confidence,
                "theme_etf_signal_confidence": _weighted_average(selected["etf_signal_confidence"], selected["aggregate_weight"]),
                "theme_score_theme_level": theme_score_theme_level,
                "theme_score_base": weighted_mean_topk,
                "theme_score_breadth_adjusted": theme_score_theme_level,
                "theme_signal_regime": str(regime or ""),
                "theme_flow_source": str(dominant_flow_source or "unknown"),
                "theme_source_type": source_type,
                "theme_trend_score": _weighted_average(selected["trend_score"], selected["aggregate_weight"]),
                "theme_activity_score": _weighted_average(selected["activity_score"], selected["aggregate_weight"]),
                "theme_flow_score": _weighted_average(selected["flow_score"], selected["aggregate_weight"]),
                "theme_stability_score": _weighted_average(selected["stability_score"], selected["aggregate_weight"]),
                "theme_etf_detail_json": json.dumps(detail, ensure_ascii=False),
            }
        )
        debug_rows.append(
            {
                "date": as_of_date,
                "theme_id": str(theme_id).upper(),
                "theme_name": str(selected.iloc[0].get("theme_name") or ""),
                "selected_etf_codes": json.dumps(selected["etf_code"].astype(str).str.zfill(6).tolist(), ensure_ascii=False),
                "selected_etf_scores": json.dumps(_to_numeric(selected["etf_theme_score"]).fillna(0.0).round(4).tolist(), ensure_ascii=False),
                "selected_etf_confidences": json.dumps(_to_numeric(selected["etf_signal_confidence"]).fillna(0.0).round(4).tolist(), ensure_ascii=False),
                "selected_etf_sources": json.dumps(selected.get("source_name", pd.Series(dtype=str)).fillna("").astype(str).tolist(), ensure_ascii=False),
                "theme_topk_count": int(len(selected)),
                "theme_total_etf_count": total_count,
                "theme_breadth": breadth,
                "theme_consistency": consistency,
                "theme_concentration": concentration,
                "theme_score_theme_level": theme_score_theme_level,
                "theme_signal_confidence": theme_signal_confidence,
                "theme_source_type": source_type,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(debug_rows)


def build_theme_candidates(
    stock_theme_df: pd.DataFrame,
    theme_etf_df: pd.DataFrame,
    theme_etf_master_path: Path = THEME_ETF_MASTER_CSV,
    transmission_config: dict[str, float | str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stock_theme_df.empty or theme_etf_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    theme_priority_df = _load_theme_priority_map(path=theme_etf_master_path)
    daily_theme_df, theme_level_debug_df = aggregate_theme_signal(
        theme_etf_df,
        top_k=TOP_K_THEME_ETF,
        min_score=THEME_MIN_ETF_SCORE,
        min_conf=THEME_MIN_ETF_CONF,
    )
    merged = stock_theme_df.merge(
        daily_theme_df,
        on=["theme_id"],
        how="inner",
        suffixes=("_map", "_daily"),
    )
    if merged.empty:
        return merged, theme_level_debug_df

    if "theme_name_daily" in merged.columns:
        merged["theme_name"] = merged["theme_name_daily"].fillna(merged.get("theme_name_map", ""))
    elif "theme_name" not in merged.columns and "theme_name_map" in merged.columns:
        merged["theme_name"] = merged["theme_name_map"]
    merged["name"] = merged["name"].fillna("").astype(str)
    merged["theme_score_base"] = _to_numeric(merged["theme_score_base"]).fillna(0.0).clip(lower=0.0, upper=100.0)
    merged["theme_score_breadth_adjusted"] = _to_numeric(merged["theme_score_breadth_adjusted"]).fillna(merged["theme_score_base"]).clip(lower=0.0, upper=100.0)
    merged["theme_score_theme_level"] = _to_numeric(merged["theme_score_theme_level"]).fillna(merged["theme_score_breadth_adjusted"]).clip(lower=0.0, upper=100.0)
    merged["theme_breadth"] = _to_numeric(merged["theme_breadth"]).fillna(_to_numeric(merged.get("theme_etf_breadth")).fillna(0.5)).clip(lower=0.0, upper=1.0)
    merged["theme_consistency"] = _to_numeric(merged["theme_consistency"]).fillna(0.5).clip(lower=0.0, upper=1.0)
    merged["theme_concentration"] = _to_numeric(merged["theme_concentration"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    merged["theme_signal_confidence"] = _to_numeric(merged["theme_signal_confidence"]).fillna(_to_numeric(merged.get("theme_etf_signal_confidence")).fillna(0.5)).clip(lower=0.0, upper=1.0)
    merged = merged[merged["theme_id"].fillna("").astype(str).str.strip() != ""].copy()
    merged = merged[merged["theme_score_breadth_adjusted"].fillna(0.0) > 0.0].copy()
    merged["etf_signal_confidence"] = _to_numeric(merged["theme_etf_signal_confidence"]).fillna(merged.apply(derive_etf_signal_confidence, axis=1)).clip(lower=0.0, upper=1.0)
    merged["mapping_quality"] = _to_numeric(merged["mapping_weight"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    merged["theme_breadth_confidence"] = (0.60 * merged["theme_breadth"] + 0.40 * (1.0 - merged["theme_concentration"])).clip(lower=0.0, upper=1.0)
    merged["stock_theme_alignment"] = 0.60
    merged["theme_weight"] = _to_numeric(merged.get("theme_weight")).fillna(_to_numeric(merged["mapping_weight"]).fillna(0.0)).clip(lower=0.0, upper=1.0)
    merged = apply_stock_theme_transmission(
        merged,
        transmission_config or resolve_stock_theme_transmission_config(),
    )
    merged = merged.merge(theme_priority_df, on="theme_id", how="left")
    merged["theme_priority"] = _to_numeric(merged["theme_priority"]).fillna(999).astype(int)
    return merged, theme_level_debug_df


def normalize_stock_theme_score(series: pd.Series) -> pd.Series:
    return _to_numeric(series).fillna(0.0).clip(lower=0.0, upper=100.0)


def pick_dominant_theme(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    work = candidate_df.copy()
    work["stock_theme_component"] = _to_numeric(work["stock_theme_component"]).fillna(0.0)
    work["mapping_weight"] = _to_numeric(work["mapping_weight"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    work["theme_weight"] = _to_numeric(work.get("theme_weight")).fillna(work["mapping_weight"]).clip(lower=0.0, upper=1.0)
    work["theme_confidence_candidate"] = _to_numeric(work["theme_confidence_candidate"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    work["theme_priority"] = _to_numeric(work["theme_priority"]).fillna(999).astype(int)
    work["is_primary"] = _to_numeric(work["is_primary"]).fillna(0).astype(int)
    # Give the curated primary theme a small deterministic edge when raw theme components are close.
    work["dominant_sort_score"] = work["stock_theme_component"] + work["is_primary"] * 0.05

    detail_rows: list[dict[str, Any]] = []
    for (as_of_date, code, name), grp in work.groupby(["date", "code", "name"], dropna=False):
        grp = grp.sort_values(
            ["dominant_sort_score", "stock_theme_component", "is_primary", "theme_weight", "mapping_weight", "theme_priority", "theme_id"],
            ascending=[False, False, False, False, False, True, True],
        ).reset_index(drop=True)
        grp["theme_rank_within_stock"] = range(1, len(grp) + 1)
        top = grp.iloc[0] if not grp.empty else None

        if top is None or float(top["stock_theme_component"]) <= 0.0:
            detail_rows.append(
                {
                    "date": as_of_date,
                    "code": str(code).zfill(6),
                    "name": str(name or ""),
                    "theme_score_raw": 0.0,
                    "theme_score": 0.0,
                    "dominant_theme": "",
                    "theme_confidence": 0.0,
                    "theme_source_count": 0,
                    "theme_score_theme_level": 0.0,
                    "theme_breadth": 0.0,
                    "theme_consistency": 0.5,
                    "theme_topk_count": 0,
                    "theme_total_etf_count": 0,
                    "theme_concentration": 1.0,
                    "theme_signal_confidence": 0.0,
                    "theme_etf_count": 0,
                    "theme_etf_breadth": 0.0,
                    "theme_signal_regime": "",
                    "theme_flow_source": "",
                    "dominant_theme_source_type": "none",
                    "theme_transmission_mode": "baseline",
                    "theme_weight_effective": 0.0,
                    "theme_confidence_raw": 0.0,
                    "theme_confidence_effective": 0.0,
                    "theme_component_blend_floor": 0.0,
                    "strong_source_gate_passed": False,
                    "component_floor_applied": False,
                    "mapping_floor_applied": False,
                    "confidence_floor_applied": False,
                    "signal_retention_ratio": 0.0,
                    "mapping_quality": 0.0,
                    "theme_breadth_confidence": 0.0,
                    "stock_theme_alignment": 0.0,
                    "theme_trend_score": 0.0,
                    "theme_activity_score": 0.0,
                    "theme_flow_score": 0.0,
                    "theme_stability_score": 0.0,
                    "theme_explain": "",
                    "theme_rank_within_stock": 0,
                    "theme_detail_json": "[]",
                }
            )
            continue

        details = []
        for _, row in grp.iterrows():
            details.append(
                {
                    "theme_id": str(row.get("theme_id") or ""),
                    "theme_name": str(row.get("theme_name") or ""),
                    "component": round(float(row.get("stock_theme_component") or 0.0), 4),
                    "mapping_weight": round(float(row.get("mapping_weight") or 0.0), 4),
                    "mapping_quality": round(float(row.get("mapping_quality") or 0.0), 4),
                    "theme_weight": round(float(row.get("theme_weight") or 0.0), 4),
                    "theme_score_base": round(float(row.get("theme_score_base") or 0.0), 4),
                    "theme_score_theme_level": round(float(row.get("theme_score_theme_level") or 0.0), 4),
                    "theme_score_breadth_adjusted": round(float(row.get("theme_score_breadth_adjusted") or 0.0), 4),
                    "theme_confidence_candidate": round(float(row.get("theme_confidence_candidate") or 0.0), 4),
                    "theme_signal_confidence": round(float(row.get("theme_signal_confidence") or 0.0), 4),
                    "theme_breadth_confidence": round(float(row.get("theme_breadth_confidence") or 0.0), 4),
                    "stock_theme_alignment": round(float(row.get("stock_theme_alignment") or 0.0), 4),
                    "theme_topk_count": int(row.get("theme_topk_count") or 0),
                    "theme_total_etf_count": int(row.get("theme_total_etf_count") or 0),
                    "theme_breadth": round(float(row.get("theme_breadth") or 0.0), 4),
                    "theme_consistency": round(float(row.get("theme_consistency") or 0.0), 4),
                    "theme_concentration": round(float(row.get("theme_concentration") or 1.0), 4),
                    "theme_source_type": str(row.get("theme_source_type") or "none"),
                    "theme_etf_count": int(row.get("theme_etf_count") or 0),
                    "theme_etf_breadth": round(float(row.get("theme_etf_breadth") or 0.0), 4),
                    "theme_signal_regime": str(row.get("theme_signal_regime") or ""),
                    "theme_flow_source": str(row.get("theme_flow_source") or ""),
                    "theme_trend_score": round(float(row.get("theme_trend_score") or 0.0), 4),
                    "theme_activity_score": round(float(row.get("theme_activity_score") or 0.0), 4),
                    "theme_flow_score": round(float(row.get("theme_flow_score") or 0.0), 4),
                    "theme_stability_score": round(float(row.get("theme_stability_score") or 0.0), 4),
                    "theme_etf_detail": json.loads(str(row.get("theme_etf_detail_json") or "[]")),
                    "is_primary": int(row.get("is_primary") or 0),
                    "match_type": str(row.get("match_type") or ""),
                    "source_note": str(row.get("source_note") or ""),
                }
            )

        theme_score_raw = float(top["stock_theme_component"])
        theme_explain = (
            f"{str(top.get('theme_name') or '')} theme selected: "
            f"theme_level_score={float(top.get('theme_score_theme_level') or 0.0):.1f}, "
            f"topk={int(top.get('theme_topk_count') or 0)}/{int(top.get('theme_total_etf_count') or 0)}, "
            f"breadth={float(top.get('theme_breadth') or 0.0):.2f}, "
            f"consistency={float(top.get('theme_consistency') or 0.0):.2f}, "
            f"concentration={float(top.get('theme_concentration') or 0.0):.2f}, "
            f"mapping={float(top.get('mapping_quality') or 0.0):.2f}, "
            f"source={str(top.get('theme_source_type') or 'none')}"
        )
        detail_rows.append(
            {
                "date": as_of_date,
                "code": str(code).zfill(6),
                "name": str(name or ""),
                "theme_score_raw": theme_score_raw,
                "theme_score": theme_score_raw,
                "dominant_theme": str(top.get("theme_name") or ""),
                "theme_confidence": float(top.get("theme_confidence_candidate") or 0.0),
                "theme_source_count": int(grp["theme_id"].nunique()),
                "theme_score_theme_level": float(top.get("theme_score_theme_level") or 0.0),
                "theme_breadth": float(top.get("theme_breadth") or 0.0),
                "theme_consistency": float(top.get("theme_consistency") or 0.5),
                "theme_topk_count": int(top.get("theme_topk_count") or 0),
                "theme_total_etf_count": int(top.get("theme_total_etf_count") or 0),
                "theme_concentration": float(top.get("theme_concentration") or 1.0),
                "theme_signal_confidence": float(top.get("theme_signal_confidence") or 0.0),
                "theme_etf_count": int(top.get("theme_etf_count") or 0),
                "theme_etf_breadth": float(top.get("theme_etf_breadth") or 0.0),
                "theme_signal_regime": str(top.get("theme_signal_regime") or ""),
                "theme_flow_source": str(top.get("theme_flow_source") or ""),
                "dominant_theme_source_type": str(top.get("theme_source_type") or "none"),
                "theme_transmission_mode": str(top.get("theme_transmission_mode") or "baseline"),
                "theme_weight_effective": float(top.get("theme_weight_effective") or 0.0),
                "theme_confidence_raw": float(top.get("theme_confidence_raw") or 0.0),
                "theme_confidence_effective": float(top.get("theme_confidence_effective") or 0.0),
                "theme_component_blend_floor": float(top.get("theme_component_blend_floor") or 0.0),
                "strong_source_gate_passed": bool(top.get("strong_source_gate_passed") or False),
                "component_floor_applied": bool(top.get("component_floor_applied") or False),
                "mapping_floor_applied": bool(top.get("mapping_floor_applied") or False),
                "confidence_floor_applied": bool(top.get("confidence_floor_applied") or False),
                "signal_retention_ratio": float(top.get("signal_retention_ratio") or 0.0),
                "mapping_quality": float(top.get("mapping_quality") or 0.0),
                "theme_breadth_confidence": float(top.get("theme_breadth_confidence") or 0.0),
                "stock_theme_alignment": float(top.get("stock_theme_alignment") or 0.0),
                "theme_trend_score": float(top.get("theme_trend_score") or 0.0),
                "theme_activity_score": float(top.get("theme_activity_score") or 0.0),
                "theme_flow_score": float(top.get("theme_flow_score") or 0.0),
                "theme_stability_score": float(top.get("theme_stability_score") or 0.0),
                "theme_explain": theme_explain,
                "theme_rank_within_stock": int(top.get("theme_rank_within_stock") or 0),
                "theme_detail_json": json.dumps(details, ensure_ascii=False),
            }
        )

    out = pd.DataFrame(detail_rows)
    out["theme_score_raw"] = _to_numeric(out["theme_score_raw"]).fillna(0.0).clip(lower=0.0)
    out["theme_score"] = normalize_stock_theme_score(out["theme_score"])
    out["theme_confidence"] = _to_numeric(out["theme_confidence"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_source_count"] = _to_numeric(out["theme_source_count"]).fillna(0).astype(int)
    out["theme_score_theme_level"] = _to_numeric(out["theme_score_theme_level"]).fillna(0.0).clip(lower=0.0, upper=100.0)
    out["theme_breadth"] = _to_numeric(out["theme_breadth"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_consistency"] = _to_numeric(out["theme_consistency"]).fillna(0.5).clip(lower=0.0, upper=1.0)
    out["theme_topk_count"] = _to_numeric(out["theme_topk_count"]).fillna(0).astype(int)
    out["theme_total_etf_count"] = _to_numeric(out["theme_total_etf_count"]).fillna(0).astype(int)
    out["theme_concentration"] = _to_numeric(out["theme_concentration"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    out["theme_signal_confidence"] = _to_numeric(out["theme_signal_confidence"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_etf_count"] = _to_numeric(out["theme_etf_count"]).fillna(0).astype(int)
    out["theme_etf_breadth"] = _to_numeric(out["theme_etf_breadth"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    for col in ["theme_trend_score", "theme_activity_score", "theme_flow_score", "theme_stability_score"]:
        out[col] = _to_numeric(out[col]).fillna(0.0).clip(lower=0.0, upper=100.0)
    out["theme_signal_regime"] = out["theme_signal_regime"].fillna("").astype(str)
    out["theme_flow_source"] = out["theme_flow_source"].fillna("").astype(str)
    out["dominant_theme_source_type"] = out["dominant_theme_source_type"].fillna("none").astype(str)
    out["theme_transmission_mode"] = out["theme_transmission_mode"].fillna("baseline").astype(str)
    out["theme_weight_effective"] = _to_numeric(out["theme_weight_effective"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_confidence_raw"] = _to_numeric(out["theme_confidence_raw"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_confidence_effective"] = _to_numeric(out["theme_confidence_effective"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_component_blend_floor"] = _to_numeric(out["theme_component_blend_floor"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["strong_source_gate_passed"] = out["strong_source_gate_passed"].fillna(False).astype(bool)
    out["component_floor_applied"] = out["component_floor_applied"].fillna(False).astype(bool)
    out["mapping_floor_applied"] = out["mapping_floor_applied"].fillna(False).astype(bool)
    out["confidence_floor_applied"] = out["confidence_floor_applied"].fillna(False).astype(bool)
    out["signal_retention_ratio"] = _to_numeric(out["signal_retention_ratio"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["mapping_quality"] = _to_numeric(out["mapping_quality"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_breadth_confidence"] = _to_numeric(out["theme_breadth_confidence"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["stock_theme_alignment"] = _to_numeric(out["stock_theme_alignment"]).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["theme_explain"] = out["theme_explain"].fillna("").astype(str)
    out["theme_rank_within_stock"] = _to_numeric(out["theme_rank_within_stock"]).fillna(0).astype(int)
    out["dominant_theme"] = out["dominant_theme"].fillna("").astype(str)
    out["theme_detail_json"] = out["theme_detail_json"].fillna("[]").astype(str)
    return out.loc[:, OUTPUT_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _build_summary(stock_theme_daily_df: pd.DataFrame) -> pd.DataFrame:
    if stock_theme_daily_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for as_of_date, grp in stock_theme_daily_df.groupby("date", dropna=False):
        dominant_counts = (
            grp.loc[grp["dominant_theme"].fillna("").astype(str).str.strip() != ""]
            .groupby("dominant_theme")["code"]
            .count()
            .sort_values(ascending=False)
        )
        top_items = list(dominant_counts.items())[:3]
        padded = top_items + [("", 0)] * (3 - len(top_items))
        rows.append(
            {
                "date": as_of_date,
                "total_stocks": int(len(grp)),
                "themed_stocks": int(grp["dominant_theme"].fillna("").astype(str).str.strip().ne("").sum()),
                "avg_theme_score": float(_to_numeric(grp["theme_score"]).fillna(0.0).mean()),
                "avg_theme_confidence": float(_to_numeric(grp["theme_confidence"]).fillna(0.0).mean()),
                "top_theme_1": padded[0][0],
                "top_theme_1_count": int(padded[0][1]),
                "top_theme_2": padded[1][0],
                "top_theme_2_count": int(padded[1][1]),
                "top_theme_3": padded[2][0],
                "top_theme_3_count": int(padded[2][1]),
            }
        )
    return pd.DataFrame(rows).loc[:, SUMMARY_COLUMNS].sort_values("date").reset_index(drop=True)


def build_stock_theme_daily(
    stock_theme_df: pd.DataFrame,
    theme_etf_df: pd.DataFrame,
    theme_etf_master_path: Path = THEME_ETF_MASTER_CSV,
    transmission_config: dict[str, float | str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_df, theme_level_debug_df = build_theme_candidates(
        stock_theme_df,
        theme_etf_df,
        theme_etf_master_path=theme_etf_master_path,
        transmission_config=transmission_config,
    )
    daily_df = pick_dominant_theme(candidate_df)
    summary_df = _build_summary(daily_df)
    stock_debug_cols = [
        "date", "code", "name", "dominant_theme", "dominant_theme_source_type", "theme_score", "theme_confidence",
        "theme_score_theme_level", "theme_breadth", "theme_consistency", "theme_topk_count", "theme_total_etf_count",
        "theme_concentration", "theme_signal_confidence", "theme_transmission_mode", "theme_weight_effective",
        "theme_confidence_raw", "theme_confidence_effective", "mapping_floor_applied", "confidence_floor_applied",
        "signal_retention_ratio", "mapping_quality", "theme_breadth_confidence",
        "stock_theme_alignment", "theme_explain",
    ]
    stock_debug_df = daily_df.loc[:, [c for c in stock_debug_cols if c in daily_df.columns]].copy()
    return daily_df, summary_df, theme_level_debug_df, stock_debug_df


def export_stock_theme_daily(
    stock_theme_daily_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    theme_level_debug_df: pd.DataFrame,
    stock_debug_df: pd.DataFrame,
    output_csv: Path = OUTPUT_CSV,
    summary_csv: Path = SUMMARY_CSV,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out = stock_theme_daily_df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[:, OUTPUT_COLUMNS]
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = summary_df.copy()
    for col in SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = pd.NA
    summary = summary.loc[:, SUMMARY_COLUMNS]
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    theme_level_debug_df.to_csv(THEME_LEVEL_DEBUG_CSV, index=False, encoding="utf-8-sig")
    stock_debug_df.to_csv(STOCK_THEME_TOPK_DEBUG_CSV, index=False, encoding="utf-8-sig")

    LOGGER.info("Saved stock theme daily CSV: %s rows=%d", output_csv.resolve(), len(out))
    LOGGER.info("Saved stock theme summary CSV: %s rows=%d", summary_csv.resolve(), len(summary))
    LOGGER.info("Saved theme-level aggregation debug CSV: %s rows=%d", THEME_LEVEL_DEBUG_CSV.resolve(), len(theme_level_debug_df))
    LOGGER.info("Saved stock theme top-k debug CSV: %s rows=%d", STOCK_THEME_TOPK_DEBUG_CSV.resolve(), len(stock_debug_df))


def _print_validation_summary(stock_theme_daily_df: pd.DataFrame) -> None:
    if stock_theme_daily_df.empty:
        print("processed_dates=0")
        print("generated_stocks=0")
        print("dominant_theme_nonempty=0")
        print("theme_score_mean=0.00 max=0.00")
        print("theme_confidence_mean=0.00 max=0.00")
        print("avg_theme_topk_count=0.00")
        print("avg_theme_breadth=0.00")
        print("avg_theme_concentration=0.00")
        print("dominant_theme_source_type_counts={}")
        print("dominant_theme_top10={}")
        return

    processed_dates = int(stock_theme_daily_df["date"].nunique())
    generated_stocks = int(len(stock_theme_daily_df))
    dominant_nonempty = int(stock_theme_daily_df["dominant_theme"].fillna("").astype(str).str.strip().ne("").sum())
    theme_score = _to_numeric(stock_theme_daily_df["theme_score"]).fillna(0.0)
    theme_conf = _to_numeric(stock_theme_daily_df["theme_confidence"]).fillna(0.0)
    topk_count = _to_numeric(stock_theme_daily_df.get("theme_topk_count")).fillna(0.0)
    breadth = _to_numeric(stock_theme_daily_df.get("theme_breadth")).fillna(0.0)
    concentration = _to_numeric(stock_theme_daily_df.get("theme_concentration")).fillna(0.0)
    source_counts = stock_theme_daily_df.get("dominant_theme_source_type", pd.Series(dtype=str)).fillna("none").astype(str).value_counts().to_dict()
    top10 = (
        stock_theme_daily_df.loc[stock_theme_daily_df["dominant_theme"].fillna("").astype(str).str.strip() != ""]
        .groupby("dominant_theme")["code"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    print(f"processed_dates={processed_dates}")
    print(f"generated_stocks={generated_stocks}")
    print(f"dominant_theme_nonempty={dominant_nonempty}")
    print(f"theme_score_mean={theme_score.mean():.2f} max={theme_score.max():.2f}")
    print(f"theme_confidence_mean={theme_conf.mean():.4f} max={theme_conf.max():.4f}")
    print(f"avg_theme_topk_count={topk_count.mean():.2f}")
    print(f"avg_theme_breadth={breadth.mean():.4f}")
    print(f"avg_theme_concentration={concentration.mean():.4f}")
    print(f"dominant_theme_source_type_counts={source_counts}")
    print(f"dominant_theme_top10={top10}")


def write_summary_md(stock_theme_daily_df: pd.DataFrame, theme_level_debug_df: pd.DataFrame) -> None:
    if stock_theme_daily_df.empty:
        SUMMARY_MD.write_text("# Stock Theme Top-k Summary\n\nNo rows generated.\n", encoding="utf-8")
        return
    latest_date = str(stock_theme_daily_df["date"].max())
    latest = stock_theme_daily_df.loc[stock_theme_daily_df["date"] == latest_date].copy()
    latest_theme_debug = theme_level_debug_df.loc[theme_level_debug_df["date"] == latest_date].copy() if not theme_level_debug_df.empty else pd.DataFrame()
    lines = [
        "# Stock Theme Top-k Summary",
        "",
        f"- latest_date: {latest_date}",
        f"- total themes processed: {int(latest_theme_debug['theme_id'].nunique()) if not latest_theme_debug.empty else 0}",
        f"- average theme_topk_count: {float(_to_numeric(latest_theme_debug.get('theme_topk_count')).fillna(0.0).mean()) if not latest_theme_debug.empty else 0.0:.2f}",
        f"- average theme_breadth: {float(_to_numeric(latest_theme_debug.get('theme_breadth')).fillna(0.0).mean()) if not latest_theme_debug.empty else 0.0:.4f}",
        f"- average theme_concentration: {float(_to_numeric(latest_theme_debug.get('theme_concentration')).fillna(0.0).mean()) if not latest_theme_debug.empty else 0.0:.4f}",
        f"- dominant_theme != (none) ratio: {float(latest['dominant_theme'].fillna('').astype(str).str.strip().ne('').mean()):.2%}",
        f"- dominant_theme_source_type counts: {latest.get('dominant_theme_source_type', pd.Series(dtype=str)).fillna('none').astype(str).value_counts().to_dict()}",
        f"- transmission_mode: {latest.get('theme_transmission_mode', pd.Series(dtype=str)).fillna('baseline').astype(str).mode().iloc[0] if not latest.empty else 'baseline'}",
        f"- avg signal_retention_ratio: {float(_to_numeric(latest.get('signal_retention_ratio')).fillna(0.0).mean()) if not latest.empty else 0.0:.4f}",
        "",
        "## Notes",
        "",
        f"- top-k threshold: top_k={TOP_K_THEME_ETF}, min_score={THEME_MIN_ETF_SCORE}, min_conf={THEME_MIN_ETF_CONF}",
        f"- transmission_config: mode={STOCK_THEME_TRANSMISSION_MODE}, mapping_floor={STOCK_THEME_MAPPING_FLOOR:.2f}, confidence_floor={STOCK_THEME_CONFIDENCE_FLOOR:.2f}, component_blend_floor={STOCK_THEME_COMPONENT_BLEND_FLOOR:.2f}, strong_source_theme_level_min={STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN:.1f}, strong_source_signal_conf_min={STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN:.2f}, strong_source_conf_floor={STOCK_THEME_STRONG_SOURCE_CONF_FLOOR:.2f}",
        "- current theme universe may still produce many single-ETF selections if master/theme coverage is thin.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    stock_theme_map_path = Path(args.stock_theme_map)
    stock_theme_map_override_path = Path(args.stock_theme_map_override) if str(args.stock_theme_map_override).strip() else None
    theme_etf_daily_path = Path(args.theme_etf_daily)
    theme_etf_master_path = Path(args.theme_etf_master)
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    global TOP_K_THEME_ETF, THEME_MIN_ETF_SCORE, THEME_MIN_ETF_CONF
    global STOCK_THEME_TRANSMISSION_MODE, STOCK_THEME_MAPPING_FLOOR, STOCK_THEME_CONFIDENCE_FLOOR
    global STOCK_THEME_COMPONENT_BLEND_FLOOR, STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN
    global STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN, STOCK_THEME_STRONG_SOURCE_CONF_FLOOR
    TOP_K_THEME_ETF = max(int(args.theme_topk_etf), 1)
    THEME_MIN_ETF_SCORE = float(args.theme_min_etf_score)
    THEME_MIN_ETF_CONF = float(args.theme_min_etf_conf)
    transmission_config = resolve_stock_theme_transmission_config(args)
    STOCK_THEME_TRANSMISSION_MODE = str(transmission_config["mode"])
    STOCK_THEME_MAPPING_FLOOR = float(transmission_config["mapping_floor"])
    STOCK_THEME_CONFIDENCE_FLOOR = float(transmission_config["confidence_floor"])
    STOCK_THEME_COMPONENT_BLEND_FLOOR = float(transmission_config["component_blend_floor"])
    STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN = float(transmission_config["strong_source_theme_level_min"])
    STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN = float(transmission_config["strong_source_signal_conf_min"])
    STOCK_THEME_STRONG_SOURCE_CONF_FLOOR = float(transmission_config["strong_source_conf_floor"])

    stock_theme_df = load_stock_theme_map(stock_theme_map_path, stock_theme_map_override_path)
    theme_etf_df = load_theme_etf_daily(theme_etf_daily_path)
    stock_theme_daily_df, summary_df, theme_level_debug_df, stock_debug_df = build_stock_theme_daily(
        stock_theme_df,
        theme_etf_df,
        theme_etf_master_path=theme_etf_master_path,
        transmission_config=transmission_config,
    )
    export_stock_theme_daily(
        stock_theme_daily_df,
        summary_df,
        theme_level_debug_df,
        stock_debug_df,
        output_csv=output_csv,
        summary_csv=summary_csv,
    )
    write_summary_md(stock_theme_daily_df, theme_level_debug_df)
    _print_validation_summary(stock_theme_daily_df)
    print(f"generated_files={[str(output_csv), str(summary_csv), str(THEME_LEVEL_DEBUG_CSV), str(STOCK_THEME_TOPK_DEBUG_CSV), str(SUMMARY_MD)]}")
    print("example=python python\\build_stock_theme_daily.py --stock-theme-map data\\stock_theme_map.csv --stock-theme-map-override data\\stock_theme_map_overrides.csv --theme-topk-etf 3 --theme-min-etf-score 50 --theme-min-etf-conf 0.45")


if __name__ == "__main__":
    main()
