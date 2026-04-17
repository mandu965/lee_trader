import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from theme_mapping_utils import (
    ensure_theme_mapping_files,
    load_stock_theme_map,
    load_theme_etf_master as _load_theme_etf_master_seed,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
PRICES_RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
PRICES_CLEAN_CSV = DATA_DIR / "prices_daily_clean.csv"

OUTPUT_CSV = OUTPUT_DIR / "theme_etf_daily.csv"
SUMMARY_CSV = OUTPUT_DIR / "theme_etf_daily_summary.csv"
VALIDATION_MD = OUTPUT_DIR / "theme_etf_validation.md"
LATEST_RANK_CSV = OUTPUT_DIR / "theme_etf_latest_rank.csv"
DEBUG_CSV = OUTPUT_DIR / "etf_theme_score_debug.csv"
DATA_OUTPUT_CSV = DATA_DIR / "theme_etf_daily.csv"
DATA_SUMMARY_CSV = DATA_DIR / "theme_etf_daily_summary.csv"
DATA_VALIDATION_MD = DATA_DIR / "theme_etf_validation.md"
DATA_LATEST_RANK_CSV = DATA_DIR / "theme_etf_latest_rank.csv"
DATA_DEBUG_CSV = DATA_DIR / "etf_theme_score_debug.csv"

LOOKBACK_CALENDAR_DAYS = 220
KOSPI_BENCHMARK_CODES = {"1001", "KOSPI", "KOSPI200", "U001", "U180"}
EPS = 1e-9
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99
OUTLIER_SCORE_CAP = 100.0
THEME_REGIME_STRONG = 70.0
THEME_REGIME_NEUTRAL = 40.0

SUB_SCORE_WEIGHTS = {
    "trend_score": 0.30,
    "activity_score": 0.20,
    "flow_score": 0.35,
    "stability_score": 0.15,
}
CONFIDENCE_WEIGHTS = {
    "data_quality_conf": 0.40,
    "flow_evidence_conf": 0.25,
    "cross_etf_support_hint": 0.20,
    "stability_conf": 0.15,
}

OUTPUT_COLUMNS = [
    "date", "theme_id", "theme", "theme_name", "etf_code", "etf_name", "source_name",
    "close", "volume", "trading_value", "trading_value_ratio_20d", "abnormal_value_5d", "turnover_ratio",
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_ratio_20d",
    "ma20", "ma60", "ma20_gap", "ma60_gap", "rs_vs_kospi_20d",
    "positive_day_ratio_20d", "volatility_20d", "nav", "nav_gap", "tracking_error_20d",
    "aum", "shares_outstanding", "flow_ratio_1d", "flow_ratio_5d", "flow_ratio_20d",
    "ret_20d_score", "ret_60d_score", "vol_ratio_score", "ma20_gap_score", "rs_vs_kospi_score",
    "trend_score", "activity_score", "flow_score", "flow_proxy_score", "stability_score",
    "etf_theme_score_raw", "etf_theme_score", "flow_data_available", "flow_source",
    "data_quality_conf", "flow_evidence_conf", "stability_conf", "cross_etf_support_hint",
    "etf_signal_confidence", "signal_regime", "theme_regime", "overheat_penalty", "explain_etf_theme",
]

SUMMARY_COLUMNS = [
    "date", "total_themes", "avg_etf_theme_score", "avg_etf_signal_confidence",
    "avg_trend_score", "avg_activity_score", "avg_flow_score", "avg_stability_score",
    "flow_data_available_ratio", "flow_source_counts_json",
    "strong_count", "neutral_count", "weak_count",
    "top_theme_1", "top_theme_1_score", "top_theme_2", "top_theme_2_score", "top_theme_3", "top_theme_3_score",
]

LOGGER = logging.getLogger("compute_theme_etf_daily")


def _mirror_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute factorized ETF-based daily theme strength scores.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format. Default: end-date - 220 calendar days.")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="End date in YYYY-MM-DD format. Default: today.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV path.")
    return parser.parse_args()


def parse_date(raw_value: str) -> date:
    return datetime.strptime(raw_value, "%Y-%m-%d").date()


def _to_numeric(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if hasattr(converted, "replace"):
        return converted.replace([np.inf, -np.inf], np.nan)
    if converted in (np.inf, -np.inf):
        return np.nan
    return converted


def _clip01(series: pd.Series) -> pd.Series:
    return _to_numeric(series).fillna(0.0).clip(lower=0.0, upper=1.0)


def winsorize_series_if_needed(series: pd.Series, lower_q: float = WINSOR_LOWER_Q, upper_q: float = WINSOR_UPPER_Q) -> pd.Series:
    values = _to_numeric(series)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    return values.clip(lower=float(valid.quantile(lower_q)), upper=float(valid.quantile(upper_q)))


def safe_rank_or_scale(series: pd.Series) -> pd.Series:
    winsorized = winsorize_series_if_needed(series)
    valid = winsorized.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    if valid.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    return (winsorized.rank(method="average", pct=True) * 100.0).fillna(0.0).clip(lower=0.0, upper=OUTLIER_SCORE_CAP)


def load_theme_etf_master(path: Path = THEME_ETF_MASTER_CSV) -> pd.DataFrame:
    if not path.exists() and path.resolve() == THEME_ETF_MASTER_CSV.resolve():
        ensure_theme_mapping_files()
    if not path.exists():
        raise FileNotFoundError(f"theme_etf_master.csv not found: {path}")
    df = _load_theme_etf_master_seed(ensure_exists=True)
    if df.empty:
        raise ValueError(f"theme_etf_master.csv is empty: {path}")
    df["theme_id"] = df["theme_id"].fillna("").astype(str).str.upper().str.strip()
    df["theme_name"] = df["theme_name"].fillna("").astype(str)
    df["etf_code"] = df["etf_code"].astype(str).str.zfill(6)
    df["etf_name"] = df["etf_name"].fillna("").astype(str)
    LOGGER.info("Loaded theme ETF master rows=%d path=%s", len(df), path)
    return df


def load_etf_price_data(etf_codes: set[str]) -> pd.DataFrame:
    required = {"date", "code", "close", "volume"}
    optional_aliases = {
        "trading_value": ["trading_value", "value", "tradingvalue", "amt", "amount"],
        "aum": ["aum", "fund_aum", "net_assets"],
        "shares_outstanding": ["shares_outstanding", "shares", "fund_shares", "listed_shares"],
        "nav": ["nav", "iopv", "nav_close"],
    }
    for path in [PRICES_RAW_CSV, PRICES_CLEAN_CSV]:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        columns = {str(c).strip().lower(): c for c in df.columns}
        if not required.issubset(columns.keys()):
            continue
        selected = {"date": columns["date"], "code": columns["code"], "close": columns["close"], "volume": columns["volume"]}
        for target, aliases in optional_aliases.items():
            for alias in aliases:
                if alias in columns:
                    selected[target] = columns[alias]
                    break
        out = df.loc[:, list(selected.values())].copy()
        out.columns = list(selected.keys())
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["code"] = out["code"].astype(str).str.zfill(6)
        for col in [c for c in out.columns if c not in {"date", "code"}]:
            out[col] = _to_numeric(out[col])
        if "trading_value" not in out.columns:
            out["trading_value"] = out["close"] * out["volume"]
        matched_count = int(out["code"].isin(etf_codes).sum())
        LOGGER.info("Loaded price CSV path=%s rows=%d matched_etf_rows=%d", path, len(out), matched_count)
        return out.dropna(subset=["date", "code", "close"]).sort_values(["code", "date"]).reset_index(drop=True)
    LOGGER.warning("No compatible local price CSV found for ETF prices")
    return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value", "aum", "shares_outstanding", "nav"])


def load_benchmark_data(price_df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if not price_df.empty:
        bench = price_df.loc[price_df["code"].isin(KOSPI_BENCHMARK_CODES)].copy()
        if not bench.empty:
            bench = bench.sort_values("date")
            bench = bench.loc[(bench["date"].dt.date >= start_date) & (bench["date"].dt.date <= end_date)].copy()
            bench = bench.groupby("date", as_index=False).agg(kospi_close=("close", "last"))
            LOGGER.info("Loaded benchmark from local price data rows=%d", len(bench))
            return bench
    LOGGER.warning("Benchmark data not found in local prices; rs_vs_kospi_20d will fallback to ret_20d")
    return pd.DataFrame(columns=["date", "kospi_close"])


def _build_theme_proxy_prices(
    local_price_df: pd.DataFrame,
    stock_theme_df: pd.DataFrame,
    theme_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if local_price_df.empty or stock_theme_df.empty:
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value"])
    mappings = stock_theme_df.loc[stock_theme_df["theme_id"].fillna("").astype(str).str.upper() == str(theme_id).upper()].copy()
    if mappings.empty:
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value"])
    mappings["code"] = mappings["code"].astype(str).str.zfill(6)
    mappings["mapping_weight"] = _to_numeric(mappings["mapping_weight"]).fillna(0.0).clip(lower=0.0)
    prices = local_price_df.loc[local_price_df["code"].isin(mappings["code"])].copy()
    if prices.empty:
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value"])
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.loc[(prices["date"].dt.date >= start_date) & (prices["date"].dt.date <= end_date)].copy()
    if prices.empty:
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value"])
    prices = prices.merge(mappings.loc[:, ["code", "mapping_weight"]], on="code", how="inner")
    prices["close"] = _to_numeric(prices["close"])
    prices["volume"] = _to_numeric(prices.get("volume")).fillna(0.0)
    prices["trading_value"] = _to_numeric(prices.get("trading_value")).fillna(prices["close"] * prices["volume"])
    prices = prices.dropna(subset=["date", "close"]).sort_values(["code", "date"]).copy()
    if prices.empty:
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value"])
    base_close = prices.groupby("code")["close"].transform("first")
    prices["rebased_close"] = np.where(base_close > 0, prices["close"] / base_close * 100.0, np.nan)
    prices["weighted_close"] = prices["rebased_close"] * prices["mapping_weight"]
    prices["weighted_volume"] = prices["volume"] * prices["mapping_weight"]
    prices["weighted_trading_value"] = prices["trading_value"] * prices["mapping_weight"]
    grouped = prices.groupby("date", as_index=False).agg(
        weighted_close=("weighted_close", "sum"),
        total_weight=("mapping_weight", "sum"),
        volume=("weighted_volume", "sum"),
        trading_value=("weighted_trading_value", "sum"),
    )
    grouped["close"] = np.where(grouped["total_weight"] > 0, grouped["weighted_close"] / grouped["total_weight"], np.nan)
    grouped["code"] = f"THEME_{str(theme_id).upper()}"
    return grouped.loc[:, ["date", "code", "close", "volume", "trading_value"]].dropna(subset=["date", "close"]).reset_index(drop=True)


def compute_etf_features(price_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["close"] = _to_numeric(out["close"])
    out["volume"] = _to_numeric(out["volume"]).fillna(0.0)
    out["trading_value"] = _to_numeric(out.get("trading_value")).fillna(out["close"] * out["volume"])
    out["aum"] = _to_numeric(out.get("aum"))
    out["shares_outstanding"] = _to_numeric(out.get("shares_outstanding"))
    out["nav"] = _to_numeric(out.get("nav"))

    out["ret_1d"] = out["close"].pct_change(1)
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_20d"] = out["close"].pct_change(20)
    out["ret_60d"] = out["close"].pct_change(60)

    out["ma20"] = out["close"].rolling(20, min_periods=10).mean()
    out["ma60"] = out["close"].rolling(60, min_periods=20).mean()
    out["ma20_gap"] = np.where(out["ma20"] > 0, out["close"] / out["ma20"] - 1.0, np.nan)
    out["ma60_gap"] = np.where(out["ma60"] > 0, out["close"] / out["ma60"] - 1.0, np.nan)

    vol_ma20 = out["volume"].rolling(20, min_periods=10).mean()
    value_ma20 = out["trading_value"].rolling(20, min_periods=10).mean()
    value_ma5 = out["trading_value"].rolling(5, min_periods=3).mean()
    out["vol_ratio_20d"] = np.where(vol_ma20 > 0, out["volume"] / vol_ma20, np.nan)
    out["trading_value_ratio_20d"] = np.where(value_ma20 > 0, out["trading_value"] / value_ma20, np.nan)
    out["abnormal_value_5d"] = np.where(value_ma20 > 0, value_ma5 / value_ma20, np.nan)

    returns = out["close"].pct_change()
    out["positive_day_ratio_20d"] = (returns.gt(0).rolling(20, min_periods=10).mean()).clip(lower=0.0, upper=1.0)
    out["volatility_20d"] = returns.rolling(20, min_periods=10).std()
    out["ret_vol_ratio_20d"] = np.where(out["volatility_20d"] > 0, out["ret_20d"] / out["volatility_20d"], np.nan)

    shares = out["shares_outstanding"]
    out["turnover_ratio"] = np.where(shares > 0, out["volume"] / shares, np.nan)

    if out["nav"].notna().any():
        out["nav_gap"] = np.where(out["nav"] > 0, out["close"] / out["nav"] - 1.0, np.nan)
        nav_ret = out["nav"].pct_change()
        close_ret = out["close"].pct_change()
        out["tracking_error_20d"] = (close_ret - nav_ret).rolling(20, min_periods=10).std()
    else:
        out["nav_gap"] = np.nan
        out["tracking_error_20d"] = np.nan

    if not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
        bench = bench.sort_values("date")
        bench["kospi_ret_20d"] = bench["kospi_close"].pct_change(20)
        out = out.merge(bench.loc[:, ["date", "kospi_ret_20d"]], on="date", how="left")
        out["rs_vs_kospi_20d"] = out["ret_20d"] - _to_numeric(out["kospi_ret_20d"])
        out = out.drop(columns=["kospi_ret_20d"])
    else:
        out["rs_vs_kospi_20d"] = out["ret_20d"]
    return out


def compute_trend_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ret_20d_score"] = safe_rank_or_scale(out["ret_20d"])
    out["ret_60d_score"] = safe_rank_or_scale(out["ret_60d"])
    out["ma20_gap_score"] = safe_rank_or_scale(out["ma20_gap"])
    out["rs_vs_kospi_score"] = safe_rank_or_scale(out["rs_vs_kospi_20d"])
    out["trend_score"] = (
        0.35 * out["ret_20d_score"]
        + 0.15 * out["ret_60d_score"]
        + 0.20 * out["ma20_gap_score"]
        + 0.30 * out["rs_vs_kospi_score"]
    ).clip(lower=0.0, upper=100.0)
    return out


def compute_activity_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["vol_ratio_score"] = safe_rank_or_scale(out["vol_ratio_20d"])
    out["trading_value_ratio_20d_score"] = safe_rank_or_scale(out["trading_value_ratio_20d"])
    out["abnormal_value_5d_score"] = safe_rank_or_scale(out["abnormal_value_5d"])
    out["turnover_ratio_score"] = safe_rank_or_scale(out["turnover_ratio"]) if out["turnover_ratio"].notna().any() else 50.0
    out["activity_score"] = (
        0.30 * out["vol_ratio_score"]
        + 0.35 * out["trading_value_ratio_20d_score"]
        + 0.25 * out["abnormal_value_5d_score"]
        + 0.10 * out["turnover_ratio_score"]
    ).clip(lower=0.0, upper=100.0)
    return out


def _compute_aum_flow_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    aum = _to_numeric(out["aum"])
    out["aum_prev_1d"] = aum.shift(1)
    out["aum_prev_5d"] = aum.shift(5)
    out["aum_prev_20d"] = aum.shift(20)
    out["aum_flow_ratio_1d"] = np.where(
        out["aum_prev_1d"].abs() > EPS,
        (aum - out["aum_prev_1d"] * (1.0 + _to_numeric(out["ret_1d"]).fillna(0.0))) / out["aum_prev_1d"].abs(),
        np.nan,
    )
    out["aum_flow_ratio_5d"] = np.where(
        out["aum_prev_5d"].abs() > EPS,
        (aum - out["aum_prev_5d"] * (1.0 + _to_numeric(out["ret_5d"]).fillna(0.0))) / out["aum_prev_5d"].abs(),
        np.nan,
    )
    out["aum_flow_ratio_20d"] = np.where(
        out["aum_prev_20d"].abs() > EPS,
        (aum - out["aum_prev_20d"] * (1.0 + _to_numeric(out["ret_20d"]).fillna(0.0))) / out["aum_prev_20d"].abs(),
        np.nan,
    )
    return out


def _compute_share_flow_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    shares = _to_numeric(out["shares_outstanding"])
    out["shares_prev_1d"] = shares.shift(1)
    out["shares_prev_5d"] = shares.shift(5)
    out["shares_prev_20d"] = shares.shift(20)
    out["shares_flow_ratio_1d"] = np.where(out["shares_prev_1d"].abs() > EPS, (shares - out["shares_prev_1d"]) / out["shares_prev_1d"].abs(), np.nan)
    out["shares_flow_ratio_5d"] = np.where(out["shares_prev_5d"].abs() > EPS, (shares - out["shares_prev_5d"]) / out["shares_prev_5d"].abs(), np.nan)
    out["shares_flow_ratio_20d"] = np.where(out["shares_prev_20d"].abs() > EPS, (shares - out["shares_prev_20d"]) / out["shares_prev_20d"].abs(), np.nan)
    return out


def compute_flow_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = _compute_aum_flow_ratios(frame)
    out = _compute_share_flow_ratios(out)
    out["price_up_with_value_expansion"] = (
        _clip01((_to_numeric(out["ret_5d"]).fillna(0.0) + 0.05) / 0.10)
        * _clip01((_to_numeric(out["trading_value_ratio_20d"]).fillna(0.0) - 0.8) / 0.8)
    )
    out["flow_proxy_score"] = (
        0.45 * safe_rank_or_scale(out["trading_value_ratio_20d"])
        + 0.25 * safe_rank_or_scale(out["abnormal_value_5d"])
        + 0.30 * safe_rank_or_scale(out["price_up_with_value_expansion"])
    ).clip(lower=0.0, upper=100.0)

    aum_score = (
        0.25 * safe_rank_or_scale(out["aum_flow_ratio_1d"])
        + 0.45 * safe_rank_or_scale(out["aum_flow_ratio_5d"])
        + 0.30 * safe_rank_or_scale(out["aum_flow_ratio_20d"])
    ).clip(lower=0.0, upper=100.0)
    shares_score = (
        0.25 * safe_rank_or_scale(out["shares_flow_ratio_1d"])
        + 0.45 * safe_rank_or_scale(out["shares_flow_ratio_5d"])
        + 0.30 * safe_rank_or_scale(out["shares_flow_ratio_20d"])
    ).clip(lower=0.0, upper=100.0)

    aum_available = out[["aum_flow_ratio_1d", "aum_flow_ratio_5d", "aum_flow_ratio_20d"]].notna().any(axis=1)
    shares_available = out[["shares_flow_ratio_1d", "shares_flow_ratio_5d", "shares_flow_ratio_20d"]].notna().any(axis=1)
    out["flow_data_available"] = aum_available | shares_available
    out["flow_source"] = np.select(
        [aum_available, shares_available],
        ["aum", "shares_outstanding"],
        default="trading_value_proxy",
    )
    out["flow_score"] = np.select(
        [aum_available, shares_available],
        [aum_score, shares_score],
        default=out["flow_proxy_score"],
    )
    out["flow_ratio_1d"] = np.select(
        [aum_available, shares_available],
        [out["aum_flow_ratio_1d"], out["shares_flow_ratio_1d"]],
        default=np.where(out["trading_value_ratio_20d"].abs() > EPS, _to_numeric(out["trading_value_ratio_20d"]) - 1.0, np.nan),
    )
    out["flow_ratio_5d"] = np.select(
        [aum_available, shares_available],
        [out["aum_flow_ratio_5d"], out["shares_flow_ratio_5d"]],
        default=np.where(out["abnormal_value_5d"].abs() > EPS, _to_numeric(out["abnormal_value_5d"]) - 1.0, np.nan),
    )
    out["flow_ratio_20d"] = np.select(
        [aum_available, shares_available],
        [out["aum_flow_ratio_20d"], out["shares_flow_ratio_20d"]],
        default=np.where(out["trading_value_ratio_20d"].abs() > EPS, _to_numeric(out["trading_value_ratio_20d"]) - 1.0, np.nan),
    )
    return out


def compute_stability_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pos_score = safe_rank_or_scale(out["positive_day_ratio_20d"])
    ret_vol_score = safe_rank_or_scale(out["ret_vol_ratio_20d"])
    nav_quality_score = (
        100.0
        - (
            0.60 * safe_rank_or_scale(_to_numeric(out["nav_gap"]).abs())
            + 0.40 * safe_rank_or_scale(_to_numeric(out["tracking_error_20d"]).abs())
        )
    ).clip(lower=0.0, upper=100.0)
    if out["nav_gap"].isna().all() and out["tracking_error_20d"].isna().all():
        nav_quality_score = pd.Series(50.0, index=out.index)

    overheat_ret = _clip01((_to_numeric(out["ret_5d"]).fillna(0.0) - 0.08) / 0.12)
    overheat_value = _clip01((_to_numeric(out["trading_value_ratio_20d"]).fillna(1.0) - 1.8) / 1.8)
    out["overheat_penalty"] = (100.0 * 0.5 * overheat_ret * overheat_value).clip(lower=0.0, upper=40.0)
    out["stability_score"] = (
        0.35 * pos_score
        + 0.35 * ret_vol_score
        + 0.30 * nav_quality_score
        - out["overheat_penalty"]
    ).clip(lower=0.0, upper=100.0)
    return out


def compute_cross_etf_support_hint(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["cross_etf_support_hint"] = 0.5
    if len(out) <= 1:
        return out
    support_rank = (out["etf_theme_score_raw"].rank(method="average", pct=True)).fillna(0.5)
    breadth_hint = min(1.0, len(out) / 3.0)
    out["cross_etf_support_hint"] = (0.60 * support_rank + 0.40 * breadth_hint).clip(lower=0.0, upper=1.0)
    return out


def compute_etf_signal_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    core_non_null = out[["ret_20d", "ret_60d", "ma20_gap", "rs_vs_kospi_20d", "vol_ratio_20d", "trading_value_ratio_20d"]].notna().mean(axis=1)
    optional_non_null = out[["aum", "shares_outstanding", "nav"]].notna().mean(axis=1)
    out["data_quality_conf"] = (0.75 * core_non_null + 0.25 * optional_non_null).clip(lower=0.0, upper=1.0)

    source_bonus = out["flow_source"].map({"aum": 1.0, "shares_outstanding": 0.8, "trading_value_proxy": 0.45}).fillna(0.35)
    strength_conf = _clip01((_to_numeric(out["flow_score"]).fillna(0.0) - 40.0) / 35.0)
    out["flow_evidence_conf"] = (0.60 * source_bonus + 0.40 * strength_conf).clip(lower=0.0, upper=1.0)
    out["stability_conf"] = _clip01(_to_numeric(out["stability_score"]).fillna(0.0) / 100.0)
    out["cross_etf_support_hint"] = _clip01(out["cross_etf_support_hint"])
    out["etf_signal_confidence"] = (
        CONFIDENCE_WEIGHTS["data_quality_conf"] * out["data_quality_conf"]
        + CONFIDENCE_WEIGHTS["flow_evidence_conf"] * out["flow_evidence_conf"]
        + CONFIDENCE_WEIGHTS["cross_etf_support_hint"] * out["cross_etf_support_hint"]
        + CONFIDENCE_WEIGHTS["stability_conf"] * out["stability_conf"]
    ).clip(lower=0.0, upper=1.0)
    return out


def classify_theme_regime(score: Any) -> str:
    value = float(_to_numeric(pd.Series([score])).fillna(0.0).iloc[0])
    if value >= THEME_REGIME_STRONG:
        return "strong"
    if value >= THEME_REGIME_NEUTRAL:
        return "neutral"
    return "weak"


def build_explain_etf_theme(row: pd.Series) -> str:
    return (
        f"trend={float(row.get('trend_score', 0.0)):.1f}(ret20={float(_to_numeric(pd.Series([row.get('ret_20d')])).fillna(0.0).iloc[0])*100:.1f}%,"
        f"rs20={float(_to_numeric(pd.Series([row.get('rs_vs_kospi_20d')])).fillna(0.0).iloc[0])*100:.1f}%), "
        f"activity={float(row.get('activity_score', 0.0)):.1f}(vol_ratio={float(_to_numeric(pd.Series([row.get('vol_ratio_20d')])).fillna(0.0).iloc[0]):.2f},"
        f"value_ratio={float(_to_numeric(pd.Series([row.get('trading_value_ratio_20d')])).fillna(0.0).iloc[0]):.2f}), "
        f"flow={float(row.get('flow_score', 0.0)):.1f}(source={row.get('flow_source', 'unknown')},"
        f"flow5={float(_to_numeric(pd.Series([row.get('flow_ratio_5d')])).fillna(0.0).iloc[0]):.4f}), "
        f"stability={float(row.get('stability_score', 0.0)):.1f}(pos_day_ratio={float(_to_numeric(pd.Series([row.get('positive_day_ratio_20d')])).fillna(0.0).iloc[0]):.2f},"
        f"overheat_penalty={float(_to_numeric(pd.Series([row.get('overheat_penalty')])).fillna(0.0).iloc[0]):.1f}), "
        f"conf={float(_to_numeric(pd.Series([row.get('etf_signal_confidence')])).fillna(0.0).iloc[0]):.2f}"
    )


def _build_summary(theme_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for as_of_date, grp in theme_df.groupby("date", dropna=False):
        top = grp.sort_values("etf_theme_score", ascending=False).head(3).reset_index(drop=True)
        flow_source_counts = grp["flow_source"].fillna("unknown").value_counts().to_dict()
        row: dict[str, Any] = {
            "date": as_of_date,
            "total_themes": int(grp["theme_id"].nunique()),
            "avg_etf_theme_score": float(_to_numeric(grp["etf_theme_score"]).fillna(0.0).mean()),
            "avg_etf_signal_confidence": float(_to_numeric(grp["etf_signal_confidence"]).fillna(0.0).mean()),
            "avg_trend_score": float(_to_numeric(grp["trend_score"]).fillna(0.0).mean()),
            "avg_activity_score": float(_to_numeric(grp["activity_score"]).fillna(0.0).mean()),
            "avg_flow_score": float(_to_numeric(grp["flow_score"]).fillna(0.0).mean()),
            "avg_stability_score": float(_to_numeric(grp["stability_score"]).fillna(0.0).mean()),
            "flow_data_available_ratio": float(grp["flow_data_available"].fillna(False).astype(bool).mean()),
            "flow_source_counts_json": json.dumps(flow_source_counts, ensure_ascii=False),
            "strong_count": int((grp["theme_regime"] == "strong").sum()),
            "neutral_count": int((grp["theme_regime"] == "neutral").sum()),
            "weak_count": int((grp["theme_regime"] == "weak").sum()),
        }
        for idx in range(3):
            row[f"top_theme_{idx + 1}"] = str(top.iloc[idx]["theme_name"]) if idx < len(top) else ""
            row[f"top_theme_{idx + 1}_score"] = float(top.iloc[idx]["etf_theme_score"]) if idx < len(top) else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _build_validation_latest_rank(theme_df: pd.DataFrame) -> pd.DataFrame:
    if theme_df.empty:
        return pd.DataFrame(columns=["date", "rank", "theme_id", "theme_name", "etf_code", "etf_name", "etf_theme_score", "etf_signal_confidence", "flow_source", "explain_etf_theme"])
    latest_date = theme_df["date"].max()
    latest = theme_df.loc[theme_df["date"] == latest_date].copy()
    latest = latest.sort_values(["etf_theme_score", "etf_signal_confidence", "theme_id"], ascending=[False, False, True]).reset_index(drop=True)
    latest["rank"] = range(1, len(latest) + 1)
    return latest.loc[:, ["date", "rank", "theme_id", "theme_name", "etf_code", "etf_name", "etf_theme_score", "etf_signal_confidence", "flow_source", "explain_etf_theme"]]


def _build_validation_markdown(theme_df: pd.DataFrame, latest_rank_df: pd.DataFrame, start_date: date, end_date: date) -> str:
    if theme_df.empty:
        return "# Theme ETF Validation\n\nNo ETF theme rows were generated.\n"
    flow_ratio = float(theme_df["flow_data_available"].fillna(False).astype(bool).mean())
    flow_counts = theme_df["flow_source"].fillna("unknown").value_counts().to_dict()
    conf = _to_numeric(theme_df["etf_signal_confidence"]).fillna(0.0)
    lines = [
        "# Theme ETF Validation",
        "",
        f"- Window: {start_date.isoformat()} to {end_date.isoformat()}",
        f"- Total ETF-theme rows: {len(theme_df)}",
        f"- Flow data available ratio: {flow_ratio:.2%}",
        f"- Flow source counts: `{json.dumps(flow_counts, ensure_ascii=False)}`",
        f"- Confidence summary: mean={conf.mean():.3f}, p50={conf.quantile(0.5):.3f}, p90={conf.quantile(0.9):.3f}",
        "",
        "## Scoring Structure",
        "",
        "- `trend_score`: ret_20d, ret_60d, ma20_gap, rs_vs_kospi_20d",
        "- `activity_score`: volume ratio, trading value ratio, abnormal value burst, turnover proxy",
        "- `flow_score`: real flow(AUM / shares outstanding) 우선, 없으면 trading value proxy fallback",
        "- `stability_score`: positive-day ratio, return-vol quality, NAV/tracking quality, overheat penalty",
        "- `etf_theme_score_raw = 0.30*trend + 0.20*activity + 0.35*flow + 0.15*stability`",
        "- `etf_theme_score`: daily robust percentile scaling of raw score",
        "- `etf_signal_confidence = 0.40*data_quality + 0.25*flow_evidence + 0.20*cross_etf_support + 0.15*stability_conf`",
        "",
        "## Latest Top Themes",
        "",
    ]
    for _, row in latest_rank_df.head(10).iterrows():
        lines.append(
            f"- #{int(row['rank'])} {row['theme_name']} / {row['etf_name']} "
            f"(score={float(row['etf_theme_score']):.1f}, conf={float(row['etf_signal_confidence']):.2f}, flow={row['flow_source']})"
        )
    lines.append("")
    return "\n".join(lines)


def export_theme_etf_validation(theme_df: pd.DataFrame, summary_df: pd.DataFrame, start_date: date, end_date: date) -> None:
    latest_rank_df = _build_validation_latest_rank(theme_df)
    latest_rank_df.to_csv(LATEST_RANK_CSV, index=False, encoding="utf-8-sig")
    _mirror_file(LATEST_RANK_CSV, DATA_LATEST_RANK_CSV)
    validation_md = _build_validation_markdown(theme_df, latest_rank_df, start_date, end_date)
    VALIDATION_MD.write_text(validation_md, encoding="utf-8")
    DATA_VALIDATION_MD.write_text(validation_md, encoding="utf-8")


def export_debug_outputs(theme_df: pd.DataFrame) -> None:
    debug_cols = [
        "date", "theme_id", "theme_name", "etf_code", "etf_name", "etf_theme_score", "etf_signal_confidence",
        "trend_score", "activity_score", "flow_score", "flow_proxy_score", "stability_score",
        "flow_data_available", "flow_source", "data_quality_conf", "flow_evidence_conf", "stability_conf", "cross_etf_support_hint",
        "trading_value", "trading_value_ratio_20d", "abnormal_value_5d", "flow_ratio_1d", "flow_ratio_5d", "flow_ratio_20d",
        "positive_day_ratio_20d", "volatility_20d", "overheat_penalty", "explain_etf_theme",
    ]
    debug_df = theme_df.loc[:, [c for c in debug_cols if c in theme_df.columns]].copy()
    DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(DEBUG_CSV, index=False, encoding="utf-8-sig")
    _mirror_file(DEBUG_CSV, DATA_DEBUG_CSV)


def build_theme_etf_daily(start_date: date, end_date: date) -> pd.DataFrame:
    theme_master = load_theme_etf_master()
    stock_theme_map = load_stock_theme_map(ensure_exists=True)
    all_price_df = load_etf_price_data(set(theme_master["etf_code"].astype(str).str.zfill(6)))
    benchmark_df = load_benchmark_data(all_price_df, start_date, end_date)

    result_frames: list[pd.DataFrame] = []
    for row in theme_master.itertuples(index=False):
        theme_id = str(row.theme_id).upper()
        theme_name = str(row.theme_name)
        etf_code = str(row.etf_code).zfill(6)
        etf_name = str(row.etf_name)

        if etf_code.isdigit() and etf_code != "000000" and (all_price_df["code"] == etf_code).any():
            price_df = all_price_df.loc[all_price_df["code"] == etf_code].copy()
            source_name = "etf_price"
        else:
            price_df = _build_theme_proxy_prices(all_price_df, stock_theme_map, theme_id, start_date, end_date)
            source_name = "theme_proxy"
        if price_df.empty:
            LOGGER.warning("Skipping theme=%s etf=%s because no price history is available", theme_id, etf_code)
            continue
        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        price_df = price_df.loc[(price_df["date"].dt.date >= start_date) & (price_df["date"].dt.date <= end_date)].copy()
        if price_df.empty:
            continue
        feature_df = compute_etf_features(price_df, benchmark_df)
        feature_df = compute_trend_score(feature_df)
        feature_df = compute_activity_score(feature_df)
        feature_df = compute_flow_score(feature_df)
        feature_df = compute_stability_score(feature_df)
        feature_df["etf_theme_score_raw"] = (
            SUB_SCORE_WEIGHTS["trend_score"] * feature_df["trend_score"]
            + SUB_SCORE_WEIGHTS["activity_score"] * feature_df["activity_score"]
            + SUB_SCORE_WEIGHTS["flow_score"] * feature_df["flow_score"]
            + SUB_SCORE_WEIGHTS["stability_score"] * feature_df["stability_score"]
        )
        feature_df["theme_id"] = theme_id
        feature_df["theme"] = theme_name
        feature_df["theme_name"] = theme_name
        feature_df["etf_code"] = etf_code
        feature_df["etf_name"] = etf_name
        feature_df["source_name"] = source_name
        result_frames.append(feature_df)

    if not result_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    theme_df = pd.concat(result_frames, ignore_index=True)
    theme_df["date"] = pd.to_datetime(theme_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    theme_df = theme_df.dropna(subset=["date"]).copy()
    theme_df["etf_theme_score"] = theme_df.groupby("date", dropna=False)["etf_theme_score_raw"].transform(safe_rank_or_scale)
    theme_groups: list[pd.DataFrame] = []
    for _, grp in theme_df.groupby(["date", "theme_id"], dropna=False):
        theme_groups.append(compute_cross_etf_support_hint(grp))
    theme_df = pd.concat(theme_groups, ignore_index=True) if theme_groups else theme_df
    theme_df = compute_etf_signal_confidence(theme_df)
    theme_df["signal_regime"] = theme_df["etf_theme_score"].apply(classify_theme_regime)
    theme_df["theme_regime"] = theme_df["signal_regime"]
    theme_df["explain_etf_theme"] = theme_df.apply(build_explain_etf_theme, axis=1)

    for col in OUTPUT_COLUMNS:
        if col not in theme_df.columns:
            theme_df[col] = np.nan
    theme_df = theme_df.loc[:, OUTPUT_COLUMNS].sort_values(["date", "etf_theme_score", "theme_id", "etf_code"], ascending=[True, False, True, True]).reset_index(drop=True)
    return theme_df


def export_theme_etf_daily(theme_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theme_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if output_path.resolve() != OUTPUT_CSV.resolve():
        OUTPUT_CSV.write_bytes(output_path.read_bytes())
    _mirror_file(OUTPUT_CSV, DATA_OUTPUT_CSV)

    summary_df = _build_summary(theme_df)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    _mirror_file(SUMMARY_CSV, DATA_SUMMARY_CSV)
    export_debug_outputs(theme_df)
    return summary_df


def _print_summary(theme_df: pd.DataFrame) -> None:
    if theme_df.empty:
        LOGGER.warning("No ETF theme rows generated")
        return
    flow_ratio = float(theme_df["flow_data_available"].fillna(False).astype(bool).mean())
    flow_counts = theme_df["flow_source"].fillna("unknown").value_counts().to_dict()
    conf = _to_numeric(theme_df["etf_signal_confidence"]).fillna(0.0)
    LOGGER.info("ETF theme rows=%d unique_dates=%d unique_themes=%d", len(theme_df), theme_df["date"].nunique(), theme_df["theme_id"].nunique())
    LOGGER.info("flow_data_available_ratio=%.2f%%", flow_ratio * 100.0)
    LOGGER.info("flow_source_counts=%s", json.dumps(flow_counts, ensure_ascii=False))
    LOGGER.info(
        "etf_signal_confidence mean=%.3f p50=%.3f p90=%.3f max=%.3f",
        float(conf.mean()),
        float(conf.quantile(0.5)),
        float(conf.quantile(0.9)),
        float(conf.max()),
    )


def main() -> int:
    setup_logging()
    args = parse_args()
    end_date = parse_date(args.end_date)
    start_date = parse_date(args.start_date) if args.start_date else (end_date - timedelta(days=LOOKBACK_CALENDAR_DAYS))
    LOGGER.info("Building factorized theme ETF daily scores start=%s end=%s", start_date, end_date)
    theme_df = build_theme_etf_daily(start_date, end_date)
    summary_df = export_theme_etf_daily(theme_df, args.output)
    export_theme_etf_validation(theme_df, summary_df, start_date, end_date)
    _print_summary(theme_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
