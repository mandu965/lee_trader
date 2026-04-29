from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_FEATURES = DATA_DIR / "features.csv"
DEFAULT_PRICES = DATA_DIR / "prices_daily_adjusted.csv"
DEFAULT_UNIVERSE = DATA_DIR / "universe.csv"
DEFAULT_MARKET = DATA_DIR / "market_status.csv"
DEFAULT_OUT = DATA_DIR / "rule_signals.csv"

STRATEGY_ID = "RULE_TREND_LIQUIDITY_V1"
ENGINE_TYPE = "rule_based"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RULE_TREND_LIQUIDITY_V1 paper signals.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--market-status-csv", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-mode", default=os.getenv("RULE_TRADING_RUN_MODE", "paper"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_by_date(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = numeric(df[col])
    return values.groupby(df["date"]).rank(pct=True, ascending=ascending, method="average") * 100.0


def get_min_trading_value(run_mode: str) -> float:
    mode = str(run_mode or "paper").strip().lower()
    defaults = {
        "paper": 500_000_000.0,
        "pilot": 1_000_000_000.0,
        "live": 2_000_000_000.0,
    }
    env_name = f"RULE_MIN_TRADING_VALUE_MA20_{mode.upper()}"
    return float(os.getenv(env_name, defaults.get(mode, defaults["paper"])))


def get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def load_features(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"features csv not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features csv must contain date and code")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.dropna(subset=["date", "code"]).sort_values(["code", "date"])
    return df


def load_prices(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "code", "open", "close"])
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    if "date" not in df.columns or "code" not in df.columns:
        return pd.DataFrame(columns=["date", "code", "open", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    out = pd.DataFrame({"date": df["date"], "code": df["code"]})
    out["open"] = numeric(df["adj_open"] if "adj_open" in df.columns else df.get("open"))
    out["close_price_raw"] = numeric(df["adj_close"] if "adj_close" in df.columns else df.get("close"))
    return out.dropna(subset=["date", "code"]).sort_values(["code", "date"])


def load_universe(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        return pd.DataFrame(columns=["code", "name", "sector", "market"])
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    if "code" not in df.columns:
        return pd.DataFrame(columns=["code", "name", "sector", "market"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in ["name", "sector", "market"]:
        if col not in df.columns:
            df[col] = None
    return df[["code", "name", "sector", "market"]].drop_duplicates("code")


def load_market(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for col in ["kospi_close", "kospi_ma20", "volatility_5d"]:
        if col in df.columns:
            df[col] = numeric(df[col])
    if "kospi_close" in df.columns:
        df["kospi_momentum_20d"] = df["kospi_close"].pct_change(20)
    else:
        df["kospi_momentum_20d"] = np.nan
    if "volatility_5d" in df.columns:
        df["volatility_5d_rising"] = df["volatility_5d"] > df["volatility_5d"].shift(5)
        vol_threshold = df["volatility_5d"].rolling(60, min_periods=20).quantile(0.8)
        df["volatility_5d_high"] = df["volatility_5d"] > vol_threshold
    else:
        df["volatility_5d_rising"] = False
        df["volatility_5d_high"] = False
    return df


def attach_price_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if not prices.empty:
        df = df.merge(prices, on=["date", "code"], how="left")
        if "close_price_raw" in df.columns:
            df["close"] = numeric(df.get("close")).fillna(numeric(df["close_price_raw"]))
            df = df.drop(columns=["close_price_raw"])
    if "open" not in df.columns:
        df["open"] = np.nan
    df["close"] = numeric(df.get("close"))
    df["volume"] = numeric(df.get("volume"))
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["previous_close"] = df.groupby("code")["close"].shift(1)
    df["open_gap"] = numeric(df["open"]) / numeric(df["previous_close"]) - 1.0
    df["expected_gap"] = (
        df.groupby("code")["open_gap"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
        .fillna(0.0)
    )
    df["expected_entry_price"] = df["close"] * (1.0 + df["expected_gap"])
    df["next_open"] = df.groupby("code")["open"].shift(-1)
    df["actual_open_gap"] = df["next_open"] / df["close"] - 1.0
    return df


def attach_market(features: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if market.empty:
        df["regime_risk_count"] = 0
        df["regime_risk_flag"] = False
        df["market_entry_allowed"] = True
        df["market_defensive_mode"] = False
        return df
    keep = [
        col
        for col in [
            "date",
            "kospi_close",
            "kospi_ma20",
            "kospi_momentum_20d",
            "volatility_5d",
            "volatility_5d_rising",
            "volatility_5d_high",
            "regime",
        ]
        if col in market.columns
    ]
    df = df.merge(market[keep], on="date", how="left")
    risk_1 = numeric(df.get("kospi_close")) < numeric(df.get("kospi_ma20"))
    risk_2 = numeric(df.get("kospi_momentum_20d")) < 0
    risk_3_left = df.get("volatility_5d_rising", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    risk_3_right = df.get("volatility_5d_high", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    risk_3 = risk_3_left | risk_3_right
    df["regime_risk_count"] = risk_1.fillna(False).astype(int) + risk_2.fillna(False).astype(int) + risk_3.fillna(False).astype(int)
    df["regime_risk_flag"] = df["regime_risk_count"] >= 2
    df["market_entry_allowed"] = ~df["regime_risk_flag"]
    df["market_defensive_mode"] = df["regime_risk_flag"]
    return df


def build_rule_scores(df: pd.DataFrame, run_mode: str) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "ma_5",
        "ma_20",
        "ma_60",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
        "vol_20",
        "close_over_ma20",
        "volume_ratio_20d",
        "value_ratio_20d",
        "liquidity_score",
        "quality_missing_ratio",
        "quality_score_confidence",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = numeric(out[col])

    out["trading_value"] = out["close"] * out["volume"]
    out["trading_value_ma_5"] = out.groupby("code")["trading_value"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    out["trading_value_ma_20"] = out.groupby("code")["trading_value"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["trading_value_ratio_5d"] = out["trading_value"] / out["trading_value_ma_5"].replace(0, np.nan)
    out["trading_value_ratio_20d"] = out["trading_value"] / out["trading_value_ma_20"].replace(0, np.nan)

    min_trading_value = get_min_trading_value(run_mode)
    out["min_trading_value_required"] = min_trading_value
    out["trading_value_pass"] = out["trading_value_ma_20"] >= min_trading_value
    out["trading_value_block_reason"] = np.where(out["trading_value_pass"], "none", f"trading_value_ma20_below_{run_mode}_threshold")

    out["gap_risk_blocked"] = out["expected_gap"].abs() > 0.03
    out["gap_risk_reason"] = np.where(out["gap_risk_blocked"], "expected_gap_abs_gt_3pct", "none")

    trend_points = (
        (out["close"] > numeric(out.get("ma_20"))).fillna(False).astype(float) * 30.0
        + (numeric(out.get("ma_20")) >= numeric(out.get("ma_60"))).fillna(False).astype(float) * 20.0
        + percentile_by_date(out, "mom_20").fillna(50.0) * 0.15
        + percentile_by_date(out, "ret_5d").fillna(50.0) * 0.10
    )
    volume_col = "trading_value_ratio_20d" if "trading_value_ratio_20d" in out.columns else "value_ratio_20d"
    trend_points += percentile_by_date(out, "volume_ratio_20d").fillna(50.0) * 0.10
    trend_points += percentile_by_date(out, volume_col).fillna(50.0) * 0.10
    rsi = numeric(out.get("rsi_14"))
    rsi_score = (100.0 - (rsi - 60.0).abs() * 2.0).clip(0, 100).fillna(50.0)
    trend_points += rsi_score * 0.05

    risk_deduction = (
        (rsi > 75).fillna(False).astype(float) * 10.0
        + (numeric(out.get("close_over_ma20")) > 0.15).fillna(False).astype(float) * 10.0
        + (percentile_by_date(out, "vol_20") > 80).fillna(False).astype(float) * 10.0
    )
    out["rule_score"] = (trend_points - risk_deduction).clip(0, 100)

    trend_component = (
        0.35 * (out["close"] > numeric(out.get("ma_20"))).fillna(False).astype(float)
        + 0.25 * (numeric(out.get("ma_20")) >= numeric(out.get("ma_60"))).fillna(False).astype(float)
        + 0.20 * (percentile_by_date(out, "mom_20").fillna(50.0) / 100.0)
        + 0.20 * (percentile_by_date(out, "ret_5d").fillna(50.0) / 100.0)
    )
    liquidity_component = (
        0.45 * (percentile_by_date(out, "trading_value_ma_20").fillna(50.0) / 100.0)
        + 0.35 * (percentile_by_date(out, "trading_value_ratio_20d").fillna(50.0) / 100.0)
        + 0.20 * (numeric(out.get("liquidity_score")).fillna(50.0) / 100.0)
    )
    stability_component = (
        0.50 * (percentile_by_date(out, "vol_20", ascending=False).fillna(50.0) / 100.0)
        + 0.30 * (rsi_score / 100.0)
        + 0.20 * (1.0 - numeric(out.get("close_over_ma20")).abs().clip(0, 0.25).fillna(0.1) / 0.25)
    )
    regime_component = np.where(out["market_entry_allowed"].fillna(True), 1.0, 0.35)
    out["trend_component"] = trend_component.clip(0, 1)
    out["liquidity_component"] = liquidity_component.clip(0, 1)
    out["stability_component"] = stability_component.clip(0, 1)
    out["regime_component"] = pd.Series(regime_component, index=out.index).clip(0, 1)
    out["rule_score_v2"] = (
        out["trend_component"]
        * out["liquidity_component"]
        * out["stability_component"]
        * out["regime_component"]
        * 100.0
    ).clip(0, 100)

    base_conditions = (
        (out["close"] > numeric(out.get("ma_20")))
        & (numeric(out.get("ma_20")) >= numeric(out.get("ma_60")))
        & (numeric(out.get("ret_5d")) > 0)
        & (numeric(out.get("mom_20")) > 0)
        & (rsi.between(45, 75))
        & (numeric(out.get("volume_ratio_20d")) >= 1.2)
        & (out["trading_value_ratio_20d"] >= 1.1)
        & out["trading_value_pass"]
        & (~out["gap_risk_blocked"])
        & out["market_entry_allowed"].fillna(True)
    )
    overheated = (
        (rsi > 80)
        | (numeric(out.get("close_over_ma20")) > 0.20)
        | (numeric(out.get("ret_5d")) > 0.25)
        | (percentile_by_date(out, "vol_20") > 90)
    ).fillna(False)
    entry_rule_score_min = get_float_env("RULE_ENTRY_RULE_SCORE_MIN", 70.0)
    entry_rule_score_v2_min = get_float_env("RULE_ENTRY_RULE_SCORE_V2_MIN", 65.0)
    strong_rule_score_min = get_float_env("RULE_STRONG_RULE_SCORE_MIN", 70.0)
    strong_rule_score_v2_min = get_float_env("RULE_STRONG_RULE_SCORE_V2_MIN", 50.0)
    out["entry_rule_score_min"] = entry_rule_score_min
    out["entry_rule_score_v2_min"] = entry_rule_score_v2_min
    out["strong_rule_score_min"] = strong_rule_score_min
    out["strong_rule_score_v2_min"] = strong_rule_score_v2_min
    out["entry_signal"] = base_conditions.fillna(False) & (~overheated) & (
        (out["rule_score"] >= entry_rule_score_min) | (out["rule_score_v2"] >= entry_rule_score_v2_min)
    )
    out["strong_entry_signal"] = base_conditions.fillna(False) & (~overheated) & (
        (out["rule_score"] >= strong_rule_score_min) & (out["rule_score_v2"] >= strong_rule_score_v2_min)
    )
    out["signal_strength"] = np.select(
        [out["strong_entry_signal"], out["entry_signal"]],
        ["strong_entry", "entry"],
        default="none",
    )
    out["strategy_id"] = STRATEGY_ID
    out["engine_type"] = ENGINE_TYPE
    out["run_mode"] = str(run_mode or "paper").lower()
    return out


def build_signals(
    features_csv: Path = DEFAULT_FEATURES,
    prices_csv: Path = DEFAULT_PRICES,
    universe_csv: Path = DEFAULT_UNIVERSE,
    market_status_csv: Path = DEFAULT_MARKET,
    run_mode: str = "paper",
) -> pd.DataFrame:
    features = load_features(features_csv)
    prices = load_prices(prices_csv)
    universe = load_universe(universe_csv)
    market = load_market(market_status_csv)
    signals = attach_price_features(features, prices)
    signals = attach_market(signals, market)
    if not universe.empty:
        signals = signals.merge(universe, on="code", how="left", suffixes=("", "_universe"))
    for col in ["name", "sector", "market"]:
        if col not in signals.columns:
            signals[col] = None
    signals = build_rule_scores(signals, run_mode)
    return signals.sort_values(["date", "rule_score_v2", "rule_score"], ascending=[True, False, False]).reset_index(drop=True)


def write_meta(out_csv: Path, signals: pd.DataFrame) -> None:
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(signals)),
        "min_date": signals["date"].min().date().isoformat() if not signals.empty else None,
        "max_date": signals["date"].max().date().isoformat() if not signals.empty else None,
        "entry_count": int(signals["entry_signal"].sum()) if "entry_signal" in signals else 0,
        "strong_entry_count": int(signals["strong_entry_signal"].sum()) if "strong_entry_signal" in signals else 0,
        "out_csv": str(out_csv),
    }
    meta_path = out_csv.with_name("rule_signals_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_csv = resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    signals = build_signals(
        features_csv=args.features_csv,
        prices_csv=args.prices_csv,
        universe_csv=args.universe_csv,
        market_status_csv=args.market_status_csv,
        run_mode=args.run_mode,
    )
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_meta(out_csv, signals)
    print(f"saved {out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
