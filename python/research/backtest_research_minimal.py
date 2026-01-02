# backtest_research_minimal.py
"""
Minimal research backtest script (CSV 기반)

기능:
- predictions.csv + prices.csv 로드
- horizon 기준 realized_return 계산
- 가중치 기반 final_score_custom 계산
- model_version 별 TopN 전략 성과 요약
- 예측 vs 실제 (pred_return_60d vs realized_return) 메트릭
- score 분위수 분석(quantile)

필요 CSV:
- predictions.csv: date, code, model_version, pred_return_60d, prob_top20_60d,
                   ret_score?, prob_score?, qual_score?, tech_score?, risk_penalty?, final_score?
- prices.csv:      date, code, close (또는 adj_close)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


# =========================
# 1. Config
# =========================


@dataclass
class ScoreWeights:
    w_ret: float = 1.0
    w_prob: float = 1.0
    w_qual: float = 1.0
    w_tech: float = 1.0
    w_risk: float = 1.0


@dataclass
class BacktestConfig:
    # 기간 / 모델
    start_date: date
    end_date: date
    model_versions: List[str]

    # 예측 파라미터
    horizon_days: int = 60
    top_n: int = 20
    score_column: str = "final_score_custom"  # 또는 "final_score"

    # 점수 가중치
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    # CSV 경로
    predictions_csv: str = os.path.join("data", "predictions.csv")
    prices_csv: str = os.path.join("data", "prices_daily_adjusted.csv")

    # 출력 경로
    output_dir: str = "outputs/backtest_minimal"


# =========================
# 2. Data loading
# =========================


def load_predictions_csv(config: BacktestConfig) -> pd.DataFrame:
    if not os.path.exists(config.predictions_csv):
        raise FileNotFoundError(f"predictions_csv not found: {config.predictions_csv}")

    df = pd.read_csv(config.predictions_csv)

    # model_version 컬럼이 없으면 기본값으로 채워 필터가 동작하도록 보정
    if "model_version" not in df.columns:
        default_mv = config.model_versions[0] if config.model_versions else "default"
        df["model_version"] = default_mv
        print(f"[WARN] model_version column missing; filled with '{default_mv}'")

    # 날짜/형변환
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 기간/모델 필터
    df = df[
        (df["date"] >= config.start_date)
        & (df["date"] <= config.end_date)
        & (df["model_version"].isin(config.model_versions))
    ].copy()

    # 점수 컬럼 체크 & 기본 score 보정(없으면 계산)
    if "ret_score" not in df.columns and "pred_return_60d" in df.columns:
        df["ret_score"] = df["pred_return_60d"] * 100.0

    if "prob_score" not in df.columns and "prob_top20_60d" in df.columns:
        df["prob_score"] = df["prob_top20_60d"] * 100.0

    for col in ["qual_score", "tech_score", "risk_penalty"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def load_prices_csv(config: BacktestConfig) -> pd.DataFrame:
    if not os.path.exists(config.prices_csv):
        raise FileNotFoundError(f"prices_csv not found: {config.prices_csv}")

    prices = pd.read_csv(config.prices_csv)
    if "close" not in prices.columns:
        if "adj_close" in prices.columns:
            prices = prices.rename(columns={"adj_close": "close"})
        else:
            raise ValueError("prices_csv must contain 'close' or 'adj_close' column")
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices[["date", "code", "close"]].copy()


# =========================
# 3. Outcomes (realized_return)
# =========================


def compute_realized_outcomes(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    df = preds.copy()

    # horizon 기준 미래 날짜 계산 (달력일 기준)
    df["date_h"] = df["date"].apply(
        lambda d: (datetime.combine(d, datetime.min.time()) + timedelta(days=horizon_days)).date()
    )

    # t 시점 가격
    price_t = prices.rename(columns={"close": "price_t"})
    df = df.merge(
        price_t[["date", "code", "price_t"]],
        on=["date", "code"],
        how="left",
    )

    # t+horizon 시점 가격
    price_th = prices.rename(columns={"date": "date_h", "close": "price_th"})
    df = df.merge(
        price_th[["date_h", "code", "price_th"]],
        on=["date_h", "code"],
        how="left",
    )

    # 유효한 가격만 수익률 계산
    df["realized_return"] = np.where(
        (df["price_t"] > 0) & (df["price_th"].notna()),
        (df["price_th"] - df["price_t"]) / df["price_t"],
        np.nan,
    )

    df = df.dropna(subset=["price_t", "price_th", "realized_return"])
    return df


# =========================
# 4. Scoring (가중치 적용)
# =========================


def apply_score_weights(df: pd.DataFrame, weights: ScoreWeights) -> pd.DataFrame:
    d = df.copy()

    for col in ["ret_score", "prob_score", "qual_score", "tech_score", "risk_penalty"]:
        if col not in d.columns:
            raise KeyError(f"Required score column missing: {col}")

    d["final_score_custom"] = (
        weights.w_ret * d["ret_score"]
        + weights.w_prob * d["prob_score"]
        + weights.w_qual * d["qual_score"]
        + weights.w_tech * d["tech_score"]
        - weights.w_risk * d["risk_penalty"]
    )

    return d


# =========================
# 5. Strategy (TopN / 포트폴리오)
# =========================


def select_top_n(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> pd.DataFrame:
    picks = (
        df.sort_values(["date", "model_version", score_col], ascending=[True, True, False])
        .groupby(["date", "model_version"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return picks


def evaluate_strategy(
    picks: pd.DataFrame,
) -> pd.DataFrame:
    """
    date, model_version 별 포트폴리오 수익률 계산
    """
    if "realized_return" not in picks.columns:
        raise KeyError("picks must contain realized_return")

    portfolio = (
        picks.groupby(["date", "model_version"], as_index=False)
        .agg(portfolio_return=("realized_return", "mean"))
    )
    return portfolio


# =========================
# 6. Metrics
# =========================


def prediction_regression_metrics(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str,
) -> dict:
    sub = df[[pred_col, true_col]].dropna().copy()
    if sub.empty:
        return {}

    e = sub[pred_col] - sub[true_col]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    corr = float(sub[[pred_col, true_col]].corr().iloc[0, 1])

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "n": int(len(sub)),
    }


def quantile_analysis(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str = "realized_return",
    q: int = 5,
) -> pd.Series | None:
    sub = df[[score_col, ret_col]].dropna().copy()
    if sub.empty:
        return None

    sub["quantile"] = pd.qcut(sub[score_col], q, labels=False) + 1
    return sub.groupby("quantile")[ret_col].mean()


def summarize_strategy_performance(portfolio: pd.DataFrame) -> dict:
    r = portfolio["portfolio_return"].dropna()
    if r.empty:
        return {}

    avg_ret = float(r.mean())
    hit_rate = float((r > 0).mean())
    std_ret = float(r.std(ddof=1))
    sharpe_like = float(avg_ret / std_ret) if std_ret > 0 else None

    return {
        "avg_return": avg_ret,
        "hit_rate": hit_rate,
        "sharpe_like": sharpe_like,
        "num_periods": int(len(r)),
    }


# =========================
# 7. Main backtest runner
# =========================


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    os.makedirs(config.output_dir, exist_ok=True)

    # 1) 데이터 로드
    preds = load_predictions_csv(config)
    prices = load_prices_csv(config)

    # 2) realized outcomes
    preds = compute_realized_outcomes(preds, prices, horizon_days=config.horizon_days)

    # 3) 가중치 기반 final_score_custom
    preds = apply_score_weights(preds, config.weights)

    summaries = []

    for mv in config.model_versions:
        df_mv = preds[preds["model_version"] == mv].copy()

        # 예측 메트릭
        pred_metrics = prediction_regression_metrics(
            df_mv, pred_col="pred_return_60d", true_col="realized_return"
        )

        # 분위수 분석
        final_q = quantile_analysis(df_mv, score_col=config.score_column)
        pred_q = quantile_analysis(df_mv, score_col="pred_return_60d")

        # TopN 매수
        picks = select_top_n(
            df_mv,
            score_col=config.score_column,
            top_n=config.top_n,
        )
        portfolio = evaluate_strategy(picks)
        strat_summary = summarize_strategy_performance(portfolio)

        summary = {
            "model_version": mv,
            "horizon_days": config.horizon_days,
            "top_n": config.top_n,
        }
        summary.update({f"pred_{k}": v for k, v in pred_metrics.items()})
        summary.update({f"strat_{k}": v for k, v in strat_summary.items()})
        summaries.append(summary)

        # 분위수 결과 저장(옵션)
        if final_q is not None:
            final_q.to_csv(
                os.path.join(
                    config.output_dir,
                    f"quantiles_final_{mv}.csv",
                ),
                header=["mean_realized_return"],
            )
        if pred_q is not None:
            pred_q.to_csv(
                os.path.join(
                    config.output_dir,
                    f"quantiles_pred_return_{mv}.csv",
                ),
                header=["mean_realized_return"],
            )

    result_df = pd.DataFrame(summaries)
    out_path = os.path.join(config.output_dir, "summary.csv")
    result_df.to_csv(out_path, index=False)
    print("=== Backtest summary ===")
    print(result_df)
    print(f"\nSaved summary to: {out_path}")

    return result_df


# =========================
# 8. Example usage
# =========================


if __name__ == "__main__":
    # TODO: 필요에 맞게 기간 / 모델 버전 조정
    cfg = BacktestConfig(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        model_versions=["v1"],   # 필요시 ["v1", "v2", ...]
        horizon_days=60,
        top_n=20,
        score_column="final_score_custom",
        weights=ScoreWeights(
            w_ret=1.0,
            w_prob=1.0,
            w_qual=1.0,
            w_tech=1.0,
            w_risk=1.0,
        ),
        predictions_csv=os.path.join("data", "predictions.csv"),
        prices_csv=os.path.join("data", "prices_daily_adjusted.csv"),
        output_dir="outputs/backtest_minimal",
    )

    run_backtest(cfg)
