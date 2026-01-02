# python/research/backtest_research.py
import os
import pandas as pd
try:
    from .config import BacktestConfig
    from .data_loader import load_predictions, load_prices, load_benchmark
    from .outcomes import compute_realized_outcomes
    from .scoring import apply_score_weights
    from .strategy import select_top_n, evaluate_strategy
    from .metrics import (
        prediction_regression_metrics,
        quantile_analysis,
        summarize_strategy_performance,
    )
except ImportError:
    # fallback when running standalone with sys.path pointing to python/
    from config import BacktestConfig
    from data_loader import load_predictions, load_prices, load_benchmark
    from outcomes import compute_realized_outcomes
    from scoring import apply_score_weights
    from strategy import select_top_n, evaluate_strategy
    from metrics import (
        prediction_regression_metrics,
        quantile_analysis,
        summarize_strategy_performance,
    )


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    # 1) 데이터 로드
    preds = load_predictions(config)
    prices = load_prices(config)
    bench = load_benchmark(config)

    # 2) prediction-level: realized outcomes 계산
    preds = compute_realized_outcomes(preds, prices, config)

    # 3) 가중치 반영해 final_score_custom 생성
    preds = apply_score_weights(preds, config)

    os.makedirs(config.output_dir, exist_ok=True)
    all_summaries = []

    for mv in config.model_versions:
        df_mv = preds[preds["model_version"] == mv].copy()

        # 4) prediction-level metrics
        pred_metrics = prediction_regression_metrics(
            df_mv, pred_col="pred_return_60d", true_col="realized_return"
        )

        # 5) score 분위수별 성과
        final_q = quantile_analysis(df_mv, score_col="final_score_custom")
        pred_q = quantile_analysis(df_mv, score_col="pred_return_60d")

        # 6) 전략: TopN 선택 + 포트폴리오 성과
        picks = select_top_n(df_mv, config)
        portfolio = evaluate_strategy(picks, config, bench)
        strat_summary = summarize_strategy_performance(portfolio)

        summary = {
            "model_version": mv,
            "horizon_days": config.horizon_days,
            **{f"pred_{k}": v for k, v in pred_metrics.items()},
            **{f"strat_{k}": v for k, v in strat_summary.items()},
        }
        all_summaries.append(summary)

        # 분위수 결과 저장(선택)
        if final_q is not None:
            final_q.to_csv(
                os.path.join(config.output_dir, f"quantiles_final_{mv}.csv"),
                header=["mean_realized_return"],
            )
        if pred_q is not None:
            pred_q.to_csv(
                os.path.join(config.output_dir, f"quantiles_pred_return_{mv}.csv"),
                header=["mean_realized_return"],
            )

        # 포트폴리오 궤적 저장(선택)
        portfolio.to_csv(
            os.path.join(config.output_dir, f"portfolio_{mv}.csv"),
            index=False,
        )

    result_df = pd.DataFrame(all_summaries)
    out_path = os.path.join(config.output_dir, "summary.csv")
    result_df.to_csv(out_path, index=False)
    print("=== Backtest summary ===")
    print(result_df)
    print(f"\nSaved summary to: {out_path}")

    return result_df
