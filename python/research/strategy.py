# python/research/strategy.py
import pandas as pd
try:
    from .config import BacktestConfig
except ImportError:
    from config import BacktestConfig


def select_top_n(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    date, model_version 기준으로 config.score_column 순서로 top_n 선택.
    """
    score_col = config.score_column
    top_n = config.top_n

    picks = (
        df
        .sort_values(["date", "model_version", score_col],
                     ascending=[True, True, False])
        .groupby(["date", "model_version"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return picks


def evaluate_strategy(
    picks: pd.DataFrame,
    config: BacktestConfig,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    picks: select_top_n 결과 + realized_return 포함
    return: date, model_version 별 포트폴리오 수익률/턴오버 집계
    """
    if "realized_return" not in picks.columns:
        raise KeyError("picks must contain realized_return")

    records = []
    prev_weights: dict[str, dict[str, float]] = {}
    for (dt, mv), grp in picks.sort_values(["date", "model_version"]).groupby(["date", "model_version"]):
        if grp.empty:
            continue

        # 기본: 1/n equal weight
        n = len(grp)
        w = 1.0 / n if n > 0 else 0.0
        curr_w = {code: w for code in grp["code"]}

        # 보유/교체 turnover (0.5 * L1)
        prev_w = prev_weights.get(mv, {})
        turnover = 0.5 * sum(abs(curr_w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in set(curr_w) | set(prev_w))

        port_ret = grp["realized_return"].mean()
        records.append(
            {
                "date": dt,
                "model_version": mv,
                "portfolio_return": port_ret,
                "turnover": turnover,
                "num_holdings": n,
            }
        )
        prev_weights[mv] = curr_w

    port = pd.DataFrame(records).sort_values(["date", "model_version"]).reset_index(drop=True)

    # 누적 수익률 (단순 곱)
    port["cum_return"] = (
        port.groupby("model_version")["portfolio_return"]
        .apply(lambda s: (1 + s).cumprod() - 1)
        .reset_index(level=0, drop=True)
    )

    # benchmark 대비 초과수익 (date, return 컬럼 기대)
    if benchmark is not None and not benchmark.empty:
        bmk = benchmark.copy()
        bmk["date"] = pd.to_datetime(bmk["date"]).dt.date
        if "return" not in bmk.columns and "close" in bmk.columns:
            bmk = bmk.sort_values("date")
            bmk["return"] = bmk["close"].pct_change()
        bmk = bmk.dropna(subset=["return"])
        bmk = bmk[["date", "return"]].rename(columns={"return": "benchmark_return"})
        port = port.merge(bmk, on="date", how="left")
        port["excess_return"] = port["portfolio_return"] - port["benchmark_return"]

    return port
