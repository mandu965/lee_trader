# python/research/data_loader.py
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
try:
    from .config import BacktestConfig
except ImportError:
    from config import BacktestConfig

# Ensure repo paths are on sys.path when running scripts directly
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (REPO_ROOT, PYTHON_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.append(_ps)

try:
    from ..db import get_engine  # when imported as package
except Exception:
    try:
        from db import get_engine    # fallback when running standalone from repo root
    except Exception:
        try:
            from python.db import get_engine  # fallback when cwd is repo root and package path missing
        except Exception:
            get_engine = None


def _get_engine_or_raise():
    if get_engine is None:
        raise RuntimeError(
            "get_engine import failed; ensure PYTHONPATH includes repo root or ./python and DATABASE_URL is set"
        )
    return get_engine()


def load_predictions(config: BacktestConfig) -> pd.DataFrame:
    """
    Expected columns (CSV mode):
      date, code, model_version,
      pred_return_60d, pred_mdd_60d, prob_top20_60d,
      ret_score, prob_score, qual_score, tech_score,
      risk_penalty, final_score (optional)
    """
    if config.prediction_source == "research":
        if config.run_id is None:
            raise ValueError("prediction_source='research' requires config.run_id")
        eng = _get_engine_or_raise()
        sql = text(
            """
            SELECT
              as_of_date AS date,
              code,
              model_version,
              horizon_days,
              pred_return_30d,
              pred_return_60d,
              pred_return_90d,
              pred_mdd_30d,
              pred_mdd_60d,
              pred_mdd_90d,
              prob_top20_30d,
              prob_top20_60d,
              prob_top20_90d,
              ret_score,
              prob_score,
              qual_score,
              tech_score,
              risk_penalty,
              final_score,
              final_score_custom
            FROM research.prediction_history
            WHERE run_id = :run_id
              AND as_of_date BETWEEN :start AND :end
              AND horizon_days = :h
            """
        )
        df = pd.read_sql(
            sql,
            eng,
            params={
                "run_id": config.run_id,
                "start": config.start_date,
                "end": config.end_date,
                "h": config.horizon_days,
            },
        )
    elif config.prediction_source == "db":
        eng = _get_engine_or_raise()
        sql = text(
            """
            SELECT
              date,
              code,
              pred_return_60d,
              pred_return_90d,
              pred_mdd_60d,
              pred_mdd_90d,
              prob_top20_60d,
              prob_top20_90d,
              ret_score,
              prob_score,
              qual_score,
              tech_score,
              risk_penalty,
              final_score
            FROM daily_ranking
            WHERE date BETWEEN :start AND :end
            """
        )
        df = pd.read_sql(sql, eng, params={"start": config.start_date, "end": config.end_date})
        # daily_ranking에 model_version이 없다면 기본값 사용
        default_mv = config.model_versions[0] if config.model_versions else "default"
        df["model_version"] = default_mv
    else:
        path = config.predictions_csv
        if not os.path.exists(path):
            raise FileNotFoundError(f"predictions_csv not found: {path}")

        df = pd.read_csv(path)

        if "model_version" not in df.columns:
            default_mv = config.model_versions[0] if config.model_versions else "default"
            df["model_version"] = default_mv

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["code"] = df["code"].astype(str)

    mv_filter = config.model_versions
    if mv_filter is None or mv_filter == ["any"]:
        mask_mv = True
    else:
        mask_mv = df["model_version"].isin(mv_filter)

    df = df[
        (df["date"] >= config.start_date)
        & (df["date"] <= config.end_date)
        & mask_mv
    ].copy()

    if "ret_score" not in df.columns and "pred_return_60d" in df.columns:
        df["ret_score"] = df["pred_return_60d"] * 100.0
    if "prob_score" not in df.columns:
        if "prob_top20_30d" in df.columns and df["prob_top20_30d"].notna().any():
            df["prob_score"] = df["prob_top20_30d"] * 100.0
        elif "prob_top20_60d" in df.columns:
            df["prob_score"] = df["prob_top20_60d"] * 100.0
        elif "prob_top20_90d" in df.columns:
            df["prob_score"] = df["prob_top20_90d"] * 100.0
    if "final_score_custom" not in df.columns:
        df["final_score_custom"] = df.get("final_score")
    fallback = df.get("final_score")
    if fallback is None:
        fallback = 0.0
    # If final_score is missing, fall back to ret_score (or pred_return_60d*100) so ranking works.
    ret_fallback = df.get("ret_score")
    if ret_fallback is None and "pred_return_60d" in df.columns:
        ret_fallback = df["pred_return_60d"] * 100.0
    if ret_fallback is None:
        ret_fallback = 0.0

    df["final_score_custom"] = (
        df["final_score_custom"]
        .fillna(fallback)
        .fillna(ret_fallback)
        .infer_objects(copy=False)
    )
    for col in ["qual_score", "tech_score", "risk_penalty"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_prices(config: BacktestConfig) -> pd.DataFrame:
    """
    Expected columns:
      date, code, close (adj_close accepted and renamed)
    """
    if config.price_source == "db":
        eng = _get_engine_or_raise()
        horizon_buf = getattr(config, "horizon_days", 0) or 0
        end_with_horizon = pd.to_datetime(config.end_date) + pd.Timedelta(days=horizon_buf)
        sql = text(
            """
            SELECT
              CAST(date AS date) AS date,
              code,
              COALESCE(adj_close, close) AS close
            FROM fact_price_daily
            WHERE CAST(date AS date) BETWEEN CAST(:start AS date) AND CAST(:end AS date)
            """
        )
        prices = pd.read_sql(
            sql,
            eng,
            params={
                "start": config.start_date,
                "end": end_with_horizon.date(),
            },
        )
    else:
        path = config.prices_csv
        if not os.path.exists(path):
            raise FileNotFoundError(f"prices_csv not found: {path}")

        prices = pd.read_csv(path)
        if "close" not in prices.columns:
            if "adj_close" in prices.columns:
                prices = prices.rename(columns={"adj_close": "close"})
            else:
                raise ValueError("prices_csv must contain 'close' or 'adj_close'")

    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["code"] = prices["code"].astype(str)
    horizon_buf = getattr(config, "horizon_days", 0) or 0
    end_with_horizon = pd.to_datetime(config.end_date) + pd.Timedelta(days=horizon_buf)
    prices = prices[
        (prices["date"] >= config.start_date)
        & (prices["date"] <= end_with_horizon.date())
    ].copy()

    return prices[["date", "code", "close"]].copy()


def load_benchmark(config: BacktestConfig) -> pd.DataFrame:
    """
    columns:
      date, close, return (return optional; derived if missing)
    """
    if not config.benchmark_code and not config.benchmark_csv:
        return pd.DataFrame()

    if config.benchmark_csv and os.path.exists(config.benchmark_csv):
        bench = pd.read_csv(config.benchmark_csv)
        bench["date"] = pd.to_datetime(bench["date"]).dt.date
        if "close" not in bench.columns:
            raise ValueError("benchmark_csv must contain 'close'")
        bench = bench.sort_values("date")
        if "return" not in bench.columns:
            bench["return"] = bench["close"].pct_change()
        bench = bench.dropna(subset=["return"])
        return bench[["date", "close", "return"]].copy()

    # DB에서 벤치마크 코드가 fact_price_daily에 있는 경우 로드 시도
    if config.benchmark_code:
        try:
            eng = _get_engine_or_raise()
            sql = text(
                """
                SELECT CAST(date AS date) AS date, COALESCE(adj_close, close) AS close
                FROM fact_price_daily
                WHERE code = :code AND CAST(date AS date) BETWEEN CAST(:start AS date) AND CAST(:end AS date)
                ORDER BY date
                """
            )
            bench = pd.read_sql(
                sql,
                eng,
                params={"code": config.benchmark_code, "start": config.start_date, "end": config.end_date},
            )
            if not bench.empty:
                bench["date"] = pd.to_datetime(bench["date"]).dt.date
                bench["return"] = bench["close"].pct_change()
                bench = bench.dropna(subset=["return"])
                return bench[["date", "close", "return"]].copy()
        except Exception:
            # optional fallback; silence to allow CSV-only workflows
            pass

    # TODO: hook into benchmark_etf_comparison.py or DB when available
    return pd.DataFrame()
