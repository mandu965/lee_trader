# python/run_pipeline.py
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sqlalchemy import text
try:
    from db import get_engine, log_pipeline_history
except Exception:
    get_engine = None
    log_pipeline_history = None

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lee_trader.db"

STEPS: List[Tuple[str, str]] = [
    ("fetch_market_data", "python/fetch_market_data.py"),   # 시장 데이터 수집
    ("fetch_top_universe", "python/fetch_top_universe.py"), # 관심 종목/유니버스 업데이트
    ("merge_universe", "python/merge_universe.py"),
    ("download_prices_kis", "python/download_prices_kis.py"),
    ("clean_prices", "python/clean_prices.py"),
    ("create_adjusted_prices", "python/create_adjusted_prices.py"),
    ("fetch_fundamentals_dart", "python/fetch_fundamentals_dart.py"),
    ("quality_builder", "python/quality_builder.py"),
    ("feature_builder", "python/feature_builder.py"),
    ("scoring", "python/scoring.py"),
    ("label_builder", "python/label_builder.py"),
    ("model_train", "python/model_train.py"),
    ("model_predict", "python/model_predict.py"),
    ("ranking_builder", "python/ranking_builder.py"),
    # ("migrate_to_sqlite", "migrate_to_sqlite.py"),
]

# 필수 종목 존재 여부 체크(샘플)
CORE_CODES = ["005930", "000660", "035420"]


def _env_model_version() -> str:
    return os.environ.get("MODEL_VERSION", "v1")


def _env_horizon_days() -> int:
    return int(os.environ.get("HORIZON_DAYS", "60"))


def _env_top_n() -> int:
    return int(os.environ.get("TOP_N", "20"))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def create_model_run_id() -> str:
    """
    Insert a row into research.dim_model_run and return run_id.
    Falls back to timestamp-based ID if DB/engine is unavailable.
    """
    fallback = f"{datetime.utcnow().isoformat()}-{os.getpid()}"

    if not get_engine:
        return fallback

    run_type = os.environ.get("RUN_TYPE", "daily_pipeline")
    model_version = _env_model_version()
    horizon_days = _env_horizon_days()
    top_n = _env_top_n()
    config_json = {}
    if os.environ.get("SCORE_WEIGHTS_JSON"):
        try:
            config_json["score_weights"] = json.loads(os.environ["SCORE_WEIGHTS_JSON"])
        except Exception:
            logging.warning("Invalid SCORE_WEIGHTS_JSON; storing empty config_json")
            config_json["score_weights"] = {}

    try:
        eng = get_engine()
        with eng.begin() as conn:
            res = conn.execute(
                text(
                    """
                    INSERT INTO research.dim_model_run
                    (run_type, model_version, horizon_days, top_n, config_json)
                    VALUES (:run_type, :model_version, :horizon_days, :top_n, :config_json)
                    RETURNING run_id
                    """
                ),
                {
                    "run_type": run_type,
                    "model_version": model_version,
                    "horizon_days": horizon_days,
                    "top_n": top_n,
                    "config_json": config_json if config_json else None,
                },
            )
            run_id_db = res.scalar_one()
            return str(run_id_db)
    except Exception as e:
        logging.warning("create_model_run_id failed, fallback run_id used: %s", e)
        return fallback


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def apply_schema_if_available() -> None:
    """
    Apply schema.sql once on startup (idempotent: CREATE IF NOT EXISTS).
    Useful in environments without shell access (e.g., Render).
    """
    if not get_engine:
        logging.info("No DB engine available -> skip schema apply")
        return

    schema_path = Path("schema.sql")
    if not schema_path.exists():
        logging.info("schema.sql not found -> skip schema apply")
        return

    try:
        sql_text = schema_path.read_text(encoding="utf-8")
        eng = get_engine()
        with eng.begin() as conn:
            conn.exec_driver_sql(sql_text)
        logging.info("schema.sql applied successfully")
    except Exception:
        logging.exception("Failed to apply schema.sql (skipped)")


def run_step(name: str, script: str) -> float:
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for step {name}: {script_path}")

    start_ts = time.perf_counter()
    logging.info("==> Running step: %s (%s)", name, script_path)
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    elapsed = time.perf_counter() - start_ts
    if result.returncode != 0:
        raise RuntimeError(f"Step {name} failed with return code {result.returncode}")
    logging.info("<== Step completed: %s (elapsed=%.2fs)", name, elapsed)
    return elapsed


def _load_codes_from_csv(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype={"code": str})
    if "code" not in df.columns:
        return set()
    return set(df["code"].astype(str).str.zfill(6).unique())


def _load_codes_from_db(table: str) -> set:
    # Prefer Postgres via SQLAlchemy engine if available
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                stmt = text(f"SELECT DISTINCT code FROM {table}")
                rows = conn.execute(stmt).fetchall()

            codes = {str(r[0]).zfill(6) for r in rows if r and r[0] is not None}
            logging.info("Loaded %d codes from Postgres table '%s'", len(codes), table)
            return codes
        except Exception as e:
            logging.warning("Postgres load failed for table '%s': %s", table, e)

    if not DB_PATH.exists():
        logging.info("No sqlite DB at %s -> skip DB check for table '%s'", DB_PATH, table)
        return set()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            rows = cur.execute(f"SELECT DISTINCT code FROM {table}").fetchall()
            codes = {str(r[0]).zfill(6) for r in rows if r and r[0] is not None}
            logging.info("Loaded %d codes from sqlite table '%s'", len(codes), table)
            return codes
    except Exception as e:
        logging.warning("sqlite load failed for table '%s': %s", table, e)
        return set()


def verify_core_codes(codes: list[str]) -> None:
    """Ensure critical codes exist in both CSV outputs and DB tables."""
    codes = [str(c).zfill(6) for c in codes]
    files = {
        "features.csv": DATA_DIR / "features.csv",
        "predictions.csv": DATA_DIR / "predictions.csv",
        "ranking_final.csv": DATA_DIR / "ranking_final.csv",
    }
    tables = ["features", "predictions", "daily_ranking"]

    missing = []

    for name, path in files.items():
        present = _load_codes_from_csv(path)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{name}: {','.join(miss)}")

    for tbl in tables:
        present = _load_codes_from_db(tbl)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{tbl} (DB): {','.join(miss)}")

    if missing:
        raise RuntimeError(f"Core code check failed -> { '; '.join(missing) }")

    logging.info("Core code check passed: %s", ", ".join(codes))


def _load_table(engine, table: str) -> pd.DataFrame:
    """Load full table from Postgres via SQLAlchemy engine."""
    return pd.read_sql(f"SELECT * FROM {table}", con=engine)


def _latest_as_of_date(df: pd.DataFrame, date_col: str = "date") -> date:
    if date_col not in df.columns or df.empty:
        return date.today()
    try:
        dates = pd.to_datetime(df[date_col]).dt.date.dropna().unique()
        if len(dates) == 0:
            return date.today()
        return max(dates)
    except Exception:
        return date.today()


def append_prediction_history(run_id: str) -> None:
    """Append today's predictions into research.prediction_history."""
    if not get_engine:
        logging.info("No DB engine available -> skip prediction_history append")
        return
    if not run_id.isdigit():
        logging.warning("run_id is not numeric (%s); skip prediction_history append", run_id)
        return

    horizon_days = _env_horizon_days()
    model_version = _env_model_version()

    try:
        eng = get_engine()
        preds = _load_table(eng, "predictions")
        if preds.empty:
            logging.warning("predictions table empty -> skip prediction_history append")
            return

        # Optional score join from daily_ranking if present
        score_cols = ["ret_score", "prob_score", "qual_score", "tech_score", "risk_penalty", "final_score", "final_score_custom"]
        try:
            ranking = _load_table(eng, "daily_ranking")
            score_subset = ranking[["date", "code"] + [c for c in score_cols if c in ranking.columns]]
            merged = preds.merge(score_subset, on=["date", "code"], how="left")
        except Exception:
            logging.warning("daily_ranking load failed; inserting predictions without score columns", exc_info=True)
            merged = preds.copy()

        as_of = _latest_as_of_date(merged, "date")
        merged["as_of_date"] = as_of
        merged["run_id"] = int(run_id)
        merged["model_version"] = model_version
        merged["horizon_days"] = horizon_days
        if "final_score_custom" not in merged.columns:
            merged["final_score_custom"] = None

        cols = [
            "run_id",
            "as_of_date",
            "code",
            "model_version",
            "horizon_days",
            "pred_return_60d",
            "pred_return_90d",
            "pred_mdd_60d",
            "pred_mdd_90d",
            "prob_top20_60d",
            "prob_top20_90d",
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "final_score",
            "final_score_custom",
        ]
        missing = [c for c in cols if c not in merged.columns]
        for c in missing:
            merged[c] = None

        out = merged[cols].copy()

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.prediction_history
                    WHERE run_id = :run_id AND as_of_date = :as_of_date
                    """
                ),
                {"run_id": int(run_id), "as_of_date": as_of},
            )
        out.to_sql("prediction_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Appended prediction_history: rows=%d, as_of=%s", len(out), as_of)
    except Exception:
        logging.exception("append_prediction_history failed")


def append_ranking_history(run_id: str) -> None:
    """Append today's ranking into research.ranking_history."""
    if not get_engine:
        logging.info("No DB engine available -> skip ranking_history append")
        return
    if not run_id.isdigit():
        logging.warning("run_id is not numeric (%s); skip ranking_history append", run_id)
        return

    horizon_days = _env_horizon_days()
    model_version = _env_model_version()
    top_n = _env_top_n()

    try:
        eng = get_engine()
        ranking = _load_table(eng, "daily_ranking")
        if ranking.empty:
            logging.warning("daily_ranking table empty -> skip ranking_history append")
            return

        as_of = _latest_as_of_date(ranking, "date")
        ranking["as_of_date"] = as_of
        ranking = ranking.sort_values(["date", "final_score"], ascending=[False, False]).copy()
        ranking["rank"] = ranking.groupby("date")["final_score"].rank(method="first", ascending=False).astype(int)
        ranking["in_top_n"] = ranking["rank"] <= top_n
        ranking["top_n"] = top_n
        ranking["run_id"] = int(run_id)
        if "model_version" in ranking.columns:
            ranking["model_version"] = ranking["model_version"].fillna(model_version)
        else:
            ranking["model_version"] = model_version
        ranking["horizon_days"] = horizon_days

        cols = [
            "run_id",
            "as_of_date",
            "code",
            "model_version",
            "horizon_days",
            "rank",
            "final_score",
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "in_top_n",
            "top_n",
        ]
        missing = [c for c in cols if c not in ranking.columns]
        for c in missing:
            ranking[c] = None

        out = ranking[cols].copy()

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.ranking_history
                    WHERE run_id = :run_id AND as_of_date = :as_of_date
                    """
                ),
                {"run_id": int(run_id), "as_of_date": as_of},
            )
        out.to_sql("ranking_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Appended ranking_history: rows=%d, as_of=%s", len(out), as_of)
    except Exception:
        logging.exception("append_ranking_history failed")


def _table_stat_query(conn, table: str, date_cols: list[str]) -> None:
    """Log max(date-col) if present and count(*) for a table."""
    # find first existing date-like column
    col = None
    try:
        schema, _, name = table.partition(".")
        if not name:  # no schema given
            schema = "public"
            name = table
        cols = conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=:schema AND table_name=:name
                """
            ),
            {"schema": schema, "name": name},
        ).fetchall()
        colnames = {c[0] for c in cols}
        if not colnames:
            logging.warning("Table stats skipped (not found): %s", table)
            return
        for cand in date_cols:
            if cand in colnames:
                col = cand
                break
    except Exception:
        col = None

    try:
        if col:
            res = conn.execute(text(f"SELECT MAX({col}) AS max_col, COUNT(*) AS cnt FROM {table}"))
            row = res.fetchone()
            logging.info("Table %s -> max_%s=%s, count=%s", table, col, row.max_col, row.cnt)
        else:
            res = conn.execute(text(f"SELECT COUNT(*) AS cnt FROM {table}"))
            row = res.fetchone()
            logging.info("Table %s -> count=%s (no date column found)", table, row.cnt)
    except Exception as e:
        logging.warning("Table stats failed for %s: %s", table, e)
        try:
            conn.rollback()
        except Exception:
            pass


def log_table_stats() -> None:
    """Log MAX(date-like) and row count for core/research tables."""
    if not get_engine:
        logging.info("No DB engine available -> skip table stats logging")
        return

    # Preferred date/timestamp columns in order
    date_cols = ["date", "as_of_date", "trade_date", "created_at"]

    tables = [
        # Core/market
        "market_status",
        "stocks",
        "prices_raw",
        "prices_clean",
        "prices_adjusted",
        "fact_price_daily",
        "quality",
        "fundamentals",
        # Features/labels/preds/ranking
        "features",
        "daily_scores",
        "labels",
        "predictions",
        "daily_ranking",
        # Trades
        "trades",
        "backtest_trades",
        # Research layer
        "research.dim_model_run",
        "research.prediction_history",
        "research.ranking_history",
        "research.backtest_outcome",
    ]

    try:
        eng = get_engine()
        with eng.connect() as conn:
            for tbl in tables:
                _table_stat_query(conn, tbl, date_cols)
            try:
                res = conn.execute(
                    text(
                        """
                        SELECT run_id, run_type, model_version, horizon_days, top_n, created_at
                        FROM research.dim_model_run
                        ORDER BY run_id DESC
                        LIMIT 5
                        """
                    )
                )
                rows = res.fetchall()
                logging.info("Recent dim_model_run (latest 5): %s", rows)
            except Exception as e:
                logging.warning("dim_model_run fetch failed: %s", e)
    except Exception:
        logging.exception("log_table_stats failed")


def _label_cols_for_horizon(h: int) -> tuple[str | None, str | None]:
    """Return (realized_return_col, realized_mdd_col) names for the given horizon days."""
    ret_col = f"realized_return_{h}d"
    mdd_col_candidates = [f"realized_mdd_{h}d", f"target_mdd_{h}d"]
    return ret_col, mdd_col_candidates


def append_backtest_outcome(run_id: str) -> None:
    """Append realized outcomes (from labels) into research.backtest_outcome."""
    if not get_engine:
        logging.info("No DB engine available -> skip backtest_outcome append")
        return
    if not run_id.isdigit():
        logging.warning("run_id is not numeric (%s); skip backtest_outcome append", run_id)
        return

    horizon_days = _env_horizon_days()
    ret_col, mdd_candidates = _label_cols_for_horizon(horizon_days)

    try:
        eng = get_engine()
        query = text(
            "SELECT run_id, as_of_date, code, horizon_days "
            "FROM research.prediction_history WHERE run_id = :run_id"
        )
        preds_hist = pd.read_sql(query, con=eng, params={"run_id": int(run_id)}, parse_dates=["as_of_date"])
        if preds_hist.empty:
            logging.warning("prediction_history empty for run_id=%s -> skip backtest_outcome append", run_id)
            return

        labels = _load_table(eng, "labels")
        if labels.empty:
            logging.warning("labels table empty -> skip backtest_outcome append")
            return
        labels["date"] = pd.to_datetime(labels["date"])

        latest_label_date = labels["date"].max()
        as_of_limit = latest_label_date
        preds_hist = preds_hist[preds_hist["as_of_date"] <= as_of_limit]
        if preds_hist.empty:
            logging.info(
                "No prediction_history rows within label coverage (latest_label_date=%s) -> skip backtest_outcome append",
                latest_label_date.date(),
            )
            return

        if ret_col not in labels.columns:
            logging.warning("%s not in labels -> skip backtest_outcome append", ret_col)
            return
        mdd_col = next((c for c in mdd_candidates if c in labels.columns), None)

        merged = preds_hist.merge(
            labels[["date", "code", ret_col] + ([mdd_col] if mdd_col else [])],
            left_on=["as_of_date", "code"],
            right_on=["date", "code"],
            how="left",
        )
        merged["realized_return"] = merged[ret_col]
        merged["realized_mdd"] = merged[mdd_col] if mdd_col else None
        merged["label_source"] = "from_labels"

        out_cols = ["run_id", "as_of_date", "code", "horizon_days", "realized_return", "realized_mdd", "label_source"]
        out = merged[out_cols].copy()
        out = out.dropna(subset=["realized_return"])
        if out.empty:
            logging.warning("No realized_return rows after merge -> skip backtest_outcome append")
            return

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.backtest_outcome
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": int(run_id)},
            )
        out.to_sql("backtest_outcome", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info(
            "Appended backtest_outcome: rows=%d, horizon=%d, ret_col=%s, mdd_col=%s",
            len(out),
            horizon_days,
            ret_col,
            mdd_col,
        )
    except Exception:
        logging.exception("append_backtest_outcome failed")


def main() -> None:
    setup_logging()
    ensure_data_dir()
    apply_schema_if_available()

    run_id = create_model_run_id()
    try:
        pipeline_start = time.perf_counter()
        for name, script in STEPS:
            if log_pipeline_history:
                log_pipeline_history(run_id, name, "start", None, None)
            elapsed = run_step(name, script)
            if log_pipeline_history:
                log_pipeline_history(run_id, name, "success", elapsed, None)

        verify_core_codes(CORE_CODES)

        # Research append (snapshot)
        append_prediction_history(run_id)
        append_ranking_history(run_id)
        append_backtest_outcome(run_id)
        log_table_stats()
        # Daily snapshots of ranking/predictions/features + metrics
        try:
            snapshot_script = Path("scripts") / "snapshot_scores.py"
            if snapshot_script.exists():
                run_step("snapshot_scores", str(snapshot_script))
            else:
                logging.warning("snapshot_scores.py not found -> skip snapshot step")
        except Exception:
            logging.exception("Snapshot step failed (pipeline continues)")

        total_elapsed = time.perf_counter() - pipeline_start
        if log_pipeline_history:
            log_pipeline_history(run_id, "pipeline", "success", total_elapsed, "all steps completed")
        logging.info("Pipeline finished successfully (total=%.2fs)", total_elapsed)
    except Exception as e:
        logging.exception("Pipeline error: %s", e)
        if log_pipeline_history:
            try:
                log_pipeline_history(run_id, "pipeline", "failed", None, str(e))
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
