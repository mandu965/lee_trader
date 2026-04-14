"""
db.py

Provides SQLAlchemy engine/session helpers and a psycopg2 raw connector
using DATABASE_URL environment variable.
"""
import os
import json
from pathlib import Path
from functools import lru_cache
from io import StringIO
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker


WALKFORWARD_REQUIRED_CONFIG_FIELDS = [
    "train_start",
    "train_end",
    "predict_start",
    "predict_end",
    "rebalance_freq",
    "universe_version",
    "score_weights",
]


def use_sqlite_mirror() -> bool:
    return os.environ.get("USE_SQLITE_MIRROR", "0").strip().lower() in ("1", "true", "yes", "y")


def use_sqlite_fallback_writes() -> bool:
    return os.environ.get("USE_SQLITE_FALLBACK_WRITES", "0").strip().lower() in ("1", "true", "yes", "y")


@lru_cache(maxsize=1)
def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        # fallback: try loading from .env in repo root
        env_path = Path(".env")
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
                url = os.environ.get("DATABASE_URL")
            except Exception:
                # silent fallback to raise below
                url = None
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return url


@lru_cache(maxsize=1)
def get_engine():
    url = get_database_url()
    # Pool tuned for small app: pool_size=5, max_overflow=2, pool_recycle=1800s, connect timeout 5s
    engine = create_engine(
        url,
        pool_size=int(os.environ.get("PG_POOL_SIZE", "5")),
        max_overflow=int(os.environ.get("PG_MAX_OVERFLOW", "2")),
        pool_timeout=int(os.environ.get("PG_POOL_TIMEOUT", "5")),
        pool_recycle=int(os.environ.get("PG_POOL_RECYCLE", "1800")),
        connect_args={"connect_timeout": int(os.environ.get("PG_CONNECT_TIMEOUT", "5"))},
        future=True,
    )
    return engine


def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def raw_psycopg2_conn():
    import psycopg2
    url = get_database_url()
    return psycopg2.connect(url, connect_timeout=int(os.environ.get("PG_CONNECT_TIMEOUT", "5")))


def copy_df(table: str, df, columns=None, truncate: bool = False) -> None:
    """
    Fast bulk load via psycopg2 copy_expert.
    """
    import psycopg2

    conn = raw_psycopg2_conn()
    try:
        with conn, conn.cursor() as cur:
            if truncate:
                cur.execute(f"TRUNCATE TABLE {table}")
            buf = StringIO()
            if columns:
                df.to_csv(buf, index=False, header=False, columns=columns)
            else:
                df.to_csv(buf, index=False, header=False)
            buf.seek(0)
            cols = f"({', '.join(columns)})" if columns else ""
            cur.copy_expert(f"COPY {table} {cols} FROM STDIN WITH (FORMAT CSV)", buf)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_unique_keys(df, keys: list[str], table: str) -> None:
    """Raise if the dataframe contains duplicate PK-like keys before save."""
    if not keys:
        return
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing key columns for duplicate check: {missing}")
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return
    sample = df.loc[dup_mask, keys].drop_duplicates().head(5).to_dict(orient="records")
    raise ValueError(f"{table}: duplicate key rows detected for {keys}; sample={sample}")


def replace_table_rows_pg(table: str, df, columns: list[str] | None = None) -> None:
    """
    Replace table rows while preserving schema, PK, and indexes.
    Postgres implementation uses TRUNCATE + COPY.
    """
    use_columns = list(columns) if columns else list(df.columns)
    out = df.copy()
    for col in use_columns:
        if col not in out.columns:
            out[col] = None
    out = out.loc[:, use_columns]

    if out.empty:
        conn = raw_psycopg2_conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return

    copy_df(table, out, columns=use_columns, truncate=True)


def replace_table_rows_sqlite(conn, table: str, df) -> None:
    """
    Replace table rows while preserving schema, PK, and indexes.
    SQLite implementation uses DELETE + append.
    """
    conn.execute(f"DELETE FROM {table}")
    if df.empty:
        return
    df.to_sql(table, conn, if_exists="append", index=False)


def log_pipeline_history(run_id: str, step: str, status: str, duration_s=None, message: str | None = None) -> None:
    """
    Best-effort logging of pipeline checkpoints to pipeline_history.
    Does not raise on failure.
    """
    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pipeline_history (run_id, step, status, duration_s, message, created_at)
                    VALUES (:run_id, :step, :status, :duration_s, :message, :created_at)
                    """
                ),
                {
                    "run_id": run_id,
                    "step": step,
                    "status": status,
                    "duration_s": duration_s,
                    "message": message,
                    "created_at": datetime.utcnow(),
                },
            )
    except Exception:
        # swallow any logging errors to avoid breaking pipeline
        pass


def create_research_model_run(
    *,
    run_type: str,
    model_version: str,
    horizon_days: int,
    top_n: int,
    train_start_date=None,
    train_end_date=None,
    config_json=None,
    comment: str | None = None,
) -> int:
    if run_type == "walkforward_backtest":
        if not isinstance(config_json, dict):
            raise ValueError("walkforward_backtest requires config_json as a dict")
        missing = [key for key in WALKFORWARD_REQUIRED_CONFIG_FIELDS if key not in config_json]
        if missing:
            raise ValueError(
                "walkforward_backtest config_json missing required fields: "
                + ", ".join(missing)
            )
    eng = get_engine()
    params = {
        "run_type": run_type,
        "model_version": model_version,
        "horizon_days": horizon_days,
        "top_n": top_n,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "config_json": json.dumps(config_json) if isinstance(config_json, (dict, list)) else config_json,
        "comment": comment,
    }
    insert_sql = text(
        """
        INSERT INTO research.dim_model_run
        (run_type, model_version, horizon_days, top_n, train_start_date, train_end_date, config_json, comment)
        VALUES (:run_type, :model_version, :horizon_days, :top_n, :train_start_date, :train_end_date, :config_json, :comment)
        RETURNING run_id
        """
    )

    for attempt in range(2):
        try:
            with eng.begin() as conn:
                res = conn.execute(insert_sql, params)
                return int(res.scalar_one())
        except IntegrityError as exc:
            if attempt == 1 or "dim_model_run_pkey" not in str(exc):
                raise
            with eng.begin() as conn:
                conn.execute(
                    text(
                        """
                        SELECT setval(
                            pg_get_serial_sequence('research.dim_model_run', 'run_id'),
                            COALESCE((SELECT MAX(run_id) FROM research.dim_model_run), 0) + 1,
                            false
                        )
                        """
                    )
                )

    raise RuntimeError("failed to create research.dim_model_run row")
