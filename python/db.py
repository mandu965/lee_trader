"""
db.py

Provides SQLAlchemy engine/session helpers and a psycopg2 raw connector
using DATABASE_URL environment variable.
"""
import os
from pathlib import Path
from functools import lru_cache
from io import StringIO
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


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
