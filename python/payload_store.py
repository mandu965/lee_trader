from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine


CREATE_PAYLOAD_TABLE_SQL = """
CREATE SCHEMA IF NOT EXISTS research;
CREATE TABLE IF NOT EXISTS research.app_payload_store (
    payload_key   TEXT PRIMARY KEY,
    payload_json  JSONB NOT NULL,
    asof_date     DATE,
    generated_at  TIMESTAMPTZ,
    source_path   TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def ensure_payload_table() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(CREATE_PAYLOAD_TABLE_SQL))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _parse_generated_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text_value, fmt)
        except ValueError:
            continue
    return None


def _parse_asof_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return pd.to_datetime(text_value, errors="coerce").date()
    except Exception:
        return None


def upsert_json_payload(
    payload_key: str,
    payload: dict[str, Any],
    *,
    asof_date: Any = None,
    generated_at: Any = None,
    source_path: str | Path | None = None,
) -> None:
    ensure_payload_table()
    clean_payload = _sanitize(payload)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO research.app_payload_store
                    (payload_key, payload_json, asof_date, generated_at, source_path, updated_at)
                VALUES
                    (:payload_key, CAST(:payload_json AS JSONB), :asof_date, :generated_at, :source_path, now())
                ON CONFLICT (payload_key) DO UPDATE
                SET payload_json = EXCLUDED.payload_json,
                    asof_date = EXCLUDED.asof_date,
                    generated_at = EXCLUDED.generated_at,
                    source_path = EXCLUDED.source_path,
                    updated_at = now()
                """
            ),
            {
                "payload_key": payload_key,
                "payload_json": json.dumps(clean_payload, ensure_ascii=False),
                "asof_date": _parse_asof_date(asof_date),
                "generated_at": _parse_generated_at(generated_at),
                "source_path": str(source_path) if source_path is not None else None,
            },
        )
