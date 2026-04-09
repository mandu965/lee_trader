from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from payload_store import upsert_json_payload


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
HISTORY_DIR = DATA_DIR / "history"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    value = json.loads(normalized)
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if df.empty:
        return []
    return df.where(pd.notna(df), None).to_dict(orient="records")


def parse_key_value_markdown(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = str(raw_line or "").strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        out[key.strip()] = value.strip()
    return out


def resolve_asof_from_payload(payload: dict[str, Any], asof_field: str | None) -> Any:
    if not asof_field:
        return None
    asof_value = payload.get(asof_field)
    if isinstance(asof_value, dict):
        return asof_value.get("latest_asof_date") or asof_value.get("latest_date")
    return asof_value


def sync_json_payload(payload_key: str, path: Path, *, asof_field: str | None = None) -> None:
    payload = read_json(path)
    if not payload:
        return
    upsert_json_payload(
        payload_key,
        payload,
        asof_date=resolve_asof_from_payload(payload, asof_field),
        generated_at=payload.get("generated_at"),
        source_path=path,
    )


def sync_history_payload(payload_key: str, path: Path, *, asof_field: str) -> None:
    rows = read_csv_rows(path)
    if not rows:
        return
    latest_row = rows[-1]
    payload = {
        "entity": payload_key,
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "row_count": len(rows),
        "latest_asof_date": latest_row.get(asof_field),
        "rows": rows,
    }
    upsert_json_payload(
        payload_key,
        payload,
        asof_date=latest_row.get(asof_field),
        generated_at=payload.get("generated_at"),
        source_path=path,
    )


def sync_inventory_payload() -> None:
    csv_path = HISTORY_DIR / "ranking_snapshot_inventory.csv"
    md_path = HISTORY_DIR / "ranking_snapshot_inventory.md"
    rows = read_csv_rows(csv_path)
    key_values = parse_key_value_markdown(md_path)
    if not rows and not key_values:
        return
    latest_asof_date = None
    if rows:
        latest_asof_date = rows[-1].get("as_of_date")
    elif key_values:
        latest_asof_date = key_values.get("latest snapshot date")
    payload = {
        "entity": "ranking_snapshot_inventory",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "summary": key_values,
        "rows": rows,
    }
    upsert_json_payload(
        "ranking_snapshot_inventory",
        payload,
        asof_date=latest_asof_date,
        generated_at=payload.get("generated_at"),
        source_path=csv_path if csv_path.exists() else md_path,
    )


def main() -> int:
    sync_json_payload("operational_daily_cycle_status", OUTPUT_DIR / "operational_daily_cycle_status.json")
    sync_json_payload(
        "shadow_quality_risk_guard_repeatability_report",
        OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json",
        asof_field="summary",
    )
    sync_history_payload(
        "operational_buy_gate_history",
        HISTORY_DIR / "operational_buy_gate_history.csv",
        asof_field="as_of_date",
    )
    sync_history_payload(
        "score_kpi_monitor_history",
        HISTORY_DIR / "score_kpi_monitor_history.csv",
        asof_field="as_of_date",
    )
    sync_inventory_payload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
