from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from python.us.trade_orchestration.config import TradeOrchestrationConfig


def _lock_path(cfg: TradeOrchestrationConfig, trade_date: str) -> Path:
    return cfg.lock_dir / f"{trade_date}.lock"


def inspect_run_lock(cfg: TradeOrchestrationConfig, *, trade_date: str) -> dict[str, object]:
    path = _lock_path(cfg, trade_date)
    if not path.exists():
        return {"lock_exists": False, "lock_path": str(path), "stale": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    now_ts = time.time()
    created_ts = float(payload.get("created_ts") or path.stat().st_mtime)
    age_seconds = max(0.0, now_ts - created_ts)
    stale = age_seconds > cfg.scheduler_lock_ttl_seconds
    return {
        "lock_exists": True,
        "lock_path": str(path),
        "stale": stale,
        "age_seconds": age_seconds,
        "payload": payload,
    }


def acquire_run_lock(
    cfg: TradeOrchestrationConfig,
    *,
    trade_date: str,
    owner: str,
) -> dict[str, object]:
    path = _lock_path(cfg, trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_lock_removed = False
    inspected = inspect_run_lock(cfg, trade_date=trade_date)
    if inspected.get("lock_exists"):
        if inspected.get("stale"):
            try:
                path.unlink(missing_ok=True)
                stale_lock_removed = True
            except Exception:
                return {
                    "lock_acquired": False,
                    "reason": "STALE_LOCK_REMOVE_FAILED",
                    "lock_path": str(path),
                    "stale_lock_removed": False,
                }
        else:
            return {
                "lock_acquired": False,
                "reason": "DUPLICATE_RUN_DETECTED",
                "lock_path": str(path),
                "stale_lock_removed": False,
            }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_ts": time.time(),
        "trade_date": trade_date,
        "owner": owner,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "lock_acquired": True,
        "lock_path": str(path),
        "stale_lock_removed": stale_lock_removed,
    }


def release_run_lock(cfg: TradeOrchestrationConfig, *, trade_date: str) -> dict[str, object]:
    path = _lock_path(cfg, trade_date)
    if not path.exists():
        return {"released": True, "lock_path": str(path), "warning": "LOCK_ALREADY_MISSING"}
    try:
        path.unlink()
        return {"released": True, "lock_path": str(path), "warning": None}
    except Exception as exc:
        return {"released": False, "lock_path": str(path), "warning": f"LOCK_RELEASE_FAILED:{exc}"}
