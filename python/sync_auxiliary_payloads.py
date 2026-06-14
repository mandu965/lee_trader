from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from payload_store import upsert_json_payload

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
HISTORY_DIR = DATA_DIR / "history"
AUTO_TRADING_POLICY_PATH = OUTPUT_DIR / "auto_trading_policy.json"
AUTO_TRADING_OPS_STATUS_PATH = OUTPUT_DIR / "auto_trading_ops_status.json"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if load_dotenv:
        load_dotenv(env_path, override=False)
        return
    # Fallback: manually parse .env when python-dotenv is unavailable
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().split("#")[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    value = json.loads(normalized)
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str, "pdno": str, "symbol": str}, low_memory=False)
    if df.empty:
        return []
    for col in ("code", "pdno", "symbol"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            mask = df[col].str.fullmatch(r"\d{1,6}", na=False)
            df.loc[mask, col] = df.loc[mask, col].str.zfill(6)
    return df.where(pd.notna(df), None).to_dict(orient="records")


def parse_key_value_markdown(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
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


def sync_rows_payload(
    payload_key: str,
    path: Path,
    *,
    rows_field: str = "items",
    asof_date: Any = None,
    generated_at: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    rows = read_csv_rows(path)
    if not rows:
        return
    payload: dict[str, Any] = {
        "entity": payload_key,
        "generated_at": generated_at or pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "row_count": len(rows),
        rows_field: rows,
    }
    if extra:
        payload.update(extra)
    upsert_json_payload(
        payload_key,
        payload,
        asof_date=asof_date,
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


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def sync_auto_trading_policy_payload() -> None:
    payload = {
        "entity": "auto_trading_policy",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "auto_trade_execute": _env_flag("AUTO_TRADE_EXECUTE", False),
        "auto_trade_allow_buy": _env_flag("AUTO_TRADE_ALLOW_BUY", False),
        "buy_approval_required": _env_flag("AUTO_TRADE_BUY_APPROVAL_REQUIRED", False),
        "confirm_configured": str(os.environ.get("AUTO_TRADE_CONFIRM_TEXT", "")).strip() == "LIVE_ORDER",
        "source": "env_snapshot",
    }
    AUTO_TRADING_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_TRADING_POLICY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_json_payload(
        "auto_trading_policy",
        payload,
        generated_at=payload.get("generated_at"),
        source_path=AUTO_TRADING_POLICY_PATH,
    )


def _today_local() -> str:
    return pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    try:
        if parsed.tzinfo is None:
            return parsed.tz_localize("Asia/Seoul")
    except Exception:
        pass
    try:
        return parsed.tz_convert("Asia/Seoul")
    except Exception:
        return parsed


def _is_today(value: Any, today_str: str) -> bool:
    text = str(value or "").strip()
    if text == today_str:
        return True
    parsed = _parse_timestamp(value)
    if parsed is None:
        return False
    return parsed.strftime("%Y-%m-%d") == today_str


def _count_where(items: list[dict[str, Any]], predicate) -> int:
    return sum(1 for item in items if predicate(item))


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {"none", "null"}:
            return text
    return None


def _scheduler_message(payload: dict[str, Any]) -> str | None:
    failure = payload.get("last_failure_details") if isinstance(payload.get("last_failure_details"), dict) else {}
    warning = payload.get("last_warning_details") if isinstance(payload.get("last_warning_details"), dict) else {}
    return _first_text(
        failure.get("error_message"),
        warning.get("error_message"),
        payload.get("last_error"),
        payload.get("status_note"),
    )


def _status_tone(success_today: bool, *, missing: bool = False, failure_today: bool = False, stopped: bool = False, stale: bool = False) -> str:
    if stopped:
        return "stopped"
    if missing:
        return "warning"
    if failure_today:
        return "risk"
    if success_today:
        return "normal"
    if stale:
        return "warning"
    return "warning"


def _generic_scheduler_card(name: str, path: Path, today_str: str) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        return {
            "name": name,
            "source_path": str(path),
            "available": False,
            "today_success": False,
            "status_tone": "warning",
            "status_label": "주의",
            "warning_reason": "scheduler_status_missing",
            "last_success_at": None,
            "last_failure_at": None,
            "last_error_message": "scheduler_status_missing",
        }
    last_success_at = payload.get("last_success_at")
    last_failure_at = payload.get("last_failure_at")
    last_warning_at = payload.get("last_warning_at")
    failure = payload.get("last_failure_details") if isinstance(payload.get("last_failure_details"), dict) else {}
    warning = payload.get("last_warning_details") if isinstance(payload.get("last_warning_details"), dict) else {}
    last_error = _scheduler_message(payload)
    today_success = _is_today(payload.get("last_success_date") or last_success_at, today_str)
    failure_today = _is_today(last_failure_at, today_str)
    warning_today = _is_today(last_warning_at, today_str)
    success_ts = _parse_timestamp(last_success_at)
    failure_ts = _parse_timestamp(last_failure_at)
    recovered_after_failure = bool(
        success_ts is not None and failure_ts is not None and success_ts > failure_ts
    )
    if recovered_after_failure:
        failure_today = False
    stale = not today_success and bool(last_success_at)
    tone = _status_tone(today_success, failure_today=failure_today, stale=stale)
    if tone == "normal" and warning_today:
        tone = "warning"
    label = "정상" if tone == "normal" else "위험" if tone == "risk" else "주의"
    return {
        "name": name,
        "source_path": str(path),
        "available": True,
        "today_success": today_success,
        "status_tone": tone,
        "status_label": label,
        "scheduler_status": payload.get("status"),
        "last_success_at": last_success_at,
        "last_failure_at": last_failure_at,
        "last_warning_at": last_warning_at,
        "last_error_message": last_error,
        "last_failure_details": failure,
        "last_warning_details": warning,
        "status_note": payload.get("status_note"),
        "configured_daily_time": payload.get("configured_daily_time"),
    }


def _close_batch_fallback_card(today_str: str) -> dict[str, Any]:
    card = _generic_scheduler_card("close_batch", OUTPUT_DIR / "auto_ops_scheduler_status.json", today_str)
    card["source"] = "scheduler_status_fallback"
    return card


def _close_batch_card(today_str: str) -> dict[str, Any]:
    path = OUTPUT_DIR / "operational_daily_cycle_status.json"
    payload = read_json(path)
    if not payload:
        fallback = _close_batch_fallback_card(today_str)
        fallback["warning_reason"] = "daily_cycle_status_missing"
        return fallback
    steps = payload.get("steps") or []
    critical_failures = [
        step for step in steps
        if bool(step.get("critical")) and str(step.get("status") or "").upper() in {"FAILED", "ERROR"}
    ]
    finished_at = payload.get("finished_at")
    finished_today = _is_today(finished_at, today_str)
    if not finished_today:
        fallback = _close_batch_fallback_card(today_str)
        fallback["source_path"] = str(path)
        fallback["warning_reason"] = "daily_cycle_status_stale"
        fallback["fallback_finished_at"] = finished_at
        return fallback
    overall = str(payload.get("overall_status") or "").upper()
    success_today = finished_today and not critical_failures and overall not in {"FAILED", "ERROR"}
    last_error = None
    if critical_failures:
        failed_step = critical_failures[0]
        last_error = _first_text(failed_step.get("error"), failed_step.get("wait_reason"), failed_step.get("name"))
    tone = _status_tone(success_today, failure_today=finished_today and not success_today, stale=bool(finished_at) and not finished_today)
    label = "정상" if tone == "normal" else "위험" if tone == "risk" else "주의"
    return {
        "name": "close_batch",
        "source_path": str(path),
        "available": True,
        "today_success": success_today,
        "status_tone": tone,
        "status_label": label,
        "overall_status": overall,
        "last_success_at": finished_at if success_today else None,
        "last_failure_at": finished_at if finished_today and not success_today else None,
        "last_error_message": last_error,
        "critical_failure_count": len(critical_failures),
        "source": "operational_daily_cycle_status",
    }


def _ai_metrics(today_str: str) -> dict[str, Any]:
    preview = read_json(OUTPUT_DIR / "order_requests_preview.json")
    execution = read_json(OUTPUT_DIR / "order_requests_execution.json")
    fills = read_json(OUTPUT_DIR / "live_order_fills.json")
    preview_is_today = _is_today(preview.get("asof_date") or preview.get("generated_at"), today_str)
    execution_is_today = _is_today(execution.get("asof_date") or execution.get("executed_at") or execution.get("generated_at"), today_str)
    preview_items = list(preview.get("items") or []) if preview_is_today else []
    execution_items = list(execution.get("items") or []) if execution_is_today else []
    fill_items = list(fills.get("items") or [])
    buy_candidates = _count_where(preview_items, lambda item: str(item.get("side") or "").upper() == "BUY")
    buy_blocked = _count_where(
        preview_items,
        lambda item: str(item.get("side") or "").upper() == "BUY"
        and (
            _first_text(item.get("blocked_reason"), item.get("entry_price_gate_reason")) is not None
            or bool(item.get("common_risk_block_reasons"))
            or (item.get("executable_now") is False)
        ),
    )
    submitted = _count_where(
        execution_items,
        lambda item: str(item.get("side") or "").upper() == "BUY"
        and str(item.get("submission_status") or "").lower() == "submitted",
    )
    filled = _count_where(
        fill_items,
        lambda item: str(item.get("side") or "").upper() == "BUY"
        and _is_today(item.get("as_of_date") or item.get("filled_at"), today_str),
    )
    return {
        "buy_candidate_count": buy_candidates,
        "buy_blocked_count": buy_blocked,
        "submitted_count": submitted,
        "filled_count": filled,
        "preview_generated_at": preview.get("generated_at"),
        "execution_generated_at": execution.get("generated_at") or execution.get("executed_at"),
        "fills_generated_at": fills.get("generated_at"),
    }


def _rule_metrics(today_str: str) -> dict[str, Any]:
    preview = read_json(OUTPUT_DIR / "rule_order_preview.json")
    execution = read_json(OUTPUT_DIR / "rule_execution_results.json")
    preview_is_today = _is_today(preview.get("as_of_date") or preview.get("generated_at"), today_str)
    execution_is_today = _is_today(execution.get("as_of_date") or execution.get("generated_at"), today_str)
    preview_items = list(preview.get("items") or []) if preview_is_today else []
    execution_summary = execution.get("summary") or {}
    buy_candidates = _count_where(preview_items, lambda item: str(item.get("side") or "").upper() == "BUY")
    buy_blocked = _count_where(
        preview_items,
        lambda item: str(item.get("side") or "").upper() == "BUY"
        and (
            not bool(item.get("order_allowed"))
            or _first_text(item.get("order_block_reason")) not in {None, "none"}
        ),
    )
    filled = int(execution_summary.get("filled_count") or 0) + int(execution_summary.get("simulated_filled_count") or 0) if execution_is_today else 0
    submitted = int(execution_summary.get("submitted_count") or 0) if execution_is_today else 0
    return {
        "buy_candidate_count": buy_candidates,
        "buy_blocked_count": buy_blocked,
        "submitted_count": submitted,
        "filled_count": filled,
        "preview_generated_at": preview.get("generated_at"),
        "execution_generated_at": execution.get("generated_at"),
        "as_of_date": execution.get("as_of_date") or preview.get("as_of_date"),
        "today_execution_available": execution_is_today,
    }


def sync_auto_trading_ops_status_payload() -> None:
    today_str = _today_local()
    controls = {
        "global_kill_switch": _env_flag("GLOBAL_KILL_SWITCH", False),
        "rule_kill_switch": _env_flag("RULE_KILL_SWITCH", False),
        "auto_trade_execute": _env_flag("AUTO_TRADE_EXECUTE", False),
        "auto_trade_allow_buy": _env_flag("AUTO_TRADE_ALLOW_BUY", False),
    }
    cards = {
        "close_batch": _close_batch_card(today_str),
        "ai_auto_buy": _generic_scheduler_card("ai_auto_buy", OUTPUT_DIR / "auto_ops_auto_buy_scheduler_status.json", today_str),
        "rule_before_open": _generic_scheduler_card("rule_before_open", OUTPUT_DIR / "rule_before_open_scheduler_status.json", today_str),
        "rule_after_open": _generic_scheduler_card("rule_after_open", OUTPUT_DIR / "rule_after_open_scheduler_status.json", today_str),
        "live_account_sync": _generic_scheduler_card("live_account_sync", OUTPUT_DIR / "auto_ops_live_account_sync_scheduler_status.json", today_str),
    }
    ai = _ai_metrics(today_str)
    rule = _rule_metrics(today_str)
    card_list = list(cards.values())
    success_times = [item.get("last_success_at") for item in card_list if item.get("last_success_at")]
    failure_times = [item.get("last_failure_at") for item in card_list if item.get("last_failure_at")]
    latest_error = next((item.get("last_error_message") for item in card_list if item.get("last_error_message")), None)
    stopped = controls["global_kill_switch"] or controls["rule_kill_switch"]
    overall_tone = "stopped" if stopped else "risk" if any(item.get("status_tone") == "risk" for item in card_list) else "warning" if any(item.get("status_tone") == "warning" for item in card_list) else "normal"
    payload = {
        "entity": "auto_trading_ops_status",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "as_of_date": today_str,
        "controls": controls,
        "cards": cards,
        "ai": ai,
        "rule": rule,
        "summary": {
            "overall_tone": overall_tone,
            "today_close_batch_success": bool(cards["close_batch"].get("today_success")),
            "today_ai_auto_buy_success": bool(cards["ai_auto_buy"].get("today_success")),
            "today_rule_before_open_success": bool(cards["rule_before_open"].get("today_success")),
            "today_rule_after_open_success": bool(cards["rule_after_open"].get("today_success")),
            "today_live_account_sync_success": bool(cards["live_account_sync"].get("today_success")),
            "latest_success_at": max(success_times) if success_times else None,
            "latest_failure_at": max(failure_times) if failure_times else None,
            "latest_error_message": latest_error,
        },
    }
    AUTO_TRADING_OPS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_TRADING_OPS_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_json_payload(
        "auto_trading_ops_status",
        payload,
        asof_date=today_str,
        generated_at=payload.get("generated_at"),
        source_path=AUTO_TRADING_OPS_STATUS_PATH,
    )


def main() -> int:
    _load_env()
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
    sync_auto_trading_policy_payload()
    sync_auto_trading_ops_status_payload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
