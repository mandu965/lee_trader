from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path

from sqlalchemy import text

from python.us.dashboard.config import DashboardConfig
from python.us.us_db import get_us_engine, relation_exists


def _safe_json_load(path: Path) -> dict[str, object] | list[object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_json_value(value: object) -> object:
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return value
        if (text_value.startswith("{") and text_value.endswith("}")) or (text_value.startswith("[") and text_value.endswith("]")):
            try:
                return json.loads(text_value)
            except Exception:
                return value
    return value


def _normalize_row(row: dict[str, object]) -> dict[str, object]:
    return {key: _normalize_json_value(value) for key, value in row.items()}


def _fetch_rows(sql_text: str, params: dict[str, object], relation_name: str) -> list[dict[str, object]]:
    if not relation_exists(relation_name):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql_text), params).mappings().all()
    return [_normalize_row(dict(row)) for row in rows]


def _find_latest_trade_date_from_dir(directory: Path, pattern: str) -> str | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in candidates:
        prefix = path.name.split("_", 1)[0]
        try:
            date.fromisoformat(prefix)
        except ValueError:
            continue
        return prefix
    return None


def _find_latest_by_mtime(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_latest_matching_json(directory: Path, pattern: str, trade_date: str | None = None) -> dict[str, object] | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in files:
        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue
        if trade_date is None or str(payload.get("trade_date") or payload.get("evaluation_date") or "").strip() == trade_date:
            payload["_source_json_path"] = str(path)
            return payload
    return None


def _load_trade_scheduler_job_logs(directory: Path, trade_date: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not directory.exists():
        return rows
    pattern = "trade_scheduler_job_*.json" if trade_date is None else f"trade_scheduler_job_{trade_date}_*.json"
    for path in sorted(directory.glob(pattern)):
        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue
        payload["_source_json_path"] = str(path)
        rows.append(payload)
    return rows


def _derive_trade_date(cfg: DashboardConfig, requested_trade_date: str | None) -> str:
    if requested_trade_date:
        return requested_trade_date
    detected = _find_latest_trade_date_from_dir(cfg.trade_report_dir, "*_integrated_trade_report.json")
    if detected:
        return detected
    readiness_path = _find_latest_by_mtime(cfg.readiness_output_dir, "*_live_readiness.json")
    if readiness_path is not None:
        prefix = readiness_path.name.split("_", 1)[0]
        try:
            date.fromisoformat(prefix)
            return prefix
        except ValueError:
            pass
    return date.today().isoformat()


def load_dashboard_raw_data(
    cfg: DashboardConfig,
    *,
    trade_date: str | None = None,
) -> dict[str, object]:
    missing_sources: list[str] = []
    load_warnings: list[str] = list(cfg.warnings)
    effective_trade_date = _derive_trade_date(cfg, trade_date)

    integrated_report = None
    orchestration_logs: list[dict[str, object]] = []
    buy_decisions: list[dict[str, object]] = []
    sell_decisions: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    paper_buy_orders: list[dict[str, object]] = []
    paper_sell_orders: list[dict[str, object]] = []
    paper_positions: list[dict[str, object]] = []
    paper_position_snapshots: list[dict[str, object]] = []
    scheduler_run_logs: list[dict[str, object]] = []
    scheduler_health_rows: list[dict[str, object]] = []
    readiness = None

    try:
        rows = _fetch_rows(
            """
            SELECT trade_date, mode, summary_json, source_json_path
            FROM trade.us_integrated_daily_report
            WHERE trade_date = :trade_date
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            LIMIT 1
            """,
            {"trade_date": effective_trade_date},
            "trade.us_integrated_daily_report",
        )
        if rows:
            integrated_report = rows[0].get("summary_json")
            if isinstance(integrated_report, dict):
                integrated_report["_source_json_path"] = rows[0].get("source_json_path")
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_integrated_daily_report:{exc}")
    if integrated_report is None:
        file_path = cfg.trade_report_dir / f"{effective_trade_date}_integrated_trade_report.json"
        payload = _safe_json_load(file_path)
        if isinstance(payload, dict):
            payload["_source_json_path"] = str(file_path)
            integrated_report = payload
        else:
            missing_sources.append("integrated_daily_report")

    try:
        orchestration_logs = _fetch_rows(
            """
            SELECT *
            FROM trade.us_trade_orchestration_log
            WHERE trade_date = :trade_date
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
            """,
            {"trade_date": effective_trade_date},
            "trade.us_trade_orchestration_log",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_trade_orchestration_log:{exc}")
    if not orchestration_logs:
        payload = _load_latest_matching_json(cfg.trade_report_dir, f"{effective_trade_date}_trade_orchestration_log_*.json")
        if isinstance(payload, dict):
            orchestration_logs = [payload]
        else:
            missing_sources.append("trade.us_trade_orchestration_log")

    try:
        buy_decisions = _fetch_rows(
            """
            SELECT *
            FROM trade.us_buy_decision_log
            WHERE trade_date = :trade_date
            ORDER BY created_at DESC NULLS LAST, symbol
            """,
            {"trade_date": effective_trade_date},
            "trade.us_buy_decision_log",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_buy_decision_log:{exc}")
    if not buy_decisions:
        payload = _load_latest_matching_json(cfg.buy_output_dir, "buy_automation_*.json", effective_trade_date)
        if isinstance(payload, dict):
            buy_decisions = list(payload.get("candidates") or [])
            paper_buy_orders = list(payload.get("paper_orders") or [])
        else:
            missing_sources.append("trade.us_buy_decision_log")

    try:
        sell_decisions = _fetch_rows(
            """
            SELECT *
            FROM trade.us_sell_decision_log
            WHERE trade_date = :trade_date
            ORDER BY created_at DESC NULLS LAST, symbol
            """,
            {"trade_date": effective_trade_date},
            "trade.us_sell_decision_log",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_sell_decision_log:{exc}")
    sell_raw_payload: dict[str, object] | None = None
    if not sell_decisions:
        payload = _load_latest_matching_json(cfg.sell_output_dir, "sell_automation_*.json", effective_trade_date)
        if isinstance(payload, dict):
            sell_raw_payload = payload
            sell_decisions = list(payload.get("decisions") or [])
            paper_positions = list(payload.get("positions") or [])
            paper_sell_orders = list(payload.get("paper_sell_orders") or [])
        else:
            missing_sources.append("trade.us_sell_decision_log")

    try:
        conflicts = _fetch_rows(
            """
            SELECT *
            FROM trade.us_trade_conflict_log
            WHERE trade_date = :trade_date
            ORDER BY created_at DESC NULLS LAST, symbol
            """,
            {"trade_date": effective_trade_date},
            "trade.us_trade_conflict_log",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_trade_conflict_log:{exc}")
    if not conflicts and orchestration_logs:
        first_log = orchestration_logs[0]
        conflicts = list(first_log.get("conflict_results") or [])
        if not conflicts:
            missing_sources.append("trade.us_trade_conflict_log")

    if not paper_buy_orders:
        try:
            paper_buy_orders = _fetch_rows(
                """
                SELECT *
                FROM trade.us_paper_order
                WHERE trade_date = :trade_date
                ORDER BY created_at DESC NULLS LAST, symbol
                """,
                {"trade_date": effective_trade_date},
                "trade.us_paper_order",
            )
        except Exception as exc:
            load_warnings.append(f"DB_LOAD_WARNING:trade.us_paper_order:{exc}")
    if not paper_buy_orders and integrated_report and isinstance(integrated_report, dict):
        paper_buy_orders = []

    if not paper_sell_orders:
        try:
            paper_sell_orders = _fetch_rows(
                """
                SELECT *
                FROM trade.us_paper_sell_order
                WHERE trade_date = :trade_date
                ORDER BY created_at DESC NULLS LAST, symbol
                """,
                {"trade_date": effective_trade_date},
                "trade.us_paper_sell_order",
            )
        except Exception as exc:
            load_warnings.append(f"DB_LOAD_WARNING:trade.us_paper_sell_order:{exc}")

    if not paper_positions:
        try:
            paper_positions = _fetch_rows(
                """
                SELECT *
                FROM trade.us_paper_position
                ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, symbol
                """,
                {},
                "trade.us_paper_position",
            )
        except Exception as exc:
            load_warnings.append(f"DB_LOAD_WARNING:trade.us_paper_position:{exc}")

    try:
        paper_position_snapshots = _fetch_rows(
            """
            SELECT *
            FROM trade.us_paper_position_snapshot
            WHERE snapshot_date = :trade_date
            ORDER BY created_at DESC NULLS LAST, symbol
            """,
            {"trade_date": effective_trade_date},
            "trade.us_paper_position_snapshot",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_paper_position_snapshot:{exc}")
    if not paper_position_snapshots and paper_positions:
        paper_position_snapshots = list(paper_positions)
    if not paper_position_snapshots:
        missing_sources.append("trade.us_paper_position_snapshot")

    try:
        scheduler_run_logs = _fetch_rows(
            """
            SELECT *
            FROM trade.us_trade_scheduler_run_log
            WHERE trade_date = :trade_date
            ORDER BY created_at DESC NULLS LAST
            """,
            {"trade_date": effective_trade_date},
            "trade.us_trade_scheduler_run_log",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_trade_scheduler_run_log:{exc}")
    if not scheduler_run_logs:
        scheduler_run_logs = _load_trade_scheduler_job_logs(cfg.trade_report_dir, effective_trade_date)
        if not scheduler_run_logs:
            missing_sources.append("trade.us_trade_scheduler_run_log")

    try:
        scheduler_health_rows = _fetch_rows(
            """
            SELECT *
            FROM trade.us_trade_scheduler_health_check
            WHERE trade_date = :trade_date
            ORDER BY created_at DESC NULLS LAST
            """,
            {"trade_date": effective_trade_date},
            "trade.us_trade_scheduler_health_check",
        )
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_trade_scheduler_health_check:{exc}")
    if not scheduler_health_rows:
        if scheduler_run_logs:
            scheduler_health_rows = [
                {
                    "trade_date": effective_trade_date,
                    "health_status": "PASS" if scheduler_run_logs[0].get("health_check_passed") else "UNKNOWN",
                    "warnings": [],
                    "errors": [],
                }
            ]
        else:
            missing_sources.append("trade.us_trade_scheduler_health_check")

    try:
        rows = _fetch_rows(
            """
            SELECT *
            FROM trade.us_buy_readiness_report
            WHERE evaluation_date = :trade_date
            ORDER BY created_at DESC NULLS LAST
            LIMIT 1
            """,
            {"trade_date": effective_trade_date},
            "trade.us_buy_readiness_report",
        )
        if rows:
            readiness = rows[0]
    except Exception as exc:
        load_warnings.append(f"DB_LOAD_WARNING:trade.us_buy_readiness_report:{exc}")
    if readiness is None:
        readiness_path = cfg.readiness_output_dir / f"{effective_trade_date}_live_readiness.json"
        payload = _safe_json_load(readiness_path)
        if isinstance(payload, dict):
            payload["_source_json_path"] = str(readiness_path)
            readiness = payload
        else:
            missing_sources.append("trade.us_buy_readiness_report")

    return {
        "trade_date": effective_trade_date,
        "loaded_at": datetime.now().isoformat(),
        "integrated_report": integrated_report,
        "orchestration_logs": orchestration_logs,
        "buy_decisions": buy_decisions,
        "sell_decisions": sell_decisions,
        "conflicts": conflicts,
        "paper_buy_orders": paper_buy_orders,
        "paper_sell_orders": paper_sell_orders,
        "paper_positions": paper_positions,
        "paper_position_snapshots": paper_position_snapshots,
        "scheduler_run_logs": scheduler_run_logs,
        "scheduler_health_rows": scheduler_health_rows,
        "readiness": readiness,
        "sell_raw_payload": sell_raw_payload,
        "missing_sources": sorted(set(missing_sources)),
        "load_warnings": load_warnings,
    }
