from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import text

from python.us.trade_orchestration.config import TradeOrchestrationConfig
from python.us.us_db import get_us_engine, relation_exists


INSERT_ORCHESTRATION_SQL = text(
    """
    INSERT INTO trade.us_trade_orchestration_log (
        orchestration_log_id,
        execution_time,
        trade_date,
        mode,
        orchestration_enabled,
        sell_executed,
        buy_executed,
        report_generated,
        success,
        fail_safe_triggered,
        conflict_count,
        conflict_summary,
        final_action_summary,
        error_message,
        created_at,
        updated_at
    ) VALUES (
        :orchestration_log_id,
        :execution_time,
        :trade_date,
        :mode,
        :orchestration_enabled,
        :sell_executed,
        :buy_executed,
        :report_generated,
        :success,
        :fail_safe_triggered,
        :conflict_count,
        CAST(:conflict_summary AS JSONB),
        CAST(:final_action_summary AS JSONB),
        :error_message,
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, mode)
    DO UPDATE SET
        execution_time = EXCLUDED.execution_time,
        orchestration_enabled = EXCLUDED.orchestration_enabled,
        sell_executed = EXCLUDED.sell_executed,
        buy_executed = EXCLUDED.buy_executed,
        report_generated = EXCLUDED.report_generated,
        success = EXCLUDED.success,
        fail_safe_triggered = EXCLUDED.fail_safe_triggered,
        conflict_count = EXCLUDED.conflict_count,
        conflict_summary = EXCLUDED.conflict_summary,
        final_action_summary = EXCLUDED.final_action_summary,
        error_message = EXCLUDED.error_message,
        updated_at = EXCLUDED.updated_at
    """
)

INSERT_CONFLICT_SQL = text(
    """
    INSERT INTO trade.us_trade_conflict_log (
        conflict_log_id,
        trade_date,
        mode,
        symbol,
        buy_allowed_after_conflict_check,
        conflict_reasons,
        related_position_id,
        related_sell_signal,
        cooldown_until,
        created_at,
        updated_at
    ) VALUES (
        :conflict_log_id,
        :trade_date,
        :mode,
        :symbol,
        :buy_allowed_after_conflict_check,
        CAST(:conflict_reasons AS JSONB),
        :related_position_id,
        CAST(:related_sell_signal AS JSONB),
        :cooldown_until,
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, mode, symbol)
    DO UPDATE SET
        buy_allowed_after_conflict_check = EXCLUDED.buy_allowed_after_conflict_check,
        conflict_reasons = EXCLUDED.conflict_reasons,
        related_position_id = EXCLUDED.related_position_id,
        related_sell_signal = EXCLUDED.related_sell_signal,
        cooldown_until = EXCLUDED.cooldown_until,
        updated_at = EXCLUDED.updated_at
    """
)

INSERT_REPORT_SQL = text(
    """
    INSERT INTO trade.us_integrated_daily_report (
        report_id,
        trade_date,
        mode,
        report_type,
        source_json_path,
        summary_json,
        created_at,
        updated_at
    ) VALUES (
        :report_id,
        :trade_date,
        :mode,
        :report_type,
        :source_json_path,
        CAST(:summary_json AS JSONB),
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, mode, report_type)
    DO UPDATE SET
        source_json_path = EXCLUDED.source_json_path,
        summary_json = EXCLUDED.summary_json,
        updated_at = EXCLUDED.updated_at
    """
)


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")
    return path


def persist_integrated_logs(
    *,
    cfg: TradeOrchestrationConfig,
    orchestration_result: dict[str, object],
    integrated_report: dict[str, object] | None,
) -> dict[str, object]:
    trade_date = str(orchestration_result.get("trade_date") or "unknown")
    mode = str(orchestration_result.get("mode") or cfg.mode).upper()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    path = _write_json(cfg.report_output_dir / f"{trade_date}_trade_orchestration_log_{timestamp}.json", orchestration_result)
    persisted = {
        "json_path": str(path),
        "db_orchestration_logs": 0,
        "db_conflict_logs": 0,
        "db_report_logs": 0,
    }

    try:
        engine = get_us_engine()
        with engine.begin() as conn:
            if relation_exists("trade.us_trade_orchestration_log"):
                conn.execute(
                    INSERT_ORCHESTRATION_SQL,
                    {
                        "orchestration_log_id": f"USTRADELOG_{trade_date}_{mode}",
                        "execution_time": datetime.now(timezone.utc),
                        "trade_date": trade_date,
                        "mode": mode,
                        "orchestration_enabled": orchestration_result.get("enabled"),
                        "sell_executed": orchestration_result.get("sell_executed"),
                        "buy_executed": orchestration_result.get("buy_executed"),
                        "report_generated": orchestration_result.get("report_generated"),
                        "success": orchestration_result.get("success"),
                        "fail_safe_triggered": orchestration_result.get("fail_safe_triggered"),
                        "conflict_count": orchestration_result.get("conflict_summary", {}).get("TOTAL_CONFLICT_BLOCKED", 0),
                        "conflict_summary": _json_text(orchestration_result.get("conflict_summary") or {}),
                        "final_action_summary": _json_text(orchestration_result.get("final_action_summary") or {}),
                        "error_message": orchestration_result.get("error"),
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                persisted["db_orchestration_logs"] = 1

            if relation_exists("trade.us_trade_conflict_log"):
                rows = []
                for index, item in enumerate(orchestration_result.get("conflict_results") or [], start=1):
                    rows.append(
                        {
                            "conflict_log_id": f"USTRADECONFLICT_{trade_date}_{mode}_{index}",
                            "trade_date": trade_date,
                            "mode": mode,
                            "symbol": item.get("symbol"),
                            "buy_allowed_after_conflict_check": item.get("buy_allowed_after_conflict_check"),
                            "conflict_reasons": _json_text(item.get("conflict_reasons") or []),
                            "related_position_id": item.get("related_position_id"),
                            "related_sell_signal": _json_text(item.get("sell_signal") or {}),
                            "cooldown_until": item.get("cooldown_until"),
                            "created_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_CONFLICT_SQL, rows)
                    persisted["db_conflict_logs"] = len(rows)

            if integrated_report is not None and relation_exists("trade.us_integrated_daily_report"):
                conn.execute(
                    INSERT_REPORT_SQL,
                    {
                        "report_id": f"USINTEGRATEDREPORT_{trade_date}_{mode}",
                        "trade_date": trade_date,
                        "mode": mode,
                        "report_type": "integrated_daily_report",
                        "source_json_path": str(cfg.report_output_dir / f"{trade_date}_integrated_trade_report.json"),
                        "summary_json": _json_text(integrated_report),
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                persisted["db_report_logs"] = 1
    except Exception:
        pass

    return persisted
