from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import text

from python.us.sell_automation.config import SellAutomationConfig
from python.us.us_db import get_us_engine, relation_exists


INSERT_DECISION_SQL = text(
    """
    INSERT INTO trade.us_sell_decision_log (
        sell_decision_id,
        trade_date,
        account_id,
        automation_mode,
        paper_position_id,
        symbol,
        decision,
        sell_action,
        sell_ratio,
        sell_quantity,
        exit_reason,
        review_required,
        applied_rules,
        latest_price,
        avg_entry_price,
        unrealized_pnl_pct,
        realized_paper_pnl,
        error_message,
        created_at,
        updated_at
    ) VALUES (
        :sell_decision_id,
        :trade_date,
        :account_id,
        :automation_mode,
        :paper_position_id,
        :symbol,
        :decision,
        :sell_action,
        :sell_ratio,
        :sell_quantity,
        :exit_reason,
        :review_required,
        CAST(:applied_rules AS JSONB),
        :latest_price,
        :avg_entry_price,
        :unrealized_pnl_pct,
        :realized_paper_pnl,
        :error_message,
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, automation_mode, paper_position_id)
    DO UPDATE SET
        decision = EXCLUDED.decision,
        sell_action = EXCLUDED.sell_action,
        sell_ratio = EXCLUDED.sell_ratio,
        sell_quantity = EXCLUDED.sell_quantity,
        exit_reason = EXCLUDED.exit_reason,
        review_required = EXCLUDED.review_required,
        applied_rules = EXCLUDED.applied_rules,
        latest_price = EXCLUDED.latest_price,
        avg_entry_price = EXCLUDED.avg_entry_price,
        unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
        realized_paper_pnl = EXCLUDED.realized_paper_pnl,
        error_message = EXCLUDED.error_message,
        updated_at = EXCLUDED.updated_at
    """
)

INSERT_SIGNAL_SQL = text(
    """
    INSERT INTO trade.us_sell_signal_log (
        sell_signal_id,
        trade_date,
        paper_position_id,
        symbol,
        rule_name,
        rule_result,
        metric_value,
        threshold_value,
        severity,
        detail,
        created_at
    ) VALUES (
        :sell_signal_id,
        :trade_date,
        :paper_position_id,
        :symbol,
        :rule_name,
        :rule_result,
        :metric_value,
        CAST(:threshold_value AS JSONB),
        :severity,
        :detail,
        :created_at
    )
    """
)

INSERT_PAPER_SELL_ORDER_SQL = text(
    """
    INSERT INTO trade.us_paper_sell_order (
        paper_sell_order_id,
        trade_date,
        paper_position_id,
        symbol,
        side,
        sell_action,
        sell_ratio,
        sell_quantity,
        sell_price_ref,
        sell_amount,
        assumed_fill_status,
        exit_reason,
        source_sell_decision_id,
        created_at,
        updated_at
    ) VALUES (
        :paper_sell_order_id,
        :trade_date,
        :paper_position_id,
        :symbol,
        :side,
        :sell_action,
        :sell_ratio,
        :sell_quantity,
        :sell_price_ref,
        :sell_amount,
        :assumed_fill_status,
        :exit_reason,
        :source_sell_decision_id,
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, paper_position_id, sell_action)
    DO UPDATE SET
        sell_ratio = EXCLUDED.sell_ratio,
        sell_quantity = EXCLUDED.sell_quantity,
        sell_price_ref = EXCLUDED.sell_price_ref,
        sell_amount = EXCLUDED.sell_amount,
        assumed_fill_status = EXCLUDED.assumed_fill_status,
        exit_reason = EXCLUDED.exit_reason,
        source_sell_decision_id = EXCLUDED.source_sell_decision_id,
        updated_at = EXCLUDED.updated_at
    """
)

INSERT_POSITION_SNAPSHOT_SQL = text(
    """
    INSERT INTO trade.us_paper_position_snapshot (
        snapshot_id,
        snapshot_date,
        paper_position_id,
        symbol,
        latest_price,
        remaining_quantity,
        highest_price_since_entry,
        unrealized_pnl,
        unrealized_pnl_pct,
        holding_days,
        status,
        data_quality_flags,
        created_at,
        updated_at
    ) VALUES (
        :snapshot_id,
        :snapshot_date,
        :paper_position_id,
        :symbol,
        :latest_price,
        :remaining_quantity,
        :highest_price_since_entry,
        :unrealized_pnl,
        :unrealized_pnl_pct,
        :holding_days,
        :status,
        CAST(:data_quality_flags AS JSONB),
        :created_at,
        :updated_at
    )
    ON CONFLICT (snapshot_date, paper_position_id)
    DO UPDATE SET
        latest_price = EXCLUDED.latest_price,
        remaining_quantity = EXCLUDED.remaining_quantity,
        highest_price_since_entry = EXCLUDED.highest_price_since_entry,
        unrealized_pnl = EXCLUDED.unrealized_pnl,
        unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
        holding_days = EXCLUDED.holding_days,
        status = EXCLUDED.status,
        data_quality_flags = EXCLUDED.data_quality_flags,
        updated_at = EXCLUDED.updated_at
    """
)


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")
    return path


def persist_sell_automation_logs(
    *,
    cfg: SellAutomationConfig,
    report: dict[str, object],
    account_id: str,
) -> dict[str, object]:
    trade_date = str(report.get("trade_date") or "unknown")
    mode = str(report.get("mode") or cfg.mode).upper()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_name = f"sell_automation_{trade_date}_{mode}_{timestamp}"
    json_path = _write_json(cfg.output_dir / f"{base_name}.json", report)

    persisted = {
        "json_path": str(json_path),
        "db_decision_logs": 0,
        "db_signal_logs": 0,
        "db_paper_sell_orders": 0,
        "db_position_snapshots": 0,
    }

    try:
        engine = get_us_engine()
        with engine.begin() as conn:
            if relation_exists("trade.us_sell_decision_log"):
                rows = []
                for item in report.get("decisions", []):
                    rows.append(
                        {
                            "sell_decision_id": item.get("sell_decision_id"),
                            "trade_date": report.get("trade_date"),
                            "account_id": account_id,
                            "automation_mode": mode,
                            "paper_position_id": item.get("paper_position_id"),
                            "symbol": item.get("symbol"),
                            "decision": item.get("decision"),
                            "sell_action": item.get("sell_action"),
                            "sell_ratio": item.get("sell_ratio"),
                            "sell_quantity": item.get("sell_quantity"),
                            "exit_reason": item.get("exit_reason"),
                            "review_required": item.get("review_required"),
                            "applied_rules": _json_text(item.get("applied_rules") or []),
                            "latest_price": item.get("latest_price"),
                            "avg_entry_price": item.get("avg_entry_price"),
                            "unrealized_pnl_pct": item.get("unrealized_pnl_pct"),
                            "realized_paper_pnl": item.get("realized_paper_pnl"),
                            "error_message": item.get("error_message"),
                            "created_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_DECISION_SQL, rows)
                    persisted["db_decision_logs"] = len(rows)

            if relation_exists("trade.us_sell_signal_log"):
                rows = []
                for item in report.get("decisions", []):
                    for index, rule in enumerate(item.get("applied_rules") or [], start=1):
                        rows.append(
                            {
                                "sell_signal_id": f"{item.get('sell_decision_id')}_{index}",
                                "trade_date": report.get("trade_date"),
                                "paper_position_id": item.get("paper_position_id"),
                                "symbol": item.get("symbol"),
                                "rule_name": rule.get("rule"),
                                "rule_result": rule.get("result"),
                                "metric_value": _json_text(rule.get("value")),
                                "threshold_value": _json_text(rule.get("threshold")),
                                "severity": "ERROR" if rule.get("action") in {"FULL_SELL", "REVIEW_REQUIRED"} else "INFO",
                                "detail": rule.get("reason"),
                                "created_at": datetime.now(timezone.utc),
                            }
                        )
                if rows:
                    conn.execute(INSERT_SIGNAL_SQL, rows)
                    persisted["db_signal_logs"] = len(rows)

            if relation_exists("trade.us_paper_sell_order"):
                rows = []
                for item in report.get("paper_sell_orders", []):
                    rows.append(
                        {
                            "paper_sell_order_id": item.get("paper_sell_order_id"),
                            "trade_date": report.get("trade_date"),
                            "paper_position_id": item.get("paper_position_id"),
                            "symbol": item.get("symbol"),
                            "side": item.get("side"),
                            "sell_action": item.get("sell_action"),
                            "sell_ratio": item.get("sell_ratio"),
                            "sell_quantity": item.get("sell_quantity"),
                            "sell_price_ref": item.get("sell_price"),
                            "sell_amount": item.get("sell_amount"),
                            "assumed_fill_status": item.get("assumed_fill_status"),
                            "exit_reason": item.get("exit_reason"),
                            "source_sell_decision_id": item.get("source_sell_decision_id"),
                            "created_at": item.get("created_at"),
                            "updated_at": item.get("updated_at"),
                        }
                    )
                if rows:
                    conn.execute(INSERT_PAPER_SELL_ORDER_SQL, rows)
                    persisted["db_paper_sell_orders"] = len(rows)

            if relation_exists("trade.us_paper_position_snapshot"):
                rows = []
                for position in report.get("positions", []):
                    rows.append(
                        {
                            "snapshot_id": f"{report.get('trade_date')}_{position.get('paper_position_id')}",
                            "snapshot_date": report.get("trade_date"),
                            "paper_position_id": position.get("paper_position_id"),
                            "symbol": position.get("symbol"),
                            "latest_price": position.get("latest_price"),
                            "remaining_quantity": position.get("remaining_quantity"),
                            "highest_price_since_entry": position.get("highest_price_since_entry"),
                            "unrealized_pnl": position.get("unrealized_pnl"),
                            "unrealized_pnl_pct": position.get("unrealized_pnl_pct"),
                            "holding_days": position.get("holding_days"),
                            "status": position.get("status"),
                            "data_quality_flags": _json_text(position.get("data_quality_flags") or []),
                            "created_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_POSITION_SNAPSHOT_SQL, rows)
                    persisted["db_position_snapshots"] = len(rows)
    except Exception:
        pass

    return persisted
