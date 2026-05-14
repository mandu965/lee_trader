from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import text

from python.us.buy_automation.config import BuyAutomationConfig
from python.us.us_db import get_us_engine, relation_exists


INSERT_CANDIDATE_SQL = text(
    """
    INSERT INTO trade.us_buy_candidate_log (
        candidate_id,
        trade_date,
        account_id,
        automation_mode,
        ranking_source,
        symbol,
        company_name,
        sector,
        rank_no,
        recommend_grade,
        total_score,
        score_detail_json,
        price_ref,
        candidate_amount_usd,
        candidate_status,
        filter_stage,
        filter_reason_code,
        filter_reason_detail,
        created_at
    ) VALUES (
        :candidate_id,
        :trade_date,
        :account_id,
        :automation_mode,
        :ranking_source,
        :symbol,
        :company_name,
        :sector,
        :rank_no,
        :recommend_grade,
        :total_score,
        CAST(:score_detail_json AS JSONB),
        :price_ref,
        :candidate_amount_usd,
        :candidate_status,
        :filter_stage,
        :filter_reason_code,
        :filter_reason_detail,
        :created_at
    )
    ON CONFLICT (trade_date, automation_mode, symbol, filter_stage)
    DO UPDATE SET
        total_score = EXCLUDED.total_score,
        price_ref = EXCLUDED.price_ref,
        candidate_amount_usd = EXCLUDED.candidate_amount_usd,
        candidate_status = EXCLUDED.candidate_status,
        filter_reason_code = EXCLUDED.filter_reason_code,
        filter_reason_detail = EXCLUDED.filter_reason_detail
    """
)

INSERT_DECISION_SQL = text(
    """
    INSERT INTO trade.us_buy_decision_log (
        decision_id,
        trade_date,
        account_id,
        automation_mode,
        symbol,
        candidate_id,
        decision,
        severity,
        decision_reason_code,
        decision_reason_detail,
        rule_tags,
        block_reasons,
        rank_no,
        recommend_grade,
        total_score,
        price_ref,
        planned_order_amount_usd,
        cooldown_until,
        conflict_checked,
        conflict_blocked,
        conflict_reasons,
        related_position_id,
        related_sell_signal,
        requires_manual_review,
        report_group,
        created_at
    ) VALUES (
        :decision_id,
        :trade_date,
        :account_id,
        :automation_mode,
        :symbol,
        :candidate_id,
        :decision,
        :severity,
        :decision_reason_code,
        :decision_reason_detail,
        CAST(:rule_tags AS JSONB),
        CAST(:block_reasons AS JSONB),
        :rank_no,
        :recommend_grade,
        :total_score,
        :price_ref,
        :planned_order_amount_usd,
        :cooldown_until,
        :conflict_checked,
        :conflict_blocked,
        CAST(:conflict_reasons AS JSONB),
        :related_position_id,
        CAST(:related_sell_signal AS JSONB),
        :requires_manual_review,
        :report_group,
        :created_at
    )
    ON CONFLICT (trade_date, automation_mode, symbol)
    DO UPDATE SET
        decision = EXCLUDED.decision,
        severity = EXCLUDED.severity,
        decision_reason_code = EXCLUDED.decision_reason_code,
        decision_reason_detail = EXCLUDED.decision_reason_detail,
        rule_tags = EXCLUDED.rule_tags,
        block_reasons = EXCLUDED.block_reasons,
        planned_order_amount_usd = EXCLUDED.planned_order_amount_usd,
        cooldown_until = EXCLUDED.cooldown_until,
        conflict_checked = EXCLUDED.conflict_checked,
        conflict_blocked = EXCLUDED.conflict_blocked,
        conflict_reasons = EXCLUDED.conflict_reasons,
        related_position_id = EXCLUDED.related_position_id,
        related_sell_signal = EXCLUDED.related_sell_signal,
        report_group = EXCLUDED.report_group
    """
)

INSERT_GUARD_SQL = text(
    """
    INSERT INTO trade.us_risk_guard_log (
        guard_log_id,
        trade_date,
        account_id,
        automation_mode,
        guard_scope,
        guard_name,
        guard_status,
        severity,
        metric_value,
        threshold_value,
        reason_code,
        reason_detail,
        raw_payload,
        created_at
    ) VALUES (
        :guard_log_id,
        :trade_date,
        :account_id,
        :automation_mode,
        :guard_scope,
        :guard_name,
        :guard_status,
        :severity,
        :metric_value,
        :threshold_value,
        :reason_code,
        :reason_detail,
        CAST(:raw_payload AS JSONB),
        :created_at
    )
    ON CONFLICT (trade_date, automation_mode, guard_scope, guard_name, account_id)
    DO UPDATE SET
        guard_status = EXCLUDED.guard_status,
        severity = EXCLUDED.severity,
        metric_value = EXCLUDED.metric_value,
        threshold_value = EXCLUDED.threshold_value,
        reason_code = EXCLUDED.reason_code,
        reason_detail = EXCLUDED.reason_detail,
        raw_payload = EXCLUDED.raw_payload
    """
)

INSERT_PAPER_ORDER_SQL = text(
    """
    INSERT INTO trade.us_paper_order (
        paper_order_id,
        trade_date,
        account_id,
        automation_mode,
        symbol,
        side,
        paper_order_qty,
        paper_order_price,
        paper_order_amount,
        assumed_fill_price,
        assumed_fill_status,
        source_decision_id,
        created_at,
        updated_at
    ) VALUES (
        :paper_order_id,
        :trade_date,
        :account_id,
        :automation_mode,
        :symbol,
        :side,
        :paper_order_qty,
        :paper_order_price,
        :paper_order_amount,
        :assumed_fill_price,
        :assumed_fill_status,
        :source_decision_id,
        :created_at,
        :updated_at
    )
    ON CONFLICT (trade_date, automation_mode, symbol, side)
    DO UPDATE SET
        paper_order_qty = EXCLUDED.paper_order_qty,
        paper_order_price = EXCLUDED.paper_order_price,
        paper_order_amount = EXCLUDED.paper_order_amount,
        assumed_fill_price = EXCLUDED.assumed_fill_price,
        assumed_fill_status = EXCLUDED.assumed_fill_status,
        source_decision_id = EXCLUDED.source_decision_id,
        updated_at = EXCLUDED.updated_at
    """
)


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")
    return path


def persist_buy_automation_logs(
    *,
    cfg: BuyAutomationConfig,
    report: dict[str, object],
    account_id: str,
) -> dict[str, object]:
    trade_date = str(report.get("trade_date") or "unknown")
    mode = str(report.get("mode") or cfg.mode).upper()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_name = f"buy_automation_{trade_date}_{mode}_{timestamp}"
    json_path = _write_json(cfg.output_dir / f"{base_name}.json", report)

    persisted = {
        "json_path": str(json_path),
        "db_candidate_logs": 0,
        "db_decision_logs": 0,
        "db_guard_logs": 0,
        "db_paper_orders": 0,
    }

    try:
        engine = get_us_engine()
        with engine.begin() as conn:
            if relation_exists("trade.us_buy_candidate_log"):
                rows = []
                for item in report.get("candidates", []):
                    rows.append(
                        {
                            "candidate_id": item["candidate_id"],
                            "trade_date": report.get("trade_date"),
                            "account_id": account_id,
                            "automation_mode": mode,
                            "ranking_source": report.get("ranking_source"),
                            "symbol": item.get("symbol"),
                            "company_name": item.get("company_name"),
                            "sector": item.get("sector"),
                            "rank_no": item.get("rank"),
                            "recommend_grade": item.get("recommend_grade"),
                            "total_score": item.get("score"),
                            "score_detail_json": _json_text(item.get("score_detail_json") or {}),
                            "price_ref": item.get("reference_price"),
                            "candidate_amount_usd": item.get("allocated_amount_usd"),
                            "candidate_status": "ALLOWED" if item.get("allowed") else "BLOCKED",
                            "filter_stage": "FINAL",
                            "filter_reason_code": (item.get("block_reasons") or [None])[0],
                            "filter_reason_detail": ", ".join(item.get("block_reasons") or []),
                            "created_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_CANDIDATE_SQL, rows)
                    persisted["db_candidate_logs"] = len(rows)

            if relation_exists("trade.us_buy_decision_log"):
                rows = []
                for item in report.get("candidates", []):
                    rows.append(
                        {
                            "decision_id": item["decision_id"],
                            "trade_date": report.get("trade_date"),
                            "account_id": account_id,
                            "automation_mode": mode,
                            "symbol": item.get("symbol"),
                            "candidate_id": item["candidate_id"],
                            "decision": "ALLOW" if item.get("allowed") else "BLOCK",
                            "severity": item.get("severity"),
                            "decision_reason_code": (item.get("block_reasons") or ["ALLOW"])[0],
                            "decision_reason_detail": ", ".join(item.get("block_reasons") or []) or "allowed",
                            "rule_tags": _json_text(item.get("applied_rules") or []),
                            "block_reasons": _json_text(item.get("block_reasons") or []),
                            "rank_no": item.get("rank"),
                            "recommend_grade": item.get("recommend_grade"),
                            "total_score": item.get("score"),
                            "price_ref": item.get("reference_price"),
                            "planned_order_amount_usd": item.get("allocated_amount_usd"),
                            "cooldown_until": item.get("cooldown_until"),
                            "conflict_checked": item.get("conflict_checked", False),
                            "conflict_blocked": item.get("conflict_blocked", False),
                            "conflict_reasons": _json_text(item.get("conflict_reasons") or []),
                            "related_position_id": item.get("related_position_id"),
                            "related_sell_signal": _json_text(item.get("related_sell_signal") or {}),
                            "requires_manual_review": True,
                            "report_group": "BUY_AUTOMATION",
                            "created_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_DECISION_SQL, rows)
                    persisted["db_decision_logs"] = len(rows)

            if relation_exists("trade.us_risk_guard_log"):
                rows = []
                for item in report.get("candidates", []):
                    rows.append(
                        {
                            "guard_log_id": item["guard_log_id"],
                            "trade_date": report.get("trade_date"),
                            "account_id": account_id,
                            "automation_mode": mode,
                            "guard_scope": "SYMBOL",
                            "guard_name": item.get("symbol"),
                            "guard_status": "PASS" if item.get("allowed") else "BLOCK",
                            "severity": item.get("severity"),
                            "metric_value": item.get("score"),
                            "threshold_value": report.get("config_snapshot", {}).get("min_score"),
                            "reason_code": (item.get("block_reasons") or [None])[0],
                            "reason_detail": ", ".join(item.get("block_reasons") or []),
                            "raw_payload": _json_text(item.get("applied_rules") or []),
                            "created_at": datetime.now(timezone.utc),
                        }
                    )
                if rows:
                    conn.execute(INSERT_GUARD_SQL, rows)
                    persisted["db_guard_logs"] = len(rows)

            if relation_exists("trade.us_paper_order"):
                rows = []
                for item in report.get("paper_orders", []):
                    rows.append(
                        {
                            "paper_order_id": item.get("paper_order_id"),
                            "trade_date": report.get("trade_date"),
                            "account_id": account_id,
                            "automation_mode": mode,
                            "symbol": item.get("symbol"),
                            "side": item.get("side"),
                            "paper_order_qty": item.get("paper_order_qty"),
                            "paper_order_price": item.get("paper_order_price"),
                            "paper_order_amount": item.get("paper_order_amount"),
                            "assumed_fill_price": item.get("assumed_fill_price"),
                            "assumed_fill_status": item.get("assumed_fill_status"),
                            "source_decision_id": item.get("source_decision_id"),
                            "created_at": item.get("created_at"),
                            "updated_at": item.get("updated_at"),
                        }
                    )
                if rows:
                    conn.execute(INSERT_PAPER_ORDER_SQL, rows)
                    persisted["db_paper_orders"] = len(rows)
    except Exception:
        pass

    return persisted
