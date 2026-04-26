from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_TRADE_INTENTS_JSON = OUTPUT_DIR / "trade_intents.json"
DEFAULT_ORDER_REQUESTS_PREVIEW_JSON = OUTPUT_DIR / "order_requests_preview.json"
DEFAULT_ORDER_REQUESTS_EXECUTION_JSON = OUTPUT_DIR / "order_requests_execution.json"
DEFAULT_LIVE_HOLDINGS_CSV = ROOT / "data" / "live_account_holdings.csv"
DEFAULT_LIVE_BALANCE_SUMMARY_JSON = OUTPUT_DIR / "live_account_balance_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync live auto-trading artifacts into normalized research ledger tables.")
    parser.add_argument("--trade-intents-json", type=Path, default=DEFAULT_TRADE_INTENTS_JSON)
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_ORDER_REQUESTS_PREVIEW_JSON)
    parser.add_argument("--order-execution-json", type=Path, default=DEFAULT_ORDER_REQUESTS_EXECUTION_JSON)
    parser.add_argument("--live-holdings-csv", type=Path, default=DEFAULT_LIVE_HOLDINGS_CSV)
    parser.add_argument("--live-balance-summary-json", type=Path, default=DEFAULT_LIVE_BALANCE_SUMMARY_JSON)
    parser.add_argument("--skip-execution", action="store_true", help="Only sync decisions and order requests.")
    parser.add_argument("--skip-position-snapshot", action="store_true", help="Skip live account position snapshot sync.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), ensure_ascii=False, default=str, allow_nan=False)


def _json_sanitize(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    return value


def _none_if_blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _date(value: Any) -> Any:
    value = _none_if_blank(value)
    if value is None:
        return None
    return str(value)[:10]


def _ts(value: Any) -> Any:
    value = _none_if_blank(value)
    if value is None:
        return None
    text_value = str(value).strip()
    try:
        return datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text_value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _num(value: Any) -> Any:
    value = _none_if_blank(value)
    if value is None:
        return None
    try:
        numeric = float(value)
        return None if pd.isna(numeric) else numeric
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Any:
    value = _none_if_blank(value)
    if value is None:
        return None
    try:
        numeric = float(value)
        return None if pd.isna(numeric) else int(numeric)
    except (TypeError, ValueError):
        return None


def ensure_tables() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS research"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_trade_decision (
                    intent_id text PRIMARY KEY,
                    as_of_date date NOT NULL,
                    code varchar(10),
                    name text,
                    source_action text,
                    intent_type text NOT NULL,
                    target_weight numeric,
                    gate_status text,
                    reason text,
                    priority integer,
                    executable boolean DEFAULT false NOT NULL,
                    policy_version text,
                    score_formula_version text,
                    gate_version text,
                    portfolio_version text,
                    holdings_source text,
                    ranking_run_id bigint,
                    ranking_rank integer,
                    final_score numeric,
                    confidence_score numeric,
                    risk_penalty numeric,
                    ret_score numeric,
                    prob_score numeric,
                    qual_score numeric,
                    tech_score numeric,
                    liquidity_score numeric,
                    safety_score numeric,
                    dominant_theme text,
                    score_driver_1 text,
                    score_driver_2 text,
                    score_driver_3 text,
                    risk_factor_1 text,
                    risk_factor_2 text,
                    action_note text,
                    payload_json jsonb NOT NULL,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    updated_at timestamptz DEFAULT now() NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_order_request (
                    request_id text PRIMARY KEY,
                    intent_id text,
                    as_of_date date,
                    generated_at timestamptz,
                    gate_status text,
                    env_dv text,
                    code varchar(10) NOT NULL,
                    name text,
                    side text NOT NULL,
                    intent_type text,
                    ord_dvsn text,
                    reference_price numeric,
                    planned_qty numeric,
                    allowed_qty numeric,
                    final_request_qty numeric,
                    target_weight numeric,
                    priority integer,
                    reason text,
                    blocked_reason text,
                    expected_hold_reason text,
                    executable_now boolean DEFAULT false NOT NULL,
                    ranking_run_id bigint,
                    ranking_rank integer,
                    final_score numeric,
                    confidence_score numeric,
                    risk_penalty numeric,
                    ret_score numeric,
                    prob_score numeric,
                    qual_score numeric,
                    tech_score numeric,
                    liquidity_score numeric,
                    safety_score numeric,
                    dominant_theme text,
                    score_driver_1 text,
                    score_driver_2 text,
                    score_driver_3 text,
                    risk_factor_1 text,
                    risk_factor_2 text,
                    action_note text,
                    payload_json jsonb NOT NULL,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    updated_at timestamptz DEFAULT now() NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_order_execution (
                    execution_id bigserial PRIMARY KEY,
                    request_id text NOT NULL,
                    intent_id text,
                    as_of_date date,
                    executed_at timestamptz,
                    submitted_at timestamptz,
                    gate_status text,
                    env_dv text,
                    code varchar(10) NOT NULL,
                    name text,
                    side text NOT NULL,
                    intent_type text,
                    ord_dvsn text,
                    reference_price numeric,
                    final_request_qty numeric,
                    submission_status text NOT NULL,
                    skip_reason text,
                    broker_order_id text,
                    broker_org_order_id text,
                    raw_response_json jsonb,
                    payload_json jsonb NOT NULL,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    updated_at timestamptz DEFAULT now() NOT NULL,
                    UNIQUE (request_id, executed_at)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_order_fill (
                    fill_id bigserial PRIMARY KEY,
                    request_id text,
                    broker_order_id text,
                    broker_org_order_id text,
                    as_of_date date,
                    filled_at timestamptz,
                    code varchar(10) NOT NULL,
                    name text,
                    side text NOT NULL,
                    filled_qty numeric NOT NULL,
                    filled_price numeric NOT NULL,
                    filled_amount numeric,
                    fee numeric,
                    tax numeric,
                    fill_status text,
                    source text DEFAULT 'kis_fill_inquiry' NOT NULL,
                    raw_response_json jsonb,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    updated_at timestamptz DEFAULT now() NOT NULL,
                    UNIQUE (broker_order_id, code, side, filled_at, filled_qty, filled_price)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_position_snapshot (
                    snapshot_id bigserial PRIMARY KEY,
                    snapshot_at timestamptz NOT NULL,
                    snapshot_date date NOT NULL,
                    env_dv text,
                    account_masked text,
                    code varchar(10) NOT NULL,
                    name text,
                    qty numeric NOT NULL,
                    avg_price numeric,
                    current_price numeric,
                    eval_amount numeric,
                    pnl_amount numeric,
                    pnl_pct numeric,
                    weight numeric,
                    status text DEFAULT 'OPEN' NOT NULL,
                    cash_amount numeric,
                    total_assets numeric,
                    payload_json jsonb NOT NULL,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    UNIQUE (snapshot_at, code)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS research.live_trade_review (
                    review_id bigserial PRIMARY KEY,
                    intent_id text,
                    request_id text,
                    code varchar(10) NOT NULL,
                    review_date date NOT NULL DEFAULT CURRENT_DATE,
                    pre_tags text[],
                    post_tags text[],
                    outcome_label text,
                    review_note text,
                    next_action_note text,
                    reviewer text,
                    created_at timestamptz DEFAULT now() NOT NULL,
                    updated_at timestamptz DEFAULT now() NOT NULL
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_trade_decision_asof_code ON research.live_trade_decision(as_of_date DESC, code)"))
        conn.execute(text("ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS ranking_run_id bigint"))
        conn.execute(text("ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS ranking_rank integer"))
        conn.execute(text("ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS final_score numeric"))
        conn.execute(text("ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS confidence_score numeric"))
        for column_def in [
            "risk_penalty numeric",
            "ret_score numeric",
            "prob_score numeric",
            "qual_score numeric",
            "tech_score numeric",
            "liquidity_score numeric",
            "safety_score numeric",
            "dominant_theme text",
            "score_driver_1 text",
            "score_driver_2 text",
            "score_driver_3 text",
            "risk_factor_1 text",
            "risk_factor_2 text",
            "action_note text",
        ]:
            conn.execute(text(f"ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS {column_def}"))
        conn.execute(text("ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS ranking_run_id bigint"))
        conn.execute(text("ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS ranking_rank integer"))
        conn.execute(text("ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS final_score numeric"))
        conn.execute(text("ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS confidence_score numeric"))
        for column_def in [
            "risk_penalty numeric",
            "ret_score numeric",
            "prob_score numeric",
            "qual_score numeric",
            "tech_score numeric",
            "liquidity_score numeric",
            "safety_score numeric",
            "dominant_theme text",
            "score_driver_1 text",
            "score_driver_2 text",
            "score_driver_3 text",
            "risk_factor_1 text",
            "risk_factor_2 text",
            "action_note text",
        ]:
            conn.execute(text(f"ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS {column_def}"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_request_asof_code ON research.live_order_request(as_of_date DESC, code)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_request_intent ON research.live_order_request(intent_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_execution_asof_code ON research.live_order_execution(as_of_date DESC, code)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_execution_status ON research.live_order_execution(submission_status, executed_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_fill_code_time ON research.live_order_fill(code, filled_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_order_fill_request ON research.live_order_fill(request_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_position_snapshot_date_code ON research.live_position_snapshot(snapshot_date DESC, code)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_live_trade_review_code_date ON research.live_trade_review(code, review_date DESC)"))


def _table_exists(conn, qualified_name: str) -> bool:
    return bool(conn.execute(text("SELECT to_regclass(:name)"), {"name": qualified_name}).scalar())


def _load_ranking_history_lookup(conn, items: list[dict[str, Any]], payload_asof: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not items or not _table_exists(conn, "research.ranking_history"):
        return {}

    keys: set[tuple[str, str]] = set()
    for item in items:
        code = _none_if_blank(item.get("code"))
        as_of_date = _date(item.get("asof_date") or payload_asof)
        if code and as_of_date:
            keys.add((as_of_date, str(code).zfill(6)))
    if not keys:
        return {}

    values_clause = ", ".join(f"(:date_{idx}, :code_{idx})" for idx, _ in enumerate(keys))
    params: dict[str, Any] = {}
    for idx, (as_of_date, code) in enumerate(keys):
        params[f"date_{idx}"] = as_of_date
        params[f"code_{idx}"] = code

    rows = conn.execute(
        text(
            f"""
            WITH lookup(as_of_date, code) AS (
                VALUES {values_clause}
            )
            SELECT DISTINCT ON (rh.as_of_date, rh.code)
                rh.as_of_date,
                rh.code,
                rh.run_id AS ranking_run_id,
                rh.rank AS ranking_rank,
                rh.final_score,
                rh.ret_score,
                rh.prob_score,
                rh.qual_score,
                rh.tech_score,
                rh.risk_penalty
            FROM research.ranking_history rh
            JOIN lookup l
              ON rh.as_of_date = CAST(l.as_of_date AS date)
             AND rh.code = l.code
            ORDER BY rh.as_of_date, rh.code, rh.run_id DESC
            """
        ),
        params,
    ).mappings().all()

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        as_of_date = str(row["as_of_date"])[:10]
        code = str(row["code"]).zfill(6)
        out[(as_of_date, code)] = dict(row)
    return out


def _merge_ranking_history_context(item: dict[str, Any], lookup: dict[tuple[str, str], dict[str, Any]], payload_asof: Any) -> dict[str, Any]:
    code = _none_if_blank(item.get("code"))
    as_of_date = _date(item.get("asof_date") or payload_asof)
    if not code or not as_of_date:
        return item
    ranking = lookup.get((as_of_date, str(code).zfill(6)))
    if not ranking:
        return item
    merged = dict(item)
    merge_map = {
        "ranking_run_id": "ranking_run_id",
        "ranking_rank": "ranking_rank",
        "final_score": "final_score",
        "ret_score": "ret_score",
        "prob_score": "prob_score",
        "qual_score": "qual_score",
        "tech_score": "tech_score",
        "risk_penalty": "risk_penalty",
    }
    for target, source in merge_map.items():
        if _none_if_blank(merged.get(target)) is None and ranking.get(source) is not None:
            merged[target] = ranking.get(source)
    return merged


def sync_trade_decisions(payload: dict[str, Any]) -> int:
    items = payload.get("intents") or []
    if not isinstance(items, list):
        return 0
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        ranking_lookup = _load_ranking_history_lookup(conn, items, payload.get("asof_date"))
        for item in items:
            item = _merge_ranking_history_context(item, ranking_lookup, payload.get("asof_date"))
            intent_id = _none_if_blank(item.get("intent_id"))
            as_of_date = _date(item.get("asof_date") or payload.get("asof_date"))
            intent_type = _none_if_blank(item.get("intent_type"))
            if not intent_id or not as_of_date or not intent_type:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO research.live_trade_decision (
                        intent_id, as_of_date, code, name, source_action, intent_type,
                        target_weight, gate_status, reason, priority, executable,
                        policy_version, score_formula_version, gate_version, portfolio_version,
                        holdings_source, ranking_run_id, ranking_rank, final_score,
                        confidence_score, risk_penalty, ret_score, prob_score, qual_score,
                        tech_score, liquidity_score, safety_score, dominant_theme,
                        score_driver_1, score_driver_2, score_driver_3,
                        risk_factor_1, risk_factor_2, action_note, payload_json, updated_at
                    )
                    VALUES (
                        :intent_id, :as_of_date, :code, :name, :source_action, :intent_type,
                        :target_weight, :gate_status, :reason, :priority, :executable,
                        :policy_version, :score_formula_version, :gate_version, :portfolio_version,
                        :holdings_source, :ranking_run_id, :ranking_rank, :final_score,
                        :confidence_score, :risk_penalty, :ret_score, :prob_score, :qual_score,
                        :tech_score, :liquidity_score, :safety_score, :dominant_theme,
                        :score_driver_1, :score_driver_2, :score_driver_3,
                        :risk_factor_1, :risk_factor_2, :action_note, CAST(:payload_json AS jsonb), now()
                    )
                    ON CONFLICT (intent_id) DO UPDATE SET
                        as_of_date = EXCLUDED.as_of_date,
                        code = EXCLUDED.code,
                        name = EXCLUDED.name,
                        source_action = EXCLUDED.source_action,
                        intent_type = EXCLUDED.intent_type,
                        target_weight = EXCLUDED.target_weight,
                        gate_status = EXCLUDED.gate_status,
                        reason = EXCLUDED.reason,
                        priority = EXCLUDED.priority,
                        executable = EXCLUDED.executable,
                        policy_version = EXCLUDED.policy_version,
                        score_formula_version = EXCLUDED.score_formula_version,
                        gate_version = EXCLUDED.gate_version,
                        portfolio_version = EXCLUDED.portfolio_version,
                        holdings_source = EXCLUDED.holdings_source,
                        ranking_run_id = EXCLUDED.ranking_run_id,
                        ranking_rank = EXCLUDED.ranking_rank,
                        final_score = EXCLUDED.final_score,
                        confidence_score = EXCLUDED.confidence_score,
                        risk_penalty = EXCLUDED.risk_penalty,
                        ret_score = EXCLUDED.ret_score,
                        prob_score = EXCLUDED.prob_score,
                        qual_score = EXCLUDED.qual_score,
                        tech_score = EXCLUDED.tech_score,
                        liquidity_score = EXCLUDED.liquidity_score,
                        safety_score = EXCLUDED.safety_score,
                        dominant_theme = EXCLUDED.dominant_theme,
                        score_driver_1 = EXCLUDED.score_driver_1,
                        score_driver_2 = EXCLUDED.score_driver_2,
                        score_driver_3 = EXCLUDED.score_driver_3,
                        risk_factor_1 = EXCLUDED.risk_factor_1,
                        risk_factor_2 = EXCLUDED.risk_factor_2,
                        action_note = EXCLUDED.action_note,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = now()
                    """
                ),
                {
                    "intent_id": intent_id,
                    "as_of_date": as_of_date,
                    "code": _none_if_blank(item.get("code")),
                    "name": _none_if_blank(item.get("name")),
                    "source_action": _none_if_blank(item.get("source_action")),
                    "intent_type": intent_type,
                    "target_weight": _num(item.get("target_weight")),
                    "gate_status": _none_if_blank(item.get("gate_status") or payload.get("gate_status")),
                    "reason": _none_if_blank(item.get("reason")),
                    "priority": _int(item.get("priority")),
                    "executable": bool(item.get("executable", False)),
                    "policy_version": _none_if_blank(payload.get("policy_version")),
                    "score_formula_version": _none_if_blank(payload.get("score_formula_version")),
                    "gate_version": _none_if_blank(payload.get("gate_version")),
                    "portfolio_version": _none_if_blank(payload.get("portfolio_version")),
                    "holdings_source": _none_if_blank(payload.get("holdings_source")),
                    "ranking_run_id": _int(item.get("ranking_run_id") or item.get("run_id")),
                    "ranking_rank": _int(item.get("ranking_rank") or item.get("rank") or item.get("buy_rank") or item.get("rank_final")),
                    "final_score": _num(item.get("final_score")),
                    "confidence_score": _num(item.get("confidence_score")),
                    "risk_penalty": _num(item.get("risk_penalty")),
                    "ret_score": _num(item.get("ret_score")),
                    "prob_score": _num(item.get("prob_score")),
                    "qual_score": _num(item.get("qual_score")),
                    "tech_score": _num(item.get("tech_score")),
                    "liquidity_score": _num(item.get("liquidity_score")),
                    "safety_score": _num(item.get("safety_score")),
                    "dominant_theme": _none_if_blank(item.get("dominant_theme")),
                    "score_driver_1": _none_if_blank(item.get("score_driver_1")),
                    "score_driver_2": _none_if_blank(item.get("score_driver_2")),
                    "score_driver_3": _none_if_blank(item.get("score_driver_3")),
                    "risk_factor_1": _none_if_blank(item.get("risk_factor_1")),
                    "risk_factor_2": _none_if_blank(item.get("risk_factor_2")),
                    "action_note": _none_if_blank(item.get("action_note")),
                    "payload_json": _json_dumps(item),
                },
            )
            count += 1
    return count


def sync_order_requests(payload: dict[str, Any]) -> int:
    items = payload.get("items") or []
    if not isinstance(items, list):
        return 0
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        ranking_lookup = _load_ranking_history_lookup(conn, items, payload.get("asof_date"))
        for item in items:
            item = _merge_ranking_history_context(item, ranking_lookup, payload.get("asof_date"))
            request_id = _none_if_blank(item.get("request_id"))
            code = _none_if_blank(item.get("code"))
            side = _none_if_blank(item.get("side"))
            if not request_id or not code or not side:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO research.live_order_request (
                        request_id, intent_id, as_of_date, generated_at, gate_status, env_dv,
                        code, name, side, intent_type, ord_dvsn, reference_price,
                        planned_qty, allowed_qty, final_request_qty, target_weight,
                        priority, reason, blocked_reason, expected_hold_reason,
                        executable_now, ranking_run_id, ranking_rank, final_score,
                        confidence_score, risk_penalty, ret_score, prob_score, qual_score,
                        tech_score, liquidity_score, safety_score, dominant_theme,
                        score_driver_1, score_driver_2, score_driver_3,
                        risk_factor_1, risk_factor_2, action_note, payload_json, updated_at
                    )
                    VALUES (
                        :request_id, :intent_id, :as_of_date, :generated_at, :gate_status, :env_dv,
                        :code, :name, :side, :intent_type, :ord_dvsn, :reference_price,
                        :planned_qty, :allowed_qty, :final_request_qty, :target_weight,
                        :priority, :reason, :blocked_reason, :expected_hold_reason,
                        :executable_now, :ranking_run_id, :ranking_rank, :final_score,
                        :confidence_score, :risk_penalty, :ret_score, :prob_score, :qual_score,
                        :tech_score, :liquidity_score, :safety_score, :dominant_theme,
                        :score_driver_1, :score_driver_2, :score_driver_3,
                        :risk_factor_1, :risk_factor_2, :action_note, CAST(:payload_json AS jsonb), now()
                    )
                    ON CONFLICT (request_id) DO UPDATE SET
                        intent_id = EXCLUDED.intent_id,
                        as_of_date = EXCLUDED.as_of_date,
                        generated_at = EXCLUDED.generated_at,
                        gate_status = EXCLUDED.gate_status,
                        env_dv = EXCLUDED.env_dv,
                        code = EXCLUDED.code,
                        name = EXCLUDED.name,
                        side = EXCLUDED.side,
                        intent_type = EXCLUDED.intent_type,
                        ord_dvsn = EXCLUDED.ord_dvsn,
                        reference_price = EXCLUDED.reference_price,
                        planned_qty = EXCLUDED.planned_qty,
                        allowed_qty = EXCLUDED.allowed_qty,
                        final_request_qty = EXCLUDED.final_request_qty,
                        target_weight = EXCLUDED.target_weight,
                        priority = EXCLUDED.priority,
                        reason = EXCLUDED.reason,
                        blocked_reason = EXCLUDED.blocked_reason,
                        expected_hold_reason = EXCLUDED.expected_hold_reason,
                        executable_now = EXCLUDED.executable_now,
                        ranking_run_id = EXCLUDED.ranking_run_id,
                        ranking_rank = EXCLUDED.ranking_rank,
                        final_score = EXCLUDED.final_score,
                        confidence_score = EXCLUDED.confidence_score,
                        risk_penalty = EXCLUDED.risk_penalty,
                        ret_score = EXCLUDED.ret_score,
                        prob_score = EXCLUDED.prob_score,
                        qual_score = EXCLUDED.qual_score,
                        tech_score = EXCLUDED.tech_score,
                        liquidity_score = EXCLUDED.liquidity_score,
                        safety_score = EXCLUDED.safety_score,
                        dominant_theme = EXCLUDED.dominant_theme,
                        score_driver_1 = EXCLUDED.score_driver_1,
                        score_driver_2 = EXCLUDED.score_driver_2,
                        score_driver_3 = EXCLUDED.score_driver_3,
                        risk_factor_1 = EXCLUDED.risk_factor_1,
                        risk_factor_2 = EXCLUDED.risk_factor_2,
                        action_note = EXCLUDED.action_note,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = now()
                    """
                ),
                {
                    "request_id": request_id,
                    "intent_id": _none_if_blank(item.get("intent_id")),
                    "as_of_date": _date(payload.get("asof_date")),
                    "generated_at": _ts(payload.get("generated_at")),
                    "gate_status": _none_if_blank(payload.get("gate_status")),
                    "env_dv": _none_if_blank(payload.get("env_dv")),
                    "code": str(code).zfill(6),
                    "name": _none_if_blank(item.get("name")),
                    "side": str(side).upper(),
                    "intent_type": _none_if_blank(item.get("intent_type")),
                    "ord_dvsn": _none_if_blank(item.get("ord_dvsn")),
                    "reference_price": _num(item.get("reference_price")),
                    "planned_qty": _num(item.get("planned_qty")),
                    "allowed_qty": _num(item.get("allowed_qty")),
                    "final_request_qty": _num(item.get("final_request_qty")),
                    "target_weight": _num(item.get("target_weight")),
                    "priority": _int(item.get("priority")),
                    "reason": _none_if_blank(item.get("reason")),
                    "blocked_reason": _none_if_blank(item.get("blocked_reason")),
                    "expected_hold_reason": _none_if_blank(item.get("expected_hold_reason")),
                    "executable_now": bool(item.get("executable_now", False)),
                    "ranking_run_id": _int(item.get("ranking_run_id") or item.get("run_id")),
                    "ranking_rank": _int(item.get("ranking_rank") or item.get("rank") or item.get("buy_rank") or item.get("rank_final")),
                    "final_score": _num(item.get("final_score")),
                    "confidence_score": _num(item.get("confidence_score")),
                    "risk_penalty": _num(item.get("risk_penalty")),
                    "ret_score": _num(item.get("ret_score")),
                    "prob_score": _num(item.get("prob_score")),
                    "qual_score": _num(item.get("qual_score")),
                    "tech_score": _num(item.get("tech_score")),
                    "liquidity_score": _num(item.get("liquidity_score")),
                    "safety_score": _num(item.get("safety_score")),
                    "dominant_theme": _none_if_blank(item.get("dominant_theme")),
                    "score_driver_1": _none_if_blank(item.get("score_driver_1")),
                    "score_driver_2": _none_if_blank(item.get("score_driver_2")),
                    "score_driver_3": _none_if_blank(item.get("score_driver_3")),
                    "risk_factor_1": _none_if_blank(item.get("risk_factor_1")),
                    "risk_factor_2": _none_if_blank(item.get("risk_factor_2")),
                    "action_note": _none_if_blank(item.get("action_note")),
                    "payload_json": _json_dumps(item),
                },
            )
            count += 1
    return count


def sync_order_executions(payload: dict[str, Any]) -> int:
    items = payload.get("items") or []
    if not isinstance(items, list):
        return 0
    engine = get_engine()
    executed_at = _ts(payload.get("executed_at"))
    count = 0
    with engine.begin() as conn:
        for item in items:
            request_id = _none_if_blank(item.get("request_id"))
            code = _none_if_blank(item.get("code"))
            side = _none_if_blank(item.get("side"))
            status = _none_if_blank(item.get("submission_status"))
            if not request_id or not code or not side or not status:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO research.live_order_execution (
                        request_id, intent_id, as_of_date, executed_at, submitted_at,
                        gate_status, env_dv, code, name, side, intent_type, ord_dvsn,
                        reference_price, final_request_qty, submission_status, skip_reason,
                        broker_order_id, broker_org_order_id, raw_response_json, payload_json,
                        updated_at
                    )
                    VALUES (
                        :request_id, :intent_id, :as_of_date, :executed_at, :submitted_at,
                        :gate_status, :env_dv, :code, :name, :side, :intent_type, :ord_dvsn,
                        :reference_price, :final_request_qty, :submission_status, :skip_reason,
                        :broker_order_id, :broker_org_order_id, CAST(:raw_response_json AS jsonb),
                        CAST(:payload_json AS jsonb), now()
                    )
                    ON CONFLICT (request_id, executed_at) DO UPDATE SET
                        intent_id = EXCLUDED.intent_id,
                        as_of_date = EXCLUDED.as_of_date,
                        submitted_at = EXCLUDED.submitted_at,
                        gate_status = EXCLUDED.gate_status,
                        env_dv = EXCLUDED.env_dv,
                        code = EXCLUDED.code,
                        name = EXCLUDED.name,
                        side = EXCLUDED.side,
                        intent_type = EXCLUDED.intent_type,
                        ord_dvsn = EXCLUDED.ord_dvsn,
                        reference_price = EXCLUDED.reference_price,
                        final_request_qty = EXCLUDED.final_request_qty,
                        submission_status = EXCLUDED.submission_status,
                        skip_reason = EXCLUDED.skip_reason,
                        broker_order_id = EXCLUDED.broker_order_id,
                        broker_org_order_id = EXCLUDED.broker_org_order_id,
                        raw_response_json = EXCLUDED.raw_response_json,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = now()
                    """
                ),
                {
                    "request_id": request_id,
                    "intent_id": _none_if_blank(item.get("intent_id")),
                    "as_of_date": _date(payload.get("asof_date")),
                    "executed_at": executed_at,
                    "submitted_at": _ts(item.get("submitted_at")),
                    "gate_status": _none_if_blank(payload.get("gate_status")),
                    "env_dv": _none_if_blank(payload.get("env_dv")),
                    "code": str(code).zfill(6),
                    "name": _none_if_blank(item.get("name")),
                    "side": str(side).upper(),
                    "intent_type": _none_if_blank(item.get("intent_type")),
                    "ord_dvsn": _none_if_blank(item.get("ord_dvsn")),
                    "reference_price": _num(item.get("reference_price")),
                    "final_request_qty": _num(item.get("final_request_qty")),
                    "submission_status": status,
                    "skip_reason": _none_if_blank(item.get("skip_reason")),
                    "broker_order_id": _none_if_blank(item.get("broker_order_id")),
                    "broker_org_order_id": _none_if_blank(item.get("broker_org_order_id")),
                    "raw_response_json": _json_dumps(item.get("raw_response")),
                    "payload_json": _json_dumps(item),
                },
            )
            count += 1
    return count


def sync_position_snapshot(*, holdings_csv: Path, summary_payload: dict[str, Any]) -> int:
    resolved = _resolve(holdings_csv)
    if not resolved.exists():
        return 0
    holdings = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if holdings.empty:
        return 0

    snapshot_at = _ts(summary_payload.get("generated_at")) or datetime.now()
    snapshot_date = snapshot_at.date().isoformat()
    derived = summary_payload.get("derived_metrics") if isinstance(summary_payload.get("derived_metrics"), dict) else {}
    cash_summary = summary_payload.get("cash_summary") if isinstance(summary_payload.get("cash_summary"), dict) else {}
    cash_amount = _num(derived.get("cash_amount") or cash_summary.get("dnca_tot_amt"))
    total_assets = _num(derived.get("total_assets") or cash_summary.get("tot_evlu_amt"))
    env_dv = _none_if_blank(summary_payload.get("env_dv"))
    account_masked = _none_if_blank(summary_payload.get("cano_masked"))

    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        for row in holdings.where(pd.notna(holdings), None).to_dict(orient="records"):
            code = _none_if_blank(row.get("code"))
            qty = _num(row.get("qty"))
            if not code or qty is None:
                continue
            payload = dict(row)
            payload["summary"] = summary_payload
            conn.execute(
                text(
                    """
                    INSERT INTO research.live_position_snapshot (
                        snapshot_at, snapshot_date, env_dv, account_masked, code, name,
                        qty, avg_price, current_price, eval_amount, pnl_amount, pnl_pct,
                        weight, status, cash_amount, total_assets, payload_json
                    )
                    VALUES (
                        :snapshot_at, :snapshot_date, :env_dv, :account_masked, :code, :name,
                        :qty, :avg_price, :current_price, :eval_amount, :pnl_amount, :pnl_pct,
                        :weight, :status, :cash_amount, :total_assets, CAST(:payload_json AS jsonb)
                    )
                    ON CONFLICT (snapshot_at, code) DO UPDATE SET
                        env_dv = EXCLUDED.env_dv,
                        account_masked = EXCLUDED.account_masked,
                        name = EXCLUDED.name,
                        qty = EXCLUDED.qty,
                        avg_price = EXCLUDED.avg_price,
                        current_price = EXCLUDED.current_price,
                        eval_amount = EXCLUDED.eval_amount,
                        pnl_amount = EXCLUDED.pnl_amount,
                        pnl_pct = EXCLUDED.pnl_pct,
                        weight = EXCLUDED.weight,
                        status = EXCLUDED.status,
                        cash_amount = EXCLUDED.cash_amount,
                        total_assets = EXCLUDED.total_assets,
                        payload_json = EXCLUDED.payload_json
                    """
                ),
                {
                    "snapshot_at": snapshot_at,
                    "snapshot_date": snapshot_date,
                    "env_dv": env_dv,
                    "account_masked": account_masked,
                    "code": str(code).zfill(6),
                    "name": _none_if_blank(row.get("name")),
                    "qty": qty,
                    "avg_price": _num(row.get("avg_price")),
                    "current_price": _num(row.get("current_price")),
                    "eval_amount": _num(row.get("eval_amount")),
                    "pnl_amount": _num(row.get("pnl_amount")),
                    "pnl_pct": _num(row.get("pnl_pct")),
                    "weight": _num(row.get("weight")),
                    "status": _none_if_blank(row.get("status")) or "OPEN",
                    "cash_amount": cash_amount,
                    "total_assets": total_assets,
                    "payload_json": _json_dumps(payload),
                },
            )
            count += 1
    return count


def sync_live_trade_ledger(
    *,
    intents_payload: dict[str, Any] | None = None,
    preview_payload: dict[str, Any] | None = None,
    execution_payload: dict[str, Any] | None = None,
    holdings_csv: Path | None = None,
    balance_summary_payload: dict[str, Any] | None = None,
) -> dict[str, int]:
    ensure_tables()
    return {
        "decisions": sync_trade_decisions(intents_payload or {}),
        "requests": sync_order_requests(preview_payload or {}),
        "executions": sync_order_executions(execution_payload or {}),
        "position_snapshots": sync_position_snapshot(
            holdings_csv=holdings_csv or DEFAULT_LIVE_HOLDINGS_CSV,
            summary_payload=balance_summary_payload or {},
        ) if holdings_csv is not None or balance_summary_payload is not None else 0,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = sync_live_trade_ledger(
        intents_payload=_read_json(args.trade_intents_json),
        preview_payload=_read_json(args.order_preview_json),
        execution_payload={} if args.skip_execution else _read_json(args.order_execution_json),
        holdings_csv=None if args.skip_position_snapshot else args.live_holdings_csv,
        balance_summary_payload={} if args.skip_position_snapshot else _read_json(args.live_balance_summary_json),
    )
    logging.info("Synced live trade ledger: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
