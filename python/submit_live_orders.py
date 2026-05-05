from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common_live_risk_guard import evaluate_common_buy_guard
from kis_client import KISClient
from kis_live_account import (
    compute_market_order_preview_qty,
    inquire_balance,
    inquire_psbl_order,
    order_cash,
    resolve_account_env,
    summarize_cash,
)
from sync_live_trade_ledger import sync_live_trade_ledger

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SYNC_WEB_DISPLAY_SCRIPT = ROOT / "python" / "sync_web_display_data.py"

INPUT_TRADE_INTENTS = OUTPUT_DIR / "trade_intents.json"
INPUT_LIVE_HOLDINGS = DATA_DIR / "live_account_holdings.csv"
INPUT_RANKING = DATA_DIR / "ranking_final.csv"
INPUT_PRICE_FALLBACK = DATA_DIR / "ranking_final.csv"
OUT_JSON = OUTPUT_DIR / "order_requests_preview.json"
OUT_MD = OUTPUT_DIR / "order_requests_preview.md"
OUT_EXEC_JSON = OUTPUT_DIR / "order_requests_execution.json"
OUT_EXEC_MD = OUTPUT_DIR / "order_requests_execution.md"
BUY_APPROVAL_JSON = OUTPUT_DIR / "order_buy_approvals.json"


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert executable trade intents into guarded order request submissions.")
    parser.add_argument("--trade-intents-json", type=Path, default=INPUT_TRADE_INTENTS)
    parser.add_argument("--live-holdings-csv", type=Path, default=INPUT_LIVE_HOLDINGS)
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--price-fallback-csv", type=Path, default=INPUT_PRICE_FALLBACK)
    parser.add_argument("--ord-dvsn", default="01", help="01 market, 00 limit")
    parser.add_argument("--execute", action="store_true", help="Actually submit guarded live orders.")
    parser.add_argument("--confirm-text", default="", help="Must be LIVE_ORDER to execute.")
    parser.add_argument("--allow-buy", action="store_true", help="Allow BUY submissions. Without this, only SELL/TRIM/EXIT are submitted.")
    parser.add_argument("--force-resubmit", action="store_true", help="Ignore successful request ids from the last execution artifact.")
    parser.add_argument("--approval-json", type=Path, default=BUY_APPROVAL_JSON, help="BUY approval request id list JSON.")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-exec-json", type=Path, default=OUT_EXEC_JSON)
    parser.add_argument("--out-exec-md", type=Path, default=OUT_EXEC_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(v: object, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{digits}f}"


def _safe_read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _json_sanitize(value: object) -> object:
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


def _num_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


def _int_or_none(value: object) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else None


def load_live_holdings(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    for col in ["qty", "avg_price", "current_price", "weight"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    return work


def load_ranking(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    for col in ["close", "buy_rank", "rank_final", "final_score", "confidence_score"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    if "buy_rank" not in work.columns or work["buy_rank"].isna().all():
        work["buy_rank"] = pd.to_numeric(work.get("rank_final"), errors="coerce")
    if work["buy_rank"].isna().all():
        work["buy_rank"] = (
            pd.to_numeric(work["final_score"], errors="coerce")
            .rank(method="first", ascending=False)
            .astype(float)
        )
    return work.sort_values(["buy_rank", "code"]).reset_index(drop=True)


def build_ranking_context(row: pd.Series, intent_row: pd.Series | None = None) -> dict[str, object]:
    intent_row = intent_row if intent_row is not None else pd.Series(dtype="object")
    def first(*keys: str) -> object:
        for key in keys:
            if key in intent_row.index and pd.notna(intent_row.get(key)):
                return intent_row.get(key)
            if key in row.index and pd.notna(row.get(key)):
                return row.get(key)
        return None

    return {
        "ranking_run_id": _int_or_none(first("ranking_run_id", "run_id")),
        "ranking_rank": _int_or_none(first("ranking_rank", "live_rank", "rank_final", "buy_rank")),
        "buy_rank": _int_or_none(first("buy_rank")),
        "rank_final": _int_or_none(first("rank_final")),
        "live_rank": _int_or_none(first("live_rank")),
        "final_score": _num_or_none(first("final_score", "live_score")),
        "live_score": _num_or_none(first("live_score")),
        "confidence_score": _num_or_none(first("confidence_score")),
        "risk_penalty": _num_or_none(first("risk_penalty")),
        "ret_score": _num_or_none(first("ret_score")),
        "prob_score": _num_or_none(first("prob_score")),
        "qual_score": _num_or_none(first("qual_score")),
        "tech_score": _num_or_none(first("tech_score")),
        "liquidity_score": _num_or_none(first("liquidity_score")),
        "safety_score": _num_or_none(first("safety_score")),
        "dominant_theme": str(first("dominant_theme") or "").strip() or None,
        "score_driver_1": str(first("score_driver_1") or "").strip() or None,
        "score_driver_2": str(first("score_driver_2") or "").strip() or None,
        "score_driver_3": str(first("score_driver_3") or "").strip() or None,
        "risk_factor_1": str(first("risk_factor_1") or "").strip() or None,
        "risk_factor_2": str(first("risk_factor_2") or "").strip() or None,
        "action_note": str(first("action_note") or "").strip() or None,
        "ai_filtered_source_used": bool(first("ai_filtered_source_used")) if first("ai_filtered_source_used") is not None else False,
        "ai_filtered_rank": _int_or_none(first("ai_filtered_rank")),
        "ai_adjusted_score": _num_or_none(first("ai_adjusted_score")),
        "entry_quality_score": _num_or_none(first("entry_quality_score")),
        "entry_quality_status": str(first("entry_quality_status") or "").strip() or None,
        "entry_quality_reasons": str(first("entry_quality_reasons") or "").strip() or None,
        "original_final_rank": _int_or_none(first("original_final_rank", "rank_final", "buy_rank")),
        "original_final_score": _num_or_none(first("original_final_score", "final_score", "live_score")),
    }


def load_price_fallback(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame(columns=["code", "close"])
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["code", "close"])
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work.get("date"), errors="coerce")
        work = work.sort_values(["date", "code"], ascending=[False, True], na_position="last")
    return work.loc[:, ["code", "close"]].dropna(subset=["close"]).drop_duplicates("code", keep="first").reset_index(drop=True)


def merge_ranking_with_prices(ranking: pd.DataFrame, price_fallback: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    if price_fallback.empty:
        return ranking
    merged = ranking.merge(
        price_fallback.rename(columns={"close": "fallback_close"}),
        on="code",
        how="left",
    )
    merged["close"] = pd.to_numeric(merged.get("close"), errors="coerce").where(
        pd.to_numeric(merged.get("close"), errors="coerce").notna(),
        pd.to_numeric(merged.get("fallback_close"), errors="coerce"),
    )
    return merged.drop(columns=["fallback_close"])


def _coalesce_allowed_qty(*values: object) -> int:
    numeric_values: list[int] = []
    for value in values:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            numeric_values.append(max(int(numeric), 0))
    if not numeric_values:
        return 0
    return min(numeric_values)


def _preview_expected_hold_reason(*, side: str, blocked_reason: str | None, final_qty: object) -> str:
    blocked = str(blocked_reason or "").strip()
    if blocked == "buy_context_unavailable":
        return "Live BUY context is unavailable, so the order stays on preview only."
    if blocked == "buy_qty_zero":
        return "Final BUY quantity is zero, so nothing will be submitted."
    if blocked == "live_price_unavailable":
        return "Live price is unavailable, so the BUY order stays on preview only."
    if blocked == "previous_close_unavailable":
        return "Previous close is unavailable, so the BUY order stays on preview only."
    if blocked == "entry_gap_up_blocked":
        return "Live price is above the soft entry gap threshold, so the BUY order stays on preview only."
    if blocked == "entry_gap_up_hard_blocked":
        return "Live price is above the hard entry gap threshold, so the BUY order stays on preview only."
    if blocked == "entry_gap_down_blocked":
        return "Live price is below the allowed entry gap threshold, so the BUY order stays on preview only."
    if blocked == "holding_qty_missing":
        return "Live holding quantity is missing, so the SELL order cannot be submitted."
    if blocked == "non_executable_intent_type":
        return "This intent type is not submit-ready."
    if blocked and str(side or "").upper() == "BUY":
        return f"BUY blocked by common live risk guard: {blocked}"
    qty = pd.to_numeric(final_qty, errors="coerce")
    side_upper = str(side or "").upper()
    if not pd.notna(qty) or int(qty) <= 0:
        return "Final order quantity is zero, so submission will be skipped."
    if not _env_flag("AUTO_TRADE_EXECUTE", False):
        return "AUTO_TRADE_EXECUTE is OFF, so this stays as preview only."
    if side_upper == "BUY" and not _env_flag("AUTO_TRADE_ALLOW_BUY", False):
        return "AUTO_TRADE_ALLOW_BUY is OFF, so BUY submission will be held."
    if side_upper == "BUY" and _env_flag("AUTO_TRADE_BUY_APPROVAL_REQUIRED", False):
        return "BUY approval is required. If request_id is not approved, submission will be skipped."
    return "No expected submit hold reason at preview stage."


def build_order_requests(
    *,
    intents_payload: dict[str, Any],
    holdings: pd.DataFrame,
    ranking: pd.DataFrame,
    ord_dvsn: str,
) -> dict[str, Any]:
    intents = pd.DataFrame(intents_payload.get("intents") or [])
    intents = intents.loc[intents.get("executable", False).fillna(False)].copy() if not intents.empty else pd.DataFrame()
    if intents.empty:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asof_date": intents_payload.get("asof_date"),
            "gate_status": intents_payload.get("gate_status"),
            "env_dv": None,
            "cash_summary": {},
            "items": [],
            "summary": {
                "request_count": 0,
                "buy_count": 0,
                "sell_count": 0,
            },
        }

    ranking_lookup = ranking.drop_duplicates("code").set_index("code") if not ranking.empty else pd.DataFrame()
    holdings_lookup = holdings.drop_duplicates("code").set_index("code") if not holdings.empty else pd.DataFrame()
    has_buy_intents = intents.get("intent_type", pd.Series(dtype="object")).astype(str).str.upper().eq("BUY").any()
    client = None
    account = None
    cash_summary: dict[str, Any] = {}
    available_cash = None
    total_assets = None
    entry_gate_config = _entry_price_gate_config()
    if has_buy_intents:
        client = KISClient.from_env()
        client.issue_access_token()
        account = resolve_account_env()
        _, summary_df = inquire_balance(client, account)
        cash_summary = summarize_cash(summary_df)
        available_cash = cash_summary.get("dnca_tot_amt")
        total_assets = cash_summary.get("tot_evlu_amt")

    items: list[dict[str, Any]] = []
    for _, row in intents.iterrows():
        code = str(row.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        ranking_row = ranking_lookup.loc[code] if not ranking_lookup.empty and code in ranking_lookup.index else pd.Series(dtype="object")
        holding_row = holdings_lookup.loc[code] if not holdings_lookup.empty and code in holdings_lookup.index else pd.Series(dtype="object")
        ranking_context = build_ranking_context(ranking_row, row)
        intent_type = str(row.get("intent_type") or "").upper()
        side = "BUY" if intent_type == "BUY" else "SELL" if intent_type in {"TRIM", "EXIT"} else "HOLD"
        reference_price = pd.to_numeric(ranking_row.get("close"), errors="coerce")
        order_price = int(reference_price) if pd.notna(reference_price) and reference_price > 0 else 0
        previous_close = int(reference_price) if pd.notna(reference_price) and reference_price > 0 else None
        live_price = None
        live_price_source = None
        entry_price_gap_pct = None
        entry_price_gate_status = None
        entry_price_gate_reason = None
        common_risk_allowed = True
        common_risk_block_reasons: list[str] = []
        common_risk_snapshot: dict[str, Any] = {}
        planned_qty = None
        allowed_qty = None
        final_qty = None
        blocked_reason = None

        if side == "BUY":
            if client is None or account is None:
                blocked_reason = "buy_context_unavailable"
                final_qty = 0
                planned_qty = 0
                allowed_qty = 0
                items.append(
                    {
                        "request_id": f"{intents_payload.get('asof_date') or 'unknown'}:{intent_type}:{code}",
                        "intent_id": row.get("intent_id"),
                        "code": code,
                        "name": str(ranking_row.get("name") or row.get("name") or "").strip() or None,
                        "side": side,
                        "intent_type": intent_type,
                        "ord_dvsn": ord_dvsn,
                        "reference_price": order_price if order_price else None,
                        "previous_close": previous_close,
                        "live_price": live_price,
                        "live_price_source": live_price_source,
                        "entry_price_gap_pct": entry_price_gap_pct,
                        "entry_price_gate_status": entry_price_gate_status,
                        "entry_price_gate_reason": entry_price_gate_reason,
                        "common_risk_allowed": common_risk_allowed,
                        "common_risk_block_reasons": common_risk_block_reasons,
                        "common_risk_snapshot": common_risk_snapshot,
                        "planned_qty": 0,
                        "allowed_qty": 0,
                        "final_request_qty": 0,
                        "target_weight": _num_or_none(row.get("target_weight")),
                        "priority": _int_or_none(row.get("priority")),
                        "reason": row.get("reason"),
                        **ranking_context,
                        "blocked_reason": blocked_reason,
                        "expected_hold_reason": _preview_expected_hold_reason(
                            side=side,
                            blocked_reason=blocked_reason,
                            final_qty=0,
                        ),
                        "executable_now": False,
                    }
                )
                continue
            live_snapshot = _fetch_live_price_snapshot(client, code)
            live_price = _int_or_none(live_snapshot.get("live_price"))
            live_price_source = str(live_snapshot.get("live_price_source") or "").strip() or None
            live_previous_close = _int_or_none(live_snapshot.get("previous_close"))
            if live_previous_close is not None:
                previous_close = live_previous_close
                order_price = live_previous_close
            gate_result = _evaluate_entry_price_gate(
                previous_close=previous_close,
                live_price=live_price,
                config=entry_gate_config,
            )
            entry_price_gap_pct = _num_or_none(gate_result.get("entry_price_gap_pct"))
            entry_price_gate_status = str(gate_result.get("entry_price_gate_status") or "").strip() or None
            entry_price_gate_reason = str(gate_result.get("entry_price_gate_reason") or "").strip() or None
            psbl = inquire_psbl_order(
                client,
                account,
                pdno=code,
                ord_unpr=str(order_price),
                ord_dvsn=ord_dvsn,
            )
            psbl_row = psbl.iloc[0] if not psbl.empty else pd.Series(dtype="object")
            nrcvb_buy_qty = pd.to_numeric(psbl_row.get("nrcvb_buy_qty"), errors="coerce")
            max_buy_qty = pd.to_numeric(psbl_row.get("max_buy_qty"), errors="coerce")
            target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
            current_position_value = pd.to_numeric(holding_row.get("eval_amount"), errors="coerce")
            planned_qty = compute_market_order_preview_qty(
                available_cash=available_cash,
                target_weight=float(target_weight) if pd.notna(target_weight) else 0.0,
                price=float(order_price) if order_price else None,
                total_assets=float(total_assets) if pd.notna(pd.to_numeric(total_assets, errors="coerce")) else None,
                current_position_value=float(current_position_value) if pd.notna(current_position_value) else None,
            )
            allowed_qty = _coalesce_allowed_qty(nrcvb_buy_qty, max_buy_qty)
            final_qty = min(planned_qty, allowed_qty)
            if not final_qty or final_qty <= 0:
                blocked_reason = "buy_qty_zero"
            elif bool(gate_result.get("blocked")):
                blocked_reason = entry_price_gate_reason
            else:
                common_risk_allowed, common_risk_block_reasons, common_risk_snapshot = evaluate_common_buy_guard(
                    {
                        "as_of_date": intents_payload.get("asof_date"),
                        "execution_date": datetime.now().strftime("%Y-%m-%d"),
                        "side": side,
                        "code": code,
                        "order_qty": int(final_qty),
                        "order_amount": float(final_qty) * float(order_price) if order_price else None,
                        "reference_price": order_price if order_price else None,
                    }
                )
                if not common_risk_allowed:
                    blocked_reason = ";".join(common_risk_block_reasons) if common_risk_block_reasons else "common_risk_blocked"
        elif side == "SELL":
            current_qty = pd.to_numeric(holding_row.get("qty"), errors="coerce")
            if not pd.notna(current_qty) or float(current_qty) <= 0:
                blocked_reason = "holding_qty_missing"
                final_qty = 0
            else:
                if intent_type == "TRIM":
                    weight = pd.to_numeric(holding_row.get("weight"), errors="coerce")
                    target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
                    trim_ratio = 0.5
                    if pd.notna(weight) and pd.notna(target_weight) and weight > 0:
                        trim_ratio = max(min((float(weight) - float(target_weight)) / float(weight), 1.0), 0.0)
                    final_qty = max(int(round(float(current_qty) * trim_ratio)), 1)
                else:
                    final_qty = int(round(float(current_qty)))
                planned_qty = final_qty
                allowed_qty = final_qty
                order_price = 0
        else:
            blocked_reason = "non_executable_intent_type"
            final_qty = 0

        items.append(
            {
                "request_id": f"{intents_payload.get('asof_date') or 'unknown'}:{intent_type}:{code}",
                "intent_id": row.get("intent_id"),
                "code": code,
                "name": str(ranking_row.get("name") or row.get("name") or "").strip() or None,
                "side": side,
                "intent_type": intent_type,
                "ord_dvsn": ord_dvsn,
                "reference_price": order_price if order_price else None,
                "previous_close": previous_close,
                "live_price": live_price,
                "live_price_source": live_price_source,
                "entry_price_gap_pct": entry_price_gap_pct,
                "entry_price_gate_status": entry_price_gate_status,
                "entry_price_gate_reason": entry_price_gate_reason,
                "common_risk_allowed": common_risk_allowed,
                "common_risk_block_reasons": common_risk_block_reasons,
                "common_risk_snapshot": common_risk_snapshot,
                "planned_qty": int(planned_qty) if pd.notna(planned_qty) else None,
                "allowed_qty": int(allowed_qty) if pd.notna(allowed_qty) else None,
                "final_request_qty": int(final_qty) if final_qty is not None else None,
                "target_weight": _num_or_none(row.get("target_weight")),
                "priority": _int_or_none(row.get("priority")),
                "reason": row.get("reason"),
                **ranking_context,
                "blocked_reason": blocked_reason,
                "expected_hold_reason": _preview_expected_hold_reason(
                    side=side,
                    blocked_reason=blocked_reason,
                    final_qty=final_qty,
                ),
                "executable_now": blocked_reason in {None, ""},
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": intents_payload.get("asof_date"),
        "gate_status": intents_payload.get("gate_status"),
        "env_dv": account.env_dv if account is not None else None,
        "cash_summary": cash_summary,
        "items": items,
        "summary": {
            "request_count": len(items),
            "buy_count": sum(1 for item in items if item["side"] == "BUY"),
            "sell_count": sum(1 for item in items if item["side"] == "SELL"),
        },
    }


def _require_execution_confirmation(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.confirm_text != "LIVE_ORDER":
        raise ValueError("execution blocked: --confirm-text LIVE_ORDER is required")


def _load_previous_success_ids(path: Path) -> set[str]:
    payload = _safe_read_json(path)
    items = payload.get("items") or []
    success_ids = {
        str(item.get("request_id") or "").strip()
        for item in items
        if str(item.get("submission_status") or "").strip() == "submitted"
    }
    return {item for item in success_ids if item}


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    value = pd.to_numeric(os.environ.get(name), errors="coerce")
    if pd.notna(value):
        return float(value)
    return float(default)


def _entry_price_gate_config() -> dict[str, object]:
    return {
        "block_up_pct": _env_float("ENTRY_GAP_BLOCK_UP_PCT", 0.03),
        "hard_block_up_pct": _env_float("ENTRY_GAP_HARD_BLOCK_UP_PCT", 0.05),
        "block_down_pct": _env_float("ENTRY_GAP_BLOCK_DOWN_PCT", -0.04),
        "block_on_live_price_missing": _env_flag("ENTRY_GAP_BLOCK_ON_LIVE_PRICE_MISSING", True),
    }


def _fetch_live_price_snapshot(client: KISClient, code: str) -> dict[str, object]:
    payload = {
        "previous_close": None,
        "live_price": None,
        "live_price_source": "unavailable",
    }
    try:
        response = client.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": str(code).zfill(6),
            },
        )
        if isinstance(response, dict):
            output = response.get("output") or {}
        elif hasattr(response, "json"):
            output = response.json().get("output") or {}
        else:
            output = {}
        payload["previous_close"] = _int_or_none(output.get("stck_sdpr"))
        payload["live_price"] = _int_or_none(output.get("stck_prpr"))
        if payload["live_price"] is not None:
            payload["live_price_source"] = "kis_quote"
    except Exception:
        logging.warning("Live price lookup failed for %s", code, exc_info=True)
    return payload


def _evaluate_entry_price_gate(
    *,
    previous_close: object,
    live_price: object,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    gate_config = config or _entry_price_gate_config()
    prev_close = pd.to_numeric(previous_close, errors="coerce")
    current_price = pd.to_numeric(live_price, errors="coerce")
    if not pd.notna(current_price) or float(current_price) <= 0:
        reason = "live_price_unavailable"
        return {
            "entry_price_gap_pct": None,
            "entry_price_gate_status": "blocked" if gate_config.get("block_on_live_price_missing") else "ok",
            "entry_price_gate_reason": reason,
            "blocked": bool(gate_config.get("block_on_live_price_missing")),
        }
    if not pd.notna(prev_close) or float(prev_close) <= 0:
        return {
            "entry_price_gap_pct": None,
            "entry_price_gate_status": "blocked",
            "entry_price_gate_reason": "previous_close_unavailable",
            "blocked": True,
        }
    gap_pct = (float(current_price) - float(prev_close)) / float(prev_close)
    if gap_pct >= float(gate_config["hard_block_up_pct"]):
        reason = "entry_gap_up_hard_blocked"
        status = "blocked"
        blocked = True
    elif gap_pct > float(gate_config["block_up_pct"]):
        reason = "entry_gap_up_blocked"
        status = "blocked"
        blocked = True
    elif gap_pct <= float(gate_config["block_down_pct"]):
        reason = "entry_gap_down_blocked"
        status = "blocked"
        blocked = True
    else:
        reason = "entry_gap_ok"
        status = "ok"
        blocked = False
    return {
        "entry_price_gap_pct": gap_pct,
        "entry_price_gate_status": status,
        "entry_price_gate_reason": reason,
        "blocked": blocked,
    }


def _load_approved_request_ids(path: Path) -> set[str]:
    payload = _safe_read_json(path)
    approved = payload.get("approved_request_ids") or payload.get("request_ids") or []
    if not isinstance(approved, list):
        return set()
    return {str(item).strip() for item in approved if str(item).strip()}


def _execution_price_for_submit(*, ord_dvsn: str, reference_price: object) -> str:
    if str(ord_dvsn).strip() == "01":
        return "0"
    price = pd.to_numeric(reference_price, errors="coerce")
    if not pd.notna(price) or float(price) <= 0:
        raise ValueError("limit order requires a positive reference_price")
    return str(int(price))


def execute_order_requests(
    *,
    preview_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    _require_execution_confirmation(args)
    previous_success_ids = set() if args.force_resubmit else _load_previous_success_ids(_resolve(args.out_exec_json))
    buy_approval_required = _env_flag("AUTO_TRADE_BUY_APPROVAL_REQUIRED", False)
    approved_request_ids = _load_approved_request_ids(args.approval_json) if buy_approval_required else set()
    client = KISClient.from_env()
    client.issue_access_token()
    account = resolve_account_env()
    entry_gate_config = _entry_price_gate_config()

    results: list[dict[str, Any]] = []
    for item in preview_payload.get("items") or []:
        request_id = str(item.get("request_id") or "").strip()
        side = str(item.get("side") or "").strip().upper()
        qty = pd.to_numeric(item.get("final_request_qty"), errors="coerce")
        blocked_reason = str(item.get("blocked_reason") or "").strip()

        result: dict[str, Any] = {
            "request_id": request_id or None,
            "intent_id": item.get("intent_id"),
            "code": item.get("code"),
            "name": item.get("name"),
            "side": side or None,
            "intent_type": item.get("intent_type"),
            "ord_dvsn": item.get("ord_dvsn") or args.ord_dvsn,
            "reference_price": item.get("reference_price"),
            "previous_close": item.get("previous_close"),
            "live_price": item.get("live_price"),
            "live_price_source": item.get("live_price_source"),
            "entry_price_gap_pct": item.get("entry_price_gap_pct"),
            "entry_price_gate_status": item.get("entry_price_gate_status"),
            "entry_price_gate_reason": item.get("entry_price_gate_reason"),
            "common_risk_allowed": item.get("common_risk_allowed"),
            "common_risk_block_reasons": item.get("common_risk_block_reasons"),
            "common_risk_snapshot": item.get("common_risk_snapshot"),
            "final_request_qty": int(qty) if pd.notna(qty) else None,
            "submitted_at": None,
            "submission_status": "skipped",
            "skip_reason": None,
            "broker_order_id": None,
            "broker_org_order_id": None,
            "raw_response": None,
        }

        if blocked_reason:
            result["skip_reason"] = blocked_reason
            results.append(result)
            continue
        if not request_id:
            result["skip_reason"] = "missing_request_id"
            results.append(result)
            continue
        if request_id in previous_success_ids:
            result["skip_reason"] = "duplicate_request_id"
            results.append(result)
            continue
        if side == "BUY" and not args.allow_buy:
            result["skip_reason"] = "buy_requires_allow_buy"
            results.append(result)
            continue
        if side == "BUY" and buy_approval_required and request_id not in approved_request_ids:
            result["skip_reason"] = "buy_approval_required"
            results.append(result)
            continue
        if side == "BUY" and not bool(item.get("common_risk_allowed", True)):
            result["skip_reason"] = ";".join(item.get("common_risk_block_reasons") or []) or "common_risk_blocked"
            results.append(result)
            continue
        if side not in {"BUY", "SELL"}:
            result["skip_reason"] = "unsupported_side"
            results.append(result)
            continue
        if not pd.notna(qty) or int(qty) <= 0:
            result["skip_reason"] = "invalid_final_request_qty"
            results.append(result)
            continue

        try:
            if side == "BUY":
                live_snapshot = _fetch_live_price_snapshot(client, str(item.get("code") or "").zfill(6))
                live_price = _int_or_none(live_snapshot.get("live_price"))
                live_price_source = str(live_snapshot.get("live_price_source") or "").strip() or None
                previous_close = _int_or_none(live_snapshot.get("previous_close")) or _int_or_none(item.get("previous_close"))
                gate_result = _evaluate_entry_price_gate(
                    previous_close=previous_close,
                    live_price=live_price,
                    config=entry_gate_config,
                )
                result["previous_close"] = previous_close
                result["live_price"] = live_price
                result["live_price_source"] = live_price_source
                result["entry_price_gap_pct"] = _num_or_none(gate_result.get("entry_price_gap_pct"))
                result["entry_price_gate_status"] = gate_result.get("entry_price_gate_status")
                result["entry_price_gate_reason"] = gate_result.get("entry_price_gate_reason")
                if bool(gate_result.get("blocked")):
                    result["skip_reason"] = str(gate_result.get("entry_price_gate_reason") or "entry_price_blocked")
                    results.append(result)
                    continue
                common_risk_allowed, common_risk_block_reasons, common_risk_snapshot = evaluate_common_buy_guard(
                    {
                        "as_of_date": preview_payload.get("asof_date"),
                        "execution_date": datetime.now().strftime("%Y-%m-%d"),
                        "side": side,
                        "code": item.get("code"),
                        "order_qty": int(qty),
                        "order_amount": float(qty) * float(item.get("reference_price") or 0) if pd.notna(qty) else None,
                        "reference_price": item.get("reference_price"),
                    }
                )
                result["common_risk_allowed"] = common_risk_allowed
                result["common_risk_block_reasons"] = common_risk_block_reasons
                result["common_risk_snapshot"] = common_risk_snapshot
                if not common_risk_allowed:
                    result["skip_reason"] = ";".join(common_risk_block_reasons) if common_risk_block_reasons else "common_risk_blocked"
                    results.append(result)
                    continue
            order_price = _execution_price_for_submit(
                ord_dvsn=str(item.get("ord_dvsn") or args.ord_dvsn),
                reference_price=item.get("reference_price"),
            )
            response_df = order_cash(
                client,
                account,
                side=side.lower(),
                pdno=str(item.get("code") or "").zfill(6),
                ord_dvsn=str(item.get("ord_dvsn") or args.ord_dvsn),
                ord_qty=str(int(qty)),
                ord_unpr=order_price,
            )
            response_row = response_df.iloc[0].to_dict() if not response_df.empty else {}
            result["submitted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["submission_status"] = "submitted"
            result["broker_order_id"] = response_row.get("ODNO") or response_row.get("odno")
            result["broker_org_order_id"] = response_row.get("KRX_FWDG_ORD_ORGNO") or response_row.get("krx_fwdg_ord_orgno")
            result["raw_response"] = response_row
        except Exception as exc:
            result["submitted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["submission_status"] = "failed"
            result["skip_reason"] = str(exc)
        results.append(result)

    return {
        "generated_at": preview_payload.get("generated_at"),
        "executed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": preview_payload.get("asof_date"),
        "gate_status": preview_payload.get("gate_status"),
        "env_dv": account.env_dv,
        "allow_buy": bool(args.allow_buy),
        "buy_approval_required": bool(buy_approval_required),
        "approved_request_count": len(approved_request_ids),
        "force_resubmit": bool(args.force_resubmit),
        "items": results,
        "summary": {
            "request_count": len(results),
            "submitted_count": sum(1 for item in results if item["submission_status"] == "submitted"),
            "failed_count": sum(1 for item in results if item["submission_status"] == "failed"),
            "skipped_count": sum(1 for item in results if item["submission_status"] == "skipped"),
        },
    }


def render_preview_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Order Requests Preview",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- asof_date: {payload.get('asof_date') or 'NA'}",
        f"- gate_status: {payload.get('gate_status') or 'NA'}",
        f"- env_dv: {payload.get('env_dv') or 'NA'}",
        f"- request_count: {payload['summary']['request_count']}",
        f"- buy_count: {payload['summary']['buy_count']}",
        f"- sell_count: {payload['summary']['sell_count']}",
        "",
        "| request_id | code | name | side | intent_type | ai_src | ai_rank | ai_adj_score | entry_q_score | entry_q_status | original_rank | original_score | ref_price | prev_close | live_price | live_src | gap_pct | gate_status | gate_reason | final_qty | executable_now | blocked_reason | expected_hold_reason | reason |",
        "| ---------- | ---- | ---- | ---- | ----------- | ------ | ------- | ------------ | ------------- | -------------- | ------------- | -------------- | --------- | ---------- | ---------- | -------- | ------- | ----------- | ----------- | --------- | -------------- | -------------- | -------------------- | ------ |",
    ]
    for item in payload["items"]:
        lines.append(
            f"| {item.get('request_id') or ''} | {item.get('code') or ''} | {item.get('name') or ''} | {item.get('side') or ''} | {item.get('intent_type') or ''} | {'Y' if item.get('ai_filtered_source_used') else 'N'} | {_fmt_num(item.get('ai_filtered_rank'), 0)} | {_fmt_num(item.get('ai_adjusted_score'), 2)} | {_fmt_num(item.get('entry_quality_score'), 2)} | {item.get('entry_quality_status') or ''} | {_fmt_num(item.get('original_final_rank'), 0)} | {_fmt_num(item.get('original_final_score'), 2)} | {_fmt_num(item.get('reference_price'), 0)} | {_fmt_num(item.get('previous_close'), 0)} | {_fmt_num(item.get('live_price'), 0)} | {item.get('live_price_source') or ''} | {_fmt_num((item.get('entry_price_gap_pct') or 0) * 100 if item.get('entry_price_gap_pct') is not None else None, 2)} | {item.get('entry_price_gate_status') or ''} | {item.get('entry_price_gate_reason') or ''} | {_fmt_num(item.get('final_request_qty'), 0)} | {'Y' if item.get('executable_now') else 'N'} | {item.get('blocked_reason') or ''} | {item.get('expected_hold_reason') or ''} | {item.get('reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_execution_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Order Requests Execution",
        "",
        f"- executed_at: {payload['executed_at']}",
        f"- asof_date: {payload.get('asof_date') or 'NA'}",
        f"- gate_status: {payload.get('gate_status') or 'NA'}",
        f"- env_dv: {payload.get('env_dv') or 'NA'}",
        f"- allow_buy: {'Y' if payload.get('allow_buy') else 'N'}",
        f"- submitted_count: {payload['summary']['submitted_count']}",
        f"- failed_count: {payload['summary']['failed_count']}",
        f"- skipped_count: {payload['summary']['skipped_count']}",
        "",
        "| request_id | code | side | qty | status | broker_order_id | skip_reason |",
        "| ---------- | ---- | ---- | --- | ------ | --------------- | ----------- |",
    ]
    for item in payload["items"]:
        lines.append(
            f"| {item.get('request_id') or ''} | {item.get('code') or ''} | {item.get('side') or ''} | {_fmt_num(item.get('final_request_qty'), 0)} | {item.get('submission_status') or ''} | {item.get('broker_order_id') or ''} | {item.get('skip_reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8-sig",
    )


def write_text(path: Path, text: str) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8-sig")


def sync_web_display_if_configured() -> None:
    if not str(os.environ.get("WEB_DATABASE_URL", "")).strip():
        return
    subprocess.run(
        [
            sys.executable,
            str(SYNC_WEB_DISPLAY_SCRIPT),
            "--skip-core",
            "--skip-paper-trading",
            "--skip-trades",
        ],
        cwd=ROOT,
        check=True,
    )


def sync_live_trade_ledger_best_effort(
    *,
    intents_payload: dict[str, Any],
    preview_payload: dict[str, Any],
    execution_payload: dict[str, Any] | None = None,
) -> None:
    try:
        result = sync_live_trade_ledger(
            intents_payload=intents_payload,
            preview_payload=preview_payload,
            execution_payload=execution_payload or {},
        )
        logging.info("Live trade ledger sync completed: %s", result)
    except Exception:
        logging.warning("Live trade ledger sync failed", exc_info=True)


def main() -> int:
    _load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    intents_payload = _safe_read_json(args.trade_intents_json)
    if not intents_payload:
        raise FileNotFoundError("trade intents payload not found")

    holdings = load_live_holdings(args.live_holdings_csv)
    ranking = merge_ranking_with_prices(
        load_ranking(args.ranking_csv),
        load_price_fallback(args.price_fallback_csv),
    )
    preview_payload = build_order_requests(
        intents_payload=intents_payload,
        holdings=holdings,
        ranking=ranking,
        ord_dvsn=args.ord_dvsn,
    )

    write_payload(args.out_json, preview_payload)
    write_text(args.out_md, render_preview_markdown(preview_payload))
    sync_live_trade_ledger_best_effort(
        intents_payload=intents_payload,
        preview_payload=preview_payload,
    )
    print(f"order_requests_preview_json: {_resolve(args.out_json)}")
    print(f"order_requests_preview_md: {_resolve(args.out_md)}")

    if not args.execute:
        try:
            sync_web_display_if_configured()
        except subprocess.CalledProcessError as exc:
            logging.warning("Post-preview web display sync failed: %s", exc)
        except Exception:
            logging.warning("Post-preview web display sync failed", exc_info=True)
        return 0

    execution_payload = execute_order_requests(preview_payload=preview_payload, args=args)
    write_payload(args.out_exec_json, execution_payload)
    write_text(args.out_exec_md, render_execution_markdown(execution_payload))
    sync_live_trade_ledger_best_effort(
        intents_payload=intents_payload,
        preview_payload=preview_payload,
        execution_payload=execution_payload,
    )
    print(f"order_requests_execution_json: {_resolve(args.out_exec_json)}")
    print(f"order_requests_execution_md: {_resolve(args.out_exec_md)}")
    try:
        sync_web_display_if_configured()
    except subprocess.CalledProcessError as exc:
        logging.warning("Post-execution web display sync failed: %s", exc)
    except Exception:
        logging.warning("Post-execution web display sync failed", exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
