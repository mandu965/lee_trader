from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from krx_price_utils import normalize_krx_limit_price
from kis_client import KISAuthError, KISClient, KISHTTPError
from kis_live_account import order_cash
from rule_account_guard import evaluate_rule_order_guard
from rule_execution_simulator import (
    append_jsonl,
    build_aborted_results,
    load_calendar,
    load_json,
    render_reconciliation,
    validate_previous_execution_completed,
    validate_trading_session,
)
from rule_market_open_snapshot import check_market_data_available, resolve_rule_account_env
from rule_paper_state_manager import default_state
from rule_trading_diagnostics import (
    append_trade_flow_log,
    build_block_reason_details,
    build_trade_flow_payload,
    debug_trade_mode_enabled,
    select_primary_block_reason,
)
from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"

DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_RESULTS = OUTPUT_DIR / "rule_execution_results.json"
DEFAULT_RECON_MD = OUTPUT_DIR / "rule_execution_reconciliation_report.md"
DEFAULT_CALENDAR = ROOT / "config" / "trading_calendar_kr.json"
DEFAULT_EXECUTION_HISTORY = OUTPUT_DIR / "rule_execution_history.jsonl"
DEFAULT_MARKET_SNAPSHOT = OUTPUT_DIR / "rule_market_open_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit RULE pilot/live orders using KIS.")
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out-results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-reconciliation-md", type=Path, default=DEFAULT_RECON_MD)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--out-execution-history-jsonl", type=Path, default=DEFAULT_EXECUTION_HISTORY)
    parser.add_argument("--out-market-snapshot-json", type=Path, default=DEFAULT_MARKET_SNAPSHOT)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _append_reason(existing: str | None, *reasons: str | None) -> str:
    parts = [part for part in str(existing or "").split(";") if part and part != "none"]
    for reason in reasons:
        text = str(reason or "").strip()
        if text and text != "none":
            parts.append(text)
    unique = list(dict.fromkeys(parts))
    return ";".join(unique) if unique else "none"


def _ord_dvsn(item: dict[str, Any]) -> str:
    if str(item.get("order_type") or "").strip().lower() == "market":
        return "01"
    return "00"


def _ord_unpr(item: dict[str, Any], ord_dvsn: str) -> str:
    if ord_dvsn == "01":
        return "0"
    price = item.get("limit_price") or item.get("original_limit_price") or item.get("expected_execution_price")
    numeric = _float(price)
    if numeric is None or numeric <= 0:
        raise ValueError("limit_price_missing")
    side = str(item.get("side") or "BUY").upper()
    normalized = normalize_krx_limit_price(numeric, side="SELL" if side == "SELL" else "BUY")
    if normalized is None or normalized <= 0:
        raise ValueError("limit_price_invalid")
    return str(normalized)


def _log(message: str) -> None:
    print(f"[RULE_ORDER_SUBMITTER] {message}")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _is_retryable_order_error(exc: Exception) -> tuple[bool, float]:
    """주문 재시도 가능 여부와 대기 시간(초)을 반환합니다.

    주문은 KIS에 도달했을 가능성이 있는 오류(타임아웃, 5xx)는 재시도하지 않습니다.
    중복 주문 방지가 최우선입니다.
    """
    if isinstance(exc, KISAuthError):
        # 토큰 만료 → 재인증 후 재시도 안전
        return True, 1.0
    if isinstance(exc, KISHTTPError):
        if exc.status_code == 429:
            # 레이트 리밋 → 요청이 처리되지 않았으므로 안전
            return True, float(exc.retry_after_sec or 5.0)
        # 5xx: 요청이 KIS에 도달했을 수 있음 → 재시도 불가
        # 4xx (429 제외): 유효성 오류 → 재시도해도 동일 결과
        return False, 0.0
    if isinstance(exc, ConnectionRefusedError):
        # TCP 연결 자체가 거절됨 → 안전
        return True, 1.0
    # TimeoutError, OSError 등: 요청이 KIS에 도달했을 수 있음 → 재시도 불가
    return False, 0.0


def _submit_order_with_retry(
    client: KISClient,
    account: Any,
    *,
    side: str,
    pdno: str,
    ord_dvsn: str,
    ord_qty: str,
    ord_unpr: str,
    code: str,
) -> tuple[pd.DataFrame, int, list[str]]:
    """order_cash 호출을 안전한 조건에서만 재시도합니다.

    Returns:
        (response_df, attempt_count, retry_reasons)
    """
    retry_enabled = os.environ.get("RULE_RETRY_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    max_retry = _env_int("RULE_MAX_RETRY_COUNT", 1) if retry_enabled else 0

    last_exc: Exception = RuntimeError("no_attempt")
    retry_reasons: list[str] = []

    for attempt in range(max_retry + 1):
        try:
            if attempt > 0:
                # 재시도 전 토큰 재발급 시도 (auth 오류 대응)
                try:
                    client.issue_access_token()
                except Exception as token_exc:
                    _log(f"token_refresh_failed attempt={attempt} code={code} error={token_exc}")
            response_df = order_cash(
                client, account,
                side=side, pdno=pdno, ord_dvsn=ord_dvsn,
                ord_qty=ord_qty, ord_unpr=ord_unpr,
            )
            return response_df, attempt + 1, retry_reasons
        except Exception as exc:
            last_exc = exc
            retryable, wait_sec = _is_retryable_order_error(exc)
            reason = f"attempt{attempt + 1}:{type(exc).__name__}"
            if attempt > 0:
                retry_reasons.append(reason)
            if not retryable or attempt >= max_retry:
                break
            retry_reasons.append(reason)
            _log(f"order_retry attempt={attempt + 1}/{max_retry} code={code} reason={exc} wait={wait_sec}s")
            time.sleep(wait_sec)

    raise last_exc


def _build_market_snapshot(symbols: list[str], out_path: Path) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": [str(symbol).zfill(6) for symbol in symbols if str(symbol).strip()],
        **check_market_data_available(symbols),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _gap_thresholds() -> tuple[float, float]:
    """환경변수로 갭 차단 임계값을 읽습니다. 기본값: 상단 +5%, 하단 -4%."""
    try:
        upper = float(os.environ.get("RULE_BUY_MAX_ACTUAL_OPEN_GAP", "0.05"))
    except (ValueError, TypeError):
        upper = 0.05
    try:
        lower = float(os.environ.get("RULE_BUY_MIN_ACTUAL_OPEN_GAP", "-0.04"))
    except (ValueError, TypeError):
        lower = -0.04
    return upper, lower


def _block_on_gap_unavailable() -> bool:
    return os.environ.get("RULE_BLOCK_ON_OPEN_GAP_UNAVAILABLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _live_order_context(item: dict[str, Any], market_row: dict[str, Any] | None, preview: dict[str, Any]) -> dict[str, Any]:
    side = str(item.get("side") or "NONE").upper()
    actual_open_gap = _float((market_row or {}).get("actual_open_gap"))
    actual_gap_reason: str | None = None
    gap_upper, gap_lower = _gap_thresholds()

    if side == "BUY":
        if market_row is None:
            # snapshot 파일에 종목코드가 없음 (매핑 실패)
            actual_gap_reason = "market_snapshot_unmapped"
        elif actual_open_gap is None:
            # snapshot row는 있지만 open_price/prev_close 값이 유효하지 않음
            if _block_on_gap_unavailable():
                actual_gap_reason = "actual_open_gap_unavailable"
        elif actual_open_gap > gap_upper:
            actual_gap_reason = "actual_open_gap_gt_5pct"
        elif actual_open_gap < gap_lower:
            actual_gap_reason = "actual_open_gap_lt_minus_4pct"

    open_price = (market_row or {}).get("open_price")
    prev_close = (market_row or {}).get("previous_close")
    _log(
        f"[RULE_ORDER_GUARD] symbol={item.get('code')} name={item.get('name')} side={side} "
        f"prev_close={prev_close} open_price={open_price} actual_open_gap={actual_open_gap} "
        f"market_row_found={market_row is not None} actual_gap_reason={actual_gap_reason}"
    )
    if market_row is not None and (open_price is None or (isinstance(open_price, (int, float)) and open_price <= 0)):
        _log(f"[RULE_ORDER_GUARD] WARNING open_price={open_price} is None/zero for symbol={item.get('code')} — actual_open_gap should be None")

    base_gap_reason = item.get("gap_risk_reason")
    gap_reason = actual_gap_reason or base_gap_reason
    return {
        "account_id": preview.get("account_id"),
        "strategy_id": preview.get("strategy_id"),
        "engine_type": preview.get("engine_type"),
        "run_mode": preview.get("run_mode"),
        "as_of_date": preview.get("as_of_date"),
        "execution_date": datetime.now().strftime("%Y-%m-%d"),
        "side": item.get("side"),
        "code": item.get("code") or item.get("symbol"),
        "order_qty": item.get("order_qty"),
        "order_amount": item.get("order_amount"),
        "reference_price": (market_row or {}).get("open_price") or item.get("limit_price") or item.get("expected_execution_price"),
        "signal_strength": item.get("signal_strength"),
        "market_defensive_mode": item.get("market_defensive_mode"),
        "gap_risk_blocked": gap_reason not in {None, "", "none"},
        "gap_risk_reason": gap_reason,
        "trading_value_pass": item.get("trading_value_block_reason") in {None, "", "none"},
        "trading_value_block_reason": item.get("trading_value_block_reason"),
        "sector_limit_pass": item.get("sector_limit_pass", True),
        "cooldown_pass": item.get("cooldown_pass", True),
        "cash_limit_pass": item.get("cash_limit_pass", True),
        "daily_order_amount_used": item.get("daily_order_amount_used", 0.0),
    }


def _submit_items(preview: dict[str, Any], market_snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    debug_trade_mode = debug_trade_mode_enabled()
    if str(market_snapshot.get("api_health_status") or "").strip().lower() == "auth_failed":
        raise RuntimeError(str(market_snapshot.get("api_failure_reason") or "market_snapshot_auth_failed"))
    account = resolve_rule_account_env()
    client = None
    if not debug_trade_mode:
        client = KISClient.from_rule_env()
        client.issue_access_token()

    snapshot_map = market_snapshot.get("snapshots") or {}
    results: list[dict[str, Any]] = []
    submitted_count = 0
    failed_count = 0
    skipped_count = 0
    submitted_buy_amount = 0.0
    submitted_sell_amount = 0.0

    for item in preview.get("items") or []:
        row = dict(item)
        row["execution_checked_at"] = datetime.now().isoformat(timespec="seconds")
        row["submitted_at"] = None
        row["raw_response"] = None
        row["broker_order_id"] = None
        row["broker_org_order_id"] = None
        row["filled_qty"] = 0
        row["filled_amount"] = 0.0
        row["avg_fill_price"] = None
        row["unfilled_qty"] = int(float(row.get("order_qty") or 0))
        row["estimated_amount"] = float(row.get("order_amount") or 0.0)
        row["daily_order_amount_used"] = submitted_buy_amount + submitted_sell_amount
        row["submit_attempted"] = False
        row["debug_trade_mode"] = debug_trade_mode

        side = str(row.get("side") or "NONE").upper()
        if side == "NONE":
            _log(
                f"skip no-order-action code={row.get('code')} name={row.get('name')} run_mode={preview.get('run_mode')}"
            )
            row["order_status"] = "planned"
            row["reconciliation_status"] = "skipped_no_order_action"
            trade_flow = build_trade_flow_payload(
                stage="execution_skip",
                run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                symbol=row.get("code"),
                score=row.get("signal_strength"),
                strategy_action=row.get("portfolio_action"),
                risk_pass=False,
                order_amount=row.get("estimated_amount"),
                price=row.get("expected_execution_price"),
                qty=row.get("order_qty"),
                final_amount=row.get("order_amount"),
                blocked=True,
                blocked_reason=row.get("order_block_reason"),
                extra={"side": side, "submit_attempted": False, "broker_result": "SKIPPED_NO_ORDER_ACTION"},
            )
            row["trade_flow"] = trade_flow
            append_trade_flow_log(trade_flow)
            results.append(row)
            continue

        market_row = snapshot_map.get(str(row.get("code") or row.get("symbol") or "").zfill(6))
        row["actual_open_gap"] = _float((market_row or {}).get("actual_open_gap"))
        row["market_data_available"] = bool((market_row or {}).get("market_data_available"))
        row["api_health_status"] = market_snapshot.get("api_health_status")
        row["api_failure_reason"] = market_snapshot.get("api_failure_reason")
        if market_row and market_row.get("open_price") is not None:
            row["expected_execution_price"] = market_row.get("open_price")

        preview_common_allowed = bool(row.get("common_risk_allowed", True))
        preview_common_reasons = list(row.get("common_risk_block_reasons") or [])
        if side == "BUY" and not preview_common_allowed:
            row["order_allowed"] = False
            row["order_block_reason"] = _append_reason(row.get("order_block_reason"), *preview_common_reasons)
            row["common_risk_allowed"] = False
            row["common_risk_block_reasons"] = preview_common_reasons
            row["common_risk_snapshot"] = row.get("common_risk_snapshot") or {"bypass_reason": "preview_common_risk_blocked"}
            row["order_status"] = "blocked"
            row["reconciliation_status"] = "blocked_before_submit"
            row["block_reason_details"] = build_block_reason_details(str(row.get("order_block_reason") or "").split(";"))
            row["primary_block_reason"] = select_primary_block_reason(row["block_reason_details"])
            _log(
                f"blocked preview-common-risk code={row.get('code')} name={row.get('name')} side={side} qty={row.get('order_qty')} "
                f"estimated_amount={row.get('estimated_amount')} run_mode={preview.get('run_mode')} "
                f"reasons={row.get('order_block_reason')}"
            )
            trade_flow = build_trade_flow_payload(
                stage="execution_guard",
                run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                symbol=row.get("code"),
                score=row.get("signal_strength"),
                strategy_action=row.get("portfolio_action"),
                risk_pass=False,
                order_amount=row.get("estimated_amount"),
                price=row.get("expected_execution_price"),
                qty=row.get("order_qty"),
                final_amount=row.get("order_amount"),
                blocked=True,
                blocked_reason=row.get("order_block_reason"),
                extra={"side": side, "submit_attempted": False, "broker_result": "POLICY_BLOCK"},
            )
            row["trade_flow"] = trade_flow
            append_trade_flow_log(trade_flow)
            skipped_count += 1
            results.append(row)
            continue

        order_allowed, guard_reasons, guard_details = evaluate_rule_order_guard(_live_order_context(row, market_row, preview))
        row["common_risk_allowed"] = bool(guard_details.get("common_risk_allowed", True))
        row["common_risk_block_reasons"] = list(guard_details.get("common_risk_block_reasons") or [])
        row["common_risk_snapshot"] = dict(guard_details.get("common_risk_snapshot") or {})
        if not order_allowed:
            row["order_allowed"] = False
            row["order_block_reason"] = _append_reason(row.get("order_block_reason"), *guard_reasons)
            row["final_guard_details"] = guard_details
            row["order_status"] = "blocked"
            row["reconciliation_status"] = "blocked_before_submit"
            row["block_reason_details"] = build_block_reason_details(str(row.get("order_block_reason") or "").split(";"))
            row["primary_block_reason"] = select_primary_block_reason(row["block_reason_details"])
            _log(
                f"blocked final-guard code={row.get('code')} name={row.get('name')} side={side} qty={row.get('order_qty')} "
                f"estimated_amount={row.get('estimated_amount')} run_mode={preview.get('run_mode')} "
                f"reasons={row.get('order_block_reason')}"
            )
            trade_flow = build_trade_flow_payload(
                stage="execution_guard",
                run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                symbol=row.get("code"),
                score=row.get("signal_strength"),
                strategy_action=row.get("portfolio_action"),
                risk_pass=False,
                order_amount=row.get("estimated_amount"),
                price=row.get("expected_execution_price"),
                qty=row.get("order_qty"),
                final_amount=row.get("order_amount"),
                blocked=True,
                blocked_reason=row.get("order_block_reason"),
                extra={"side": side, "submit_attempted": False, "broker_result": "POLICY_BLOCK"},
            )
            row["trade_flow"] = trade_flow
            append_trade_flow_log(trade_flow)
            skipped_count += 1
            results.append(row)
            continue

        try:
            ord_dvsn = _ord_dvsn(row)
            ord_unpr = _ord_unpr(row, ord_dvsn)
            row["final_guard_details"] = guard_details
            if debug_trade_mode:
                row["submitted_at"] = datetime.now().isoformat(timespec="seconds")
                row["order_status"] = "debug_skipped"
                row["reconciliation_status"] = "debug_trade_mode_skip"
                row["order_block_reason"] = _append_reason(row.get("order_block_reason"), "debug_trade_mode")
                row["block_reason_details"] = build_block_reason_details(str(row.get("order_block_reason") or "").split(";"))
                row["primary_block_reason"] = select_primary_block_reason(row["block_reason_details"])
                trade_flow = build_trade_flow_payload(
                    stage="execution_debug_skip",
                    run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                    symbol=row.get("code"),
                    score=row.get("signal_strength"),
                    strategy_action=row.get("portfolio_action"),
                    risk_pass=True,
                    order_amount=row.get("estimated_amount"),
                    price=row.get("expected_execution_price"),
                    qty=row.get("order_qty"),
                    final_amount=row.get("order_amount"),
                    blocked=False,
                    blocked_reason="debug_trade_mode",
                    extra={"side": side, "submit_attempted": False, "broker_result": "DEBUG_SKIP"},
                )
                row["trade_flow"] = trade_flow
                append_trade_flow_log(trade_flow)
                results.append(row)
                continue
            _log(
                f"approve submit code={row.get('code')} name={row.get('name')} side={side} qty={row.get('order_qty')} "
                f"estimated_amount={row.get('estimated_amount')} run_mode={preview.get('run_mode')} "
                f"order_type={row.get('order_type')} ord_dvsn={ord_dvsn} ord_unpr={ord_unpr} "
                f"daily_used={row.get('daily_order_amount_used')}"
            )
            row["submit_attempted"] = True
            code_str = str(row.get("code") or row.get("symbol") or "").zfill(6)
            response_df, attempt_count, retry_reasons = _submit_order_with_retry(
                client,
                account,
                side=side.lower(),
                pdno=code_str,
                ord_dvsn=ord_dvsn,
                ord_qty=str(int(float(row.get("order_qty") or 0))),
                ord_unpr=ord_unpr,
                code=code_str,
            )
            response_row = response_df.iloc[0].to_dict() if not response_df.empty else {}
            row["submitted_at"] = datetime.now().isoformat(timespec="seconds")
            row["raw_response"] = response_row
            row["broker_order_id"] = response_row.get("ODNO") or response_row.get("odno")
            row["broker_org_order_id"] = response_row.get("KRX_FWDG_ORD_ORGNO") or response_row.get("krx_fwdg_ord_orgno")
            row["submit_attempt_count"] = attempt_count
            row["submit_retry_reasons"] = retry_reasons
            row["order_status"] = "submitted"
            row["reconciliation_status"] = "submitted_pending_fill_check"
            submitted_count += 1
            order_amount = float(row.get("order_amount") or 0.0)
            if side == "BUY":
                submitted_buy_amount += order_amount
            elif side == "SELL":
                submitted_sell_amount += order_amount
            row["block_reason_details"] = build_block_reason_details(str(row.get("order_block_reason") or "").split(";"))
            row["primary_block_reason"] = select_primary_block_reason(row["block_reason_details"])
            trade_flow = build_trade_flow_payload(
                stage="execution_submit",
                run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                symbol=row.get("code"),
                score=row.get("signal_strength"),
                strategy_action=row.get("portfolio_action"),
                risk_pass=True,
                order_amount=row.get("estimated_amount"),
                price=row.get("expected_execution_price"),
                qty=row.get("order_qty"),
                final_amount=row.get("order_amount"),
                blocked=False,
                blocked_reason="none",
                extra={"side": side, "submit_attempted": True, "broker_result": "SUBMITTED"},
            )
            row["trade_flow"] = trade_flow
            append_trade_flow_log(trade_flow)
            _log(
                f"submitted code={row.get('code')} name={row.get('name')} side={side} qty={row.get('order_qty')} "
                f"estimated_amount={row.get('estimated_amount')} broker_order_id={row.get('broker_order_id')}"
            )
        except Exception as exc:
            row["submitted_at"] = datetime.now().isoformat(timespec="seconds")
            retryable, _ = _is_retryable_order_error(exc)
            failure_tag = "order_submit_failed"
            # 타임아웃·5xx는 주문이 KIS에 도달했을 수 있으므로 별도 표시
            if not retryable and not isinstance(exc, (KISAuthError, ConnectionRefusedError)):
                failure_tag = "order_submit_failed_possibly_sent"
            row["order_block_reason"] = _append_reason(row.get("order_block_reason"), failure_tag, str(exc))
            row["submit_attempt_count"] = row.get("submit_attempt_count") or 1
            row["submit_retry_reasons"] = row.get("submit_retry_reasons") or []
            row["order_status"] = "failed"
            row["reconciliation_status"] = "submit_failed"
            row["block_reason_details"] = build_block_reason_details(str(row.get("order_block_reason") or "").split(";"))
            row["primary_block_reason"] = select_primary_block_reason(row["block_reason_details"])
            trade_flow = build_trade_flow_payload(
                stage="execution_api_error",
                run_id=str(preview.get("run_id") or row.get("run_id") or ""),
                symbol=row.get("code"),
                score=row.get("signal_strength"),
                strategy_action=row.get("portfolio_action"),
                risk_pass=False,
                order_amount=row.get("estimated_amount"),
                price=row.get("expected_execution_price"),
                qty=row.get("order_qty"),
                final_amount=row.get("order_amount"),
                blocked=True,
                blocked_reason=row.get("order_block_reason"),
                extra={"side": side, "submit_attempted": True, "broker_result": "API_ERROR", "broker_error_message": str(exc)},
            )
            row["trade_flow"] = trade_flow
            append_trade_flow_log(trade_flow)
            _log(
                f"submit failed code={row.get('code')} name={row.get('name')} side={side} qty={row.get('order_qty')} "
                f"estimated_amount={row.get('estimated_amount')} run_mode={preview.get('run_mode')} error={exc}"
            )
            failed_count += 1
        results.append(row)

        # KIS EGW00201 (초당 거래건수 초과) 방지: 주문 제출 시도 후 딜레이
        # RULE_ORDER_INTERVAL_SEC 환경변수로 조정 가능 (기본 0.5초)
        if row.get("submit_attempted"):
            _interval = float(os.environ.get("RULE_ORDER_INTERVAL_SEC", "0.5"))
            if _interval > 0:
                time.sleep(_interval)

    return results, {
        "requested_count": len(results),
        "request_count": len(results),
        "submitted_count": submitted_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "debug_skipped_count": sum(1 for row in results if row.get("order_status") == "debug_skipped"),
        "filled_count": 0,
        "partial_filled_count": 0,
        "unfilled_count": 0,
        "canceled_count": 0,
        "simulated_filled_count": 0,
        "simulated_unfilled_count": 0,
        "buy_submitted_amount": submitted_buy_amount,
        "sell_submitted_amount": submitted_sell_amount,
    }


def main() -> None:
    args = parse_args()
    preview = load_json(args.order_preview_json, {})
    state = default_state()

    out_results = resolve(args.out_results_json)
    out_recon = resolve(args.out_reconciliation_md)
    out_execution_history = resolve(args.out_execution_history_jsonl)
    out_market_snapshot = resolve(args.out_market_snapshot_json)
    calendar = load_calendar(args.calendar_json)
    session_status = validate_trading_session(datetime.now(), str(preview.get("as_of_date") or ""), calendar)
    reconciliation_status = validate_previous_execution_completed(out_results, out_recon, str(preview.get("as_of_date") or ""))

    if not preview:
        results = build_aborted_results(
            {},
            state,
            "order_preview_missing",
            {"trading_day_valid": None, "trading_day_reason": "preview_missing"},
            reconciliation_status,
        )
    elif str(preview.get("run_mode") or "").lower() not in {"pilot", "live"}:
        results = build_aborted_results(
            preview,
            state,
            "order_submitter_requires_pilot_or_live",
            session_status,
            reconciliation_status,
        )
    elif not session_status.get("session_valid"):
        results = build_aborted_results(
            preview,
            state,
            str(session_status.get("session_reason") or "session_invalid"),
            session_status,
            reconciliation_status,
        )
    elif reconciliation_status.get("new_orders_blocked_by_reconciliation"):
        results = build_aborted_results(
            preview,
            state,
            str(reconciliation_status.get("reconciliation_block_reason") or "reconciliation_guard_blocked"),
            session_status,
            reconciliation_status,
        )
    else:
        try:
            buy_symbols = [
                str(item.get("code") or item.get("symbol") or "").zfill(6)
                for item in preview.get("items") or []
                if str(item.get("side") or "").upper() == "BUY"
            ]
            _log(
                f"start submit run_mode={preview.get('run_mode')} as_of_date={preview.get('as_of_date')} "
                f"buy_symbol_count={len(buy_symbols)}"
            )
            market_snapshot = _build_market_snapshot(buy_symbols, out_market_snapshot)
            if str(market_snapshot.get("api_health_status") or "").strip().lower() == "auth_failed":
                raise RuntimeError(str(market_snapshot.get("api_failure_reason") or "kis_auth_failed"))
            items, summary = _submit_items(preview, market_snapshot)
            results = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "as_of_date": preview.get("as_of_date"),
                "account_id": preview.get("account_id"),
                "strategy_id": preview.get("strategy_id"),
                "engine_type": preview.get("engine_type"),
                "run_mode": preview.get("run_mode"),
                "run_id": preview.get("run_id"),
                "debug_trade_mode": debug_trade_mode_enabled(),
                "paper_only": False,
                "trading_day_valid": session_status.get("trading_day_valid"),
                "trading_day_reason": session_status.get("trading_day_reason"),
                "market_data_available": market_snapshot.get("market_data_available"),
                "api_health_status": market_snapshot.get("api_health_status"),
                "api_failure_reason": market_snapshot.get("api_failure_reason"),
                "order_run_aborted": False,
                "order_run_abort_reason": None,
                "previous_reconciliation_found": reconciliation_status.get("previous_reconciliation_found"),
                "previous_reconciliation_status": reconciliation_status.get("previous_reconciliation_status"),
                "new_orders_blocked_by_reconciliation": reconciliation_status.get("new_orders_blocked_by_reconciliation"),
                "reconciliation_block_reason": reconciliation_status.get("reconciliation_block_reason"),
                "items": items,
                "summary": summary,
            }
        except Exception as exc:
            _log(
                f"abort submit run_mode={preview.get('run_mode')} as_of_date={preview.get('as_of_date')} reason={exc}"
            )
            results = build_aborted_results(
                preview,
                state,
                str(exc),
                session_status,
                reconciliation_status,
            )

    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_recon.write_text(render_reconciliation(results, state), encoding="utf-8")
    append_jsonl(out_execution_history, results)
    print(f"saved {out_results}")
    print(f"saved {out_recon}")
    print(f"appended {out_execution_history}")


if __name__ == "__main__":
    main()
