from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rule_signal_builder import ROOT


OUTPUT_DIR = ROOT / "outputs"
DEFAULT_TRADE_FLOW_LOG = OUTPUT_DIR / "rule_trade_flow.jsonl"

DEBUG_ENV_NAME = "DEBUG_TRADE_MODE"
DEFAULT_TZ_NAME = "Asia/Seoul"

BLOCK_REASON = {
    "WEEKLY_LOSS_LIMIT": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "주간 손실 제한에 도달해 신규 주문이 차단되었습니다.",
        "recommended_action": "주간 손익 계산 기준과 계좌 스냅샷을 먼저 확인하세요.",
    },
    "ORDER_QTY_ZERO": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "계산된 주문 수량이 0주라서 주문을 만들 수 없습니다.",
        "recommended_action": "주문 금액과 현재가, 최소 주문 수량을 확인하세요.",
    },
    "MIN_ORDER_AMOUNT": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "최종 주문 금액이 최소 주문 금액보다 작아 차단되었습니다.",
        "recommended_action": "최소 주문 금액 설정과 주문 금액 계산식을 확인하세요.",
    },
    "PRICE_TOO_HIGH": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "현재가가 주문 가능 금액보다 높아 최소 1주도 살 수 없습니다.",
        "recommended_action": "주문 금액을 늘리거나 더 낮은 가격대 종목을 선택하세요.",
    },
    "CASH_SHORTAGE": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "가용 현금 또는 한도 부족으로 주문이 차단되었습니다.",
        "recommended_action": "현금 잔고와 일간/주간 주문 한도를 확인하세요.",
    },
    "MARKET_CLOSED": {
        "category": "POLICY_BLOCK",
        "severity": "WARNING",
        "user_message_ko": "현재 주문 가능한 장 시간이 아니어서 주문이 차단되었습니다.",
        "recommended_action": "장중 여부와 거래일 캘린더를 확인하세요.",
    },
    "INVALID_PRICE": {
        "category": "SYSTEM_ERROR",
        "severity": "ERROR",
        "user_message_ko": "유효한 가격을 계산하지 못해 주문을 생성할 수 없습니다.",
        "recommended_action": "시세 데이터와 예상 체결가 계산 결과를 확인하세요.",
    },
    "API_ERROR": {
        "category": "API_ERROR",
        "severity": "ERROR",
        "user_message_ko": "증권사 주문 API 호출 중 오류가 발생했습니다.",
        "recommended_action": "브로커 응답 코드와 네트워크/인증 상태를 확인하세요.",
    },
    "GAP_RISK_NEGATIVE": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "당일 실제 시가가 전일 기준가 대비 -4% 초과 하락하여 갭 하락 방어 정책에 의해 매수 차단되었습니다.",
        "recommended_action": "장 시작 시가와 전일 기준가를 비교하고, 갭 차단 임계값 설정을 확인하세요.",
    },
    "GAP_RISK_POSITIVE": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "당일 실제 시가가 전일 기준가 대비 +5% 초과 상승하여 과열 진입 방어 정책에 의해 매수 차단되었습니다.",
        "recommended_action": "갭 상승 임계값 설정과 해당 종목의 시가 데이터를 확인하세요.",
    },
    "GAP_RISK_UNAVAILABLE": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "실제 시가 또는 전일 기준가를 확인할 수 없어 갭 계산이 불가능해 안전 차단되었습니다.",
        "recommended_action": "장 시작 시가 스냅샷 파일(rule_market_open_snapshot.json)과 KIS API 응답을 확인하세요.",
    },
    "MARKET_SNAPSHOT_UNMAPPED": {
        "category": "POLICY_BLOCK",
        "severity": "BLOCKED",
        "user_message_ko": "장 시작 시가 스냅샷에서 해당 종목을 찾지 못해 주문이 안전 차단되었습니다. (매핑 실패)",
        "recommended_action": "종목코드 6자리 정규화 여부와 market snapshot 수집 결과를 확인하세요.",
    },
    "NO_ORDER_ACTION": {
        "category": "POLICY_BLOCK",
        "severity": "INFO",
        "user_message_ko": "해당 종목은 주문 대상이 아닙니다 (NONE 액션).",
        "recommended_action": "시그널 결과와 포트폴리오 액션을 확인하세요.",
    },
}

RAW_REASON_TO_BLOCK_REASON = {
    "weekly_loss_limit_reached": "WEEKLY_LOSS_LIMIT",
    "order_qty_zero": "ORDER_QTY_ZERO",
    "final_order_amount_below_min_order_amount": "MIN_ORDER_AMOUNT",
    "price_too_high_for_minimum_shares": "PRICE_TOO_HIGH",
    "cash_limit_failed": "CASH_SHORTAGE",
    "daily_buy_amount_limit_exceeded": "CASH_SHORTAGE",
    "weekly_buy_amount_limit_exceeded": "CASH_SHORTAGE",
    "order_submit_failed": "API_ERROR",
    "order_submit_failed_possibly_sent": "API_ERROR",
    "limit_price_missing": "INVALID_PRICE",
    "market_status_missing": "MARKET_CLOSED",
    # Gap risk — 실제 갭 하락/상승
    "actual_open_gap_lt_minus_4pct": "GAP_RISK_NEGATIVE",
    "actual_open_gap_lt_threshold": "GAP_RISK_NEGATIVE",
    "actual_open_gap_gt_5pct": "GAP_RISK_POSITIVE",
    "actual_open_gap_gt_threshold": "GAP_RISK_POSITIVE",
    # Gap risk — 데이터 없음 (실제 갭 하락과 구분)
    "actual_open_gap_unavailable": "GAP_RISK_UNAVAILABLE",
    "actual_open_price_unavailable": "GAP_RISK_UNAVAILABLE",
    "prev_close_unavailable": "GAP_RISK_UNAVAILABLE",
    "actual_open_gap_blocked": "GAP_RISK_UNAVAILABLE",
    # Snapshot 매핑 실패 (종목코드 미발견)
    "market_snapshot_unmapped": "MARKET_SNAPSHOT_UNMAPPED",
    # 주문 액션 없음
    "no_order_action": "NO_ORDER_ACTION",
}


def debug_trade_mode_enabled() -> bool:
    return str(os.getenv(DEBUG_ENV_NAME, "0")).strip().lower() in {"1", "true", "yes", "on"}


def generate_rule_run_id(prefix: str = "rule-trade") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _sanitize_token(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace(" ", "_")


def append_trade_flow_log(payload: dict[str, Any], out_path: Path | None = None) -> None:
    target = out_path or DEFAULT_TRADE_FLOW_LOG
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    log_line = (
        "[TRADE_FLOW] "
        f"stage={_sanitize_token(payload.get('stage'))} "
        f"run_id={_sanitize_token(payload.get('run_id'))} "
        f"symbol={_sanitize_token(payload.get('symbol'))} "
        f"score={_sanitize_token(payload.get('score'))} "
        f"strategy_action={_sanitize_token(payload.get('strategy_action'))} "
        f"risk_pass={_sanitize_token(payload.get('risk_pass'))} "
        f"order_amount={_sanitize_token(payload.get('order_amount'))} "
        f"price={_sanitize_token(payload.get('price'))} "
        f"qty={_sanitize_token(payload.get('qty'))} "
        f"final_amount={_sanitize_token(payload.get('final_amount'))} "
        f"blocked={_sanitize_token(payload.get('blocked'))} "
        f"blocked_reason={_sanitize_token(payload.get('blocked_reason'))}"
    )
    print(log_line)


def build_trade_flow_payload(
    *,
    stage: str,
    run_id: str,
    symbol: Any,
    score: Any,
    strategy_action: Any,
    risk_pass: Any,
    order_amount: Any,
    price: Any,
    qty: Any,
    final_amount: Any,
    blocked: Any,
    blocked_reason: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "run_id": run_id,
        "symbol": str(symbol or "").strip() or None,
        "score": score,
        "strategy_action": strategy_action,
        "risk_pass": risk_pass,
        "order_amount": order_amount,
        "price": price,
        "qty": qty,
        "final_amount": final_amount,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
    }
    if extra:
        payload.update(extra)
    return payload


def build_order_sizing_details(
    *,
    side: str,
    base_order_amount: float,
    adjusted_order_amount: float,
    current_price: float | None,
    raw_qty: int,
    final_qty: int,
    min_order_amount: float,
    minimum_shares: int,
    required_amount: float | None,
    adjustment_applied: bool,
    price_too_high: bool,
) -> dict[str, Any]:
    final_order_amount = adjusted_order_amount if final_qty > 0 else adjusted_order_amount
    return {
        "side": side,
        "base_order_amount": base_order_amount,
        "adjusted_order_amount": adjusted_order_amount,
        "current_price": current_price,
        "raw_qty": raw_qty,
        "final_qty": final_qty,
        "min_order_amount": min_order_amount,
        "final_order_amount": final_order_amount,
        "minimum_shares": minimum_shares,
        "required_amount": required_amount,
        "adjustment_applied": adjustment_applied,
        "price_too_high": price_too_high,
    }


def normalize_block_reason(raw_reason: Any) -> str | None:
    reason = str(raw_reason or "").strip()
    if not reason or reason == "none":
        return None
    return RAW_REASON_TO_BLOCK_REASON.get(reason, None)


def build_block_reason_detail(raw_reason: Any) -> dict[str, Any]:
    normalized = normalize_block_reason(raw_reason)
    meta = BLOCK_REASON.get(normalized or "", {})
    return {
        "raw_reason": str(raw_reason or "").strip() or None,
        "block_reason": normalized or "UNMAPPED",
        "category": meta.get("category", "POLICY_BLOCK"),
        "severity": meta.get("severity", "INFO"),
        "user_message_ko": meta.get("user_message_ko", "운영자가 원문 차단 사유를 확인해야 합니다."),
        "recommended_action": meta.get("recommended_action", "원문 로그와 입력 데이터를 확인하세요."),
    }


def build_block_reason_details(reasons: list[Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason in reasons:
        key = str(reason or "").strip()
        if not key or key == "none" or key in seen:
            continue
        details.append(build_block_reason_detail(key))
        seen.add(key)
    return details


def select_primary_block_reason(details: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not details:
        return None
    priority = {
        "WEEKLY_LOSS_LIMIT": 100,
        "PRICE_TOO_HIGH": 90,
        "ORDER_QTY_ZERO": 80,
        "MIN_ORDER_AMOUNT": 70,
        "CASH_SHORTAGE": 60,
        "MARKET_CLOSED": 50,
        "INVALID_PRICE": 45,
        "API_ERROR": 40,
        "GAP_RISK_NEGATIVE": 35,
        "GAP_RISK_POSITIVE": 35,
        "GAP_RISK_UNAVAILABLE": 30,
        "MARKET_SNAPSHOT_UNMAPPED": 28,
        "NO_ORDER_ACTION": 5,
    }
    return sorted(details, key=lambda item: priority.get(str(item.get("block_reason")), 0), reverse=True)[0]
