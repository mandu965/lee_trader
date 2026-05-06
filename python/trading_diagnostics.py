from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from typing import Any


KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(KST)


def generate_run_id(*, prefix: str = "live-auto", now: datetime | None = None) -> str:
    ts = now or now_kst()
    return f"{prefix}-{ts.strftime('%Y%m%d-%H%M%S')}"


def build_market_context(now: datetime | None = None) -> dict[str, Any]:
    kst_now = now.astimezone(KST) if now is not None and now.tzinfo else (now or now_kst())
    utc_now = kst_now.astimezone(UTC)
    weekday = kst_now.weekday()
    hour_min = kst_now.hour * 60 + kst_now.minute
    if weekday >= 5:
        status = "HOLIDAY"
        status_ko = "휴장"
    elif hour_min < 9 * 60:
        status = "PREOPEN"
        status_ko = "장전"
    elif hour_min < 15 * 60 + 30:
        status = "OPEN"
        status_ko = "장중"
    else:
        status = "AFTER_HOURS"
        status_ko = "장후"
    return {
        "market_status": status,
        "market_status_ko": status_ko,
        "is_trading_day": weekday < 5,
        "server_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "utc_timestamp": utc_now.isoformat(timespec="seconds"),
        "kst_timestamp": kst_now.isoformat(timespec="seconds"),
        "kst_date": kst_now.strftime("%Y-%m-%d"),
        "kst_time": kst_now.strftime("%H:%M:%S"),
        "timezone": "Asia/Seoul",
    }


def split_reason_text(reason: object) -> list[str]:
    text = str(reason or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part and part.strip()]


def extract_live_grade(*, raw_reason: object = None, fallback: object = None) -> str | None:
    fallback_text = str(fallback or "").strip().upper()
    if fallback_text:
        return fallback_text
    for token in split_reason_text(raw_reason):
        match = re.search(r"live grade\s+([A-D])", token, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _diag(
    *,
    block_type: str,
    severity: str,
    message_ko: str,
    raw_reason: str,
    recommended_action: str,
    policy_status: str | None = None,
) -> dict[str, Any]:
    return {
        "policy_status": policy_status,
        "block_type": block_type,
        "severity": severity,
        "user_message_ko": message_ko,
        "message_ko": message_ko,
        "raw_reason": raw_reason,
        "recommended_action": recommended_action,
    }


def build_policy_diagnostics(
    *,
    raw_reason: object,
    intent_type: object = None,
    blocked_reason: object = None,
    entry_quality_status: object = None,
    live_grade: object = None,
) -> dict[str, Any]:
    tokens = split_reason_text(raw_reason)
    diagnostics: list[dict[str, Any]] = []
    side = str(intent_type or "").upper()
    resolved_live_grade = extract_live_grade(raw_reason=raw_reason, fallback=live_grade)

    is_buy = side == "BUY"

    for token in tokens:
        lower = token.lower()
        if "live grade" in lower and "block new buy" in lower:
            diagnostics.append(
                _diag(
                    policy_status="BLOCK" if is_buy else "SELL_ONLY",
                    block_type="LIVE_GRADE_BLOCK",
                    severity="BLOCKED" if is_buy else "INFO",
                    message_ko=(
                        f"live grade {resolved_live_grade or '?'} 상태로 신규 매수가 차단되었습니다."
                        if is_buy
                        else f"live grade {resolved_live_grade or '?'} 상태라 신규 매수는 막혀 있고 현재 결정은 매도/축소 중심입니다."
                    ),
                    raw_reason=token,
                    recommended_action=(
                        "최근 실거래 성과와 live grade 기준을 확인하세요."
                        if is_buy
                        else "보유 축소/정리 사유와 신규 매수 차단 사유를 분리해서 확인하세요."
                    ),
                )
            )
        elif "recent_10_trade_return_below_minus_2pct" in lower:
            diagnostics.append(
                _diag(
                    policy_status="BLOCK" if is_buy else "SELL_ONLY",
                    block_type="RISK_LIMIT",
                    severity="BLOCKED" if is_buy else "INFO",
                    message_ko=(
                        "최근 10건 거래 수익률이 -2% 이하라 신규 매수가 차단되었습니다."
                        if is_buy
                        else "최근 실거래 성과가 약해 신규 매수는 차단되어 있고 현재 결정은 매도/축소 중심입니다."
                    ),
                    raw_reason=token,
                    recommended_action=(
                        "실거래 표본이 더 쌓일 때까지 관찰하거나 성과 기준을 점검하세요."
                        if is_buy
                        else "실거래 성과 기준과 현재 보유 조정 사유를 함께 확인하세요."
                    ),
                )
            )
        elif "confidence unavailable" in lower:
            diagnostics.append(
                _diag(
                    policy_status="BLOCK" if is_buy else "SELL_ONLY",
                    block_type="CONFIDENCE_MISSING",
                    severity="BLOCKED" if is_buy else "INFO",
                    message_ko=(
                        "신뢰도 점수를 계산할 수 없어 신규 매수가 차단되었습니다."
                        if is_buy
                        else "신뢰도 부족으로 신규 매수는 차단되어 있고 현재 결정은 매도/축소 중심입니다."
                    ),
                    raw_reason=token,
                    recommended_action=(
                        "confidence 산출 입력 데이터와 최신 보정 파일을 확인하세요."
                        if is_buy
                        else "confidence 입력 데이터와 현재 보유 조정 판단을 함께 확인하세요."
                    ),
                )
            )
        elif lower.startswith("entry_quality_status=watch"):
            diagnostics.append(
                _diag(
                    policy_status="WATCH",
                    block_type="POLICY_BLOCK",
                    severity="WARNING",
                    message_ko="진입 품질이 WATCH 상태여서 우선 관찰 대상으로 분류되었습니다.",
                    raw_reason=token,
                    recommended_action="entry quality 이유와 가격/유동성 상태를 함께 확인하세요.",
                )
            )
        elif lower.startswith("target_weight_fallback="):
            diagnostics.append(
                _diag(
                    policy_status="WATCH",
                    block_type="POLICY_BLOCK",
                    severity="INFO",
                    message_ko="목표 비중 기본값이 적용되었습니다.",
                    raw_reason=token,
                    recommended_action="target weight 산출 입력이 누락되지 않았는지 확인하세요.",
                )
            )

    blocked_text = str(blocked_reason or "").strip()
    if blocked_text:
        lower = blocked_text.lower()
        if "market_closed" in lower:
            diagnostics.append(
                _diag(
                    policy_status="BLOCK",
                    block_type="MARKET_TIME_BLOCK",
                    severity="BLOCKED",
                    message_ko="시장 운영 시간이 아니라 주문이 차단되었습니다.",
                    raw_reason=blocked_text,
                    recommended_action="KST 기준 주문 가능 시간을 확인하세요.",
                )
            )
        elif lower in {"live_price_unavailable", "previous_close_unavailable"} or lower.startswith("entry_gap_"):
            diagnostics.append(
                _diag(
                    policy_status="WATCH",
                    block_type="MARKET_TIME_BLOCK",
                    severity="WARNING",
                    message_ko="실시간 가격 조건을 만족하지 않아 주문이 보류되었습니다.",
                    raw_reason=blocked_text,
                    recommended_action="장중 시세 수집과 진입 가격 가드 기준을 확인하세요.",
                )
            )

    if not diagnostics and side in {"TRIM", "EXIT", "SELL"}:
        diagnostics.append(
            _diag(
                policy_status="SELL_ONLY",
                block_type="POLICY_BLOCK",
                severity="INFO",
                message_ko="현재 결정은 신규 매수보다 보유 축소 또는 매도 중심입니다.",
                raw_reason=str(raw_reason or ""),
                recommended_action="보유 비중과 교체 사유를 확인하세요.",
            )
        )

    if not diagnostics and str(entry_quality_status or "").upper() == "WATCH":
        diagnostics.append(
            _diag(
                policy_status="WATCH",
                block_type="POLICY_BLOCK",
                severity="WARNING",
                message_ko="진입 품질이 WATCH 상태입니다.",
                raw_reason=str(raw_reason or ""),
                recommended_action="entry quality 상세 사유를 확인하세요.",
            )
        )

    severity_rank = {"ERROR": 4, "BLOCKED": 3, "WARNING": 2, "INFO": 1}
    diagnostics = sorted(
        diagnostics,
        key=lambda item: severity_rank.get(str(item.get("severity") or "").upper(), 0),
        reverse=True,
    )

    policy_status = "ALLOW"
    if any(item["severity"] == "BLOCKED" for item in diagnostics):
        policy_status = "BLOCK"
    elif any(item["policy_status"] == "SELL_ONLY" for item in diagnostics):
        policy_status = "SELL_ONLY"
    elif any(item["severity"] == "WARNING" for item in diagnostics):
        policy_status = "WATCH"

    primary = diagnostics[0] if diagnostics else _diag(
        policy_status="ALLOW",
        block_type="POLICY_BLOCK",
        severity="INFO",
        message_ko="정책 차단 사유가 없습니다.",
        raw_reason=str(raw_reason or ""),
        recommended_action="현재 정책 기준상 추가 확인만 필요합니다.",
    )
    primary["policy_status"] = policy_status
    return {
        "policy_status": policy_status,
        "block_type": primary["block_type"],
        "severity": primary["severity"],
        "user_message_ko": primary["user_message_ko"],
        "raw_reason": str(raw_reason or ""),
        "recommended_action": primary["recommended_action"],
        "diagnostics": diagnostics,
    }


def parse_broker_error(error_text: object) -> dict[str, Any]:
    text = str(error_text or "").strip()
    payload = {
        "raw_error": text,
        "broker_error_code": None,
        "broker_error_message": None,
        "rt_cd": None,
    }
    if not text:
        return payload
    rt_match = re.search(r"rt_cd=([A-Za-z0-9_-]+)", text)
    code_match = re.search(r"msg_cd=([A-Za-z0-9_-]+)", text)
    msg_match = re.search(r"msg1=(.+)$", text)
    if rt_match:
        payload["rt_cd"] = rt_match.group(1)
    if code_match:
        payload["broker_error_code"] = code_match.group(1)
    if msg_match:
        payload["broker_error_message"] = msg_match.group(1).strip()
    return payload


def build_broker_diagnostic(
    *,
    error_text: object,
    env_dv: object,
    market_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_broker_error(error_text)
    code = parsed.get("broker_error_code")
    message = parsed.get("broker_error_message") or str(error_text or "")
    inferred_causes: list[str] = []
    actions: list[str] = []
    if code == "APBK0919":
        inferred_causes = [
            "장 종료 후 주문",
            "서버 날짜/시간 또는 KST 변환 문제",
            "휴장일 또는 거래일 처리 문제",
            f"mock/live 환경 날짜 불일치 가능성 ({env_dv or 'unknown'})",
        ]
        actions = [
            "서버 timezone 설정 확인",
            "KST 기준 장운영일 확인",
            "주문 가능 시간 확인",
            "KIS mock/live 설정 확인",
        ]
    return {
        "policy_status": "BLOCK",
        "block_type": "BROKER_REJECT",
        "severity": "ERROR",
        "user_message_ko": f"증권사 주문이 거절되었습니다: {message}",
        "raw_reason": str(error_text or ""),
        "recommended_action": actions[0] if actions else "브로커 응답 전문을 확인하세요.",
        "broker_error_code": code,
        "broker_error_message": message,
        "rt_cd": parsed.get("rt_cd"),
        "inferred_causes": inferred_causes,
        "actions": actions,
        "market_context": market_context or build_market_context(),
    }
