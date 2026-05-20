"""AI 자동매매 포지션 리스크 평가 및 상태 추적.

매일 live_account_holdings.csv를 읽어 positions의 entry_date/peak_price를
outputs/ai_position_state.json에 누적 기록하고,
stop_loss / trailing_stop / max_holding_days 기준으로 청산 신호를 반환한다.

주요 환경변수:
  AI_POSITION_RISK_ENABLED      (기본 true)  — 전체 on/off
  AI_POSITION_STOP_LOSS_PCT     (기본 0.07)  — 7% 손절
  AI_TRAILING_STOP_PCT          (기본 0.05)  — 고점 대비 5% 하락 시 청산
  AI_TRAILING_STOP_MIN_PROFIT   (기본 0.03)  — trailing 발동 최소 수익 3%
  AI_MAX_HOLDING_DAYS           (기본 15)    — 최대 보유일 (pnl < 2% 시 청산)
  AI_MAX_HOLDING_DAYS_HARD_CAP  (기본 45)    — 하드캡: pnl >= 2%여도 초과 시 강제 청산
  AI_POSITION_TAKE_PROFIT_PCT   (기본 0.0)   — 익절 기준 (0=비활성)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AI_POSITION_STATE_PATH = ROOT / "outputs" / "ai_position_state.json"

logger = logging.getLogger(__name__)


# ── 환경변수 헬퍼 ──────────────────────────────────────────────────────────────

def _cfg_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _cfg_bool(name: str, default: bool = True) -> bool:
    val = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return val in {"1", "true", "yes", "on"}


# ── 상태 파일 I/O ─────────────────────────────────────────────────────────────

def load_ai_position_state() -> dict:
    """outputs/ai_position_state.json 로드. 없으면 빈 구조 반환."""
    if not AI_POSITION_STATE_PATH.exists():
        return {"positions": {}}
    try:
        return json.loads(AI_POSITION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ai_position_state load failed: %s — empty state used", exc)
        return {"positions": {}}


def save_ai_position_state(state: dict) -> None:
    """상태를 파일에 저장. 실패해도 파이프라인을 중단하지 않는다."""
    try:
        state["generated_at"] = datetime.now().isoformat(timespec="seconds")
        AI_POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AI_POSITION_STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("ai_position_state save failed: %s", exc)


# ── 상태 갱신 ─────────────────────────────────────────────────────────────────

def update_position_state_from_holdings(holdings_df: pd.DataFrame) -> dict:
    """sync_live_account_holdings 실행 후 호출.

    신규 포지션은 entry_date/entry_price 기록,
    기존 포지션은 peak_price 갱신,
    holdings에서 사라진 포지션은 제거한다.

    상태 파일이 유실된 경우, 현재 보유 종목은 state_recovered=True로 마킹되어
    max_holding_days 체크가 억제된다. (하드캡은 여전히 적용)
    """
    state = load_ai_position_state()
    positions: dict[str, dict] = state.get("positions") or {}
    # 기존 포지션이 없으면 상태 파일이 유실된 것으로 간주
    state_was_empty = not positions
    today = datetime.now().strftime("%Y-%m-%d")

    current_codes: set[str] = set()
    for _, row in holdings_df.iterrows():
        code = str(row.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        current_codes.add(code)

        avg_price = float(row.get("avg_price") or 0)
        current_price = float(row.get("current_price") or 0)

        if code not in positions:
            state_recovered = state_was_empty
            if state_recovered:
                logger.warning(
                    "AI_POSITION_STATE_RECOVERED | code=%s — state file was missing, "
                    "entry_date set to today(%s); max_holding_days check suppressed until "
                    "AI_MAX_HOLDING_DAYS_HARD_CAP",
                    code, today,
                )
            else:
                logger.info(
                    "AI_POSITION_STATE_NEW | code=%s entry_date=%s entry_price=%.0f",
                    code, today, avg_price,
                )
            positions[code] = {
                "code": code,
                "name": str(row.get("name") or ""),
                "entry_date": today,
                "entry_price": avg_price,
                "peak_price": max(current_price, avg_price),
                "peak_date": today,
                "first_seen_at": datetime.now().isoformat(timespec="seconds"),
                "state_recovered": state_recovered,
            }
        else:
            existing = positions[code]
            old_peak = float(existing.get("peak_price") or 0)
            if current_price > old_peak:
                existing["peak_price"] = current_price
                existing["peak_date"] = today

    # 청산된 포지션 정리
    closed = set(positions.keys()) - current_codes
    for code in closed:
        logger.info("AI_POSITION_STATE_CLOSED | code=%s", code)
        del positions[code]

    state["positions"] = positions
    save_ai_position_state(state)
    return state


# ── 리스크 평가 ───────────────────────────────────────────────────────────────

def evaluate_ai_position_risk(
    code: str,
    pnl_pct: float | None,
    position_state: dict | None = None,
    as_of_date: str | None = None,
) -> tuple[str | None, str | None]:
    """AI 포지션 리스크 평가.

    Returns:
        (action, reason) — action은 "exit" / "reduce" / None
    """
    if not _cfg_bool("AI_POSITION_RISK_ENABLED", True):
        return None, None

    stop_loss_pct = _cfg_float("AI_POSITION_STOP_LOSS_PCT", 0.07)
    trailing_stop_pct = _cfg_float("AI_TRAILING_STOP_PCT", 0.05)
    trailing_stop_min_profit = _cfg_float("AI_TRAILING_STOP_MIN_PROFIT", 0.03)
    max_holding_days = _cfg_int("AI_MAX_HOLDING_DAYS", 15)
    take_profit_pct = _cfg_float("AI_POSITION_TAKE_PROFIT_PCT", 0.0)

    # 1. Stop loss ─────────────────────────────────────────────────────────────
    if pnl_pct is not None and pnl_pct <= -abs(stop_loss_pct):
        reason = (
            f"stop_loss_exit pnl={pnl_pct:.2%} <= -{stop_loss_pct:.0%}"
        )
        logger.warning(
            "AI_POSITION_RISK_BLOCK | code=%s reason=%s", code, reason
        )
        return "exit", reason

    pos = ((position_state or {}).get("positions") or {}).get(code, {})

    # 2. Trailing stop ─────────────────────────────────────────────────────────
    entry_price = float(pos.get("entry_price") or 0)
    peak_price = float(pos.get("peak_price") or 0)
    if (
        entry_price > 0
        and peak_price > 0
        and pnl_pct is not None
    ):
        highest_return = (peak_price / entry_price) - 1.0
        drawdown_from_high = pnl_pct - highest_return
        if (
            highest_return >= trailing_stop_min_profit
            and drawdown_from_high <= -abs(trailing_stop_pct)
        ):
            reason = (
                f"trailing_stop_exit peak_return={highest_return:.2%} "
                f"drawdown={drawdown_from_high:.2%}"
            )
            logger.warning(
                "AI_POSITION_RISK_BLOCK | code=%s reason=%s", code, reason
            )
            return "exit", reason

    # 3. Take profit (기본 비활성) ─────────────────────────────────────────────
    if take_profit_pct > 0 and pnl_pct is not None and pnl_pct >= take_profit_pct:
        reason = f"take_profit pnl={pnl_pct:.2%} >= {take_profit_pct:.0%}"
        logger.info("AI_POSITION_RISK_BLOCK | code=%s reason=%s", code, reason)
        return "reduce", reason

    # 4. Max holding days ──────────────────────────────────────────────────────
    max_holding_hard_cap = _cfg_int("AI_MAX_HOLDING_DAYS_HARD_CAP", 45)
    entry_date_str = pos.get("entry_date")
    state_recovered = bool(pos.get("state_recovered", False))
    if entry_date_str and as_of_date:
        try:
            entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
            as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
            holding_days = (as_of_dt - entry_dt).days

            # 하드캡: state_recovered 여부와 무관하게 무조건 적용
            if holding_days > max_holding_hard_cap:
                reason = (
                    f"max_holding_hard_cap_exit days={holding_days} > {max_holding_hard_cap}"
                    + (" [state_recovered]" if state_recovered else "")
                    + (f" pnl={pnl_pct:.2%}" if pnl_pct is not None else "")
                )
                logger.warning(
                    "AI_POSITION_RISK_BLOCK | code=%s reason=%s", code, reason
                )
                return "exit", reason

            if holding_days > max_holding_days:
                if state_recovered:
                    # 상태 파일 유실 복구 포지션: entry_date 신뢰 불가 → 소프트 체크 생략
                    logger.warning(
                        "AI_POSITION_RISK_WARN | code=%s max_holding_days exceeded "
                        "but state_recovered=True, soft check suppressed (days=%d)",
                        code, holding_days,
                    )
                elif pnl_pct is None or pnl_pct < 0.02:
                    # 수익률 2% 미만인 경우만 청산 (수익권은 하드캡까지 유지)
                    action = "exit" if (pnl_pct is None or pnl_pct <= 0) else "reduce"
                    reason = (
                        f"max_holding_days_{action} days={holding_days} > {max_holding_days}"
                    )
                    logger.warning(
                        "AI_POSITION_RISK_BLOCK | code=%s reason=%s", code, reason
                    )
                    return action, reason
        except ValueError:
            pass

    return None, None
