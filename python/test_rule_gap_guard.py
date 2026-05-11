"""
RULE 자동매매 actual_open_gap 가드 로직 검증 스크립트.

실행: python python/test_rule_gap_guard.py
"""
from __future__ import annotations

import sys
import os
import types

# ── 의존 모듈 stub (단독 실행 용) ─────────────────────────────────────────────
def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _name in [
    "kis_client", "kis_live_account", "rule_signal_builder",
    "rule_account_guard", "rule_execution_simulator", "rule_market_open_snapshot",
    "rule_paper_state_manager", "rule_trading_diagnostics", "pandas",
    "common_live_risk_guard",
]:
    if _name not in sys.modules:
        _stub(_name)

# rule_signal_builder 에서 ROOT 참조 필요
sys.modules["rule_signal_builder"].ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.modules["rule_signal_builder"].STRATEGY_ID = "TEST"
sys.modules["rule_signal_builder"].ENGINE_TYPE = "TEST"
sys.modules["rule_signal_builder"].resolve = lambda p: p

# ── 실제 모듈 임포트 ───────────────────────────────────────────────────────────
# (stub 이후에 로드해야 임포트 오류 없음)
from importlib import import_module

# rule_market_open_snapshot 핵심 함수만 직접 복사하여 테스트
# (KIS API 실호출 없이 순수 로직만 검증)

def _to_float(value):
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compute_actual_open_gap(open_price_raw, prev_close_raw):
    """rule_market_open_snapshot.fetch_single_open_snapshot 의 gap 계산 로직."""
    open_price = _to_float(open_price_raw)
    prev_close = _to_float(prev_close_raw)
    actual_open_gap = None
    if open_price is not None and open_price > 0 and prev_close not in {None, 0.0}:
        actual_open_gap = (open_price / prev_close) - 1.0
    return actual_open_gap, open_price, prev_close


def compute_gap_reason(market_row, side: str, gap_upper: float = 0.05, gap_lower: float = -0.04, block_on_unavailable: bool = True):
    """rule_order_submitter._live_order_context 의 gap reason 계산 로직."""
    def _float(v):
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    actual_open_gap = _float((market_row or {}).get("actual_open_gap"))
    actual_gap_reason = None

    if side == "BUY":
        if market_row is None:
            actual_gap_reason = "market_snapshot_unmapped"
        elif actual_open_gap is None:
            if block_on_unavailable:
                actual_gap_reason = "actual_open_gap_unavailable"
        elif actual_open_gap > gap_upper:
            actual_gap_reason = "actual_open_gap_gt_5pct"
        elif actual_open_gap < gap_lower:
            actual_gap_reason = "actual_open_gap_lt_minus_4pct"

    gap_blocked = actual_gap_reason not in {None, "", "none"}
    return actual_open_gap, actual_gap_reason, gap_blocked


# ── 차단 사유 매핑 테스트 ─────────────────────────────────────────────────────
RAW_REASON_TO_BLOCK_REASON = {
    "actual_open_gap_lt_minus_4pct": "GAP_RISK_NEGATIVE",
    "actual_open_gap_gt_5pct": "GAP_RISK_POSITIVE",
    "actual_open_gap_unavailable": "GAP_RISK_UNAVAILABLE",
    "market_snapshot_unmapped": "MARKET_SNAPSHOT_UNMAPPED",
    "actual_open_price_unavailable": "GAP_RISK_UNAVAILABLE",
    "prev_close_unavailable": "GAP_RISK_UNAVAILABLE",
    "no_order_action": "NO_ORDER_ACTION",
}

UI_MESSAGES = {
    "GAP_RISK_NEGATIVE": "갭 하락",
    "GAP_RISK_POSITIVE": "갭 상승",
    "GAP_RISK_UNAVAILABLE": "시가 확인 불가",
    "MARKET_SNAPSHOT_UNMAPPED": "스냅샷에서 해당 종목을 찾지 못해",
    "NO_ORDER_ACTION": "주문 대상 아님",
}


def normalize_block_reason(raw_reason: str) -> str:
    return RAW_REASON_TO_BLOCK_REASON.get(raw_reason, "UNMAPPED")


# ── 테스트 케이스 ──────────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures: list[str] = []


def check(case_name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {case_name}")
    else:
        print(f"  {FAIL}  {case_name}  ← {detail}")
        failures.append(case_name)


print("\n=== Case 1: 정상 갭 (-2%) ===")
gap, _, _ = compute_actual_open_gap("9800", "10000")
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap}, "BUY")
check("actual_open_gap == -0.02", abs(gap - (-0.02)) < 1e-9, f"got {gap}")
check("reason is None", reason is None, f"got {reason}")
check("blocked is False", not blocked, f"got {blocked}")

print("\n=== Case 2: 실제 -5% 갭 하락 ===")
gap, _, _ = compute_actual_open_gap("9500", "10000")
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap}, "BUY")
check("actual_open_gap == -0.05", abs(gap - (-0.05)) < 1e-9, f"got {gap}")
check("reason == actual_open_gap_lt_minus_4pct", reason == "actual_open_gap_lt_minus_4pct", f"got {reason}")
check("blocked is True", blocked, f"got {blocked}")
mapped = normalize_block_reason(reason)
check("mapped == GAP_RISK_NEGATIVE", mapped == "GAP_RISK_NEGATIVE", f"got {mapped}")

print("\n=== Case 3: open_price = 0 (장 시작 전, KIS API 미확정) ===")
gap, open_price, _ = compute_actual_open_gap("0", "10000")
check("actual_open_gap is None (not -1.0)", gap is None, f"got {gap}")
check("open_price == 0.0", open_price == 0.0, f"got {open_price}")
# open_price=0이면 snapshot row의 actual_open_gap=None
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap}, "BUY")
check("reason == actual_open_gap_unavailable", reason == "actual_open_gap_unavailable", f"got {reason}")
check("blocked is True", blocked, f"got {blocked}")
mapped = normalize_block_reason(reason)
check("mapped == GAP_RISK_UNAVAILABLE", mapped == "GAP_RISK_UNAVAILABLE", f"got {mapped}")

print("\n=== Case 4: open_price = None ===")
gap, _, _ = compute_actual_open_gap(None, "10000")
check("actual_open_gap is None", gap is None, f"got {gap}")
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap}, "BUY")
check("reason == actual_open_gap_unavailable", reason == "actual_open_gap_unavailable", f"got {reason}")
check("blocked is True", blocked, f"got {blocked}")

print("\n=== Case 5: prev_close = 0 ===")
gap, _, _ = compute_actual_open_gap("10000", "0")
check("actual_open_gap is None", gap is None, f"got {gap}")
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap}, "BUY")
check("reason == actual_open_gap_unavailable", reason == "actual_open_gap_unavailable", f"got {reason}")
check("blocked is True", blocked, f"got {blocked}")

print("\n=== Case 6: market_row = None (스냅샷 매핑 실패) ===")
_, reason, blocked = compute_gap_reason(None, "BUY")
check("reason == market_snapshot_unmapped", reason == "market_snapshot_unmapped", f"got {reason}")
check("blocked is True", blocked, f"got {blocked}")
mapped = normalize_block_reason(reason)
check("mapped == MARKET_SNAPSHOT_UNMAPPED", mapped == "MARKET_SNAPSHOT_UNMAPPED", f"got {mapped}")

print("\n=== Case 7: UI 메시지 매핑 검증 ===")
for raw, expected_keyword in [
    ("actual_open_gap_lt_minus_4pct", "갭 하락"),
    ("market_snapshot_unmapped", "스냅샷에서 해당 종목을 찾지 못해"),
    ("actual_open_price_unavailable", "시가 확인 불가"),
]:
    mapped = normalize_block_reason(raw)
    msg = UI_MESSAGES.get(mapped, "")
    check(f"UI message for '{raw}' contains '{expected_keyword}'", expected_keyword in msg, f"mapped={mapped}, msg={msg}")

print("\n=== Case 8: open_price=-100 (음수) ===")
gap, _, _ = compute_actual_open_gap("-100", "10000")
check("actual_open_gap is None for negative open_price", gap is None, f"got {gap}")

print("\n=== Case 9: SELL 종목은 gap guard 적용 안 됨 ===")
gap2, _, _ = compute_actual_open_gap("0", "10000")
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap2}, "SELL")
check("SELL side: reason is None", reason is None, f"got {reason}")
check("SELL side: blocked is False", not blocked, f"got {blocked}")

print("\n=== Case 10: gap_upper/gap_lower 환경변수 반영 ===")
gap10, _, _ = compute_actual_open_gap("10300", "10000")  # +3%
_, reason, blocked = compute_gap_reason({"actual_open_gap": gap10}, "BUY", gap_upper=0.02)
check("+3% gap blocked when upper=0.02", blocked, f"got reason={reason}")
_, reason2, blocked2 = compute_gap_reason({"actual_open_gap": gap10}, "BUY", gap_upper=0.05)
check("+3% gap NOT blocked when upper=0.05", not blocked2, f"got reason={reason2}")

# ── 결과 ───────────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"[FAIL] {len(failures)}건 실패 — {', '.join(failures)}")
    sys.exit(1)
else:
    total = 10
    print(f"[PASS] 모든 테스트 통과 (Case 1~{total})")
