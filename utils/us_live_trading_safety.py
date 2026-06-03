from __future__ import annotations

from utils.us_live_risk_policy import load_us_live_risk_policy


def assert_us_live_pre_trade_only(*, policy_id: str | None = None, message: str | None = None) -> None:
    policy = load_us_live_risk_policy(policy_id)
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    if not bool(safety.get("real_order_blocked", True)):
        raise RuntimeError("US_LIVE_REAL_ORDER_BLOCKED must be true for pre-trade check scripts.")
    if message:
        print(message)
