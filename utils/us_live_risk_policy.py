from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in {".", "e", "E"}):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml_mapping(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, remainder = line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not remainder:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue
        current[key] = _parse_scalar(remainder)
    return root


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def resolve_us_live_risk_policy_path() -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    raw = str(os.environ.get("US_LIVE_RISK_POLICY_FILE", "config/us_stock_live_risk_policy.yaml")).strip()
    path = Path(raw or "config/us_stock_live_risk_policy.yaml")
    return path if path.is_absolute() else root_dir / path


def _load_profiles() -> dict[str, Any]:
    path = resolve_us_live_risk_policy_path()
    if not path.exists():
        raise FileNotFoundError(f"US live risk policy file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = _load_simple_yaml_mapping(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"US live risk policy must be a mapping: {path}")
    return payload


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "US_LIVE_TRADING_ENABLED": ("safety", "live_trading_enabled"),
    "US_LIVE_ORDER_ENABLED": ("safety", "live_order_enabled"),
    "US_LIVE_BUY_ENABLED": ("safety", "buy_enabled"),
    "US_LIVE_SELL_ENABLED": ("safety", "sell_enabled"),
    "US_LIVE_REQUIRE_MANUAL_APPROVAL": ("safety", "require_manual_approval"),
    "US_LIVE_REAL_ORDER_BLOCKED": ("safety", "real_order_blocked"),
    "US_LIVE_ACCOUNT_ID": ("account", "account_id"),
    "US_LIVE_MIN_CASH_WEIGHT": ("account", "min_cash_weight"),
    "US_LIVE_MAX_POSITION_COUNT": ("account", "max_position_count"),
    "US_LIVE_DEFAULT_ORDER_TYPE": ("order", "default_order_type"),
    "US_LIVE_ALLOW_MARKET_ORDER": ("order", "allow_market_order"),
    "US_LIVE_MAX_ORDER_AMOUNT_USD": ("order", "max_order_amount_usd"),
    "US_LIVE_MIN_ORDER_AMOUNT_USD": ("order", "min_order_amount_usd"),
    "US_LIVE_MAX_DAILY_BUY_AMOUNT_USD": ("order", "max_daily_buy_amount_usd"),
    "US_LIVE_MAX_DAILY_SELL_AMOUNT_USD": ("order", "max_daily_sell_amount_usd"),
    "US_LIVE_MAX_DAILY_ORDER_COUNT": ("order", "max_daily_order_count"),
    "US_LIVE_MAX_DAILY_NEW_BUYS": ("order", "max_daily_new_buys"),
    "US_LIVE_MAX_ORDER_RETRY": ("order", "max_order_retry"),
    "US_LIVE_MAX_DAILY_ORDER_FAILURES": ("order", "max_daily_order_failures"),
    "US_LIVE_MAX_POSITION_WEIGHT": ("position", "max_position_weight"),
    "US_LIVE_MAX_SYMBOL_POSITION_AMOUNT_USD": ("position", "max_symbol_position_amount_usd"),
    "US_LIVE_MAX_SECTOR_WEIGHT": ("sector", "max_sector_weight"),
    "US_LIVE_BLOCK_LEVERAGED_ETF": ("instrument", "block_leveraged_etf"),
    "US_LIVE_BLOCK_INVERSE_ETF": ("instrument", "block_inverse_etf"),
    "US_LIVE_ALLOW_ETF": ("instrument", "allow_etf"),
    "US_LIVE_BLOCK_BUY_ON_SPY_DROP_PCT": ("market", "block_buy_on_spy_drop_pct"),
    "US_LIVE_BLOCK_BUY_ON_QQQ_DROP_PCT": ("market", "block_buy_on_qqq_drop_pct"),
    "US_LIVE_BLOCK_BUY_ON_SYMBOL_GAP_UP_PCT": ("market", "block_buy_on_symbol_gap_up_pct"),
    "US_LIVE_BLOCK_BUY_ON_SYMBOL_GAP_DOWN_PCT": ("market", "block_buy_on_symbol_gap_down_pct"),
    "US_LIVE_MAX_SYMBOL_VOLATILITY_20D": ("market", "max_symbol_volatility_20d"),
    "US_LIVE_BLOCK_BEAR_HIGH_VOL_REGIME": ("market", "block_bear_high_vol_regime"),
    "US_LIVE_REGULAR_SESSION_ONLY": ("time", "regular_session_only"),
    "US_LIVE_BLOCK_FIRST_MINUTES_AFTER_OPEN": ("time", "block_first_minutes_after_open"),
    "US_LIVE_BLOCK_LAST_MINUTES_BEFORE_CLOSE": ("time", "block_last_minutes_before_close"),
    "US_LIVE_BLOCK_PREMARKET": ("time", "block_premarket"),
    "US_LIVE_BLOCK_AFTERHOURS": ("time", "block_afterhours"),
    "US_LIVE_APPROVAL_EXPIRES_MINUTES": ("approval", "approval_expires_minutes"),
    "US_LIVE_NOTIFY_ENABLED": ("notification", "notify_on_candidate"),
    "US_LIVE_KILL_SWITCH_NOTIFY_ENABLED": ("notification", "notify_on_kill_switch"),
    "US_LIVE_APPROVAL_NOTIFY_ENABLED": ("notification", "notify_on_approval"),
}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str


def load_us_live_risk_policy(policy_id: str | None = None) -> dict[str, Any]:
    policy_key = _safe_str(policy_id, os.environ.get("US_LIVE_RISK_POLICY_ID", "US_LIVE_RULE_V1")).upper()
    profiles = _load_profiles()
    raw_profile = profiles.get(policy_key, {})
    if not isinstance(raw_profile, dict):
        raise ValueError(f"US live risk policy profile must be a mapping: {policy_key}")
    profile = _deep_copy(raw_profile)
    for env_name, path in ENV_OVERRIDES.items():
        if env_name in os.environ:
            _set_path(profile, path, _parse_scalar(str(os.environ.get(env_name, "")).strip()))
    profile["policy_id"] = policy_key
    profile["config_path"] = str(resolve_us_live_risk_policy_path())
    return profile


def collect_us_live_risk_policy_issues(policy: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    account = policy.get("account", {}) if isinstance(policy.get("account"), dict) else {}
    order = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    position = policy.get("position", {}) if isinstance(policy.get("position"), dict) else {}
    sector = policy.get("sector", {}) if isinstance(policy.get("sector"), dict) else {}

    def require(condition: bool, code: str, message: str, *, warning: bool = False) -> None:
        if condition:
            return
        issues.append(ValidationIssue("WARNING" if warning else "ERROR", code, message))

    require(bool(safety.get("live_trading_enabled")) is False, "live_trading_enabled_not_false", "live_trading_enabled should remain false for SAFE_DEFAULT")
    require(bool(safety.get("live_order_enabled")) is False, "live_order_enabled_not_false", "live_order_enabled should remain false for SAFE_DEFAULT")
    require(bool(safety.get("buy_enabled")) is False, "buy_enabled_not_false", "buy_enabled should remain false for SAFE_DEFAULT")
    require(bool(safety.get("sell_enabled")) is False, "sell_enabled_not_false", "sell_enabled should remain false for SAFE_DEFAULT")
    require(bool(safety.get("require_manual_approval")) is True, "manual_approval_disabled", "require_manual_approval should remain true")
    require(bool(safety.get("real_order_blocked")) is True, "real_order_blocked_disabled", "real_order_blocked should remain true")
    require(bool(order.get("allow_market_order")) is False, "market_order_enabled", "allow_market_order should remain false for SAFE_DEFAULT")

    max_order_amount = _safe_float(order.get("max_order_amount_usd"))
    min_order_amount = _safe_float(order.get("min_order_amount_usd"))
    max_daily_buy = _safe_float(order.get("max_daily_buy_amount_usd"))
    max_daily_order_count = _safe_float(order.get("max_daily_order_count"))
    max_daily_new_buys = _safe_float(order.get("max_daily_new_buys"))
    min_cash_weight = _safe_float(account.get("min_cash_weight"))
    max_position_weight = _safe_float(position.get("max_position_weight"))
    max_sector_weight = _safe_float(sector.get("max_sector_weight"))

    require(max_order_amount is not None and max_order_amount > 0, "max_order_amount_missing", "max_order_amount_usd must be a positive number")
    require(min_order_amount is not None and min_order_amount > 0, "min_order_amount_missing", "min_order_amount_usd must be a positive number")
    if max_order_amount is not None:
        require(max_order_amount <= 50, "max_order_amount_too_high", "max_order_amount_usd exceeds Micro Live SAFE_DEFAULT recommendation", warning=True)
    if min_order_amount is not None and max_order_amount is not None:
        require(min_order_amount <= max_order_amount, "min_order_gt_max_order", "min_order_amount_usd must not exceed max_order_amount_usd")
    if max_daily_buy is not None:
        require(max_daily_buy <= 100, "max_daily_buy_too_high", "max_daily_buy_amount_usd exceeds Micro Live SAFE_DEFAULT recommendation", warning=True)
    if max_daily_order_count is not None:
        require(max_daily_order_count <= 3, "max_daily_order_count_too_high", "max_daily_order_count exceeds SAFE_DEFAULT recommendation", warning=True)
    if max_daily_new_buys is not None:
        require(max_daily_new_buys <= 1, "max_daily_new_buys_too_high", "max_daily_new_buys exceeds SAFE_DEFAULT recommendation", warning=True)
    if min_cash_weight is not None:
        require(0 < min_cash_weight <= 1, "min_cash_weight_invalid", "min_cash_weight must be between 0 and 1")
        require(min_cash_weight >= 0.50, "min_cash_weight_too_low", "min_cash_weight is below SAFE_DEFAULT recommendation", warning=True)
    if max_position_weight is not None:
        require(0 < max_position_weight <= 1, "max_position_weight_invalid", "max_position_weight must be between 0 and 1")
    if max_sector_weight is not None:
        require(0 < max_sector_weight <= 1, "max_sector_weight_invalid", "max_sector_weight must be between 0 and 1")
    return issues


def validate_us_live_risk_policy(policy: dict[str, Any]) -> list[str]:
    return [item.code for item in collect_us_live_risk_policy_issues(policy)]


def print_us_live_risk_policy_summary(policy: dict[str, Any]) -> None:
    safety = policy.get("safety", {}) if isinstance(policy.get("safety"), dict) else {}
    account = policy.get("account", {}) if isinstance(policy.get("account"), dict) else {}
    order = policy.get("order", {}) if isinstance(policy.get("order"), dict) else {}
    position = policy.get("position", {}) if isinstance(policy.get("position"), dict) else {}
    sector = policy.get("sector", {}) if isinstance(policy.get("sector"), dict) else {}
    print(f"Policy ID: {policy.get('policy_id')}")
    print("Safety Flags:")
    print(f"- live_trading_enabled: {bool(safety.get('live_trading_enabled'))}")
    print(f"- live_order_enabled: {bool(safety.get('live_order_enabled'))}")
    print(f"- buy_enabled: {bool(safety.get('buy_enabled'))}")
    print(f"- sell_enabled: {bool(safety.get('sell_enabled'))}")
    print(f"- require_manual_approval: {bool(safety.get('require_manual_approval'))}")
    print(f"- real_order_blocked: {bool(safety.get('real_order_blocked'))}")
    print("Risk Limits:")
    print(f"- max_order_amount_usd: {order.get('max_order_amount_usd')}")
    print(f"- max_daily_buy_amount_usd: {order.get('max_daily_buy_amount_usd')}")
    print(f"- max_daily_order_count: {order.get('max_daily_order_count')}")
    print(f"- max_position_weight: {position.get('max_position_weight')}")
    print(f"- max_sector_weight: {sector.get('max_sector_weight')}")
    print(f"- min_cash_weight: {account.get('min_cash_weight')}")
