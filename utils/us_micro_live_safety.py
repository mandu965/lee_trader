from __future__ import annotations

import os


SAFETY_MESSAGE = "[SAFETY] Micro order mock/sandbox only. Real order APIs are blocked."


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def assert_us_micro_mock_only(message: str | None = None) -> None:
    if not _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"):
        raise RuntimeError("US_MICRO_REAL_ORDER_BLOCKED must be true for Phase 7-2 micro order scripts.")
    if _flag("US_MICRO_ALLOW_LIVE", "false"):
        raise RuntimeError("US_MICRO_ALLOW_LIVE must remain false for Phase 7-2. execution_mode=LIVE requests are explicitly blocked.")
    if not _flag("US_MICRO_REQUIRE_APPROVAL", "true"):
        raise RuntimeError("US_MICRO_REQUIRE_APPROVAL must remain true for Phase 7-2.")
    if not _flag("US_MICRO_REQUIRE_PRECHECK", "true"):
        raise RuntimeError("US_MICRO_REQUIRE_PRECHECK must remain true for Phase 7-2.")
    print(message or SAFETY_MESSAGE)


def resolve_micro_execution_mode(requested: str | None = None) -> str:
    mode = str(requested or os.environ.get("US_MICRO_EXECUTION_MODE", "MOCK")).strip().upper() or "MOCK"
    if mode == "LIVE":
        raise RuntimeError("SANDBOX is not a live-account order. LIVE is not used in Phase 7-2. execution_mode=LIVE requests are explicitly blocked.")
    if mode not in {"MOCK", "SANDBOX"}:
        raise ValueError(f"Unsupported execution_mode: {mode}")
    if mode == "SANDBOX" and not _flag("US_MICRO_ALLOW_SANDBOX", "false"):
        raise RuntimeError("US_MICRO_ALLOW_SANDBOX must be true to use SANDBOX mode.")
    return mode


def sandbox_config_snapshot() -> dict[str, object]:
    return {
        "execution_mode": resolve_micro_execution_mode("SANDBOX" if _flag("US_MICRO_ALLOW_SANDBOX", "false") else _text("US_MICRO_EXECUTION_MODE", "MOCK")),
        "allow_sandbox": _flag("US_MICRO_ALLOW_SANDBOX", "false"),
        "allow_live": _flag("US_MICRO_ALLOW_LIVE", "false"),
        "real_order_blocked": _flag("US_MICRO_REAL_ORDER_BLOCKED", "true"),
        "sandbox_order_enabled": _flag("US_SANDBOX_ORDER_ENABLED", "false"),
        "sandbox_broker_name": _text("US_SANDBOX_BROKER_NAME", "NONE") or "NONE",
        "sandbox_base_url_present": bool(_text("US_SANDBOX_BASE_URL")),
        "sandbox_api_key_present": bool(_text("US_SANDBOX_API_KEY")),
        "sandbox_api_secret_present": bool(_text("US_SANDBOX_API_SECRET")),
        "sandbox_require_approval": _flag("US_SANDBOX_REQUIRE_APPROVAL", "true"),
        "sandbox_require_precheck": _flag("US_SANDBOX_REQUIRE_PRECHECK", "true"),
        "sandbox_require_kill_switch_clear": _flag("US_SANDBOX_REQUIRE_KILL_SWITCH_CLEAR", "true"),
        "sandbox_max_order_amount_usd": _safe_float(_text("US_SANDBOX_MAX_ORDER_AMOUNT_USD", "50")),
        "sandbox_max_daily_order_count": _safe_float(_text("US_SANDBOX_MAX_DAILY_ORDER_COUNT", "3")),
        "sandbox_max_daily_new_buys": _safe_float(_text("US_SANDBOX_MAX_DAILY_NEW_BUYS", "1")),
        "sandbox_allow_market_order": _flag("US_SANDBOX_ALLOW_MARKET_ORDER", "false"),
        "sandbox_default_order_type": _text("US_SANDBOX_DEFAULT_ORDER_TYPE", "LIMIT").upper() or "LIMIT",
    }


def validate_us_micro_sandbox_config() -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []
    snapshot = sandbox_config_snapshot()

    if snapshot["allow_live"]:
        errors.append("US_MICRO_ALLOW_LIVE must remain false.")
    if not snapshot["real_order_blocked"]:
        errors.append("US_MICRO_REAL_ORDER_BLOCKED must remain true.")
    if snapshot["sandbox_max_order_amount_usd"] is None or float(snapshot["sandbox_max_order_amount_usd"]) > 50:
        errors.append("US_SANDBOX_MAX_ORDER_AMOUNT_USD must be <= 50.")
    if snapshot["sandbox_allow_market_order"]:
        errors.append("US_SANDBOX_ALLOW_MARKET_ORDER must remain false.")
    if snapshot["sandbox_broker_name"] == "NONE":
        warnings.append("Sandbox broker is not configured.")
    if not snapshot["sandbox_base_url_present"]:
        warnings.append("Sandbox base URL is missing.")
    if not snapshot["sandbox_api_key_present"]:
        warnings.append("Sandbox API key is missing.")
    if not snapshot["sandbox_api_secret_present"]:
        warnings.append("Sandbox API secret is missing.")

    result = "SAFE_DEFAULT"
    if errors:
        result = "ERROR"
    elif warnings:
        result = "WARNING"
    return {"result": result, "errors": errors, "warnings": warnings, "snapshot": snapshot}


def assert_sandbox_enabled() -> None:
    assert_us_micro_mock_only()
    if not _flag("US_MICRO_ALLOW_SANDBOX", "false"):
        raise RuntimeError("US_MICRO_ALLOW_SANDBOX must be true for sandbox execution.")
    if not _flag("US_SANDBOX_ORDER_ENABLED", "false"):
        raise RuntimeError("US_SANDBOX_ORDER_ENABLED must be true for sandbox execution.")
    if not _flag("US_SANDBOX_REQUIRE_APPROVAL", "true"):
        raise RuntimeError("US_SANDBOX_REQUIRE_APPROVAL must remain true for Phase 7-2.")
    if not _flag("US_SANDBOX_REQUIRE_PRECHECK", "true"):
        raise RuntimeError("US_SANDBOX_REQUIRE_PRECHECK must remain true for Phase 7-2.")
    if not _flag("US_SANDBOX_REQUIRE_KILL_SWITCH_CLEAR", "true"):
        raise RuntimeError("US_SANDBOX_REQUIRE_KILL_SWITCH_CLEAR must remain true for Phase 7-2.")
