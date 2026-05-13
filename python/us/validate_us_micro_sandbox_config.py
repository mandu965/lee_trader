from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.us_micro_live_safety import SAFETY_MESSAGE, assert_us_micro_mock_only, validate_us_micro_sandbox_config


def main() -> int:
    assert_us_micro_mock_only(SAFETY_MESSAGE)
    result = validate_us_micro_sandbox_config()
    snapshot = result["snapshot"]
    print("[US Micro Sandbox Config Validation]")
    print("")
    print(f"Execution Mode: {snapshot.get('execution_mode')}")
    print(f"Allow Sandbox: {snapshot.get('allow_sandbox')}")
    print(f"Allow Live: {snapshot.get('allow_live')}")
    print(f"Real Order Blocked: {snapshot.get('real_order_blocked')}")
    print(f"Sandbox Order Enabled: {snapshot.get('sandbox_order_enabled')}")
    print(f"Sandbox Broker: {snapshot.get('sandbox_broker_name')}")
    print("")
    print(f"Result: {result.get('result')}")
    if result["errors"]:
        print("")
        print("Errors:")
        for item in result["errors"]:
            print(f"- {item}")
    if result["warnings"]:
        print("")
        print("Warnings:")
        for item in result["warnings"]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
