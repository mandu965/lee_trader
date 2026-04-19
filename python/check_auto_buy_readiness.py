from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def summarize(label: str, ok: bool, detail: str) -> str:
    state = "OK" if ok else "WARN"
    return f"[{state}] {label}: {detail}"


def main() -> int:
    gate = read_json(OUTPUTS / "operational_buy_gate.json")
    intents = read_json(OUTPUTS / "trade_intents.json")
    preview = read_json(OUTPUTS / "order_requests_preview.json")
    approvals = read_json(OUTPUTS / "order_buy_approvals.json")
    token_cache = read_json(OUTPUTS / "kis_access_token_cache.json")
    scheduler = read_json(OUTPUTS / "auto_ops_auto_buy_scheduler_status.json")

    lines: list[str] = []
    warnings = 0

    gate_status = str(gate.get("overall_status") or "").strip()
    gate_ok = gate_status in {"PILOT", "BUY_ALLOWED", "WATCH"}
    if not gate_ok:
        warnings += 1
    lines.append(summarize("gate", gate_ok, gate_status or "missing"))

    intent_rows = intents.get("intents") or []
    buy_intents = [row for row in intent_rows if str(row.get("intent_type") or "").upper() == "BUY" and bool(row.get("executable"))]
    sell_intents = [row for row in intent_rows if str(row.get("intent_type") or "").upper() in {"EXIT", "TRIM"} and bool(row.get("executable"))]
    intent_ok = bool(intent_rows)
    if not intent_ok:
        warnings += 1
    lines.append(
        summarize(
            "trade_intents",
            intent_ok,
            f"total={len(intent_rows)} buy_executable={len(buy_intents)} sell_executable={len(sell_intents)}",
        )
    )

    preview_rows = preview.get("items") or []
    executable_preview = [row for row in preview_rows if bool(row.get("executable_now"))]
    preview_ok = bool(preview_rows)
    if not preview_ok:
        warnings += 1
    lines.append(
        summarize(
            "order_preview",
            preview_ok,
            f"total={len(preview_rows)} executable_now={len(executable_preview)} gate={preview.get('gate_status') or '-'}",
        )
    )

    approved_request_ids = approvals.get("approved_request_ids") or approvals.get("request_ids") or []
    approval_ok = bool(approved_request_ids) or not buy_intents
    if not approval_ok:
        warnings += 1
    lines.append(
        summarize(
            "buy_approval",
            approval_ok,
            f"approved_request_ids={len(approved_request_ids)} expected_buy_candidates={len(buy_intents)}",
        )
    )

    token_ok = bool(token_cache.get("access_token")) and bool(token_cache.get("expires_at"))
    if not token_ok:
        warnings += 1
    lines.append(
        summarize(
            "kis_token_cache",
            token_ok,
            f"expires_at={token_cache.get('expires_at') or '-'} cached_at={token_cache.get('cached_at') or '-'}",
        )
    )

    scheduler_status = str(scheduler.get("status") or "").strip() or "missing"
    lines.append(
        summarize(
            "scheduler_status",
            scheduler_status in {"idle", "error", "running"},
            f"status={scheduler_status} last_success_at={scheduler.get('last_success_at') or '-'} last_failure_at={scheduler.get('last_failure_at') or '-'}",
        )
    )
    lines.append(
        summarize(
            "scheduler_config",
            str(scheduler.get("configured_daily_time") or "").strip() == "09:30",
            f"configured_daily_time={scheduler.get('configured_daily_time') or '-'}",
        )
    )

    print("Auto-buy readiness check")
    print("========================")
    for line in lines:
        print(line)

    if buy_intents and not approved_request_ids:
        print("")
        print("Action needed:")
        print("- BUY intents exist, but order_buy_approvals.json is empty, so new BUY orders will be skipped.")
        print("- If you want real BUY execution on 2026-04-21, fill approved_request_ids before market open.")

    print("")
    print(f"warnings={warnings}")
    return 0 if warnings == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
