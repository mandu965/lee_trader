from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from python.us.notification.channel_router import run_notification_channels
from python.us.notification.config import load_notification_config
from python.us.notification.notification_payload_loader import load_notification_payload


def _parse_channels(raw_value: str | None) -> tuple[str, ...] | None:
    if not raw_value:
        return None
    values = tuple(item.strip().upper() for item in raw_value.split(",") if item.strip())
    return values or None


def _render_console_summary(result: dict[str, object]) -> str:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    lines = [
        "[US NOTIFICATION ADAPTER]",
        f"trade_date={payload.get('trade_date')}",
        f"mode={result.get('mode')}",
        f"severity={result.get('severity')}",
        f"paper_trading_only={str(payload.get('paper_trading_only')).lower()}",
        f"live_orders_executed={str(payload.get('live_orders_executed')).lower()}",
        "",
        "channels:",
    ]
    for channel_name, channel_result in (result.get("channels") or {}).items():
        status = channel_result.get("status")
        reason = channel_result.get("reason")
        suffix = f" {reason}" if reason else ""
        lines.append(f"- {channel_name}: {status}{suffix}")
    lines.extend(
        [
            "",
            f"approval_required={str(result.get('approval_required')).lower()}",
            f"approval_status={result.get('approval_status')}",
            "",
            "output:",
        ]
    )
    file_result = (result.get("channels") or {}).get("FILE") or {}
    for key in ("path", "text_path", "latest_path", "latest_text_path"):
        if file_result.get(key):
            lines.append(f"- {file_result.get(key)}")
    return "\n".join(lines)


def run_notification_adapter(
    *,
    trade_date: str | None = None,
    channels_override: tuple[str, ...] | None = None,
    force: bool = False,
    emit_console: bool = True,
) -> dict[str, object]:
    cfg = load_notification_config()
    if not cfg.enabled and not force:
        result = {
            "notification_executed": False,
            "mode": cfg.mode,
            "severity": "INFO",
            "channels": {},
            "approval_required": False,
            "approval_status": None,
            "payload": {},
            "warnings": list(cfg.warnings),
            "errors": ["NOTIFICATION_ADAPTER_DISABLED"],
            "pipeline_should_fail": False,
        }
        if emit_console:
            print(_render_console_summary(result))
        return result

    payload_result = load_notification_payload(cfg, trade_date=trade_date)
    result = run_notification_channels(cfg, payload_result=payload_result, channels_override=channels_override, emit_console=emit_console)
    result["warnings"] = list(cfg.warnings) + list(result.get("warnings") or [])
    if emit_console:
        print(_render_console_summary(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run US Paper Trading notification adapter dry-run.")
    parser.add_argument("--trade-date")
    parser.add_argument("--channels")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_notification_adapter(
        trade_date=args.trade_date,
        channels_override=_parse_channels(args.channels),
        force=args.force,
        emit_console=True,
    )
    if result.get("pipeline_should_fail"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
