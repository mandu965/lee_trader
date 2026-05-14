from __future__ import annotations

from pathlib import Path

from python.us.dashboard.config import load_dashboard_config
from python.us.dashboard.dashboard_data_loader import load_dashboard_raw_data
from python.us.dashboard.dashboard_json_writer import write_dashboard_outputs
from python.us.dashboard.dashboard_report_generator import build_dashboard_payload


def run_dashboard_scheduler_integration(
    *,
    trade_date: str | None = None,
    force: bool = False,
    formats: tuple[str, ...] | None = None,
) -> dict[str, object]:
    cfg = load_dashboard_config()
    result: dict[str, object] = {
        "dashboard_enabled": cfg.enabled,
        "dashboard_executed": False,
        "success": True,
        "trade_date": trade_date,
        "json_report_path": None,
        "markdown_report_path": None,
        "latest_json_path": None,
        "latest_markdown_path": None,
        "payload": None,
        "warnings": list(cfg.warnings),
        "errors": [],
        "pipeline_should_fail": False,
    }

    run_enabled = cfg.enabled or force
    if not run_enabled:
        result["warnings"].append("DASHBOARD_SCHEDULER_DISABLED")
        return result

    try:
        raw_data = load_dashboard_raw_data(cfg, trade_date=trade_date)
        payload = build_dashboard_payload(raw_data, cfg)
        paths = write_dashboard_outputs(payload, cfg, formats=formats or cfg.formats)
        result.update(
            {
                "dashboard_executed": True,
                "success": True,
                "trade_date": str((payload.get("meta") or {}).get("trade_date") or trade_date or ""),
                "json_report_path": paths.get("json"),
                "markdown_report_path": paths.get("markdown"),
                "latest_json_path": paths.get("latest_json"),
                "latest_markdown_path": paths.get("latest_markdown"),
                "payload": payload,
            }
        )
        if cfg.require_json_report and not result["json_report_path"]:
            result["success"] = False
            result["errors"].append("DASHBOARD_JSON_REPORT_REQUIRED")
        if cfg.require_markdown_report and not result["markdown_report_path"]:
            result["success"] = False
            result["errors"].append("DASHBOARD_MARKDOWN_REPORT_REQUIRED")
        if not result["success"]:
            result["pipeline_should_fail"] = cfg.fail_pipeline_on_error
    except Exception as exc:
        result["success"] = False
        result["errors"].append(f"DASHBOARD_REPORT_GENERATION_FAILED:{exc}")
        result["pipeline_should_fail"] = cfg.fail_pipeline_on_error
    return result
