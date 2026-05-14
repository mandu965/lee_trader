from __future__ import annotations

import json
from pathlib import Path

from python.us.dashboard.config import DashboardConfig
from python.us.dashboard.dashboard_markdown_renderer import render_dashboard_markdown


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def write_dashboard_outputs(
    payload: dict[str, object],
    cfg: DashboardConfig,
    *,
    formats: tuple[str, ...] | None = None,
) -> dict[str, str]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trade_date = str((payload.get("meta") or {}).get("trade_date") or "unknown")
    selected_formats = formats or cfg.formats
    paths: dict[str, str] = {}

    if "json" in selected_formats:
        json_path = cfg.output_dir / f"{trade_date}_dashboard.json"
        latest_json_path = cfg.output_dir / "latest_dashboard.json"
        json_text = _json_text(payload)
        json_path.write_text(json_text, encoding="utf-8")
        latest_json_path.write_text(json_text, encoding="utf-8")
        paths["json"] = str(json_path)
        paths["latest_json"] = str(latest_json_path)

    if "markdown" in selected_formats:
        md_path = cfg.output_dir / f"{trade_date}_dashboard.md"
        latest_md_path = cfg.output_dir / "latest_dashboard.md"
        markdown = render_dashboard_markdown(payload)
        md_path.write_text(markdown, encoding="utf-8")
        latest_md_path.write_text(markdown, encoding="utf-8")
        paths["markdown"] = str(md_path)
        paths["latest_markdown"] = str(latest_md_path)

    return paths
