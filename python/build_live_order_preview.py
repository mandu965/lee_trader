from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from submit_live_orders import (
    INPUT_LIVE_HOLDINGS,
    OUT_JSON as INPUT_ORDER_REQUESTS_PREVIEW,
    INPUT_PRICE_FALLBACK,
    INPUT_RANKING,
    INPUT_TRADE_INTENTS,
    build_order_requests,
    load_live_holdings,
    load_price_fallback,
    load_ranking,
    merge_ranking_with_prices,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

OUT_JSON = OUTPUT_DIR / "live_order_preview.json"
OUT_MD = OUTPUT_DIR / "live_order_preview.md"
INPUT_GATE = OUTPUT_DIR / "operational_buy_gate.json"

STATUS_BUY_ALLOWED = "BUY_ALLOWED"
STATUS_PILOT = "PILOT"
STATUS_WATCH = "WATCH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live-account order preview aligned with guarded order requests.")
    parser.add_argument("--order-requests-preview-json", type=Path, default=INPUT_ORDER_REQUESTS_PREVIEW)
    parser.add_argument("--trade-intents-json", type=Path, default=INPUT_TRADE_INTENTS)
    parser.add_argument("--live-holdings-csv", type=Path, default=INPUT_LIVE_HOLDINGS)
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--price-fallback-csv", type=Path, default=INPUT_PRICE_FALLBACK)
    parser.add_argument("--gate-json", type=Path, default=INPUT_GATE)
    parser.add_argument("--ord-dvsn", default="01", help="01 market, 00 limit")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _safe_read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _fmt_num(v: object, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{digits}f}"


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _clean_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def resolve_limited_entry_mode(gate: dict[str, Any]) -> str | None:
    explicit = str(gate.get("limited_entry_mode") or "").strip().lower()
    if explicit:
        return explicit
    gate_status = str(gate.get("overall_status") or "").upper()
    if gate_status == STATUS_WATCH:
        return "watch"
    if gate_status == STATUS_PILOT:
        return "pilot"
    return None


def convert_order_preview_to_live_preview(
    *,
    preview_payload: dict[str, Any],
    intents_payload: dict[str, Any],
    holdings: pd.DataFrame,
    ranking: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    items = preview_payload.get("items") or []
    preview_df = pd.DataFrame(items)
    if preview_df.empty:
        preview_df = pd.DataFrame(columns=["code", "name", "reference_price", "allowed_qty", "planned_qty", "final_request_qty", "blocked_reason"])

    preview_df["code"] = preview_df.get("code", pd.Series(dtype="object")).fillna("").astype(str).str.zfill(6)
    preview_df["side"] = preview_df.get("side", pd.Series(dtype="object")).fillna("").astype(str).str.upper()
    preview_df = preview_df.loc[preview_df["side"].eq("BUY")].copy()

    holdings_lookup = holdings.drop_duplicates("code").set_index("code") if not holdings.empty else pd.DataFrame()
    ranking_lookup = ranking.drop_duplicates("code").set_index("code") if not ranking.empty else pd.DataFrame()

    intent_rows = pd.DataFrame(intents_payload.get("intents") or [])
    if not intent_rows.empty:
        intent_rows["code"] = intent_rows.get("code", pd.Series(dtype="object")).fillna("").astype(str).str.zfill(6)
        intent_rows["intent_type"] = intent_rows.get("intent_type", pd.Series(dtype="object")).fillna("").astype(str).str.upper()
        intent_rows = intent_rows.loc[intent_rows["intent_type"].eq("BUY")].copy()
        intent_rows = intent_rows.sort_values(["priority", "code"], ascending=[False, True], na_position="last")
        intent_rows = intent_rows.drop_duplicates("code", keep="first")
        preview_df = preview_df.merge(
            intent_rows.loc[:, ["code", "intent_type", "reason", "target_weight"]].rename(
                columns={"reason": "intent_reason"}
            ),
            on="code",
            how="left",
        )

    rows: list[dict[str, Any]] = []
    for _, row in preview_df.iterrows():
        code = str(row.get("code") or "").zfill(6)
        ranking_row = ranking_lookup.loc[code] if not ranking_lookup.empty and code in ranking_lookup.index else pd.Series(dtype="object")
        held_now = bool(not holdings_lookup.empty and code in holdings_lookup.index)
        rows.append(
            {
                "code": code,
                "name": _clean_text(row.get("name")) or _clean_text(ranking_row.get("name")),
                "buy_rank": pd.to_numeric(ranking_row.get("buy_rank"), errors="coerce"),
                "final_score": pd.to_numeric(ranking_row.get("final_score"), errors="coerce"),
                "confidence_score": pd.to_numeric(ranking_row.get("confidence_score"), errors="coerce"),
                "reference_price": pd.to_numeric(row.get("reference_price"), errors="coerce"),
                "held_now": held_now,
                "nrcvb_buy_qty": pd.to_numeric(row.get("allowed_qty"), errors="coerce"),
                "max_buy_qty": pd.to_numeric(row.get("allowed_qty"), errors="coerce"),
                "planned_qty": pd.to_numeric(row.get("planned_qty"), errors="coerce"),
                "final_preview_qty": pd.to_numeric(row.get("final_request_qty"), errors="coerce"),
                "target_weight": pd.to_numeric(row.get("target_weight"), errors="coerce"),
                "blocked_reason": _clean_text(row.get("blocked_reason")) or "",
                "reason": _clean_text(row.get("intent_reason")) or _clean_text(row.get("reason")),
            }
        )

    cleaned_rows = [{key: _clean_value(value) for key, value in row.items()} for row in rows]
    source_gate_status = str(gate.get("overall_status") or "").upper() or None
    runtime_gate_status = str(preview_payload.get("gate_status") or source_gate_status or "").upper() or None
    runtime_override_applied = bool(
        source_gate_status
        and runtime_gate_status
        and source_gate_status != runtime_gate_status
    )
    gate_display_status = (
        f"{source_gate_status} (runtime {runtime_gate_status} override)"
        if runtime_override_applied
        else runtime_gate_status
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": preview_payload.get("asof_date"),
        "env_dv": preview_payload.get("env_dv"),
        "gate_status": runtime_gate_status,
        "gate_source_status": source_gate_status,
        "gate_runtime_status": runtime_gate_status,
        "gate_display_status": gate_display_status,
        "runtime_override_applied": runtime_override_applied,
        "limited_entry_mode": resolve_limited_entry_mode(gate),
        "cash_summary": preview_payload.get("cash_summary") or {},
        "items": cleaned_rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Live Order Preview",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- asof_date: {payload.get('asof_date') or 'NA'}",
        f"- env_dv: {payload.get('env_dv') or 'NA'}",
        f"- gate_status: {payload.get('gate_status') or 'NA'}",
        f"- gate_display_status: {payload.get('gate_display_status') or payload.get('gate_status') or 'NA'}",
        f"- limited_entry_mode: {payload.get('limited_entry_mode') or 'none'}",
        f"- available_cash: {_fmt_num((payload.get('cash_summary') or {}).get('dnca_tot_amt'))}",
        "",
        "| code | name | buy_rank | ref_price | held_now | planned_qty | final_preview_qty | blocked_reason |",
        "| ---- | ---- | -------- | --------- | -------- | ----------- | ----------------- | -------------- |",
    ]
    for item in payload.get("items") or []:
        lines.append(
            f"| {item.get('code') or ''} | {item.get('name') or ''} | {_fmt_num(item.get('buy_rank'), 0)} | {_fmt_num(item.get('reference_price'), 0)} | {'Y' if item.get('held_now') else 'N'} | {_fmt_num(item.get('planned_qty'), 0)} | {_fmt_num(item.get('final_preview_qty'), 0)} | {item.get('blocked_reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    holdings = load_live_holdings(args.live_holdings_csv)
    ranking = merge_ranking_with_prices(
        load_ranking(args.ranking_csv),
        load_price_fallback(args.price_fallback_csv),
    )
    order_preview = _safe_read_json(args.order_requests_preview_json)
    intents_payload = _safe_read_json(args.trade_intents_json)
    if not order_preview:
        if not intents_payload:
            raise FileNotFoundError("trade intents payload not found")
        order_preview = build_order_requests(
            intents_payload=intents_payload,
            holdings=holdings,
            ranking=ranking,
            ord_dvsn=args.ord_dvsn,
        )
    gate = _safe_read_json(args.gate_json)
    payload = convert_order_preview_to_live_preview(
        preview_payload=order_preview,
        intents_payload=intents_payload,
        holdings=holdings,
        ranking=ranking,
        gate=gate,
    )

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    out_md.write_text(render_markdown(payload), encoding="utf-8-sig")
    print(f"preview_json: {out_json}")
    print(f"preview_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
