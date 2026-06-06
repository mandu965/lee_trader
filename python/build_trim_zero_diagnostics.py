from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_ORDER_PREVIEW_JSON = OUTPUT_DIR / "order_requests_preview.json"
DEFAULT_TRADE_INTENTS_JSON = OUTPUT_DIR / "trade_intents.json"
DEFAULT_HOLDINGS_CSV = DATA_DIR / "live_account_holdings.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "trim_zero_diagnostics.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "trim_zero_diagnostics.md"

TRIM_RATIO_EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose TRIM requests skipped by trim_ratio_zero.")
    parser.add_argument("--order-preview-json", type=Path, default=DEFAULT_ORDER_PREVIEW_JSON)
    parser.add_argument("--trade-intents-json", type=Path, default=DEFAULT_TRADE_INTENTS_JSON)
    parser.add_argument("--holdings-csv", type=Path, default=DEFAULT_HOLDINGS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"json not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _read_holdings(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"holdings csv not found: {resolved}")
    frame = pd.read_csv(resolved, dtype={"code": str}, encoding="utf-8-sig")
    if frame.empty:
        return pd.DataFrame(columns=["code"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def _num_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _int_or_none(value: object) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(round(float(numeric)))


def _pick_name(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text and "?" not in text and "\ufffd" not in text:
            return text
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _lookup_by_code(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        code = str(item.get("code") or "").zfill(6)
        if code and code != "000000":
            lookup[code] = item
    return lookup


def _diagnose_trim(
    *,
    blocked_reason: str,
    holding_exists: bool,
    current_weight: float | None,
    target_weight: float | None,
    trim_ratio: float | None,
    final_qty: int | None,
) -> tuple[str, str]:
    if not holding_exists:
        return "holding_missing", "Live holding row is missing, so the TRIM request cannot be sized."
    if current_weight is None or target_weight is None or current_weight <= 0:
        return "weight_data_missing", "Current or target weight is missing, so the TRIM ratio cannot be calculated."
    if blocked_reason == "trim_ratio_zero" or (trim_ratio is not None and trim_ratio < TRIM_RATIO_EPSILON):
        return (
            "target_weight_at_or_above_current_weight",
            "Current weight and target weight are effectively the same, or target is higher, so no shares need to be sold.",
        )
    if final_qty is not None and final_qty <= 0:
        return "final_qty_zero", "The calculated final quantity is zero, so the order is skipped."
    return "trim_has_executable_quantity", "The TRIM request has a positive sell quantity."


def build_diagnostics(
    *,
    order_preview: dict[str, Any],
    trade_intents: dict[str, Any],
    holdings: pd.DataFrame,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now()
    preview_items = list(order_preview.get("items") or [])
    intent_items = list(trade_intents.get("intents") or [])
    intent_lookup = _lookup_by_code(intent_items)
    holdings_lookup = holdings.drop_duplicates("code").set_index("code") if not holdings.empty else pd.DataFrame()

    trim_preview_items = [
        item
        for item in preview_items
        if str(item.get("intent_type") or "").upper() == "TRIM"
        or str(item.get("blocked_reason") or "").strip() == "trim_ratio_zero"
    ]

    rows: list[dict[str, Any]] = []
    for item in trim_preview_items:
        code = str(item.get("code") or "").zfill(6)
        intent = intent_lookup.get(code, {})
        holding = holdings_lookup.loc[code] if not holdings_lookup.empty and code in holdings_lookup.index else pd.Series(dtype="object")
        holding_exists = not holding.empty

        current_qty = _int_or_none(holding.get("qty"))
        current_weight = _num_or_none(holding.get("weight"))
        target_weight = _num_or_none(item.get("target_weight"))
        if target_weight is None:
            target_weight = _num_or_none(intent.get("target_weight"))
        current_price = _num_or_none(holding.get("current_price"))
        current_value = _num_or_none(holding.get("eval_amount"))
        final_qty = _int_or_none(item.get("final_request_qty"))
        blocked_reason = str(item.get("blocked_reason") or "").strip()

        trim_ratio: float | None = None
        if current_weight is not None and target_weight is not None and current_weight > 0:
            trim_ratio = max(min((current_weight - target_weight) / current_weight, 1.0), 0.0)

        expected_sell_qty: int | None = None
        if current_qty is not None and trim_ratio is not None:
            expected_sell_qty = 0 if trim_ratio < TRIM_RATIO_EPSILON else max(int(round(current_qty * trim_ratio)), 1)

        target_value: float | None = None
        value_delta: float | None = None
        if current_value is not None and current_weight is not None and current_weight > 0 and target_weight is not None:
            total_assets_estimate = current_value / current_weight
            target_value = total_assets_estimate * target_weight
            value_delta = current_value - target_value

        diagnosis_code, diagnosis_message = _diagnose_trim(
            blocked_reason=blocked_reason,
            holding_exists=holding_exists,
            current_weight=current_weight,
            target_weight=target_weight,
            trim_ratio=trim_ratio,
            final_qty=final_qty,
        )

        rows.append(
            {
                "code": code,
                "name": _pick_name(holding.get("name"), item.get("name"), intent.get("name")),
                "intent_type": str(item.get("intent_type") or intent.get("intent_type") or "").upper(),
                "blocked_reason": blocked_reason or None,
                "executable_now": bool(item.get("executable_now")),
                "current_qty": current_qty,
                "final_request_qty": final_qty,
                "expected_sell_qty_from_weights": expected_sell_qty,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_delta": None if current_weight is None or target_weight is None else current_weight - target_weight,
                "trim_ratio": trim_ratio,
                "current_price": current_price,
                "current_value": current_value,
                "target_value_estimate": target_value,
                "value_delta_estimate": value_delta,
                "ranking_rank": _int_or_none(item.get("ranking_rank") or intent.get("ranking_rank")),
                "buy_rank": _int_or_none(item.get("buy_rank") or intent.get("buy_rank")),
                "reason": item.get("reason") or intent.get("reason"),
                "diagnosis_code": diagnosis_code,
                "diagnosis_message": diagnosis_message,
            }
        )

    diagnosis_counts = Counter(row["diagnosis_code"] for row in rows)
    return {
        "run_id": order_preview.get("run_id") or trade_intents.get("run_id"),
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "order_preview_generated_at": order_preview.get("generated_at"),
            "trade_intents_generated_at": trade_intents.get("generated_at"),
            "asof_date": order_preview.get("asof_date") or trade_intents.get("asof_date"),
        },
        "summary": {
            "preview_item_count": len(preview_items),
            "trim_preview_count": len(trim_preview_items),
            "trim_ratio_zero_count": sum(1 for row in rows if row.get("blocked_reason") == "trim_ratio_zero"),
            "executable_trim_count": sum(1 for row in rows if row.get("executable_now")),
            "diagnosis_counts": dict(diagnosis_counts),
        },
        "items": rows,
    }


def _fmt_pct(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _fmt_num(value: object, digits: int = 6) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _fmt_int(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{int(round(float(numeric))):,}"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    source = payload.get("source") or {}
    rows = list(payload.get("items") or [])

    lines = [
        "# TRIM zero diagnostics",
        "",
        "## Summary",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- asof_date: {source.get('asof_date')}",
        f"- run_id: {payload.get('run_id')}",
        f"- trim_preview_count: {summary.get('trim_preview_count', 0)}",
        f"- trim_ratio_zero_count: {summary.get('trim_ratio_zero_count', 0)}",
        f"- executable_trim_count: {summary.get('executable_trim_count', 0)}",
        "",
        "## Diagnosis",
        "",
    ]
    if not rows:
        lines.append("_No TRIM preview items found._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| code | name | blocked | current_weight | target_weight | delta | trim_ratio | qty | expected_sell_qty | diagnosis |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("code") or ""),
                    str(row.get("name") or ""),
                    str(row.get("blocked_reason") or ""),
                    _fmt_pct(row.get("current_weight")),
                    _fmt_pct(row.get("target_weight")),
                    _fmt_num(row.get("weight_delta"), 10),
                    _fmt_num(row.get("trim_ratio"), 10),
                    _fmt_int(row.get("current_qty")),
                    _fmt_int(row.get("expected_sell_qty_from_weights")),
                    str(row.get("diagnosis_code") or ""),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `trim_ratio_zero` means the submit preview calculated `(current_weight - target_weight) / current_weight` as nearly zero.",
            "- In this state, the order layer is not finding a sellable reduction amount; it is skipping the TRIM safely.",
            "- The usual cause is semantic: a TRIM intent can mean `out of top10 but within hold range top20; live grade C`, not necessarily `sell some shares now`.",
            "- If these rows repeat, review whether the intent label should distinguish `TRIM_REVIEW` from executable `TRIM`; that would be a later behavior/design change, not required for order safety.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], out_json: Path, out_md: Path) -> None:
    resolved_json = _resolve(out_json)
    resolved_md = _resolve(out_md)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_md.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    resolved_md.write_text(render_markdown(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_diagnostics(
        order_preview=_read_json(args.order_preview_json),
        trade_intents=_read_json(args.trade_intents_json),
        holdings=_read_holdings(args.holdings_csv),
    )
    write_outputs(payload, args.out_json, args.out_md)
    print(f"trim_zero_diagnostics_json: {_resolve(args.out_json)}")
    print(f"trim_zero_diagnostics_md: {_resolve(args.out_md)}")


if __name__ == "__main__":
    main()
