from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kis_client import KISClient
from kis_live_account import (
    compute_market_order_preview_qty,
    inquire_balance,
    inquire_psbl_order,
    resolve_account_env,
    summarize_cash,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_TRADE_INTENTS = OUTPUT_DIR / "trade_intents.json"
INPUT_LIVE_HOLDINGS = DATA_DIR / "live_account_holdings.csv"
INPUT_RANKING = DATA_DIR / "buy_candidates_top5.csv"
OUT_JSON = OUTPUT_DIR / "order_requests_preview.json"
OUT_MD = OUTPUT_DIR / "order_requests_preview.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert executable trade intents into dry-run order request previews.")
    parser.add_argument("--trade-intents-json", type=Path, default=INPUT_TRADE_INTENTS)
    parser.add_argument("--live-holdings-csv", type=Path, default=INPUT_LIVE_HOLDINGS)
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--ord-dvsn", default="01", help="01 market, 00 limit")
    parser.add_argument("--execute", action="store_true", help="Reserved for future live submission. Currently preview only.")
    parser.add_argument("--confirm-text", default="", help="Reserved safety text for future live submission.")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(v: object, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{digits}f}"


def _safe_read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_live_holdings(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    for col in ["qty", "avg_price", "current_price", "weight"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    return work


def load_ranking(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    for col in ["close", "buy_rank", "final_score", "confidence_score"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    return work.sort_values(["buy_rank", "code"]).reset_index(drop=True)


def build_order_requests(
    *,
    intents_payload: dict[str, Any],
    holdings: pd.DataFrame,
    ranking: pd.DataFrame,
    ord_dvsn: str,
) -> dict[str, Any]:
    intents = pd.DataFrame(intents_payload.get("intents") or [])
    intents = intents.loc[intents.get("executable", False).fillna(False)].copy() if not intents.empty else pd.DataFrame()
    if intents.empty:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asof_date": intents_payload.get("asof_date"),
            "gate_status": intents_payload.get("gate_status"),
            "env_dv": None,
            "cash_summary": {},
            "items": [],
            "summary": {
                "request_count": 0,
                "buy_count": 0,
                "sell_count": 0,
            },
        }

    ranking_lookup = ranking.drop_duplicates("code").set_index("code") if not ranking.empty else pd.DataFrame()
    holdings_lookup = holdings.drop_duplicates("code").set_index("code") if not holdings.empty else pd.DataFrame()
    has_buy_intents = intents.get("intent_type", pd.Series(dtype="object")).astype(str).str.upper().eq("BUY").any()
    client = None
    account = None
    cash_summary: dict[str, Any] = {}
    available_cash = None
    if has_buy_intents:
        client = KISClient.from_env()
        client.issue_access_token()
        account = resolve_account_env()
        _, summary_df = inquire_balance(client, account)
        cash_summary = summarize_cash(summary_df)
        available_cash = cash_summary.get("dnca_tot_amt")

    items: list[dict[str, Any]] = []
    for _, row in intents.iterrows():
        code = str(row.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        ranking_row = ranking_lookup.loc[code] if not ranking_lookup.empty and code in ranking_lookup.index else pd.Series(dtype="object")
        holding_row = holdings_lookup.loc[code] if not holdings_lookup.empty and code in holdings_lookup.index else pd.Series(dtype="object")
        intent_type = str(row.get("intent_type") or "").upper()
        side = "BUY" if intent_type == "BUY" else "SELL" if intent_type in {"TRIM", "EXIT"} else "HOLD"
        reference_price = pd.to_numeric(ranking_row.get("close"), errors="coerce")
        order_price = int(reference_price) if pd.notna(reference_price) and reference_price > 0 else 0
        planned_qty = None
        allowed_qty = None
        final_qty = None
        blocked_reason = None

        if side == "BUY":
            if client is None or account is None:
                blocked_reason = "buy_context_unavailable"
                final_qty = 0
                planned_qty = 0
                allowed_qty = 0
                items.append(
                    {
                        "request_id": f"{intents_payload.get('asof_date') or 'unknown'}:{intent_type}:{code}",
                        "intent_id": row.get("intent_id"),
                        "code": code,
                        "name": str(ranking_row.get("name") or row.get("name") or "").strip() or None,
                        "side": side,
                        "intent_type": intent_type,
                        "ord_dvsn": ord_dvsn,
                        "reference_price": order_price if order_price else None,
                        "planned_qty": 0,
                        "allowed_qty": 0,
                        "final_request_qty": 0,
                        "target_weight": pd.to_numeric(row.get("target_weight"), errors="coerce"),
                        "priority": pd.to_numeric(row.get("priority"), errors="coerce"),
                        "reason": row.get("reason"),
                        "blocked_reason": blocked_reason,
                        "executable_now": False,
                    }
                )
                continue
            psbl = inquire_psbl_order(
                client,
                account,
                pdno=code,
                ord_unpr=str(order_price),
                ord_dvsn=ord_dvsn,
            )
            psbl_row = psbl.iloc[0] if not psbl.empty else pd.Series(dtype="object")
            nrcvb_buy_qty = pd.to_numeric(psbl_row.get("nrcvb_buy_qty"), errors="coerce")
            max_buy_qty = pd.to_numeric(psbl_row.get("max_buy_qty"), errors="coerce")
            target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
            planned_qty = compute_market_order_preview_qty(
                available_cash=available_cash,
                target_weight=float(target_weight) if pd.notna(target_weight) else 0.0,
                price=float(order_price) if order_price else None,
            )
            allowed_qty = int(max(nrcvb_buy_qty, max_buy_qty)) if pd.notna(nrcvb_buy_qty) or pd.notna(max_buy_qty) else 0
            final_qty = min(planned_qty, allowed_qty) if allowed_qty and allowed_qty > 0 else planned_qty
            if not final_qty or final_qty <= 0:
                blocked_reason = "buy_qty_zero"
        elif side == "SELL":
            current_qty = pd.to_numeric(holding_row.get("qty"), errors="coerce")
            if not pd.notna(current_qty) or float(current_qty) <= 0:
                blocked_reason = "holding_qty_missing"
                final_qty = 0
            else:
                if intent_type == "TRIM":
                    weight = pd.to_numeric(holding_row.get("weight"), errors="coerce")
                    target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
                    trim_ratio = 0.5
                    if pd.notna(weight) and pd.notna(target_weight) and weight > 0:
                        trim_ratio = max(min((float(weight) - float(target_weight)) / float(weight), 1.0), 0.0)
                    final_qty = max(int(round(float(current_qty) * trim_ratio)), 1)
                else:
                    final_qty = int(round(float(current_qty)))
                planned_qty = final_qty
                allowed_qty = final_qty
                order_price = 0
        else:
            blocked_reason = "non_executable_intent_type"
            final_qty = 0

        items.append(
            {
                "request_id": f"{intents_payload.get('asof_date') or 'unknown'}:{intent_type}:{code}",
                "intent_id": row.get("intent_id"),
                "code": code,
                "name": str(ranking_row.get("name") or row.get("name") or "").strip() or None,
                "side": side,
                "intent_type": intent_type,
                "ord_dvsn": ord_dvsn,
                "reference_price": order_price if order_price else None,
                "planned_qty": int(planned_qty) if pd.notna(planned_qty) else None,
                "allowed_qty": int(allowed_qty) if pd.notna(allowed_qty) else None,
                "final_request_qty": int(final_qty) if final_qty is not None else None,
                "target_weight": pd.to_numeric(row.get("target_weight"), errors="coerce"),
                "priority": pd.to_numeric(row.get("priority"), errors="coerce"),
                "reason": row.get("reason"),
                "blocked_reason": blocked_reason,
                "executable_now": blocked_reason in {None, ""},
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": intents_payload.get("asof_date"),
        "gate_status": intents_payload.get("gate_status"),
        "env_dv": account.env_dv if account is not None else None,
        "cash_summary": cash_summary,
        "items": items,
        "summary": {
            "request_count": len(items),
            "buy_count": sum(1 for item in items if item["side"] == "BUY"),
            "sell_count": sum(1 for item in items if item["side"] == "SELL"),
        },
    }


def main() -> int:
    args = parse_args()
    if args.execute:
        raise ValueError("Live execution is not enabled yet. This command currently supports dry-run preview only.")

    intents_payload = _safe_read_json(args.trade_intents_json)
    if not intents_payload:
        raise FileNotFoundError("trade intents payload not found")

    holdings = load_live_holdings(args.live_holdings_csv)
    ranking = load_ranking(args.ranking_csv)
    payload = build_order_requests(
        intents_payload=intents_payload,
        holdings=holdings,
        ranking=ranking,
        ord_dvsn=args.ord_dvsn,
    )

    lines = [
        "# Order Requests Preview",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- asof_date: {payload.get('asof_date') or 'NA'}",
        f"- gate_status: {payload.get('gate_status') or 'NA'}",
        f"- env_dv: {payload.get('env_dv') or 'NA'}",
        f"- request_count: {payload['summary']['request_count']}",
        f"- buy_count: {payload['summary']['buy_count']}",
        f"- sell_count: {payload['summary']['sell_count']}",
        "",
        "| request_id | code | name | side | intent_type | ref_price | final_qty | executable_now | blocked_reason | reason |",
        "| ---------- | ---- | ---- | ---- | ----------- | --------- | --------- | -------------- | -------------- | ------ |",
    ]
    for item in payload["items"]:
        lines.append(
            f"| {item.get('request_id') or ''} | {item.get('code') or ''} | {item.get('name') or ''} | {item.get('side') or ''} | {item.get('intent_type') or ''} | {_fmt_num(item.get('reference_price'), 0)} | {_fmt_num(item.get('final_request_qty'), 0)} | {'Y' if item.get('executable_now') else 'N'} | {item.get('blocked_reason') or ''} | {item.get('reason') or ''} |"
        )
    lines.append("")

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"order_requests_preview_json: {out_json}")
    print(f"order_requests_preview_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
