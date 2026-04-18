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

INPUT_CANDIDATES = DATA_DIR / "ranking_final.csv"
INPUT_PRICE_FALLBACK = DATA_DIR / "ranking_final.csv"
INPUT_GATE = OUTPUT_DIR / "operational_buy_gate.json"
INPUT_LIVE_HOLDINGS = DATA_DIR / "live_account_holdings.csv"
OUT_JSON = OUTPUT_DIR / "live_order_preview.json"
OUT_MD = OUTPUT_DIR / "live_order_preview.md"
STATUS_BUY_ALLOWED = "BUY_ALLOWED"
STATUS_PILOT = "PILOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live-account order preview without sending real orders.")
    parser.add_argument("--candidates-csv", type=Path, default=INPUT_CANDIDATES)
    parser.add_argument("--price-fallback-csv", type=Path, default=INPUT_PRICE_FALLBACK)
    parser.add_argument("--gate-json", type=Path, default=INPUT_GATE)
    parser.add_argument("--live-holdings-csv", type=Path, default=INPUT_LIVE_HOLDINGS)
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.20)
    parser.add_argument("--ord-dvsn", default="01", help="01 market, 00 limit")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path), dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    for col in ["buy_rank", "rank_final", "final_score", "confidence_score", "close"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    if "buy_rank" not in work.columns or work["buy_rank"].isna().all():
        work["buy_rank"] = pd.to_numeric(work.get("rank_final"), errors="coerce")
    if work["buy_rank"].isna().all():
        work["buy_rank"] = (
            pd.to_numeric(work["final_score"], errors="coerce")
            .rank(method="first", ascending=False)
            .astype(float)
        )
    return work.sort_values(["buy_rank", "code"]).reset_index(drop=True)


def load_price_fallback(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame(columns=["code", "close"])
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["code", "close"])
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work.get("date"), errors="coerce")
        work = work.sort_values(["date", "code"], ascending=[False, True], na_position="last")
    return work.loc[:, ["code", "close"]].dropna(subset=["close"]).drop_duplicates("code", keep="first").reset_index(drop=True)


def merge_candidates_with_prices(candidates: pd.DataFrame, price_fallback: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if price_fallback.empty:
        return candidates
    merged = candidates.merge(
        price_fallback.rename(columns={"close": "fallback_close"}),
        on="code",
        how="left",
    )
    merged["close"] = pd.to_numeric(merged.get("close"), errors="coerce").where(
        pd.to_numeric(merged.get("close"), errors="coerce").notna(),
        pd.to_numeric(merged.get("fallback_close"), errors="coerce"),
    )
    return merged.drop(columns=["fallback_close"])


def load_holdings_codes(path: Path) -> set[str]:
    resolved = _resolve(path)
    if not resolved.exists():
        return set()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty or "code" not in df.columns:
        return set()
    return set(df["code"].astype(str).str.zfill(6).tolist())


def load_gate(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _fmt_num(v: object, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{digits}f}"


def _fmt_pct(v: object, digits: int = 2) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def main() -> int:
    args = parse_args()
    candidates = merge_candidates_with_prices(
        load_candidates(args.candidates_csv),
        load_price_fallback(args.price_fallback_csv),
    )
    held_codes = load_holdings_codes(args.live_holdings_csv)
    gate = load_gate(args.gate_json)

    client = KISClient.from_env()
    client.issue_access_token()
    account = resolve_account_env()
    _, summary_df = inquire_balance(client, account)
    cash_summary = summarize_cash(summary_df)
    available_cash = cash_summary.get("dnca_tot_amt")

    selected = candidates.head(args.target_count).copy()
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        code = str(row["code"]).zfill(6)
        ref_price = pd.to_numeric(row.get("close"), errors="coerce")
        order_price = int(ref_price) if pd.notna(ref_price) and ref_price > 0 else 0
        psbl = inquire_psbl_order(
            client,
            account,
            pdno=code,
            ord_unpr=str(order_price),
            ord_dvsn=args.ord_dvsn,
        )
        psbl_row = psbl.iloc[0] if not psbl.empty else pd.Series(dtype="object")
        nrcvb_buy_qty = pd.to_numeric(psbl_row.get("nrcvb_buy_qty"), errors="coerce")
        max_buy_qty = pd.to_numeric(psbl_row.get("max_buy_qty"), errors="coerce")
        planned_qty = compute_market_order_preview_qty(
            available_cash=available_cash,
            total_assets=cash_summary.get("tot_evlu_amt"),
            target_weight=args.max_position_weight,
            current_position_value=None,
            price=float(order_price) if order_price else None,
        )
        allowed_values = [int(v) for v in [nrcvb_buy_qty, max_buy_qty] if pd.notna(v)]
        allowed_qty = min(allowed_values) if allowed_values else 0
        final_qty = min(planned_qty, allowed_qty) if allowed_qty > 0 else 0
        gate_status = str(gate.get("overall_status") or "").upper()
        gate_buy_enabled = gate_status in {STATUS_BUY_ALLOWED, STATUS_PILOT}
        rows.append(
            {
                "code": code,
                "name": row.get("name"),
                "buy_rank": pd.to_numeric(row.get("buy_rank"), errors="coerce"),
                "final_score": pd.to_numeric(row.get("final_score"), errors="coerce"),
                "confidence_score": pd.to_numeric(row.get("confidence_score"), errors="coerce"),
                "reference_price": order_price if order_price else None,
                "held_now": code in held_codes,
                "nrcvb_buy_qty": nrcvb_buy_qty,
                "max_buy_qty": max_buy_qty,
                "planned_qty": planned_qty,
                "final_preview_qty": final_qty,
                "blocked_reason": "already_held" if code in held_codes else ("gate_not_buy_enabled" if not gate_buy_enabled else ("buy_qty_zero" if final_qty <= 0 else "")),
            }
        )

    preview_df = pd.DataFrame(rows)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_dv": account.env_dv,
        "gate_status": gate.get("overall_status"),
        "cash_summary": cash_summary,
        "items": preview_df.where(pd.notna(preview_df), None).to_dict(orient="records"),
    }

    lines = [
        "# Live Order Preview",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- env_dv: {account.env_dv}",
        f"- gate_status: {gate.get('overall_status')}",
        f"- available_cash: {_fmt_num(available_cash)}",
        "",
        "| code | name | buy_rank | ref_price | held_now | planned_qty | final_preview_qty | blocked_reason |",
        "| ---- | ---- | -------- | --------- | -------- | ----------- | ----------------- | -------------- |",
    ]
    for item in payload["items"]:
        lines.append(
            f"| {item.get('code') or ''} | {item.get('name') or ''} | {_fmt_num(item.get('buy_rank'), 0)} | {_fmt_num(item.get('reference_price'), 0)} | {'Y' if item.get('held_now') else 'N'} | {_fmt_num(item.get('planned_qty'), 0)} | {_fmt_num(item.get('final_preview_qty'), 0)} | {item.get('blocked_reason') or ''} |"
        )
    lines.append("")

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"preview_json: {out_json}")
    print(f"preview_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
