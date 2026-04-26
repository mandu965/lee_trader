from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from kis_client import KISClient
from kis_live_account import inquire_daily_ccld, resolve_account_env
from sync_live_trade_ledger import ensure_tables


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUT_JSON = OUTPUT_DIR / "live_order_fills.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync KIS daily order fills into research.live_order_fill.")
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD or YYYYMMDD. Defaults to recent execution date window.")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD or YYYYMMDD. Defaults to today.")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--query-all", action="store_true", help="Query all orders in the date range instead of known submitted broker orders.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    return text_value.replace("-", "")[:8]


def _num(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _side_from_row(row: pd.Series) -> str:
    for key in ("sll_buy_dvsn_cd_name", "sll_buy_dvsn_cd"):
        value = str(row.get(key) or "").strip().upper()
        if value in {"BUY", "매수", "02"}:
            return "BUY"
        if value in {"SELL", "매도", "01"}:
            return "SELL"
    return ""


def _code_from_row(row: pd.Series) -> str:
    for key in ("pdno", "pdno_code", "prdt_code"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.zfill(6)
    return ""


def _name_from_row(row: pd.Series) -> str | None:
    for key in ("prdt_name", "prdt_name1", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def _broker_order_id_from_row(row: pd.Series) -> str | None:
    for key in ("odno", "orgn_odno", "ODNO"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def _filled_qty_from_row(row: pd.Series) -> float | None:
    for key in ("tot_ccld_qty", "ccld_qty", "ord_qty"):
        value = _num(row.get(key))
        if value is not None and value > 0:
            return value
    ord_qty = _num(row.get("ord_qty"))
    rmn_qty = _num(row.get("rmn_qty"))
    if ord_qty is not None and rmn_qty is not None:
        return max(ord_qty - rmn_qty, 0.0)
    return None


def _filled_price_from_row(row: pd.Series, filled_qty: float | None) -> float | None:
    for key in ("avg_prvs", "avg_pric", "ccld_unpr", "ord_unpr"):
        value = _num(row.get(key))
        if value is not None and value > 0:
            return value
    amount = _num(row.get("tot_ccld_amt"))
    if amount is not None and filled_qty:
        return amount / filled_qty
    return None


def _filled_at_from_row(row: pd.Series, fallback_date: str) -> datetime | None:
    order_date = str(row.get("ord_dt") or fallback_date or "").replace("-", "")[:8]
    time_value = str(row.get("ord_tmd") or row.get("ccld_tmd") or "").strip()
    if order_date and time_value:
        time_value = time_value.zfill(6)[:6]
        try:
            return datetime.strptime(order_date + time_value, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    if order_date:
        try:
            return datetime.strptime(order_date, "%Y%m%d")
        except ValueError:
            return None
    return None


def load_known_submitted_orders(start_date: str, end_date: str) -> list[dict[str, Any]]:
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT request_id, broker_order_id, broker_org_order_id, code, side, as_of_date, executed_at
                FROM research.live_order_execution
                WHERE submission_status = 'submitted'
                  AND broker_order_id IS NOT NULL
                  AND COALESCE(as_of_date, executed_at::date) BETWEEN CAST(:start_date AS date) AND CAST(:end_date AS date)
                ORDER BY COALESCE(executed_at, submitted_at) DESC
                """
            ),
            {"start_date": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}", "end_date": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"},
        ).mappings().all()
    return [dict(row) for row in rows]


def insert_fill_rows(fill_rows: list[dict[str, Any]]) -> int:
    if not fill_rows:
        return 0
    ensure_tables()
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        for row in fill_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO research.live_order_fill (
                        request_id, broker_order_id, broker_org_order_id, as_of_date, filled_at,
                        code, name, side, filled_qty, filled_price, filled_amount,
                        fee, tax, fill_status, source, raw_response_json, updated_at
                    )
                    VALUES (
                        :request_id, :broker_order_id, :broker_org_order_id, :as_of_date, :filled_at,
                        :code, :name, :side, :filled_qty, :filled_price, :filled_amount,
                        :fee, :tax, :fill_status, :source, CAST(:raw_response_json AS jsonb), now()
                    )
                    ON CONFLICT (broker_order_id, code, side, filled_at, filled_qty, filled_price) DO UPDATE SET
                        request_id = EXCLUDED.request_id,
                        broker_org_order_id = EXCLUDED.broker_org_order_id,
                        as_of_date = EXCLUDED.as_of_date,
                        name = EXCLUDED.name,
                        filled_amount = EXCLUDED.filled_amount,
                        fee = EXCLUDED.fee,
                        tax = EXCLUDED.tax,
                        fill_status = EXCLUDED.fill_status,
                        source = EXCLUDED.source,
                        raw_response_json = EXCLUDED.raw_response_json,
                        updated_at = now()
                    """
                ),
                row,
            )
            count += 1
    return count


def build_fill_rows(frame: pd.DataFrame, *, request_map: dict[str, dict[str, Any]], fallback_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.where(pd.notna(frame), None).to_dict(orient="records"):
        row = pd.Series(raw)
        broker_order_id = _broker_order_id_from_row(row)
        code = _code_from_row(row)
        side = _side_from_row(row)
        filled_qty = _filled_qty_from_row(row)
        filled_price = _filled_price_from_row(row, filled_qty)
        if not broker_order_id or not code or not side or not filled_qty or not filled_price:
            continue
        request = request_map.get(broker_order_id, {})
        filled_at = _filled_at_from_row(row, fallback_date)
        as_of_date = str(row.get("ord_dt") or request.get("as_of_date") or fallback_date).replace("-", "")[:8]
        rows.append(
            {
                "request_id": request.get("request_id"),
                "broker_order_id": broker_order_id,
                "broker_org_order_id": request.get("broker_org_order_id") or str(row.get("orgn_odno") or "").strip() or None,
                "as_of_date": f"{as_of_date[:4]}-{as_of_date[4:6]}-{as_of_date[6:8]}" if len(as_of_date) == 8 else None,
                "filled_at": filled_at,
                "code": code,
                "name": _name_from_row(row),
                "side": side,
                "filled_qty": filled_qty,
                "filled_price": filled_price,
                "filled_amount": _num(row.get("tot_ccld_amt")) or (filled_qty * filled_price),
                "fee": _num(row.get("ord_tlex") or row.get("fee")),
                "tax": _num(row.get("stax") or row.get("tax")),
                "fill_status": "FILLED" if (_num(row.get("rmn_qty")) or 0.0) == 0.0 else "PARTIAL_FILLED",
                "source": "kis_inquire_daily_ccld",
                "raw_response_json": json.dumps(raw, ensure_ascii=False, default=str),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    end_date = _date_text(args.end_date) or date.today().strftime("%Y%m%d")
    start_date = _date_text(args.start_date) or (datetime.strptime(end_date, "%Y%m%d").date() - timedelta(days=args.lookback_days)).strftime("%Y%m%d")

    ensure_tables()
    known_orders = [] if args.query_all else load_known_submitted_orders(start_date, end_date)
    request_map = {str(row.get("broker_order_id")): row for row in known_orders if row.get("broker_order_id")}

    if not args.query_all and not request_map:
        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_date": start_date,
            "end_date": end_date,
            "known_order_count": 0,
            "api_row_count": 0,
            "fill_row_count": 0,
            "inserted_count": 0,
            "items": [],
            "skip_reason": "no_submitted_broker_orders",
        }
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logging.info("Skip live order fill sync: no submitted broker orders in range")
        print(f"live_order_fills_json: {out_json}")
        return 0

    client = KISClient.from_env()
    client.issue_access_token()
    account = resolve_account_env()

    frames: list[pd.DataFrame] = []
    if args.query_all:
        frame, _ = inquire_daily_ccld(client, account, start_date=start_date, end_date=end_date)
        frames.append(frame)
    elif not request_map:
        frames = []
    else:
        for order in known_orders:
            frame, _ = inquire_daily_ccld(
                client,
                account,
                start_date=start_date,
                end_date=end_date,
                pdno=str(order.get("code") or ""),
                odno=str(order.get("broker_order_id") or ""),
            )
            frames.append(frame)

    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    fill_rows = build_fill_rows(combined, request_map=request_map, fallback_date=end_date)
    inserted = insert_fill_rows(fill_rows)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": start_date,
        "end_date": end_date,
        "known_order_count": len(known_orders),
        "api_row_count": int(len(combined)),
        "fill_row_count": len(fill_rows),
        "inserted_count": inserted,
        "items": fill_rows,
    }
    out_json = _resolve(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logging.info("Synced live order fills: api_rows=%d fill_rows=%d inserted=%d", len(combined), len(fill_rows), inserted)
    print(f"live_order_fills_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
