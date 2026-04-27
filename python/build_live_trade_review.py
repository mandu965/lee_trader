from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from db import get_engine
from payload_store import upsert_json_payload
from sync_live_trade_ledger import ensure_tables


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_trade_review_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_trade_review_report.md"
DEFAULT_HORIZONS = (0, 1, 3, 5, 10)
REVIEWER = "auto_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatic post-trade review rows from live fills.")
    parser.add_argument("--as-of-date", default="", help="Execution basis date YYYY-MM-DD. Defaults to all fills.")
    parser.add_argument("--review-date", default="", help="Review date YYYY-MM-DD. Defaults to latest price date.")
    parser.add_argument("--prices-csv", type=Path, default=DEFAULT_PRICES_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _num(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def load_prices(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"price csv not found: {resolved}")
    prices = pd.read_csv(resolved, usecols=["date", "code", "adj_close"], dtype={"code": str}, low_memory=False)
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.dropna(subset=["date", "code", "adj_close"]).sort_values(["code", "date"]).reset_index(drop=True)
    return prices


def load_fill_rows(as_of_date: str) -> list[dict[str, Any]]:
    params = {"as_of_date": as_of_date} if as_of_date else {}
    filter_sql = "AND e.as_of_date = CAST(:as_of_date AS date)" if as_of_date else ""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                WITH latest_execution AS (
                    SELECT DISTINCT ON (request_id)
                        request_id, broker_order_id, intent_id, as_of_date, intent_type, submission_status, executed_at, updated_at
                    FROM research.live_order_execution
                    ORDER BY request_id, COALESCE(executed_at, updated_at) DESC
                )
                SELECT
                    f.fill_id,
                    f.request_id,
                    COALESCE(f.broker_order_id, e.broker_order_id) AS broker_order_id,
                    f.code,
                    f.name,
                    f.side,
                    f.as_of_date AS fill_as_of_date,
                    f.filled_at,
                    f.filled_qty,
                    f.filled_price,
                    f.filled_amount,
                    e.intent_id,
                    e.as_of_date AS execution_as_of_date,
                    e.intent_type,
                    e.submission_status,
                    r.ranking_rank,
                    r.final_score,
                    r.confidence_score,
                    r.risk_penalty,
                    r.reason
                FROM research.live_order_fill f
                LEFT JOIN latest_execution e
                  ON e.request_id = f.request_id
                LEFT JOIN research.live_order_request r
                  ON r.request_id = f.request_id
                WHERE f.request_id IS NOT NULL
                  {filter_sql}
                ORDER BY f.filled_at, f.request_id
                """
            ),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def _price_points(prices: pd.DataFrame, code: str, fill_date: str) -> dict[int, dict[str, Any]]:
    series = prices.loc[(prices["code"] == str(code).zfill(6)) & (prices["date"] >= fill_date), ["date", "adj_close"]]
    series = series.reset_index(drop=True)
    out: dict[int, dict[str, Any]] = {}
    for horizon in DEFAULT_HORIZONS:
        if horizon < len(series):
            row = series.iloc[horizon]
            out[horizon] = {"date": str(row["date"]), "price": float(row["adj_close"])}
    return out


def _signed_return(side: str, fill_price: float, mark_price: float) -> float:
    if fill_price <= 0 or mark_price <= 0:
        return 0.0
    side_upper = str(side or "").upper()
    if side_upper == "BUY":
        return mark_price / fill_price - 1.0
    return fill_price / mark_price - 1.0


def _outcome_label(best_horizon: int | None, signed_return: float | None) -> str:
    if best_horizon is None or signed_return is None:
        return "pending_price_data"
    if best_horizon == 0:
        return "same_day_observed"
    if signed_return >= 0.03:
        return "positive"
    if signed_return <= -0.03:
        return "negative"
    return "neutral"


def build_review_items(fills: list[dict[str, Any]], prices: pd.DataFrame, review_date: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in fills:
        fill_dt = pd.to_datetime(row.get("filled_at"), errors="coerce")
        fill_date = fill_dt.strftime("%Y-%m-%d") if pd.notna(fill_dt) else ""
        code = str(row.get("code") or "").zfill(6)
        fill_price = _num(row.get("filled_price")) or 0.0
        points = _price_points(prices, code, fill_date) if fill_date and code else {}

        returns: dict[str, dict[str, Any]] = {}
        best_horizon = None
        best_signed_return = None
        for horizon, point in points.items():
            signed = _signed_return(str(row.get("side") or ""), fill_price, float(point["price"]))
            returns[f"d{horizon}"] = {
                "date": point["date"],
                "price": point["price"],
                "signed_return": signed,
            }
            best_horizon = horizon
            best_signed_return = signed

        label = _outcome_label(best_horizon, best_signed_return)
        tags = [
            f"side:{str(row.get('side') or '').upper()}",
            f"intent:{row.get('intent_type') or 'UNKNOWN'}",
            f"basis:{row.get('execution_as_of_date') or '-'}",
        ]
        if best_horizon is not None:
            tags.append(f"horizon:d{best_horizon}")
        if row.get("risk_penalty") is not None:
            tags.append(f"risk_penalty:{float(row['risk_penalty']):.2f}")

        return_text = (
            f"d{best_horizon}_signed_return={best_signed_return:.2%}"
            if best_horizon is not None and best_signed_return is not None
            else "return_pending"
        )
        note = (
            f"fill_date={fill_date}, fill_price={fill_price:.2f}, {return_text}, "
            f"rank={row.get('ranking_rank') or '-'}, confidence={row.get('confidence_score') or '-'}"
        )
        next_action = "Review again after more post-fill prices mature." if best_horizon is not None and best_horizon < 5 else "No immediate action from automatic review."

        items.append(
            {
                "intent_id": row.get("intent_id"),
                "request_id": row.get("request_id"),
                "code": code,
                "name": row.get("name"),
                "side": row.get("side"),
                "intent_type": row.get("intent_type"),
                "review_date": review_date,
                "fill_date": fill_date,
                "filled_at": row.get("filled_at"),
                "filled_qty": _num(row.get("filled_qty")),
                "filled_price": fill_price,
                "outcome_label": label,
                "pre_tags": tags[:3],
                "post_tags": tags,
                "review_note": note,
                "next_action_note": next_action,
                "returns": returns,
                "reason": row.get("reason"),
            }
        )
    return items


def write_review_rows(items: list[dict[str, Any]], review_date: str) -> int:
    ensure_tables()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM research.live_trade_review
                WHERE review_date = CAST(:review_date AS date)
                  AND reviewer = :reviewer
                """
            ),
            {"review_date": review_date, "reviewer": REVIEWER},
        )
        if not items:
            return 0
        conn.execute(
            text(
                """
                INSERT INTO research.live_trade_review (
                    intent_id, request_id, code, review_date, pre_tags, post_tags,
                    outcome_label, review_note, next_action_note, reviewer, updated_at
                )
                VALUES (
                    :intent_id, :request_id, :code, CAST(:review_date AS date), :pre_tags, :post_tags,
                    :outcome_label, :review_note, :next_action_note, :reviewer, now()
                )
                """
            ),
            [
                {
                    "intent_id": item.get("intent_id"),
                    "request_id": item.get("request_id"),
                    "code": item.get("code"),
                    "review_date": review_date,
                    "pre_tags": item.get("pre_tags") or [],
                    "post_tags": item.get("post_tags") or [],
                    "outcome_label": item.get("outcome_label"),
                    "review_note": item.get("review_note"),
                    "next_action_note": item.get("next_action_note"),
                    "reviewer": REVIEWER,
                }
                for item in items
            ],
        )
    return len(items)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Trade Review Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- review_date: `{report['review_date']}`",
        f"- as_of_date: `{report.get('as_of_date') or '-'}`",
        f"- reviewed_count: `{report['reviewed_count']}`",
        f"- price_latest_date: `{report['price_latest_date']}`",
        "",
        "## Outcome Summary",
        "",
        "| outcome | count |",
        "| --- | ---: |",
    ]
    for row in report["outcome_counts"]:
        lines.append(f"| {row['outcome_label']} | {row['count']} |")
    if not report["outcome_counts"]:
        lines.append("| - | 0 |")

    lines.extend(
        [
            "",
            "## Reviewed Fills",
            "",
            "| request_id | code | side | intent | fill_price | outcome | note |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for item in report["items"]:
        lines.append(
            "| {request_id} | {code} | {side} | {intent_type} | {filled_price:.2f} | {outcome_label} | {review_note} |".format(
                request_id=item.get("request_id") or "",
                code=item.get("code") or "",
                side=item.get("side") or "",
                intent_type=item.get("intent_type") or "",
                filled_price=float(item.get("filled_price") or 0.0),
                outcome_label=item.get("outcome_label") or "",
                review_note=str(item.get("review_note") or "").replace("|", "/"),
            )
        )
    if not report["items"]:
        lines.append("| - | - | - | - | 0 | - | - |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prices = load_prices(args.prices_csv)
    price_latest_date = str(prices["date"].max()) if not prices.empty else ""
    review_date = str(args.review_date or price_latest_date or datetime.now().date().isoformat())
    fills = load_fill_rows(str(args.as_of_date or "").strip())
    items = build_review_items(fills, prices, review_date)
    inserted = write_review_rows(items, review_date)
    outcome_counts = (
        pd.DataFrame(items).groupby("outcome_label").size().rename("count").reset_index().to_dict(orient="records")
        if items
        else []
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review_date": review_date,
        "as_of_date": str(args.as_of_date or ""),
        "price_latest_date": price_latest_date,
        "reviewed_count": inserted,
        "outcome_counts": outcome_counts,
        "items": items,
    }

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    upsert_json_payload("live_trade_review_report", report, asof_date=review_date, generated_at=report["generated_at"], source_path=out_json)
    logging.info("Built live trade review rows=%d output=%s", inserted, out_json)
    print(f"live_trade_review_report_json: {out_json}")
    print(f"live_trade_review_report_md: {out_md}")
    print(f"reviewed_count: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
