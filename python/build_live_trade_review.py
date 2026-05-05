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
                    r.engine_type,
                    r.strategy_id,
                    r.run_mode,
                    r.source_score_date,
                    r.ranking_rank,
                    r.final_score,
                    r.prob_score,
                    r.ret_score,
                    r.tech_score,
                    COALESCE(r.quality_score, r.qual_score) AS quality_score,
                    r.confidence_score,
                    r.calibrated_confidence,
                    r.live_confidence_grade,
                    r.liquidity_score,
                    r.market_regime,
                    r.previous_close,
                    r.live_price,
                    r.entry_price_gap_pct,
                    r.entry_gate_status,
                    r.entry_gate_reason,
                    r.buy_reason,
                    r.sell_reason,
                    r.portfolio_action_reason,
                    r.risk_penalty,
                    r.reason,
                    c.realized_return AS realized_return_until_exit,
                    c.holding_days AS closed_holding_days
                FROM research.live_order_fill f
                LEFT JOIN latest_execution e
                  ON e.request_id = f.request_id
                LEFT JOIN research.live_order_request r
                  ON r.request_id = f.request_id
                LEFT JOIN analytics.live_closed_trade c
                  ON c.sell_fill_id = f.fill_id
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
        review_dt = pd.to_datetime(review_date, errors="coerce")
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
        holding_days = int(_num(row.get("closed_holding_days")) or 0) if row.get("closed_holding_days") is not None else None
        if holding_days is None and pd.notna(fill_dt) and pd.notna(review_dt):
            holding_days = max((review_dt.date() - fill_dt.date()).days, 0)
        strategy_return = _num(row.get("realized_return_until_exit"))
        if strategy_return is None:
            strategy_return = best_signed_return
        review_status = "pending_price_data" if best_horizon is None or best_signed_return is None else "auto_review_complete"
        if str(row.get("side") or "").upper() == "SELL":
            exit_reason = row.get("sell_reason") or row.get("portfolio_action_reason")
        else:
            exit_reason = None
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
            f"rank={row.get('ranking_rank') or '-'}, confidence={row.get('confidence_score') or '-'}, "
            f"engine={row.get('engine_type') or '-'}, entry_gate={row.get('entry_gate_status') or '-'}"
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
                "engine_type": row.get("engine_type"),
                "strategy_id": row.get("strategy_id"),
                "run_mode": row.get("run_mode"),
                "source_score_date": row.get("source_score_date"),
                "final_score": _num(row.get("final_score")),
                "prob_score": _num(row.get("prob_score")),
                "ret_score": _num(row.get("ret_score")),
                "tech_score": _num(row.get("tech_score")),
                "quality_score": _num(row.get("quality_score")),
                "confidence_score": _num(row.get("confidence_score")),
                "calibrated_confidence": _num(row.get("calibrated_confidence")),
                "live_confidence_grade": row.get("live_confidence_grade"),
                "liquidity_score": _num(row.get("liquidity_score")),
                "market_regime": row.get("market_regime"),
                "previous_close": _num(row.get("previous_close")),
                "live_price": _num(row.get("live_price")),
                "entry_price_gap_pct": _num(row.get("entry_price_gap_pct")),
                "entry_gate_status": row.get("entry_gate_status"),
                "entry_gate_reason": row.get("entry_gate_reason"),
                "buy_reason": row.get("buy_reason"),
                "sell_reason": row.get("sell_reason"),
                "portfolio_action_reason": row.get("portfolio_action_reason"),
                "benchmark_name": None,
                "benchmark_return_until_exit": None,
                "strategy_return": strategy_return,
                "excess_return": None,
                "holding_days": holding_days,
                "exit_reason": exit_reason,
                "review_status": review_status,
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
                    outcome_label, review_note, next_action_note,
                    engine_type, strategy_id, run_mode, source_score_date,
                    final_score, prob_score, ret_score, tech_score, quality_score,
                    confidence_score, calibrated_confidence, live_confidence_grade,
                    liquidity_score, market_regime, previous_close, live_price,
                    entry_price_gap_pct, entry_gate_status, entry_gate_reason,
                    buy_reason, sell_reason, portfolio_action_reason,
                    benchmark_name, benchmark_return_until_exit, strategy_return,
                    excess_return, holding_days, exit_reason, review_status,
                    reviewer, updated_at
                )
                VALUES (
                    :intent_id, :request_id, :code, CAST(:review_date AS date), :pre_tags, :post_tags,
                    :outcome_label, :review_note, :next_action_note,
                    :engine_type, :strategy_id, :run_mode, CAST(:source_score_date AS date),
                    :final_score, :prob_score, :ret_score, :tech_score, :quality_score,
                    :confidence_score, :calibrated_confidence, :live_confidence_grade,
                    :liquidity_score, :market_regime, :previous_close, :live_price,
                    :entry_price_gap_pct, :entry_gate_status, :entry_gate_reason,
                    :buy_reason, :sell_reason, :portfolio_action_reason,
                    :benchmark_name, :benchmark_return_until_exit, :strategy_return,
                    :excess_return, :holding_days, :exit_reason, :review_status,
                    :reviewer, now()
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
                    "engine_type": item.get("engine_type"),
                    "strategy_id": item.get("strategy_id"),
                    "run_mode": item.get("run_mode"),
                    "source_score_date": item.get("source_score_date"),
                    "final_score": item.get("final_score"),
                    "prob_score": item.get("prob_score"),
                    "ret_score": item.get("ret_score"),
                    "tech_score": item.get("tech_score"),
                    "quality_score": item.get("quality_score"),
                    "confidence_score": item.get("confidence_score"),
                    "calibrated_confidence": item.get("calibrated_confidence"),
                    "live_confidence_grade": item.get("live_confidence_grade"),
                    "liquidity_score": item.get("liquidity_score"),
                    "market_regime": item.get("market_regime"),
                    "previous_close": item.get("previous_close"),
                    "live_price": item.get("live_price"),
                    "entry_price_gap_pct": item.get("entry_price_gap_pct"),
                    "entry_gate_status": item.get("entry_gate_status"),
                    "entry_gate_reason": item.get("entry_gate_reason"),
                    "buy_reason": item.get("buy_reason"),
                    "sell_reason": item.get("sell_reason"),
                    "portfolio_action_reason": item.get("portfolio_action_reason"),
                    "benchmark_name": item.get("benchmark_name"),
                    "benchmark_return_until_exit": item.get("benchmark_return_until_exit"),
                    "strategy_return": item.get("strategy_return"),
                    "excess_return": item.get("excess_return"),
                    "holding_days": item.get("holding_days"),
                    "exit_reason": item.get("exit_reason"),
                    "review_status": item.get("review_status"),
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
            "| request_id | engine | code | side | intent | entry_gate | review_status | holding_days | strategy_return | outcome | note |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in report["items"]:
        lines.append(
            "| {request_id} | {engine_type} | {code} | {side} | {intent_type} | {entry_gate_status} | {review_status} | {holding_days} | {strategy_return} | {outcome_label} | {review_note} |".format(
                request_id=item.get("request_id") or "",
                engine_type=item.get("engine_type") or "",
                code=item.get("code") or "",
                side=item.get("side") or "",
                intent_type=item.get("intent_type") or "",
                entry_gate_status=item.get("entry_gate_status") or "-",
                review_status=item.get("review_status") or "-",
                holding_days=item.get("holding_days") if item.get("holding_days") is not None else "-",
                strategy_return=(
                    f"{float(item.get('strategy_return')) * 100:.2f}%"
                    if item.get("strategy_return") is not None
                    else "-"
                ),
                outcome_label=item.get("outcome_label") or "",
                review_note=str(item.get("review_note") or "").replace("|", "/"),
            )
        )
    if not report["items"]:
        lines.append("| - | - | - | - | - | - | - | 0 | - | - | - |")
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
