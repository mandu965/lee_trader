from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

PORTFOLIO_PATH = OUTPUT_DIR / "rule_portfolio_plan.json"
PREVIEW_PATH = OUTPUT_DIR / "rule_order_preview.json"
PAPER_STATE_PATH = OUTPUT_DIR / "rule_account_paper_state.json"
LIVE_STATE_PATH = OUTPUT_DIR / "rule_account_live_state.json"
BACKTEST_PATH = OUTPUT_DIR / "rule_strategy_backtest_report.json"
EXECUTION_PATH = OUTPUT_DIR / "rule_execution_results.json"
SIGNALS_CSV_PATH = DATA_DIR / "rule_signals.csv"
SUMMARY_OUT = OUTPUT_DIR / "rule_dashboard_summary.json"
SIGNALS_OUT = OUTPUT_DIR / "rule_signals_latest.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    value = json.loads(normalized)
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_num(value: Any) -> float | int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def to_iso_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) >= 10:
        return text[:10]
    return text


def parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def has_account_state(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("generated_at") or payload.get("as_of_date") or payload.get("positions"))


def choose_rule_account_state(
    *,
    run_mode: str,
    paper_state: dict[str, Any],
    live_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    paper_ready = has_account_state(paper_state)
    live_ready = has_account_state(live_state)
    if run_mode in {"pilot", "live"} and live_ready:
        return "live", live_state
    if live_ready and not paper_ready:
        return "live", live_state
    if paper_ready and not live_ready:
        return "paper", paper_state
    if live_ready and paper_ready:
        live_dt = parse_iso_datetime(live_state.get("generated_at"))
        paper_dt = parse_iso_datetime(paper_state.get("generated_at"))
        if live_dt and (paper_dt is None or live_dt >= paper_dt):
            return "live", live_state
    return "paper", paper_state


def normalize_rule_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": to_iso_date(row.get("date")),
        "code": str(row.get("code") or "").strip().zfill(6),
        "name": row.get("name") or None,
        "sector": row.get("sector") or None,
        "market": row.get("market") or None,
        "close": to_num(row.get("close")),
        "open": to_num(row.get("open")),
        "expected_gap": to_num(row.get("expected_gap")),
        "expected_entry_price": to_num(row.get("expected_entry_price")),
        "actual_open_gap": to_num(row.get("actual_open_gap")),
        "trading_value": to_num(row.get("trading_value")),
        "trading_value_ma_20": to_num(row.get("trading_value_ma_20")),
        "trading_value_pass": to_bool(row.get("trading_value_pass")),
        "trading_value_block_reason": row.get("trading_value_block_reason") or "none",
        "gap_risk_blocked": to_bool(row.get("gap_risk_blocked")),
        "gap_risk_reason": row.get("gap_risk_reason") or "none",
        "market_defensive_mode": to_bool(row.get("market_defensive_mode")),
        "market_entry_allowed": to_bool(row.get("market_entry_allowed")),
        "rule_score": to_num(row.get("rule_score")),
        "rule_score_v2": to_num(row.get("rule_score_v2")),
        "liquidity_score": to_num(row.get("liquidity_score")),
        "vol_20": to_num(row.get("vol_20")),
        "entry_signal": to_bool(row.get("entry_signal")),
        "strong_entry_signal": to_bool(row.get("strong_entry_signal")),
        "signal_strength": row.get("signal_strength") or "none",
        "strategy_id": row.get("strategy_id") or None,
        "engine_type": row.get("engine_type") or None,
        "run_mode": row.get("run_mode") or None,
    }


def load_latest_rule_signals() -> tuple[str | None, list[dict[str, Any]]]:
    portfolio = read_json(PORTFOLIO_PATH)
    preview = read_json(PREVIEW_PATH)
    portfolio_items = portfolio.get("items") or []
    preview_items = preview.get("items") or []
    if portfolio.get("as_of_date") and portfolio_items:
        preview_by_code = {
            str(item.get("code") or item.get("symbol") or "").strip().zfill(6): item
            for item in preview_items
        }
        items: list[dict[str, Any]] = []
        for item in portfolio_items:
            code = str(item.get("code") or "").strip().zfill(6)
            preview_item = preview_by_code.get(code, {})
            items.append(
                {
                    "date": portfolio.get("as_of_date"),
                    "code": code,
                    "name": item.get("name") or preview_item.get("name"),
                    "sector": item.get("sector"),
                    "market": None,
                    "close": None,
                    "open": None,
                    "expected_gap": None,
                    "expected_entry_price": to_num(item.get("expected_entry_price")) or to_num(preview_item.get("expected_execution_price")),
                    "actual_open_gap": None,
                    "trading_value": None,
                    "trading_value_ma_20": None,
                    "trading_value_pass": str(item.get("trading_value_block_reason") or "none") == "none",
                    "trading_value_block_reason": item.get("trading_value_block_reason") or "none",
                    "gap_risk_blocked": str(item.get("gap_risk_reason") or "none") != "none",
                    "gap_risk_reason": item.get("gap_risk_reason") or "none",
                    "market_defensive_mode": bool(item.get("market_defensive_mode")),
                    "market_entry_allowed": not bool(item.get("market_defensive_mode")),
                    "rule_score": to_num(item.get("rule_score")),
                    "rule_score_v2": to_num(item.get("rule_score_v2")),
                    "liquidity_score": to_num(item.get("liquidity_score")),
                    "vol_20": to_num(item.get("vol_20")),
                    "entry_signal": bool(item.get("entry_signal")),
                    "strong_entry_signal": bool(item.get("strong_entry_signal")),
                    "signal_strength": item.get("signal_strength") or preview_item.get("signal_strength") or "none",
                    "strategy_id": portfolio.get("strategy_id") or preview.get("strategy_id"),
                    "engine_type": portfolio.get("engine_type") or preview.get("engine_type"),
                    "run_mode": portfolio.get("run_mode") or preview.get("run_mode"),
                }
            )
        latest_date = str(portfolio.get("as_of_date"))
    else:
        rows = read_csv_rows(SIGNALS_CSV_PATH)
        if not rows:
            return None, []
        latest_date = max((to_iso_date(row.get("date")) or "" for row in rows), default=None)
        items = [normalize_rule_signal_row(row) for row in rows if (to_iso_date(row.get("date")) or "") == latest_date]

    strength_rank = {"strong_entry": 2, "entry": 1, "none": 0}
    items.sort(
        key=lambda item: (
            -(strength_rank.get(str(item.get("signal_strength") or "none"), 0)),
            -(float(item.get("rule_score_v2") or 0)),
            -(float(item.get("rule_score") or 0)),
        )
    )
    return latest_date, items


def summarize_counts_by(items: list[dict[str, Any]], key: str, fallback: str = "none") -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        raw = item.get(key)
        value = str(raw if raw not in (None, "") else fallback).strip() or fallback
        counts[value] = counts.get(value, 0) + 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def build_rule_dashboard_summary() -> dict[str, Any]:
    latest_date, signal_items = load_latest_rule_signals()
    portfolio = read_json(PORTFOLIO_PATH)
    preview = read_json(PREVIEW_PATH)
    paper_state = read_json(PAPER_STATE_PATH)
    live_state = read_json(LIVE_STATE_PATH)
    backtest = read_json(BACKTEST_PATH)
    execution = read_json(EXECUTION_PATH)
    run_mode = str(execution.get("run_mode") or preview.get("run_mode") or "").lower()
    account_mode, account_state = choose_rule_account_state(
        run_mode=run_mode,
        paper_state=paper_state,
        live_state=live_state,
    )

    portfolio_items = portfolio.get("items") or []
    preview_items = preview.get("items") or []
    positions = account_state.get("positions") or []

    return {
        "generated_at": preview.get("generated_at") or portfolio.get("generated_at") or backtest.get("generated_at"),
        "as_of_date": preview.get("as_of_date") or portfolio.get("as_of_date") or latest_date,
        "strategy_id": preview.get("strategy_id") or portfolio.get("strategy_id") or (signal_items[0].get("strategy_id") if signal_items else None),
        "engine_type": preview.get("engine_type") or portfolio.get("engine_type") or (signal_items[0].get("engine_type") if signal_items else None),
        "run_mode": preview.get("run_mode") or portfolio.get("run_mode") or (signal_items[0].get("run_mode") if signal_items else None),
        "counts": {
            "total_candidates": len(signal_items),
            "entry_signal_count": sum(1 for item in signal_items if item.get("entry_signal")),
            "strong_entry_count": sum(1 for item in signal_items if item.get("strong_entry_signal")),
            "gap_risk_blocked_count": sum(1 for item in signal_items if item.get("gap_risk_blocked")),
            "trading_value_failed_count": sum(1 for item in signal_items if not item.get("trading_value_pass")),
            "defensive_mode_count": sum(1 for item in signal_items if item.get("market_defensive_mode")),
            "hold_count": int(to_num((portfolio.get("summary") or {}).get("hold_count")) or 0),
            "buy_count": int(to_num((portfolio.get("summary") or {}).get("buy_count")) or 0),
            "reduce_count": int(to_num((portfolio.get("summary") or {}).get("reduce_count")) or 0),
            "exit_count": int(to_num((portfolio.get("summary") or {}).get("exit_count")) or 0),
            "skip_count": int(to_num((portfolio.get("summary") or {}).get("skip_count")) or 0),
            "preview_request_count": int(to_num((preview.get("summary") or {}).get("request_count")) or 0),
            "preview_buy_count": int(to_num((preview.get("summary") or {}).get("buy_preview_count")) or 0),
            "preview_sell_count": int(to_num((preview.get("summary") or {}).get("sell_preview_count")) or 0),
            "preview_allowed_count": int(to_num((preview.get("summary") or {}).get("order_allowed_count")) or 0),
            "execution_submitted_count": int(to_num((execution.get("summary") or {}).get("submitted_count")) or 0),
            "execution_failed_count": int(to_num((execution.get("summary") or {}).get("failed_count")) or 0),
            "execution_skipped_count": int(to_num((execution.get("summary") or {}).get("skipped_count")) or 0),
            "execution_filled_count": int(to_num((execution.get("summary") or {}).get("filled_count")) or 0),
            "execution_partial_filled_count": int(to_num((execution.get("summary") or {}).get("partial_filled_count")) or 0),
            "execution_unfilled_count": int(to_num((execution.get("summary") or {}).get("unfilled_count")) or 0),
            "execution_canceled_count": int(to_num((execution.get("summary") or {}).get("canceled_count")) or 0),
            "execution_simulated_filled_count": int(to_num((execution.get("summary") or {}).get("simulated_filled_count")) or 0),
            "execution_simulated_unfilled_count": int(to_num((execution.get("summary") or {}).get("simulated_unfilled_count")) or 0),
            "paper_position_count": len(positions),
            "account_position_count": len(positions),
        },
        "distributions": {
            "signal_strength": summarize_counts_by(signal_items, "signal_strength", "none"),
            "portfolio_action_reason": summarize_counts_by(portfolio_items, "portfolio_action_reason", "none")[:10],
            "order_block_reason": summarize_counts_by(preview_items, "order_block_reason", "none")[:10],
            "trading_value_block_reason": summarize_counts_by([item for item in signal_items if not item.get("trading_value_pass")], "trading_value_block_reason", "none"),
            "gap_risk_reason": summarize_counts_by([item for item in signal_items if item.get("gap_risk_blocked")], "gap_risk_reason", "none"),
        },
        "top_candidates": {
            "strong": [item for item in signal_items if item.get("strong_entry_signal")][:10],
            "entry_only": [item for item in signal_items if item.get("entry_signal") and not item.get("strong_entry_signal")][:10],
        },
        "account_mode": account_mode,
        "account_as_of_date": account_state.get("as_of_date"),
        "account_generated_at": account_state.get("generated_at"),
        "paper_state": {
            "total_equity": to_num(account_state.get("total_equity")),
            "cash": to_num(account_state.get("cash")),
            "recent_trade_count": len(account_state.get("recent_trades") or []),
            "cooldown_count": len(account_state.get("cooldown_codes") or []),
            "positions": positions,
        },
        "account_state": {
            "total_equity": to_num(account_state.get("total_equity")),
            "cash": to_num(account_state.get("cash")),
            "recent_trade_count": len(account_state.get("recent_trades") or []),
            "cooldown_count": len(account_state.get("cooldown_codes") or []),
            "positions": positions,
        },
        "execution": execution or {},
        "backtest": {
            "summary": (backtest.get("summary") or {}),
            "portfolio_equity_curve": (backtest.get("portfolio_equity_curve") or {}),
            "score_distribution": (backtest.get("score_distribution") or {}),
            "trading_value_distribution": (backtest.get("trading_value_distribution") or {}),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    latest_date, items = load_latest_rule_signals()
    summary = build_rule_dashboard_summary()
    signals_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": latest_date,
        "strength": "all",
        "count": len(items),
        "items": items,
    }
    write_json(SUMMARY_OUT, summary)
    write_json(SIGNALS_OUT, signals_payload)
    print(f"saved {SUMMARY_OUT}")
    print(f"saved {SIGNALS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
