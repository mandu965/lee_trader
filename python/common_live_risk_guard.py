from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_HOLDINGS_CSV = DATA_DIR / "live_account_holdings.csv"
DEFAULT_BALANCE_JSON = OUTPUT_DIR / "live_account_balance_summary.json"
DEFAULT_FILLS_JSON = OUTPUT_DIR / "live_order_fills.json"
DEFAULT_MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "common_live_risk_guard.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "common_live_risk_guard_report.md"


def _flag(name: str, default: str) -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    raw = resolved.read_text(encoding="utf-8-sig")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    value = json.loads(normalized)
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def _num(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text else ""


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_date(value: Any) -> datetime | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def _date_text(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None else None


def _minutes_since(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return max((now - ts).total_seconds() / 60.0, 0.0)


def _load_balance_payload(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _load_fills_payload(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _load_market_status(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True) if "date" in out.columns else out


def _holdings_generated_at(balance_payload: dict[str, Any], holdings_csv: Path) -> datetime | None:
    ts = _parse_timestamp(balance_payload.get("generated_at"))
    if ts is not None:
        return ts
    resolved = _resolve(holdings_csv)
    if resolved.exists():
        return datetime.fromtimestamp(resolved.stat().st_mtime)
    return None


def _fills_generated_at(fills_payload: dict[str, Any], fills_json: Path) -> datetime | None:
    ts = _parse_timestamp(fills_payload.get("generated_at"))
    if ts is not None:
        return ts
    resolved = _resolve(fills_json)
    if resolved.exists():
        return datetime.fromtimestamp(resolved.stat().st_mtime)
    return None


def _market_status_latest_row(market_status: pd.DataFrame) -> dict[str, Any]:
    if market_status.empty:
        return {}
    return market_status.where(pd.notna(market_status), None).iloc[-1].to_dict()


def _derive_daily_loss_pct(balance_payload: dict[str, Any]) -> float | None:
    summary_row = balance_payload.get("summary_row") or {}
    direct = _num(summary_row.get("asst_icdc_erng_rt"))
    if direct is not None:
        if direct > 1.0:
            return direct / 100.0
        return direct
    delta_amount = _num(summary_row.get("asst_icdc_amt"))
    previous_total = _num(summary_row.get("bfdy_tot_asst_evlu_amt"))
    if delta_amount is None or previous_total in {None, 0.0}:
        return None
    return delta_amount / previous_total


def _derive_weekly_loss_pct(balance_payload: dict[str, Any]) -> float | None:
    derived_metrics = balance_payload.get("derived_metrics") or {}
    direct = _num(derived_metrics.get("weekly_loss_pct"))
    if direct is not None:
        return direct
    current_total = _num(derived_metrics.get("total_assets"))
    week_start_total = _num(derived_metrics.get("week_start_total_assets"))
    if current_total is None or week_start_total in {None, 0.0}:
        return None
    return (current_total / week_start_total) - 1.0


def _derived_metric(balance_payload: dict[str, Any], key: str) -> float | None:
    return _num((balance_payload.get("derived_metrics") or {}).get(key))


def _coalesce_order_amount(order_context: dict[str, Any]) -> float | None:
    explicit = _num(order_context.get("order_amount"))
    if explicit is not None:
        return explicit
    qty = _num(order_context.get("order_qty"))
    price = _num(order_context.get("reference_price"))
    if qty is None or price is None:
        return None
    return qty * price


def _sum_buy_amount(
    fill_items: list[dict[str, Any]],
    *,
    start_date: datetime,
    end_date: datetime,
) -> float:
    total = 0.0
    for item in fill_items:
        side = str(item.get("side") or "").upper()
        if side != "BUY":
            continue
        item_dt = _parse_date(item.get("as_of_date") or item.get("filled_at"))
        if item_dt is None or item_dt < start_date or item_dt > end_date:
            continue
        total += _num(item.get("filled_amount")) or 0.0
    return total


def _same_symbol_buy_filled_today(fill_items: list[dict[str, Any]], code: str, as_of_date: datetime) -> bool:
    if not code:
        return False
    target_date = _date_text(as_of_date)
    for item in fill_items:
        if str(item.get("side") or "").upper() != "BUY":
            continue
        if _normalize_code(item.get("code")) != code:
            continue
        item_date = _date_text(_parse_date(item.get("as_of_date") or item.get("filled_at")))
        if item_date == target_date:
            return True
    return False


def evaluate_common_buy_guard(order_context: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    now = _parse_timestamp(order_context.get("now")) or datetime.now()
    signal_date = _parse_date(order_context.get("as_of_date"))
    effective_trade_date = (
        _parse_date(order_context.get("execution_date"))
        or _parse_date(order_context.get("trade_date"))
        or signal_date
        or now.replace(hour=0, minute=0, second=0, microsecond=0)
    )
    market_reference_date = signal_date or effective_trade_date

    side = str(order_context.get("side") or "BUY").upper()
    code = _normalize_code(order_context.get("code"))
    order_amount = _coalesce_order_amount(order_context)

    holdings_csv = Path(order_context.get("holdings_csv") or DEFAULT_HOLDINGS_CSV)
    balance_json = Path(order_context.get("balance_json") or DEFAULT_BALANCE_JSON)
    fills_json = Path(order_context.get("fills_json") or DEFAULT_FILLS_JSON)
    market_status_csv = Path(order_context.get("market_status_csv") or DEFAULT_MARKET_STATUS_CSV)

    balance_payload = _load_balance_payload(balance_json)
    fills_payload = _load_fills_payload(fills_json)
    market_status = _load_market_status(market_status_csv)

    holdings_sync_at = _parse_timestamp(order_context.get("holdings_generated_at")) or _holdings_generated_at(balance_payload, holdings_csv)
    fills_sync_at = _parse_timestamp(order_context.get("fills_generated_at")) or _fills_generated_at(fills_payload, fills_json)

    holdings_sync_age_minutes = _minutes_since(holdings_sync_at, now)
    fills_sync_age_minutes = _minutes_since(fills_sync_at, now)

    market_latest = _market_status_latest_row(market_status)
    market_latest_date = _parse_date(market_latest.get("date"))
    market_status_missing = not bool(market_latest)
    # 전일 종가 기준 데이터는 장전(09:30 이전)에 갱신되지 않으므로
    # 정확한 날짜 일치 대신 나이(일수) 기반 허용 범위를 사용한다.
    _market_status_max_age_days = int(_float_env("GLOBAL_MARKET_STATUS_MAX_AGE_DAYS", 5.0))
    if market_latest_date is not None:
        _market_age_days = (effective_trade_date - market_latest_date).days
        market_status_is_latest = _market_age_days <= _market_status_max_age_days
    else:
        market_status_is_latest = False
    market_defensive_mode = bool(order_context.get("market_defensive_mode")) if "market_defensive_mode" in order_context else (not bool(market_latest.get("market_up")) if market_latest else False)

    fill_items = fills_payload.get("items") or []
    if not isinstance(fill_items, list):
        fill_items = []

    week_start = effective_trade_date - timedelta(days=effective_trade_date.weekday())
    daily_buy_amount_used = _num(order_context.get("daily_buy_amount_used"))
    if daily_buy_amount_used is None:
        daily_buy_amount_used = _sum_buy_amount(fill_items, start_date=effective_trade_date, end_date=effective_trade_date)
    weekly_buy_amount_used = _num(order_context.get("weekly_buy_amount_used"))
    if weekly_buy_amount_used is None:
        weekly_buy_amount_used = _sum_buy_amount(fill_items, start_date=week_start, end_date=effective_trade_date)

    daily_loss_pct = _num(order_context.get("daily_loss_pct"))
    if daily_loss_pct is None:
        daily_loss_pct = _derive_daily_loss_pct(balance_payload)
    weekly_loss_pct = _num(order_context.get("weekly_loss_pct"))
    if weekly_loss_pct is None:
        weekly_loss_pct = _derive_weekly_loss_pct(balance_payload)
    weekly_realized_pnl = _derived_metric(balance_payload, "weekly_realized_pnl")
    weekly_unrealized_pnl = _derived_metric(balance_payload, "weekly_unrealized_pnl")
    weekly_total_pnl = _num(order_context.get("weekly_total_pnl"))
    if weekly_total_pnl is None:
        weekly_total_pnl = _derived_metric(balance_payload, "weekly_total_pnl")
    if weekly_total_pnl is None:
        weekly_total_pnl = _derived_metric(balance_payload, "weekly_asset_change_amount")
    week_start_total_assets = _derived_metric(balance_payload, "week_start_total_assets")
    weekly_fill_count = _derived_metric(balance_payload, "weekly_fill_count")
    weekly_buy_fill_count = _derived_metric(balance_payload, "weekly_buy_fill_count")
    weekly_sell_fill_count = _derived_metric(balance_payload, "weekly_sell_fill_count")
    weekly_external_cash_flow_amount = _derived_metric(balance_payload, "weekly_external_cash_flow_amount")
    weekly_loss_guard_basis = (balance_payload.get("derived_metrics") or {}).get("weekly_loss_guard_basis")
    weekly_loss_source = (balance_payload.get("derived_metrics") or {}).get("weekly_loss_source")
    weekly_reset_mode = (balance_payload.get("derived_metrics") or {}).get("weekly_reset_mode")
    week_start_date = (balance_payload.get("derived_metrics") or {}).get("week_start_date")
    week_start_snapshot_at = (balance_payload.get("derived_metrics") or {}).get("week_start_snapshot_at")
    timezone_name = (balance_payload.get("derived_metrics") or {}).get("timezone") or "Asia/Seoul"

    reasons: list[str] = []
    if side != "BUY":
        snapshot = {
            "as_of": now.isoformat(timespec="seconds"),
            "side": side,
            "code": code or None,
            "buy_allowed": True,
            "block_reasons": [],
            "bypass_reason": "non_buy_side",
        }
        return True, [], snapshot

    if not code:
        reasons.append("code_missing")
    if order_amount is None or order_amount <= 0:
        reasons.append("order_amount_missing")

    if _flag("GLOBAL_KILL_SWITCH", "0"):
        reasons.append("global_kill_switch_on")

    if _flag("GLOBAL_BLOCK_BUY_ON_SYNC_STALE", "1"):
        max_age_minutes = _float_env("GLOBAL_SYNC_MAX_AGE_MINUTES", 30.0)
        if holdings_sync_age_minutes is None:
            reasons.append("holdings_sync_missing")
        elif holdings_sync_age_minutes > max_age_minutes:
            reasons.append("holdings_sync_stale")
        if fills_sync_age_minutes is None:
            reasons.append("fills_sync_missing")
        elif fills_sync_age_minutes > max_age_minutes:
            reasons.append("fills_sync_stale")

    if _flag("GLOBAL_BLOCK_BUY_ON_MARKET_STATUS_MISSING", "1"):
        if market_status_missing or not market_status_is_latest:
            reasons.append("market_status_missing")

    if _flag("GLOBAL_BLOCK_BUY_ON_MARKET_DEFENSIVE", "1"):
        if not market_status_missing and market_status_is_latest and market_defensive_mode:
            reasons.append("market_defensive_mode")

    if _flag("GLOBAL_BLOCK_SAME_SYMBOL_BUY_SAME_DAY", "1") and code:
        if _same_symbol_buy_filled_today(fill_items, code, effective_trade_date):
            reasons.append("same_symbol_buy_already_filled_today")

    daily_buy_limit = _float_env("GLOBAL_MAX_DAILY_BUY_AMOUNT", 500_000.0)
    weekly_buy_limit = _float_env("GLOBAL_MAX_WEEKLY_BUY_AMOUNT", 1_500_000.0)
    if order_amount is not None and daily_buy_amount_used + order_amount > daily_buy_limit:
        reasons.append("daily_buy_amount_limit_exceeded")
    if order_amount is not None and weekly_buy_amount_used + order_amount > weekly_buy_limit:
        reasons.append("weekly_buy_amount_limit_exceeded")

    daily_loss_limit = _float_env("GLOBAL_MAX_DAILY_LOSS_PCT", 0.01)
    weekly_loss_limit = _float_env("GLOBAL_MAX_WEEKLY_LOSS_PCT", 0.03)
    if daily_loss_pct is None:
        reasons.append("daily_loss_pct_unavailable")
    elif daily_loss_pct <= -abs(daily_loss_limit):
        reasons.append("daily_loss_limit_reached")
    if weekly_loss_pct is None:
        if _flag("GLOBAL_BLOCK_BUY_ON_WEEKLY_LOSS_UNAVAILABLE", "1"):
            reasons.append("weekly_loss_pct_unavailable")
    elif weekly_loss_pct <= -abs(weekly_loss_limit):
        reasons.append("weekly_loss_limit_reached")

    unique_reasons = list(dict.fromkeys(reasons))
    snapshot = {
        "as_of": now.isoformat(timespec="seconds"),
        "as_of_date": _date_text(signal_date or effective_trade_date),
        "effective_trade_date": _date_text(effective_trade_date),
        "market_reference_date": _date_text(market_reference_date),
        "side": side,
        "code": code or None,
        "order_amount": order_amount,
        "buy_allowed": not unique_reasons,
        "block_reasons": unique_reasons,
        "global_kill_switch": _flag("GLOBAL_KILL_SWITCH", "0"),
        "daily_buy_amount_used": daily_buy_amount_used,
        "daily_buy_amount_limit": daily_buy_limit,
        "weekly_buy_amount_used": weekly_buy_amount_used,
        "weekly_buy_amount_limit": weekly_buy_limit,
        "daily_loss_pct": daily_loss_pct,
        "daily_loss_limit_pct": -abs(daily_loss_limit),
        "weekly_loss_pct": weekly_loss_pct,
        "weekly_loss_limit_pct": -abs(weekly_loss_limit),
        "weekly_realized_pnl": weekly_realized_pnl,
        "weekly_unrealized_pnl": weekly_unrealized_pnl,
        "weekly_total_pnl": weekly_total_pnl,
        "weekly_loss_rate": weekly_loss_pct,
        "weekly_limit": -abs(weekly_loss_limit),
        "weekly_blocked": weekly_loss_pct is not None and weekly_loss_pct <= -abs(weekly_loss_limit),
        "week_start_total_assets": week_start_total_assets,
        "week_start_date": week_start_date,
        "week_start_snapshot_at": week_start_snapshot_at,
        "weekly_fill_count": weekly_fill_count,
        "weekly_buy_fill_count": weekly_buy_fill_count,
        "weekly_sell_fill_count": weekly_sell_fill_count,
        "weekly_external_cash_flow_amount": weekly_external_cash_flow_amount,
        "weekly_loss_guard_basis": weekly_loss_guard_basis,
        "weekly_loss_source": weekly_loss_source,
        "weekly_reset_mode": weekly_reset_mode,
        "timezone": timezone_name,
        "paper_live_scope": str(order_context.get("run_mode") or ""),
        "holdings_sync_at": holdings_sync_at.isoformat(timespec="seconds") if holdings_sync_at else None,
        "holdings_sync_age_minutes": holdings_sync_age_minutes,
        "fills_sync_at": fills_sync_at.isoformat(timespec="seconds") if fills_sync_at else None,
        "fills_sync_age_minutes": fills_sync_age_minutes,
        "sync_max_age_minutes": _float_env("GLOBAL_SYNC_MAX_AGE_MINUTES", 30.0),
        "market_status_missing": market_status_missing,
        "market_status_latest_date": _date_text(market_latest_date),
        "market_status_is_latest": market_status_is_latest,
        "market_defensive_mode": market_defensive_mode,
        "same_symbol_buy_already_filled_today": _same_symbol_buy_filled_today(fill_items, code, effective_trade_date) if code else False,
        "data_sources": {
            "holdings_csv": str(_resolve(holdings_csv)),
            "balance_json": str(_resolve(balance_json)),
            "fills_json": str(_resolve(fills_json)),
            "market_status_csv": str(_resolve(market_status_csv)),
        },
    }
    return not unique_reasons, unique_reasons, snapshot


def _write_outputs(payload: dict[str, Any], out_json: Path, out_md: Path) -> None:
    resolved_json = _resolve(out_json)
    resolved_md = _resolve(out_md)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_md.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Common Live Risk Guard Report",
        "",
        f"- as_of: `{payload.get('as_of')}`",
        f"- as_of_date: `{payload.get('as_of_date')}`",
        f"- buy_allowed: `{payload.get('buy_allowed')}`",
        f"- block_reasons: `{', '.join(payload.get('block_reasons') or []) or 'none'}`",
        f"- code: `{payload.get('code') or '-'}`",
        f"- order_amount: `{payload.get('order_amount')}`",
        f"- daily_buy_amount_used: `{payload.get('daily_buy_amount_used')}` / `{payload.get('daily_buy_amount_limit')}`",
        f"- weekly_buy_amount_used: `{payload.get('weekly_buy_amount_used')}` / `{payload.get('weekly_buy_amount_limit')}`",
        f"- daily_loss_pct: `{payload.get('daily_loss_pct')}` / limit `{payload.get('daily_loss_limit_pct')}`",
        f"- weekly_loss_pct: `{payload.get('weekly_loss_pct')}` / limit `{payload.get('weekly_loss_limit_pct')}`",
        f"- holdings_sync_age_minutes: `{payload.get('holdings_sync_age_minutes')}`",
        f"- fills_sync_age_minutes: `{payload.get('fills_sync_age_minutes')}`",
        f"- market_status_latest_date: `{payload.get('market_status_latest_date')}`",
        f"- market_status_is_latest: `{payload.get('market_status_is_latest')}`",
        f"- market_defensive_mode: `{payload.get('market_defensive_mode')}`",
        f"- same_symbol_buy_already_filled_today: `{payload.get('same_symbol_buy_already_filled_today')}`",
        "",
        "## Data Sources",
        "",
    ]
    for key, value in (payload.get("data_sources") or {}).items():
        lines.append(f"- {key}: `{value}`")
    resolved_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test_payloads() -> list[tuple[str, dict[str, Any], bool, list[str]]]:
    base = {
        "side": "BUY",
        "code": "005930",
        "order_amount": 100000.0,
        "as_of_date": "2026-04-30",
        "now": "2026-04-30T10:00:00",
        "holdings_generated_at": "2026-04-30T09:55:00",
        "fills_generated_at": "2026-04-30T09:55:00",
        "market_defensive_mode": False,
        "daily_buy_amount_used": 0.0,
        "weekly_buy_amount_used": 0.0,
        "daily_loss_pct": -0.001,
        "weekly_loss_pct": -0.002,
        "market_status_csv": DEFAULT_MARKET_STATUS_CSV,
    }
    return [
        (
            "healthy_allowed",
            {**base},
            True,
            [],
        ),
        (
            "kill_switch",
            {**base},
            False,
            ["global_kill_switch_on"],
        ),
        (
            "stale_sync",
            {
                **base,
                "holdings_generated_at": "2026-04-30T08:00:00",
                "fills_generated_at": "2026-04-30T08:00:00",
            },
            False,
            ["holdings_sync_stale", "fills_sync_stale"],
        ),
        (
            "market_status_missing",
            {
                **base,
                "market_status_csv": DATA_DIR / "_missing_market_status.csv",
            },
            False,
            ["market_status_missing"],
        ),
        (
            "loss_limit",
            {
                **base,
                "daily_loss_pct": -0.02,
                "weekly_loss_pct": -0.04,
            },
            False,
            ["daily_loss_limit_reached", "weekly_loss_limit_reached"],
        ),
        (
            "non_buy_bypass",
            {
                **base,
                "side": "SELL",
            },
            True,
            [],
        ),
    ]


def run_self_test(out_json: Path, out_md: Path) -> int:
    original_kill_switch = os.environ.get("GLOBAL_KILL_SWITCH")
    scenarios: list[dict[str, Any]] = []
    try:
        for name, context, expected_allowed, expected_reasons in _self_test_payloads():
            if name == "kill_switch":
                os.environ["GLOBAL_KILL_SWITCH"] = "1"
            else:
                os.environ["GLOBAL_KILL_SWITCH"] = "0"
            allowed, reasons, snapshot = evaluate_common_buy_guard(context)
            if allowed != expected_allowed:
                raise AssertionError(f"{name}: expected allowed={expected_allowed}, got allowed={allowed}, reasons={reasons}")
            missing_expected = [reason for reason in expected_reasons if reason not in reasons]
            if missing_expected:
                raise AssertionError(f"{name}: missing expected reasons {missing_expected}; got={reasons}")
            scenarios.append(
                {
                    "name": name,
                    "allowed": allowed,
                    "reasons": reasons,
                    "snapshot": snapshot,
                }
            )
    finally:
        if original_kill_switch is None:
            os.environ.pop("GLOBAL_KILL_SWITCH", None)
        else:
            os.environ["GLOBAL_KILL_SWITCH"] = original_kill_switch

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "self_test": True,
        "scenario_count": len(scenarios),
        "passed": True,
        "scenarios": scenarios,
    }
    _write_outputs(payload, out_json, out_md)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone common live BUY risk guard.")
    parser.add_argument("--context-json", type=Path, default=None, help="JSON file with order_context.")
    parser.add_argument("--side", default="BUY")
    parser.add_argument("--code", default="")
    parser.add_argument("--order-amount", type=float, default=None)
    parser.add_argument("--order-qty", type=float, default=None)
    parser.add_argument("--reference-price", type=float, default=None)
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--daily-loss-pct", type=float, default=None)
    parser.add_argument("--weekly-loss-pct", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _build_context_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.context_json:
        payload = _read_json(args.context_json)
        if payload:
            return payload
    context: dict[str, Any] = {
        "side": args.side,
        "code": args.code,
        "as_of_date": args.as_of_date or datetime.now().strftime("%Y-%m-%d"),
    }
    if args.order_amount is not None:
        context["order_amount"] = args.order_amount
    if args.order_qty is not None:
        context["order_qty"] = args.order_qty
    if args.reference_price is not None:
        context["reference_price"] = args.reference_price
    if args.daily_loss_pct is not None:
        context["daily_loss_pct"] = args.daily_loss_pct
    if args.weekly_loss_pct is not None:
        context["weekly_loss_pct"] = args.weekly_loss_pct
    return context


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test(args.out_json, args.out_md)

    context = _build_context_from_args(args)
    allowed, reasons, snapshot = evaluate_common_buy_guard(context)
    _write_outputs(snapshot, args.out_json, args.out_md)
    print(json.dumps({"allowed": allowed, "reasons": reasons}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
