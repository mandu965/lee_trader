from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from apply_analytics_views import apply_analytics_views
from db import get_engine
from payload_store import upsert_json_payload
from report_json import sanitize_for_json, write_json_strict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT_JSON = OUTPUT_DIR / "live_closed_trade_report.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "live_closed_trade_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build realized live closed-trade KPI report.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--skip-ensure-views", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    engine = get_engine()
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]


def _one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.0f}"


def _sample_status(observed_count: int) -> str:
    if observed_count < 30:
        return "INSUFFICIENT_SAMPLE"
    if observed_count < 100:
        return "MONITOR_ONLY"
    return "ACTIONABLE"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_bucket(value: Any) -> str:
    rank = _float(value)
    if rank is None:
        return "rank_unknown"
    if rank <= 3:
        return "rank_1_3"
    if rank <= 8:
        return "rank_4_8"
    if rank <= 20:
        return "rank_9_20"
    return "rank_21_plus"


def _load_fill_rows() -> list[dict[str, Any]]:
    return _rows(
        """
        SELECT
            fill_id,
            request_id,
            intent_id,
            as_of_date,
            filled_at,
            code,
            name,
            side,
            intent_type,
            filled_qty,
            filled_price,
            filled_amount,
            fee,
            tax,
            ranking_run_id,
            ranking_rank,
            final_score,
            confidence_score,
            risk_penalty,
            qual_score,
            dominant_theme,
            guard_applied_bucket,
            guard_penalty_bucket,
            production_rank_bucket,
            shadow_rank_bucket,
            shadow_rank_delta_bucket
        FROM analytics.live_trade_fact
        WHERE filled_at IS NOT NULL
          AND filled_qty IS NOT NULL
          AND filled_price IS NOT NULL
        ORDER BY code, filled_at, fill_id
        """
    )


def _load_position_snapshots() -> dict[str, list[dict[str, Any]]]:
    rows = _rows(
        """
        SELECT
            snapshot_at,
            code,
            name,
            qty,
            avg_price
        FROM research.live_position_snapshot
        WHERE snapshot_at IS NOT NULL
          AND code IS NOT NULL
          AND qty IS NOT NULL
          AND avg_price IS NOT NULL
        ORDER BY code, snapshot_at
        """
    )
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        snapshots[str(row.get("code") or "").zfill(6)].append(row)
    return snapshots


def _snapshot_basis_for_sell(
    snapshots: dict[str, list[dict[str, Any]]],
    *,
    code: str,
    sell_time: Any,
    requested_qty: float,
) -> dict[str, Any] | None:
    if requested_qty <= 0:
        return None
    candidates = snapshots.get(str(code).zfill(6), [])
    best: dict[str, Any] | None = None
    for row in candidates:
        snapshot_at = row.get("snapshot_at")
        if snapshot_at is None or sell_time is None:
            continue
        if snapshot_at < sell_time:
            best = row
        else:
            break
    if not best:
        return None
    snapshot_qty = _float(best.get("qty")) or 0.0
    avg_price = _float(best.get("avg_price")) or 0.0
    if snapshot_qty <= 0 or avg_price <= 0:
        return None
    use_qty = min(requested_qty, snapshot_qty)
    return {
        "snapshot_at": best.get("snapshot_at"),
        "snapshot_qty": snapshot_qty,
        "matched_qty": use_qty,
        "avg_price": avg_price,
    }


def _build_fifo_closed_trades(
    fills: list[dict[str, Any]],
    snapshots: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed: list[dict[str, Any]] = []

    for row in fills:
        code = str(row.get("code") or "").zfill(6)
        side = str(row.get("side") or "").upper()
        qty = _float(row.get("filled_qty")) or 0.0
        price = _float(row.get("filled_price")) or 0.0
        amount = _float(row.get("filled_amount"))
        fee = _float(row.get("fee")) or 0.0
        tax = _float(row.get("tax")) or 0.0
        if not code or qty <= 0 or price <= 0:
            continue

        if side == "BUY":
            lots[code].append(
                {
                    "fill_id": row.get("fill_id"),
                    "request_id": row.get("request_id"),
                    "filled_at": row.get("filled_at"),
                    "remaining_qty": qty,
                    "original_qty": qty,
                    "price": price,
                    "amount": amount if amount is not None else qty * price,
                    "remaining_fee": fee,
                }
            )
            continue

        if side != "SELL":
            continue

        remaining_sell_qty = qty
        matched_qty = 0.0
        matched_buy_cost = 0.0
        matched_buy_fee = 0.0
        match_lots: list[dict[str, Any]] = []

        while remaining_sell_qty > 0 and lots[code]:
            lot = lots[code][0]
            lot_qty = float(lot.get("remaining_qty") or 0.0)
            if lot_qty <= 0:
                lots[code].popleft()
                continue
            use_qty = min(remaining_sell_qty, lot_qty)
            lot_price = float(lot.get("price") or 0.0)
            lot_fee = float(lot.get("remaining_fee") or 0.0)
            fee_used = lot_fee * (use_qty / lot_qty) if lot_qty > 0 else 0.0

            matched_qty += use_qty
            matched_buy_cost += use_qty * lot_price
            matched_buy_fee += fee_used
            match_lots.append(
                {
                    "buy_fill_id": lot.get("fill_id"),
                    "buy_request_id": lot.get("request_id"),
                    "buy_filled_at": lot.get("filled_at"),
                    "matched_qty": use_qty,
                    "buy_price": lot_price,
                    "allocated_buy_fee": fee_used,
                }
            )

            lot["remaining_qty"] = lot_qty - use_qty
            lot["remaining_fee"] = lot_fee - fee_used
            remaining_sell_qty -= use_qty
            if float(lot.get("remaining_qty") or 0.0) <= 1e-9:
                lots[code].popleft()

        fallback_basis = None
        if remaining_sell_qty > 1e-9:
            fallback_basis = _snapshot_basis_for_sell(
                snapshots,
                code=code,
                sell_time=row.get("filled_at"),
                requested_qty=remaining_sell_qty,
            )
            if fallback_basis:
                fallback_qty = float(fallback_basis["matched_qty"])
                fallback_price = float(fallback_basis["avg_price"])
                matched_qty += fallback_qty
                matched_buy_cost += fallback_qty * fallback_price
                match_lots.append(
                    {
                        "basis_source": "position_snapshot_avg_price",
                        "snapshot_at": fallback_basis.get("snapshot_at"),
                        "snapshot_qty": fallback_basis.get("snapshot_qty"),
                        "matched_qty": fallback_qty,
                        "buy_price": fallback_price,
                        "allocated_buy_fee": 0.0,
                    }
                )
                remaining_sell_qty -= fallback_qty

        sell_fee_used = fee * (matched_qty / qty) if qty > 0 else 0.0
        sell_tax_used = tax * (matched_qty / qty) if qty > 0 else 0.0
        sell_proceeds = matched_qty * price
        avg_buy_price = matched_buy_cost / matched_qty if matched_qty > 0 else None
        realized_gross_pnl = sell_proceeds - matched_buy_cost if matched_qty > 0 else None
        realized_net_pnl = (
            sell_proceeds - matched_buy_cost - matched_buy_fee - sell_fee_used - sell_tax_used
            if matched_qty > 0
            else None
        )
        realized_return = realized_net_pnl / matched_buy_cost if matched_buy_cost > 0 and realized_net_pnl is not None else None
        if matched_qty <= 0:
            match_status = "BUY_BASIS_MISSING"
        elif remaining_sell_qty > 1e-9:
            match_status = "PARTIAL_BASIS"
        else:
            match_status = "MATCHED"
        match_method = "FIFO"
        if fallback_basis and matched_qty > 0:
            match_method = "FIFO_WITH_SNAPSHOT_AVG_FALLBACK"

        closed.append(
            {
                "sell_fill_id": row.get("fill_id"),
                "sell_request_id": row.get("request_id"),
                "intent_id": row.get("intent_id"),
                "as_of_date": row.get("as_of_date"),
                "closed_at": row.get("filled_at"),
                "code": code,
                "name": row.get("name"),
                "intent_type": row.get("intent_type"),
                "sell_qty": qty,
                "sell_price": price,
                "sell_amount": amount if amount is not None else qty * price,
                "sell_fee": fee,
                "sell_tax": tax,
                "matched_qty": matched_qty,
                "unmatched_qty": max(remaining_sell_qty, 0.0),
                "avg_buy_price": avg_buy_price,
                "matched_buy_cost": matched_buy_cost if matched_qty > 0 else None,
                "allocated_buy_fee": matched_buy_fee if matched_qty > 0 else None,
                "realized_gross_pnl": realized_gross_pnl,
                "realized_net_pnl": realized_net_pnl,
                "realized_return": realized_return,
                "match_status": match_status,
                "match_method": match_method,
                "match_lots": match_lots,
                "ranking_run_id": row.get("ranking_run_id"),
                "ranking_rank": row.get("ranking_rank"),
                "rank_bucket": _rank_bucket(row.get("ranking_rank")),
                "final_score": row.get("final_score"),
                "confidence_score": row.get("confidence_score"),
                "risk_penalty": row.get("risk_penalty"),
                "qual_score": row.get("qual_score"),
                "dominant_theme": row.get("dominant_theme"),
                "guard_applied_bucket": row.get("guard_applied_bucket"),
                "guard_penalty_bucket": row.get("guard_penalty_bucket"),
                "production_rank_bucket": row.get("production_rank_bucket"),
                "shadow_rank_bucket": row.get("shadow_rank_bucket"),
                "shadow_rank_delta_bucket": row.get("shadow_rank_delta_bucket"),
            }
        )

    return sorted(closed, key=lambda item: (str(item.get("closed_at") or ""), int(item.get("sell_fill_id") or 0)), reverse=True)


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(part) or "UNKNOWN" for part in keys)
        bucket = grouped.setdefault(
            key,
            {
                **{part: key[idx] for idx, part in enumerate(keys)},
                "count": 0,
                "observed_count": 0,
                "realized_net_pnl": 0.0,
                "_returns": [],
                "_wins": [],
            },
        )
        bucket["count"] += 1
        realized_return = _float(row.get("realized_return"))
        realized_net_pnl = _float(row.get("realized_net_pnl"))
        if realized_return is not None:
            bucket["observed_count"] += 1
            bucket["_returns"].append(realized_return)
            bucket["_wins"].append(1.0 if realized_return > 0 else 0.0)
        if realized_net_pnl is not None:
            bucket["realized_net_pnl"] += realized_net_pnl

    out: list[dict[str, Any]] = []
    for bucket in grouped.values():
        returns = bucket.pop("_returns")
        wins = bucket.pop("_wins")
        bucket["realized_net_pnl"] = bucket["realized_net_pnl"] if bucket["observed_count"] else None
        bucket["avg_realized_return"] = sum(returns) / len(returns) if returns else None
        bucket["win_rate"] = sum(wins) / len(wins) if wins else None
        out.append(bucket)
    return sorted(out, key=lambda item: (-(int(item.get("observed_count") or 0)), -(int(item.get("count") or 0)), str(item)))


def build_report() -> dict[str, Any]:
    closed = _build_fifo_closed_trades(_load_fill_rows(), _load_position_snapshots())
    returns = [_float(row.get("realized_return")) for row in closed if _float(row.get("realized_return")) is not None]
    pnl_values = [_float(row.get("realized_net_pnl")) for row in closed if _float(row.get("realized_net_pnl")) is not None]
    latest_closed_date = None
    if closed and closed[0].get("closed_at"):
        latest_closed_date = str(closed[0]["closed_at"])[:10]
    overview = {
        "latest_closed_date": latest_closed_date,
        "closed_trade_count": len(closed),
        "observed_count": len(returns),
        "realized_net_pnl": sum(pnl_values) if pnl_values else 0.0,
        "avg_realized_return": sum(returns) / len(returns) if returns else None,
        "win_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else None,
        "max_loss": min(returns) if returns else None,
        "unmatched_count": sum(1 for row in closed if row.get("match_status") != "MATCHED"),
        "partial_basis_count": sum(1 for row in closed if row.get("match_status") == "PARTIAL_BASIS"),
        "buy_basis_missing_count": sum(1 for row in closed if row.get("match_status") == "BUY_BASIS_MISSING"),
        "snapshot_fallback_count": sum(
            1 for row in closed if row.get("match_method") == "FIFO_WITH_SNAPSHOT_AVG_FALLBACK"
        ),
        "match_method": "FIFO_WITH_SNAPSHOT_AVG_FALLBACK",
    }
    by_intent = _aggregate(closed, ["intent_type"])
    by_rank = _aggregate(closed, ["rank_bucket"])
    by_guard = _aggregate(closed, ["guard_penalty_bucket", "shadow_rank_delta_bucket"])
    recent = closed[:20]

    observed_count = int(overview.get("observed_count") or 0)
    warnings: list[str] = []
    if observed_count < 30:
        warnings.append("Closed-trade observed_count is below 30; use this report as monitoring only.")
    if int(overview.get("unmatched_count") or 0) > 0:
        warnings.append("Some SELL fills do not have a complete prior BUY basis.")
    if int(overview.get("snapshot_fallback_count") or 0) > 0:
        warnings.append("Some closed trades use position snapshot avg_price as fallback basis; treat lot-level PnL as approximate.")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_closed_date": overview.get("latest_closed_date"),
        "overview": overview,
        "sample_status": _sample_status(observed_count),
        "by_intent": by_intent,
        "by_rank_bucket": by_rank,
        "by_quality_guard": by_guard,
        "recent_closed_trades": recent,
        "warnings": warnings,
    }
    return sanitize_for_json(report)


def _render_group(title: str, rows: list[dict[str, Any]], key: str) -> list[str]:
    lines = ["", f"## {title}", "", "| group | count | observed | pnl | avg_return | win_rate |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        group = row.get(key) or " / ".join(str(row.get(part) or "-") for part in ("guard_penalty_bucket", "shadow_rank_delta_bucket"))
        lines.append(
            f"| {group} | {row.get('count') or 0} | {row.get('observed_count') or 0} | "
            f"{_num(row.get('realized_net_pnl'))} | {_pct(row.get('avg_realized_return'))} | {_pct(row.get('win_rate'))} |"
        )
    if not rows:
        lines.append("| - | 0 | 0 | - | - | - |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    lines = [
        "# Live Closed Trade Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- latest_closed_date: `{report.get('latest_closed_date') or '-'}`",
        f"- sample_status: `{report.get('sample_status')}`",
        f"- closed_trade_count: `{overview.get('closed_trade_count') or 0}`",
        f"- observed_count: `{overview.get('observed_count') or 0}`",
        f"- realized_net_pnl: `{_num(overview.get('realized_net_pnl'))}`",
        f"- avg_realized_return: `{_pct(overview.get('avg_realized_return'))}`",
        f"- win_rate: `{_pct(overview.get('win_rate'))}`",
        f"- max_loss: `{_pct(overview.get('max_loss'))}`",
        f"- match_method: `{overview.get('match_method') or '-'}`",
        f"- unmatched_count: `{overview.get('unmatched_count') or 0}`",
        f"- partial_basis_count: `{overview.get('partial_basis_count') or 0}`",
        f"- snapshot_fallback_count: `{overview.get('snapshot_fallback_count') or 0}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(_render_group("By Intent", report["by_intent"], "intent_type"))
    lines.extend(_render_group("By Rank Bucket", report["by_rank_bucket"], "rank_bucket"))
    lines.extend(_render_group("By Quality Guard", report["by_quality_guard"], ""))
    lines.extend(
        [
            "",
            "## Recent Closed Trades",
            "",
            "| closed_at | code | name | qty | matched | unmatched | sell | avg_buy | pnl | return | match |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["recent_closed_trades"]:
        lines.append(
            f"| {row.get('closed_at') or '-'} | {row.get('code') or '-'} | {row.get('name') or '-'} | "
            f"{_num(row.get('sell_qty'))} | {_num(row.get('matched_qty'))} | {_num(row.get('unmatched_qty'))} | "
            f"{_num(row.get('sell_price'))} | {_num(row.get('avg_buy_price'))} | "
            f"{_num(row.get('realized_net_pnl'))} | {_pct(row.get('realized_return'))} | {row.get('match_status') or '-'} |"
        )
    if not report["recent_closed_trades"]:
        lines.append("| - | - | - | 0 | 0 | 0 | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.skip_ensure_views:
        apply_analytics_views()
    report = build_report()
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    write_json_strict(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    upsert_json_payload(
        "live_closed_trade_report",
        report,
        asof_date=report.get("latest_closed_date"),
        generated_at=report.get("generated_at"),
        source_path=out_json,
    )
    print(f"live_closed_trade_report_json: {out_json}")
    print(f"live_closed_trade_report_md: {out_md}")
    print(f"sample_status: {report.get('sample_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
