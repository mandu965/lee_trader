from __future__ import annotations

from datetime import datetime, timezone
import uuid


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_paper_sell_order(
    *,
    trade_date: str,
    decision: dict[str, object],
    position: dict[str, object],
    mode: str = "PAPER",
) -> dict[str, object]:
    latest_price = _safe_float(position.get("latest_price"))
    avg_entry_price = _safe_float(position.get("avg_entry_price"))
    sell_quantity = _safe_float(decision.get("sell_quantity")) or 0.0
    sell_amount = round((latest_price or 0.0) * sell_quantity, 6)
    realized_paper_pnl = None
    realized_paper_pnl_pct = None
    if latest_price is not None and avg_entry_price is not None:
        realized_paper_pnl = round((latest_price - avg_entry_price) * sell_quantity, 6)
        if avg_entry_price > 0:
            realized_paper_pnl_pct = round((latest_price / avg_entry_price) - 1.0, 6)

    now = datetime.now(timezone.utc)
    return {
        "paper_sell_order_id": f"USPAPERSELL_{uuid.uuid4().hex[:24].upper()}",
        "paper_position_id": position.get("paper_position_id"),
        "symbol": str(position.get("symbol") or "").upper(),
        "trade_date": trade_date,
        "mode": mode,
        "side": "SELL",
        "sell_action": decision.get("requested_sell_action") or decision.get("sell_action"),
        "sell_quantity": round(sell_quantity, 6),
        "sell_price": round(latest_price, 6) if latest_price is not None else None,
        "sell_amount": sell_amount,
        "realized_paper_pnl": realized_paper_pnl,
        "realized_paper_pnl_pct": realized_paper_pnl_pct,
        "exit_reason": decision.get("exit_reason"),
        "assumed_fill_status": "ASSUMED_FILLED" if latest_price is not None else "PRICE_UNAVAILABLE",
        "created_at": now,
        "updated_at": now,
        "source_sell_decision_id": decision.get("sell_decision_id"),
        "limitation_note": "Paper-only assumed fill using latest available close/reference price. Fees, taxes, FX, and broker execution are excluded.",
    }
