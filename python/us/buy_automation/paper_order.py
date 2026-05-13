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


def build_paper_order(
    *,
    trade_date: str,
    symbol: str,
    allocated_amount_usd: float,
    reference_price: float | None,
    mode: str = "PAPER",
) -> dict[str, object]:
    price = _safe_float(reference_price)
    amount = max(0.0, float(allocated_amount_usd or 0.0))
    qty = round(amount / price, 6) if price and price > 0 else 0.0
    now = datetime.now(timezone.utc)
    fill_status = "PENDING"
    return {
        "paper_order_id": f"USPAPERBUY_{uuid.uuid4().hex[:24].upper()}",
        "trade_date": trade_date,
        "mode": mode,
        "symbol": str(symbol).upper(),
        "side": "BUY",
        "paper_order_qty": qty,
        "paper_order_price": round(price, 6) if price else None,
        "paper_order_amount": round(amount, 6),
        "assumed_fill_price": round(price, 6) if price else None,
        "assumed_fill_status": fill_status,
        "created_at": now,
        "updated_at": now,
    }
