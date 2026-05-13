from __future__ import annotations

import math
from typing import Literal


KrxSide = Literal["BUY", "SELL"]


def krx_tick_size(price: float) -> int:
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def normalize_krx_limit_price(price: float | int | None, *, side: KrxSide = "BUY") -> int | None:
    if price is None:
        return None
    value = float(price)
    if not math.isfinite(value) or value <= 0:
        return None
    tick = krx_tick_size(value)
    if side == "SELL":
        normalized = math.floor(value / tick) * tick
    else:
        normalized = math.ceil(value / tick) * tick
    return int(max(normalized, tick))
