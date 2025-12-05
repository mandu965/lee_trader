# schemas.py

from pydantic import BaseModel
from datetime import date
from typing import List, Literal

class ActionItem(BaseModel):
    code: str
    name: str
    market: str
    sector: str
    close: float
    pred_return_60d: float
    pred_mdd_60d: float
    prob_top20_60d: float
    final_score: float | None = None
    reason: str  # 왜 이 액션에 들어갔는지 짧게

class TodayActionResponse(BaseModel):
    date: date
    horizon: Literal["60d", "90d"]
    buy_candidates: List[ActionItem]
    add_candidates: List[ActionItem]
    trim_candidates: List[ActionItem]
