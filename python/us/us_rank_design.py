from __future__ import annotations

from datetime import date
import json


US_RECOMMEND_GRADES: tuple[str, ...] = (
    "STRONG_BUY",
    "BUY",
    "WATCH",
    "HOLD",
    "EXCLUDE",
)

US_RANKING_SCORE_WEIGHTS: dict[str, float] = {
    "momentum_score": 25.0,
    "relative_strength_score": 20.0,
    "fundamental_score": 20.0,
    "growth_score": 15.0,
    "valuation_score": 10.0,
    "risk_score_floor": -10.0,
    "risk_score_ceiling": 0.0,
}

US_RANKING_GRADE_GUIDE: dict[str, str] = {
    "STRONG_BUY": "total_score >= 80 with acceptable data quality and no excessive risk penalty",
    "BUY": "70 <= total_score < 80",
    "WATCH": "60 <= total_score < 70",
    "HOLD": "50 <= total_score < 60",
    "EXCLUDE": "total_score < 50 or excluded by universe, quality, or risk rules",
}


def build_score_detail_json(symbol: str) -> str:
    payload = {
        "symbol": symbol,
        "version": "phase3_2_design_sample",
        "momentum": {
            "score": 21.5,
            "max_score": 25,
            "reasons": [
                "20d return above peer median",
                "price above 60d moving average",
                "trend persistence confirmed",
            ],
        },
        "relative_strength": {
            "score": 16.2,
            "max_score": 20,
            "spy_relative_return_20d": 0.035,
            "qqq_relative_return_20d": 0.018,
        },
        "fundamental": {
            "score": 17.0,
            "max_score": 20,
            "roe": 0.28,
            "operating_margin": 0.31,
            "debt_to_equity": 0.42,
        },
        "growth": {
            "score": 12.5,
            "max_score": 15,
            "revenue_growth": 0.15,
            "earnings_growth": 0.18,
        },
        "valuation": {
            "score": 6.0,
            "max_score": 10,
            "trailing_pe": 32.4,
            "forward_pe": 27.1,
            "price_to_book": 8.5,
        },
        "risk": {
            "score": -3.0,
            "max_penalty": -10,
            "reasons": [
                "short-term volatility elevated",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def build_sample_rank_rows(*, trade_date: date, source: str) -> list[dict[str, object]]:
    base_rows = [
        ("AAPL", 1, "STRONG_BUY", 82.5, 23.0, 17.5, 18.5, 13.5, 8.0, -2.0, "Apple Inc.", "Technology", "Consumer Electronics"),
        ("MSFT", 2, "STRONG_BUY", 80.0, 22.0, 16.0, 19.0, 13.0, 9.0, -2.0, "Microsoft Corporation", "Technology", "Software"),
        ("NVDA", 3, "BUY", 76.5, 24.0, 18.5, 14.0, 12.5, 8.5, -1.0, "NVIDIA Corporation", "Technology", "Semiconductors"),
        ("AMZN", 4, "BUY", 72.0, 18.0, 15.5, 14.5, 13.0, 12.0, -1.0, "Amazon.com, Inc.", "Consumer Discretionary", "Internet Retail"),
        ("GOOGL", 5, "WATCH", 67.5, 16.5, 14.5, 17.0, 11.0, 10.0, -1.5, "Alphabet Inc.", "Communication Services", "Internet Content & Information"),
    ]
    rows: list[dict[str, object]] = []
    for symbol, rank_no, grade, total, momentum, rs, fundamental, growth, valuation, risk, company_name, sector, industry in base_rows:
        rows.append(
            {
                "trade_date": trade_date,
                "symbol": symbol,
                "rank_no": rank_no,
                "recommend_grade": grade,
                "total_score": total,
                "momentum_score": momentum,
                "relative_strength_score": rs,
                "fundamental_score": fundamental,
                "growth_score": growth,
                "valuation_score": valuation,
                "risk_score": risk,
                "feature_quality_score": 95.0 - rank_no,
                "universe_group": "NASDAQ100,SP500",
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "market_cap": 1000000000000.0 + (6 - rank_no) * 100000000000.0,
                "avg_volume": 45000000.0 + (6 - rank_no) * 5000000.0,
                "is_etf": False,
                "is_active": True,
                "data_status": "READY",
                "exclude_reason": None,
                "reason_summary": f"{grade} sample row for Phase 3-2 ranking table validation.",
                "score_detail_json": build_score_detail_json(symbol),
                "source": source,
            }
        )
    return rows
