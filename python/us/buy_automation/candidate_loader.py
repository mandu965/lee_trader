from __future__ import annotations

from datetime import date, timedelta
import json

from sqlalchemy import text

from python.us.buy_automation.config import BuyAutomationConfig
from python.us.us_db import (
    fetch_latest_daily_feature_snapshots,
    fetch_latest_financial_feature_snapshots,
    fetch_latest_relative_strength_snapshots,
    fetch_meta_us_universe_rows,
    fetch_price_history_for_tickers,
    fetch_rank_component_rows_between,
    get_us_engine,
    relation_exists,
)


LATEST_TRADE_DATE_SQL = text(
    """
    SELECT MAX(trade_date) AS trade_date
    FROM recommend.us_stock_rank_daily
    WHERE source = :source
      AND (:requested_trade_date IS NULL OR trade_date <= :requested_trade_date)
    """
)


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_probability(row: dict[str, object]) -> float | None:
    for key in ("probability", "prob", "predict_prob", "model_probability"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    raw_detail = row.get("score_detail_json")
    if isinstance(raw_detail, str) and raw_detail.strip():
        try:
            detail = json.loads(raw_detail)
        except json.JSONDecodeError:
            return None
        if isinstance(detail, dict):
            for key in ("probability", "prob", "model_probability"):
                value = _safe_float(detail.get(key))
                if value is not None:
                    return value
    return None


def _latest_trade_date(source: str, requested_trade_date: date | None = None) -> date | None:
    if not relation_exists("recommend.us_stock_rank_daily"):
        return None
    engine = get_us_engine()
    with engine.connect() as conn:
        return conn.execute(
            LATEST_TRADE_DATE_SQL,
            {
                "source": source,
                "requested_trade_date": requested_trade_date,
            },
        ).scalar()


def _group_price_history(price_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in price_rows:
        ticker = str(row.get("ticker") or "").upper()
        grouped.setdefault(ticker, []).append(row)
    return grouped


def _build_candidate(
    *,
    rank_row: dict[str, object],
    meta_row: dict[str, object] | None,
    price_history: list[dict[str, object]],
    daily_feature: dict[str, object] | None,
    relative_strength: dict[str, object] | None,
    financial_feature: dict[str, object] | None,
) -> dict[str, object]:
    latest_price = price_history[-1] if price_history else {}
    previous_price = price_history[-2] if len(price_history) >= 2 else {}
    close_price = _safe_float(latest_price.get("adj_close_price")) or _safe_float(latest_price.get("close_price"))
    open_price = _safe_float(latest_price.get("open_price"))
    prev_close = _safe_float(previous_price.get("adj_close_price")) or _safe_float(previous_price.get("close_price"))
    gap_up_pct = ((open_price / prev_close) - 1.0) if open_price and prev_close else None
    intraday_change_pct = ((close_price / open_price) - 1.0) if close_price and open_price else None
    probability = _extract_probability(rank_row)

    return {
        "trade_date": rank_row.get("trade_date"),
        "symbol": str(rank_row.get("symbol") or "").upper(),
        "rank": int(rank_row.get("rank_no") or 999999),
        "score": _safe_float(rank_row.get("total_score")),
        "probability": probability,
        "reference_price": close_price,
        "close_price": close_price,
        "open_price": open_price,
        "prev_close_price": prev_close,
        "gap_up_pct": gap_up_pct,
        "intraday_change_pct": intraday_change_pct,
        "sector": rank_row.get("sector") or (meta_row or {}).get("sector"),
        "industry": rank_row.get("industry") or (meta_row or {}).get("industry"),
        "company_name": rank_row.get("company_name") or (meta_row or {}).get("company_name"),
        "recommend_grade": str(rank_row.get("recommend_grade") or "").upper(),
        "data_status": str(rank_row.get("data_status") or "").upper(),
        "exclude_reason": rank_row.get("exclude_reason"),
        "score_detail_json": rank_row.get("score_detail_json"),
        "source_table": "recommend.us_stock_rank_daily",
        "daily_feature": daily_feature or {},
        "relative_strength": relative_strength or {},
        "financial_feature": financial_feature or {},
        "financial_quality_score": _safe_float((financial_feature or {}).get("financial_quality_score")),
        "volatility_20d": _safe_float((daily_feature or {}).get("volatility_20d")),
        "loader_filter_passed": True,
        "loader_filter_reasons": [],
    }


def load_buy_candidates(
    cfg: BuyAutomationConfig,
    *,
    requested_trade_date: date | None = None,
) -> dict[str, object]:
    try:
        rank_exists = relation_exists("recommend.us_stock_rank_daily")
    except Exception as exc:
        return {
            "trade_date": None,
            "candidates": [],
            "events": [{"severity": "ERROR", "reason_code": "RANKING_LOOKUP_FAILED", "detail": str(exc)}],
        }

    if not rank_exists:
        return {
            "trade_date": None,
            "candidates": [],
            "events": [{"severity": "ERROR", "reason_code": "RANKING_TABLE_MISSING", "detail": "recommend.us_stock_rank_daily is unavailable."}],
        }

    try:
        trade_date = _latest_trade_date(cfg.ranking_source, requested_trade_date=requested_trade_date)
        if trade_date is None:
            return {
                "trade_date": None,
                "candidates": [],
                "events": [{"severity": "WARNING", "reason_code": "RANKING_DATA_MISSING", "detail": "No ranking rows were found for the requested source/date."}],
            }
        if requested_trade_date is not None and trade_date != requested_trade_date:
            return {
                "trade_date": trade_date,
                "candidates": [],
                "events": [
                    {
                        "severity": "ERROR",
                        "reason_code": "RANKING_DATA_STALE",
                        "detail": (
                            f"Ranking source {cfg.ranking_source} is stale. "
                            f"requested_trade_date={requested_trade_date.isoformat()} "
                            f"latest_trade_date={trade_date.isoformat()}"
                        ),
                    }
                ],
            }

        rank_rows = fetch_rank_component_rows_between(start_date=trade_date, end_date=trade_date, source=cfg.ranking_source)
        rank_rows = sorted(rank_rows, key=lambda row: (int(row.get("rank_no") or 999999), str(row.get("symbol") or "")))
        top_rows = [row for row in rank_rows if int(row.get("rank_no") or 999999) <= cfg.top_n][: cfg.top_n]
        if not top_rows:
            return {
                "trade_date": trade_date,
                "candidates": [],
                "events": [{"severity": "WARNING", "reason_code": "TOP_N_EMPTY", "detail": f"No ranking rows matched Top N={cfg.top_n}."}],
            }

        symbols = [str(row.get("symbol") or "").upper() for row in top_rows]
        meta_map = {str(row.get("symbol") or "").upper(): row for row in fetch_meta_us_universe_rows() if str(row.get("symbol") or "").upper() in symbols}
        price_history_rows = fetch_price_history_for_tickers(symbols, end_date=trade_date)
        price_history_map = _group_price_history(price_history_rows)
        daily_feature_map = fetch_latest_daily_feature_snapshots(symbols, trade_date=trade_date)
        relative_strength_map = fetch_latest_relative_strength_snapshots(symbols, trade_date=trade_date)
        financial_feature_map = fetch_latest_financial_feature_snapshots(symbols, trade_date=trade_date)

        candidates: list[dict[str, object]] = []
        for row in top_rows:
            symbol = str(row.get("symbol") or "").upper()
            candidate = _build_candidate(
                rank_row=row,
                meta_row=meta_map.get(symbol),
                price_history=price_history_map.get(symbol, []),
                daily_feature=daily_feature_map.get(symbol),
                relative_strength=relative_strength_map.get(symbol),
                financial_feature=financial_feature_map.get(symbol),
            )
            if candidate["score"] is None or float(candidate["score"]) < cfg.min_score:
                candidate["loader_filter_passed"] = False
                candidate["loader_filter_reasons"].append("MIN_SCORE_NOT_MET")
            if cfg.min_prob > 0:
                probability = candidate.get("probability")
                if probability is None:
                    candidate["loader_filter_passed"] = False
                    candidate["loader_filter_reasons"].append("PROBABILITY_MISSING")
                elif float(probability) < cfg.min_prob:
                    candidate["loader_filter_passed"] = False
                    candidate["loader_filter_reasons"].append("MIN_PROBABILITY_NOT_MET")
            candidates.append(candidate)

        return {
            "trade_date": trade_date,
            "candidates": candidates,
            "events": [],
            "lookback_start_date": trade_date - timedelta(days=10),
            "source": cfg.ranking_source,
        }
    except Exception as exc:
        return {
            "trade_date": None,
            "candidates": [],
            "events": [{"severity": "ERROR", "reason_code": "CANDIDATE_LOAD_FAILED", "detail": str(exc)}],
        }
