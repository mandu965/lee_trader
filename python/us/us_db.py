from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from sqlalchemy import text

try:
    from python.db import get_engine
except ImportError:
    from db import get_engine


def get_us_engine():
    """
    Shared database entry point for Project C US-only modules.

    Phase 1-1 keeps DB access minimal and does not create business logic.
    """
    return get_engine()


READ_UNIVERSE_TICKERS_SQL = text(
    """
    SELECT ticker, is_active, added_date, removed_date
    FROM market.us_stock_universe
    WHERE universe_tag = :universe_tag
      AND is_active = 'Y'
    ORDER BY ticker
    """
)

READ_LAST_TRADE_DATES_SQL = text(
    """
    SELECT ticker, MAX(trade_date) AS last_trade_date
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
    GROUP BY ticker
    """
)

READ_UNIVERSE_COUNTS_SQL = text(
    """
    SELECT
        COUNT(*)::integer AS total_ticker_count,
        COUNT(*) FILTER (WHERE is_active = 'Y')::integer AS active_ticker_count
    FROM market.us_stock_universe
    WHERE universe_tag = :universe_tag
    """
)

READ_PRICE_STATS_SQL = text(
    """
    SELECT
        ticker,
        COUNT(*)::integer AS row_count,
        MAX(trade_date) AS last_trade_date
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date <= :as_of_date
      AND data_source = :data_source
    GROUP BY ticker
    ORDER BY ticker
    """
)

READ_ANOMALY_STATS_SQL = text(
    """
    SELECT
        ticker,
        COUNT(*)::integer AS anomaly_count,
        MAX(trade_date) AS last_anomaly_date
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date <= :as_of_date
      AND data_source = :data_source
      AND (
          close_price <= 0
          OR open_price <= 0
          OR high_price <= 0
          OR low_price <= 0
          OR high_price < low_price
          OR volume < 0
          OR trade_date IS NULL
          OR ticker IS NULL
      )
    GROUP BY ticker
    ORDER BY ticker
    """
)

READ_ORPHAN_TICKERS_SQL = text(
    """
    SELECT DISTINCT p.ticker
    FROM market.us_stock_daily_price AS p
    LEFT JOIN market.us_stock_universe AS u
      ON u.universe_tag = :universe_tag
     AND u.ticker = p.ticker
     AND u.is_active = 'Y'
    WHERE p.trade_date <= :as_of_date
      AND p.data_source = :data_source
      AND u.ticker IS NULL
    ORDER BY p.ticker
    """
)

READ_PRICE_HISTORY_SQL = text(
    """
    SELECT
        trade_date,
        ticker,
        open_price,
        high_price,
        low_price,
        close_price,
        adj_close_price,
        volume
    FROM market.us_stock_daily_price
    WHERE ticker = :ticker
      AND (:end_date IS NULL OR trade_date <= :end_date)
    ORDER BY trade_date
    """
)

UPSERT_FINANCIAL_STATEMENT_SQL = text(
    """
    INSERT INTO raw.us_stock_financial_statement (
        ticker,
        market,
        period_type,
        fiscal_date,
        reported_date,
        currency,
        revenue,
        gross_profit,
        operating_income,
        net_income,
        ebitda,
        total_assets,
        total_liabilities,
        total_equity,
        operating_cash_flow,
        investing_cash_flow,
        financing_cash_flow,
        free_cash_flow,
        source,
        source_updated_at,
        collected_at,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :market,
        :period_type,
        :fiscal_date,
        :reported_date,
        :currency,
        :revenue,
        :gross_profit,
        :operating_income,
        :net_income,
        :ebitda,
        :total_assets,
        :total_liabilities,
        :total_equity,
        :operating_cash_flow,
        :investing_cash_flow,
        :financing_cash_flow,
        :free_cash_flow,
        :source,
        :source_updated_at,
        :collected_at,
        now(),
        now()
    )
    ON CONFLICT (ticker, period_type, fiscal_date, source) DO UPDATE SET
        market = EXCLUDED.market,
        reported_date = EXCLUDED.reported_date,
        currency = EXCLUDED.currency,
        revenue = EXCLUDED.revenue,
        gross_profit = EXCLUDED.gross_profit,
        operating_income = EXCLUDED.operating_income,
        net_income = EXCLUDED.net_income,
        ebitda = EXCLUDED.ebitda,
        total_assets = EXCLUDED.total_assets,
        total_liabilities = EXCLUDED.total_liabilities,
        total_equity = EXCLUDED.total_equity,
        operating_cash_flow = EXCLUDED.operating_cash_flow,
        investing_cash_flow = EXCLUDED.investing_cash_flow,
        financing_cash_flow = EXCLUDED.financing_cash_flow,
        free_cash_flow = EXCLUDED.free_cash_flow,
        source_updated_at = EXCLUDED.source_updated_at,
        collected_at = EXCLUDED.collected_at,
        updated_at = now()
    """
)

UPSERT_FINANCIAL_METRIC_SQL = text(
    """
    INSERT INTO raw.us_stock_financial_metric (
        ticker,
        market,
        period_type,
        fiscal_date,
        reported_date,
        currency,
        eps,
        roe,
        roa,
        shares_outstanding,
        market_cap,
        per,
        pbr,
        psr,
        ev_ebitda,
        debt_to_equity,
        current_ratio,
        dividend_yield,
        source,
        source_updated_at,
        collected_at,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :market,
        :period_type,
        :fiscal_date,
        :reported_date,
        :currency,
        :eps,
        :roe,
        :roa,
        :shares_outstanding,
        :market_cap,
        :per,
        :pbr,
        :psr,
        :ev_ebitda,
        :debt_to_equity,
        :current_ratio,
        :dividend_yield,
        :source,
        :source_updated_at,
        :collected_at,
        now(),
        now()
    )
    ON CONFLICT (ticker, period_type, fiscal_date, source) DO UPDATE SET
        market = EXCLUDED.market,
        reported_date = EXCLUDED.reported_date,
        currency = EXCLUDED.currency,
        eps = EXCLUDED.eps,
        roe = EXCLUDED.roe,
        roa = EXCLUDED.roa,
        shares_outstanding = EXCLUDED.shares_outstanding,
        market_cap = EXCLUDED.market_cap,
        per = EXCLUDED.per,
        pbr = EXCLUDED.pbr,
        psr = EXCLUDED.psr,
        ev_ebitda = EXCLUDED.ev_ebitda,
        debt_to_equity = EXCLUDED.debt_to_equity,
        current_ratio = EXCLUDED.current_ratio,
        dividend_yield = EXCLUDED.dividend_yield,
        source_updated_at = EXCLUDED.source_updated_at,
        collected_at = EXCLUDED.collected_at,
        updated_at = now()
    """
)

UPSERT_PRICE_SQL = text(
    """
    INSERT INTO market.us_stock_daily_price (
        trade_date,
        ticker,
        open_price,
        high_price,
        low_price,
        close_price,
        adj_close_price,
        volume,
        data_source,
        created_at,
        updated_at
    ) VALUES (
        :trade_date,
        :ticker,
        :open_price,
        :high_price,
        :low_price,
        :close_price,
        :adj_close_price,
        :volume,
        :data_source,
        now(),
        now()
    )
    ON CONFLICT (trade_date, ticker) DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        adj_close_price = EXCLUDED.adj_close_price,
        volume = EXCLUDED.volume,
        data_source = EXCLUDED.data_source,
        updated_at = now()
    """
)

INSERT_COLLECT_LOG_SQL = text(
    """
    INSERT INTO market.us_stock_data_collect_log (
        collect_date,
        ticker,
        universe_tag,
        data_source,
        status,
        row_count,
        start_date,
        end_date,
        error_message,
        run_stage,
        created_at,
        updated_at
    ) VALUES (
        :collect_date,
        :ticker,
        :universe_tag,
        :data_source,
        :status,
        :row_count,
        :start_date,
        :end_date,
        :error_message,
        :run_stage,
        now(),
        now()
    )
    """
)

UPSERT_FEATURE_SQL = text(
    """
    INSERT INTO feature.us_stock_feature_daily (
        feature_date,
        ticker,
        ret_5d,
        ret_10d,
        ret_20d,
        ret_60d,
        volume_avg_20d,
        volatility_20d,
        ma_20,
        ma_60,
        price_above_ma20_flag,
        price_above_ma60_flag,
        created_at,
        updated_at
    ) VALUES (
        :feature_date,
        :ticker,
        :ret_5d,
        :ret_10d,
        :ret_20d,
        :ret_60d,
        :volume_avg_20d,
        :volatility_20d,
        :ma_20,
        :ma_60,
        :price_above_ma20_flag,
        :price_above_ma60_flag,
        now(),
        now()
    )
    ON CONFLICT (feature_date, ticker) DO UPDATE SET
        ret_5d = EXCLUDED.ret_5d,
        ret_10d = EXCLUDED.ret_10d,
        ret_20d = EXCLUDED.ret_20d,
        ret_60d = EXCLUDED.ret_60d,
        volume_avg_20d = EXCLUDED.volume_avg_20d,
        volatility_20d = EXCLUDED.volatility_20d,
        ma_20 = EXCLUDED.ma_20,
        ma_60 = EXCLUDED.ma_60,
        price_above_ma20_flag = EXCLUDED.price_above_ma20_flag,
        price_above_ma60_flag = EXCLUDED.price_above_ma60_flag,
        updated_at = now()
    """
)

UPSERT_QUALITY_REPORT_SQL = text(
    """
    INSERT INTO market.us_stock_data_quality_daily (
        check_date,
        universe_tag,
        total_ticker_count,
        active_ticker_count,
        price_ok_count,
        price_missing_count,
        stale_ticker_count,
        failed_ticker_count,
        quality_status,
        summary,
        created_at,
        updated_at
    ) VALUES (
        :check_date,
        :universe_tag,
        :total_ticker_count,
        :active_ticker_count,
        :price_ok_count,
        :price_missing_count,
        :stale_ticker_count,
        :failed_ticker_count,
        :quality_status,
        :summary,
        now(),
        now()
    )
    ON CONFLICT (check_date, universe_tag) DO UPDATE SET
        total_ticker_count = EXCLUDED.total_ticker_count,
        active_ticker_count = EXCLUDED.active_ticker_count,
        price_ok_count = EXCLUDED.price_ok_count,
        price_missing_count = EXCLUDED.price_missing_count,
        stale_ticker_count = EXCLUDED.stale_ticker_count,
        failed_ticker_count = EXCLUDED.failed_ticker_count,
        quality_status = EXCLUDED.quality_status,
        summary = EXCLUDED.summary,
        updated_at = now()
    """
)


UPSERT_UNIVERSE_SQL = text(
    """
    INSERT INTO market.us_stock_universe (
        ticker,
        name,
        sector,
        industry,
        universe_tag,
        is_active,
        added_date,
        removed_date,
        data_source,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :name,
        :sector,
        :industry,
        :universe_tag,
        :is_active,
        :added_date,
        :removed_date,
        :data_source,
        now(),
        now()
    )
    ON CONFLICT (universe_tag, ticker) DO UPDATE SET
        name = EXCLUDED.name,
        sector = EXCLUDED.sector,
        industry = EXCLUDED.industry,
        is_active = EXCLUDED.is_active,
        added_date = COALESCE(market.us_stock_universe.added_date, EXCLUDED.added_date),
        removed_date = EXCLUDED.removed_date,
        data_source = EXCLUDED.data_source,
        updated_at = now()
    """
)


DEACTIVATE_ALL_IF_EMPTY_SQL = text(
    """
    UPDATE market.us_stock_universe
    SET is_active = 'N',
        removed_date = COALESCE(removed_date, :removed_date),
        updated_at = now()
    WHERE universe_tag = :universe_tag
      AND is_active = 'Y'
    """
)


def fetch_universe_rows(universe_tag: str) -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_UNIVERSE_TICKERS_SQL, {"universe_tag": universe_tag}).mappings().all()
    return [dict(row) for row in rows]


def fetch_active_tickers(universe_tag: str) -> list[str]:
    return [str(row["ticker"]).upper() for row in fetch_universe_rows(universe_tag)]


def fetch_last_trade_dates(tickers: list[str]) -> dict[str, date]:
    if not tickers:
        return {}
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_LAST_TRADE_DATES_SQL, {"tickers": tickers}).mappings().all()
    out: dict[str, date] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        trade_date = row["last_trade_date"]
        if trade_date is not None:
            out[ticker] = trade_date
    return out


def fetch_universe_counts(universe_tag: str) -> dict[str, int]:
    engine = get_us_engine()
    with engine.connect() as conn:
        row = conn.execute(READ_UNIVERSE_COUNTS_SQL, {"universe_tag": universe_tag}).mappings().one()
    return {
        "total_ticker_count": int(row["total_ticker_count"] or 0),
        "active_ticker_count": int(row["active_ticker_count"] or 0),
    }


def fetch_price_stats(tickers: list[str], *, as_of_date: date, data_source: str) -> dict[str, dict[str, object]]:
    if not tickers:
        return {}
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_PRICE_STATS_SQL,
            {"tickers": tickers, "as_of_date": as_of_date, "data_source": data_source},
        ).mappings().all()
    return {str(row["ticker"]).upper(): dict(row) for row in rows}


def fetch_anomaly_stats(tickers: list[str], *, as_of_date: date, data_source: str) -> dict[str, dict[str, object]]:
    if not tickers:
        return {}
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_ANOMALY_STATS_SQL,
            {"tickers": tickers, "as_of_date": as_of_date, "data_source": data_source},
        ).mappings().all()
    return {str(row["ticker"]).upper(): dict(row) for row in rows}


def fetch_orphan_tickers(*, universe_tag: str, as_of_date: date, data_source: str) -> list[str]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_ORPHAN_TICKERS_SQL,
            {"universe_tag": universe_tag, "as_of_date": as_of_date, "data_source": data_source},
        ).mappings().all()
    return [str(row["ticker"]).upper() for row in rows]


def fetch_price_history(ticker: str, *, end_date: date | None = None) -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_PRICE_HISTORY_SQL,
            {"ticker": ticker, "end_date": end_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def upsert_universe_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_UNIVERSE_SQL, rows)
    return len(rows)


def deactivate_missing_universe_rows(universe_tag: str, keep_tickers: list[str], removed_date) -> int:
    engine = get_us_engine()
    with engine.begin() as conn:
        if keep_tickers:
            placeholders = {f"ticker_{idx}": ticker for idx, ticker in enumerate(keep_tickers)}
            bind_names = ", ".join(f":ticker_{idx}" for idx in range(len(keep_tickers)))
            stmt = text(
                f"""
                UPDATE market.us_stock_universe
                SET is_active = 'N',
                    removed_date = COALESCE(removed_date, :removed_date),
                    updated_at = now()
                WHERE universe_tag = :universe_tag
                  AND is_active = 'Y'
                  AND ticker NOT IN ({bind_names})
                """
            )
            params = {"universe_tag": universe_tag, "removed_date": removed_date, **placeholders}
            result = conn.execute(stmt, params)
            return int(result.rowcount or 0)
        result = conn.execute(
            DEACTIVATE_ALL_IF_EMPTY_SQL,
            {"universe_tag": universe_tag, "removed_date": removed_date},
        )
        return int(result.rowcount or 0)


def upsert_price_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_PRICE_SQL, rows)
    return len(rows)


def insert_collect_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_COLLECT_LOG_SQL, rows)
    return len(rows)


def upsert_quality_report(row: dict[str, object]) -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_QUALITY_REPORT_SQL, row)


def upsert_feature_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_FEATURE_SQL, rows)
    return len(rows)


def upsert_financial_statement_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_FINANCIAL_STATEMENT_SQL, rows)
    return len(rows)


def upsert_financial_metric_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_FINANCIAL_METRIC_SQL, rows)
    return len(rows)


def upsert_financial_rows(
    statement_rows: Iterable[dict[str, object]],
    metric_rows: Iterable[dict[str, object]],
) -> tuple[int, int]:
    statement_rows = list(statement_rows)
    metric_rows = list(metric_rows)
    if not statement_rows and not metric_rows:
        return 0, 0
    engine = get_us_engine()
    with engine.begin() as conn:
        if statement_rows:
            conn.execute(UPSERT_FINANCIAL_STATEMENT_SQL, statement_rows)
        if metric_rows:
            conn.execute(UPSERT_FINANCIAL_METRIC_SQL, metric_rows)
    return len(statement_rows), len(metric_rows)
