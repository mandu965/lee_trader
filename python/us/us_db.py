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

READ_LATEST_PRICE_TRADE_DATE_SQL = text(
    """
    SELECT MAX(trade_date) AS latest_trade_date
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date <= :as_of_date
      AND data_source = :data_source
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

READ_PRICE_HISTORY_FOR_TICKERS_SQL = text(
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
    WHERE ticker = ANY(:tickers)
      AND (:end_date IS NULL OR trade_date <= :end_date)
    ORDER BY ticker, trade_date
    """
)

READ_FINANCIAL_STATEMENT_ROWS_SQL = text(
    """
    SELECT
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
        collected_at
    FROM raw.us_stock_financial_statement
    WHERE ticker = ANY(:tickers)
      AND period_type = ANY(:period_types)
      AND fiscal_date >= :min_fiscal_date
    ORDER BY ticker, period_type, fiscal_date
    """
)

READ_FINANCIAL_METRIC_ROWS_SQL = text(
    """
    SELECT
        ticker,
        market,
        period_type,
        fiscal_date,
        reported_date,
        currency,
        eps,
        forward_eps,
        roe,
        roa,
        shares_outstanding,
        market_cap,
        per,
        forward_pe,
        peg_ratio,
        pbr,
        psr,
        ev_ebitda,
        debt_to_equity,
        current_ratio,
        dividend_yield,
        analyst_target_price,
        analyst_recommendation,
        analyst_count,
        source,
        source_updated_at,
        collected_at
    FROM raw.us_stock_financial_metric
    WHERE ticker = ANY(:tickers)
      AND period_type = ANY(:period_types)
      AND fiscal_date >= :min_fiscal_date
    ORDER BY ticker, period_type, fiscal_date
    """
)

READ_DAILY_FEATURE_ROWS_SQL = text(
    """
    SELECT
        feature_date AS trade_date,
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
        price_above_ma60_flag
    FROM feature.us_stock_feature_daily
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, feature_date
    """
)

READ_RELATIVE_STRENGTH_ROWS_SQL = text(
    """
    SELECT *
    FROM feature.us_stock_relative_strength_daily
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, trade_date
    """
)

READ_FINANCIAL_FEATURE_ROWS_SQL = text(
    """
    SELECT *
    FROM feature.us_stock_financial_feature
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, fiscal_date
    """
)

READ_LABEL_ROWS_SQL = text(
    """
    SELECT *
    FROM label.us_stock_label_daily
    WHERE ticker = ANY(:tickers)
    ORDER BY ticker, trade_date
    """
)

READ_META_US_UNIVERSE_SQL = text(
    """
    SELECT *
    FROM meta.us_stock_universe
    ORDER BY symbol
    """
)

READ_US_RANK_ROWS_SQL = text(
    """
    SELECT *
    FROM recommend.us_stock_rank_daily
    WHERE symbol = ANY(:symbols)
    ORDER BY symbol, trade_date
    """
)

READ_US_RANK_ROWS_BETWEEN_SQL = text(
    """
    SELECT
        trade_date,
        symbol,
        rank_no,
        recommend_grade,
        total_score,
        data_status,
        exclude_reason,
        source
    FROM recommend.us_stock_rank_daily
    WHERE trade_date BETWEEN :start_date AND :end_date
      AND source = :source
    ORDER BY trade_date, rank_no ASC NULLS LAST, symbol ASC
    """
)

READ_US_RANK_COMPONENT_ROWS_BETWEEN_SQL = text(
    """
    SELECT
        trade_date,
        symbol,
        rank_no,
        recommend_grade,
        total_score,
        momentum_score,
        relative_strength_score,
        fundamental_score,
        growth_score,
        valuation_score,
        risk_score,
        feature_quality_score,
        universe_group,
        company_name,
        sector,
        industry,
        market_cap,
        avg_volume,
        is_etf,
        is_active,
        data_status,
        exclude_reason,
        reason_summary,
        score_detail_json::text AS score_detail_json,
        source
    FROM recommend.us_stock_rank_daily
    WHERE trade_date BETWEEN :start_date AND :end_date
      AND source = :source
    ORDER BY trade_date, symbol
    """
)

READ_US_WEIGHT_CONFIG_ROWS_SQL = text(
    """
    SELECT *
    FROM research.us_stock_rule_weight_config
    ORDER BY weight_config_id
    """
)

READ_US_WEIGHT_EXPERIMENT_SUMMARY_ROWS_SQL = text(
    """
    SELECT *
    FROM research.us_stock_weight_experiment_backtest_summary
    WHERE experiment_id = :experiment_id
    ORDER BY weight_config_id, strategy_name, holding_days
    """
)

READ_US_FORWARD_TEST_ROWS_SQL = text(
    """
    SELECT *
    FROM research.us_stock_rank_forward_test
    WHERE forward_test_id = :forward_test_id
      AND (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:strategy_name IS NULL OR strategy_name = :strategy_name)
      AND (:holding_days IS NULL OR holding_days = :holding_days)
      AND (:status IS NULL OR status = :status)
    ORDER BY trade_date, strategy_name, holding_days, rank_no NULLS LAST, symbol
    """
)

READ_US_FORWARD_TEST_SUMMARY_ROWS_SQL = text(
    """
    SELECT *
    FROM research.us_stock_rank_forward_test_summary
    WHERE forward_test_id = :forward_test_id
      AND (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:strategy_name IS NULL OR strategy_name = :strategy_name)
      AND (:holding_days IS NULL OR holding_days = :holding_days)
    ORDER BY trade_date, strategy_name, holding_days
    """
)

READ_US_PAPER_ACCOUNT_ROWS_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_account
    WHERE (:account_id IS NULL OR account_id = :account_id)
    ORDER BY account_id
    """
)

READ_US_PAPER_ORDER_ROWS_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_order
    WHERE (:paper_order_id IS NULL OR paper_order_id = :paper_order_id)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:side IS NULL OR side = :side)
      AND (:status IS NULL OR status = :status)
      AND (:strategy_name IS NULL OR strategy_name = :strategy_name)
    ORDER BY trade_date, created_at, symbol
    """
)

READ_US_PAPER_FILL_ROWS_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_fill
    WHERE (:paper_order_id IS NULL OR paper_order_id = :paper_order_id)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:trade_date IS NULL OR trade_date = :trade_date)
    ORDER BY trade_date, created_at, symbol
    """
)

READ_US_PAPER_POSITION_ROWS_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_position
    WHERE (:account_id IS NULL OR account_id = :account_id)
      AND (:status IS NULL OR status = :status)
    ORDER BY status, symbol
    """
)

READ_US_PAPER_ACCOUNT_SNAPSHOT_ROWS_SQL = text(
    """
    SELECT *
    FROM paper.us_stock_paper_account_snapshot
    WHERE (:account_id IS NULL OR account_id = :account_id)
      AND (:snapshot_date IS NULL OR snapshot_date = :snapshot_date)
    ORDER BY snapshot_date DESC, account_id
    """
)

READ_US_LIVE_KILL_SWITCH_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_kill_switch
    WHERE (:kill_switch_id IS NULL OR kill_switch_id = :kill_switch_id)
      AND (:scope IS NULL OR scope = :scope)
    ORDER BY kill_switch_id
    """
)

READ_US_LIVE_KILL_SWITCH_EVENT_LOG_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_kill_switch_event_log
    WHERE (:kill_switch_id IS NULL OR kill_switch_id = :kill_switch_id)
      AND (:scope IS NULL OR scope = :scope)
    ORDER BY created_at DESC, event_id DESC
    """
)

READ_US_LIVE_DAILY_RISK_USAGE_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_daily_risk_usage
    WHERE (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:policy_id IS NULL OR policy_id = :policy_id)
      AND (:account_id IS NULL OR account_id = :account_id)
    ORDER BY trade_date DESC, policy_id, account_id
    """
)

READ_US_LIVE_ORDER_BLOCK_LOG_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_order_block_log
    WHERE (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:policy_id IS NULL OR policy_id = :policy_id)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:symbol IS NULL OR symbol = :symbol)
    ORDER BY created_at DESC, block_id DESC
    """
)

READ_US_LIVE_ORDER_APPROVAL_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_order_approval
    WHERE (:approval_id IS NULL OR approval_id = :approval_id)
      AND (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:status IS NULL OR approval_status = :status)
    ORDER BY requested_at DESC, approval_id DESC
    """
)

READ_US_LIVE_ORDER_APPROVAL_EVENT_LOG_ROWS_SQL = text(
    """
    SELECT *
    FROM risk.us_stock_live_order_approval_event_log
    WHERE (:approval_id IS NULL OR approval_id = :approval_id)
    ORDER BY created_at DESC, event_id DESC
    """
)

READ_US_MICRO_ORDER_REQUEST_ROWS_SQL = text(
    """
    SELECT *
    FROM live.us_stock_micro_order_request
    WHERE (:micro_order_id IS NULL OR micro_order_id = :micro_order_id)
      AND (:approval_id IS NULL OR approval_id = :approval_id)
      AND (:trade_date IS NULL OR trade_date = :trade_date)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:status IS NULL OR request_status = :status)
      AND (:execution_mode IS NULL OR execution_mode = :execution_mode)
    ORDER BY created_at DESC, micro_order_id DESC
    """
)

READ_US_MICRO_ORDER_EVENT_LOG_ROWS_SQL = text(
    """
    SELECT *
    FROM live.us_stock_micro_order_event_log
    WHERE (:micro_order_id IS NULL OR micro_order_id = :micro_order_id)
    ORDER BY created_at DESC, event_id DESC
    """
)

READ_US_MICRO_ORDER_FILL_ROWS_SQL = text(
    """
    SELECT *
    FROM live.us_stock_micro_order_fill
    WHERE (:micro_order_id IS NULL OR micro_order_id = :micro_order_id)
      AND (:broker_order_id IS NULL OR broker_order_id = :broker_order_id)
    ORDER BY fill_time DESC NULLS LAST, created_at DESC, micro_fill_id DESC
    """
)

READ_US_MICRO_RECONCILIATION_RESULT_ROWS_SQL = text(
    """
    SELECT *
    FROM live.us_stock_micro_reconciliation_result
    WHERE (:recon_run_id IS NULL OR recon_run_id = :recon_run_id)
      AND (:recon_date IS NULL OR recon_date = :recon_date)
      AND (:account_id IS NULL OR account_id = :account_id)
      AND (:recon_type IS NULL OR recon_type = :recon_type)
      AND (:severity IS NULL OR severity = :severity)
    ORDER BY created_at DESC, recon_id DESC
    """
)

READ_US_MICRO_RECONCILIATION_EVENT_LOG_ROWS_SQL = text(
    """
    SELECT *
    FROM live.us_stock_micro_reconciliation_event_log
    WHERE (:recon_run_id IS NULL OR recon_run_id = :recon_run_id)
      AND (:event_type IS NULL OR event_type = :event_type)
    ORDER BY created_at DESC, event_id DESC
    """
)

READ_PRICE_ROWS_FOR_TICKERS_BETWEEN_SQL = text(
    """
    SELECT
        trade_date,
        ticker,
        close_price,
        adj_close_price,
        volume
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date BETWEEN :start_date AND :end_date
    ORDER BY ticker, trade_date
    """
)

READ_MIXED_PRICE_ROWS_FOR_TICKERS_BETWEEN_SQL = text(
    """
    SELECT
        trade_date,
        ticker,
        close_price,
        adj_close_price,
        volume,
        'stock' AS asset_type
    FROM market.us_stock_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date BETWEEN :start_date AND :end_date
    UNION ALL
    SELECT
        trade_date,
        ticker,
        close AS close_price,
        adj_close AS adj_close_price,
        volume,
        'etf' AS asset_type
    FROM market.us_etf_daily_price
    WHERE ticker = ANY(:tickers)
      AND trade_date BETWEEN :start_date AND :end_date
    ORDER BY ticker, trade_date
    """
)

READ_MARKET_REGIME_ROWS_BETWEEN_SQL = text(
    """
    SELECT *
    FROM research.us_market_regime_daily
    WHERE trade_date BETWEEN :start_date AND :end_date
    ORDER BY trade_date
    """
)

READ_LATEST_MACRO_SNAPSHOT_SQL = text(
    """
    SELECT trade_date, vix_close, vix_ret_20d, spy_ret_20d,
           spy_above_ma200, qqq_ret_20d, market_regime
    FROM feature.us_macro_daily
    WHERE trade_date <= :trade_date
    ORDER BY trade_date DESC
    LIMIT 1
    """
)

READ_LATEST_DAILY_FEATURE_SNAPSHOTS_SQL = text(
    """
    SELECT DISTINCT ON (ticker)
        feature_date AS trade_date,
        ticker,
        ret_1d,
        ret_3d,
        ret_5d,
        ret_10d,
        ret_20d,
        ret_60d,
        ret_252d,
        volume_avg_20d,
        volume_ratio_20d,
        volatility_20d,
        ma_20,
        ma_60,
        ma_200,
        price_vs_ma200,
        price_above_ma20_flag,
        price_above_ma60_flag,
        rsi_14,
        atr_14_norm,
        bb_position,
        high_52w_ratio,
        sector_rel_ret_20d,
        sector_rel_ret_60d,
        sector_rank_pct
    FROM feature.us_stock_feature_daily
    WHERE ticker = ANY(:tickers)
      AND feature_date <= :trade_date
    ORDER BY ticker, feature_date DESC
    """
)

READ_LATEST_RELATIVE_STRENGTH_SNAPSHOTS_SQL = text(
    """
    SELECT DISTINCT ON (ticker)
        ticker,
        trade_date,
        ret_5d,
        ret_20d,
        ret_60d,
        ret_120d,
        ret_252d,
        spy_ret_5d,
        spy_ret_20d,
        spy_ret_60d,
        spy_ret_120d,
        spy_ret_252d,
        qqq_ret_5d,
        qqq_ret_20d,
        qqq_ret_60d,
        qqq_ret_120d,
        qqq_ret_252d,
        rs_spy_5d,
        rs_spy_20d,
        rs_spy_60d,
        rs_spy_120d,
        rs_spy_252d,
        rs_qqq_5d,
        rs_qqq_20d,
        rs_qqq_60d,
        rs_qqq_120d,
        rs_qqq_252d,
        rs_spy_20d_rank_pct,
        rs_spy_60d_rank_pct,
        rs_qqq_20d_rank_pct,
        rs_qqq_60d_rank_pct,
        source
    FROM feature.us_stock_relative_strength_daily
    WHERE ticker = ANY(:tickers)
      AND trade_date <= :trade_date
    ORDER BY ticker, trade_date DESC, source
    """
)

READ_LATEST_FINANCIAL_FEATURE_SNAPSHOTS_SQL = text(
    """
    SELECT DISTINCT ON (ticker)
        ticker,
        market,
        period_type,
        fiscal_date,
        reported_date,
        source,
        revenue,
        gross_profit,
        operating_income,
        net_income,
        ebitda,
        eps,
        total_assets,
        total_liabilities,
        total_equity,
        operating_cash_flow,
        free_cash_flow,
        shares_outstanding,
        market_cap,
        revenue_growth_yoy,
        revenue_growth_qoq,
        net_income_growth_yoy,
        net_income_growth_qoq,
        eps_growth_yoy,
        eps_growth_qoq,
        free_cash_flow_growth_yoy,
        free_cash_flow_growth_qoq,
        gross_margin,
        operating_margin,
        net_margin,
        ebitda_margin,
        roe,
        roa,
        debt_to_equity,
        debt_ratio,
        equity_ratio,
        current_ratio,
        free_cash_flow_margin,
        per,
        forward_pe,
        peg_ratio,
        pbr,
        psr,
        ev_ebitda,
        dividend_yield,
        fcf_yield,
        asset_turnover,
        gross_margin_trend,
        revenue_growth_accel,
        roic_approx,
        analyst_target_price,
        analyst_recommendation,
        analyst_count,
        analyst_target_upside,
        financial_quality_score,
        financial_growth_score,
        financial_value_score
    FROM feature.us_stock_financial_feature
    WHERE ticker = ANY(:tickers)
      AND COALESCE(reported_date, fiscal_date) <= :trade_date
    ORDER BY
        ticker,
        COALESCE(reported_date, fiscal_date) DESC,
        fiscal_date DESC,
        CASE period_type
            WHEN 'ttm' THEN 1
            WHEN 'trailing' THEN 2
            WHEN 'quarterly' THEN 3
            WHEN 'annual' THEN 4
            ELSE 9
        END
    """
)

READ_ACTIVE_META_US_UNIVERSE_SQL = text(
    """
    WITH recent_price AS (
        SELECT
            ticker,
            AVG(volume)::numeric AS avg_volume_20d,
            MAX(trade_date) AS latest_trade_date
        FROM (
            SELECT
                ticker,
                trade_date,
                volume,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
            FROM market.us_stock_daily_price
        ) ranked
        WHERE rn <= 20
        GROUP BY ticker
    ),
    latest_financial AS (
        SELECT DISTINCT ON (ticker)
            ticker,
            market_cap,
            fiscal_date
        FROM feature.us_stock_financial_feature
        ORDER BY ticker, fiscal_date DESC
    ),
    daily_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_feature_daily
    ),
    rs_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_relative_strength_daily
    ),
    financial_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_financial_feature
    )
    SELECT
        u.symbol,
        u.company_name,
        u.market,
        u.sector,
        u.industry,
        u.universe_group,
        u.is_active,
        u.is_etf,
        u.is_leveraged,
        u.is_inverse,
        u.source,
        u.market_cap,
        u.avg_volume,
        u.currency,
        u.country,
        u.exchange,
        u.first_included_date,
        u.last_checked_date,
        u.exclude_reason,
        u.feature_quality_score,
        COALESCE(u.market_cap, lf.market_cap) AS effective_market_cap,
        COALESCE(u.avg_volume, rp.avg_volume_20d) AS effective_avg_volume,
        COALESCE(
            u.feature_quality_score,
            (CASE WHEN dfp.ticker IS NOT NULL THEN 40 ELSE 0 END) +
            (CASE WHEN rsp.ticker IS NOT NULL THEN 30 ELSE 0 END) +
            (CASE WHEN ffp.ticker IS NOT NULL THEN 30 ELSE 0 END)
        ) AS effective_feature_quality_score,
        rp.latest_trade_date
    FROM meta.us_stock_universe u
    LEFT JOIN recent_price rp
      ON rp.ticker = u.symbol
    LEFT JOIN latest_financial lf
      ON lf.ticker = u.symbol
    LEFT JOIN daily_feature_presence dfp
      ON dfp.ticker = u.symbol
    LEFT JOIN rs_feature_presence rsp
      ON rsp.ticker = u.symbol
    LEFT JOIN financial_feature_presence ffp
      ON ffp.ticker = u.symbol
    WHERE u.is_active = true
      AND (:include_etf = true OR COALESCE(u.is_etf, false) = false)
      AND (:exclude_leveraged = false OR COALESCE(u.is_leveraged, false) = false)
      AND (:exclude_inverse = false OR COALESCE(u.is_inverse, false) = false)
      AND rp.latest_trade_date IS NOT NULL
      AND COALESCE(COALESCE(u.market_cap, lf.market_cap), 0) >= :min_market_cap
      AND COALESCE(COALESCE(u.avg_volume, rp.avg_volume_20d), 0) >= :min_avg_volume
      AND COALESCE(
            u.feature_quality_score,
            (CASE WHEN dfp.ticker IS NOT NULL THEN 40 ELSE 0 END) +
            (CASE WHEN rsp.ticker IS NOT NULL THEN 30 ELSE 0 END) +
            (CASE WHEN ffp.ticker IS NOT NULL THEN 30 ELSE 0 END)
          ) >= :min_feature_quality_score
    ORDER BY u.symbol
    """
)

UPSERT_FINANCIAL_FEATURE_SQL = text(
    """
    INSERT INTO feature.us_stock_financial_feature (
        ticker,
        market,
        period_type,
        fiscal_date,
        reported_date,
        source,
        revenue,
        gross_profit,
        operating_income,
        net_income,
        ebitda,
        eps,
        forward_eps,
        total_assets,
        total_liabilities,
        total_equity,
        operating_cash_flow,
        free_cash_flow,
        shares_outstanding,
        market_cap,
        revenue_growth_yoy,
        revenue_growth_qoq,
        net_income_growth_yoy,
        net_income_growth_qoq,
        eps_growth_yoy,
        eps_growth_qoq,
        free_cash_flow_growth_yoy,
        free_cash_flow_growth_qoq,
        gross_margin,
        operating_margin,
        net_margin,
        ebitda_margin,
        roe,
        roa,
        debt_to_equity,
        debt_ratio,
        equity_ratio,
        current_ratio,
        free_cash_flow_margin,
        fcf_yield,
        asset_turnover,
        gross_margin_trend,
        revenue_growth_accel,
        roic_approx,
        analyst_target_price,
        analyst_recommendation,
        analyst_count,
        analyst_target_upside,
        per,
        forward_pe,
        peg_ratio,
        pbr,
        psr,
        ev_ebitda,
        dividend_yield,
        financial_quality_score,
        financial_growth_score,
        financial_value_score,
        raw_collected_at,
        feature_created_at,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :market,
        :period_type,
        :fiscal_date,
        :reported_date,
        :source,
        :revenue,
        :gross_profit,
        :operating_income,
        :net_income,
        :ebitda,
        :eps,
        :forward_eps,
        :total_assets,
        :total_liabilities,
        :total_equity,
        :operating_cash_flow,
        :free_cash_flow,
        :shares_outstanding,
        :market_cap,
        :revenue_growth_yoy,
        :revenue_growth_qoq,
        :net_income_growth_yoy,
        :net_income_growth_qoq,
        :eps_growth_yoy,
        :eps_growth_qoq,
        :free_cash_flow_growth_yoy,
        :free_cash_flow_growth_qoq,
        :gross_margin,
        :operating_margin,
        :net_margin,
        :ebitda_margin,
        :roe,
        :roa,
        :debt_to_equity,
        :debt_ratio,
        :equity_ratio,
        :current_ratio,
        :free_cash_flow_margin,
        :fcf_yield,
        :asset_turnover,
        :gross_margin_trend,
        :revenue_growth_accel,
        :roic_approx,
        :analyst_target_price,
        :analyst_recommendation,
        :analyst_count,
        :analyst_target_upside,
        :per,
        :forward_pe,
        :peg_ratio,
        :pbr,
        :psr,
        :ev_ebitda,
        :dividend_yield,
        :financial_quality_score,
        :financial_growth_score,
        :financial_value_score,
        :raw_collected_at,
        :feature_created_at,
        now(),
        now()
    )
    ON CONFLICT (ticker, period_type, fiscal_date, source) DO UPDATE SET
        market = EXCLUDED.market,
        reported_date = EXCLUDED.reported_date,
        revenue = EXCLUDED.revenue,
        gross_profit = EXCLUDED.gross_profit,
        operating_income = EXCLUDED.operating_income,
        net_income = EXCLUDED.net_income,
        ebitda = EXCLUDED.ebitda,
        eps = EXCLUDED.eps,
        forward_eps = EXCLUDED.forward_eps,
        total_assets = EXCLUDED.total_assets,
        total_liabilities = EXCLUDED.total_liabilities,
        total_equity = EXCLUDED.total_equity,
        operating_cash_flow = EXCLUDED.operating_cash_flow,
        free_cash_flow = EXCLUDED.free_cash_flow,
        shares_outstanding = EXCLUDED.shares_outstanding,
        market_cap = EXCLUDED.market_cap,
        revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
        revenue_growth_qoq = EXCLUDED.revenue_growth_qoq,
        net_income_growth_yoy = EXCLUDED.net_income_growth_yoy,
        net_income_growth_qoq = EXCLUDED.net_income_growth_qoq,
        eps_growth_yoy = EXCLUDED.eps_growth_yoy,
        eps_growth_qoq = EXCLUDED.eps_growth_qoq,
        free_cash_flow_growth_yoy = EXCLUDED.free_cash_flow_growth_yoy,
        free_cash_flow_growth_qoq = EXCLUDED.free_cash_flow_growth_qoq,
        gross_margin = EXCLUDED.gross_margin,
        operating_margin = EXCLUDED.operating_margin,
        net_margin = EXCLUDED.net_margin,
        ebitda_margin = EXCLUDED.ebitda_margin,
        roe = EXCLUDED.roe,
        roa = EXCLUDED.roa,
        debt_to_equity = EXCLUDED.debt_to_equity,
        debt_ratio = EXCLUDED.debt_ratio,
        equity_ratio = EXCLUDED.equity_ratio,
        current_ratio = EXCLUDED.current_ratio,
        free_cash_flow_margin = EXCLUDED.free_cash_flow_margin,
        fcf_yield = EXCLUDED.fcf_yield,
        asset_turnover = EXCLUDED.asset_turnover,
        gross_margin_trend = EXCLUDED.gross_margin_trend,
        revenue_growth_accel = EXCLUDED.revenue_growth_accel,
        roic_approx = EXCLUDED.roic_approx,
        analyst_target_price = EXCLUDED.analyst_target_price,
        analyst_recommendation = EXCLUDED.analyst_recommendation,
        analyst_count = EXCLUDED.analyst_count,
        analyst_target_upside = EXCLUDED.analyst_target_upside,
        per = EXCLUDED.per,
        forward_pe = EXCLUDED.forward_pe,
        peg_ratio = EXCLUDED.peg_ratio,
        pbr = EXCLUDED.pbr,
        psr = EXCLUDED.psr,
        ev_ebitda = EXCLUDED.ev_ebitda,
        dividend_yield = EXCLUDED.dividend_yield,
        financial_quality_score = EXCLUDED.financial_quality_score,
        financial_growth_score = EXCLUDED.financial_growth_score,
        financial_value_score = EXCLUDED.financial_value_score,
        raw_collected_at = EXCLUDED.raw_collected_at,
        feature_created_at = EXCLUDED.feature_created_at,
        updated_at = now()
    """
)

UPSERT_RELATIVE_STRENGTH_SQL = text(
    """
    INSERT INTO feature.us_stock_relative_strength_daily (
        ticker,
        market,
        trade_date,
        price_column_used,
        ret_5d,
        ret_20d,
        ret_60d,
        ret_120d,
        ret_252d,
        spy_ret_5d,
        spy_ret_20d,
        spy_ret_60d,
        spy_ret_120d,
        spy_ret_252d,
        qqq_ret_5d,
        qqq_ret_20d,
        qqq_ret_60d,
        qqq_ret_120d,
        qqq_ret_252d,
        rs_spy_5d,
        rs_spy_20d,
        rs_spy_60d,
        rs_spy_120d,
        rs_spy_252d,
        rs_qqq_5d,
        rs_qqq_20d,
        rs_qqq_60d,
        rs_qqq_120d,
        rs_qqq_252d,
        rs_spy_20d_rank_pct,
        rs_spy_60d_rank_pct,
        rs_qqq_20d_rank_pct,
        rs_qqq_60d_rank_pct,
        source,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :market,
        :trade_date,
        :price_column_used,
        :ret_5d,
        :ret_20d,
        :ret_60d,
        :ret_120d,
        :ret_252d,
        :spy_ret_5d,
        :spy_ret_20d,
        :spy_ret_60d,
        :spy_ret_120d,
        :spy_ret_252d,
        :qqq_ret_5d,
        :qqq_ret_20d,
        :qqq_ret_60d,
        :qqq_ret_120d,
        :qqq_ret_252d,
        :rs_spy_5d,
        :rs_spy_20d,
        :rs_spy_60d,
        :rs_spy_120d,
        :rs_spy_252d,
        :rs_qqq_5d,
        :rs_qqq_20d,
        :rs_qqq_60d,
        :rs_qqq_120d,
        :rs_qqq_252d,
        :rs_spy_20d_rank_pct,
        :rs_spy_60d_rank_pct,
        :rs_qqq_20d_rank_pct,
        :rs_qqq_60d_rank_pct,
        :source,
        now(),
        now()
    )
    ON CONFLICT (ticker, trade_date, source) DO UPDATE SET
        market = EXCLUDED.market,
        price_column_used = EXCLUDED.price_column_used,
        ret_5d = EXCLUDED.ret_5d,
        ret_20d = EXCLUDED.ret_20d,
        ret_60d = EXCLUDED.ret_60d,
        ret_120d = EXCLUDED.ret_120d,
        ret_252d = EXCLUDED.ret_252d,
        spy_ret_5d = EXCLUDED.spy_ret_5d,
        spy_ret_20d = EXCLUDED.spy_ret_20d,
        spy_ret_60d = EXCLUDED.spy_ret_60d,
        spy_ret_120d = EXCLUDED.spy_ret_120d,
        spy_ret_252d = EXCLUDED.spy_ret_252d,
        qqq_ret_5d = EXCLUDED.qqq_ret_5d,
        qqq_ret_20d = EXCLUDED.qqq_ret_20d,
        qqq_ret_60d = EXCLUDED.qqq_ret_60d,
        qqq_ret_120d = EXCLUDED.qqq_ret_120d,
        qqq_ret_252d = EXCLUDED.qqq_ret_252d,
        rs_spy_5d = EXCLUDED.rs_spy_5d,
        rs_spy_20d = EXCLUDED.rs_spy_20d,
        rs_spy_60d = EXCLUDED.rs_spy_60d,
        rs_spy_120d = EXCLUDED.rs_spy_120d,
        rs_spy_252d = EXCLUDED.rs_spy_252d,
        rs_qqq_5d = EXCLUDED.rs_qqq_5d,
        rs_qqq_20d = EXCLUDED.rs_qqq_20d,
        rs_qqq_60d = EXCLUDED.rs_qqq_60d,
        rs_qqq_120d = EXCLUDED.rs_qqq_120d,
        rs_qqq_252d = EXCLUDED.rs_qqq_252d,
        rs_spy_20d_rank_pct = EXCLUDED.rs_spy_20d_rank_pct,
        rs_spy_60d_rank_pct = EXCLUDED.rs_spy_60d_rank_pct,
        rs_qqq_20d_rank_pct = EXCLUDED.rs_qqq_20d_rank_pct,
        rs_qqq_60d_rank_pct = EXCLUDED.rs_qqq_60d_rank_pct,
        updated_at = now()
    """
)

CREATE_US_RANK_BACKTEST_RESULT_TABLE_SQL = text(
    """
    CREATE SCHEMA IF NOT EXISTS research
    """
)

CREATE_US_RANK_BACKTEST_RESULT_DETAIL_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_backtest_result (
        backtest_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        selection_rule VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        holding_days INTEGER NOT NULL,
        entry_date DATE,
        entry_price NUMERIC(18,6),
        exit_date DATE,
        exit_price NUMERIC(18,6),
        return_pct NUMERIC(18,6),
        spy_return_pct NUMERIC(18,6),
        qqq_return_pct NUMERIC(18,6),
        universe_avg_return_pct NUMERIC(18,6),
        excess_return_vs_spy NUMERIC(18,6),
        excess_return_vs_qqq NUMERIC(18,6),
        excess_return_vs_universe NUMERIC(18,6),
        win_flag INTEGER,
        win_vs_spy_flag INTEGER,
        win_vs_qqq_flag INTEGER,
        win_vs_universe_flag INTEGER,
        data_status VARCHAR(50),
        exclude_reason TEXT,
        source VARCHAR(50) DEFAULT 'rank_rule_v1',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (backtest_id, trade_date, strategy_name, symbol, holding_days)
    )
    """
)

ALTER_US_RANK_BACKTEST_RESULT_ADD_STRATEGY_NAME_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    ADD COLUMN IF NOT EXISTS strategy_name VARCHAR(100)
    """
)

ALTER_US_RANK_BACKTEST_RESULT_ADD_SELECTION_RULE_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    ADD COLUMN IF NOT EXISTS selection_rule VARCHAR(100)
    """
)

ALTER_US_RANK_BACKTEST_RESULT_SET_STRATEGY_NAME_SQL = text(
    """
    UPDATE research.us_stock_rank_backtest_result
    SET strategy_name = COALESCE(strategy_name, 'UNKNOWN_STRATEGY')
    WHERE strategy_name IS NULL
    """
)

ALTER_US_RANK_BACKTEST_RESULT_SET_SELECTION_RULE_SQL = text(
    """
    UPDATE research.us_stock_rank_backtest_result
    SET selection_rule = COALESCE(selection_rule, 'unknown')
    WHERE selection_rule IS NULL
    """
)

ALTER_US_RANK_BACKTEST_RESULT_STRATEGY_NAME_NOT_NULL_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    ALTER COLUMN strategy_name SET NOT NULL
    """
)

ALTER_US_RANK_BACKTEST_RESULT_SELECTION_RULE_NOT_NULL_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    ALTER COLUMN selection_rule SET NOT NULL
    """
)

DROP_US_RANK_BACKTEST_RESULT_PKEY_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    DROP CONSTRAINT IF EXISTS us_stock_rank_backtest_result_pkey
    """
)

ADD_US_RANK_BACKTEST_RESULT_PKEY_SQL = text(
    """
    ALTER TABLE research.us_stock_rank_backtest_result
    ADD CONSTRAINT us_stock_rank_backtest_result_pkey
    PRIMARY KEY (backtest_id, trade_date, strategy_name, symbol, holding_days)
    """
)

CREATE_US_RANK_BACKTEST_RESULT_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_backtest_result_trade_date
    ON research.us_stock_rank_backtest_result (trade_date, holding_days, symbol)
    """
)

CREATE_US_RANK_BACKTEST_RESULT_SYMBOL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_backtest_result_symbol
    ON research.us_stock_rank_backtest_result (symbol, trade_date)
    """
)

CREATE_US_RANK_BACKTEST_SUMMARY_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_backtest_summary (
        backtest_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        selection_rule VARCHAR(100) NOT NULL,
        holding_days INTEGER NOT NULL,
        selected_count INTEGER,
        avg_return_pct NUMERIC(18,6),
        median_return_pct NUMERIC(18,6),
        win_rate NUMERIC(18,6),
        avg_spy_return_pct NUMERIC(18,6),
        avg_qqq_return_pct NUMERIC(18,6),
        avg_universe_return_pct NUMERIC(18,6),
        avg_excess_return_vs_spy NUMERIC(18,6),
        avg_excess_return_vs_qqq NUMERIC(18,6),
        avg_excess_return_vs_universe NUMERIC(18,6),
        win_rate_vs_spy NUMERIC(18,6),
        win_rate_vs_qqq NUMERIC(18,6),
        win_rate_vs_universe NUMERIC(18,6),
        best_symbol VARCHAR(20),
        best_return_pct NUMERIC(18,6),
        worst_symbol VARCHAR(20),
        worst_return_pct NUMERIC(18,6),
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (backtest_id, trade_date, strategy_name, selection_rule, holding_days)
    )
    """
)

CREATE_US_RANK_BACKTEST_SUMMARY_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_backtest_summary_trade_date
    ON research.us_stock_rank_backtest_summary (trade_date, strategy_name, holding_days)
    """
)

CREATE_US_MARKET_REGIME_DAILY_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_market_regime_daily (
        trade_date DATE NOT NULL,
        spy_close NUMERIC(18,6),
        spy_return_20d NUMERIC(18,6),
        spy_return_60d NUMERIC(18,6),
        spy_ma20 NUMERIC(18,6),
        spy_ma60 NUMERIC(18,6),
        spy_volatility_20d NUMERIC(18,6),
        qqq_close NUMERIC(18,6),
        qqq_return_20d NUMERIC(18,6),
        qqq_return_60d NUMERIC(18,6),
        qqq_ma20 NUMERIC(18,6),
        qqq_ma60 NUMERIC(18,6),
        qqq_volatility_20d NUMERIC(18,6),
        spy_regime VARCHAR(30),
        qqq_regime VARCHAR(30),
        vol_regime VARCHAR(30),
        market_regime VARCHAR(50),
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date)
    )
    """
)

CREATE_US_MARKET_REGIME_DAILY_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_market_regime_daily_market_regime
    ON research.us_market_regime_daily (market_regime, trade_date)
    """
)

CREATE_US_RANK_BACKTEST_REGIME_SUMMARY_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_backtest_regime_summary (
        backtest_id VARCHAR(100) NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        selection_rule VARCHAR(100) NOT NULL,
        holding_days INTEGER NOT NULL,
        regime_type VARCHAR(50) NOT NULL,
        regime_value VARCHAR(50) NOT NULL,
        test_days INTEGER,
        selected_count_avg NUMERIC(18,6),
        avg_return_pct NUMERIC(18,6),
        median_return_pct NUMERIC(18,6),
        win_rate NUMERIC(18,6),
        avg_excess_return_vs_spy NUMERIC(18,6),
        avg_excess_return_vs_qqq NUMERIC(18,6),
        avg_excess_return_vs_universe NUMERIC(18,6),
        win_rate_vs_spy NUMERIC(18,6),
        win_rate_vs_qqq NUMERIC(18,6),
        win_rate_vs_universe NUMERIC(18,6),
        best_trade_date DATE,
        best_avg_return_pct NUMERIC(18,6),
        worst_trade_date DATE,
        worst_avg_return_pct NUMERIC(18,6),
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (
            backtest_id,
            strategy_name,
            holding_days,
            regime_type,
            regime_value
        )
    )
    """
)

CREATE_US_RANK_BACKTEST_REGIME_SUMMARY_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_backtest_regime_summary_lookup
    ON research.us_stock_rank_backtest_regime_summary (backtest_id, regime_type, regime_value)
    """
)

CREATE_US_RULE_WEIGHT_CONFIG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rule_weight_config (
        weight_config_id VARCHAR(100) NOT NULL,
        description TEXT,
        momentum_weight NUMERIC(10,4),
        relative_strength_weight NUMERIC(10,4),
        fundamental_weight NUMERIC(10,4),
        growth_weight NUMERIC(10,4),
        valuation_weight NUMERIC(10,4),
        risk_penalty_weight NUMERIC(10,4),
        is_active BOOLEAN DEFAULT true,
        is_baseline BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (weight_config_id)
    )
    """
)

CREATE_US_RULE_WEIGHT_CONFIG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rule_weight_config_active
    ON research.us_stock_rule_weight_config (is_active, is_baseline)
    """
)

CREATE_US_RANK_WEIGHT_EXPERIMENT_RESULT_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_weight_experiment_result (
        experiment_id VARCHAR(100) NOT NULL,
        weight_config_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        momentum_score NUMERIC(10,4),
        relative_strength_score NUMERIC(10,4),
        fundamental_score NUMERIC(10,4),
        growth_score NUMERIC(10,4),
        valuation_score NUMERIC(10,4),
        risk_score NUMERIC(10,4),
        reason_summary TEXT,
        score_detail_json TEXT,
        data_status VARCHAR(50),
        exclude_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (experiment_id, weight_config_id, trade_date, symbol)
    )
    """
)

CREATE_US_RANK_WEIGHT_EXPERIMENT_RESULT_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_weight_experiment_result_lookup
    ON research.us_stock_rank_weight_experiment_result (experiment_id, weight_config_id, trade_date, rank_no)
    """
)

CREATE_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_weight_experiment_backtest_summary (
        experiment_id VARCHAR(100) NOT NULL,
        weight_config_id VARCHAR(100) NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        selection_rule VARCHAR(100) NOT NULL,
        holding_days INTEGER NOT NULL,
        test_days INTEGER,
        selected_count_avg NUMERIC(18,6),
        avg_return_pct NUMERIC(18,6),
        median_return_pct NUMERIC(18,6),
        win_rate NUMERIC(18,6),
        avg_excess_return_vs_spy NUMERIC(18,6),
        avg_excess_return_vs_qqq NUMERIC(18,6),
        avg_excess_return_vs_universe NUMERIC(18,6),
        win_rate_vs_spy NUMERIC(18,6),
        win_rate_vs_qqq NUMERIC(18,6),
        win_rate_vs_universe NUMERIC(18,6),
        avg_return_bull NUMERIC(18,6),
        avg_return_bear NUMERIC(18,6),
        avg_return_high_vol NUMERIC(18,6),
        score_rank INTEGER,
        risk_adjusted_rank INTEGER,
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (
            experiment_id,
            weight_config_id,
            strategy_name,
            holding_days
        )
    )
    """
)

CREATE_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_weight_experiment_backtest_summary_lookup
    ON research.us_stock_weight_experiment_backtest_summary (experiment_id, strategy_name, holding_days)
    """
)

CREATE_US_RANK_FORWARD_TEST_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_forward_test (
        forward_test_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        holding_days INTEGER NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        selection_rule VARCHAR(100),
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        company_name VARCHAR(255),
        sector VARCHAR(100),
        industry VARCHAR(150),
        weight_config_id VARCHAR(100) DEFAULT 'RULE_V1_BASELINE',
        source VARCHAR(50) DEFAULT 'rule_v1',
        entry_date DATE,
        entry_price NUMERIC(18,6),
        target_exit_date DATE,
        exit_date DATE,
        exit_price NUMERIC(18,6),
        return_pct NUMERIC(18,6),
        spy_entry_price NUMERIC(18,6),
        spy_exit_price NUMERIC(18,6),
        spy_return_pct NUMERIC(18,6),
        qqq_entry_price NUMERIC(18,6),
        qqq_exit_price NUMERIC(18,6),
        qqq_return_pct NUMERIC(18,6),
        excess_return_vs_spy NUMERIC(18,6),
        excess_return_vs_qqq NUMERIC(18,6),
        win_flag INTEGER,
        win_vs_spy_flag INTEGER,
        win_vs_qqq_flag INTEGER,
        market_regime VARCHAR(50),
        spy_regime VARCHAR(30),
        qqq_regime VARCHAR(30),
        vol_regime VARCHAR(30),
        status VARCHAR(50),
        data_status VARCHAR(50),
        exclude_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (
          forward_test_id,
          trade_date,
          strategy_name,
          symbol,
          holding_days
        )
    )
    """
)

CREATE_US_RANK_FORWARD_TEST_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_forward_test_status
    ON research.us_stock_rank_forward_test (forward_test_id, status, holding_days, trade_date)
    """
)

CREATE_US_RANK_FORWARD_TEST_SYMBOL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_forward_test_symbol
    ON research.us_stock_rank_forward_test (symbol, trade_date, holding_days)
    """
)

CREATE_US_RANK_FORWARD_TEST_SUMMARY_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS research.us_stock_rank_forward_test_summary (
        forward_test_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        strategy_name VARCHAR(100) NOT NULL,
        holding_days INTEGER NOT NULL,
        selected_count INTEGER,
        completed_count INTEGER,
        active_count INTEGER,
        pending_count INTEGER,
        error_count INTEGER,
        avg_return_pct NUMERIC(18,6),
        median_return_pct NUMERIC(18,6),
        win_rate NUMERIC(18,6),
        avg_spy_return_pct NUMERIC(18,6),
        avg_qqq_return_pct NUMERIC(18,6),
        avg_excess_return_vs_spy NUMERIC(18,6),
        avg_excess_return_vs_qqq NUMERIC(18,6),
        win_rate_vs_spy NUMERIC(18,6),
        win_rate_vs_qqq NUMERIC(18,6),
        best_symbol VARCHAR(20),
        best_return_pct NUMERIC(18,6),
        worst_symbol VARCHAR(20),
        worst_return_pct NUMERIC(18,6),
        status VARCHAR(50),
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (
          forward_test_id,
          trade_date,
          strategy_name,
          holding_days
        )
    )
    """
)

CREATE_US_RANK_FORWARD_TEST_SUMMARY_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_rank_forward_test_summary_lookup
    ON research.us_stock_rank_forward_test_summary (forward_test_id, strategy_name, holding_days, trade_date)
    """
)

CREATE_US_PAPER_SCHEMA_SQL = text(
    """
    CREATE SCHEMA IF NOT EXISTS paper
    """
)

CREATE_US_LIVE_RISK_SCHEMA_SQL = text(
    """
    CREATE SCHEMA IF NOT EXISTS risk
    """
)

CREATE_US_MICRO_LIVE_SCHEMA_SQL = text(
    """
    CREATE SCHEMA IF NOT EXISTS live
    """
)

CREATE_US_PAPER_ACCOUNT_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS paper.us_stock_paper_account (
        account_id VARCHAR(100) NOT NULL,
        account_name VARCHAR(200),
        base_currency VARCHAR(10) DEFAULT 'USD',
        initial_cash NUMERIC(24,6),
        cash_balance NUMERIC(24,6),
        reserved_cash NUMERIC(24,6) DEFAULT 0,
        market_value NUMERIC(24,6) DEFAULT 0,
        equity_value NUMERIC(24,6) DEFAULT 0,
        realized_pnl NUMERIC(24,6) DEFAULT 0,
        unrealized_pnl NUMERIC(24,6) DEFAULT 0,
        total_pnl NUMERIC(24,6) DEFAULT 0,
        status VARCHAR(50) DEFAULT 'ACTIVE',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (account_id)
    )
    """
)

CREATE_US_PAPER_ORDER_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS paper.us_stock_paper_order (
        paper_order_id VARCHAR(100) NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        order_type VARCHAR(20) DEFAULT 'MARKET',
        order_qty NUMERIC(24,6),
        order_price NUMERIC(18,6),
        order_amount NUMERIC(24,6),
        limit_price NUMERIC(18,6),
        source VARCHAR(50),
        strategy_name VARCHAR(100),
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        status VARCHAR(50),
        reason TEXT,
        reject_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (paper_order_id)
    )
    """
)

CREATE_US_PAPER_FILL_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS paper.us_stock_paper_fill (
        paper_fill_id VARCHAR(100) NOT NULL,
        paper_order_id VARCHAR(100) NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        filled_qty NUMERIC(24,6),
        filled_price NUMERIC(18,6),
        filled_amount NUMERIC(24,6),
        commission NUMERIC(18,6) DEFAULT 0,
        slippage_amount NUMERIC(18,6) DEFAULT 0,
        fill_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (paper_fill_id)
    )
    """
)

CREATE_US_PAPER_POSITION_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS paper.us_stock_paper_position (
        account_id VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        qty NUMERIC(24,6),
        avg_price NUMERIC(18,6),
        cost_amount NUMERIC(24,6),
        last_price NUMERIC(18,6),
        market_value NUMERIC(24,6),
        unrealized_pnl NUMERIC(24,6),
        unrealized_pnl_pct NUMERIC(18,6),
        realized_pnl NUMERIC(24,6) DEFAULT 0,
        last_trade_date DATE,
        last_price_date DATE,
        status VARCHAR(50) DEFAULT 'OPEN',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (account_id, symbol)
    )
    """
)

CREATE_US_PAPER_ACCOUNT_SNAPSHOT_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS paper.us_stock_paper_account_snapshot (
        account_id VARCHAR(100) NOT NULL,
        snapshot_date DATE NOT NULL,
        cash_balance NUMERIC(24,6),
        reserved_cash NUMERIC(24,6),
        market_value NUMERIC(24,6),
        equity_value NUMERIC(24,6),
        realized_pnl NUMERIC(24,6),
        unrealized_pnl NUMERIC(24,6),
        total_pnl NUMERIC(24,6),
        total_pnl_pct NUMERIC(18,6),
        daily_return_pct NUMERIC(18,6),
        spy_return_pct NUMERIC(18,6),
        qqq_return_pct NUMERIC(18,6),
        excess_return_vs_spy NUMERIC(18,6),
        excess_return_vs_qqq NUMERIC(18,6),
        position_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (account_id, snapshot_date)
    )
    """
)

CREATE_US_PAPER_ACCOUNT_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_paper_order_account_trade_date
    ON paper.us_stock_paper_order (account_id, trade_date, symbol)
    """
)

CREATE_US_PAPER_FILL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_paper_fill_account_trade_date
    ON paper.us_stock_paper_fill (account_id, trade_date, symbol)
    """
)

CREATE_US_PAPER_POSITION_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_paper_position_status
    ON paper.us_stock_paper_position (account_id, status, symbol)
    """
)

CREATE_US_PAPER_SNAPSHOT_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_paper_account_snapshot_lookup
    ON paper.us_stock_paper_account_snapshot (account_id, snapshot_date DESC)
    """
)

CREATE_US_LIVE_KILL_SWITCH_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_kill_switch (
        kill_switch_id VARCHAR(100) NOT NULL,
        scope VARCHAR(50) NOT NULL,
        target_value VARCHAR(100),
        is_active BOOLEAN NOT NULL DEFAULT false,
        reason_code VARCHAR(100),
        reason_detail TEXT,
        activated_at TIMESTAMP,
        activated_by VARCHAR(100),
        cleared_at TIMESTAMP,
        cleared_by VARCHAR(100),
        clear_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (kill_switch_id)
    )
    """
)

ALTER_US_LIVE_KILL_SWITCH_ADD_TARGET_VALUE_SQL = text(
    """
    ALTER TABLE risk.us_stock_live_kill_switch
    ADD COLUMN IF NOT EXISTS target_value VARCHAR(100)
    """
)

CREATE_US_LIVE_KILL_SWITCH_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_kill_switch_scope
    ON risk.us_stock_live_kill_switch (scope, is_active)
    """
)

CREATE_US_LIVE_KILL_SWITCH_TARGET_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_kill_switch_target
    ON risk.us_stock_live_kill_switch (scope, target_value, is_active)
    """
)

CREATE_US_LIVE_DAILY_RISK_USAGE_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_daily_risk_usage (
        trade_date DATE NOT NULL,
        policy_id VARCHAR(100) NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        buy_order_count INTEGER DEFAULT 0,
        sell_order_count INTEGER DEFAULT 0,
        total_order_count INTEGER DEFAULT 0,
        new_buy_count INTEGER DEFAULT 0,
        buy_amount_usd NUMERIC(24,6) DEFAULT 0,
        sell_amount_usd NUMERIC(24,6) DEFAULT 0,
        total_order_amount_usd NUMERIC(24,6) DEFAULT 0,
        failed_order_count INTEGER DEFAULT 0,
        rejected_order_count INTEGER DEFAULT 0,
        blocked_order_count INTEGER DEFAULT 0,
        realized_pnl_usd NUMERIC(24,6),
        unrealized_pnl_usd NUMERIC(24,6),
        daily_pnl_usd NUMERIC(24,6),
        daily_pnl_pct NUMERIC(18,6),
        max_position_weight NUMERIC(18,6),
        max_sector_weight NUMERIC(18,6),
        cash_weight NUMERIC(18,6),
        data_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, policy_id, account_id)
    )
    """
)

CREATE_US_LIVE_DAILY_RISK_USAGE_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_daily_risk_usage_account
    ON risk.us_stock_live_daily_risk_usage (account_id, trade_date DESC)
    """
)

CREATE_US_LIVE_ORDER_BLOCK_LOG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_order_block_log (
        block_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        policy_id VARCHAR(100),
        account_id VARCHAR(100),
        symbol VARCHAR(20),
        side VARCHAR(10),
        candidate_source VARCHAR(100),
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        requested_order_amount_usd NUMERIC(24,6),
        requested_qty NUMERIC(24,6),
        requested_order_type VARCHAR(20),
        block_reason_code VARCHAR(100) NOT NULL,
        block_reason_detail TEXT,
        check_stage VARCHAR(50),
        severity VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (block_id)
    )
    """
)

CREATE_US_LIVE_ORDER_BLOCK_LOG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_order_block_log_trade_date
    ON risk.us_stock_live_order_block_log (trade_date, symbol, side)
    """
)

CREATE_US_LIVE_KILL_SWITCH_EVENT_LOG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_kill_switch_event_log (
        event_id VARCHAR(100) NOT NULL,
        kill_switch_id VARCHAR(100) NOT NULL,
        scope VARCHAR(50) NOT NULL,
        target_value VARCHAR(100),
        event_type VARCHAR(50) NOT NULL,
        reason_code VARCHAR(100),
        reason_detail TEXT,
        trigger_source VARCHAR(100),
        trigger_ref_id VARCHAR(100),
        performed_by VARCHAR(100),
        before_is_active BOOLEAN,
        after_is_active BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id)
    )
    """
)

CREATE_US_LIVE_KILL_SWITCH_EVENT_LOG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_kill_switch_event_log_lookup
    ON risk.us_stock_live_kill_switch_event_log (kill_switch_id, created_at DESC)
    """
)

CREATE_US_LIVE_ORDER_APPROVAL_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_order_approval (
        approval_id VARCHAR(120) NOT NULL,
        trade_date DATE NOT NULL,
        policy_id VARCHAR(100) NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        candidate_source VARCHAR(100),
        strategy_name VARCHAR(100),
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        requested_order_type VARCHAR(20),
        requested_limit_price NUMERIC(18,6),
        requested_qty NUMERIC(24,6),
        requested_order_amount_usd NUMERIC(24,6),
        precheck_decision VARCHAR(30),
        precheck_reason_codes TEXT,
        precheck_summary TEXT,
        approval_status VARCHAR(30) NOT NULL,
        requested_by VARCHAR(100),
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        approved_by VARCHAR(100),
        approved_at TIMESTAMP,
        approval_reason TEXT,
        rejected_by VARCHAR(100),
        rejected_at TIMESTAMP,
        reject_reason TEXT,
        expired_at TIMESTAMP,
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (approval_id)
    )
    """
)

CREATE_US_LIVE_ORDER_APPROVAL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_order_approval_lookup
    ON risk.us_stock_live_order_approval (approval_status, trade_date, account_id, symbol, side)
    """
)

CREATE_US_LIVE_ORDER_APPROVAL_EVENT_LOG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS risk.us_stock_live_order_approval_event_log (
        event_id VARCHAR(120) NOT NULL,
        approval_id VARCHAR(120) NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        before_status VARCHAR(30),
        after_status VARCHAR(30),
        reason_code VARCHAR(100),
        reason_detail TEXT,
        performed_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id)
    )
    """
)

CREATE_US_LIVE_ORDER_APPROVAL_EVENT_LOG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_live_order_approval_event_lookup
    ON risk.us_stock_live_order_approval_event_log (approval_id, created_at DESC)
    """
)

CREATE_US_MICRO_ORDER_REQUEST_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS live.us_stock_micro_order_request (
        micro_order_id VARCHAR(120) NOT NULL,
        approval_id VARCHAR(120),
        policy_id VARCHAR(100) NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        order_type VARCHAR(20) NOT NULL,
        limit_price NUMERIC(18,6),
        order_qty NUMERIC(24,6),
        order_amount_usd NUMERIC(24,6),
        candidate_source VARCHAR(100),
        strategy_name VARCHAR(100),
        rank_no INTEGER,
        recommend_grade VARCHAR(20),
        total_score NUMERIC(10,4),
        precheck_decision VARCHAR(30),
        precheck_reason_codes TEXT,
        precheck_summary TEXT,
        execution_mode VARCHAR(30) NOT NULL,
        broker_name VARCHAR(50),
        request_status VARCHAR(50) NOT NULL,
        request_payload TEXT,
        response_payload TEXT,
        broker_order_id VARCHAR(120),
        last_broker_status VARCHAR(100),
        last_sync_at TIMESTAMP,
        filled_qty NUMERIC(24,6),
        remaining_qty NUMERIC(24,6),
        avg_filled_price NUMERIC(18,6),
        filled_amount_usd NUMERIC(24,6),
        sync_status VARCHAR(50),
        sync_error TEXT,
        reject_reason_code VARCHAR(100),
        reject_reason_detail TEXT,
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (micro_order_id)
    )
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_LAST_BROKER_STATUS_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS last_broker_status VARCHAR(100)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_LAST_SYNC_AT_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMP
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_FILLED_QTY_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS filled_qty NUMERIC(24,6)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_REMAINING_QTY_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS remaining_qty NUMERIC(24,6)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_AVG_FILLED_PRICE_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS avg_filled_price NUMERIC(18,6)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_FILLED_AMOUNT_USD_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS filled_amount_usd NUMERIC(24,6)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_SYNC_STATUS_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS sync_status VARCHAR(50)
    """
)

ALTER_US_MICRO_ORDER_REQUEST_ADD_SYNC_ERROR_SQL = text(
    """
    ALTER TABLE live.us_stock_micro_order_request
    ADD COLUMN IF NOT EXISTS sync_error TEXT
    """
)

CREATE_US_MICRO_ORDER_REQUEST_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_order_request_lookup
    ON live.us_stock_micro_order_request (trade_date, account_id, request_status, execution_mode)
    """
)

CREATE_US_MICRO_ORDER_REQUEST_APPROVAL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_order_request_approval
    ON live.us_stock_micro_order_request (approval_id, created_at DESC)
    """
)

CREATE_US_MICRO_ORDER_EVENT_LOG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS live.us_stock_micro_order_event_log (
        event_id VARCHAR(120) NOT NULL,
        micro_order_id VARCHAR(120) NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        before_status VARCHAR(50),
        after_status VARCHAR(50),
        event_source VARCHAR(100),
        reason_code VARCHAR(100),
        reason_detail TEXT,
        request_payload TEXT,
        response_payload TEXT,
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id)
    )
    """
)

CREATE_US_MICRO_ORDER_EVENT_LOG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_order_event_log_lookup
    ON live.us_stock_micro_order_event_log (micro_order_id, created_at DESC)
    """
)

CREATE_US_MICRO_ORDER_FILL_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS live.us_stock_micro_order_fill (
        micro_fill_id VARCHAR(120) NOT NULL,
        micro_order_id VARCHAR(120) NOT NULL,
        broker_order_id VARCHAR(120),
        broker_fill_id VARCHAR(120),
        account_id VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        filled_qty NUMERIC(24,6),
        filled_price NUMERIC(18,6),
        filled_amount_usd NUMERIC(24,6),
        commission_usd NUMERIC(18,6),
        fee_usd NUMERIC(18,6),
        fill_time TIMESTAMP,
        fill_date DATE,
        liquidity_flag VARCHAR(30),
        raw_fill_payload TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (micro_fill_id)
    )
    """
)

CREATE_US_MICRO_RECONCILIATION_RESULT_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS live.us_stock_micro_reconciliation_result (
        recon_id VARCHAR(120) NOT NULL,
        recon_run_id VARCHAR(160) NOT NULL,
        recon_date DATE NOT NULL,
        account_id VARCHAR(100) NOT NULL,
        execution_mode VARCHAR(30) NOT NULL,
        recon_type VARCHAR(50) NOT NULL,
        symbol VARCHAR(20),
        micro_order_id VARCHAR(120),
        broker_order_id VARCHAR(120),
        internal_qty NUMERIC(24,6),
        broker_qty NUMERIC(24,6),
        qty_diff NUMERIC(24,6),
        internal_amount_usd NUMERIC(24,6),
        broker_amount_usd NUMERIC(24,6),
        amount_diff_usd NUMERIC(24,6),
        internal_cash_usd NUMERIC(24,6),
        broker_cash_usd NUMERIC(24,6),
        cash_diff_usd NUMERIC(24,6),
        internal_status VARCHAR(50),
        broker_status VARCHAR(50),
        recon_status VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,
        reason_code VARCHAR(100),
        reason_detail TEXT,
        raw_internal_payload TEXT,
        raw_broker_payload TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (recon_id)
    )
    """
)

CREATE_US_MICRO_RECONCILIATION_RESULT_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_reconciliation_result_lookup
    ON live.us_stock_micro_reconciliation_result (recon_date, account_id, recon_type, severity, created_at DESC)
    """
)

CREATE_US_MICRO_RECONCILIATION_RUN_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_reconciliation_result_run
    ON live.us_stock_micro_reconciliation_result (recon_run_id, created_at DESC)
    """
)

CREATE_US_MICRO_RECONCILIATION_EVENT_LOG_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS live.us_stock_micro_reconciliation_event_log (
        event_id VARCHAR(120) NOT NULL,
        recon_run_id VARCHAR(160) NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        account_id VARCHAR(100),
        execution_mode VARCHAR(30),
        message TEXT,
        severity VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id)
    )
    """
)

CREATE_US_MICRO_RECONCILIATION_EVENT_LOG_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_reconciliation_event_log_lookup
    ON live.us_stock_micro_reconciliation_event_log (recon_run_id, created_at DESC)
    """
)

CREATE_US_MICRO_ORDER_FILL_INDEX_SQL = text(
    """
    CREATE INDEX IF NOT EXISTS ix_us_stock_micro_order_fill_lookup
    ON live.us_stock_micro_order_fill (micro_order_id, fill_time DESC, broker_fill_id)
    """
)

UPSERT_US_RANK_BACKTEST_RESULT_SQL = text(
    """
    INSERT INTO research.us_stock_rank_backtest_result (
        backtest_id,
        trade_date,
        strategy_name,
        selection_rule,
        symbol,
        rank_no,
        recommend_grade,
        total_score,
        holding_days,
        entry_date,
        entry_price,
        exit_date,
        exit_price,
        return_pct,
        spy_return_pct,
        qqq_return_pct,
        universe_avg_return_pct,
        excess_return_vs_spy,
        excess_return_vs_qqq,
        excess_return_vs_universe,
        win_flag,
        win_vs_spy_flag,
        win_vs_qqq_flag,
        win_vs_universe_flag,
        data_status,
        exclude_reason,
        source,
        created_at,
        updated_at
    ) VALUES (
        :backtest_id,
        :trade_date,
        :strategy_name,
        :selection_rule,
        :symbol,
        :rank_no,
        :recommend_grade,
        :total_score,
        :holding_days,
        :entry_date,
        :entry_price,
        :exit_date,
        :exit_price,
        :return_pct,
        :spy_return_pct,
        :qqq_return_pct,
        :universe_avg_return_pct,
        :excess_return_vs_spy,
        :excess_return_vs_qqq,
        :excess_return_vs_universe,
        :win_flag,
        :win_vs_spy_flag,
        :win_vs_qqq_flag,
        :win_vs_universe_flag,
        :data_status,
        :exclude_reason,
        :source,
        now(),
        now()
    )
    ON CONFLICT (backtest_id, trade_date, strategy_name, symbol, holding_days) DO UPDATE SET
        selection_rule = EXCLUDED.selection_rule,
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        entry_date = EXCLUDED.entry_date,
        entry_price = EXCLUDED.entry_price,
        exit_date = EXCLUDED.exit_date,
        exit_price = EXCLUDED.exit_price,
        return_pct = EXCLUDED.return_pct,
        spy_return_pct = EXCLUDED.spy_return_pct,
        qqq_return_pct = EXCLUDED.qqq_return_pct,
        universe_avg_return_pct = EXCLUDED.universe_avg_return_pct,
        excess_return_vs_spy = EXCLUDED.excess_return_vs_spy,
        excess_return_vs_qqq = EXCLUDED.excess_return_vs_qqq,
        excess_return_vs_universe = EXCLUDED.excess_return_vs_universe,
        win_flag = EXCLUDED.win_flag,
        win_vs_spy_flag = EXCLUDED.win_vs_spy_flag,
        win_vs_qqq_flag = EXCLUDED.win_vs_qqq_flag,
        win_vs_universe_flag = EXCLUDED.win_vs_universe_flag,
        data_status = EXCLUDED.data_status,
        exclude_reason = EXCLUDED.exclude_reason,
        source = EXCLUDED.source,
        updated_at = now()
    """
)

UPSERT_US_RANK_BACKTEST_SUMMARY_SQL = text(
    """
    INSERT INTO research.us_stock_rank_backtest_summary (
        backtest_id,
        trade_date,
        strategy_name,
        selection_rule,
        holding_days,
        selected_count,
        avg_return_pct,
        median_return_pct,
        win_rate,
        avg_spy_return_pct,
        avg_qqq_return_pct,
        avg_universe_return_pct,
        avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe,
        win_rate_vs_spy,
        win_rate_vs_qqq,
        win_rate_vs_universe,
        best_symbol,
        best_return_pct,
        worst_symbol,
        worst_return_pct,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :backtest_id,
        :trade_date,
        :strategy_name,
        :selection_rule,
        :holding_days,
        :selected_count,
        :avg_return_pct,
        :median_return_pct,
        :win_rate,
        :avg_spy_return_pct,
        :avg_qqq_return_pct,
        :avg_universe_return_pct,
        :avg_excess_return_vs_spy,
        :avg_excess_return_vs_qqq,
        :avg_excess_return_vs_universe,
        :win_rate_vs_spy,
        :win_rate_vs_qqq,
        :win_rate_vs_universe,
        :best_symbol,
        :best_return_pct,
        :worst_symbol,
        :worst_return_pct,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (backtest_id, trade_date, strategy_name, selection_rule, holding_days) DO UPDATE SET
        selected_count = EXCLUDED.selected_count,
        avg_return_pct = EXCLUDED.avg_return_pct,
        median_return_pct = EXCLUDED.median_return_pct,
        win_rate = EXCLUDED.win_rate,
        avg_spy_return_pct = EXCLUDED.avg_spy_return_pct,
        avg_qqq_return_pct = EXCLUDED.avg_qqq_return_pct,
        avg_universe_return_pct = EXCLUDED.avg_universe_return_pct,
        avg_excess_return_vs_spy = EXCLUDED.avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq = EXCLUDED.avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe = EXCLUDED.avg_excess_return_vs_universe,
        win_rate_vs_spy = EXCLUDED.win_rate_vs_spy,
        win_rate_vs_qqq = EXCLUDED.win_rate_vs_qqq,
        win_rate_vs_universe = EXCLUDED.win_rate_vs_universe,
        best_symbol = EXCLUDED.best_symbol,
        best_return_pct = EXCLUDED.best_return_pct,
        worst_symbol = EXCLUDED.worst_symbol,
        worst_return_pct = EXCLUDED.worst_return_pct,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

UPSERT_US_MARKET_REGIME_DAILY_SQL = text(
    """
    INSERT INTO research.us_market_regime_daily (
        trade_date,
        spy_close,
        spy_return_20d,
        spy_return_60d,
        spy_ma20,
        spy_ma60,
        spy_volatility_20d,
        qqq_close,
        qqq_return_20d,
        qqq_return_60d,
        qqq_ma20,
        qqq_ma60,
        qqq_volatility_20d,
        spy_regime,
        qqq_regime,
        vol_regime,
        market_regime,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :trade_date,
        :spy_close,
        :spy_return_20d,
        :spy_return_60d,
        :spy_ma20,
        :spy_ma60,
        :spy_volatility_20d,
        :qqq_close,
        :qqq_return_20d,
        :qqq_return_60d,
        :qqq_ma20,
        :qqq_ma60,
        :qqq_volatility_20d,
        :spy_regime,
        :qqq_regime,
        :vol_regime,
        :market_regime,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (trade_date) DO UPDATE SET
        spy_close = EXCLUDED.spy_close,
        spy_return_20d = EXCLUDED.spy_return_20d,
        spy_return_60d = EXCLUDED.spy_return_60d,
        spy_ma20 = EXCLUDED.spy_ma20,
        spy_ma60 = EXCLUDED.spy_ma60,
        spy_volatility_20d = EXCLUDED.spy_volatility_20d,
        qqq_close = EXCLUDED.qqq_close,
        qqq_return_20d = EXCLUDED.qqq_return_20d,
        qqq_return_60d = EXCLUDED.qqq_return_60d,
        qqq_ma20 = EXCLUDED.qqq_ma20,
        qqq_ma60 = EXCLUDED.qqq_ma60,
        qqq_volatility_20d = EXCLUDED.qqq_volatility_20d,
        spy_regime = EXCLUDED.spy_regime,
        qqq_regime = EXCLUDED.qqq_regime,
        vol_regime = EXCLUDED.vol_regime,
        market_regime = EXCLUDED.market_regime,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

UPSERT_US_RANK_BACKTEST_REGIME_SUMMARY_SQL = text(
    """
    INSERT INTO research.us_stock_rank_backtest_regime_summary (
        backtest_id,
        strategy_name,
        selection_rule,
        holding_days,
        regime_type,
        regime_value,
        test_days,
        selected_count_avg,
        avg_return_pct,
        median_return_pct,
        win_rate,
        avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe,
        win_rate_vs_spy,
        win_rate_vs_qqq,
        win_rate_vs_universe,
        best_trade_date,
        best_avg_return_pct,
        worst_trade_date,
        worst_avg_return_pct,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :backtest_id,
        :strategy_name,
        :selection_rule,
        :holding_days,
        :regime_type,
        :regime_value,
        :test_days,
        :selected_count_avg,
        :avg_return_pct,
        :median_return_pct,
        :win_rate,
        :avg_excess_return_vs_spy,
        :avg_excess_return_vs_qqq,
        :avg_excess_return_vs_universe,
        :win_rate_vs_spy,
        :win_rate_vs_qqq,
        :win_rate_vs_universe,
        :best_trade_date,
        :best_avg_return_pct,
        :worst_trade_date,
        :worst_avg_return_pct,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (backtest_id, strategy_name, holding_days, regime_type, regime_value) DO UPDATE SET
        selection_rule = EXCLUDED.selection_rule,
        test_days = EXCLUDED.test_days,
        selected_count_avg = EXCLUDED.selected_count_avg,
        avg_return_pct = EXCLUDED.avg_return_pct,
        median_return_pct = EXCLUDED.median_return_pct,
        win_rate = EXCLUDED.win_rate,
        avg_excess_return_vs_spy = EXCLUDED.avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq = EXCLUDED.avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe = EXCLUDED.avg_excess_return_vs_universe,
        win_rate_vs_spy = EXCLUDED.win_rate_vs_spy,
        win_rate_vs_qqq = EXCLUDED.win_rate_vs_qqq,
        win_rate_vs_universe = EXCLUDED.win_rate_vs_universe,
        best_trade_date = EXCLUDED.best_trade_date,
        best_avg_return_pct = EXCLUDED.best_avg_return_pct,
        worst_trade_date = EXCLUDED.worst_trade_date,
        worst_avg_return_pct = EXCLUDED.worst_avg_return_pct,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

UPSERT_US_RULE_WEIGHT_CONFIG_SQL = text(
    """
    INSERT INTO research.us_stock_rule_weight_config (
        weight_config_id,
        description,
        momentum_weight,
        relative_strength_weight,
        fundamental_weight,
        growth_weight,
        valuation_weight,
        risk_penalty_weight,
        is_active,
        is_baseline,
        created_at,
        updated_at
    ) VALUES (
        :weight_config_id,
        :description,
        :momentum_weight,
        :relative_strength_weight,
        :fundamental_weight,
        :growth_weight,
        :valuation_weight,
        :risk_penalty_weight,
        :is_active,
        :is_baseline,
        now(),
        now()
    )
    ON CONFLICT (weight_config_id) DO UPDATE SET
        description = EXCLUDED.description,
        momentum_weight = EXCLUDED.momentum_weight,
        relative_strength_weight = EXCLUDED.relative_strength_weight,
        fundamental_weight = EXCLUDED.fundamental_weight,
        growth_weight = EXCLUDED.growth_weight,
        valuation_weight = EXCLUDED.valuation_weight,
        risk_penalty_weight = EXCLUDED.risk_penalty_weight,
        is_active = EXCLUDED.is_active,
        is_baseline = EXCLUDED.is_baseline,
        updated_at = now()
    """
)

UPSERT_US_RANK_WEIGHT_EXPERIMENT_RESULT_SQL = text(
    """
    INSERT INTO research.us_stock_rank_weight_experiment_result (
        experiment_id,
        weight_config_id,
        trade_date,
        symbol,
        rank_no,
        recommend_grade,
        total_score,
        momentum_score,
        relative_strength_score,
        fundamental_score,
        growth_score,
        valuation_score,
        risk_score,
        reason_summary,
        score_detail_json,
        data_status,
        exclude_reason,
        created_at,
        updated_at
    ) VALUES (
        :experiment_id,
        :weight_config_id,
        :trade_date,
        :symbol,
        :rank_no,
        :recommend_grade,
        :total_score,
        :momentum_score,
        :relative_strength_score,
        :fundamental_score,
        :growth_score,
        :valuation_score,
        :risk_score,
        :reason_summary,
        :score_detail_json,
        :data_status,
        :exclude_reason,
        now(),
        now()
    )
    ON CONFLICT (experiment_id, weight_config_id, trade_date, symbol) DO UPDATE SET
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        momentum_score = EXCLUDED.momentum_score,
        relative_strength_score = EXCLUDED.relative_strength_score,
        fundamental_score = EXCLUDED.fundamental_score,
        growth_score = EXCLUDED.growth_score,
        valuation_score = EXCLUDED.valuation_score,
        risk_score = EXCLUDED.risk_score,
        reason_summary = EXCLUDED.reason_summary,
        score_detail_json = EXCLUDED.score_detail_json,
        data_status = EXCLUDED.data_status,
        exclude_reason = EXCLUDED.exclude_reason,
        updated_at = now()
    """
)

UPSERT_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_SQL = text(
    """
    INSERT INTO research.us_stock_weight_experiment_backtest_summary (
        experiment_id,
        weight_config_id,
        strategy_name,
        selection_rule,
        holding_days,
        test_days,
        selected_count_avg,
        avg_return_pct,
        median_return_pct,
        win_rate,
        avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe,
        win_rate_vs_spy,
        win_rate_vs_qqq,
        win_rate_vs_universe,
        avg_return_bull,
        avg_return_bear,
        avg_return_high_vol,
        score_rank,
        risk_adjusted_rank,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :experiment_id,
        :weight_config_id,
        :strategy_name,
        :selection_rule,
        :holding_days,
        :test_days,
        :selected_count_avg,
        :avg_return_pct,
        :median_return_pct,
        :win_rate,
        :avg_excess_return_vs_spy,
        :avg_excess_return_vs_qqq,
        :avg_excess_return_vs_universe,
        :win_rate_vs_spy,
        :win_rate_vs_qqq,
        :win_rate_vs_universe,
        :avg_return_bull,
        :avg_return_bear,
        :avg_return_high_vol,
        :score_rank,
        :risk_adjusted_rank,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (experiment_id, weight_config_id, strategy_name, holding_days) DO UPDATE SET
        selection_rule = EXCLUDED.selection_rule,
        test_days = EXCLUDED.test_days,
        selected_count_avg = EXCLUDED.selected_count_avg,
        avg_return_pct = EXCLUDED.avg_return_pct,
        median_return_pct = EXCLUDED.median_return_pct,
        win_rate = EXCLUDED.win_rate,
        avg_excess_return_vs_spy = EXCLUDED.avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq = EXCLUDED.avg_excess_return_vs_qqq,
        avg_excess_return_vs_universe = EXCLUDED.avg_excess_return_vs_universe,
        win_rate_vs_spy = EXCLUDED.win_rate_vs_spy,
        win_rate_vs_qqq = EXCLUDED.win_rate_vs_qqq,
        win_rate_vs_universe = EXCLUDED.win_rate_vs_universe,
        avg_return_bull = EXCLUDED.avg_return_bull,
        avg_return_bear = EXCLUDED.avg_return_bear,
        avg_return_high_vol = EXCLUDED.avg_return_high_vol,
        score_rank = EXCLUDED.score_rank,
        risk_adjusted_rank = EXCLUDED.risk_adjusted_rank,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

UPSERT_US_RANK_FORWARD_TEST_SQL = text(
    """
    INSERT INTO research.us_stock_rank_forward_test (
        forward_test_id,
        trade_date,
        symbol,
        holding_days,
        strategy_name,
        selection_rule,
        rank_no,
        recommend_grade,
        total_score,
        company_name,
        sector,
        industry,
        weight_config_id,
        source,
        entry_date,
        entry_price,
        target_exit_date,
        exit_date,
        exit_price,
        return_pct,
        spy_entry_price,
        spy_exit_price,
        spy_return_pct,
        qqq_entry_price,
        qqq_exit_price,
        qqq_return_pct,
        excess_return_vs_spy,
        excess_return_vs_qqq,
        win_flag,
        win_vs_spy_flag,
        win_vs_qqq_flag,
        market_regime,
        spy_regime,
        qqq_regime,
        vol_regime,
        status,
        data_status,
        exclude_reason,
        created_at,
        updated_at
    ) VALUES (
        :forward_test_id,
        :trade_date,
        :symbol,
        :holding_days,
        :strategy_name,
        :selection_rule,
        :rank_no,
        :recommend_grade,
        :total_score,
        :company_name,
        :sector,
        :industry,
        :weight_config_id,
        :source,
        :entry_date,
        :entry_price,
        :target_exit_date,
        :exit_date,
        :exit_price,
        :return_pct,
        :spy_entry_price,
        :spy_exit_price,
        :spy_return_pct,
        :qqq_entry_price,
        :qqq_exit_price,
        :qqq_return_pct,
        :excess_return_vs_spy,
        :excess_return_vs_qqq,
        :win_flag,
        :win_vs_spy_flag,
        :win_vs_qqq_flag,
        :market_regime,
        :spy_regime,
        :qqq_regime,
        :vol_regime,
        :status,
        :data_status,
        :exclude_reason,
        now(),
        now()
    )
    ON CONFLICT (forward_test_id, trade_date, strategy_name, symbol, holding_days) DO UPDATE SET
        selection_rule = EXCLUDED.selection_rule,
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        company_name = EXCLUDED.company_name,
        sector = EXCLUDED.sector,
        industry = EXCLUDED.industry,
        weight_config_id = EXCLUDED.weight_config_id,
        source = EXCLUDED.source,
        entry_date = EXCLUDED.entry_date,
        entry_price = EXCLUDED.entry_price,
        target_exit_date = EXCLUDED.target_exit_date,
        exit_date = EXCLUDED.exit_date,
        exit_price = EXCLUDED.exit_price,
        return_pct = EXCLUDED.return_pct,
        spy_entry_price = EXCLUDED.spy_entry_price,
        spy_exit_price = EXCLUDED.spy_exit_price,
        spy_return_pct = EXCLUDED.spy_return_pct,
        qqq_entry_price = EXCLUDED.qqq_entry_price,
        qqq_exit_price = EXCLUDED.qqq_exit_price,
        qqq_return_pct = EXCLUDED.qqq_return_pct,
        excess_return_vs_spy = EXCLUDED.excess_return_vs_spy,
        excess_return_vs_qqq = EXCLUDED.excess_return_vs_qqq,
        win_flag = EXCLUDED.win_flag,
        win_vs_spy_flag = EXCLUDED.win_vs_spy_flag,
        win_vs_qqq_flag = EXCLUDED.win_vs_qqq_flag,
        market_regime = EXCLUDED.market_regime,
        spy_regime = EXCLUDED.spy_regime,
        qqq_regime = EXCLUDED.qqq_regime,
        vol_regime = EXCLUDED.vol_regime,
        status = EXCLUDED.status,
        data_status = EXCLUDED.data_status,
        exclude_reason = EXCLUDED.exclude_reason,
        updated_at = now()
    """
)

UPSERT_US_RANK_FORWARD_TEST_SUMMARY_SQL = text(
    """
    INSERT INTO research.us_stock_rank_forward_test_summary (
        forward_test_id,
        trade_date,
        strategy_name,
        holding_days,
        selected_count,
        completed_count,
        active_count,
        pending_count,
        error_count,
        avg_return_pct,
        median_return_pct,
        win_rate,
        avg_spy_return_pct,
        avg_qqq_return_pct,
        avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq,
        win_rate_vs_spy,
        win_rate_vs_qqq,
        best_symbol,
        best_return_pct,
        worst_symbol,
        worst_return_pct,
        status,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :forward_test_id,
        :trade_date,
        :strategy_name,
        :holding_days,
        :selected_count,
        :completed_count,
        :active_count,
        :pending_count,
        :error_count,
        :avg_return_pct,
        :median_return_pct,
        :win_rate,
        :avg_spy_return_pct,
        :avg_qqq_return_pct,
        :avg_excess_return_vs_spy,
        :avg_excess_return_vs_qqq,
        :win_rate_vs_spy,
        :win_rate_vs_qqq,
        :best_symbol,
        :best_return_pct,
        :worst_symbol,
        :worst_return_pct,
        :status,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (forward_test_id, trade_date, strategy_name, holding_days) DO UPDATE SET
        selected_count = EXCLUDED.selected_count,
        completed_count = EXCLUDED.completed_count,
        active_count = EXCLUDED.active_count,
        pending_count = EXCLUDED.pending_count,
        error_count = EXCLUDED.error_count,
        avg_return_pct = EXCLUDED.avg_return_pct,
        median_return_pct = EXCLUDED.median_return_pct,
        win_rate = EXCLUDED.win_rate,
        avg_spy_return_pct = EXCLUDED.avg_spy_return_pct,
        avg_qqq_return_pct = EXCLUDED.avg_qqq_return_pct,
        avg_excess_return_vs_spy = EXCLUDED.avg_excess_return_vs_spy,
        avg_excess_return_vs_qqq = EXCLUDED.avg_excess_return_vs_qqq,
        win_rate_vs_spy = EXCLUDED.win_rate_vs_spy,
        win_rate_vs_qqq = EXCLUDED.win_rate_vs_qqq,
        best_symbol = EXCLUDED.best_symbol,
        best_return_pct = EXCLUDED.best_return_pct,
        worst_symbol = EXCLUDED.worst_symbol,
        worst_return_pct = EXCLUDED.worst_return_pct,
        status = EXCLUDED.status,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

UPSERT_US_PAPER_ACCOUNT_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_account (
        account_id,
        account_name,
        base_currency,
        initial_cash,
        cash_balance,
        reserved_cash,
        market_value,
        equity_value,
        realized_pnl,
        unrealized_pnl,
        total_pnl,
        status,
        created_at,
        updated_at
    ) VALUES (
        :account_id,
        :account_name,
        :base_currency,
        :initial_cash,
        :cash_balance,
        :reserved_cash,
        :market_value,
        :equity_value,
        :realized_pnl,
        :unrealized_pnl,
        :total_pnl,
        :status,
        now(),
        now()
    )
    ON CONFLICT (account_id) DO UPDATE SET
        account_name = EXCLUDED.account_name,
        base_currency = EXCLUDED.base_currency,
        initial_cash = EXCLUDED.initial_cash,
        cash_balance = EXCLUDED.cash_balance,
        reserved_cash = EXCLUDED.reserved_cash,
        market_value = EXCLUDED.market_value,
        equity_value = EXCLUDED.equity_value,
        realized_pnl = EXCLUDED.realized_pnl,
        unrealized_pnl = EXCLUDED.unrealized_pnl,
        total_pnl = EXCLUDED.total_pnl,
        status = EXCLUDED.status,
        updated_at = now()
    """
)

UPSERT_US_PAPER_ORDER_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_order (
        paper_order_id,
        account_id,
        trade_date,
        symbol,
        side,
        order_type,
        order_qty,
        order_price,
        order_amount,
        limit_price,
        source,
        strategy_name,
        rank_no,
        recommend_grade,
        total_score,
        status,
        reason,
        reject_reason,
        created_at,
        updated_at
    ) VALUES (
        :paper_order_id,
        :account_id,
        :trade_date,
        :symbol,
        :side,
        :order_type,
        :order_qty,
        :order_price,
        :order_amount,
        :limit_price,
        :source,
        :strategy_name,
        :rank_no,
        :recommend_grade,
        :total_score,
        :status,
        :reason,
        :reject_reason,
        now(),
        now()
    )
    ON CONFLICT (paper_order_id) DO UPDATE SET
        order_type = EXCLUDED.order_type,
        order_qty = EXCLUDED.order_qty,
        order_price = EXCLUDED.order_price,
        order_amount = EXCLUDED.order_amount,
        limit_price = EXCLUDED.limit_price,
        source = EXCLUDED.source,
        strategy_name = EXCLUDED.strategy_name,
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        status = EXCLUDED.status,
        reason = EXCLUDED.reason,
        reject_reason = EXCLUDED.reject_reason,
        updated_at = now()
    """
)

UPSERT_US_PAPER_ACCOUNT_SNAPSHOT_SQL = text(
    """
    INSERT INTO paper.us_stock_paper_account_snapshot (
        account_id,
        snapshot_date,
        cash_balance,
        reserved_cash,
        market_value,
        equity_value,
        realized_pnl,
        unrealized_pnl,
        total_pnl,
        total_pnl_pct,
        daily_return_pct,
        spy_return_pct,
        qqq_return_pct,
        excess_return_vs_spy,
        excess_return_vs_qqq,
        position_count,
        created_at
    ) VALUES (
        :account_id,
        :snapshot_date,
        :cash_balance,
        :reserved_cash,
        :market_value,
        :equity_value,
        :realized_pnl,
        :unrealized_pnl,
        :total_pnl,
        :total_pnl_pct,
        :daily_return_pct,
        :spy_return_pct,
        :qqq_return_pct,
        :excess_return_vs_spy,
        :excess_return_vs_qqq,
        :position_count,
        now()
    )
    ON CONFLICT (account_id, snapshot_date) DO UPDATE SET
        cash_balance = EXCLUDED.cash_balance,
        reserved_cash = EXCLUDED.reserved_cash,
        market_value = EXCLUDED.market_value,
        equity_value = EXCLUDED.equity_value,
        realized_pnl = EXCLUDED.realized_pnl,
        unrealized_pnl = EXCLUDED.unrealized_pnl,
        total_pnl = EXCLUDED.total_pnl,
        total_pnl_pct = EXCLUDED.total_pnl_pct,
        daily_return_pct = EXCLUDED.daily_return_pct,
        spy_return_pct = EXCLUDED.spy_return_pct,
        qqq_return_pct = EXCLUDED.qqq_return_pct,
        excess_return_vs_spy = EXCLUDED.excess_return_vs_spy,
        excess_return_vs_qqq = EXCLUDED.excess_return_vs_qqq,
        position_count = EXCLUDED.position_count
    """
)

UPSERT_US_LIVE_KILL_SWITCH_SQL = text(
    """
    INSERT INTO risk.us_stock_live_kill_switch (
        kill_switch_id,
        scope,
        target_value,
        is_active,
        reason_code,
        reason_detail,
        activated_at,
        activated_by,
        cleared_at,
        cleared_by,
        clear_reason,
        created_at,
        updated_at
    ) VALUES (
        :kill_switch_id,
        :scope,
        :target_value,
        :is_active,
        :reason_code,
        :reason_detail,
        :activated_at,
        :activated_by,
        :cleared_at,
        :cleared_by,
        :clear_reason,
        now(),
        now()
    )
    ON CONFLICT (kill_switch_id) DO UPDATE SET
        scope = EXCLUDED.scope,
        target_value = EXCLUDED.target_value,
        is_active = EXCLUDED.is_active,
        reason_code = EXCLUDED.reason_code,
        reason_detail = EXCLUDED.reason_detail,
        activated_at = EXCLUDED.activated_at,
        activated_by = EXCLUDED.activated_by,
        cleared_at = EXCLUDED.cleared_at,
        cleared_by = EXCLUDED.cleared_by,
        clear_reason = EXCLUDED.clear_reason,
        updated_at = now()
    """
)

INSERT_US_LIVE_KILL_SWITCH_EVENT_LOG_SQL = text(
    """
    INSERT INTO risk.us_stock_live_kill_switch_event_log (
        event_id,
        kill_switch_id,
        scope,
        target_value,
        event_type,
        reason_code,
        reason_detail,
        trigger_source,
        trigger_ref_id,
        performed_by,
        before_is_active,
        after_is_active,
        created_at
    ) VALUES (
        :event_id,
        :kill_switch_id,
        :scope,
        :target_value,
        :event_type,
        :reason_code,
        :reason_detail,
        :trigger_source,
        :trigger_ref_id,
        :performed_by,
        :before_is_active,
        :after_is_active,
        now()
    )
    """
)

UPSERT_US_LIVE_DAILY_RISK_USAGE_SQL = text(
    """
    INSERT INTO risk.us_stock_live_daily_risk_usage (
        trade_date,
        policy_id,
        account_id,
        buy_order_count,
        sell_order_count,
        total_order_count,
        new_buy_count,
        buy_amount_usd,
        sell_amount_usd,
        total_order_amount_usd,
        failed_order_count,
        rejected_order_count,
        blocked_order_count,
        realized_pnl_usd,
        unrealized_pnl_usd,
        daily_pnl_usd,
        daily_pnl_pct,
        max_position_weight,
        max_sector_weight,
        cash_weight,
        data_status,
        created_at,
        updated_at
    ) VALUES (
        :trade_date,
        :policy_id,
        :account_id,
        :buy_order_count,
        :sell_order_count,
        :total_order_count,
        :new_buy_count,
        :buy_amount_usd,
        :sell_amount_usd,
        :total_order_amount_usd,
        :failed_order_count,
        :rejected_order_count,
        :blocked_order_count,
        :realized_pnl_usd,
        :unrealized_pnl_usd,
        :daily_pnl_usd,
        :daily_pnl_pct,
        :max_position_weight,
        :max_sector_weight,
        :cash_weight,
        :data_status,
        now(),
        now()
    )
    ON CONFLICT (trade_date, policy_id, account_id) DO UPDATE SET
        buy_order_count = EXCLUDED.buy_order_count,
        sell_order_count = EXCLUDED.sell_order_count,
        total_order_count = EXCLUDED.total_order_count,
        new_buy_count = EXCLUDED.new_buy_count,
        buy_amount_usd = EXCLUDED.buy_amount_usd,
        sell_amount_usd = EXCLUDED.sell_amount_usd,
        total_order_amount_usd = EXCLUDED.total_order_amount_usd,
        failed_order_count = EXCLUDED.failed_order_count,
        rejected_order_count = EXCLUDED.rejected_order_count,
        blocked_order_count = EXCLUDED.blocked_order_count,
        realized_pnl_usd = EXCLUDED.realized_pnl_usd,
        unrealized_pnl_usd = EXCLUDED.unrealized_pnl_usd,
        daily_pnl_usd = EXCLUDED.daily_pnl_usd,
        daily_pnl_pct = EXCLUDED.daily_pnl_pct,
        max_position_weight = EXCLUDED.max_position_weight,
        max_sector_weight = EXCLUDED.max_sector_weight,
        cash_weight = EXCLUDED.cash_weight,
        data_status = EXCLUDED.data_status,
        updated_at = now()
    """
)

INSERT_US_LIVE_ORDER_BLOCK_LOG_SQL = text(
    """
    INSERT INTO risk.us_stock_live_order_block_log (
        block_id,
        trade_date,
        policy_id,
        account_id,
        symbol,
        side,
        candidate_source,
        rank_no,
        recommend_grade,
        total_score,
        requested_order_amount_usd,
        requested_qty,
        requested_order_type,
        block_reason_code,
        block_reason_detail,
        check_stage,
        severity,
        created_at
    ) VALUES (
        :block_id,
        :trade_date,
        :policy_id,
        :account_id,
        :symbol,
        :side,
        :candidate_source,
        :rank_no,
        :recommend_grade,
        :total_score,
        :requested_order_amount_usd,
        :requested_qty,
        :requested_order_type,
        :block_reason_code,
        :block_reason_detail,
        :check_stage,
        :severity,
        now()
    )
    ON CONFLICT (block_id) DO NOTHING
    """
)

UPSERT_LABEL_SQL = text(
    """
    INSERT INTO label.us_stock_label_daily (
        ticker,
        market,
        trade_date,
        price_column_used,
        future_ret_5d,
        future_ret_20d,
        future_ret_60d,
        future_ret_20d_rank_pct,
        future_ret_60d_rank_pct,
        label_positive_20d,
        label_positive_60d,
        label_top20_20d,
        label_top20_60d,
        source,
        label_created_at,
        created_at,
        updated_at
    ) VALUES (
        :ticker,
        :market,
        :trade_date,
        :price_column_used,
        :future_ret_5d,
        :future_ret_20d,
        :future_ret_60d,
        :future_ret_20d_rank_pct,
        :future_ret_60d_rank_pct,
        :label_positive_20d,
        :label_positive_60d,
        :label_top20_20d,
        :label_top20_60d,
        :source,
        :label_created_at,
        now(),
        now()
    )
    ON CONFLICT (ticker, trade_date, source) DO UPDATE SET
        market = EXCLUDED.market,
        price_column_used = EXCLUDED.price_column_used,
        future_ret_5d = EXCLUDED.future_ret_5d,
        future_ret_20d = EXCLUDED.future_ret_20d,
        future_ret_60d = EXCLUDED.future_ret_60d,
        future_ret_20d_rank_pct = EXCLUDED.future_ret_20d_rank_pct,
        future_ret_60d_rank_pct = EXCLUDED.future_ret_60d_rank_pct,
        label_positive_20d = EXCLUDED.label_positive_20d,
        label_positive_60d = EXCLUDED.label_positive_60d,
        label_top20_20d = EXCLUDED.label_top20_20d,
        label_top20_60d = EXCLUDED.label_top20_60d,
        label_created_at = EXCLUDED.label_created_at,
        updated_at = now()
    """
)

UPSERT_META_US_UNIVERSE_SQL = text(
    """
    INSERT INTO meta.us_stock_universe (
        symbol,
        company_name,
        market,
        sector,
        industry,
        universe_group,
        is_active,
        is_etf,
        is_leveraged,
        is_inverse,
        source,
        market_cap,
        avg_volume,
        currency,
        country,
        exchange,
        first_included_date,
        last_checked_date,
        exclude_reason,
        feature_quality_score,
        created_at,
        updated_at
    ) VALUES (
        :symbol,
        :company_name,
        :market,
        :sector,
        :industry,
        :universe_group,
        :is_active,
        :is_etf,
        :is_leveraged,
        :is_inverse,
        :source,
        :market_cap,
        :avg_volume,
        :currency,
        :country,
        :exchange,
        :first_included_date,
        :last_checked_date,
        :exclude_reason,
        :feature_quality_score,
        now(),
        now()
    )
    ON CONFLICT (symbol) DO UPDATE SET
        company_name = EXCLUDED.company_name,
        market = EXCLUDED.market,
        sector = EXCLUDED.sector,
        industry = EXCLUDED.industry,
        universe_group = EXCLUDED.universe_group,
        is_active = EXCLUDED.is_active,
        is_etf = EXCLUDED.is_etf,
        is_leveraged = EXCLUDED.is_leveraged,
        is_inverse = EXCLUDED.is_inverse,
        source = EXCLUDED.source,
        market_cap = EXCLUDED.market_cap,
        avg_volume = EXCLUDED.avg_volume,
        currency = EXCLUDED.currency,
        country = EXCLUDED.country,
        exchange = EXCLUDED.exchange,
        first_included_date = COALESCE(meta.us_stock_universe.first_included_date, EXCLUDED.first_included_date),
        last_checked_date = EXCLUDED.last_checked_date,
        exclude_reason = EXCLUDED.exclude_reason,
        feature_quality_score = EXCLUDED.feature_quality_score,
        updated_at = now()
    """
)

UPSERT_US_RANK_SQL = text(
    """
    INSERT INTO recommend.us_stock_rank_daily (
        trade_date,
        symbol,
        rank_no,
        recommend_grade,
        total_score,
        momentum_score,
        relative_strength_score,
        fundamental_score,
        growth_score,
        valuation_score,
        risk_score,
        feature_quality_score,
        universe_group,
        company_name,
        sector,
        industry,
        market_cap,
        avg_volume,
        is_etf,
        is_active,
        data_status,
        exclude_reason,
        reason_summary,
        score_detail_json,
        source,
        created_at,
        updated_at
    ) VALUES (
        :trade_date,
        :symbol,
        :rank_no,
        :recommend_grade,
        :total_score,
        :momentum_score,
        :relative_strength_score,
        :fundamental_score,
        :growth_score,
        :valuation_score,
        :risk_score,
        :feature_quality_score,
        :universe_group,
        :company_name,
        :sector,
        :industry,
        :market_cap,
        :avg_volume,
        :is_etf,
        :is_active,
        :data_status,
        :exclude_reason,
        :reason_summary,
        CAST(:score_detail_json AS jsonb),
        :source,
        now(),
        now()
    )
    ON CONFLICT (trade_date, symbol, source) DO UPDATE SET
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        momentum_score = EXCLUDED.momentum_score,
        relative_strength_score = EXCLUDED.relative_strength_score,
        fundamental_score = EXCLUDED.fundamental_score,
        growth_score = EXCLUDED.growth_score,
        valuation_score = EXCLUDED.valuation_score,
        risk_score = EXCLUDED.risk_score,
        feature_quality_score = EXCLUDED.feature_quality_score,
        universe_group = EXCLUDED.universe_group,
        company_name = EXCLUDED.company_name,
        sector = EXCLUDED.sector,
        industry = EXCLUDED.industry,
        market_cap = EXCLUDED.market_cap,
        avg_volume = EXCLUDED.avg_volume,
        is_etf = EXCLUDED.is_etf,
        is_active = EXCLUDED.is_active,
        data_status = EXCLUDED.data_status,
        exclude_reason = EXCLUDED.exclude_reason,
        reason_summary = EXCLUDED.reason_summary,
        score_detail_json = EXCLUDED.score_detail_json,
        source = EXCLUDED.source,
        updated_at = now()
    """
)

DELETE_US_RANK_ROWS_SQL = text(
    """
    DELETE FROM recommend.us_stock_rank_daily
    WHERE trade_date = :trade_date
      AND source = :source
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
        forward_eps,
        roe,
        roa,
        shares_outstanding,
        market_cap,
        per,
        forward_pe,
        peg_ratio,
        pbr,
        psr,
        ev_ebitda,
        debt_to_equity,
        current_ratio,
        dividend_yield,
        analyst_target_price,
        analyst_recommendation,
        analyst_count,
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
        :forward_eps,
        :roe,
        :roa,
        :shares_outstanding,
        :market_cap,
        :per,
        :forward_pe,
        :peg_ratio,
        :pbr,
        :psr,
        :ev_ebitda,
        :debt_to_equity,
        :current_ratio,
        :dividend_yield,
        :analyst_target_price,
        :analyst_recommendation,
        :analyst_count,
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
        forward_eps = EXCLUDED.forward_eps,
        roe = EXCLUDED.roe,
        roa = EXCLUDED.roa,
        shares_outstanding = EXCLUDED.shares_outstanding,
        market_cap = EXCLUDED.market_cap,
        per = EXCLUDED.per,
        forward_pe = EXCLUDED.forward_pe,
        peg_ratio = EXCLUDED.peg_ratio,
        pbr = EXCLUDED.pbr,
        psr = EXCLUDED.psr,
        ev_ebitda = EXCLUDED.ev_ebitda,
        debt_to_equity = EXCLUDED.debt_to_equity,
        current_ratio = EXCLUDED.current_ratio,
        dividend_yield = EXCLUDED.dividend_yield,
        analyst_target_price = EXCLUDED.analyst_target_price,
        analyst_recommendation = EXCLUDED.analyst_recommendation,
        analyst_count = EXCLUDED.analyst_count,
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

UPSERT_US_LIVE_ORDER_APPROVAL_SQL = text(
    """
    INSERT INTO risk.us_stock_live_order_approval (
        approval_id,
        trade_date,
        policy_id,
        account_id,
        symbol,
        side,
        candidate_source,
        strategy_name,
        rank_no,
        recommend_grade,
        total_score,
        requested_order_type,
        requested_limit_price,
        requested_qty,
        requested_order_amount_usd,
        precheck_decision,
        precheck_reason_codes,
        precheck_summary,
        approval_status,
        requested_by,
        requested_at,
        approved_by,
        approved_at,
        approval_reason,
        rejected_by,
        rejected_at,
        reject_reason,
        expired_at,
        expires_at,
        created_at,
        updated_at
    ) VALUES (
        :approval_id,
        :trade_date,
        :policy_id,
        :account_id,
        :symbol,
        :side,
        :candidate_source,
        :strategy_name,
        :rank_no,
        :recommend_grade,
        :total_score,
        :requested_order_type,
        :requested_limit_price,
        :requested_qty,
        :requested_order_amount_usd,
        :precheck_decision,
        :precheck_reason_codes,
        :precheck_summary,
        :approval_status,
        :requested_by,
        COALESCE(:requested_at, now()),
        :approved_by,
        :approved_at,
        :approval_reason,
        :rejected_by,
        :rejected_at,
        :reject_reason,
        :expired_at,
        :expires_at,
        now(),
        now()
    )
    ON CONFLICT (approval_id) DO UPDATE SET
        trade_date = EXCLUDED.trade_date,
        policy_id = EXCLUDED.policy_id,
        account_id = EXCLUDED.account_id,
        symbol = EXCLUDED.symbol,
        side = EXCLUDED.side,
        candidate_source = EXCLUDED.candidate_source,
        strategy_name = EXCLUDED.strategy_name,
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        requested_order_type = EXCLUDED.requested_order_type,
        requested_limit_price = EXCLUDED.requested_limit_price,
        requested_qty = EXCLUDED.requested_qty,
        requested_order_amount_usd = EXCLUDED.requested_order_amount_usd,
        precheck_decision = EXCLUDED.precheck_decision,
        precheck_reason_codes = EXCLUDED.precheck_reason_codes,
        precheck_summary = EXCLUDED.precheck_summary,
        approval_status = EXCLUDED.approval_status,
        requested_by = EXCLUDED.requested_by,
        requested_at = EXCLUDED.requested_at,
        approved_by = EXCLUDED.approved_by,
        approved_at = EXCLUDED.approved_at,
        approval_reason = EXCLUDED.approval_reason,
        rejected_by = EXCLUDED.rejected_by,
        rejected_at = EXCLUDED.rejected_at,
        reject_reason = EXCLUDED.reject_reason,
        expired_at = EXCLUDED.expired_at,
        expires_at = EXCLUDED.expires_at,
        updated_at = now()
    """
)

INSERT_US_LIVE_ORDER_APPROVAL_EVENT_LOG_SQL = text(
    """
    INSERT INTO risk.us_stock_live_order_approval_event_log (
        event_id,
        approval_id,
        event_type,
        before_status,
        after_status,
        reason_code,
        reason_detail,
        performed_by,
        created_at
    ) VALUES (
        :event_id,
        :approval_id,
        :event_type,
        :before_status,
        :after_status,
        :reason_code,
        :reason_detail,
        :performed_by,
        now()
    )
    """
)

UPSERT_US_MICRO_ORDER_REQUEST_SQL = text(
    """
    INSERT INTO live.us_stock_micro_order_request (
        micro_order_id,
        approval_id,
        policy_id,
        account_id,
        trade_date,
        symbol,
        side,
        order_type,
        limit_price,
        order_qty,
        order_amount_usd,
        candidate_source,
        strategy_name,
        rank_no,
        recommend_grade,
        total_score,
        precheck_decision,
        precheck_reason_codes,
        precheck_summary,
        execution_mode,
        broker_name,
        request_status,
        request_payload,
        response_payload,
        broker_order_id,
        last_broker_status,
        last_sync_at,
        filled_qty,
        remaining_qty,
        avg_filled_price,
        filled_amount_usd,
        sync_status,
        sync_error,
        reject_reason_code,
        reject_reason_detail,
        created_by,
        created_at,
        updated_at
    ) VALUES (
        :micro_order_id,
        :approval_id,
        :policy_id,
        :account_id,
        :trade_date,
        :symbol,
        :side,
        :order_type,
        :limit_price,
        :order_qty,
        :order_amount_usd,
        :candidate_source,
        :strategy_name,
        :rank_no,
        :recommend_grade,
        :total_score,
        :precheck_decision,
        :precheck_reason_codes,
        :precheck_summary,
        :execution_mode,
        :broker_name,
        :request_status,
        :request_payload,
        :response_payload,
        :broker_order_id,
        :last_broker_status,
        :last_sync_at,
        :filled_qty,
        :remaining_qty,
        :avg_filled_price,
        :filled_amount_usd,
        :sync_status,
        :sync_error,
        :reject_reason_code,
        :reject_reason_detail,
        :created_by,
        COALESCE(:created_at, now()),
        COALESCE(:updated_at, now())
    )
    ON CONFLICT (micro_order_id) DO UPDATE SET
        approval_id = EXCLUDED.approval_id,
        policy_id = EXCLUDED.policy_id,
        account_id = EXCLUDED.account_id,
        trade_date = EXCLUDED.trade_date,
        symbol = EXCLUDED.symbol,
        side = EXCLUDED.side,
        order_type = EXCLUDED.order_type,
        limit_price = EXCLUDED.limit_price,
        order_qty = EXCLUDED.order_qty,
        order_amount_usd = EXCLUDED.order_amount_usd,
        candidate_source = EXCLUDED.candidate_source,
        strategy_name = EXCLUDED.strategy_name,
        rank_no = EXCLUDED.rank_no,
        recommend_grade = EXCLUDED.recommend_grade,
        total_score = EXCLUDED.total_score,
        precheck_decision = EXCLUDED.precheck_decision,
        precheck_reason_codes = EXCLUDED.precheck_reason_codes,
        precheck_summary = EXCLUDED.precheck_summary,
        execution_mode = EXCLUDED.execution_mode,
        broker_name = EXCLUDED.broker_name,
        request_status = EXCLUDED.request_status,
        request_payload = EXCLUDED.request_payload,
        response_payload = EXCLUDED.response_payload,
        broker_order_id = EXCLUDED.broker_order_id,
        last_broker_status = EXCLUDED.last_broker_status,
        last_sync_at = EXCLUDED.last_sync_at,
        filled_qty = EXCLUDED.filled_qty,
        remaining_qty = EXCLUDED.remaining_qty,
        avg_filled_price = EXCLUDED.avg_filled_price,
        filled_amount_usd = EXCLUDED.filled_amount_usd,
        sync_status = EXCLUDED.sync_status,
        sync_error = EXCLUDED.sync_error,
        reject_reason_code = EXCLUDED.reject_reason_code,
        reject_reason_detail = EXCLUDED.reject_reason_detail,
        created_by = EXCLUDED.created_by,
        updated_at = now()
    """
)

UPSERT_US_MICRO_ORDER_FILL_SQL = text(
    """
    INSERT INTO live.us_stock_micro_order_fill (
        micro_fill_id,
        micro_order_id,
        broker_order_id,
        broker_fill_id,
        account_id,
        symbol,
        side,
        filled_qty,
        filled_price,
        filled_amount_usd,
        commission_usd,
        fee_usd,
        fill_time,
        fill_date,
        liquidity_flag,
        raw_fill_payload,
        created_at,
        updated_at
    ) VALUES (
        :micro_fill_id,
        :micro_order_id,
        :broker_order_id,
        :broker_fill_id,
        :account_id,
        :symbol,
        :side,
        :filled_qty,
        :filled_price,
        :filled_amount_usd,
        :commission_usd,
        :fee_usd,
        :fill_time,
        :fill_date,
        :liquidity_flag,
        :raw_fill_payload,
        COALESCE(:created_at, now()),
        COALESCE(:updated_at, now())
    )
    ON CONFLICT (micro_fill_id) DO UPDATE SET
        micro_order_id = EXCLUDED.micro_order_id,
        broker_order_id = EXCLUDED.broker_order_id,
        broker_fill_id = EXCLUDED.broker_fill_id,
        account_id = EXCLUDED.account_id,
        symbol = EXCLUDED.symbol,
        side = EXCLUDED.side,
        filled_qty = EXCLUDED.filled_qty,
        filled_price = EXCLUDED.filled_price,
        filled_amount_usd = EXCLUDED.filled_amount_usd,
        commission_usd = EXCLUDED.commission_usd,
        fee_usd = EXCLUDED.fee_usd,
        fill_time = EXCLUDED.fill_time,
        fill_date = EXCLUDED.fill_date,
        liquidity_flag = EXCLUDED.liquidity_flag,
        raw_fill_payload = EXCLUDED.raw_fill_payload,
        updated_at = now()
    """
)

UPSERT_US_MICRO_RECONCILIATION_RESULT_SQL = text(
    """
    INSERT INTO live.us_stock_micro_reconciliation_result (
        recon_id,
        recon_run_id,
        recon_date,
        account_id,
        execution_mode,
        recon_type,
        symbol,
        micro_order_id,
        broker_order_id,
        internal_qty,
        broker_qty,
        qty_diff,
        internal_amount_usd,
        broker_amount_usd,
        amount_diff_usd,
        internal_cash_usd,
        broker_cash_usd,
        cash_diff_usd,
        internal_status,
        broker_status,
        recon_status,
        severity,
        reason_code,
        reason_detail,
        raw_internal_payload,
        raw_broker_payload,
        created_at
    ) VALUES (
        :recon_id,
        :recon_run_id,
        :recon_date,
        :account_id,
        :execution_mode,
        :recon_type,
        :symbol,
        :micro_order_id,
        :broker_order_id,
        :internal_qty,
        :broker_qty,
        :qty_diff,
        :internal_amount_usd,
        :broker_amount_usd,
        :amount_diff_usd,
        :internal_cash_usd,
        :broker_cash_usd,
        :cash_diff_usd,
        :internal_status,
        :broker_status,
        :recon_status,
        :severity,
        :reason_code,
        :reason_detail,
        :raw_internal_payload,
        :raw_broker_payload,
        COALESCE(:created_at, now())
    )
    ON CONFLICT (recon_id) DO UPDATE SET
        recon_run_id = EXCLUDED.recon_run_id,
        recon_date = EXCLUDED.recon_date,
        account_id = EXCLUDED.account_id,
        execution_mode = EXCLUDED.execution_mode,
        recon_type = EXCLUDED.recon_type,
        symbol = EXCLUDED.symbol,
        micro_order_id = EXCLUDED.micro_order_id,
        broker_order_id = EXCLUDED.broker_order_id,
        internal_qty = EXCLUDED.internal_qty,
        broker_qty = EXCLUDED.broker_qty,
        qty_diff = EXCLUDED.qty_diff,
        internal_amount_usd = EXCLUDED.internal_amount_usd,
        broker_amount_usd = EXCLUDED.broker_amount_usd,
        amount_diff_usd = EXCLUDED.amount_diff_usd,
        internal_cash_usd = EXCLUDED.internal_cash_usd,
        broker_cash_usd = EXCLUDED.broker_cash_usd,
        cash_diff_usd = EXCLUDED.cash_diff_usd,
        internal_status = EXCLUDED.internal_status,
        broker_status = EXCLUDED.broker_status,
        recon_status = EXCLUDED.recon_status,
        severity = EXCLUDED.severity,
        reason_code = EXCLUDED.reason_code,
        reason_detail = EXCLUDED.reason_detail,
        raw_internal_payload = EXCLUDED.raw_internal_payload,
        raw_broker_payload = EXCLUDED.raw_broker_payload
    """
)

INSERT_US_MICRO_RECONCILIATION_EVENT_LOG_SQL = text(
    """
    INSERT INTO live.us_stock_micro_reconciliation_event_log (
        event_id,
        recon_run_id,
        event_type,
        account_id,
        execution_mode,
        message,
        severity,
        created_at
    ) VALUES (
        :event_id,
        :recon_run_id,
        :event_type,
        :account_id,
        :execution_mode,
        :message,
        :severity,
        COALESCE(:created_at, now())
    )
    """
)

INSERT_US_MICRO_ORDER_EVENT_LOG_SQL = text(
    """
    INSERT INTO live.us_stock_micro_order_event_log (
        event_id,
        micro_order_id,
        event_type,
        before_status,
        after_status,
        event_source,
        reason_code,
        reason_detail,
        request_payload,
        response_payload,
        created_by,
        created_at
    ) VALUES (
        :event_id,
        :micro_order_id,
        :event_type,
        :before_status,
        :after_status,
        :event_source,
        :reason_code,
        :reason_detail,
        :request_payload,
        :response_payload,
        :created_by,
        COALESCE(:created_at, now())
    )
    """
)

UPSERT_FEATURE_SQL = text(
    """
    INSERT INTO feature.us_stock_feature_daily (
        feature_date,
        ticker,
        ret_1d,
        ret_3d,
        ret_5d,
        ret_10d,
        ret_20d,
        ret_60d,
        ret_252d,
        volume_avg_20d,
        volume_ratio_20d,
        volatility_20d,
        ma_20,
        ma_60,
        ma_200,
        price_vs_ma200,
        price_above_ma20_flag,
        price_above_ma60_flag,
        rsi_14,
        atr_14_norm,
        bb_position,
        high_52w_ratio,
        created_at,
        updated_at
    ) VALUES (
        :feature_date,
        :ticker,
        :ret_1d,
        :ret_3d,
        :ret_5d,
        :ret_10d,
        :ret_20d,
        :ret_60d,
        :ret_252d,
        :volume_avg_20d,
        :volume_ratio_20d,
        :volatility_20d,
        :ma_20,
        :ma_60,
        :ma_200,
        :price_vs_ma200,
        :price_above_ma20_flag,
        :price_above_ma60_flag,
        :rsi_14,
        :atr_14_norm,
        :bb_position,
        :high_52w_ratio,
        now(),
        now()
    )
    ON CONFLICT (feature_date, ticker) DO UPDATE SET
        ret_1d = EXCLUDED.ret_1d,
        ret_3d = EXCLUDED.ret_3d,
        ret_5d = EXCLUDED.ret_5d,
        ret_10d = EXCLUDED.ret_10d,
        ret_20d = EXCLUDED.ret_20d,
        ret_60d = EXCLUDED.ret_60d,
        ret_252d = EXCLUDED.ret_252d,
        volume_avg_20d = EXCLUDED.volume_avg_20d,
        volume_ratio_20d = EXCLUDED.volume_ratio_20d,
        volatility_20d = EXCLUDED.volatility_20d,
        ma_20 = EXCLUDED.ma_20,
        ma_60 = EXCLUDED.ma_60,
        ma_200 = EXCLUDED.ma_200,
        price_vs_ma200 = EXCLUDED.price_vs_ma200,
        price_above_ma20_flag = EXCLUDED.price_above_ma20_flag,
        price_above_ma60_flag = EXCLUDED.price_above_ma60_flag,
        rsi_14 = EXCLUDED.rsi_14,
        atr_14_norm = EXCLUDED.atr_14_norm,
        bb_position = EXCLUDED.bb_position,
        high_52w_ratio = EXCLUDED.high_52w_ratio,
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


def fetch_latest_price_trade_date(tickers: list[str], *, as_of_date: date, data_source: str) -> date | None:
    if not tickers:
        return None
    engine = get_us_engine()
    with engine.connect() as conn:
        row = conn.execute(
            READ_LATEST_PRICE_TRADE_DATE_SQL,
            {"tickers": tickers, "as_of_date": as_of_date, "data_source": data_source},
        ).mappings().one()
    return row.get("latest_trade_date")


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


def fetch_price_history_for_tickers(tickers: list[str], *, end_date: date | None = None) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_PRICE_HISTORY_FOR_TICKERS_SQL,
            {"tickers": tickers, "end_date": end_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_financial_statement_rows(
    tickers: list[str], *, period_types: list[str], min_fiscal_date: date
) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_FINANCIAL_STATEMENT_ROWS_SQL,
            {"tickers": tickers, "period_types": period_types, "min_fiscal_date": min_fiscal_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_financial_metric_rows(
    tickers: list[str], *, period_types: list[str], min_fiscal_date: date
) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_FINANCIAL_METRIC_ROWS_SQL,
            {"tickers": tickers, "period_types": period_types, "min_fiscal_date": min_fiscal_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_daily_feature_rows(tickers: list[str]) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_DAILY_FEATURE_ROWS_SQL, {"tickers": tickers}).mappings().all()
    return [dict(row) for row in rows]


def fetch_relative_strength_feature_rows(tickers: list[str]) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_RELATIVE_STRENGTH_ROWS_SQL, {"tickers": tickers}).mappings().all()
    return [dict(row) for row in rows]


def fetch_financial_feature_rows(tickers: list[str]) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_FINANCIAL_FEATURE_ROWS_SQL, {"tickers": tickers}).mappings().all()
    return [dict(row) for row in rows]


def fetch_label_rows(tickers: list[str]) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_LABEL_ROWS_SQL, {"tickers": tickers}).mappings().all()
    return [dict(row) for row in rows]


def fetch_meta_us_universe_rows() -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_META_US_UNIVERSE_SQL).mappings().all()
    return [dict(row) for row in rows]


def fetch_latest_daily_feature_snapshots(tickers: list[str], *, trade_date: date) -> dict[str, dict[str, object]]:
    if not tickers or not relation_exists("feature.us_stock_feature_daily"):
        return {}
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_LATEST_DAILY_FEATURE_SNAPSHOTS_SQL,
            {"tickers": tickers, "trade_date": trade_date},
        ).mappings().all()
    return {str(row["ticker"]).upper(): dict(row) for row in rows}


def fetch_latest_relative_strength_snapshots(tickers: list[str], *, trade_date: date) -> dict[str, dict[str, object]]:
    if not tickers or not relation_exists("feature.us_stock_relative_strength_daily"):
        return {}
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_LATEST_RELATIVE_STRENGTH_SNAPSHOTS_SQL,
            {"tickers": tickers, "trade_date": trade_date},
        ).mappings().all()
    return {str(row["ticker"]).upper(): dict(row) for row in rows}


def fetch_latest_financial_feature_snapshots(tickers: list[str], *, trade_date: date) -> dict[str, dict[str, object]]:
    if not tickers or not relation_exists("feature.us_stock_financial_feature"):
        return {}
    ensure_us_financial_feature_reported_date_column()
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_LATEST_FINANCIAL_FEATURE_SNAPSHOTS_SQL,
            {"tickers": tickers, "trade_date": trade_date},
        ).mappings().all()
    return {str(row["ticker"]).upper(): dict(row) for row in rows}


def fetch_latest_macro_snapshot(*, trade_date: date) -> dict[str, object] | None:
    if not relation_exists("feature.us_macro_daily"):
        return None
    engine = get_us_engine()
    with engine.connect() as conn:
        row = conn.execute(READ_LATEST_MACRO_SNAPSHOT_SQL, {"trade_date": trade_date}).mappings().first()
    return dict(row) if row else None


def fetch_us_rank_rows(symbols: list[str]) -> list[dict[str, object]]:
    if not symbols:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_US_RANK_ROWS_SQL, {"symbols": symbols}).mappings().all()
    return [dict(row) for row in rows]


def fetch_rank_rows_between(*, start_date: date, end_date: date, source: str) -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_RANK_ROWS_BETWEEN_SQL,
            {"start_date": start_date, "end_date": end_date, "source": source},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_rank_component_rows_between(*, start_date: date, end_date: date, source: str) -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_RANK_COMPONENT_ROWS_BETWEEN_SQL,
            {"start_date": start_date, "end_date": end_date, "source": source},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_price_rows_for_tickers_between(*, tickers: list[str], start_date: date, end_date: date) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_PRICE_ROWS_FOR_TICKERS_BETWEEN_SQL,
            {"tickers": tickers, "start_date": start_date, "end_date": end_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_mixed_price_rows_for_tickers_between(*, tickers: list[str], start_date: date, end_date: date) -> list[dict[str, object]]:
    if not tickers:
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_MIXED_PRICE_ROWS_FOR_TICKERS_BETWEEN_SQL,
            {"tickers": tickers, "start_date": start_date, "end_date": end_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_market_regime_rows_between(*, start_date: date, end_date: date) -> list[dict[str, object]]:
    if not relation_exists("research.us_market_regime_daily"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_MARKET_REGIME_ROWS_BETWEEN_SQL,
            {"start_date": start_date, "end_date": end_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_weight_config_rows() -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(READ_US_WEIGHT_CONFIG_ROWS_SQL).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_weight_experiment_summary_rows(*, experiment_id: str) -> list[dict[str, object]]:
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_WEIGHT_EXPERIMENT_SUMMARY_ROWS_SQL,
            {"experiment_id": experiment_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_forward_test_rows(
    *,
    forward_test_id: str,
    trade_date: date | None = None,
    strategy_name: str | None = None,
    holding_days: int | None = None,
    status: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("research.us_stock_rank_forward_test"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_FORWARD_TEST_ROWS_SQL,
            {
                "forward_test_id": forward_test_id,
                "trade_date": trade_date,
                "strategy_name": strategy_name,
                "holding_days": holding_days,
                "status": status,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_forward_test_summary_rows(
    *,
    forward_test_id: str,
    trade_date: date | None = None,
    strategy_name: str | None = None,
    holding_days: int | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("research.us_stock_rank_forward_test_summary"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_FORWARD_TEST_SUMMARY_ROWS_SQL,
            {
                "forward_test_id": forward_test_id,
                "trade_date": trade_date,
                "strategy_name": strategy_name,
                "holding_days": holding_days,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_paper_account_rows(*, account_id: str | None = None) -> list[dict[str, object]]:
    if not relation_exists("paper.us_stock_paper_account"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_PAPER_ACCOUNT_ROWS_SQL,
            {"account_id": account_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_paper_order_rows(
    *,
    paper_order_id: str | None = None,
    account_id: str | None = None,
    trade_date: date | None = None,
    side: str | None = None,
    status: str | None = None,
    strategy_name: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("paper.us_stock_paper_order"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_PAPER_ORDER_ROWS_SQL,
            {
                "paper_order_id": paper_order_id,
                "account_id": account_id,
                "trade_date": trade_date,
                "side": side,
                "status": status,
                "strategy_name": strategy_name,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_paper_fill_rows(
    *,
    paper_order_id: str | None = None,
    account_id: str | None = None,
    trade_date: date | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("paper.us_stock_paper_fill"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_PAPER_FILL_ROWS_SQL,
            {
                "paper_order_id": paper_order_id,
                "account_id": account_id,
                "trade_date": trade_date,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_paper_position_rows(*, account_id: str | None = None, status: str | None = None) -> list[dict[str, object]]:
    if not relation_exists("paper.us_stock_paper_position"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_PAPER_POSITION_ROWS_SQL,
            {"account_id": account_id, "status": status},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_paper_account_snapshot_rows(
    *,
    account_id: str | None = None,
    snapshot_date: date | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("paper.us_stock_paper_account_snapshot"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_PAPER_ACCOUNT_SNAPSHOT_ROWS_SQL,
            {"account_id": account_id, "snapshot_date": snapshot_date},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_kill_switch_rows(*, kill_switch_id: str | None = None, scope: str | None = None) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_kill_switch"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_KILL_SWITCH_ROWS_SQL,
            {"kill_switch_id": kill_switch_id, "scope": scope},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_kill_switch_event_log_rows(
    *,
    kill_switch_id: str | None = None,
    scope: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_kill_switch_event_log"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_KILL_SWITCH_EVENT_LOG_ROWS_SQL,
            {"kill_switch_id": kill_switch_id, "scope": scope},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_daily_risk_usage_rows(
    *,
    trade_date: date | None = None,
    policy_id: str | None = None,
    account_id: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_daily_risk_usage"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_DAILY_RISK_USAGE_ROWS_SQL,
            {"trade_date": trade_date, "policy_id": policy_id, "account_id": account_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_order_block_log_rows(
    *,
    trade_date: date | None = None,
    policy_id: str | None = None,
    account_id: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_order_block_log"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_ORDER_BLOCK_LOG_ROWS_SQL,
            {"trade_date": trade_date, "policy_id": policy_id, "account_id": account_id, "symbol": symbol},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_order_approval_rows(
    *,
    approval_id: str | None = None,
    trade_date: date | None = None,
    account_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_order_approval"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_ORDER_APPROVAL_ROWS_SQL,
            {
                "approval_id": approval_id,
                "trade_date": trade_date,
                "account_id": account_id,
                "status": status,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_live_order_approval_event_log_rows(*, approval_id: str | None = None) -> list[dict[str, object]]:
    if not relation_exists("risk.us_stock_live_order_approval_event_log"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_LIVE_ORDER_APPROVAL_EVENT_LOG_ROWS_SQL,
            {"approval_id": approval_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_micro_order_request_rows(
    *,
    micro_order_id: str | None = None,
    approval_id: str | None = None,
    trade_date: date | None = None,
    account_id: str | None = None,
    status: str | None = None,
    execution_mode: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("live.us_stock_micro_order_request"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_MICRO_ORDER_REQUEST_ROWS_SQL,
            {
                "micro_order_id": micro_order_id,
                "approval_id": approval_id,
                "trade_date": trade_date,
                "account_id": account_id,
                "status": status,
                "execution_mode": execution_mode,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_micro_order_event_log_rows(*, micro_order_id: str | None = None) -> list[dict[str, object]]:
    if not relation_exists("live.us_stock_micro_order_event_log"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_MICRO_ORDER_EVENT_LOG_ROWS_SQL,
            {"micro_order_id": micro_order_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_micro_order_fill_rows(
    *,
    micro_order_id: str | None = None,
    broker_order_id: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("live.us_stock_micro_order_fill"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_MICRO_ORDER_FILL_ROWS_SQL,
            {
                "micro_order_id": micro_order_id,
                "broker_order_id": broker_order_id,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_micro_reconciliation_result_rows(
    *,
    recon_run_id: str | None = None,
    recon_date: date | None = None,
    account_id: str | None = None,
    recon_type: str | None = None,
    severity: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("live.us_stock_micro_reconciliation_result"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_MICRO_RECONCILIATION_RESULT_ROWS_SQL,
            {
                "recon_run_id": recon_run_id,
                "recon_date": recon_date,
                "account_id": account_id,
                "recon_type": recon_type,
                "severity": severity,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_us_micro_reconciliation_event_log_rows(
    *,
    recon_run_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, object]]:
    if not relation_exists("live.us_stock_micro_reconciliation_event_log"):
        return []
    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            READ_US_MICRO_RECONCILIATION_EVENT_LOG_ROWS_SQL,
            {
                "recon_run_id": recon_run_id,
                "event_type": event_type,
            },
        ).mappings().all()
    return [dict(row) for row in rows]


def relation_exists(relation_name: str) -> bool:
    engine = get_us_engine()
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT to_regclass(:relation_name)"),
            {"relation_name": relation_name},
        ).scalar()
    return value is not None


def ensure_us_financial_feature_reported_date_column() -> None:
    if not relation_exists("feature.us_stock_financial_feature"):
        return
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE feature.us_stock_financial_feature
                ADD COLUMN IF NOT EXISTS reported_date DATE
                """
            )
        )


def get_active_us_stock_universe(
    *,
    min_market_cap: float,
    min_avg_volume: float,
    min_feature_quality_score: float,
    include_etf: bool,
    exclude_leveraged: bool,
    exclude_inverse: bool,
) -> list[dict[str, object]]:
    has_price_table = relation_exists("market.us_stock_daily_price")
    has_daily_feature = relation_exists("feature.us_stock_feature_daily")
    has_relative_strength = relation_exists("feature.us_stock_relative_strength_daily")
    has_financial_feature = relation_exists("feature.us_stock_financial_feature")

    if not has_price_table:
        return []

    latest_financial_cte = ""
    latest_financial_join = ""
    daily_feature_cte = ""
    daily_feature_join = ""
    rs_feature_cte = ""
    rs_feature_join = ""
    financial_feature_cte = ""
    financial_feature_join = ""
    score_terms: list[str] = []

    if has_financial_feature:
        latest_financial_cte = """
    ,
    latest_financial AS (
        SELECT DISTINCT ON (ticker)
            ticker,
            market_cap,
            fiscal_date
        FROM feature.us_stock_financial_feature
        ORDER BY ticker, fiscal_date DESC
    )
"""
        latest_financial_join = """
    LEFT JOIN latest_financial lf
      ON lf.ticker = u.symbol
"""
        financial_feature_cte = """
    ,
    financial_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_financial_feature
    )
"""
        financial_feature_join = """
    LEFT JOIN financial_feature_presence ffp
      ON ffp.ticker = u.symbol
"""
        score_terms.append("(CASE WHEN ffp.ticker IS NOT NULL THEN 30 ELSE 0 END)")

    if has_daily_feature:
        daily_feature_cte = """
    ,
    daily_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_feature_daily
    )
"""
        daily_feature_join = """
    LEFT JOIN daily_feature_presence dfp
      ON dfp.ticker = u.symbol
"""
        score_terms.append("(CASE WHEN dfp.ticker IS NOT NULL THEN 40 ELSE 0 END)")

    if has_relative_strength:
        rs_feature_cte = """
    ,
    rs_feature_presence AS (
        SELECT DISTINCT ticker
        FROM feature.us_stock_relative_strength_daily
    )
"""
        rs_feature_join = """
    LEFT JOIN rs_feature_presence rsp
      ON rsp.ticker = u.symbol
"""
        score_terms.append("(CASE WHEN rsp.ticker IS NOT NULL THEN 30 ELSE 0 END)")

    effective_market_cap_expr = "COALESCE(u.market_cap, 0)"
    if has_financial_feature:
        effective_market_cap_expr = "COALESCE(u.market_cap, lf.market_cap)"

    effective_feature_score_expr = "u.feature_quality_score"
    if score_terms:
        effective_feature_score_expr = f"COALESCE(u.feature_quality_score, {' + '.join(score_terms)})"

    stmt = text(
        f"""
    WITH recent_price AS (
        SELECT
            ticker,
            AVG(volume)::numeric AS avg_volume_20d,
            MAX(trade_date) AS latest_trade_date
        FROM (
            SELECT
                ticker,
                trade_date,
                volume,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
            FROM market.us_stock_daily_price
        ) ranked
        WHERE rn <= 20
        GROUP BY ticker
    )
{latest_financial_cte}{daily_feature_cte}{rs_feature_cte}{financial_feature_cte}
    SELECT
        u.symbol,
        u.company_name,
        u.market,
        u.sector,
        u.industry,
        u.universe_group,
        u.is_active,
        u.is_etf,
        u.is_leveraged,
        u.is_inverse,
        u.source,
        u.market_cap,
        u.avg_volume,
        u.currency,
        u.country,
        u.exchange,
        u.first_included_date,
        u.last_checked_date,
        u.exclude_reason,
        u.feature_quality_score,
        {effective_market_cap_expr} AS effective_market_cap,
        COALESCE(u.avg_volume, rp.avg_volume_20d) AS effective_avg_volume,
        {effective_feature_score_expr} AS effective_feature_quality_score,
        rp.latest_trade_date
    FROM meta.us_stock_universe u
    LEFT JOIN recent_price rp
      ON rp.ticker = u.symbol
{latest_financial_join}{daily_feature_join}{rs_feature_join}{financial_feature_join}
    WHERE u.is_active = true
      AND (:include_etf = true OR COALESCE(u.is_etf, false) = false)
      AND (:exclude_leveraged = false OR COALESCE(u.is_leveraged, false) = false)
      AND (:exclude_inverse = false OR COALESCE(u.is_inverse, false) = false)
      AND rp.latest_trade_date IS NOT NULL
      AND COALESCE({effective_market_cap_expr}, 0) >= :min_market_cap
      AND COALESCE(COALESCE(u.avg_volume, rp.avg_volume_20d), 0) >= :min_avg_volume
      AND COALESCE({effective_feature_score_expr}, 0) >= :min_feature_quality_score
    ORDER BY u.symbol
    """
    )

    engine = get_us_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            stmt,
            {
                "min_market_cap": min_market_cap,
                "min_avg_volume": min_avg_volume,
                "min_feature_quality_score": min_feature_quality_score,
                "include_etf": include_etf,
                "exclude_leveraged": exclude_leveraged,
                "exclude_inverse": exclude_inverse,
            },
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


def upsert_financial_feature_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    ensure_us_financial_feature_reported_date_column()
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_FINANCIAL_FEATURE_SQL, rows)
    return len(rows)


def upsert_relative_strength_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_RELATIVE_STRENGTH_SQL, rows)
    return len(rows)


def upsert_label_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_LABEL_SQL, rows)
    return len(rows)


def upsert_meta_us_universe_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_META_US_UNIVERSE_SQL, rows)
    return len(rows)


def upsert_us_rank_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_SQL, rows)
    return len(rows)


def ensure_us_rank_backtest_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_TABLE_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_DETAIL_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_ADD_STRATEGY_NAME_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_ADD_SELECTION_RULE_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_SET_STRATEGY_NAME_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_SET_SELECTION_RULE_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_STRATEGY_NAME_NOT_NULL_SQL)
        conn.execute(ALTER_US_RANK_BACKTEST_RESULT_SELECTION_RULE_NOT_NULL_SQL)
        conn.execute(DROP_US_RANK_BACKTEST_RESULT_PKEY_SQL)
        conn.execute(ADD_US_RANK_BACKTEST_RESULT_PKEY_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_INDEX_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_SYMBOL_INDEX_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_SUMMARY_TABLE_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_SUMMARY_INDEX_SQL)


def ensure_us_market_regime_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_TABLE_SQL)
        conn.execute(CREATE_US_MARKET_REGIME_DAILY_TABLE_SQL)
        conn.execute(CREATE_US_MARKET_REGIME_DAILY_INDEX_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_REGIME_SUMMARY_TABLE_SQL)
        conn.execute(CREATE_US_RANK_BACKTEST_REGIME_SUMMARY_INDEX_SQL)


def ensure_us_weight_experiment_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_TABLE_SQL)
        conn.execute(CREATE_US_RULE_WEIGHT_CONFIG_TABLE_SQL)
        conn.execute(CREATE_US_RULE_WEIGHT_CONFIG_INDEX_SQL)
        conn.execute(CREATE_US_RANK_WEIGHT_EXPERIMENT_RESULT_TABLE_SQL)
        conn.execute(CREATE_US_RANK_WEIGHT_EXPERIMENT_RESULT_INDEX_SQL)
        conn.execute(CREATE_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_TABLE_SQL)
        conn.execute(CREATE_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_INDEX_SQL)


def ensure_us_forward_test_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_RANK_BACKTEST_RESULT_TABLE_SQL)
        detail_exists = conn.execute(text("SELECT to_regclass('research.us_stock_rank_forward_test')")).scalar() is not None
        summary_exists = conn.execute(text("SELECT to_regclass('research.us_stock_rank_forward_test_summary')")).scalar() is not None
        if not detail_exists:
            conn.execute(CREATE_US_RANK_FORWARD_TEST_TABLE_SQL)
        conn.execute(CREATE_US_RANK_FORWARD_TEST_INDEX_SQL)
        conn.execute(CREATE_US_RANK_FORWARD_TEST_SYMBOL_INDEX_SQL)
        if not summary_exists:
            conn.execute(CREATE_US_RANK_FORWARD_TEST_SUMMARY_TABLE_SQL)
        conn.execute(CREATE_US_RANK_FORWARD_TEST_SUMMARY_INDEX_SQL)


def ensure_us_paper_trading_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_PAPER_SCHEMA_SQL)
        conn.execute(CREATE_US_PAPER_ACCOUNT_TABLE_SQL)
        conn.execute(CREATE_US_PAPER_ORDER_TABLE_SQL)
        conn.execute(CREATE_US_PAPER_FILL_TABLE_SQL)
        conn.execute(CREATE_US_PAPER_POSITION_TABLE_SQL)
        conn.execute(CREATE_US_PAPER_ACCOUNT_SNAPSHOT_TABLE_SQL)
        conn.execute(CREATE_US_PAPER_ACCOUNT_INDEX_SQL)
        conn.execute(CREATE_US_PAPER_FILL_INDEX_SQL)
        conn.execute(CREATE_US_PAPER_POSITION_INDEX_SQL)
        conn.execute(CREATE_US_PAPER_SNAPSHOT_INDEX_SQL)


def ensure_us_live_risk_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_LIVE_RISK_SCHEMA_SQL)
        conn.execute(CREATE_US_LIVE_KILL_SWITCH_TABLE_SQL)
        conn.execute(ALTER_US_LIVE_KILL_SWITCH_ADD_TARGET_VALUE_SQL)
        conn.execute(CREATE_US_LIVE_DAILY_RISK_USAGE_TABLE_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_BLOCK_LOG_TABLE_SQL)
        conn.execute(CREATE_US_LIVE_KILL_SWITCH_EVENT_LOG_TABLE_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_APPROVAL_TABLE_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_APPROVAL_EVENT_LOG_TABLE_SQL)
        conn.execute(CREATE_US_LIVE_KILL_SWITCH_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_KILL_SWITCH_TARGET_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_DAILY_RISK_USAGE_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_BLOCK_LOG_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_KILL_SWITCH_EVENT_LOG_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_APPROVAL_INDEX_SQL)
        conn.execute(CREATE_US_LIVE_ORDER_APPROVAL_EVENT_LOG_INDEX_SQL)


def ensure_us_micro_live_tables() -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(CREATE_US_MICRO_LIVE_SCHEMA_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_REQUEST_TABLE_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_LAST_BROKER_STATUS_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_LAST_SYNC_AT_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_FILLED_QTY_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_REMAINING_QTY_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_AVG_FILLED_PRICE_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_FILLED_AMOUNT_USD_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_SYNC_STATUS_SQL)
        conn.execute(ALTER_US_MICRO_ORDER_REQUEST_ADD_SYNC_ERROR_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_EVENT_LOG_TABLE_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_FILL_TABLE_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_REQUEST_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_REQUEST_APPROVAL_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_EVENT_LOG_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_ORDER_FILL_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_RECONCILIATION_RESULT_TABLE_SQL)
        conn.execute(CREATE_US_MICRO_RECONCILIATION_RESULT_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_RECONCILIATION_RUN_INDEX_SQL)
        conn.execute(CREATE_US_MICRO_RECONCILIATION_EVENT_LOG_TABLE_SQL)
        conn.execute(CREATE_US_MICRO_RECONCILIATION_EVENT_LOG_INDEX_SQL)


def upsert_us_rank_backtest_result_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_BACKTEST_RESULT_SQL, rows)
    return len(rows)


def upsert_us_rank_backtest_summary_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_BACKTEST_SUMMARY_SQL, rows)
    return len(rows)


def upsert_us_market_regime_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_MARKET_REGIME_DAILY_SQL, rows)
    return len(rows)


def upsert_us_rank_backtest_regime_summary_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_BACKTEST_REGIME_SUMMARY_SQL, rows)
    return len(rows)


def upsert_us_rule_weight_config_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RULE_WEIGHT_CONFIG_SQL, rows)
    return len(rows)


def upsert_us_rank_weight_experiment_result_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_WEIGHT_EXPERIMENT_RESULT_SQL, rows)
    return len(rows)


def upsert_us_weight_experiment_backtest_summary_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_WEIGHT_EXPERIMENT_BACKTEST_SUMMARY_SQL, rows)
    return len(rows)


def upsert_us_forward_test_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_FORWARD_TEST_SQL, rows)
    return len(rows)


def upsert_us_forward_test_summary_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_RANK_FORWARD_TEST_SUMMARY_SQL, rows)
    return len(rows)


def upsert_us_paper_account_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_PAPER_ACCOUNT_SQL, rows)
    return len(rows)


def upsert_us_paper_order_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_PAPER_ORDER_SQL, rows)
    return len(rows)


def upsert_us_paper_account_snapshot_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_PAPER_ACCOUNT_SNAPSHOT_SQL, rows)
    return len(rows)


def upsert_us_live_kill_switch_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_LIVE_KILL_SWITCH_SQL, rows)
    return len(rows)


def upsert_us_live_daily_risk_usage_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_LIVE_DAILY_RISK_USAGE_SQL, rows)
    return len(rows)


def insert_us_live_order_block_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_US_LIVE_ORDER_BLOCK_LOG_SQL, rows)
    return len(rows)


def upsert_us_live_order_approval_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_LIVE_ORDER_APPROVAL_SQL, rows)
    return len(rows)


def insert_us_live_order_approval_event_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_US_LIVE_ORDER_APPROVAL_EVENT_LOG_SQL, rows)
    return len(rows)


def upsert_us_micro_order_request_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_MICRO_ORDER_REQUEST_SQL, rows)
    return len(rows)


def insert_us_micro_order_event_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_US_MICRO_ORDER_EVENT_LOG_SQL, rows)
    return len(rows)


def upsert_us_micro_order_fill_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_MICRO_ORDER_FILL_SQL, rows)
    return len(rows)


def upsert_us_micro_reconciliation_result_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(UPSERT_US_MICRO_RECONCILIATION_RESULT_SQL, rows)
    return len(rows)


def insert_us_micro_reconciliation_event_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_US_MICRO_RECONCILIATION_EVENT_LOG_SQL, rows)
    return len(rows)


def insert_us_live_kill_switch_event_log_rows(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(INSERT_US_LIVE_KILL_SWITCH_EVENT_LOG_SQL, rows)
    return len(rows)


def reset_us_paper_account(account_id: str) -> None:
    engine = get_us_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM paper.us_stock_paper_account_snapshot WHERE account_id = :account_id"), {"account_id": account_id})
        conn.execute(text("DELETE FROM paper.us_stock_paper_position WHERE account_id = :account_id"), {"account_id": account_id})
        conn.execute(text("DELETE FROM paper.us_stock_paper_fill WHERE account_id = :account_id"), {"account_id": account_id})
        conn.execute(text("DELETE FROM paper.us_stock_paper_order WHERE account_id = :account_id"), {"account_id": account_id})
        conn.execute(text("DELETE FROM paper.us_stock_paper_account WHERE account_id = :account_id"), {"account_id": account_id})


def delete_us_rank_rows(*, trade_date: date, source: str) -> int:
    engine = get_us_engine()
    with engine.begin() as conn:
        result = conn.execute(
            DELETE_US_RANK_ROWS_SQL,
            {"trade_date": trade_date, "source": source},
        )
    return int(result.rowcount or 0)
