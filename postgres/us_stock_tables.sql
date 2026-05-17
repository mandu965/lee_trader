-- =============================================================================
-- US Stock Module DB 스키마 생성
-- 생성일: 2026-05-16
-- 용도: Lee_trader_us Phase A 백필 및 이후 Paper Trading 지원
-- =============================================================================

-- 스키마 생성
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS feature;
CREATE SCHEMA IF NOT EXISTS label;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS recommend;
CREATE SCHEMA IF NOT EXISTS paper;
CREATE SCHEMA IF NOT EXISTS risk;

-- =============================================================================
-- raw: 원시 수집 데이터
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw.us_stock_financial_statement (
    ticker              TEXT        NOT NULL,
    market              TEXT,
    period_type         TEXT        NOT NULL,  -- 'annual' | 'quarterly'
    fiscal_date         DATE        NOT NULL,
    reported_date       DATE,
    currency            TEXT,
    revenue             NUMERIC,
    gross_profit        NUMERIC,
    operating_income    NUMERIC,
    net_income          NUMERIC,
    ebitda              NUMERIC,
    total_assets        NUMERIC,
    total_liabilities   NUMERIC,
    total_equity        NUMERIC,
    operating_cash_flow NUMERIC,
    investing_cash_flow NUMERIC,
    financing_cash_flow NUMERIC,
    free_cash_flow      NUMERIC,
    source              TEXT        NOT NULL,
    source_updated_at   TIMESTAMPTZ,
    collected_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, period_type, fiscal_date, source)
);

CREATE TABLE IF NOT EXISTS raw.us_stock_financial_metric (
    ticker              TEXT        NOT NULL,
    market              TEXT,
    period_type         TEXT        NOT NULL,
    fiscal_date         DATE        NOT NULL,
    reported_date       DATE,
    currency            TEXT,
    eps                 NUMERIC,
    forward_eps         NUMERIC,
    roe                 NUMERIC,
    roa                 NUMERIC,
    shares_outstanding  NUMERIC,
    market_cap          NUMERIC,
    per                 NUMERIC,    -- trailing P/E
    forward_pe          NUMERIC,    -- forward P/E (analyst consensus)
    peg_ratio           NUMERIC,    -- price/earnings-to-growth
    pbr                 NUMERIC,
    psr                 NUMERIC,
    ev_ebitda           NUMERIC,
    debt_to_equity      NUMERIC,
    current_ratio       NUMERIC,
    dividend_yield      NUMERIC,
    source              TEXT        NOT NULL,
    source_updated_at   TIMESTAMPTZ,
    collected_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, period_type, fiscal_date, source)
);

-- =============================================================================
-- feature: 피처 테이블 (기술적·재무·상대강도)
-- =============================================================================

CREATE TABLE IF NOT EXISTS feature.us_stock_financial_feature (
    ticker                      TEXT        NOT NULL,
    market                      TEXT,
    period_type                 TEXT        NOT NULL,
    fiscal_date                 DATE        NOT NULL,
    source                      TEXT        NOT NULL,
    revenue                     NUMERIC,
    gross_profit                NUMERIC,
    operating_income            NUMERIC,
    net_income                  NUMERIC,
    ebitda                      NUMERIC,
    eps                         NUMERIC,
    total_assets                NUMERIC,
    total_liabilities           NUMERIC,
    total_equity                NUMERIC,
    operating_cash_flow         NUMERIC,
    free_cash_flow              NUMERIC,
    shares_outstanding          NUMERIC,
    market_cap                  NUMERIC,
    revenue_growth_yoy          NUMERIC,
    revenue_growth_qoq          NUMERIC,
    net_income_growth_yoy       NUMERIC,
    net_income_growth_qoq       NUMERIC,
    eps_growth_yoy              NUMERIC,
    eps_growth_qoq              NUMERIC,
    free_cash_flow_growth_yoy   NUMERIC,
    free_cash_flow_growth_qoq   NUMERIC,
    gross_margin                NUMERIC,
    operating_margin            NUMERIC,
    net_margin                  NUMERIC,
    ebitda_margin               NUMERIC,
    roe                         NUMERIC,
    roa                         NUMERIC,
    debt_to_equity              NUMERIC,
    debt_ratio                  NUMERIC,
    equity_ratio                NUMERIC,
    current_ratio               NUMERIC,
    free_cash_flow_margin       NUMERIC,
    per                         NUMERIC,
    pbr                         NUMERIC,
    psr                         NUMERIC,
    ev_ebitda                   NUMERIC,
    dividend_yield              NUMERIC,
    financial_quality_score     NUMERIC,
    financial_growth_score      NUMERIC,
    financial_value_score       NUMERIC,
    raw_collected_at            TIMESTAMPTZ,
    feature_created_at          TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, period_type, fiscal_date, source)
);

CREATE TABLE IF NOT EXISTS feature.us_stock_relative_strength_daily (
    ticker              TEXT        NOT NULL,
    market              TEXT,
    trade_date          DATE        NOT NULL,
    price_column_used   TEXT,
    ret_5d              NUMERIC,
    ret_20d             NUMERIC,
    ret_60d             NUMERIC,
    ret_120d            NUMERIC,
    ret_252d            NUMERIC,
    spy_ret_5d          NUMERIC,
    spy_ret_20d         NUMERIC,
    spy_ret_60d         NUMERIC,
    spy_ret_120d        NUMERIC,
    spy_ret_252d        NUMERIC,
    qqq_ret_5d          NUMERIC,
    qqq_ret_20d         NUMERIC,
    qqq_ret_60d         NUMERIC,
    qqq_ret_120d        NUMERIC,
    qqq_ret_252d        NUMERIC,
    rs_spy_5d           NUMERIC,
    rs_spy_20d          NUMERIC,
    rs_spy_60d          NUMERIC,
    rs_spy_120d         NUMERIC,
    rs_spy_252d         NUMERIC,
    rs_qqq_5d           NUMERIC,
    rs_qqq_20d          NUMERIC,
    rs_qqq_60d          NUMERIC,
    rs_qqq_120d         NUMERIC,
    rs_qqq_252d         NUMERIC,
    rs_spy_20d_rank_pct NUMERIC,
    rs_spy_60d_rank_pct NUMERIC,
    rs_qqq_20d_rank_pct NUMERIC,
    rs_qqq_60d_rank_pct NUMERIC,
    source              TEXT        NOT NULL DEFAULT 'yfinance',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date, source)
);

-- =============================================================================
-- label: ML 학습용 레이블
-- =============================================================================

CREATE TABLE IF NOT EXISTS label.us_stock_label_daily (
    ticker                  TEXT        NOT NULL,
    market                  TEXT,
    trade_date              DATE        NOT NULL,
    price_column_used       TEXT,
    future_ret_5d           NUMERIC,
    future_ret_20d          NUMERIC,
    future_ret_60d          NUMERIC,
    future_ret_20d_rank_pct NUMERIC,
    future_ret_60d_rank_pct NUMERIC,
    label_positive_20d      BOOLEAN,
    label_positive_60d      BOOLEAN,
    label_top20_20d         BOOLEAN,
    label_top20_60d         BOOLEAN,
    source                  TEXT        NOT NULL DEFAULT 'yfinance',
    label_created_at        TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date, source)
);

-- =============================================================================
-- meta: universe 메타 정보
-- =============================================================================

CREATE TABLE IF NOT EXISTS meta.us_stock_universe (
    symbol              TEXT        NOT NULL PRIMARY KEY,
    company_name        TEXT,
    market              TEXT,
    sector              TEXT,
    industry            TEXT,
    universe_group      TEXT,
    is_active           BOOLEAN,
    is_etf              BOOLEAN,
    is_leveraged        BOOLEAN,
    is_inverse          BOOLEAN,
    source              TEXT,
    market_cap          NUMERIC,
    avg_volume          NUMERIC,
    currency            TEXT,
    country             TEXT,
    exchange            TEXT,
    first_included_date DATE,
    last_checked_date   DATE,
    exclude_reason      TEXT,
    feature_quality_score NUMERIC,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- recommend: 랭킹 결과
-- =============================================================================

CREATE TABLE IF NOT EXISTS recommend.us_stock_rank_daily (
    trade_date              DATE        NOT NULL,
    symbol                  TEXT        NOT NULL,
    rank_no                 INTEGER,
    recommend_grade         TEXT,
    total_score             NUMERIC,
    momentum_score          NUMERIC,
    relative_strength_score NUMERIC,
    fundamental_score       NUMERIC,
    growth_score            NUMERIC,
    valuation_score         NUMERIC,
    risk_score              NUMERIC,
    feature_quality_score   NUMERIC,
    universe_group          TEXT,
    company_name            TEXT,
    sector                  TEXT,
    industry                TEXT,
    market_cap              NUMERIC,
    avg_volume              NUMERIC,
    is_etf                  BOOLEAN,
    is_active               BOOLEAN,
    data_status             TEXT,
    exclude_reason          TEXT,
    reason_summary          TEXT,
    score_detail_json       JSONB,
    source                  TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_us_stock_rank_daily_trade_date
    ON recommend.us_stock_rank_daily (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_stock_rank_daily_grade
    ON recommend.us_stock_rank_daily (trade_date DESC, recommend_grade);

-- =============================================================================
-- research: 포워드 테스트
-- =============================================================================

CREATE TABLE IF NOT EXISTS research.us_stock_rank_forward_test (
    forward_test_id         TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    symbol                  TEXT        NOT NULL,
    holding_days            INTEGER     NOT NULL,
    strategy_name           TEXT        NOT NULL,
    selection_rule          TEXT,
    rank_no                 INTEGER,
    recommend_grade         TEXT,
    total_score             NUMERIC,
    company_name            TEXT,
    sector                  TEXT,
    industry                TEXT,
    weight_config_id        TEXT,
    source                  TEXT,
    entry_date              DATE,
    entry_price             NUMERIC,
    target_exit_date        DATE,
    exit_date               DATE,
    exit_price              NUMERIC,
    return_pct              NUMERIC,
    spy_entry_price         NUMERIC,
    spy_exit_price          NUMERIC,
    spy_return_pct          NUMERIC,
    qqq_entry_price         NUMERIC,
    qqq_exit_price          NUMERIC,
    qqq_return_pct          NUMERIC,
    excess_return_vs_spy    NUMERIC,
    excess_return_vs_qqq    NUMERIC,
    win_flag                BOOLEAN,
    win_vs_spy_flag         BOOLEAN,
    win_vs_qqq_flag         BOOLEAN,
    market_regime           TEXT,
    spy_regime              TEXT,
    qqq_regime              TEXT,
    vol_regime              TEXT,
    status                  TEXT,
    data_status             TEXT,
    exclude_reason          TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (forward_test_id, trade_date, strategy_name, symbol, holding_days)
);

CREATE TABLE IF NOT EXISTS research.us_stock_rank_forward_test_summary (
    forward_test_id             TEXT        NOT NULL,
    trade_date                  DATE        NOT NULL,
    strategy_name               TEXT        NOT NULL,
    holding_days                INTEGER     NOT NULL,
    selected_count              INTEGER,
    completed_count             INTEGER,
    active_count                INTEGER,
    pending_count               INTEGER,
    error_count                 INTEGER,
    avg_return_pct              NUMERIC,
    median_return_pct           NUMERIC,
    win_rate                    NUMERIC,
    avg_spy_return_pct          NUMERIC,
    avg_qqq_return_pct          NUMERIC,
    avg_excess_return_vs_spy    NUMERIC,
    avg_excess_return_vs_qqq    NUMERIC,
    win_rate_vs_spy             NUMERIC,
    win_rate_vs_qqq             NUMERIC,
    best_symbol                 TEXT,
    best_return_pct             NUMERIC,
    worst_symbol                TEXT,
    worst_return_pct            NUMERIC,
    status                      TEXT,
    data_status                 TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (forward_test_id, trade_date, strategy_name, holding_days)
);

-- =============================================================================
-- paper: 페이퍼 트레이딩 (Phase B 전환 시 사용)
-- =============================================================================

CREATE TABLE IF NOT EXISTS paper.us_stock_paper_account (
    account_id          TEXT        NOT NULL PRIMARY KEY,
    account_name        TEXT,
    initial_cash        NUMERIC,
    base_currency       TEXT        DEFAULT 'USD',
    strategy_name       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper.us_stock_paper_order (
    paper_order_id      TEXT        NOT NULL PRIMARY KEY,
    account_id          TEXT        NOT NULL,
    trade_date          DATE        NOT NULL,
    symbol              TEXT        NOT NULL,
    side                TEXT        NOT NULL,  -- 'BUY' | 'SELL'
    order_type          TEXT,
    quantity            NUMERIC,
    amount_usd          NUMERIC,
    limit_price         NUMERIC,
    fill_price          NUMERIC,
    status              TEXT,
    strategy_name       TEXT,
    rank_no             INTEGER,
    total_score         NUMERIC,
    recommend_grade     TEXT,
    source              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper.us_stock_paper_fill (
    paper_order_id      TEXT        NOT NULL,
    account_id          TEXT        NOT NULL,
    trade_date          DATE        NOT NULL,
    symbol              TEXT        NOT NULL,
    side                TEXT        NOT NULL,
    fill_price          NUMERIC,
    quantity            NUMERIC,
    amount_usd          NUMERIC,
    commission          NUMERIC,
    slippage_bps        NUMERIC,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_order_id)
);

CREATE TABLE IF NOT EXISTS paper.us_stock_paper_position (
    account_id              TEXT        NOT NULL,
    symbol                  TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'OPEN',
    quantity                NUMERIC,
    avg_entry_price         NUMERIC,
    current_price           NUMERIC,
    unrealized_pnl_pct      NUMERIC,
    highest_price_since_entry NUMERIC,
    holding_days            INTEGER,
    entry_date              DATE,
    exit_date               DATE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, symbol, status)
);

CREATE TABLE IF NOT EXISTS paper.us_stock_paper_account_snapshot (
    account_id          TEXT        NOT NULL,
    snapshot_date       DATE        NOT NULL,
    cash_usd            NUMERIC,
    portfolio_value_usd NUMERIC,
    total_value_usd     NUMERIC,
    position_count      INTEGER,
    daily_return_pct    NUMERIC,
    cumulative_return_pct NUMERIC,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, snapshot_date)
);

-- =============================================================================
-- risk: 킬스위치
-- =============================================================================

CREATE TABLE IF NOT EXISTS risk.us_stock_live_kill_switch (
    kill_switch_id      TEXT        NOT NULL PRIMARY KEY,
    account_id          TEXT,
    status              TEXT,
    triggered_at        TIMESTAMPTZ,
    triggered_by        TEXT,
    reason              TEXT,
    reset_at            TIMESTAMPTZ,
    reset_by            TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
