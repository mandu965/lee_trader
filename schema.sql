-- Postgres schema for Lee_trader
-- Use: psql -d <db> -f schema.sql

-- 1) Stocks master
CREATE TABLE IF NOT EXISTS stocks (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    market      TEXT,
    sector      TEXT,
    listed_at   DATE,
    delisted_at DATE
);

-- 2) Market status (regime info)
CREATE TABLE IF NOT EXISTS market_status (
    date            DATE PRIMARY KEY,
    kospi_close     NUMERIC,
    kospi_ma20      NUMERIC,
    volatility_5d   NUMERIC,
    foreign_net_5d  NUMERIC,
    market_up       BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_market_status_date_desc ON market_status(date DESC);

-- 3) Prices (raw / clean / adjusted)
CREATE TABLE IF NOT EXISTS prices_raw (
    date   DATE NOT NULL,
    code   TEXT NOT NULL,
    open   NUMERIC,
    high   NUMERIC,
    low    NUMERIC,
    close  NUMERIC,
    volume NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_raw_code_date ON prices_raw(code, date);

CREATE TABLE IF NOT EXISTS prices_clean (
    date   DATE NOT NULL,
    code   TEXT NOT NULL,
    open   NUMERIC,
    high   NUMERIC,
    low    NUMERIC,
    close  NUMERIC,
    volume NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_clean_code_date ON prices_clean(code, date);

CREATE TABLE IF NOT EXISTS prices_adjusted (
    date      DATE NOT NULL,
    code      TEXT NOT NULL,
    adj_open  NUMERIC,
    adj_high  NUMERIC,
    adj_low   NUMERIC,
    adj_close NUMERIC,
    volume    NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_adjusted_code_date ON prices_adjusted(code, date);

-- 4) Fact price (optional enriched)
CREATE TABLE IF NOT EXISTS fact_price_daily (
    date          DATE NOT NULL,
    code          TEXT NOT NULL,
    open          NUMERIC,
    high          NUMERIC,
    low           NUMERIC,
    close         NUMERIC,
    adj_close     NUMERIC,
    volume        NUMERIC,
    value         NUMERIC,
    market_cap    NUMERIC,
    listed_shares NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_fact_price_code_date ON fact_price_daily(code, date);

-- 5) Investor flow raw source (normalized by investor_type)
CREATE TABLE IF NOT EXISTS flow_daily (
    date              DATE         NOT NULL,
    code              VARCHAR(6)   NOT NULL,
    investor_type     VARCHAR(32)  NOT NULL,
    net_buy_amount    NUMERIC,
    net_buy_volume    NUMERIC,
    raw_payload_hash  VARCHAR(64)  NOT NULL,
    collected_at      TIMESTAMPTZ  NOT NULL,
    source_endpoint   VARCHAR(128) NOT NULL,
    market_div_code   VARCHAR(8)   NOT NULL,
    input_date        VARCHAR(8)   NOT NULL,
    tr_id             VARCHAR(32)  NOT NULL,
    raw_payload_json  JSONB,
    created_run_id    BIGINT,
    fetch_status      VARCHAR(32),
    error_code        VARCHAR(64),
    error_message     TEXT,
    is_partial_page   BOOLEAN      NOT NULL DEFAULT false,
    PRIMARY KEY (date, code, investor_type)
);
CREATE INDEX IF NOT EXISTS idx_flow_daily_code_date_desc ON flow_daily(code, date DESC);
CREATE INDEX IF NOT EXISTS idx_flow_daily_date_investor_type ON flow_daily(date, investor_type);

-- 6) Fundamentals (aggregated factors)
CREATE TABLE IF NOT EXISTS fundamentals (
    date           DATE NOT NULL,
    code           TEXT NOT NULL,
    roe            NUMERIC,
    op_margin      NUMERIC,
    debt_ratio     NUMERIC,
    ocf_to_assets  NUMERIC,
    net_margin     NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_code_date ON fundamentals(code, date);

-- 7) Quality score (forward-filled)
CREATE TABLE IF NOT EXISTS quality (
    date           DATE NOT NULL,
    code           TEXT NOT NULL,
    quality_score  NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_quality_code_date ON quality(code, date);

-- 8) Features (model inputs)
CREATE TABLE IF NOT EXISTS features (
    date             DATE NOT NULL,
    code             TEXT NOT NULL,
    close            NUMERIC,
    ret_1d           NUMERIC,
    ret_5d           NUMERIC,
    ret_10d          NUMERIC,
    mom_20           NUMERIC,
    ma_5             NUMERIC,
    ma_20            NUMERIC,
    ma_60            NUMERIC,
    close_over_ma20  NUMERIC,
    vol_20           NUMERIC,
    vol_60           NUMERIC,
    rsi_14           NUMERIC,
    volume           NUMERIC,
    vol_ma_20        NUMERIC,
    vol_ratio_20     NUMERIC,
    quality_score    NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_features_code_date ON features(code, date);

-- 9) Labels (training targets)
CREATE TABLE IF NOT EXISTS labels (
    date               DATE NOT NULL,
    code               TEXT NOT NULL,
    target_60d         NUMERIC,
    target_90d         NUMERIC,
    target_log_60d     NUMERIC,
    target_log_90d     NUMERIC,
    target_mdd_60d     NUMERIC,
    target_mdd_90d     NUMERIC,
    target_60d_top20   NUMERIC,
    target_90d_top20   NUMERIC,
    realized_price_60d NUMERIC,
    realized_price_90d NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_labels_code_date ON labels(code, date);

-- 10) Predictions (model outputs)
CREATE TABLE IF NOT EXISTS predictions (
    date             DATE NOT NULL,
    code             TEXT NOT NULL,
    pred_return_60d  NUMERIC,
    pred_return_90d  NUMERIC,
    pred_mdd_60d     NUMERIC,
    pred_mdd_90d     NUMERIC,
    prob_top20_60d   NUMERIC,
    prob_top20_90d   NUMERIC,
    score            NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_predictions_code_date ON predictions(code, date);

-- 11) Daily scores (technical)
CREATE TABLE IF NOT EXISTS daily_scores (
    date       DATE NOT NULL,
    code       TEXT NOT NULL,
    score      NUMERIC,
    composite  NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_scores_code_date ON daily_scores(code, date);

-- 12) Daily ranking (final)
CREATE TABLE IF NOT EXISTS daily_ranking (
    date               DATE NOT NULL,
    code               TEXT NOT NULL,
    close              NUMERIC,
    pred_return_60d    NUMERIC,
    pred_return_90d    NUMERIC,
    pred_mdd_60d       NUMERIC,
    pred_mdd_90d       NUMERIC,
    prob_top20_60d     NUMERIC,
    prob_top20_90d     NUMERIC,
    score              NUMERIC,
    score_score        NUMERIC,
    composite          NUMERIC,
    quality_score      NUMERIC,
    name               TEXT,
    market             TEXT,
    sector             TEXT,
    tech_score         NUMERIC,
    pred_score         NUMERIC,
    ret_score          NUMERIC,
    prob_score         NUMERIC,
    qual_score         NUMERIC,
    safety_score       NUMERIC,
    liquidity_score    NUMERIC,
    final_score        NUMERIC,
    risk_penalty       NUMERIC,
    market_up          BOOLEAN,
    market_status_date DATE,
    market_kospi_close NUMERIC,
    market_kospi_ma20  NUMERIC,
    market_vol_5d      NUMERIC,
    market_foreign_5d  NUMERIC,
    generated_at       TIMESTAMPTZ,
    model_version      TEXT,
    score_formula_version TEXT,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_ranking_code_date ON daily_ranking(code, date);
CREATE INDEX IF NOT EXISTS idx_daily_ranking_date_final ON daily_ranking(date, final_score DESC);

-- 13) Trades (manual/live)
CREATE TABLE IF NOT EXISTS trades (
    trade_id    BIGSERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    side        TEXT NOT NULL, -- BUY/SELL
    code        TEXT NOT NULL,
    name        TEXT,
    market      TEXT,
    sector      TEXT,
    qty         NUMERIC,
    price       NUMERIC,
    amount      NUMERIC,
    fee         NUMERIC,
    memo        TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(code);

-- 14) Backtest trades (historical)
CREATE TABLE IF NOT EXISTS backtest_trades (
    id               BIGSERIAL PRIMARY KEY,
    trade_date       DATE NOT NULL,
    strategy         TEXT NOT NULL,
    code             TEXT NOT NULL,
    final_score      NUMERIC,
    pred_return_60d  NUMERIC,
    realized_return  NUMERIC NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_date ON backtest_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_code ON backtest_trades(code);

-- 15) Pipeline history (checkpoint per step)
CREATE TABLE IF NOT EXISTS pipeline_history (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL,
    step        TEXT NOT NULL,
    status      TEXT NOT NULL,
    duration_s  NUMERIC,
    message     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pipeline_history_run_step ON pipeline_history(run_id, step);
CREATE INDEX IF NOT EXISTS idx_pipeline_history_created ON pipeline_history(created_at DESC);

-- 16) Theme master (ETF-driven theme dictionary)
CREATE TABLE IF NOT EXISTS theme_master (
    theme_code         TEXT PRIMARY KEY,
    theme_name         TEXT NOT NULL UNIQUE,
    theme_group        TEXT,
    theme_description  TEXT,
    display_order      INTEGER,
    is_active          BOOLEAN NOT NULL DEFAULT true,
    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE theme_master IS 'Theme master used as the canonical dictionary for ETF-based theme classification.';
CREATE INDEX IF NOT EXISTS idx_theme_master_is_active ON theme_master(is_active, display_order, theme_code);

-- 17) ETF master (ETF metadata used by theme pipeline)
CREATE TABLE IF NOT EXISTS etf_master (
    etf_code               VARCHAR(12) PRIMARY KEY,
    etf_name               TEXT NOT NULL,
    market                 TEXT,
    asset_class            TEXT,
    issuer_name            TEXT,
    reference_index_name   TEXT,
    listing_date           DATE,
    delisting_date         DATE,
    is_active              BOOLEAN NOT NULL DEFAULT true,
    created_at             TIMESTAMPTZ DEFAULT now(),
    updated_at             TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE etf_master IS 'ETF master storing core metadata and listing status used by the theme pipeline.';
CREATE INDEX IF NOT EXISTS idx_etf_master_is_active ON etf_master(is_active, etf_code);
CREATE INDEX IF NOT EXISTS idx_etf_master_listing_date ON etf_master(listing_date DESC, etf_code);

-- 18) ETF-theme mapping (manual/curated theme assignment per ETF)
CREATE TABLE IF NOT EXISTS etf_theme_map (
    etf_code             VARCHAR(12) NOT NULL REFERENCES etf_master(etf_code),
    theme_code           TEXT        NOT NULL REFERENCES theme_master(theme_code),
    mapping_source       TEXT        NOT NULL DEFAULT 'manual',
    mapping_confidence   NUMERIC,
    is_primary           BOOLEAN     NOT NULL DEFAULT false,
    valid_from           DATE        NOT NULL DEFAULT CURRENT_DATE,
    valid_to             DATE,
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (etf_code, theme_code, valid_from),
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
COMMENT ON TABLE etf_theme_map IS 'Mapping table that manages ETF-to-theme assignments, validity windows, and confidence.';
CREATE INDEX IF NOT EXISTS idx_etf_theme_map_theme_code_valid_from ON etf_theme_map(theme_code, valid_from DESC, etf_code);
CREATE INDEX IF NOT EXISTS idx_etf_theme_map_etf_code_valid_to ON etf_theme_map(etf_code, valid_to, theme_code);

-- 19) ETF holdings snapshot (daily ETF constituent snapshot)
CREATE TABLE IF NOT EXISTS etf_holdings_snapshot (
    as_of_date         DATE         NOT NULL,
    etf_code           VARCHAR(12)  NOT NULL REFERENCES etf_master(etf_code),
    stock_code         TEXT         NOT NULL REFERENCES stocks(code),
    stock_name         TEXT,
    holding_weight     NUMERIC,
    holding_quantity   NUMERIC,
    market_value       NUMERIC,
    rank_in_etf        INTEGER,
    source_name        TEXT         NOT NULL,
    collected_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    raw_payload_json   JSONB,
    PRIMARY KEY (as_of_date, etf_code, stock_code)
);
COMMENT ON TABLE etf_holdings_snapshot IS 'Daily snapshot of ETF constituent stocks and their portfolio weights.';
CREATE INDEX IF NOT EXISTS idx_etf_holdings_snapshot_stock_code_asof ON etf_holdings_snapshot(stock_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_etf_holdings_snapshot_etf_code_asof ON etf_holdings_snapshot(etf_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_etf_holdings_snapshot_asof_etf_code ON etf_holdings_snapshot(as_of_date, etf_code);

-- 20) ETF signal daily (ETF price/NAV strength signal)
CREATE TABLE IF NOT EXISTS etf_signal_daily (
    as_of_date               DATE         NOT NULL,
    etf_code                 VARCHAR(12)  NOT NULL REFERENCES etf_master(etf_code),
    close_price              NUMERIC,
    nav_price                NUMERIC,
    nav_gap_pct              NUMERIC,
    return_1d                NUMERIC,
    return_5d                NUMERIC,
    return_20d               NUMERIC,
    volume                   NUMERIC,
    trading_value            NUMERIC,
    aum_amount               NUMERIC,
    relative_strength_score  NUMERIC,
    signal_score             NUMERIC,
    signal_payload_json      JSONB,
    created_at               TIMESTAMPTZ DEFAULT now(),
    updated_at               TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (as_of_date, etf_code)
);
COMMENT ON TABLE etf_signal_daily IS 'Daily ETF signal fact table storing price, NAV, and relative-strength metrics.';
CREATE INDEX IF NOT EXISTS idx_etf_signal_daily_etf_code_asof ON etf_signal_daily(etf_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_etf_signal_daily_asof_score ON etf_signal_daily(as_of_date, signal_score DESC, etf_code);

-- 21) Stock theme exposure daily (theme exposure propagated from ETF holdings)
CREATE TABLE IF NOT EXISTS stock_theme_exposure_daily (
    as_of_date             DATE         NOT NULL,
    stock_code             TEXT         NOT NULL REFERENCES stocks(code),
    theme_code             TEXT         NOT NULL REFERENCES theme_master(theme_code),
    exposure_score         NUMERIC      NOT NULL,
    exposure_weight        NUMERIC,
    supporting_etf_count   INTEGER      NOT NULL DEFAULT 0,
    primary_etf_code       VARCHAR(12)  REFERENCES etf_master(etf_code),
    calc_version           TEXT,
    created_at             TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (as_of_date, stock_code, theme_code)
);
COMMENT ON TABLE stock_theme_exposure_daily IS 'Daily stock-level theme exposure propagated from ETF holdings snapshots.';
CREATE INDEX IF NOT EXISTS idx_stock_theme_exposure_daily_stock_code_asof ON stock_theme_exposure_daily(stock_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_theme_exposure_daily_asof_stock_code ON stock_theme_exposure_daily(as_of_date, stock_code);
CREATE INDEX IF NOT EXISTS idx_stock_theme_exposure_daily_theme_code_asof ON stock_theme_exposure_daily(theme_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_theme_exposure_daily_asof_theme_score ON stock_theme_exposure_daily(as_of_date, theme_code, exposure_score DESC, stock_code);
CREATE INDEX IF NOT EXISTS idx_stock_theme_exposure_daily_primary_etf_code_asof ON stock_theme_exposure_daily(primary_etf_code, as_of_date DESC);

-- 22) Theme score daily (final theme strength score by day)
CREATE TABLE IF NOT EXISTS theme_score_daily (
    as_of_date         DATE         NOT NULL,
    theme_code         TEXT         NOT NULL REFERENCES theme_master(theme_code),
    theme_score        NUMERIC      NOT NULL,
    signal_score       NUMERIC,
    breadth_count      INTEGER      NOT NULL DEFAULT 0,
    leader_etf_code    VARCHAR(12)  REFERENCES etf_master(etf_code),
    leader_stock_code  TEXT         REFERENCES stocks(code),
    calc_version       TEXT,
    created_at         TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (as_of_date, theme_code)
);
COMMENT ON TABLE theme_score_daily IS 'Daily final theme score table with representative ETF and stock references.';
CREATE INDEX IF NOT EXISTS idx_theme_score_daily_theme_code_asof ON theme_score_daily(theme_code, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_theme_score_daily_asof_score ON theme_score_daily(as_of_date, theme_score DESC, theme_code);
CREATE INDEX IF NOT EXISTS idx_theme_score_daily_leader_etf_code_asof ON theme_score_daily(leader_etf_code, as_of_date DESC);

-- ============================
-- Research / Backtest layer
-- ============================
CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.app_payload_store (
    payload_key   TEXT PRIMARY KEY,
    payload_json  JSONB NOT NULL,
    asof_date     DATE,
    generated_at  TIMESTAMPTZ,
    source_path   TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_app_payload_store_asof ON research.app_payload_store(asof_date DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS research.dim_model_run (
    run_id           BIGSERIAL PRIMARY KEY,
    run_type         VARCHAR(32)  NOT NULL, -- 'daily_pipeline', 'backtest_offline', 'grid_search', ...
    model_version    VARCHAR(50)  NOT NULL,
    horizon_days     INTEGER      NOT NULL,
    top_n            INTEGER      DEFAULT 20,
    train_start_date DATE,
    train_end_date   DATE,
    config_json      JSONB,
    created_at       TIMESTAMPTZ  DEFAULT now(),
    comment          TEXT
);
CREATE INDEX IF NOT EXISTS idx_dim_model_run_type ON research.dim_model_run(run_type);

CREATE TABLE IF NOT EXISTS research.prediction_history (
    run_id             BIGINT       REFERENCES research.dim_model_run(run_id),
    as_of_date         DATE         NOT NULL,
    code               VARCHAR(10)  NOT NULL,
    model_version      VARCHAR(50)  NOT NULL,
    horizon_days       INTEGER      NOT NULL,
    pred_return_60d    NUMERIC,
    pred_return_90d    NUMERIC,
    pred_mdd_60d       NUMERIC,
    pred_mdd_90d       NUMERIC,
    prob_top20_60d     NUMERIC,
    prob_top20_90d     NUMERIC,
    ret_score          NUMERIC,
    prob_score         NUMERIC,
    qual_score         NUMERIC,
    tech_score         NUMERIC,
    risk_penalty       NUMERIC,
    final_score        NUMERIC,
    final_score_custom NUMERIC,
    created_at         TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, as_of_date, code, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_pred_hist_asof_model ON research.prediction_history(as_of_date, model_version);
CREATE INDEX IF NOT EXISTS idx_pred_hist_code_model ON research.prediction_history(code, model_version);

CREATE TABLE IF NOT EXISTS research.ranking_history (
    run_id        BIGINT       REFERENCES research.dim_model_run(run_id),
    as_of_date    DATE         NOT NULL,
    code          VARCHAR(10)  NOT NULL,
    model_version VARCHAR(50)  NOT NULL,
    horizon_days  INTEGER      NOT NULL,
    rank          INTEGER      NOT NULL,
    final_score   NUMERIC      NOT NULL,
    score_formula_version TEXT,
    ret_score     NUMERIC,
    prob_score    NUMERIC,
    qual_score    NUMERIC,
    tech_score    NUMERIC,
    risk_penalty  NUMERIC,
    in_top_n      BOOLEAN      NOT NULL DEFAULT false,
    top_n         INTEGER      NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, as_of_date, code)
);
CREATE INDEX IF NOT EXISTS idx_rank_hist_run_asof ON research.ranking_history(run_id, as_of_date, rank);
CREATE INDEX IF NOT EXISTS idx_rank_hist_asof_model ON research.ranking_history(as_of_date, model_version);

-- Optional: realized outcomes per pick (can be derived on the fly)
CREATE TABLE IF NOT EXISTS research.backtest_outcome (
    run_id          BIGINT       REFERENCES research.dim_model_run(run_id),
    as_of_date      DATE         NOT NULL,
    code            VARCHAR(10)  NOT NULL,
    horizon_days    INTEGER      NOT NULL,
    realized_return NUMERIC,
    realized_mdd    NUMERIC,
    label_source    VARCHAR(32), -- 'from_labels', 'from_price'
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, as_of_date, code, horizon_days)
);

CREATE TABLE IF NOT EXISTS research.paper_trading_run (
    paper_run_id          BIGSERIAL PRIMARY KEY,
    run_tag               TEXT         NOT NULL UNIQUE,
    source_mode           TEXT         NOT NULL,
    asof_date             DATE,
    hold_days             INTEGER      NOT NULL,
    initial_nav           NUMERIC,
    entry_fee_bps         NUMERIC,
    exit_fee_bps          NUMERIC,
    entry_slippage_bps    NUMERIC,
    exit_slippage_bps     NUMERIC,
    positions_row_count   INTEGER      NOT NULL DEFAULT 0,
    nav_row_count         INTEGER      NOT NULL DEFAULT 0,
    source_positions_csv  TEXT,
    source_nav_csv        TEXT,
    source_report_md      TEXT,
    comment               TEXT,
    created_at            TIMESTAMPTZ  DEFAULT now(),
    updated_at            TIMESTAMPTZ  DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_paper_trading_run_asof ON research.paper_trading_run(asof_date DESC, paper_run_id DESC);

CREATE TABLE IF NOT EXISTS research.paper_trading_position (
    paper_run_id           BIGINT       NOT NULL REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE,
    strategy               TEXT         NOT NULL,
    code                   VARCHAR(10)  NOT NULL,
    name                   TEXT,
    entry_date             DATE         NOT NULL,
    planned_exit_date      DATE,
    exit_date              DATE,
    entry_price_close      NUMERIC,
    entry_exec_price       NUMERIC,
    exit_price_close       NUMERIC,
    exit_exec_price        NUMERIC,
    shares                 NUMERIC,
    entry_notional_gross   NUMERIC,
    exit_notional_net      NUMERIC,
    entry_cost_amount      NUMERIC,
    exit_cost_amount       NUMERIC,
    gross_return           NUMERIC,
    net_return             NUMERIC,
    source_rank            INTEGER,
    selection_stage        TEXT,
    dominant_theme         TEXT,
    confidence_score       NUMERIC,
    final_score            NUMERIC,
    status                 TEXT,
    created_at             TIMESTAMPTZ  DEFAULT now(),
    PRIMARY KEY (paper_run_id, strategy, code, entry_date)
);
CREATE INDEX IF NOT EXISTS idx_paper_trading_position_lookup ON research.paper_trading_position(strategy, entry_date DESC, code);

CREATE TABLE IF NOT EXISTS research.paper_trading_nav (
    paper_run_id             BIGINT       NOT NULL REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE,
    strategy                 TEXT         NOT NULL,
    date                     DATE         NOT NULL,
    cash                     NUMERIC,
    market_value             NUMERIC,
    nav                      NUMERIC,
    daily_return             NUMERIC,
    active_position_count    INTEGER,
    opened_today             INTEGER,
    duplicate_skip_count     INTEGER,
    deployed_cash            NUMERIC,
    cumulative_return        NUMERIC,
    running_nav_max          NUMERIC,
    drawdown                 NUMERIC,
    closed_trade_count       INTEGER,
    closed_win_rate          NUMERIC,
    closed_win_count         INTEGER,
    closed_trade_count_cum   INTEGER,
    closed_win_count_cum     INTEGER,
    created_at               TIMESTAMPTZ  DEFAULT now(),
    PRIMARY KEY (paper_run_id, strategy, date)
);
CREATE INDEX IF NOT EXISTS idx_paper_trading_nav_lookup ON research.paper_trading_nav(strategy, date DESC);
