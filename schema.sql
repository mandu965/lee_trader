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

-- 5) Fundamentals (aggregated factors)
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

-- 6) Quality score (forward-filled)
CREATE TABLE IF NOT EXISTS quality (
    date           DATE NOT NULL,
    code           TEXT NOT NULL,
    quality_score  NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_quality_code_date ON quality(code, date);

-- 7) Features (model inputs)
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

-- 8) Labels (training targets)
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

-- 9) Predictions (model outputs)
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

-- 10) Daily scores (technical)
CREATE TABLE IF NOT EXISTS daily_scores (
    date       DATE NOT NULL,
    code       TEXT NOT NULL,
    score      NUMERIC,
    composite  NUMERIC,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_scores_code_date ON daily_scores(code, date);

-- 11) Daily ranking (final)
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
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_ranking_code_date ON daily_ranking(code, date);
CREATE INDEX IF NOT EXISTS idx_daily_ranking_date_final ON daily_ranking(date, final_score DESC);

-- 12) Trades (manual/live)
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

-- 13) Backtest trades (historical)
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

-- 14) Pipeline history (checkpoint per step)
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
