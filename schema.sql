-- Lee_trader SQLite schema (aligned to current pipeline tables)
PRAGMA foreign_keys = ON;

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
    date           DATE PRIMARY KEY,
    kospi_close    REAL,
    kospi_ma20     REAL,
    volatility_5d  REAL,
    foreign_net_5d REAL,
    market_up      INTEGER
);

-- 3) Prices (raw / clean / adjusted)
CREATE TABLE IF NOT EXISTS prices_raw (
    date   DATE NOT NULL,
    code   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_raw_code_date ON prices_raw(code, date);

CREATE TABLE IF NOT EXISTS prices_clean (
    date   DATE NOT NULL,
    code   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_clean_code_date ON prices_clean(code, date);

CREATE TABLE IF NOT EXISTS prices_adjusted (
    date      DATE NOT NULL,
    code      TEXT NOT NULL,
    adj_open  REAL,
    adj_high  REAL,
    adj_low   REAL,
    adj_close REAL,
    volume    REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_prices_adjusted_code_date ON prices_adjusted(code, date);

-- 4) Fact price (enriched, optional)
CREATE TABLE IF NOT EXISTS fact_price_daily (
    date          DATE NOT NULL,
    code          TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    adj_close     REAL,
    volume        REAL,
    value         REAL,
    market_cap    REAL,
    listed_shares REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_fact_price_code_date ON fact_price_daily(code, date);

-- 5) Fundamentals (aggregated factors)
CREATE TABLE IF NOT EXISTS fundamentals (
    date           DATE NOT NULL,
    code           TEXT NOT NULL,
    roe            REAL,
    op_margin      REAL,
    debt_ratio     REAL,
    ocf_to_assets  REAL,
    net_margin     REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_code_date ON fundamentals(code, date);

-- 6) Quality score (forward-filled)
CREATE TABLE IF NOT EXISTS quality (
    date           DATE NOT NULL,
    code           TEXT NOT NULL,
    quality_score  REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_quality_code_date ON quality(code, date);

-- 7) Features (model inputs)
CREATE TABLE IF NOT EXISTS features (
    date             DATE NOT NULL,
    code             TEXT NOT NULL,
    close            REAL,
    ret_1d           REAL,
    ret_5d           REAL,
    ret_10d          REAL,
    mom_20           REAL,
    ma_5             REAL,
    ma_20            REAL,
    ma_60            REAL,
    close_over_ma20  REAL,
    vol_20           REAL,
    vol_60           REAL,
    rsi_14           REAL,
    volume           REAL,
    vol_ma_20        REAL,
    vol_ratio_20     REAL,
    quality_score    REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_features_code_date ON features(code, date);

-- 8) Labels (training targets)
CREATE TABLE IF NOT EXISTS labels (
    date               DATE NOT NULL,
    code               TEXT NOT NULL,
    target_60d         REAL,
    target_90d         REAL,
    target_log_60d     REAL,
    target_log_90d     REAL,
    target_mdd_60d     REAL,
    target_mdd_90d     REAL,
    target_60d_top20   REAL,
    target_90d_top20   REAL,
    realized_price_60d REAL,
    realized_price_90d REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_labels_code_date ON labels(code, date);

-- 9) Predictions (model outputs)
CREATE TABLE IF NOT EXISTS predictions (
    date             DATE NOT NULL,
    code             TEXT NOT NULL,
    pred_return_60d  REAL,
    pred_return_90d  REAL,
    pred_mdd_60d     REAL,
    pred_mdd_90d     REAL,
    prob_top20_60d   REAL,
    prob_top20_90d   REAL,
    score            REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_predictions_code_date ON predictions(code, date);

-- 10) Daily scores (technical)
CREATE TABLE IF NOT EXISTS daily_scores (
    date       DATE NOT NULL,
    code       TEXT NOT NULL,
    score      REAL,
    composite  REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_scores_code_date ON daily_scores(code, date);

-- 11) Daily ranking (final)
CREATE TABLE IF NOT EXISTS daily_ranking (
    date               DATE NOT NULL,
    code               TEXT NOT NULL,
    close              REAL,
    pred_return_60d    REAL,
    pred_return_90d    REAL,
    pred_mdd_60d       REAL,
    pred_mdd_90d       REAL,
    prob_top20_60d     REAL,
    prob_top20_90d     REAL,
    score              REAL,
    score_score        REAL,
    composite          REAL,
    quality_score      REAL,
    name               TEXT,
    market             TEXT,
    sector             TEXT,
    tech_score         REAL,
    pred_score         REAL,
    ret_score          REAL,
    prob_score         REAL,
    qual_score         REAL,
    safety_score       REAL,
    liquidity_score    REAL,
    final_score        REAL,
    risk_penalty       REAL,
    market_up          INTEGER,
    market_status_date DATE,
    market_kospi_close REAL,
    market_kospi_ma20  REAL,
    market_vol_5d      REAL,
    market_foreign_5d  REAL,
    generated_at       TEXT,
    model_version      TEXT,
    PRIMARY KEY (date, code)
);
CREATE INDEX IF NOT EXISTS idx_daily_ranking_code_date ON daily_ranking(code, date);

-- 12) Trades (manual/live)
CREATE TABLE IF NOT EXISTS trades (
    trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    date        DATE NOT NULL,
    side        TEXT NOT NULL, -- BUY/SELL
    code        TEXT NOT NULL,
    name        TEXT,
    market      TEXT,
    sector      TEXT,
    qty         REAL,
    price       REAL,
    amount      REAL,
    fee         REAL,
    memo        TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(code);

-- 13) Backtest trades (historical)
CREATE TABLE IF NOT EXISTS backtest_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date       DATE NOT NULL,
    strategy         TEXT NOT NULL,
    code             TEXT NOT NULL,
    final_score      REAL,
    pred_return_60d  REAL,
    realized_return  REAL NOT NULL,
    created_at       TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_date ON backtest_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_code ON backtest_trades(code);
