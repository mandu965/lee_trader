-- =============================================================================
-- US Trade Schema — buy/sell automation 로그 테이블
-- 생성일: 2026-05-16
-- 용도: US Paper Trading 의사결정·주문·포지션 스냅샷 저장
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS trade;

-- ---------------------------------------------------------------------------
-- 매수 후보 로그
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_buy_candidate_log (
    candidate_id            TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    account_id              TEXT        NOT NULL,
    automation_mode         TEXT        NOT NULL,   -- SHADOW | PAPER | LIVE
    ranking_source          TEXT,
    symbol                  TEXT        NOT NULL,
    company_name            TEXT,
    sector                  TEXT,
    rank_no                 INTEGER,
    recommend_grade         TEXT,
    total_score             NUMERIC,
    score_detail_json       JSONB,
    price_ref               NUMERIC,
    candidate_amount_usd    NUMERIC,
    candidate_status        TEXT,
    filter_stage            TEXT,
    filter_reason_code      TEXT,
    filter_reason_detail    TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (candidate_id),
    UNIQUE (trade_date, automation_mode, symbol, filter_stage)
);

-- ---------------------------------------------------------------------------
-- 매수 의사결정 로그
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_buy_decision_log (
    decision_id             TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    account_id              TEXT        NOT NULL,
    automation_mode         TEXT        NOT NULL,
    symbol                  TEXT        NOT NULL,
    candidate_id            TEXT,
    decision                TEXT,       -- ALLOWED | BLOCKED
    severity                TEXT,
    decision_reason_code    TEXT,
    decision_reason_detail  TEXT,
    rule_tags               JSONB,
    block_reasons           JSONB,
    rank_no                 INTEGER,
    recommend_grade         TEXT,
    total_score             NUMERIC,
    price_ref               NUMERIC,
    planned_order_amount_usd NUMERIC,
    cooldown_until          DATE,
    conflict_checked        BOOLEAN,
    conflict_blocked        BOOLEAN,
    conflict_reasons        JSONB,
    related_position_id     TEXT,
    related_sell_signal     JSONB,
    requires_manual_review  BOOLEAN,
    report_group            TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (decision_id),
    UNIQUE (trade_date, automation_mode, symbol)
);

-- ---------------------------------------------------------------------------
-- 리스크 가드 로그
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_risk_guard_log (
    guard_log_id            TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    account_id              TEXT        NOT NULL,
    automation_mode         TEXT        NOT NULL,
    guard_scope             TEXT        NOT NULL,
    guard_name              TEXT        NOT NULL,
    guard_status            TEXT,       -- PASS | FAIL | WARN
    severity                TEXT,
    metric_value            NUMERIC,
    threshold_value         NUMERIC,
    reason_code             TEXT,
    reason_detail           TEXT,
    raw_payload             JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (guard_log_id),
    UNIQUE (trade_date, automation_mode, guard_scope, guard_name, account_id)
);

-- ---------------------------------------------------------------------------
-- 매수 페이퍼 주문
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_paper_order (
    paper_order_id          TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    account_id              TEXT        NOT NULL,
    automation_mode         TEXT        NOT NULL,
    symbol                  TEXT        NOT NULL,
    side                    TEXT        NOT NULL DEFAULT 'BUY',
    paper_order_qty         NUMERIC,
    paper_order_price       NUMERIC,
    paper_order_amount      NUMERIC,
    assumed_fill_price      NUMERIC,
    assumed_fill_status     TEXT,
    source_decision_id      TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_order_id),
    UNIQUE (trade_date, automation_mode, symbol, side)
);

-- ---------------------------------------------------------------------------
-- 매도 의사결정 로그
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_sell_decision_log (
    sell_decision_id        TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    account_id              TEXT        NOT NULL,
    automation_mode         TEXT        NOT NULL,
    paper_position_id       TEXT,
    symbol                  TEXT        NOT NULL,
    decision                TEXT,       -- FULL_SELL | PARTIAL_SELL | HOLD | REVIEW_REQUIRED
    sell_action             TEXT,
    sell_ratio              NUMERIC,
    sell_quantity           NUMERIC,
    exit_reason             TEXT,
    review_required         BOOLEAN,
    applied_rules           JSONB,
    latest_price            NUMERIC,
    avg_entry_price         NUMERIC,
    unrealized_pnl_pct      NUMERIC,
    realized_paper_pnl      NUMERIC,
    error_message           TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sell_decision_id),
    UNIQUE (trade_date, automation_mode, paper_position_id)
);

-- ---------------------------------------------------------------------------
-- 매도 신호 로그
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_sell_signal_log (
    sell_signal_id          TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    paper_position_id       TEXT,
    symbol                  TEXT        NOT NULL,
    rule_name               TEXT,
    rule_result             TEXT,       -- PASS | FAIL | UNKNOWN
    metric_value            TEXT,
    threshold_value         JSONB,
    severity                TEXT,
    detail                  TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sell_signal_id)
);

-- ---------------------------------------------------------------------------
-- 매도 페이퍼 주문
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_paper_sell_order (
    paper_sell_order_id     TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    paper_position_id       TEXT,
    symbol                  TEXT        NOT NULL,
    side                    TEXT        NOT NULL DEFAULT 'SELL',
    sell_action             TEXT,
    sell_ratio              NUMERIC,
    sell_quantity           NUMERIC,
    sell_price_ref          NUMERIC,
    sell_amount             NUMERIC,
    assumed_fill_status     TEXT,
    exit_reason             TEXT,
    source_sell_decision_id TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_sell_order_id),
    UNIQUE (trade_date, paper_position_id, sell_action)
);

-- ---------------------------------------------------------------------------
-- 포지션 스냅샷 (일별)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade.us_paper_position_snapshot (
    snapshot_id             TEXT        NOT NULL,
    snapshot_date           DATE        NOT NULL,
    paper_position_id       TEXT        NOT NULL,
    symbol                  TEXT        NOT NULL,
    latest_price            NUMERIC,
    remaining_quantity      NUMERIC,
    highest_price_since_entry NUMERIC,
    unrealized_pnl          NUMERIC,
    unrealized_pnl_pct      NUMERIC,
    holding_days            INTEGER,
    status                  TEXT,
    data_quality_flags      JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (snapshot_id),
    UNIQUE (snapshot_date, paper_position_id)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_us_buy_candidate_log_date ON trade.us_buy_candidate_log (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_buy_decision_log_date  ON trade.us_buy_decision_log  (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_sell_decision_log_date ON trade.us_sell_decision_log  (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_paper_order_date       ON trade.us_paper_order        (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_paper_sell_order_date  ON trade.us_paper_sell_order   (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_us_pos_snapshot_date      ON trade.us_paper_position_snapshot (snapshot_date DESC);
