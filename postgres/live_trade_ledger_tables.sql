CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.live_trade_decision (
    intent_id text PRIMARY KEY,
    as_of_date date NOT NULL,
    code character varying(10),
    name text,
    source_action text,
    intent_type text NOT NULL,
    target_weight numeric,
    gate_status text,
    reason text,
    priority integer,
    executable boolean DEFAULT false NOT NULL,
    policy_version text,
    score_formula_version text,
    gate_version text,
    portfolio_version text,
    holdings_source text,
    ranking_run_id bigint,
    ranking_rank integer,
    final_score numeric,
    confidence_score numeric,
    risk_penalty numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    liquidity_score numeric,
    safety_score numeric,
    dominant_theme text,
    score_driver_1 text,
    score_driver_2 text,
    score_driver_3 text,
    risk_factor_1 text,
    risk_factor_2 text,
    action_note text,
    payload_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS research.live_order_request (
    request_id text PRIMARY KEY,
    intent_id text,
    as_of_date date,
    generated_at timestamp with time zone,
    gate_status text,
    env_dv text,
    code character varying(10) NOT NULL,
    name text,
    side text NOT NULL,
    intent_type text,
    ord_dvsn text,
    reference_price numeric,
    planned_qty numeric,
    allowed_qty numeric,
    final_request_qty numeric,
    target_weight numeric,
    priority integer,
    reason text,
    blocked_reason text,
    expected_hold_reason text,
    executable_now boolean DEFAULT false NOT NULL,
    ranking_run_id bigint,
    ranking_rank integer,
    final_score numeric,
    confidence_score numeric,
    risk_penalty numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    liquidity_score numeric,
    safety_score numeric,
    dominant_theme text,
    score_driver_1 text,
    score_driver_2 text,
    score_driver_3 text,
    risk_factor_1 text,
    risk_factor_2 text,
    action_note text,
    payload_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS research.live_order_execution (
    execution_id bigserial PRIMARY KEY,
    request_id text NOT NULL,
    intent_id text,
    as_of_date date,
    executed_at timestamp with time zone,
    submitted_at timestamp with time zone,
    gate_status text,
    env_dv text,
    code character varying(10) NOT NULL,
    name text,
    side text NOT NULL,
    intent_type text,
    ord_dvsn text,
    reference_price numeric,
    final_request_qty numeric,
    submission_status text NOT NULL,
    skip_reason text,
    broker_order_id text,
    broker_org_order_id text,
    raw_response_json jsonb,
    payload_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    UNIQUE (request_id, executed_at)
);

CREATE TABLE IF NOT EXISTS research.live_order_fill (
    fill_id bigserial PRIMARY KEY,
    request_id text,
    broker_order_id text,
    broker_org_order_id text,
    as_of_date date,
    filled_at timestamp with time zone,
    code character varying(10) NOT NULL,
    name text,
    side text NOT NULL,
    filled_qty numeric NOT NULL,
    filled_price numeric NOT NULL,
    filled_amount numeric,
    fee numeric,
    tax numeric,
    fill_status text,
    source text DEFAULT 'kis_fill_inquiry'::text NOT NULL,
    raw_response_json jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    UNIQUE (broker_order_id, code, side, filled_at, filled_qty, filled_price)
);

CREATE TABLE IF NOT EXISTS research.live_position_snapshot (
    snapshot_id bigserial PRIMARY KEY,
    snapshot_at timestamp with time zone NOT NULL,
    snapshot_date date NOT NULL,
    env_dv text,
    account_masked text,
    code character varying(10) NOT NULL,
    name text,
    qty numeric NOT NULL,
    avg_price numeric,
    current_price numeric,
    eval_amount numeric,
    pnl_amount numeric,
    pnl_pct numeric,
    weight numeric,
    status text DEFAULT 'OPEN'::text NOT NULL,
    cash_amount numeric,
    total_assets numeric,
    payload_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    UNIQUE (snapshot_at, code)
);

CREATE TABLE IF NOT EXISTS research.live_trade_review (
    review_id bigserial PRIMARY KEY,
    intent_id text,
    request_id text,
    code character varying(10) NOT NULL,
    review_date date DEFAULT CURRENT_DATE NOT NULL,
    pre_tags text[],
    post_tags text[],
    outcome_label text,
    review_note text,
    next_action_note text,
    reviewer text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS ranking_run_id bigint;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS ranking_rank integer;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS final_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS confidence_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS risk_penalty numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS ret_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS prob_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS qual_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS tech_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS liquidity_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS safety_score numeric;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS dominant_theme text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS score_driver_1 text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS score_driver_2 text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS score_driver_3 text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS risk_factor_1 text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS risk_factor_2 text;
ALTER TABLE research.live_trade_decision ADD COLUMN IF NOT EXISTS action_note text;

ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS ranking_run_id bigint;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS ranking_rank integer;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS final_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS confidence_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS risk_penalty numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS ret_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS prob_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS qual_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS tech_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS liquidity_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS safety_score numeric;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS dominant_theme text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS score_driver_1 text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS score_driver_2 text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS score_driver_3 text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS risk_factor_1 text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS risk_factor_2 text;
ALTER TABLE research.live_order_request ADD COLUMN IF NOT EXISTS action_note text;

CREATE INDEX IF NOT EXISTS idx_live_trade_decision_asof_code ON research.live_trade_decision USING btree (as_of_date DESC, code);
CREATE INDEX IF NOT EXISTS idx_live_order_request_asof_code ON research.live_order_request USING btree (as_of_date DESC, code);
CREATE INDEX IF NOT EXISTS idx_live_order_request_intent ON research.live_order_request USING btree (intent_id);
CREATE INDEX IF NOT EXISTS idx_live_order_execution_asof_code ON research.live_order_execution USING btree (as_of_date DESC, code);
CREATE INDEX IF NOT EXISTS idx_live_order_execution_status ON research.live_order_execution USING btree (submission_status, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_order_fill_code_time ON research.live_order_fill USING btree (code, filled_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_order_fill_request ON research.live_order_fill USING btree (request_id);
CREATE INDEX IF NOT EXISTS idx_live_position_snapshot_date_code ON research.live_position_snapshot USING btree (snapshot_date DESC, code);
CREATE INDEX IF NOT EXISTS idx_live_trade_review_code_date ON research.live_trade_review USING btree (code, review_date DESC);
CREATE INDEX IF NOT EXISTS idx_live_trade_review_request ON research.live_trade_review USING btree (request_id);
CREATE INDEX IF NOT EXISTS idx_live_trade_review_intent ON research.live_trade_review USING btree (intent_id);
