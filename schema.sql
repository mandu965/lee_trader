<<<<<<< HEAD
﻿--
-- PostgreSQL database dump
--


-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: research; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA research;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: backtest_trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.backtest_trades (
    id bigint NOT NULL,
    trade_date date NOT NULL,
    strategy text NOT NULL,
    code text NOT NULL,
    final_score numeric,
    pred_return_60d numeric,
    realized_return numeric NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: backtest_trades_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.backtest_trades_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: backtest_trades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.backtest_trades_id_seq OWNED BY public.backtest_trades.id;


--
-- Name: daily_ranking; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.daily_ranking (
    date date NOT NULL,
    code text NOT NULL,
    close numeric,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    score numeric,
    score_score numeric,
    composite numeric,
    quality_score numeric,
    name text,
    market text,
    sector text,
    tech_score numeric,
    pred_score numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    safety_score numeric,
    liquidity_score numeric,
    final_score numeric,
    risk_penalty numeric,
    market_up boolean,
    market_status_date date,
    market_kospi_close numeric,
    market_kospi_ma20 numeric,
    market_vol_5d numeric,
    market_foreign_5d numeric,
    generated_at timestamp with time zone,
    model_version text,
    score_formula_version text,
    rank_final integer,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    prob_score_raw double precision,
    prob_score_missing boolean,
    prob_rank_pct double precision,
    ret_score_missing boolean,
    qual_score_missing boolean,
    tech_score_missing boolean,
    safety_score_missing boolean,
    liquidity_score_missing boolean,
    ret_score_fallback_used boolean,
    prob_score_fallback_used boolean,
    qual_score_fallback_used boolean,
    tech_score_fallback_used boolean,
    safety_score_fallback_used boolean,
    liquidity_score_fallback_used boolean,
    fallback_count integer,
    component_coverage_ratio double precision,
    data_maturity_score double precision,
    model_reliability_score double precision,
    signal_agreement_score double precision,
    regime_fitness_score double precision,
    confidence_label text,
    confidence_reason text,
    confidence_score_research double precision,
    confidence_score_operational double precision,
    confidence_label_research text,
    confidence_label_operational text,
    quality_flag boolean,
    quality_gate_applied boolean,
    quality_penalty_ratio double precision,
    shadow_quality_gate_applied boolean,
    shadow_quality_penalty_ratio double precision,
    shadow_final_score_quality_gate double precision,
    shadow_rank_quality_gate integer,
    quality_gate_experiment text,
    score_explain_summary text,
    score_explain_strengths text,
    score_explain_risks text,
    score_explain_confidence text,
    score_explain_regime text,
    explain text,
    score_driver_1 text,
    score_driver_2 text,
    score_driver_3 text,
    score_drag_1 text,
    score_drag_2 text,
    top_driver_1 text,
    top_driver_2 text,
    top_driver_3 text,
    risk_factor_1 text,
    risk_factor_2 text,
    action_note text,
    score_explain_json text,
    score_contribution_ret double precision,
    score_contribution_prob double precision,
    score_contribution_tech double precision,
    score_contribution_qual double precision,
    score_contribution_safety double precision,
    score_contribution_liquidity double precision,
    score_contribution_theme double precision,
    score_contribution_risk double precision,
    return_score double precision,
    probability_score double precision,
    technical_score double precision,
    valuation_score double precision,
    theme_score double precision,
    dominant_theme text,
    theme_confidence double precision,
    theme_score_effective double precision,
    final_score_before_theme double precision,
    final_score_v2_before_theme double precision,
    final_score_v3 double precision,
    score_diff_v2 double precision,
    score_diff_v3 double precision,
    v3_vs_v2_diff double precision,
    theme_overlay_mode text,
    theme_overlay_anchor text,
    theme_delta_raw double precision,
    theme_overlay_formula text,
    theme_delta_vs_base double precision,
    theme_delta_positive double precision,
    theme_positive_part double precision,
    theme_negative_part double precision,
    theme_overlay_gain double precision,
    theme_overlay_cap double precision,
    theme_overlay_signed_component double precision,
    theme_overlay_positive_component double precision,
    theme_overlay_negative_component double precision,
    theme_overlay_applied double precision,
    theme_overlay_capped boolean,
    theme_overlay_soft_conf_gate double precision,
    theme_uplift_applied boolean,
    theme_penalty_applied boolean,
    shadow_theme_weight_raw double precision,
    shadow_theme_weight double precision,
    shadow_theme_weight_effective double precision,
    shadow_base_weight double precision,
    shadow_floor_applied boolean,
    shadow_theme_score_effective double precision,
    shadow_final_score_v3 double precision,
    shadow_score_diff_v3 double precision,
    shadow_rank_v3 integer,
    shadow_explain text,
    regime_reason text,
    weight_profile text,
    live_rank integer,
    live_score double precision,
    live_score_source text,
    shadow_quality_risk_guard_penalty double precision,
    shadow_quality_risk_guard_applied boolean,
    shadow_final_score_quality_risk_guard double precision,
    shadow_rank_quality_risk_guard integer
);


--
-- Name: daily_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.daily_scores (
    date date NOT NULL,
    code text NOT NULL,
    score numeric,
    composite numeric
);


--
-- Name: etf_holdings_snapshot; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_holdings_snapshot (
    as_of_date date NOT NULL,
    etf_code character varying(12) NOT NULL,
    stock_code text NOT NULL,
    stock_name text,
    holding_weight numeric,
    holding_quantity numeric,
    market_value numeric,
    rank_in_etf integer,
    source_name text NOT NULL,
    collected_at timestamp with time zone DEFAULT now() NOT NULL,
    raw_payload_json jsonb
);


--
-- Name: TABLE etf_holdings_snapshot; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_holdings_snapshot IS 'Daily snapshot of ETF constituent stocks and their portfolio weights.';


--
-- Name: etf_master; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_master (
    etf_code character varying(12) NOT NULL,
    etf_name text NOT NULL,
    market text,
    asset_class text,
    issuer_name text,
    reference_index_name text,
    listing_date date,
    delisting_date date,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE etf_master; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_master IS 'ETF master storing core metadata and listing status used by the theme pipeline.';


--
-- Name: etf_signal_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_signal_daily (
    as_of_date date NOT NULL,
    etf_code character varying(12) NOT NULL,
    close_price numeric,
    nav_price numeric,
    nav_gap_pct numeric,
    return_1d numeric,
    return_5d numeric,
    return_20d numeric,
    volume numeric,
    trading_value numeric,
    aum_amount numeric,
    relative_strength_score numeric,
    signal_score numeric,
    signal_payload_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE etf_signal_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_signal_daily IS 'Daily ETF signal fact table storing price, NAV, and relative-strength metrics.';


--
-- Name: etf_theme_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_theme_map (
    etf_code character varying(12) NOT NULL,
    theme_code text NOT NULL,
    mapping_source text DEFAULT 'manual'::text NOT NULL,
    mapping_confidence numeric,
    is_primary boolean DEFAULT false NOT NULL,
    valid_from date DEFAULT CURRENT_DATE NOT NULL,
    valid_to date,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT etf_theme_map_check CHECK (((valid_to IS NULL) OR (valid_to >= valid_from)))
);


--
-- Name: TABLE etf_theme_map; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_theme_map IS 'Mapping table that manages ETF-to-theme assignments, validity windows, and confidence.';


--
-- Name: fact_price_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fact_price_daily (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    adj_close numeric,
    volume numeric,
    value numeric,
    market_cap numeric,
    listed_shares numeric
);


--
-- Name: features; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.features (
    date date NOT NULL,
    code text NOT NULL,
    close numeric,
    ret_1d numeric,
    ret_5d numeric,
    ret_10d numeric,
    mom_20 numeric,
    ma_5 numeric,
    ma_20 numeric,
    ma_60 numeric,
    close_over_ma20 numeric,
    vol_20 numeric,
    vol_60 numeric,
    rsi_14 numeric,
    volume numeric,
    vol_ma_20 numeric,
    vol_ratio_20 numeric,
    quality_score numeric,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    volume_ratio_5d double precision,
    volume_ratio_20d double precision,
    value_ratio_5d double precision,
    value_ratio_20d double precision,
    volume_score double precision,
    liquidity_score double precision
);


--
-- Name: flow_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.flow_daily (
    date date NOT NULL,
    code character varying(6) NOT NULL,
    investor_type character varying(32) NOT NULL,
    net_buy_amount numeric,
    net_buy_volume numeric,
    raw_payload_hash character varying(64) NOT NULL,
    collected_at timestamp with time zone NOT NULL,
    source_endpoint character varying(128) NOT NULL,
    market_div_code character varying(8) NOT NULL,
    input_date character varying(8) NOT NULL,
    tr_id character varying(32) NOT NULL,
    raw_payload_json jsonb,
    created_run_id bigint,
    fetch_status character varying(32),
    error_code character varying(64),
    error_message text,
    is_partial_page boolean DEFAULT false NOT NULL
);


--
-- Name: fundamentals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fundamentals (
    date date NOT NULL,
    code text NOT NULL,
    roe numeric,
    op_margin numeric,
    debt_ratio numeric,
    ocf_to_assets numeric,
    net_margin numeric
);


--
-- Name: labels; Type: TABLE; Schema: public; Owner: -
--

=======
﻿--
-- PostgreSQL database dump
--


-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: research; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA research;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: backtest_trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.backtest_trades (
    id bigint NOT NULL,
    trade_date date NOT NULL,
    strategy text NOT NULL,
    code text NOT NULL,
    final_score numeric,
    pred_return_60d numeric,
    realized_return numeric NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: backtest_trades_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.backtest_trades_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: backtest_trades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.backtest_trades_id_seq OWNED BY public.backtest_trades.id;


--
-- Name: daily_ranking; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.daily_ranking (
    date date NOT NULL,
    code text NOT NULL,
    close numeric,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    score numeric,
    score_score numeric,
    composite numeric,
    quality_score numeric,
    name text,
    market text,
    sector text,
    tech_score numeric,
    pred_score numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    safety_score numeric,
    liquidity_score numeric,
    final_score numeric,
    risk_penalty numeric,
    market_up boolean,
    market_status_date date,
    market_kospi_close numeric,
    market_kospi_ma20 numeric,
    market_vol_5d numeric,
    market_foreign_5d numeric,
    generated_at timestamp with time zone,
    model_version text,
    score_formula_version text,
    rank_final integer,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    prob_score_raw double precision,
    prob_score_missing boolean,
    prob_rank_pct double precision,
    ret_score_missing boolean,
    qual_score_missing boolean,
    tech_score_missing boolean,
    safety_score_missing boolean,
    liquidity_score_missing boolean,
    ret_score_fallback_used boolean,
    prob_score_fallback_used boolean,
    qual_score_fallback_used boolean,
    tech_score_fallback_used boolean,
    safety_score_fallback_used boolean,
    liquidity_score_fallback_used boolean,
    fallback_count integer,
    component_coverage_ratio double precision,
    data_maturity_score double precision,
    model_reliability_score double precision,
    signal_agreement_score double precision,
    regime_fitness_score double precision,
    confidence_label text,
    confidence_reason text,
    confidence_score_research double precision,
    confidence_score_operational double precision,
    confidence_label_research text,
    confidence_label_operational text,
    quality_flag boolean,
    quality_gate_applied boolean,
    quality_penalty_ratio double precision,
    shadow_quality_gate_applied boolean,
    shadow_quality_penalty_ratio double precision,
    shadow_final_score_quality_gate double precision,
    shadow_rank_quality_gate integer,
    quality_gate_experiment text,
    score_explain_summary text,
    score_explain_strengths text,
    score_explain_risks text,
    score_explain_confidence text,
    score_explain_regime text,
    explain text,
    score_driver_1 text,
    score_driver_2 text,
    score_driver_3 text,
    score_drag_1 text,
    score_drag_2 text,
    top_driver_1 text,
    top_driver_2 text,
    top_driver_3 text,
    risk_factor_1 text,
    risk_factor_2 text,
    action_note text,
    score_explain_json text,
    score_contribution_ret double precision,
    score_contribution_prob double precision,
    score_contribution_tech double precision,
    score_contribution_qual double precision,
    score_contribution_safety double precision,
    score_contribution_liquidity double precision,
    score_contribution_theme double precision,
    score_contribution_risk double precision,
    return_score double precision,
    probability_score double precision,
    technical_score double precision,
    valuation_score double precision,
    theme_score double precision,
    dominant_theme text,
    theme_confidence double precision,
    theme_score_effective double precision,
    final_score_before_theme double precision,
    final_score_v2_before_theme double precision,
    final_score_v3 double precision,
    score_diff_v2 double precision,
    score_diff_v3 double precision,
    v3_vs_v2_diff double precision,
    theme_overlay_mode text,
    theme_overlay_anchor text,
    theme_delta_raw double precision,
    theme_overlay_formula text,
    theme_delta_vs_base double precision,
    theme_delta_positive double precision,
    theme_positive_part double precision,
    theme_negative_part double precision,
    theme_overlay_gain double precision,
    theme_overlay_cap double precision,
    theme_overlay_signed_component double precision,
    theme_overlay_positive_component double precision,
    theme_overlay_negative_component double precision,
    theme_overlay_applied double precision,
    theme_overlay_capped boolean,
    theme_overlay_soft_conf_gate double precision,
    theme_uplift_applied boolean,
    theme_penalty_applied boolean,
    shadow_theme_weight_raw double precision,
    shadow_theme_weight double precision,
    shadow_theme_weight_effective double precision,
    shadow_base_weight double precision,
    shadow_floor_applied boolean,
    shadow_theme_score_effective double precision,
    shadow_final_score_v3 double precision,
    shadow_score_diff_v3 double precision,
    shadow_rank_v3 integer,
    shadow_explain text,
    regime_reason text,
    weight_profile text,
    live_rank integer,
    live_score double precision,
    live_score_source text,
    shadow_quality_risk_guard_penalty double precision,
    shadow_quality_risk_guard_applied boolean,
    shadow_final_score_quality_risk_guard double precision,
    shadow_rank_quality_risk_guard integer
);


--
-- Name: daily_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.daily_scores (
    date date NOT NULL,
    code text NOT NULL,
    score numeric,
    composite numeric
);


--
-- Name: etf_holdings_snapshot; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_holdings_snapshot (
    as_of_date date NOT NULL,
    etf_code character varying(12) NOT NULL,
    stock_code text NOT NULL,
    stock_name text,
    holding_weight numeric,
    holding_quantity numeric,
    market_value numeric,
    rank_in_etf integer,
    source_name text NOT NULL,
    collected_at timestamp with time zone DEFAULT now() NOT NULL,
    raw_payload_json jsonb
);


--
-- Name: TABLE etf_holdings_snapshot; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_holdings_snapshot IS 'Daily snapshot of ETF constituent stocks and their portfolio weights.';


--
-- Name: etf_master; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_master (
    etf_code character varying(12) NOT NULL,
    etf_name text NOT NULL,
    market text,
    asset_class text,
    issuer_name text,
    reference_index_name text,
    listing_date date,
    delisting_date date,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE etf_master; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_master IS 'ETF master storing core metadata and listing status used by the theme pipeline.';


--
-- Name: etf_signal_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_signal_daily (
    as_of_date date NOT NULL,
    etf_code character varying(12) NOT NULL,
    close_price numeric,
    nav_price numeric,
    nav_gap_pct numeric,
    return_1d numeric,
    return_5d numeric,
    return_20d numeric,
    volume numeric,
    trading_value numeric,
    aum_amount numeric,
    relative_strength_score numeric,
    signal_score numeric,
    signal_payload_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE etf_signal_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_signal_daily IS 'Daily ETF signal fact table storing price, NAV, and relative-strength metrics.';


--
-- Name: etf_theme_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.etf_theme_map (
    etf_code character varying(12) NOT NULL,
    theme_code text NOT NULL,
    mapping_source text DEFAULT 'manual'::text NOT NULL,
    mapping_confidence numeric,
    is_primary boolean DEFAULT false NOT NULL,
    valid_from date DEFAULT CURRENT_DATE NOT NULL,
    valid_to date,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT etf_theme_map_check CHECK (((valid_to IS NULL) OR (valid_to >= valid_from)))
);


--
-- Name: TABLE etf_theme_map; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.etf_theme_map IS 'Mapping table that manages ETF-to-theme assignments, validity windows, and confidence.';


--
-- Name: fact_price_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fact_price_daily (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    adj_close numeric,
    volume numeric,
    value numeric,
    market_cap numeric,
    listed_shares numeric
);


--
-- Name: features; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.features (
    date date NOT NULL,
    code text NOT NULL,
    close numeric,
    ret_1d numeric,
    ret_5d numeric,
    ret_10d numeric,
    mom_20 numeric,
    ma_5 numeric,
    ma_20 numeric,
    ma_60 numeric,
    close_over_ma20 numeric,
    vol_20 numeric,
    vol_60 numeric,
    rsi_14 numeric,
    volume numeric,
    vol_ma_20 numeric,
    vol_ratio_20 numeric,
    quality_score numeric,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    volume_ratio_5d double precision,
    volume_ratio_20d double precision,
    value_ratio_5d double precision,
    value_ratio_20d double precision,
    volume_score double precision,
    liquidity_score double precision
);


--
-- Name: flow_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.flow_daily (
    date date NOT NULL,
    code character varying(6) NOT NULL,
    investor_type character varying(32) NOT NULL,
    net_buy_amount numeric,
    net_buy_volume numeric,
    raw_payload_hash character varying(64) NOT NULL,
    collected_at timestamp with time zone NOT NULL,
    source_endpoint character varying(128) NOT NULL,
    market_div_code character varying(8) NOT NULL,
    input_date character varying(8) NOT NULL,
    tr_id character varying(32) NOT NULL,
    raw_payload_json jsonb,
    created_run_id bigint,
    fetch_status character varying(32),
    error_code character varying(64),
    error_message text,
    is_partial_page boolean DEFAULT false NOT NULL
);


--
-- Name: fundamentals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fundamentals (
    date date NOT NULL,
    code text NOT NULL,
    roe numeric,
    op_margin numeric,
    debt_ratio numeric,
    ocf_to_assets numeric,
    net_margin numeric
);


--
-- Name: labels; Type: TABLE; Schema: public; Owner: -
--

>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
CREATE TABLE public.labels (
    date date NOT NULL,
    code text NOT NULL,
    target_30d numeric,
    target_60d numeric,
    target_90d numeric,
    target_log_30d numeric,
    target_log_60d numeric,
    target_log_90d numeric,
    target_mdd_30d numeric,
    target_mdd_60d numeric,
    target_mdd_90d numeric,
    target_30d_top20 numeric,
    target_60d_top20 numeric,
    target_90d_top20 numeric,
    realized_return_30d numeric,
    realized_return_60d numeric,
    realized_return_90d numeric
);
<<<<<<< HEAD


--
-- Name: market_status; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.market_status (
    date date NOT NULL,
    kospi_close numeric,
    kospi_ma20 numeric,
    volatility_5d numeric,
    foreign_net_5d numeric,
    market_up boolean
);


--
-- Name: page_view_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.page_view_events (
    id bigint NOT NULL,
    visitor_id text NOT NULL,
    path text NOT NULL,
    referrer text,
    user_agent text,
    ip_hash text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: page_view_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.page_view_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: page_view_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.page_view_events_id_seq OWNED BY public.page_view_events.id;


--
-- Name: pipeline_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_history (
    id bigint NOT NULL,
    run_id text NOT NULL,
    step text NOT NULL,
    status text NOT NULL,
    duration_s numeric,
    message text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: pipeline_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pipeline_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_history_id_seq OWNED BY public.pipeline_history.id;


--
-- Name: predictions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.predictions (
    date date NOT NULL,
    code text NOT NULL,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    score numeric
);


--
-- Name: prices_adjusted; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_adjusted (
    date date NOT NULL,
    code text NOT NULL,
    adj_open numeric,
    adj_high numeric,
    adj_low numeric,
    adj_close numeric,
    volume numeric
);


--
-- Name: prices_clean; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_clean (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric
);


--
-- Name: prices_raw; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_raw (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric
);


--
-- Name: quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.quality (
    date date NOT NULL,
    code text NOT NULL,
    quality_score numeric,
    roe double precision,
    op_margin double precision,
    current_ratio double precision,
    debt_ratio double precision,
    ocf_to_assets double precision,
    quality_raw_score double precision,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    net_margin double precision
);


--
-- Name: stock_theme_exposure_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stock_theme_exposure_daily (
    as_of_date date NOT NULL,
    stock_code text NOT NULL,
    theme_code text NOT NULL,
    exposure_score numeric NOT NULL,
    exposure_weight numeric,
    supporting_etf_count integer DEFAULT 0 NOT NULL,
    primary_etf_code character varying(12),
    calc_version text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE stock_theme_exposure_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.stock_theme_exposure_daily IS 'Daily stock-level theme exposure propagated from ETF holdings snapshots.';


--
-- Name: stocks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stocks (
    code text NOT NULL,
    name text NOT NULL,
    market text,
    sector text,
    listed_at date,
    delisted_at date
);


--
-- Name: theme_master; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theme_master (
    theme_code text NOT NULL,
    theme_name text NOT NULL,
    theme_group text,
    theme_description text,
    display_order integer,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE theme_master; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.theme_master IS 'Theme master used as the canonical dictionary for ETF-based theme classification.';


--
-- Name: theme_score_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theme_score_daily (
    as_of_date date NOT NULL,
    theme_code text NOT NULL,
    theme_score numeric NOT NULL,
    signal_score numeric,
    breadth_count integer DEFAULT 0 NOT NULL,
    leader_etf_code character varying(12),
    leader_stock_code text,
    calc_version text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE theme_score_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.theme_score_daily IS 'Daily final theme score table with representative ETF and stock references.';


--
-- Name: trade_audit_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trade_audit_log (
    audit_id bigint NOT NULL,
    trade_id bigint,
    action text NOT NULL,
    trade_snapshot jsonb NOT NULL,
    actor text,
    reason text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: trade_audit_log_audit_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.trade_audit_log_audit_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: trade_audit_log_audit_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.trade_audit_log_audit_id_seq OWNED BY public.trade_audit_log.audit_id;


--
-- Name: trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trades (
    trade_id bigint NOT NULL,
    date date NOT NULL,
    side text NOT NULL,
    code text NOT NULL,
    name text,
    market text,
    sector text,
    qty numeric,
    price numeric,
    amount numeric,
    fee numeric,
    memo text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: trades_trade_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.trades_trade_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: trades_trade_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.trades_trade_id_seq OWNED BY public.trades.trade_id;


--
-- Name: app_payload_store; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.app_payload_store (
    payload_key text NOT NULL,
    payload_json jsonb NOT NULL,
    asof_date date,
    generated_at timestamp with time zone,
    source_path text,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: backtest_outcome; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.backtest_outcome (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    horizon_days integer NOT NULL,
    realized_return numeric,
    realized_mdd numeric,
    label_source character varying(32),
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: dim_model_run; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.dim_model_run (
    run_id bigint NOT NULL,
    run_type character varying(32) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    top_n integer DEFAULT 20,
    train_start_date date,
    train_end_date date,
    config_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    comment text
);


--
-- Name: dim_model_run_run_id_seq; Type: SEQUENCE; Schema: research; Owner: -
--

CREATE SEQUENCE research.dim_model_run_run_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: dim_model_run_run_id_seq; Type: SEQUENCE OWNED BY; Schema: research; Owner: -
--

ALTER SEQUENCE research.dim_model_run_run_id_seq OWNED BY research.dim_model_run.run_id;


--
-- Name: paper_trading_nav; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_nav (
    paper_run_id bigint NOT NULL,
    strategy text NOT NULL,
    date date NOT NULL,
    cash numeric,
    market_value numeric,
    nav numeric,
    daily_return numeric,
    active_position_count integer,
    opened_today integer,
    duplicate_skip_count integer,
    deployed_cash numeric,
    cumulative_return numeric,
    running_nav_max numeric,
    drawdown numeric,
    closed_trade_count integer,
    closed_win_rate numeric,
    closed_win_count integer,
    closed_trade_count_cum integer,
    closed_win_count_cum integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_position; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_position (
    paper_run_id bigint NOT NULL,
    strategy text NOT NULL,
    code character varying(10) NOT NULL,
    name text,
    entry_date date NOT NULL,
    planned_exit_date date,
    exit_date date,
    entry_price_close numeric,
    entry_exec_price numeric,
    exit_price_close numeric,
    exit_exec_price numeric,
    shares numeric,
    entry_notional_gross numeric,
    exit_notional_net numeric,
    entry_cost_amount numeric,
    exit_cost_amount numeric,
    gross_return numeric,
    net_return numeric,
    source_rank integer,
    selection_stage text,
    dominant_theme text,
    confidence_score numeric,
    final_score numeric,
    status text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_run; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_run (
    paper_run_id bigint NOT NULL,
    run_tag text NOT NULL,
    source_mode text NOT NULL,
    asof_date date,
    hold_days integer NOT NULL,
    initial_nav numeric,
    entry_fee_bps numeric,
    exit_fee_bps numeric,
    entry_slippage_bps numeric,
    exit_slippage_bps numeric,
    positions_row_count integer DEFAULT 0 NOT NULL,
    nav_row_count integer DEFAULT 0 NOT NULL,
    source_positions_csv text,
    source_nav_csv text,
    source_report_md text,
    comment text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_run_paper_run_id_seq; Type: SEQUENCE; Schema: research; Owner: -
--

CREATE SEQUENCE research.paper_trading_run_paper_run_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: paper_trading_run_paper_run_id_seq; Type: SEQUENCE OWNED BY; Schema: research; Owner: -
--

ALTER SEQUENCE research.paper_trading_run_paper_run_id_seq OWNED BY research.paper_trading_run.paper_run_id;


--
-- Name: prediction_history; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.prediction_history (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    risk_penalty numeric,
    final_score numeric,
    final_score_custom numeric,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: ranking_history; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.ranking_history (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    rank integer NOT NULL,
    final_score numeric NOT NULL,
    score_formula_version text,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    risk_penalty numeric,
    in_top_n boolean DEFAULT false NOT NULL,
    top_n integer NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: backtest_trades id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backtest_trades ALTER COLUMN id SET DEFAULT nextval('public.backtest_trades_id_seq'::regclass);


--
-- Name: page_view_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.page_view_events ALTER COLUMN id SET DEFAULT nextval('public.page_view_events_id_seq'::regclass);


--
-- Name: pipeline_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_history ALTER COLUMN id SET DEFAULT nextval('public.pipeline_history_id_seq'::regclass);


--
-- Name: trade_audit_log audit_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trade_audit_log ALTER COLUMN audit_id SET DEFAULT nextval('public.trade_audit_log_audit_id_seq'::regclass);


--
-- Name: trades trade_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trades ALTER COLUMN trade_id SET DEFAULT nextval('public.trades_trade_id_seq'::regclass);


--
-- Name: dim_model_run run_id; Type: DEFAULT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.dim_model_run ALTER COLUMN run_id SET DEFAULT nextval('research.dim_model_run_run_id_seq'::regclass);


--
-- Name: paper_trading_run paper_run_id; Type: DEFAULT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run ALTER COLUMN paper_run_id SET DEFAULT nextval('research.paper_trading_run_paper_run_id_seq'::regclass);


--
-- Name: backtest_trades backtest_trades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backtest_trades
    ADD CONSTRAINT backtest_trades_pkey PRIMARY KEY (id);


--
-- Name: daily_ranking daily_ranking_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daily_ranking
    ADD CONSTRAINT daily_ranking_pkey PRIMARY KEY (date, code);


--
-- Name: daily_scores daily_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daily_scores
    ADD CONSTRAINT daily_scores_pkey PRIMARY KEY (date, code);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_pkey PRIMARY KEY (as_of_date, etf_code, stock_code);


--
-- Name: etf_master etf_master_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_master
    ADD CONSTRAINT etf_master_pkey PRIMARY KEY (etf_code);


--
-- Name: etf_signal_daily etf_signal_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_signal_daily
    ADD CONSTRAINT etf_signal_daily_pkey PRIMARY KEY (as_of_date, etf_code);


--
-- Name: etf_theme_map etf_theme_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_pkey PRIMARY KEY (etf_code, theme_code, valid_from);


--
-- Name: fact_price_daily fact_price_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fact_price_daily
    ADD CONSTRAINT fact_price_daily_pkey PRIMARY KEY (date, code);


--
-- Name: features features_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features
    ADD CONSTRAINT features_pkey PRIMARY KEY (date, code);


--
-- Name: flow_daily flow_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.flow_daily
    ADD CONSTRAINT flow_daily_pkey PRIMARY KEY (date, code, investor_type);


--
-- Name: fundamentals fundamentals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fundamentals
    ADD CONSTRAINT fundamentals_pkey PRIMARY KEY (date, code);


--
-- Name: labels labels_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.labels
    ADD CONSTRAINT labels_pkey PRIMARY KEY (date, code);


--
-- Name: market_status market_status_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.market_status
    ADD CONSTRAINT market_status_pkey PRIMARY KEY (date);


--
-- Name: page_view_events page_view_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.page_view_events
    ADD CONSTRAINT page_view_events_pkey PRIMARY KEY (id);


--
-- Name: pipeline_history pipeline_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_history
    ADD CONSTRAINT pipeline_history_pkey PRIMARY KEY (id);


--
-- Name: predictions predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (date, code);


--
-- Name: prices_adjusted prices_adjusted_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_adjusted
    ADD CONSTRAINT prices_adjusted_pkey PRIMARY KEY (date, code);


--
-- Name: prices_clean prices_clean_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_clean
    ADD CONSTRAINT prices_clean_pkey PRIMARY KEY (date, code);


--
-- Name: prices_raw prices_raw_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_raw
    ADD CONSTRAINT prices_raw_pkey PRIMARY KEY (date, code);


--
-- Name: quality quality_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.quality
    ADD CONSTRAINT quality_pkey PRIMARY KEY (date, code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_pkey PRIMARY KEY (as_of_date, stock_code, theme_code);


--
-- Name: stocks stocks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stocks
    ADD CONSTRAINT stocks_pkey PRIMARY KEY (code);


--
-- Name: theme_master theme_master_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_master
    ADD CONSTRAINT theme_master_pkey PRIMARY KEY (theme_code);


--
-- Name: theme_master theme_master_theme_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_master
    ADD CONSTRAINT theme_master_theme_name_key UNIQUE (theme_name);


--
-- Name: theme_score_daily theme_score_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_pkey PRIMARY KEY (as_of_date, theme_code);


--
-- Name: trade_audit_log trade_audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trade_audit_log
    ADD CONSTRAINT trade_audit_log_pkey PRIMARY KEY (audit_id);


--
-- Name: trades trades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trades
    ADD CONSTRAINT trades_pkey PRIMARY KEY (trade_id);


--
-- Name: app_payload_store app_payload_store_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.app_payload_store
    ADD CONSTRAINT app_payload_store_pkey PRIMARY KEY (payload_key);


--
-- Name: backtest_outcome backtest_outcome_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.backtest_outcome
    ADD CONSTRAINT backtest_outcome_pkey PRIMARY KEY (run_id, as_of_date, code, horizon_days);


--
-- Name: dim_model_run dim_model_run_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.dim_model_run
    ADD CONSTRAINT dim_model_run_pkey PRIMARY KEY (run_id);


--
-- Name: paper_trading_nav paper_trading_nav_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_nav
    ADD CONSTRAINT paper_trading_nav_pkey PRIMARY KEY (paper_run_id, strategy, date);


--
-- Name: paper_trading_position paper_trading_position_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_position
    ADD CONSTRAINT paper_trading_position_pkey PRIMARY KEY (paper_run_id, strategy, code, entry_date);


--
-- Name: paper_trading_run paper_trading_run_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run
    ADD CONSTRAINT paper_trading_run_pkey PRIMARY KEY (paper_run_id);


--
-- Name: paper_trading_run paper_trading_run_run_tag_key; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run
    ADD CONSTRAINT paper_trading_run_run_tag_key UNIQUE (run_tag);


--
-- Name: prediction_history prediction_history_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.prediction_history
    ADD CONSTRAINT prediction_history_pkey PRIMARY KEY (run_id, as_of_date, code, horizon_days);


--
-- Name: ranking_history ranking_history_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.ranking_history
    ADD CONSTRAINT ranking_history_pkey PRIMARY KEY (run_id, as_of_date, code);


--
-- Name: idx_backtest_trades_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_backtest_trades_code ON public.backtest_trades USING btree (code);


--
-- Name: idx_backtest_trades_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_backtest_trades_date ON public.backtest_trades USING btree (trade_date);


--
-- Name: idx_daily_ranking_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_ranking_code_date ON public.daily_ranking USING btree (code, date);


--
-- Name: idx_daily_ranking_date_final; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_ranking_date_final ON public.daily_ranking USING btree (date, final_score DESC);


--
-- Name: idx_daily_scores_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_scores_code_date ON public.daily_scores USING btree (code, date);


--
-- Name: idx_etf_holdings_snapshot_asof_etf_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_asof_etf_code ON public.etf_holdings_snapshot USING btree (as_of_date, etf_code);


--
-- Name: idx_etf_holdings_snapshot_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_etf_code_asof ON public.etf_holdings_snapshot USING btree (etf_code, as_of_date DESC);


--
-- Name: idx_etf_holdings_snapshot_stock_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_stock_code_asof ON public.etf_holdings_snapshot USING btree (stock_code, as_of_date DESC);


--
-- Name: idx_etf_master_is_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_master_is_active ON public.etf_master USING btree (is_active, etf_code);


--
-- Name: idx_etf_master_listing_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_master_listing_date ON public.etf_master USING btree (listing_date DESC, etf_code);


--
-- Name: idx_etf_signal_daily_asof_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_signal_daily_asof_score ON public.etf_signal_daily USING btree (as_of_date, signal_score DESC, etf_code);


--
-- Name: idx_etf_signal_daily_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_signal_daily_etf_code_asof ON public.etf_signal_daily USING btree (etf_code, as_of_date DESC);


--
-- Name: idx_etf_theme_map_etf_code_valid_to; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_theme_map_etf_code_valid_to ON public.etf_theme_map USING btree (etf_code, valid_to, theme_code);


--
-- Name: idx_etf_theme_map_theme_code_valid_from; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_theme_map_theme_code_valid_from ON public.etf_theme_map USING btree (theme_code, valid_from DESC, etf_code);


--
-- Name: idx_fact_price_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_fact_price_code_date ON public.fact_price_daily USING btree (code, date);


--
-- Name: idx_features_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_features_code_date ON public.features USING btree (code, date);


--
-- Name: idx_flow_daily_code_date_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_flow_daily_code_date_desc ON public.flow_daily USING btree (code, date DESC);


--
-- Name: idx_flow_daily_date_investor_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_flow_daily_date_investor_type ON public.flow_daily USING btree (date, investor_type);


--
-- Name: idx_fundamentals_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_fundamentals_code_date ON public.fundamentals USING btree (code, date);


--
-- Name: idx_labels_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_labels_code_date ON public.labels USING btree (code, date);


--
-- Name: idx_market_status_date_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_market_status_date_desc ON public.market_status USING btree (date DESC);


--
-- Name: idx_page_view_events_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_created_at ON public.page_view_events USING btree (created_at DESC);


--
-- Name: idx_page_view_events_path_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_path_created_at ON public.page_view_events USING btree (path, created_at DESC);


--
-- Name: idx_page_view_events_visitor_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_visitor_created_at ON public.page_view_events USING btree (visitor_id, created_at DESC);


--
-- Name: idx_pipeline_history_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_history_created ON public.pipeline_history USING btree (created_at DESC);


--
-- Name: idx_pipeline_history_run_step; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_history_run_step ON public.pipeline_history USING btree (run_id, step);


--
-- Name: idx_predictions_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_predictions_code_date ON public.predictions USING btree (code, date);


--
-- Name: idx_prices_adjusted_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_adjusted_code_date ON public.prices_adjusted USING btree (code, date);


--
-- Name: idx_prices_clean_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_clean_code_date ON public.prices_clean USING btree (code, date);


--
-- Name: idx_prices_raw_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_raw_code_date ON public.prices_raw USING btree (code, date);


--
-- Name: idx_quality_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_quality_code_date ON public.quality USING btree (code, date);


--
-- Name: idx_stock_theme_exposure_daily_asof_stock_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_asof_stock_code ON public.stock_theme_exposure_daily USING btree (as_of_date, stock_code);


--
-- Name: idx_stock_theme_exposure_daily_asof_theme_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_asof_theme_score ON public.stock_theme_exposure_daily USING btree (as_of_date, theme_code, exposure_score DESC, stock_code);


--
-- Name: idx_stock_theme_exposure_daily_primary_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_primary_etf_code_asof ON public.stock_theme_exposure_daily USING btree (primary_etf_code, as_of_date DESC);


--
-- Name: idx_stock_theme_exposure_daily_stock_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_stock_code_asof ON public.stock_theme_exposure_daily USING btree (stock_code, as_of_date DESC);


--
-- Name: idx_stock_theme_exposure_daily_theme_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_theme_code_asof ON public.stock_theme_exposure_daily USING btree (theme_code, as_of_date DESC);


--
-- Name: idx_theme_master_is_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_master_is_active ON public.theme_master USING btree (is_active, display_order, theme_code);


--
-- Name: idx_theme_score_daily_asof_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_asof_score ON public.theme_score_daily USING btree (as_of_date, theme_score DESC, theme_code);


--
-- Name: idx_theme_score_daily_leader_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_leader_etf_code_asof ON public.theme_score_daily USING btree (leader_etf_code, as_of_date DESC);


--
-- Name: idx_theme_score_daily_theme_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_theme_code_asof ON public.theme_score_daily USING btree (theme_code, as_of_date DESC);


--
-- Name: idx_trade_audit_trade_id_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trade_audit_trade_id_created ON public.trade_audit_log USING btree (trade_id, created_at DESC);


--
-- Name: idx_trades_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_code ON public.trades USING btree (code);


--
-- Name: idx_trades_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_code_date ON public.trades USING btree (code, date);


--
-- Name: idx_trades_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_date ON public.trades USING btree (date);


--
-- Name: idx_app_payload_store_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_app_payload_store_asof ON research.app_payload_store USING btree (asof_date DESC, updated_at DESC);


--
-- Name: idx_dim_model_run_type; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_dim_model_run_type ON research.dim_model_run USING btree (run_type);


--
-- Name: idx_paper_trading_nav_lookup; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_nav_lookup ON research.paper_trading_nav USING btree (strategy, date DESC);


--
-- Name: idx_paper_trading_position_lookup; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_position_lookup ON research.paper_trading_position USING btree (strategy, entry_date DESC, code);


--
-- Name: idx_paper_trading_run_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_run_asof ON research.paper_trading_run USING btree (asof_date DESC, paper_run_id DESC);


--
-- Name: idx_pred_hist_asof_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_pred_hist_asof_model ON research.prediction_history USING btree (as_of_date, model_version);


--
-- Name: idx_pred_hist_code_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_pred_hist_code_model ON research.prediction_history USING btree (code, model_version);


--
-- Name: idx_rank_hist_asof_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_rank_hist_asof_model ON research.ranking_history USING btree (as_of_date, model_version);


--
-- Name: idx_rank_hist_run_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_rank_hist_run_asof ON research.ranking_history USING btree (run_id, as_of_date, rank);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_stock_code_fkey FOREIGN KEY (stock_code) REFERENCES public.stocks(code);


--
-- Name: etf_signal_daily etf_signal_daily_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_signal_daily
    ADD CONSTRAINT etf_signal_daily_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_theme_map etf_theme_map_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_theme_map etf_theme_map_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_primary_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_primary_etf_code_fkey FOREIGN KEY (primary_etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_stock_code_fkey FOREIGN KEY (stock_code) REFERENCES public.stocks(code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: theme_score_daily theme_score_daily_leader_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_leader_etf_code_fkey FOREIGN KEY (leader_etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: theme_score_daily theme_score_daily_leader_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_leader_stock_code_fkey FOREIGN KEY (leader_stock_code) REFERENCES public.stocks(code);


--
-- Name: theme_score_daily theme_score_daily_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: backtest_outcome backtest_outcome_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.backtest_outcome
    ADD CONSTRAINT backtest_outcome_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- Name: paper_trading_nav paper_trading_nav_paper_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_nav
    ADD CONSTRAINT paper_trading_nav_paper_run_id_fkey FOREIGN KEY (paper_run_id) REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE;


--
-- Name: paper_trading_position paper_trading_position_paper_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_position
    ADD CONSTRAINT paper_trading_position_paper_run_id_fkey FOREIGN KEY (paper_run_id) REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE;


--
-- Name: prediction_history prediction_history_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.prediction_history
    ADD CONSTRAINT prediction_history_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- Name: ranking_history ranking_history_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.ranking_history
    ADD CONSTRAINT ranking_history_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- PostgreSQL database dump complete
--


=======


--
-- Name: market_status; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.market_status (
    date date NOT NULL,
    kospi_close numeric,
    kospi_ma20 numeric,
    volatility_5d numeric,
    foreign_net_5d numeric,
    market_up boolean
);


--
-- Name: page_view_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.page_view_events (
    id bigint NOT NULL,
    visitor_id text NOT NULL,
    path text NOT NULL,
    referrer text,
    user_agent text,
    ip_hash text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: page_view_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.page_view_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: page_view_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.page_view_events_id_seq OWNED BY public.page_view_events.id;


--
-- Name: pipeline_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_history (
    id bigint NOT NULL,
    run_id text NOT NULL,
    step text NOT NULL,
    status text NOT NULL,
    duration_s numeric,
    message text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: pipeline_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pipeline_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_history_id_seq OWNED BY public.pipeline_history.id;


--
-- Name: predictions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.predictions (
    date date NOT NULL,
    code text NOT NULL,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    score numeric
);


--
-- Name: prices_adjusted; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_adjusted (
    date date NOT NULL,
    code text NOT NULL,
    adj_open numeric,
    adj_high numeric,
    adj_low numeric,
    adj_close numeric,
    volume numeric
);


--
-- Name: prices_clean; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_clean (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric
);


--
-- Name: prices_raw; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices_raw (
    date date NOT NULL,
    code text NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    volume numeric
);


--
-- Name: quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.quality (
    date date NOT NULL,
    code text NOT NULL,
    quality_score numeric,
    roe double precision,
    op_margin double precision,
    current_ratio double precision,
    debt_ratio double precision,
    ocf_to_assets double precision,
    quality_raw_score double precision,
    quality_factor_count integer,
    quality_missing_ratio double precision,
    quality_score_confidence double precision,
    net_margin double precision
);


--
-- Name: stock_theme_exposure_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stock_theme_exposure_daily (
    as_of_date date NOT NULL,
    stock_code text NOT NULL,
    theme_code text NOT NULL,
    exposure_score numeric NOT NULL,
    exposure_weight numeric,
    supporting_etf_count integer DEFAULT 0 NOT NULL,
    primary_etf_code character varying(12),
    calc_version text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE stock_theme_exposure_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.stock_theme_exposure_daily IS 'Daily stock-level theme exposure propagated from ETF holdings snapshots.';


--
-- Name: stocks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stocks (
    code text NOT NULL,
    name text NOT NULL,
    market text,
    sector text,
    listed_at date,
    delisted_at date
);


--
-- Name: theme_master; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theme_master (
    theme_code text NOT NULL,
    theme_name text NOT NULL,
    theme_group text,
    theme_description text,
    display_order integer,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE theme_master; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.theme_master IS 'Theme master used as the canonical dictionary for ETF-based theme classification.';


--
-- Name: theme_score_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theme_score_daily (
    as_of_date date NOT NULL,
    theme_code text NOT NULL,
    theme_score numeric NOT NULL,
    signal_score numeric,
    breadth_count integer DEFAULT 0 NOT NULL,
    leader_etf_code character varying(12),
    leader_stock_code text,
    calc_version text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE theme_score_daily; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.theme_score_daily IS 'Daily final theme score table with representative ETF and stock references.';


--
-- Name: trade_audit_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trade_audit_log (
    audit_id bigint NOT NULL,
    trade_id bigint,
    action text NOT NULL,
    trade_snapshot jsonb NOT NULL,
    actor text,
    reason text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: trade_audit_log_audit_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.trade_audit_log_audit_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: trade_audit_log_audit_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.trade_audit_log_audit_id_seq OWNED BY public.trade_audit_log.audit_id;


--
-- Name: trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trades (
    trade_id bigint NOT NULL,
    date date NOT NULL,
    side text NOT NULL,
    code text NOT NULL,
    name text,
    market text,
    sector text,
    qty numeric,
    price numeric,
    amount numeric,
    fee numeric,
    memo text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: trades_trade_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.trades_trade_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: trades_trade_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.trades_trade_id_seq OWNED BY public.trades.trade_id;


--
-- Name: app_payload_store; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.app_payload_store (
    payload_key text NOT NULL,
    payload_json jsonb NOT NULL,
    asof_date date,
    generated_at timestamp with time zone,
    source_path text,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: backtest_outcome; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.backtest_outcome (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    horizon_days integer NOT NULL,
    realized_return numeric,
    realized_mdd numeric,
    label_source character varying(32),
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: dim_model_run; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.dim_model_run (
    run_id bigint NOT NULL,
    run_type character varying(32) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    top_n integer DEFAULT 20,
    train_start_date date,
    train_end_date date,
    config_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    comment text
);


--
-- Name: dim_model_run_run_id_seq; Type: SEQUENCE; Schema: research; Owner: -
--

CREATE SEQUENCE research.dim_model_run_run_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: dim_model_run_run_id_seq; Type: SEQUENCE OWNED BY; Schema: research; Owner: -
--

ALTER SEQUENCE research.dim_model_run_run_id_seq OWNED BY research.dim_model_run.run_id;


--
-- Name: paper_trading_nav; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_nav (
    paper_run_id bigint NOT NULL,
    strategy text NOT NULL,
    date date NOT NULL,
    cash numeric,
    market_value numeric,
    nav numeric,
    daily_return numeric,
    active_position_count integer,
    opened_today integer,
    duplicate_skip_count integer,
    deployed_cash numeric,
    cumulative_return numeric,
    running_nav_max numeric,
    drawdown numeric,
    closed_trade_count integer,
    closed_win_rate numeric,
    closed_win_count integer,
    closed_trade_count_cum integer,
    closed_win_count_cum integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_position; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_position (
    paper_run_id bigint NOT NULL,
    strategy text NOT NULL,
    code character varying(10) NOT NULL,
    name text,
    entry_date date NOT NULL,
    planned_exit_date date,
    exit_date date,
    entry_price_close numeric,
    entry_exec_price numeric,
    exit_price_close numeric,
    exit_exec_price numeric,
    shares numeric,
    entry_notional_gross numeric,
    exit_notional_net numeric,
    entry_cost_amount numeric,
    exit_cost_amount numeric,
    gross_return numeric,
    net_return numeric,
    source_rank integer,
    selection_stage text,
    dominant_theme text,
    confidence_score numeric,
    final_score numeric,
    status text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_run; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.paper_trading_run (
    paper_run_id bigint NOT NULL,
    run_tag text NOT NULL,
    source_mode text NOT NULL,
    asof_date date,
    hold_days integer NOT NULL,
    initial_nav numeric,
    entry_fee_bps numeric,
    exit_fee_bps numeric,
    entry_slippage_bps numeric,
    exit_slippage_bps numeric,
    positions_row_count integer DEFAULT 0 NOT NULL,
    nav_row_count integer DEFAULT 0 NOT NULL,
    source_positions_csv text,
    source_nav_csv text,
    source_report_md text,
    comment text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: paper_trading_run_paper_run_id_seq; Type: SEQUENCE; Schema: research; Owner: -
--

CREATE SEQUENCE research.paper_trading_run_paper_run_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: paper_trading_run_paper_run_id_seq; Type: SEQUENCE OWNED BY; Schema: research; Owner: -
--

ALTER SEQUENCE research.paper_trading_run_paper_run_id_seq OWNED BY research.paper_trading_run.paper_run_id;


--
-- Name: prediction_history; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.prediction_history (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    pred_return_60d numeric,
    pred_return_90d numeric,
    pred_mdd_60d numeric,
    pred_mdd_90d numeric,
    prob_top20_60d numeric,
    prob_top20_90d numeric,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    risk_penalty numeric,
    final_score numeric,
    final_score_custom numeric,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: ranking_history; Type: TABLE; Schema: research; Owner: -
--

CREATE TABLE research.ranking_history (
    run_id bigint NOT NULL,
    as_of_date date NOT NULL,
    code character varying(10) NOT NULL,
    model_version character varying(50) NOT NULL,
    horizon_days integer NOT NULL,
    rank integer NOT NULL,
    final_score numeric NOT NULL,
    score_formula_version text,
    ret_score numeric,
    prob_score numeric,
    qual_score numeric,
    tech_score numeric,
    risk_penalty numeric,
    in_top_n boolean DEFAULT false NOT NULL,
    top_n integer NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: backtest_trades id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backtest_trades ALTER COLUMN id SET DEFAULT nextval('public.backtest_trades_id_seq'::regclass);


--
-- Name: page_view_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.page_view_events ALTER COLUMN id SET DEFAULT nextval('public.page_view_events_id_seq'::regclass);


--
-- Name: pipeline_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_history ALTER COLUMN id SET DEFAULT nextval('public.pipeline_history_id_seq'::regclass);


--
-- Name: trade_audit_log audit_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trade_audit_log ALTER COLUMN audit_id SET DEFAULT nextval('public.trade_audit_log_audit_id_seq'::regclass);


--
-- Name: trades trade_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trades ALTER COLUMN trade_id SET DEFAULT nextval('public.trades_trade_id_seq'::regclass);


--
-- Name: dim_model_run run_id; Type: DEFAULT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.dim_model_run ALTER COLUMN run_id SET DEFAULT nextval('research.dim_model_run_run_id_seq'::regclass);


--
-- Name: paper_trading_run paper_run_id; Type: DEFAULT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run ALTER COLUMN paper_run_id SET DEFAULT nextval('research.paper_trading_run_paper_run_id_seq'::regclass);


--
-- Name: backtest_trades backtest_trades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backtest_trades
    ADD CONSTRAINT backtest_trades_pkey PRIMARY KEY (id);


--
-- Name: daily_ranking daily_ranking_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daily_ranking
    ADD CONSTRAINT daily_ranking_pkey PRIMARY KEY (date, code);


--
-- Name: daily_scores daily_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.daily_scores
    ADD CONSTRAINT daily_scores_pkey PRIMARY KEY (date, code);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_pkey PRIMARY KEY (as_of_date, etf_code, stock_code);


--
-- Name: etf_master etf_master_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_master
    ADD CONSTRAINT etf_master_pkey PRIMARY KEY (etf_code);


--
-- Name: etf_signal_daily etf_signal_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_signal_daily
    ADD CONSTRAINT etf_signal_daily_pkey PRIMARY KEY (as_of_date, etf_code);


--
-- Name: etf_theme_map etf_theme_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_pkey PRIMARY KEY (etf_code, theme_code, valid_from);


--
-- Name: fact_price_daily fact_price_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fact_price_daily
    ADD CONSTRAINT fact_price_daily_pkey PRIMARY KEY (date, code);


--
-- Name: features features_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.features
    ADD CONSTRAINT features_pkey PRIMARY KEY (date, code);


--
-- Name: flow_daily flow_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.flow_daily
    ADD CONSTRAINT flow_daily_pkey PRIMARY KEY (date, code, investor_type);


--
-- Name: fundamentals fundamentals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fundamentals
    ADD CONSTRAINT fundamentals_pkey PRIMARY KEY (date, code);


--
-- Name: labels labels_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.labels
    ADD CONSTRAINT labels_pkey PRIMARY KEY (date, code);


--
-- Name: market_status market_status_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.market_status
    ADD CONSTRAINT market_status_pkey PRIMARY KEY (date);


--
-- Name: page_view_events page_view_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.page_view_events
    ADD CONSTRAINT page_view_events_pkey PRIMARY KEY (id);


--
-- Name: pipeline_history pipeline_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_history
    ADD CONSTRAINT pipeline_history_pkey PRIMARY KEY (id);


--
-- Name: predictions predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (date, code);


--
-- Name: prices_adjusted prices_adjusted_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_adjusted
    ADD CONSTRAINT prices_adjusted_pkey PRIMARY KEY (date, code);


--
-- Name: prices_clean prices_clean_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_clean
    ADD CONSTRAINT prices_clean_pkey PRIMARY KEY (date, code);


--
-- Name: prices_raw prices_raw_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices_raw
    ADD CONSTRAINT prices_raw_pkey PRIMARY KEY (date, code);


--
-- Name: quality quality_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.quality
    ADD CONSTRAINT quality_pkey PRIMARY KEY (date, code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_pkey PRIMARY KEY (as_of_date, stock_code, theme_code);


--
-- Name: stocks stocks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stocks
    ADD CONSTRAINT stocks_pkey PRIMARY KEY (code);


--
-- Name: theme_master theme_master_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_master
    ADD CONSTRAINT theme_master_pkey PRIMARY KEY (theme_code);


--
-- Name: theme_master theme_master_theme_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_master
    ADD CONSTRAINT theme_master_theme_name_key UNIQUE (theme_name);


--
-- Name: theme_score_daily theme_score_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_pkey PRIMARY KEY (as_of_date, theme_code);


--
-- Name: trade_audit_log trade_audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trade_audit_log
    ADD CONSTRAINT trade_audit_log_pkey PRIMARY KEY (audit_id);


--
-- Name: trades trades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trades
    ADD CONSTRAINT trades_pkey PRIMARY KEY (trade_id);


--
-- Name: app_payload_store app_payload_store_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.app_payload_store
    ADD CONSTRAINT app_payload_store_pkey PRIMARY KEY (payload_key);


--
-- Name: backtest_outcome backtest_outcome_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.backtest_outcome
    ADD CONSTRAINT backtest_outcome_pkey PRIMARY KEY (run_id, as_of_date, code, horizon_days);


--
-- Name: dim_model_run dim_model_run_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.dim_model_run
    ADD CONSTRAINT dim_model_run_pkey PRIMARY KEY (run_id);


--
-- Name: paper_trading_nav paper_trading_nav_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_nav
    ADD CONSTRAINT paper_trading_nav_pkey PRIMARY KEY (paper_run_id, strategy, date);


--
-- Name: paper_trading_position paper_trading_position_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_position
    ADD CONSTRAINT paper_trading_position_pkey PRIMARY KEY (paper_run_id, strategy, code, entry_date);


--
-- Name: paper_trading_run paper_trading_run_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run
    ADD CONSTRAINT paper_trading_run_pkey PRIMARY KEY (paper_run_id);


--
-- Name: paper_trading_run paper_trading_run_run_tag_key; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_run
    ADD CONSTRAINT paper_trading_run_run_tag_key UNIQUE (run_tag);


--
-- Name: prediction_history prediction_history_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.prediction_history
    ADD CONSTRAINT prediction_history_pkey PRIMARY KEY (run_id, as_of_date, code, horizon_days);


--
-- Name: ranking_history ranking_history_pkey; Type: CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.ranking_history
    ADD CONSTRAINT ranking_history_pkey PRIMARY KEY (run_id, as_of_date, code);


--
-- Name: idx_backtest_trades_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_backtest_trades_code ON public.backtest_trades USING btree (code);


--
-- Name: idx_backtest_trades_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_backtest_trades_date ON public.backtest_trades USING btree (trade_date);


--
-- Name: idx_daily_ranking_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_ranking_code_date ON public.daily_ranking USING btree (code, date);


--
-- Name: idx_daily_ranking_date_final; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_ranking_date_final ON public.daily_ranking USING btree (date, final_score DESC);


--
-- Name: idx_daily_scores_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_daily_scores_code_date ON public.daily_scores USING btree (code, date);


--
-- Name: idx_etf_holdings_snapshot_asof_etf_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_asof_etf_code ON public.etf_holdings_snapshot USING btree (as_of_date, etf_code);


--
-- Name: idx_etf_holdings_snapshot_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_etf_code_asof ON public.etf_holdings_snapshot USING btree (etf_code, as_of_date DESC);


--
-- Name: idx_etf_holdings_snapshot_stock_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_holdings_snapshot_stock_code_asof ON public.etf_holdings_snapshot USING btree (stock_code, as_of_date DESC);


--
-- Name: idx_etf_master_is_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_master_is_active ON public.etf_master USING btree (is_active, etf_code);


--
-- Name: idx_etf_master_listing_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_master_listing_date ON public.etf_master USING btree (listing_date DESC, etf_code);


--
-- Name: idx_etf_signal_daily_asof_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_signal_daily_asof_score ON public.etf_signal_daily USING btree (as_of_date, signal_score DESC, etf_code);


--
-- Name: idx_etf_signal_daily_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_signal_daily_etf_code_asof ON public.etf_signal_daily USING btree (etf_code, as_of_date DESC);


--
-- Name: idx_etf_theme_map_etf_code_valid_to; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_theme_map_etf_code_valid_to ON public.etf_theme_map USING btree (etf_code, valid_to, theme_code);


--
-- Name: idx_etf_theme_map_theme_code_valid_from; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_etf_theme_map_theme_code_valid_from ON public.etf_theme_map USING btree (theme_code, valid_from DESC, etf_code);


--
-- Name: idx_fact_price_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_fact_price_code_date ON public.fact_price_daily USING btree (code, date);


--
-- Name: idx_features_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_features_code_date ON public.features USING btree (code, date);


--
-- Name: idx_flow_daily_code_date_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_flow_daily_code_date_desc ON public.flow_daily USING btree (code, date DESC);


--
-- Name: idx_flow_daily_date_investor_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_flow_daily_date_investor_type ON public.flow_daily USING btree (date, investor_type);


--
-- Name: idx_fundamentals_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_fundamentals_code_date ON public.fundamentals USING btree (code, date);


--
-- Name: idx_labels_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_labels_code_date ON public.labels USING btree (code, date);


--
-- Name: idx_market_status_date_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_market_status_date_desc ON public.market_status USING btree (date DESC);


--
-- Name: idx_page_view_events_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_created_at ON public.page_view_events USING btree (created_at DESC);


--
-- Name: idx_page_view_events_path_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_path_created_at ON public.page_view_events USING btree (path, created_at DESC);


--
-- Name: idx_page_view_events_visitor_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_page_view_events_visitor_created_at ON public.page_view_events USING btree (visitor_id, created_at DESC);


--
-- Name: idx_pipeline_history_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_history_created ON public.pipeline_history USING btree (created_at DESC);


--
-- Name: idx_pipeline_history_run_step; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_history_run_step ON public.pipeline_history USING btree (run_id, step);


--
-- Name: idx_predictions_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_predictions_code_date ON public.predictions USING btree (code, date);


--
-- Name: idx_prices_adjusted_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_adjusted_code_date ON public.prices_adjusted USING btree (code, date);


--
-- Name: idx_prices_clean_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_clean_code_date ON public.prices_clean USING btree (code, date);


--
-- Name: idx_prices_raw_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_raw_code_date ON public.prices_raw USING btree (code, date);


--
-- Name: idx_quality_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_quality_code_date ON public.quality USING btree (code, date);


--
-- Name: idx_stock_theme_exposure_daily_asof_stock_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_asof_stock_code ON public.stock_theme_exposure_daily USING btree (as_of_date, stock_code);


--
-- Name: idx_stock_theme_exposure_daily_asof_theme_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_asof_theme_score ON public.stock_theme_exposure_daily USING btree (as_of_date, theme_code, exposure_score DESC, stock_code);


--
-- Name: idx_stock_theme_exposure_daily_primary_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_primary_etf_code_asof ON public.stock_theme_exposure_daily USING btree (primary_etf_code, as_of_date DESC);


--
-- Name: idx_stock_theme_exposure_daily_stock_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_stock_code_asof ON public.stock_theme_exposure_daily USING btree (stock_code, as_of_date DESC);


--
-- Name: idx_stock_theme_exposure_daily_theme_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_stock_theme_exposure_daily_theme_code_asof ON public.stock_theme_exposure_daily USING btree (theme_code, as_of_date DESC);


--
-- Name: idx_theme_master_is_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_master_is_active ON public.theme_master USING btree (is_active, display_order, theme_code);


--
-- Name: idx_theme_score_daily_asof_score; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_asof_score ON public.theme_score_daily USING btree (as_of_date, theme_score DESC, theme_code);


--
-- Name: idx_theme_score_daily_leader_etf_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_leader_etf_code_asof ON public.theme_score_daily USING btree (leader_etf_code, as_of_date DESC);


--
-- Name: idx_theme_score_daily_theme_code_asof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theme_score_daily_theme_code_asof ON public.theme_score_daily USING btree (theme_code, as_of_date DESC);


--
-- Name: idx_trade_audit_trade_id_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trade_audit_trade_id_created ON public.trade_audit_log USING btree (trade_id, created_at DESC);


--
-- Name: idx_trades_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_code ON public.trades USING btree (code);


--
-- Name: idx_trades_code_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_code_date ON public.trades USING btree (code, date);


--
-- Name: idx_trades_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_date ON public.trades USING btree (date);


--
-- Name: idx_app_payload_store_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_app_payload_store_asof ON research.app_payload_store USING btree (asof_date DESC, updated_at DESC);


--
-- Name: idx_dim_model_run_type; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_dim_model_run_type ON research.dim_model_run USING btree (run_type);


--
-- Name: idx_paper_trading_nav_lookup; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_nav_lookup ON research.paper_trading_nav USING btree (strategy, date DESC);


--
-- Name: idx_paper_trading_position_lookup; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_position_lookup ON research.paper_trading_position USING btree (strategy, entry_date DESC, code);


--
-- Name: idx_paper_trading_run_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_paper_trading_run_asof ON research.paper_trading_run USING btree (asof_date DESC, paper_run_id DESC);


--
-- Name: idx_pred_hist_asof_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_pred_hist_asof_model ON research.prediction_history USING btree (as_of_date, model_version);


--
-- Name: idx_pred_hist_code_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_pred_hist_code_model ON research.prediction_history USING btree (code, model_version);


--
-- Name: idx_rank_hist_asof_model; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_rank_hist_asof_model ON research.ranking_history USING btree (as_of_date, model_version);


--
-- Name: idx_rank_hist_run_asof; Type: INDEX; Schema: research; Owner: -
--

CREATE INDEX idx_rank_hist_run_asof ON research.ranking_history USING btree (run_id, as_of_date, rank);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_holdings_snapshot etf_holdings_snapshot_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_holdings_snapshot
    ADD CONSTRAINT etf_holdings_snapshot_stock_code_fkey FOREIGN KEY (stock_code) REFERENCES public.stocks(code);


--
-- Name: etf_signal_daily etf_signal_daily_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_signal_daily
    ADD CONSTRAINT etf_signal_daily_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_theme_map etf_theme_map_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_etf_code_fkey FOREIGN KEY (etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: etf_theme_map etf_theme_map_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.etf_theme_map
    ADD CONSTRAINT etf_theme_map_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_primary_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_primary_etf_code_fkey FOREIGN KEY (primary_etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_stock_code_fkey FOREIGN KEY (stock_code) REFERENCES public.stocks(code);


--
-- Name: stock_theme_exposure_daily stock_theme_exposure_daily_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stock_theme_exposure_daily
    ADD CONSTRAINT stock_theme_exposure_daily_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: theme_score_daily theme_score_daily_leader_etf_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_leader_etf_code_fkey FOREIGN KEY (leader_etf_code) REFERENCES public.etf_master(etf_code);


--
-- Name: theme_score_daily theme_score_daily_leader_stock_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_leader_stock_code_fkey FOREIGN KEY (leader_stock_code) REFERENCES public.stocks(code);


--
-- Name: theme_score_daily theme_score_daily_theme_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theme_score_daily
    ADD CONSTRAINT theme_score_daily_theme_code_fkey FOREIGN KEY (theme_code) REFERENCES public.theme_master(theme_code);


--
-- Name: backtest_outcome backtest_outcome_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.backtest_outcome
    ADD CONSTRAINT backtest_outcome_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- Name: paper_trading_nav paper_trading_nav_paper_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_nav
    ADD CONSTRAINT paper_trading_nav_paper_run_id_fkey FOREIGN KEY (paper_run_id) REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE;


--
-- Name: paper_trading_position paper_trading_position_paper_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.paper_trading_position
    ADD CONSTRAINT paper_trading_position_paper_run_id_fkey FOREIGN KEY (paper_run_id) REFERENCES research.paper_trading_run(paper_run_id) ON DELETE CASCADE;


--
-- Name: prediction_history prediction_history_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.prediction_history
    ADD CONSTRAINT prediction_history_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- Name: ranking_history ranking_history_run_id_fkey; Type: FK CONSTRAINT; Schema: research; Owner: -
--

ALTER TABLE ONLY research.ranking_history
    ADD CONSTRAINT ranking_history_run_id_fkey FOREIGN KEY (run_id) REFERENCES research.dim_model_run(run_id);


--
-- PostgreSQL database dump complete
--


>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
