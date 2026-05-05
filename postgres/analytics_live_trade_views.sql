CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.ranking_quality_guard_shadow AS
WITH latest_ranking AS (
    SELECT DISTINCT ON (rh.as_of_date, rh.code)
        rh.run_id,
        rh.as_of_date,
        rh.code,
        rh.model_version,
        rh.horizon_days,
        rh.rank AS production_rank,
        rh.final_score,
        rh.score_formula_version,
        rh.ret_score,
        rh.prob_score,
        rh.qual_score,
        rh.tech_score,
        rh.risk_penalty,
        rh.in_top_n,
        rh.top_n,
        rh.created_at
    FROM research.ranking_history rh
    ORDER BY rh.as_of_date, rh.code, rh.run_id DESC, rh.top_n DESC NULLS LAST
),
scored AS (
    SELECT
        latest_ranking.*,
        (
            CASE WHEN COALESCE(latest_ranking.qual_score, 50) < 20 THEN 6 ELSE 0 END
            + CASE WHEN COALESCE(latest_ranking.risk_penalty, 0) >= 12 THEN 4 ELSE 0 END
        )::numeric AS shadow_quality_risk_guard_penalty
    FROM latest_ranking
)
SELECT
    scored.*,
    (scored.shadow_quality_risk_guard_penalty > 0) AS shadow_quality_risk_guard_applied,
    (scored.final_score - scored.shadow_quality_risk_guard_penalty) AS shadow_final_score_quality_risk_guard,
    RANK() OVER (
        PARTITION BY scored.as_of_date
        ORDER BY (scored.final_score - scored.shadow_quality_risk_guard_penalty) DESC NULLS LAST, scored.code
    )::integer AS shadow_rank_quality_risk_guard
FROM scored;

CREATE OR REPLACE VIEW analytics.live_trade_fact AS
WITH latest_execution AS (
    SELECT DISTINCT ON (request_id)
        request_id,
        intent_id,
        as_of_date,
        executed_at,
        submitted_at,
        gate_status,
        env_dv,
        code,
        name,
        side,
        intent_type,
        ord_dvsn,
        reference_price,
        previous_close,
        live_price,
        entry_price_gap_pct,
        entry_gate_status,
        entry_gate_reason,
        final_request_qty,
        engine_type,
        strategy_id,
        run_mode,
        source_score_date,
        final_score,
        confidence_score,
        calibrated_confidence,
        live_confidence_grade,
        prob_score,
        ret_score,
        tech_score,
        quality_score,
        liquidity_score,
        market_regime,
        buy_reason,
        sell_reason,
        portfolio_action_reason,
        submission_status,
        skip_reason,
        broker_order_id,
        broker_org_order_id,
        updated_at
    FROM research.live_order_execution
    ORDER BY request_id, COALESCE(executed_at, submitted_at, updated_at) DESC NULLS LAST
)
SELECT
    f.fill_id,
    f.request_id,
    COALESCE(f.request_id, e.request_id, r.request_id) AS resolved_request_id,
    COALESCE(e.intent_id, r.intent_id, d.intent_id) AS intent_id,
    COALESCE(e.as_of_date, r.as_of_date, d.as_of_date, f.as_of_date) AS as_of_date,
    f.filled_at,
    COALESCE(f.code, e.code, r.code, d.code) AS code,
    COALESCE(f.name, e.name, r.name, d.name) AS name,
    COALESCE(f.side, e.side, r.side) AS side,
    COALESCE(e.intent_type, r.intent_type, d.intent_type) AS intent_type,
    f.filled_qty,
    f.filled_price,
    COALESCE(f.filled_amount, f.filled_qty * f.filled_price) AS filled_amount,
    f.fee,
    f.tax,
    COALESCE(r.ranking_run_id, d.ranking_run_id, q.run_id) AS ranking_run_id,
    COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) AS ranking_rank,
    COALESCE(e.final_score, r.final_score, d.final_score, q.final_score) AS final_score,
    COALESCE(e.confidence_score, r.confidence_score, d.confidence_score) AS confidence_score,
    COALESCE(r.risk_penalty, d.risk_penalty, q.risk_penalty) AS risk_penalty,
    COALESCE(e.ret_score, r.ret_score, d.ret_score, q.ret_score) AS ret_score,
    COALESCE(e.prob_score, r.prob_score, d.prob_score, q.prob_score) AS prob_score,
    COALESCE(r.qual_score, d.qual_score, q.qual_score) AS qual_score,
    COALESCE(e.tech_score, r.tech_score, d.tech_score, q.tech_score) AS tech_score,
    COALESCE(e.liquidity_score, r.liquidity_score, d.liquidity_score) AS liquidity_score,
    COALESCE(r.safety_score, d.safety_score) AS safety_score,
    COALESCE(r.dominant_theme, d.dominant_theme) AS dominant_theme,
    COALESCE(r.score_driver_1, d.score_driver_1) AS score_driver_1,
    COALESCE(r.score_driver_2, d.score_driver_2) AS score_driver_2,
    COALESCE(r.score_driver_3, d.score_driver_3) AS score_driver_3,
    COALESCE(r.risk_factor_1, d.risk_factor_1) AS risk_factor_1,
    COALESCE(r.risk_factor_2, d.risk_factor_2) AS risk_factor_2,
    COALESCE(e.gate_status, r.gate_status, d.gate_status) AS gate_status,
    d.source_action,
    e.submission_status,
    COALESCE(f.broker_order_id, e.broker_order_id) AS broker_order_id,
    q.shadow_quality_risk_guard_applied,
    q.shadow_quality_risk_guard_penalty,
    q.shadow_final_score_quality_risk_guard,
    q.shadow_rank_quality_risk_guard,
    q.production_rank,
    (COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) - q.shadow_rank_quality_risk_guard)::integer AS shadow_rank_delta,
    CASE
        WHEN q.shadow_quality_risk_guard_applied THEN 'guard_applied'
        WHEN q.shadow_quality_risk_guard_applied = false THEN 'guard_not_applied'
        ELSE 'guard_unknown'
    END AS guard_applied_bucket,
    CASE
        WHEN q.shadow_quality_risk_guard_penalty IS NULL THEN 'penalty_unknown'
        WHEN q.shadow_quality_risk_guard_penalty = 0 THEN 'penalty_0'
        WHEN q.shadow_quality_risk_guard_penalty = 4 THEN 'penalty_4'
        WHEN q.shadow_quality_risk_guard_penalty = 6 THEN 'penalty_6'
        WHEN q.shadow_quality_risk_guard_penalty >= 10 THEN 'penalty_10'
        ELSE 'penalty_other'
    END AS guard_penalty_bucket,
    CASE
        WHEN COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) IS NULL THEN 'rank_unknown'
        WHEN COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) <= 3 THEN 'rank_1_3'
        WHEN COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) <= 8 THEN 'rank_4_8'
        WHEN COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) <= 20 THEN 'rank_9_20'
        ELSE 'rank_21_plus'
    END AS rank_bucket,
    CASE
        WHEN COALESCE(r.final_score, d.final_score, q.final_score) IS NULL THEN 'score_unknown'
        WHEN COALESCE(r.final_score, d.final_score, q.final_score) >= 80 THEN 'score_80_plus'
        WHEN COALESCE(r.final_score, d.final_score, q.final_score) >= 70 THEN 'score_70_80'
        WHEN COALESCE(r.final_score, d.final_score, q.final_score) >= 60 THEN 'score_60_70'
        ELSE 'score_under_60'
    END AS final_score_bucket,
    CASE
        WHEN COALESCE(r.confidence_score, d.confidence_score) IS NULL THEN 'confidence_unknown'
        WHEN COALESCE(r.confidence_score, d.confidence_score) >= 90 THEN 'confidence_90_plus'
        WHEN COALESCE(r.confidence_score, d.confidence_score) >= 80 THEN 'confidence_80_90'
        WHEN COALESCE(r.confidence_score, d.confidence_score) >= 70 THEN 'confidence_70_80'
        ELSE 'confidence_under_70'
    END AS confidence_bucket,
    CASE
        WHEN COALESCE(r.risk_penalty, d.risk_penalty, q.risk_penalty) IS NULL THEN 'risk_unknown'
        WHEN COALESCE(r.risk_penalty, d.risk_penalty, q.risk_penalty) >= 12 THEN 'risk_12_plus'
        WHEN COALESCE(r.risk_penalty, d.risk_penalty, q.risk_penalty) >= 6 THEN 'risk_6_12'
        WHEN COALESCE(r.risk_penalty, d.risk_penalty, q.risk_penalty) > 0 THEN 'risk_0_6'
        ELSE 'risk_0'
    END AS risk_penalty_bucket,
    CASE
        WHEN q.production_rank IS NULL THEN 'production_rank_unknown'
        WHEN q.production_rank <= 20 THEN 'production_top20'
        WHEN q.production_rank <= 50 THEN 'production_top50'
        ELSE 'production_51_plus'
    END AS production_rank_bucket,
    CASE
        WHEN q.shadow_rank_quality_risk_guard IS NULL THEN 'shadow_rank_unknown'
        WHEN q.shadow_rank_quality_risk_guard <= 20 THEN 'shadow_top20'
        WHEN q.shadow_rank_quality_risk_guard <= 50 THEN 'shadow_top50'
        ELSE 'shadow_51_plus'
    END AS shadow_rank_bucket,
    CASE
        WHEN (COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) - q.shadow_rank_quality_risk_guard) > 0 THEN 'shadow_rank_up'
        WHEN (COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) - q.shadow_rank_quality_risk_guard) < 0 THEN 'shadow_rank_down'
        WHEN (COALESCE(r.ranking_rank, d.ranking_rank, q.production_rank) - q.shadow_rank_quality_risk_guard) = 0 THEN 'shadow_rank_same'
        ELSE 'shadow_rank_unknown'
    END AS shadow_rank_delta_bucket,
    COALESCE(e.engine_type, r.engine_type, d.engine_type) AS engine_type,
    COALESCE(e.strategy_id, r.strategy_id, d.strategy_id) AS strategy_id,
    COALESCE(e.run_mode, r.run_mode, d.run_mode) AS run_mode,
    COALESCE(e.source_score_date, r.source_score_date, d.source_score_date, COALESCE(e.as_of_date, r.as_of_date, d.as_of_date, f.as_of_date)) AS source_score_date,
    COALESCE(e.calibrated_confidence, r.calibrated_confidence, d.calibrated_confidence) AS calibrated_confidence,
    COALESCE(e.live_confidence_grade, r.live_confidence_grade, d.live_confidence_grade) AS live_confidence_grade,
    COALESCE(e.quality_score, r.quality_score, d.quality_score, r.qual_score, d.qual_score, q.qual_score) AS quality_score,
    COALESCE(e.market_regime, r.market_regime, d.market_regime) AS market_regime,
    COALESCE(e.previous_close, r.previous_close) AS previous_close,
    COALESCE(e.live_price, r.live_price) AS live_price,
    COALESCE(e.entry_price_gap_pct, r.entry_price_gap_pct) AS entry_price_gap_pct,
    COALESCE(e.entry_gate_status, r.entry_gate_status) AS entry_gate_status,
    COALESCE(e.entry_gate_reason, r.entry_gate_reason) AS entry_gate_reason,
    COALESCE(e.buy_reason, r.buy_reason, d.buy_reason) AS buy_reason,
    COALESCE(e.sell_reason, r.sell_reason, d.sell_reason) AS sell_reason,
    COALESCE(e.portfolio_action_reason, r.portfolio_action_reason, d.portfolio_action_reason) AS portfolio_action_reason
FROM research.live_order_fill f
LEFT JOIN latest_execution e
  ON e.request_id = f.request_id
LEFT JOIN research.live_order_request r
  ON r.request_id = f.request_id
LEFT JOIN research.live_trade_decision d
  ON d.intent_id = COALESCE(e.intent_id, r.intent_id)
LEFT JOIN analytics.ranking_quality_guard_shadow q
  ON q.as_of_date = COALESCE(e.as_of_date, r.as_of_date, d.as_of_date, f.as_of_date)
 AND q.code = COALESCE(f.code, e.code, r.code, d.code);

CREATE OR REPLACE VIEW analytics.live_trade_return_fact AS
WITH horizons(horizon) AS (
    VALUES (0), (1), (3), (5)
)
SELECT
    ltf.fill_id,
    ltf.request_id,
    ltf.resolved_request_id,
    ltf.intent_id,
    ltf.as_of_date,
    ltf.filled_at,
    ltf.code,
    ltf.name,
    ltf.side,
    ltf.intent_type,
    ltf.filled_qty,
    ltf.filled_price,
    ltf.filled_amount,
    ltf.fee,
    ltf.tax,
    ltf.ranking_run_id,
    ltf.ranking_rank,
    ltf.final_score,
    ltf.confidence_score,
    ltf.risk_penalty,
    ltf.ret_score,
    ltf.prob_score,
    ltf.qual_score,
    ltf.tech_score,
    ltf.liquidity_score,
    ltf.safety_score,
    ltf.dominant_theme,
    ltf.score_driver_1,
    ltf.score_driver_2,
    ltf.score_driver_3,
    ltf.risk_factor_1,
    ltf.risk_factor_2,
    ltf.gate_status,
    ltf.source_action,
    ltf.submission_status,
    ltf.broker_order_id,
    ltf.shadow_quality_risk_guard_applied,
    ltf.shadow_quality_risk_guard_penalty,
    ltf.shadow_final_score_quality_risk_guard,
    ltf.shadow_rank_quality_risk_guard,
    ltf.production_rank,
    ltf.shadow_rank_delta,
    ltf.guard_applied_bucket,
    ltf.guard_penalty_bucket,
    ltf.rank_bucket,
    ltf.final_score_bucket,
    ltf.confidence_bucket,
    ltf.risk_penalty_bucket,
    ltf.production_rank_bucket,
    ltf.shadow_rank_bucket,
    ltf.shadow_rank_delta_bucket,
    horizons.horizon,
    price_point.price_date,
    price_point.mark_price,
    CASE
        WHEN price_point.mark_price IS NULL OR ltf.filled_price <= 0 THEN NULL
        WHEN UPPER(COALESCE(ltf.side, '')) = 'BUY' THEN price_point.mark_price / ltf.filled_price - 1
        ELSE ltf.filled_price / price_point.mark_price - 1
    END AS signed_return,
    (price_point.mark_price IS NOT NULL) AS is_observed,
    ltf.engine_type,
    ltf.strategy_id,
    ltf.run_mode,
    ltf.source_score_date,
    ltf.calibrated_confidence,
    ltf.live_confidence_grade,
    ltf.quality_score,
    ltf.market_regime,
    ltf.previous_close,
    ltf.live_price,
    ltf.entry_price_gap_pct,
    ltf.entry_gate_status,
    ltf.entry_gate_reason,
    ltf.buy_reason,
    ltf.sell_reason,
    ltf.portfolio_action_reason
FROM analytics.live_trade_fact ltf
CROSS JOIN horizons
LEFT JOIN LATERAL (
    SELECT ranked.price_date, ranked.mark_price
    FROM (
        SELECT
            p.date AS price_date,
            p.adj_close AS mark_price,
            (ROW_NUMBER() OVER (ORDER BY p.date) - 1)::integer AS price_horizon
        FROM public.prices_adjusted p
        WHERE p.code = ltf.code
          AND p.date >= ltf.filled_at::date
          AND p.adj_close IS NOT NULL
    ) ranked
    WHERE ranked.price_horizon = horizons.horizon
    LIMIT 1
) price_point ON true;

CREATE OR REPLACE VIEW analytics.live_review_kpi AS
WITH grouped AS (
    SELECT
        horizon,
        COALESCE(intent_type, 'UNKNOWN') AS intent_type,
        rank_bucket,
        final_score_bucket,
        confidence_bucket,
        risk_penalty_bucket,
        COALESCE(dominant_theme, 'theme_unknown') AS dominant_theme,
        COALESCE(gate_status, 'gate_unknown') AS market_gate,
        COUNT(*)::integer AS count,
        COUNT(signed_return)::integer AS observed_count,
        AVG(signed_return) AS avg_return,
        SUM(signed_return * filled_amount) / NULLIF(SUM(CASE WHEN signed_return IS NOT NULL THEN filled_amount END), 0) AS weighted_avg_return,
        AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
        AVG(CASE WHEN signed_return > 0 THEN signed_return END) AS avg_win,
        AVG(CASE WHEN signed_return < 0 THEN signed_return END) AS avg_loss,
        AVG(signed_return) AS expectancy
    FROM analytics.live_trade_return_fact
    GROUP BY
        horizon, COALESCE(intent_type, 'UNKNOWN'), rank_bucket, final_score_bucket,
        confidence_bucket, risk_penalty_bucket, COALESCE(dominant_theme, 'theme_unknown'),
        COALESCE(gate_status, 'gate_unknown')
)
SELECT
    grouped.*,
    CASE
        WHEN observed_count < 30 THEN 'INSUFFICIENT_SAMPLE'
        WHEN observed_count < 100 THEN 'MONITOR_ONLY'
        ELSE 'ACTIONABLE'
    END AS sample_status,
    ABS(avg_win / NULLIF(avg_loss, 0)) AS payoff_ratio
FROM grouped;

CREATE OR REPLACE VIEW analytics.live_score_bucket_kpi AS
WITH grouped AS (
    SELECT
        horizon,
        final_score_bucket,
        confidence_bucket,
        risk_penalty_bucket,
        rank_bucket,
        COUNT(*)::integer AS count,
        COUNT(signed_return)::integer AS observed_count,
        AVG(signed_return) AS avg_return,
        SUM(signed_return * filled_amount) / NULLIF(SUM(CASE WHEN signed_return IS NOT NULL THEN filled_amount END), 0) AS weighted_avg_return,
        AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
        AVG(CASE WHEN signed_return > 0 THEN signed_return END) AS avg_win,
        AVG(CASE WHEN signed_return < 0 THEN signed_return END) AS avg_loss,
        AVG(signed_return) AS expectancy
    FROM analytics.live_trade_return_fact
    GROUP BY horizon, final_score_bucket, confidence_bucket, risk_penalty_bucket, rank_bucket
)
SELECT
    grouped.*,
    CASE
        WHEN observed_count < 30 THEN 'INSUFFICIENT_SAMPLE'
        WHEN observed_count < 100 THEN 'MONITOR_ONLY'
        ELSE 'ACTIONABLE'
    END AS sample_status,
    ABS(avg_win / NULLIF(avg_loss, 0)) AS payoff_ratio
FROM grouped;

CREATE OR REPLACE VIEW analytics.live_quality_guard_kpi AS
WITH grouped AS (
    SELECT
        horizon,
        COALESCE(shadow_quality_risk_guard_applied, false) AS shadow_quality_risk_guard_applied,
        guard_penalty_bucket,
        shadow_rank_delta_bucket,
        production_rank_bucket,
        shadow_rank_bucket,
        COUNT(*)::integer AS count,
        COUNT(signed_return)::integer AS observed_count,
        AVG(signed_return) AS avg_return,
        SUM(signed_return * filled_amount) / NULLIF(SUM(CASE WHEN signed_return IS NOT NULL THEN filled_amount END), 0) AS weighted_avg_return,
        AVG(CASE WHEN signed_return > 0 THEN 1.0 WHEN signed_return IS NOT NULL THEN 0.0 END) AS win_rate,
        AVG(signed_return) AS expectancy,
        AVG(CASE WHEN signed_return < 0 THEN signed_return END) AS avg_downside_return,
        MIN(signed_return) AS max_loss
    FROM analytics.live_trade_return_fact
    GROUP BY
        horizon, COALESCE(shadow_quality_risk_guard_applied, false), guard_penalty_bucket,
        shadow_rank_delta_bucket, production_rank_bucket, shadow_rank_bucket
)
SELECT
    grouped.*,
    CASE
        WHEN observed_count < 30 THEN 'INSUFFICIENT_SAMPLE'
        WHEN observed_count < 100 THEN 'MONITOR_ONLY'
        ELSE 'ACTIONABLE'
    END AS sample_status
FROM grouped;

CREATE OR REPLACE VIEW analytics.live_daily_account_nav AS
WITH latest_snapshot AS (
    SELECT DISTINCT ON (snapshot_date)
        snapshot_date,
        snapshot_at,
        cash_amount,
        total_assets
    FROM research.live_position_snapshot
    ORDER BY snapshot_date, snapshot_at DESC
),
positions AS (
    SELECT
        p.snapshot_date,
        p.snapshot_at,
        COUNT(*) FILTER (WHERE COALESCE(p.qty, 0) > 0)::integer AS open_position_count,
        COALESCE(SUM(p.eval_amount), 0)::numeric AS position_eval_amount,
        COALESCE(SUM(p.pnl_amount), 0)::numeric AS unrealized_pnl_amount
    FROM research.live_position_snapshot p
    JOIN latest_snapshot latest
      ON latest.snapshot_date = p.snapshot_date
     AND latest.snapshot_at = p.snapshot_at
    GROUP BY p.snapshot_date, p.snapshot_at
)
SELECT
    positions.snapshot_date,
    positions.snapshot_at,
    positions.open_position_count,
    positions.position_eval_amount,
    positions.unrealized_pnl_amount,
    latest_snapshot.cash_amount,
    COALESCE(latest_snapshot.total_assets, positions.position_eval_amount + COALESCE(latest_snapshot.cash_amount, 0)) AS total_assets,
    LAG(COALESCE(latest_snapshot.total_assets, positions.position_eval_amount + COALESCE(latest_snapshot.cash_amount, 0)))
        OVER (ORDER BY positions.snapshot_date) AS prev_total_assets,
    (
        COALESCE(latest_snapshot.total_assets, positions.position_eval_amount + COALESCE(latest_snapshot.cash_amount, 0))
        / NULLIF(
            LAG(COALESCE(latest_snapshot.total_assets, positions.position_eval_amount + COALESCE(latest_snapshot.cash_amount, 0)))
                OVER (ORDER BY positions.snapshot_date),
            0
        )
        - 1
    ) AS daily_return
FROM positions
JOIN latest_snapshot
  ON latest_snapshot.snapshot_date = positions.snapshot_date
 AND latest_snapshot.snapshot_at = positions.snapshot_at;

CREATE OR REPLACE VIEW analytics.live_closed_trade AS
WITH sell_fills AS (
    SELECT *
    FROM analytics.live_trade_fact
    WHERE UPPER(COALESCE(side, '')) = 'SELL'
),
closed AS (
    SELECT
        s.fill_id AS sell_fill_id,
        s.request_id AS sell_request_id,
        s.intent_id,
        s.as_of_date,
        s.filled_at AS closed_at,
        s.code,
        s.name,
        s.intent_type,
        s.filled_qty AS sell_qty,
        s.filled_price AS sell_price,
        s.filled_amount AS sell_amount,
        s.fee AS sell_fee,
        s.tax AS sell_tax,
        buy_basis.buy_qty_before_sell,
        buy_basis.buy_amount_before_sell,
        buy_basis.buy_fee_before_sell,
        buy_basis.avg_buy_price,
        LEAST(s.filled_qty, buy_basis.buy_qty_before_sell) AS matched_qty,
        s.ranking_run_id,
        s.ranking_rank,
        s.final_score,
        s.confidence_score,
        s.risk_penalty,
        s.qual_score,
        s.dominant_theme,
        s.guard_applied_bucket,
        s.guard_penalty_bucket,
        s.production_rank_bucket,
        s.shadow_rank_bucket,
        s.shadow_rank_delta_bucket,
        s.engine_type,
        s.strategy_id,
        s.run_mode,
        s.source_score_date,
        s.live_confidence_grade,
        s.quality_score,
        s.entry_gate_status,
        s.entry_gate_reason,
        s.buy_reason,
        s.sell_reason,
        s.portfolio_action_reason,
        buy_basis.first_buy_filled_at
    FROM sell_fills s
    LEFT JOIN LATERAL (
        SELECT
            SUM(b.filled_qty) AS buy_qty_before_sell,
            SUM(COALESCE(b.filled_amount, b.filled_qty * b.filled_price)) AS buy_amount_before_sell,
            SUM(COALESCE(b.fee, 0)) AS buy_fee_before_sell,
            SUM(COALESCE(b.filled_amount, b.filled_qty * b.filled_price)) / NULLIF(SUM(b.filled_qty), 0) AS avg_buy_price,
            MIN(b.filled_at) AS first_buy_filled_at
        FROM analytics.live_trade_fact b
        WHERE UPPER(COALESCE(b.side, '')) = 'BUY'
          AND b.code = s.code
          AND b.filled_at <= s.filled_at
    ) buy_basis ON true
)
SELECT
    closed.sell_fill_id,
    closed.sell_request_id,
    closed.intent_id,
    closed.as_of_date,
    closed.closed_at,
    closed.code,
    closed.name,
    closed.intent_type,
    closed.sell_qty,
    closed.sell_price,
    closed.sell_amount,
    closed.sell_fee,
    closed.sell_tax,
    closed.buy_qty_before_sell,
    closed.buy_amount_before_sell,
    closed.buy_fee_before_sell,
    closed.avg_buy_price,
    closed.matched_qty,
    closed.ranking_run_id,
    closed.ranking_rank,
    closed.final_score,
    closed.confidence_score,
    closed.risk_penalty,
    closed.qual_score,
    closed.dominant_theme,
    closed.guard_applied_bucket,
    closed.guard_penalty_bucket,
    closed.production_rank_bucket,
    closed.shadow_rank_bucket,
    closed.shadow_rank_delta_bucket,
    CASE
        WHEN matched_qty IS NULL OR avg_buy_price IS NULL THEN NULL
        ELSE (sell_price - avg_buy_price) * matched_qty
    END AS realized_gross_pnl,
    CASE
        WHEN matched_qty IS NULL OR buy_qty_before_sell IS NULL OR buy_qty_before_sell = 0 THEN NULL
        ELSE COALESCE(buy_fee_before_sell, 0) * matched_qty / buy_qty_before_sell
    END AS allocated_buy_fee,
    CASE
        WHEN matched_qty IS NULL OR avg_buy_price IS NULL THEN NULL
        ELSE
            (sell_price - avg_buy_price) * matched_qty
            - COALESCE(sell_fee, 0)
            - COALESCE(sell_tax, 0)
            - CASE
                WHEN buy_qty_before_sell IS NULL OR buy_qty_before_sell = 0 THEN 0
                ELSE COALESCE(buy_fee_before_sell, 0) * matched_qty / buy_qty_before_sell
              END
    END AS realized_net_pnl,
    CASE
        WHEN matched_qty IS NULL OR avg_buy_price IS NULL OR avg_buy_price <= 0 THEN NULL
        ELSE
            (
                (sell_price - avg_buy_price) * matched_qty
                - COALESCE(sell_fee, 0)
                - COALESCE(sell_tax, 0)
                - CASE
                    WHEN buy_qty_before_sell IS NULL OR buy_qty_before_sell = 0 THEN 0
                    ELSE COALESCE(buy_fee_before_sell, 0) * matched_qty / buy_qty_before_sell
                  END
            ) / NULLIF(avg_buy_price * matched_qty, 0)
    END AS realized_return,
    CASE
        WHEN buy_qty_before_sell IS NULL THEN 'BUY_BASIS_MISSING'
        WHEN sell_qty > buy_qty_before_sell THEN 'PARTIAL_BASIS'
        ELSE 'MATCHED'
    END AS match_status,
    closed.engine_type,
    closed.strategy_id,
    closed.run_mode,
    closed.source_score_date,
    closed.live_confidence_grade,
    closed.quality_score,
    closed.entry_gate_status,
    closed.entry_gate_reason,
    closed.buy_reason,
    closed.sell_reason,
    closed.portfolio_action_reason,
    CASE
        WHEN first_buy_filled_at IS NULL THEN NULL
        ELSE GREATEST((closed_at::date - first_buy_filled_at::date), 0)
    END::integer AS holding_days
FROM closed;
