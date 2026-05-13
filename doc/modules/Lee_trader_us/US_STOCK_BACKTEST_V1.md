# US Stock Rank Backtest V1

> 문서 역할: `상세 참고 문서`
>
> Phase 4 backtest와 분석 흐름을 자세히 설명하는 문서다.

## Purpose

Phase 4-1 validates whether stored US stock ranking snapshots showed usable forward performance.

This is not auto-trading.

- Phase 4 uses stored ranking snapshots for research and validation only.
- No buy or sell order is created.
- Korean live-trading logic remains untouched.

## Backtest Question

Core question:

```text
If a symbol was selected by the Rule-based rank on trade_date,
did it outperform over the next 5 / 20 / 60 trading days?
```

Initial comparison targets:

- SPY excess return
- QQQ excess return
- same-date ranked-universe average return

## Data Flow

```text
recommend.us_stock_rank_daily
    ->
market.us_stock_daily_price
    ->
scripts/backtest_us_stock_rank_strategy.py
    ->
research.us_stock_rank_backtest_result
research.us_stock_rank_backtest_summary
```

## Related Tables

`recommend.us_stock_rank_daily`
- stored daily rank snapshot
- input for strategy selection

`market.us_stock_daily_price`
- next-trading-day entry lookup
- forward exit lookup
- SPY / QQQ benchmark return lookup

`research.us_stock_rank_backtest_result`
- symbol-level backtest rows
- one row per `backtest_id + trade_date + symbol + holding_days`

`research.us_stock_rank_backtest_summary`
- date-level strategy summary
- one row per `backtest_id + trade_date + strategy_name + holding_days`

## Strategy Set

Default strategies:

- `US_RANK_TOP5`: `rank_no <= 5`
- `US_RANK_TOP10`: `rank_no <= 10`
- `US_RANK_TOP20`: `rank_no <= 20`
- `US_RANK_BUY_OR_BETTER`: `recommend_grade in ('STRONG_BUY', 'BUY')`
- `US_RANK_STRONG_BUY`: `recommend_grade = 'STRONG_BUY'`

Holding periods:

- `5`
- `20`
- `60`

## Entry / Exit Price Rule

To avoid look-ahead bias:

- do not use `trade_date` close as entry
- use the next available trading-day close after `trade_date`
- use the close price `holding_days` trading sessions after `entry_date`

Current implementation price basis:

- `adj_close_price` when available
- fallback to `close_price`

This keeps split-adjusted series usable while still following next-session close timing.

## Return Formula

```text
return_pct = (exit_price - entry_price) / entry_price
spy_return_pct = (SPY exit_price - SPY entry_price) / SPY entry_price
qqq_return_pct = (QQQ exit_price - QQQ entry_price) / QQQ entry_price
excess_return_vs_spy = return_pct - spy_return_pct
excess_return_vs_qqq = return_pct - qqq_return_pct
excess_return_vs_universe = return_pct - universe_avg_return_pct
```

Flag rules:

```text
win_flag = 1 if return_pct > 0 else 0
win_vs_spy_flag = 1 if excess_return_vs_spy > 0 else 0
win_vs_qqq_flag = 1 if excess_return_vs_qqq > 0 else 0
win_vs_universe_flag = 1 if excess_return_vs_universe > 0 else 0
```

## Look-Ahead Bias Guardrail

Required rule:

```text
Use stored ranking rows for trade_date.
Entry uses the next trading day after trade_date.
Exit uses a later trading day after entry_date.
Do not rebuild old ranks from newer features for backtest purposes.
```

If historical ranking rows are missing, recreate them only with that historical `trade_date` snapshot logic.

## Data Status Rules

Symbol-level `data_status`:

- `OK`
- `MISSING_ENTRY_PRICE`
- `NOT_ENOUGH_FORWARD_DATA`
- `MISSING_EXIT_PRICE`
- `PARTIAL_BENCHMARK_DATA`

Summary-level `data_status`:

- `OK`
- `NO_SELECTION`
- `NO_VALID_RETURNS`
- `PARTIAL_FORWARD_DATA`
- `PARTIAL_BENCHMARK_DATA`

Interpretation:

- `NOT_ENOUGH_FORWARD_DATA` usually means the requested holding window extends past the latest loaded price date
- `PARTIAL_BENCHMARK_DATA` means SPY or QQQ comparison could not be computed for part of the set

## Commands

Dry-run:

```powershell
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --holding-days 5,20,60 --dry-run
```

DB upsert:

```powershell
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --holding-days 5,20,60
```

Single strategy:

```powershell
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --strategy TOP20
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --strategy BUY_OR_BETTER
```

Fixed backtest ID:

```powershell
python scripts/backtest_us_stock_rank_strategy.py --start-date 2026-01-01 --end-date 2026-05-12 --backtest-id US_RANK_RULE_V1_TEST
```

Performance report:

```powershell
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format console
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format markdown
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --format csv
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --strategy US_RANK_TOP20 --holding-days 20
python scripts/report_us_stock_rank_backtest.py --backtest-id US_RANK_RULE_V1_TEST --symbol NVDA
```

## Backtest ID

Default format:

```text
US_RANK_RULE_V1_20260101_20260512_HD5_20_60
```

## Validation SQL

Strategy-level aggregate:

```sql
SELECT
    strategy_name,
    selection_rule,
    holding_days,
    COUNT(*) AS test_days,
    AVG(avg_return_pct) AS avg_return,
    AVG(avg_excess_return_vs_spy) AS avg_excess_spy,
    AVG(avg_excess_return_vs_qqq) AS avg_excess_qqq,
    AVG(win_rate) AS avg_win_rate
FROM research.us_stock_rank_backtest_summary
WHERE backtest_id = 'US_RANK_RULE_V1_TEST'
GROUP BY strategy_name, selection_rule, holding_days
ORDER BY strategy_name, holding_days;
```

Top20 by day:

```sql
SELECT
    trade_date,
    strategy_name,
    holding_days,
    selected_count,
    avg_return_pct,
    avg_excess_return_vs_spy,
    win_rate,
    data_status
FROM research.us_stock_rank_backtest_summary
WHERE backtest_id = 'US_RANK_RULE_V1_TEST'
  AND strategy_name = 'US_RANK_TOP20'
ORDER BY trade_date, holding_days;
```

Symbol detail:

```sql
SELECT
    trade_date,
    symbol,
    rank_no,
    recommend_grade,
    total_score,
    holding_days,
    entry_price,
    exit_price,
    return_pct,
    excess_return_vs_spy,
    data_status
FROM research.us_stock_rank_backtest_result
WHERE backtest_id = 'US_RANK_RULE_V1_TEST'
ORDER BY trade_date, rank_no, holding_days;
```

## Operations Notes

- `--dry-run` must not write DB rows
- same `backtest_id` re-run should upsert, not duplicate
- missing rank dates are skipped
- Phase 4 output is research evidence, not a trading instruction

## Phase 4-2 Report Notes

- compare strategies by `holding_days`, `avg_excess_return_vs_spy`, `avg_excess_return_vs_qqq`, `win_rate_vs_spy`, then `avg_return_pct`
- `Best Candidate` is only a candidate for deeper validation
- markdown/csv report output defaults to `outputs/us_stock_backtest`
- if `result` rows are sparse but `summary` rows exist, strategy comparison can still be rendered from summary data

## Phase 4-3 Regime Analysis

Purpose:

- separate performance by market environment instead of only using strategy-wide averages
- compare strategy behavior by month, quarter, and benchmark regime
- leave score-weight changes for Phase 4-4

### Regime Definition

`BULL`
- `SPY 60d return > 0`
- `SPY close > SPY 60d moving average`

`BEAR`
- `SPY 60d return < 0`
- `SPY close < SPY 60d moving average`

`SIDEWAYS`
- any state that is neither `BULL` nor `BEAR`

`HIGH_VOL`
- `SPY 20d volatility >= US_REGIME_SPY_VOL20_HIGH_THRESHOLD`
- or `QQQ 20d volatility >= US_REGIME_QQQ_VOL20_HIGH_THRESHOLD`

`LOW_VOL`
- benchmark volatility is below both high-vol thresholds

Combined `market_regime`:

- `BULL_LOW_VOL`
- `BULL_HIGH_VOL`
- `BEAR_LOW_VOL`
- `BEAR_HIGH_VOL`
- `SIDEWAYS_LOW_VOL`
- `SIDEWAYS_HIGH_VOL`
- `UNKNOWN`

### Regime Tables

`research.us_market_regime_daily`
- daily SPY / QQQ benchmark snapshot
- trend regime, volatility regime, combined regime

`research.us_stock_rank_backtest_regime_summary`
- aggregated regime / month / quarter / year summary
- keyed by `backtest_id + strategy_name + holding_days + regime_type + regime_value`

### Regime Commands

Build regime rows:

```powershell
python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12 --dry-run
python scripts/build_us_market_regime_daily.py --start-date 2026-01-01 --end-date 2026-05-12
```

Analyze by regime:

```powershell
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format console
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format markdown
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --format csv
python scripts/analyze_us_stock_backtest_by_regime.py --backtest-id US_RANK_RULE_V1_TEST --strategy US_RANK_TOP20 --holding-days 20
```

### Period Analysis

- `MONTH`
- `QUARTER`
- `YEAR`

These are grouped from `research.us_stock_rank_backtest_summary.trade_date`.

### Data Quality Rules

- if benchmark lookback is insufficient, use `UNKNOWN` or `INSUFFICIENT_LOOKBACK`
- if benchmark rows do not exist for a backtest `trade_date`, regime analysis reports `UNKNOWN`
- if a regime bucket has fewer than `US_REGIME_MIN_TEST_DAYS_WARNING` rows, print a low-sample warning

### Interpretation Notes

- best/worst regime means a review priority, not a trading signal
- if `BULL` works and `BEAR` does not, the strategy may be trend-following
- if `HIGH_VOL` is weak, stronger risk filtering may be worth reviewing in Phase 4-4
- do not change Rule weights in Phase 4-3

### Current Sample Constraint

The current local sample DB has:

- rank backtest summary on `2026-05-11`
- SPY/QQQ ETF benchmark rows only through `2026-05-08`

Because of that date gap, current regime analysis reports `UNKNOWN` for `2026-05-11`. This is a benchmark freshness issue, not an order-path issue.

## Phase 4-4 Weight Experiment

Purpose:

- keep `RULE_V1_BASELINE` fixed as the production reference
- generate alternative weight candidates without touching the operational ranking table
- compare candidates by excess return, win rate, and regime defense before any forward test

### Baseline

`RULE_V1_BASELINE`

- `momentum_weight = 25`
- `relative_strength_weight = 20`
- `fundamental_weight = 20`
- `growth_weight = 15`
- `valuation_weight = 10`
- `risk_penalty_weight = 10`

### Candidate Set

- `RULE_V1_MOMENTUM_PLUS`
- `RULE_V1_QUALITY_PLUS`
- `RULE_V1_GROWTH_PLUS`
- `RULE_V1_RISK_DEFENSIVE`
- `RULE_V1_VALUE_BALANCED`

### Experiment Tables

`research.us_stock_rule_weight_config`
- weight candidate definitions

`research.us_stock_rank_weight_experiment_result`
- symbol-level reweighted rank snapshot
- separated from `recommend.us_stock_rank_daily`

`research.us_stock_weight_experiment_backtest_summary`
- aggregated candidate performance summary
- stores rank-based comparison fields such as `score_rank` and `risk_adjusted_rank`

### Experiment Commands

```powershell
python scripts/experiment_us_stock_rule_weights.py --start-date 2026-01-01 --end-date 2026-05-12 --weight-configs RULE_V1_BASELINE,RULE_V1_MOMENTUM_PLUS,RULE_V1_QUALITY_PLUS --holding-days 20 --dry-run
python scripts/experiment_us_stock_rule_weights.py --start-date 2026-01-01 --end-date 2026-05-12 --weight-configs ALL --holding-days 5,20,60 --experiment-id US_RULE_WEIGHT_EXP_001
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format console
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format markdown
python scripts/report_us_stock_rule_weight_experiment.py --experiment-id US_RULE_WEIGHT_EXP_001 --format csv
```

### Comparison Criteria

- `avg_excess_return_vs_spy`
- `avg_excess_return_vs_qqq`
- `win_rate_vs_spy`
- `win_rate_vs_qqq`
- `avg_return_bear`
- `avg_return_high_vol`
- sample sufficiency and data stability

### Candidate Judgment

`PROMOTE_CANDIDATE`
- excess return and win rate improve over baseline
- bear / high-vol defense does not worsen
- test days are sufficient

`WATCH_CANDIDATE`
- partial improvement or insufficient sample

`REJECT_CANDIDATE`
- major metrics and defense both worsen versus baseline

### Overfitting Guardrails

- do not promote a candidate from one short window alone
- prefer candidates that remain stable across multiple regimes
- if `TOP5` improves but `TOP10` / `TOP20` degrade, confidence should stay low
- compare win rate, excess return, and drawdown defense together
- any candidate still requires forward testing before operational review

### Safety Note

- Phase 4-4 does not change operational Rule weights
- experiment outputs are research artifacts only
- even `PROMOTE_CANDIDATE` means a forward-test candidate, not a live-trading switch

## Phase 4-5 Forward Test

Purpose:

- track whether newly generated recommendation snapshots lead to favorable outcomes after 5/20/60 trading sessions
- observe current live-date recommendation quality without placing orders

### Forward Test vs Backtest

- Backtest uses historical recommendation snapshots with already-known forward prices.
- Forward Test registers the snapshot first and fills entry/exit performance only when time passes.

### Forward Test vs Paper Trading

- Forward Test tracks recommendation outcomes only.
- Paper Trading simulates a position and portfolio lifecycle.

### Forward Test Tables

`research.us_stock_rank_forward_test`
- strategy/symbol/holding-day detail rows
- status transitions from registration through completion

`research.us_stock_rank_forward_test_summary`
- date/strategy/holding-day progress and completed-performance summary

### Forward Test Commands

```powershell
python scripts/register_us_stock_forward_test.py --trade-date 2026-05-12 --forward-test-id US_RANK_FORWARD_RULE_V1 --holding-days 5,20,60
python scripts/update_us_stock_forward_entry.py --as-of-date 2026-05-13 --forward-test-id US_RANK_FORWARD_RULE_V1
python scripts/update_us_stock_forward_exit.py --as-of-date 2026-06-12 --forward-test-id US_RANK_FORWARD_RULE_V1
python scripts/update_us_stock_forward_summary.py --forward-test-id US_RANK_FORWARD_RULE_V1
python scripts/report_us_stock_forward_test.py --forward-test-id US_RANK_FORWARD_RULE_V1 --format console
```

### State Transition

- `PENDING_ENTRY`: registered, waiting for next-session entry price
- `ACTIVE`: entry price captured and exit horizon still open
- `PENDING_EXIT`: exit horizon reached but exit price still missing
- `COMPLETED`: exit price and forward return computed
- `SKIPPED`: excluded from tracking due to policy or missing mandatory setup
- `ERROR`: invalid price or unexpected update failure

### Look-Ahead Bias Controls

- register from the stored `trade_date` ranking snapshot only
- use the next US trading day as `entry_date`
- use the future holding-session target after `entry_date` as `exit_date`
- do not fill future results before the target date arrives

### Forward Test Completion Gate

- accumulate at least 20 to 60 trade dates
- complete enough 5d and 20d samples to compare strategies
- keep benchmark excess-return coverage stable
- confirm forward results do not diverge materially from backtest expectations
- only after that should Phase 5 Paper Trading be considered

### Safety Note

- Forward Test is not live trading
- Forward Test is not Paper Trading
- Forward Test results alone must not trigger automatic order or broker API behavior

## Phase 4 Next Step

After Phase 4-1:

- Phase 4-2: strategy-level performance report
- Phase 4-3: market-regime and period analysis
- Phase 4-4: score-weight adjustment candidates
- Phase 4-5: forward-test operations structure
