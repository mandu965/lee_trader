# US Stock Ranking V1

> 문서 역할: `상세 참고 문서`
>
> Phase 3 ranking 구조와 운영 예시를 자세히 설명하는 문서다.

## Purpose

This document is the Phase 3 integrated operations guide for the Project C US stock Rule-based recommendation and ranking engine v1.

Phase 3 scope:

- recommendation universe management
- Rule-based score calculation
- ranking result storage
- Top N reporting
- excluded-row review
- validation and anomaly checks

This is not auto-trading.

- Phase 3 results are recommendation and observation priority only.
- Phase 3 results must not be treated as automatic buy or sell instructions.
- Korean live trading logic and KIS order execution remain out of scope.

## Phase 3 Goal

Phase 3 builds an operator-reviewable US stock recommendation ranking engine.

- Phase 3-1: recommendation universe master
- Phase 3-2: ranking result storage contract
- Phase 3-3: Rule-based scoring
- Phase 3-4: Top N report generation
- Phase 3-5: explainability and validation
- Phase 3-6: operations documentation and sample execution summary

## Data Flow

```text
[Universe]
meta.us_stock_universe
    ->
[Price / Technical Feature]
feature.us_stock_feature_daily
    ->
[Fundamental Feature]
feature.us_stock_financial_feature
    ->
[Relative Strength Feature]
feature.us_stock_relative_strength_daily
    ->
[Rule Score Calculation]
scripts/calculate_us_stock_rule_scores.py
    ->
[Ranking Result]
recommend.us_stock_rank_daily
    ->
[Report / Validation]
scripts/report_us_stock_top_rank.py
scripts/validate_us_stock_rank_daily.py
```

## Stage I/O

`scripts/init_us_stock_universe.py`

Input:
- static and source universe membership
- Project C universe policy flags

Output:
- `meta.us_stock_universe`

`python -m python.us.build_us_features`

Input:
- `market.us_stock_daily_price`

Output:
- `feature.us_stock_feature_daily`

`python -m python.us.build_us_financial_features`

Input:
- `raw.us_stock_financial_statement`
- `raw.us_stock_financial_metric`

Output:
- `feature.us_stock_financial_feature`

`python -m python.us.build_us_relative_strength_features`

Input:
- `market.us_stock_daily_price`

Output:
- `feature.us_stock_relative_strength_daily`

`scripts/calculate_us_stock_rule_scores.py`

Input:
- `meta.us_stock_universe`
- `feature.us_stock_feature_daily`
- `feature.us_stock_financial_feature`
- `feature.us_stock_relative_strength_daily`
- `market.us_stock_daily_price`

Output:
- `recommend.us_stock_rank_daily`

`scripts/report_us_stock_top_rank.py`

Input:
- `recommend.us_stock_rank_daily`

Output:
- console output
- markdown report under `outputs/us_stock_top_rank/`
- csv report under `outputs/us_stock_top_rank/`

`scripts/validate_us_stock_rank_daily.py`

Input:
- `recommend.us_stock_rank_daily`

Output:
- validation summary
- optional markdown validation report under `outputs/us_stock_top_rank/`

## Related Tables

`meta.us_stock_universe`
- recommendation candidate master
- active/inactive policy
- ETF / leveraged / inverse flags
- exclude policy metadata

`market.us_stock_daily_price`
- raw US price history
- source for price-derived snapshots and fallbacks

`feature.us_stock_feature_daily`
- price/technical features
- used by `momentum_score`

`feature.us_stock_financial_feature`
- financial quality, growth, and valuation inputs
- used by `fundamental_score`, `growth_score`, `valuation_score`

`feature.us_stock_relative_strength_daily`
- SPY/QQQ relative strength features
- used by `relative_strength_score`

`recommend.us_stock_rank_daily`
- final Rule-based ranking result table
- source for Top N reports and validation

## Related Scripts

`scripts/init_us_stock_universe.py`
- initialize or refresh recommendation universe rows

`scripts/calculate_us_stock_rule_scores.py`
- calculate Rule-based scores
- assign grade and rank
- upsert `recommend.us_stock_rank_daily`

`scripts/report_us_stock_top_rank.py`
- show Top N eligible rows
- show symbol detail
- show excluded rows
- build console/markdown/csv outputs

`scripts/validate_us_stock_rank_daily.py`
- validate stored ranking rows
- check score range, JSON, grade consistency, exclude reason, and anomaly warnings

Supporting module commands:

- `python -m python.us.build_us_features`
- `python -m python.us.build_us_financial_features`
- `python -m python.us.build_us_relative_strength_features`

No dedicated `scripts/*.py` wrappers exist yet for those three feature builders.

## ENV

Core Phase 3 settings:

```env
US_UNIVERSE_MIN_MARKET_CAP=10000000000
US_UNIVERSE_MIN_AVG_VOLUME=1000000
US_UNIVERSE_MIN_FEATURE_QUALITY_SCORE=40
US_UNIVERSE_INCLUDE_ETF=true
US_UNIVERSE_EXCLUDE_LEVERAGED=true
US_UNIVERSE_EXCLUDE_INVERSE=true

US_RANK_MIN_FEATURE_QUALITY_SCORE=40
US_RANK_APPLY_FUNDAMENTAL_QUALITY_TO_ETF=false
US_RANK_VOLATILITY_20D_THRESHOLD=0.05
US_RANK_VOLATILITY_60D_THRESHOLD=0.04
US_RANK_RETURN_20D_OVERHEAT_THRESHOLD=0.25
US_RANK_STRONG_BUY_SCORE=80
US_RANK_BUY_SCORE=70
US_RANK_WATCH_SCORE=60
US_RANK_HOLD_SCORE=50

US_RANK_REPORT_EMAIL_ENABLED=false
US_RANK_REPORT_TOP_N=20
US_RANK_REPORT_OUTPUT_DIR=outputs/us_stock_top_rank
US_RANK_REPORT_LOG_LEVEL=INFO
```

Meaning summary:

- universe vars: recommendation-universe filter policy
- rank vars: score cutoffs and risk thresholds
- report vars: report size, output location, notifier toggle

## Standard Daily Execution Order

This is a recommendation and validation pipeline, not an order pipeline.

1. Refresh recommendation universe  
   `python scripts/init_us_stock_universe.py`

2. Build baseline price features  
   `python -m python.us.build_us_features --universe NASDAQ100`

3. Build financial features  
   `python -m python.us.build_us_financial_features`

4. Build relative strength features  
   `python -m python.us.build_us_relative_strength_features`

5. Calculate Rule ranking  
   `python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --top-n 20`

6. Review Top N console report  
   `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format console`

7. Generate markdown report  
   `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format markdown`

8. Generate csv report  
   `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format csv`

9. Validate stored rows  
   `python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12`

If the requested Korea-local date is not a US session date, score calculation may resolve to the previous US trade date.

## Sample Commands

Development verification:

```powershell
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --symbols AAPL,MSFT,NVDA --dry-run
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format console
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --show-excluded --limit 50
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12
```

Auto-calculate when a rank snapshot is missing:

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format console --auto-calculate
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA --auto-calculate
```

## Top 20 Report Review

Default command:

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format console
```

What to check:

- `rank_no` ascending order
- `recommend_grade <> EXCLUDE` in default Top N
- `Category`, `Tags`, `Risk`, and `Reason`
- validation summary block after the table

## Symbol Detail Review

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA
```

What to check:

- grade and total score
- section-by-section score breakdown
- per-section reasons and missing fields
- `reason_category`
- `reason_tags`
- `data_status`
- `exclude_reason`

## Excluded Rows Review

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --show-excluded --limit 50
```

What to check:

- `recommend_grade = EXCLUDE`
- `exclude_reason`
- `data_status`
- `feature_quality_score`
- repeated data-insufficiency patterns

`EXCLUDE` is not a sell signal.

It only means the symbol is excluded from recommendation ranking for that snapshot.

## Validation Script

```powershell
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --top-n 20
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --fail-on-error
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --output markdown
```

Validation checks:

- component score range
- total score range
- grade consistency
- `score_detail_json` parsing
- missing `reason_summary`
- missing `exclude_reason` on `EXCLUDE`
- anomaly warnings

## Score Formula

```text
total_score =
momentum_score
+ relative_strength_score
+ fundamental_score
+ growth_score
+ valuation_score
+ risk_score
```

Ranges:

```text
momentum_score:           0 ~ 25
relative_strength_score:  0 ~ 20
fundamental_score:        0 ~ 20
growth_score:             0 ~ 15
valuation_score:          0 ~ 10
risk_score:              -10 ~ 0
total_score:              0 ~ 100
```

Grade cutoffs:

```text
STRONG_BUY: total_score >= 80
BUY:        total_score >= 70
WATCH:      total_score >= 60
HOLD:       total_score >= 50
EXCLUDE:    total_score < 50 or excluded by policy/data rules
```

These grades are ranking priorities, not automatic buy signals.

## Explainability Rules

`reason_summary`
- short operator-facing summary
- built only from deterministic Rule logic

`reason_category`
- dominant explanation type

`reason_tags`
- detailed strength, weakness, risk, and data-coverage tags

Reason categories:

```text
MOMENTUM_LEADER
RELATIVE_STRENGTH_LEADER
QUALITY_COMPOUNDER
GROWTH_LEADER
VALUE_CANDIDATE
ETF_CORE
WATCHLIST
RISK_HIGH
DATA_INSUFFICIENT
EXCLUDED
```

Representative tags:

```text
strong_momentum
weak_momentum
spy_outperform
qqq_outperform
high_roe
high_margin
positive_growth
cheap_valuation
expensive_valuation
low_volatility
high_volatility
low_quality_data
etf
missing_fundamental
missing_relative_strength
```

## Data Status And Exclude Reason

`data_status` values:

```text
OK
PARTIAL_DATA
MISSING_PRICE_FEATURE
MISSING_FUNDAMENTAL
MISSING_RELATIVE_STRENGTH
LOW_FEATURE_QUALITY
EXCLUDED
ERROR
```

Interpretation:

- `OK`: required inputs are present
- `PARTIAL_DATA`: ranking was still computed but optional layers are materially incomplete
- `MISSING_PRICE_FEATURE`: effective-date price-derived inputs are missing
- `MISSING_FUNDAMENTAL`: non-ETF fundamental layer is missing
- `MISSING_RELATIVE_STRENGTH`: relative-strength layer is missing
- `LOW_FEATURE_QUALITY`: quality threshold is below policy minimum
- `EXCLUDED`: policy or universe exclusion
- `ERROR`: reserved for future hard-failure handling

Typical `exclude_reason` values in current implementation:

- universe row inactive
- leveraged ETF excluded
- inverse ETF excluded
- price-derived features missing for the effective trade date
- feature quality below minimum threshold
- total score below HOLD threshold

Again:

- `EXCLUDE` is not a sell signal
- `EXCLUDE` means the symbol was excluded from that recommendation snapshot

## Direct SQL

Top 20 eligible rows:

```sql
SELECT
    trade_date,
    rank_no,
    symbol,
    company_name,
    recommend_grade,
    total_score,
    momentum_score,
    relative_strength_score,
    fundamental_score,
    growth_score,
    valuation_score,
    risk_score,
    reason_summary
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
  AND recommend_grade <> 'EXCLUDE'
ORDER BY rank_no
LIMIT 20;
```

Symbol history:

```sql
SELECT
    trade_date,
    symbol,
    rank_no,
    recommend_grade,
    total_score,
    reason_summary
FROM recommend.us_stock_rank_daily
WHERE symbol = 'NVDA'
ORDER BY trade_date DESC;
```

Excluded rows:

```sql
SELECT
    trade_date,
    symbol,
    company_name,
    recommend_grade,
    total_score,
    data_status,
    exclude_reason
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
  AND recommend_grade = 'EXCLUDE'
ORDER BY symbol;
```

Grade distribution:

```sql
SELECT
    recommend_grade,
    COUNT(*) AS cnt,
    AVG(total_score) AS avg_total_score
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
GROUP BY recommend_grade
ORDER BY recommend_grade;
```

## Troubleshooting

### Top 20 Is Empty

Check:

1. whether `recommend.us_stock_rank_daily` has rows for that date
2. whether `scripts/calculate_us_stock_rule_scores.py` ran
3. whether `meta.us_stock_universe` has active rows
4. whether the effective-date feature layers exist
5. whether all rows became `EXCLUDE`

If the request date is `2026-05-12` and US market data only exists up to `2026-05-11`, the score script may write rows for `2026-05-11`.

### A Symbol Is Missing

Check:

1. whether the symbol exists in `meta.us_stock_universe`
2. whether `is_active = true`
3. whether price features exist
4. whether missing fundamental / relative strength coverage pushed the row to `EXCLUDE`
5. `exclude_reason`

### Scores Look Too High Or Too Low

Check:

1. `score_detail_json`
2. feature value units
3. return / growth / debt-to-equity normalization
4. `scripts/validate_us_stock_rank_daily.py`
5. anomaly warnings

### JSON Parsing Fails

Check:

1. `score_detail_json` storage type
2. date / numeric serialization
3. whether the snapshot should be recalculated

### DB Connection Fails

Check:

1. `DATABASE_URL`
2. `PG_CONNECT_TIMEOUT`
3. whether `.env` accidentally contains inline comments on the same `DATABASE_URL` line

Known local issue during the Phase 3-6 sample run:

- the workspace `.env` contained an inline comment on `DATABASE_URL`
- sample execution used a temporary shell override to avoid that parse issue

## Phase 3 Completion Checklist

```text
[Phase 3 Completion Checklist]

DB
- [ ] meta.us_stock_universe exists
- [ ] recommend.us_stock_rank_daily exists
- [ ] ranking indexes exist

Data
- [ ] active universe rows exist
- [ ] effective-date price features exist
- [ ] effective-date financial features exist
- [ ] effective-date relative strength features exist

Calculation
- [ ] Rule scorer runs
- [ ] total_score stays within 0..100
- [ ] risk_score stays within -10..0
- [ ] rank_no is assigned
- [ ] recommend_grade is assigned

Reporting
- [ ] Top 20 console output runs
- [ ] markdown report runs
- [ ] csv report runs
- [ ] symbol detail output runs
- [ ] excluded-row output runs

Validation
- [ ] validate script runs
- [ ] score_detail_json parses
- [ ] EXCLUDE rows have exclude_reason
- [ ] warning/error summary renders

Operations
- [ ] ENV doc updated
- [ ] execution order documented
- [ ] troubleshooting documented
- [ ] separation from auto-trading documented
```

## Phase 4 Entry Conditions

Phase 4 is backtest and performance validation, not auto-trading.

Entry conditions:

1. structure exists to accumulate at least 20 to 60 trading days of ranking snapshots
2. re-running the same date produces stable upsert behavior
3. Top N output is understandable to an operator
4. `EXCLUDE` and data-missing reasons remain explicit
5. validation script catches score anomalies
6. ranking results can be linked to return labels
7. the ranking path remains completely separated from real-trading logic

Required interpretation:

- Phase 3 results are recommendation and observation priority.
- Phase 3 results alone must not trigger automatic buying or live trading.
- Only after Phase 4 backtest and performance validation should paper trading or auto-trading be considered.

## Phase 4 Hand-Off

Phase 4 starts after Phase 3 ranking snapshots are stable enough to evaluate historically.

Phase 4-1 scope:

- read stored rows from `recommend.us_stock_rank_daily`
- compute next-session entry and forward exit returns
- compare against SPY, QQQ, and ranked-universe average
- write research-only outputs to:
  - `research.us_stock_rank_backtest_result`
  - `research.us_stock_rank_backtest_summary`

Reference:

- [US_STOCK_BACKTEST_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md)

## Sample Execution Result

Sample executions were run on May 12, 2026 in the current workspace.

Observed date behavior:

- requested date: `2026-05-12`
- effective US rank date resolved by scorer: `2026-05-11`

Result summary:

- `python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --symbols AAPL,MSFT,NVDA --dry-run`: success
- `python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12`: success, wrote `2026-05-11` rows
- `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --format console --auto-calculate`: no eligible Top N rows because the current snapshot is entirely `EXCLUDE`
- `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-11 --symbol NVDA`: success
- `python scripts/report_us_stock_top_rank.py --trade-date 2026-05-11 --show-excluded --limit 50 --format markdown`: success
- `python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-11`: success

Validation sample metrics for `2026-05-11`:

- total checked: `545`
- valid: `538`
- warnings: `7`
- errors: `0`
- invalid JSON: `0`
- exclude without reason: `0`

The dominant current-state issue is not score corruption but data completeness:

- many rows are `PARTIAL_DATA`
- all current rows are `EXCLUDE`
- fundamental and relative-strength coverage is incomplete in the present sample dataset

## Related Documents

- [README.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/README.md)
- [CONTEXT.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/CONTEXT.md)
- [FLOW.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/FLOW.md)
- [ENV.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/ENV.md)
- [OPERATIONS.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/OPERATIONS.md)
- [RANKING.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/RANKING.md)
