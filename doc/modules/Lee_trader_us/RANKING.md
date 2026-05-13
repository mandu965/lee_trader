# US Stock Ranking Storage Design

## Purpose

Phase 3-2 defines the storage contract for daily US stock recommendation rankings.

- target table: `recommend.us_stock_rank_daily`
- scope: ranking result persistence only
- excluded: score calculation, order generation, paper trading, live trading

## Table Purpose

`recommend.us_stock_rank_daily` stores one canonical ranking row per `trade_date + symbol`.

- `trade_date`: ranking base date
- `symbol`: US stock ticker
- `rank_no`: final rank for the date
- `recommend_grade`: recommendation grade snapshot
- score columns: component scores that Phase 3-3 will calculate
- snapshot columns: copied universe metadata used for review and debugging
- `reason_summary`: short recommendation rationale
- `score_detail_json`: detailed score breakdown for debugging
- `source`: ranking producer tag such as `rule_v1`

## Column Notes

| Column | Meaning |
| --- | --- |
| `trade_date` | ranking base date |
| `symbol` | US ticker |
| `rank_no` | final daily rank |
| `recommend_grade` | `STRONG_BUY` / `BUY` / `WATCH` / `HOLD` / `EXCLUDE` |
| `total_score` | final composite score |
| `momentum_score` | price momentum contribution |
| `relative_strength_score` | SPY/QQQ relative strength contribution |
| `fundamental_score` | profitability / quality contribution |
| `growth_score` | growth contribution |
| `valuation_score` | valuation contribution |
| `risk_score` | negative penalty value, stored as `0` to `-10` |
| `feature_quality_score` | feature coverage / completeness signal |
| `universe_group` | source universe snapshot such as `NASDAQ100,SP500` |
| `company_name`, `sector`, `industry` | metadata snapshot for reporting |
| `market_cap`, `avg_volume` | liquidity / size snapshot |
| `is_etf`, `is_active` | universe status snapshot |
| `data_status`, `exclude_reason` | ranking eligibility state |
| `reason_summary` | short recommendation explanation |
| `score_detail_json` | JSON breakdown for audit/debug |
| `source` | ranking version tag |

## Recommendation Grades

- `STRONG_BUY`: `total_score >= 80` and acceptable quality/risk state
- `BUY`: `70 <= total_score < 80`
- `WATCH`: `60 <= total_score < 70`
- `HOLD`: `50 <= total_score < 60`
- `EXCLUDE`: `total_score < 50` or excluded by universe/quality/risk rules

Phase 3-2 documents the grade contract only. Actual grade assignment starts in Phase 3-3.

## Score Structure

Initial Rule-based score target is 100 points.

- `momentum_score`: 25
- `relative_strength_score`: 20
- `fundamental_score`: 20
- `growth_score`: 15
- `valuation_score`: 10
- `risk_score`: `0` to `-10`

Recommended formula:

```text
total_score =
momentum_score
+ relative_strength_score
+ fundamental_score
+ growth_score
+ valuation_score
+ risk_score
```

## risk_score Convention

`risk_score` is stored as a negative penalty.

- `0`: no penalty
- `-5`: five-point penalty
- `-10`: maximum penalty

This keeps `total_score` additive and easier to debug in SQL.

## score_detail_json Structure

Phase 3-3 and later phases should serialize score evidence into a JSON object with these sections:

- `momentum`
- `relative_strength`
- `fundamental`
- `growth`
- `valuation`
- `risk`

Example shape:

```json
{
  "momentum": {
    "score": 21.5,
    "max_score": 25,
    "reasons": ["20d return above peer median"]
  },
  "relative_strength": {
    "score": 16.2,
    "max_score": 20,
    "spy_relative_return_20d": 0.035,
    "qqq_relative_return_20d": 0.018
  },
  "risk": {
    "score": -3.0,
    "max_penalty": -10,
    "reasons": ["short-term volatility elevated"]
  }
}
```

PostgreSQL `jsonb` is used for direct inspection and future indexing flexibility.

## Query Examples

Top 20 for one date:

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
ORDER BY rank_no
LIMIT 20;
```

Ranking history for one symbol:

```sql
SELECT
    trade_date,
    symbol,
    rank_no,
    recommend_grade,
    total_score
FROM recommend.us_stock_rank_daily
WHERE symbol = 'AAPL'
ORDER BY trade_date DESC;
```

`STRONG_BUY` candidates:

```sql
SELECT
    trade_date,
    rank_no,
    symbol,
    company_name,
    total_score,
    reason_summary
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
  AND recommend_grade = 'STRONG_BUY'
ORDER BY rank_no;
```

## Phase 3-3 Link

Phase 3-3 should:

1. read active candidates from `meta.us_stock_universe`
2. join `feature.us_stock_feature_daily`
3. join `feature.us_stock_relative_strength_daily`
4. join `feature.us_stock_financial_feature`
5. calculate component scores and `recommend_grade`
6. upsert final rows into `recommend.us_stock_rank_daily`

The storage contract is intentionally wider than the first Rule-based version so later model or hybrid rankings can reuse the same table with a different `source` value.

## Phase 3-3 Rule Score Logic

### Purpose

The Phase 3-3 scorer converts daily US candidate data into deterministic ranking rows without touching trading logic.

### Score Structure

```text
total_score =
momentum_score
+ relative_strength_score
+ fundamental_score
+ growth_score
+ valuation_score
+ risk_score
```

`total_score` is capped to `0..100`.

### momentum_score

- `ret_20d > 0`: `+5`
- `ret_60d > 0`: `+5`
- `ret_120d > 0`: `+5`
- `close > ma_20`: `+3`
- `close > ma_60`: `+4`
- `ma_20 > ma_60`: `+3`

### relative_strength_score

- `rs_spy_20d > 0`: `+5`
- `rs_spy_60d > 0`: `+5`
- `rs_qqq_20d > 0`: `+5`
- `rs_qqq_60d > 0`: `+5`

### fundamental_score

- `roe > 0.15`: `+5`
- `operating_margin > 0.15`: `+5`
- `profit_margin > 0.10`: `+4`
- `debt_to_equity <= 0.5 / 1.0 / 2.0`: `+3 / +2 / +1`
- `feature_quality_score >= 60`: `+3`

`debt_to_equity` values above `10` are normalized as percentage-style inputs, for example `42 -> 0.42`.

### growth_score

- `revenue_growth > 0.10`: `+5`
- `revenue_growth > 0.20`: extra `+2`
- `earnings_growth > 0.10`: `+5`
- `earnings_growth > 0.20`: extra `+2`
- both positive: `+1`

### valuation_score

- `0 < trailing_pe <= 20`: `+4`
- `0 < forward_pe <= 25`: `+3`
- `0 < price_to_book <= 5`: `+3`

### risk_score

- `volatility_20d > threshold`: `-3`
- `volatility_60d > threshold`: `-3`
- `ret_20d > overheat threshold`: `-2`
- `feature_quality_score < min threshold`: `-2`
- insufficient price history: `-5`

`risk_score` is capped to `-10..0`.

### Grade Rules

- `STRONG_BUY`: `total_score >= 80`
- `BUY`: `70 <= total_score < 80`
- `WATCH`: `60 <= total_score < 70`
- `HOLD`: `50 <= total_score < 60`
- `EXCLUDE`: below `50` or forced exclusion

Forced exclusion:

- inactive universe row
- leveraged ETF
- inverse ETF
- missing price for the effective trade date
- feature quality below threshold for non-ETF, or ETF when the ETF quality flag is enabled

### score_detail_json

The stored JSON includes:

- per-section `score`, `inputs`, `missing_fields`, and `reasons`
- root `meta` with raw total, capped total, data status, and exclusion reason

### reason_summary

The summary is Rule-based.

- mention the strongest 1-2 positive sections
- mention one risk penalty when present
- mention non-ready data status when needed
- keep it within 1-2 short sentences

### Missing Data Rules

- missing universe row: excluded before ranking
- missing price for the effective trade date: `EXCLUDE`
- missing fundamental / relative strength / valuation input: section score `0`
- missing fields are recorded in `score_detail_json`
- source feature tables are never backfilled with `0`

### Commands

```powershell
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --dry-run
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --symbols AAPL,MSFT,NVDA --dry-run
python scripts/calculate_us_stock_rule_scores.py --trade-date 2026-05-12 --top-n 20
```

### Effective Trade Date Note

If the requested date is not a US session date or newer than the latest loaded US market data, the scorer logs the resolved effective US trade date and writes rows for that effective date.

## Phase 3-4 Top Rank Report

### Purpose

Phase 3-4 exposes `recommend.us_stock_rank_daily` in operator-facing report formats.

- output purpose: review, validation, and audit
- excluded purpose: order generation, buy/sell execution, live-trading linkage

### Script

- wrapper: `scripts/report_us_stock_top_rank.py`
- implementation: `python/us/report_us_stock_top_rank.py`

### Commands

```powershell
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --grade BUY
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --grade STRONG_BUY
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --symbol NVDA
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --format markdown
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --format csv
python scripts/report_us_stock_top_rank.py --trade-date 2026-05-12 --top-n 20 --auto-calculate
```

### Default Query Rule

Default Top N selection uses:

```sql
SELECT
    trade_date,
    rank_no,
    symbol,
    company_name,
    sector,
    industry,
    recommend_grade,
    total_score,
    momentum_score,
    relative_strength_score,
    fundamental_score,
    growth_score,
    valuation_score,
    risk_score,
    feature_quality_score,
    reason_summary
FROM recommend.us_stock_rank_daily
WHERE trade_date = DATE '2026-05-12'
  AND recommend_grade <> 'EXCLUDE'
  AND rank_no IS NOT NULL
ORDER BY rank_no ASC
LIMIT 20;
```

### Console Output

- fixed-width table for fast operator review
- `reason_summary` is truncated in console output
- daily summary statistics are printed after the Top N table

### Markdown Output

- report title, source metadata, and exclusion note
- Top N summary table
- per-symbol recommendation reason section
- daily check-point statistics

Markdown output is saved under `outputs/us_stock_top_rank/` by default.

### CSV Output

CSV output is saved under `outputs/us_stock_top_rank/` by default with UTF-8 encoding.

CSV columns:

- `trade_date`
- `rank_no`
- `symbol`
- `company_name`
- `sector`
- `industry`
- `recommend_grade`
- `total_score`
- `momentum_score`
- `relative_strength_score`
- `fundamental_score`
- `growth_score`
- `valuation_score`
- `risk_score`
- `feature_quality_score`
- `reason_summary`

### Symbol Detail

`--symbol` prints one-row detail with:

- rank and grade
- component score breakdown
- `reason_summary`
- `score_detail_json` preview in console
- full `score_detail_json` in markdown when requested

### Summary Statistics

The report script calculates daily statistics for the full ranking snapshot:

- total ranked rows
- eligible rows after `EXCLUDE`
- `STRONG_BUY` / `BUY` / `WATCH` / `HOLD` / `EXCLUDE` counts
- average / max / min `total_score`
- average `feature_quality_score`
- average `momentum_score`
- average `relative_strength_score`
- average `fundamental_score`
- average `risk_score`

### Missing Data Handling

- no rows for `trade_date`: prompt to run `calculate_us_stock_rule_scores.py`
- empty Top N after filters: prompt to review `EXCLUDE` or grade filters
- missing symbol row: explain that the symbol may be outside the universe or the ranking may not be generated
- invalid `score_detail_json`: log a warning and keep the report output alive
- missing output directory: create it automatically

### Optional Notification

- `US_RANK_REPORT_EMAIL_ENABLED=false` by default
- when enabled, the existing notifier module is used on a best-effort basis
- notification failure must not break report generation

### Safety Note

Phase 3-4 reports are recommendation artifacts only.

- not a buy order
- not a sell order
- not connected to Korean real-trading logic

## Phase 3-5 Explainability Design

### reason_summary

`reason_summary` is still Rule-based.

Each summary should try to include:

- one or two strength drivers
- one weakness or risk note
- data-quality or partial-data note when applicable
- ETF-specific wording for ETF rows
- short grade rationale

### reason_category

Stored in `score_detail_json.meta.reason_category`.

Recommended values:

- `MOMENTUM_LEADER`
- `RELATIVE_STRENGTH_LEADER`
- `QUALITY_COMPOUNDER`
- `GROWTH_LEADER`
- `VALUE_CANDIDATE`
- `ETF_CORE`
- `WATCHLIST`
- `RISK_HIGH`
- `DATA_INSUFFICIENT`
- `EXCLUDED`

### reason_tags

Stored in `score_detail_json.meta.reason_tags`.

Representative tags:

- `strong_momentum`
- `weak_momentum`
- `spy_outperform`
- `qqq_outperform`
- `high_roe`
- `high_margin`
- `positive_growth`
- `cheap_valuation`
- `expensive_valuation`
- `low_volatility`
- `high_volatility`
- `low_quality_data`
- `etf`
- `leveraged_etf_excluded`
- `inverse_etf_excluded`
- `missing_fundamental`
- `missing_relative_strength`

### score_detail_json meta extension

The `meta` section now includes:

```json
{
  "symbol": "NVDA",
  "trade_date": "2026-05-11",
  "data_status": "PARTIAL_DATA",
  "exclude_reason": "Total score below HOLD threshold 50.",
  "reason_category": "DATA_INSUFFICIENT",
  "reason_tags": [
    "strong_momentum",
    "missing_relative_strength",
    "missing_fundamental",
    "expensive_valuation"
  ],
  "grade_rationale": "Recommendation was excluded because total score below hold threshold 50.",
  "data_flags": {
    "missing_price_feature": false,
    "missing_relative_strength": true,
    "missing_fundamental": true,
    "low_feature_quality": false,
    "insufficient_price_history": false
  }
}
```

### data_status values

- `OK`: required ranking inputs are present
- `MISSING_PRICE_FEATURE`: price-derived inputs are missing for the effective trade date
- `MISSING_FUNDAMENTAL`: non-ETF row is missing most fundamental inputs
- `MISSING_RELATIVE_STRENGTH`: relative-strength inputs are missing
- `LOW_FEATURE_QUALITY`: feature quality threshold is below policy minimum
- `PARTIAL_DATA`: ranking is still possible but multiple optional layers are missing
- `EXCLUDED`: universe or policy exclusion
- `ERROR`: reserved for future hard calculation failures

### exclude_reason rule

`exclude_reason` should be present for every `EXCLUDE` row.

Typical cases:

- universe/policy exclusion
- missing effective-date price features
- feature-quality exclusion
- total score below `HOLD` cutoff

### Validation Script

- script: `scripts/validate_us_stock_rank_daily.py`
- module: `python/us/validate_us_stock_rank_daily.py`

Commands:

```powershell
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --top-n 20
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --fail-on-error
python scripts/validate_us_stock_rank_daily.py --trade-date 2026-05-12 --output markdown
```

Validation checks:

- component score ranges
- `total_score` range
- grade consistency
- `score_detail_json` parsing
- missing `reason_summary`
- missing `exclude_reason` for `EXCLUDE`
- anomaly warnings such as very high `20d` return or extreme valuation multiples

### Top 20 Detail Usage

- default Top N report shows `Category`, `Tags`, `Risk`, and `Reason`
- `--symbol` expands per-section reasons and missing-field notes
- `--show-excluded` prints excluded rows with `exclude_reason` and `data_status`

### Operator Warning Note

Validation warnings are not trade signals.

- they indicate review-worthy conditions
- they do not override ranking policy by themselves
