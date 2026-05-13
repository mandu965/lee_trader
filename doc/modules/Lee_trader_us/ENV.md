# Lee_trader_us ENV

> 문서 역할: `현재 기준 문서`
>
> 현재 사용 중인 US 관련 ENV와 safety flag 기준 문서다. 설정을 바꾸기 전 먼저 확인한다.

## Purpose

This document describes the environment variables reserved for Project C US stock preparation.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_STOCK_ENABLED` | `false` | Master switch for the US stock pipeline | US-only pipeline | `run_us_daily_pipeline.py` skips unless `--force` is provided |
| `US_STOCK_UNIVERSE` | `NASDAQ100` | Universe tag to manage | universe | Phase 1 default universe |
| `US_STOCK_DATA_SOURCE` | `yfinance` | US daily price data source | price collection | Phase 1 supports only `yfinance` |
| `US_STOCK_PRICE_BACKFILL_YEARS` | `5` | Default initial history range | backfill | Overridden by explicit start date |
| `US_STOCK_PRICE_START_DATE` | blank | Explicit backfill start date | backfill | Optional |
| `US_STOCK_PRICE_END_DATE` | blank | Explicit collection end date | backfill / replay | Optional |
| `US_STOCK_STALE_DAYS_LIMIT` | `3` | Data freshness threshold | quality validation | Calendar-day based in Phase 1 |
| `US_STOCK_BATCH_SIZE` | `20` | Batch size for collection | price collection | Used by `download_us_prices.py` |
| `US_STOCK_REQUEST_SLEEP_SEC` | `1` | Sleep between batches | price collection | Used by `download_us_prices.py` |
| `US_PAPER_TRADING_ENABLED` | `false` | Reserved for future paper trading | future phases | Not used in this phase |
| `US_LIVE_TRADING_ENABLED` | `false` | Reserved for future live trading | future phases | Must not trigger any order code |
| `US_LIVE_BROKER` | `none` | Reserved broker selector | future phases | Must remain unused in this phase |

## Phase 1-6 Rules

- `run_us_daily_pipeline.py` is independent from Korean schedulers.
- `US_STOCK_ENABLED=false` means the pipeline is skipped by default.
- `--force` is only a manual override for standalone execution.
- `US_PAPER_TRADING_ENABLED` is ignored in this phase.
- `US_LIVE_TRADING_ENABLED` is ignored in this phase and must not trigger any order function.

## Phase 3-1 Universe Filter Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_UNIVERSE_MIN_MARKET_CAP` | `10000000000` | Minimum recommendation market cap | recommendation universe | Uses latest financial feature `market_cap` or meta override |
| `US_UNIVERSE_MIN_AVG_VOLUME` | `1000000` | Minimum recent average volume | recommendation universe | Uses recent 20 trading days from `market.us_stock_daily_price` |
| `US_UNIVERSE_MIN_FEATURE_QUALITY_SCORE` | `40` | Minimum feature-quality threshold | recommendation universe | Default derived score is based on feature presence |
| `US_UNIVERSE_INCLUDE_ETF` | `true` | Include standard ETFs in the recommendation universe | recommendation universe | Leveraged/inverse ETF rules still apply |
| `US_UNIVERSE_EXCLUDE_LEVERAGED` | `true` | Exclude leveraged ETFs | recommendation universe | Recommended to keep enabled |
| `US_UNIVERSE_EXCLUDE_INVERSE` | `true` | Exclude inverse ETFs | recommendation universe | Recommended to keep enabled |

### Universe Notes

- `meta.us_stock_universe` is a recommendation master table, not the same as `market.us_stock_universe`.
- `market.us_stock_universe` tracks collection membership snapshots.
- `meta.us_stock_universe` tracks recommendation eligibility, ETF flags, and exclude reasons.

## Phase 3-2 Ranking Table Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_RANKING_TABLE_ENABLED` | `0` | Reserved ranking table switch | ranking storage | Phase 3-2 keeps scoring disabled even if this is set |
| `US_RANKING_TARGET_TABLE` | `recommend.us_stock_rank_daily` | Ranking result table | ranking storage | Used by validation/helper code |
| `US_RANKING_DEFAULT_SOURCE` | `rule_v1` | Default ranking producer tag | ranking storage | Future versions may use `model_v1` or `hybrid_v1` |
| `US_RANKING_VERIFY_SAMPLE_TRADE_DATE` | `2099-12-31` | Safe validation date for sample inserts | validation | Avoids collisions with real trading dates |
| `US_RANKING_LOG_LEVEL` | `INFO` | Validation helper log level | validation | Used by `verify_us_stock_rank_table.py` |

### Ranking Notes

- `recommend.us_stock_rank_daily` is storage only in Phase 3-2.
- `recommend_grade` is documented now and calculated in Phase 3-3.
- `risk_score` is stored as a negative penalty.
- `score_detail_json` is stored as PostgreSQL `jsonb`.

## Phase 3-3 Rule Ranking Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_RULE_RANKING_ENABLED` | `1` | Master switch for the Rule scorer | ranking | Does not connect to order logic |
| `US_RANK_MIN_FEATURE_QUALITY_SCORE` | `40` | Minimum quality threshold for non-ETF exclusion | ranking | ETF handling is split by the next flag |
| `US_RANK_APPLY_FUNDAMENTAL_QUALITY_TO_ETF` | `false` | Apply the same minimum-quality exclusion to ETFs | ranking | Default keeps ETF quality handling looser |
| `US_RANK_VOLATILITY_20D_THRESHOLD` | `0.05` | 20d volatility penalty threshold | ranking risk | Expected unit is raw daily stddev |
| `US_RANK_VOLATILITY_60D_THRESHOLD` | `0.04` | 60d volatility penalty threshold | ranking risk | Computed from price history fallback |
| `US_RANK_RETURN_20D_OVERHEAT_THRESHOLD` | `0.25` | 20d overheat threshold | ranking risk | Above this level the risk penalty increases |
| `US_RANK_STRONG_BUY_SCORE` | `80` | `STRONG_BUY` cutoff | ranking grade | |
| `US_RANK_BUY_SCORE` | `70` | `BUY` cutoff | ranking grade | |
| `US_RANK_WATCH_SCORE` | `60` | `WATCH` cutoff | ranking grade | |
| `US_RANK_HOLD_SCORE` | `50` | `HOLD` cutoff | ranking grade | |

### Rule Ranking Notes

- The scorer uses `risk_score` as a negative penalty from `0` to `-10`.
- If the requested date is not a US trading date, the scorer automatically uses the latest available US trade date on or before the requested date.
- Missing fundamental or relative-strength inputs become `0` score for that section and are recorded in `score_detail_json`.

## Phase 3-4 Rank Report Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_RANK_REPORT_OUTPUT_DIR` | `outputs/us_stock_top_rank` | Output directory for markdown/csv reports | reporting | Directory is auto-created when needed |
| `US_RANK_REPORT_TOP_N` | `20` | Default Top N report size | reporting | CLI `--top-n` overrides it |
| `US_RANK_REPORT_EMAIL_ENABLED` | `false` | Optional notifier/email-style delivery switch | reporting notification | Default must stay off |
| `US_RANK_REPORT_LOG_LEVEL` | `INFO` | Report script log level | reporting | Console output is still plain stdout |

### Rank Report Notes

- `report_us_stock_top_rank.py` is a read/report tool only.
- The generated Top N report is not a buy/sell instruction.
- If `US_RANK_REPORT_EMAIL_ENABLED=false`, the script only writes local files and console output.
- If `US_RANK_REPORT_EMAIL_ENABLED=true`, the existing notifier module is used on a best-effort basis and report failure must not block the ranking pipeline.

## Phase 3-5 Explainability Notes

- `reason_summary` remains Rule-based and does not call any LLM API.
- `reason_category` and `reason_tags` are stored inside `score_detail_json.meta`.
- `data_status` is an operator-review field and is separate from order logic.
- validation warnings are review diagnostics, not buy/sell signals.

## Phase 4-2 Backtest Report Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BACKTEST_REPORT_OUTPUT_DIR` | `outputs/us_stock_backtest` | Output directory for markdown/csv backtest reports | backtest reporting | Directory is auto-created when needed |
| `US_BACKTEST_REPORT_DEFAULT_FORMAT` | `console` | Default backtest report format | backtest reporting | CLI `--format` overrides it |
| `US_BACKTEST_REPORT_RECENT_DAYS` | `10` | Recent-day section size | backtest reporting | Used in markdown trend sections |
| `US_BACKTEST_REPORT_BEST_WORST_LIMIT` | `10` | Best/worst day section size | backtest reporting | Used by strategy detail and markdown output |
| `US_BACKTEST_MIN_TEST_DAYS_WARNING` | `30` | Minimum test-day threshold for warning text | backtest interpretation | Short samples trigger caution text |
| `US_BACKTEST_MISSING_RATE_WARNING` | `0.1` | Missing-rate warning threshold | backtest interpretation | Values above this lower confidence wording |
| `US_BACKTEST_REPORT_LOG_LEVEL` | `INFO` | Backtest report log level | backtest reporting | Console report still prints plain stdout |

### Backtest Report Notes

- `report_us_stock_rank_backtest.py` reads research tables only.
- `Best Candidate` means a follow-up review candidate, not a live-trading recommendation.
- Backtest report output must stay separated from order execution and Korean trading paths.

## Phase 4-3 Market Regime Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_REGIME_SPY_VOL20_HIGH_THRESHOLD` | `0.025` | SPY 20d volatility threshold for `HIGH_VOL` | regime analysis | Raw daily stddev threshold |
| `US_REGIME_QQQ_VOL20_HIGH_THRESHOLD` | `0.030` | QQQ 20d volatility threshold for `HIGH_VOL` | regime analysis | Used as a secondary high-vol trigger |
| `US_REGIME_MIN_TEST_DAYS_WARNING` | `20` | Minimum sample size before warning text | regime analysis | Small regime buckets print caution text |
| `US_REGIME_REPORT_OUTPUT_DIR` | `outputs/us_stock_backtest` | Output directory for markdown/csv regime reports | regime reporting | Shared with backtest report outputs by default |
| `US_REGIME_REPORT_DEFAULT_FORMAT` | `console` | Default regime analysis format | regime reporting | CLI `--format` overrides it |
| `US_REGIME_LOG_LEVEL` | `INFO` | Regime build/report log level | regime build/report | Console report still writes plain stdout |

### Market Regime Notes

- `build_us_market_regime_daily.py` uses `trade_date`-time information only.
- `HIGH_VOL` / `LOW_VOL` is based on benchmark 20d volatility and must not use future returns.
- `analyze_us_stock_backtest_by_regime.py` is a research/report layer only.
- Regime analysis output is not an order signal and must remain separated from Korean trading logic.

## Phase 4-4 Weight Experiment Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_WEIGHT_EXPERIMENT_OUTPUT_DIR` | `outputs/us_stock_weight_experiment` | Output directory for markdown/csv experiment reports | weight experiment reporting | Directory is auto-created when needed |
| `US_WEIGHT_EXPERIMENT_DEFAULT_HOLDING_DAYS` | `5,20,60` | Default holding-day set | weight experiment | CLI `--holding-days` overrides it |
| `US_WEIGHT_EXPERIMENT_DEFAULT_STRATEGIES` | `TOP5,TOP10,TOP20,BUY_OR_BETTER` | Default strategy set | weight experiment | Uses the same strategy aliases as Phase 4-1 |
| `US_WEIGHT_EXPERIMENT_MIN_TEST_DAYS` | `30` | Minimum sample threshold for candidate promotion | weight experiment interpretation | Smaller samples default to watch-only |
| `US_WEIGHT_EXPERIMENT_BASELINE` | `RULE_V1_BASELINE` | Baseline weight config ID | weight experiment comparison | Used for delta and candidate judgment |
| `US_WEIGHT_EXPERIMENT_LOG_LEVEL` | `INFO` | Experiment/report log level | weight experiment | Console output still writes plain stdout |

### Weight Experiment Notes

- Phase 4-4 keeps the operational baseline unchanged.
- Experiment results are stored in `research` tables only and must not overwrite `recommend.us_stock_rank_daily`.
- `PROMOTE_CANDIDATE` means a forward-test candidate, not an immediate production change.

## Phase 4-5 Forward Test Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_FORWARD_TEST_ENABLED` | `false` | Optional scheduler gate for forward-test tasks | forward test scheduler | Must stay off by default |
| `US_FORWARD_TEST_ID` | `US_RANK_FORWARD_RULE_V1` | Default forward-test tracking ID | forward test | CLI overrides it |
| `US_FORWARD_TEST_HOLDING_DAYS` | `5,20,60` | Default holding-day set | forward test | Used by register/update/report scripts |
| `US_FORWARD_TEST_STRATEGIES` | `TOP5,TOP10,TOP20,BUY_OR_BETTER,STRONG_BUY` | Default strategy aliases | forward test | Same aliases as backtest |
| `US_FORWARD_TEST_OUTPUT_DIR` | `outputs/us_stock_forward_test` | Output directory for markdown/csv reports | forward test reporting | Directory is auto-created |
| `US_FORWARD_TEST_AUTO_REGISTER` | `false` | Optional scheduler registration step | forward test scheduler | Disabled by default |
| `US_FORWARD_TEST_AUTO_UPDATE` | `true` | Optional scheduler entry/exit refresh step | forward test scheduler | Still guarded by `US_FORWARD_TEST_ENABLED` |
| `US_FORWARD_TEST_AUTO_REPORT` | `false` | Optional scheduler report generation step | forward test scheduler | Disabled by default |
| `US_FORWARD_TEST_LOG_LEVEL` | `INFO` | Forward-test log level | forward test | Console output stays plain stdout |

### Forward Test Notes

- Forward Test is not live trading.
- Forward Test is not Paper Trading.
- The register step stores the recommendation snapshot first and fills returns only after time passes.
- Entry uses the next US trading day after `trade_date`.
- Exit uses the session target after `entry_date`.
- Forward Test remains separated from broker APIs and Korean trading logic.

## Phase 5-1 Paper Trading Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_PAPER_TRADING_ENABLED` | `false` | Master switch for future US paper-trading flows | paper trading | Phase 5-1 creates structure only |
| `US_PAPER_ACCOUNT_ID` | `US_PAPER_RULE_V1` | Default US paper account ID | paper trading | Used by account init and later paper flows |
| `US_PAPER_INITIAL_CASH` | `100000` | Initial virtual cash balance | paper trading | USD default |
| `US_PAPER_BASE_CURRENCY` | `USD` | Base account currency | paper trading | Current Phase 5-1 assumes USD only |
| `US_PAPER_MAX_POSITIONS` | `20` | Maximum target positions | paper trading policy | Reserved for Phase 5-2+ |
| `US_PAPER_MAX_POSITION_WEIGHT` | `0.10` | Max weight per position | paper trading risk | Reserved for Phase 5-2+ |
| `US_PAPER_MAX_SECTOR_WEIGHT` | `0.30` | Max sector concentration | paper trading risk | Reserved for Phase 5-2+ |
| `US_PAPER_MIN_CASH_WEIGHT` | `0.05` | Minimum cash reserve weight | paper trading risk | Reserved for Phase 5-2+ |
| `US_PAPER_MAX_DAILY_NEW_BUYS` | `5` | Max daily new entries | paper trading risk | Reserved for Phase 5-2+ |
| `US_PAPER_ALLOW_FRACTIONAL_SHARES` | `true` | Allow fractional virtual shares | paper trading execution | Reserved for Phase 5-2+ |
| `US_PAPER_MIN_ORDER_AMOUNT` | `100` | Minimum virtual order amount | paper trading execution | Used by Phase 5-2 order generation |
| `US_PAPER_COMMISSION_PER_TRADE` | `0` | Flat commission assumption | paper trading execution | Reserved for later fill simulation |
| `US_PAPER_SLIPPAGE_BPS` | `5` | Slippage assumption in bps | paper trading execution | Reserved for later fill simulation |
| `US_PAPER_REAL_ORDER_BLOCKED` | `true` | Hard safety flag blocking real-order paths | paper trading safety | Must remain `true` |
| `US_PAPER_OUTPUT_DIR` | `outputs/us_stock_paper_trading` | Output directory for future paper reports | paper trading | Reserved for later phases |
| `US_PAPER_REPORT_OUTPUT_DIR` | `outputs/us_stock_paper_trading` | Paper report output directory | paper trading reporting | Used by Phase 5-4 report writer |
| `US_PAPER_USE_PREVIOUS_CLOSE_IF_MISSING` | `true` | Allow previous-close fallback for missing snapshot-date prices | paper trading valuation | Used by Phase 5-4 snapshot update |
| `US_PAPER_AUTO_SNAPSHOT_ENABLED` | `false` | Optional scheduler flag for daily paper snapshot refresh | paper trading operations | Default disabled |
| `US_PAPER_AUTO_REPORT_ENABLED` | `false` | Optional scheduler flag for paper report generation | paper trading operations | Default disabled |
| `US_PAPER_REBALANCE_ENABLED` | `false` | Master switch for paper rebalance planning policy | paper trading rebalance | Must stay off by default |
| `US_PAPER_REBALANCE_FREQUENCY` | `DAILY` | Rebalance cadence hint | paper trading rebalance | `DAILY` or `WEEKLY` |
| `US_PAPER_REBALANCE_SELL_FIRST` | `true` | Plan SELL actions before BUY actions | paper trading rebalance | Used by plan/order generation |
| `US_PAPER_REBALANCE_ALLOW_REBUY_SAME_DAY` | `false` | Allow same-day rebuy after a sell candidate | paper trading rebalance | Default blocks churn |
| `US_PAPER_REBALANCE_MIN_AMOUNT` | `100` | Minimum rebalance amount per order idea | paper trading rebalance | Small rebalance diffs are rejected |
| `US_PAPER_REBALANCE_MIN_WEIGHT_DIFF` | `0.02` | Minimum target/current weight gap | paper trading rebalance | Prevents low-signal top-up orders |
| `US_PAPER_REBALANCE_FULL_SELL_ON_RANK_EXIT` | `true` | Fully sell holdings that leave the allowed rank bucket | paper trading rebalance | Applies only to paper positions |
| `US_PAPER_REBALANCE_FULL_SELL_ON_GRADE_DOWNGRADE` | `true` | Fully sell holdings that downgrade into sell grades | paper trading rebalance | Applies only to paper positions |
| `US_PAPER_VALIDATION_ENABLED` | `true` | Enable paper operating validation flow | paper trading validation | Used by Phase 5-5 validation/reporting |
| `US_PAPER_VALIDATION_FAIL_ON_ERROR` | `false` | Optional non-zero exit on validation errors | paper trading validation | Safe default stays non-failing |
| `US_PAPER_SCHEDULER_ENABLED` | `false` | Master switch for optional paper scheduler steps | paper trading scheduler | Must stay off by default |
| `US_PAPER_SCHEDULER_ACCOUNT_ID` | `US_PAPER_RULE_V1` | Default paper account for scheduled steps | paper trading scheduler | Only for paper review automation |
| `US_PAPER_SCHEDULER_RUN_REBALANCE_PLAN` | `true` | Allow scheduled rebalance-plan generation | paper trading scheduler | Review step only |
| `US_PAPER_SCHEDULER_GENERATE_ORDERS` | `false` | Allow scheduled paper-order generation | paper trading scheduler | Explicit opt-in only |
| `US_PAPER_SCHEDULER_SIMULATE_FILLS` | `false` | Allow scheduled paper fill simulation | paper trading scheduler | Explicit opt-in only |
| `US_PAPER_SCHEDULER_UPDATE_SNAPSHOT` | `false` | Allow scheduled paper snapshot refresh | paper trading scheduler | Explicit opt-in only |
| `US_PAPER_SCHEDULER_VALIDATE` | `true` | Allow scheduled validation step | paper trading scheduler | Review-only |
| `US_PAPER_SCHEDULER_REPORT` | `true` | Allow scheduled paper report generation | paper trading scheduler | Review-only |
| `US_PAPER_LOG_LEVEL` | `INFO` | Paper-trading structure log level | paper trading | Used by initialization script |

### Paper Trading Notes

- Paper Trading is not real trading.
- Paper Trading scripts must not call Alpaca, KIS, or any broker order API.
- `US_PAPER_REAL_ORDER_BLOCKED=true` must remain the default.
- Phase 5-4 uses `US_PAPER_USE_PREVIOUS_CLOSE_IF_MISSING` when the snapshot-date close is unavailable.
- Phase 5-5 uses the `US_PAPER_REBALANCE_*` variables to keep paper-only rebalance planning consistent with order generation.
- `US_PAPER_SCHEDULER_*` variables define optional review automation only and must not trigger any real broker flow.

## Phase 6-1 Live-Trading Policy Draft Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_LIVE_TRADING_ENABLED` | `false` | Master live-trading gate | live trading draft | Must stay off by default |
| `US_LIVE_ORDER_ENABLED` | `false` | Real order submission gate | live trading draft | Separate from general live mode |
| `US_LIVE_BUY_ENABLED` | `false` | BUY-side live execution gate | live trading draft | Must stay off by default |
| `US_LIVE_SELL_ENABLED` | `false` | SELL-side live execution gate | live trading draft | Must stay off by default |
| `US_LIVE_REQUIRE_MANUAL_APPROVAL` | `true` | Require operator approval before Micro Live order release | live trading draft | Recommended for Phase 7 entry |
| `US_LIVE_KILL_SWITCH_ENABLED` | `true` | Enable live kill-switch enforcement | live trading draft | Policy default is conservative |
| `US_LIVE_MAX_ORDER_AMOUNT_USD` | `50` | Max single live order size in USD | live trading draft | Micro Live starts very small |
| `US_LIVE_MAX_DAILY_BUY_AMOUNT_USD` | `100` | Max daily live BUY amount in USD | live trading draft | Connectivity validation first |
| `US_LIVE_MAX_DAILY_ORDER_COUNT` | `3` | Max daily live order count | live trading draft | Conservative initial cap |
| `US_LIVE_MAX_DAILY_NEW_BUYS` | `1` | Max new BUY symbols per day | live trading draft | One new BUY candidate only |
| `US_LIVE_MAX_POSITION_WEIGHT` | `0.05` | Max live position weight | live trading draft | Early live concentration cap |
| `US_LIVE_MAX_SECTOR_WEIGHT` | `0.20` | Max live sector weight | live trading draft | Early live concentration cap |
| `US_LIVE_MIN_CASH_WEIGHT` | `0.50` | Min live cash reserve | live trading draft | High cash buffer by design |
| `US_LIVE_MAX_POSITION_COUNT` | `5` | Max live holdings count | live trading draft | Small basket only |
| `US_LIVE_ALLOW_MARKET_ORDER` | `false` | Allow market orders in live mode | live trading draft | Limit-first policy |
| `US_LIVE_DEFAULT_ORDER_TYPE` | `LIMIT` | Default live order type | live trading draft | Market orders discouraged early |
| `US_LIVE_BUY_LIMIT_BUFFER_PCT` | `0.005` | BUY limit buffer vs reference price | live trading draft | Policy draft only |
| `US_LIVE_SELL_LIMIT_BUFFER_PCT` | `0.005` | SELL limit buffer vs reference price | live trading draft | Policy draft only |
| `US_LIVE_BLOCK_LEVERAGED_ETF` | `true` | Block leveraged ETFs in live mode | live trading draft | Default safety block |
| `US_LIVE_BLOCK_INVERSE_ETF` | `true` | Block inverse ETFs in live mode | live trading draft | Default safety block |
| `US_LIVE_BLOCK_BUY_ON_SPY_DROP_PCT` | `-0.02` | Block new BUY on large SPY drop | live trading draft | Market selloff guard |
| `US_LIVE_BLOCK_BUY_ON_QQQ_DROP_PCT` | `-0.025` | Block new BUY on large QQQ drop | live trading draft | Tech selloff guard |
| `US_LIVE_BLOCK_BUY_ON_SYMBOL_GAP_UP_PCT` | `0.05` | Block BUY after excessive gap up | live trading draft | Gap-risk guard |
| `US_LIVE_BLOCK_BUY_ON_SYMBOL_GAP_DOWN_PCT` | `-0.05` | Block BUY after excessive gap down | live trading draft | Falling-knife guard |
| `US_LIVE_MAX_ORDER_RETRY` | `1` | Max live order retry count | live trading draft | No infinite retry loops |
| `US_LIVE_MAX_DAILY_ORDER_FAILURES` | `3` | Max daily live order failures before escalation | live trading draft | Kill-switch candidate threshold |
| `US_LIVE_NOTIFY_ENABLED` | `true` | Enable live-order safety notifications | live trading draft | Policy draft only |
| `US_LIVE_KILL_SWITCH_NOTIFY_ENABLED` | `true` | Enable kill-switch activation/clear notifications | live trading draft | Notification is best-effort only |
| `US_LIVE_APPROVAL_NOTIFY_ENABLED` | `true` | Enable approval-request lifecycle notifications | live trading draft | Best-effort only; DB state is primary |
| `US_LIVE_APPROVAL_EXPIRES_MINUTES` | `30` | Default approval-request expiry window in minutes | live trading draft | Used by Phase 6-5 approval flow |

### Live-Trading Draft Notes

- Phase 6-1 adds policy-draft ENV only.
- These values do not activate live trading by themselves.
- Real-order implementation is not part of this phase.
- All live-trading default gates must stay `false` until later safety layers are implemented.

## Phase 6-2 Live Risk Policy Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_LIVE_REAL_ORDER_BLOCKED` | `true` | Hard safety flag that must stay true before any broker attachment review | live risk policy | Phase 6-2 still does not submit live orders |
| `US_LIVE_ACCOUNT_ID` | `US_LIVE_RULE_V1` | Placeholder live account identifier for risk-state tracking rows | live risk state | Not a broker account lookup key in this phase |
| `US_LIVE_RISK_POLICY_ID` | `US_LIVE_RULE_V1` | Default YAML profile ID for live risk policy loading | live risk policy | Used by validation/init scripts |
| `US_LIVE_RISK_POLICY_FILE` | `config/us_stock_live_risk_policy.yaml` | YAML file path for live risk policy defaults | live risk policy | Relative to repo root by default |
| `US_LIVE_MIN_ORDER_AMOUNT_USD` | `10` | Minimum Micro Live order size | live risk policy | Conservative lower bound |
| `US_LIVE_MAX_DAILY_SELL_AMOUNT_USD` | `500` | Max daily SELL notional tracked by policy | live risk policy | Review-only default in this phase |
| `US_LIVE_MAX_SYMBOL_POSITION_AMOUNT_USD` | `250` | Max notional per single symbol | live risk policy | Complements weight-based cap |
| `US_LIVE_ALLOW_ETF` | `true` | Allow standard ETF candidates if other blocks pass | live risk policy | Leveraged/inverse ETF blocks still apply |
| `US_LIVE_MAX_SYMBOL_VOLATILITY_20D` | `0.05` | Max allowed 20d volatility for new BUY candidates | live risk policy | Future pre-trade guard input |
| `US_LIVE_BLOCK_BEAR_HIGH_VOL_REGIME` | `true` | Block new BUYs in `BEAR_HIGH_VOL` regime | live risk policy | Phase 6-2 stores policy only |
| `US_LIVE_REGULAR_SESSION_ONLY` | `true` | Restrict future live orders to regular session | live risk policy | Time-window enforcement comes later |
| `US_LIVE_BLOCK_FIRST_MINUTES_AFTER_OPEN` | `15` | Block early-session orders for the first N minutes | live risk policy | Conservative Micro Live default |
| `US_LIVE_BLOCK_LAST_MINUTES_BEFORE_CLOSE` | `15` | Block late-session orders for the last N minutes | live risk policy | Conservative Micro Live default |
| `US_LIVE_BLOCK_PREMARKET` | `true` | Block premarket orders | live risk policy | Must stay blocked by default |
| `US_LIVE_BLOCK_AFTERHOURS` | `true` | Block after-hours orders | live risk policy | Must stay blocked by default |

### Phase 6-2 Risk Notes

- Phase 6-2 adds YAML + ENV policy structure only.
- `config/us_stock_live_risk_policy.yaml` remains the reviewed baseline, while ENV is an override layer for controlled testing.
- `risk.us_stock_live_kill_switch`, `risk.us_stock_live_daily_risk_usage`, and `risk.us_stock_live_order_block_log` are status/audit tables only in this phase.
- Phase 6-4 adds `risk.us_stock_live_kill_switch_event_log` and scoped `target_value` support for kill-switch management.
- Phase 6-5 adds `risk.us_stock_live_order_approval` and `risk.us_stock_live_order_approval_event_log`.
- Phase 6-5 uses `US_LIVE_APPROVAL_NOTIFY_ENABLED` and `US_LIVE_APPROVAL_EXPIRES_MINUTES` for approval-request handling only.
- SAFE_DEFAULT means:
  - live trading/order/buy/sell gates remain `false`
  - manual approval remains `true`
  - `US_LIVE_REAL_ORDER_BLOCKED=true`
  - market orders remain disabled
  - Micro Live limits stay very small
- No script in Phase 6-2 may call a broker API or read a real account balance.

## Phase 7-1 To 7-4 Micro Order And Sync Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_MICRO_LIVE_ENABLED` | `false` | Master gate for Micro Live review features | Micro Live | Must stay off by default |
| `US_MICRO_EXECUTION_MODE` | `MOCK` | Default Micro execution mode | Micro Live | `MOCK`, `SANDBOX`, `LIVE` |
| `US_MICRO_ALLOW_SANDBOX` | `false` | Allow sandbox review path | Micro Live | Explicit operator opt-in |
| `US_MICRO_ALLOW_LIVE` | `false` | Allow gated live review path | Micro Live | Still blocked by other live gates |
| `US_MICRO_REAL_ORDER_BLOCKED` | `true` | Hard stop for real-order path | Micro Live safety | Must remain true by default |
| `US_MICRO_REQUIRE_APPROVAL` | `true` | Require approval before Micro order creation/review | Micro Live safety | Conservative default |
| `US_MICRO_REQUIRE_PRECHECK` | `true` | Require pre-trade recheck before Micro review | Micro Live safety | Conservative default |
| `US_MICRO_MOCK_FORCE_REJECT` | `false` | Force mock rejection scenario | mock test | Test-only |
| `US_MICRO_MOCK_FORCE_FAIL` | `false` | Force mock failure scenario | mock test | Test-only |
| `US_MICRO_MOCK_REJECT_SYMBOLS` | blank | Comma-separated mock reject symbols | mock test | Optional |
| `US_MICRO_MOCK_FAIL_SYMBOLS` | blank | Comma-separated mock failure symbols | mock test | Optional |
| `US_SANDBOX_ORDER_ENABLED` | `false` | Allow sandbox send path | sandbox | Explicit opt-in only |
| `US_SANDBOX_BROKER_NAME` | `NONE` | Sandbox broker label | sandbox | Placeholder unless configured |
| `US_SANDBOX_REQUIRE_APPROVAL` | `true` | Require approval before sandbox send | sandbox safety | Conservative default |
| `US_SANDBOX_REQUIRE_PRECHECK` | `true` | Require pre-trade check before sandbox send | sandbox safety | Conservative default |
| `US_SANDBOX_REQUIRE_KILL_SWITCH_CLEAR` | `true` | Require kill-switch clear before sandbox send | sandbox safety | Conservative default |
| `US_SANDBOX_MAX_ORDER_AMOUNT_USD` | `50` | Max sandbox order size | sandbox safety | Very small by design |
| `US_SANDBOX_MAX_DAILY_ORDER_COUNT` | `3` | Max sandbox daily order count | sandbox safety | Conservative default |
| `US_SANDBOX_MAX_DAILY_NEW_BUYS` | `1` | Max sandbox new BUY count | sandbox safety | Conservative default |
| `US_SANDBOX_ALLOW_MARKET_ORDER` | `false` | Allow sandbox market order | sandbox safety | Default block |
| `US_SANDBOX_DEFAULT_ORDER_TYPE` | `LIMIT` | Default sandbox order type | sandbox | Limit-first |
| `US_MICRO_ORDER_SYNC_ENABLED` | `false` | Enable broker status sync | Phase 7-4 sync | Must stay off by default |
| `US_MICRO_ALLOW_LIVE_STATUS_QUERY` | `false` | Allow live status query | Phase 7-4 sync | Explicit opt-in only |
| `US_MICRO_SYNC_INCLUDE_FILLS` | `true` | Include fill lookup during sync | Phase 7-4 sync | Read-only behavior |
| `US_MICRO_SYNC_REAL_ORDER_BLOCKED` | `true` | Safety guard for sync path | Phase 7-4 sync | Must remain true by default |

### Phase 7-1 To 7-4 Notes

- Micro order review remains heavily gated.
- `LIVE_ACCEPTED` is not a fill.
- status sync does not create any new order.
- fill sync does not finalize position truth by itself.

## Phase 7-5 Reconciliation Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_MICRO_RECON_ENABLED` | `false` | Master switch for Micro reconciliation | reconciliation | Must stay off by default |
| `US_MICRO_RECON_EXECUTION_MODE` | `MOCK` | Default reconciliation mode | reconciliation | `MOCK`, `SANDBOX`, `LIVE` |
| `US_MICRO_ALLOW_LIVE_ACCOUNT_QUERY` | `false` | Allow live account/position query | reconciliation safety | Explicit opt-in only |
| `US_MICRO_RECON_REAL_ORDER_BLOCKED` | `true` | Safety guard for reconciliation | reconciliation safety | Must remain true |
| `US_MICRO_RECON_TOLERANCE_QTY` | `0.000001` | Quantity tolerance | reconciliation | Used for fill/position comparison |
| `US_MICRO_RECON_TOLERANCE_AMOUNT_USD` | `1.00` | Amount tolerance in USD | reconciliation | Used for fill comparison |
| `US_MICRO_RECON_TOLERANCE_CASH_USD` | `1.00` | Cash tolerance in USD | reconciliation | Used for cash comparison |
| `US_MICRO_RECON_TRIGGER_KILL_ON_CRITICAL` | `true` | Allow explicit critical-trigger path | reconciliation | Still opt-in at command level |
| `US_MICRO_RECON_REPORT_OUTPUT_DIR` | `output/us_stock_micro_live` | Markdown reconciliation output dir | reconciliation reporting | Auto-created |

### Reconciliation Notes

- reconciliation records mismatch, but does not auto-correct by trading
- live account query stays disabled by default
- critical mismatch becomes a kill-switch recommendation candidate

## Phase 7-6 Operations Report Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_MICRO_OPS_REPORT_NOTIFY_ENABLED` | `false` | Enable operations-report notification | ops reporting | Default off |
| `US_MICRO_OPS_NOTIFY_ON_CRITICAL` | `true` | Notify on critical operations state | ops reporting | Best-effort only |
| `US_MICRO_OPS_NOTIFY_ON_ERROR` | `true` | Notify on error operations state | ops reporting | Best-effort only |
| `US_MICRO_OPS_REPORT_OUTPUT_DIR` | `output/us_stock_micro_live` | Output directory for ops markdown/csv | ops reporting | Auto-created |
| `US_MICRO_OPS_STALE_OPEN_ORDER_MINUTES` | `60` | Minutes before open order becomes stale | ops reporting | Action-required threshold |

### Operations Report Notes

- the operations report is read/report only
- notification failure must not fail report generation
- action-required output is guidance, not an execution trigger

## Phase 8-1 Limited BUY Automation Design Variables

Phase 8-1 is design only. The variables below are proposed for later implementation and must not be interpreted as permission to place real BUY orders.

### Immediately Usable Design ENV

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_AUTOMATION_MODE` | `SHADOW` | BUY automation operating mode | Phase 8 design | `SHADOW`, `PAPER`, `LIVE` |
| `US_BUY_AUTOMATION_ENABLED` | `false` | Master gate for BUY evaluation flow | Phase 8 design | Should stay `false` until a shadow-only runner exists |
| `US_BUY_TOP_N` | `5` | Maximum ranked symbols considered for BUY review | candidate filter | Conservative default |
| `US_BUY_MIN_GRADE` | `BUY` | Minimum recommendation grade for BUY review | candidate filter | Suggested values: `BUY` or `STRONG_BUY` |
| `US_BUY_MIN_TOTAL_SCORE` | `70` | Minimum ranking score for BUY review | candidate filter | Align with current Rule grade threshold |
| `US_BUY_MAX_DAILY_SYMBOLS` | `1` | Max new BUY symbols per day | risk cap | Keep extremely small at first |
| `US_BUY_MAX_DAILY_AMOUNT_USD` | `100` | Max daily BUY notional | risk cap | Separate from existing live caps for design clarity |
| `US_BUY_MAX_PER_SYMBOL_AMOUNT_USD` | `100` | Max BUY notional per symbol | risk cap | Conservative default |
| `US_BUY_MIN_PRICE` | `5` | Minimum price allowed for automated BUY review | symbol filter | Avoid sub-scale low-price names |
| `US_BUY_MAX_PRICE` | `500` | Maximum price allowed for automated BUY review | symbol filter | Keeps sizing simple |
| `US_BUY_MAX_SYMBOL_VOLATILITY_20D` | `0.05` | Max recent volatility for automated BUY review | symbol filter | Reuses current daily-feature concept |
| `US_BUY_REQUIRE_FINANCIAL_DATA` | `true` | Require minimum financial-data presence | data quality | Fail-safe default |
| `US_BUY_REQUIRE_RELATIVE_STRENGTH` | `true` | Require relative-strength evidence | candidate filter | Fail-safe default |
| `US_BUY_COOLDOWN_DAYS` | `10` | Block repeat BUY on the same symbol during cooldown | risk cap | Applies across review history |
| `US_BUY_FAILSAFE_ON_DATA_ERROR` | `true` | Default to `BLOCK` on missing/stale/error state | safety | Must remain `true` |
| `US_BUY_BLOCK_ON_KILL_SWITCH` | `true` | Force BUY block when matching kill switch exists | safety | Must remain `true` |
| `US_BUY_REPORT_OUTPUT_DIR` | `output/us_stock_buy_automation` | Output directory for future BUY automation reports | reporting | Design target only |

### Future-Feature Design ENV

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_MIN_PROBABILITY` | `0.60` | Minimum model probability for BUY review | future ML filter | Not active until model output exists |
| `US_BUY_MAX_GAP_UP_PCT` | `0.05` | Max allowed gap-up before BUY block | future intraday risk | Requires trusted same-day reference |
| `US_BUY_MAX_INTRADAY_CHANGE_PCT` | `0.08` | Max intraday chase threshold | future intraday risk | Requires sub-daily or same-session feed |
| `US_BUY_MAX_VIX_LEVEL` | `25` | Volatility hard-block threshold | future market risk | Only after VIX feed is integrated |
| `US_BUY_EARNINGS_BLACKOUT_DAYS` | `3` | Days around earnings release to block BUY | future event risk | Requires earnings calendar data |
| `US_BUY_MAX_SECTOR_WEIGHT` | `0.20` | Max sector concentration after BUY | future portfolio risk | Needs reliable live/paper exposure mapping |

### LIVE-Only Design ENV

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_LIVE_ENABLED` | `false` | Future explicit BUY live gate | future live mode | Must stay `false` |
| `US_BUY_LIVE_REQUIRE_MANUAL_APPROVAL` | `true` | Keep manual approval mandatory for BUY release | future live mode | Conservative default |
| `US_BUY_LIVE_REQUIRE_RECON_OK` | `true` | Require recent reconciliation without critical mismatch | future live mode | Depends on Phase 7-5 stability |
| `US_BUY_LIVE_REQUIRE_OPS_HEALTH_BELOW` | `ATTENTION` | Maximum allowed ops severity before BUY release | future live mode | Example policy only |
| `US_BUY_LIVE_ACCOUNT_CASH_BUFFER_USD` | `1000` | Minimum cash buffer after BUY release | future live mode | Needs live account-state trust |
| `US_BUY_LIVE_RELEASE_BY_OPERATOR` | blank | Future operator identifier for release approval | future live mode | Audit/control field |

### Phase 8-1 Design Notes

- `US_BUY_AUTOMATION_MODE=LIVE` must not activate any real BUY code in Phase 8-1.
- `US_BUY_AUTOMATION_ENABLED=false` should remain the safe default until a SHADOW-only implementation exists.
- existing live safety flags such as `US_LIVE_ORDER_ENABLED=false` and `US_LIVE_REAL_ORDER_BLOCKED=true` remain the higher-priority gates.
- Phase 8-1 adds design-level config only; it does not weaken any Phase 6 / Phase 7 safety default.

### Phase 8-2 Skeleton Notes

- Phase 8-2 now reads these ENV values in a real SHADOW/PAPER skeleton.
- `US_BUY_MIN_SCORE` is supported as a compatibility alias for `US_BUY_MIN_TOTAL_SCORE`.
- if `US_BUY_MIN_SCORE` is set like `0.70`, the loader interprets it as `70` on the current 0-100 ranking scale.
- `US_BUY_MIN_PROB` is optional; if it is greater than `0` and no probability field exists, the candidate is blocked fail-safe.
- `US_BUY_AUTOMATION_ENABLED=0` still runs the evaluation pipeline, but final candidates are blocked with `AUTOMATION_DISABLED`.
- `US_BUY_AUTOMATION_MODE=LIVE` is accepted syntactically but is blocked with `LIVE_NOT_IMPLEMENTED`.

## Phase 8-3 BUY Report Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_REPORT_ENABLED` | `true` | Enable standalone BUY report generation path | reporting | Report failure must not change BUY decisions |
| `US_BUY_REPORT_OUTPUT_DIR` | `reports/lee_trader_us/buy_automation` | Final JSON/Markdown report output directory | reporting | Separate from raw execution log directory |
| `US_BUY_REPORT_FORMAT` | `json,markdown` | Preferred persisted report formats | reporting | Console output is still available by CLI |
| `US_BUY_LOG_INPUT_DIR` | `output/us_stock_buy_automation` | Raw BUY automation JSON input directory | reporting | Defaults to Phase 8-2 execution log path |
| `US_BUY_BENCHMARK_SYMBOL` | `SPY` | Benchmark symbol for PAPER performance comparison | paper performance | Current implementation uses daily close history only |
| `US_BUY_REPORT_INCLUDE_PAPER_PERFORMANCE` | `true` | Include PAPER performance section in report | reporting | If no paper orders exist, section stays empty |
| `US_BUY_REPORT_FAIL_ON_INVALID_LOG` | `false` | Raise operator-visible error on invalid decision logs | reporting validation | Current default is warn, not hard fail |

### Phase 8-3 Notes

- `US_BUY_REPORT_OUTPUT_DIR` is the final report directory, not the raw Phase 8-2 execution log directory.
- `US_BUY_LOG_INPUT_DIR` lets the report layer read old SHADOW/PAPER runs without rerunning the BUY pipeline.
- benchmark comparison is informational only and must not be treated as a release signal for LIVE.

## Phase 8-4 BUY Scheduler Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_SCHEDULER_ENABLED` | `0` | Master switch for BUY scheduler integration | scheduler | Must stay off by default |
| `US_BUY_SCHEDULER_RUN_AUTOMATION` | `1` | Run BUY automation stage inside scheduler job | scheduler | Can disable automation while leaving report path enabled |
| `US_BUY_SCHEDULER_RUN_REPORT` | `1` | Run BUY report generation inside scheduler job | scheduler | Report-only use is allowed |
| `US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR` | `0` | Re-raise BUY scheduler failure to parent pipeline | scheduler safety | Safe default keeps failures isolated |
| `US_BUY_SCHEDULER_ALLOW_LIVE` | `0` | Reserved explicit scheduler live flag | scheduler safety | Phase 8-4 still blocks LIVE even if set |
| `US_BUY_SCHEDULER_TRADE_DATE` | blank | Optional forced trade date for scheduler job | scheduler | If blank, current pipeline/reference date is used |
| `US_BUY_SCHEDULER_MAX_RUNTIME_SECONDS` | `300` | Soft runtime threshold for scheduler job | scheduler | Exceeding it records `SCHEDULER_TIMEOUT` |
| `US_BUY_SCHEDULER_LOG_LEVEL` | `INFO` | Scheduler-job log level | scheduler | Console output remains concise |

### Phase 8-4 Notes

- `US_BUY_SCHEDULER_ENABLED=0` means the parent pipeline must skip the BUY scheduler stage entirely.
- `US_BUY_SCHEDULER_FAIL_PIPELINE_ON_ERROR=0` means BUY automation remains an isolated review stage and must not break the broader US data path.
- `US_BUY_AUTOMATION_MODE=LIVE` is blocked in the scheduler with `LIVE_DISABLED_IN_SCHEDULER`.
- Phase 8-4 still does not permit broker calls, account lookup, or real BUY order execution.

## Phase 8-5 Readiness Variables

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_BUY_READINESS_ENABLED` | `1` | Master switch for readiness evaluation | readiness | Evaluation only, never activates LIVE |
| `US_BUY_READINESS_REPORT_OUTPUT_DIR` | `reports/lee_trader_us/buy_automation/readiness` | Output directory for readiness JSON/Markdown reports | readiness reporting | Auto-created |
| `US_BUY_READINESS_BENCHMARK_SYMBOL` | `SPY` | Primary benchmark for readiness comparison | readiness performance | `SPY` default |
| `US_BUY_READINESS_COMPARE_QQQ` | `1` | Whether to include secondary `QQQ` comparison | readiness performance | Informational comparison |
| `US_BUY_READINESS_MIN_EXCESS_RETURN_PCT` | `0` | Minimum excess-return threshold vs primary benchmark | readiness policy | Safe default is non-negative |
| `US_BUY_MIN_SHADOW_DAYS` | `20` | Minimum SHADOW operating days before review | readiness policy | Conservative baseline |
| `US_BUY_MIN_PAPER_DAYS` | `60` | Minimum PAPER operating days before review | readiness policy | Conservative baseline |
| `US_BUY_MIN_PAPER_ORDERS` | `20` | Minimum PAPER order count before review | readiness policy | Conservative baseline |
| `US_BUY_MIN_WIN_RATE_PCT` | `50` | Minimum win-rate threshold | readiness policy | Converted to `0.50` internally |
| `US_BUY_MAX_DRAWDOWN_PCT` | `15` | Maximum allowed drawdown | readiness policy | Converted to `0.15` internally |
| `US_BUY_MAX_DATA_MISSING_RATE_PCT` | `5` | Maximum allowed data-missing rate | readiness policy | Converted to decimal internally |
| `US_BUY_MIN_SCHEDULER_SUCCESS_RATE_PCT` | `95` | Minimum scheduler success-rate threshold | readiness policy | Converted to `0.95` internally |
| `US_BUY_REQUIRE_POSITIVE_EXCESS_RETURN` | `1` | Require positive excess return before review-ready state | readiness policy | Conservative default |
| `US_BUY_REQUIRE_MANUAL_APPROVAL` | `1` | Keep manual approval mandatory even when review-ready | readiness policy | Must remain true |
| `US_BUY_SCHEDULER_RUN_READINESS` | `0` | Optional future scheduler hook for readiness report | scheduler readiness | Default off |

### Phase 8-5 Notes

- `live_ready=true` means review-eligible only.
- readiness evaluation must not change any LIVE ENV value automatically.
- benchmark data missing means fail-safe `NOT_READY`.
- manual approval remains mandatory regardless of score.
