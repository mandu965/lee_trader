# AI ENV

## Purpose

This document summarizes environment variables used by the AI pipeline and live auto-trading flow.

## Core Data / DB

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `DATABASE_URL` | none | Primary research database connection | pipeline, training, sync |
| `WEB_DATABASE_URL` | none | Web payload sync database connection | `sync_web_display_data.py` |
| `USE_SQLITE_MIRROR` | `0` | Enable SQLite mirror reads | pipeline |
| `USE_SQLITE_FALLBACK_WRITES` | `0` | Enable SQLite fallback writes | pipeline |

## Model / Ranking

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `MODEL_VERSION` | project default | Active model version | train / predict |
| `HORIZON_DAYS` | project default | Prediction horizon | train / predict |
| `TOP_N` | project default | Top-N extraction size | ranking / recommendation |
| `SCORE_FORMULA_VERSION` | blank | Optional score formula override flag | ranking |

## Financial Momentum (Phase 1~8)

| Variable | Default | Description | Scope | 활성화 시점 |
| --- | --- | --- | --- | --- |
| `FINANCIAL_FEATURE_ENABLED` | `0` | 재무 모멘텀 shadow 계산 활성화 | `ranking_builder.py` | Phase 4 완료 후 |
| `FINANCIAL_SCORE_OVERLAY_ENABLED` | `0` | shadow → live final_score 실반영 (Phase 7) | `ranking_builder.py` | Phase 6 백테스트 통과 후 |
| `FINANCIAL_BUY_GATE_ENABLED` | `0` | 수량 축소 / BUY 차단 활성화 (Phase 8) | `apply_execution_policy.py` | Phase 7 검증 후 |
| `FINANCIAL_OP_MARGIN_DROP_THRESHOLD` | `-2.0` | 영업이익률 QoQ 하락 임계값 (pp) | `build_financial_momentum_features.py` | |
| `FINANCIAL_SLOWDOWN_CONSECUTIVE_QUARTERS` | `2` | SLOWING 판정 연속 분기 수 | `build_financial_momentum_features.py` | |

## C-1 공매도 잔고 (fetch_short_interest.py)

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `SHORT_INTEREST_YEARS_BACK` | `2` | 백필 기간 (연) | `fetch_short_interest.py` |
| `SHORT_INTEREST_SLEEP_SEC` | `1.5` | 날짜 간 슬립 (초) | `fetch_short_interest.py` |

## Live Auto Trading

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `AUTO_TRADE_EXECUTE` | `0` | Enable real order submission | `submit_live_orders.py` | Keep disabled by default |
| `AUTO_TRADE_ALLOW_BUY` | `0` | Allow BUY order submission | live auto trade | SELL-only if disabled |
| `AUTO_TRADE_BUY_APPROVAL_REQUIRED` | `0` | Require manual BUY approval | live auto trade | Operational safety gate |
| `AUTO_TRADE_FORCE_RESUBMIT` | `0` | Ignore previous successful request ids | live auto trade | Use carefully |

## KR AI 포지션 리스크 (ai_position_risk.py)

| Variable | Default | 서버 권장 | Description | Scope |
| --- | --- | --- | --- | --- |
| `AI_MAX_HOLDING_DAYS` | `30` | `30` | 최대 보유일 — 초과 시 pnl < 0% 이면 청산, 0% 이상은 hard_cap까지 HOLD | `ai_position_risk.py` |
| `AI_MAX_HOLDING_DAYS_HARD_CAP` | `45` | `45` | 하드캡 — pnl ≥ 0%여도 초과 시 강제 청산 | `ai_position_risk.py` |
| `AI_TRAILING_STOP_PCT` | `0.05` | `0.10` | 고점 대비 하락률 임계값 (trailing stop 발동) | `ai_position_risk.py` |
| `AI_TRAILING_STOP_MIN_PROFIT` | `0.03` | `0.08` | trailing stop 발동 최소 수익률 — 미달 시 trailing 미발동 | `ai_position_risk.py` |

## 진입 갭 필터 (submit_live_orders.py)

갭 필터는 코드 기본값으로 이미 작동 중. `.env` 미설정 시에도 아래 기본값이 적용됨.

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `ENTRY_GAP_BLOCK_UP_PCT` | `0.03` | 갭 상승 +3% 이상 → 소프트 차단 (당일 체결 불가) | `submit_live_orders.py` |
| `ENTRY_GAP_HARD_BLOCK_UP_PCT` | `0.05` | 갭 상승 +5% 이상 → 하드 차단 | `submit_live_orders.py` |
| `ENTRY_GAP_BLOCK_DOWN_PCT` | `-0.04` | 갭 하락 -4% 이하 → 차단 | `submit_live_orders.py` |
| `ENTRY_GAP_BLOCK_ON_LIVE_PRICE_MISSING` | `1` | 실시간 가격 미수신 시 차단 여부 | `submit_live_orders.py` |

## Common Live Risk Guard (매수 안전 게이트)

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `GLOBAL_MAX_DAILY_BUY_RATIO` | `0` (비활성) | 일일 매수 한도: 총자산 대비 비율. 설정 시 절대 금액보다 우선. 예: `0.30` = 총자산 30% | `common_live_risk_guard.py` |
| `GLOBAL_MAX_WEEKLY_BUY_RATIO` | `0` (비활성) | 주간 매수 한도: 총자산 대비 비율. 예: `0.60` = 총자산 60% | `common_live_risk_guard.py` |
| `GLOBAL_MAX_DAILY_BUY_AMOUNT` | `500000` | 일일 매수 한도 절대 금액 (RATIO 미설정 시 fallback) | `common_live_risk_guard.py` |
| `GLOBAL_MAX_WEEKLY_BUY_AMOUNT` | `1500000` | 주간 매수 한도 절대 금액 (RATIO 미설정 시 fallback) | `common_live_risk_guard.py` |
| `GLOBAL_BLOCK_BUY_ON_DAILY_LOSS_UNAVAILABLE` | `1` | 일일 손실률 조회 실패 시 매수 차단 여부. `0`으로 완화 가능 | `common_live_risk_guard.py` |
| `GLOBAL_BLOCK_BUY_ON_WEEKLY_LOSS_UNAVAILABLE` | `1` | 주간 손실률 조회 실패 시 매수 차단 여부 | `common_live_risk_guard.py` |
| `GLOBAL_MAX_DAILY_LOSS_PCT` | `0.01` | 일일 손실 한도 (1% 초과 시 매수 차단) | `common_live_risk_guard.py` |
| `GLOBAL_MAX_WEEKLY_LOSS_PCT` | `0.03` | 주간 손실 한도 (3% 초과 시 매수 차단) | `common_live_risk_guard.py` |
| `GLOBAL_KILL_SWITCH` | `0` | 전체 매수 긴급 차단 | `common_live_risk_guard.py` |
| `GLOBAL_SYNC_MAX_AGE_MINUTES` | `30` | holdings/fills 동기화 최대 허용 경과 시간(분) | `common_live_risk_guard.py` |
| `MARKET_GUARD_FILE_PATH` | `data/market_guard_kill.json` | Market Guard 킬스위치 파일 경로 (ROOT 기준 상대 경로 허용) | `common_live_risk_guard.py` |

> **비율 기반 한도 사용 예시** (`.env`):
> ```env
> GLOBAL_MAX_DAILY_BUY_RATIO=0.30    # 총자산의 30%/일
> GLOBAL_MAX_WEEKLY_BUY_RATIO=0.60   # 총자산의 60%/주
> ```
> 비율 설정 시 입출금으로 총자산이 변경돼도 한도가 자동 추적됩니다.  
> 총자산 조회 실패 시 `GLOBAL_MAX_DAILY_BUY_AMOUNT` 절대 금액으로 자동 fallback합니다.

## Alerts (과제 4-B)

설정된 채널에만 발송. 미설정 시 콘솔로 fallback. 파일 로그는 항상 기록.
웹 확인: `/alerts.html` (`GET /api/alerts`)

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `SLACK_WEBHOOK_URL` | blank | Slack Incoming Webhook URL | KPI / live auto-trade alerts |
| `TELEGRAM_BOT_TOKEN` | blank | Telegram Bot 토큰 | KPI / live auto-trade alerts |
| `TELEGRAM_CHAT_ID` | blank | Telegram 채팅 ID | KPI / live auto-trade alerts |
| `ALERT_EMAIL_SMTP_HOST` | blank | SMTP 서버 호스트 | KPI / live auto-trade alerts |
| `ALERT_EMAIL_SMTP_PORT` | `587` | SMTP 포트 | alerts |
| `ALERT_EMAIL_SMTP_USER` | blank | SMTP 사용자 (미설정 시 FROM 주소 사용) | alerts |
| `ALERT_EMAIL_SMTP_PASSWORD` | blank | SMTP 비밀번호 | alerts |
| `ALERT_EMAIL_FROM` | blank | 발신 이메일 주소 | alerts |
| `ALERT_EMAIL_TO` | blank | 수신 이메일 주소 | alerts |
| `ALERT_MIN_SCORE_THRESHOLD` | `40` | 상위 20개 평균 final_score 경보 기준치 | `score_kpi_monitor.py` |
| `ALERT_LOG_PATH` | `outputs/alert_log.json` | 알림 로그 파일 경로 | notifier |
| `ALERT_LOG_MAX_ENTRIES` | `200` | 로그 파일 최대 보관 건수 | notifier |

## KIS Auth

| Variable | Default | Description | Scope |
| --- | --- | --- | --- |
| `KIS_BASE_URL` | none | KIS base URL | AI / RULE shared |
| `KIS_APP_KEY` | none | KIS app key | AI path |
| `KIS_APP_SECRET` | none | KIS app secret | AI path |
| `KIS_CANO` | none | Account number | live sync / live orders |
| `KIS_ACNT_PRDT_CD` | none | Account product code | live sync / live orders |

## KIS Retry

| Variable | Default | Description | Note |
| --- | --- | --- | --- |
| `KIS_MAX_RETRY` | `3` | Maximum retry count | Preferred variable |
| `KIS_RETRY_WAIT_SEC` | `1` | Initial retry wait seconds | Preferred variable |
| `KIS_RETRY_BACKOFF_FACTOR` | `2` | Retry backoff multiplier | Example: `1s -> 2s -> 4s` |
| `KIS_RETRY_BACKOFF_MAX_SEC` | `30` | Maximum retry wait seconds | Backoff cap |
| `KIS_TIMEOUT_SEC` | `20` | Per-request timeout seconds | Minimum practical value is 5 |

## Retry Notes

- `429`: wait using `Retry-After` when present, then retry.
- `5xx`: retry with backoff.
- Other `4xx`: fail immediately.
- `order_cash` and `order_rvsecncl` should remain `no_retry=True` to avoid duplicate orders.
- If retries are exhausted, the system should log at critical level and attempt notifier delivery without stopping the main flow.

## Project C Phase 2-2: US Financial Collector

The following variables are used by the standalone US stock financial collector.
The collector is implemented, but it is not attached to production schedulers.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_FINANCIAL_COLLECT_ENABLED` | `0` | Master switch for the standalone US financial collector | Project C US financial | Must stay disabled by default |
| `US_FINANCIAL_SOURCE` | `yfinance` | Financial data source | standalone collector | Only `yfinance` is supported now |
| `US_FINANCIAL_PERIOD_TYPES` | `annual,quarterly` | Period types to request | standalone collector | Only `annual` and `quarterly` are supported now |
| `US_FINANCIAL_LOOKBACK_YEARS` | `5` | History depth filter for fiscal periods | standalone collector | Older periods are skipped |
| `US_FINANCIAL_MAX_TICKERS_PER_RUN` | `100` | Per-run ticker cap | standalone collector | Prevents accidental large runs |
| `US_FINANCIAL_SLEEP_SEC` | `1.0` | Sleep between ticker requests | standalone collector | Conservative default |
| `US_FINANCIAL_RETRY_COUNT` | `3` | Retry count on temporary ticker failures | standalone collector | Applied per ticker |
| `US_FINANCIAL_RETRY_SLEEP_SEC` | `5` | Retry wait seconds | standalone collector | Applied between retries |
| `US_FINANCIAL_FAIL_FAST` | `0` | Stop on first ticker failure when enabled | standalone collector | Default keeps ticker failures isolated |
| `US_FINANCIAL_WRITE_MODE` | `upsert` | DB write mode | standalone collector | Only `upsert` is supported now |
| `US_FINANCIAL_LOG_LEVEL` | `INFO` | Collector log level | standalone collector | Safe default |

### Operating Notes

- `US_FINANCIAL_COLLECT_ENABLED=0` keeps the collector detached from current production schedulers.
- The collector is standalone only and is not wired into Korean schedulers.
- `US_FINANCIAL_*` settings must remain independent from Korean AI, RULE, and live-order flows.

## Project C Phase 2-3: US Financial Feature Builder

The following variables are used by the standalone US financial feature builder.
This builder reads raw financial tables and writes a separate financial feature table.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_FINANCIAL_FEATURE_BUILD_ENABLED` | `0` | Master switch for the standalone US financial feature builder | Project C US financial | Must stay disabled by default |
| `US_FINANCIAL_FEATURE_SOURCE_STATEMENT_TABLE` | `raw.us_stock_financial_statement` | Raw statement source table | standalone feature builder | Current implementation supports this table only |
| `US_FINANCIAL_FEATURE_SOURCE_METRIC_TABLE` | `raw.us_stock_financial_metric` | Raw metric source table | standalone feature builder | Current implementation supports this table only |
| `US_FINANCIAL_FEATURE_TARGET_TABLE` | `feature.us_stock_financial_feature` | Target financial feature table | standalone feature builder | Separate from daily price features |
| `US_FINANCIAL_FEATURE_PERIOD_TYPES` | `annual,quarterly` | Period types to build | standalone feature builder | Only `annual` and `quarterly` are supported now |
| `US_FINANCIAL_FEATURE_LOOKBACK_YEARS` | `5` | Source fiscal history depth | standalone feature builder | Older periods are ignored |
| `US_FINANCIAL_FEATURE_WRITE_MODE` | `upsert` | DB write mode | standalone feature builder | Only `upsert` is supported now |
| `US_FINANCIAL_FEATURE_LOG_LEVEL` | `INFO` | Builder log level | standalone feature builder | Safe default |

### Operating Notes

- `US_FINANCIAL_FEATURE_BUILD_ENABLED=0` keeps the builder detached from current production schedulers and domestic trading flows.
- The builder is standalone only and must not be attached to `run_pipeline.py`, `run_live_auto_trade_cycle.py`, or `run_daily_scheduler.py`.
- Financial features are stored in a separate table because `fiscal_date` / `period_type` do not align with `feature.us_stock_feature_daily`'s daily `feature_date` axis.
- Null-heavy raw financial inputs are expected. Missing fields should not be treated as batch failure.

## Project C Phase 2-4: US Relative Strength Builder

The following variables are used by the standalone US relative strength builder.

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_RELATIVE_STRENGTH_BUILD_ENABLED` | `0` | Master switch for standalone relative strength generation | Project C US price features | Must stay disabled by default |
| `US_RELATIVE_STRENGTH_SOURCE_TABLE` | `market.us_stock_daily_price` | Source price table | standalone relative strength builder | Current implementation supports this table only |
| `US_RELATIVE_STRENGTH_TARGET_TABLE` | `feature.us_stock_relative_strength_daily` | Target relative strength table | standalone relative strength builder | Separate from existing Phase 1 daily features |
| `US_RELATIVE_STRENGTH_BENCHMARKS` | `SPY,QQQ` | Fixed benchmark list | standalone relative strength builder | Only `SPY,QQQ` is supported in this phase |
| `US_RELATIVE_STRENGTH_WINDOWS` | `5,20,60,120,252` | Trading-day return windows | standalone relative strength builder | Used for return and relative strength features |
| `US_RELATIVE_STRENGTH_PRICE_COLUMN` | `auto` | Price column selection mode | standalone relative strength builder | `adj_close_price` first, `close_price` fallback |
| `US_RELATIVE_STRENGTH_WRITE_MODE` | `upsert` | DB write mode | standalone relative strength builder | Only `upsert` is supported now |
| `US_RELATIVE_STRENGTH_LOG_LEVEL` | `INFO` | Builder log level | standalone relative strength builder | Safe default |

### Relative Strength Notes

- `US_RELATIVE_STRENGTH_BUILD_ENABLED=0` keeps the builder detached from domestic production schedulers.
- `SPY` and `QQQ` price rows must already exist in `market.us_stock_daily_price`.
- If either benchmark is missing, the related benchmark return and relative strength fields are left null with warning logs.

## Project C Phase 2-5: US Label Builder

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_LABEL_BUILD_ENABLED` | `0` | Master switch for standalone label generation | Project C US labels | Must stay disabled by default |
| `US_LABEL_SOURCE_PRICE_TABLE` | `market.us_stock_daily_price` | Source price table | standalone label builder | Current implementation supports this table only |
| `US_LABEL_TARGET_TABLE` | `label.us_stock_label_daily` | Target label table | standalone label builder | Separate label layer |
| `US_LABEL_PRICE_COLUMN` | `auto` | Price column selection mode | standalone label builder | `adj_close_price` first, `close_price` fallback |
| `US_LABEL_WINDOWS` | `5,20,60` | Forward-return windows | standalone label builder | Trading-day based |
| `US_LABEL_TOP_PERCENTILE` | `0.20` | Top-percentile cutoff | standalone label builder | Used for `label_top20_*` |
| `US_LABEL_MIN_UNIVERSE_SIZE` | `30` | Minimum same-date universe size | standalone label builder | Smaller dates keep top20 labels null |
| `US_LABEL_EXCLUDE_BENCHMARKS` | `SPY,QQQ` | Excluded benchmark tickers | standalone label builder | Excluded from top20 label universe |
| `US_LABEL_WRITE_MODE` | `upsert` | DB write mode | standalone label builder | Only `upsert` is supported now |
| `US_LABEL_LOG_LEVEL` | `INFO` | Builder log level | standalone label builder | Safe default |

## Project C Phase 2-5: US Dataset Validator

| Variable | Default | Description | Scope | Note |
| --- | --- | --- | --- | --- |
| `US_DATASET_VALIDATE_ENABLED` | `0` | Master switch for standalone dataset validation | Project C US dataset | Must stay disabled by default |
| `US_DATASET_FEATURE_TABLE` | `feature.us_stock_feature_daily` | Base daily feature table | standalone dataset validator | Current implementation supports this table only |
| `US_DATASET_FINANCIAL_FEATURE_TABLE` | `feature.us_stock_financial_feature` | Financial feature table | standalone dataset validator | Used for row-count and leakage notes |
| `US_DATASET_RELATIVE_STRENGTH_TABLE` | `feature.us_stock_relative_strength_daily` | Relative strength feature table | standalone dataset validator | Joined on `ticker + trade_date` |
| `US_DATASET_LABEL_TABLE` | `label.us_stock_label_daily` | Label table | standalone dataset validator | Joined on `ticker + trade_date` |
| `US_DATASET_REPORT_PATH` | `reports/us_stock_dataset_validation.md` | Markdown report output path | standalone dataset validator | Relative paths resolve from repo root |
| `US_DATASET_LOG_LEVEL` | `INFO` | Validator log level | standalone dataset validator | Safe default |

### Label / Dataset Notes

- Both label building and dataset validation are standalone only.
- `US_LABEL_BUILD_ENABLED=0` and `US_DATASET_VALIDATE_ENABLED=0` keep them detached from domestic schedulers.
- Financial feature as-of join is intentionally not auto-applied in this phase because `reported_date`-aware leakage control is not fully implemented.
