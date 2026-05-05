# Lee_trader_rule Flow

## Execution Flow

### 1. After-Close

실행 파일:

- `python/run_rule_after_close_cycle.py`

내부 순서:

1. `rule_signal_builder.py`
2. `rule_backtest.py`
3. `rule_portfolio_manager.py`
4. `rule_order_preview_builder.py`
5. `rule_daily_report.py`

주요 산출물:

- `data/rule_signals.csv`
- `outputs/rule_strategy_backtest_report.json`
- `outputs/rule_portfolio_plan.json`
- `outputs/rule_order_preview.json`

### 2. Before-Open

실행 파일:

- `python/run_rule_before_open_cycle.py`

분기:

- `paper` -> `python/rule_execution_simulator.py`
- `pilot`, `live` -> `python/rule_order_submitter.py`

주요 검증:

- 거래일 검증
- 장전 허용 시간 검증
- 이전 실행 reconciliation 검증
- RULE order guard 검증

주요 산출물:

- `outputs/rule_execution_results.json`
- `outputs/rule_execution_reconciliation_report.md`

### 3. After-Open

실행 파일:

- `python/run_rule_after_open_cycle.py`

주요 역할:

- `python/rule_order_fill_sync.py`
- live/pilot 체결 상태 동기화
- 필요 시 미체결 후속 처리

### 4. Web Payload

실행 파일:

- `python/build_rule_web_payloads.py`
- `python/sync_web_display_data.py`

주요 역할:

- RULE summary/signals payload 생성
- web DB / display payload 반영
- `node/index.js`의 `/api/rule/*` 응답 소스 갱신

## Data Flow

### Input

- `data/features.csv`
- `data/prices_daily_adjusted.csv`
- `data/universe.csv`
- `data/market_status.csv`
- `outputs/rule_account_paper_state.json`
- `outputs/rule_account_live_state.json`
- `config/trading_calendar_kr.json`

### Transform

1. feature/price/universe/market 입력 병합
2. RULE 점수 및 signal 생성
3. backtest 요약 생성
4. 포트폴리오 액션 계산
5. 주문 preview 생성
6. paper 시뮬레이션 또는 pilot/live 제출
7. 화면용 summary/payload 생성

### Output

- `data/rule_signals.csv`
- `outputs/rule_strategy_backtest_report.json`
- `outputs/rule_portfolio_plan.json`
- `outputs/rule_order_preview.json`
- `outputs/rule_execution_results.json`
- `outputs/rule_dashboard_summary.json`
- `outputs/rule_signals_latest.json`

## Main Guards

- `RULE_TRADING_RUN_MODE`
- `RULE_LIVE_ENABLED`
- `RULE_ORDER_SUBMIT_ENABLED`
- `RULE_KILL_SWITCH`
- `GLOBAL_KILL_SWITCH`
- `RULE_BEFORE_OPEN_START_TIME`
- `RULE_BEFORE_OPEN_END_TIME`
- `trading_calendar_kr.json`

## Notes

- RULE은 `paper`, `pilot`, `live`를 같은 흐름에서 모드별로 분기합니다.
- live 계좌 동기화 파일이 최신이어도, preview/execution payload가 stale이면 화면이 예전 상태를 보여줄 수 있습니다.
- 장전 실행은 `after-close -> before-open -> sync_web_display_data` 순서를 맞춰야 합니다.
