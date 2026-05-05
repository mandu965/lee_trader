# Lee_trader_rule Flow

## 실행 흐름
1. 장마감 후
   - `run_rule_after_close_cycle.py`
   - `rule_signal_builder.py`
   - `rule_backtest.py`
   - `rule_portfolio_manager.py`
   - `rule_order_preview_builder.py`
   - `rule_daily_report.py`
2. 장시작 전
   - `run_rule_before_open_cycle.py`
   - paper: `rule_execution_simulator.py`
   - live/pilot: `rule_order_submitter.py`
3. 장시작 후
   - `run_rule_after_open_cycle.py`
   - live/pilot: `rule_order_fill_sync.py`
4. 웹 노출
   - `build_rule_web_payloads.py`
   - `sync_web_display_data.py`
   - `node/index.js` `/api/rule/*`

## 주요 함수 호출 순서
- 신호
  - `load_features()`
  - `load_prices()`
  - `load_universe()`
  - `load_market()`
  - `attach_price_features()`
  - `attach_market()`
  - `build_rule_scores()`
- 포트폴리오
  - `build_rule_portfolio_plan()`
  - 내부 보조
    - `position_frame()`
    - `recent_trade_codes()`
    - `evaluate_position_risk()`
- 주문 프리뷰
  - `build_rule_order_preview()`
  - `evaluate_rule_order_guard()`
- 실행
  - paper: `validate_trading_session() -> validate_previous_execution_completed() -> evaluate_preview_item() -> apply_filled_orders_to_state()`
  - live: `_build_market_snapshot() -> _submit_items() -> order_cash()`

## 데이터 흐름
- `features.csv` + `prices_daily_adjusted.csv` + `universe.csv` + `market_status.csv`
  -> `data/rule_signals.csv`
- `rule_signals.csv` + `prices_daily_adjusted.csv`
  -> `outputs/rule_strategy_backtest_report.json`
- `rule_signals.csv` + account state JSON
  -> `outputs/rule_portfolio_plan.json`
- `rule_portfolio_plan.json`
  -> `outputs/rule_order_preview.json`
- preview + calendar + signals + market snapshot
  -> `outputs/rule_execution_results.json`
- execution / state / backtest / signals
  -> `outputs/rule_dashboard_summary.json`, `outputs/rule_signals_latest.json`

## 외부 의존성
- Python
  - `pandas`
  - `numpy`
- 서비스
  - KIS API
  - Postgres via `sync_web_display_data.py`
- 설정
  - `config/trading_calendar_kr.json`
  - RULE 관련 환경 변수

## 확인 필요
- `run_rule_after_open_cycle.py`가 호출하는 `rule_order_fill_sync.py`의 내부 동기화 범위는 이번 문서 작성 범위에서 상세 확인이 부족하다.
