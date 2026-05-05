# Lee_trader_rule Context

## 상세 설명
- 룰 모듈은 `RULE_TREND_LIQUIDITY_V1` 전략을 중심으로 동작한다.
- after-close 단계에서 신호와 포트폴리오 계획을 만들고, before-open 단계에서 paper/live 실행을 수행하며, after-open 단계에서 live fill sync를 수행한다.
- 웹 노출은 `build_rule_web_payloads.py`와 `sync_web_display_data.py`, `node/index.js`의 `/api/rule/*` 경로를 통해 연결된다.

## 전략/로직 개요
- 신호 생성
  - `rule_signal_builder.py`는 `features`, `prices`, `universe`, `market_status`를 결합한다.
  - 거래대금 20일 평균, gap risk, 시장 방어 모드, RSI, 추세/유동성/안정성 점수를 이용해 `rule_score`, `rule_score_v2`, `entry_signal`, `strong_entry_signal`를 계산한다.
- 포트폴리오 계획
  - `rule_portfolio_manager.py`는 최대 보유 수, 종목 비중, 섹터 비중, 현금 비중, 쿨다운, 보유기간, stop loss, trailing stop 규칙을 반영해 `buy/hold/reduce/exit/skip` 액션을 만든다.
- 주문 프리뷰
  - `rule_order_preview_builder.py`는 `rule_portfolio_plan.json`을 읽어 `order_qty`, `order_amount`, `limit_price`, `order_allowed`, `order_block_reason`을 계산한다.
- 실행
  - paper 모드는 `rule_execution_simulator.py`
  - pilot/live 모드는 `rule_order_submitter.py`
  - live 모드는 KIS 시가 snapshot과 `evaluate_rule_order_guard()`를 추가로 반영한다.

## 운영상 주의사항
- `RULE_TRADING_RUN_MODE`에 따라 before-open 실행 대상이 paper simulator 또는 live submitter로 갈린다.
- `rule_order_submitter.py`는 preview의 `run_mode`가 `pilot` 또는 `live`가 아니면 중단한다.
- before-open 세션 시간은 `RULE_BEFORE_OPEN_START_TIME`, `RULE_BEFORE_OPEN_END_TIME` 환경 변수에 의해 제한된다.
- `config/trading_calendar_kr.json`이 잘못되면 실행일 validation이 실패할 수 있다.
- live snapshot / order submit은 KIS 인증 실패 시 `auth_failed`, `market_data_unavailable` 상태로 빠질 수 있다.

## 다른 모듈과의 관계
- `Lee_trader_ai`
  - 입력으로 `data/features.csv`, `data/market_status.csv`를 공유한다.
  - 웹 payload 동기화 경로는 동일하다.
- `Lee_trader_backTest`
  - `rule_backtest.py`는 룰 모듈 내부 백테스트이며, walk-forward 백테스트와는 별도 저장 흐름이다.
- `node/index.js`
  - `/api/rule/summary`
  - `/api/rule/signals/latest`
  - `/api/rule/portfolio-plan`
  - `/api/rule/order-preview`
  - `/api/rule/paper-state`
  - `/api/rule/backtest-summary`
  - `/api/rule/execution-results`
  - `/api/rule/execution-history`

## 확인 필요
- RULE 전략의 원래 운영계좌 식별자와 외부 운영 절차는 코드 밖 문서 의존성이 있을 수 있다.
