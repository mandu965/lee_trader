# Lee_trader_rule

## 모듈 목적
- 실제 저장소에는 `Lee_trader_rule` 폴더가 없으며, 이 문서는 `python/rule_*` 계열 파일과 해당 웹/API 연결부를 기준으로 정리한다.
- 목적은 규칙 기반 자동매매 전략 `RULE_TREND_LIQUIDITY_V1`의 신호 생성, 포트폴리오 계획, 주문 프리뷰, 모의체결/실체결, 계좌 상태 동기화, 대시보드 산출을 수행하는 것이다.

## 핵심 기능
- `python/rule_signal_builder.py`: 룰 신호 생성
- `python/rule_backtest.py`: 룰 신호의 D+1 open 기반 백테스트
- `python/rule_portfolio_manager.py`: 포지션/리스크 기반 포트폴리오 액션 산출
- `python/rule_order_preview_builder.py`: 주문 프리뷰 JSON 생성
- `python/rule_execution_simulator.py`: paper 모드 모의 실행
- `python/rule_order_submitter.py`: pilot/live 실주문 제출
- `python/rule_live_account_snapshot.py`: live 계좌 상태 동기화
- `python/build_rule_web_payloads.py`: 룰 대시보드용 JSON 생성

## 입력 데이터
- CSV
  - `data/features.csv`
  - `data/prices_daily_adjusted.csv`
  - `data/universe.csv`
  - `data/market_status.csv`
  - `data/rule_signals.csv`
  - `data/rule_account_live_holdings.csv`
- JSON
  - `outputs/rule_account_paper_state.json`
  - `outputs/rule_account_live_state.json`
  - `config/trading_calendar_kr.json`
- 환경 변수
  - `RULE_TRADING_RUN_MODE`
  - `RULE_MIN_TRADING_VALUE_MA20_*`
  - `RULE_ENTRY_RULE_SCORE_MIN`
  - `RULE_STRONG_RULE_SCORE_MIN`
  - `RULE_MAX_POSITIONS`
  - `RULE_MAX_HOLDING_DAYS`
  - KIS RULE 계좌 환경 변수 (`KIS_RULE_CANO`, `KIS_RULE_ACNT_PRDT_CD` 등)

## 출력 데이터
- `data/rule_signals.csv`
- `outputs/rule_strategy_backtest_report.json`
- `outputs/rule_portfolio_plan.json`
- `outputs/rule_order_preview.json`
- `outputs/rule_execution_results.json`
- `outputs/rule_account_paper_state.json`
- `outputs/rule_account_live_state.json`
- `outputs/rule_dashboard_summary.json`
- `outputs/rule_signals_latest.json`
- `data/rule_account_live_holdings.csv`

## 주요 실행 파일
- `python/run_rule_after_close_cycle.py`
- `python/run_rule_before_open_cycle.py`
- `python/run_rule_after_open_cycle.py`
- `python/rule_signal_builder.py`
- `python/rule_order_submitter.py`
- `python/rule_execution_simulator.py`
- `python/build_rule_web_payloads.py`
