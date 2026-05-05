# Lee_trader_rule Context

## 개요

- 이 모듈은 `RULE_TREND_LIQUIDITY_V1` 기반 RULE 자동매매 흐름을 다룹니다.
- 핵심 운영 사이클은 `after-close -> before-open -> after-open`입니다.
- 화면, API, 산출물, 계좌 동기화까지 RULE 전용 경로로 분리해서 관리합니다.

## 운영 흐름

- `after-close`
  - `rule_signal_builder.py`가 신호를 생성합니다.
  - `rule_backtest.py`, `rule_portfolio_manager.py`, `rule_order_preview_builder.py`가 후속 산출물을 만듭니다.
  - `build_rule_web_payloads.py`가 RULE 대시보드용 payload를 생성합니다.
- `before-open`
  - `run_rule_before_open_cycle.py`가 거래일, 시간, 계좌, 실행 모드를 확인합니다.
  - `paper`면 `rule_execution_simulator.py`로 갑니다.
  - `pilot/live`면 `rule_order_submitter.py`로 가며 KIS 인증과 주문 가드를 다시 확인합니다.
- `after-open`
  - `rule_order_fill_sync.py`와 `rule_live_account_snapshot.py`가 체결과 계좌 상태를 정리합니다.

## 핵심 판단 기준

- 신호 생성:
  - 거래대금, gap risk, 시장 방어 상태, RSI, 추세 기반으로 `entry_signal`, `strong_entry_signal` 등을 계산합니다.
- 포트폴리오 계획:
  - 보유 종목 수, 종목/섹터 비중, cash 비중, 보유일, stop loss, trailing stop, reduce/exit 기준을 반영합니다.
- 주문 preview:
  - `order_qty`, `order_amount`, `order_allowed`, `order_block_reason`를 생성합니다.
- 실주문 제출:
  - `paper/pilot/live` 분기
  - kill switch
  - 시간창
  - 거래일
  - KIS 인증
  - 계좌 상태
  - 금액 상한
  - 수량 0 초과
  - BUY/SELL 최종 guard

## 운영상 주의점

- RULE은 AI 일반 실자동매매와 계좌, 앱키, payload를 섞지 않습니다.
- `rule_account_guard.py`, `rule_order_preview_builder.py`, `rule_order_submitter.py`는 하나의 변경 묶음으로 봐야 합니다.
- `config/trading_calendar_kr.json`은 before-open 실행 가능 여부에 직접 영향을 줍니다.
- `RULE_TRADING_RUN_MODE`, `RULE_LIVE_ENABLED`, `RULE_ORDER_SUBMIT_ENABLED`, `RULE_KILL_SWITCH`는 실행 결과에 직접 반영됩니다.
- stale preview나 stale summary가 남으면 화면이 paper처럼 보일 수 있으므로 `build_rule_web_payloads.py`와 `sync_web_display_data.py` 갱신이 같이 필요합니다.

## 연관 모듈

- `Lee_trader_ai`
  - 일부 입력 데이터와 웹 payload 저장 경로를 공유합니다.
- `Lee_trader_backTest`
  - RULE 포트폴리오 백테스트 결과 해석과 연결됩니다.
- `node/index.js`
  - `/api/rule/*` 응답을 제공합니다.

## 확인 포인트

- RULE 실계좌 동기화는 일반 AI 실계좌 경로와 분리되어 있는지
- 최신 preview/execution 결과가 실제 화면 payload까지 반영되었는지
- live 관련 변경이 paper simulator 동작을 깨지 않았는지
