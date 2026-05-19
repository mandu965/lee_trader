# Lee_trader_rule File Index

## 목적

RULE 자동매매 모듈에서 실제 운영에 직접 영향을 주는 핵심 파일만 추려 둔 인덱스입니다.
코드 수정 전에는 이 문서와 [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/FLOW.md>), [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/OPERATIONS.md>)를 함께 보는 것을 기준으로 합니다.

## 핵심 파일

| 파일 | 역할 | 수정 위험도 | 함께 확인할 파일 |
| --- | --- | --- | --- |
| `python/run_rule_after_close_cycle.py` | after-close 전체 오케스트레이션 실행 | 높음 | `rule_signal_builder.py`, `rule_portfolio_manager.py`, `rule_order_preview_builder.py`, `build_rule_web_payloads.py` |
| `python/run_rule_before_open_cycle.py` | before-open 실행, 장전 가드 확인, paper/pilot/live 분기 | 높음 | `rule_market_open_snapshot.py`, `rule_execution_simulator.py`, `rule_order_submitter.py`, `rule_account_guard.py` |
| `python/run_rule_after_open_cycle.py` | after-open 체결 동기화와 후속 정리 | 중간 | `rule_order_fill_sync.py`, `rule_live_account_snapshot.py` |
| `python/rule_signal_builder.py` | RULE 진입 후보와 신호 산출 | 높음 | `rule_backtest.py`, `rule_portfolio_manager.py`, `rule_formula_review.py` |
| `python/rule_portfolio_manager.py` | 보유 상태를 반영해 buy/hold/reduce/exit 계획 생성 | 높음 | `rule_order_preview_builder.py`, `rule_paper_state_manager.py`, `rule_live_account_snapshot.py` |
| `python/rule_order_preview_builder.py` | 주문 초안과 `order_allowed` 판단 생성 | 매우 높음 | `rule_account_guard.py`, `rule_portfolio_manager.py`, `build_rule_web_payloads.py` |
| `python/rule_account_guard.py` | 주문 차단 사유, 금액 한도, kill switch, pilot/live guard 판단 | 매우 높음 | `rule_order_preview_builder.py`, `rule_order_submitter.py`, `run_rule_before_open_cycle.py` |
| `python/rule_execution_simulator.py` | paper 모드 체결 시뮬레이션과 reconciliation 처리 | 높음 | `rule_paper_state_manager.py`, `run_rule_before_open_cycle.py` |
| `python/rule_order_submitter.py` | pilot/live 실주문 제출 직전 검증과 KIS 주문 호출 | 매우 높음 | `kis_client.py`, `rule_account_guard.py`, `rule_market_open_snapshot.py`, `rule_order_fill_sync.py` |
| `python/rule_market_open_snapshot.py` | 장전 시세와 장 상태 스냅샷 저장 | 높음 | `kis_client.py`, `run_rule_before_open_cycle.py` |
| `python/rule_live_account_snapshot.py` | RULE 실계좌 잔고/보유 상태 동기화 | 높음 | `kis_client.py`, `build_rule_web_payloads.py`, `sync_web_display_data.py` |
| `python/rule_order_fill_sync.py` | RULE 체결 결과를 수집하고 execution 결과에 반영 | 높음 | `rule_order_submitter.py`, `rule_live_account_snapshot.py` |
| `python/rule_paper_state_manager.py` | paper 계좌 상태와 주문 적용 이력 관리 | 중간 | `rule_execution_simulator.py`, `rule_portfolio_manager.py` |
| `python/build_rule_web_payloads.py` | RULE 대시보드용 summary/detail payload 생성 | 높음 | `node/index.js`, `node/public/rule-auto-trading.js`, `sync_web_display_data.py` |
| `node/index.js` | RULE API 응답 제공 | 높음 | `build_rule_web_payloads.py`, `node/public/rule-auto-trading.js` |
| `node/public/rule-auto-trading.js` | RULE 운영 화면 렌더링 | 중간 | `node/index.js`, `outputs/rule_dashboard_summary.json` |

## 보조 파일

| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `python/rule_backtest.py` | RULE 신호 기준 성과 요약 생성 | after-close 품질 검토용 |
| `python/rule_portfolio_backtest.py` | RULE 포트폴리오 백테스트 | 전략 검증용 |
| `python/rule_daily_report.py` | RULE 운영 리포트 생성 | 대시보드/운영 문서 보강용 |
| `python/rule_formula_review.py` | RULE 식과 기준 리뷰 | 전략 점검용 |
| `config/trading_calendar_kr.json` | 한국 거래일/휴장일 기준 | before-open 실행 가능 여부에 직접 영향 |

## 수정 원칙

- `rule_account_guard.py`, `rule_order_preview_builder.py`, `rule_order_submitter.py`는 한 세트로 봅니다.
- `paper`, `pilot`, `live` 분기를 건드릴 때는 `run_rule_before_open_cycle.py`와 `rule_execution_results.json` 구조를 같이 확인합니다.
- 실주문 관련 변경은 guard, 로그, 상한 중심으로만 접근하고 KIS 호출부는 단순화하지 않습니다.
- RULE 계좌 동기화 관련 수정은 AI 일반 실계좌 경로와 섞지 않습니다.

## Recent Analysis

- [20260519_로직진단.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/20260519_로직진단.md>)
