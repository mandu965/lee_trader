# Lee_trader_rule File Index

## 소스 파일 목록
| 파일 | 역할 | 수정 가능 여부 | 수정 시 주의사항 |
| --- | --- | --- | --- |
| `python/rule_signal_builder.py` | 룰 신호 생성, `rule_signals.csv` 저장 | 핵심 파일, 신중 수정 | 출력 컬럼은 후속 포트폴리오/백테스트/대시보드가 직접 사용 |
| `python/rule_backtest.py` | 룰 신호 D+1 open 백테스트 리포트 생성 | 수정 가능 | `rule_signals.csv`, 가격 CSV 컬럼 계약 유지 |
| `python/rule_portfolio_manager.py` | 보유 종목 평가와 포트폴리오 액션 결정 | 핵심 파일, 신중 수정 | 상태 JSON 및 preview builder와 강하게 결합 |
| `python/rule_order_preview_builder.py` | 주문 preview JSON 생성 | 핵심 파일, 신중 수정 | `order_allowed`, `order_block_reason` 계약 유지 |
| `python/rule_execution_simulator.py` | paper 모드 실행 / state 반영 / reconciliation | 핵심 파일, 신중 수정 | 모의 state 이력과 preview 상태 전이를 함께 봐야 함 |
| `python/rule_order_submitter.py` | live/pilot 실주문 제출 | 제한적 수정 권장 | KIS 실주문 호출 포함 |
| `python/rule_market_open_snapshot.py` | KIS 시가 snapshot 수집 | 제한적 수정 권장 | 인증/재시도/장시작 데이터 의존 |
| `python/rule_live_account_snapshot.py` | live 계좌 상태 JSON/CSV 동기화 | 신중 수정 | live holdings CSV와 UI payload가 의존 |
| `python/rule_paper_state_manager.py` | paper account state 반영 | 신중 수정 | preview 재적용 방지 `last_applied_order_ids` 유지 필요 |
| `python/rule_account_guard.py` | 주문 차단 로직 | 핵심 파일, 신중 수정 | preview/live submit 모두 같은 guard를 사용 |
| `python/build_rule_web_payloads.py` | 룰 대시보드 JSON 집계 | 수정 가능 | `/api/rule/*` 응답 구조와 연결 |
| `python/run_rule_after_close_cycle.py` | after-close 오케스트레이션 | 신중 수정 | 신호, 백테스트, 포트폴리오, preview, report 순서 보장 필요 |
| `python/run_rule_before_open_cycle.py` | before-open 오케스트레이션 | 신중 수정 | run_mode 분기만 단순하지만 실제 실행 대상이 중요 |
| `python/run_rule_after_open_cycle.py` | after-open fill sync 진입점 | 수정 가능 | live 모드에서만 동작 |
| `node/public/rule-auto-trading.js` | 룰 자동매매 UI | 수정 가능 | `/api/rule/*` 응답 필드명 변경 시 함께 수정 |
| `node/index.js` | 룰 API 엔드포인트 제공 | 신중 수정 | `/api/rule/*` 경로와 payload key 유지 |

## 수정 기준
- `rule_order_submitter.py`, `rule_market_open_snapshot.py`는 실계좌/KIS 호출이 있어서 직접 변경 전 검증 환경이 필요하다.
- `rule_signal_builder.py`의 컬럼명 변경은 `rule_backtest.py`, `rule_portfolio_manager.py`, `build_rule_web_payloads.py`까지 연쇄 영향이 있다.

## 확인 필요
- `rule_daily_report.py`, `rule_order_fill_sync.py`, `rule_formula_review.py`의 운영상 중요도는 확인됐지만 이번 문서 범위에서는 전체 로직을 끝까지 추적하지 않았다.
