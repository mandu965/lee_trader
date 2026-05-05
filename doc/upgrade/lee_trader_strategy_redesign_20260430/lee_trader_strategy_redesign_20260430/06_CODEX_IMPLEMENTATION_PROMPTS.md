# Codex 작업 지시문 모음

작성일: 2026-04-30  
목적: Lee Trader 시스템 개선을 Codex로 단계별 실행하기 위한 지시문

---

## 사용 원칙

각 프롬프트는 한 번에 하나씩 실행한다.  
실제 자동매매 서버에 반영하기 전 반드시 local 또는 preview 모드에서 확인한다.

공통 지시:

```text
실제 주문 제출 로직은 기본값 OFF를 유지한다.
BUY 차단 조건은 보수적으로 적용한다.
SELL/EXIT은 신규 BUY 차단과 별도로 다룬다.
기존 outputs/report 포맷을 깨뜨리지 말고 필드를 추가하는 방식으로 구현한다.
환경변수 기본값은 안전한 방향으로 둔다.
테스트 가능한 작은 단위로 수정한다.
```

---

# Prompt 1. 공통 Live Risk Guard 모듈 추가

```text
Lee Trader 프로젝트를 기준으로 실전 자동매매 공통 리스크 차단 모듈을 추가해주세요.

목표:
- AI와 RULE 자동매매 모두에서 사용할 수 있는 common_live_risk_guard.py를 생성합니다.
- BUY 주문 직전 공통 차단 조건을 평가합니다.
- SELL/EXIT은 별도 판단 대상으로 두고, 신규 BUY 차단에 집중합니다.

신규 파일:
- python/common_live_risk_guard.py

필수 함수:
- evaluate_common_buy_guard(order_context: dict) -> tuple[bool, list[str], dict]

확인할 환경변수:
- GLOBAL_KILL_SWITCH 기본 0
- GLOBAL_MAX_DAILY_BUY_AMOUNT 기본 500000
- GLOBAL_MAX_WEEKLY_BUY_AMOUNT 기본 1500000
- GLOBAL_MAX_DAILY_LOSS_PCT 기본 0.01
- GLOBAL_MAX_WEEKLY_LOSS_PCT 기본 0.03
- GLOBAL_BLOCK_BUY_ON_SYNC_STALE 기본 1
- GLOBAL_SYNC_MAX_AGE_MINUTES 기본 30
- GLOBAL_BLOCK_SAME_SYMBOL_BUY_SAME_DAY 기본 1
- GLOBAL_BLOCK_BUY_ON_MARKET_DEFENSIVE 기본 1
- GLOBAL_BLOCK_BUY_ON_MARKET_STATUS_MISSING 기본 1

초기 구현은 파일/DB가 없는 경우에도 죽지 않게 safe fallback으로 작성해주세요.
단, 정보가 없으면 신규 BUY는 차단하는 방향으로 처리해주세요.

출력:
- outputs/common_live_risk_guard.json
- outputs/common_live_risk_guard_report.md

완료 후:
- 간단한 unit-style self test 또는 --self-test 옵션을 추가해주세요.
- 기존 실주문 로직은 아직 연결하지 말고 모듈 단독으로 실행 가능하게 해주세요.
```

---

# Prompt 2. AI submit_live_orders.py에 진입가격 게이트 추가

```text
Lee Trader 프로젝트의 python/submit_live_orders.py를 수정해주세요.

목표:
- AI 기반 BUY 주문 preview 생성 시 전일 종가 대비 현재가 괴리율을 계산합니다.
- 괴리율 기준을 벗어나면 BUY를 blocked_reason으로 차단합니다.
- 실제 주문 제출 직전에도 동일 조건이 반영되도록 합니다.

추가 환경변수:
- ENTRY_GAP_BLOCK_UP_PCT 기본 0.03
- ENTRY_GAP_HARD_BLOCK_UP_PCT 기본 0.05
- ENTRY_GAP_BLOCK_DOWN_PCT 기본 -0.04
- ENTRY_GAP_BLOCK_ON_LIVE_PRICE_MISSING 기본 1

추가 필드:
- previous_close
- live_price
- live_price_source
- entry_price_gap_pct
- entry_price_gate_status
- entry_price_gate_reason

우선 현재가 조회 함수가 이미 있으면 재사용하고, 없으면 KIS 현재가 조회를 안전하게 감싸는 helper를 추가해주세요.
현재가 조회 실패 시 기본적으로 BUY는 차단하고 preview only로 남겨주세요.

차단 기준:
- live_price unavailable: live_price_unavailable
- gap > +5%: entry_gap_up_hard_blocked
- gap > +3%: entry_gap_up_blocked
- gap <= -4%: entry_gap_down_blocked

기존 AUTO_TRADE_EXECUTE, AUTO_TRADE_ALLOW_BUY, AUTO_TRADE_CONFIRM_TEXT 안전장치는 절대 약화하지 마세요.
리포트 markdown에도 entry price gate 결과를 표시해주세요.
```

---

# Prompt 3. RULE order guard에 common_live_risk_guard 연결

```text
Lee Trader 프로젝트에서 RULE 기반 자동매매의 주문 차단 로직을 강화해주세요.

대상 파일:
- python/rule_account_guard.py
- python/rule_order_preview_builder.py
- 필요 시 python/rule_order_submitter.py

목표:
- BUY 주문에 대해 common_live_risk_guard.evaluate_common_buy_guard() 결과를 반영합니다.
- 공통 guard가 차단하면 RULE BUY 주문은 제출되지 않아야 합니다.
- SELL/EXIT은 신규 BUY 차단 조건과 분리합니다.

프리뷰 JSON에 추가할 필드:
- common_risk_allowed
- common_risk_block_reasons
- common_risk_snapshot

기존 차단 조건은 유지해야 합니다:
- paper_mode_no_order_submission
- rule_live_disabled
- rule_order_submit_disabled
- kill_switch_on
- buy_requires_strong_entry
- market_defensive_mode
- gap_risk_blocked
- trading_value_failed
- sector_limit_failed
- cooldown_failed
- cash_limit_failed
- order_amount_exceeds_limit

완료 후 rule_order_preview.json에서 차단 사유가 명확히 확인되도록 해주세요.
```

---

# Prompt 4. live trade ledger 필드 확장

```text
Lee Trader 프로젝트의 실전 거래 기록/리뷰 구조를 확장해주세요.

대상 후보 파일:
- postgres/live_trade_ledger_tables.sql
- postgres/analytics_live_trade_views.sql
- python/sync_live_trade_ledger.py
- python/build_live_trade_review.py
- python/build_live_trade_review_summary.py
- python/build_live_kpi_daily_report.py

목표:
각 실전 거래에 대해 다음 정보를 저장/표시할 수 있게 합니다.

필드 후보:
- engine_type
- strategy_id
- run_mode
- source_score_date
- final_score
- prob_score
- ret_score
- tech_score
- quality_score
- confidence_score
- calibrated_confidence
- live_confidence_grade
- liquidity_score
- market_regime
- previous_close
- live_price
- entry_price_gap_pct
- entry_gate_status
- entry_gate_reason
- buy_reason
- sell_reason
- portfolio_action_reason
- benchmark_name
- benchmark_return_until_exit
- strategy_return
- excess_return
- holding_days
- exit_reason
- review_status

요구사항:
- 기존 DB가 깨지지 않도록 ALTER TABLE 방식 또는 migration-safe 방식으로 작성합니다.
- 값이 없는 경우 NULL 허용합니다.
- 기존 리포트가 깨지지 않도록 backward compatible하게 처리합니다.
- markdown 리포트에는 핵심 필드만 요약 표시합니다.
```

---

# Prompt 5. AI live_confidence_grade 도입

```text
Lee Trader 프로젝트의 AI confidence 사용 방식을 개선해주세요.

대상 후보 파일:
- python/calibrate_operational_confidence.py
- python/build_confidence_calibration_report.py
- python/build_confidence_score_v2.py
- python/apply_execution_policy.py

목표:
raw confidence를 그대로 실전 비중에 사용하지 않고 live_confidence_grade를 생성/사용합니다.

등급:
- A: 표본 충분, hit rate/return/excess return 양호
- B: 표본 일부 충분, 성과 보통 이상
- C: 표본 부족 또는 성과 불명확
- D: 성과 부진 또는 drawdown 과다

기본 규칙:
- bucket sample < 20이면 최대 C
- excess return < -1%이면 D
- recent 10 trade return < -2%이면 D
- 성과 정보가 없으면 C

apply_execution_policy.py에서:
- A: 기존 표준 비중
- B: weight scale 0.5
- C: weight scale 0.2 또는 watch only
- D: BUY 차단

outputs에 다음을 생성/확장:
- confidence_live_grade_map.json
- confidence_live_grade_report.md

기존 confidence_score_v2는 제거하지 말고 병행 사용하세요.
```

---

# Prompt 6. RULE 최대 보유일/손절/트레일링 스탑 추가

```text
Lee Trader 프로젝트의 RULE 기반 포트폴리오 관리 로직을 보강해주세요.

대상 파일:
- python/rule_portfolio_manager.py
- 필요 시 rule state/ledger 관련 파일

목표:
보유 종목에 대해 최대 보유일, 손절, 트레일링 스탑을 적용합니다.

환경변수:
- RULE_MAX_HOLDING_DAYS 기본 10
- RULE_MAX_HOLDING_DAYS_PROFIT_BUFFER 기본 0.02
- RULE_STOP_LOSS_PCT 기본 0.05
- RULE_TRAILING_STOP_PCT 기본 0.04
- RULE_TRAILING_STOP_MIN_PROFIT_PCT 기본 0.03

추가 판단:
- holding_days > RULE_MAX_HOLDING_DAYS and return < +2%: reduce 또는 exit
- return <= -5%: exit 후보
- highest_return_since_entry >= +3% and drawdown_from_high <= -4%: reduce 또는 exit

추가 사유 코드:
- max_holding_days_exit
- stop_loss_exit
- trailing_stop_reduce
- trailing_stop_exit

주의:
- 보유일/진입가/최고가 데이터가 없으면 기존 로직을 유지하고, missing data reason을 기록합니다.
- 신규 BUY 로직은 건드리지 말고 보유 종목 action 판단만 보강합니다.
```

---

# Prompt 7. 운영 상태 대시보드용 payload 추가

```text
Lee Trader 프로젝트에 실서버 자동매매 운영 상태 대시보드용 payload를 추가해주세요.

대상 후보:
- python/sync_auxiliary_payloads.py
- python/run_daily_scheduler.py
- node/public/ops-readiness.js
- node/public/live-auto-trading.js
- node/public/rule-auto-trading.js

목표:
웹 UI에서 다음 항목을 볼 수 있게 합니다.

표시 항목:
- 오늘 종가배치 성공 여부
- 오늘 AI 자동매수 실행 여부
- 오늘 RULE before-open 실행 여부
- 오늘 RULE after-open 실행 여부
- 오늘 live account sync 성공 여부
- 마지막 성공 시각
- 마지막 실패 시각
- 마지막 에러 메시지
- 오늘 BUY 후보 수
- 오늘 BUY 차단 수
- 오늘 실주문 제출 수
- 오늘 체결 수
- GLOBAL_KILL_SWITCH 상태
- RULE_KILL_SWITCH 상태
- AUTO_TRADE_EXECUTE 상태
- AUTO_TRADE_ALLOW_BUY 상태

outputs JSON 또는 payload store에 저장하고, 기존 UI를 깨지 않게 카드 형태로 추가해주세요.
```

---

# Prompt 8. Master Risk Manager 설계 및 preview 통합

```text
Lee Trader 프로젝트에 AI/RULE 통합 주문 승인 레이어를 설계해주세요.

1차 목표는 실주문 연결이 아니라 preview 통합입니다.

신규 파일 후보:
- python/master_risk_manager.py

입력:
- outputs/order_requests_preview.json 또는 AI preview
- outputs/rule_order_preview.json
- live holdings
- live fills
- market status
- common_live_risk_guard 결과

출력:
- outputs/master_approved_orders.json
- outputs/master_blocked_orders.json
- outputs/master_risk_summary.json
- outputs/master_risk_summary.md

승인 기준:
- 동일 종목 중복 BUY 차단
- engine별 일일 예산 제한
- 전체 일일 신규 BUY 제한
- 섹터/테마 총 노출 제한
- 현금 비중 하한 유지
- common risk guard 통과
- entry price gate 통과

주의:
- 1차 구현에서는 실제 주문 제출과 연결하지 않습니다.
- AI/RULE 각각의 기존 주문 흐름은 유지합니다.
- preview 단계에서만 통합 판단 결과를 보여주세요.
```
