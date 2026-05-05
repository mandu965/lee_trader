# Lee Trader Codex 개발 마스터 체크리스트

작성일: 2026-04-30  
목적: AI/RULE 자동매매 개선 작업을 하루에 끝내지 않고, 여러 날에 걸쳐 안전하게 진행하기 위한 개발 체크리스트

---

## 0. 개발 운영 원칙

현재 Lee Trader는 실제 서버에서 자동매매가 이루어지고 있으므로, 개발 우선순위는 다음 순서로 고정한다.

```text
1. 실전 주문 사고 방지
2. 신규 BUY 차단 조건 강화
3. 체결/잔고 동기화 신뢰성 확보
4. 주문 사유와 차단 사유 기록
5. 실전 거래 리뷰 데이터 축적
6. confidence와 전략 비중 개선
7. AI/RULE 통합 리스크 관리
8. 모델 고도화
```

### 절대 원칙

```text
[ ] 실제 주문 제출 기본값은 OFF 유지
[ ] BUY 차단 조건은 보수적으로 적용
[ ] SELL/EXIT은 신규 BUY 차단과 분리
[ ] 기존 outputs/report 포맷은 깨지지 않게 필드 추가 방식 사용
[ ] 환경변수 기본값은 안전한 방향으로 설정
[ ] 한 번에 하나의 Codex Prompt만 실행
[ ] Prompt 실행 후 반드시 git diff 확인
[ ] Prompt 실행 후 반드시 preview/local 검증
[ ] 실서버 반영 전 반드시 kill switch/env 확인
```

---

# P0. 즉시 적용 영역: 실전 안전장치 강화

## P0-1. common_live_risk_guard.py 신규 생성

관련 문서:
- `02_LIVE_RISK_CONTROL_SPEC.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 1

### 목표

AI와 RULE 모두에서 공통으로 사용할 신규 BUY 차단 모듈을 만든다.

### 개발 체크리스트

```text
[ ] python/common_live_risk_guard.py 생성
[ ] evaluate_common_buy_guard(order_context: dict) 함수 생성
[ ] GLOBAL_KILL_SWITCH 검사
[ ] GLOBAL_MAX_DAILY_BUY_AMOUNT 검사
[ ] GLOBAL_MAX_WEEKLY_BUY_AMOUNT 검사
[ ] GLOBAL_MAX_DAILY_LOSS_PCT 검사
[ ] GLOBAL_MAX_WEEKLY_LOSS_PCT 검사
[ ] GLOBAL_BLOCK_BUY_ON_SYNC_STALE 검사
[ ] GLOBAL_SYNC_MAX_AGE_MINUTES 검사
[ ] GLOBAL_BLOCK_SAME_SYMBOL_BUY_SAME_DAY 검사
[ ] GLOBAL_BLOCK_BUY_ON_MARKET_DEFENSIVE 검사
[ ] GLOBAL_BLOCK_BUY_ON_MARKET_STATUS_MISSING 검사
[ ] 파일/DB 누락 시 죽지 않도록 safe fallback 처리
[ ] 정보 부족 시 신규 BUY는 차단하도록 처리
[ ] outputs/common_live_risk_guard.json 생성
[ ] outputs/common_live_risk_guard_report.md 생성
[ ] --self-test 또는 unit-style self test 추가
```

### 검증 체크리스트

```text
[ ] GLOBAL_KILL_SWITCH=1이면 BUY 차단
[ ] 동기화 파일 누락 시 BUY 차단
[ ] market_status 누락 시 BUY 차단
[ ] 당일 동일 종목 BUY 이력 있으면 차단
[ ] 한도 초과 시 차단
[ ] SELL/EXIT 로직은 건드리지 않음
[ ] python python/common_live_risk_guard.py --self-test 통과
[ ] outputs/common_live_risk_guard_report.md에서 차단 사유 확인 가능
```

### 완료 기준

```text
[ ] 공통 guard 단독 실행 가능
[ ] AI/RULE 연결 전에도 결과 파일 생성 가능
[ ] 차단 사유가 list[str]로 명확히 반환됨
[ ] risk_snapshot에 판단 근거가 남음
```

---

## P0-2. AI submit_live_orders.py 진입가격 게이트 추가

관련 문서:
- `03_AI_CONFIDENCE_AND_ENTRY_STRATEGY.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 2

### 목표

전일 종가 기준 점수가 높더라도 실제 매수 시점 가격이 불리하면 AI BUY를 차단한다.

### 개발 체크리스트

```text
[ ] python/submit_live_orders.py 수정 전 백업 또는 git 상태 확인
[ ] ENTRY_GAP_BLOCK_UP_PCT 기본 0.03 추가
[ ] ENTRY_GAP_HARD_BLOCK_UP_PCT 기본 0.05 추가
[ ] ENTRY_GAP_BLOCK_DOWN_PCT 기본 -0.04 추가
[ ] ENTRY_GAP_BLOCK_ON_LIVE_PRICE_MISSING 기본 1 추가
[ ] previous_close 필드 추가
[ ] live_price 필드 추가
[ ] live_price_source 필드 추가
[ ] entry_price_gap_pct 필드 추가
[ ] entry_price_gate_status 필드 추가
[ ] entry_price_gate_reason 필드 추가
[ ] 현재가 조회 helper 추가 또는 기존 함수 재사용
[ ] 현재가 조회 실패 시 BUY 차단
[ ] +3% 초과 시 entry_gap_up_blocked
[ ] +5% 초과 시 entry_gap_up_hard_blocked
[ ] -4% 이하 시 entry_gap_down_blocked
[ ] 기존 AUTO_TRADE_EXECUTE 안전장치 유지
[ ] 기존 AUTO_TRADE_ALLOW_BUY 안전장치 유지
[ ] 기존 AUTO_TRADE_CONFIRM_TEXT 안전장치 유지
[ ] markdown report에 entry price gate 결과 표시
```

### 검증 체크리스트

```text
[ ] previous_close=10000, live_price=10200이면 통과
[ ] previous_close=10000, live_price=10350이면 차단
[ ] previous_close=10000, live_price=10500이면 hard block
[ ] previous_close=10000, live_price=9600이면 차단
[ ] live_price 조회 실패 시 차단
[ ] preview JSON에 필드가 모두 존재
[ ] 실제 주문 제출 직전에도 동일 조건 재검증
[ ] 기존 preview/report 소비 로직이 깨지지 않음
```

### 완료 기준

```text
[ ] AI BUY 후보가 가격 괴리율 기준으로 차단 가능
[ ] 차단 사유가 order_requests_preview.json 또는 report에 표시
[ ] 실주문 안전장치가 약화되지 않음
```

---

## P0-3. RULE 주문 guard에 common_live_risk_guard 연결

관련 문서:
- `04_RULE_STRATEGY_HARDENING_PLAN.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 3

### 목표

RULE BUY도 공통 리스크 guard를 반드시 통과하게 만든다.

### 개발 체크리스트

```text
[ ] python/rule_account_guard.py에 common_live_risk_guard import
[ ] BUY 주문에만 evaluate_common_buy_guard 적용
[ ] SELL/EXIT은 신규 BUY 차단 조건과 분리
[ ] python/rule_order_preview_builder.py에 common_risk_allowed 추가
[ ] common_risk_block_reasons 추가
[ ] common_risk_snapshot 추가
[ ] python/rule_order_submitter.py에서 제출 직전 재검증
[ ] 기존 차단 사유 유지
[ ] paper_mode_no_order_submission 유지
[ ] rule_live_disabled 유지
[ ] rule_order_submit_disabled 유지
[ ] kill_switch_on 유지
[ ] buy_requires_strong_entry 유지
[ ] market_defensive_mode 유지
[ ] gap_risk_blocked 유지
[ ] trading_value_failed 유지
[ ] sector_limit_failed 유지
[ ] cooldown_failed 유지
[ ] cash_limit_failed 유지
[ ] order_amount_exceeds_limit 유지
```

### 검증 체크리스트

```text
[ ] RULE BUY 후보에 common_risk_allowed 표시
[ ] common guard 차단 시 RULE BUY 미제출
[ ] SELL/EXIT 후보는 common BUY 차단 때문에 사라지지 않음
[ ] rule_order_preview.json에서 차단 사유 확인 가능
[ ] 기존 RULE 차단 사유가 사라지지 않음
```

### 완료 기준

```text
[ ] RULE BUY는 기존 guard + common guard를 모두 통과해야 제출 가능
[ ] preview와 submit 단계 결과가 일관됨
```

---

# P1. 실전 리뷰 데이터 강화

## P1-1. live trade ledger 필드 확장

관련 문서:
- `01_PRIORITY_ROADMAP.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 4

### 개발 체크리스트

```text
[ ] postgres/live_trade_ledger_tables.sql migration-safe 방식 확인
[ ] ALTER TABLE ADD COLUMN IF NOT EXISTS 방식 적용
[ ] engine_type 추가
[ ] strategy_id 추가
[ ] run_mode 추가
[ ] source_score_date 추가
[ ] final_score 추가
[ ] prob_score 추가
[ ] ret_score 추가
[ ] tech_score 추가
[ ] quality_score 추가
[ ] confidence_score 추가
[ ] calibrated_confidence 추가
[ ] live_confidence_grade 추가
[ ] liquidity_score 추가
[ ] market_regime 추가
[ ] previous_close 추가
[ ] live_price 추가
[ ] entry_price_gap_pct 추가
[ ] entry_gate_status 추가
[ ] entry_gate_reason 추가
[ ] buy_reason 추가
[ ] sell_reason 추가
[ ] portfolio_action_reason 추가
[ ] benchmark_name 추가
[ ] benchmark_return_until_exit 추가
[ ] strategy_return 추가
[ ] excess_return 추가
[ ] holding_days 추가
[ ] exit_reason 추가
[ ] review_status 추가
[ ] sync_live_trade_ledger.py backward compatible 처리
[ ] build_live_trade_review.py 핵심 필드 표시
[ ] build_live_kpi_daily_report.py 요약 반영
```

### 검증 체크리스트

```text
[ ] 기존 DB에 migration 재실행해도 실패하지 않음
[ ] 기존 리포트가 깨지지 않음
[ ] 값 없는 필드는 NULL 허용
[ ] 신규 체결 건에 engine_type과 strategy_id 저장
[ ] 주문 당시 점수와 entry gap이 저장
[ ] 매도 후 holding_days와 excess_return 계산 가능
```

---

# P2. AI confidence 재정의

## P2-1. live_confidence_grade 도입

관련 문서:
- `03_AI_CONFIDENCE_AND_ENTRY_STRATEGY.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 5

### 개발 체크리스트

```text
[ ] confidence_live_grade_map.json 생성 로직 추가
[ ] confidence_live_grade_report.md 생성 로직 추가
[ ] bucket_sample_count < 20이면 최대 C 처리
[ ] excess return < -1%이면 D 처리
[ ] recent 10 trade return < -2%이면 D 처리
[ ] 성과 정보 없으면 C 처리
[ ] A/B/C/D 등급 규칙 함수화
[ ] apply_execution_policy.py에서 grade별 weight scale 적용
[ ] A: 1.0 적용
[ ] B: 0.5 적용
[ ] C: 0.2 또는 watch only 적용
[ ] D: BUY 차단
[ ] 기존 confidence_score_v2 제거하지 않음
```

### 검증 체크리스트

```text
[ ] 표본 부족 bucket은 A/B가 나오지 않음
[ ] 성과 부진 bucket은 D로 강등
[ ] D등급은 BUY 차단
[ ] C등급은 소액 또는 watch only
[ ] 리포트에서 등급 산정 근거 확인 가능
```

---

# P3. RULE 전략 고도화

## P3-1. 최대 보유일/손절/트레일링 스탑

관련 문서:
- `04_RULE_STRATEGY_HARDENING_PLAN.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 6

### 개발 체크리스트

```text
[ ] RULE_MAX_HOLDING_DAYS 기본 10 추가
[ ] RULE_MAX_HOLDING_DAYS_PROFIT_BUFFER 기본 0.02 추가
[ ] RULE_STOP_LOSS_PCT 기본 0.05 추가
[ ] RULE_TRAILING_STOP_PCT 기본 0.04 추가
[ ] RULE_TRAILING_STOP_MIN_PROFIT_PCT 기본 0.03 추가
[ ] holding_days 계산
[ ] entry_price 확보
[ ] highest_price_since_entry 확보 또는 없으면 missing reason 기록
[ ] return <= -5%이면 stop_loss_exit 후보
[ ] holding_days > 10 and return < +2%이면 reduce/exit 후보
[ ] highest_return >= +3% and drawdown_from_high <= -4%이면 reduce/exit 후보
[ ] 신규 BUY 로직은 변경하지 않음
[ ] missing data 시 기존 로직 유지
```

### 검증 체크리스트

```text
[ ] 손절 조건 발동 시 exit reason 기록
[ ] 최대 보유일 조건 발동 시 reason 기록
[ ] 트레일링 스탑 조건 발동 시 reason 기록
[ ] 데이터 부족 시 기존 HOLD/EXIT 판단 유지
[ ] RULE BUY 로직에 부작용 없음
```

---

# P4. 운영 상태 대시보드 payload

## P4-1. 실서버 운영 상태 payload 추가

관련 문서:
- `05_OPERATIONS_AND_MONITORING_CHECKLIST.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 7

### 개발 체크리스트

```text
[ ] 오늘 종가배치 성공 여부 payload 추가
[ ] 오늘 AI 자동매수 실행 여부 추가
[ ] 오늘 RULE before-open 실행 여부 추가
[ ] 오늘 RULE after-open 실행 여부 추가
[ ] live account sync 성공 여부 추가
[ ] 마지막 성공 시각 추가
[ ] 마지막 실패 시각 추가
[ ] 마지막 에러 메시지 추가
[ ] 오늘 BUY 후보 수 추가
[ ] 오늘 BUY 차단 수 추가
[ ] 오늘 실주문 제출 수 추가
[ ] 오늘 체결 수 추가
[ ] GLOBAL_KILL_SWITCH 상태 추가
[ ] RULE_KILL_SWITCH 상태 추가
[ ] AUTO_TRADE_EXECUTE 상태 추가
[ ] AUTO_TRADE_ALLOW_BUY 상태 추가
[ ] 기존 UI를 깨지 않고 카드 형태로 추가
```

### 검증 체크리스트

```text
[ ] outputs JSON만 봐도 오늘 운영 상태 판단 가능
[ ] 웹 UI에서 상태 카드 확인 가능
[ ] scheduler 실패/미실행 구분 가능
[ ] kill switch 상태 표시 가능
```

---

# P5. Master Risk Manager

## P5-1. preview 통합부터 구현

관련 문서:
- `00_MASTER_STRATEGY_REDEFINITION.md`
- `06_CODEX_IMPLEMENTATION_PROMPTS.md` Prompt 8

### 개발 체크리스트

```text
[ ] python/master_risk_manager.py 생성
[ ] AI order preview 입력 처리
[ ] RULE order preview 입력 처리
[ ] live holdings 입력 처리
[ ] live fills 입력 처리
[ ] market status 입력 처리
[ ] 동일 종목 중복 BUY 차단
[ ] engine별 일일 예산 제한
[ ] 전체 일일 신규 BUY 제한
[ ] 섹터/테마 노출 제한
[ ] 현금 비중 하한 유지
[ ] common risk guard 결과 반영
[ ] entry price gate 결과 반영
[ ] outputs/master_approved_orders.json 생성
[ ] outputs/master_blocked_orders.json 생성
[ ] outputs/master_risk_summary.json 생성
[ ] outputs/master_risk_summary.md 생성
[ ] 실제 주문 제출과는 연결하지 않음
```

### 검증 체크리스트

```text
[ ] 같은 종목이 AI/RULE 양쪽에 있으면 하나 이상 차단
[ ] 예산 초과 시 차단
[ ] common guard 차단 항목은 승인되지 않음
[ ] entry price gate 차단 항목은 승인되지 않음
[ ] 기존 AI/RULE 주문 흐름은 그대로 유지
```

---

## 매일 작업 종료 체크리스트

```text
[ ] 오늘 실행한 Codex Prompt 번호 기록
[ ] 수정 파일 목록 기록
[ ] git diff 검토
[ ] 로컬 실행 결과 기록
[ ] outputs 생성 여부 확인
[ ] report markdown 확인
[ ] 테스트 실패 항목 기록
[ ] 실서버 반영 여부 기록
[ ] 실서버 반영했다면 env/kill switch 상태 기록
[ ] 다음 작업 Prompt 번호 기록
[ ] 미해결 리스크 기록
```

---

## 실서버 반영 전 최종 체크리스트

```text
[ ] AUTO_TRADE_EXECUTE 값 확인
[ ] AUTO_TRADE_ALLOW_BUY 값 확인
[ ] AUTO_TRADE_CONFIRM_TEXT 값 확인
[ ] GLOBAL_KILL_SWITCH 값 확인
[ ] RULE_KILL_SWITCH 값 확인
[ ] RULE_ORDER_SUBMIT_ENABLED 값 확인
[ ] KIS API 모드 확인
[ ] 계좌번호 확인
[ ] paper/pilot/live run mode 확인
[ ] 오늘 이미 주문된 종목 확인
[ ] 보유잔고 동기화 최신성 확인
[ ] 체결내역 동기화 최신성 확인
[ ] market_status 최신성 확인
[ ] preview 결과 확인
[ ] 차단 사유 확인
[ ] 예상 주문금액 확인
```
