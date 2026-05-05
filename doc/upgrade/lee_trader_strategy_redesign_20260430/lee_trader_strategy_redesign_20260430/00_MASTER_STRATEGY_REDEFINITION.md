# Lee Trader 자동매매 전략 재정의 마스터 문서

작성일: 2026-04-30  
대상 시스템: `lee_trader` Git 소스 기준  
대상 엔진: AI 기반 자동매매, RULE 기반 자동매매  
운영 상태: 실제 서버 자동매매 진행 중, 표본 부족, 전략 업데이트 필요

---

## 1. 결론

현재 시스템은 단순 추천기가 아니라 다음 흐름을 가진 실제 자동매매 운영 시스템이다.

```text
데이터 수집/가공
→ AI 점수 또는 RULE 신호 생성
→ 후보 선정
→ 게이트/리스크 필터
→ 포트폴리오 목표 비중 산출
→ 주문 프리뷰 생성
→ 실주문 제출
→ 체결/보유 동기화
→ 실전 리뷰/리포트 생성
```

구조 자체는 발전 가능성이 있다. 하지만 현재 단계에서 가장 큰 리스크는 모델이 아니라 **실전 운영 통제의 부족**이다.

따라서 전략 재수립의 핵심은 다음이다.

```text
수익 확대보다 손실 제한 우선
모델 개선보다 실전 검증 체계 우선
자동매수 확대보다 진입가격 통제 우선
AI/RULE 병행보다 통합 노출관리 우선
점수 상승보다 실제 체결 후 성과 추적 우선
```

---

## 2. 현재 시스템의 강점

### 2.1 AI 기반 자동매매 강점

확인된 주요 흐름:

- `model_train.py`, `model_train_optuna.py`
- `build_confidence_score_v2.py`
- `calibrate_operational_confidence.py`
- `build_operational_buy_gate.py`
- `apply_execution_policy.py`
- `build_trade_intents.py`
- `submit_live_orders.py`
- `run_live_auto_trade_cycle.py`

강점:

1. 단일 모델 예측값만 쓰지 않고 final score, confidence, quality, liquidity, risk penalty 등 복수 축을 사용한다.
2. `production_v1.yaml`에서 운영 버전과 threshold를 관리한다.
3. `AUTO_TRADE_EXECUTE`, `AUTO_TRADE_ALLOW_BUY`, `AUTO_TRADE_CONFIRM_TEXT=LIVE_ORDER` 등 실주문 안전장치가 있다.
4. 실주문 후 `sync_live_account_holdings.py`, `sync_live_order_fills.py`, `build_live_trade_review.py`, `build_live_kpi_daily_report.py` 같은 리뷰 루틴이 있다.
5. 프리뷰 기반 주문 구조라 완전 무방비 주문은 아니다.

### 2.2 RULE 기반 자동매매 강점

확인된 주요 흐름:

- `rule_signal_builder.py`
- `rule_portfolio_manager.py`
- `rule_order_preview_builder.py`
- `rule_order_submitter.py`
- `rule_account_guard.py`
- `run_rule_after_close_cycle.py`
- `run_rule_before_open_cycle.py`
- `run_rule_after_open_cycle.py`

강점:

1. `RULE_TREND_LIQUIDITY_V1`로 전략 ID가 명확하다.
2. paper/pilot/live run mode 구분이 있다.
3. `RULE_LIVE_ENABLED`, `RULE_ORDER_SUBMIT_ENABLED`, `RULE_KILL_SWITCH`가 있다.
4. `strong_entry`가 아닌 BUY를 차단한다.
5. 방어장세, 갭 리스크, 거래대금 미달, 섹터 한도, 쿨다운, 현금 한도를 확인한다.
6. RULE 계좌 분리 구조가 존재한다.

---

## 3. 현재 시스템의 핵심 약점

### 3.1 수익성 검증보다 자동화가 앞서 있다

자동주문 구조는 이미 상당히 진행되어 있지만, 실전 표본은 아직 부족하다. 이 상태에서 주문 규모를 키우면 모델의 기대값이 양수인지 확인하기 전에 손실이 누적될 수 있다.

현재 판단:

```text
운영 자동화 수준: 중상
주문 안전장치 수준: 중상
수익성 검증 수준: 낮음~중간
confidence 신뢰도: 아직 검증 필요
실전 확대 적합성: 소액 파일럿 단계
```

### 3.2 AI final score는 좋은 종목 순위이지 좋은 매수가가 아니다

전일 종가 기준으로 점수가 높아도 다음날 시초가/현재가가 급등하면 매수 기대값은 훼손된다.

위험한 흐름:

```text
전일 종가 기준 상위 점수
→ 다음날 갭 상승
→ 09:30 자동매수
→ 고점 추격
→ 손실 또는 낮은 기대수익
```

따라서 AI 자동매수에는 **실시간 진입가격 게이트**가 반드시 추가되어야 한다.

### 3.3 confidence score는 아직 실전 비중 산정에 직접 쓰면 안 된다

문서와 구조상 confidence calibration이 존재하지만, 표본 부족 상태에서는 confidence가 실제 수익 확률을 충분히 설명한다고 보기 어렵다.

운영 원칙:

```text
raw confidence는 참고용
calibrated confidence는 제한적 사용
실전 비중은 live confidence grade로만 결정
```

### 3.4 AI와 RULE이 별도 엔진으로 움직이지만 총 노출 통합 관리가 약하다

AI와 RULE이 서로 다른 계좌 또는 자금을 사용하더라도 전체 투자자산 관점에서는 하나의 포트폴리오다. 같은 종목, 같은 섹터, 같은 시장 국면에 중복 노출될 수 있다.

필요한 구조:

```text
AI 후보
RULE 후보
→ Master Risk Manager
→ 총 노출/중복/현금/시장상태/일손실 확인
→ 최종 주문 승인
```

---

## 4. 전략 재정의 원칙

### 원칙 1. 자동매매의 1차 목표는 수익이 아니라 생존이다

초기 단계에서는 월 수익률보다 다음 지표가 더 중요하다.

```text
주문 누락 없음
체결 동기화 정상
실현손익 기록 정상
진입 사유 기록 정상
손실 제한 정상 작동
재주문/중복주문 없음
```

### 원칙 2. 신규 매수는 보수적으로 줄이고, 리포트는 과하게 늘린다

표본 부족 상태에서는 매매를 늘려서 데이터를 만들고 싶어지지만, 실제 돈이 들어간다면 반대로 해야 한다.

```text
매수 횟수는 제한
매수 사유 기록은 상세화
진입가격 통제는 강화
사후분석은 자동화
```

### 원칙 3. AI와 RULE은 경쟁시키되, 동시에 확대하지 않는다

운영 방식:

```text
RULE: 먼저 소액 실전 파일럿
AI: 프리뷰/소액 실전/감시 병행
2~4주 후 실제 성과로 증액 여부 판단
```

### 원칙 4. 실전 데이터가 쌓이기 전까지는 성과보다 무결성을 본다

2~4주 파일럿 기간의 판단 기준:

```text
체결 누락률
주문 실패율
동기화 실패율
중복주문 여부
진입가격 괴리율
손절/청산 작동 여부
전략별 실제 수익률
benchmark 대비 초과수익
```

---

## 5. 전체 개선 우선순위

### P0. 즉시 적용: 실전 안전장치 강화

목표: 손실 확대와 운영 사고 방지

작업:

1. AI/RULE 공통 일일 손실 제한 추가
2. 일일 총 매수금액 제한 추가
3. 동일 종목 당일 재매수 차단
4. 체결 동기화 실패 시 신규 매수 차단
5. API 장애/데이터 누락 시 신규 매수 차단
6. 실시간 진입가격 괴리율 게이트 추가

완료 기준:

```text
장 시작 전/장중/장마감 후 상태가 불완전하면 BUY가 생성되지 않는다.
일 손실/주간 손실 기준 도달 시 신규매수가 차단된다.
전일 종가 대비 과도한 갭 상승 종목은 자동매수되지 않는다.
```

### P1. 1주 내 적용: 실전 리뷰 데이터 모델 강화

목표: 매매 결과를 전략 개선에 사용할 수 있게 만들기

작업:

1. 주문 당시 점수 구성요소 저장
2. 주문 당시 시장상태 저장
3. 주문 당시 진입가격 괴리율 저장
4. AI/RULE engine_type과 strategy_id 저장
5. 매도 사유와 보유기간 저장
6. benchmark 대비 초과수익 저장

완료 기준:

```text
각 체결 건마다 왜 샀는지, 어떤 조건에서 샀는지, 결과가 어땠는지 추적 가능하다.
```

### P2. 2주 내 적용: confidence calibration 재구성

목표: confidence를 실전 비중 결정에 쓸 수 있게 만들기

작업:

1. raw_confidence와 calibrated_confidence 분리
2. live_confidence_grade A/B/C/D 도입
3. 표본 부족 구간은 자동으로 C 이하 처리
4. confidence 구간별 실현수익률/승률 리포트 생성
5. grade별 주문 비중 제한 적용

완료 기준:

```text
confidence가 높다는 이유만으로 비중이 커지지 않는다.
실제 성과가 검증된 confidence 구간만 표준 비중을 받을 수 있다.
```

### P3. 2~3주 내 적용: RULE 전략 고도화

목표: RULE을 실전 파일럿 엔진으로 안정화

작업:

1. 청산 규칙 명문화
2. 최대 보유일 추가
3. 트레일링 스탑 추가
4. 20일선 이탈/거래량 급감 조건 강화
5. 시장 방어모드 전환 시 신규매수 차단 및 보유 축소
6. RULE 성과 리포트 분리

완료 기준:

```text
RULE은 왜 샀고, 왜 보유하며, 왜 줄이고, 왜 파는지가 완전히 설명된다.
```

### P4. 3~4주 내 적용: AI/RULE 통합 리스크 관리자

목표: 엔진별 주문이 아니라 계좌 전체 위험 기준으로 주문 승인

작업:

1. Master Risk Manager 신규 모듈 설계
2. AI/RULE 후보를 통합 주문 후보로 변환
3. 동일 종목 중복 차단
4. 섹터/테마/시장 총 노출 제한
5. 엔진별 예산 제한
6. 전체 현금 비중 제한

완료 기준:

```text
AI와 RULE이 같은 날 같은 위험에 중복 베팅하지 않는다.
최종 주문은 항상 Master Risk Manager를 통과해야 한다.
```

### P5. 1개월 이상: 모델/전략 성과 검증 강화

목표: 장기적으로 돈이 되는 조건만 남기기

작업:

1. walk-forward split 확대
2. 실제 체결가 기준 성과 계산
3. 수수료/슬리피지 반영
4. top20/top10/top5 성과 비교
5. AI vs RULE vs benchmark 비교
6. 시장 국면별 성과 분리

완료 기준:

```text
어떤 시장에서 어떤 엔진이 돈을 벌고 잃는지 수치로 판단 가능하다.
```

---

## 6. 운영 권고

현재 실제 서버에서 자동매매가 이루어지고 있으므로, 개선 작업 중 운영 모드는 다음을 권장한다.

```text
AI 자동매수: 신규 BUY 최소화 또는 승인제
RULE 자동매수: pilot 소액 유지
자동 SELL/EXIT: 기존 규칙 유지하되 로그 강화
수동 개입: 가능하게 유지
전체 증액: 보류
```

운영 금액 확대 조건:

```text
최소 20건 이상 실전 체결 표본
체결/동기화 오류 0건 또는 원인 통제 완료
일별 리포트 누락 0건
AI/RULE별 손익 분리 가능
benchmark 대비 초과수익 양수
최대낙폭 허용범위 내
```

---

## 7. 최종 방향

Lee Trader의 다음 단계는 “더 똑똑한 AI”가 아니라 **더 안전한 운용 체계**다.

우선순위는 다음과 같다.

```text
1. 손실 제한
2. 진입가격 통제
3. 체결/보유 동기화 신뢰성
4. 실전 리뷰 데이터 축적
5. confidence 보정
6. RULE 청산 고도화
7. AI/RULE 통합 노출관리
8. 모델 고도화
```

이 순서가 지켜지면 시스템은 개인 자동매매 도구를 넘어, 신뢰 가능한 종목 추천/운영 리포트 서비스로 발전할 수 있다.
