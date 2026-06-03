# Lee Trader AI 개선 — Claude Code 프롬프트 플레이북

> **사용법**: 각 과제 시작 시 해당 프롬프트를 **그대로 복사**해서 Claude Code에 붙여넣으세요.
> 작업 순서는 우선순위 기준입니다. 즉시 착수(🔴) → 1~2개월(🟡) → 3개월 이상(🟢) 순으로 진행하세요.

---

## 목차

| 우선순위 | 과제 | 섹션 |
|---|---|---|
| 🔴 즉시 | 2-A Shadow 승격 기준 확정 | [바로가기](#2-a-shadow--production-승격-기준-확정) |
| 🔴 즉시 | 3-A BUY 승인 조건부 자동화 | [바로가기](#3-a-buy-승인-조건부-자동화) |
| 🔴 즉시 | 4-B KPI 이상 자동 알림 | [바로가기](#4-b-kpi-이상-자동-알림) |
| 🔴 즉시 | 5-A KIS API 오류 복구 강화 | [바로가기](#5-a-kis-api-오류-복구-강화) |
| 🟡 1~2개월 | 1-A 피처 다양화 (섹터 로테이션) | [바로가기](#1-a-피처-다양화--섹터-로테이션) |
| 🟡 1~2개월 | 1-B 멀티 Horizon 앙상블 | [바로가기](#1-b-멀티-horizon-앙상블) |
| 🟡 1~2개월 | 1-C 모델 Drift 감지 | [바로가기](#1-c-모델-drift-감지--자동-재학습-트리거) |
| 🟡 1~2개월 | 2-B Regime 가중치 자동화 | [바로가기](#2-b-regime-가중치-데이터-기반-자동화) |
| 🟡 1~2개월 | 2-C final_score 버전 정리 | [바로가기](#2-c-final_score-버전-정리) |
| 🟡 1~2개월 | 3-B 부분 체결 대응 | [바로가기](#3-b-부분-체결-대응-로직) |
| 🟡 1~2개월 | 4-A 실거래 Outcome 피드백 루프 | [바로가기](#4-a-실거래-outcome-피드백-루프) |
| 🟡 1~2개월 | 4-C Walkforward REJECTED 자동 대응 | [바로가기](#4-c-walkforward-rejected-자동-대응) |
| 🟡 1~2개월 | 5-B DART 재무 지연 대응 | [바로가기](#5-b-dart-재무-지연-대응) |
| 🟢 3개월~ | 3-C 동적 포지션 사이징 고도화 | [바로가기](#3-c-동적-포지션-사이징-고도화) |
| 🟢 3개월~ | 5-C 장중 이벤트 선택적 반영 | [바로가기](#5-c-장중-이벤트-선택적-반영-옵션) |
| 🟢 3개월~ | 6-A RULE-AI 통합 성과 대시보드 | [바로가기](#6-a-rule-ai-통합-성과-대시보드) |
| 🟢 3개월~ | 6-B 종목 상세 예측 이력 추가 | [바로가기](#6-b-종목-상세-화면-예측-이력-추가) |
| 🟢 3개월~ | 6-C 모바일 대응 | [바로가기](#6-c-모바일-대응) |

---

---

# 🔴 즉시 착수 과제

---

## 2-A Shadow → Production 승격 기준 확정

```
지금부터 과제 2-A (shadow → production 승격 기준 확정)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_score/CONTEXT.md
- doc/modules/Lee_trader_score/RUNTIME_SORTING.md
- doc/modules/Lee_trader_backTest/CONTEXT.md
- python/build_walkforward_acceptance.py
- python/production_config.py
- outputs/walkforward_acceptance.json (있으면)
- outputs/walk_forward_score_validation.md (있으면)

배경:
- quality_risk_guard shadow가 sidecar로 붙어 있으나 production 승격 기준이 명문화되지 않은 상태
- walkforward REJECTED/ACCEPTED 판정이 있지만 shadow를 언제 production으로 올릴지 기준이 없음

작업 목표:
1. 현재 shadow(quality_risk_guard)의 성능 지표를 walkforward 결과에서 확인
2. shadow → production 승격 조건 정의
   - 조건 예시: top20 > top50 ordering 3개월 이상 유지 + MDD 개선 확인
   - 조건은 코드에서 읽을 수 있는 수치 기반으로 작성
3. doc/shadow_promotion_criteria.md 문서 신규 작성
4. production_config.py에 SCORE_FORMULA_VERSION 환경변수 기반 feature flag 구현
   - 현재 shadow를 production으로 전환할 수 있는 스위치 역할
5. 전환 테스트: SCORE_FORMULA_VERSION 변경 시 ranking 결과가 정상적으로 바뀌는지 확인

주의사항:
- ranking_builder.py 수정 시 점수·순위·UI·주문에 동시 영향이 가므로 shadow 비교 먼저 실행
- AUTO_TRADE_EXECUTE=0 상태 유지
- 작업 완료 후 doc/modules/Lee_trader_score/CONTEXT.md 갱신
```

---

## 3-A BUY 승인 조건부 자동화

```
지금부터 과제 3-A (BUY 승인 조건부 자동화)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/ENV.md
- python/submit_live_orders.py
- python/common_live_risk_guard.py
- python/build_trade_intents.py
- python/build_live_order_preview.py
- outputs/order_buy_approvals.json (있으면)
- outputs/trade_intents.json (있으면)

배경:
- 현재 AUTO_TRADE_BUY_APPROVAL_REQUIRED 플래그로 모든 BUY가 수동 승인에 의존
- confidence_score와 buy_eligibility 조건이 명확한데도 사람이 매번 확인해야 하는 구조
- 조건이 충분히 좋은 종목은 자동으로 승인하여 운영 부담 감소 필요

작업 목표:
1. 자동 승인 조건 정의 및 문서 작성 (doc/auto_buy_approval_policy.md)
   - 조건: confidence_score ≥ 70 AND buy_eligibility = BUY_ALLOWED AND live_rank ≤ 10
   - 초기에는 보수적으로 설정 (나중에 완화 가능하도록 환경변수화)
2. submit_live_orders.py에 조건부 자동 승인 로직 추가
   - 조건 충족 시 approval_source = "auto" 로 order_buy_approvals.json에 기록
   - 조건 미충족 시 기존 수동 승인 흐름 유지
3. order_buy_approvals.json 스키마에 approval_source, auto_approval_reason 필드 추가
4. common_live_risk_guard.py에 자동 승인 guard 연동
   - 자동 승인된 종목도 risk guard를 통과해야 최종 제출
5. 환경변수 추가 (.env.example 갱신)
   - AUTO_TRADE_AUTO_APPROVE_MIN_CONFIDENCE=70
   - AUTO_TRADE_AUTO_APPROVE_MAX_RANK=10

주의사항:
- AUTO_TRADE_EXECUTE=0 상태로 작업 (실주문 차단 유지)
- submit_live_orders.py와 run_live_auto_trade_cycle.py는 실주문 경로이므로 로그 중심으로만 먼저 확인
- 변경 후 반드시 paper trading 환경에서 3일 이상 검증 후 실계좌 적용
- 작업 완료 후 doc/modules/Lee_trader_ai/OPERATIONS.md 갱신
```

---

## 4-B KPI 이상 자동 알림

```
지금부터 과제 4-B (KPI 이상 자동 알림)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/ENV.md
- python/score_kpi_monitor.py
- python/run_daily_scheduler.py
- python/run_live_auto_trade_cycle.py
- .env.example

배경:
- walkforward REJECTED, KPI 경고, 주문 오류 등 이상 상황 발생 시 운영자가 화면을 직접 확인해야만 알 수 있음
- 자동 알림이 없어 이상 상황 대응이 늦어지는 경우 발생

작업 목표:
1. python/notifier.py 신규 작성
   - Slack Webhook 알림 함수 구현
   - 알림 레벨 정의: INFO / WARNING / CRITICAL
   - SLACK_WEBHOOK_URL 환경변수 없으면 콘솔 출력으로 fallback (운영 중단 없이)
2. score_kpi_monitor.py에 알림 호출 연동
   - 알림 조건:
     a. walkforward_acceptance = REJECTED
     b. 상위 20개 종목 평균 final_score가 기준치 이하
     c. buy_eligibility = BUY_ALLOWED 종목이 0개
3. run_live_auto_trade_cycle.py에 알림 연동
   - 알림 조건:
     a. 주문 제출 실패 (KIS API 오류)
     b. 체결 동기화 실패
4. .env.example에 환경변수 추가
   - SLACK_WEBHOOK_URL=
   - ALERT_MIN_SCORE_THRESHOLD=40
5. doc/alert_policy.md 신규 작성
   - 알림 조건 / 레벨 / 채널 정의

주의사항:
- notifier.py는 알림 실패 시 절대 예외를 밖으로 던지지 않도록 설계 (try/except로 감싸기)
- 알림 모듈 오류가 메인 파이프라인을 멈추면 안 됨
- 작업 완료 후 doc/modules/Lee_trader_ai/OPERATIONS.md와 ENV.md 갱신
```

---

## 5-A KIS API 오류 복구 강화

```
지금부터 과제 5-A (KIS API 오류 복구 강화)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/ENV.md
- python/kis_client.py
- python/kis_live_account.py
- python/download_prices_kis.py
- .env.example

배경:
- KIS API 호출 시 rate limit(429), timeout, 일시적 서버 오류가 발생할 수 있음
- 현재 재시도 로직이 없거나 단순 1회 재시도 수준이라 파이프라인 중단으로 이어짐
- 장 시작 직후나 이벤트 집중 시간대에 rate limit 발생 빈도 높음

작업 목표:
1. kis_client.py에 exponential backoff 재시도 데코레이터 구현
   - 기본 설정: 최대 3회 재시도, 초기 대기 1초, 배수 2 (1s → 2s → 4s)
   - 429(rate limit) 응답 시: Retry-After 헤더 확인 후 해당 시간만큼 대기
   - 5xx 서버 오류: 재시도
   - 4xx 클라이언트 오류(429 제외): 재시도 없이 즉시 실패 처리
2. 재시도 파라미터 환경변수화 (.env.example 갱신)
   - KIS_MAX_RETRY=3
   - KIS_RETRY_WAIT_SEC=1
   - KIS_RETRY_BACKOFF_FACTOR=2
3. 재시도 초과 시 동작 정의
   - 로그에 오류 상세 기록
   - notifier.py가 있으면 CRITICAL 알림 전송 (없으면 로그로만)
4. download_prices_kis.py에 데코레이터 적용 확인
5. kis_live_account.py 주요 함수에 데코레이터 적용

주의사항:
- 데코레이터 적용 후 기존 함수 시그니처가 바뀌면 안 됨
- 재시도 로직이 실주문 함수(order_cash 등)에 적용되면 중복 주문 위험 있음
  → 실주문 함수에는 재시도 데코레이터 적용 금지, 별도 처리
- 작업 완료 후 doc/modules/Lee_trader_ai/ENV.md 갱신
```

---

---

# 🟡 1~2개월 과제

---

## 1-A 피처 다양화 — 섹터 로테이션

```
지금부터 과제 1-A (피처 다양화 - 섹터 로테이션 지표 추가)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/FILE_INDEX.md
- doc/feature_gap_analysis.md
- python/feature_builder.py
- python/quality_builder.py
- data/universe.csv (헤더만 확인)
- data/features.csv (헤더만 확인)

배경:
- 현재 feature는 개별 종목의 가격/모멘텀/기술/품질 위주
- 섹터 레벨 상대 강도(어느 섹터가 지금 outperform 중인지)가 feature에 없음
- universe.csv에 sector 컬럼이 이미 있어 추가 데이터 수집 없이 계산 가능
- flow_daily 종목별 수급은 소스 불안정으로 이번 작업에서 제외

작업 목표:
1. feature_builder.py에 섹터 로테이션 feature 추가
   - sector_ret_5d: 해당 종목이 속한 섹터의 최근 5일 평균 수익률
   - sector_ret_20d: 섹터 최근 20일 평균 수익률
   - sector_relative_strength: 종목 수익률 - 섹터 평균 수익률 (상대 강도)
   - sector_rank: 전체 섹터 중 해당 섹터의 최근 수익률 순위 (1=최강)
2. 신규 feature 컬럼이 features.csv와 DB features 테이블에 정상 적재되는지 확인
3. ranking_builder.py에서 신규 feature 활용 여부 검토
   - tech_score 계산에 sector_relative_strength 반영 가능 여부 확인
   - 바로 반영하지 않아도 되고, 컬럼만 추가해두는 것도 OK
4. 신규 feature 추가 전/후 features.csv 비교 확인

주의사항:
- feature 추가 후 model_train.py 재학습 필요 여부 확인
- 새 컬럼 추가로 기존 컬럼이 깨지면 안 됨
- 작업 완료 후 doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신
```

---

## 1-B 멀티 Horizon 앙상블

```
지금부터 과제 1-B (멀티 Horizon 앙상블)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_score/CONTEXT.md
- doc/modules/Lee_trader_score/RUNTIME_SORTING.md
- python/model_predict.py
- python/ranking_builder.py
- python/scoring/final_score.py
- data/predictions.csv (헤더 + 상위 5행 확인)

배경:
- 현재 모델은 60d / 90d 예측값을 따로 계산하지만 ranking에서 하나만 선택해 사용
- 두 horizon을 가중 평균하면 단일 horizon보다 안정적인 예측 가능
- ret_score 입력을 앙상블 값으로 전환하는 것이 핵심

작업 목표:
1. model_predict.py에 앙상블 컬럼 추가
   - pred_return_ensemble = pred_return_60d * 0.6 + pred_return_90d * 0.4
   - pred_mdd_ensemble = pred_mdd_60d * 0.6 + pred_mdd_90d * 0.4
   - 가중치는 환경변수로 설정 가능하게 (ENSEMBLE_WEIGHT_60D=0.6)
2. predictions.csv 및 DB predictions 테이블에 앙상블 컬럼 추가
   - schema.sql 컬럼 추가 DDL 작성
3. ranking_builder.py의 ret_score 계산에 앙상블 컬럼 반영
   - 기존 pred_return_60d 사용 부분을 pred_return_ensemble로 교체
   - 기존 컬럼도 유지 (하위 호환)
4. shadow 비교 실행
   - 앙상블 적용 전/후 final_score 분포 비교
   - ranking 상위 20개 변화 확인
5. sync_web_display_data.py payload에 앙상블 컬럼 반영

주의사항:
- ranking_builder.py 수정은 점수·순위·UI·주문에 동시 영향 → shadow 비교 먼저
- DB 컬럼 추가 후 기존 파이프라인이 오류 없이 돌아가는지 확인
- 작업 완료 후 doc/modules/Lee_trader_score/CONTEXT.md 갱신
```

---

## 1-C 모델 Drift 감지 / 자동 재학습 트리거

```
지금부터 과제 1-C (모델 drift 감지 및 자동 재학습 트리거)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- python/score_kpi_monitor.py
- python/run_daily_scheduler.py
- python/model_train.py
- python/notifier.py (4-B 완료 후 존재)

배경:
- 예측 분포가 시장 변화로 이탈해도 자동 감지 수단이 없음
- 운영자가 KPI 화면을 매일 확인하지 않으면 모델 성능 저하를 늦게 발견함
- drift 감지 → 알림 → (선택) 재학습 트리거 흐름이 필요

작업 목표:
1. score_kpi_monitor.py에 drift 감지 로직 추가
   - 감지 조건:
     a. prob_top20_60d 평균이 N일 연속으로 임계값 이하 (기본: 0.35 × 5일)
     b. 상위 20개 종목의 realized_return vs pred_return 오차가 기준 초과
   - 임계값은 환경변수로 설정 (DRIFT_PROB_THRESHOLD=0.35, DRIFT_CONSECUTIVE_DAYS=5)
2. drift 감지 이력 저장
   - DB에 research.model_drift_log 테이블 신규 생성 (schema.sql DDL 작성)
   - 컬럼: date, drift_type, metric_value, threshold, triggered_action, created_at
3. drift 감지 시 동작
   - CRITICAL 알림 전송 (notifier.py 연동)
   - AUTO_RETRAIN=1 환경변수가 설정된 경우 model_train.py 자동 실행
   - AUTO_RETRAIN=0 (기본)이면 알림만 발송 후 수동 재학습 유도
4. run_daily_scheduler.py에 drift 감지 스텝 추가

주의사항:
- 자동 재학습은 기본적으로 비활성화 (AUTO_RETRAIN=0)
- 재학습 트리거 시 기존 model.pkl 백업 먼저
- 4-B (notifier.py)가 완료된 후 진행 권장
- 작업 완료 후 doc/modules/Lee_trader_ai/OPERATIONS.md 갱신
```

---

## 2-B Regime 가중치 데이터 기반 자동화

```
지금부터 과제 2-B (regime 가중치 데이터 기반 자동화)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_score/CONTEXT.md
- doc/modules/Lee_trader_score/RUNTIME_SORTING.md
- python/scoring/final_score.py
- python/ranking_builder.py
- python/production_config.py
- config/production_v1.yaml
- data/experiments/best_weight.json
- data/experiments/best_weight_by_regime.json
- python/research/generate_weight_grid.py

배경:
- bull / neutral / defensive regime별 가중치가 config 파일에 수동 하드코딩
- walkforward 결과를 보고 운영자가 직접 수정해야 하는 구조
- 최적 가중치를 데이터에서 자동 탐색하는 파이프라인 필요

작업 목표:
1. generate_weight_grid.py 검토 및 regime별 분리 탐색 기능 추가
   - 기존 그리드 서치를 bull / neutral / defensive 각각 별도 실행
   - 각 regime의 walkforward 기간 데이터만 사용하여 최적 가중치 탐색
2. 탐색 결과를 data/experiments/best_weight_by_regime.json에 자동 저장
3. production_config.py에서 best_weight_by_regime.json을 읽어 가중치 적용
   - USE_AUTO_REGIME_WEIGHTS=1 환경변수로 on/off 가능
   - 기본값은 0 (기존 수동 설정 유지)
4. 기존 수동 가중치 vs 자동 탐색 가중치 비교 리포트 생성
   - outputs/weight_comparison_report.md
5. 환경변수 추가 (.env.example 갱신)
   - USE_AUTO_REGIME_WEIGHTS=0

주의사항:
- 가중치 변경은 ranking 전체에 영향 → shadow 비교 후 검토
- USE_AUTO_REGIME_WEIGHTS는 기본 0으로 유지
- 작업 완료 후 doc/modules/Lee_trader_score/CONTEXT.md 갱신
```

---

## 2-C final_score 버전 정리

```
지금부터 과제 2-C (final_score 버전 정리)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_score/CONTEXT.md
- doc/modules/Lee_trader_score/RUNTIME_SORTING.md
- python/ranking_builder.py
- python/sync_web_display_data.py
- data/ranking_final.csv (헤더만 확인)

배경:
- ranking_final.csv에 final_score / final_score_v2 / final_score_v3 / live_score 혼재
- 각 컬럼의 역할과 운영 기준이 코드 외부에 명문화되어 있지 않음
- 어떤 컬럼이 실제 주문 판단에 사용되는지 명확히 정리 필요

작업 목표:
1. ranking_builder.py 분석
   - 각 score 버전의 계산 방식과 쓰임새 파악
   - final_score / v2 / v3 / live_score의 차이 정리
2. doc/score_column_definitions.md 신규 작성
   - 컬럼명 / 계산 방식 / 운영 사용 여부 / 연구 전용 여부 표로 정리
3. 운영 기준 명문화
   - 실주문 판단 기준: final_score (v1) 으로 단일화
   - v2, v3는 research 전용으로 명시
4. sync_web_display_data.py payload에서 research 전용 컬럼 노출 범위 정리
   - 운영자 화면에 불필요한 버전 컬럼 정리 (숨김 처리 또는 제거)
5. doc/modules/Lee_trader_score/RUNTIME_SORTING.md 갱신

주의사항:
- 컬럼 제거는 하지 않음 (하위 호환 유지) — 노출 범위만 정리
- 운영 기준 변경 없이 문서화 + 코드 주석만으로도 충분
- 작업 완료 후 doc/modules/Lee_trader_score/RUNTIME_SORTING.md 갱신
```

---

## 3-B 부분 체결 대응 로직

```
지금부터 과제 3-B (부분 체결 대응 로직)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/FLOW.md
- python/sync_live_order_fills.py
- python/submit_live_orders.py
- python/kis_client.py
- python/kis_live_account.py
- outputs/order_requests_execution.json (있으면)

배경:
- 현재 시장가 주문(ord_dvsn=01) 위주라 대부분 전량 체결되지만 유동성 낮은 종목에서 부분 체결 가능
- 부분 체결 시 잔여 수량 처리 로직 없음
- 부분 체결이 반복되면 포트폴리오 비중이 의도와 달라짐

작업 목표:
1. KIS API 체결 응답에서 부분 체결 상태 확인
   - 체결 상태 코드 및 체결 수량 vs 주문 수량 비교 방법 파악
2. sync_live_order_fills.py에 부분 체결 감지 로직 추가
   - 체결 수량 < 주문 수량이면 부분 체결로 분류
   - outputs/partial_fill_queue.json에 잔여 수량 기록
3. submit_live_orders.py에 재제출 로직 추가
   - partial_fill_queue.json 확인 후 잔여 수량 재주문
   - 최대 재제출 횟수: AUTO_TRADE_MAX_RESUBMIT=2 (환경변수)
   - 재제출 간격: AUTO_TRADE_RESUBMIT_WAIT_SEC=60 (환경변수)
4. 최대 재제출 초과 시
   - 잔여 수량 로그 기록
   - notifier.py 알림 전송 (4-B 완료 후 연동)
5. .env.example 환경변수 추가
6. build_live_trade_consistency_report.py에 부분 체결 통계 추가

주의사항:
- AUTO_TRADE_EXECUTE=0 상태로 작업
- 재제출 로직이 중복 주문을 만들지 않도록 request_id 체크 필수
- 4-B (notifier.py) 완료 후 알림 연동
- 작업 완료 후 doc/modules/Lee_trader_ai/FLOW.md 갱신
```

---

## 4-A 실거래 Outcome 피드백 루프

```
지금부터 과제 4-A (실거래 outcome 피드백 루프)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_backTest/CONTEXT.md
- doc/modules/Lee_trader_backTest/FLOW.md
- python/sync_live_trade_ledger.py
- python/run_daily_scheduler.py
- schema.sql (research 스키마 테이블 확인)

배경:
- paper trading 추적은 잘 되어 있지만 실계좌 체결 결과가 모델 재학습에 피드백되지 않음
- 실거래 수익률과 모델 예측 수익률 비교가 수동 작업
- 피드백 루프가 없으면 모델이 실거래 환경의 특성을 학습하지 못함

작업 목표:
1. sync_live_trade_ledger.py 분석
   - 현재 체결/거래 데이터 구조 파악
   - research.backtest_outcome 테이블 스키마 확인
2. 실거래 outcome을 research.backtest_outcome에 적재하는 로직 추가
   - 체결 완료된 거래의 실제 수익률 계산
   - 동일 종목/날짜의 모델 예측값(predictions 테이블)과 매핑
3. python/build_live_vs_prediction_report.py 신규 작성
   - 실거래 수익률 vs 모델 예측 수익률 비교
   - 오차가 큰 케이스 식별
   - outputs/live_vs_prediction_report.md 생성
4. run_daily_scheduler.py에 비교 리포트 스텝 추가
5. doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신
   - 신규 스크립트와 산출물 등록

주의사항:
- research 스키마 테이블 수정 시 schema.sql도 함께 갱신
- 실거래 데이터는 민감하므로 리포트에 금액보다 비율/순위 위주로 표현
- 작업 완료 후 doc/modules/Lee_trader_backTest/CONTEXT.md 갱신
```

---

## 4-C Walkforward REJECTED 자동 대응

```
지금부터 과제 4-C (walkforward REJECTED 자동 대응)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_backTest/CONTEXT.md
- python/build_walkforward_acceptance.py
- python/build_operational_buy_gate.py
- python/notifier.py (4-B 완료 후 존재)
- outputs/walkforward_acceptance.json (있으면)
- outputs/operational_buy_gate.json (있으면)

배경:
- walkforward REJECTED 시 운영자가 화면을 직접 확인해야만 알 수 있음
- REJECTED 상태에서도 BUY gate가 자동으로 HOLD로 전환되지 않아 수동 개입 필요
- REJECTED 원인 분석도 수동으로 해야 함

작업 목표:
1. build_walkforward_acceptance.py 분석
   - REJECTED 판정 조건 및 결과 저장 방식 파악
2. REJECTED 시 operational_buy_gate 자동 HOLD 전환 연동
   - build_operational_buy_gate.py에서 walkforward_acceptance.json 읽어
   - REJECTED이면 gate 상태를 HOLD로 강제 설정
3. python/build_walkforward_rejection_report.py 신규 작성
   - REJECTED 원인 자동 분석 (어떤 지표에서 기준 미달인지)
   - outputs/walkforward_rejection_report.md 생성
4. REJECTED 시 CRITICAL 알림 전송 (notifier.py 연동)
5. 운영자 OVERRIDE CLI 인터페이스 구현
   - python/ops_override.py 신규 작성
   - 사용법: python ops_override.py --action=allow-buy --reason="수동 확인 완료"
   - OVERRIDE 이력을 outputs/ops_override_log.json에 기록

주의사항:
- OVERRIDE는 반드시 --reason 인자가 있어야 실행 (이력 추적 목적)
- 4-B (notifier.py) 완료 후 진행 권장
- 작업 완료 후 doc/modules/Lee_trader_ai/OPERATIONS.md 갱신
```

---

## 5-B DART 재무 지연 대응

```
지금부터 과제 5-B (DART 재무 지연 대응)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- python/fetch_fundamentals_dart.py
- python/quality_builder.py
- python/score_kpi_monitor.py
- data/fundamentals.csv (헤더 + 상위 5행 확인)
- data/quality.csv (헤더 + 상위 5행 확인)

배경:
- DART 재무 데이터는 분기 공시 이후 수집되지만 공시 날짜가 fundamentals.csv에 없음
- 오래된 재무 데이터 기반으로 quality_score가 계산되어도 운영자가 알 수 없음
- 특히 결산 후 수개월이 지난 데이터를 최신처럼 사용하는 위험

작업 목표:
1. fetch_fundamentals_dart.py에 데이터 수집 날짜 기록
   - fundamentals.csv에 data_as_of 컬럼 추가 (공시 기준일)
   - collected_at 컬럼 추가 (수집 시각)
2. quality_builder.py에 staleness 패널티 로직 추가
   - data_as_of 기준으로 경과 일수 계산
   - 경과 180일 초과: quality_score_confidence를 0.7 이하로 낮춤
   - 경과 360일 초과: quality_score_confidence를 0.5 이하로 낮춤
   - staleness_days 컬럼을 quality.csv에 추가
3. score_kpi_monitor.py에 staleness 경고 추가
   - 전체 유니버스 중 staleness_days > 180인 종목 비율이 30% 초과 시 경고
4. notifier.py 알림 연동 (4-B 완료 후)
5. doc/modules/Lee_trader_ai/CONTEXT.md 갱신

주의사항:
- fundamentals.csv 컬럼 추가 시 기존 파이프라인이 오류 없이 돌아가는지 확인
- staleness 패널티는 quality_score_confidence만 조정 (quality_score 자체는 건드리지 않음)
- 작업 완료 후 doc/modules/Lee_trader_ai/CONTEXT.md 갱신
```

---

---

# 🟢 3개월 이상 과제

---

## 3-C 동적 포지션 사이징 고도화

```
지금부터 과제 3-C (동적 포지션 사이징 고도화)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- python/build_trade_intents.py
- python/submit_live_orders.py
- python/common_live_risk_guard.py
- python/run_paper_trading_ledger.py
- data/ranking_final.csv (헤더 확인)

배경:
- 현재 confidence_score 55/70/85 구간별 단순 비중 규칙 사용
- 포트폴리오 전체 변동성 관리 없이 개별 종목 비중만 조정
- 종목 수가 많아질수록 전체 포트폴리오 리스크가 예상보다 커질 수 있음

작업 목표:
1. 현재 사이징 로직 분석 (build_trade_intents.py)
   - confidence 구간별 비중 규칙 파악
   - 포트폴리오 전체 비중 합산 방식 파악
2. 변동성 역비례 사이징 구현 (Vol Parity 방식)
   - 각 종목의 vol_20 기반으로 비중 역산
   - 비중 = (1 / vol_20) / sum(1 / vol_20 for all candidates)
   - confidence_score가 낮을수록 계산된 비중에 추가 축소
3. 포트폴리오 volatility budget 파라미터 추가
   - MAX_PORTFOLIO_VOL=0.15 (연간 15% 변동성 한도, 환경변수)
   - 합산 포트폴리오 변동성이 한도 초과 시 전체 비중 스케일다운
4. paper trading에서 기존 사이징 vs 신규 사이징 shadow 비교
   - 1개월 이상 비교 후 신규 사이징 전환 결정
5. .env.example 환경변수 추가

주의사항:
- AUTO_TRADE_EXECUTE=0 상태로 작업
- 사이징 변경은 실계좌 적용 전 paper trading 1개월 이상 검증 필수
- 3-A (BUY 자동화), 3-B (부분 체결) 완료 후 진행 권장
- 작업 완료 후 doc/modules/Lee_trader_ai/OPERATIONS.md 갱신
```

---

## 5-C 장중 이벤트 선택적 반영 (옵션)

```
지금부터 과제 5-C (장중 이벤트 선택적 반영)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/FLOW.md
- python/fetch_fundamentals_dart.py
- python/run_intraday_refresh.py
- python/build_operational_buy_gate.py
- config/trading_calendar_kr.json

배경:
- 현재 운영 원칙: 전일 종가 기준 고정 (운영 안정성 우선)
- 이 원칙은 유지하되, 장중 중요 공시 발생 시 해당 종목만 선택적으로 BLOCK 처리 필요
- 예: 유상증자 공시, 감사의견 거절, 관리종목 지정 등

작업 목표:
1. DART OpenAPI 실시간 공시 엔드포인트 검토
   - 장중 공시 조회 가능한 API 확인 (DART Open API 문서 참조)
   - 호출 빈도 및 rate limit 파악
2. python/intraday_event_guard.py 신규 작성
   - 주요 공시 유형 정의 (BLOCK 유발 공시 리스트)
   - DART API 호출 → 공시 파싱 → BLOCK 대상 종목 추출
   - outputs/intraday_event_blocks.json 생성
3. run_intraday_refresh.py에 이벤트 가드 스텝 추가
4. build_operational_buy_gate.py에서 intraday_event_blocks.json 읽어
   - BLOCK 대상 종목의 buy_eligibility를 BLOCK으로 강제 설정
   - 차단 사유: intraday_event_block
5. .env.example 환경변수 추가
   - INTRADAY_EVENT_GUARD_ENABLED=0 (기본 비활성화)
   - DART_API_KEY=

주의사항:
- INTRADAY_EVENT_GUARD_ENABLED 기본값은 0 (전일 종가 원칙 유지)
- DART API 키 발급 필요 (https://opendart.fss.or.kr)
- 장중 호출 빈도는 30분 1회 이하 권장 (rate limit 여유 확보)
- 작업 완료 후 doc/modules/Lee_trader_ai/FLOW.md 갱신
```

---

## 6-A RULE-AI 통합 성과 대시보드

```
지금부터 과제 6-A (RULE-AI 통합 성과 대시보드)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/FILE_INDEX.md
- doc/modules/Lee_trader_rule/CONTEXT.md
- node/index.js
- node/public/ops-unified-nav.js
- node/public/ops-readiness.html (구조 참고)
- schema.sql (trades, daily_ranking 테이블 확인)

배경:
- AI 경로와 RULE 경로의 성과를 나란히 비교하는 화면이 없음
- 운영자가 어느 경로가 더 좋은 성과를 내는지 파악하기 어려움

작업 목표:
1. DB에서 AI / RULE 성과 데이터 집계 쿼리 설계
   - 집계 지표: 누적 수익률, 최대 낙폭(MDD), 승률, 평균 보유 기간
   - 기간별 필터: 1개월 / 3개월 / 전체
2. node/index.js에 API 라우트 추가
   - GET /api/performance-comparison?period=1m|3m|all
   - AI 성과 / RULE 성과 / 비교 데이터를 JSON으로 반환
3. node/public/comparison.html 신규 작성
   - 누적 수익률 비교 차트 (Chart.js)
   - MDD / 승률 / 평균 보유 기간 비교 카드
4. node/public/comparison.js 신규 작성
   - API 호출 및 차트 렌더링
5. ops-unified-nav.js 네비게이션에 성과 비교 화면 추가
6. doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신

주의사항:
- AI 경로와 RULE 경로 계좌/앱키가 분리되어 있으므로 데이터 혼용 주의
- 4-A (실거래 outcome 피드백) 완료 후 진행하면 데이터가 더 풍부함
- 작업 완료 후 doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신
```

---

## 6-B 종목 상세 화면 예측 이력 추가

```
지금부터 과제 6-B (종목 상세 화면 예측 이력 추가)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/FILE_INDEX.md
- node/index.js
- node/public/detail.html
- python/sync_web_display_data.py
- schema.sql (research.prediction_history 테이블 확인)

배경:
- 종목 상세 화면에서 오늘의 예측값만 볼 수 있고 과거 예측 이력이 없음
- 예측이 일관성 있게 나오는 종목인지, 아니면 매일 바뀌는지 알 수 없음
- 예측 vs 실제 수익률 비교도 불가

작업 목표:
1. research.prediction_history 테이블 데이터 구조 확인
2. node/index.js /api/stocks/:code 응답에 prediction_history 배열 추가
   - 최근 30일 예측 이력 반환
   - 컬럼: date, pred_return_60d, pred_return_ensemble (있으면), prob_top20_60d, final_score
3. node/public/detail.html에 예측 이력 섹션 추가
   - 예측 vs 실제 수익률 mini 차트 (Chart.js)
   - 예측 일관성 지표 (표준편차 등)
4. sync_web_display_data.py에 prediction_history payload 반영 검토
   - API가 직접 DB에서 읽어도 되면 생략 가능

주의사항:
- 1-B (앙상블) 완료 후 진행하면 pred_return_ensemble 컬럼 활용 가능
- 상세 화면 수정 시 기존 차트/카드 렌더링이 깨지지 않도록 주의
- 작업 완료 후 doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신
```

---

## 6-C 모바일 대응

```
지금부터 과제 6-C (운영자 화면 모바일 대응)를 진행합니다.

작업 시작 전 아래 문서와 파일을 먼저 읽어주세요:
- doc/modules/Lee_trader_ai/FILE_INDEX.md
- node/public/ops-readiness.html
- node/public/live-auto-trading.html
- node/public/ops-unified-nav.js
- node/public/ranking.html

배경:
- 운영자 화면이 desktop 위주로 설계되어 모바일에서 레이아웃이 깨짐
- 핵심 판정(READINESS / BUY GATE / KPI)을 외부에서 빠르게 확인하기 어려움

작업 목표:
1. 각 화면의 모바일 breakpoint 분석
   - 현재 CSS에서 미디어 쿼리 사용 여부 확인
   - 모바일(375px), 태블릿(768px) 기준으로 문제 지점 파악
2. ops-readiness.html 모바일 대응
   - 핵심 판정 카드 (READINESS / BUY GATE / KPI)를 모바일 상단에 고정 노출
   - 상세 테이블은 아코디언(펼치기/접기)으로 처리
3. live-auto-trading.html 모바일 대응
   - 주문 미리보기 테이블을 모바일에서 카드 형태로 전환
4. ranking.html 모바일 대응
   - 컬럼 수가 많은 테이블은 핵심 컬럼만 모바일에서 표시
   - 전체 컬럼은 가로 스크롤 또는 상세 팝업으로 처리
5. ops-unified-nav.js 네비게이션 모바일 햄버거 메뉴 적용

주의사항:
- 기존 desktop 레이아웃이 깨지면 안 됨 (미디어 쿼리로만 분기)
- 모바일 실기기 또는 Chrome DevTools로 테스트
- 작업 완료 후 doc/modules/Lee_trader_ai/FILE_INDEX.md 갱신
```

---

---

## 작업 완료 공통 체크리스트

> 모든 과제 완료 시 아래를 확인하세요.

```
방금 완료한 작업의 마무리 체크를 진행해주세요.

1. 코드 변경 단위 실행 검증 완료 여부 확인
2. 관련 산출물(CSV/JSON/MD) 정상 생성 확인
3. 영향받는 모듈 문서 갱신 여부 확인
   - doc/modules/ 아래 CONTEXT.md / FILE_INDEX.md / OPERATIONS.md
4. 환경변수 추가 시 .env.example 갱신 여부 확인
5. AUTO_TRADE_EXECUTE=0 상태 유지 확인
6. 실주문 관련 변경 시 paper trading 환경 검증 계획 확인
7. 이번 작업에서 변경된 파일 목록 요약해줘
8. 다음 권장 과제가 무엇인지 알려줘
```

---

*Lee Trader AI 개선 Claude Code 프롬프트 플레이북 v1.0 | 2026-05-06*
