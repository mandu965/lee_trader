# Lee Trader KR 운영 핵심 컬럼 사전

작성일: 2026-05-26
마지막 갱신: 2026-05-29 — confidence_score는 OOF calibration으로 산출 (기존 누설 calibration 해소)
범위: KR 운영 판단에 직접 쓰는 핵심 컬럼만 정리

> 점수 산출 체인 전체는 `doc/score_column_definitions.md`, 오늘 변경 이력은 `doc/20260529_시스템변경이력.md` 참조.

---

## 1. 목적

이 문서는 KR 운영자가 매일 실제로 봐야 하는 핵심 컬럼만 추려 설명한다.

다루는 범위:

- `data/ranking_final.csv`
- `outputs/operational_buy_gate.json`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `data/live_account_holdings.csv`

다루지 않는 범위:

- 연구용 비교 컬럼 전체
- 디버그용 중간 산출물 전체
- 미국 주식 관련 컬럼

---

## 2. 운영 해석 원칙

운영 판단은 아래 순서로 본다.

1. 게이트 상태: `overall_status`, `daily_cycle_status`
2. 랭킹 기준: `live_rank`, `live_score`
3. 신뢰도: `confidence_score`, `confidence_grade`
4. 실행 메모: `action_note`, `risk_factor_*`
5. 주문 가능성: `entry_eligible`, `policy_status`, `blocked_reason`

중요 원칙:

- `live_score`가 실제 운영 정렬 기준이다.
- `rank_final`은 사실상 `live_rank`와 같은 운영 표시 alias로 본다.
- `final_score_v2`, `final_score_v3`, `pred_score` 같은 비교용 컬럼은 운영 우선순위 판단에 직접 쓰지 않는다.

---

## 3. 랭킹 핵심 컬럼

대상 파일: `data/ranking_final.csv`

## 3-1. 가장 먼저 볼 컬럼

| 컬럼 | 의미 | 왜 중요한가 | 운영 해석 |
| --- | --- | --- | --- |
| `live_rank` | 실제 운영 순위 | 종목 우선순위의 최상위 기준 | 숫자가 낮을수록 우선 검토 |
| `rank_final` | 운영 표시 순위 | 화면/API 호환용 | `live_rank`와 같이 본다 |
| `live_score` | 실제 정렬 기준 점수 | 종목 간 비교의 실질 점수 | 높을수록 우선 |
| `live_score_source` | `live_score`가 어디서 왔는지 | 점수 기준 변경 여부 확인 | 현재는 보통 `final_score` |
| `confidence_score` | 운영 신뢰도 점수 | 진입/유지/비중 판단에 직접 영향 | 낮으면 보수적으로 해석 |
| `confidence_grade` | 신뢰도 등급 | 빠른 정성 판단 | 낮은 등급이면 주의 |
| `action_note` | 실행 메모 | 운영자에게 바로 행동 힌트 제공 | BUY/HOLD/WATCH 성격 해석 |

## 3-2. 점수 축 핵심 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `ret_score` | 예측 수익률 축 점수 | 상방 기대가 얼마나 강한지 |
| `prob_score` | 상위권 진입 확률 축 점수 | 예측 신호의 상대 우위 |
| `tech_score` | 기술적 흐름 점수 | 추세, RSI, 거래량 흐름 |
| `qual_score` | 재무 품질 점수 | 재무 건전성/수익성 품질 |
| `flow_score` | 수급 점수 | 외국인·기관 순매수 흐름 |
| `risk_penalty` | 리스크 감점 | 예상 손실위험이 크면 불리 |

핵심 해석:

- `live_score`는 위 점수 축과 `risk_penalty`가 합성된 결과다.
- `risk_penalty`가 높으면 다른 점수가 좋아도 실제 순위가 밀릴 수 있다.

## 3-3. 신뢰도 관련 핵심 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `confidence_score` | 최종 운영 신뢰도 수치 | 1차 신뢰 기준 |
| `confidence_label` | 신뢰도 라벨 | 빠른 구간 판별 |
| `confidence_grade` | 운영 등급 | 실행/보류 해석 보조 |
| `component_coverage_ratio` | 구성 요소 커버리지 | 데이터 결손 정도 확인 |
| `data_maturity_score` | 데이터 성숙도 | 이력 부족 여부 판단 |
| `model_reliability_score` | 모델 안정성 축 | 모델 신뢰 근거 |
| `signal_agreement_score` | 신호 일치도 | 여러 축이 같은 방향인지 |
| `regime_fitness_score` | 현재 시장 적합도 | 지금 레짐과 맞는지 |

실무 해석:

- `confidence_score`만 보지 말고 `confidence_reason`도 함께 본다.
- `component_coverage_ratio`가 낮거나 `data_maturity_score`가 낮으면 과신하면 안 된다.

## 3-4. 설명용 핵심 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `score_driver_1` | 가장 큰 점수 상승 요인 | 왜 상위권인지 한 줄 요약 |
| `score_driver_2` | 두 번째 상승 요인 | 보조 강점 |
| `score_driver_3` | 세 번째 상승 요인 | 보조 강점 |
| `risk_factor_1` | 가장 큰 위험 요인 | 왜 보수적으로 봐야 하는지 |
| `risk_factor_2` | 두 번째 위험 요인 | 보조 위험 |
| `top_positive_factor` | 최상위 긍정 팩터 | 설명 텍스트 보조 |
| `top_negative_factor` | 최상위 부정 팩터 | 설명 텍스트 보조 |
| `explain_text` | 요약 설명 | 화면/운영 메모용 |
| `score_explain_summary` | 설명 요약 | 빠른 브리핑용 |

## 3-5. 시장/상황 해석 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `regime` | 현재 시장 레짐 | bull / neutral / defensive |
| `regime_reason` | 레짐 판단 이유 | 왜 현재 가중치가 적용됐는지 |
| `market_up` | 시장 우호 여부 | 시장 전반 우호성 |
| `market_kospi_close` | KOSPI 종가 | 시장 수준 참고 |
| `market_kospi_ma20` | KOSPI 20일선 | 추세 참고 |
| `market_vol_5d` | 최근 변동성 | 과열/불안정 참고 |
| `market_foreign_5d` | 외국인 5일 수급 | 수급 우호성 참고 |

## 3-6. 실전에서 자주 함께 보는 조합

### 진입 검토 조합

- `live_rank`
- `live_score`
- `confidence_score`
- `action_note`
- `risk_factor_1`

### 과열 여부 점검 조합

- `ret_5d`
- `ret_10d`
- `rsi_14`
- `liquidity_score`
- `action_note`

### 수급 확인 조합

- `flow_foreign_net_5d`
- `flow_inst_net_5d`
- `flow_score`

---

## 4. 게이트 핵심 컬럼

대상 파일: `outputs/operational_buy_gate.json`

## 4-1. 가장 먼저 볼 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `overall_status` | 전체 매수 허용 상태 | `BUY_ALLOWED`, `PILOT`, `WATCH`, `HOLD`, `BLOCK` |
| `daily_cycle_status` | 당일 사이클 상태 | 오늘 바로 매수 가능한지/대기인지 |
| `primary_bucket` | 판단 기준 bucket | 현재 어떤 평가 bucket 기준인지 |
| `asof_date` | 기준일 | 당일 산출물 정합성 확인 |
| `generated_at` | 생성 시각 | 최신성 확인 |

## 4-2. 운영자 핵심 확인 포인트

| 경로 | 의미 | 운영 해석 |
| --- | --- | --- |
| `decisions[].benchmark.matured_dates_max` | 성숙 benchmark 수 | 너무 적으면 승격 보류 |
| `decisions[].confidence_v2.trusted_ratio_top20` | 상위 20종목 trusted 비율 | 낮으면 자동매수 확대에 불리 |
| `decisions[].walkforward_acceptance.status` | 워크포워드 판정 | `REJECTED`면 강한 보수 해석 |
| `decisions[].buyability.buy_now_count` | 바로 매수 가능한 수 | 0이면 실제 자동매수 어려움 |
| `decisions[].buyability.watchlist_count` | 관찰 대상 수 | 검토 대상 풀 |
| `decisions[].buyability.blocked_count` | 차단 종목 수 | 현재 차단 강도 |
| `decisions[].buyability.paper_only_count` | paper 수준 종목 수 | 실매수 전환 불가 후보 수 |

실무 해석:

- `overall_status` 하나만 보지 말고 `buy_now_count`와 `walkforward_acceptance.status`를 같이 본다.
- `WATCH`라도 `buy_now_count`가 0이면 사실상 즉시 매수는 어렵다.

---

## 5. 거래 의도 핵심 컬럼

대상 파일: `outputs/trade_intents.json`

## 5-1. 가장 먼저 볼 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `intent` 또는 `action` | 최종 행동 유형 | BUY / HOLD / TRIM / EXIT / REVIEW |
| `code` | 종목코드 | 대상 식별 |
| `name` | 종목명 | 대상 식별 |
| `ranking_rank` | 랭킹 기준 순위 | 왜 이 종목이 잡혔는지 |
| `final_score` | 점수 | 상대 우선순위 |
| `confidence_score` | 신뢰도 | 진입/유지 강도 |
| `target_weight` | 목표 비중 | 실제 주문 크기 판단 |
| `reason` 또는 `action_note` | 행동 이유 | 운영자가 즉시 이해할 근거 |

## 5-2. 보유 종목 판단 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `current_weight` | 현재 비중 | 유지/축소 기준 |
| `policy_cap_weight` | 정책상 허용 비중 | 초과 여부 확인 |
| `held_now` | 현재 보유 여부 | 신규/기보유 구분 |
| `entry_eligible` | 진입 가능 여부 | BUY 후보인지 여부 |
| `confidence_band` | 신뢰도 구간 | 비중 축소/확대 판단 |
| `position_guidance` | 비중 가이드 | 실행 메모 |

---

## 6. 주문 프리뷰 핵심 컬럼

대상 파일: `outputs/order_requests_preview.json`

## 6-1. 가장 먼저 볼 컬럼

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `side` | 주문 방향 | BUY / SELL |
| `code` | 종목코드 | 주문 대상 |
| `qty` | 주문 수량 | 실제 주문 크기 |
| `price` 또는 `price_ref` | 주문 기준 가격 | 시장가/기준가 판단 |
| `policy_status` | 정책 허용 상태 | ALLOW / BLOCK 등 |
| `blocked_reason` 또는 `policy_reason` | 제출 보류/차단 사유 | 왜 주문이 나가지 않았는지 |
| `confidence_score` | 신뢰도 | 주문 강도 판단 |
| `ranking_rank` | 순위 | 왜 선택됐는지 |
| `action_note` | 실행 메모 | 운영 해석 보조 |

## 6-2. 운영 시 꼭 보는 조합

- `side`
- `qty`
- `policy_status`
- `blocked_reason`
- `confidence_score`
- `ranking_rank`

핵심 해석:

- `policy_status=BLOCK`면 랭킹이 높아도 주문하면 안 된다.
- BUY 프리뷰는 `AUTO_TRADE_ALLOW_BUY`와 별개로 정책 차단 여부를 먼저 본다.
- `blocked_reason=trim_ratio_zero`는 장애성 차단이 아니다. 현재 비중과 목표 비중이 같거나 목표가 더 높아 계산상 매도수량이 0주인 TRIM 안전 스킵이다.

---

## 7. 보유 데이터 핵심 컬럼

대상 파일: `data/live_account_holdings.csv`

| 컬럼 | 의미 | 운영 해석 |
| --- | --- | --- |
| `code` | 종목코드 | 보유 식별 |
| `name` | 종목명 | 보유 식별 |
| `qty` | 수량 | 매도/추가매수 판단 |
| `avg_price` | 평균 매입가 | 손익 계산 기준 |
| `current_price` | 현재가 | 현재 평가 기준 |
| `eval_amount` | 평가금액 | 노출 규모 |
| `pnl_amount` | 손익금액 | 절대 손익 |
| `pnl_pct` | 손익률 | 상대 손익 |
| `weight` | 포트폴리오 비중 | 집중도 판단 |
| `status` | 보유 상태 | 동기화 상태 참고 |

---

## 8. 운영 우선순위별 추천 뷰

## 8-1. 매수 후보 빠른 점검용

- `live_rank`
- `name`
- `live_score`
- `confidence_score`
- `action_note`
- `risk_factor_1`

## 8-2. 게이트 점검용

- `overall_status`
- `daily_cycle_status`
- `buy_now_count`
- `watchlist_count`
- `blocked_count`
- `walkforward_acceptance.status`

## 8-3. 주문 전 최종 확인용

- `side`
- `code`
- `qty`
- `policy_status`
- `blocked_reason`
- `confidence_score`

---

## 9. 절대 혼용하면 안 되는 컬럼

| 컬럼 | 이유 |
| --- | --- |
| `final_score_v2` | 비교용 레퍼런스 |
| `final_score_v3` | 테마 overlay 비교 성격이 강함 |
| `pred_score` | 레거시 비교용 |
| `prob_score_raw` | 절대확률 표현으로 운영 상대점수와 다름 |
| `confidence_score` 단독 | 중요하지만 반드시 게이트/랭크와 같이 봐야 함 |

핵심 원칙:

- 실제 종목 선택은 `live_rank`, `live_score`, `overall_status` 중심으로 본다.
- `confidence_score`는 보조 강도 판단 축이지, 단독 진입 스위치로 쓰면 안 된다.

---

## 10. 함께 봐야 하는 문서

- [doc/20260525_KR_PRD.md](/d:/ai/lee_trader/doc/20260525_KR_PRD.md)
- [doc/20260525_KR_데이터카탈로그.md](/d:/ai/lee_trader/doc/20260525_KR_%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%B9%B4%ED%83%88%EB%A1%9C%EA%B7%B8.md)
- [doc/20260526_KR_일일운영SOP.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9D%BC%EC%9D%BC%EC%9A%B4%EC%98%81SOP.md)
- [doc/score_column_definitions.md](/d:/ai/lee_trader/doc/score_column_definitions.md)
- [doc/운영 준비 체크리스트.md](/d:/ai/lee_trader/doc/%EC%9A%B4%EC%98%81%20%EC%A4%80%EB%B9%84%20%EC%B2%B4%ED%81%AC%EB%A6%AC%EC%8A%A4%ED%8A%B8.md)
