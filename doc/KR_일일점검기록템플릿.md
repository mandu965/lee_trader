# Lee Trader KR 일일 점검 기록 템플릿

작성일: 2026-05-26  
용도: KR 운영자가 매일 실행 결과와 판단 내용을 남기는 기록 템플릿

---

## 사용 방법

- 이 문서를 복사해서 날짜별 기록 파일로 사용한다.
- 파일명 예시: `doc/logs/2026-05-26_KR_일일점검기록.md`
- 숫자/상태/판단은 가능한 한 실제 산출물 기준으로 적는다.
- “괜찮아 보임” 같은 표현보다 파일명, 상태값, 수치, 보류 이유를 남긴다.

---

## 기본 정보

| 항목 | 기록 |
| --- | --- |
| 점검일 | `YYYY-MM-DD` |
| 점검 시작 시각 |  |
| 점검 종료 시각 |  |
| 점검자 |  |
| 기준 거래일 |  |
| 전체 코멘트 |  |

---

## 1. 종가 배치 상태

### 1-1. 실행 상태

| 항목 | 값 |
| --- | --- |
| 종가 배치 성공 여부 | `성공 / 실패` |
| 실행 방식 | `자동 / 수동 / local 수동` |
| 실패 시 실패 단계 |  |
| 실패 시 로그 위치 |  |

### 1-2. 날짜 정합성

| 파일 | 최신 날짜 | 정상 여부 |
| --- | --- | --- |
| `data/market_status.csv` |  |  |
| `data/features.csv` |  |  |
| `data/predictions.csv` |  |  |
| `data/ranking_final.csv` |  |  |

### 1-3. 메모

```text
예:
- 네 파일 최신 날짜 모두 2026-05-26로 일치
- ranking_final 생성 정상
```

---

## 2. 운영 리프레시 상태

### 2-1. serving 정합성

| 파일 | `asof_date` | 정상 여부 |
| --- | --- | --- |
| `serving/daily_recommendations.json` |  |  |
| `serving/buy_gate_status.json` |  |  |
| `serving/model_portfolio.json` |  |  |

### 2-2. 운영 게이트

| 항목 | 값 |
| --- | --- |
| `overall_status` |  |
| `daily_cycle_status` |  |
| `primary_bucket` |  |
| `buy_now_count` |  |
| `watchlist_count` |  |
| `blocked_count` |  |
| `paper_only_count` |  |

### 2-3. 게이트 해석

```text
예:
- overall_status=WATCH, daily_cycle_status=WAIT
- buy_now_count=0으로 즉시 자동매수는 어려움
```

---

## 3. 랭킹 상위 후보 점검

### 3-1. 상위 5개 핵심 체크

| code | name | live_rank | live_score | confidence_score | action_note | risk_factor_1 | 메모 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

### 3-2. 상위 후보 해석

```text
예:
- 상위 5개 중 2개는 confidence_score 80 이상
- 1개는 ret_5d 과열 + RSI 과열로 보류
```

---

## 4. 자동매매/주문 프리뷰 상태

### 4-1. 자동매매 슬롯 상태

| 항목 | 값 |
| --- | --- |
| `scheduler-auto-buy` 실행 여부 |  |
| 09:30 슬롯 결과 |  |
| 10:00 슬롯 결과 |  |
| `order_requests_preview.json` 생성 여부 |  |
| `order_requests_execution.json` 생성 여부 |  |

### 4-2. 주문 프리뷰 요약

| 항목 | 값 |
| --- | --- |
| BUY 후보 수 |  |
| SELL/TRIM 후보 수 |  |
| `policy_blocked_count` |  |
| 주요 차단 사유 |  |

### 4-3. 주문 판단 메모

```text
예:
- BUY 후보 2건 모두 policy_status=BLOCK
- blocked_reason은 liquidity / gate hold 조합
```

---

## 5. 라이브 계좌/체결/리뷰 상태

### 5-1. 계좌 상태

| 항목 | 값 |
| --- | --- |
| `data/live_account_holdings.csv` 최신화 여부 |  |
| 총 보유 종목 수 |  |
| 최대 비중 종목 |  |
| 현금/노출 관련 특이사항 |  |

### 5-2. 체결/리뷰 상태

| 항목 | 값 |
| --- | --- |
| 체결 동기화 성공 여부 |  |
| 리뷰 요약 생성 여부 |  |
| consistency report 생성 여부 |  |
| 특이 손익/오류 여부 |  |

---

## 6. KPI 및 게이트 심화 점검

| 항목 | 값 |
| --- | --- |
| `walkforward_acceptance.status` |  |
| `trusted_ratio_top20` |  |
| `matured_benchmark_dates_max` |  |
| `top20_mean_confidence_score` |  |
| KPI overall status |  |

### 해석 메모

```text
예:
- walkforward_acceptance=REJECTED 유지
- trusted_ratio_top20은 0.40으로 전일과 동일
```

---

## 7. 장애 및 조치 이력

| 시각 | 장애/이슈 | 영향 범위 | 조치 | 결과 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |

---

## 8. 수동 조치 기록

| 항목 | 실행 여부 | 명령어/조치 | 결과 |
| --- | --- | --- | --- |
| 수동 close batch |  |  |  |
| 운영 리프레시 재실행 |  |  |  |
| web sync 재실행 |  |  |  |
| node-api 재기동 |  |  |  |
| 기타 |  |  |  |

---

## 9. 최종 판단

### 9-1. 당일 결론

| 항목 | 결론 |
| --- | --- |
| 시스템 정상 여부 | `정상 / 주의 / 장애` |
| 자동매수 판단 | `허용 / 제한 / 보류` |
| 수동 검토 필요 여부 | `예 / 아니오` |
| 즉시 후속 조치 필요 여부 | `예 / 아니오` |

### 9-2. 운영자 결론 메모

```text
예:
- 시스템은 정상이나 게이트는 WATCH 유지
- 상위 후보 중 즉시 매수감은 없음
- 다음 확인 포인트는 matured benchmark dates 증가 여부
```

---

## 10. 다음 액션

| 우선순위 | 액션 | 담당 | 예정 시각/일자 |
| --- | --- | --- | --- |
| P1 |  |  |  |
| P2 |  |  |  |
| P3 |  |  |  |

---

## 11. 빠른 복사용 체크 문구

```text
[배치]
- market_status/features/predictions/ranking_final 날짜 일치 여부:

[게이트]
- overall_status:
- daily_cycle_status:
- buy_now/watchlist/blocked:

[상위 후보]
- top1:
- top2:
- top3:

[주문]
- preview 생성 여부:
- execution 생성 여부:
- blocked_reason 요약:

[계좌/리뷰]
- holdings 최신화:
- fills sync:
- review summary:

[결론]
- 당일 운영 상태:
- 자동매수 판단:
- 후속 액션:
```

---

## 12. 관련 문서

- [doc/20260526_KR_일일운영SOP.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9D%BC%EC%9D%BC%EC%9A%B4%EC%98%81SOP.md)
- [doc/20260526_KR_운영핵심컬럼사전.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9A%B4%EC%98%81%ED%95%B5%EC%8B%AC%EC%BB%AC%EB%9F%BC%EC%82%AC%EC%A0%84.md)
- [doc/20260525_KR_PRD.md](/d:/ai/lee_trader/doc/20260525_KR_PRD.md)
