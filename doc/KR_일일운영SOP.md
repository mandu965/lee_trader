# Lee Trader KR 일일 운영 SOP

작성일: 2026-05-26  
범위: 한국 주식(KR) 일일 운영 절차  
목적: 운영자가 하루 단위로 무엇을 언제 확인하고, 문제가 생기면 어떤 순서로 복구할지 표준화한다.

---

## 1. 운영 원칙

- 운영 해석의 기준 점수는 `live_score`다.
- 종목 우선순위는 `live_rank`와 `rank_final` 기준으로 본다.
- 매수 허용 여부는 `operational_buy_gate.json`의 `overall_status`와 `daily_cycle_status`를 먼저 본다.
- 실주문은 `AUTO_TRADE_EXECUTE=1`과 `AUTO_TRADE_CONFIRM_TEXT=LIVE_ORDER`가 함께 맞을 때만 가능하다.
- BUY 주문은 추가로 `AUTO_TRADE_ALLOW_BUY=1`이 필요하다.

---

## 2. 하루 운영 타임라인

### 오전 장전/장초

- 자동매매 슬롯 준비 상태 확인
- 실계좌 연결 상태 확인
- 전일 종가 기준 산출물이 최신인지 확인

### 장중 12:00

- `scheduler-recovery` 장중 refresh 결과 확인
- 게이트 상태와 상위 후보 변화 확인

### 장중 09:30 / 10:00

- `scheduler-auto-buy` 실행 결과 확인
- 주문 프리뷰 생성 여부와 차단 사유 확인

### 종가 후 18:10

- 종가 배치 성공 여부 확인
- `ranking_final.csv`, `predictions.csv`, `market_status.csv` 날짜 정합성 확인
- 운영 리프레시와 화면 반영 상태 확인

### 저녁 18:00 / 이후

- 실계좌 동기화, 체결 동기화, 리뷰 산출물 확인

---

## 3. 아침 시작 체크

## 3-1. 먼저 볼 파일

- `outputs/auto_ops_auto_buy_scheduler_status.json`
- `outputs/auto_ops_live_account_sync_scheduler_status.json`
- `outputs/operational_buy_gate.json`
- `data/ranking_final.csv`
- `serving/daily_recommendations.json`

## 3-2. 확인 항목

- 산출물 기준일이 최신 거래일인지
- `overall_status`가 전일 대비 급락하지 않았는지
- `daily_cycle_status`가 `WAIT`인지, 실행 가능 상태인지
- `buy_now_count`, `watchlist_count`, `blocked_count` 분포가 비정상적으로 바뀌지 않았는지

---

## 4. 장중 운영 절차

## 4-1. 자동매매 슬롯 확인

확인 파일:

- `outputs/auto_ops_auto_buy_scheduler_status.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`

확인 항목:

- 슬롯 실행 성공 여부
- 프리뷰가 생성됐는지
- `policy_status=BLOCK` 비율이 과도하지 않은지
- BUY가 막혔다면 `blocked_reason`과 `overall_status`가 일치하는지

## 4-2. 장중 refresh 확인

확인 파일:

- `outputs/auto_ops_recovery_scheduler_status.json`
- `outputs/operational_buy_gate.json`

확인 항목:

- 장중 refresh가 성공했는지
- `overall_status`, `daily_cycle_status` 변동이 있었는지
- 상위 후보의 `live_score`, `confidence_score`, `action_note`가 과도하게 뒤집히지 않았는지

---

## 5. 종가 후 운영 절차

## 5-1. 종가 배치 성공 여부 확인

확인 파일:

- `data/market_status.csv`
- `data/features.csv`
- `data/predictions.csv`
- `data/ranking_final.csv`

확인 항목:

- 네 파일의 최신 날짜가 모두 같은지
- `ranking_final.csv`가 비어 있지 않은지
- 상위권 종목의 `live_score`, `live_rank`, `confidence_score`가 생성됐는지

## 5-2. 운영 리프레시 확인

확인 파일:

- `outputs/operational_buy_gate.json`
- `serving/daily_recommendations.json`
- `serving/buy_gate_status.json`
- `serving/model_portfolio.json`

확인 항목:

- 세 serving 파일의 `asof_date`가 종가 배치 날짜와 일치하는지
- `operational_buy_gate.json`이 생성됐는지
- 게이트 상태와 추천 payload가 논리적으로 맞는지

## 5-3. 화면 반영 확인

필요 시 실행:

```powershell
docker compose up -d --build node-api
```

확인 항목:

- API/화면에서 최신 날짜가 보이는지
- `live_score`, `rank_final`, 추천 종목 정보가 최신인지

---

## 6. 라이브 계좌/체결/리뷰 확인

## 6-1. 확인 파일

- `data/live_account_holdings.csv`
- `outputs/live_account_balance_summary.json`
- `outputs/live_trade_review_summary.json`
- `outputs/live_trade_consistency_report.json`

## 6-2. 확인 항목

- 잔고 동기화가 최신 시각으로 반영됐는지
- 체결 동기화 실패가 없는지
- 리뷰 요약이 생성됐는지
- 손익과 보유 비중이 비정상적으로 튀지 않았는지

---

## 7. 매일 봐야 하는 핵심 컬럼

상세 설명 문서:

- [doc/20260526_KR_운영핵심컬럼사전.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9A%B4%EC%98%81%ED%95%B5%EC%8B%AC%EC%BB%AC%EB%9F%BC%EC%82%AC%EC%A0%84.md)

실무상 우선순위:

1. 게이트: `overall_status`, `daily_cycle_status`
2. 랭킹: `live_rank`, `live_score`
3. 신뢰도: `confidence_score`, `confidence_grade`
4. 실행 메모: `action_note`, `risk_factor_1`
5. 주문 프리뷰: `policy_status`, `blocked_reason`, `qty`

---

## 8. 이상 징후 체크리스트

아래 중 하나라도 보이면 추가 점검이 필요하다.

- `ranking_final.csv` 최신 날짜가 전일에서 멈춤
- `operational_buy_gate.json` 미생성
- `overall_status`가 갑자기 `BLOCK` 또는 `HOLD`로 급락
- `buy_now_count=0`이 계속 반복
- `confidence_score` 상위권 평균이 급락
- `order_requests_preview.json` 미생성
- `policy_blocked_count`가 비정상적으로 급증
- `live_account_holdings.csv` 최신화 실패

---

## 9. 장애 대응 절차

## 9-1. 종가 배치 실패

실행:

```powershell
python python\run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

로컬 직접 실행:

```powershell
.venv\Scripts\python.exe python\run_manual_close_batch.py --local
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

확인:

- `market_status`, `features`, `predictions`, `ranking_final` 날짜 일치
- serving 파일 `asof_date` 일치
- 필요 시 `sync_web_display_data.py`, `node-api` 재기동

## 9-2. 운영 리프레시 불일치

증상:

- `ranking_final.csv`는 최신인데 serving 파일 날짜가 다름

대응:

1. `MARKET_DATE` 고정 실행 여부 점검
2. `python python/run_operational_refresh.py` 재실행
3. `serving/*.json` `asof_date` 재확인

## 9-3. 웹 반영 불일치

대응:

```powershell
python python\sync_web_display_data.py
docker compose up -d --build node-api
```

## 9-4. 주문 제출 실패

확인 순서:

1. `order_requests_preview.json` 생성 여부
2. `policy_status`와 `blocked_reason`
3. `AUTO_TRADE_EXECUTE`, `AUTO_TRADE_ALLOW_BUY`, `AUTO_TRADE_CONFIRM_TEXT`
4. 계좌/현금/주문가능수량
5. `operational_buy_gate.json` 상태

원칙:

- 주문 실패는 즉시 재시도보다 차단 사유 확인이 우선이다.

## 9-5. 체결/리뷰 동기화 실패

원칙:

- `CRITICAL` 성격으로 다룬다.
- `sync_live_order_fills.py`와 후속 리뷰 산출물 생성 여부를 같이 점검한다.

---

## 10. 수동 운영 명령 모음

## 10-1. 수동 close batch

```powershell
python python\run_manual_close_batch.py
```

## 10-2. 로컬 close batch

```powershell
.venv\Scripts\python.exe python\run_manual_close_batch.py --local
```

## 10-3. 운영 리프레시 재실행

```powershell
python python\run_operational_refresh.py
```

## 10-4. 웹 동기화

```powershell
python python\sync_web_display_data.py
```

## 10-5. Node API 재기동

```powershell
docker compose up -d --build node-api
```

## 10-6. 스케줄러 로그 확인

```powershell
docker compose logs -f scheduler
docker compose logs -f scheduler-recovery
docker compose logs -f scheduler-auto-buy
docker compose logs -f scheduler-live-account-sync
```

---

## 11. 매일 기록할 항목

- 실행 성공/실패
- 최신 기준일
- `overall_status`
- `daily_cycle_status`
- `buy_now_count`
- `watchlist_count`
- `blocked_count`
- 수동 검토 종목
- 실제 매수 여부
- 보류 이유
- 장애 발생 여부와 복구 조치

---

## 12. 주간 점검 항목

- `score_kpi_monitor.md` ALERT/WARNING 추세
- matured benchmark dates 증가 여부
- confidence calibration readiness 상태
- 상위 추천 종목의 섹터/테마 쏠림
- 추천 시점과 실제 체결 시점 가격 차이
- `walkforward_acceptance` 상태 변화

---

## 13. 관련 문서

- [doc/20260526_KR_장애대응런북.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9E%A5%EC%95%A0%EB%8C%80%EC%9D%91%EB%9F%B0%EB%B6%81.md)
- [doc/20260526_KR_일일점검기록템플릿.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9D%BC%EC%9D%BC%EC%A0%90%EA%B2%80%EA%B8%B0%EB%A1%9D%ED%85%9C%ED%94%8C%EB%A6%BF.md)
- [doc/20260525_KR_PRD.md](/d:/ai/lee_trader/doc/20260525_KR_PRD.md)
- [doc/20260525_KR_데이터카탈로그.md](/d:/ai/lee_trader/doc/20260525_KR_%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%B9%B4%ED%83%88%EB%A1%9C%EA%B7%B8.md)
- [doc/20260526_KR_운영핵심컬럼사전.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9A%B4%EC%98%81%ED%95%B5%EC%8B%AC%EC%BB%AC%EB%9F%BC%EC%82%AC%EC%A0%84.md)
- [doc/manual_close_batch.md](/d:/ai/lee_trader/doc/manual_close_batch.md)
- [doc/운영자 가이드.md](/d:/ai/lee_trader/doc/%EC%9A%B4%EC%98%81%EC%9E%90%20%EA%B0%80%EC%9D%B4%EB%93%9C.md)
