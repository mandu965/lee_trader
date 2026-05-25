# Lee Trader KR 장애 대응 런북

작성일: 2026-05-26  
범위: 한국 주식(KR) 운영 장애 대응  
목적: 장애 유형별로 증상, 우선 확인 항목, 원인 후보, 즉시 조치, 복구 완료 기준을 표준화한다.

---

## 1. 사용 원칙

- 장애 대응은 “증상 확인 → 영향 범위 확인 → 즉시 조치 → 복구 확인” 순서로 진행한다.
- 무조건 재실행부터 하지 않는다. 먼저 어떤 산출물이 어디서 끊겼는지 확인한다.
- `CRITICAL` 성격 장애는 주문/체결/리뷰 계층 실패를 의미하므로 우선순위를 높게 둔다.
- 종가 배치와 장중 refresh는 분리해서 해석한다.

---

## 2. 공통 확인 파일

장애 유형과 무관하게 먼저 확인할 파일:

- `outputs/auto_ops_scheduler_status.json`
- `outputs/auto_ops_recovery_scheduler_status.json`
- `outputs/auto_ops_auto_buy_scheduler_status.json`
- `outputs/auto_ops_live_account_sync_scheduler_status.json`
- `outputs/operational_buy_gate.json`
- `data/ranking_final.csv`
- `serving/daily_recommendations.json`

공통 확인 포인트:

- 최신 기준일
- 마지막 성공 시각
- 실패 단계명
- 산출물 생성 여부
- serving `asof_date` 일치 여부

---

## 3. 장애 유형 A: 종가 배치 실패

## 증상

- `auto_ops_scheduler_status.json`이 `error`
- `ranking_final.csv` 최신 날짜가 갱신되지 않음
- `manual close batch` 로그에서 중단

## 우선 확인

- `data/market_status.csv`
- `data/features.csv`
- `data/predictions.csv`
- `data/ranking_final.csv`
- `outputs/auto_ops_scheduler_status.json`

## 원인 후보

- 가격/재무/수급 수집 실패
- feature 또는 prediction 단계 실패
- ranking 생성 실패
- close batch 후속 단계 실패

## 즉시 조치

```powershell
python python\run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

로컬 직접 실행:

```powershell
.venv\Scripts\python.exe python\run_manual_close_batch.py --local
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 복구 완료 기준

- `market_status`, `features`, `predictions`, `ranking_final` 최신 날짜 일치
- `[DONE] manual close batch completed market_date=...` 확인
- serving `asof_date` 일치

---

## 4. 장애 유형 B: 운영 리프레시 불일치

## 증상

- `ranking_final.csv`는 최신인데 `daily_recommendations.json`이나 `buy_gate_status.json` 날짜가 이전 날짜
- `operational_buy_gate.json` 미생성 또는 stale

## 우선 확인

- `serving/daily_recommendations.json`
- `serving/buy_gate_status.json`
- `serving/model_portfolio.json`
- `outputs/operational_buy_gate.json`

## 원인 후보

- `run_operational_refresh.py` 중간 실패
- `MARKET_DATE` 불일치
- 후속 payload export 단계 누락

## 즉시 조치

```powershell
python python\run_operational_refresh.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

필요 시 `MARKET_DATE`를 고정해 재실행한다.

## 복구 완료 기준

- serving 3개 파일의 `asof_date`가 종가 배치 기준일과 일치
- `operational_buy_gate.json` 최신 생성 시각 확인

---

## 5. 장애 유형 C: 웹/화면 반영 불일치

## 증상

- 로컬 CSV는 최신인데 화면/API는 이전 데이터
- ranking history 또는 trend 화면이 비정상

## 우선 확인

- `python/sync_web_display_data.py` 실행 로그
- `serving/*.json`
- `outputs/web_display_sync.json`

## 원인 후보

- web sync 실패
- ranking history fallback 경로 문제
- `node-api` 재기동 누락

## 즉시 조치

```powershell
python python\sync_web_display_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

강한 재적재가 필요할 때:

```powershell
python python\sync_web_display_data.py --reset-first
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 복구 완료 기준

- API/화면 최신 날짜 반영
- ranking trend/history 데이터 정상
- `web_display_sync.json` 성공 상태 확인

---

## 6. 장애 유형 D: 자동매매 슬롯 실패

## 증상

- `auto_ops_auto_buy_scheduler_status.json` 실패
- `trade_intents.json`, `order_requests_preview.json` 또는 `order_requests_execution.json` 미생성

## 우선 확인

- `outputs/auto_ops_auto_buy_scheduler_status.json`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`

## 원인 후보

- 운영 리프레시 실패
- 거래 의도 생성 실패
- 주문 프리뷰/실주문 단계 실패
- 환경변수 승인 조건 불일치

## 즉시 조치

확인 순서:

1. `trade_intents.json` 생성 여부
2. `order_requests_preview.json` 생성 여부
3. `policy_status`, `blocked_reason`
4. `AUTO_TRADE_EXECUTE`, `AUTO_TRADE_ALLOW_BUY`, `AUTO_TRADE_CONFIRM_TEXT`

필요 시:

```powershell
python python\run_operational_refresh.py --with-live-account --skip-live-preview
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\submit_live_orders.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 복구 완료 기준

- `trade_intents.json` 생성
- `order_requests_preview.json` 생성
- 실패 원인이 정책 차단인지 시스템 오류인지 분리 확인

---

## 7. 장애 유형 E: 주문 제출 실패

## 증상

- `submit_live_orders.py` 실패
- 주문 프리뷰는 있으나 execution 결과가 비정상
- `CRITICAL` 알림 발생

## 우선 확인

- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`

## 원인 후보

- 계좌/현금/주문 가능 수량 부족
- 정책 차단
- 확인 문구/환경변수 불일치
- 브로커 API 실패

## 즉시 조치

확인 순서:

1. `policy_status=BLOCK`인지 확인
2. `blocked_reason` 확인
3. `AUTO_TRADE_EXECUTE=1` 여부 확인
4. `AUTO_TRADE_CONFIRM_TEXT=LIVE_ORDER` 여부 확인
5. BUY라면 `AUTO_TRADE_ALLOW_BUY=1` 여부 확인

원칙:

- 정책 차단을 시스템 장애로 오판하지 않는다.
- 브로커 실패면 재시도보다 계좌 상태와 API 응답을 먼저 본다.

## 복구 완료 기준

- 정책 차단인지 시스템 장애인지 명확히 분류
- 실행 가능 시 preview/execution 아티팩트 정상 생성

---

## 8. 장애 유형 F: 라이브 계좌/체결 동기화 실패

## 증상

- `auto_ops_live_account_sync_scheduler_status.json` 실패
- `live_account_holdings.csv` stale
- 리뷰/consistency 산출물 미생성

## 우선 확인

- `outputs/auto_ops_live_account_sync_scheduler_status.json`
- `data/live_account_holdings.csv`
- `outputs/live_trade_review_summary.json`
- `outputs/live_trade_consistency_report.json`

## 원인 후보

- 계좌 조회 실패
- 체결 조회 실패
- 리뷰 후속 단계 실패

## 즉시 조치

```powershell
python python\sync_live_account_holdings.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\sync_live_order_fills.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

필요 시 강제 전체 체결 조회:

```powershell
.venv\Scripts\python.exe python\sync_live_order_fills.py --query-all --start-date YYYYMMDD --end-date YYYYMMDD
```

## 복구 완료 기준

- `live_account_holdings.csv` 최신화
- fills sync 성공
- review summary / consistency report 재생성

---

## 9. 장애 유형 G: 게이트 상태 급락

## 증상

- `overall_status`가 갑자기 `BLOCK` 또는 `HOLD`
- `buy_now_count=0` 급변

## 우선 확인

- `outputs/operational_buy_gate.json`
- `outputs/walkforward_acceptance.json`
- `outputs/score_kpi_monitor.md`
- `data/confidence_score_v2.json`

## 원인 후보

- walkforward 판정 악화
- confidence trusted ratio 저하
- benchmark 성숙도 부족
- 유동성/집중도 위험 증가

## 즉시 조치

- 우선 자동매수 확대를 멈춘다.
- `overall_status`만 보지 말고 `buy_now_count`, `walkforward_acceptance.status`, `trusted_ratio_top20`를 함께 본다.
- 필요 시 당일은 수동 검토 모드로 전환한다.

## 복구 완료 기준

- 급락 원인이 데이터 부족인지 실제 성과 악화인지 분리됨
- 운영자 결론 메모가 기록됨

---

## 10. 장애 유형 H: 배포 후 이상

## 증상

- 배포 직후 산출물 날짜 불일치
- scheduler 동작은 하나 화면/serving 반영이 어긋남

## 우선 확인

- `doc/git 배포 절차.md`
- `runtime_snapshot` 사용 여부
- 최신 로컬 산출물 기준 배포 여부

## 즉시 조치

필요 시 전체 재배포 순서:

1. `run_manual_close_batch.py`
2. `export_trades_csv.py`
3. `backup_*zip`
4. `export_git_release.py`
5. 배포용 git push
6. web sync 및 node-api 확인

## 복구 완료 기준

- 로컬 기준 산출물과 배포본 기준일 일치
- 화면/API 최신 반영

---

## 11. CRITICAL 대응 원칙

아래는 즉시 우선 확인한다.

- 주문 제출 실패
- 체결 동기화 실패
- 계좌 동기화 실패 후 리뷰 계층 연쇄 실패

대응 원칙:

- 원인 분류 전 자동 재시도를 반복하지 않는다.
- 실패 단계, exit code, 관련 산출물 생성 여부를 먼저 기록한다.
- 당일 운영 결론에 “실주문 보류/허용” 판단을 남긴다.

---

## 12. 복구 완료 체크리스트

- 최신 날짜 정합성 회복
- `operational_buy_gate.json` 정상 생성
- `daily_recommendations.json` 정상 생성
- `trade_intents.json` 정상 생성
- `order_requests_preview.json` 정상 생성
- `live_account_holdings.csv` 최신화
- 화면/API 최신 반영
- 점검 기록 템플릿에 장애와 조치 이력 기록

---

## 13. 관련 문서

- [doc/20260526_KR_일일운영SOP.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9D%BC%EC%9D%BC%EC%9A%B4%EC%98%81SOP.md)
- [doc/20260526_KR_일일점검기록템플릿.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9D%BC%EC%9D%BC%EC%A0%90%EA%B2%80%EA%B8%B0%EB%A1%9D%ED%85%9C%ED%94%8C%EB%A6%BF.md)
- [doc/20260526_KR_운영핵심컬럼사전.md](/d:/ai/lee_trader/doc/20260526_KR_%EC%9A%B4%EC%98%81%ED%95%B5%EC%8B%AC%EC%BB%AC%EB%9F%BC%EC%82%AC%EC%A0%84.md)
- [doc/manual_close_batch.md](/d:/ai/lee_trader/doc/manual_close_batch.md)
- [doc/git 배포 절차.md](/d:/ai/lee_trader/doc/git%20%EB%B0%B0%ED%8F%AC%20%EC%A0%88%EC%B0%A8.md)
