# 실서버 자동매매 운영/모니터링 체크리스트

작성일: 2026-04-30  
목적: 실제 서버에서 자동매매가 실행되는 동안 매일 확인할 운영 항목 정의

---

## 1. 매일 장 시작 전 체크리스트

시간: 08:30~09:00

```text
[ ] Docker 컨테이너 전체 기동 상태 확인
[ ] postgres 정상 확인
[ ] node-api 정상 확인
[ ] scheduler 정상 확인
[ ] scheduler-auto-buy 정상 확인
[ ] scheduler-live-account-sync 정상 확인
[ ] scheduler-rule-before-open 정상 확인
[ ] KIS API 인증 정상 확인
[ ] 계좌 잔고 조회 정상 확인
[ ] 보유종목 동기화 정상 확인
[ ] 전일 종가 배치 성공 확인
[ ] market_status 최신 날짜 확인
[ ] ranking/latest payload 최신 날짜 확인
[ ] AI/RULE 주문 예정 후보 확인
[ ] GLOBAL_KILL_SWITCH=0 여부 확인
[ ] RULE_KILL_SWITCH=0 여부 확인
```

---

## 2. 장 시작 직후 체크리스트

시간: 09:00~09:40

```text
[ ] RULE before-open cycle 성공 여부 확인
[ ] RULE after-open cycle 실행 여부 확인
[ ] AI auto-buy scheduler 실행 여부 확인
[ ] 주문 preview 생성 여부 확인
[ ] BUY 차단 사유 확인
[ ] 실제 제출 주문 수 확인
[ ] 체결 여부 확인
[ ] 미체결 주문 존재 여부 확인
[ ] 같은 종목 중복 주문 여부 확인
[ ] 전일 종가 대비 체결가 괴리율 확인
[ ] 체결 후 holdings sync 성공 여부 확인
[ ] 체결 후 fills sync 성공 여부 확인
```

---

## 3. 장중 체크리스트

시간: 10:00, 14:00

```text
[ ] scheduler-live-account-sync 성공 여부
[ ] 보유수량과 웹 표시 수량 일치 여부
[ ] 체결내역과 ledger 일치 여부
[ ] 미체결 주문 방치 여부
[ ] 일일 손실 제한 접근 여부
[ ] API 오류 발생 여부
[ ] 로그에 exception 발생 여부
```

---

## 4. 장 마감 후 체크리스트

시간: 16:00~18:30

```text
[ ] 종가 배치 성공 여부
[ ] ranking 최신화 여부
[ ] final_score 생성 여부
[ ] operational_buy_gate 생성 여부
[ ] portfolio/intents 생성 여부
[ ] live trade review 생성 여부
[ ] live KPI daily report 생성 여부
[ ] RULE after-close cycle 성공 여부
[ ] RULE signal/portfolio/preview 생성 여부
[ ] 웹 DB sync 성공 여부
[ ] 다음날 주문 후보 수 확인
[ ] BUY_ALLOWED/PILOT/WATCH/BLOCK 분포 확인
```

---

## 5. 주간 체크리스트

매주 금요일 장마감 후 또는 주말

```text
[ ] AI 실전 거래 수
[ ] RULE 실전 거래 수
[ ] AI 승률
[ ] RULE 승률
[ ] AI 평균 수익률
[ ] RULE 평균 수익률
[ ] benchmark 대비 초과수익
[ ] 최대낙폭
[ ] confidence grade별 성과
[ ] entry gap 구간별 성과
[ ] BUY 차단 사유 Top 10
[ ] 손절/청산 사유별 성과
[ ] 중복주문/동기화오류 발생 여부
[ ] 다음 주 매수한도 조정 여부
```

---

## 6. 장애 대응 기준

## 6.1 즉시 신규매수 중지해야 하는 경우

```text
보유잔고 조회 실패
체결내역 동기화 실패
계좌 평가금액 이상치
오늘 주문 성공 여부 불명확
동일 종목 중복 주문 발생
KIS API 인증 오류 반복
데이터 날짜가 전일 기준으로 최신이 아님
ranking/top20 파일 누락
market_status 누락
```

조치:

```bash
GLOBAL_KILL_SWITCH=1
AUTO_TRADE_ALLOW_BUY=0
RULE_KILL_SWITCH=1
```

---

## 6.2 전체 자동매매 중지해야 하는 경우

```text
일 손실 한도 초과
주간 손실 한도 초과
잘못된 계좌로 주문 시도
수량 계산 오류
주문 가격 이상치
API 장애로 주문 결과 불명확
DB sync 실패로 웹과 실제 계좌 불일치
```

조치:

```bash
AUTO_TRADE_EXECUTE=0
AUTO_TRADE_ALLOW_BUY=0
RULE_ORDER_SUBMIT_ENABLED=0
GLOBAL_KILL_SWITCH=1
RULE_KILL_SWITCH=1
```

---

## 7. 확인 명령어 예시

### 컨테이너 상태

```bash
docker compose ps
```

### 최근 로그

```bash
docker logs lee_trader_scheduler --tail 100
docker logs lee_trader_scheduler_auto_buy --tail 100
docker logs lee_trader_scheduler_live_account_sync --tail 100
docker logs lee_trader_scheduler_rule_before_open --tail 100
docker logs lee_trader_scheduler_rule_after_open --tail 100
docker logs lee_trader_scheduler_rule_after_close --tail 100
```

### outputs 상태 확인

```bash
ls -lt outputs | head -50
```

### 자동매수 상태 파일 확인

```bash
cat outputs/auto_ops_auto_buy_scheduler_status.json
cat outputs/auto_ops_live_account_sync_scheduler_status.json
cat outputs/rule_before_open_scheduler_status.json
cat outputs/rule_after_open_scheduler_status.json
cat outputs/rule_after_close_scheduler_status.json
```

---

## 8. 웹 UI에 추가하면 좋은 운영 카드

```text
오늘 종가배치 상태
오늘 AI 자동매수 상태
오늘 RULE 자동매수 상태
오늘 체결동기화 상태
오늘 BUY 후보 수
오늘 BUY 차단 수
오늘 실주문 제출 수
오늘 실제 체결 수
오늘 손익
주간 손익
현재 kill switch 상태
마지막 에러
```

---

## 9. 운영 판단 기준

### 정상

```text
모든 스케줄 정상
동기화 정상
BUY 차단 사유가 합리적
실제 주문과 preview 일치
웹 표시와 계좌 일치
```

### 주의

```text
스케줄 일부 지연
BUY 후보 과다
entry gap 차단 과다
confidence C/D 과다
동기화 지연
```

### 위험

```text
체결 내역 누락
보유수량 불일치
동일 종목 중복 주문
일 손실 한도 근접
API 오류 반복
```

### 중지

```text
주문 결과 불명확
계좌 불일치
동기화 실패 지속
손실 한도 초과
수량/가격 계산 오류
```
