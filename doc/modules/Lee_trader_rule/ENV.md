# RULE ENV

## Purpose

이 문서는 RULE 자동매매 모듈에서 자주 쓰는 환경변수와 운영 영향 범위를 정리합니다.

## Core Runtime

| 변수명 | 기본값 | 설명 | 영향 범위 | 주의 |
| --- | --- | --- | --- | --- |
| `RULE_TRADING_RUN_MODE` | `paper` | `paper`, `pilot`, `live` 중 실행 모드 | after-close, before-open, after-open | 기본값은 안전하게 `paper` 유지 |
| `RULE_LIVE_ENABLED` | `0` | live/pilot 주문 경로 허용 플래그 | order submitter | `1`이어야 실주문 경로 가능 |
| `RULE_ORDER_SUBMIT_ENABLED` | `0` | 주문 제출 허용 플래그 | order submitter | `1`이어야 제출 가능 |
| `RULE_KILL_SWITCH` | `0` | RULE 전용 긴급 차단 | RULE buy guard | `1`이면 RULE 주문 차단 |
| `GLOBAL_KILL_SWITCH` | `0` | 공통 긴급 차단 | common live risk guard | RULE preview/block reason에도 영향 가능 |

## Time / Calendar

| 변수명 | 기본값 | 설명 | 영향 범위 | 주의 |
| --- | --- | --- | --- | --- |
| `RULE_BEFORE_OPEN_START_TIME` | `08:55` | 장전 주문 허용 시작 시각 | before-open cycle | 시간 밖이면 제출 중단 |
| `RULE_BEFORE_OPEN_END_TIME` | `09:30` | 장전 주문 허용 종료 시각 | before-open cycle | 시간 밖이면 제출 중단 |

관련 파일:
- [config/trading_calendar_kr.json](/d:/ai/lee_trader/config/trading_calendar_kr.json)

## Account / Auth

| 변수명 | 기본값 | 설명 | 영향 범위 | 주의 |
| --- | --- | --- | --- | --- |
| `KIS_RULE_CANO` | 없음 | RULE 계좌 앞 8자리 | RULE 계좌 조회/주문 | 민감정보, 문서에 값 기록 금지 |
| `KIS_RULE_ACNT_PRDT_CD` | 없음 | RULE 계좌 상품코드 | RULE 계좌 조회/주문 | 계좌번호와 조합 일치 필요 |
| `KIS_APP_RULE_KEY` | 없음 | RULE 전용 KIS 앱키 | RULE KISClient 인증 | 없으면 공용 키 fallback |
| `KIS_APP_RULE_SECRET` | 없음 | RULE 전용 KIS 시크릿 | RULE KISClient 인증 | 없으면 공용 키 fallback |
| `KIS_BASE_URL` | 없음 | KIS 실전/모의 URL | RULE/AI 공통 | 실전/모의 계좌와 일치 필요 |

## Order Limits

| 변수명 | 기본값 | 설명 | 영향 범위 | 주의 |
| --- | --- | --- | --- | --- |
| `RULE_MIN_ORDER_AMOUNT` | `100000` | 최소 주문금액 | RULE guard | 너무 작으면 차단 |
| `RULE_MAX_ORDER_AMOUNT` | `1000000` | 1건 최대 주문금액 | RULE guard | 초과 시 차단 |
| `RULE_MAX_DAILY_ORDER_AMOUNT` | 없음 | 일일 주문금액 상한 | RULE guard | 누적 초과 시 차단 |
| `RULE_PILOT_MAX_ORDER_AMOUNT` | 없음 | pilot 1건 추가 상한 | pilot preview/guard | pilot 소액 검증용 |
| `RULE_PILOT_MAX_ORDER_QTY` | 없음 | pilot 수량 상한 | pilot preview/guard | pilot 소량 검증용 |
| `RULE_BUY_LIMIT_BUFFER` | `0.01` | 매수 지정가 버퍼 (기준가 × (1 + 버퍼)). 변동성 낮은 종목에 적용 | order preview builder | 기본 1% 버퍼, 현행 동작 유지 |
| `RULE_BUY_LIMIT_BUFFER_HIGH_VOL` | `0.02` | 고변동성 종목 매수 버퍼. `vol_20` percentile > 70인 종목에 적용 | order preview builder | 미체결 방지용 확대 버퍼 |

## Portfolio / Exit

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `RULE_MAX_POSITIONS` | `5` | 최대 보유 종목 수 | portfolio manager |
| `RULE_MAX_POSITION_WEIGHT` | `0.15` | 종목당 최대 비중 | portfolio manager |
| `RULE_NEW_ENTRY_WEIGHT` | `0.05` | 신규 진입 목표 비중 | portfolio manager |
| `RULE_MIN_CASH_WEIGHT` | `0.20` | 최소 현금 비중 | portfolio manager |
| `RULE_MAX_SECTOR_WEIGHT` | `0.35` | 섹터 최대 비중 | portfolio manager |
| `RULE_COOLDOWN_DAYS` | `5` | 재진입 쿨다운 | portfolio manager |
| `RULE_ALLOW_ENTRY_SIGNAL` | `0` | `entry_signal`(완화 진입) 허용 플래그. `1`이어야 활성화. 기본 off(안전) | portfolio manager |
| `RULE_ENTRY_ALLOW_RATIO` | `0.6` | 완화 진입 허용 포지션 비율. 보유+예정 포지션 수 < `RULE_MAX_POSITIONS * 이 값`일 때만 `entry_signal` 허용 | portfolio manager |
| `RULE_MAX_HOLDING_DAYS` | `10` | 최대 보유일 | portfolio manager/backtest |
| `RULE_MAX_HOLDING_DAYS_DEFENSIVE` | `7` | `market_defensive_mode`일 때 단축 보유일. `RULE_MAX_HOLDING_DAYS`보다 짧게 설정 | portfolio manager |
| `RULE_PROFIT_TARGET_PCT` | `0.0` | 익절 목표 수익률. `0.0`이면 비활성(기본). `0.08` 설정 시 +8% 도달 → 30% 매도(reduce) | portfolio manager |
| `RULE_STOP_LOSS_PCT` | `0.05` | 손절 기준 | portfolio manager/backtest |
| `RULE_TRAILING_STOP_PCT` | `0.04` | trailing stop 기준 | portfolio manager/backtest |
| `RULE_TRAILING_STOP_MIN_PROFIT_PCT` | `0.03` | trailing stop 최소 이익 기준 | portfolio manager/backtest |
