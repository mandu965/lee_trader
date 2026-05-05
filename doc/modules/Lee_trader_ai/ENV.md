# AI ENV

## Purpose

이 문서는 AI 선별/자동매매 모듈의 핵심 환경변수를 정리합니다.

## Core Data / DB

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `DATABASE_URL` | 없음 | 주 데이터베이스 연결 | pipeline, training, sync |
| `WEB_DATABASE_URL` | 없음 | web payload sync 대상 DB | sync_web_display_data |
| `USE_SQLITE_MIRROR` | `0` | SQLite mirror 사용 여부 | pipeline |
| `USE_SQLITE_FALLBACK_WRITES` | `0` | fallback write 사용 여부 | pipeline |

## Model / Ranking

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `MODEL_VERSION` | 프로젝트 기준 | 현재 운영 모델 버전 | train/predict |
| `HORIZON_DAYS` | 프로젝트 기준 | 예측 horizon | train/predict |
| `TOP_N` | 프로젝트 기준 | 후보 추출 상한 | ranking/recommendation |

## Live Auto Trading

| 변수명 | 기본값 | 설명 | 영향 범위 | 주의 |
| --- | --- | --- | --- | --- |
| `AUTO_TRADE_EXECUTE` | `0` | 실주문 실행 여부 | submit_live_orders | 기본값은 보수적으로 유지 |
| `AUTO_TRADE_ALLOW_BUY` | `0` | 신규 매수 허용 여부 | auto trading | BUY 차단 해제용 |
| `AUTO_TRADE_BUY_APPROVAL_REQUIRED` | `0` | 수동 승인 필요 여부 | auto trading | 운영 정책용 |
| `AUTO_TRADE_FORCE_RESUBMIT` | `0` | 재제출 강제 여부 | auto trading | 중복 제출 주의 |

## KIS Auth

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `KIS_BASE_URL` | 없음 | KIS base URL | AI/RULE 공통 |
| `KIS_APP_KEY` | 없음 | 공용 KIS 앱키 | AI 경로 |
| `KIS_APP_SECRET` | 없음 | 공용 KIS 시크릿 | AI 경로 |
| `KIS_CANO` | 없음 | 일반 실계좌 앞 8자리 | live sync / live orders |
| `KIS_ACNT_PRDT_CD` | 없음 | 일반 실계좌 상품코드 | live sync / live orders |
