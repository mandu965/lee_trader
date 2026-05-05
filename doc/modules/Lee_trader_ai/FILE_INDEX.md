# Lee_trader_ai File Index

## 소스 파일 목록
| 파일 | 역할 | 수정 가능 여부 | 수정 시 주의사항 |
| --- | --- | --- | --- |
| `python/run_pipeline.py` | 일일 AI 파이프라인 실행 순서 정의 및 후속 리포트/동기화 제어 | 신중 수정 | `STEPS` 순서 변경 시 전체 산출물 의존성이 깨질 수 있음 |
| `python/model_train.py` | LightGBM 학습, `model.pkl` 저장 | 신중 수정 | feature/label 컬럼 계약과 호환되어야 함 |
| `python/model_predict.py` | 최신 feature snapshot 기반 예측 및 `public.predictions` 저장 | 신중 수정 | 출력 컬럼은 `PREDICTIONS_DB_COLUMNS`와 맞아야 함 |
| `python/ranking_builder.py` | 운영 점수 계산, `data/ranking_final.csv` 및 `public.daily_ranking` 저장 | 핵심 파일, 매우 신중 | 웹/API/DB가 다수 컬럼에 의존 |
| `python/strategy_core.py` | 실행 정책 조합 및 전략 평가 래퍼 | 신중 수정 | `apply_execution_policy.py` 반환 스키마에 의존 |
| `python/submit_live_orders.py` | 주문 프리뷰/실제 제출, KIS 호출, 실행 결과 저장 | 제한적 수정 권장 | 실주문 리스크가 있음 |
| `python/run_live_auto_trade_cycle.py` | 실거래 자동 사이클 오케스트레이션 | 신중 수정 | 제출, 체결, 리뷰, 웹 sync가 연쇄 실행됨 |
| `python/sync_web_display_data.py` | core table / JSON payload를 웹 DB로 동기화 | 신중 수정 | `research.app_payload_store` payload key 계약 유지 필요 |
| `python/db.py` | DB 연결, bulk replace, `research.dim_model_run` 생성 | 신중 수정 | 전 모듈 공통 DB 진입점 |
| `node/index.js` | AI 결과 조회 API 제공 | 신중 수정 | `/api/ranking`, `/api/top20`, `/api/trade-intents`, `/api/order-requests-preview`, `/api/live-account/*` 경로와 프런트 연동 |
| `node/public/ranking.js` | 랭킹 화면 클라이언트 | 수정 가능 | API 응답 필드명 변경 시 함께 수정 필요 |
| `node/public/live-auto-trading.js` | 라이브 자동매매 UI | 수정 가능 | payload key 및 API path 의존 |
| `node/public/manual-trading.js` | 수동매매/오더 요청 UI | 수정 가능 | 주문 preview/execution JSON 구조 의존 |

## 수정 기준
- 기본적으로 문서 외 AI 실운영 로직은 변경 영향이 크므로 수정 전 배치, 웹 API, DB 스키마를 같이 확인해야 한다.
- `ranking_builder.py`, `submit_live_orders.py`, `sync_web_display_data.py`는 “수정 가능”보다 “변경 영향 범위가 큰 핵심 파일”로 보는 것이 맞다.

## 확인 필요
- AI 모듈에 `python/run_operational_refresh.py`, `python/build_trade_intents.py`를 포함할지 여부는 코드 관계상 포함이 타당하지만, 저장소 내 명시적 모듈 정의는 없다.
