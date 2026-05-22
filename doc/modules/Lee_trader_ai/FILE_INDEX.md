# Lee_trader_ai File Index

## 목적

AI 추천, 주문 preview, 실자동매매, 웹 동기화까지 포함한 핵심 파일 인덱스입니다.
AI 쪽은 배치, DB, 웹이 강하게 연결되어 있으므로 단일 파일만 보고 수정하지 않는 것을 원칙으로 합니다.

## 핵심 파일

| 파일 | 역할 | 수정 위험도 | 함께 확인할 파일 |
| --- | --- | --- | --- |
| `python/run_pipeline.py` | 일일 AI 파이프라인 실행 순서 제어 | 높음 | `feature_builder.py`, `model_predict.py`, `ranking_builder.py`, `sync_web_display_data.py` |
| `python/feature_builder.py` | 학습/예측 공통 feature 생성 | 높음 | `label_builder.py`, `quality_builder.py`, `model_train.py` |
| `python/model_train.py` | 모델 학습과 `model.pkl` 저장 | 높음 | `feature_builder.py`, `label_builder.py`, `production_config.py` |
| `python/model_predict.py` | 최신 feature 기준 예측 생성과 DB 적재 | 높음 | `model_train.py`, `ranking_builder.py`, `db.py` |
| `python/ranking_builder.py` | 최종 점수, 정렬 순위, 운영용 ranking 산출 | 매우 높음 | `scoring/final_score.py`, `score_explainer.py`, `sync_web_display_data.py` |
| `python/build_trade_intents.py` | AI 후보에서 실제 매매 의도 JSON 생성 | 높음 | `build_live_order_preview.py`, `build_operational_buy_gate.py`, `submit_live_orders.py` |
| `python/build_live_order_preview.py` | 주문 preview와 허용/차단 사유 생성 | 높음 | `build_trade_intents.py`, `submit_live_orders.py`, `common_live_risk_guard.py` |
| `python/submit_live_orders.py` | AI 실주문 제출과 실행 결과 저장 | 매우 높음 | `kis_client.py`, `sync_live_order_fills.py`, `run_live_auto_trade_cycle.py` |
| `python/run_live_auto_trade_cycle.py` | 실자동매매 전체 사이클 실행 | 매우 높음 | `build_trade_intents.py`, `submit_live_orders.py`, `sync_live_order_fills.py`, `sync_live_account_holdings.py` |
| `python/sync_live_account_holdings.py` | 일반 실계좌 보유/잔고 동기화 | 높음 | `kis_live_account.py`, `sync_web_display_data.py` |
| `python/sync_live_order_fills.py` | 실주문 체결 내역 동기화 | 높음 | `submit_live_orders.py`, `sync_live_trade_ledger.py` |
| `python/sync_live_trade_ledger.py` | 체결과 주문 결과를 거래 ledger로 정리 | 높음 | `sync_live_order_fills.py`, `build_live_trade_review.py` |
| `python/sync_web_display_data.py` | JSON 산출물과 DB payload 동기화 | 매우 높음 | `db.py`, `node/index.js`, `export_serving_payloads.py` |
| `python/db.py` | 공통 DB 연결과 bulk upsert 유틸리티 | 높음 | 대부분의 Python 배치 |
| `node/index.js` | 랭킹, 자동매매, 계좌 상태 API 제공 | 매우 높음 | `node/public/ranking.js`, `node/public/live-auto-trading.js`, `sync_web_display_data.py` |
| `node/public/ranking.js` | 메인 랭킹 화면 렌더링 | 중간 | `node/index.js`, `outputs/*ranking*` |
| `node/public/live-auto-trading.js` | AI 자동매매 운영 화면 렌더링 | 높음 | `node/index.js`, `outputs/order_requests_preview.json` |
| `node/public/manual-trading.js` | 수동 주문 요청/검토 UI | 중간 | `node/index.js`, `submit_live_orders.py` |

## 데이터 수집 파일

| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `python/fetch_fundamentals_dart.py` | DART 연간 재무 수집 | `data/fundamentals.csv` 출력 |
| `python/fetch_financials_dart_quarterly.py` | DART 분기 누적 재무 수집 (Phase 1) | `data/dart/financial_quarterly.csv` |
| `python/fetch_short_interest.py` | 공매도 잔고 비율 수집 — pykrx (C-1) | `data/short_interest.csv` |

## 재무 모멘텀 파이프라인 (Phase 1~8)

| 파일 | Phase | 역할 | 비고 |
| --- | --- | --- | --- |
| `python/fetch_financials_dart_quarterly.py` | 1 | DART 분기 수집 | 서버 실행 필요 |
| `python/build_financial_momentum_features.py` | 2 | true quarterly 역산·YoY/QoQ·구간 분류·점수 계산 | 서버 실행 필요 |
| `python/feature_builder.py` `merge_financial_momentum()` | 3 | point-in-time merge → features.csv | Phase 2 완료 후 자동 반영 |
| `python/ranking_builder.py` `attach_fin_momentum_shadow()` | 4/7 | shadow overlay (4) + live 반영 (7: `FINANCIAL_SCORE_OVERLAY_ENABLED=1`) | |
| `python/apply_execution_policy.py` `resolve_fin_momentum_gate()` | 8 | 수량 축소·BUY 차단 (`FINANCIAL_BUY_GATE_ENABLED=1`) | |

## 보조 파일

| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `python/build_operational_buy_gate.py` | 운영용 BUY gate 생성 | preview 차단 사유 해석에 중요 |
| `python/common_live_risk_guard.py` | 공통 live risk guard | BUY 제한 로직 추적용 |
| `python/build_live_trade_review.py` | 거래 리뷰 리포트 생성 | 운영 검토용 |
| `python/build_live_trade_review_summary.py` | 리뷰 요약 산출 | 웹/문서 반영용 |
| `python/export_serving_payloads.py` | 외부 노출용 serving payload 생성 | 배포 payload 확인용 |
| `node/public/ops-unified-nav.js` | 운영자 네비게이션 제어 | 화면 권한 정책 변경 시 확인 |

## 수정 원칙

- `ranking_builder.py` 변경은 점수, 정렬, UI 노출, 주문 후보에 동시에 영향을 줍니다.
- `submit_live_orders.py`와 `run_live_auto_trade_cycle.py`는 실주문 경로이므로 로그와 guard 중심으로만 변경합니다.
- 웹 payload 관련 수정은 `sync_web_display_data.py`, `node/index.js`, 프론트 JS를 같이 확인합니다.
- KR Rule 자동매매는 2026-05-22 기준 운영 중단 (`RULE_LIVE_ENABLED=0`). AI 경로는 독립적으로 유지됨.
