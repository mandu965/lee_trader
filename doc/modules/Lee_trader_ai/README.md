# Lee_trader_ai

## Purpose

이 모듈은 AI 기반 선별, 랭킹, 자동매매 후보 해석과 실자동매매 운영 흐름을 정리합니다.

범위:
- 데이터 수집과 feature 생성
- 모델 학습/예측
- 최종 점수와 랭킹 생성
- AI 주문 preview / execution
- live 계좌/체결 동기화
- 운영 화면과 web payload 반영

## Main Files

- `python/run_pipeline.py`
- `python/model_train.py`
- `python/model_predict.py`
- `python/ranking_builder.py`
- `python/run_operational_refresh.py`
- `python/run_live_auto_trade_cycle.py`
- `python/submit_live_orders.py`
- `python/sync_live_account_holdings.py`
- `python/sync_live_order_fills.py`
- `python/sync_web_display_data.py`

## Feature Pipeline

| 스크립트 | 역할 | 출력 |
|---|---|---|
| `python/feature_builder.py` | 가격·기술·수급·재무 feature 통합 | `data/features.csv` |
| `python/quality_builder.py` | 재무 품질 점수 + A-3 YoY 성장률 | `data/quality.csv` |
| `python/fetch_fundamentals_dart.py` | DART 연간 재무 수집 | `data/fundamentals.csv` |
| `python/fetch_short_interest.py` | 공매도 잔고 비율 수집 (pykrx, C-1) | `data/short_interest.csv` |

## Financial Momentum (Phase 1~4 완료, 2026-05-15)

분기 재무 모멘텀 기능. 설계 문서: [FINANCIAL_MOMENTUM_DESIGN.md](FINANCIAL_MOMENTUM_DESIGN.md)

| 스크립트 | Phase | 역할 |
|---|---|---|
| `python/fetch_financials_dart_quarterly.py` | 1 | DART 분기 누적 재무 수집 |
| `python/build_financial_momentum_features.py` | 2 | true quarterly 역산, YoY/QoQ, Phase 분류, 점수 계산 |
| `python/feature_builder.py` (merge_financial_momentum) | 3 | point-in-time merge → features.csv 반영 |
| `python/ranking_builder.py` (attach_fin_momentum_shadow) | 4 | shadow ranking overlay (live_score 미영향) |

**실행 순서:**
```bash
python fetch_financials_dart_quarterly.py      # Phase 1: DART 수집
python build_financial_momentum_features.py    # Phase 2: feature 계산
python feature_builder.py                      # Phase 3: features.csv 반영
python ranking_builder.py                      # Phase 4: shadow_fin_rank 포함
```

**Shadow 출력 컬럼 (ranking_final.csv):**
- `shadow_fin_momentum_adj` — overlay 가감점 (ACCELERATING +5 ~ DECLINING -10)
- `shadow_fin_final_score` — 재무 모멘텀 반영 가상 점수
- `shadow_fin_rank` / `shadow_fin_rank_diff` — 가상 랭크 및 원래 랭크 대비 변화량
- `shadow_fin_hard_risk_triggered` — 실적 훼손 위험 종목 flag

## Main Outputs

- `data/predictions.csv`
- `data/ranking_final.csv`
- `data/dart/financial_quarterly.csv`
- `data/dart/financial_momentum_quarterly.csv`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`
- `outputs/live_trade_review_report.json`
- `serving/daily_recommendations.json`

## Read First

- [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/CONTEXT.md>)
- [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FLOW.md>)
- [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FILE_INDEX.md>)
- [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/ENV.md>)
- [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/OPERATIONS.md>)
