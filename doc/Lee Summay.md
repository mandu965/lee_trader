# Lee Trader 프로젝트 백과사전

기준일: 2026-04-07 KST

이 문서는 Lee_trader 프로젝트를 코드 기준으로 설명하는 백과사전형 요약 문서입니다.  
목표는 "이 프로젝트가 무엇을 만들고, 어떤 데이터와 로직으로 움직이며, 운영자가 무엇을 확인해야 하는지"를 한 문서에서 이해할 수 있게 정리하는 것입니다.

주요 기준 파일
- [run_pipeline.py](/d:/ai/Lee_trader/python/run_pipeline.py)
- [run_operational_refresh.py](/d:/ai/Lee_trader/python/run_operational_refresh.py)
- [ranking_builder.py](/d:/ai/Lee_trader/python/ranking_builder.py)
- [index.js](/d:/ai/Lee_trader/node/index.js)
- [운영자 가이드.md](/d:/ai/Lee_trader/doc/운영자%20가이드.md)

---

## 1. 프로젝트 한 줄 정의

Lee_trader는 한국 주식 유니버스를 대상으로,

1. 시장/가격/재무 데이터를 수집하고  
2. feature와 label을 만들고  
3. 모델 예측과 점수식을 결합해  
4. 최종 랭킹, 절대 매수 판단, 운영 gate, 모의투자/실계좌 보조 자료를 생성하는  
리서치 + 운영 연결형 주식 추천 시스템입니다.

이 프로젝트는 단순히 "상위 종목을 뽑는 모델"이 아니라, 다음까지 포함합니다.

- 데이터 적재
- 점수 계산
- 워크포워드 검증
- 운영 readiness / buy gate 판단
- 추천 화면 / 운영자 화면 / 수동매매 화면
- 모의투자 추적
- 실계좌 잔고 조회와 주문 미리보기

---

## 2. 최상위 구조

루트 주요 디렉터리

- [python](/d:/ai/Lee_trader/python)
  - 데이터 파이프라인, 모델, 점수식, 운영 산출물 생성 스크립트
- [node](/d:/ai/Lee_trader/node)
  - API 서버와 웹 UI
- [data](/d:/ai/Lee_trader/data)
  - 핵심 CSV/JSON 데이터와 스냅샷/히스토리
- [outputs](/d:/ai/Lee_trader/outputs)
  - 운영 리포트, 검증 리포트, 운영 상태 JSON
- [serving](/d:/ai/Lee_trader/serving)
  - API/UI가 바로 읽는 serving payload
- [config](/d:/ai/Lee_trader/config)
  - 운영 설정
- [doc](/d:/ai/Lee_trader/doc)
  - 운영 문서, 개선 메모, 설계 문서
- [scripts](/d:/ai/Lee_trader/scripts)
  - PowerShell 실행 스크립트와 작업 스케줄러 등록 도구
- [postgres](/d:/ai/Lee_trader/postgres)
  - DB 초기화 관련 파일

---

## 3. 시스템 전체 흐름

### 연구/생성 파이프라인

핵심 엔트리
- [run_pipeline.py](/d:/ai/Lee_trader/python/run_pipeline.py)

주요 순서

1. fetch_market_data
2. fetch_top_universe
3. merge_universe
4. download_prices_kis
5. clean_prices
6. create_adjusted_prices
7. fetch_fundamentals_dart
8. quality_builder
9. feature_builder
10. label_builder
11. model_train
12. model_predict
13. ranking_builder

즉,
시장/유니버스 -> 가격/재무 -> 품질/특징량 -> 라벨 -> 학습 -> 예측 -> 최종 랭킹
구조입니다.

### 운영 후처리 파이프라인

핵심 엔트리
- [run_operational_refresh.py](/d:/ai/Lee_trader/python/run_operational_refresh.py)

주요 순서

1. run_theme_shadow_daily.py
2. buy_candidate_builder.py
3. build_buy_candidate_comparison.py
4. build_operational_buy_gate.py
5. run_paper_trading_ledger.py
6. sync_paper_trading_db.py
7. export_serving_payloads.py

즉,
랭킹 -> 운영 후보 -> 운영 gate -> 페이퍼 트레이딩 -> serving payload
구조입니다.

---

## 4. 프로젝트의 핵심 개념

### 4.1 final_score

- 오늘 후보들 사이의 상대 순위 점수
- 운영 추천 리스트에서 어느 종목이 위에 올라올지 정하는 중심 점수
- 현재 운영 직접 반영 축:
  - ret_score
  - prob_score
  - tech_score
  - qual_score
  - risk_penalty

### 4.2 buy_eligibility_score

- 종목 하나를 지금 사도 되는지 보는 절대 기준 점수
- final_score와 별개
- 결과 상태:
  - BUY_ALLOWED
  - WATCH
  - BLOCK

즉,
- final_score: "오늘 후보들 중 상대적으로 누가 더 좋은가"
- buy_eligibility: "이 종목을 지금 실제로 진입 가능한가"

### 4.3 confidence_score

- 모델 신뢰도 축
- 단순 표시용이 아니라 실제 비중 판단에도 연결
- 현재 운영 해석:
  - <55: 진입 금지
  - 55~70: 소액
  - 70~85: 표준
  - 85+: 확대 가능

### 4.4 operational_buy_gate

- 종목 개별 점수가 아니라 운영 전체 상태를 보는 gate
- 운영자 화면에서 BUY_ALLOWED / HOLD / BLOCK 등으로 해석되는 상위 운영 기준선

### 4.5 market regime

- 시장을 bull / neutral / defensive 식으로 해석
- 종목 점수 자체보다 운영 해석과 gate 보정에 중요

### 4.6 walkforward_acceptance

- 최근 워크포워드 결과가 운영 승격 기준을 만족하는지 보는 acceptance 판정
- 현재 REJECTED여도 스크립트 실패와는 다를 수 있음
- 즉, "계산 실패"가 아니라 "실제 판정 결과"일 수 있음

### 4.7 shadow

- production 점수를 바로 바꾸기 전에, 별도 실험 점수를 sidecar 형태로 같이 계산하는 방식
- 현재 대표 사례:
  - quality_risk_guard shadow

---

## 5. 데이터 계층 백과사전

## 5.1 원천 데이터

### data/universe.csv

- 생성: [fetch_top_universe.py](/d:/ai/Lee_trader/python/fetch_top_universe.py)
- 의미: 어떤 종목을 분석 대상으로 삼을지 결정하는 유니버스 마스터
- 주요 컬럼:
  - code
  - name
  - market
  - sector
- DB 테이블: stocks

### data/prices_daily_clean.csv

- 생성 흐름:
  - [download_prices_kis.py](/d:/ai/Lee_trader/python/download_prices_kis.py)
  - [clean_prices.py](/d:/ai/Lee_trader/python/clean_prices.py)
- 의미: 일봉 가격 정리본
- 주요 컬럼:
  - date
  - code
  - open
  - high
  - low
  - close
  - volume

### data/prices_daily_adjusted.csv

- 생성:
  - [create_adjusted_prices.py](/d:/ai/Lee_trader/python/create_adjusted_prices.py)
- 의미: 수정주가 반영 가격 데이터
- 모델과 라벨 계산에서 실제로 중요한 가격 기준

### data/market_status.csv

- 생성:
  - [fetch_market_data.py](/d:/ai/Lee_trader/python/fetch_market_data.py)
- 의미: 시장 상태 판단 데이터
- 주요 컬럼:
  - date
  - kospi_close
  - kospi_ma20
  - volatility_5d
  - foreign_net_5d
  - market_up
- DB 테이블: market_status

### data/fundamentals.csv

- 생성:
  - [fetch_fundamentals_dart.py](/d:/ai/Lee_trader/python/fetch_fundamentals_dart.py)
- 의미: 재무 원천 데이터
- 주요 컬럼:
  - roe
  - op_margin
  - net_margin
  - debt_ratio
  - ocf_to_assets
- DB 테이블: fundamentals

### flow_daily

- 문서:
  - [flow_daily_schema.md](/d:/ai/Lee_trader/doc/flow_daily_schema.md)
- 의미: 투자자 수급 데이터
- grain:
  - (date, code, investor_type)
- 현재 메인 랭킹 핵심축은 아니지만 확장 분석 축으로 중요

---

## 5.2 가공 데이터

### data/quality.csv

- 생성:
  - [quality_builder.py](/d:/ai/Lee_trader/python/quality_builder.py)
- 의미: 재무 데이터를 품질 점수로 변환한 결과
- 주요 컬럼:
  - quality_raw_score
  - quality_score
  - quality_factor_count
  - quality_missing_ratio
  - quality_score_confidence
- DB 테이블: quality

### data/features.csv

- 생성:
  - [feature_builder.py](/d:/ai/Lee_trader/python/feature_builder.py)
- 의미: 모델 입력 특징량
- 주요 컬럼 범주:
  - 수익률/모멘텀
  - 이동평균/추세
  - 변동성
  - RSI
  - 거래량/유동성
  - 품질 점수 결합
- DB 테이블: features

### data/labels.csv

- 생성:
  - [label_builder.py](/d:/ai/Lee_trader/python/label_builder.py)
- 의미: 모델 학습용 정답
- 주요 컬럼:
  - target_30d, target_60d, target_90d
  - target_mdd_30d, target_mdd_60d, target_mdd_90d
  - target_30d_top20, target_60d_top20, target_90d_top20
  - realized_return_30d, realized_return_60d, realized_return_90d
- DB 테이블: labels

---

## 5.3 모델 데이터

### data/model.pkl

- 생성:
  - [model_train.py](/d:/ai/Lee_trader/python/model_train.py)
- 의미: 학습된 모델 번들
- 내부 구성:
  - 회귀 모델
  - 분류 모델
  - 타깃 정보

### data/predictions.csv

- 생성:
  - [model_predict.py](/d:/ai/Lee_trader/python/model_predict.py)
- 의미: 최신 feature에 대한 모델 예측 결과
- 주요 컬럼:
  - pred_return_60d
  - pred_return_90d
  - pred_mdd_60d
  - pred_mdd_90d
  - prob_top20_60d
  - prob_top20_90d
  - score
- DB 테이블: predictions

---

## 5.4 운영 핵심 결과 데이터

### data/ranking_final.csv

- 생성:
  - [ranking_builder.py](/d:/ai/Lee_trader/python/ranking_builder.py)
- 의미: 최종 운영 랭킹
- 이 프로젝트에서 가장 중요한 산출물
- 주요 컬럼 범주:
  - 메타:
    - date, code, name, market, sector
  - 모델 출력:
    - pred_return_60d, pred_return_90d
    - pred_mdd_60d, pred_mdd_90d
    - prob_top20_60d, prob_top20_90d
  - 중간 점수:
    - ret_score
    - prob_score
    - qual_score
    - tech_score
    - safety_score
    - liquidity_score
    - valuation_score
    - risk_penalty
  - 최종 결과:
    - final_score_raw
    - final_score
    - final_score_v2
    - final_score_v3
    - live_score
    - live_rank
    - live_score_source
  - shadow:
    - shadow_quality_risk_guard_penalty
    - shadow_final_score_quality_risk_guard
    - shadow_rank_quality_risk_guard
- theme 최종 정책:
  - production 점수축은 final_score
  - final_score_v3는 theme overlay 비교용 연구 컬럼
  - dominant_theme는 설명축과 포트폴리오 cap 확인용
- DB 테이블: daily_ranking

### serving/daily_recommendations.json

- 생성:
  - [export_serving_payloads.py](/d:/ai/Lee_trader/python/export_serving_payloads.py)
- 의미: 프론트/UI가 직접 읽기 쉬운 종합 serving payload
- buy_eligibility, selection, 추천 설명 등이 함께 정리됨

### outputs/operational_buy_gate.json

- 생성:
  - [build_operational_buy_gate.py](/d:/ai/Lee_trader/python/build_operational_buy_gate.py)
- 의미: 운영 전체 상태 판정
- 운영자 화면의 BUY GATE 근거가 됨

### outputs/walkforward_acceptance.json

- 생성:
  - [build_walkforward_acceptance.py](/d:/ai/Lee_trader/python/build_walkforward_acceptance.py)
- 의미: 워크포워드 acceptance 결과

### data/score_kpi_monitor.json 또는 outputs/score_kpi_monitor.json

- 생성:
  - [score_kpi_monitor.py](/d:/ai/Lee_trader/python/score_kpi_monitor.py)
- 의미: 운영 KPI 요약

---

## 5.5 운영/매매 보조 데이터

### buy candidates 계열

- 관련 스크립트:
  - [buy_candidate_builder.py](/d:/ai/Lee_trader/python/buy_candidate_builder.py)
  - [build_buy_candidate_comparison.py](/d:/ai/Lee_trader/python/build_buy_candidate_comparison.py)
- 의미: 실질적인 매수 검토 후보 압축

### paper trading 계열

- 관련 스크립트:
  - [run_paper_trading_ledger.py](/d:/ai/Lee_trader/python/run_paper_trading_ledger.py)
  - [sync_paper_trading_db.py](/d:/ai/Lee_trader/python/sync_paper_trading_db.py)
- 산출물:
  - paper_trading_positions.csv
  - paper_trading_nav.csv
  - paper_trading_report.md

### live account 계열

- 관련 스크립트:
  - [sync_live_account_holdings.py](/d:/ai/Lee_trader/python/sync_live_account_holdings.py)
  - [build_live_order_preview.py](/d:/ai/Lee_trader/python/build_live_order_preview.py)
  - [kis_manual_order.py](/d:/ai/Lee_trader/python/kis_manual_order.py)
- 의미:
  - 실계좌 자동매수가 아니라
  - 잔고 조회 + 주문 미리보기 + 수동 주문 지원

---

## 5.6 연구/백테스트 데이터

대표 테이블

- research.dim_model_run
- research.prediction_history
- research.ranking_history
- research.backtest_outcome
- research.paper_trading_run
- research.paper_trading_position
- research.paper_trading_nav

주요 역할

- 워크포워드 검증
- 모델 버전 추적
- 랭킹 이력 축적
- 운영 전 승격 검토

---

## 6. 파이프라인 기준일 원칙

현재 프로젝트는 당일 장중 실시간 체결가 기반 시스템이 아닙니다.

운영 원칙

- 기본 기준일은 전일 종가
- 가격 수집 종료일도 기본적으로 전일
- market_status도 직전 완료 거래일까지만 사용

따라서,

- 2026-04-07 장중 실행
- 산출물 AS OF 2026-04-06

은 정상 동작입니다.

이 원칙은 다음 파일들에 반영되어 있습니다.

- [download_prices_kis.py](/d:/ai/Lee_trader/python/download_prices_kis.py)
- [fetch_market_data.py](/d:/ai/Lee_trader/python/fetch_market_data.py)
- [run_pipeline.py](/d:/ai/Lee_trader/python/run_pipeline.py)

---

## 7. 운영 흐름 백과사전

## 7.1 데일리 운영

권장 중심 명령

powershell
python python/run_operational_refresh.py


이 한 명령으로 보통 아래가 묶여 돌아갑니다.

- theme shadow
- buy candidates
- buy gate
- paper trading
- serving payload export

그 뒤 API 반영

powershell
docker compose up -d --build node-api


---

## 7.2 스케줄러 운영

주요 컨테이너

- scheduler
- scheduler-recovery

원칙

- 메인 scheduler는 16:00 마감 기준 정식 배치
- scheduler-recovery는 이름은 그대로지만 현재는 12:00 장중 refresh 역할
- 현재 자동운영은 `12:00 장중 + 16:00 마감` 2회 구조

---

## 7.3 운영자가 보는 핵심 판정

운영자 화면에서 중요한 것은 다음 4축입니다.

- READINESS
- BUY GATE
- KPI
- 오늘 후보

대표 해석

- WAIT
  - 아직 만기 표본이 부족하거나 준비 상태가 충분치 않음
- BLOCK
  - 운영 전체 기준선상 공격적 신규 진입이 부적절
- ALERT
  - KPI 경고가 존재
- WATCH
  - 종목은 볼 수 있지만 즉시 진입은 보수 해석

---

## 8. 웹/UI 구조 백과사전

핵심 서버
- [index.js](/d:/ai/Lee_trader/node/index.js)

정적 서빙
- express.static(path.join(__dirname, "public"))

즉, [node/public](/d:/ai/Lee_trader/node/public) 아래 HTML/JS가 실제 화면입니다.

### 주요 화면

- [index.html](/d:/ai/Lee_trader/node/public/index.html)
  - 메인 추천
- [ranking.html](/d:/ai/Lee_trader/node/public/ranking.html)
  - 연구/추천 종목 화면
- [ops-readiness.html](/d:/ai/Lee_trader/node/public/ops-readiness.html)
  - 운영자 화면
- [manual-trading.html](/d:/ai/Lee_trader/node/public/manual-trading.html)
  - 수동매매 화면
- [holdings.html](/d:/ai/Lee_trader/node/public/holdings.html)
  - 보유 종목 화면
- [paper-trading.html](/d:/ai/Lee_trader/node/public/paper-trading.html)
  - 모의투자 화면
- [detail.html](/d:/ai/Lee_trader/node/public/detail.html)
  - 종목 상세 화면

### 상세 화면의 현재 특징

- final_score와 buy_eligibility를 분리해서 읽게 만듦
- shadow guard 비교 박스가 있음
- 판단 문장에 shadow guard +N위 개선이 붙을 수 있음
- Shadow 개선, Guard 경고 칩과 보조 문구가 붙음
- 우측 상단에 상세 / 메인 / 연구·추천 / 운영자 / 수동매매 / 보유종목 / 모의투자 네비가 있음

---

## 9. 주요 API 백과사전

대표 API 예시

- /api/ranking
  - 추천 랭킹 목록
- /api/stocks/:code
  - 종목 상세 데이터
- /api/ops-readiness
  - 운영자 화면 요약
- /api/manual-trading/summary
  - 수동매매 요약
- /api/paper-trading/summary
  - 모의투자 요약
- /api/live-account/summary
  - 실계좌 요약
- /api/live-account/holdings
  - 실계좌 보유
- /api/live-account/order-preview
  - 주문 미리보기

### /api/stocks/:code의 의미

이 API는 상세 화면의 핵심입니다.

포함 데이터

- 가격 row 배열
- 최신 가격/RSI/변동성
- 모델 예측값
- 점수 분해
- buy_eligibility
- shadow_quality_risk_guard_*
- 설명 문구

즉, 상세 화면은 이 API 하나로 대부분의 종목 상태를 그립니다.

---

## 10. shadow / 실험 체계 백과사전

이 프로젝트는 production 점수를 바로 뒤집지 않고, shadow를 통해 먼저 관찰하는 방식이 강합니다.

### 대표 shadow 흐름

- theme shadow
- quality/risk guard shadow
- walkforward acceptance / validation
- top20 meaningfulness

### quality_risk_guard shadow

목적

- top20 < top50처럼 상위권 ordering이 깨지는 문제를 완화
- 저품질 / 고위험 종목이 top20을 과도하게 잠식하는지 보정

현재 상태

- ranking pipeline sidecar로 붙어 있음
- API/UI에도 비교용으로 노출됨
- 아직 production 승격 전

---

## 11. 워크포워드와 acceptance 백과사전

관련 파일

- [build_walk_forward_score_validation.py](/d:/ai/Lee_trader/python/build_walk_forward_score_validation.py)
- [build_walkforward_acceptance.py](/d:/ai/Lee_trader/python/build_walkforward_acceptance.py)
- [walk_forward_score_validation.md](/d:/ai/Lee_trader/outputs/walk_forward_score_validation.md)
- [walkforward_acceptance.md](/d:/ai/Lee_trader/outputs/walkforward_acceptance.md)

역할

- 최신 점수 체계가 실제로 상위 종목 선별력을 갖는지 확인
- top20 > top50 > universe ordering이 유지되는지 확인
- 평균 수익률, MDD, monotonicity 등을 운영 기준으로 점검

중요한 점

- REJECTED는 스크립트 실패와 다름
- 현재는 계산이 끝난 뒤 실제로 운영 기준 미달이란 뜻일 수 있음

---

## 12. confidence 백과사전

관련 파일

- [build_confidence_score_v2.py](/d:/ai/Lee_trader/python/build_confidence_score_v2.py)
- [build_confidence_calibration_map.py](/d:/ai/Lee_trader/python/build_confidence_calibration_map.py)
- [score_kpi_monitor.py](/d:/ai/Lee_trader/python/score_kpi_monitor.py)

핵심 개념

- raw confidence
- provisional calibration
- operational calibration

현재 상태

- snapshot 히스토리가 충분하지 않으면 insufficient_history가 정상
- 임시로 walkforward_provisional 기준을 보조 해석으로 사용
- 이 값은 final_score를 대체하지 않음

운영 해석 우선순위

1. final_score
2. raw confidence_score
3. provisional calibration note

---

## 13. 실계좌와 수동매매 백과사전

이 프로젝트는 현재 완전 자동 주문 시스템이 아닙니다.

실제 구조

- 시스템이 후보를 압축
- gate와 근거를 정리
- 운영자가 HTS/MTS에서 확인
- 필요 시 수동 주문

실계좌 1단계

- 잔고 조회
- 주문 미리보기
- 수동 주문

즉, 자동 execution보다 운영 보조와 통제에 더 가깝습니다.

---

## 14. 지금 이 프로젝트에서 특히 중요한 최근 정리

### 14.1 절대 매수 판단 분리

- ranking과 buy_eligibility를 분리
- 추천 상위라도 WATCH/BLOCK이면 바로 매수하지 않도록 해석 강화

### 14.2 confidence 비중 규칙 반영

- confidence가 실제 target weight / cap에 연결

### 14.3 전일 종가 기준 고정

- 장중 미완성 일봉이 운영 판단을 흔들지 않도록 기준일 통일

### 14.4 shadow 상세 화면 반영

- 상세 화면에서 production 판단과 shadow 개선 여지를 함께 읽을 수 있게 정리

### 14.5 상세 화면 복구

- escapeHtml() 누락으로 인한 JS 중단을 복구
- 차트와 우측 카드가 정상 렌더링되게 수정

---

## 15. 운영자가 제일 자주 봐야 하는 파일

우선순위 기준으로 보면 아래가 핵심입니다.

1. [ranking_final.csv](/d:/ai/Lee_trader/data/ranking_final.csv)
2. [predictions.csv](/d:/ai/Lee_trader/data/predictions.csv)
3. [daily_recommendations.json](/d:/ai/Lee_trader/serving/daily_recommendations.json)
4. [operational_buy_gate.json](/d:/ai/Lee_trader/outputs/operational_buy_gate.json)
5. [walkforward_acceptance.json](/d:/ai/Lee_trader/outputs/walkforward_acceptance.json)
6. [score_kpi_monitor.json](/d:/ai/Lee_trader/data/score_kpi_monitor.json)
7. [paper_trading_positions.csv](/d:/ai/Lee_trader/data/paper_trading_positions.csv)
8. [paper_trading_nav.csv](/d:/ai/Lee_trader/data/paper_trading_nav.csv)

이 8개를 보면

- 오늘 추천이 무엇인지
- 왜 BLOCK/WATCH인지
- 운영 전체 상태가 어떤지
- 모의 추적이 어떻게 되고 있는지

를 대부분 파악할 수 있습니다.

---

## 16. 파일별 한 줄 사전

- universe.csv: 어떤 종목을 볼지 정한 목록
- market_status.csv: 시장 상태 판단표
- fundamentals.csv: 재무 원천 데이터
- quality.csv: 재무 품질 점수
- features.csv: 모델 입력 상태표
- labels.csv: 모델 학습 정답
- model.pkl: 학습된 모델
- predictions.csv: 미래 수익/확률/낙폭 예측
- ranking_final.csv: 최종 운영 순위표
- daily_recommendations.json: 프론트용 추천 payload
- operational_buy_gate.json: 운영 전체 gate 판정
- walkforward_acceptance.json: 워크포워드 acceptance 결과
- score_kpi_monitor.json: 운영 KPI 요약

---

## 17. 결론

이 프로젝트는 단순 추천 엔진이 아니라,

- 데이터 수집
- 모델 예측
- 점수화
- 운영 gate
- shadow 검증
- 모의투자
- 수동매매/실계좌 보조
- 웹 운영 도구

가 하나의 흐름으로 이어진 운영형 리서치 플랫폼입니다.

프로젝트를 이해할 때 가장 중요한 축은 다음 셋입니다.

1. run_pipeline.py가 만드는 연구/랭킹 축
2. run_operational_refresh.py가 만드는 운영 산출물 축
3. node/index.js와 node/public/*가 보여주는 운영/UI 축

즉,
이 프로젝트의 본질은
예측 모델 자체보다
예측을 운영 가능한 형태로 해석하고 통제하는 체계
에 있습니다.
<<<<<<< HEAD



=======



>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
