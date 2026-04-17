# Source Analysis

## 목적

이 문서는 현재 코드 기준으로 랭킹 생성 파이프라인의 파일별 역할과 데이터 가치를 정리한다.

체크리스트:
- [x] 파일별 역할
- [x] 입력/출력
- [x] 추천 점수 직접 반영 여부
- [x] 데이터 가치 평가
- [x] 현재 구조의 강점
- [x] 현재 구조의 한계
- [x] 향후 개선 우선순위

## 핵심 구조 요약

현재 추천 점수 생성의 운영 기준은 [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)다.

핵심 흐름:
1. 시장/유니버스/가격/재무/특징 데이터 준비
2. `label_builder`로 학습 타깃 생성
3. `model_train`으로 모델 학습
4. `model_predict`로 예측값 생성
5. `ranking_builder`로 최종 점수와 랭킹 생성

## 파일별 역할

### 1. [`python/run_pipeline.py`](/d:/ai/Lee_trader/python/run_pipeline.py)

- 역할:
  전체 일일 파이프라인 실행기
- 입력:
  환경변수, 각 Python step 스크립트
- 출력:
  각 단계 실행 결과, DB 적재, history append
- 추천 점수 직접 반영 여부:
  아니오
- 데이터 가치:
  오케스트레이션 가치가 큼. 점수 공식 자체를 가지지는 않음

### 2. [`python/model_predict.py`](/d:/ai/Lee_trader/python/model_predict.py)

- 역할:
  최신 feature를 사용해 예측값 생성
- 입력:
  `data/model.pkl`, `data/features.csv`
- 출력:
  `data/predictions.csv`
- 추천 점수 직접 반영 여부:
  예, 간접적으로 반영
- 데이터 가치:
  매우 높음. `ret_score`, `pred_score`, `prob_score`, `risk_penalty` 입력의 핵심 원천

### 3. [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)

- 역할:
  최종 점수 계산 단일 원본 파일
- 입력:
  `predictions.csv`, `scores_final.csv`, `features.csv`, `universe.csv`, `market_status.csv`
- 출력:
  `ranking_final.csv`, `daily_ranking` 테이블
- 추천 점수 직접 반영 여부:
  예, 직접 반영
- 데이터 가치:
  최상위. 운영 랭킹 산출의 핵심

### 4. [`python/rebalance_ranking.py`](/d:/ai/Lee_trader/python/rebalance_ranking.py)

- 역할:
  리밸런싱용 커스텀 랭킹 계산
- 입력:
  `research.prediction_history`, sector/feature 데이터
- 출력:
  `outputs/rebalance/ranking_*.csv`
- 추천 점수 직접 반영 여부:
  운영 일일 랭킹에는 직접 반영되지 않음
- 데이터 가치:
  중간. 운영 점수의 보조 활용 경로

### 5. [`python/quality_builder.py`](/d:/ai/Lee_trader/python/quality_builder.py)

- 역할:
  품질/재무 기반 점수 소스 생성
- 입력:
  재무 데이터
- 출력:
  후속 feature 또는 quality 관련 컬럼
- 추천 점수 직접 반영 여부:
  예, 간접 반영
- 데이터 가치:
  높음. `qual_score`의 근간

### 6. [`python/feature_builder.py`](/d:/ai/Lee_trader/python/feature_builder.py)

- 역할:
  모델 입력 feature 및 일부 랭킹 입력 컬럼 생성
- 입력:
  가격, 재무, 기술 지표 재료
- 출력:
  `data/features.csv`
- 추천 점수 직접 반영 여부:
  예, 직접 반영에 가까운 간접 반영
- 데이터 가치:
  매우 높음. `qual_score`, `safety_score`, `liquidity_score` 및 모델 입력의 기반

### 7. [`python/label_builder.py`](/d:/ai/Lee_trader/python/label_builder.py)

- 역할:
  학습용 정답 라벨 생성
- 입력:
  가격/성과 데이터
- 출력:
  `labels`
- 추천 점수 직접 반영 여부:
  직접은 아님
- 데이터 가치:
  매우 높음. 학습 품질에 직접 영향

### 8. [`python/research/scoring.py`](/d:/ai/Lee_trader/python/research/scoring.py)

- 역할:
  연구/백테스트용 `final_score_custom` 계산
- 입력:
  `ret_score`, `prob_score`, `qual_score`, `tech_score`, `risk_penalty`
- 출력:
  `final_score_custom`
- 추천 점수 직접 반영 여부:
  운영 랭킹에는 아니오
- 데이터 가치:
  연구용으로는 높음, 운영용 기준으로는 분리된 보조 모듈

## 추천 점수에 직접 반영되는 데이터

### 직접 반영됨

- `predictions.csv`
  - `pred_return_60d`
  - `pred_return_90d`
  - `pred_mdd_60d`
  - `pred_mdd_90d`
  - `prob_top20_60d`
- `scores_final.csv`
  - `composite` 또는 `score_score`
- `features.csv`
  - `quality_score`
  - `vol_20`
  - `vol_60`
  - `vol_ma_20`
  - `volume`
- `market_status.csv`
  - `market_up`
  - `kospi_close`
  - `kospi_ma20`

### 직접 반영되지 않지만 중요함

- `universe.csv`
  종목 메타데이터 제공
- `labels`
  모델 학습 품질과 결과에 영향

## 데이터 가치 평가

### 가장 가치가 높은 입력

1. `predictions.csv`
   모델의 미래 예측이므로 최종 점수의 핵심 신호
2. `features.csv`
   quality, safety, liquidity와 모델 feature를 동시에 지탱
3. `market_status.csv`
   동일 종목도 시장 상태에 따라 다른 가중치를 받게 만듦

### 상대적으로 보조적인 입력

1. `universe.csv`
   순위 계산보다는 메타정보에 가까움
2. `scores_final.csv`
   기술 점수 소스지만 구조에 따라 `composite` 품질에 크게 의존

## 현재 구조의 강점

- [x] 운영 점수 계산의 진실원천이 `ranking_builder.py`로 정리되어 있음
- [x] 점수 항목별 함수 분리가 되어 있어 유지보수가 쉬움
- [x] 상대점수와 절대점수가 명확히 섞여 있어 편향을 줄이기 좋음
- [x] bull / defensive 레짐 분리로 시장 국면 적응력이 있음
- [x] `risk_penalty`로 단순 고수익 추종을 제어함
- [x] `run_pipeline.py` 기준 흐름이 `label_builder -> model_train -> model_predict -> ranking_builder`로 명확함

## 현재 구조의 한계

- [ ] `final_score`와 `final_score_v2`가 함께 존재해 운영 기준 점수 정의가 헷갈릴 수 있음
- [ ] `pred_score`는 계산되지만 운영 `final_score`에서 직접 쓰이지 않아 의미 정리가 필요함
- [ ] `RISK_MDD_THRESHOLD`, `RISK_PENALTY_SCALE` 상수는 현재 직접 사용되지 않음
- [ ] `scores_final.csv`의 `composite` / `score_score` 중 무엇이 주 소스인지 데이터 의존성이 남아 있음
- [ ] 일부 research 코드에서 fallback 점수 생성 규칙이 운영 점수 체계와 완전히 일치하지 않을 수 있음

## 향후 개선 우선순위

### 우선순위 1

- `final_score`와 `final_score_v2`의 운영 역할을 명확히 분리
- 운영 기준 점수가 무엇인지 API/DB/문서까지 통일

### 우선순위 2

- `pred_score`를 유지할지, 실제 운영 산식에 넣을지, 완전히 reference로 둘지 결정
- 미사용 상수와 중복 컬럼 정리

### 우선순위 3

- `scores_final.csv` 기술 점수 source schema를 고정
- `composite` vs `score_score` fallback 구조를 문서화 또는 단일화

### 우선순위 4

- research 경로와 운영 경로의 점수 정의 차이를 문서로 명확히 분리
- 백테스트용 `final_score_custom`과 운영 `final_score` 비교 리포트 자동화

## 유지보수 체크포인트

체크리스트:
- [ ] `ranking_builder.py`를 수정할 때는 문서도 같이 갱신할 것
- [ ] 새 입력 컬럼 추가 시 상대점수/절대점수 여부를 먼저 정의할 것
- [ ] 레짐 판정 기준을 바꾸면 가중치 의미도 함께 검토할 것
- [ ] `risk_penalty` 수정 시 실전 drawdown 통제 목적이 유지되는지 확인할 것
- [ ] research score와 운영 score를 혼동하지 않도록 파일 책임을 분리해서 유지할 것
