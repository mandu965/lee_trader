# Lee_trader_score Context

## 상세 설명
- 점수 모듈의 기준점은 `python/ranking_builder.py`다.
- 이 파일은 예측값, quality/technical/market 정보를 합쳐서 종목별 운영 점수를 계산한다.
- 실제 component 계산의 상당 부분은 shared utility인 `python/scoring/final_score.py`로 이동되어 있다.
- 따라서 점수 해석은 `ranking_builder.py` 단독이 아니라 다음 조합으로 봐야 한다.
  - 입력 merge: `ranking_builder.py`
  - component score 계산: `scoring/final_score.py`
  - explain 컬럼 부착: `score_explainer.py`
  - confidence 해석 보강: `build_confidence_score_v2.py`

## 로직 개요
- 입력 결합
  - `predictions.csv`에서 `pred_return_60d`, `pred_return_90d`, `prob_top20_60d`, `pred_mdd_60d`, `pred_mdd_90d`
  - `scores_final.csv`에서 legacy tech source 후보 `composite`, `score_score`
  - `features.csv`에서 quality, volatility, liquidity, RSI, MA, valuation 관련 컬럼
  - `universe.csv`에서 `name`, `sector`, `market`
  - `market_status.csv`에서 regime 판별용 market context
- component score
  - `tech_score`
  - `ret_score`
  - `prob_score`
  - `qual_score`
  - diagnostic only:
    - `safety_score`
    - `liquidity_score`
    - `valuation_score`
- regime / risk
  - market regime를 `bull`, `neutral`, `defensive` 중 하나로 정리
  - regime마다 가중치 프로필을 다르게 적용
  - `pred_mdd_60d`, `pred_mdd_90d` 기반 `risk_penalty` 차감
- theme overlay
  - baseline `final_score`는 운영 기준점
  - `final_score_v2`는 theme 점수 직접 혼합 비교축
  - `final_score_v3`는 theme confidence 반영 비교축
  - `live_rank`는 runtime flag에 따라 `final_score` 또는 `final_score_v3`를 사용

## 운영상 주의사항
- `final_score`는 운영 기준선이라 downstream 의존성이 가장 크다.
- `final_score_v2`, `final_score_v3`는 theme 영향 비교와 live 정렬 실험까지 포함하므로, 변경 시 top20 구성이 달라질 수 있다.
- `public.daily_ranking` 저장 컬럼이 매우 많아서 점수 컬럼명 변경은 API/DB/UI에 직접 영향을 준다.
- `prob_score`는 운영 정책상 `prob_top20_60d`만 사용한다. `prob_top20_90d`는 연구용 보조 축이다.
- `pred_score`는 legacy/research comparison 용이고, baseline 운영 점수의 직접 축은 `ret_score`다.

## 다른 모듈과의 관계
- `Lee_trader_ai`
  - 전체 파이프라인 문맥을 공유한다.
- `Lee_trader_backTest`
  - 비슷한 점수 개념을 백테스트 이력 계산에 재사용한다.
- `Lee_trader_rule`
  - rule engine 자체 점수는 별도지만, 운영 UI/DB payload에서는 AI 점수 컬럼과 함께 해석될 수 있다.

## 확인 필요
- 실제 live 정렬 기준은 runtime theme overlay flag에 따라 `final_score` 또는 `final_score_v3`로 달라진다.
- 따라서 운영 기준을 문서화할 때는 "현재 설정의 runtime flag"까지 같이 기록해야 한다.
