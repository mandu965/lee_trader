# Lee_trader_score

## 모듈 목적
- 이 문서는 종목별 최종 점수 산출 경로를 따로 설명하기 위한 모듈 문서다.
- 실제 별도 소스 디렉토리가 있는 것은 아니고, 현재 저장소에서는 `python/ranking_builder.py`와 `python/scoring/final_score.py`가 운영 점수 계산의 중심이다.
- 목적은 각 종목의 `final_score`, `final_score_v2`, `final_score_v3`, `live_rank`, `rank_final`이 어떻게 만들어지는지 추적 가능하게 만드는 것이다.

## 핵심 기능
- `python/scoring/final_score.py`
  - component score 계산
  - market regime 부착
  - risk penalty 계산
  - baseline `final_score` 계산
- `python/ranking_builder.py`
  - 입력 데이터 merge
  - shared scoring 호출
  - theme overlay 적용
  - `final_score_v2`, `final_score_v3`, `live_rank`, `rank_final` 생성
  - `data/ranking_final.csv` 및 `public.daily_ranking` 저장
- `python/score_explainer.py`
  - 점수 설명용 요약 컬럼 생성
- `python/build_confidence_score_v2.py`
  - 점수 자체와 별개로 confidence 축을 보강하고 downstream 해석에 사용
- `doc/modules/Lee_trader_score/RUNTIME_SORTING.md`
  - 실제 운영에서 어떤 점수 컬럼이 rank 기준인지 설명

## 입력 데이터
- `data/predictions.csv`
- `data/scores_final.csv`
- `data/features.csv`
- `data/universe.csv`
- `data/market_status.csv`
- optional theme overlay source
  - `output/stock_theme_daily.csv`
  - 또는 DB overlay source

## 출력 데이터
- `data/ranking_final.csv`
- `outputs/score_breakdown_debug.csv`
- `outputs/confidence_diagnostics_snapshot.csv`
- `outputs/theme_score_impact_compare.csv`
- `public.daily_ranking`

## 최종 점수 개요
- baseline 운영 점수:
  - `final_score = w_ret * ret_score + w_prob * prob_score + w_tech * tech_score + w_qual * qual_score - w_risk_penalty * risk_penalty`
- comparison score:
  - `final_score_v2 = base_weight * final_score + theme_weight * theme_score`
- theme confidence adjusted score:
  - `final_score_v3 = base_weight * final_score + theme_weight * theme_score_effective`
- live 정렬 컬럼:
  - theme live 반영 플래그가 켜져 있으면 `final_score_v3`
  - 아니면 `final_score`

## 주요 실행 파일
- `python/ranking_builder.py`
- `python/scoring/final_score.py`
- `python/score_explainer.py`
- `python/run_pipeline.py`
