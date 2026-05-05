# Lee_trader_score Flow

## 실행 흐름
1. `python/run_pipeline.py`
2. `python/model_predict.py`
3. `python/ranking_builder.py`
4. shared score utilities
   - `compute_component_scores`
   - `attach_market_columns`
   - `apply_baseline_final_score`
   - `compute_score_explain`
5. theme overlay and ranking
   - `apply_theme_overlay_v2`
   - `apply_theme_overlay_v3`
   - `_apply_shadow_theme_overlay_v3`
6. 저장
   - `data/ranking_final.csv`
   - `public.daily_ranking`

## 주요 함수 호출 순서
- `ranking_builder.build_ranking()`
  - prediction/features/universe/market merge
  - `apply_theme_overlay(base)` 이전 baseline score build
  - shared scoring:
    - `compute_component_scores(base)`
    - `attach_market_columns(...)`
    - `apply_baseline_final_score(base)`
  - theme 관련 후처리:
    - `apply_theme_overlay_v2(base)`
    - `apply_theme_overlay_v3(base)`
    - `_apply_shadow_theme_overlay_v3(base)`
  - rank 생성:
    - `live_rank`
    - `rank_final`
    - `rank_v2`
- `scoring/final_score.py`
  - `compute_tech_score()`
  - `compute_ret_and_pred_scores()`
  - `compute_prob_score()`
  - `compute_qual_score()`
  - `compute_safety_score()`
  - `compute_liquidity_score()`
  - `compute_valuation_score()`
  - `compute_risk_penalty()`
  - `resolve_core_weight_profile()`
  - `compute_score_explain()`

## 데이터 흐름
- `predictions.csv`
  + `scores_final.csv`
  + `features.csv`
  + `universe.csv`
  + `market_status.csv`
  -> merged base frame
- merged base frame
  -> component scores
  -> regime columns
  -> `final_score`
- `final_score` + theme inputs
  -> `final_score_v2`
  -> `final_score_v3`
  -> `shadow_final_score_v3`
- score outputs
  -> `live_rank`
  -> `rank_final`
  -> `rank_v2`
  -> `data/ranking_final.csv`

## 최종 점수 산식
- baseline:
  - `final_score = w_ret*ret_score + w_prob*prob_score + w_tech*tech_score + w_qual*qual_score - w_risk_penalty*risk_penalty`
- v2:
  - `final_score_v2 = base_weight * final_score + theme_weight * theme_score`
- v3:
  - `final_score_v3 = base_weight * final_score + theme_weight * theme_score_effective`

## rank 결정
- `live_rank`
  - runtime flag가 theme live 반영이면 `final_score_v3` 기준
  - 아니면 `final_score` 기준
- `rank_final`
  - 현재 구현상 `live_rank`와 같은 값으로 맞춰진다
- `rank_v2`
  - `final_score_v2` 비교용 rank

## 확인 필요
- theme overlay runtime 설정이 꺼져 있으면 baseline `final_score` 중심 운영이 된다.
- theme overlay runtime 설정이 켜져 있으면 `live_rank`와 `rank_final` 해석 시 `final_score_v3` 사용 여부를 함께 봐야 한다.
