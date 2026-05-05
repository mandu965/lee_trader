# Lee_trader_score Flow

## Execution Flow

### 1. Prediction Input

입력:

- `data/predictions.csv`
- `data/scores_final.csv`
- `data/features.csv`
- `data/universe.csv`
- `data/market_status.csv`

### 2. Base Score Build

주요 파일:

- `python/ranking_builder.py`
- `python/scoring/final_score.py`

대표 단계:

1. prediction/features/universe/market 병합
2. component score 계산
3. regime column 부착
4. baseline `final_score` 계산

### 3. Theme / Runtime Overlay

주요 단계:

- `final_score_v2`
- `final_score_v3`
- shadow theme overlay

runtime 설정에 따라:

- `live_rank`
- `rank_final`

기준 점수 컬럼이 달라질 수 있습니다.

### 4. Ranking Output

출력:

- `data/ranking_final.csv`
- `public.daily_ranking`
- 점수 보조 debug 산출물

## Score Layers

### Baseline

`final_score`

### Overlay

- `final_score_v2`
- `final_score_v3`

### Runtime Rank

- `live_rank`
- `rank_final`

## Main Checks

- 현재 운영 기준이 `final_score`인지 `final_score_v3`인지
- `live_rank`와 `rank_final`이 같은 기준인지
- theme overlay가 실제로 켜져 있는지
- confidence/score explain이 화면 해석과 일치하는지

## Notes

- 운영 정렬 기준은 [RUNTIME_SORTING.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/RUNTIME_SORTING.md>)를 우선 확인합니다.
- 점수 산식 변경은 ranking, UI, 주문 후보 해석까지 연쇄 영향이 있습니다.
