# Lee_trader_score

## Purpose

이 모듈은 종목별 최종 점수와 운영 정렬 기준을 정리합니다.

범위:
- `final_score` 산출
- `final_score_v2`, `final_score_v3` 산출
- `live_rank`, `rank_final` 결정
- theme overlay와 runtime sorting 해석
- 점수 관련 디버그 산출물 확인

## Main Files

- `python/ranking_builder.py`
- `python/scoring/final_score.py`
- `python/score_explainer.py`
- `python/build_confidence_score_v2.py`
- `python/run_pipeline.py`

## Main Outputs

- `data/ranking_final.csv`
- `outputs/score_breakdown_debug.csv`
- `outputs/confidence_diagnostics_snapshot.csv`
- `outputs/theme_score_impact_compare.csv`

## Read First

- [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/CONTEXT.md>)
- [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/FLOW.md>)
- [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/FILE_INDEX.md>)
- [RUNTIME_SORTING.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/RUNTIME_SORTING.md>)
- [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/ENV.md>)
- [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/OPERATIONS.md>)
