# Lee_trader_score File Index

## 목적

종목별 최종 점수, 파생 점수, 정렬 순위, 설명 컬럼을 만드는 핵심 파일 인덱스입니다.
점수 모듈은 수식 변경 하나가 ranking, top 후보, 주문 preview까지 연쇄 영향을 주므로 변경 범위를 먼저 확인해야 합니다.

## 핵심 파일

| 파일 | 역할 | 수정 위험도 | 함께 확인할 파일 |
| --- | --- | --- | --- |
| `python/ranking_builder.py` | 운영용 최종 점수와 순위 산출의 중심 엔진 | 매우 높음 | `scoring/final_score.py`, `score_explainer.py`, `build_confidence_score_v2.py` |
| `python/scoring/final_score.py` | `final_score` 기본 계산식과 penalty/overlay 조합 | 매우 높음 | `ranking_builder.py`, `production_config.py` |
| `python/score_explainer.py` | 점수 설명 문구와 driver 컬럼 생성 | 높음 | `ranking_builder.py`, `node/index.js` |
| `python/build_confidence_score_v2.py` | confidence score와 confidence band 산출 | 높음 | `ranking_builder.py`, `build_operational_buy_gate.py` |
| `python/production_config.py` | 운영 점수 관련 runtime 설정 로드 | 높음 | `ranking_builder.py`, `config/production_v1.yaml` |
| `python/run_pipeline.py` | 점수 산출까지 포함한 일일 파이프라인 진입점 | 중간 | `ranking_builder.py`, `model_predict.py` |
| `python/sync_web_display_data.py` | 점수 결과를 DB와 payload로 동기화 | 높음 | `ranking_builder.py`, `node/index.js` |
| `node/index.js` | ranking API와 점수 설명 응답 | 높음 | `node/public/ranking.js`, `node/public/score-check.js` |
| `node/public/ranking.js` | 랭킹 화면 점수 표시 | 중간 | `node/index.js` |
| `node/public/score-check.js` | 점수 검증 화면 렌더링 | 중간 | `node/index.js`, `score_explainer.py` |

## 보조 파일

| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `python/check_score_explain.py` | 설명 컬럼 검증 | score explain 회귀 확인용 |
| `python/check_final_score_dominance.py` | 특정 점수 축 지배 여부 점검 | 점수 편향 점검용 |
| `python/check_confidence_score.py` | confidence 분포 확인 | 운영 진입 품질 점검용 |
| `python/build_confidence_calibration_report.py` | confidence calibration 리포트 | 해석용 문서 생성 |
| `config/production_v1.yaml` | theme overlay, ranking runtime 설정 | 현재 운영 정렬 기준 확인용 |

## 수정 원칙

- `final_score`, `live_rank`, `rank_final` 기준이 바뀌면 [RUNTIME_SORTING.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/RUNTIME_SORTING.md>)를 같이 갱신합니다.
- 점수 수식 실험과 운영 기준 변경은 같은 커밋에 섞지 않는 편이 안전합니다.
- 점수 설명 컬럼 변경은 API 응답과 화면 문구까지 함께 확인합니다.
