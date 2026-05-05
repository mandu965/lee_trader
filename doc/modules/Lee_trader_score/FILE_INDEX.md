# Lee_trader_score File Index

## 핵심 파일 목록
| 파일 | 역할 | 수정 가능 여부 | 수정 시 주의사항 |
| --- | --- | --- | --- |
| `python/ranking_builder.py` | 운영 점수 계산 총괄, `ranking_final.csv` 및 `daily_ranking` 저장 | 핵심 파일, 매우 신중 | 점수 컬럼과 rank 컬럼이 API/DB/UI에 직접 연결된다 |
| `python/scoring/final_score.py` | shared component score, regime, risk penalty, baseline final score 계산 | 핵심 파일, 매우 신중 | `final_score` 산식의 기준선이다 |
| `python/score_explainer.py` | 점수 설명용 summary/driver 컬럼 생성 | 수정 가능 | UI 설명 문구와 downstream payload 해석이 변한다 |
| `python/build_confidence_score_v2.py` | confidence score 및 live confidence 등급 산출 | 수정 가능 | 점수 자체는 아니지만 운영 해석 기준이 바뀐다 |
| `python/run_pipeline.py` | 점수 생성 전후 배치 순서 제어 | 수정 가능 | `ranking_builder.py` 호출 순서 변경 시 전체 산출물이 달라질 수 있다 |
| `python/sync_web_display_data.py` | 점수 결과를 payload / DB에 적재 | 수정 가능 | `research.app_payload_store`, `public.daily_ranking` 계약 유지 필요 |
| `node/index.js` | ranking 관련 API 제공 | 수정 가능 | `final_score`, `rank_final`, explain 컬럼 응답과 직접 연결된다 |

## 수정 기준
- `final_score` 기준선 변경은 사실상 운영 정책 변경이다.
- `final_score_v2`, `final_score_v3` 변경은 theme 반영 정책 변경으로 보고 top20 변화를 같이 검증해야 한다.
- `rank_final`, `live_rank` 계산 기준을 바꾸면 주문 후보와 UI 정렬이 동시에 바뀐다.

## 확인 필요
- `ranking_builder.py` 안에는 baseline 점수 외에도 theme overlay, shadow overlay, experiment 출력이 많이 섞여 있다.
- 운영 기준선을 바꾸는 수정과 실험용 비교축 수정을 분리해서 다루는 것이 안전하다.
