# Score ENV

## Purpose

이 문서는 점수 산출과 정렬 기준에 영향을 주는 핵심 환경/설정 요소를 정리합니다.

## Main Inputs

| 항목 | 설명 | 영향 범위 |
| --- | --- | --- |
| `data/predictions.csv` | 모델 예측값 | ranking |
| `data/features.csv` | feature 원본 | scoring |
| `data/market_status.csv` | 시장 상태 | regime/risk 해석 |
| `config/production_v1.yaml` | production ranking 정책 | runtime sorting |

## Runtime Notes

직접적인 `.env` 변수보다 설정 파일과 입력 CSV의 영향이 더 큽니다.

주요 확인 대상:

- `ranking.theme_overlay.enabled`
- `ranking.theme_overlay.mode`
- confidence 관련 보조 산출물
