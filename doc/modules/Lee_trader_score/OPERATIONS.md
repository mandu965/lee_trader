# Score Operations

## Purpose

이 문서는 점수 산출 결과를 확인할 때 기본적으로 보는 파일과 점검 순서를 정리합니다.

## Main Commands

```powershell
python python/run_pipeline.py
```

또는 ranking 관련 단일 경로 확인:

```powershell
python python/ranking_builder.py
```

## Key Outputs

- `data/ranking_final.csv`
- `outputs/score_breakdown_debug.csv`
- `outputs/confidence_diagnostics_snapshot.csv`
- `outputs/theme_score_impact_compare.csv`
