# BackTest ENV

## Purpose

이 문서는 백테스트/워크포워드 모듈에서 성과 해석에 직접 영향을 주는 환경변수를 정리합니다.

## Core Variables

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `MODEL_VERSION` | 프로젝트 기준 | 대상 모델 버전 | walkforward |
| `HORIZON_DAYS` | 프로젝트 기준 | 예측 horizon | walkforward |
| `TOP_N` | 프로젝트 기준 | ranking cut | backtest ranking |
| `DATABASE_URL` | 없음 | 결과 적재 DB | history tables |

## RULE Portfolio Backtest Related

| 변수명 | 기본값 | 설명 | 영향 범위 |
| --- | --- | --- | --- |
| `RULE_MAX_POSITIONS` | `5` | 최대 보유 종목 | rule portfolio backtest |
| `RULE_NEW_ENTRY_WEIGHT` | `0.10` | 신규 진입 비중 | rule portfolio backtest |
| `RULE_MAX_POSITION_WEIGHT` | `0.20` | 종목 비중 상한 | rule portfolio backtest |
| `RULE_MIN_CASH_WEIGHT` | `0.20` | 최소 현금 비중 | rule portfolio backtest |
| `RULE_MAX_SECTOR_WEIGHT` | `0.35` | 섹터 비중 상한 | rule portfolio backtest |
| `RULE_COOLDOWN_DAYS` | `5` | 재진입 쿨다운 | rule portfolio backtest |
| `RULE_MAX_HOLDING_DAYS` | `10` | 최대 보유일 | rule portfolio backtest |
| `RULE_STOP_LOSS_PCT` | `0.05` | 손절 기준 | rule portfolio backtest |
| `RULE_TRAILING_STOP_PCT` | `0.04` | trailing stop 기준 | rule portfolio backtest |
| `RULE_TRAILING_STOP_MIN_PROFIT_PCT` | `0.03` | trailing stop 최소 이익 기준 | rule portfolio backtest |
