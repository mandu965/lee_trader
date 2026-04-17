# Walk-Forward Backtest

## 목적

이 문서는 현재 코드 기준 walk-forward backtest 실행 방법을 정리한다.

- 단일 `data/model.pkl`을 전체 기간에 백필하는 방식이 아니다.
- 분기별 재학습한다.
- 각 분기 run은 expanding window 방식으로 학습 구간을 누적 확장한다.
- 각 run은 `research.dim_model_run`에 등록되고, 결과는 `research.prediction_history`, `research.ranking_history`, `research.backtest_outcome`에 `run_id` 기준으로 적재된다.

## 실행 흐름

`python/walkforward_backtest.py`는 각 분기 window마다 아래 순서로 실행한다.

1. `model_train.py`
2. `build_backtest_predictions.py`
3. `build_backtest_ranking.py`
4. `build_backtest_outcome.py`

각 run의 모델 파일은 `artifacts/models/` 아래에 저장된다.

예시 파일명:

- `artifacts/models/run_123_wf_h60_20241231.pkl`

## 현재 기본 설정

- horizon: `60`
- top_n: `20`
- 재학습 주기: 분기별
- 학습 방식: expanding window
- 첫 학습 시작 조건: 기본 `12개월` 이상 학습 구간 확보 후 첫 분기말부터 시작

## 실행 예시

```powershell
python python\walkforward_backtest.py ^
  --universe-version universe_20260313 ^
  --score-weights-json "{\"ret\":0.4,\"prob\":0.25,\"qual\":0.15,\"tech\":0.10,\"bias\":0.10}"
```

기간을 제한해서 실행:

```powershell
python python\walkforward_backtest.py ^
  --start-date 2021-01-01 ^
  --end-date 2025-12-31 ^
  --universe-version universe_20260313 ^
  --score-weights-json "{\"ret\":0.4,\"prob\":0.25,\"qual\":0.15,\"tech\":0.10,\"bias\":0.10}"
```

최소 학습 기간을 조정해서 실행:

```powershell
python python\walkforward_backtest.py ^
  --min-train-months 18 ^
  --universe-version universe_20260313 ^
  --score-weights-json "{\"ret\":0.4,\"prob\":0.25,\"qual\":0.15,\"tech\":0.10,\"bias\":0.10}"
```

## 개별 구성요소 설명

### model_train.py

- `--train-end-date` 인자를 지원한다.
- 학습 데이터는 `date <= train_end_date` 조건으로 제한된다.
- model pack에는 아래 메타데이터가 포함된다.
  - `model_version`
  - `train_end_date`
  - `trained_at`

예시:

```powershell
python python\model_train.py ^
  --horizons 60 ^
  --train-end-date 2024-12-31 ^
  --model-version wf_h60_20241231 ^
  --output-pkl artifacts\models\run_test.pkl
```

### build_backtest_predictions.py

- 지정된 model pack으로 지정 기간 예측을 생성한다.
- `run_id`, `model_version`, `horizon_days`를 붙여 `research.prediction_history`에 적재한다.

### build_backtest_ranking.py

- 같은 `run_id`의 prediction history를 읽어서 날짜별 rank를 계산한다.
- 결과를 `research.ranking_history`에 적재한다.

### build_backtest_outcome.py

- 같은 `run_id`의 prediction history와 `labels`를 조인한다.
- 실현 수익률 / 실현 MDD를 계산해 `research.backtest_outcome`에 적재한다.

## 검증 방법

### 1. dim_model_run 생성 여부

```sql
SELECT run_id, run_type, model_version, horizon_days, top_n, train_start_date, train_end_date, created_at
FROM research.dim_model_run
WHERE run_type = 'walkforward_backtest'
ORDER BY run_id DESC;
```

기대 결과:

- 분기별 run이 생성되어 있어야 한다.
- `train_end_date`가 분기말 기준으로 증가해야 한다.

### 2. prediction_history 적재 여부

```sql
SELECT run_id, COUNT(*) AS rows, MIN(as_of_date), MAX(as_of_date)
FROM research.prediction_history
GROUP BY run_id
ORDER BY run_id DESC;
```

기대 결과:

- 각 run마다 예측 구간 row가 존재해야 한다.
- `as_of_date` 범위가 해당 분기 prediction window와 일치해야 한다.

### 3. ranking_history 적재 여부

```sql
SELECT run_id, COUNT(*) AS rows, MIN(rank), MAX(rank)
FROM research.ranking_history
GROUP BY run_id
ORDER BY run_id DESC;
```

기대 결과:

- 각 run마다 ranking row가 존재해야 한다.
- 날짜별로 `rank=1..N` 구조가 형성되어야 한다.

### 4. backtest_outcome 적재 여부

```sql
SELECT run_id, COUNT(*) AS rows, AVG(realized_return) AS avg_ret
FROM research.backtest_outcome
GROUP BY run_id
ORDER BY run_id DESC;
```

기대 결과:

- 각 run마다 outcome row가 존재해야 한다.
- `realized_return`이 비어 있지 않아야 한다.

### 5. 모델 메타데이터 확인

```powershell
@'
import pickle
from pathlib import Path

path = Path("artifacts/models")
model = sorted(path.glob("*.pkl"))[-1]
with open(model, "rb") as f:
    pack = pickle.load(f)

print("model:", model)
print("model_version:", pack.get("model_version"))
print("train_end_date:", pack.get("train_end_date"))
print("trained_at:", pack.get("trained_at"))
'@ | python -
```

기대 결과:

- `model_version`
- `train_end_date`
- `trained_at`

값이 존재해야 한다.

## 해석 주의

- 각 run은 해당 run의 `train_end_date` 이전 데이터만으로 학습된다.
- 이후 기간만 예측하므로, 단일 모델을 전체 기간에 적용하는 백필보다 실제 운용에 가깝다.
- 첫 버전은 `horizon=60`, `top_n=20`만 고정 지원한다.
